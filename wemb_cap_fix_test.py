"""FIX ATTEMPT for the real Wemb explosion found in wemb_windup_instrument.py: by rep 6 of the
standard identical-fact-x10 diagnostic, _adaptive_normalize_v2's own second-moment tracker (`v`,
updated via a RATE-LIMITED EMA, |d|//MOMENTUM_K per call) lags so far behind a sudden correlated
gradient spike that `rms=isqrt(v)` stays small while the numerator (built from the CURRENT, already-
large gradient) keeps growing -- `scaled = numres // denom` then explodes (measured: 2-3 units/call
for reps 1-5, then 32 BILLION units on rep 6 onward), saturating MAX_EMB=5000 every call from then on.
Same failure SHAPE as the already-known/already-fixed bias runaway (244M magnitude), different
location (Wemb's RMS normalizer, never examined before tonight).

FIX (Bonin-style: cap the DEMANDED MOVEMENT before it enters the residual accumulator, not after --
the one Bonin mechanism already partially validated on the level weights, applied here to the thing
that's actually exploding): clip `scaled` to +-WEMB_CAP_MULT*NORM_TARGET before it's added to `res`.
Discarded amount is explicit, not silent -- same honesty standard _residual_update_wide already holds
itself to for decay/clip (see check_conservation's "discarded" accounting in the real file).

Tests, in order: (1) does the cap actually kill the billion-unit explosion, at several cap strengths;
(2) does CE still behave sanely afterward; (3) THE REAL TEST per this project's own hard-learned
lesson tonight -- what does cluster.generate() actually produce, checked by eye, not inferred from CE.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick
from native_quinary_charlm import THRESHOLD, EQ, MAX_EMB, EMB_DECAY_DIV, NORM_TARGET
from lutgtlm_ingest import _line_batches, enc, ITOS, bytes_to_text


class WembCapBrick(FixedStepBrick):
    """WEMB_CAP_MULT=None reproduces the real production bug exactly (uncapped baseline). A numeric
    value caps `scaled` (the per-call demanded movement into Wemb's residual) to that many multiples
    of NORM_TARGET -- inserted AFTER _adaptive_normalize_v2, BEFORE accumulation into `res`, exactly
    where Bonin's own fix inserts its cap for the level weights."""
    WEMB_CAP_MULT = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.wemb_log = []
        self.wemb_discarded = 0

    def _residual_update_wide(self, name, W, grad):
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize_v2(name, mom_int)
        if name == "Wemb" and self.WEMB_CAP_MULT is not None:
            lim = int(self.WEMB_CAP_MULT * NORM_TARGET)
            pre = scaled
            scaled = np.clip(scaled, -lim, lim)
            self.wemb_discarded += int(np.abs(pre - scaled).sum())
        res = self._res.get(name, np.zeros_like(W, dtype=np.int64))
        res = res + scaled
        step = np.sign(res) * (np.abs(res) // THRESHOLD)
        res_after = res - step * THRESHOLD
        W_new = W - step
        decay = np.sign(W_new) * (np.abs(W_new) // EMB_DECAY_DIV)
        W_new = W_new - decay
        clipped = np.clip(W_new, -MAX_EMB, MAX_EMB)
        self._res[name] = res_after
        if name == "Wemb":
            self.wemb_log.append({"step": step.copy(), "W": clipped.copy()})
        return clipped


def teacher_forced_ce(brick, line):
    ids = enc(line)
    logits = brick.forward_logits(ids[:-1])
    targets = np.array(ids[1:])
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return float(-logp[np.arange(len(targets)), targets].mean())

def teach(brick, line, lr_scale=1.0):
    for b in _line_batches(line):
        brick.learn_batch(b, lr_scale=lr_scale, corrob_eligible=False)


RANDOM_BASELINE = np.log(256)
FACT = "User: What is the capital of France?\nLucy: The capital of France is Paris."
PROMPT = "User: What is the capital of France?\nLucy: "
fact_char_ids = sorted(set(enc(FACT)))

configs = {
    "NoCap (reproduces bug)": None,
    "Cap=1x NORM_TARGET    ": 1.0,
    "Cap=2x NORM_TARGET    ": 2.0,
    "Cap=5x NORM_TARGET    ": 5.0,
    "Cap=10x NORM_TARGET   ": 10.0,
}
bricks = {}
for name, mult in configs.items():
    b = WembCapBrick(V=256, D=40, seed=0)
    b.WEMB_CAP_MULT = mult
    bricks[name] = b

print(f"random baseline: {RANDOM_BASELINE:.3f} nats, THRESHOLD={THRESHOLD}, NORM_TARGET={NORM_TARGET}\n")

print("=== Part 1: CE and max step magnitude (fact-cols only), per rep ===")
header = f"{'rep':>4} | " + " | ".join(f"{n} CE" for n in configs)
print(header)
step_mag_log = {name: [] for name in configs}
for rep in range(1, 11):
    row = [f"{rep:>4}"]
    for name, brick in bricks.items():
        teach(brick, FACT)
        ce = teacher_forced_ce(brick, FACT)
        step_fact = brick.wemb_log[-1]["step"][:, fact_char_ids]
        step_mag_log[name].append(int(np.abs(step_fact).max()))
        row.append(f"{ce:>8.3f}")
    print(" | ".join(row))

print("\n=== Part 2: max |step| magnitude on Wemb (fact cols), per rep, per config ===")
header2 = f"{'rep':>4} | " + " | ".join(configs.keys())
print(header2)
for rep in range(10):
    row = [f"{rep+1:>4}"]
    for name in configs:
        row.append(f"{step_mag_log[name][rep]:>20,d}" if step_mag_log[name][rep] > 999 else f"{step_mag_log[name][rep]:>20d}")
    print(" | ".join(row))

print("\n=== Part 3: THE REAL TEST -- actual generated text after 10 reps, same prompt, each config ===")
print(f"Prompt: {PROMPT!r}\n")
seed_ids = enc(PROMPT)
for name, brick in bricks.items():
    out = brick.generate(seed_ids, ITOS, n=60, temp=0.7, seed=0)
    text = bytes_to_text(out)
    print(f"[{name}] discarded_total={getattr(brick, 'wemb_discarded', 0):>15,d}")
    print(f"  -> {text!r}\n")
