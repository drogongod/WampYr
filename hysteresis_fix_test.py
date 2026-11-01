"""Hysteresis fix, ported from the REAL Bonin NeurIPS repo (bitdiffusion/src/bitdiffusion/
optim.py:84-92, ed_hyst_step) -- their own branch table independently names the same "avalanche"
phenomenon this project found. Their fix: raise the flip threshold right after a weight flips, via
a per-weight refractory counter, then relax it back down. 100% no-shadow compliant -- refract is
small per-weight integer/float state, same category as the existing _res/_mom buffers, never a
master copy of the weight.

Tests on this project's own real diagnostic (one-shot-fact teach x N reps, D=40), same methodology
discipline as every other fix this investigation: real generate() output is the verdict, not CE.

Applied to NativeQuinaryCharLM's ORIGINAL _residual_update_level (the variable-step mechanism where
windup/avalanche were actually measured in the diary) -- NOT FixedStepBrick's override, since that
override (fixed +-1/call) was itself a reaction to windup and this is testing a different, more
targeted answer to the same problem.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from native_quinary_charlm import NativeQuinaryCharLM, THRESHOLD, EQ
from lutgtlm_ingest import _line_batches, enc, ITOS, bytes_to_text


class HystBrick(NativeQuinaryCharLM):
    """HYST_DELTA=None reproduces the real production baseline (plain _residual_update_level,
    flat THRESHOLD, no hysteresis). A numeric value enables hysteresis: right after a weight flips,
    its effective threshold is THRESHOLD*(1+HYST_DELTA), decaying by 1/REFRACT_MAX of that bonus
    per subsequent non-flipping call."""
    HYST_DELTA = None
    REFRACT_MAX = 4

    def _residual_update_level(self, name, level, grad):
        if self.HYST_DELTA is None:
            return super()._residual_update_level(name, level, grad)
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize(name, mom_int)
        res = self._res.get(name, np.zeros_like(level, dtype=np.int64))
        res = res + scaled
        refract = self._mom.get(name + "_refract", np.zeros_like(level, dtype=np.float64))
        eff_threshold = THRESHOLD * (1.0 + self.HYST_DELTA * refract / self.REFRACT_MAX)
        step = np.sign(res) * (np.abs(res) // eff_threshold).astype(np.int64)
        desired = level.astype(np.int64) - step
        new_level = np.clip(desired, -2, 2).astype(np.int8)
        actual_step = level.astype(np.int64) - new_level.astype(np.int64)
        flipped = actual_step != 0
        res = res - actual_step * THRESHOLD
        refract = np.where(flipped, float(self.REFRACT_MAX), np.maximum(refract - 1.0, 0.0))
        self._mom[name + "_refract"] = refract
        self._res[name] = res
        return new_level


def teacher_forced_ce(brick, line):
    ids = enc(line)
    logits = brick.forward_logits(ids[:-1])
    targets = np.array(ids[1:])
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return float(-logp[np.arange(len(targets)), targets].mean())


def teach(brick, line):
    for b in _line_batches(line):
        brick.learn_batch(b, lr_scale=1.0, corrob_eligible=False)


RANDOM_BASELINE = np.log(256)
FACT = "User: What is the capital of France?\nLucy: The capital of France is Paris."
PROMPT = "User: What is the capital of France?\nLucy: "

print(f"random baseline: {RANDOM_BASELINE:.3f} nats, THRESHOLD={THRESHOLD}\n")

configs = {
    "Baseline (no hyst, reproduces windup)": None,
    "Hyst delta=1.0 ": 1.0,
    "Hyst delta=3.0 ": 3.0,
    "Hyst delta=8.0 ": 8.0,
}
bricks = {name: HystBrick(V=256, D=40, seed=0) for name in configs}
for name, delta in configs.items():
    bricks[name].HYST_DELTA = delta

print("=== Part 1: teacher-forced CE per rep (20 reps -- the windup bug needed 6-8 to show up) ===")
header = f"{'rep':>4} | " + " | ".join(configs.keys())
print(header)
for rep in range(1, 21):
    row = [f"{rep:>4}"]
    for name in configs:
        teach(bricks[name], FACT)
        ce = teacher_forced_ce(bricks[name], FACT)
        row.append(f"{ce:>10.3f}")
    print(" | ".join(row))

print("\n=== Part 2: THE REAL TEST -- actual generated text after 20 reps ===")
print(f"Prompt: {PROMPT!r}\n")
seed_ids = enc(PROMPT)
for name, brick in bricks.items():
    for temp_label, temp in (("t0.7", 0.7), ("greedy", 0.01)):
        out = brick.generate(seed_ids, ITOS, n=60, temp=temp, seed=0)
        text = bytes_to_text(out)
        print(f"[{name}] {temp_label} -> {text!r}")
    print()
