"""Instruments Wemb's (embedding) own residual accumulator for the same windup signature already
confirmed on Win/Wh/Wgx/Wgh. Nothing built or tested in the whole 2026-10-29/30 investigation touched
Wemb -- every fix/test subclassed only _residual_update_level. Wemb uses a DIFFERENT, unguarded
formula (_residual_update_wide: L2/RMS normalize + VARIABLE multi-level-per-call step,
native_quinary_charlm.py lines 331-366) -- the same variable-step formula the project's own comment
says "measured worse" when tried on the level weights, here applied unmodified to the one set of
weights that feeds every single timestep's input (x_t = Wemb/EQ).

Measures, per rep, for ONLY the Wemb columns corresponding to characters that actually appear in the
taught fact: |res|/THRESHOLD after each call's accumulation (the same windup signature Qwen's
diagnostic found at 27-34x on the recurrent weights), and how many magnitude-units the variable step
actually moved in that call (>1 = a multi-level jump in one shot, never possible on any other weight
in this architecture since those are hard-capped at +-1/call).

Not touching production code or any real checkpoint -- fresh in-memory brick only, same discipline as
every other scratch test this session.
"""
import sys
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick
from native_quinary_charlm import THRESHOLD, EQ, MAX_EMB, EMB_DECAY_DIV
from lutgtlm_ingest import _line_batches, enc


class WembDiagBrick(FixedStepBrick):
    """Exact copy of NativeQuinaryCharLM._residual_update_wide's real math (native_quinary_charlm.py
    lines 331-366), unmodified, with instrumentation added. Copied verbatim rather than monkeypatched
    so the real formula can't silently drift from what's measured here -- the only additions are the
    `self.wemb_log` capture and dropping the _total_in/_total_consumed/_total_discarded bookkeeping
    (pure accounting, doesn't affect the actual weight/residual values being measured)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.wemb_log = []  # one dict per learn_batch call, Wemb only

    def _residual_update_wide(self, name, W, grad):
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize_v2(name, mom_int)
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
            self.wemb_log.append({
                "res_after": res_after.copy(),   # (D, V)
                "step": step.copy(),             # (D, V) -- |step|>1 = multi-level jump in one call
                "W": clipped.copy(),
            })
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
fact_char_ids = sorted(set(enc(FACT)))

brick = WembDiagBrick(V=256, D=40, seed=0)

print(f"random baseline: {RANDOM_BASELINE:.3f} nats")
print(f"THRESHOLD={THRESHOLD}  distinct characters in fact: {len(fact_char_ids)}\n")

header = (f"{'rep':>4} | {'CE':>7} | {'max|res|/THR':>13} | {'%cols>1xTHR':>12} | "
          f"{'%cols>5xTHR':>12} | {'max step mag':>13} | {'max|Wemb|':>10} | {'MAX_EMB':>8}")
print(header)
for rep in range(1, 11):
    teach(brick, FACT)
    ce = teacher_forced_ce(brick, FACT)
    log = brick.wemb_log[-1]
    res_fact = log["res_after"][:, fact_char_ids]
    step_fact = log["step"][:, fact_char_ids]
    W_fact = log["W"][:, fact_char_ids]

    windup_ratio = np.abs(res_fact) / THRESHOLD
    pct_over_1x = 100.0 * np.mean(windup_ratio > 1.0)
    pct_over_5x = 100.0 * np.mean(windup_ratio > 5.0)
    max_ratio = windup_ratio.max()
    max_step = int(np.abs(step_fact).max())
    max_w = int(np.abs(W_fact).max())

    print(f"{rep:>4} | {ce:>7.3f} | {max_ratio:>13.2f} | {pct_over_1x:>11.1f}% | "
          f"{pct_over_5x:>11.1f}% | {max_step:>13d} | {max_w:>10d} | {MAX_EMB:>8d}")

print("\n--- rep 1 detail: was even the FIRST update already a multi-level jump for any fact char? ---")
log1 = brick.wemb_log[0]
step1 = log1["step"][:, fact_char_ids]
jumps = np.abs(step1)
print(f"max |step| on rep 1 (fact cols): {int(jumps.max())}  "
      f"(>1 means Wemb moved more than one unit for that (hidden-dim,char) pair on the FIRST teach)")
n_multi = int(np.sum(jumps > 1))
print(f"count of (hidden-dim,char) pairs with |step|>1 on rep 1: {n_multi} / {jumps.size}")
