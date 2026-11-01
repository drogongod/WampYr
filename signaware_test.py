"""Adapts Bonin's sign-aware firing condition to quinary's multi-level case. Their actual rule
(binary only): a flip fires only when accumulated evidence points AWAY FROM the weight's current
sign -- i.e. evidence that CONFIRMS the current state does nothing (nowhere better to go for a
binary weight); only CONTRADICTING evidence, once it clears threshold, acts.

For quinary this doesn't port literally (a weight has room to move further from zero, unlike
binary), but the INTENT generalizes as an ASYMMETRIC threshold: moving TOWARD zero (retreating from
a commitment) should be cheap/normal; moving AWAY from zero (deepening a commitment, e.g. 0->+-1,
+-1->+-2) should require MORE evidence. This also directly matches Qwen's own suggestion from the
cross-model handoff (asymmetric per-direction thresholds, harder near the extremes) -- independent
convergence on the same idea from a different angle.

AWAY_MULT: how many multiples of THRESHOLD are required to fire a step that moves |level| further
from zero. Moving toward zero always uses the normal THRESHOLD.
"""
import sys
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick
from native_quinary_charlm import THRESHOLD
from lutgtlm_ingest import _line_batches, enc


class SignAwareBrick(FixedStepBrick):
    LEVEL_MIN = -2
    LEVEL_MAX = 2
    AWAY_MULT = 1.0  # 1.0 reproduces production exactly (sanity check)

    def _residual_update_level(self, name, level, grad):
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize(name, mom_int)
        res = self._res.get(name, np.zeros_like(level, dtype=np.int64))
        res = res + scaled

        lvl = level.astype(np.int64)
        step_dir = -np.sign(res)  # direction this step would move the level, our convention
        # "moving away from zero": the step direction matches (or extends) the current level's own
        # sign, or the level is currently 0 (any first commitment counts as "away").
        moving_away = ((np.sign(lvl) == step_dir) & (lvl != 0)) | (lvl == 0)
        eff_threshold = np.where(moving_away, THRESHOLD * self.AWAY_MULT, THRESHOLD).astype(np.int64)

        step = np.where(np.abs(res) >= eff_threshold, np.sign(res), 0)
        desired = lvl - step
        new_level = np.clip(desired, self.LEVEL_MIN, self.LEVEL_MAX).astype(np.int8)
        actual_step = lvl - new_level.astype(np.int64)
        # consume at the effective threshold actually required for the direction that fired
        res = res - actual_step * eff_threshold
        self._res[name] = res
        return new_level


class TernaryFixedStepBrick(FixedStepBrick):
    LEVEL_MIN = -1
    LEVEL_MAX = 1
    def _residual_update_level(self, name, level, grad):
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize(name, mom_int)
        res = self._res.get(name, np.zeros_like(level, dtype=np.int64))
        res = res + scaled
        step = np.where(np.abs(res) >= THRESHOLD, np.sign(res), 0)
        desired = level.astype(np.int64) - step
        new_level = np.clip(desired, self.LEVEL_MIN, self.LEVEL_MAX).astype(np.int8)
        actual_step = level.astype(np.int64) - new_level.astype(np.int64)
        res = res - actual_step * THRESHOLD
        self._res[name] = res
        return new_level


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
print(f"random baseline: {RANDOM_BASELINE:.3f} nats\n")

configs = {
    "Quinary-NoFix       ": 1.0,
    "Quinary-SignAware x2": 2.0,
    "Quinary-SignAware x4": 4.0,
    "Quinary-SignAware x8": 8.0,
}
bricks = {name: SignAwareBrick(V=256, D=40, seed=0) for name in configs}
for name, mult in configs.items():
    bricks[name].AWAY_MULT = mult
ternary = TernaryFixedStepBrick(V=256, D=40, seed=0)

header = f"{'rep':>4} | " + " | ".join(configs.keys()) + " | Ternary(ref)"
print(header)
for rep in range(1, 11):
    row = [f"{rep:>4}"]
    for name, brick in bricks.items():
        teach(brick, FACT)
        ce = teacher_forced_ce(brick, FACT)
        row.append(f"{ce:>8.3f}" + " " * (len(name) - 8))
    teach(ternary, FACT)
    row.append(f"{teacher_forced_ce(ternary, FACT):>8.3f}")
    print(" | ".join(row))
