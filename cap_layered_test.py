"""Tests Bonin's anti-avalanche CAP (on the input drive, before accumulation) layered with the two
things that already helped on their own: ternary range restriction, and real QAR (signal-amplified
stochastic routing from arXiv:2609.21834). NOT layered with reset (confirmed a negative combo last
test -- reset swamps cap's distinct contribution). 8 conditions: {quinary, ternary} x {nofix, cap,
qar16, cap+qar16}.
"""
import sys
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick
from native_quinary_charlm import THRESHOLD
from lutgtlm_ingest import _line_batches, enc


class FlexBrick(FixedStepBrick):
    LEVEL_MIN = -2
    LEVEL_MAX = 2
    CAP = None     # units of THRESHOLD; None = no cap
    GAMMA = 0.0    # QAR power-amplifier exponent; 0 = deterministic (no QAR)

    def __init__(self, *a, rng_seed=12345, **kw):
        super().__init__(*a, **kw)
        self._rng = np.random.default_rng(rng_seed)

    def _residual_update_level(self, name, level, grad):
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize(name, mom_int)
        if self.CAP is not None:
            lim = int(self.CAP * THRESHOLD)
            scaled = np.clip(scaled, -lim, lim)
        res = self._res.get(name, np.zeros_like(level, dtype=np.int64))
        res = res + scaled

        if self.GAMMA > 0 or self.GAMMA == -1:  # GAMMA==-1 signals "use QAR machinery at gamma=0 (uniform stochastic)"
            a = np.abs(res).astype(np.float64) / THRESHOLD
            if self.GAMMA > 0 and a.ndim == 2:
                ag = a ** self.GAMMA
                row_mean = ag.mean(axis=1, keepdims=True)
                b = np.where(row_mean > 0, ag / np.where(row_mean > 0, row_mean, 1.0), 1.0)
            else:
                b = np.ones_like(a)
            p_flip = np.clip(a * b, 0.0, 1.0)
            roll = self._rng.random(size=level.shape)
            do_flip = (roll < p_flip) & (res != 0)
            step = np.where(do_flip, np.sign(res), 0).astype(np.int64)
        else:
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
    "Quinary-NoFix      ": dict(LMIN=-2, LMAX=2, CAP=None, GAMMA=0.0),
    "Quinary-Cap1.0     ": dict(LMIN=-2, LMAX=2, CAP=1.0,  GAMMA=0.0),
    "Quinary-QAR16      ": dict(LMIN=-2, LMAX=2, CAP=None, GAMMA=16.0),
    "Quinary-Cap+QAR16  ": dict(LMIN=-2, LMAX=2, CAP=1.0,  GAMMA=16.0),
    "Ternary-NoFix      ": dict(LMIN=-1, LMAX=1, CAP=None, GAMMA=0.0),
    "Ternary-Cap1.0     ": dict(LMIN=-1, LMAX=1, CAP=1.0,  GAMMA=0.0),
    "Ternary-QAR16      ": dict(LMIN=-1, LMAX=1, CAP=None, GAMMA=16.0),
    "Ternary-Cap+QAR16  ": dict(LMIN=-1, LMAX=1, CAP=1.0,  GAMMA=16.0),
}
bricks = {}
for name, cfg in configs.items():
    b = FlexBrick(V=256, D=40, seed=0)
    b.LEVEL_MIN, b.LEVEL_MAX, b.CAP, b.GAMMA = cfg["LMIN"], cfg["LMAX"], cfg["CAP"], cfg["GAMMA"]
    bricks[name] = b

header = f"{'rep':>4} | " + " | ".join(configs.keys())
print(header)
for rep in range(1, 11):
    row = [f"{rep:>4}"]
    for name, brick in bricks.items():
        teach(brick, FACT)
        ce = teacher_forced_ce(brick, FACT)
        row.append(f"{ce:>8.3f}" + " " * (len(name) - 8))
    print(" | ".join(row))
