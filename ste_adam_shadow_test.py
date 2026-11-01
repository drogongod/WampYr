"""THE UNEXPLORED ANGLE: a float32-shadow + real Adam + STE variant ALREADY EXISTS in this
codebase (D:\\flybrain\\quinary_simd_batched_charlm.py, QuinarySIMDBatchedCharLM) -- it was argued
away earlier ("BitNet/Zhao/Xu prove a float shadow isn't NECESSARY for quinary to work in
general") but never actually A/B'd against THIS specific failing case: the one-shot-fact-x10
diagnostic that every residual-carry-based fix (windup reset, avalanche cap, QAR, ternary range,
sign-aware, Wemb cap) failed to fix for real generation. "Not necessary in general" != "doesn't
help HERE." This is the real, direct test, same methodology discipline as every other test
tonight: real sampled generate() output is the verdict, not CE alone.

The class has no generate()/forward_logits() (training-step-only file) -- added here as a
subclass, NOT touching the original file, same as every other scratch test's convention of
leaving the engine's real class alone.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"D:\flybrain")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from quinary_simd_batched_charlm import QuinarySIMDBatchedCharLM, quinary_quant_planes, activation_quant_int8, simd_mm
from lutgtlm_ingest import _line_batches, enc, ITOS, bytes_to_text


class STEGenBrick(QuinarySIMDBatchedCharLM):
    def _quant_planes_current(self):
        S1_o, S2_o, ws_o = quinary_quant_planes(self.Wo)
        S1_wx, S2_wx, ws_wx = quinary_quant_planes(np.concatenate([self.Win, self.Wh], axis=1), use_max_scale=True)
        S1_wg, S2_wg, ws_wg = quinary_quant_planes(np.concatenate([self.Wgx, self.Wgh], axis=1), use_max_scale=True)
        return S1_o, S2_o, ws_o, S1_wx, S2_wx, ws_wx, S1_wg, S2_wg, ws_wg

    def forward_logits(self, idxs):
        D = self.D
        S1_o, S2_o, ws_o, S1_wx, S2_wx, ws_wx, S1_wg, S2_wg, ws_wg = self._quant_planes_current()
        h = np.zeros((D, 1), np.float32)
        T = len(idxs)
        logits = np.empty((T, self.V), np.float32)
        for t in range(T):
            x = self.Wemb[:, [idxs[t]]]
            xh = np.concatenate([x, h], axis=0)
            xhq, xhs = activation_quant_int8(xh)
            pc = simd_mm(S1_wx, S2_wx, ws_wx, xhq, xhs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_wg, S2_wg, ws_wg, xhq, xhs) + self.bg[:, None]
            g = 1.0 / (1.0 + np.exp(-pg))
            h = g * c + (1 - g) * h
            hq, hs = activation_quant_int8(h)
            logits[t] = simd_mm(S1_o, S2_o, ws_o, hq, hs).flatten()
        return logits

    def generate(self, seed_idx, itos, n=60, temp=0.7, seed=0):
        rng = np.random.default_rng(seed)
        D = self.D
        S1_o, S2_o, ws_o, S1_wx, S2_wx, ws_wx, S1_wg, S2_wg, ws_wg = self._quant_planes_current()
        h = np.zeros((D, 1), np.float32)
        out = list(seed_idx)

        def step(ix, h):
            x = self.Wemb[:, [ix]]
            xh = np.concatenate([x, h], axis=0)
            xhq, xhs = activation_quant_int8(xh)
            pc = simd_mm(S1_wx, S2_wx, ws_wx, xhq, xhs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_wg, S2_wg, ws_wg, xhq, xhs) + self.bg[:, None]
            g = 1.0 / (1.0 + np.exp(-pg))
            return g * c + (1 - g) * h

        for ix in seed_idx[:-1]:
            h = step(ix, h)
        ix = seed_idx[-1]
        for _ in range(n):
            h = step(ix, h)
            hq, hs = activation_quant_int8(h)
            logit = simd_mm(S1_o, S2_o, ws_o, hq, hs).flatten()
            p = np.exp((logit - logit.max()) / max(1e-3, temp)); p /= p.sum()
            ix = int(rng.choice(len(p), p=p)); out.append(ix)
            if itos[ix] == "\n":
                break
        return "".join(itos[i] for i in out)


def teacher_forced_ce(brick, line):
    ids = enc(line)
    logits = brick.forward_logits(ids[:-1]).astype(np.float64)
    targets = np.array(ids[1:])
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return float(-logp[np.arange(len(targets)), targets].mean())


def teach(brick, line, lr):
    for b in _line_batches(line):
        brick.learn_batch(b, lr=lr)


RANDOM_BASELINE = np.log(256)
FACT = "User: What is the capital of France?\nLucy: The capital of France is Paris."
PROMPT = "User: What is the capital of France?\nLucy: "

print(f"random baseline: {RANDOM_BASELINE:.3f} nats\n")

lrs = {"lr=0.003 (file default)": 0.003, "lr=0.01": 0.01, "lr=0.03": 0.03}
bricks = {name: STEGenBrick(V=256, D=40, seed=0) for name in lrs}

print("=== Part 1: teacher-forced CE per rep, per lr (same D=40, same one-shot-fact-x10 diagnostic) ===")
header = f"{'rep':>4} | " + " | ".join(lrs.keys())
print(header)
for rep in range(1, 11):
    row = [f"{rep:>4}"]
    for name, lr in lrs.items():
        teach(bricks[name], FACT, lr)
        ce = teacher_forced_ce(bricks[name], FACT)
        row.append(f"{ce:>22.3f}")
    print(" | ".join(row))

print("\n=== Part 2: THE REAL TEST -- actual generated text after 10 reps, same prompt, each lr ===")
print(f"Prompt: {PROMPT!r}\n")
seed_ids = enc(PROMPT)
for name, brick in bricks.items():
    out = brick.generate(seed_ids, ITOS, n=60, temp=0.7, seed=0)
    text = bytes_to_text(out)
    print(f"[{name}]")
    print(f"  -> {text!r}\n")
