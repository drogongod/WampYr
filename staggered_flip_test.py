"""Two real fixes for the CONFIRMED avalanche mechanism (discrete_endpoint_audit.py: reps 1-4,
individually-correct flips destructively interact, interaction +2 to +7, destroying ~80-90% of
available benefit), compared head-to-head:

1. RATE-CAP / TIME-DELAY (Jonathan's question): commit only the top-K most-confident candidate
   flips this call (ranked by |accumulated residual|); the rest are left UNCONSUMED (still >=
   THRESHOLD) so they carry forward and are reconsidered next call -- a real, cheap, no-forward-
   pass-needed staggering mechanism. Reduces to the exact unmodified baseline when K=None.
2. VALIDATED SUBSET (Astra's independent review, section 6.2): rank the same candidates, then
   greedily accept/reject each by actually measuring real teacher-forced loss on a frozen snapshot
   -- commit only candidates that measurably help (or don't hurt), skip the rest (same deferral
   semantics as policy 1, not a full stop). Capped to the top VALIDATED_TOP_N candidates by
   magnitude for tractability (up to 5351 flips/call were measured; evaluating all would be
   enormously expensive).

Both policies reimplement ONLY the tail of the real learn_batch (the 5 matrix updates) -- the
forward/backward BPTT pass is copied verbatim from native_quinary_charlm.py, not reimplemented or
approximated, so this stays a faithful comparison, not a different algorithm.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick
from native_quinary_charlm import (THRESHOLD, EQ, level_to_planes, activation_quant_int8, simd_mm,
                                    sig_stable, DH_CARRY_CLIP, ACCUM_GRAD_CLIP)
from lutgtlm_ingest import _line_batches, enc, ITOS, bytes_to_text

MATRICES = ["Win", "Wh", "Wgx", "Wgh", "Wo"]
FACT = "User: What is the capital of France?\nLucy: The capital of France is Paris."
PROMPT_PREFIX = "User: What is the capital of France?\nLucy: "
ANSWER_START = len(enc(PROMPT_PREFIX))
RANDOM_BASELINE = np.log(256)


def loss_per_token(brick, line):
    ids = enc(line)
    logits = brick.forward_logits(ids[:-1])
    targets = np.array(ids[1:])
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return -logp[np.arange(len(targets)), targets]


def answer_only_nll(brick, line):
    nll = loss_per_token(brick, line)
    return float(nll[max(0, ANSWER_START - 1):].mean())


def full_line_nll(brick, line):
    return float(loss_per_token(brick, line).mean())


class StaggeredFlipBrick(FixedStepBrick):
    POLICY = None          # None = exact baseline, "cap" = rate-cap, "validated" = validated subset
    MAX_FLIPS_PER_CALL = None
    VALIDATED_TOP_N = 150

    def learn_batch(self, idxs_batch, lr_scale=1.0, corrob_eligible=True):
        if self.POLICY is None:
            return super().learn_batch(idxs_batch, lr_scale, corrob_eligible)

        self.consume_pending_relay()
        B = len(idxs_batch); T = len(idxs_batch[0])
        idxs_arr = np.array(idxs_batch, dtype=np.int64)
        D = self.D

        S1_in, S2_in = level_to_planes(self.Win); S1_h, S2_h = level_to_planes(self.Wh)
        S1_gx, S2_gx = level_to_planes(self.Wgx); S1_gh, S2_gh = level_to_planes(self.Wgh)
        S1_o, S2_o = level_to_planes(self.Wo)
        S1_inT, S2_inT = level_to_planes(self.Win.T); S1_hT, S2_hT = level_to_planes(self.Wh.T)
        S1_gxT, S2_gxT = level_to_planes(self.Wgx.T); S1_ghT, S2_ghT = level_to_planes(self.Wgh.T)
        S1_oT, S2_oT = level_to_planes(self.Wo.T)

        X = (self.Wemb[:, idxs_arr.T]).astype(np.float64) / EQ
        h = np.zeros((D, B), np.float64)
        H = []
        logits = np.empty((T, self.V, B), np.float64)
        for t in range(T):
            x_t = X[:, t, :]
            xq, xs = activation_quant_int8(x_t); hq, hs = activation_quant_int8(h)
            pc = simd_mm(S1_in, S2_in, xq, xs) + simd_mm(S1_h, S2_h, hq, hs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_gx, S2_gx, xq, xs) + simd_mm(S1_gh, S2_gh, hq, hs) + self.bg[:, None]
            g = sig_stable(pg)
            hn = g * c + (1 - g) * h
            H.append((x_t, h, c, g, hn)); h = hn
            hnq, hns = activation_quant_int8(hn)
            logits[t] = simd_mm(S1_o, S2_o, hnq, hns)

        gWemb = np.zeros_like(self.Wemb, dtype=np.float64)
        gWin = np.zeros((D, D)); gWh = np.zeros((D, D)); gb = np.zeros(D)
        gWgx = np.zeros((D, D)); gWgh = np.zeros((D, D)); gbg = np.zeros(D)
        gWo = np.zeros((self.V, D))
        dh = np.zeros((D, B)); loss = np.zeros(B); n = 0

        for t in range(T - 2, -1, -1):
            tgt = idxs_arr[:, t + 1]
            lg = logits[t]
            p = np.exp(lg - lg.max(axis=0, keepdims=True)); p /= p.sum(axis=0, keepdims=True)
            loss += -np.log(p[tgt, np.arange(B)] + 1e-9); n += 1
            dl = p.copy(); dl[tgt, np.arange(B)] -= 1.0
            x_t, hp, c, g, hn = H[t]
            gWo += dl @ hn.T
            dlq, dls = activation_quant_int8(dl)
            dh = np.clip(dh + simd_mm(S1_oT, S2_oT, dlq, dls), -DH_CARRY_CLIP, DH_CARRY_CLIP)
            dg = dh * (c - hp); dc = dh * g; dhp = dh * (1 - g)
            dpc = dc * (1 - c * c); dpg = dg * g * (1 - g)
            gWin += dpc @ x_t.T; gWh += dpc @ hp.T; gb += dpc.sum(axis=1)
            gWgx += dpg @ x_t.T; gWgh += dpg @ hp.T; gbg += dpg.sum(axis=1)
            dpcq, dpcs = activation_quant_int8(dpc); dpgq, dpgs = activation_quant_int8(dpg)
            dxemb = simd_mm(S1_inT, S2_inT, dpcq, dpcs) + simd_mm(S1_gxT, S2_gxT, dpgq, dpgs)
            np.add.at(gWemb, (slice(None), idxs_arr[:, t]), dxemb)
            dh = np.clip(dhp + simd_mm(S1_hT, S2_hT, dpcq, dpcs) + simd_mm(S1_ghT, S2_ghT, dpgq, dpgs),
                         -DH_CARRY_CLIP, DH_CARRY_CLIP)

        scale_int = lambda G: np.clip(np.round(G * lr_scale), -ACCUM_GRAD_CLIP, ACCUM_GRAD_CLIP).astype(np.int64)
        grads = {"Win": gWin, "Wh": gWh, "Wgx": gWgx, "Wgh": gWgh, "Wo": gWo}

        proposals = {}  # name -> (level_before, res_new, step)
        for name in MATRICES:
            level = getattr(self, name)
            mom_int = self._integer_ema(name, scale_int(grads[name]))
            scaled = self._adaptive_normalize(name, mom_int)
            res_prev = self._res.get(name, np.zeros_like(level, dtype=np.int64))
            res_new = res_prev + scaled
            self._total_in[name] = self._total_in.get(name, 0) + int(scaled.sum())
            step = np.where(np.abs(res_new) >= THRESHOLD, np.sign(res_new), 0)
            proposals[name] = (level.copy(), res_new, step)

        candidates = []  # (name, ij, magnitude)
        for name, (level, res_new, step) in proposals.items():
            for ij in np.argwhere(step != 0):
                ij = tuple(ij)
                candidates.append((name, ij, abs(int(res_new[ij]))))
        candidates.sort(key=lambda c: -c[2])

        if self.POLICY == "cap":
            k = len(candidates) if self.MAX_FLIPS_PER_CALL is None else self.MAX_FLIPS_PER_CALL
            commit_set = set((n, ij) for n, ij, _ in candidates[:k])
        elif self.POLICY == "validated":
            commit_set = set()
            eval_pool = candidates[:self.VALIDATED_TOP_N]
            trial_levels = {name: proposals[name][0].copy() for name in MATRICES}
            base_loss = answer_only_nll(self, FACT)  # weights still at pre-update levels here
            cur_loss = base_loss
            for name, ij, mag in eval_pool:
                step_val = proposals[name][2][ij]
                old_val = trial_levels[name][ij]
                trial_levels[name][ij] = int(np.clip(old_val - step_val, -2, 2))
                for m in MATRICES:
                    setattr(self, m, trial_levels[m])
                new_loss = answer_only_nll(self, FACT)
                if new_loss <= cur_loss + 1e-9:
                    commit_set.add((name, ij))
                    cur_loss = new_loss
                else:
                    trial_levels[name][ij] = old_val  # reject, restore trial copy
                    for m in MATRICES:
                        setattr(self, m, trial_levels[m])
            for name in MATRICES:
                setattr(self, name, proposals[name][0])  # restore REAL levels before committing below
        else:
            raise ValueError(self.POLICY)

        for name, (level, res_new, step) in proposals.items():
            mask = np.zeros_like(step, dtype=bool)
            for n2, ij in commit_set:
                if n2 == name:
                    mask[ij] = True
            actual_step = np.where(mask, step, 0)
            desired = level.astype(np.int64) - actual_step
            new_level = np.clip(desired, -2, 2).astype(np.int8)
            realized = level.astype(np.int64) - new_level.astype(np.int64)
            res_final = res_new - realized * THRESHOLD
            self._total_consumed[name] = self._total_consumed.get(name, 0) + int((realized * THRESHOLD).sum())
            self._res[name] = res_final
            setattr(self, name, new_level)

        self.Wemb = self._residual_update_wide("Wemb", self.Wemb, scale_int(gWemb * EQ))
        new_b = self.b - lr_scale * 1e-4 * gb
        new_bg = self.bg - lr_scale * 1e-4 * gbg
        self.b, self._last_excess_b = self._clip_and_shed(new_b, 6.0)
        self.bg, self._last_excess_bg = self._clip_and_shed(new_bg, 6.0)
        return loss / max(1, n)


def teach(brick, line):
    for b in _line_batches(line):
        brick.learn_batch(b, lr_scale=1.0, corrob_eligible=False)


print(f"random baseline: {RANDOM_BASELINE:.3f} nats\n")

configs = {
    "Baseline (no cap)   ": (None, None),
    "Cap K=10            ": ("cap", 10),
    "Cap K=30            ": ("cap", 30),
    "Cap K=60            ": ("cap", 60),
    "Validated subset    ": ("validated", None),
}
bricks = {}
for name, (policy, k) in configs.items():
    b = StaggeredFlipBrick(V=256, D=40, seed=0)
    b.POLICY = policy
    if policy == "cap":
        b.MAX_FLIPS_PER_CALL = k
    bricks[name] = b

N_REPS = 15
print("=== answer-only NLL per rep ===")
header = f"{'rep':>4} | " + " | ".join(configs.keys())
print(header)
for rep in range(1, N_REPS + 1):
    row = [f"{rep:>4}"]
    for name, brick in bricks.items():
        teach(brick, FACT)
        ce = answer_only_nll(brick, FACT)
        row.append(f"{ce:>8.3f}")
    print(" | ".join(row))

print("\n=== THE REAL TEST -- actual generated text after 15 reps ===")
seed_ids = enc(PROMPT_PREFIX)
for name, brick in bricks.items():
    for lbl, temp in (("t0.7", 0.7), ("greedy", 0.01)):
        out = bytes_to_text(brick.generate(seed_ids, ITOS, n=60, temp=temp, seed=0))
        print(f"[{name}] {lbl} -> {out!r}")
    print()
