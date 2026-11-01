"""Push the validated-subset result (lucy_west_validated_subset_best_result_20261031.md) further on
its own three open items:
  1. Push past rep 40 to see if the still-falling NLL trend ever crosses into exact correct text.
  2. Extend the real-loss validated-subset check to Wemb (previously exempt -- went through the
     normal unconditional _residual_update_wide every call, no matter what the other 5 matrices
     decided).
  3. Raise VALIDATED_TOP_N above 150 to check whether candidates outside that window matter.

Wemb candidates are computed with the exact same elementwise mechanics _residual_update_wide uses
(_integer_ema + _adaptive_normalize_v2 -> res -> step = sign(res)*(|res|//THRESHOLD), variable
magnitude, never the FixedStepBrick +-1 override, which only overrides _residual_update_LEVEL).
They are merged into ONE ranked candidate pool together with the 5 matrices' +-1 flips, so the
validated-greedy budget (VALIDATED_TOP_N) is spent on whichever candidates matter most, not split
by an arbitrary per-matrix quota. Decay+clip (the two deliberate, already-tracked lossy mechanisms
in _residual_update_wide) are applied once, at commit time, to the whole matrix exactly as
production does every call -- these are elementwise and order-independent, so doing them after
instead of during the greedy loop changes nothing about correctness.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick
from native_quinary_charlm import (THRESHOLD, EQ, MAX_EMB, EMB_DECAY_DIV, level_to_planes,
                                    activation_quant_int8, simd_mm, sig_stable, DH_CARRY_CLIP,
                                    ACCUM_GRAD_CLIP)
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


class StaggeredFlipBrickV2(FixedStepBrick):
    POLICY = None          # None, "cap", "validated" (matrices only), "validated_wemb" (+ Wemb)
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

        proposals = {}  # name -> (level_before, res_new, step)   step is +-1 only (level matrices)
        for name in MATRICES:
            level = getattr(self, name)
            mom_int = self._integer_ema(name, scale_int(grads[name]))
            scaled = self._adaptive_normalize(name, mom_int)
            res_prev = self._res.get(name, np.zeros_like(level, dtype=np.int64))
            res_new = res_prev + scaled
            self._total_in[name] = self._total_in.get(name, 0) + int(scaled.sum())
            step = np.where(np.abs(res_new) >= THRESHOLD, np.sign(res_new), 0)
            proposals[name] = (level.copy(), res_new, step)

        wemb_candidates_included = self.POLICY == "validated_wemb"
        wemb_prop = None
        if wemb_candidates_included:
            wemb_grad_int = scale_int(gWemb * EQ)
            wmom_int = self._integer_ema("Wemb", wemb_grad_int)
            wscaled = self._adaptive_normalize_v2("Wemb", wmom_int)
            wres_prev = self._res.get("Wemb", np.zeros_like(self.Wemb, dtype=np.int64))
            wres_new = wres_prev + wscaled
            self._total_in["Wemb"] = self._total_in.get("Wemb", 0) + int(wscaled.sum())
            wstep = np.sign(wres_new) * (np.abs(wres_new) // THRESHOLD)  # variable magnitude
            wemb_prop = (self.Wemb.copy(), wres_new, wstep)

        candidates = []  # (name, ij, magnitude, step_val)
        for name, (level, res_new, step) in proposals.items():
            for ij in np.argwhere(step != 0):
                ij = tuple(ij)
                candidates.append((name, ij, abs(int(res_new[ij])), int(step[ij])))
        if wemb_candidates_included:
            level, res_new, step = wemb_prop
            for ij in np.argwhere(step != 0):
                ij = tuple(ij)
                candidates.append(("Wemb", ij, abs(int(res_new[ij])), int(step[ij])))
        candidates.sort(key=lambda c: -c[2])

        if self.POLICY == "cap":
            k = len(candidates) if self.MAX_FLIPS_PER_CALL is None else self.MAX_FLIPS_PER_CALL
            commit_set = set((n, ij) for n, ij, _, _ in candidates[:k])
        elif self.POLICY in ("validated", "validated_wemb"):
            commit_set = set()
            eval_pool = candidates[:self.VALIDATED_TOP_N]
            trial_levels = {name: proposals[name][0].copy() for name in MATRICES}
            trial_wemb = wemb_prop[0].copy() if wemb_candidates_included else None
            base_loss = answer_only_nll(self, FACT)  # weights still at pre-update levels here
            cur_loss = base_loss
            for name, ij, mag, step_val in eval_pool:
                if name == "Wemb":
                    old_val = int(trial_wemb[ij])
                    trial_wemb[ij] = old_val - step_val  # full variable-magnitude step, no clip yet
                    self.Wemb = trial_wemb
                else:
                    old_val = trial_levels[name][ij]
                    trial_levels[name][ij] = int(np.clip(old_val - step_val, -2, 2))
                    for m in MATRICES:
                        setattr(self, m, trial_levels[m])
                new_loss = answer_only_nll(self, FACT)
                if new_loss <= cur_loss + 1e-9:
                    commit_set.add((name, ij))
                    cur_loss = new_loss
                else:
                    if name == "Wemb":
                        trial_wemb[ij] = old_val
                        self.Wemb = trial_wemb
                    else:
                        trial_levels[name][ij] = old_val
                        for m in MATRICES:
                            setattr(self, m, trial_levels[m])
            for name in MATRICES:
                setattr(self, name, proposals[name][0])  # restore REAL levels before committing below
            if wemb_candidates_included:
                self.Wemb = wemb_prop[0]  # restore REAL Wemb before committing below
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

        if wemb_candidates_included:
            level, res_new, step = wemb_prop
            mask = np.zeros_like(step, dtype=bool)
            for n2, ij in commit_set:
                if n2 == "Wemb":
                    mask[ij] = True
            realized_step = np.where(mask, step, 0)
            W_mid = level.astype(np.int64) - realized_step
            decay = np.sign(W_mid) * (np.abs(W_mid) // EMB_DECAY_DIV)
            W_new = W_mid - decay
            clipped = np.clip(W_new, -MAX_EMB, MAX_EMB)
            discarded = int(decay.sum()) + int((W_new - clipped).sum())
            self._total_discarded = getattr(self, "_total_discarded", {})
            self._total_discarded["Wemb"] = self._total_discarded.get("Wemb", 0) + discarded
            self._total_consumed["Wemb"] = self._total_consumed.get("Wemb", 0) + int((realized_step * THRESHOLD).sum())
            self._res["Wemb"] = res_new - realized_step * THRESHOLD
            self.Wemb = clipped
        else:
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
    "Validated (matrices only, N=150)": ("validated", None, 150),
    "Validated+Wemb N=150            ": ("validated_wemb", None, 150),
    "Validated+Wemb N=400            ": ("validated_wemb", None, 400),
}
bricks = {}
for name, (policy, k, topn) in configs.items():
    b = StaggeredFlipBrickV2(V=256, D=40, seed=0)
    b.POLICY = policy
    b.VALIDATED_TOP_N = topn
    bricks[name] = b

N_REPS = 60
PRINT_EVERY = 5
seed_ids = enc(PROMPT_PREFIX)
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
    if rep % PRINT_EVERY == 0 or rep == N_REPS:
        for name, brick in bricks.items():
            out = bytes_to_text(brick.generate(seed_ids, ITOS, n=60, temp=0.01, seed=0))
            print(f"    [rep {rep:>3} greedy] [{name.strip()}] {out!r}")
