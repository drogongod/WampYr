"""Cheaper candidate-selection for the validated-subset mechanism
(lucy_west_validated_subset_exact_generation_breakthrough_20261101.md). The UNCAPPED version that
achieves the 0.002-0.004 nat floor real-loss-checks every single candidate flip (up to ~5351/call)
with its OWN full forward pass -- the real cost driver for scaling to a real training set, not raw
FLOPs (this net is tiny, D=40) but sheer Python/numpy call COUNT.

ADAPTIVE BATCH-DOUBLING WITH BISECTION ("validated_binary"): candidates are still ranked and
processed in the same order, but instead of testing one at a time, a whole contiguous RANGE is
tentatively committed and checked with ONE forward pass. If the range's net effect doesn't increase
loss, the WHOLE range is accepted in that one pass (this is where the savings come from -- most of
a sorted candidate list, once the biggest disruptive flips are past, nets out fine in large groups).
If a range's net effect DOES increase loss, it is recursively bisected into two halves, each
re-tested the same way, down to individual candidates when a range of size 1 still fails (which
is then simply rejected/deferred, exactly as the original one-at-a-time rule would).

HONEST SCOPE: this is NOT provably bit-identical to strict sequential one-at-a-time testing -- a
large accepted batch can let through an individually-marginal flip that one-at-a-time would have
rejected at that exact moment, as long as the GROUP's net effect still doesn't increase loss. This
is a real, different greedy variant, not a cached/sped-up recomputation of the same algorithm, and
is benchmarked here against the real uncapped ground truth to see whether it costs any measured
quality, not assumed to be free.
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

FWD_PASS_COUNT = [0]


def loss_per_token(brick, line):
    FWD_PASS_COUNT[0] += 1
    ids = enc(line)
    logits = brick.forward_logits(ids[:-1])
    targets = np.array(ids[1:])
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return -logp[np.arange(len(targets)), targets]


def answer_only_nll(brick, line):
    nll = loss_per_token(brick, line)
    return float(nll[max(0, ANSWER_START - 1):].mean())


class StaggeredFlipBrickV4(FixedStepBrick):
    POLICY = None          # "validated_wemb" (uncapped ground truth), "validated_binary", "validated_gallop"
    VALIDATED_TOP_N = None  # None = uncapped
    MIN_BATCH = 1           # bisection floor for validated_binary
    MAX_BATCH = 256         # growth ceiling for validated_gallop

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

        proposals = {}
        for name in MATRICES:
            level = getattr(self, name)
            mom_int = self._integer_ema(name, scale_int(grads[name]))
            scaled = self._adaptive_normalize(name, mom_int)
            res_prev = self._res.get(name, np.zeros_like(level, dtype=np.int64))
            res_new = res_prev + scaled
            self._total_in[name] = self._total_in.get(name, 0) + int(scaled.sum())
            step = np.where(np.abs(res_new) >= THRESHOLD, np.sign(res_new), 0)
            proposals[name] = (level.copy(), res_new, step)

        wemb_grad_int = scale_int(gWemb * EQ)
        wmom_int = self._integer_ema("Wemb", wemb_grad_int)
        wscaled = self._adaptive_normalize_v2("Wemb", wmom_int)
        wres_prev = self._res.get("Wemb", np.zeros_like(self.Wemb, dtype=np.int64))
        wres_new = wres_prev + wscaled
        self._total_in["Wemb"] = self._total_in.get("Wemb", 0) + int(wscaled.sum())
        wstep = np.sign(wres_new) * (np.abs(wres_new) // THRESHOLD)
        wemb_prop = (self.Wemb.copy(), wres_new, wstep)

        candidates = []
        for name, (level, res_new, step) in proposals.items():
            for ij in np.argwhere(step != 0):
                ij = tuple(ij)
                candidates.append((name, ij, abs(int(res_new[ij])), int(step[ij])))
        level, res_new, step = wemb_prop
        for ij in np.argwhere(step != 0):
            ij = tuple(ij)
            candidates.append(("Wemb", ij, abs(int(res_new[ij])), int(step[ij])))
        candidates.sort(key=lambda c: -c[2])
        if self.VALIDATED_TOP_N is not None:
            candidates = candidates[:self.VALIDATED_TOP_N]

        trial_levels = {name: proposals[name][0].copy() for name in MATRICES}
        trial_wemb = wemb_prop[0].copy()

        def apply_entry(name, ij, step_val):
            if name == "Wemb":
                trial_wemb[ij] -= step_val
                self.Wemb = trial_wemb
            else:
                trial_levels[name][ij] = int(np.clip(trial_levels[name][ij] - step_val, -2, 2))
                for m in MATRICES:
                    setattr(self, m, trial_levels[m])

        def revert_entry(name, ij, old_val):
            if name == "Wemb":
                trial_wemb[ij] = old_val
                self.Wemb = trial_wemb
            else:
                trial_levels[name][ij] = old_val
                for m in MATRICES:
                    setattr(self, m, trial_levels[m])

        commit_set = set()
        cur_loss = [answer_only_nll(self, FACT)]  # mutable cell, baseline at pre-update weights

        if self.POLICY == "validated_binary":
            def validate_range(lo, hi):
                if lo >= hi:
                    return
                old_vals = []
                for (name, ij, mag, step_val) in candidates[lo:hi]:
                    old_vals.append((name, ij, (trial_wemb[ij] if name == "Wemb" else trial_levels[name][ij])))
                    apply_entry(name, ij, step_val)
                new_loss = answer_only_nll(self, FACT)
                if new_loss <= cur_loss[0] + 1e-9:
                    for (name, ij, mag, step_val) in candidates[lo:hi]:
                        commit_set.add((name, ij))
                    cur_loss[0] = new_loss
                    return
                for (name, ij, old_val) in old_vals:
                    revert_entry(name, ij, old_val)
                if hi - lo <= self.MIN_BATCH:
                    return  # single-candidate range, confirmed net-harmful at this point: reject
                mid = (lo + hi) // 2
                validate_range(lo, mid)
                validate_range(mid, hi)

            validate_range(0, len(candidates))
        elif self.POLICY == "validated_gallop":
            idx = 0
            batch_size = 1
            N = len(candidates)
            while idx < N:
                hi = min(idx + batch_size, N)
                old_vals = []
                for (name, ij, mag, step_val) in candidates[idx:hi]:
                    old_vals.append((name, ij, (trial_wemb[ij] if name == "Wemb" else trial_levels[name][ij])))
                    apply_entry(name, ij, step_val)
                new_loss = answer_only_nll(self, FACT)
                if new_loss <= cur_loss[0] + 1e-9:
                    for (name, ij, mag, step_val) in candidates[idx:hi]:
                        commit_set.add((name, ij))
                    cur_loss[0] = new_loss
                    idx = hi
                    batch_size = min(batch_size * 2, self.MAX_BATCH)  # grow on success
                else:
                    for (name, ij, old_val) in old_vals:
                        revert_entry(name, ij, old_val)
                    if batch_size == 1:
                        idx += 1  # confirmed-bad single candidate, same cost as brute force: reject
                    else:
                        batch_size = max(1, batch_size // 2)  # shrink and retry from the SAME idx
        elif self.POLICY == "validated_wemb":
            for name, ij, mag, step_val in candidates:
                old_val = trial_wemb[ij] if name == "Wemb" else trial_levels[name][ij]
                apply_entry(name, ij, step_val)
                new_loss = answer_only_nll(self, FACT)
                if new_loss <= cur_loss[0] + 1e-9:
                    commit_set.add((name, ij))
                    cur_loss[0] = new_loss
                else:
                    revert_entry(name, ij, old_val)
        else:
            raise ValueError(self.POLICY)

        for name in MATRICES:
            setattr(self, name, proposals[name][0])
        self.Wemb = wemb_prop[0]

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
        self._total_consumed["Wemb"] = self._total_consumed.get("Wemb", 0) + int((realized_step * THRESHOLD).sum())
        self._res["Wemb"] = res_new - realized_step * THRESHOLD
        self.Wemb = clipped

        new_b = self.b - lr_scale * 1e-4 * gb
        new_bg = self.bg - lr_scale * 1e-4 * gbg
        self.b, self._last_excess_b = self._clip_and_shed(new_b, 6.0)
        self.bg, self._last_excess_bg = self._clip_and_shed(new_bg, 6.0)
        return loss / max(1, n)


def teach(brick, line):
    for b in _line_batches(line):
        brick.learn_batch(b, lr_scale=1.0, corrob_eligible=False)


print(f"random baseline: {RANDOM_BASELINE:.3f} nats\n")

brick_ground = StaggeredFlipBrickV4(V=256, D=40, seed=0); brick_ground.POLICY = "validated_wemb"
brick_gallop = StaggeredFlipBrickV4(V=256, D=40, seed=0); brick_gallop.POLICY = "validated_gallop"
seed_ids = enc(PROMPT_PREFIX)

print(f"{'rep':>4} | {'NLL ground':>11} | {'fwd ground':>10} | {'NLL gallop':>11} | {'fwd gallop':>10} | speedup")
total_ground = total_gallop = 0
for rep in range(1, 31):
    FWD_PASS_COUNT[0] = 0
    teach(brick_ground, FACT)
    ce_g = answer_only_nll(brick_ground, FACT)
    fwd_g = FWD_PASS_COUNT[0]
    FWD_PASS_COUNT[0] = 0
    teach(brick_gallop, FACT)
    ce_b = answer_only_nll(brick_gallop, FACT)
    fwd_b = FWD_PASS_COUNT[0]
    total_ground += fwd_g; total_gallop += fwd_b
    speedup = fwd_g / max(1, fwd_b)
    print(f"{rep:>4} | {ce_g:>11.4f} | {fwd_g:>10} | {ce_b:>11.4f} | {fwd_b:>10} | {speedup:>6.1f}x")
    if rep % 5 == 0 or rep in (1, 2, 3):
        out_g = bytes_to_text(brick_ground.generate(seed_ids, ITOS, n=50, temp=0.01, seed=0))
        out_b = bytes_to_text(brick_gallop.generate(seed_ids, ITOS, n=50, temp=0.01, seed=0))
        print(f"      [ground ] {out_g!r}")
        print(f"      [gallop ] {out_b!r}")

print(f"\nTOTAL forward passes over 30 reps: ground={total_ground}  gallop={total_gallop}  "
      f"overall speedup={total_ground/max(1,total_gallop):.1f}x")
