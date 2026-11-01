"""Coherent-growth / correction-by-addition test, directly testing the user's hypothesis after the
multi-fact curriculum negative result (lucy_west_galloping_multifact_curriculum_negative_20261101.md):
the 8-country test forced UNRELATED facts into one brick with no router -- exactly the condition the
real architecture's similarity-based routing (BrickCluster.route()) exists to prevent. This test uses
ONE growing, thematically COHERENT passage instead (all about the same subject), extended stage by
stage the way real incremental memory grows -- nothing is ever deleted, each stage adds more text on
top of the last, matching this project's residual-carry/"can't delete, only add" design and the
tea-patina imperfect-memory philosophy (lucy_west_tea_patina_imperfect_memory_philosophy_20261029.md).

The sharpest sub-test: stage 5 ends on "...He likes English muffins." and stage 6 appends
"English muffins are not actually English." -- a real correction added, not substituted. Question:
does the trained brick's generation actually pick up and reflect the correction, or does the earlier
"breakfast list" framing dominate and drown it out?

Uses the real validated-subset ground-truth mechanism (validated_wemb, the one proven to produce
exact correct generation with zero float shadow weight) -- galloping is not used here since the
prior test showed it doesn't help past a brick's first-ever fact, and this is one fact (one growing
passage), not a multi-fact curriculum.
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
RANDOM_BASELINE = np.log(256)
PROMPT = "User: Tell me about Jonathan.\nLucy: "
ANSWER_START = len(enc(PROMPT))
REPS_PER_STAGE = 4
VALIDATED_TOP_N = 1500  # capped for wall-clock; this is the proven ground-truth mechanism, just budgeted

s1 = "Jonathan was born in England."
s2 = s1 + " He was born in Liverpool."
s3 = s2 + " He likes English breakfasts."
s4 = s3 + " English breakfasts are made with English scrambled eggs, English crepes, and pig in a hole."
s5 = s4 + " He likes English muffins."
s6 = s5 + " English muffins are not actually English."
STAGES = [s1, s2, s3, s4, s5, s6]

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


class StaggeredFlipBrickV6(FixedStepBrick):
    POLICY = "validated_wemb"
    VALIDATED_TOP_N = VALIDATED_TOP_N

    def learn_batch(self, idxs_batch, lr_scale=1.0, corrob_eligible=True, fact_line=None):
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
        cur_loss = [answer_only_nll(self, fact_line)]

        for name, ij, mag, step_val in candidates:
            old_val = trial_wemb[ij] if name == "Wemb" else trial_levels[name][ij]
            apply_entry(name, ij, step_val)
            new_loss = answer_only_nll(self, fact_line)
            if new_loss <= cur_loss[0] + 1e-9:
                commit_set.add((name, ij))
                cur_loss[0] = new_loss
            else:
                revert_entry(name, ij, old_val)

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
        brick.learn_batch(b, lr_scale=1.0, corrob_eligible=False, fact_line=line)


print(f"random baseline: {RANDOM_BASELINE:.3f} nats")
print(f"{len(STAGES)} cumulative stages, {REPS_PER_STAGE} reps each, VALIDATED_TOP_N={VALIDATED_TOP_N}\n")

brick = StaggeredFlipBrickV6(V=256, D=40, seed=0)
seed_ids = enc(PROMPT)

total_fwd = 0
for i, stage_text in enumerate(STAGES, 1):
    line = PROMPT + stage_text
    print(f"--- stage {i}: target = {stage_text!r}")
    for rep in range(1, REPS_PER_STAGE + 1):
        FWD_PASS_COUNT[0] = 0
        teach(brick, line)
        ce = answer_only_nll(brick, line)
        fwd = FWD_PASS_COUNT[0]
        total_fwd += fwd
        print(f"    rep {rep} | NLL {ce:.4f} | fwd {fwd}")
    out = bytes_to_text(brick.generate(seed_ids, ITOS, n=max(60, len(stage_text) + 20), temp=0.01, seed=0))
    print(f"    [gen] {out!r}")

print(f"\nTOTAL forward passes over {len(STAGES)} stages x {REPS_PER_STAGE} reps: {total_fwd}")

print("\n--- FINAL CHECK: full final passage NLL + generation ---")
final_line = PROMPT + STAGES[-1]
print(f"final NLL: {answer_only_nll(brick, final_line):.4f}")
out = bytes_to_text(brick.generate(seed_ids, ITOS, n=len(STAGES[-1]) + 30, temp=0.01, seed=0))
print(f"[final gen] {out!r}")
