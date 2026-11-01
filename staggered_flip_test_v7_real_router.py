"""Tests Jonathan's "enough resolution, not infinite categories" argument directly: the 8-country
curriculum test (lucy_west_galloping_multifact_curriculum_negative_20261101.md) forced 8 unrelated
facts into ONE brick with no router, which is a known, expected failure mode -- the real system has
BrickCluster.route() specifically to send different-enough content to different bricks and
similar-enough content to the same one (an online, similarity-threshold-based "category" system,
exactly the alphabet/nations analogy). That test never actually exercised the real router. This one
does: the EXACT SAME 8 facts, same reps, but through the real, UNMODIFIED BrickCluster machinery
(route/learn_line/relay/_spawn), with only _spawn() swapped to produce validated-subset-capable
bricks instead of plain FixedStepBrick.

Key design choice: the validated-subset brick's learn_batch keeps the EXACT signature production
code calls (idxs_batch, lr_scale, corrob_eligible) -- no extra line-text parameter -- by finding the
answer-start position directly inside idxs_batch itself (searching for the "Lucy: " byte marker that
every taught line shares), so it's a true drop-in _spawn() swap, not a parallel reimplementation of
learn_line/route/relay.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick, BrickCluster, _OnlineGaussian
from native_quinary_charlm import (THRESHOLD, EQ, MAX_EMB, EMB_DECAY_DIV, level_to_planes,
                                    activation_quant_int8, simd_mm, sig_stable, DH_CARRY_CLIP,
                                    ACCUM_GRAD_CLIP)
from lutgtlm_ingest import _line_batches, enc, ITOS, bytes_to_text

MATRICES = ["Win", "Wh", "Wgx", "Wgh", "Wo"]
RANDOM_BASELINE = np.log(256)
VALIDATED_TOP_N = 3000
MARKER = enc("Lucy: ")

FACTS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Japan", "Tokyo"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Brazil", "Brasilia"),
]
REPS_PER_FACT = 1  # true one-shot teaching, matching this project's actual north star (see
# LUCYWEST_INDEPENDENT_TECHNICAL_HANDOFF_20261031.md section 1: "single exposure is the target,
# not forty repeated presentations") -- also avoids an artifact found in a sanity check: repeating
# the IDENTICAL line through learn_line() repeatedly isn't a case the real router is designed for
# (production normally filters exact-duplicate lines before they ever reach learn_line at all;
# here, repeating a line raises its own trigrams' document frequency every call, shrinking its
# own TF-IDF/IDF-weighted feature vector toward zero and visibly destabilizing routing -- a
# confound from testing an unsupported input shape, not a finding about route() itself).


def make_lines(country, capital):
    prompt = f"User: What is the capital of {country}?\nLucy: "
    line = prompt + f"The capital of {country} is {capital}."
    return prompt, line


def find_answer_start(ids):
    ids = list(ids)
    m = len(MARKER)
    for i in range(len(ids) - m, -1, -1):
        if ids[i:i + m] == MARKER:
            return i + m
    return 0


FWD_PASS_COUNT = [0]


def loss_per_token_ids(brick, ids):
    FWD_PASS_COUNT[0] += 1
    logits = brick.forward_logits(ids[:-1])
    targets = np.array(ids[1:])
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return -logp[np.arange(len(targets)), targets]


def answer_only_nll_ids(brick, ids):
    nll = loss_per_token_ids(brick, ids)
    astart = find_answer_start(ids)
    return float(nll[max(0, astart - 1):].mean())


class StaggeredFlipBrickV7(FixedStepBrick):
    """Same validated-subset mechanism as v4/v5/v6, but learn_batch keeps the EXACT production
    signature (idxs_batch, lr_scale, corrob_eligible) -- validates against idxs_batch itself
    (assumes B==1, true for every single-line call BrickCluster.learn_line makes), not a separately
    passed fact string. This is what makes it a safe drop-in via _spawn() alone."""
    VALIDATED_TOP_N = VALIDATED_TOP_N

    def learn_batch(self, idxs_batch, lr_scale=1.0, corrob_eligible=True):
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
        ids_for_validation = idxs_batch[0]
        cur_loss = [answer_only_nll_ids(self, ids_for_validation)]

        for name, ij, mag, step_val in candidates:
            old_val = trial_wemb[ij] if name == "Wemb" else trial_levels[name][ij]
            apply_entry(name, ij, step_val)
            new_loss = answer_only_nll_ids(self, ids_for_validation)
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


class ValidatedBrickCluster(BrickCluster):
    """ONLY _spawn() is overridden -- route()/learn_line()/_relay_excess()/consolidate() are the
    REAL, unmodified production code. This is the whole point: test whether the real router's
    similarity-threshold categorization has enough resolution to keep unrelated facts apart,
    combined with the real per-brick fix (validated-subset), with zero other production logic
    touched."""
    def _spawn(self):
        idx = len(self.bricks)
        self.bricks.append(StaggeredFlipBrickV7(self.V, self.D_brick, seed=self.seed + 1000 * idx))
        self.brick_pos.append(_OnlineGaussian())
        self.pending_visited.append([])
        self.brick_taught_lines.append([])
        self._region_fresh = False
        return idx


def nll_and_gen_for_fact(cluster, prompt, line):
    features = cluster.tfidf_sparse(line)
    idx = cluster.route(features, for_training=False)
    brick = cluster.bricks[idx]
    ids = enc(line)
    nll = answer_only_nll_ids(brick, ids)
    out = bytes_to_text(brick.generate(enc(prompt), ITOS, n=40, temp=0.01, seed=0))
    return idx, nll, out


print(f"random baseline: {RANDOM_BASELINE:.3f} nats")
print(f"{len(FACTS)} facts, {REPS_PER_FACT} reps each, through the REAL BrickCluster router\n")

cluster = ValidatedBrickCluster(V=256, D_brick=40, seed=0)

fact_lines = []
for country, capital in FACTS:
    prompt, line = make_lines(country, capital)
    fact_lines.append((country, capital, prompt, line))

total_fwd = 0
print(f"{'fact':>8} | {'rep':>3} | {'brick':>5} | {'#bricks':>7} | {'loss':>8}")
for country, capital, prompt, line in fact_lines:
    batches = list(_line_batches(line))
    for rep in range(1, REPS_PER_FACT + 1):
        FWD_PASS_COUNT[0] = 0
        idx, mean_loss = cluster.learn_line(batches, line, lr_scale=1.0)
        fwd = FWD_PASS_COUNT[0]
        total_fwd += fwd
        print(f"{country:>8} | {rep:>3} | {idx:>5} | {len(cluster.bricks):>7} | {mean_loss:>8.4f} | fwd={fwd}")
    idx2, nll, out = nll_and_gen_for_fact(cluster, prompt, line)
    print(f"         routed-to-brick={idx2}  NLL={nll:.4f}")
    print(f"         [gen] {out!r}")

print(f"\nTOTAL forward passes over {len(FACTS)} facts x {REPS_PER_FACT} reps: {total_fwd}")
print(f"Total bricks spawned: {len(cluster.bricks)}")

print("\n--- RETENTION CHECK: re-test every fact once more, no further training, real inference routing ---")
for country, capital, prompt, line in fact_lines:
    idx, nll, out = nll_and_gen_for_fact(cluster, prompt, line)
    print(f"{country:>8} | routed-to-brick={idx}  NLL={nll:.4f}")
    print(f"         [gen] {out!r}")
