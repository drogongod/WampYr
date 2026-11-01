"""Cross-category extension of the real-router test (lucy_west_real_router_prevents_collapse_20261101.md).
That test showed the real, unmodified BrickCluster router auto-separates 8 unrelated capital-facts
into 3 bricks with no catastrophic forgetting. This test adds a SECOND, thematically distinct cluster
of content -- the Jonathan/English-breakfast growing passage from v6 (staggered_flip_test_v6_coherent_growth.py)
-- through the SAME real router, to test genuine cross-category separation: does a completely different
subject (a growing biographical passage, not a fact-lookup Q&A) get routed away from the capital
facts, and does teaching it cause any measurable interference with facts already learned?

Caveat going in, Jonathan's own words: "that was a quick example of principle, I don't know if it's
actually the best to use to represent a bit-level model." Noted -- this is a convenience probe using
material already built for v6, not a claim that this is the ideal or most representative test case.
Treat the result as one more data point on the same question, not a definitive final word.

REPS_PER_FACT=1 and REPS_PER_STAGE=1 throughout: matches the fix found in v7 (repeating an IDENTICAL
line destabilizes the real router's TF-IDF document-frequency bookkeeping, which production never
actually does -- it filters exact duplicates before learn_line()). The Jonathan passage's stages are
not identical repeats of each other (each is strictly longer), so this isn't the same failure mode,
but one-shot teaching is also simply the project's actual target, so it's kept as the default here too.
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
REPS_PER_FACT = 1

JON_PROMPT = "User: Tell me about Jonathan.\nLucy: "
j1 = "Jonathan was born in England."
j2 = j1 + " He was born in Liverpool."
j3 = j2 + " He likes English breakfasts."
j4 = j3 + " English breakfasts are made with English scrambled eggs, English crepes, and pig in a hole."
j5 = j4 + " He likes English muffins."
j6 = j5 + " English muffins are not actually English."
JON_STAGES = [j1, j2, j3, j4, j5, j6]
REPS_PER_STAGE = 1


def make_fact_lines(country, capital):
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


class StaggeredFlipBrickV8(FixedStepBrick):
    """Identical mechanism to StaggeredFlipBrickV7 -- validated-subset, production learn_batch
    signature, answer-start found via byte marker inside idxs_batch itself. Duplicated rather than
    imported so this file stays self-contained and importing it never re-runs v7's own experiment."""
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
    def _spawn(self):
        idx = len(self.bricks)
        self.bricks.append(StaggeredFlipBrickV8(self.V, self.D_brick, seed=self.seed + 1000 * idx))
        self.brick_pos.append(_OnlineGaussian())
        self.pending_visited.append([])
        self.brick_taught_lines.append([])
        self._region_fresh = False
        return idx


def nll_and_gen(cluster, prompt, line, gen_n=40):
    features = cluster.tfidf_sparse(line)
    idx = cluster.route(features, for_training=False)
    brick = cluster.bricks[idx]
    ids = enc(line)
    nll = answer_only_nll_ids(brick, ids)
    out = bytes_to_text(brick.generate(enc(prompt), ITOS, n=gen_n, temp=0.01, seed=0))
    return idx, nll, out


print(f"random baseline: {RANDOM_BASELINE:.3f} nats")
print(f"{len(FACTS)} capital facts + {len(JON_STAGES)} Jonathan stages, through the REAL BrickCluster router\n")

cluster = ValidatedBrickCluster(V=256, D_brick=40, seed=0)

fact_lines = []
for country, capital in FACTS:
    prompt, line = make_fact_lines(country, capital)
    fact_lines.append((country, capital, prompt, line))

total_fwd = 0
print("=== phase 1: teach 8 capital facts (one-shot) ===")
print(f"{'fact':>8} | {'rep':>3} | {'brick':>5} | {'#bricks':>7} | {'loss':>8}")
for country, capital, prompt, line in fact_lines:
    batches = list(_line_batches(line))
    for rep in range(1, REPS_PER_FACT + 1):
        FWD_PASS_COUNT[0] = 0
        idx, mean_loss = cluster.learn_line(batches, line, lr_scale=1.0)
        fwd = FWD_PASS_COUNT[0]
        total_fwd += fwd
        print(f"{country:>8} | {rep:>3} | {idx:>5} | {len(cluster.bricks):>7} | {mean_loss:>8.4f} | fwd={fwd}")

print(f"\nbricks after phase 1: {len(cluster.bricks)}")

print("\n=== phase 2: teach 6 growing Jonathan stages (one-shot per stage) ===")
jon_lines = [JON_PROMPT + s for s in JON_STAGES]
print(f"{'stage':>5} | {'brick':>5} | {'#bricks':>7} | {'loss':>8}")
for i, (stage_text, line) in enumerate(zip(JON_STAGES, jon_lines), 1):
    batches = list(_line_batches(line))
    for rep in range(1, REPS_PER_STAGE + 1):
        FWD_PASS_COUNT[0] = 0
        idx, mean_loss = cluster.learn_line(batches, line, lr_scale=1.0)
        fwd = FWD_PASS_COUNT[0]
        total_fwd += fwd
        print(f"{i:>5} | {idx:>5} | {len(cluster.bricks):>7} | {mean_loss:>8.4f} | fwd={fwd}")

print(f"\nTOTAL forward passes: {total_fwd}")
print(f"Total bricks spawned: {len(cluster.bricks)}")

print("\n--- RETENTION CHECK: capital facts, real inference routing, no further training ---")
fact_bricks = []
for country, capital, prompt, line in fact_lines:
    idx, nll, out = nll_and_gen(cluster, prompt, line)
    fact_bricks.append(idx)
    print(f"{country:>8} | routed-to-brick={idx}  NLL={nll:.4f}")
    print(f"         [gen] {out!r}")

print("\n--- RETENTION CHECK: final Jonathan passage, real inference routing ---")
final_jon_line = jon_lines[-1]
jon_idx, jon_nll, jon_out = nll_and_gen(cluster, JON_PROMPT, final_jon_line, gen_n=len(JON_STAGES[-1]) + 30)
print(f"Jonathan(final) | routed-to-brick={jon_idx}  NLL={jon_nll:.4f}")
print(f"         [gen] {jon_out!r}")

print("\n--- SUMMARY: which bricks hold which category? ---")
print(f"capital-fact bricks used: {sorted(set(fact_bricks))}")
print(f"Jonathan-passage brick: {jon_idx}")
print(f"shared with any capital fact? {'YES' if jon_idx in set(fact_bricks) else 'no'}")
