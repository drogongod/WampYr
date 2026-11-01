# Handoff: Quinary weight-training oscillation bug — need fresh ideas

I'm pasting this into multiple AI models to get independent perspectives. Full context below so you don't need anything else. I have a strong suspicion we (me + a previous Claude session, weeks ago) already solved a version of this and I've lost the thread of what the fix was — if this rings a bell, say so.

## The architecture (so the problem makes sense)

A small recurrent character-level language model ("brick," ~40-dim hidden state, byte-level vocab of 256) where **every weight is a quinary integer level: one of exactly `{-2, -1, 0, 1, 2}`**, no float shadow weights anywhere. It's a GRU-style cell (input gate `c = tanh(...)`, update gate `g = sigmoid(...)`, `h' = g*c + (1-g)*h`), trained with full BPTT.

The weight-update rule (`_residual_update_level`), applied after each training call:
1. Compute the real-valued gradient for a weight matrix.
2. Run it through an integer EMA (momentum) and an L1-style adaptive per-weight normalizer (rescales so weights with naturally larger/smaller gradient magnitudes take a comparable number of calls to move).
3. Accumulate the normalized signal into a per-weight integer **residual** (`res`), carried exactly across calls — nothing is ever silently dropped sub-threshold.
4. Once `|res| >= THRESHOLD` (a fixed constant, currently 500), the weight's integer level takes exactly **one step of ±1**, clipped to `[-2, 2]`. The consumed amount (`±THRESHOLD`) is subtracted back out of `res`.

This exact mechanism (EMA + adaptive-normalize + deterministic threshold-crossing single-step) is a real, independently-known pattern — later found it's essentially "Bop" (Helwegen et al. 2019, "Latent weights do not exist").

Many small bricks like this exist in a cluster; a router sends each new fact/line to whichever brick's content it's most similar to (or spawns a new brick). Not relevant to the bug itself, just context.

## The bug

Lucy (the product this feeds) is taught facts one line at a time — "User: what is the capital of France?\nLucy: The capital of France is Paris." — often just **once**. When she's later asked about that fact, generation comes out as garbled bytes, not the taught sentence.

**Confirmed via teacher-forced evaluation (feeding the real correct context at every step, so autoregressive error-accumulation is ruled out):** the WEIGHTS themselves never encode the fact. Cross-entropy on the taught line, after a single clean training call: **7.98 nats** (random-uniform baseline over 256 bytes = **5.55 nats**). So even one clean update doesn't get anywhere close to fluent — it's barely better than noise.

**It gets WORSE with more repetition, not better.** Teaching the exact same line 10 more times:

```
rep:  1     2     3     4     5     6     7     8     9     10
CE:  8.52  8.83  9.42  10.67 12.63 13.95 14.21 14.93 13.10 11.37
```

It climbs for 8 reps, then partially recovers — genuine oscillation, not a stable plateau, not simple overfitting-then-convergence.

## What's been ruled out tonight (each tested directly, not just reasoned about)

1. **"Too few discrete levels is inherently unstable."** Disproven — BitNet b1.58 (ternary, `{-1,0,1}`, *fewer* levels than quinary) is proven to match FP32 LLM quality at scale. Discreteness-per-se is not the limit.

2. **"Needs a continuous float shadow weight to train well."** Disproven — two real, verified papers train low-bit weights directly with zero shadow weight and it works fine: Zhao et al., "Direct Quantized Training of Language Models with Stochastic Rounding" (arXiv:2412.04787), and Xu/Li/Ma/Cui, "Beyond Shadow Weights: Quantization-Aware Training as Quantized-Endpoint Descent" (arXiv:2609.21834, introduces "QAR").

3. **A real, separate, fixed bug**: a `-inf` sentinel in the routing/merging code made the very first 1-2 facts in any brick's life force-merge with whatever existed already regardless of topic. Fixed (replaced with a neutral `0.0`). Contributing cause, not sufficient on its own — the oscillation above was measured on a single isolated brick with this bug already fixed, so it's not the root cause of the main problem.

4. **Range matters, a lot, for THIS specific update rule**: a clean, controlled test — identical setup, identical fact, only difference is clipping the weight level to `[-1,1]` (ternary) instead of `[-2,2]` (quinary) — dropped the same 10-rep trajectory from the 8.5→14.9 climb above down to a stable **4.5–6.3 nats band**, briefly dipping *below* the random baseline. This is NOT a claim that "base-3 beats base-5" as number systems (math is base-independent, nothing physical is literally base-3 or base-5, it's binary hardware underneath either way) — it's specifically that **a weight can only move one level per call, so correcting a bad commitment at the ±2 extreme takes up to twice as many steps as at ±1**, and this naive rule has no sense of how close it already is to correct, so it overshoots. Wider available range = more room to overshoot before self-correction, for this one specific algorithm.

5. **Real QAR, implemented faithfully from the actual paper** (not a broken toy someone pasted me — caught that one separately, its custom autograd backward never actually fired due to a basic PyTorch mistake, so it silently trained nothing for 5 "successful-looking" epochs): replacing the deterministic threshold-crossing with a probability-based stochastic transition, where coordinates with locally stronger signal (within a weight's own row) get amplified routing priority (paper's "QAR-γ", γ=16 in their own best setting too) — this **substantially helped quinary** (roughly halved the peak, from ~16 down to ~10) without shrinking the range at all. Combined with the ternary range restriction, γ=16 gave the best result of the whole session: **ending at 3.77 nats, below random baseline, still trending down at rep 10.**

6. **Overflow "shed" (discard excess when a weight is pinned at the boundary and still being pushed)**: mixed — helps for a few reps, then ends up worse than doing nothing.

7. **Overflow "carry" (a second, paired weight that absorbs the excess once the first is pinned, added in — NOT multiplied by the base, pure addition, matching how a real carry-adder or finger-counting works, no "reconstruct the value" step)**: made it dramatically **worse** — 8.5 → 29.2 by rep 10. Mechanistic read: giving the already-overshooting weight *more room* (effective range now ±4 instead of ±2) just let it overshoot harder in the same direction, because the carry weight fills up in the same direction almost as fast as the primary did.

## The design philosophy constraint (please respect this, it's not optional)

No floating-point shadow weights, no reintroducing multiplication/"times tables" to reconstruct a combined value from parts (a real carry mechanism in hardware or in base-N finger-counting never multiplies to get the total — it's pure addition and threshold/overflow detection at each position, and reconstructing a human-readable total afterward is separate from how the arithmetic itself works). Fixes should stay inside that primitive set: integer addition, comparison, threshold-crossing, simple EMA/normalization. This project has an established, deliberate reason for all of this (speed, losslessness, closer to how real low-bit networks like BitNet actually train) — "more precision to be safe" is explicitly the wrong instinct here, already corrected multiple times historically.

Also relevant, a recalled (half-remembered) design principle from this project's history: "PairedRelay" — when one unit's capacity overflows, the **excess is relayed to a different, similarity-matched unit, never discarded.** This already exists and works for a *different* part of this architecture (bias terms), just not yet successfully adapted to the core weight-level update (attempt #7 above was a first, naive try at this and made things worse — open question whether a smarter version of the same idea would behave differently).

## The actual ask

We (I + an earlier Claude session, some weeks before this one) apparently hit this same wall before and found *something* that worked — I don't remember what. If this shape of problem is familiar from the literature, your own training, or just pattern-matches to something real: what's the fix? Specifically interested in:
- Why a deterministic, no-curvature-awareness, fixed-step discrete update oscillates on short/repeated/highly-correlated signal in a way smarter discrete methods (real QAR, DQT) don't as badly.
- Whether there's a way to get carry/overflow's benefit (never losing signal, same primitive set) without it just amplifying the existing overshoot problem.
- Any genuinely different angle we haven't tried.

## RESOLVED 2026-11-01 — root cause confirmed, working fix found, no float shadow anywhere

The CE-vs-reps numbers above were never the real diagnostic — they measure the discrete,
thresholded WEIGHT, but every prior attempt (ternary-range restriction, real QAR, overflow
shed/carry, hysteresis, Wemb RMS cap, a rate-cap "time delay") tuned the update RULE while
assuming its chosen gradient-sign direction was correct. None of them checked that assumption
against reality.

**Real measurement (`discrete_endpoint_audit.py`): the assumption was false in the specific way
that explains everything above.** Snapshotting weights before/after one real training call,
isolating each individual flip the rule actually committed, and measuring the REAL finite-
difference loss at that exact discrete endpoint showed: at reps 1-4, 61-80 of 80 sampled flips are
individually CORRECT in isolation (helpful, not harmful) — but committing all ~120-2200 real flips
of a call SIMULTANEOUSLY produces a measured interaction term of +2 to +7 nats, destroying roughly
80-90% of the benefit each flip would have given alone. This is "flip-avalanche," quantitatively
confirmed for the first time rather than just named. (At reps 5-6 the dominant failure shifts to
individually-wrong flips — likely staleness from the earlier avalanche damage, not an independent
cause.)

**The fix, directly targeted at the now-measured mechanism, not another blind rule tweak**
(`staggered_flip_test.py` → `staggered_flip_test_v2.py`/`_v3.py`, all in this `engine/` directory):
instead of committing every candidate flip a call proposes at once, rank all candidates (both the
5 level-matrices' ±1 flips AND, critically, Wemb's own variable-magnitude flips — previously exempt
from every fix tried) by confidence, then greedily accept/reject each one by actually re-measuring
real teacher-forced loss against a frozen trial snapshot, in order, committing only what measurably
helps. Deferred (rejected or unevaluated) candidates are NOT discarded — they stay in their
residual buffer, still intact, reconsidered next call, so nothing is lost, consistent with this
project's existing residual-carry conservation guarantee.

**Result: the real unmodified `FixedStepBrick` generates the literal, byte-for-byte correct
sentence "The capital of France is Paris."** — greedy decode, no float shadow weight anywhere,
training or inference, the project's standing constraint fully intact. Budget determines how fast
and how far: N=1000 crosses into the exact sentence at rep 30 and plateaus around 0.157-0.169
nats; going UNCAPPED (the full real candidate pool, no top-N truncation at all) crosses into the
exact sentence by **rep 8** and keeps falling to a floor of **0.002-0.004 nats by rep 20-40** — no
ceiling found yet, more real-loss-checked budget was strictly better at every size tried
(150→400→600→1000→uncapped). The only thing left after the trained sentence is a loop of
plausible-looking fragments, not corruption — expected, since nothing in this training loop ever
taught a stopping point; not a flaw in the weights or the mechanism. Full numbers, honest scope
(single fact/seed so far, real compute cost of N checked forward-passes per call not yet reduced to
something deployable), and the generated-text tables across every budget tested:
`lucy_west_validated_subset_exact_generation_breakthrough_20261101.md` (and its precursor,
`lucy_west_discrete_endpoint_audit_avalanche_confirmed_20261031.md`, for the audit methodology
itself) in the project memory.

Answers the original ask directly: the deterministic fixed-step rule oscillates not because
discreteness or range is inherently unstable (both already correctly suspected as red herrings
here), but because it never checked whether its OWN proposed batch of flips was jointly beneficial
before committing all of them at once — a search/coordination failure, not a representational one.

## FOLLOW-UP 2026-11-01 — the open cost question: cheaper candidate selection

The uncapped fix above real-loss-checks every single candidate flip with its own full forward
pass (up to ~5351/call) -- correct, but the clear blocker for scaling this to a real training set
rather than one hammered fact. Two cheaper selection strategies tested (`staggered_flip_test_v4.py`,
same directory), both against the real uncapped ground truth, same fact/seed, instrumented with an
exact forward-pass counter:

1. **Top-down bisection** (test whole ranges at once, shrink by halving on failure): a genuine
   NEGATIVE result. Cheap in the earliest reps (373x speedup at rep 1) but becomes WORSE than brute
   force once training matures (3.2x MORE total forward passes over 10 reps) because it pays for a
   wasted check on every large range that fails, and failures get common precisely as the real
   avalanche problem (the thing this whole fix exists to defeat) gets denser. Unbounded-bad in the
   worst case -- a structural flaw, not a tuning mistake.

2. **Galloping/doubling search** (grow batch size only after a success, snap back toward 1 on any
   failure): bounded. Plateaus at a fixed 0.7-0.8x overhead instead of spiraling, with quality
   matching or slightly beating ground truth throughout (both reach the exact correct sentence by
   rep 10). Not a net win on a 30-rep same-fact stress test (31% more total passes over the whole
   run, because most of that benchmark's workload sits in the fully-converged tail where the
   plateau dominates) -- but a large, real win specifically in reps 1-4 (65.9x, 3.2x, 2.2x, 1.2x),
   which is the regime a real multi-fact curriculum mostly lives in (most facts seen only a
   handful of times, not drilled dozens of times running). Full numbers:
   `lucy_west_galloping_candidate_selection_result_20261101.md` and
   `lucy_west_binary_bisection_candidate_selection_negative_20261101.md` in project memory.

Neither is wired into production. The honest open question going into real training-set work:
whether galloping's early-rep win is enough in practice, or whether the mature-fact plateau cost
needs a further fix (a smarter growth schedule, a cheaper real-loss proxy, or accepting the cost
as the price of exactness) before this is affordable at scale.
