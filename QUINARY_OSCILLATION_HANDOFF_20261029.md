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
   rep 10, and gallop's NLL is actually lower than ground's at several reps, e.g. rep 10: 0.0235 vs
   0.0400). Not a net win on a 30-rep same-fact stress test (confirmed final totals: ground=102,207
   forward passes, gallop=133,858, overall 0.8x -- i.e. 31% more total passes over the whole run,
   because most of that benchmark's workload sits in the fully-converged tail where the plateau
   dominates) -- but a large, real win specifically in reps 1-4 (65.9x, 3.2x, 2.2x, 1.2x), which is
   the regime a real multi-fact curriculum mostly lives in (most facts seen only a handful of times,
   not drilled dozens of times running). Full numbers:
   `lucy_west_galloping_candidate_selection_result_20261101.md` and
   `lucy_west_binary_bisection_candidate_selection_negative_20261101.md` in project memory.

Neither is wired into production. The honest open question going into real training-set work:
whether galloping's early-rep win is enough in practice, or whether the mature-fact plateau cost
needs a further fix (a smarter growth schedule, a cheaper real-loss proxy, or accepting the cost
as the price of exactness) before this is affordable at scale.

**Status as of 2026-11-01: this arc is closed pending real multi-fact training data.** The core bug
(flip-avalanche) is solved and verified with zero float shadow weight. The cost question has a
real, bounded, partial answer (galloping) but not a final one -- the next real step is testing it
against an actual multi-fact curriculum once one exists, not further single-fact tuning.

## CORRECTION 2026-11-01, same day: the multi-fact test ran, and galloping does NOT win on it

Built that multi-fact curriculum test immediately (`staggered_flip_test_v5_multifact.py`, same
directory) -- 8 distinct facts (country/capital pairs), 4 reps each, block curriculum, one brick.
**Result: the "galloping wins the real-curriculum regime" hypothesis is FALSE.** Only the very
first fact ever taught to a blank brick (France) showed galloping's big speedup (65.9x/3.2x/2.2x/
1.2x, matching the earlier single-fact benchmark exactly). Every fact taught after that ran
galloping at 0.4-0.8x -- MORE expensive than ground truth, not less -- with measurably worse
quality too (by the Canada fact, galloping's NLL was stuck at 6.8-9.2 nats of garbage while ground
truth cleanly reached 0.8-1.2 nats). Overall across all 8 facts x 4 reps: ground=179,694 forward
passes, gallop=252,095, speedup=0.7x -- worse than even the single-fact stress test's 0.8x.

**Mechanistic read:** galloping's early win depends on the brick being near its blank random
initialization, where most proposed flips are obviously correct and large batches succeed easily.
A real curriculum only provides that blank state ONCE, before the first fact ever -- not once per
new fact. Once any prior learning exists, newly proposed flips interact with already-meaningful
weights in ways that are no longer obviously net-beneficial in large groups, so galloping's
batches fail more and its cost reverts to (here, past) brute force.

**Separately, both policies showed near-total catastrophic forgetting of earlier facts** in a
post-curriculum retention check -- expected, not a candidate-selection artifact: this test used
ONE brick for all 8 unrelated facts, which is not how production works (`BrickCluster.route()`
exists specifically to send different facts to different bricks). Read as confirming why that
routing layer is load-bearing, not as new evidence against the core fix.

**Honest state of the cost question as of this update: no cheaper candidate-selection method has
yet been found that beats brute-force one-at-a-time validated-subset selection on anything but a
brick's very first exposure ever.** Full numbers:
`lucy_west_galloping_multifact_curriculum_negative_20261101.md` in project memory (corrects
`lucy_west_galloping_candidate_selection_result_20261101.md`'s closing argument).

## FOLLOW-UP 2026-11-01, same day: coherent vs. unrelated content -- the actual explanation for the catastrophic forgetting above

The 8-country test above forced UNRELATED facts into one brick with no router -- a known, expected
failure mode (`BrickCluster.route()` exists specifically to send different facts to different,
similarity-matched bricks so this never happens in production). Tested directly whether COHERENT,
thematically-related cumulative content (one growing passage about a single subject, extended stage
by stage, a correction appended at the end rather than anything deleted) avoids that destructive
interference (`staggered_flip_test_v6_coherent_growth.py`).

**Positive result: no catastrophic NLL jumps, ever**, across 6 stages growing a passage from 30 to
240+ characters -- every stage transition bumped mildly then recovered within that same stage,
never spiking the way unrelated facts did (there, individual facts jumped from ~1-2 nats to 6-9
nats once later unrelated content was layered on). NLL stayed in a tight 2.4-4.2 nat band the whole
way, trending mildly down overall.

**Inconclusive on the sharpest question** (does an appended correction -- "English muffins are not
actually English" -- actually show up in generation, or get drowned out): greedy generation was
garbage at every single stage, including the very first and simplest one alone. Not evidence against
the hypothesis -- this run was deliberately budget-capped (1500 candidates/call, 4 reps/stage) for
wall-clock reasons, well short of the uncapped-budget/30-40-rep regime the single-fact breakthrough
needed to cross from garbage into literal correct generation. Final NLL 2.59 nats is more than an
order of magnitude above the ~0.002-0.16 nat floor that correlated with actual fluent output there.
Needs a rerun at the resourcing level that worked before this specific question is settled. Full
numbers: `lucy_west_coherent_growth_correction_inconclusive_20261101.md` in project memory.

## FOLLOW-UP 2026-11-01, same day: the real router, unmodified -- confirms it was a capacity-allocation artifact all along

Jonathan's direct pushback on the 8-country forgetting result: that test forced unrelated facts
into one brick with `BrickCluster.route()` bypassed entirely, which is a known, designed-for
failure mode -- the real question is whether the router has "enough resolution" (his
alphabet/nations analogy: you don't need unbounded categories, just enough that similar content
clusters together and different content gets isolated) to prevent this in actual use.

Tested directly (`staggered_flip_test_v7_real_router.py`): a `BrickCluster` subclass with ONLY
`_spawn()` overridden to produce validated-subset-capable bricks -- `route()`, `learn_line()`,
relay, and consolidate are the real, unmodified production code. Also fixed a methodology mismatch
a sanity check caught first: repeating the identical line 4+ times (this session's earlier stress-
test convention) destabilizes the real router, because production normally filters an exact-
duplicate line before it ever reaches `learn_line()` -- repeating it anyway shrinks that line's own
TF-IDF features toward zero via rising document frequency, an artifact of testing an unsupported
input shape, not a routing bug. Switched to true one-shot teaching (matching this project's actual
north star: "single exposure is the target, not forty repeated presentations").

**Result: the same 8 country/capital facts split into only 3 bricks (6 shared one brick, 2 got
their own) with NO catastrophic forgetting anywhere.** The shared brick's worst drift was +1.08
nats (Egypt) -- nowhere near random baseline (5.545), let alone past it; two facts (France, Italy)
actually IMPROVED after more facts were layered on top. The two isolated-brick facts retained
perfectly, bit-for-bit. Confirms directly: the 8-country test's catastrophic forgetting was a
capacity-allocation artifact from bypassing the router, not a flaw in the validated-subset fix
itself. Honest nuance: which specific facts got split looks somewhat noisy/order-dependent, not
cleanly semantic -- but the mechanism tolerated that noise without collapsing either way, which is
a more robust property than requiring perfect categorization. Full numbers:
`lucy_west_real_router_prevents_collapse_20261101.md` in project memory.

## FOLLOW-UP 2026-11-01, same day: cross-category probe is MIXED, not a confirmation

Jonathan's own caveat going in: the Jonathan/breakfast passage "was a quick example of principle, I
don't know if it's actually the best to use to represent a bit-level model." Correct to flag --
the test below is asymmetric (8 short repeated-template Q&A facts vs. one growing narrative), a
noisy probe, not a clean controlled one.

Extended the real-router test (`staggered_flip_test_v8_cross_category.py`): same 8 capital facts,
same real `BrickCluster`, same cluster instance -- then taught the 6-stage growing Jonathan passage
through it afterward, one-shot per stage, no reset. **Result: the Jonathan content did NOT spawn a
new brick.** All 6 stages routed into brick 2, the SAME brick Brazil already occupied (brick
assignment from phase 1 was unchanged: brick 0 = France/Italy/Spain/Japan/Egypt/Canada, brick 1 =
Germany alone, brick 2 = Brazil alone).

Retention: Brazil's NLL after the Jonathan teaching = 7.18 nats, second-worst of all 9 items
checked -- but Germany (isolated, untouched by anything) is WORSE at 7.89, so this run cannot
cleanly attribute Brazil's number to interference from Jonathan versus ordinary weak one-shot
retention in a thin/isolated brick. No mid-run checkpoint was taken (Brazil's NLL right after
phase 1, before Jonathan), so cause is unresolved either way. This is a more mixed result than the
first real-router test, not a repeat confirmation of it -- the router grouped a genuinely different
subject into an existing brick rather than carving out new territory, suggesting its TF-IDF/trigram
similarity may track structural/surface overlap more than semantic topic. Full numbers:
`lucy_west_cross_category_router_shares_brick_20261101.md` in project memory.

## FOLLOW-UP 2026-11-01, same day: confirmed WHY, with real margins printed

Reconstructed cluster state right before Jonathan stage 1 was taught and printed `route()`'s
actual per-brick margins. Two real mechanisms, both already in the code, neither a bug:

1. `n_docs=8` after phase 1, `IDF_MIN_DOCS=20` -- TF-IDF weighting NEVER activated anywhere in
   this test (max 13 docs ever seen). Every feature vector was raw trigram counts the whole time,
   exactly the regime the module's own 2026-10-09 comment already warned about (unrelated real
   English prose measuring ~0.965 cosine-similar on raw counts).
2. The established 6-fact brick actually REJECTED Jonathan's content outright (margin=-0.0217,
   below the strict `>0` bar) -- it was never "merged" there. The winner was a singleton brick
   (Brazil, margin=0.1048, beating Germany's 0.0102) because `_OnlineGaussian`'s adaptive variance
   floor deliberately widens to full background width at n=1 ("don't reject on this basis before
   you have evidence") -- making a freshly-spawned singleton the MOST lenient brick in the cluster,
   by design, regardless of its content. Brazil beat Germany only by incidental raw-vector
   proximity to Jonathan's text, not semantic signal.

Conclusion: this isn't a router failure, it's two real safeguards (IDF, variance-floor tightening)
never getting enough data (only 14 lines total) to engage. The "enough resolution" question remains
untested at adequate scale (~20+ real lines, content actually merging into experienced bricks, not
freshly-spawned singletons) by either real-router run so far. Full numbers:
`lucy_west_cross_category_why_shared_brick_mechanism_20261101.md` in project memory.

## FOLLOW-UP 2026-11-01, same day: three candidate fixes tested, two falsified, one works with a tradeoff

Jonathan's ask: "we still need a way to make certain two unlike bricks do not occupy the same
brick when all else is even." Four candidates tested against the real routing trace:

- Raise the spawn-margin bar above 0 -- FALSIFIED. Legitimate capital-fact merges (margins as low
  as 0.0008-0.0050) are as thin as, or thinner than, the spurious Jonathan-into-Brazil merges
  (0.0237-0.1048). No threshold separates the two; the spurious ones are often the LARGER margins.
- Lower IDF_MIN_DOCS so IDF engages sooner -- FALSIFIED, makes it worse. At IDF_MIN_DOCS<=2, every
  capital fact AND every Jonathan stage collapse into one brick -- IDF from too few real documents
  is itself too noisy, exactly what the existing comment on this constant already warned about.
- Lexical-overlap backstop against `brick_taught_lines` -- doesn't discriminate either: legitimate
  merges share ZERO content words with each other, only the template.
- Force-spawn unconditionally while `n_docs < IDF_MIN_DOCS` -- WORKS, real tradeoff. Confirmed:
  zero cross-category sharing (14/14 facts spawn separately during cold start). Cost: nothing
  merges during that window either, including facts that plausibly should share a brick. Bounded
  in production (4.2MB corpus clears n_docs>=20 almost immediately) -- a one-time bootstrap tax.

**Open question, not yet tested:** does the SAME n=1 variance-floor leniency re-create this failure
for a freshly-spawned brick inside an already-mature, IDF-active cluster (cluster n_docs>=20, but
the new brick's own n=1)? The floor formula only looks at the brick's own n, not the cluster's
n_docs, so this is plausible and unresolved. If cold-start-only, the force-spawn gate is a complete
fix; if not, the real fix lives in `_OnlineGaussian`'s variance floor itself (analogous to the
existing BIAS_CORROB_REUSE_THRESHOLD=3 "don't trust low-evidence signal yet" precedent), not the
cluster-level IDF gate. Needs a real large-scale trace to test. Not yet wired into production,
decision pending. Full numbers: `lucy_west_cross_category_fix_candidates_tested_20261101.md` in
project memory.

## ⛔ NORTH STAR CORRECTION 2026-11-01, same day: the whole "cross-category" framing above was wrong

Jonathan's direct correction: **LucyWEST's router organizes by MATHEMATICAL signature similarity,
never by subject matter.** `BrickCluster.route()` does not need to know whether content is
geography or biography -- it compares numerical feature vectors and routes by affinity.
**Two sufficiently similar signatures sharing a brick is intentional, a feature, not a defect.**
Approximate organization (occasional "misplaced pages") is explicitly acceptable -- 99.9%
human-topic-correct placement is NOT a mandatory bar. Do not add subject classifiers, semantic
category gates, or topic labels as a routing condition -- that would be a real departure from the
actual architecture, not a fix.

**This directly overturns the framing of the three sections immediately above.** "Jonathan's
passage shared Brazil's brick despite being a different topic" was never actually a problem by this
project's real objective. Re-examining the SAME data through the corrected lens (stability/
interference, never topic purity): Brazil's post-share NLL (7.18 nats) was actually BETTER than
isolated, completely untouched Germany's (7.89 nats) -- the data never showed sharing causing worse
outcomes than not sharing. The "force-spawn while n_docs<IDF_MIN_DOCS" fix candidate is very likely
NOT the right move to ship, now that the thing it was built to prevent (topically-mismatched
merging) is understood not to be a real target -- it would trade away genuine mathematical
self-organization/compression specifically to chase topic purity that was never the goal.

**What remains legitimate, unchanged:** the underlying engineering findings -- unreliable
small-sample IDF estimates, unusually permissive singleton variance at n=1, Gaussian acceptance
calibration -- are still real and still worth fixing. Any fix to those must preserve mathematical
organization, evaluated by learning stability/interference/retention/recoverability, never by
whether a human agrees with the apparent topic grouping. Full text and the corrected evaluation
criteria: `lucy_west_mathematical_memory_ecology_north_star_20261101.md` in project memory --
read this before touching `BrickCluster.route()` again.
