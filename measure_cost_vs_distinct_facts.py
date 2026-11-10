"""Measures real learn_batch() cost as a function of how many DISTINCT real facts one brick has
been taught -- not repeated exposure to the same fact (already measured earlier tonight), but the
actual production pattern: one real teaching exposure per distinct fact. This is the missing
number needed to answer the real question: does Wolfpack's capped-at-5-facts-per-brick policy
actually cost more or less total compute than today's system, which lets one brick absorb
arbitrarily many distinct facts (up to 126, confirmed in the real pilot)?

SAFETY: fresh, never-saved cluster, single brick captured once and reused directly (same proven
pattern as tonight's earlier validated-subset cost work) -- never touches the live checkpoint.
"""
import json
import sys
import time
from pathlib import Path

ENGINE_DIR = Path(r"F:\LucyWEST\PrivateLucyWEST\engine")
sys.path.insert(0, str(ENGINE_DIR))

FRESH_STATE_PATH = ENGINE_DIR / "_cost_vs_facts_fresh_state_never_saved.npz"
if FRESH_STATE_PATH.exists():
    FRESH_STATE_PATH.unlink()

import lutgtlm_ingest as LI
import brickcluster_engine as BCE
LI.STATE = str(FRESH_STATE_PATH)

CLUSTER, _seen, _ingests = LI.load_or_init()

# Pull real, genuinely distinct facts from the live question corpus -- same topic area so they're
# plausible candidates for sharing ONE brick (matching what Wolfpack's own similarity gate would
# actually group together), not random unrelated content.
qp = Path(r"C:\Users\Azazel\Desktop\Amelia's Dataset\NLP Pilot\nlp_questions.jsonl")
rows = []
with qp.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

# take facts from the SAME source_file, since those are the ones plausibly similar enough to
# share a brick under Wolfpack's real similarity gate
by_source = {}
for r in rows:
    by_source.setdefault(r["source_file"], []).append(r)
src = max(by_source, key=lambda k: len(by_source[k]))
facts = by_source[src][:8]
print(f"Using {len(facts)} real facts from source_file={src!r}\n")

brick_idx = None
results = []
for n, r in enumerate(facts, start=1):
    text = f"User: {r['question'].strip()}\nLucy: {r['ground_truth_answer'].strip()}"
    batches = LI._line_batches(text)
    if not batches:
        continue
    if brick_idx is None:
        brick_idx = CLUSTER.route(CLUSTER.tfidf_sparse(text), for_training=True)
    brick = CLUSTER.bricks[brick_idx]

    t0 = time.time()
    loss = brick.learn_batch(batches[0], lr_scale=BCE.LR_SCALE, corrob_eligible=True)
    elapsed = time.time() - t0
    results.append((n, elapsed, float(loss.mean()) if hasattr(loss, "mean") else float(loss)))
    print(f"  fact #{n}: learn_batch() took {elapsed:.2f}s, loss={results[-1][2]:.3f}")

print("\nn_distinct_facts  learn_batch_seconds  loss")
for n, t, l in results:
    print(f"{n:>16}  {t:>19.2f}  {l:.3f}")

if FRESH_STATE_PATH.exists():
    FRESH_STATE_PATH.unlink()
