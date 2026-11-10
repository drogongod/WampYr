"""Real-corpus validation of wolfpack_librarian.py, same methodology as the prototype's v3 (which
found and fixed 3 real bugs and confirmed the founder/joiner retrieval split: 94.7% vs 61.5%).
Tests the production module against the same real NLP Pilot question corpus, read-only.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import lutgtlm_ingest as LI
import brick_cluster_charlm as BCC
from wolfpack_librarian import WolfpackMainIndex

FRESH_STATE_PATH = Path(__file__).parent / "_wolfpack_test_fresh_state_never_saved.npz"
if FRESH_STATE_PATH.exists():
    FRESH_STATE_PATH.unlink()
LI.STATE = str(FRESH_STATE_PATH)
CLUSTER, _seen, _ingests = LI.load_or_init()


def tfidf_vec(text: str) -> np.ndarray:
    idx, val = CLUSTER.tfidf_sparse(text)
    dense = np.zeros(BCC.FEATURE_DIM, dtype=np.float64)
    if len(idx):
        dense[idx] = val
    CLUSTER._update_doc_freq(text)  # test-harness responsibility, not the module's (see docstring)
    return dense


def load_real_corpus(n_source_files=8, n_per_file=10):
    path = Path(r"C:\Users\Azazel\Desktop\Amelia's Dataset\NLP Pilot\nlp_questions.jsonl")
    by_source = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = f"User: {row['question'].strip()}\nLucy: {row['ground_truth_answer'].strip()}"
            by_source[row["source_file"]].append(text)
    random.seed(0)
    sources = sorted(by_source.keys())
    chosen = random.sample(sources, min(n_source_files, len(sources)))
    fact_to_source = {}
    all_facts = []
    for src in chosen:
        for text in random.sample(by_source[src], min(n_per_file, len(by_source[src]))):
            all_facts.append(text)
            fact_to_source[text] = src
    return all_facts, fact_to_source


def main():
    all_facts, fact_to_source = load_real_corpus(n_source_files=8, n_per_file=10)
    print(f"Loaded {len(all_facts)} real facts from {len(set(fact_to_source.values()))} "
          f"real source files.\n")

    random.seed(1)
    workload = list(all_facts)
    for _ in range(min(200, len(all_facts) * 2)):
        workload.append(random.choice(all_facts))
    random.shuffle(workload)

    main_idx = WolfpackMainIndex(dim=BCC.FEATURE_DIM, region_capacity=10)
    seen_brick_for_fact = {}
    founder_facts = set()
    retrieval_correct = retrieval_total = 0
    founder_correct = founder_total = 0
    nonfounder_correct = nonfounder_total = 0
    region_source_members = defaultdict(set)

    for text in workload:
        vec = tfidf_vec(text)
        was_founder_before = text in founder_facts
        n_regions_before = main_idx.stats()["n_regions"]
        brick_id, was_retrieval, lib_id = main_idx.dispatch(text, vec)
        n_regions_after = main_idx.stats()["n_regions"]

        if text in seen_brick_for_fact:
            retrieval_total += 1
            hit = was_retrieval and brick_id == seen_brick_for_fact[text]
            retrieval_correct += int(hit)
            if was_founder_before:
                founder_total += 1
                founder_correct += int(hit)
            else:
                nonfounder_total += 1
                nonfounder_correct += int(hit)
        else:
            seen_brick_for_fact[text] = brick_id
            if n_regions_after > n_regions_before:
                founder_facts.add(text)
        region_source_members[lib_id].add(fact_to_source[text])

    stats = main_idx.stats()
    print(f"Distinct facts: {len(all_facts)}, workload size: {len(workload)}")
    print(f"Regions/librarians created: {stats['n_regions']}")
    print(f"Bricks created total: {stats['n_bricks']}")
    print(f"Re-exposure retrieval accuracy: {retrieval_correct}/{retrieval_total} = "
          f"{100*retrieval_correct/max(1,retrieval_total):.1f}%")
    print(f"  -- founding facts:     {founder_correct}/{founder_total} = "
          f"{100*founder_correct/max(1,founder_total):.1f}%")
    print(f"  -- non-founding facts: {nonfounder_correct}/{nonfounder_total} = "
          f"{100*nonfounder_correct/max(1,nonfounder_total):.1f}%  (prototype baseline: 61.5%, "
          f"this module adds medoid re-anchoring + top-K fallback to improve this)")
    mixed = sum(1 for s in region_source_members.values() if len(s) > 1)
    print(f"{mixed}/{stats['n_regions']} regions mixed more than one real source file.")


if __name__ == "__main__":
    main()
