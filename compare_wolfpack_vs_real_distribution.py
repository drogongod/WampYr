"""Does the fixed Wolfpack librarian actually prevent the concentration problem that motivated
this whole investigation? Replays the REAL NLP Pilot's own already-processed items, in their real
order, through WolfpackMainIndex (capacity-bounded), and compares brick-distribution stats against
what the real system actually did (read-only, from lucy_nlp_pilot_results.jsonl -- never touches
the live checkpoint or the live process at all).
"""
import json
from collections import Counter
from pathlib import Path

from wolfpack_librarian import WolfpackMainIndex, stable_vec, freeze_reference_idf


def main():
    results_path = Path(r"C:\Users\Azazel\Desktop\Amelia's Dataset\NLP Pilot\lucy_nlp_pilot_results.jsonl")
    questions_path = Path(r"C:\Users\Azazel\Desktop\Amelia's Dataset\NLP Pilot\nlp_questions.jsonl")

    real_rows = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                real_rows.append(json.loads(line))

    real_brick_counts = Counter(r["brick_index"] for r in real_rows if r.get("brick_index") is not None)
    print(f"REAL system: {len(real_rows)} items -> {len(real_brick_counts)} distinct bricks, "
          f"max reuse = {max(real_brick_counts.values())}, "
          f"bricks reused >10x = {sum(1 for v in real_brick_counts.values() if v > 10)}, "
          f"bricks reused >50x = {sum(1 for v in real_brick_counts.values() if v > 50)}")

    all_texts = []
    with questions_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_texts.append(json.loads(line))
    freeze_reference_idf([f"User: {r['question'].strip()}\nLucy: {r['ground_truth_answer'].strip()}"
                          for r in all_texts])
    print(f"Froze reference IDF from {len(all_texts)} real questions.\n")

    for capacity in [10, 20, 40]:
        main_idx = WolfpackMainIndex(dim=8192, region_capacity=capacity)
        brick_counts = Counter()
        for r in real_rows:
            text = f"User: {r['question'].strip()}\nLucy: {r['ground_truth_answer'].strip()}"
            vec = stable_vec(text)
            brick_id, was_retrieval, lib_id = main_idx.dispatch(text, vec)
            brick_counts[brick_id] += 1
        stats = main_idx.stats()
        print(f"WOLFPACK (region_capacity={capacity}): {len(real_rows)} items -> "
              f"{stats['n_regions']} regions, {stats['n_bricks']} bricks, "
              f"max reuse = {max(brick_counts.values())}, "
              f"bricks reused >10x = {sum(1 for v in brick_counts.values() if v > 10)}, "
              f"bricks reused >50x = {sum(1 for v in brick_counts.values() if v > 50)}")


if __name__ == "__main__":
    main()
