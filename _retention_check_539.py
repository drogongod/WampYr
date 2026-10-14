#!/usr/bin/env python3
"""Retention check against the real, journaled 539-fact checkpoint from the interrupted
1000-fact run (MERGE_SLACK=0.08). Reuses the exact same host-off check
(engine._lucy_candidate) as amelia_teaches_lucy.py's own final retention loop, driven by the
REAL taught (question, answer) pairs from the immediately-flushed journal file, not the
buffered console log."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))
import brickcluster_engine as E
import brick_cluster_charlm as BC

CHECKPOINT = os.path.join(os.path.dirname(__file__), "data", "brickcluster",
                           "checkpoints", "amelia_teaches_lucy_mind.npz")
E.LI.STATE = CHECKPOINT

log_path = os.path.join(os.path.dirname(__file__), "data", "brickcluster",
                         "experience", "amelia_teaching_log.jsonl")
with open(log_path, encoding="utf-8") as f:
    taught = [json.loads(l) for l in f]

engine = E.BrickClusterEngine()
print(f"[loaded] n_bricks={len(engine.cluster.bricks)} seen={engine.seen} taught_facts={len(taught)}")

matched_count = 0
for row in taught:
    q, a = row["question"], row["answer"]
    candidate, error, matched = engine._lucy_candidate(q)
    matched_count += int(matched)

print(f"\n[RETENTION RESULT] {matched_count}/{len(taught)} matched host-off "
      f"({100*matched_count/len(taught):.1f}%) -- n_bricks={len(engine.cluster.bricks)} "
      f"MERGE_SLACK={BC.MERGE_SLACK}")
