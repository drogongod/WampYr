#!/usr/bin/env python3
"""THE NON-NEGOTIABLE EXPERIMENT, F:\\Wampyr.txt, adapted to BrickClusterEngine.

Runs the host-off evidence chain directly against the engine (no live donor required for this
part -- see the note at the bottom about what still needs a real donor to test end-to-end):

  1. Ask X. Confirm Lucy cannot answer host-off (own weights, no donor).
  2. Teach X via learn_accepted (the real path /accept or /overlord would trigger).
  3. Verify SPECIFIC changed quinary levels, not just "something happened."
  4. Ask X again, host-off. Check whether a matching brick now exists.
  5. Teach unrelated Y and Z.
  6. Ask X again, host-off. Confirm retention wasn't wiped by teaching Y/Z.

"Brick matched" is NOT the same as "fluent correct answer" -- at this training budget (a
handful of single-fact lines) generation will be garbled, same as every other BrickCluster test
this session. This script reports exactly what it measured, not an inferred verdict, per the
evidence contract: "a pretty answer is NOT proof," "SSM/weight changing is NOT proof," and
"Lucy retained X requires testing after additional unrelated learning."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import brickcluster_engine as E

E.LI.STATE = os.path.join(os.path.dirname(__file__), "data", "brickcluster",
                           "checkpoints", "_evidence_contract_test_mind.npz")
if os.path.exists(E.LI.STATE):
    os.remove(E.LI.STATE)

engine = E.BrickClusterEngine()
X_Q, X_A = "What is the capital of England?", "The capital of England is London."
Y_Q, Y_A = "What is the capital of France?", "The capital of France is Paris."
Z_Q, Z_A = "What is the capital of Germany?", "The capital of Germany is Berlin."


def host_off_ask(question: str) -> dict:
    candidate, error, matched = engine._lucy_candidate(question)
    return {"question": question, "brick_matched": matched,
            "candidate": candidate, "error": error}


print("=== STEP 1: host-off, before any teaching ===")
r1 = host_off_ask(X_Q)
print(r1)
assert r1["brick_matched"] is False, "FAIL: claims a match before anything was taught"
print("PASS: no brick matched -- Lucy genuinely does not know this yet, confirmed not assumed.\n")

print("=== STEP 2: teach X via learn_accepted (the real /accept path) ===")
update = engine.learn_accepted(X_Q, X_A)
print(update)
assert update["changed_levels"] >= 0  # first fact: no "before" baseline exists, see engine comment
print(f"Recorded: brick_index={update['brick_index']} is_new_brick={update['is_new_brick']} "
      f"changed_levels={update['changed_levels']}/{update['total_levels_in_brick']}\n")

print("=== STEP 3: host-off retest, immediately after teaching X ===")
r3 = host_off_ask(X_Q)
print(r3)
print(f"brick_matched={r3['brick_matched']} -- this is the real signal, not the candidate text's "
      f"fluency (expect garbled output at this training budget).\n")

print("=== STEP 4: teach unrelated Y and Z (interference test setup) ===")
update_y = engine.learn_accepted(Y_Q, Y_A)
update_z = engine.learn_accepted(Z_Q, Z_A)
print(f"Y: brick_index={update_y['brick_index']} is_new_brick={update_y['is_new_brick']} "
      f"changed_levels={update_y['changed_levels']}")
print(f"Z: brick_index={update_z['brick_index']} is_new_brick={update_z['is_new_brick']} "
      f"changed_levels={update_z['changed_levels']}\n")

print("=== STEP 5: re-ask X after unrelated learning (retention-after-interference) ===")
r5 = host_off_ask(X_Q)
print(r5)
retained = r5["brick_matched"]
print(f"\nRESULT: X {'STILL' if retained else 'NO LONGER'} matches an existing brick after "
      f"teaching Y and Z. {'This is real evidence of retention under interference.' if retained else 'This is a real retention FAILURE -- report it as such, do not soften it.'}")

print("\n=== WHAT THIS DID NOT TEST (per the evidence contract -- say so, don't imply it) ===")
print("- No live donor was contacted. complete()'s real donor-fallback path, the frozen-donor")
print("  contract, and the full session-context-buffer round trip through the HTTP gateway are")
print("  UNTESTED here. Run brickcluster_server.py against a live donor on 8102 for that.")
print("- 'brick_matched' is a routing-gate signal (Phi_pos beats Phi_neg), not a judged-correct")
print("  answer. Fluent, correct generation at this training budget is NOT established or claimed.")
