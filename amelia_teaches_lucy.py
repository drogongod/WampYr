#!/usr/bin/env python3
"""Amelia-voiced donor asks Lucy questions, bulk-teaching run.

Same 27B donor (vampire-lucywest) plays TWO roles via two separate system prompts:
  - TEACHER: given Amelia's character sheet, generates one new factual question per round.
  - ANSWERER: the existing neutral donor persona inside BrickClusterEngine.complete(), answers
    the question for Lucy exactly as the real vampire mechanism already does.

HONEST FLAG, per the user's own discussion before running this: this is a BULK run with
AUTO-ACCEPT, not an individually human-reviewed /accept for each fact. That is a real,
deliberate reduction in verification rigor relative to the evidence contract's normal human-gate
-- every accepted row is logged with auto_accepted=true so this is traceable, not hidden.

Consolidation is NOT run per-fact (the regression found earlier today) -- it runs ONCE at the
end via engine.sleep(), matching "sleep was always the plan."
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))
import brickcluster_engine as E

N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SLEEP_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CHECKPOINT = os.path.join(os.path.dirname(__file__), "data", "brickcluster",
                           "checkpoints", "amelia_teaches_lucy_mind.npz")
E.LI.STATE = CHECKPOINT

IDENTITY_PATH = r"C:\Users\Azazel\Desktop\Amelia's Dataset\Identity_Amelia_short.txt"
AMELIA_IDENTITY = open(IDENTITY_PATH, encoding="utf-8").read().strip()

TEACHER_SYSTEM = (
    AMELIA_IDENTITY + "\n\n"
    "You are quizzing a student named Lucy. Your ONLY job right now: ask exactly ONE short, "
    "clear, factual question she could learn from. Never repeat a question already asked (a "
    "list of already-asked questions follows). Reply with ONLY the question text, nothing else "
    "-- no preamble, no commentary, no quotation marks."
)

# 2026-10-11: EXPLICIT topic-category rotation, Jonathan's fix. Leaving topic choice to the
# model's own judgment (the old prompt's "vary the topic across rounds" instruction) still
# converged hard: the real 1000-fact run hit a 62-64% duplicate-skip rate by the back half --
# the model kept circling back to the same "obvious" trivia within whichever topics it drifted
# to, even with a 300-question avoid-list. Forcing a SPECIFIC category each round, round-robin,
# plus an explicit "not the most famous/obvious fact" instruction, is a direct attempt to push
# into less-exhausted territory rather than hoping self-diversification holds at scale.
CATEGORIES = [
    "geography", "history", "physics", "chemistry", "biology", "mathematics/arithmetic",
    "literature", "movies", "music", "video games", "mythology and folklore", "sports",
    "art and architecture", "technology and computing", "food and cooking",
    "animals and nature", "space and astronomy", "famous historical figures",
    "language and etymology", "inventions and discoveries",
]


def category_for(fact_index: int) -> str:
    return CATEGORIES[fact_index % len(CATEGORIES)]


AVOID_LIST_WINDOW = 300  # 2026-10-10: was 40, empirically found to be the real cause of a 44%
# duplicate-skip rate -- 244/245 skips in that run were exact repeats of a question asked MORE
# than 40 positions back, i.e. genuinely outside what the teacher could see, not the model
# ignoring the list (it respected it 244/245 times when the prior question WAS visible). Donor
# n_ctx=8192; ~300 questions at ~15 tokens each is ~4500 tokens, leaving real headroom for the
# system prompt + instructions + generation. Not unlimited (full 1000-question history could
# overflow context by itself) but a large, deliberate improvement over 40, not a guess.


def build_teacher_body(asked_already: list[str], category: str) -> dict:
    """The ONE place this request body is built -- diagnostics must call this, not re-type the
    prompt inline, after a real bug (2026-10-11) where a diagnostic script hardcoded a stale copy
    of the instruction and silently tested the wrong prompt for a full debugging cycle."""
    avoid = "\n".join(f"- {q}" for q in asked_already[-AVOID_LIST_WINDOW:]) or "(none yet)"
    return {
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Already asked, do not repeat any of these:\n{avoid}\n\n"
                                         f"Your question for THIS round must be about: {category}. "
                                         "Pick ONE specific fact in that category and ask about it "
                                         "directly -- the first reasonable fact that comes to mind "
                                         "is fine, do not deliberate over whether it's famous "
                                         "enough. Ask your next question now."},
        ],
        "model": "vampire-lucywest", "stream": False, "temperature": 0.9, "max_tokens": 600,
        # 2026-10-11, FULL debugging arc, corrected understanding (earlier comments in this spot
        # were wrong, not just incomplete -- kept replaced rather than silently edited elsewhere
        # in the file, this is the one place it matters):
        # 1. Tried chat_template_kwargs={"enable_thinking": False} -- REAL, CONFIRMED finding via
        #    direct curl: this server hangs indefinitely (60s+, zero response) when that field is
        #    present. Removing it alone restored a normal 2-second response. Do not re-add it.
        # 2. Then set max_tokens=150 alone and assumed that fixed it (one real test showed 64/150
        #    tokens used cleanly) -- WRONG, and the mistake was trusting client-side CPU time as a
        #    "is it stuck" signal: near-zero CPU during a live network wait is NORMAL (the client
        #    just blocks on a socket read), not evidence of a hang. Two validation runs were
        #    killed early on that flawed signal while they may have been behaving correctly.
        # 3. The REAL failure, found by letting a run reach its own natural exit instead of
        #    killing it early: all 120 retry attempts (40 facts * 3) returned an EMPTY content
        #    string, every single time, then the script correctly rejected each as empty and
        #    exhausted its retry budget after ~4 minutes -- which is exactly what had looked like
        #    a "hang" from the outside. Root cause: the REAL system prompt (Amelia's full identity
        #    + the category-rotation instruction) is longer and harder than the simplified prompt
        #    used in the isolated curl test, so reasoning likely consumed the entire 150-token
        #    budget, leaving nothing for the actual question -- the same "thinking ate the budget"
        #    failure this project has already documented elsewhere (mamba3_parasite_loop.py).
        #    Fix: 600 tokens, matching the OTHER donor calls in this codebase that already handle
        #    reasoning+answer together successfully at that budget.
        # 4. Even with 600 tokens AND the softened "pick ONE fact, don't deliberate" instruction
        #    (replacing "not the most famous/obvious fact... less-exhausted detail", which was
        #    directly observed via _diag_empty_responses.py pushing the model into long
        #    generate-and-reject-candidate reasoning spirals, 75-80% finish_reason="length"):
        #    re-verify with that same diagnostic before trusting this at scale again. Do not
        #    assume a prompt edit fixed an empty-response problem without re-measuring it.
    }


def ask_amelia_for_question(asked_already: list[str], category: str) -> str:
    body = build_teacher_body(asked_already, category)
    response = E.post_json(E.REFERENCE_URL.rstrip("/") + "/v1/chat/completions", body, timeout=120)
    text = response["choices"][0]["message"].get("content") or ""
    return text.strip().strip('"').strip()


def canonical(q: str) -> str:
    return " ".join(q.lower().split())


def main():
    engine = E.BrickClusterEngine()
    print(f"[start] n_bricks={len(engine.cluster.bricks)} seen={engine.seen}")
    print(f"[plan] {N_FACTS} facts, auto-accept, sleep() every {SLEEP_EVERY} facts "
          f"(route cost scales with brick count -- periodic sleep keeps per-fact latency from "
          f"climbing across a long run, without reintroducing the per-accept regression) +\n")

    asked_text: list[str] = []
    asked_keys: set[str] = set()
    taught: list[tuple[str, str]] = []
    log_path = os.path.join(os.path.dirname(__file__), "data", "brickcluster",
                             "experience", "amelia_teaching_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    i = 0
    attempts = 0
    while i < N_FACTS and attempts < N_FACTS * 3:
        attempts += 1
        category = category_for(i)
        try:
            question = ask_amelia_for_question(asked_text, category)
        except Exception as exc:
            print(f"  [teacher error] {type(exc).__name__}: {exc}, retrying")
            continue
        key = canonical(question)
        # 2026-10-11: real bad data found, not hypothesized -- "What was" and a mid-sentence cut
        # at "...renamed in 19" both got accepted as real questions and taught to Lucy as genuine
        # Q&A pairs, because neither was empty or >300 chars. A truncated generation (finish_reason
        # "length" cutting off mid-sentence) almost never ends in real sentence-final punctuation;
        # checking for that catches it without needing the raw finish_reason here.
        truncated = bool(question) and question.rstrip()[-1:] not in "?."
        if not question or key in asked_keys or len(question) > 300 or truncated:
            reason = "truncated" if truncated else "duplicate/empty/too-long"
            print(f"  [skip {reason}] {question!r}")
            continue
        asked_keys.add(key)
        asked_text.append(question)

        body = {"messages": [{"role": "user", "content": question}], "stream": False}
        try:
            response = engine.complete(body)
        except Exception as exc:
            print(f"  [complete error on {question!r}] {type(exc).__name__}: {exc}")
            continue
        route = response["vampire_route"]
        donor_answer = response["choices"][0]["message"]["content"]
        try:
            result = engine.accept_experience(route["session_id"], route["experience_id"],
                                               action="accept")
        except Exception as exc:
            print(f"  [accept error] {type(exc).__name__}: {exc}")
            continue
        update = result["weight_update"]
        i += 1
        print(f"[{i}/{N_FACTS}] ({category}) Q: {question}")
        print(f"       A: {donor_answer}")
        print(f"       brick={update.get('brick_index')} new={update.get('is_new_brick')} "
              f"changed_levels={update.get('changed_levels')} n_bricks={update.get('n_bricks')}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question": question, "answer": donor_answer,
                                 "auto_accepted": True, "weight_update": update,
                                 "at_ns": time.time_ns()}, ensure_ascii=False) + "\n")
        taught.append((question, donor_answer))

        if i % SLEEP_EVERY == 0:
            t0 = time.time()
            sleep_result = engine.sleep()
            print(f"       [periodic sleep @ {i}] {sleep_result} ({time.time()-t0:.1f}s)")

    print(f"\n[before sleep] n_bricks={len(engine.cluster.bricks)}")
    sleep_result = engine.sleep()
    print(f"[sleep] {sleep_result}")

    print("\n=== retention check: host-off, all taught questions ===")
    matched_count = 0
    for q, a in taught:
        candidate, error, matched = engine._lucy_candidate(q)
        matched_count += int(matched)
        print(f"  matched={matched!s:5}  {q!r}")
    print(f"\n[final] {matched_count}/{len(taught)} matched host-off; "
          f"n_bricks={len(engine.cluster.bricks)} seen={engine.seen}")


if __name__ == "__main__":
    main()
