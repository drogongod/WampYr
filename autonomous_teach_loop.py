"""The "fallback, cheap, continuous" background learning loop (2026-10-29), Jonathan's own
long-standing design, only just actually built: "It was always an idea as a fallback if the local
learning failed we could use normal back propagation on a tiny single brick. Continuously train the
model piece by piece check by check. The whole idea is as a very cheap process to have continuous
run learning." Confirmed via `cross_system_bug_checklist`-style grep earlier the same night that no
autonomous/background training loop existed anywhere in the engine -- every learning event required
a live human to explicitly type a chat turn, or run a bulk script by hand.

NOT a new learning mechanism. Every single thing this loop does is a call to
BrickClusterEngine.accept_experience(..., action="accept") -- the SAME already-tested, already-live
path a human uses by typing "y" in talk_to_lucy.py. "accept" (never "overlord") is the only action
ever used here, by design: it can only ever train on the donor's OWN real logged answer
(experience["donor_answer"]), never fabricate or substitute new content -- this loop automates
WHETHER something unreviewed gets absorbed, never WHAT gets absorbed.

Safety properties, all deliberate:
 - GRACE PERIOD (AUTO_TEACH_GRACE_S): only considers an experience once it has sat pending for a
   while -- gives a human actually chatting right now the normal window to explicitly reject or
   correct it first, same spirit as not grabbing something out from under an in-progress decision.
 - ONE AT A TIME (AUTO_TEACH_MAX_PER_CYCLE): never bursts through a backlog -- matches "train one
   brick at a time," and keeps any single cycle's compute cost tiny regardless of backlog size.
 - RACE-SAFE: accept_experience() itself re-validates the experience is still pending and
   undecided at call time (the same check a concurrent human /v1/feedback call would also hit) --
   if a human beat this loop to a decision in the meantime, the call raises ValueError and this
   loop just logs and skips it, never double-trains or overrides a human's own choice.
 - SEPARATELY AUDITABLE: every auto-accept this loop makes is ALSO appended to its own
   AUTONOMOUS_LOG (distinct from the human_feedback.jsonl a real human action writes to), so it is
   always possible later to tell which facts a human reviewed vs. which were absorbed unattended --
   transparency applied prospectively, not just incident disclosure after the fact.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

AUTO_TEACH_ENABLED = True       # module-level kill switch -- set False (or set the env var
# VAMPIRE_AUTO_TEACH=0, checked in start()) to disable without removing the code.
AUTO_TEACH_INTERVAL_S = 60      # how often to check for newly-eligible pending experiences.
AUTO_TEACH_GRACE_S = 300        # an experience must have sat pending this long before being
# eligible -- see module docstring's "GRACE PERIOD" note.
AUTO_TEACH_MAX_PER_CYCLE = 1    # see module docstring's "ONE AT A TIME" note.


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def _eligible_candidates(experience_log: Path, grace_s: float, now_ns: int,
                          min_at_ns: int | None = None) -> list[dict]:
    """Same two conditions accept_experience() itself enforces (pending status, no existing
    human_feedback row for this id) PLUS the grace-period age gate -- mirrors, does not bypass,
    the real engine's own eligibility rule. min_at_ns, when set, additionally excludes anything
    logged BEFORE that instant -- see start()'s own comment: a real ~18-day-old, 1277-item backlog
    (the 2026-10-11/12 Amelia-voiced MERGE_SLACK calibration run, legitimate project history but
    of unclear redundancy against the current checkpoint) was found sitting in the real log the
    first time this ran live; Jonathan did not recognize it and asked to hold off, so the loop was
    changed to only ever act on experiences logged AFTER it started, never an inherited backlog,
    until a human explicitly decides what to do with the old one."""
    rows = list(_iter_jsonl(experience_log))
    decided_ids = {row.get("experience_id") for row in rows if row.get("kind") == "human_feedback"}
    out = []
    for row in rows:
        if row.get("kind") == "human_feedback":
            continue
        if row.get("status") != "untrusted_pending_human_or_judge":
            continue
        if row.get("id") in decided_ids:
            continue
        at_ns = row.get("at_ns")
        if not isinstance(at_ns, int) or (now_ns - at_ns) < grace_s * 1e9:
            continue
        if min_at_ns is not None and at_ns < min_at_ns:
            continue
        out.append(row)
    out.sort(key=lambda r: r.get("at_ns", 0))   # oldest-pending-first, fair FIFO ordering
    return out


def _run_cycle(engine, experience_log: Path, autonomous_log, min_at_ns: int | None) -> int:
    candidates = _eligible_candidates(experience_log, AUTO_TEACH_GRACE_S, time.time_ns(), min_at_ns)
    trained = 0
    for exp in candidates:   # iterate the FULL list, not a fixed slice -- 2026-10-29 bug fix: the
        # original version sliced candidates[:AUTO_TEACH_MAX_PER_CYCLE] BEFORE trying any of them,
        # so a single permanently-failing item (found live: the real oldest pending row had an
        # empty donor_answer, which accept_experience() correctly refuses) occupied that one slot
        # forever and the loop could never advance to anything behind it -- confirmed live, 34
        # consecutive cycles, zero real trains, same experience_id skipped every single time.
        if trained >= AUTO_TEACH_MAX_PER_CYCLE:
            break
        session_id, experience_id = exp.get("session_id"), exp.get("id")
        try:
            result = engine.accept_experience(session_id, experience_id, action="accept")
        except ValueError as exc:
            # Either already decided by a human in the meantime (the race-safety this is SUPPOSED
            # to hit), or permanently untrainable (e.g. empty donor_answer) -- either way, log and
            # move on to the NEXT candidate in this same cycle rather than stopping here.
            autonomous_log({"kind": "autonomous_skip", "experience_id": experience_id,
                             "reason": f"{type(exc).__name__}: {exc}", "at_ns": time.time_ns()})
            continue
        autonomous_log({"kind": "autonomous_auto_accept", "experience_id": experience_id,
                         "session_id": session_id, "prompt": exp.get("prompt"),
                         "donor_answer": exp.get("donor_answer"), "weight_update": result,
                         "at_ns": time.time_ns()})
        trained += 1
    return trained


def _loop(engine, experience_log: Path, autonomous_log, stop_event: threading.Event,
          min_at_ns: int):
    print(f"[autonomous_teach_loop] started -- checking every {AUTO_TEACH_INTERVAL_S}s, "
          f"grace period {AUTO_TEACH_GRACE_S}s, max {AUTO_TEACH_MAX_PER_CYCLE} fact/cycle, "
          f"only considering experiences logged after this start (no inherited backlog)",
          flush=True)
    while not stop_event.is_set():
        try:
            with engine.reload_lock, engine.inference_lock:
                n = _run_cycle(engine, experience_log, autonomous_log, min_at_ns)
            if n:
                print(f"[autonomous_teach_loop] auto-accepted {n} pending fact(s) this cycle",
                      flush=True)
        except Exception as exc:   # a cycle failing must never kill the whole background loop --
            print(f"[autonomous_teach_loop] cycle error (will retry next cycle): "
                  f"{type(exc).__name__}: {exc}", flush=True)
        stop_event.wait(AUTO_TEACH_INTERVAL_S)


def start(engine, experience_log: Path, autonomous_log_path: Path) -> threading.Event | None:
    """Spawns the loop as a daemon thread; returns the stop_event (call .set() to stop it), or
    None if disabled. Daemon=True so it never blocks process exit.

    2026-10-29, real live finding: the first time this ran against the real experience log, it
    found 1277 real-but-18-day-old pending experiences (a legitimate prior MERGE_SLACK calibration
    run, NOT contamination -- see _eligible_candidates' own comment) that Jonathan did not
    recognize and asked to hold off on. min_at_ns is captured as THIS CALL's own start time, so the
    loop only ever acts on experiences that become pending from this point forward -- an inherited
    backlog, however old or large, is never auto-consumed without a separate, explicit decision."""
    import os
    if not AUTO_TEACH_ENABLED or os.environ.get("VAMPIRE_AUTO_TEACH") == "0":
        print("[autonomous_teach_loop] disabled (AUTO_TEACH_ENABLED=False or "
              "VAMPIRE_AUTO_TEACH=0)", flush=True)
        return None

    def autonomous_log(record: dict) -> None:
        autonomous_log_path.parent.mkdir(parents=True, exist_ok=True)
        with autonomous_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    min_at_ns = time.time_ns()
    stop_event = threading.Event()
    thread = threading.Thread(target=_loop,
                               args=(engine, experience_log, autonomous_log, stop_event, min_at_ns),
                               daemon=True, name="autonomous-teach-loop")
    thread.start()
    return stop_event
