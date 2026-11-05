"""Stage B shakedown: Lucy answers diary-derived questions for real, consulting Ornith (her host)
only when her own weights don't already have it, then the gated vote mechanism
(BrickCluster.learn_line(gated=True, confirmed=...)) learns from the result in the SAME turn --
not a batch/offline DPO pass. This is the first real end-to-end run of that loop.

Runs against a SANDBOX COPY of the real checkpoint (amelia_teaches_lucy_mind.SHAKEDOWN_SANDBOX_*)
-- the real mind at amelia_teaches_lucy_mind.npz is never opened by this script. Merge the sandbox
checkpoint into the real one only on explicit instruction.

Host consult goes through the real standalone llama-server driver (server.py's ensure_server(),
port 8102 -- the same one brickcluster_engine.complete() uses via _reference()), NEVER Ollama.
2026-11-04 fix: an earlier version of this script called Ollama directly, which was a mistake --
the user's explicit decision for Stage B was "write a driver that calls Ornith directly, ollama a
unessary failure point." Ollama's own keep-alive then held the ~9.65GB model loaded on a schedule
this script didn't control, which is what left the concurrently-running Stage A extractor without
enough memory and got it killed by the harness's low-memory reaper. This script now starts/stops
the standalone server itself.

Host consultation is UNGROUNDED by default (2026-11-04, reverted back from the 2026-11-03 grounded
experiment): Ornith answers from its own knowledge only, never given the diary's evidence_quote.
The 2026-11-03 version handed Ornith the evidence_quote directly because a fully blind host can't
confirm a project-internal benchmark number it has no way to know, and a wrong first guess into an
empty brick bypasses the vote gate entirely (first-ever teach always proceeds). That made the
relay safe but circular -- Lucy was learning to reproduce quotes fed to her host, through her host,
which is closer to recording than to evolving. The user's correction: "the goal is a free evolving
LLM not a tape recorder" -- an ungrounded host that's sometimes wrong, with Lucy's bricks converging
on the truth across repeated confirmed=True/False exposure via the vote mechanism, IS the real
test, not a failure mode to engineer around. Pass --grounded to run the old evidence_quote-fed mode
for comparison; it is no longer the default.

Confirmation signal is a lexical Jaccard-overlap heuristic against the diary's ground_truth_answer
-- NOT a semantic judge. Deliberately crude and disclosed as such; the raw score is logged on every
item so it can be audited and replaced later. >=CONFIRM_HIGH -> confirmed=True, <=CONFIRM_LOW ->
confirmed=False, else confirmed=None (treated as mere usage, a no-op under the gate).

2026-11-04, generalized beyond the diary corpus: --questions-file/--results-file/--checkpoint let
this same driver run against a different source (e.g. the Shakespeare/Marlowe/Plato/romance
language-skill pilot) with its own separate sandbox checkpoint -- the gated-vote teaching
mechanics don't change per corpus, only the questions and the checkpoint do. For that pilot, run
with --grounded: the goal there is real language exposure (the user's framing: "Ornith is not the
person we are training, we are training Lucy West" -- Ornith's job is relaying/grounding the real
source text, same proxy role as the diary extractor, not supplying its own guesses to be tested
against for correctness). The ungrounded ("ask Ornith from its own knowledge, let the vote gate
correct wrong guesses over time") mode is specifically for testing convergence against private,
project-internal facts a general host could never otherwise know -- not for a literature corpus
where grounded, faithful relay IS the whole point.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

import os
import subprocess
import urllib.request
import uuid as uuid_mod

import lutgtlm_ingest as LI  # noqa: E402
import brickcluster_engine as BCE  # noqa: E402  (import resets LI.STATE to the REAL path as a
# side effect -- see module docstring in brickcluster_engine.py line ~74. We override it again,
# below, AFTER this import, before constructing any engine.)

SANDBOX_CHECKPOINT = BCE.CHECKPOINT_DIR / "amelia_teaches_lucy_mind.SHAKEDOWN_SANDBOX_20261103.npz"
DATASET_DIR = Path(r"C:\Users\Azazel\Desktop\Amelia's Dataset\Diary Dataset")
QUESTIONS_FILE = DATASET_DIR / "diary_questions.jsonl"
RESULTS_FILE = DATASET_DIR / "lucy_shakedown_results.jsonl"
# 2026-11-04: the three paths above are defaults for the diary corpus specifically. --questions-
# file/--results-file/--checkpoint (added for the literature/romance language-skill pilot) let
# this same driver run against a different corpus + its own separate sandbox checkpoint, without
# forking the script -- the gated-vote teaching loop itself doesn't change per corpus.

HOST_TIMEOUT = 180
HOST_MAX_TOKENS = 400

# 2026-11-04 FIX #2: an earlier version of this script called Ollama directly, which was already
# wrong (the user's explicit decision for Stage B was "write a driver that calls Ornith directly,
# ollama a unessary failure point") -- Ollama's own keep-alive held the ~9.65GB model loaded on a
# schedule this script didn't control, starving the concurrently-running Stage A extractor of
# memory and getting it reaped. The first fix routed through server.py's ensure_server(), which
# launches a standalone llama-server.exe -- correct in spirit, but that module hardcodes the binary
# on H:\ (a second drive this script doesn't otherwise depend on). Per the user: inline the launch
# here directly, using the C:\ copy -- one fewer external module, one fewer drive dependency.
HOST_EXE = Path(r"C:\llama.cpp\llama-server.exe")
HOST_MODEL_GGUF = Path(r"F:\LLM\Ornith-1.5-9B-uncensored-GGUF\Ornith-1.5-9B-uncensored.Q8_0.gguf")
HOST_URL = "http://127.0.0.1:8102"
HOST_ALIAS = "vampire-lucywest"  # must match brickcluster_engine.REFERENCE_MODEL exactly


def _host_ready():
    try:
        with urllib.request.urlopen(HOST_URL + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_host():
    """Start the standalone llama-server directly from C:\\llama.cpp if nothing is already
    listening on 8102; verify by alias (the donor ROLE) if something already is, same convention
    as the pre-existing server.py. Returns the Popen handle only if THIS call started it, so the
    caller knows whether it's responsible for stopping it again."""
    if _host_ready():
        with urllib.request.urlopen(HOST_URL + "/v1/models", timeout=5) as r:
            models = json.load(r).get("data", [])
        if not any(HOST_ALIAS == m.get("id") for m in models):
            raise RuntimeError(f"Port 8102 is occupied by a different model; no process was stopped.")
        return None
    if not HOST_EXE.is_file() or not HOST_MODEL_GGUF.is_file():
        raise FileNotFoundError(f"Host binary or model missing: {HOST_EXE}, {HOST_MODEL_GGUF}")
    log_dir = ENGINE_DIR / "data" / "server"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"server_{uuid_mod.uuid4().hex}.log"
    args = [str(HOST_EXE), "-m", str(HOST_MODEL_GGUF), "--host", "127.0.0.1", "--port", "8102",
            "--alias", HOST_ALIAS, "-c", "8192", "-np", "1", "-ngl", "99", "--reasoning", "auto",
            "--no-mmap"]
    # --no-mmap (2026-11-05, real RAM-pressure finding): default llama.cpp mmaps the whole GGUF
    # file to load it and keeps that mapping resident even after copying every tensor to the GPU
    # (-ngl 99 here). Measured directly: with mmap, this 9.53GB Q8_0 file left llama-server's
    # WorkingSet (real physical RSS) at ~10.8GB despite the model living on the GPU (confirmed via
    # nvidia-smi, ~8.7GB VRAM, unchanged by this flag). --no-mmap dropped real RSS to ~1.5GB with
    # identical GPU usage -- this was the dominant cause of the low-memory reaper killing teaching
    # runs tonight, far bigger than anything on Lucy's own (BrickCluster) side.
    with log_path.open("xb") as f:
        proc = subprocess.Popen(args, cwd=str(ENGINE_DIR), env=os.environ.copy(), stdout=f, stderr=f,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    print(f"Starting standalone host from C:\\ (pid {proc.pid}). Log: {log_path}", flush=True)
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError(f"Host exited; inspect {log_path}")
        if _host_ready():
            print(f"Host ready: {HOST_URL}", flush=True)
            return proc
        time.sleep(1)
    raise RuntimeError(f"Host still loading; inspect {log_path}. It was not stopped.")

CONFIRM_HIGH = 0.30
CONFIRM_LOW = 0.08

# 2026-11-03 fix: a host REFUSAL/uncertainty answer ("I can't reliably answer", "doesn't clearly
# specify") scores low Jaccard for the wrong reason -- it isn't asserting anything false, it's
# declining to assert at all. The first real 5-item shakedown run showed this get misclassified as
# confirmed=False (a verified CONTRADICTION), which trained Lucy's brick on the host's hedge text
# as if it were a taught fact. A refusal must always be treated as confirmed=None (mere non-info,
# a no-op), never as a contradiction, regardless of its similarity score.
REFUSAL_MARKERS = (
    "don't have", "doesn't have", "do not have", "doesn't specify", "does not specify",
    "doesn't clearly", "does not clearly", "can't confirm", "cannot confirm", "can't reliably",
    "cannot reliably", "can't tell", "cannot tell", "i don't know", "i do not know",
    "unable to", "not sure", "no way to know", "rather not guess", "won't guess", "wouldn't guess",
    "can't determine", "cannot determine", "isn't clear", "is not clear", "i can't",
    # 2026-11-04 widened: a real result row had Ornith say "the reference material provided
    # doesn't answer your question... doesn't mention any team, a mistake" -- a genuine refusal,
    # none of the markers above matched it, and it scored confirmed=False (a contradiction)
    # instead of confirmed=None. Caught only by the vote gate's stale-belief protection that time
    # (blocked_stale_belief); would NOT have been caught on a brand-new empty brick, since a
    # first-ever teach bypasses the gate regardless of confirmed.
    "doesn't answer", "does not answer", "doesn't mention", "does not mention",
    "doesn't address", "does not address", "no mention of", "doesn't contain", "does not contain",
    "didn't include", "did not include", "didn't mention", "did not mention",
)


def is_refusal(text):
    t = (text or "").lower()
    return any(marker in t for marker in REFUSAL_MARKERS)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "for",
    "and", "or", "with", "that", "this", "it", "as", "by", "at", "from", "but", "not", "no", "if",
    "so", "did", "do", "does", "what", "why", "how", "which", "who", "when", "where", "there",
    "than", "then", "its", "into", "out", "up", "down", "over", "under", "about", "all", "any",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    # NO length filter (2026-11-03 fix): the first real shakedown run showed this project's diary
    # answers are often short numerics/units ("45.58", "ms", "1x", "0.23x") -- a len>2 floor meant
    # to drop connective words was silently discarding exactly the content these answers consist
    # of, scoring a verbatim-correct host answer ("...was 45.58 ms/window") as only 0.11 similar.
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in STOPWORDS}


def jaccard(a, b):
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def ask_ornith(engine, question, context=None):
    if context:
        system = (
            "You are being consulted as a knowledgeable host with access to the following "
            "reference material. Answer the question directly and concisely in 1-3 sentences, "
            "using the reference material when it's relevant. If the material doesn't actually "
            "answer the question, say so plainly instead of guessing.\n\n"
            f"REFERENCE MATERIAL:\n{context}"
        )
    else:
        system = (
            "Answer the question directly and concisely in 1-3 sentences, from your own "
            "general knowledge only. If you genuinely don't know, say so plainly instead of "
            "guessing."
        )
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "max_tokens": HOST_MAX_TOKENS,
    }
    try:
        # engine._reference() is the real production call used by BrickClusterEngine.complete() --
        # same loopback-only validation, same donor model, no separate Ollama policy to fight.
        resp = engine._reference(body)
        return (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip(), None
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def load_questions():
    items = []
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_done_keys():
    done = set()
    if RESULTS_FILE.is_file():
        with RESULTS_FILE.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                done.add(d.get("item_key"))
    return done


def item_key(item):
    return f"{item.get('source_file')}::{item.get('section_header')}::{item.get('question')}"


GROUNDED_MODE = False  # set from --grounded in main() before the loop starts


def process_one(engine, item):
    question = item["question"]
    ground_truth = item.get("ground_truth_answer", "")
    prompt_text = f"User: {question}\nLucy:"

    t0 = time.time()
    candidate, candidate_error, brick_matched, candidate_source = engine._lucy_candidate(question)
    own_weights_hit = bool(brick_matched and candidate and candidate_source == "retrieved")

    host_consulted = False
    host_answer, host_error = None, None
    if own_weights_hit:
        final_answer = candidate
    else:
        host_consulted = True
        ctx = item.get("evidence_quote") if GROUNDED_MODE else None
        host_answer, host_error = ask_ornith(engine, question, context=ctx)
        final_answer = host_answer

    sim = jaccard(final_answer, ground_truth)
    if not final_answer.strip() or is_refusal(final_answer):
        confirmed = None
    elif sim >= CONFIRM_HIGH:
        confirmed = True
    elif sim <= CONFIRM_LOW:
        confirmed = False
    else:
        confirmed = None

    line_text = f"User: {question.strip()[-4096:]}\nLucy: {final_answer.strip()[:2048]}"
    batches = LI._line_batches(line_text)
    result = {
        "item_key": item_key(item), "question": question, "ground_truth_answer": ground_truth,
        "own_weights_hit": own_weights_hit, "lucy_candidate": candidate,
        "lucy_candidate_source": candidate_source, "lucy_candidate_error": candidate_error,
        "host_consulted": host_consulted, "host_answer": host_answer, "host_error": host_error,
        "final_answer_shown_as_lucys": final_answer, "jaccard_sim": round(sim, 4),
        "confirmed": confirmed, "elapsed_s": round(time.time() - t0, 1),
    }
    if not batches:
        result["train_status"] = "line_too_short"
        result["vote_before"] = result["vote_after"] = None
        result["blocked_stale_belief"] = False
        result["flipped"] = False
        result["brick_index"] = None
        return result

    predicted_idx = engine.cluster.route(engine.cluster.tfidf_sparse(line_text), for_training=False)
    vote_before = (engine.cluster.vote_count[predicted_idx]
                   if predicted_idx is not None and predicted_idx < len(engine.cluster.vote_count)
                   else None)
    taught_before = (list(engine.cluster.brick_taught_lines[predicted_idx])
                      if predicted_idx is not None
                      and predicted_idx < len(engine.cluster.brick_taught_lines) else [])
    n_bricks_before = len(engine.cluster.bricks)
    idx2, loss = engine.cluster.learn_line(batches, line_text, lr_scale=BCE.LR_SCALE,
                                            gated=True, confirmed=confirmed)
    vote_after = engine.cluster.vote_count[idx2] if idx2 < len(engine.cluster.vote_count) else None
    blocked = (loss is None and confirmed is False)
    # flipped = a real contradiction that actually trained (loss is not None), on a brick that
    # already had taught content before this call (a first-ever teach always trains regardless of
    # confirmed and isn't a "flip" of anything). NOT "vote_after == 1" -- that's also true of a
    # plain decrement (e.g. 2 -> 1, correctly blocked) and wrongly flagged a blocked call as a
    # flip in the first real 5-item shakedown run.
    flipped = (confirmed is False and loss is not None and bool(taught_before))
    result.update({
        "train_status": "blocked_stale_belief" if blocked else (
            "no_update_usage" if (loss is None and confirmed is None) else "updated"),
        "brick_index": int(idx2), "vote_before": vote_before, "vote_after": vote_after,
        "blocked_stale_belief": blocked, "flipped": flipped,
        "loss": loss, "is_new_brick": idx2 >= n_bricks_before,
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--grounded", action="store_true",
                     help="feed Ornith the diary's evidence_quote (old 2026-11-03 mode); "
                          "default is ungrounded (Ornith answers from its own knowledge only)")
    ap.add_argument("--questions-file", type=str, default=None,
                     help="override QUESTIONS_FILE (default: the diary corpus)")
    ap.add_argument("--results-file", type=str, default=None,
                     help="override RESULTS_FILE (default: the diary corpus's results path)")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="override SANDBOX_CHECKPOINT (default: the diary shakedown sandbox); "
                          "always a SANDBOX path -- never point this at the real mind checkpoint")
    args = ap.parse_args()

    global GROUNDED_MODE, QUESTIONS_FILE, RESULTS_FILE, SANDBOX_CHECKPOINT
    GROUNDED_MODE = args.grounded
    if args.questions_file:
        QUESTIONS_FILE = Path(args.questions_file)
    if args.results_file:
        RESULTS_FILE = Path(args.results_file)
    if args.checkpoint:
        SANDBOX_CHECKPOINT = Path(args.checkpoint)

    LI.STATE = str(SANDBOX_CHECKPOINT)
    print("=" * 70)
    print("LUCY LIVE SHAKEDOWN (Stage B) -- gated vote mechanism, real host consult")
    print(f"Host grounding: {'GROUNDED (evidence_quote fed to host)' if GROUNDED_MODE else 'UNGROUNDED (host answers from its own knowledge)'}")
    print(f"Checkpoint (SANDBOX, never the real mind): {LI.STATE}")
    print(f"Questions:  {QUESTIONS_FILE}")
    print(f"Results:    {RESULTS_FILE}")
    print("=" * 70)

    started_proc = ensure_host()  # None if already running elsewhere; a Popen handle if THIS run
    # just started it -- only then do we stop it ourselves when done, below.
    print(f"Host: {HOST_URL} (standalone llama-server from C:\\, "
          f"{'started by this run' if started_proc else 'already running'})")

    engine = BCE.BrickClusterEngine()
    print(f"Loaded sandbox cluster: {len(engine.cluster.bricks)} bricks, seen={engine.seen} chars")

    items = load_questions()
    done = load_done_keys() if args.resume else set()
    if args.limit:
        items = items[:args.limit]

    counts = {"own_weights": 0, "host_consulted": 0, "confirmed_true": 0, "confirmed_false": 0,
              "confirmed_none": 0, "blocked_stale_belief": 0, "flipped": 0, "errors": 0,
              "line_too_short": 0, "processed": 0}

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with RESULTS_FILE.open("a", encoding="utf-8") as out:
            for i, item in enumerate(items):
                key = item_key(item)
                if key in done:
                    continue
                try:
                    result = process_one(engine, item)
                except Exception as exc:
                    result = {"item_key": key, "question": item.get("question"),
                               "train_status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
                    counts["errors"] += 1
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

                if result.get("own_weights_hit"):
                    counts["own_weights"] += 1
                if result.get("host_consulted"):
                    counts["host_consulted"] += 1
                c = result.get("confirmed")
                if c is True:
                    counts["confirmed_true"] += 1
                elif c is False:
                    counts["confirmed_false"] += 1
                elif "confirmed" in result:
                    counts["confirmed_none"] += 1
                if result.get("blocked_stale_belief"):
                    counts["blocked_stale_belief"] += 1
                if result.get("flipped"):
                    counts["flipped"] += 1
                if result.get("train_status") == "line_too_short":
                    counts["line_too_short"] += 1
                counts["processed"] += 1

                src = result.get("host_answer_source", "own" if result.get("own_weights_hit") else "host")
                print(f"[{i+1}/{len(items)}] ({src}) conf={c} sim={result.get('jaccard_sim')} "
                      f"status={result.get('train_status')} :: {item.get('question','')[:80]}")

                if counts["processed"] % args.save_every == 0:
                    LI.save(engine.cluster, engine.seen, engine.ingests)
                    print(f"  [checkpoint saved, {counts['processed']} processed so far]")

        LI.save(engine.cluster, engine.seen, engine.ingests)
        print("=" * 70)
        print(f"DONE. {json.dumps(counts)}")
        print(f"n_bricks now: {len(engine.cluster.bricks)}")
        print("=" * 70)
    finally:
        # Stop the host ourselves if we're the one who started it -- don't leave a ~9-10GB model
        # resident on an unmanaged timer the way Ollama's keep_alive did (that's what starved the
        # concurrently-running Stage A extractor of memory earlier tonight).
        if started_proc is not None:
            print(f"Stopping standalone host (pid {started_proc.pid}), started by this run...")
            started_proc.terminate()
            try:
                started_proc.wait(timeout=15)
            except Exception:
                started_proc.kill()


if __name__ == "__main__":
    main()
