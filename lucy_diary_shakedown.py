"""
*** REWRITTEN 2026-11-08, Claude -- fixes a real architectural mistake in yesterday's rebuild. ***

Jonathan caught it directly: yesterday's version reinvented its own parallel own-weights/host
logic (my own `_own_weights_lookup`/`_generate_candidate`, my own ask_host()) instead of using the
REAL, already-built, already evidence-tested production engine, brickcluster_engine.
BrickClusterEngine -- found by reading brickcluster_engine.py directly, then cross-checking
against WampYr_Mamba-3's evidence_contract_test_programming_agent.py (GitHub, public, already
proven correct): own-weights-first via `_lucy_candidate()`/`complete()`, with a REAL calibrated
retrieval-cosine-floor (0.5, from an actual measured failure case, not guessed), and -- the
concrete thing I had backwards -- code answers verified by ACTUALLY RUNNING THEM, not by fuzzy
text comparison. "A pretty answer is NOT proof" applies to code exactly as much as to trivia.

What changed from yesterday:
  - Own-weights check + host consult: now `brickcluster_engine.BrickClusterEngine.complete()` --
    the SAME calibrated logic already proven in production, not a reimplementation of it.
  - The one piece of the real engine actually modified: `_reference()`'s donor call normally goes
    over HTTP to a llama-server.exe host (see brickcluster_engine.py's `post_json`/`_reference`).
    Per Jonathan 2026-11-07 ("no Ollama... use PYTHON"), that HTTP call is replaced with the same
    in-process llama-cpp-python call built yesterday, by monkey-patching the module-level
    `post_json` function brickcluster_engine.py itself calls by bare name -- every other line of
    brickcluster_engine.py runs completely unmodified.
  - Verification for CODE answers: real `exec()`, following evidence_contract_test_programming_
    agent.py's exact pattern, instead of jaccard. Jaccard is KEPT, but only as the fallback for
    non-code trivia (diary/NLP/literature), where no execution oracle exists.
  - Training: still goes through brick_cluster_charlm.BrickCluster.learn_line(gated=True,
    confirmed=...) -- the confirmation-gated vote mechanism rebuilt and verified against ~7,700
    real surviving result rows yesterday (lucy_west_memory_architecture_locked_20261102.md). That
    mechanism's correctness is independent evidence from the engine-architecture mistake above: the
    vote_before/vote_after/blocked_stale_belief/flipped fields and their exact train_status mapping
    were read directly off real historical logs, not guessed, and still hold. What was wrong
    yesterday was NOT the vote mechanism -- it was bypassing the real engine's own-weights/host
    logic and using jaccard to verify code. BrickClusterEngine.accept_experience()/learn_accepted()
    itself calls learn_line() UNGATED (no confirmed= at all) -- that path assumes a live human
    makes the accept decision each time, which doesn't exist in this unattended bulk run, so this
    script calls learn_line(gated=True, ...) directly rather than going through accept_experience().

HONEST GAP, stated plainly: real per-problem execution verification (hand-built test cases, like
evidence_contract_test_programming_agent.py's reverse_string test) does not exist for a generic
600K-row dataset -- building that would be its own project. What IS real here: the ground_truth
code is checked to actually parse (ast.parse), and the HOST'S candidate code is actually exec()'d
and must run without crashing to count as verified -- a genuine, checkable pass/fail, not a text
comparison, but narrower than a real per-problem unit-test suite. For non-Python or non-code rows,
jaccard-vs-ground-truth remains the (already-disclosed, still-not-byte-recovered) fallback.
"""
import ast
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
import uuid
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

# 2026-11-08 BUG, found live by the shakedown: the item-215 "crash" that prompted the resource_guard
# work was NOT the only way this script can die mid-run. This run died again at item 215 -- same
# number, DIFFERENT cause -- on a plain UnicodeEncodeError raised by the progress print below
# (`print(f"[{processed_this_run}/...")`), which writes an excerpt of real dataset/model text
# straight to the Windows console's cp1252 encoding. The item's result was already safely written
# to RESULTS_FILE (opened with encoding="utf-8") one line earlier -- only the console print crashed
# -- but an unhandled exception there still kills the whole process. Fix: force stdout/stderr to
# UTF-8 with errors="replace" so an unencodable character degrades to "?" on screen instead of
# taking down the run; the results file's own correctness is untouched by this either way.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import brickcluster_engine as E
import resource_guard as RG
import lutgtlm_ingest as LI  # noqa: F401 -- E already imports/configures this; kept as a direct
# import too since this script calls LI.ARCH/LI._line_batches directly below.

# Same in-process host groundwork as yesterday (REWRITTEN 2026-11-07, per Jonathan: no Ollama).
HOST_MODEL_GGUF = Path(
    r"C:\Users\Azazel\.ollama\models\blobs\sha256-05e0878744b36b7c55385ecb8aaf47bddf0b43f720dbfac5c79e489c02abca8f"
)
HOST_N_CTX = 8192
HOST_N_GPU_LAYERS = -1
_llm = None


def ensure_model():
    """Loads the host model in-process, once. No subprocess, no HTTP server, no Ollama."""
    global _llm
    if _llm is not None:
        return _llm
    if not HOST_MODEL_GGUF.is_file():
        raise FileNotFoundError(f"Host model missing: {HOST_MODEL_GGUF}.")
    from llama_cpp import Llama
    print(f"Loading host model in-process: {HOST_MODEL_GGUF} "
          f"({HOST_MODEL_GGUF.stat().st_size / 1e9:.2f} GB)...", flush=True)
    _llm = Llama(model_path=str(HOST_MODEL_GGUF), n_ctx=HOST_N_CTX,
                 n_gpu_layers=HOST_N_GPU_LAYERS, use_mmap=False, verbose=False)
    print("Host model loaded.", flush=True)
    return _llm


def _strip_reasoning(content):
    """This model's chat template leaks raw <think>...</think> reasoning into content (found live
    2026-11-07) -- keep only text after the LAST </think>, unchanged otherwise."""
    if "</think>" in content:
        return content.rsplit("</think>", 1)[1].strip()
    return content.strip()


HOST_MAX_TOKENS = 2048  # found live, 2026-11-08: the old default of 300 (carried over from
# yesterday's trivia-only prompt) let a reasoning model burn its entire budget on hidden
# <think>...</think> reasoning for a code question and get cut off BEFORE emitting any real
# answer -- the "answer" that came back was literally unfinished reasoning prose, which exec()
# correctly rejected as a SyntaxError (not a model-capability failure, a budget failure). Same
# documented failure mode nlp_extract_questions.py already warned about: "the fix is a bigger
# budget, not a retry" -- this does NOT retry, it gives real room to think before answering.
# SUPERSEDED same day by enable_thinking=False below: even 2048 was not enough for some real
# diary questions (4/187 real-run failures, "host exhausted 2048 tokens on reasoning, never
# reached an answer") -- raising the budget further was just delaying the same failure, not
# fixing it. Left at 2048 now as the ANSWER budget (plenty, since no hidden reasoning competes
# for it anymore), not the reasoning budget.

_chat_formatter = None


def _get_chat_formatter(llm):
    """Builds a Jinja2ChatFormatter directly from this model's own embedded chat template, once.
    Needed because llama_cpp 0.3.35's Llama.create_chat_completion (checked its real signature
    directly, no kwargs passthrough to the template) exposes no way to set `enable_thinking` --
    the one thing this model's chat template actually branches on to suppress reasoning. Confirmed
    live by reading tokenizer.chat_template out of the GGUF's own metadata: a Qwen3-style template
    with `{%- if enable_thinking is defined and enable_thinking is false %}<think>\\n\\n</think>\\n\\n
    {%- else %}<think>\\n{%- endif %}` immediately before the assistant turn -- rendering with
    enable_thinking=False pre-closes the think block IN THE PROMPT, so the model never opens one.

    THE BUG THIS GUARDS AGAINST, explicitly, so it can never silently recur: on 2026-11-07 the
    host was moved from Ollama's HTTP API (where `"think": false` already handled this, see
    lucy_donor_token_level_translator_correction.md) to this in-process call, and the thinking-
    suppression fix was NOT carried over in the move -- nobody decided against it, it was just an
    untransferred fix, and it cost 4/187 real items in the next day's run (host_call_failed,
    "host exhausted 2048 tokens on reasoning, never reached an answer") before being caught and
    re-applied on 2026-11-08 (lucy_diary_shakedown_enable_thinking_fix_20261108.md). A silent
    assumption that `enable_thinking` does something just because this model's template supports
    it is exactly how that regression could repeat under a different disguise -- e.g. if
    HOST_MODEL_GGUF is ever pointed at a different donor model whose template does NOT define
    `enable_thinking` at all, this code would render a prompt where the variable is simply ignored
    (Jinja treats an undefined `enable_thinking` reference inside `is defined` as False, so the
    `else` branch runs and a real `<think>\n` block opens, unsuppressed) -- not an error, just the
    exact same budget-exhaustion bug back again, silently, with no signal pointing at the cause.
    The check below turns that silent failure mode into a loud, fixable one at load time instead:
    """
    global _chat_formatter
    if _chat_formatter is None:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        tmpl = llm.metadata["tokenizer.chat_template"]
        if "enable_thinking" not in tmpl:
            raise RuntimeError(
                "HOST_MODEL_GGUF's chat template does not reference `enable_thinking` -- the "
                "thinking-suppression fix this script relies on (see _get_chat_formatter's "
                "docstring) has no effect on this template, so reasoning would run unsuppressed "
                "and HOST_MAX_TOKENS could again be exhausted on hidden <think> reasoning before "
                "any real answer is produced (the exact 2026-11-08 regression this check exists "
                "to catch, not a hypothetical). If this model genuinely has no thinking mode, "
                "remove this check deliberately; if it DOES support one, find its real toggle "
                "(may not be named enable_thinking) and wire it into _get_chat_formatter before "
                "trusting this script's HOST_MAX_TOKENS budget again."
            )
        _chat_formatter = Jinja2ChatFormatter(template=tmpl, eos_token="<|im_end|>",
                                               bos_token="", add_generation_prompt=True)
    return _chat_formatter


def _inprocess_post_json(url, body, timeout=300):
    """Monkey-patch target for brickcluster_engine.post_json -- _reference() calls this bare name
    expecting an OpenAI-shaped chat-completion dict back from an HTTP POST; this returns the same
    shape from the in-process model instead, so _reference()/complete() run completely unmodified
    otherwise. `url` is accepted and ignored (there is no real endpoint anymore).

    REWRITTEN 2026-11-08 to call enable_thinking=False (see _get_chat_formatter) via the
    low-level create_completion API instead of create_chat_completion -- this is the in-process
    equivalent of the "think": false fix the lost F:-drive version applied to Ornith over Ollama's
    HTTP API (lucy_donor_token_level_translator_correction.md) for the exact same root cause: a
    reasoning model burning its whole token budget on hidden <think> chain-of-thought and getting
    cut off before emitting any real answer. Measured live against a real question that was
    failing in today's run ("conceptual mistake blocking the model of serotonin"): thinking ON ->
    finish_reason=length, truncated mid-reasoning, no answer; thinking OFF -> finish_reason=stop,
    clean real answer in 390 tokens."""
    llm = ensure_model()
    fmt = _get_chat_formatter(llm)
    rendered = fmt(messages=body.get("messages", []), enable_thinking=False)
    max_tokens = body.get("max_tokens", HOST_MAX_TOKENS)
    out = llm.create_completion(
        prompt=rendered.prompt,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
        temperature=body.get("temperature", 0.3),
    )
    choice = out.get("choices", [{}])[0]
    text = choice.get("text", "")
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        # Thinking is suppressed now (no <think> block ever opens), so a length cutoff means the
        # ANSWER ITSELF ran out of room -- a real, different failure than the old reasoning-
        # exhaustion case, not the same bug recurring under a new name.
        raise TimeoutError(
            f"host exhausted {max_tokens} tokens on its answer and got cut off before finishing "
            f"(thinking suppressed via enable_thinking=False -- this is a real answer-length "
            f"budget problem, not reasoning overhead)"
        )
    return {
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": _strip_reasoning(text)}}],
        "usage": out.get("usage", {}),
    }


E.post_json = _inprocess_post_json  # the ONLY behavioral change made to the real engine.


def _judge_answer_relevance(question, answer):
    """2026-11-08, Jonathan directly: a canned guardrail non-answer trained as if it were real
    content is categorically worse than a missing fact, and the prefix-list refusal guard above
    only catches refusals that OPEN with a known phrase -- his own example slips past it entirely:
    asked "what is Shakespeare's greatest play," getting back a suicide-hotline disclaimer, which
    starts with "I'm sorry if you feel..." not any phrase this script already checks for. His own
    proposed fix: Lucy herself isn't smart enough yet to notice a hotline number has nothing to do
    with Shakespeare (true chicken-and-egg -- she needs the judgment to filter bad training, but
    needs good training to develop the judgment) -- but something ELSE already in the pipeline IS
    smart enough: the donor model itself, asked a second, NARROW question instead of trusting its
    own first answer blind. Same "Frozen LLM Judge" pattern already in his own idea-reservoir file
    (parasite_mamba.txt's A-GRAE section) and the same spirit as [[galatea_pair_selfsustaining]]'s
    incorruptible-judge-floor requirement.

    Deliberately narrow and cheap: one yes/no-shaped question, no chain-of-thought needed (already
    running enable_thinking=False), small max_tokens. This is NOT asking the donor to re-answer or
    re-verify factual correctness (jaccard/exec already do that) -- only whether the answer is even
    ABOUT the question's actual subject, which catches a non-sequitur regardless of why it happened
    (a safety disclaimer, a context mix-up, anything), without requiring Lucy's own capability to
    exist at all. Returns True if the answer looks relevant, False if it looks like a mismatch, and
    True (fail open -- do not contaminate training on a failed judge call) if the judge call itself
    errors."""
    judge_prompt = (
        f"Question asked: {question}\n\nProposed answer: {answer}\n\n"
        f"Does the proposed answer actually address the SUBJECT of the question above, even if it "
        f"is wrong, incomplete, or a refusal due to lack of knowledge? Or is it something unrelated "
        f"to the question's actual subject -- for example a generic safety disclaimer, crisis "
        f"hotline, or boilerplate response that has nothing to do with what was actually asked? "
        f"Reply with exactly one word: RELEVANT or MISMATCH."
    )
    try:
        llm = ensure_model()
        fmt = _get_chat_formatter(llm)
        rendered = fmt(messages=[{"role": "user", "content": judge_prompt}], enable_thinking=False)
        out = llm.create_completion(prompt=rendered.prompt, max_tokens=10,
                                     stop=["<|im_end|>"], temperature=0.0)
        verdict = out["choices"][0]["text"].strip().upper()
        return "MISMATCH" not in verdict
    except Exception:
        return True


GROUNDED_MODE = False
SANDBOX_CHECKPOINT = E.CHECKPOINT_DIR / "amelia_teaches_lucy_mind.SHAKEDOWN_SANDBOX_20261103.npz"
DATASET_DIR = Path(r"C:\Users\Azazel\Desktop\Amelia's Dataset")
QUESTIONS_FILE = DATASET_DIR / "diary_questions.jsonl"
RESULTS_FILE = DATASET_DIR / "lucy_shakedown_results.jsonl"

# RECONSTRUCTED fallback for non-code rows only (see module docstring). Unchanged from yesterday.
CONFIRM_JACCARD = 0.35
CONTRADICT_JACCARD = 0.12


def _word_set(s):
    return set(re.findall(r"[a-z0-9']+", (s or "").lower()))


def _jaccard(a, b):
    A, B = _word_set(a), _word_set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)


def _extract_code(text):
    """Same extract_code() pattern as evidence_contract_test_programming_agent.py: pull the first
    fenced code block, or fall back to the whole text if there is no fence."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    return m.group(1).strip() if m else (text or "").strip()


def _is_python_question(ground_truth_answer):
    """RECONSTRUCTED, deliberately narrow: true execution verification needs hand-built test
    cases per problem (what evidence_contract_test_programming_agent.py does); this dataset has
    none. Only treat a row as code-verifiable when the GROUND TRUTH answer itself parses as valid
    Python -- a real, checkable fact (ast.parse succeeding), not a guess about the row's topic."""
    code = _extract_code(ground_truth_answer)
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _verify_runs(candidate_text):
    """REAL exec()-based verification, same "a pretty answer is NOT proof" discipline as
    evidence_contract_test_programming_agent.py's STEP 3. Honest scope: no structured test cases
    exist for this generic dataset, so the strongest check available is "does the candidate code
    actually parse and execute without crashing" -- a real pass/fail, not a hand-built behavioral
    test. Returns (ran_ok, error)."""
    code = _extract_code(candidate_text)
    if not code:
        return False, "no code extracted from candidate"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<lucy_candidate>", "exec"), {})
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def process_item(engine, row):
    """Processes one question row via the REAL engine (BrickClusterEngine.complete() for
    own-weights-check + host-fallback, exactly as production does it), verifies the answer (real
    exec() for Python-code rows, jaccard fallback otherwise), then trains via the confirmation-
    gated vote mechanism (brick_cluster_charlm.BrickCluster.learn_line(gated=True, confirmed=...),
    independently verified against real historical logs yesterday -- see module docstring)."""
    t0 = time.time()
    question = row["question"]
    ground_truth = row.get("ground_truth_answer", "")
    evidence_quote = row.get("evidence_quote", "")
    item_key = f"{row.get('source_file', '')}::{row.get('section_header', '')}::{question}"

    if GROUNDED_MODE and evidence_quote:
        user_content = f"Reference material: {evidence_quote}\n\nQuestion: {question}"
    else:
        user_content = question
    body = {"messages": [{"role": "user", "content": user_content}], "stream": False}
    try:
        response = engine.complete(body)  # the REAL own-weights-check-then-host-fallback call.
    except Exception as exc:
        # _complete_locked() re-raises a failed donor call as RuntimeError (see brickcluster_
        # engine.py's _complete_locked) -- uncaught, that would crash the ENTIRE run on one bad
        # item (e.g. the host running out of its reasoning budget, found live). A single item's
        # host failure must not take down a multi-thousand-item bulk run.
        return {
            "item_key": item_key, "question": question, "ground_truth_answer": ground_truth,
            "own_weights_hit": False, "lucy_candidate": "", "lucy_candidate_source": None,
            "lucy_candidate_error": None, "host_consulted": True, "host_answer": None,
            "host_error": f"{type(exc).__name__}: {exc}", "final_answer_shown_as_lucys": "",
            "verification_method": None, "exec_ran_ok": None, "exec_error": None,
            "jaccard_sim": None, "confirmed": None, "elapsed_s": round(time.time() - t0, 1),
            "train_status": "host_call_failed", "brick_index": None, "vote_before": None,
            "vote_after": None, "blocked_stale_belief": False, "flipped": False, "loss": None,
            "is_new_brick": False, "experience_id": None, "session_id": None,
        }
    route = response["vampire_route"]
    final_answer = response["choices"][0]["message"].get("content") or ""

    own_weights_hit = route["source"] == "lucy_own_weights"
    host_consulted = route["source"] == "frozen_donor"
    lucy_candidate = route["lucy_candidate"]
    lucy_candidate_source = route["lucy_candidate_source"]
    lucy_candidate_error = route["lucy_candidate_error"]

    # 2026-11-08, found live after the full 223-item run: GROUNDED_MODE defaults False, and this
    # run was launched without --grounded, so the host got the bare question with ZERO reference
    # material for a corpus that is entirely private project internals (every one of the 223 rows
    # has a real evidence_quote the host never saw). Measured result: 203/223 host answers were
    # refusals ("I don't have specific information about..."), and 201 of those got trained into
    # a brick as if they were real content -- Lucy was being taught to recite confusion, not facts.
    # This guard stops that from EVER happening silently again, in any mode: a refusal-shaped
    # answer is never fed to learn_line, grounded or not, since a grounded run can still refuse on
    # a genuinely hard item. This does NOT fix the root cause (wrong mode for this corpus) -- that
    # needs a fresh checkpoint re-run with --grounded; this only stops the symptom from corrupting
    # weights in the meantime.
    #
    # Split into two categories same day, per Jonathan directly: a lack-of-knowledge refusal is a
    # different, far less dangerous thing than a SAFETY/GUARDRAIL canned non-answer being trained
    # as if it were real content on a legitimate historical or literary question -- his own
    # examples: real scholarly debate on Carthaginian/Phoenician child sacrifice (stillbirth/birth-
    # defect burial grounds vs. live sacrifice is a genuine, real historical question, not
    # "exploitation content") getting a generic exploitation-of-children refusal trained as the
    # answer; or Romeo and Juliet's actual text getting a suicide-prevention disclaimer trained in
    # as if it were part of the play. Checked tonight's real 223 rows directly: 0 genuine policy
    # refusals occurred (the donor is an explicitly "uncensored" finetune, and the 2 near-matches
    # were honest epistemic caution, not guardrail language) -- this is a real structural gap, not
    # yet a fired incident, closed BEFORE it could produce one rather than after. A policy refusal
    # is NOT just skipped like an epistemic one -- it's tagged distinctly so a human sees it,
    # because the project's own standing position (over_refusal_is_the_disease.md,
    # galatea_manipulate_not_lie.md) is that a corporate-liability non-answer is never equivalent
    # to truth, and silently absorbing one the same way as "I don't know" would hide exactly the
    # case most worth a human looking at.
    _final_lower = final_answer.strip().lower()
    host_epistemic_refused = host_consulted and final_answer.strip().startswith((
        "I don't have", "I do not have", "I'm not certain", "I'm not aware", "I'm not familiar",
        "I am not aware", "I'm unable to", "I don't recognize", "I want to be upfront",
        "It looks like your question is missing", "Could you let me know", "Could you clarify",
    ))
    host_policy_refused = host_consulted and not host_epistemic_refused and any(m in _final_lower for m in (
        "i can't provide", "i cannot provide", "i'm not able to provide", "i won't generate",
        "i cannot create", "i can't create", "i must decline", "i'm not comfortable",
        "against my guidelines", "content policy", "i can't assist with", "i cannot assist with",
        "i'm unable to provide", "i don't feel comfortable", "this could be harmful",
        "i can't help with that", "i cannot help with that", "exploitation of children",
        "i'm not going to provide", "i won't be able to help with that",
    ))
    # Catches what the prefix-list checks above structurally cannot: a mismatch that doesn't open
    # with any known refusal phrase (Jonathan's own example -- a suicide-hotline disclaimer in
    # response to a Shakespeare question starts "I'm sorry if you feel..."). Only spends the extra
    # judge call on host-consulted, non-refused rows -- an already-detected refusal doesn't need a
    # second opinion, and own-weights hits never reach the host at all.
    host_mismatch = (host_consulted and not host_epistemic_refused and not host_policy_refused
                      and not _judge_answer_relevance(question, final_answer))
    host_refused = host_epistemic_refused or host_policy_refused or host_mismatch

    is_code = _is_python_question(ground_truth)
    if is_code and host_consulted:
        verification_method = "exec"
        ran_ok, exec_error = _verify_runs(final_answer)
        confirmed = ran_ok  # a real pass/fail, never ambiguous -- unlike jaccard, there is no
        # "ambiguous" middle ground for "did it crash".
        jaccard_sim = round(_jaccard(final_answer, ground_truth), 4)  # still logged, diagnostic
        # only -- NOT what confirmed is derived from for this row.
    else:
        verification_method = "jaccard"
        ran_ok, exec_error = None, None
        jaccard_sim = round(_jaccard(final_answer, ground_truth), 4)
        confirmed = (True if jaccard_sim >= CONFIRM_JACCARD
                     else False if jaccard_sim <= CONTRADICT_JACCARD else None)

    # Already known from her own weights: production's own rule (LUCY_SKIP_DONOR_WHEN_MATCHED)
    # is that a verbatim-retrieved answer is handed back WITHOUT any new training -- nothing to
    # confirm, contradict, or train, she already had it.
    if own_weights_hit:
        train_status, idx, loss = "already_known_no_training_needed", None, None
        vote_before = vote_after = None
        is_new_brick = False
    elif host_refused:
        # Distinct status strings, not just one -- a policy refusal or a judge-caught mismatch on
        # a legitimate question is worth a human's eyes (it means the donor itself produced
        # something actively wrong, not merely absent); a lack-of-knowledge refusal is routine.
        if host_policy_refused:
            train_status = "host_policy_refused_FLAG_FOR_REVIEW"
        elif host_mismatch:
            train_status = "host_answer_mismatch_FLAG_FOR_REVIEW"
        else:
            train_status = "host_refused_not_trained"
        idx, loss = None, None
        vote_before = vote_after = None
        is_new_brick = False
    else:
        taught_line = f"User: {question}\nLucy: {final_answer.strip()[:2048]}"
        batches = LI._line_batches(taught_line)
        probe_idx = engine.cluster.route(engine.cluster.tfidf_sparse(taught_line),
                                          for_training=False)
        # route(for_training=False) only ever returns None on a genuinely empty cluster (zero
        # bricks yet -- confirmed by reading route() directly); a fresh spawn always starts at
        # vote_count=1 (_spawn()'s own init).
        is_new_brick = (probe_idx is None
                        or len(engine.cluster.brick_taught_lines[probe_idx]) == 0)
        vote_before = (1 if probe_idx is None or probe_idx >= len(engine.cluster.vote_count)
                        else engine.cluster.vote_count[probe_idx])
        confirmed_for_gate = True if is_new_brick else confirmed
        if batches:
            idx, loss = engine.cluster.learn_line(batches, taught_line, lr_scale=1.0, gated=True,
                                                   confirmed=confirmed_for_gate)
        else:
            idx, loss = probe_idx, None  # too short to form even one training window.
        vote_after = (vote_before if idx is None or idx >= len(engine.cluster.vote_count)
                       else engine.cluster.vote_count[idx])
        if loss is not None:
            train_status = "updated"
        elif confirmed_for_gate is False:
            train_status = "blocked_stale_belief"
        else:
            train_status = "no_update_usage"

    blocked_stale_belief = train_status == "blocked_stale_belief"
    flipped = bool((not own_weights_hit) and confirmed is False and loss is not None)

    return {
        "item_key": item_key,
        "question": question,
        "ground_truth_answer": ground_truth,
        "own_weights_hit": own_weights_hit,
        "lucy_candidate": lucy_candidate,
        "lucy_candidate_source": lucy_candidate_source,
        "lucy_candidate_error": lucy_candidate_error,
        "host_consulted": host_consulted,
        "host_answer": final_answer if host_consulted else None,
        "host_error": None,
        "final_answer_shown_as_lucys": final_answer,
        "verification_method": verification_method,
        "exec_ran_ok": ran_ok,
        "exec_error": exec_error,
        "jaccard_sim": jaccard_sim,
        "confirmed": confirmed,
        "elapsed_s": round(time.time() - t0, 1),
        "train_status": train_status,
        "brick_index": idx,
        "vote_before": vote_before,
        "vote_after": vote_after,
        "blocked_stale_belief": blocked_stale_belief,
        "flipped": flipped,
        "loss": loss,
        "is_new_brick": is_new_brick,
        "experience_id": route.get("experience_id"),
        "session_id": route.get("session_id"),
    }


def _atomic_save(engine, path):
    """Same atomic-write fix as yesterday -- np.savez() direct-to-final-path is exactly the call
    that crashed with OSError when the drive vanished mid-write during the original incident."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp_{uuid.uuid4().hex}.npz")
    state = engine.cluster.export_state()
    np.savez(tmp_path, ARCH=LI.ARCH, seen=engine.seen, ingests=engine.ingests, **state)
    os.replace(tmp_path, path)


def main():
    # VERIFIED (original lines ~339-374, exact):
    import argparse
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

    E.LI.STATE = str(SANDBOX_CHECKPOINT)
    print("=" * 70)
    print("LUCY LIVE SHAKEDOWN (Stage B) -- real BrickClusterEngine, gated vote mechanism")
    print(f"Host grounding: {'GROUNDED' if GROUNDED_MODE else 'UNGROUNDED'}")
    print(f"Checkpoint (SANDBOX, never the real mind): {E.LI.STATE}")
    print(f"Questions:  {QUESTIONS_FILE}")
    print(f"Results:    {RESULTS_FILE}")
    print("=" * 70)

    # 2026-11-08: added after this exact script died at 215/223 with exit code 1 and no captured
    # traceback (stdout had been redirected to a file that no longer exists). Checked for outside
    # interference first (Windows event logs, OOM-killer events, crash dumps, disk health) -- found
    # NONE, which points at an ordinary unhandled exception rather than something external. Most
    # likely real cause: several heavy diagnostic scripts were run concurrently with this pipeline
    # that same window, competing for RAM. Startup floor is checked ONCE here, before the ~9.5GB
    # host model even loads -- a tight-but-survivable startup would otherwise look fine for the
    # first several items and only fail later once memory fragments further.
    MIN_FREE_RAM_STARTUP_GB = 6.0
    free_gb = RG.available_gb()
    if free_gb < MIN_FREE_RAM_STARTUP_GB:
        raise RuntimeError(
            f"Only {free_gb:.1f}GB RAM free, below the {MIN_FREE_RAM_STARTUP_GB}GB startup floor "
            f"(host model alone is ~9.5GB). Close other processes before starting -- this is the "
            f"exact resource-contention class that killed this script's 2026-11-08 run silently; "
            f"refusing to start into the same hole rather than fail later with no traceback."
        )
    RG.mark_running("lucy_diary_shakedown")

    ensure_model()
    engine = E.BrickClusterEngine()

    done_keys = set()
    if args.resume and RESULTS_FILE.is_file():
        with RESULTS_FILE.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_keys.add(json.loads(line)["item_key"])
                except Exception:
                    continue
        print(f"Resuming: {len(done_keys)} items already done.")

    rows = []
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MIN_FREE_RAM_PER_ITEM_GB = 2.0  # lower than the startup floor on purpose -- this is a running-
    # low backoff during a long job, not a refusal to even start; the model's own ~9.5GB is already
    # resident by this point and isn't part of what's draining further.
    processed_this_run = 0
    try:
        with RESULTS_FILE.open("a", encoding="utf-8") as out:
            for row in rows:
                if args.limit is not None and processed_this_run >= args.limit:
                    break
                item_key = f"{row.get('source_file', '')}::{row.get('section_header', '')}::{row['question']}"
                if item_key in done_keys:
                    continue

                free_gb = RG.available_gb()
                backoff_tries = 0
                while free_gb < MIN_FREE_RAM_PER_ITEM_GB and backoff_tries < 6:
                    print(f"  [low RAM: {free_gb:.1f}GB free, below {MIN_FREE_RAM_PER_ITEM_GB}GB "
                          f"floor -- pausing 10s before this item, try {backoff_tries + 1}/6]",
                          flush=True)
                    time.sleep(10)
                    free_gb = RG.available_gb()
                    backoff_tries += 1

                # 2026-11-08: this call used to be unguarded -- ANY exception (a resource failure,
                # a data edge case, anything) killed the whole process with no row written and no
                # traceback captured (the exact failure mode that lost the 2026-11-08 215/223 run).
                # Now: any exception is caught, logged as a real row with its full traceback, and
                # the run continues to the next item instead of dying silently.
                try:
                    result = process_item(engine, row)
                except Exception as exc:
                    result = {
                        "item_key": item_key, "question": row.get("question"),
                        "ground_truth_answer": row.get("ground_truth_answer"),
                        "train_status": "exception",
                        "exec_error": f"{type(exc).__name__}: {exc}",
                        "exception_traceback": traceback.format_exc(),
                        "free_ram_gb_at_failure": RG.available_gb(),
                    }
                    print(f"  [EXCEPTION on this item, logged and continuing: "
                          f"{type(exc).__name__}: {exc}]", flush=True)

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                processed_this_run += 1

                print(f"[{processed_this_run}/{len(rows) - len(done_keys)}] "
                      f"({'host' if result.get('host_consulted') else 'own'}) "
                      f"via={result.get('verification_method')} conf={result.get('confirmed')} "
                      f"status={result.get('train_status')} :: {str(result.get('question'))[:80]!r}",
                      flush=True)

                if processed_this_run % args.save_every == 0:
                    _atomic_save(engine, SANDBOX_CHECKPOINT)
                    print(f"  [checkpoint saved, {processed_this_run} processed so far]", flush=True)

        _atomic_save(engine, SANDBOX_CHECKPOINT)
        print("=" * 70)
        print(f"Done. {processed_this_run} new items processed this run; "
              f"{len(done_keys) + processed_this_run} total done; "
              f"{len(engine.cluster.vote_count)} bricks.")
    finally:
        RG.mark_stopped()


if __name__ == "__main__":
    main()
