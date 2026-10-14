"""LucyWest interaction route, running on WAMPYR -- a quinary BrickCluster as her own native,
continuously self-training weights. NOT a Mamba/SSM architecture; see the honest statement below.

ARCHITECTURE, STATED PLAINLY (no SSM anywhere in this file -- verified, not asserted):
  - brick_cluster_charlm.py's BrickCluster is a PROVEN substrate: content-routed, independently
    full-BPTT-trained quinary bricks with an LSL-style density gate, growth, and consolidation.
    Real result at production scale (D:\\flybrain session, 2026-10-10): settled to 134 stable
    bricks on a real 4.2M-char corpus, five real root-cause bugs found and fixed, 6.9x speed fix
    verified. Real result at this project's own scale (F:\\my-agent-vampire-lucy, 2026-10-12): a
    clean 1000-fact teaching run retained 71.1% host-off (own weights only, no donor), vs a 47.3%
    baseline before tuning -- see data/brickcluster/experience/amelia_teaching_log.jsonl.
  - Each brick's own recurrence (native_quinary_charlm.py: Win/Wh/Wgx/Wgh/Wo) is a small, quinary
    -quantized GATED RECURRENT CELL -- structurally closest to a minimal GRU -- NOT a state-space
    model. No input-dependent selective B/C projection, no discretized state-transition matrix, no
    continuous-time Delta step. This is a different, simpler recurrence than Mamba's, by design,
    not a mislabeled copy of it.
  - This substrate keeps the SAME donor-vampire MECHANISM Wampyr.txt specifies (own weights
    first, frozen donor as fallback, explicit human accept/overlord before any weight update) --
    only the recurrence inside "Lucy's own weights" is Wampyr's, not an SSM's.

WHAT IS HONESTLY DIFFERENT FROM AN SSM-BASED SUBSTRATE, NOT HIDDEN:
  - BrickCluster has NO per-session recurrent SSM state. There is no hidden vector that carries
    forward turn-to-turn the way a real Mamba SSM state does. This engine keeps an EXPLICIT
    rolling text buffer per session (the last few turns, concatenated into the prompt) as an
    honest, much cruder stand-in -- a bounded window, not a compressed fold. Turns that age out
    of the window are dropped, not compressed. It is never reported as a recurrent state; see
    SessionContext below.
  - BrickCluster is now TRUE byte-level (2026-10-10, V=49->256 in lutgtlm_ingest.py): enc()
    encodes real text to its raw UTF-8 bytes directly, no case-folding, no dropped characters.
    generate()'s raw output is a Latin-1-passthrough string (see ITOS/bytes_to_text in
    lutgtlm_ingest.py) and MUST be passed through LI.bytes_to_text() before being treated as
    real text -- _lucy_candidate already does this; any new caller of cluster.generate() must too.
  - "Changed weights" evidence here means changed quinary LEVELS (-2..2) in the routed-to
    brick's Win/Wh/Wgx/Wgh/Wo arrays -- diffed before/after a real learn_batch call, not asserted.

THE EVIDENCE CONTRACT THIS ENGINE MUST STILL MEET (F:\\Wampyr.txt, unchanged by the
substrate swap): "Lucy learned X" requires host-off behavioral evidence; "weights changed"
requires exact tensor/level deltas; "Lucy retained X" requires testing after unrelated learning.
Nothing in this file claims learning happened without that evidence -- see
evidence_contract_test.py for the actual non-negotiable experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

import lutgtlm_ingest as LI
from brick_cluster_charlm import BrickCluster

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "brickcluster"
STATE_DIR = DATA / "state"
CHECKPOINT_DIR = DATA / "checkpoints"
EXPERIENCE_LOG = DATA / "experience" / "teacher_experience_untrusted.jsonl"

# CRITICAL: lutgtlm_ingest.py's own default STATE points at Galatea's REAL persistent mind
# (C:/galatea_gtlm/lutgtlm/mind.npz). Override it immediately, before any load/save call, so
# this engine can never touch that file -- this Lucy build gets its own, separate checkpoint.
LI.STATE = str(CHECKPOINT_DIR / "lucy_mind.npz")

REFERENCE_MODEL = "vampire-lucywest"
REFERENCE_URL = os.environ.get("VAMPIRE_REFERENCE_URL", "http://127.0.0.1:8102")
MAX_HOST_TURNS = max(1, int(os.environ.get("VAMPIRE_HOST_TURNS", "1")))
MAX_SESSION_TURNS = max(1, int(os.environ.get("VAMPIRE_SESSION_TURNS", "6")))
LR_SCALE = float(os.environ.get("VAMPIRE_BRICK_LR_SCALE", "1.0"))

WEIGHT_ARRAY_NAMES = ("Win", "Wh", "Wgx", "Wgh", "Wo")  # level-restricted (-2..2); Wemb excluded,
# same reason the ternary engine's own trit count only covers the actually-quantized arrays.


def post_json(url: str, body: dict, timeout: float = 300) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SessionContext:
    """Explicit rolling-text stand-in for session memory -- NOT a recurrent SSM state. Just the
    last MAX_SESSION_TURNS (role, text) pairs, persisted as plain JSON per session. Honest about
    what it is: a context window, same category as the ternary engine's own donor-side
    MAX_HOST_TURNS windowing, not a claim of persistent latent state."""

    @staticmethod
    def _path(session_id: str) -> Path:
        key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return STATE_DIR / f"session_{key}.json"

    @classmethod
    def load(cls, session_id: str) -> list[dict]:
        path = cls._path(session_id)
        if not path.is_file():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    @classmethod
    def save(cls, session_id: str, turns: list[dict]) -> Path:
        path = cls._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        turns = turns[-MAX_SESSION_TURNS:]
        partial = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.partial")
        partial.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")
        os.replace(partial, path)
        return path

    @staticmethod
    def render(turns: list[dict]) -> str:
        return "".join(f"\n{t['role']}: {t['text']}\n" for t in turns)


class BrickClusterEngine:
    """Same external contract as mamba3_ternary_gateway.TernaryMamba3Engine (complete,
    accept_experience, learn_accepted, observe_correction, health) so the donor-routing/feedback
    plumbing is substrate-agnostic. Internals use BrickCluster instead of a ternary Mamba-3 SSM."""

    def __init__(self):
        self.reload_lock = threading.RLock()
        self.inference_lock = threading.RLock()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.cluster, self.seen, self.ingests = LI.load_or_init()
        self.checkpoint_path = Path(LI.STATE)
        self.checkpoint_sha256 = (sha256_file(self.checkpoint_path)
                                   if self.checkpoint_path.is_file() else None)
        self.reload_error = None
        self._taught_exact_lines = self._load_taught_exact_lines()

    @staticmethod
    def _load_taught_exact_lines() -> set[str]:
        """Rebuilds the exact-text dedup set from the human_feedback journal (not a new side
        file -- reuses the existing persisted record rather than adding bookkeeping) so a
        restarted engine still skips exact repeats, not just within one process lifetime."""
        path = DATA / "experience" / "human_feedback.jsonl"
        seen = set()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                prompt, answer = row.get("prompt"), row.get("answer")
                if isinstance(prompt, str) and isinstance(answer, str):
                    seen.add(f"User: {prompt.strip()[-4096:]}\nLucy: {answer.strip()[:2048]}")
        return seen

    # ---- generation (the "own weights, host-off" attempt) ----

    def _real_margin(self, features) -> tuple[int | None, float]:
        """BrickCluster.route(for_training=False) ALWAYS returns the best-available brick once
        ANY brick exists -- "inference never spawns, always returns the best-available brick" is
        correct behavior for actually generating, but means `idx is not None` is NOT a confidence
        signal once n_bricks>=1 (found 2026-10-10 teaching unrelated facts: it reported "matched"
        for "2+2" and "who wrote Romeo and Juliet" immediately after only ever learning about
        England's capital -- there was no second brick to be confused with, it was reporting
        "a brick exists" not "this brick fits"). Recomputes the SAME margin route() uses
        internally so the caller can apply the SAME margin>0 threshold route() uses when
        for_training=True, without triggering a spawn."""
        if not self.cluster.bricks:
            return None, float("-inf")
        import brick_cluster_charlm as bcc
        gvar = self.cluster._global_var()
        neg_ld = (self.cluster.neg.log_density(features, bcc.VAR_FLOOR_FRAC, gvar)
                  if self.cluster.neg.n >= 2 else float("-inf"))
        margins = [pos.log_density(features, bcc.VAR_FLOOR_FRAC, gvar) - neg_ld
                   for pos in self.cluster.brick_pos]
        best_idx = int(np.argmax(margins))
        return best_idx, margins[best_idx]

    def _lucy_candidate(self, prompt_text: str) -> tuple[str, str | None, bool]:
        """Returns (candidate_text, error, brick_matched). brick_matched=True requires a REAL
        positive margin (the same threshold route() uses to decide whether to spawn during
        training), not merely "some brick exists" -- see _real_margin's docstring for the bug
        this fixes."""
        seed_text = f"User: {prompt_text}\nLucy:"
        try:
            features = self.cluster.tfidf_sparse(seed_text)
            idx, margin = self._real_margin(features)
            if idx is None or margin <= 0.0:
                return "", None, False
            ids = LI.enc(seed_text)[-121:]
            if len(ids) < 3:
                return "", None, False
            raw = self.cluster.generate(ids, LI.ITOS, seed_text, n=96, temp=0.0)
            out = LI.bytes_to_text(raw)
            return out.strip(), None, True
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}", False

    def _reference(self, body: dict) -> dict:
        donor_url = body.get("vampire_donor_url", REFERENCE_URL)
        forwarded = dict(body)
        for key in ("vampire_session_id", "vampire_donor_url"):
            forwarded.pop(key, None)
        forwarded["model"] = REFERENCE_MODEL
        forwarded["stream"] = False
        messages = body.get("messages", [])
        if isinstance(messages, list):
            stable = [m for m in messages if isinstance(m, dict)
                      and m.get("role") in {"system", "developer"}]
            dialogue = [m for m in messages if isinstance(m, dict)
                        and m.get("role") not in {"system", "developer"}]
            forwarded["messages"] = stable + dialogue[-(MAX_HOST_TURNS * 2 + 1):]
        if not isinstance(donor_url, str) or not donor_url.startswith(
                ("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("vampire_donor_url must be a local loopback endpoint")
        return post_json(donor_url.rstrip("/") + "/v1/chat/completions", forwarded)

    def complete(self, body: dict) -> dict:
        with self.reload_lock, self.inference_lock:
            return self._complete_locked(body)

    def _complete_locked(self, body: dict) -> dict:
        if not isinstance(body, dict) or body.get("stream"):
            raise ValueError("Lucy accepts a JSON chat request with stream=false")
        session_id = body.get("vampire_session_id") or uuid.uuid4().hex
        turns = SessionContext.load(session_id)
        messages = body.get("messages", [])
        prompt_text = ""
        for message in reversed(messages):
            if (isinstance(message, dict) and message.get("role") == "user"
                    and isinstance(message.get("content"), str)):
                prompt_text = message["content"]
                break
        context_text = SessionContext.render(turns)
        full_prompt = context_text + prompt_text
        candidate, candidate_error, brick_matched = self._lucy_candidate(full_prompt)
        try:
            response = self._reference(body)
            source = "frozen_donor"
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Frozen donor request failed: {type(exc).__name__}: {exc}") from exc
        try:
            answer = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Donor returned no chat-completion answer") from exc
        turns = turns + [{"role": "user", "text": prompt_text}]
        if answer:
            turns.append({"role": "donor", "text": answer})
        state_path = SessionContext.save(session_id, turns)
        experience = {"id": uuid.uuid4().hex, "session_id": session_id, "prompt": prompt_text,
                      "donor_answer": answer, "source": REFERENCE_MODEL,
                      "status": "untrusted_pending_human_or_judge",
                      "checkpoint_sha256": self.checkpoint_sha256, "at_ns": time.time_ns()}
        _append_jsonl(EXPERIENCE_LOG, experience)
        response["model"] = "lucywest-brickcluster-donor-assisted"
        response["vampire_route"] = {
            "source": source, "architecture": LI.ARCH,
            "architecture_family": "wampyr_quinary_gated_rnn_brickcluster",
            "substrate_is_mamba3": False, "substrate_is_ssm": False,
            "n_bricks": len(self.cluster.bricks),
            "checkpoint": str(self.checkpoint_path), "checkpoint_sha256": self.checkpoint_sha256,
            "session_id": session_id,
            "session_memory_kind": "explicit_rolling_text_buffer_not_recurrent_state",
            "lucy_own_weights_checked_before_donor": True, "lucy_brick_matched_existing": brick_matched,
            "lucy_candidate": candidate, "lucy_candidate_error": candidate_error,
            "donor_answer_is_not_lucy_generation": True,
            "state_file": str(state_path), "donor_model": REFERENCE_MODEL,
            "experience_id": experience["id"], "experience_requires_acceptance": True,
        }
        return response

    # ---- training (only on explicit human accept/overlord) ----

    def accept_experience(self, session_id: str, experience_id: str, *,
                           action: str, answer: str | None = None) -> dict:
        if not isinstance(action, str) or action not in {"accept", "overlord"}:
            raise ValueError("feedback action must be 'accept' or 'overlord'")
        if (not isinstance(session_id, str) or not session_id
                or not isinstance(experience_id, str) or not experience_id):
            raise ValueError("session_id and experience_id are required")
        with self.reload_lock, self.inference_lock:
            if not EXPERIENCE_LOG.is_file():
                raise ValueError("unknown experience_id")
            rows = [json.loads(line) for line in EXPERIENCE_LOG.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            experience = next((row for row in rows
                                if row.get("id") == experience_id and
                                row.get("session_id") == session_id and
                                row.get("status") == "untrusted_pending_human_or_judge"), None)
            if experience is None:
                raise ValueError("experience is unknown, belongs to another session, or is not pending")
            if any(row.get("kind") == "human_feedback" and
                   row.get("experience_id") == experience_id for row in rows):
                raise ValueError("this experience already has a final human decision")
            chosen = experience.get("donor_answer") if action == "accept" else answer
            if not isinstance(chosen, str) or not chosen.strip():
                raise ValueError("overlord feedback requires a nonempty answer")
            prompt = experience.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("experience has no trainable prompt")
            result = self.learn_accepted(prompt, chosen)
            _append_jsonl(DATA / "experience" / "human_feedback.jsonl", {
                "kind": "human_feedback", "experience_id": experience_id,
                "session_id": session_id, "prompt": prompt, "answer": chosen, "action": action,
                "weight_update": result, "at_ns": time.time_ns(),
            })
            _append_jsonl(EXPERIENCE_LOG, {
                "kind": "human_feedback", "experience_id": experience_id,
                "session_id": session_id, "status": "accepted_by_human", "action": action,
                "checkpoint": result.get("checkpoint"), "at_ns": time.time_ns(),
            })
            return {"status": "human_choice_recorded", "action": action,
                    "experience_id": experience_id, "weight_update": result}

    def learn_accepted(self, prompt: str, answer: str, context: list | None = None) -> dict:
        if not prompt.strip() or not answer.strip():
            raise ValueError("accepted Lucy training requires a nonempty prompt and answer")
        with self.reload_lock, self.inference_lock:
            line_text = f"User: {prompt.strip()[-4096:]}\nLucy: {answer.strip()[:2048]}"
            # EXACT-DUPLICATE FAST PATH (Jonathan, 2026-10-10): identical content has zero
            # distance under the same pull-alike/push-unalike logic Wolfpack already uses --
            # it shouldn't cost "approximately zero," it should cost ACTUALLY zero. No route(),
            # no learn_line(), no save(). This is the training-time counterpart to BrickCluster's
            # gate already routing an exact repeat to the SAME brick rather than spawning a new
            # one -- that part was already free; this closes the remaining gap (it still retrained
            # on content it had fully learned already).
            if line_text in self._taught_exact_lines:
                return {"status": "skipped_exact_duplicate", "architecture": LI.ARCH,
                        "n_bricks": len(self.cluster.bricks), "changed_levels": 0,
                        "checkpoint": str(self.checkpoint_path),
                        "checkpoint_sha256": self.checkpoint_sha256}
            batches = LI._line_batches(line_text)
            if not batches:
                return {"status": "line_too_short", "line_text": line_text}
            # PREDICT first with a READ-ONLY probe (for_training=False never spawns -- see
            # BrickCluster.route) so we can snapshot before weights change, WITHOUT calling the
            # real (possibly-spawning) route() twice for one line. Calling the real one twice
            # would risk spawning two different bricks for a single accepted fact -- this probe
            # avoids that by construction, not by luck.
            features = self.cluster.tfidf_sparse(line_text)
            predicted_idx = self.cluster.route(features, for_training=False)
            before = None
            if predicted_idx is not None:
                brick = self.cluster.bricks[predicted_idx]
                before = {name: getattr(brick, name).copy() for name in WEIGHT_ARRAY_NAMES}
            n_bricks_before = len(self.cluster.bricks)
            idx2, loss = self.cluster.learn_line(batches, line_text, lr_scale=LR_SCALE)
            is_new_brick = idx2 >= n_bricks_before
            after_brick = self.cluster.bricks[idx2]
            changed_levels = 0
            total_levels = 0
            if before is not None and idx2 == predicted_idx:
                for name in WEIGHT_ARRAY_NAMES:
                    after_arr = getattr(after_brick, name)
                    changed_levels += int(np.sum(before[name] != after_arr))
                    total_levels += after_arr.size
            else:
                # route() spawned a brick before learn_line ran, then learn_line routed again and
                # picked a different one (can happen if the probe itself changed gate state) --
                # report size instead of a mismatched diff rather than compare unrelated arrays.
                total_levels = sum(getattr(after_brick, n).size for n in WEIGHT_ARRAY_NAMES)
            self.seen += sum(len(b) for b in batches)
            # NO consolidate() here (2026-10-10, real regression found: calling it after every
            # single accepted fact collapsed 6 diverse facts into 1 overdiluted brick that
            # matched NONE of them anymore, same small-sample permissive-merge mechanism found
            # in the main D:\flybrain session). "Sleep was always the plan" (Jonathan) -- online
            # teaching now accumulates bricks; consolidation is a separate, deliberate call --
            # see sleep().
            LI.save(self.cluster, self.seen, self.ingests)
            self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
            self._taught_exact_lines.add(line_text)
            return {
                "status": "levels_changed" if changed_levels else "no_level_changed",
                "architecture": LI.ARCH,
                "brick_index": idx2, "is_new_brick": bool(is_new_brick),
                "n_bricks": len(self.cluster.bricks),
                "changed_levels": changed_levels, "total_levels_in_brick": total_levels,
                "loss": loss, "checkpoint": str(self.checkpoint_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "floating_master_weights": False, "host_free_generalization_tested": False,
            }

    def sleep(self) -> dict:
        """The deliberate, separate consolidation pass -- run explicitly (e.g. after a batch of
        teaching, or on a schedule), never automatically per-fact. Same mechanism
        BrickCluster.consolidate() always was; the fix is in the CALLER's cadence, not the
        mechanism itself (that's already proven correct at production scale in the main
        D:\\flybrain session, 134 bricks on a real 4.2M-char corpus)."""
        with self.reload_lock, self.inference_lock:
            n_before = len(self.cluster.bricks)
            merged = self.cluster.consolidate()
            LI.save(self.cluster, self.seen, self.ingests)
            self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
            return {"n_bricks_before": n_before, "n_bricks_after": len(self.cluster.bricks),
                    "merged": merged, "checkpoint_sha256": self.checkpoint_sha256}

    def observe_correction(self, session_id: str, answer: str) -> dict:
        turns = SessionContext.load(session_id)
        turns = turns + [{"role": "user_correction", "text": answer}]
        path = SessionContext.save(session_id, turns)
        return {"state_saved": True, "state_file": str(path)}

    def health(self) -> dict:
        return {"status": "ok", "gateway_model": "lucywest-brickcluster",
                "architecture": LI.ARCH,
                "architecture_family": "wampyr_quinary_gated_rnn_brickcluster",
                "substrate_is_mamba3": False, "substrate_is_ssm": False,
                "n_bricks": len(self.cluster.bricks),
                "checkpoint": str(self.checkpoint_path), "checkpoint_sha256": self.checkpoint_sha256,
                "seen_chars": self.seen, "ingests": self.ingests,
                "persistent_representation": "per-brick quinary levels (-2..2), full-BPTT trained",
                "floating_master_weights": False,
                "session_memory_kind": "explicit_rolling_text_buffer_not_recurrent_state",
                "donor_model": REFERENCE_MODEL, "donor_weights_frozen": True,
                "reload_error": self.reload_error}
