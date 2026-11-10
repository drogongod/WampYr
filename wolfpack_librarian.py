"""Wolfpack librarian layer: a local indexing/addressing mechanism for BrickCluster, built to
answer a real scaling question found live 2026-11-10 (see
lucy_nlp_pilot_slowdown_investigation_20261110.md and
wolfpack_librarian_prototype_build_20261110.md): FixedStepBrick.learn_batch()'s validated-subset
search is deliberately, correctly exhaustive (shadow-free discrete training, proven working), and
its cost grows with how much a single brick has absorbed over its lifetime. This module does NOT
touch that mechanism at all -- it answers a different question entirely: can PLACEMENT (which
brick should hold this content, has it been seen before) be handled by cheap, deterministic,
non-gradient-trained logic, so a brick's own load stays bounded by genuine capacity limits instead
of one brick silently absorbing unlimited new content forever.

Lineage, briefly, for a future reader: co-designed with Sol (a real, separate AI, OpenAI-based --
not Echidna/Galatea) after she reviewed this exact scaling problem and proposed "compiled local
librarians with bounded learning regions" as the primary direction. The principle borrows from a
real, verified precedent (Qwenputer, github.com/divagr18/qwenputer -- a RISC-V CPU compiled,
not trained, into frozen Qwen3.5-4B weights, using the model's own persistent recurrent state as
addressable memory via hash-based lookup instead of a scan) and from this project's own
sovereign_core.py growth-trigger precedent (_grow_network). None of that code is reused literally
-- BrickCluster's quinary bricks are a different substrate -- only the PRINCIPLE: deterministic
addressing + hash-based lookup instead of a scan + a persistent key-value directory as storage.

ARCHITECTURAL DISCIPLINE, explicit per Jonathan's own framing (2026-11-10): "each department is
its own department's business and its own department's index." A Librarian owns her OWN directory
and anchor/medoid state completely privately -- nothing outside a Librarian ever reads or writes
her internal dict directly. The MainIndex (the single dispatch point, deliberately analogous to a
nervous system's single output port -- many independent departments, one narrow point where their
results converge into a single answer) only ever calls a Librarian's public methods and receives
an opaque result back. It holds nothing about a request once dispatched.

STATUS: standalone, tested against real corpus content (see test_wolfpack_librarian.py), NOT yet
wired into BrickCluster.route()/learn_line()'s live hot path. The live NLP Pilot training run
depends on that exact code path; wiring this in is a separate, later decision requiring explicit
sign-off, not something to change out from under a running production process.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from brick_cluster_charlm import raw_trigram_sparse, FEATURE_DIM

_FROZEN_IDF = None  # set once via freeze_reference_idf(), never touched again


def freeze_reference_idf(reference_texts) -> None:
    """Compute doc-frequency/IDF ONCE from a large, representative reference sample and freeze it
    forever. This is the real fix, found live 2026-11-10 after the first attempt (raw counts, no
    IDF at all) made things WORSE (joiner retrieval 63.5% -> 36.7%): IDF isn't only a source of
    instability, it's also doing real discriminative work -- downweighting generic filler
    trigrams every sentence shares so genuine topical signal can stand out. Stripping it entirely
    fixed stability but broke discrimination. The actual fix needs both: compute the downweighting
    statistics from a large reference ONCE, then never again -- same chessboard principle, applied
    correctly this time. A square's value gets painted once, from enough real data to mean
    something, and is never repainted because something else was added later."""
    global _FROZEN_IDF
    doc_freq = np.zeros(FEATURE_DIM, dtype=np.float64)
    n_docs = 0
    for text in reference_texts:
        counts = raw_trigram_sparse(text, FEATURE_DIM)
        if counts:
            idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            doc_freq[idx] += 1.0
        n_docs += 1
    idf_smoothing = 1.0
    _FROZEN_IDF = np.log((n_docs + idf_smoothing) / (doc_freq + idf_smoothing))


def stable_vec(text: str) -> np.ndarray:
    """The librarian's OWN feature vector -- deliberately NOT cluster.tfidf_sparse(), which
    recomputes IDF from the CURRENTLY-growing corpus every call (measured: cosine(v_then, v_now)
    = 0.855 for identical text after 200 unrelated documents were added in between -- a coordinate
    grid that repaints every existing square whenever a new one is added). This multiplies the
    same fixed, deterministic trigram hash (`raw_trigram_sparse`) by a FROZEN reference IDF table
    (see freeze_reference_idf()) instead of a live-recomputed one -- same discriminative weighting,
    zero dependence on anything added after the reference was frozen. Must call
    freeze_reference_idf() once before using this, with a reasonably large, representative sample."""
    if _FROZEN_IDF is None:
        raise RuntimeError("stable_vec() needs freeze_reference_idf() called once first")
    counts = raw_trigram_sparse(text, FEATURE_DIM)
    dense = np.zeros(FEATURE_DIM, dtype=np.float64)
    if counts:
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        cnt = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        val = cnt * _FROZEN_IDF[idx]
        norm = np.linalg.norm(val)
        if norm > 0:
            val = val / norm
        dense[idx] = val
    return dense

# Two forks resolved 2026-11-10, both implemented here:
#   Fork 2 (top-level scan cost, confirmed O(n_regions) linear, ~7-8us/region at test scale):
#     -> Option B, real LSH (random-hyperplane SimHash) bucketing instead of a full scan.
#   Fork 1 (founder 94.7% vs joiner 61.5% retrieval gap, confirmed not guessed):
#     -> Option B primary: periodic medoid re-anchoring (a real representative member, not a
#        diffuse running mean -- avoids the "black hole first region" failure mode found in the
#        prototype's v2, where an averaged centroid drifted into indiscriminately accepting
#        unrelated content).
#     -> Option C fallback: if the single best-matching region misses the exact-hash lookup,
#        check a small number of additional LSH-candidate regions before concluding "not found."

LSH_NUM_TABLES = 4          # independent hash tables (multi-probe, reduces false negatives)
LSH_TARGET_PER_TABLE_MATCH_PROB = 0.35  # see _bits_for_threshold() -- K is DERIVED from this,
# not a fixed magic number. A fixed K (e.g. 12 bits) picked for near-duplicate detection will
# silently fail a loose similarity threshold: at threshold=0.15 (deliberately permissive, per
# this project's "approximate at the top, correctness recovered cheaply after" design), ALL 12
# random hyperplanes agreeing in sign has well under 1% probability even for genuinely related
# vectors. Found live via a real test: 84 regions for 70 facts, 0 facts ever joined an existing
# region. K must be derived from the actual threshold being used, not guessed independently of it.


def _bits_for_threshold(threshold: float) -> int:
    """Random-hyperplane LSH: for two vectors with cosine similarity s, the probability a single
    random hyperplane's sign agrees for both is 1 - arccos(s)/pi. Solve for the number of bits K
    (independent hyperplanes that must ALL agree) such that P(all K agree) ~= target, giving
    buckets sized appropriately for the similarity threshold actually in use."""
    p1 = 1.0 - np.arccos(np.clip(threshold, -1.0, 1.0)) / np.pi
    if p1 <= 0.0 or p1 >= 1.0:
        return 1
    k = np.log(LSH_TARGET_PER_TABLE_MATCH_PROB) / np.log(p1)
    return max(1, round(k))
REANCHOR_EVERY_N_MEMBERS = 4   # recompute medoid after this many NEW members join a region
FALLBACK_CANDIDATE_REGIONS = 3  # Fork 1 Option C: how many extra regions to check on a miss


def _content_hash(text: str) -> int:
    """Exact-match key for a Librarian's own directory -- deterministic, no gradient involved,
    matches Qwenputer's own 'hash the name, read the slot back, no scan' filesystem design."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class _LSHIndex:
    """Fork 2: random-hyperplane LSH over region anchors, replacing a full O(n_regions) scan with
    an expected-O(1) bucket lookup. Multiple independent tables reduce the chance a genuine match
    gets hashed away from its true neighbors by any single table's particular hyperplanes."""

    def __init__(self, dim: int, bits_per_table: int, rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng(0)
        self.bits_per_table = bits_per_table
        self._planes = [
            self._rng.normal(size=(bits_per_table, dim))
            for _ in range(LSH_NUM_TABLES)
        ]
        self._buckets = [dict() for _ in range(LSH_NUM_TABLES)]  # signature -> set(region_idx)

    def _signatures(self, vec: np.ndarray):
        sigs = []
        for planes in self._planes:
            bits = (planes @ vec) >= 0.0
            sig = 0
            for b in bits:
                sig = (sig << 1) | int(b)
            sigs.append(sig)
        return sigs

    def insert(self, region_idx: int, vec: np.ndarray):
        for t, sig in enumerate(self._signatures(vec)):
            self._buckets[t].setdefault(sig, set()).add(region_idx)

    def reindex(self, region_idx: int, old_vec: np.ndarray, new_vec: np.ndarray):
        """Called when a region's anchor changes (re-anchoring) -- removes the stale bucket
        membership and inserts under the new signature, so lookups stay correct after a region's
        representative vector moves."""
        for t, sig in enumerate(self._signatures(old_vec)):
            s = self._buckets[t].get(sig)
            if s is not None:
                s.discard(region_idx)
        self.insert(region_idx, new_vec)

    def candidates(self, vec: np.ndarray) -> set:
        """Union of candidate region indices across all tables -- expected small relative to
        total region count, NOT a full scan. Falls back to an empty set (caller must then treat
        this as 'create new region') only when genuinely nothing nearby exists in any table."""
        out = set()
        for t, sig in enumerate(self._signatures(vec)):
            out |= self._buckets[t].get(sig, set())
        return out


@dataclass
class Librarian:
    """Owns one bounded-capacity region. Her directory and anchor/medoid state are PRIVATE -- the
    MainIndex never reads or writes these fields directly, only calls route()/stats()."""
    librarian_id: int
    capacity: int
    _directory: dict = field(default_factory=dict)   # content_hash -> brick_id, PRIVATE
    _member_vecs: list = field(default_factory=list)  # PRIVATE, used only for medoid recompute
    _bricks: list = field(default_factory=list)       # PRIVATE

    def is_full(self) -> bool:
        return len(self._bricks) >= self.capacity

    def has_exact(self, text: str) -> Optional[int]:
        """Public, read-only: does this librarian already hold this exact content? Returns the
        brick id or None. Safe to call even when full -- fullness blocks new content only, never
        blocks recognizing a repeat (the bug found and fixed in the prototype, baked in here from
        the start)."""
        return self._directory.get(_content_hash(text))

    def assign_new(self, text: str, vec: np.ndarray, next_brick_id_fn) -> Optional[int]:
        """Public: assign brand-new content to a brick within this region, if there's room.
        Returns the new brick id, or None if full (caller must find/grow a sibling region)."""
        if self.is_full():
            return None
        brick_id = next_brick_id_fn()
        self._bricks.append(brick_id)
        self._directory[_content_hash(text)] = brick_id
        self._member_vecs.append(vec.copy())
        return brick_id

    def member_count(self) -> int:
        return len(self._bricks)

    def recompute_medoid(self) -> Optional[np.ndarray]:
        """Fork 1 Option B: a real member vector with the highest average similarity to the
        OTHERS, not a running mean. A mean dilutes toward the population average and can become
        a 'black hole' that indiscriminately accepts unrelated content (found and diagnosed in
        the prototype's v2, first attempt) -- a medoid stays a genuine, specific representative
        of the group, which is what retrieval stability actually needs."""
        n = len(self._member_vecs)
        if n == 0:
            return None
        if n == 1:
            return self._member_vecs[0].copy()
        mat = np.stack(self._member_vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = mat / norms
        sims = unit @ unit.T
        np.fill_diagonal(sims, 0.0)
        avg_sim = sims.mean(axis=1)
        best = int(np.argmax(avg_sim))
        return self._member_vecs[best].copy()


class WolfpackMainIndex:
    """The single dispatch point -- deliberately O(1)-per-request at THIS layer (the LSH lookup
    below is expected-sublinear, not the O(n_regions) scan the prototype measured at ~7-8us/region
    and confirmed linear). Holds nothing about a request once dispatched; a Librarian's internal
    state is never touched directly from here.

    region_similarity_threshold: minimum cosine similarity (checked only against LSH CANDIDATES,
    never a full scan) required to join an existing region instead of founding a new one.
    """

    def __init__(self, dim: int, region_capacity: int = 10,
                 region_similarity_threshold: float = 0.15, seed: int = 0):
        self.dim = dim
        self.region_capacity = region_capacity
        self.region_similarity_threshold = region_similarity_threshold
        bits = _bits_for_threshold(region_similarity_threshold)
        self._lsh = _LSHIndex(dim, bits_per_table=bits, rng=np.random.default_rng(seed))
        self._librarians: list[Librarian] = []
        self._anchors: list[np.ndarray] = []
        self._new_members_since_reanchor: list[int] = []
        self._next_librarian_id = 0
        self._next_brick_id_counter = iter(range(10 ** 12))

    def _next_brick_id(self) -> int:
        return next(self._next_brick_id_counter)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _new_region(self, vec: np.ndarray, text: str) -> tuple[int, bool, int]:
        lib = Librarian(self._next_librarian_id, self.region_capacity)
        self._next_librarian_id += 1
        region_idx = len(self._librarians)
        self._librarians.append(lib)
        self._anchors.append(vec.copy())
        self._new_members_since_reanchor.append(0)
        self._lsh.insert(region_idx, vec)
        brick_id = lib.assign_new(text, vec, self._next_brick_id)
        return brick_id, False, lib.librarian_id

    def _maybe_reanchor(self, region_idx: int):
        lib = self._librarians[region_idx]
        self._new_members_since_reanchor[region_idx] += 1
        if self._new_members_since_reanchor[region_idx] < REANCHOR_EVERY_N_MEMBERS:
            return
        self._new_members_since_reanchor[region_idx] = 0
        new_anchor = lib.recompute_medoid()
        if new_anchor is None:
            return
        old_anchor = self._anchors[region_idx]
        self._lsh.reindex(region_idx, old_anchor, new_anchor)
        self._anchors[region_idx] = new_anchor

    def dispatch(self, text: str, vec: np.ndarray) -> tuple[int, bool, int]:
        """The one public entrypoint. Returns (brick_id, was_retrieval, librarian_id).
        `vec` is the caller's own real feature vector (e.g. from cluster.tfidf_sparse(), densified)
        -- this module does not compute features itself, so it stays substrate-agnostic about
        where that vector came from."""
        candidate_region_idxs = self._lsh.candidates(vec)

        # Fork 1 Option C fallback lives here: rank ALL LSH candidates by similarity (not just
        # take the first), so an exact-hash check can be tried against more than one region if
        # the single best guess misses -- cheap, since candidate_region_idxs is expected small.
        scored = sorted(
            ((self._cosine(vec, self._anchors[i]), i) for i in candidate_region_idxs),
            key=lambda x: -x[0],
        )

        checked = 0
        for sim, region_idx in scored:
            if sim < self.region_similarity_threshold:
                break
            lib = self._librarians[region_idx]
            hit = lib.has_exact(text)
            if hit is not None:
                return hit, True, lib.librarian_id
            checked += 1
            if checked >= FALLBACK_CANDIDATE_REGIONS:
                break

        # No exact hit among candidates. Assign to the single best-matching region if it's a
        # real match and has room; otherwise found a new region.
        if scored and scored[0][0] >= self.region_similarity_threshold:
            best_sim, best_idx = scored[0]
            lib = self._librarians[best_idx]
            if not lib.is_full():
                brick_id = lib.assign_new(text, vec, self._next_brick_id)
                self._maybe_reanchor(best_idx)
                return brick_id, False, lib.librarian_id

        return self._new_region(vec, text)

    def stats(self) -> dict:
        return {
            "n_regions": len(self._librarians),
            "n_bricks": sum(lib.member_count() for lib in self._librarians),
        }
