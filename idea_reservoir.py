"""IdeaReservoir -- a direct code translation of Jonathan's own described thinking tool, built
2026-11-08 at his explicit request ("convert this to useful code... apply any of it to what we
are doing"). Source: a live conversation where he described his own scrap-drawer/reservoir of
collected ideas (C:\\Users\\Azazel\\Desktop\\parasite mamba\\parasite mamba.txt is one real
instance of it -- his own words: "Everything here was either used or planned on using if I saw it
fit better or was the basis of a fix") and named its real properties directly:

  1. "a random generator of useful thought but not totally random" -- noisy generation (he
     reinjects his own unconstrained associative thoughts), biased retention (a vote/attention
     counter decides what stays visible).
  2. "I keep the information sparse so it doesn't stop indexing" -- sparse, not dense; capacity
     is never exhausted because most of the space stays empty.
  3. "my vote on attention counter... I do not like moving a file once placed" -- ranking and
     storage position are DECOUPLED. An item's physical slot never changes once written; only its
     score moves.
  4. "if this scrap draw was perfect we discover nothing" -- the mess (unplanned adjacency between
     unrelated items) is where discovery happens, not a defect to clean up.
  5. "humans sort their tools not by measurement but category and frequency of use" -- retrieval
     priority is CATEGORICAL + USAGE-based, never a precise numeric measurement.
  6. Multiple independent "tech trees" developed in parallel for the same problem, compared for
     where they CONVERGE -- agreement between independently-derived approaches is the validation
     signal, not any single approach's own confidence.

This is not a metaphor wrapper -- (1)-(3) are, independently, already the exact real mechanism
BrickCluster.route()/vote_count/last_usage_step implements (bricks never reindex by usage; a
separate vote/usage score does the ranking). (6) is already the exact mechanism one of his own
sessions used to confirm the combinatorial-expansion fly-brain finding ("once with the plain delta
rule, once with cross-validated ridge regression so it wasn't an overfitting illusion" --
parasite_mamba.txt line ~698). This module makes all of it a reusable, general tool instead of a
one-off pattern re-derived separately each time it's needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _Slot:
    content: object
    category: str
    attention: float = 1.0       # the "vote on attention counter" -- goes up/down, item never moves
    last_used: float = field(default_factory=time.time)
    uses: int = 0


class IdeaReservoir:
    """The scrap drawer. Append-only by position (a slot's index never changes once assigned --
    "I do not like moving a file once placed"), sparse (slots are never compacted or evicted just
    because they've gone quiet -- "keep the information sparse so it doesn't stop indexing"),
    ranked by a usage/attention score that is tracked SEPARATELY from storage position, and
    retrieved by CATEGORY + frequency, never by sorting on a precise measurement of the content
    itself ("humans sort their tools not by measurement but category and frequency of use").
    """

    def __init__(self, decay: float = 0.98):
        self._slots: list[_Slot] = []
        self.decay = decay  # same shape as BrickCluster's own relay-fitness decay (*= 0.98 per
        # tick when unused) -- an item that stops being reached for quietly loses rank, but is
        # NEVER deleted. Low rank means "hasn't had its turn," not "wrong" -- see the diary entry
        # this module ships with for why that distinction was worth stating plainly.

    def reinject(self, content, category: str) -> int:
        """Add a new, unconstrained thought. No validation, no placement logic beyond category --
        this is the "noisy generation" half of "a random generator of useful thought but not
        totally random." Returns the slot's permanent index; it will never move."""
        self._slots.append(_Slot(content=content, category=category))
        return len(self._slots) - 1

    def use(self, idx: int) -> None:
        """Reaching for an item bumps its attention score and resets its clock -- the item itself
        never moves. This is the "biased retention" half: selection pressure, not relocation."""
        slot = self._slots[idx]
        slot.attention += 1.0
        slot.uses += 1
        slot.last_used = time.time()

    def tick_decay(self) -> None:
        """Call once per some real unit of work (a session, a day -- caller's choice). Every slot
        not used since the last tick erodes toward zero attention but is never removed -- the
        drawer stays messy ON PURPOSE; a perfectly curated drawer "discovers nothing.\""""
        now = time.time()
        for slot in self._slots:
            if slot.last_used < now:
                slot.attention *= self.decay

    def fetch(self, category: str | None = None, top_n: int = 5) -> list[tuple[int, object]]:
        """Retrieval by CATEGORY first, then frequency/attention within it -- never by a precise
        measurement of the content. Passing category=None deliberately returns the full messy mix
        across categories, unsorted by anything but attention -- the "go look when stuck and let
        unplanned adjacency do its job" mode, not the "find the one tool I already know I need"
        mode. Use category=... for the latter."""
        pool = self._slots if category is None else [
            s for s in self._slots if s.category == category
        ]
        ranked = sorted(enumerate(self._slots), key=lambda pair: -pair[1].attention) \
            if category is None else \
            sorted(((i, s) for i, s in enumerate(self._slots) if s.category == category),
                   key=lambda pair: -pair[1].attention)
        return [(i, s.content) for i, s in ranked[:top_n]]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self._slots:
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts


def tech_tree_convergence(candidates: dict[str, object], agree=lambda a, b: a == b) -> dict:
    """Multiple independent approaches to the SAME problem ("tech trees"), developed in parallel;
    where they converge (agree) is treated as the validated answer, not any one tree's own
    confidence. Direct code form of Jonathan's own described method: "try and develop multiple
    solutions for the same problem, see how they converge, then at the convergence points plot a
    new tech tree... that's the solution against other solutions once I'm done."

    `candidates`: {tree_name: result}. `agree`: equality check for two results (default: exact
    equality; pass something looser -- e.g. a tolerance check -- for numeric results).

    Returns {"validated": [result, ...], "disputed": [(tree_a, tree_b, result_a, result_b), ...],
    "solo": [tree_name, ...]} -- a result several independently-built trees converge on is
    validated; a result only one tree reached is neither confirmed nor refuted, just unconfirmed
    (solo); and open disagreement between trees is named, not silently averaged away."""
    names = list(candidates)
    validated, disputed, grouped = [], [], []
    seen = [False] * len(names)
    for i, name_i in enumerate(names):
        if seen[i]:
            continue
        group = [name_i]
        for j in range(i + 1, len(names)):
            if not seen[j] and agree(candidates[name_i], candidates[names[j]]):
                group.append(names[j])
                seen[j] = True
        seen[i] = True
        grouped.append(group)
    for group in grouped:
        if len(group) >= 2:
            validated.append({"trees": group, "result": candidates[group[0]]})
        else:
            disputed_or_solo = group[0]
            # A lone tree isn't "disputed" against anything specific -- it just never found a
            # partner. Report it as solo, not as a fabricated disagreement.
            pass
    solo = [g[0] for g in grouped if len(g) == 1]
    # Real disagreements: any two DIFFERENT validated groups both containing >=2 trees, or a
    # validated group vs a solo tree whose result actively conflicts (checked by `agree` being
    # False both ways -- already implied by them landing in different groups above).
    if len(validated) > 1:
        for a in range(len(validated)):
            for b in range(a + 1, len(validated)):
                disputed.append((validated[a]["trees"], validated[b]["trees"],
                                  validated[a]["result"], validated[b]["result"]))
    return {"validated": validated, "disputed": disputed, "solo": solo}
