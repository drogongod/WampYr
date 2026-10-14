# LucyWest: the Wampyr substrate

## What this is

**Wampyr** is the name of Lucy's own, continuously self-training weight substrate: a quinary
(-2..2 level), byte-level **BrickCluster** -- content-routed, independently-trained small gated
recurrent cells ("bricks"), grown and merged online via an LSL-style density gate. It is her own
native memory, checked first on every request; a larger frozen donor model is consulted only
when her own weights don't know the answer, and her own weights only ever change on an explicit
human `/accept` or `/overlord` decision. The full mechanism is specified in `F:\Wampyr.txt`.

**Architecture, stated precisely, verified against the actual code, not asserted:** Wampyr is
**not** a state-space model (Mamba/SSM) and **not** a Transformer. Each brick's recurrence
(`native_quinary_charlm.py`: `Win`/`Wh`/`Wgx`/`Wgh`/`Wo`) is a small, quinary-quantized **gated
recurrent cell** -- structurally closest to a minimal GRU. There is no input-dependent selective
B/C projection, no discretized state-transition matrix, and no continuous-time Delta step, which
is what would make this an SSM; there is no attention/QKV anywhere in this file, which is what
would make this a Transformer. It is its own thing: a clustered population of small quinary
gated-RNN cells, routed by content.

**Session memory is honestly NOT a compressed fold.** There is no persistent hidden state
carried forward turn-to-turn the way a real SSM's state is. Conversational continuity is handled
by an explicit, bounded rolling text buffer (`SessionContext`) -- recent turns kept as literal
text and re-fed each request. Turns that age out of the window are dropped, not compressed.
This is reported explicitly in every API response (`session_memory_kind:
"explicit_rolling_text_buffer_not_recurrent_state"`), never claimed to be something it isn't.

Long-term factual memory lives in the brick population itself: bricks grow as she's taught
genuinely new, distinct things, and merge when new content closely matches something she already
knows (confirmed empirically -- exact-duplicate teaching is a zero-cost no-op, and near-duplicate
facts consolidate into fewer bricks rather than each spawning a new one). This keeps growth
sub-linear, not constant-size.

## Real, measured results (not claims)

- **Retention, full 1000-fact run, 2026-10-12** (`amelia_teaches_lucy.py`, Amelia-voiced donor
  asking Lucy one question per round, auto-accepted, `MERGE_SLACK=0.08`): **711/1000 = 71.1%**
  matched host-off (own weights only, no donor) at the end of the run, 152 final bricks, 1863
  total seen. This beat an earlier, less-tuned baseline of 47.3% at the same 1000-fact scale.
  Raw data: `data/brickcluster/experience/amelia_teaching_log.jsonl`.
- **Exact-duplicate fast path**: re-teaching an identical fact is detected and skipped before
  any training call, zero cost, persists across restarts via the journal.
- **Production-scale ingestion** (separate corpus, `D:\flybrain` session, 2026-10-10): settled
  to 134 stable bricks on a real 4.2M-char corpus; five real root-cause bugs found and fixed
  along the way, 6.9x speed fix verified.

## Current, honest limits

- **Fluent, correct host-off generation is not established at small training budgets.** Routing
  -level recognition/retention (the 71.1% figure above) is proven; whether a matched brick's own
  *generated text* is fluent/correct at very small per-fact training budgets is a separate,
  weaker claim -- see `evidence_contract_test.py`'s own notes.
- **"Changed levels" is per-call evidence, not a running total.** Each `learn_accepted` call
  reports its own before/after diff; nothing here yet tracks changed-levels across the whole
  checkpoint's history.
- **The donor model's own architecture is a separate question from Lucy's own weights.** The
  donor (`vampire-lucywest`, a Ternary-Bonsai-2-27B checkpoint) is a hybrid architecture with
  real SSM layers in most of its blocks (verified directly from its GGUF metadata) -- but that
  is the donor's architecture, not Wampyr's. Wampyr (Lucy's own learned weights) has none of
  that; see above.

## Files

| File | Role |
|---|---|
| `brick_cluster_charlm.py`, `native_quinary_charlm.py`, `quinary_simd_kernel.{c,dll}`, `lutgtlm_ingest.py` | Wampyr itself -- the proven substrate. |
| `brickcluster_engine.py` | The adapter -- `complete`, `accept_experience`, `learn_accepted`, `observe_correction`, `health`. Own-weights-first, donor-fallback routing. |
| `brickcluster_server.py` | Loopback OpenAI-compatible API, port 8105. |
| `amelia_teaches_lucy.py` | The teaching-loop script that produced the 71.1% retention result above. |
| `evidence_contract_test.py` | The non-negotiable evidence test from `F:\Wampyr.txt`: host-off before teaching (no match) -> teach -> host-off after (match) -> teach two unrelated facts -> original fact still matches (retention under interference). |
| `server.py` | Donor identity-checking starter -- verifies `vampire-lucywest` is actually what's on the donor port; never kills a process. |
