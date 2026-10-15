# 2026-10-15 — The givens layer: hydration writes thought 0

**Status:** MEASURED. Items 1-5 accepted by the operator 2026-10-15,
pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md` ("Givens
layer") before the probe was written.

**What holds (VERIFIED-IN-CODE, TEST-PINNED):**
- Thought layer 0 holds the givens of the epistemic sudoku; only hydration
  writes it. A write to layer 0 from a hypothesis is refused.
- A hypothesis offering a different value for a given's lane is refused and
  recorded as a contradiction (row, thought, lanes, offered cells); the
  given's own value is agreement and is not stored. Between hypotheses the
  higher thought still wins.
- Hydration writes CausalEdge64 bits 59..63 as one `EpistemicState5` through
  `with_epistemic_raw5` (the deprecated `with_reasoning_band` is not used).
  Topology from the ontology; certification capped by the proofs hydration
  ran: membership earns Associated, a resolving relation earns Related.
  Relations the ontology claims as Causes hydrate to Related at most.
- One read-only table per (classid, sealed version) behind a `LazyLock`
  registry; the read path never hydrates; recomputing gives the same bytes,
  whatever order the declarations arrive in.

**Probe:** `crates/cognitive-shader-driver/examples/givens_layer_probe.rs`.
7 tests written first; 8 disable runs red. Tier `v4 (avx512f)`, 65,536 rows,
63 hypothesis thoughts: hydration 34.1 ns per row; write with the given
check 73.2 ns per offered cell; merged point read over 30 lanes 2.3 µs with
givens, 1.7 µs without (the overlay merge, about 21 entries per row);
count over written words 0.19 ns per row.

**Correction to the accepted wording:** item 3 said "band in 61..63". Those
bits have a declared reading since D-EPI-CANON-0: `Certification3` (Open,
Associated, Related, Supports, CausalCandidate, Causes). "Relation, not
causal" is `Related`; the cap is checked with `Certification3::entails`.

**OPEN:** the merged read is microseconds at this density; the overlay
merge remains D-HL-5's open cost. Bits 50..52 untouched (O2). Persistence of
the contradiction records is D-HL-7's question.
