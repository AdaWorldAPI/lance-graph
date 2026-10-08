# 2026-10-08 — D-RPF-0: CE64 field predicate over resident `NodeRow` bytes

**Status:** MEASURED, TEST-PINNED
(`crates/lance-graph-mask-risc/tests/resident_ce64_predicate.rs`, 6 tests;
`cargo test -p lance-graph-mask-risc --test resident_ce64_predicate`).
Test-only: the library is unchanged; `causal-edge` and `lance-graph-contract`
are dev-dependencies of mask-risc.

## Claim and result

A Pearl3, Inference (raw 4-bit nibble) or Epi5 predicate on any of the four
`CausalEdge64` words in `ValueTenant::MaterializedEdges` runs in place through
`Pred::MatchFacet16Strided` over two 16-byte windows (edges 0,1 and 2,3), with
no extracted lane. Executor and oracle agree, and both equal the CE64 named
accessors, row by row, for every value of every field on every edge
(3 tiles + 37 rows). `raw5 >= t` is a union of at most 6 patterns and equals
the accessor for all 32 thresholds on all four edges.

The predicate is built from `isa::Field::span()`; the oracle reads the CE64
accessors, which never consult that table. Offsets come from
`ValueTenant::value_offset()`.

## Falsifiers

- can fire: a care shifted up by one bit disagrees with the accessor for some
  value of every field;
- can stay silent: rewriting every byte of every row except the cared field
  leaves the mask unchanged;
- anti-vacuity: each field has a value admitting a strict, non-empty minority
  (< 1/3) of rows. It fired on the first run: the fixture generator (`>> 11`
  LCG) left bits 53..63 zero in every word, so Epi5 was constant. Replaced by
  SplitMix64.

Disable runs (anchor asserted, each red): window `k` instead of `k / 2` (4),
half always 0 (3), threshold without the equality pattern (1), silent-arm noise
keeping none of the base field (1).

## OPEN

None for D-RPF-0. D-RPF-1 adds the admission plane in front of these patterns.
