# 2026-11-15 — PmLedger on abakus::pm; #2086 merged without a final-head review

STATUS: done | OUTCOME: `crates/xlog-relcomp-probe` `PmLedger` no longer carries
a copy of the `P × M` arithmetic; it calls `ndarray::hpc::abakus::pm`
(ndarray #433, one source of truth) through `PmSum::into_saturated_parts`.
The i64-model tests (6/6) and the D-XLOG-PM-1 triangle-pairing arm give the
same counts as before (0 of 47,646 batches wrong). Disable: a local ndarray
patch that zeroes `add_pm`'s carry fails 4 of the 6 ledger tests.

## Governance exception: #2086

lance-graph #2086 (cube safety P0) merged on 2026-11-15 10:34:28 UTC at head
`ece628bc4`. Its stated gate was an independent review of the final head. The
only independent review on record is Codex's, of `a551a32`, two pushes
earlier; no review covered `a19545a90` (the pre-check regression test) or the
rebased `ece628bc4`. CI was green on `ece628bc4`. This is recorded as a merge
without its review gate, not reclassified as reviewed. A retrospective review
of `ece628bc4` would close the assurance gap; it has not been done.

OPEN: retrospective review of #2086 at `ece628bc4`; P4 economics (including
checked `i64` arithmetic in the escape); wasm tiers of `tests/add_pm.rs` not in
ndarray CI.
