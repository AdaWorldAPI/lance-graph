## 2026-08-19 — E-CROSS-VERSION-IDENTITY-MIGRATES-BLIND-SO-IT-FAILS-CLOSED-1

**Status:** FINDING (5+3 council on the cascade-seal spec, ratified v3 at
`.claude/plans/cascade-seal-register-grid-v1.md`; novel content only per
the G9 rule — petal-16 / identity-split / twice / FNV-deletion are in the
two entries below and are cited, not restated).

**The finding:** when the durable batch identity changes hash algorithm
(FNV → the accumulated root, behind the new `seal_version` Arrow column
on `cycle_store_schema`), a cross-version fence-mismatch is
**undecidable by construction**: same-content and divergent-content
resubmissions are indistinguishable across algorithms without reading
payload bytes, and NO reconcile-adjacent path loads them —
`find_frame` projects `["batch_hash"]` only (cycle_sink.rs:577) and
`scan_sealed` mints `payload: Vec::new()` (cycle_sink.rs:1045). Adding a
read to decide it would be a post-finalization FULL-CYCLE payload rescan,
forbidden verbatim by seal req 1. Therefore the only sound rule is
**fail closed**: same-version compares exactly as today; cross-version →
`CommitError::Ambiguous` (its own doc: "may or may not be durable") →
Escalate. Gate G7(e) pins "zero payload dereferences on any reconcile
path" with an instrumented counter.

**How it was found — the maxim worked twice on one rule:** the council
killed two successive W3 designs. v1's reconcile-by-fence-chain-alone
failed OPEN (silent accept of genuine divergence — found independently
by two savants). v2's recompute-and-compare replaced it and was then
killed from opposite ends by two reviewers: its "rows already loaded"
premise was false at the cited comparison site (hash-only projection)
AND false on the restart path (empty payload mint), while the spec's own
restatement of req 1 had dropped the load-bearing "full-cycle" qualifier
that made the conflict visible. Both withdrawals are retained in the
plan's ledger (L6, L23), not deleted.

**Cross-refs:** the two entries below (register grid; accumulated seal);
`seal-vs-temporal-ordering-information.md` (G1's tie-density scope,
PROBE-SEAL-TIE-DENSITY now a W4 arm); I-LEGACY-API-FEATURE-GATED (the
`seal_version` gate mirrors `ENVELOPE_LAYOUT_VERSION`).

