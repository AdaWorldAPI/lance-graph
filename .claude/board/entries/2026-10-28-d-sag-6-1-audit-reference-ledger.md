# D-SAG-6-1: reference ledger and stale-status corrections for the D-SAG-6 audit

**Status:** PROPOSED / DOCS ONLY. Plan: [d-sag-6-1-audit-reference-ledger-v1.md](../../plans/d-sag-6-1-audit-reference-ledger-v1.md). Parent: D-SAG-6 (#1760). No code, test or authorization; the D-SAG-6-1 audit itself is still unstarted.

## What this records

- References the D-SAG-6-1 audit needs that #1760 §7 does not list: ruff `ruff_spo_triplet` (`ir.rs`, `expand.rs`, `reassemble.rs`, `role.rs`), ruff's behavioural-ore and R2IL oracle plans, OGAR `ogar-from-ruff`, the merged CFA docs (#1813, #1814, #1818), #1792, #1799, `spog_tenants.rs`, `epistemic_state5.rs`, `lance-graph-cypher-quack`, D-DTM-1, the `mcp.exo.red` workbench entry, the ndarray mask/popcount primitives for the first fixture, and crai-rs CR-H0/CR-H1 (private).
- MEASURED: ruff `f2cb523` has SPO frontends for C++, C#, Python, Ruby, MeTTa and SQLAlchemy, none for Rust; `ProposedMethod` has zero hits under lance-graph `crates/`.
- Stale status corrected: crai-rs CR-H0 lists #1813–#1830 as open (all merged); #1831 calls crai-rs the successor to #1760 (crai-rs is a sidecar that waits on D-SAG-6-1).

## Open

Whether D-SAG-6-1 proposes a Rust frontend for ruff; where `ProposedMethod` is defined before D-SAG-6-2 needs it.
