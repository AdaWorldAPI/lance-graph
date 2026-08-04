# Agent tag-file: audit-one-sided-bounds

**Task:** report-only sweep for `E-THE-GATE-ASSERTED-A-CORPUS-IT-NEVER-SAW-1`'s named
sibling check — one-sided upper-bound assertions on counts derived from parsing/
filtering/truncation across `crates/`.

**Output:** `/tmp/audit_one_sided_bounds.md`.

**Result:** 54 assertion sites classified (via 4 targeted grep passes on
`.len()`/`.count()`/named count-scalars, cross-checked against a 408-hit generic
`assert!(<=|<)` baseline for coverage auditing only): 20 AT RISK, 14 PAIRED,
20 SAFE-CAPACITY.

**Nothing edited.** No `.rs` files touched, no `cargo` run. `crates/lance-graph-planner/
examples/blw_tenant.rs` was not touched (per brief, another agent owns it).
