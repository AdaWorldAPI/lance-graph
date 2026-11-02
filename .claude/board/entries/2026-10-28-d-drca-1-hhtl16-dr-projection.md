# D-DRCA-1: HHTL16 Cartesian direct-routing projection proof

**Status:** MEASURED, lab only. Code: `crates/dr-math-binding-probe/src/hhtl16.rs`, test `tests/hhtl16_projection.rs`, costs `examples/hhtl16_stages.rs`; results in the crate README, section D-DRCA-1. Contract: the 2026-10-28 DR/DT amendment (`.claude/plans/d-drdt-dr-dt-ownership-amendment-v1.md`, separate PR).

## Result

- The Cartesian address is the existing `Morton8x8` code over a bound DR64 register; no new address type.
- YZ rotation is an address map over one stored relation; over all 4,096 cells the rotated read equals the stored read, on an asymmetric fixture.
- Identity × Property, Identity × Identity, Property × Property and cross-table Identity × Property (through a witnessed FK, `relcomp::Chain`) match an independent scalar oracle on a two-table fixture with shared local ordinals.
- Equal local ordinals of different tables are refused by `Chain` domain typing; a foreign frame owner is refused at equal revision numbers.
- Three-valued reads keep TRUE, FALSE and UNKNOWN apart in the property orientation; a changed scope or coordinate revision refuses.
- Warm projections make 0 global→local lookups and 0 heap allocations. Measured (one host): rotated cell read 0.19 ns, property column 10 ns, Property × Property Boolean 818 ns, cross-table `FK ∘ R` 881 ns (4 allocations, in `Chain::eval`), cold `LocalMap` preparation 1,050 ns.
- Six disable runs each turn their named test red.

## Open

DR256::WHERE and the stay / rebind / masked-read choice; focus above 64; a SIMD column read; `Chain::eval` copies leaf rows (512 B each) and allocates; real source owner, ACL and ClassView registry; DT method semantics (DR #4 handoff).
