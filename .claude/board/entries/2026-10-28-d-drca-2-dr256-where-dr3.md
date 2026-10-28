# D-DRCA-2: DR256::WHERE × DR#3

**Status:** MEASURED, lab only. Code: `crates/dr-math-binding-probe/src/where_dr3.rs`, test `tests/dr256_where_dr3.rs`, costs `examples/dr256_where_stages.rs`; results in the crate README, section D-DRCA-2. Builds on D-DRCA-1 (`hhtl16`, #1834). Contract: `.claude/plans/d-drdt-dr-dt-ownership-amendment-v1.md` (#1835).

## Result

- A three-valued global WHERE (Quack `Draft::bind` + `Filter::sql_where`, mask-risc `Keep`, 65,536 addresses) projects into a 64-slot DR frame by `map.len()` probes through the counted `LocalMap`. No new predicate evaluator, routing engine or mask store.
- Every read is re-admitted against owner, source, codebook, frame, register frame, axis, DR method, focus, scope and criteria; each refusal is pinned by a test against an independent source-ID oracle.
- UNKNOWN stays apart from FALSE in the selection, its negation and a three-valued property column.
- Projection, admission and folds allocate nothing. Warm read 71–90 ns; fresh read 44–73 µs (WHERE dominates). Warm reuse breaks even at N = 1 and pays from N = 2 (1.96–2.03×), 8.1–8.4× at N = 8; at N = 64 fresh and rebind exceed 1,000 µs (failed gate), warm stays at 46 µs.
- Sixteen disable runs each turn a test red. The first pass found the projection-time seal checks unpinned; a test was added.
- The #1792 control still reproduces (`reused R2 1032 vs fresh 31`).

## Open

A detached `Keep` projected under a NEW seal is accepted: the seal attests the caller's claim, not mask provenance (needs a producer-side frame on Quack's bound query, D-NFR S5). `Criteria` is caller-supplied. The FALSE set costs a second WHERE program. One 64-slot frame per table; no DR256 resident register; no scattered-mask reuse.
