# D-DRCA-2: DR256::WHERE × DR#3

**Status:** MEASURED, lab only. Code: `crates/dr-math-binding-probe/src/where_dr3.rs`, test `tests/dr256_where_dr3.rs`, costs `examples/dr256_where_stages.rs`; results in the crate README, section D-DRCA-2. Builds on D-DRCA-1 (`hhtl16`, #1834). Contract: `.claude/plans/d-drdt-dr-dt-ownership-amendment-v1.md` (#1835).

## Result

- One Quack WHERE (`Draft::bind` + `Filter::sql_where`, mask-risc `Keep`, 65,536 addresses) projects into a 64-slot DR frame by `map.len()` probes through the counted `LocalMap`. No new predicate evaluator, routing engine, mask store or epistemic sidecar.
- Epistemic absence (`$null`, NaN, SQL NULL, unset validity, K3 UNKNOWN) is one logical value. WHERE selects TRUE; the unselected slots are FALSE or UNKNOWN and no complement is offered. A negation is its own WHERE, so `NOT UNKNOWN` stays UNKNOWN. A three-valued property keeps value + validity. No Epi5 state is derived from absence.
- Every read is re-admitted against owner, source, codebook, frame, register frame, axis, DR method (the ClassId-bound `MethodSignature`), focus, scope and criteria; each refusal is pinned by a test against an independent source-ID oracle.
- Projection, admission and folds allocate nothing. Measured back to back on the same fixture, the single WHERE halves the fresh read (41.5–43.2 µs → 20.6 µs). Warm reuse pays from N = 2 (2.0×), 7.9–8.0× at N = 8; at N = 64 fresh and rebind exceed 1,000 µs (failed gate), warm stays at 24–25 µs.
- Sixteen disable runs each turn a test red. The #1792 control still reproduces (`reused R2 1032 vs fresh 31`).

## Open

A detached `Keep` projected under a NEW seal is accepted: the seal attests the caller's claim, not mask provenance (independent gate, D-NFR S5: a producer-side frame on Quack's bound query). `Criteria` is caller-supplied. One 64-slot frame per table; no DR256 resident register; no scattered-mask reuse.
