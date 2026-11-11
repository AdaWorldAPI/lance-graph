# D-DRDT-CLASS-GOVERNANCE-1 — Bypass matrix

Status: census, docs only. Read at main 238b6d1e4 (2026-11-11).

Question: which code paths execute a mask-RISC program without first going through DT64 admission (`BindStore::admit`)?

Method: each site was found by search, then the enclosing function was read. Search was used only to find candidates. Paths are relative to `crates/`. Nothing was built or run.

## Reference: the admitted path

- `lance-graph-r2il/src/binding.rs:322-334` (`prepare`): `r.guards.check(...)`, `check_route(...)`, `store.resolve(...)`, `store.admit(r.method)`.
- `lance-graph-r2il/src/binding.rs:377` (`run`) calls `run_steps`, which calls `call_step` on a `FoldDialect`. The fold reaches `execute_into` at `lance-graph-r2il/src/lib.rs:789` (`finalize_and_run`).
- `lance-graph-r2il/src/binding_native.rs:203` (`prepare_native`) and `:351` (`run_native`) take a `Prepared` plan. They call `store.check_issuer`, `guards.check` and `check_route`. They do not call `admit` themselves. `run_native` also requires an `ExecPermit` (`:367`-`:369`). They execute through `execute_compiled` (`:385`), not `execute_into`.
- `BindStore::admit` is at `lance-graph-dt64/src/lib.rs:1180`. It checks: class known, method known, signature revision, guard witness, binding digest, issuer. It does not check authorization.

Route key used below:
- ADMITTED: the enclosing function, or the documented path to it, calls `BindStore::admit`.
- BOUND: a DT64 `LaneBinding` check only, no method admission.
- DIRECT: neither.
- INTERNAL: the definition or a wrapper inside mask-risc.

## Table (the 40 candidate lines)

Crate status comes from the root `Cargo.toml` `members` list as given by the orchestrator. It was not re-derived, except `lance-graph-sap` (listed in the root `exclude`, line 53) and `lance-graph-relalg` (own `[workspace]`, listed at root line 133).

### Non-test sites

| file:line | crate status | prod/test | route | program source | evidence (function read) |
|---|---|---|---|---|---|
| lance-graph-r2il/src/lib.rs:789 | MEMBER | PROD | ADMITTED when reached via `binding::run`; DIRECT for any caller that builds a `FoldDialect` itself (see "Beyond the list") | R2IL fold (`FoldDialect`) | `finalize_and_run` (lib.rs:770). `FoldDialect::new` is `pub` (lib.rs:389). |
| lance-graph-cypher-quack/src/lib.rs:729 | MEMBER | PROD | DIRECT | `lance_graph_quack::lower` (lib.rs:305) | `Compiled::execute` (lib.rs:705). No dt64 in the crate's Cargo.toml. |
| lance-graph-xlog/src/lib.rs:220 | MEMBER | PROD | DIRECT (see note 1) | `lance_graph_quack::lower` (lib.rs:214) | `run_lanes` (lib.rs:205), called by `run` (lib.rs:198) |
| lance-graph-report/src/render.rs:320 | MEMBER | PROD | DIRECT | `lance_graph_quack::lower` (render.rs:308) | `Terminal::detail_csv` (render.rs:300). `lib` depends on dt64, but this function does not use it. |
| lance-graph-mask-risc/src/graph_hop.rs:160 | MEMBER | PROD | DIRECT | `Program::new` (graph_hop.rs:66) | `run` (graph_hop.rs:101). `#[cfg(test)]` starts at :175. |
| foundry-parity-lab/src/method.rs:302 | EXCLUDED | PROD | ADMITTED | not determined (no `lower(` or `Program::new` hit in the file) | `PreparedMethod::run` (:295). `prepare` calls `world.store.admit(r)` at :217. |
| crepe-plan-probe/src/coaddressed.rs:396 | EXCLUDED | PROD | DIRECT | `Program::new` (:377) | `project` (:370) |
| grammar-constraints-probe/src/when.rs:499 | EXCLUDED | PROD | ADMITTED at compose time | `lance_graph_quack::lower` (in `compose`) | `Composed::run` (:452). `compose` calls `store.admit(mref)` at :421. `run` itself does not re-admit. |
| grammar-constraints-probe/src/lib.rs:577 | EXCLUDED | PROD | DIRECT | `Program::new` (:321, :381) | `run` (:538). Re-checks the DR frame only. |
| mork-pathmap-probe/src/rmap.rs:816 | EXCLUDED | PROD | DIRECT | `Program::new` (:785) | `hop` (:794) |
| mork-pathmap-probe/src/rmap.rs:1309 | EXCLUDED | PROD | DIRECT | UNKNOWN (`prog` origin not read; `box_program` at :1163 builds programs) | `intersect_planes` (:1286) |
| mork-pathmap-probe/src/join.rs:552 | EXCLUDED | PROD | DIRECT | `Program::new` (:541) | `joint` (:512). The module doc (join.rs:7) says the binding is "not admitted". |
| drisa-register-probe/src/lib.rs:721 | EXCLUDED | PROD | DIRECT | UNKNOWN for `derive_gt` (not traced); `MethodOp` programs use `Program::new` (:562-:590) | `derive_gt` (:702). dt64 use is `dr` and `ValueDemand` only. |
| cognitive-polyfill-probe/src/lib.rs:409 | EXCLUDED | PROD | DIRECT | `Program::new` (:387) | `run` (:400) |
| lance-graph-dir-sim/src/exec.rs:24 | EXCLUDED (not in `members`) | PROD | DIRECT | `lance_graph_quack::lower` (exec.rs:19) | `run` (exec.rs:22). dt64 is a `[dev-dependencies]` entry only (Cargo.toml:50-51), and no `dt64` string appears in `src/`. |
| jc-batch-probe/src/lib.rs:290 | EXCLUDED | PROD | DIRECT | `Program::new(Vec::new(), terminal)` (:285) | `exec` (:278) |
| lance-graph-sap/src/query.rs:90 | EXCLUDED (root `exclude`) | PROD | DIRECT | `lance_graph_quack::lower` (query.rs:38) | `execute_into` method (:76) |
| lance-graph-sap/src/query.rs:109 | EXCLUDED (root `exclude`) | PROD | DIRECT | `lance_graph_quack::lower` (query.rs:46) | `select_into` (:102) |
| lance-graph-sap/src/query.rs:76 | EXCLUDED | n/a | INTERNAL | n/a | This line is the definition of a method named `execute_into` on a sap type, not a call to mask-risc's. |
| lance-graph-mask-risc/src/exec.rs:769 | MEMBER | PROD | INTERNAL | n/a | `execute` (exec.rs:763), a thin wrapper over `execute_into` (doc at exec.rs:755-762). |
| lance-graph-mask-risc/src/exec.rs:1343 | MEMBER | PROD | INTERNAL | n/a | Definition of `execute_into`. |
| lance-graph-mask-risc/src/reference.rs:983 | MEMBER | PROD | INTERNAL | n/a | `reference_execute` calls `reference_execute_into`, the row-by-row oracle. It is not a call to `execute_into`. |
| lance-graph-mask-risc/src/reference.rs:991 | MEMBER | PROD | INTERNAL | n/a | Definition of `reference_execute_into`. |

### Test sites

| file:line | crate status | prod/test | route | program source | evidence (function read) |
|---|---|---|---|---|---|
| lance-graph-quack/src/lib.rs:2884 | MEMBER | TEST | DIRECT | not traced | helper `exec` (:2873), inside `mod tests` (:2511) |
| lance-graph-quack/src/lib.rs:2896 | MEMBER | TEST | DIRECT | not traced | helper `exec_mask` (:2892), inside `mod tests` |
| lance-graph-quack/src/lib.rs:3726 | MEMBER | TEST | DIRECT | not traced | `semantic_group_sum_collapses_the_materialized_k_fold_forest` (:3660), `#[test]` at :3659 |
| lance-graph-quack/src/lib.rs:3789 | MEMBER | TEST | DIRECT | not traced | same test fn (:3660) |
| lance-graph-quack/src/lib.rs:4174 | MEMBER | TEST | DIRECT | not traced | `eq_u32_via_matches_an_independent_per_row_reading` (:4130) |
| lance-graph-quack/src/lib.rs:4235 | MEMBER | TEST | DIRECT | not traced | `semijoin_matches_an_independent_per_row_reading` (:4194) |
| lance-graph-quack/src/lib.rs:4287 | MEMBER | TEST | DIRECT | not traced | `scatter_or_u32_matches_an_independent_per_row_reading` (:4255) |
| lance-graph-quack/src/lib.rs:4334 | MEMBER | TEST | DIRECT | not traced | `group_sum_i32_matches_an_independent_per_row_reading` (:4304) |
| lance-graph-quack/src/lib.rs:4410 | MEMBER | TEST | DIRECT | not traced | `group_sum_via_i32_matches_an_independent_per_row_reading` (:4354) |
| lance-graph-quack/src/lib.rs:4513 | MEMBER | TEST | DIRECT | not traced | `rows_of` (:4498), `#[cfg(test)]` at :4426 |
| lance-graph-quack/src/lib.rs:4855 | MEMBER | TEST | DIRECT | not traced | `rows_of` (:4827), `#[cfg(test)]` at :4741 |
| lance-graph-quack/src/lib.rs:5076 | MEMBER | TEST | DIRECT | not traced | `match_facet16_lowers_and_reads_all_sixteen_bytes` (:5029), `#[cfg(test)]` at :5020 |
| lance-graph-quack/src/lib.rs:5182 | MEMBER | TEST | DIRECT | not traced | `count` (:5173), `#[cfg(test)]` at :5094 |
| lance-graph-quack/src/bind.rs:634 | MEMBER | TEST | DIRECT | `lower(q)` (:626) | `run` (:608), inside `mod tests` (:540) |
| lance-graph-quack/src/bind.rs:892 | MEMBER | TEST | DIRECT | `lower(q)` (:890) | `count_state` (:880), inside `mod tests` |
| lance-graph-report/src/lane_source/tests.rs:296 | MEMBER | TEST | BOUND (not fully read) | not read | `bound_lanes_are_the_batch_lanes_and_fold_with_scalar_parity` (:233). File imports `LaneBinding`, `BindStore` (:14-16). |
| lance-graph-report/src/lane_source/tests.rs:297 | MEMBER | TEST | BOUND (not fully read) | not read | same test fn |

Note 1 (xlog): `run_lanes` has no admission. The public `support` (lib.rs:255) and `bag_count` (lib.rs:280) are free functions with no `MethodBinding` argument. `bag_count_admitted` (lib.rs:384) takes a `&MethodBinding` and checks its shape (population, effect words). It does not call `admit`; the module doc (lib.rs:34-35) says it runs only from a binding the store admitted, and the caller supplies it. `Intake` re-checks the DR `MapTable` frame (lib.rs:378-380), which is not DT64 method admission.

## Summary counts

Counted over the 35 call sites (the 40 candidate lines minus 4 INTERNAL lines in mask-risc and the `lance-graph-sap/src/query.rs:76` definition).

| route | PRODUCTION in MEMBER | PRODUCTION in EXCLUDED | TEST | total |
|---|---|---|---|---|
| ADMITTED | 1 (r2il lib.rs:789) | 2 (foundry method.rs:302, grammar when.rs:499) | 0 | 3 |
| BOUND | 0 | 0 | 2 (report lane_source/tests.rs:296, :297) | 2 |
| DIRECT | 4 (cypher-quack 729, xlog 220, report render 320, mask-risc graph_hop 160) | 11 (crepe 396, grammar lib 577, mork 816/1309/552, drisa 721, cognitive-polyfill 409, dir-sim 24, jc-batch 290, sap 90/109) | 15 (quack lib.rs 13, quack bind.rs 2) | 30 |
| total | 5 | 13 | 17 | 35 |

## Beyond the list (found while reading, not counted above)

- `lance-graph-mask-risc/src/exec.rs:763` `execute` (the wrapper) is called by:
  - `drisa-register-probe/src/lib.rs:694`: DIRECT, EXCLUDED, PROD.
  - `foundry-parity-lab/src/arm_b.rs:275`: EXCLUDED, PROD. Lanes come from a `LaneBinding` through `bind.rs:171` (`bind_lanes`). BOUND. No `.admit(` call in `arm_b.rs`.
  - `foundry-parity-lab/src/arm_c.rs:206`: EXCLUDED, PROD. `arm_c.rs:130` defines its own `admit(&self, menu)`, which is not `BindStore::admit`. Route UNKNOWN.
- `lance-graph-r2il/src/lib.rs:389` `FoldDialect::new` is `pub`. Other callers: `lance-graph-relalg/src/lower.rs:112` (EXCLUDED crate, own `[workspace]`; no `admit` or `dt64` match in its `src/`) and `r2il-mask-abi-probe/tests/row_bridge.rs` (many test sites). Those paths reach `finalize_and_run` without `binding::run`: DIRECT.
- Programs in the DIRECT rows come from `lance_graph_quack::lower` or hand-built `Program::new`. Only the R2IL fold row comes from the fold.

## What you could not determine

- foundry-parity-lab `method.rs:302`: program source. No lowering call in the file; I did not read `prepare` past the `admit` line (:217).
- mork-pathmap-probe `rmap.rs:1309`: where `prog` comes from.
- drisa-register-probe `lib.rs:721` `derive_gt`: program source, and whether anything upstream of it admits. I only saw the `dr` and `ValueDemand` imports from dt64.
- report `lane_source/tests.rs:296/297`: classified BOUND from the file's imports and the test name. I did not read the test body or confirm which programs `sum` and `count` are.
- foundry-parity-lab `arm_c.rs:206`: route (see above).
- quack test sites in `lib.rs`: program source not traced. They are inside `#[cfg(test)]` modules or `#[test]` functions, found by reading upward for `mod tests` and `#[cfg(test)]`; I did not read each test body.
- EXCLUDED status for the probe crates comes from the orchestrator's `members` list ("everything else is excluded"). I did not read the root `exclude` list in full.
- Whether any other workspace code (outside `crates/*/src`) executes programs: not searched.

## Second entry points not covered

These are separate execution routes that never call `execute_into`. They are named only, not analysed.
- DataFusion / Cypher: `CypherQuery::execute` at `lance-graph/src/query.rs:241`, which dispatches by `ExecutionStrategy` (`DataFusion`, `BlasGraph`, `LanceNative`).
- Authorization in `lance-graph-callcenter`: `authorize_read` at `lance-graph-callcenter/src/unified_bridge.rs:362` (also `authorize_write` :379, `authorize_act` :396). It evaluates a policy and emits an audit event.

## Dependency on `lance-graph-dt64` (from each `Cargo.toml`)

- `lance-graph-mask-risc/Cargo.toml`: no match for `dt64` (case-insensitive grep). No dependency.
- `lance-graph-quack/Cargo.toml`: no match. No dependency.
- `lance-graph-cypher-quack/Cargo.toml`: no match. No dependency.
- `lance-graph-report/Cargo.toml`: yes, line 20: `lance-graph-dt64 = { path = "../lance-graph-dt64" }`, with the comment at lines 18-19: "The DT64 lane socket (`lane_source`). Depends on the contract only, so report -> dt64 adds no cycle."
- `lance-graph-xlog/Cargo.toml`: yes, line 13: `lance-graph-dt64 = { path = "../lance-graph-dt64" }`.
- Also read: `lance-graph-r2il/Cargo.toml:29` `lance-graph-dt64 = { path = "../lance-graph-dt64" }`; `lance-graph-dir-sim/Cargo.toml:51` has it under `[dev-dependencies]` only.

## What follows from the table

- The dependency direction means mask-risc and quack cannot call `admit`: neither depends on dt64. Admission exists only in code that sits above both and chooses to use it (r2il `binding`, foundry-parity-lab, grammar-constraints-probe `compose`).
- Among PRODUCTION sites in MEMBER crates, 4 of 5 are DIRECT. Of the 4 DIRECT, 3 take a program from `lance_graph_quack::lower` and 1 (`graph_hop`) is hand-built.
- `admit` does not check authorization (`lance-graph-dt64/src/lib.rs:1180`), so ADMITTED here means method identity, revision, guard and issuer checks only.
