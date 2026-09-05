# D-OIF-1 census — re-derived from the tree (main thread, 2026-09-05)

> Operator brief: re-open `open-ideas-fetch-v1` / #1185 and re-derive D-OIF-1 from the current
> architecture before any code. Four read-only Sonnet tracers (DataFusion policy stack; RBAC /
> ClassView / WideFieldMask; Kanban/Rubicon lifecycle; production DataFusion/Lance consumers)
> plus main-thread reads. Every classification is a call-path fact from a real entry point
> (`main.rs`, `src/bin/*`, handlers, PyO3 constructors) — never a `pub mod`, feature flag,
> registration helper, test, or comment. Ruling landed the same day:
> E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1 (closes the one seam left open in §4).

## 1. Verdict

`policy_hash_v1` is the last visible piece of an execution model that never reached production.
The DataFusion policy stack is test-only; no `add_optimizer_rule`/`add_analyzer_rule` anywhere;
the only live query surface (Python bindings) has no policy step; RBAC's canonical successor is
transport-only; lifecycle is SoA-owned as hypothesised. **D-OIF-1 = A (SUPERSEDED / REMOVE).**

## 2. Census

| old obligation | current owner | production evidence | action |
|---|---|---|---|
| `policy_hash_v1` / `NotYetWiredHashUdf` | none | `policy.rs:279-340`; only via `mask_expr` (`:137`), reached only from tests `:600-828`; no `register_udf` of that name | REMOVE (VACANCY) |
| `RedactionMode::*` | none | `policy.rs:71`; every construction in `mod tests` | REMOVE with module |
| `ColumnMaskRewriter` / `ColumnMaskRegistry` / `PolicyRewriter` / `PolicyKind` | none | one impl, zero non-test callers; `unified_bridge.rs:12`, `lib.rs:114` mention it as future | REMOVE (SUPERSEDED by project-then-query) |
| `RowEncryptionPolicy` / `DifferentialPrivacyPolicy` (+registries) | none | `policy.rs:344-420` "stub, no executor yet", zero callers | REMOVE with module |
| `RlsRewriter` / `RlsPolicyRegistry` / `MembraneRegistry::with_rls` / `postgrest.rs` dispatcher stub | duplicates canonical `ClassRbac::row_scope` (`rbac.rs:166`, default `None`) | `rls.rs:284` only `mod tests`; `postgrest.rs:940-960` comment + `Err("not yet implemented")`; CI runs `--features query --lib` only | REGRADE grace-period duplicate; keep until `row_scope` enforcement proven |
| `register_vsa_udfs` | none | `vsa_udfs.rs:574`, `pub use` `lib.rs:69`, zero call sites | separate query-side card; grace |
| DataFusion optimizer/analyzer registration | none | zero calls repo-wide | nothing to remove |
| `ClassView` | contract `class_view.rs:946`; impls `RegistryClassView`, `WikidataClassView`, `OgarClassView` | tests, planner probes, OGAR doc-ir/render | RETAIN |
| `WideFieldMask` | contract `class_view.rs:243` (`intersect` `:385`) | `recipe_vocab`, `standing_mask`, `step_mask`, `selection`, `attention_facet`, OGAR `doc-ir::project::field_mask` (`project.rs:75`) | RETAIN |
| `ogar-rbac` | no such crate | machinery lives in `lance-graph-contract::rbac` + `lance-graph-rbac` + `lance-graph-ogar::rbac_impl` | REGRADE the name in docs |
| `ClassRbac` / `authorize()` / `OgarRbac` | canonical path, TRANSPORT ONLY | `authorize.rs:60` callers only tests; `rbac_impl.rs:64` "§6 follow-on"; `medcare_actor.rs:100` `// TODO` | RETAIN; enforcement = missing implementation |
| `effective_mask = classview_mask ∧ role_mask` | nowhere computed | `ClassRbac::field_mask` default `FieldMask::FULL`, narrow (`rbac.rs:176-184`); charter C1.4 retype not done | spec stands; implement on canonical path, never as UDF |
| `UnifiedBridge::authorize_{read,write,act}` | string-keyed `Policy` gate | `unified_bridge.rs:362-410`, own tests only | RETAIN, TRANSPORT ONLY |
| `KanbanColumn` / `KanbanMove` / `try_advance_phase` | SoA (`MailboxSoA`) | write `mailbox_soa.rs:953` via `soa_view.rs:311`; applied `cycle_driver.rs:560`; `emit_bootstrap_intent` (`owner_adapter.rs:92`, `cycle_driver.rs:726`) casts an INTENT, the seal applies | RETAIN; invariant holds |
| `KanbanActor` / `KanbanMsg` | deleted | `kanban_actor.rs:1-33` tombstone | — |
| baton / `CollapseGateEmission` / `emit()` | removed | prose only: `transaction/{interactive,bulk}.rs`, `episodic_edges.rs:259`, CLAUDE.md (annotated) | REGRADE prose as stale |
| `ActionState` (OGAR) | action semantics, `ogar-vocab/src/lib.rs:676` | wire state, not a `KanbanColumn` | RETAIN, distinct layer |
| `CommitHook` | prose only | no definition anywhere | VACANCY; regrade prose |
| `ogar-loco` / `ogar-r2il` | execution/reasoning | zero refs to kanban, RBAC, `try_advance_phase` | RETAIN; no impersonation |

## 3. Heckhausen / Rubicon mapping (verified, no second controller)

`Planning` = pre-decisional; `advance_on_gate` (`kanban.rs:206`) = the crossing, decided by the MUL
gate; `CognitiveWork` = committed action; `Evaluation` = post-actional; `Commit/Plan/Prune` = absorbing.
`emit_bootstrap_intent` casts an intent into `BatchWriter`; the seal applies it through
`try_advance_phase`. No actor decides a transition.

## 4. Removal cone / retain / regrade

- REMOVE (one PR): `crates/lance-graph-callcenter/src/policy.rs` entire; `pub mod policy` gate `lib.rs:114-124`;
  `.claude/patterns.md:89` row; module-header `policy` feature note.
- RETAIN: `rls.rs` + `MembraneRegistry::with_rls` (grace; replacement `row_scope` unproven); `unified_bridge.rs`;
  `lance-graph-rbac/src/authorize.rs`; `contract/src/rbac.rs`; `contract/src/class_view.rs`; `rbac_impl.rs`;
  `datafusion_planner`, `sql_query`, `graph_table`, Python bindings (all grace).
- REGRADE: `unified_bridge.rs:12` + `super-domain-rbac-tenancy-v1.md` §3.9/§13.1 (stage 4 is
  `ClassRbac::field_mask ∩ ClassView` → column list, never a rewrite rule); `postgrest.rs:940-960`;
  baton prose; OGAR `CommitHook` prose.
- MISSING IMPLEMENTATION (canonical path, not this PR): `field_mask` retype to `WideFieldMask` (C1.4);
  `authorize → {scope, mask}` (keystone §5 stage 2); the projection consumer — Lance reads / loco
  programs take the authorized column list (the DataFusion variant of this seam is CLOSED by the
  2026-09-05 ruling); `PROBE-OGAR-RBAC-AUTHORIZE` step 5; `medcare_actor.rs:100`.

## 5. Stale claims to correct in #1185

§2 (D-OIF-1) → retirement plan; §2.2 D-OIF-1-DEC (hash + key) moot; G1–G8 withdrawn; `PolicyHashUdf` /
`register_policy_udfs` deliverables dropped; STATUS_BOARD `D-OIF-1` / `D-OIF-1-DEC` → SUPERSEDED;
IDEAS.md `IDEA-POLICY-HASH-UDF` → Superseded (its "blocker is the UDF body" flip was itself wrong: the
blocker was the absence of any consumer); OGAR DISCOVERY-MAP "no impl ClassRbac under crates/" is true of
OGAR only. Tracer error caught: `emit_bootstrap_intent` exists (`owner_adapter.rs:92`).
