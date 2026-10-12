# 2026-10-12 — Class-qualified methods compile into Quack algebra (D-CMH-1)

**Status:** MEASURED for the slice below; everything past it is OPEN.
Code: `crates/lance-graph-glove-parity/src/{method,call}.rs`,
`crates/lance-graph-r2il/src/procs.rs`. Tests: `tests/method.rs`,
`tests/polyfill.rs` (glove-parity), `tests/procs.rs` (r2il). Plan context:
OGAR #357 (RFC, unmerged), lance-graph #1501 (plan, unmerged).

## What is executable

```text
SELECT <concept>::<method>(<table>) FROM <table>
  → resolve_call: concept via ogar_vocab::canonical_concept_id,
                  method label → MethodKey, payload = FROM table = bound table
  → MethodTable (prepared once per glove): SchemaSeal, ReadingId, Universe,
                  WideFieldMask required ⊆ available (Exact|Converted claims)
  → MethodBody::Query { filter: [Cond], folds: [Fold], law: Law }   (data)
  → Quack Query per fold → lower → mask-risc          (frontend 1)
  → R2IL procs::select_u32 + count|sum|group_sum      (frontend 2, parity)
  → Law finisher → Answer::Values | Answer::Truths
```

`src/` names no method. Every method in the tests is declared there as data:
`count_billable`, `count_not_billable`, `billable_hours` (two folds),
`count_by_duration` (refused), `submit_for_approval` (action, never run),
`revision` and `raw_sums` (NARS fixture concept).

Rust supplies two reusable layers, each written once: the **algebra**
(conditions, `Count`, `Sum`, `GroupSum`, lowered by both frontends) and the
**laws** (certified finishers: `Values`, `NarsRevision`). A method composes
them and cannot supply its own.

## Measured

- `billable_hours` via the call text = the same query written by hand in
  Quack = R2IL procs called by hand = a row loop, at n ∈ {1, 64, 65, 4097}.
- `revision` on both frontends = a float oracle (|Δ| < 1e-3) and OpenNARS's
  three reference revisions to two places; a statement with no evidence
  answers `None`.
- Refusals, each with a disable run that turns its test red: concept, method,
  payload, table, extra SQL; schema seal (two swapped positions, every name
  kept); reading; population; nullable field; basis of 257; evidence not
  selected as independent; law shape; negated condition and summed lane on
  each frontend; pure call of an action.

## Capability / insufficiency ledger

| Method / need | Missing algebra | Counterexample | Proposed extension | Oracle |
|---|---|---|---|---|
| NARS revision: exclude a repeated `(statement, stamp)` | A first-occurrence mask over an ordered composite key (`key[i] != key[i-1]`). Today the payload's producer computes it. | `a_repeated_stamp_inflates_the_raw_sums`: Σw 19 500 instead of 10 500, c 0.951 instead of 0.91. | Domain-neutral `FirstOfRun(keys…)` predicate over a lane ordered by the key, refused at run when out of order (same precondition `CountKeyRunsU32` already enforces). Reusable for any dedup (IAM, SAP, Odoo). | Row loop with a seen-set (`oracle` in polyfill.rs). |
| Grouped folds: a key outside `0..groups` | The terminal drops it silently. | `evidence_outside_the_declared_domain_is_dropped_silently` (pinned as a gap). | A dropped-row count beside the sink, or a refusal flag; executor-level, not per method. | Count of rows with key ≥ groups. |
| NULL / UNKNOWN | Conditions are two-valued; nullable fields are refused. | `a_nullable_field_is_refused`. | Bind through Quack's `sql_where` (three-valued `WHERE` already exists there) and declare a null policy per law. | SQL three-valued truth table. |
| `sql(query)` frontend | No SQL path reaches Quack (lance-graph's SQL path is DataFusion, in its grace period). `resolve_call` recognises one form. | Any other SQL shape → `Unrecognised`. | A real frontend hands its parsed call to the same `MethodCall`; nothing else changes. | — |
| NARS concept | Not minted in OGAR's codebook. | `SELECT nars::revision(..)` → `UnknownConcept`. | Mint through OGAR; the fixture `0xFFF0` then goes. | — |
| Method identity | OGAR has `MethodDecl` (harvested ORM methods, no key) and #357's `LogicMethodSpec` (RFC only). `MethodKey` is a fixture. | — | Key = `ogar_loco::Inventory::key_of` once methods are minted there; this table becomes the prepared sidecar, not a second inventory. | — |
| Reading and population ids | No ClassView reading id and no plane generation id exist. Both are fixtures here. | — | Issue from the ClassView and the plane owner. | — |
| Compute bodies (fields derived from fields) | Not attempted. The compute-DAG check is fail-open past 64 positions (`10→70→10` reads acyclic). | — | Fix the DAG check to 256 before any compute body lands. | — |
| Fused folds | One mask-risc program per fold; the selection is re-emitted per fold. | — | A multi-sink terminal; an optimisation only, no semantic change. | Same answers. |

No new mask-risc opcode or R2IL op was added. One R2IL change: `POP_COUNT`
now counts a resident plane in place (mask-risc's `Count` already accepted
one).

## Metrics so far

- Methods declared: 7. Executable with existing algebra: 5 (one needs the
  producer's mark, see row 1). Refused by design: 2.
- New reusable algebra members: grouped `SUM` exposed to methods (the
  terminal existed). New laws: 1 (`NarsRevision`). New ISA: 0. Domain Rust
  adapters: 0.

## Extension criteria

A new algebra member lands only with (1) a method that needs it, (2) a
counterexample showing no composition of existing members gives the answer,
(3) an independent oracle, (4) both frontends or a stated reason one cannot,
and (5) a second domain that can use it. A new law lands only with an
independent oracle and the body shape it requires checked at preparation.
