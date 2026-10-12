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
| Runtime declarations | Done (step 2): `decl::parse_decls` + `MethodTable::prepare_with`; one build loads any declarations file. Grammar: `COUNT(*)`, `SUM`, `GROUP_SUM … BY … IN`, `WHERE = / != … AND …`; everything else refused by name. | — | — | Row loop over indicator text (`tests/declared.rs`); the compiled twin's programs. |
| **Next: compositional formulas** — a declared method reused by another, e.g. `is_billable := billable = true`, `billable_hours := SUM(duration WHERE is_billable())`, `billable_count := COUNT(WHERE is_billable())`. | No method reference in `MethodBody`; a predicate method has no body shape (it returns a selection, not folds). Today `… WHERE is_billable()` is refused as `method reference`. | `unsupported_formulas_are_refused_by_name`. | A predicate body (`Select{filter}`) and a `Cond::Method(key)` resolved at preparation, inlined into the caller's conjunction, cycles refused; no new Rust per method. Tracked as D-CMH-G1. | The inlined method's programs equal the hand-written conjunction's. |
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

## Experiments A–E (same day)

Machine: `-Ctarget-cpu=x86-64-v3` (the repo default), AVX-512 host, shared
and noisy; hot paths interleaved per rep, medians and minima reported.
Command: `cargo run --release --example method_cost -- <rows> 301`.

**A. The seam.** `SELECT billable_work_entry::billable_hours(cats) FROM cats`
= the same query written in Quack = R2IL procs = a row loop, at n ∈ {1, 64,
65, 4097}. The prepared method now holds its Quack programs, lowered once;
the tests assert `programs() == [lower(direct query)]`, so a call executes
the same program a direct query does and no other evaluator exists. There
is no SQL frontend into Quack: "direct Quack" is the query built through
Quack's binder, not parsed SQL.

**B. Frozen executor.** At e984885, `hours_by_billable` (hours grouped by the
billable code) was declared in `tests/method.rs` only; that commit's stat is
one file. Quack: correct. R2IL: a typed refusal (`WrongOperandKind`), because
its SUM, GROUP_SUM and KEEP took a mask only from a computed slot and a body
with no condition hands them the live plane. Closed in the next commit by one
shared mask resolution in R2IL (domain-neutral; no new op), after which both
frontends agree. So the method needed no Rust; the executor had a gap that
the first condition-free method exposed.

**C. NARS.** Independent positive and conflicting evidence: matches the float
oracle and OpenNARS's three reference points. A repeated stamp, a replayed
delta, and the same stamp in a second hypothesis layer leave the truth
unchanged, given the producer's two marks (replay by `delta_id`, independence
by `(statement, stamp)`). Retraction: not declared; a negative-weight delta is
exact for an asserted stamp and silently wrong for a phantom one (pinned).
The oracle is the NAL weight formula in floating point plus the OpenNARS
points from `lance-graph-planner/tests/nal_revision_oracle.rs`; no external
NARS binary was run.

**D. Faults.** Each is a typed refusal before execution, or a red test:
swapped lanes (both frontends disagree with the oracle); schema seal; reading;
population; nullable field read as a value (the refusal test goes red when the
guard is removed); duplicate stamps (refused body / counterexample); law shape;
basis of 257. A compute body has no `MethodBody` variant, so it cannot be
written, rather than being refused at the 64-position DAG limit.

**E. Cost** (1,048,576 rows; 65,536 in brackets):

| path | median | min | ns/row (min) | allocs/call |
|---|---|---|---|---|
| cold: prepare one method | 0.8 µs | | | 13 (1.1 KB) |
| cold: resolve call text | 0.2 µs | | | 1 |
| cold: lower one query | 0.2 µs | | | 4 |
| warm method call, Quack | 600 µs (56) | 447 µs (33) | 0.43 (0.51) | 2 (2.1 KB) |
| warm method call, R2IL | 605 µs (56) | 448 µs (35) | 0.43 (0.53) | 5 (16.8 KB) |
| direct Quack, lowered once | 599 µs (54) | 442 µs (37) | 0.42 (0.56) | 0 |
| scalar row loop | 366 µs (23) | 300 µs (10) | 0.29 (0.15) | 0 |

The interface costs nothing measurable over direct Quack once prepared: the
remaining per-call cost is a scratch arena and the answer vector. Both are
1.5–3× slower than a fused scalar loop, because the program writes a mask
and then sums it (two passes over 8 MB of lanes); that is the executor's
cost, the same for a hand-written Quack query. Fusion of filter into fold is
the lever, not the method layer. Not measured: decoded bytes (lanes are read
resident), and fused-vs-unfused lowering (no fused terminal exists).

**Metrics after the experiments.** Methods declared: 8; executable on both
frontends: 6; refused by design: 2. New algebra members: 0 since the freeze;
executor fixes: 1 (R2IL plane masks, found by B). New laws: 1. New ISA: 0.
Domain Rust: 0.
