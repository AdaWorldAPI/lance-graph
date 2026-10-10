# Handoff → Odoo session: land the analytic-line pivot (D-XGP-4)

> From: cross-glove parity session (2026-10-10). To: the odoo-rs / Ruff
> harvest session. Self-contained. You do not need to read anything about SAP,
> BAPI, IDoc, SCOT or transports.
> Plan: `lance-graph/.claude/plans/cross-glove-business-parity-v1.md` (§3, §5, §7).
> Status: the contract is PROPOSED. **Do not start until PR-B (contract types)
> and PR-C (basis v1) have merged.** The contract may still change in review.

## What you are building

You are mapping Odoo `account.analytic.line` onto the canonical field basis of
OGAR concept `BILLABLE_WORK_ENTRY` (`0x0103`), as **data**, plus a
classification of the model's harvested behavior. Nothing more.

## 1. Landing crate / module

- Code: `odoo-rs/crates/od-ontology/src/glove.rs` (new module), behind the
  existing `fieldmask` feature, which already pulls `lance-graph-contract`.
- Data: `odoo-rs/crates/od-ontology/data/glove/account_analytic_line.pivot.tsv`.
  This file is generated; never hand-edit it.

## 2. Accepted input types and required fields

You produce exactly one `lance_graph_contract::glove::GlovePivot`, built from
the TSV. The types are in the plan, §3.1.

| field | required value |
|---|---|
| `app_prefix` | `OdooPort::APP_PREFIX` (`0x0002`), read through `ogar_vocab`; **never a literal** |
| `concept` | `ogar_vocab::class_ids::BILLABLE_WORK_ENTRY`, read; never `0x0103` typed out |
| `basis_version` | the version of `ogar_class_view::basis::BILLABLE_WORK_ENTRY` you validated against |
| `entries[]` | one `PivotEntry{native, canonical, grade}` per Odoo field you map |

`native` is the field's position in Odoo's `ClassView` for
`account.analytic.line`. That is the same ordinal `ogar-class-view`'s
`lift_object_view` assigns: attributes first, then associations.

TSV columns: `native_name, native_ordinal, canonical_name, grade, conversion, source_pin`.
`source_pin` is `file:line` in the Odoo source at the commit the harvest ran on.

## 3. Ownership and dependency direction

- You **own** the pivot and its TSV.
- OGAR owns the basis. lance-graph owns the types and the parity probe.
- **Permitted edges:** `od-ontology → lance-graph-contract`,
  `od-ontology → ogar-class-view`, `od-ontology → ogar-vocab`,
  `od-ontology → ogar-from-ruff`.
- **Forbidden:**
  - any SAP crate (`lance-graph-sap`, `-sap-sim`);
  - `lance-graph-quack`, `mask-risc` or `report` as normal dependencies;
  - any edit to OGAR facet keys or `class_ids`.

## 4. Versioning and compatibility

- A pivot declares the `basis_version` it was written for. A basis may only
  append fields.
- When the basis gains fields, you may add entries. You never renumber.
- If the basis version you need does not exist, or a field you need is not in
  it, **STOP** and raise it in session. Do not add a field to the basis
  yourself; the basis lives in OGAR.
- Regenerate the TSV whenever the harvest commit changes, and update
  `source_pin` with it.

## 5. One complete example mapping

Odoo base `analytic` module, `account.analytic.line` fields
(`lance-graph-ontology/src/odoo_blueprint/extracted/analytic.rs:482`).

The rows below are examples only. Basis v1 is not final, so the canonical names
can change in review; take the real ones from the basis once it lands.

```tsv
native_name	native_ordinal	canonical_name	grade	conversion	source_pin
date	1	work_date	Exact	-	odoo/addons/analytic/models/analytic_line.py:<line>
unit_amount	3	quantity	Converted	uom_time_to_hours	odoo/addons/analytic/models/analytic_line.py:<line>
company_id	7	company	Exact	-	odoo/addons/analytic/models/analytic_line.py:<line>
amount	2	cost_amount	Exact	-	odoo/addons/analytic/models/analytic_line.py:<line>
name	0	description	Exact	-	odoo/addons/analytic/models/analytic_line.py:<line>
user_id	6	worker	Hypothesized	-	odoo/addons/analytic/models/analytic_line.py:<line>
partner_id	5	customer	Hypothesized	-	odoo/addons/analytic/models/analytic_line.py:<line>
```

`native_ordinal` must be **read** from the ClassView you lift. The numbers
above are illustrative.

Why `user_id` is `Hypothesized`: a login user is not an HR employee. `worker`
waits on operator ruling 2 in the plan (§10). If the ruling is "HR employee",
you will also need the `hr_timesheet` harvest (`employee_id`, `project_id`,
`task_id`), which is not extracted today.

## 6. Behavior classification (same TSV directory)

Write `account_analytic_line.behavior.tsv` with one row per harvested method.
Each row gets exactly one class from the plan, §7:

`DeclarativeFact | OgarPrimitive | ExecutableActionDef | ResidualWrapper | Hypothesized | Unsupported`

Carry the existing OGAR value where one exists:

- the `KausalSpec` variant for `OgarPrimitive`;
- the `ActionDef.identity` for `ExecutableActionDef`;
- the `ResidualRepresentation.unresolved_reason` for `ResidualWrapper`.

Known methods (`data/odoo_inheritance_manifest.ndjson:31-34`):

- `_compute_general_account_id`
- `_check_general_account_id`
- `_compute_partner_id`
- `on_change_unit_amount`

## 7. Tests you must add

Positive:

- The pivot validates against the basis (laws 1–5 from contract, plan §3.2).
- Every `Exact`/`Converted` entry's Odoo field type matches the canonical
  domain after conversion.
- The TSV round-trips into the `GlovePivot` byte-for-byte.
- `app_prefix` and `concept` equal the `ogar_vocab` values (no literals).

Negative (each must fail when its guard is removed; commit first, then disable):

- Mapping `unit_amount` as `Exact` (no UoM conversion) is rejected by law 2.
- A duplicate `native` ordinal is rejected by law 1.
- An entry pointing past the basis length is rejected.
- `RecomputeDag`: a constructed DAG that orders `_check_general_account_id`
  before `_compute_general_account_id` is detected (plan N7).

## 8. What you must implement

1. A `glove.rs` loader: TSV → `GlovePivot`, plus validation tests.
2. A generator from the ruff harvest (`ruff_python_spo` → `ModelGraph` →
   OGAR `ClassView` ordinals) → TSV. The generator is code, and the TSV is its
   output.
3. The behavior classification TSV and its loader.
4. A committed Odoo **row fixture** for the parity probe:
   `account_analytic_line.rows.tsv`, with 20–50 rows and only fields graded
   `Exact`/`Converted`. Synthetic values, no real customer data. Include rows
   in a non-time UoM so that negative control N1 has something to refuse.

## 9. What you must NOT implement

- Anything SAP-side, any `SapPort`, any CATS mapping.
- A query, filter, fold, binder, lane view or report: these live in
  lance-graph (the probe crate).
- An edit to the canonical basis, to OGAR facet keys or to `class_ids`.
- Promoting any grade because two fields "look the same" or because
  ARM/CE64 evidence correlates them. Only the parity probe promotes, after
  review.
- Executing `ActionDef`s, writing to a database, or touching transactions,
  authorization or company scope.
- A second field vocabulary. The pivot **is** the mapping.

## 10. Done means

The PR into odoo-rs contains:

- the module;
- the three TSVs: pivot, behavior and rows;
- the tests, green under
  `cargo test -p od-ontology --features fieldmask`;
- the disable runs, recorded in the PR body;
- the harvest commit the TSVs were generated from.

The lance-graph probe (PR-F) then consumes the pivot TSV and the rows TSV
unchanged.
