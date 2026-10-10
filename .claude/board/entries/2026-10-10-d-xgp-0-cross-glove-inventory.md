# 2026-10-10 — D-XGP-0: cross-glove inventory — SAP and Odoo share a type, not a meaning

**Status:** MEASURED (read-only inventory). Plan: `.claude/plans/cross-glove-business-parity-v1.md` §1.

Findings that change what can be claimed:

- `lance-graph-sap/tests/view_convergence.rs` uses Odoo `account.move` (216 fields), not `account.analytic.line`. It shows both gloves use `WideFieldMask`, not that any field corresponds.
- The "SAP" schema in `lance-graph-sap` is SIMAF's time-tracking DTO, not CATSDB. No CATSDB / `BAPI_CATS_INSERT` field list exists in SIMAF or SIMAFPort, and there is no SAP runtime evidence.
- Quack `Registrar` registers name + width only and has no production implementation (`quack/src/bind.rs:53-55,364-412`). `Binder` is the field-level seam.
- CATS hours carry a batch-local decimal scale (`sap/src/bind.rs:65-98`); cross-batch sums need an explicit rescale.
- Odoo base `account.analytic.line` has no `employee_id`, `project_id` or `task_id` (they come from `hr_timesheet` / `project`, not extracted).
- `0x000B` is `HubSpoPort`; the next free `APP_PREFIX` at OGAR `2fbae6e` is `0x000C`.
- "4×SPOG" is defined nowhere (searched OGAR crates/docs/board and lance-graph crates/.claude); OGAR docs name 3× SPOG quads. `Register128` is shipped and content-blind.

OPEN: canonical basis v1 for `0x0103` (operator rulings, plan §10).

CORRECTION (same day): `OntologyRegistry` already provides the cross-bridge field identity (same URI ⇒ shared `entity_type_id`, kind-agnostic; `registry.rs:585-618`). The CATS path bypasses Quack `Binder` (`query.rs:21-62`). The missing piece is one adapter: CATS `impl Binder` plus a registry front. The proposed `glove.rs` is superseded (plan §C).
