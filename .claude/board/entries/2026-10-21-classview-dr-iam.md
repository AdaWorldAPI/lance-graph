# 2026-10-21 — ClassView × DR, IAM: a behavior compiled from what the class knows

**Status:** MEASURED (lab, `crates/dir-rmap-probe`). **Basis:** operator
kickoff "Quack Binding | ClassView | DR" and its add-on "compile behavior
through relationship mapping" (2026-10-21). No new relationship,
association, method-descriptor or planner type.

## What runs

- The user class declares, through the existing `ClassView::compute_dag`,
  that `effective_roles` (position 1) is computed from `member_of`
  (position 0).
- A direct-routing table maps position 1 to the existing compiled closure
  fold `Compiled::roles_of`.
- `tests/classview_dr.rs`: the routed behavior equals dir-sim's own walk and
  the crepe oracle for every user; a doubly supported role survives losing
  one support; a class that does not declare the behavior computes nothing.
  Disable runs: a wrong route fails two tests, ignoring the declaration
  fails the silence test.

## Measured (`examples/classview_dr_cost.rs`, 8,000 users, 1,495 groups)

Per user, every answer checked equal to the direct path first:

| route | ns |
|---|---|
| direct dir-sim walk (existing path, negative control) | ~450,000 |
| direct prepared CSR walk | ~190 |
| compiled fold, called directly | ~13–19 |
| ClassView × DR via `execute_compute_dag` per row | ~100 |
| ClassView × DR, declaration resolved once per class | ~13–19 (±0.4 over the fold) |

Preparation (`prepare` + `compile`, once per version): FAILED GATE
(millisecond scale), the same preparation D-RMAP-DIR-1 already carries.

## Findings, per stratum

- **Correctness:** equal to dir-sim and the oracle (test-pinned).
- **Dispatch:** `execute_compute_dag` recomputes the topological order and
  allocates on every call (~85 ns per row). Resolving the declaration once
  per class removes it. ClassView decides once; only the routed fold runs
  per row.
- **Binding legality:** group slots are dir-sim ordinals of one snapshot
  version; the probe's `Witness` is lab-local. The contract-level
  frozen-codebook witness is still missing (shared with D-CMR-5, D-RU-6).
- **SAP (bypass):** `BAPI_ORDINALS` is already a position map over
  `CatsSchema`'s ClassView basis; existing tests (`edges.rs`
  `bapi_order_is_a_permutation_over_canonical_field_identity`, byte pins)
  pass. KEEP_DIRECT: no DR table, no new code.

## Open

- Whether to add a once-per-class resolution helper next to
  `execute_compute_dag` in the contract (the measured dispatch cost).
- The Odoo method case and relalg as a routed method are not run yet.
- Preparation fails the gate.
