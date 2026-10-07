# 2026-10-07 — D-PEARL-PROD-0: Pearl's ladder in production

**Code:** `lance_graph_contract::certification`, `lance_graph_planner::pearl`.
**Status:** TEST-PINNED (14 tests in the planner suite; 12 disable runs red
against the production module). CURRENT-CONTRACT for the certification folds.

## Placement

| piece | home | why there |
|---|---|---|
| certification obligations (P7a folds, builder) | `lance_graph_contract::certification` | pure mask arithmetic over contract types (`SupportLedger`, `DatasetVersion`, `Certification3`); the contract is zero-dependency |
| `reason` / `revise` / `hydrate` | `lance_graph_planner::pearl` | needs `CausalEdge64` (causal-edge) and `counterfactual_replay`; the planner is the crate that depends on both, the same reason `dismech_counterfactual` lives there |

`revise` and `hydrate` take the caller's declared reading (`Epi5Declarations`,
class, rail, provenance) instead of the probe's fixed class; an undeclared
class is refused.

## What changed for CI

The probe's tests needed the driver's `with-planner` feature, and CI runs the
driver with default features only, so the 14 falsifiers never ran in CI. They
now run in `cargo test -p lance-graph-planner`. The examples keep their
behaviour: `examples/shared/certification_model.rs` re-exports the contract
module (P7a 13/13 and the reading-conflict probe 9/9 pass unchanged), and
`pearl_ladder_probe` prints the same script through the production modules.

## Open

- No runtime caller. Nothing in production builds a sealed population or a
  sealed chain and hands it to `reason`; the API is in place for the first
  producer. (`ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE` describes the same
  gap from the register's side.)
- `CertificationModel` is capped at 64 units (one `u64` per mask).
- The planner carries four older clippy findings in `nested_bands.rs` and
  `cache/nars_engine.rs`; CI does not gate planner clippy. `pearl.rs` is clean.
