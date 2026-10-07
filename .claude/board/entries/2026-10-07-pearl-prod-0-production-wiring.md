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
- ~~`CertificationModel` is capped at 64 units (one `u64` per mask).~~ Lifted in the same PR; see "Any population width" below.
- The planner carries four older clippy findings in `nested_bands.rs` and
  `cache/nars_engine.rs`; CI does not gate planner clippy. `pearl.rs` is clean.

## Rename (same PR)

`dismech_replay` and `dismech_counterfactual` never read a DisMech predicate:
the ordinal is an opaque `u8` witness. They are now `chain_replay` and
`chain_counterfactual`. The one DisMech-specific piece — `chain_step_predicate`,
`UnmintedOrdinal`, `validate_chain` — moved to `dismech_admission`. The old
module paths stay as `#[deprecated]` re-export aliases, pinned by
`the_old_module_paths_still_name_the_same_items`. No behaviour change: planner
lib 450 passed / 3 ignored, `house_differential` 6/6, `reasoning_band_probe` 7/7.

## Any population width (same PR)

The certification folds use only intersection, difference and a population
count, so `CertificationModel` / `ModelBuilder` / `compare` / `count` are now
generic over `PopulationMask` — a sub-trait of the existing
`revision::EvidenceMask` (the mask `dismech_candidates` already uses) that adds
`count`, `full` and `unit`. `u64` is the default type parameter, so every
existing caller is unchanged; `[u64; N]` holds `64 * N` units, and
`[u64; 1024]` matches the 64k-row cycle `persist_sink` seals. `pearl::Evidence`
and `pearl::reason` take the same parameter. `ModelBuilder::new()` stays on
`u64` so it infers without annotation; `default()` builds any width.

Falsifiers (contract `certification::tests`, planner `pearl::tests`):
- `every_carrier_width_certifies_a_fixture_identically`: four fixtures
  (CausalCandidate, Open, Related, Causes) give the same six folds in `u64`,
  `[u64; 2]` and `[u64; 1024]`.
- `units_past_the_first_word_are_counted` and
  `a_population_wider_than_64_units_is_decided` (with an equal-rates silence
  twin). Disable run: counting only word 0 of `[u64; N]` turns both red.
- `units_stop_at_capacity`, `the_builder_refuses_past_capacity`.
- `the_operators_read_a_population_wider_than_64_units`: SO earns
  CausalCandidate, PO earns Causes and writes it back, and SP reports no
  confounding, over 300 + 200 units past the first word. Null arms earn nothing.

## What else was checked for reuse, and why it is not wired

- **No producer of sealed evidence exists yet.** The shader driver emits edges
  whose Pearl projection comes from resonance predicates, with no population
  behind them; `persist_sink` is storage-only; `AuditedRelation` has no
  callers. The natural first producers are `lance-graph-arm-discovery`
  (`RowMasks` are row bitsets, so a mined rule A→B is an SO question over
  exposed = rows with A, outcome = rows with B, capped observationally) and a
  sealed 64k cycle. Neither is wired: arm-discovery has one source per
  dataset, and `MIN_SOURCES = 2`, so a rule would be `TooFewSources` until it
  is decided what counts as a distinct source (basins, environments, datasets).
- **`cache::nars_engine` "Pearl rung 2/3"** (`Inference::Intervention` /
  `Counterfactual`) is truth arithmetic over two heads (abduction ×0.85,
  deduction ×0.70), not executed arms or a replay. It writes truth and the
  mantissa, never bits 59..63, so it does not bypass the certification rule;
  it is not routed into `pearl` because it measures nothing `pearl` could
  certify from.
- **`AuditedRelation::is_intervention_established`** treats one
  `InterventionBacked` receipt plus a causal classification as established,
  without arms and below `MIN_SOURCES`. That is a second, weaker definition of
  the same question. It has no callers; left as is, recorded here.
