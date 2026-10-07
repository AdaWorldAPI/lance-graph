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
`UnmintedOrdinal`, `validate_chain` — moved to `dismech_admission` (later in this PR: `chain_admission`, keyed by classid; see below). The old
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

## Generic, specific via classid as SPO-G (same PR)

A chain step's predicate ordinal only means something inside a vocabulary,
and the vocabulary is chosen by the chain's classid. The loco floor is
shared: `0x90` is DisMech `causes`, NARS recipe #1, and the first r2il op.
OGAR already routes this way (`VocabularyRegistry::resolve_classid`,
concept = `classid >> 16`, DisMech = `0x0333`); the admission check did not.

- `planner::chain_admission` replaces `dismech_admission`.
  `validate_chain(classid, chain)` and `chain_step_predicate(classid, step)`
  route by the concept half (G) through `PALETTES` (today one entry, DisMech)
  and refuse an unmirrored concept with `Unadmitted::UnknownPalette` instead of
  reading its bytes as DisMech's.
- `contract::dismech_evidence::DISMECH_CONCEPT_ID = 0x0333`, fused in
  `lance_graph_ogar::parity::assert_dismech_palette_parity` (passes against
  OGAR; disable run with `0x0334` fails with the concept-id message).
- The deprecated `dismech_replay` alias keeps DisMech-bound wrappers, pinned to
  answer exactly as admission under `DISMECH_CLASSID`.
- Tests: `the_same_byte_is_admitted_only_under_the_classid_whose_palette_mints_it`
  (RO `0x0306` is refused, even for an empty chain) and
  `only_the_concept_half_routes` (four app prefixes route the same; another
  concept with the same low half does not). Disable run: `palette_of` ignoring
  the classid turns both red.

## Palette × evidence: citations into a quorum (same PR)

`contract::dismech_evidence::citation_quorum` folds `(CitationKey, Supports)`
pairs for one relation into the contract's `ontology_warrant::Quorum`:

- the source is the citation identity, so a citation repeated across rows is
  one source and `PMID:1` / `ORPHA:1` are two;
- `SUPPORT` corroborates, `REFUTE` conflicts, a citation saying both counts on
  both sides and is reported in `both_ways` (kept, not removed);
- `NO_EVIDENCE` is silence (the quorum's own rule); `PARTIAL` is silence by
  policy pin, matching `dismech_candidates`, which keeps it inert.

This is the per-relation input the W2b field map needs (`+` agreement, `−`
disagreement, `0` silence) and the one P7's downgrade rule read. The field
map itself (global sweep, convergence, node-level hydrate) is not built; W2b
is still a proposal awaiting scope. Disable runs: no deduplication,
`PARTIAL` as support, and two-sided citations counted once each turn their
named tests red.

## reasoning_band_probe (#1360, P7) against the last 25 PRs

| P7 feature | where it lives now | relation |
|---|---|---|
| Pearl mask selects the test (SO/PO/SPO/SP) | `planner::pearl::Operation::of` (#1391) | duplicate |
| a pass is computed, never supplied | `pearl::Measured` private fields (#1391) | duplicate |
| intervention: one trial, treated rate higher | `CertificationModel::causes` (#1369, ≥ 2 intervention sources) | superseded, stricter |
| counterfactual: removal attack | `chain_counterfactual` (W3); P7a: removal is not the causal test; SPO earns nothing in `pearl` | superseded |
| confounding via `simpsons_paradox_risk` | `pearl` SP | same check, different effect: P7 caps the band; `pearl` never demotes an intervention-certified `Causes` (f6) |
| contradicting `Quorum` caps (minority) or drops (majority) the band | no counterpart in `pearl` (rise-only) | **open**; #1379 demotes on `ContradictionPreserved`; #1368/#1380 fold observations into a quorum; `citation_quorum` now produces the input |
| band written through `ReasoningBand` names | deprecated (D-EPI-MIG-0) | probe-only legacy |

Also related in the window: #1371 (Witness as an SPO-G sub-context under the
same `classid >> 16` rule; its open `ISS-MAILBOX-ROUTES-WITNESS-WITHOUT-GRAPH`
is the routing-ignores-G defect `chain_admission` avoids) and #1370 (recipe
eligibility read from the joint 59..63 code).

OPEN: one demotion policy. `pearl` only raises, P7 lowered on contradiction,
#1379 lowers on a suspended revision. Which (if any) contradiction lowers a
certification is undecided.

## Review fixes (codex + CodeRabbit on #1391)

- `revise(&Measured, Reading)` takes no edge: `Measured` binds the edge it
  measured, so one measurement cannot promote another edge.
- An SO measurement whose weakest rung cannot compare (`associated()` is
  `Err`) reports `Ungrounded::Model`, so missing evidence is no longer read as
  a grounded negative.
- `hydrate` promotes only when the edge is the `a → y` relation whose path it
  checked.
- `Reaction::classify` compares terminal `(frequency, confidence)` only. The
  old length comparison made `Inert` unreachable, since a cut always removes a
  step. Measured: no cut on the test chains reaches `Inert` (the quantized
  revision moves the truth even for a zero-confidence step), so the rule is
  pinned at the classifier.
- Each fix has a test, and each fails under a disable run.
