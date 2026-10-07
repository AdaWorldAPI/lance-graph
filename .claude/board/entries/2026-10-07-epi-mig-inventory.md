# 2026-10-07 — D-EPI-MIG-INVENTORY: every reader and writer of CE64 bits 59..63

Operator decision (2026-10-07): bits 59..63 are ONE declared field,
`EpistemicState5` — a dense 5-bit codebook over valid epistemic
conjunctions. `contract::band_reading`'s split reading (59..60 truth/topology
lens, 61..63 band) and the P7a certification band survive only as
compatibility projections / translations. This entry is the inventory taken
on `main` @ `1c91fa4` (after #1380) BEFORE any semantic change.

Method: `grep` located candidates for `truth`/`truth_raw`/`with_truth`/
`set_truth`/`topology`/`with_topology`/`reasoning_band`/`with_reasoning_band`/
`spare`/`with_spare`/`set_spare`/`with_routing`/`temporal()`/`TRUTH_*`/
`SPARE_*`/`>> 59`/`raw5`/`EPI_LAW`/`EpistemicState5`/`band_reading`/
`ReasoningBand`/`CausalTopology`/`TrustTexture` across `crates/`; every hit
was then opened and read. False positives (same method name, other type)
are listed so they are not re-checked.

Actions: MIGRATE / PROJECT / WRAP / REFUSE / REMOVE / PROBE-ONLY.

## Physical layer (`causal-edge`)

| path | r/w | kind | interpretation | joint? | action |
|---|---|---|---|---|---|
| `edge.rs` `truth()/truth_raw()/topology()` | R | prod API | 2-bit lens (Trust / Topology) | half | PROJECT — legacy lens, raw half kept as physical accessor |
| `edge.rs` `spare()/reasoning_band()` | R | prod API | 3-bit lens (spare / historical band) | half | PROJECT — legacy lens |
| `edge.rs` `with_truth/set_truth/with_topology/with_routing` | W | prod API | writes 59..60 only | half | WRAP — `#[deprecated]` → joint writer |
| `edge.rs` `with_spare/set_spare/with_reasoning_band` | W | prod API | writes 61..63 only | half | WRAP — `#[deprecated]` → joint writer |
| `edge.rs` (none) | — | — | no joint accessor exists | — | MIGRATE — add `epistemic_raw5` / `with_epistemic_raw5` |
| `edge.rs` `pack()` v2 / `pack_v2()` | W | prod | zeroes 59..63 | joint | PROJECT — code 0 = canonical `Open`, no change |
| `edge.rs` `compose()` → `pack()` | W | prod | result 59..63 = 0 | joint | PROJECT — composition certifies nothing: `Open` |
| `edge.rs` `temporal()` / `network.rs` `evidence_trail` sort | R | prod (deprecated) | bits 52..63 composite incl. 59..63 | joint-as-noise | REFUSE semantically — already deprecated; ordering is documented meaningless; listed as remaining |
| `edge.rs` `set_temporal()` v1 | W | v1 compat | v1 temporal | — | v2 no-op, unchanged |
| `edge_v3.rs` `from_v1` p[8]/p[9] | R | prod lift | raw copy of both halves | joint (two copies) | MIGRATE — copy `epistemic_raw5` jointly; provenance rule unchanged |
| `edge_v3.rs` `to_ce64` `set_truth`+`set_spare` | W | prod lift | raw copy back | joint (two writes) | MIGRATE — `with_epistemic_raw5` |
| `edge_v3.rs` tests | R/W | test | raw round trip | — | keep (allow deprecated) |
| `v2_layout_tests.rs` | R/W | test | field isolation of the halves | half | keep (allow deprecated) — physical layout tests |

## Contract (`lance-graph-contract`)

| path | r/w | kind | interpretation | class/rail-aware | provenance-gated | action |
|---|---|---|---|---|---|---|
| `band_reading.rs` | R (declares) | prod contract | split: `TruthLens` 59..60 + `BandPresence` 61..63 | yes | yes | WRAP — regraded legacy/transitional; no longer owns 59..63; `EdgeProvenance` reused unchanged |
| `class_view.rs` `band_reading()` | R | prod contract | default `ZERO_FALLBACK` | yes | — | WRAP — doc points at the canonical reading |
| (absent) | — | — | no joint 5-bit reading | — | — | MIGRATE — new `epistemic_state5` module |

## Production consumers

| path | r/w | interpretation | gated? | can map to State5? | action |
|---|---|---|---|---|---|
| `lance-graph-planner/src/dismech_counterfactual.rs:251` `CounterfactualRole{topology, band}` | R | legacy lens pair of the cut edge, no provenance | no | the edge's raw5 yes; meaning needs a declaration the call site does not have | MIGRATE — carry `epistemic_raw5`; `topology`/`band` kept as legacy projections (doc); test fixture to joint writer |
| `lance-graph-arm-discovery/src/translator.rs` doc | — | prose: "61-63 have been ReasoningBand" | — | — | docs: corrected |
| `lance-graph-planner/src/cache/nars_engine.rs` `set_truth` | — | **false positive** (`SpoHead`) | — | — | none |
| `lance-graph-cognitive/src/search/cognitive.rs` `with_truth` | — | **false positive** (`CognitiveAtom`) | — | — | none |
| `deepnsm/examples/causal_edge_v3_facet.rs` `set_truth(f,c)` | — | **false positive** (V3 facet truth pair, not CE64 59..63) | — | — | none |
| `lance-graph-contract/src/mul.rs`, supervisor `probe_ignition` `TrustTexture` | — | **false positive** (`contract::mul::TrustTexture`) | — | — | none |
| Mailbox / shader (`cognitive-shader-driver/src`, supervisor src, `lance-graph/src`) | — | no reader or writer of 59..63 found | — | — | none (streamdto probe pins "the shader writes no epistemic state") |

## Probes (`cognitive-shader-driver/examples`, `lance-graph-planner/examples`)

| path | r/w | interpretation | joint? | action |
|---|---|---|---|---|
| `shared/affordance_law.rs` `EPI_LAW`, `raw5`, `measure` | R | joint raw5, classless, own codebook | joint | MIGRATE — codebook = contract `CODEBOOK_V1`; `measure` projects through a declaration (class, rail, generation, provenance) |
| `affordance_measurement_probe.rs` `stamp` (`with_spare`+`with_truth`) | W | joint via two halves | joint | MIGRATE — joint writer |
| `ce64_cycle_survival_probe.rs` `stamp` | W | same | joint | MIGRATE |
| `ce64_nextstate_probe.rs` `stamp` + raw mask write | W | same; raw mask is the measured instruction shape | joint | MIGRATE `stamp`; raw mask pinned equal to the canonical writer (PROBE-ONLY) |
| `revision_epistemic_writer_probe.rs` (#1379) `stamp` | W | same | joint | MIGRATE |
| `streamdto_circuit_probe.rs` | R + raw W | raw5 + mask poke for a fixture | joint | MIGRATE reads via shared law (no stamp change needed) |
| `shared/certification_reading.rs` (P7a) `stamp` / `read` | W/R | **61..63 only**, P7a contract via `band_reading::project_band` | half | MIGRATE — stamp = legacy translation (grounding × contract) → canonical code → joint write; read = `project_state5` → certification projection |
| `relational_certification_probe.rs` (P7a) | W/R | via shared; raw `with_reasoning_band(6,7,Meta)` fixtures | half | MIGRATE — fixtures declare `Direct` grounding; Contributes/CausalCandidate under Direct have no canonical code → REFUSE (listed open) |
| `epistemic_reading_conflict_probe.rs` (#1378) | R | both readings | — | MIGRATE → conformance probe |
| `reasoning_band_probe.rs` (D-GSO-7, #1360) | W/R | historical `ReasoningBand` ladder on 61..63 | half | PROBE-ONLY — historical reading; canonical projection REFUSES it (no declared translation); listed remaining |
| `probe_revision_kanban_hinge.rs` | W/R | historical topology + `ReasoningBand::Causal` fixture | half | PROBE-ONLY — listed remaining |
| `entropy_topology_probe.rs` | R | topology lens via `band_reading::project_truth` | half | PROBE-ONLY — listed remaining |
| `lance-graph-planner/examples/probe_four_plane_causal_medium.rs` | W/R | historical topology + band | half | PROBE-ONLY — listed remaining |
| `witness_angle_probe.rs`, `spog_witness_probe.rs` | — | `EdgeProvenance` only | — | none |

## Why historical readings are REFUSED, not translated

The historical `ReasoningBand` (`Surface, Association, Relation, Causal,
Counterfactual, Perspective, Meta, Transcendent`) names reasoning LEVELS, and
P7a (#1369) already showed they are not the certification contract (old
`Meta`/`Transcendent` sit numerically above `Causes`). The `TrustTexture`
lens on 59..60 is an epistemic texture, not grounding. Neither producer
declared conjunction semantics, so no translation table is written for them:
canonical projection of their bits refuses (`UndeclaredCode` or, for a class
declared only under the historical reading, `UndeclaredClass`).

The only legacy translation declared is P7a certification × legacy
`CausalTopology` grounding, whose obligations were measured in #1369.
