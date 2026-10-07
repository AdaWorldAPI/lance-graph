# 2026-10-07 — D-EPI-POP-0 / D-EPI-LEGACY-DEPROJECT-0: populations, and what the old 59..63 readers were asking

Builds on #1381 (head `47bd36b`, unmerged): `EpistemicState5 = Topology2 ×
Certification3`, `raw5 = topology | certification << 2`, 24 meaningful / 8
reserved. That layout is given here, not revisited.

The rule of this wave: the historical enums are lenses, not dimensions. For
every reader and writer the question is what the caller was asking, and the
answer goes to the subsystem that owns it. No ordinal translation
(`ReasoningBand → Certification3`, CE64 `TrustTexture → Topology2`) is
used anywhere.

## 1. D-EPI-POP-0 — the population operator

`lance_graph_contract::epistemic_state5`:

- `Population = u32`: bit `r` is the state whose raw5 is `r`.
- `const fn facts_population(required: Facts) -> Population`: bit `r` is set
  iff code `r` is meaningful and its facts include every required fact.
  Derived from `COMPILED_FACTS_V1` (which is derived from the two factor
  tables), so there is no second semantic table.
- `EpistemicState5::bit()`: one state's bit.

One edge carries a coordinate (`u5`); one semantic condition is a population
over all coordinates (`u32`, transient, never stored on an edge). Many edges:
population filter, then fold.

Pinned (exact sets, not only sizes): `CAUSES` = {20,21,22,23};
`RELATED` = 8..23 (16); `IND_UNKNOWN | RELATED` = {10,14,18,22};
`IND_KNOWN | SUPPORTS` = {13,17,21}; `TOPOLOGY_UNKNOWN | CAUSES` = {23};
`DIRECT | ASSOCIATED` = {4,8,12,16,20}; `required = 0` = the 24 meaningful
states; `DIRECT | INDIRECT` = ∅. Defining falsifier, exhaustive: for every
requirement over the declared facts and every `raw in 0..32`,
`bit(raw) ∈ population ⇔ facts_v1(raw)` asserts the requirement (reserved ⇒
false). Conjunction is intersection. A topology factor update holding the
certification moves a state only between topology-conditioned populations,
and vice versa. Disable runs: any-of semantics, reserved read as fact-less —
both red.

## 2. Occurrence census (repository-wide, `47bd36b`)

Finders: `ReasoningBand`, `reasoning_band`, `with_reasoning_band`,
`TrustTexture`, `truth`/`truth_raw`/`with_truth`/`set_truth`/`with_routing`,
`topology`/`with_topology`, `spare`/`with_spare`/`set_spare`,
`TRUTH_*`/`SPARE_*`/`EPISTEMIC_*`, `epistemic_raw5`/`with_epistemic_raw5`,
`state[`, `COMPILED_FACTS_V1[`, `facts_v1(`, `decode`, `asserts(`, and the
#1370 codes 3/7/17/25/30 in epistemic contexts.

### 2a. Readers and writers of bits 59..63 through a legacy lens

| path | symbol | R/W | kind | type | semantic question | class |
|---|---|---|---|---|---|---|
| `lance-graph-planner/src/dismech_counterfactual.rs` | `EdgeRole::band` | R | production | `ReasoningBand` | "does the load-bearing edge EXPLAIN the chain or merely RELATE to it" (module doc) — a certification question, answerable only under the cut edge's class declaration | EXACT-COORDINATE via `epistemic_raw5` (already present); `band` REMOVED |
| same, tests | `reasoning_band()` | R | test | `ReasoningBand` | that `EdgeRole` reports the cut edge's own bits | EXACT-COORDINATE: pin raw5 + decoded factors |
| `cognitive-shader-driver/examples/probe_revision_kanban_hinge.rs` | `with_reasoning_band(Causal)`, F8 | W+R | probe | `ReasoningBand` | non-interference: does a nontrivial value in 59..63 survive the control loop untouched? The value's meaning is never used | EXACT-COORDINATE: a meaningful nontrivial state, pinned by raw5 + factors |
| `lance-graph-planner/examples/probe_four_plane_causal_medium.rs` | `with_reasoning_band(Causal)`, FP3, band_before | W+R | probe | `ReasoningBand` | its thesis assigns 61..63 a fourth plane, "THROUGH WHAT lens?" — a reasoning lens; the checks are non-interference of that plane | EXECUTION-CONTEXT (the lens belongs with `RungLevel` / thinking style, not CE64) → COMPAT: migrating rewrites the probe's thesis |
| `cognitive-shader-driver/examples/reasoning_band_probe.rs` | D-GSO-7 ladder | W+R | probe | `ReasoningBand` + `band_reading` projector | earned rungs: `Association` ← corroborated observation, `Causal` ← one passed intervention, `Counterfactual` ← removal attack | superseded by P7a (whose `Causes` needs ≥ 2 intervention-backed sources, and which states removal is not the causal path); `Counterfactual` rung = COUNTERFACTUAL (`dismech_counterfactual`); probe kept as shipped → COMPAT (obsolete) |
| `cognitive-shader-driver/examples/relational_certification_probe.rs` | `Meta`/`Transcendent` `to_bits_3() << 2` | W | probe | `ReasoningBand` | do the historical ordinals land on reserved certifications and refuse? | REFUSE witness (keep) |
| `cognitive-shader-driver/examples/epistemic_reading_conflict_probe.rs` | `with_reasoning_band(Causal)` | W | probe | `ReasoningBand` | a class declared under the historical reading is not consumed as canonical | REFUSE witness (keep) |
| same | `split()` (`topology()` + `spare()`) | R | probe | `CausalTopology`, untyped spare | old lenses vs canonical projection (the 24/0/8 conformance) | COMPAT comparison (keep) |
| `cognitive-shader-driver/examples/revision_epistemic_writer_probe.rs` | `spare()`, `truth_raw()` | R | probe | untyped | the joint writer equals the two halves | COMPAT comparison (keep) |
| `cognitive-shader-driver/examples/ce64_nextstate_probe.rs` | `via_two_writers` (`with_spare`, `with_truth`) | W | probe | CE64 `TrustTexture` | the joint writer equals the deprecated two-writer path | COMPAT comparison (keep) |
| `causal-edge/src/edge_v3.rs` | V3 lift `truth_raw()` / `spare()` | R | production | untyped | copy bits 59..63 into V3 bytes 8/9 | EXACT-COORDINATE (physical): read through `epistemic_raw5()`, bit-identical; V3 byte layout unchanged |
| `causal-edge/src/edge.rs` | `with_routing` doc | — | doc | CE64 `TrustTexture` | claims `MailboxSoA::dispatch_cycle()` stamps routing — no such function exists | doc corrected |
| `causal-edge` `layout.rs` / `edge.rs` / `lib.rs` / `v2_layout_tests.rs` / `edge_v3.rs` tests | definitions, re-exports, own tests | — | compatibility | both | the lens surface itself | COMPAT |
| `lance-graph-contract/src/band_reading.rs`, `class_view.rs::band_reading` | `BandDeclarations::project_band` | R | compatibility | `ReasoningBand` raw | the historical declared projector; only consumer is `reasoning_band_probe`; no `ClassView` overrides `band_reading` | COMPAT |
| `lance-graph-ogar/src/recipe_vocab.rs`, `probe_stamp_morton_cascade.rs`, `arm-discovery/src/translator.rs`, `epistemic_state5.rs` docs | prose | — | docs | — | — | FALSE-POSITIVE (doc mentions) |

### 2b. Homonyms (no bits 59..63)

`Triplet::with_truth` (arigraph, osint), `nars_engine::set_truth`,
`CognitiveAtom::with_truth`, the deepnsm V3-facet example's `set_truth(f, c)`,
`RungLevel` (shares four `ReasoningBand` names at different ordinals),
`verb_lexicon::epistemic_reading` — all NARS truth or unrelated:
FALSE-POSITIVE.

## 3. `TrustTexture` collision map

| type | variants | repr / storage | writers / readers | purpose |
|---|---|---|---|---|
| `causal_edge::layout::TrustTexture` | Crystalline, Solid, Fuzzy, Murky | `repr(u8)`, the historical lens over bits 59..60 | no production writer or reader; `causal-edge` tests, `ce64_nextstate_probe`'s comparison writer | historical gate texture (its doc: Proceed / Proceed / Sandbox / Compass) |
| `lance_graph_contract::mul::TrustTexture` | Calibrated, Overconfident, Uncertain, Underconfident | `repr(u8)`, MUL outputs and i4 batch slices | `mul`, `collapse_gate`, `kanban`, `action`, `sensorium`, `exploration`, `deepnsm-v2`, supervisor cycle driver | felt vs demonstrated competence |
| `lance_graph_planner::mul::trust::TrustTexture` | Crystalline, Solid, Fuzzy, Murky, Dissonant | 5 buckets over an f32 trust score | planner `mul::gate`, `thinking::style` | trust-score quantizer |
| `lance_graph::graph::arigraph::orchestrator::TrustTexture` | Crystalline, Fibrous, Fuzzy | serde enum | orchestrator | source / environment reliability factor |

Bridge search (`as u8`, `transmute`, `From`/`TryFrom`/`Into`, `from_bits_*` /
`to_bits_*` on a texture, match-and-repack, `with_truth`, `truth_raw`): none
between any two of the four, and none from any of them onto `Topology2`. The
only ordinal casts are `causal-edge`'s tests of its own discriminants. The
CE64 lens shares four names with the planner type, which has five buckets: an
ordinal cast would not even be total. OQ-MCAL-1 (which MUL vocabulary is
canonical) is unresolved. No bridge exists, and none was added.

## 4. ReasoningBand callers — semantic intent

| historical use | intent found | home |
|---|---|---|
| dismech `EdgeRole::band` | "explains vs relates" | certification coordinate of `epistemic_raw5`, under the cut edge's declaration |
| hinge `Causal` | a nontrivial witness value | any meaningful state; the coordinate is pinned, its meaning unused |
| four-plane `Causal` lens | the reasoning lens a hypothesis is read at | execution context (`RungLevel` / style), not CE64 — COMPAT |
| D-GSO-7 `Association` / `Causal` | earned by observation / one intervention | P7a obligations (different policy: P7a `Causes` ≥ 2 sources) — no name mapping |
| D-GSO-7 `Counterfactual` | survived a removal attack | counterfactual machinery (`dismech_counterfactual`) |
| `Meta` / `Transcendent` (P7a) | none — refusal witnesses | reserved certifications refuse |
| `Relation`, `Perspective` | no caller | — |

## 5. CE64 TrustTexture callers — semantic intent

No production reader or writer. The V3 lift copies raw bits (physical, no
lens) and now reads them jointly. The nextstate comparison writer is kept.
`band_reading`'s `Trust` lens refuses under the canonical reading. Topology
queries already read `CausalTopology` / `Topology2`; no `TrustTexture` caller
meant topology, MUL or conflict.

## 6. Stale raw5 / code assertions

The model failure (writer migrated 7 → 4, expectation still `state[7]`) was
fixed in `350e364`. This census re-checked every `state[N]`, `EPI_LAW[N]`,
`with_epistemic_raw5(N)`, `write_code`, `CODE_*` and `decode` literal
against `topology | certification << 2` and its comment: all agree. The #1370
codes surviving in comments are provenance notes ("was code 7"). No new stale
assertion.

## 7. Code-oriented checks turned into population queries

- `dismech_counterfactual` test: "explains vs relates" is membership of the
  cut edge's code in `facts_population(fact::CAUSES)`; fixtures
  `IndirectKnown × Causes` (21) and `IndirectKnown × Related` (9), each pinned
  by raw5, decoded factors and the role's topology.
- `epistemic_reading_conflict_probe::a_certified_causes_reads_as_causes_under_every_topology`:
  `20 + t.ordinal()` → each stamp pinned to its coordinate, and the four
  stamped codes together equal `facts_population(CAUSES)`.
- `revision_epistemic_writer_probe` (`for start in (0..32).filter(EPI_LAW is_some)`):
  → the members of `facts_population(0)`.
- `affordance_measurement_probe::reserved_codes_refuse_…`
  (`filter(EPI_LAW is_none)`): → the complement `!facts_population(0)`.

The remaining raw literals are coordinates by intent (a fixture's concrete
state, or a test about code arithmetic itself), each pinned with its
decoded factors where it drives behaviour.

## 8. What remains, and what cannot move

- Canonical: `EpistemicState5`, `Topology2`, `Certification3`, `Facts`,
  `facts_population`, the affordance law.
- Independent: MUL (both vocabularies), witness / provenance, revision /
  contradiction, `dismech_counterfactual`, Pearl `CausalMask`, NARS F/C,
  inference mantissa, `RungLevel`.
- Compatibility only: `causal_edge::layout::{TrustTexture, ReasoningBand}`,
  their deprecated writers and plain readers, `band_reading`'s projector.
  Production consumers after this wave: **none**. Load-bearing only as (a)
  the V1-feature stubs, (b) the old-vs-new comparison and refusal witnesses,
  (c) the two historical probes. They could be deprecated harder (readers
  too) or moved behind a `legacy-59-63` feature once those probes are
  retired; deletion is not proposed — `band_reading` is the declared
  projector for data stamped under the historical reading, and no replay
  inventory of such data exists yet.
- Cannot migrate without invention: `probe_four_plane_causal_medium` (its
  thesis puts a reasoning lens on 61..63) and `reasoning_band_probe` (its
  rungs are a different policy from P7a's). Both stay as records.
- No population helper beyond `facts_population` / `bit`: no query repeats
  three times.

## 9. Counts

ReasoningBand caller sites (bits 59..63, outside `causal-edge`'s own
definitions and tests): 6. Each has exactly one primary class:

| site | was really | primary class |
|---|---|---|
| `dismech EdgeRole::band` | a certification question (explains vs relates) | EXACT-COORDINATE (field removed; test asks `facts_population(CAUSES)`) |
| kanban hinge witness | a non-interference witness | EXACT-COORDINATE (`IndirectUnknown × Causes` = 22) |
| four-plane probe | a reasoning lens | EXECUTION-CONTEXT → COMPAT |
| `reasoning_band_probe` | D-GSO-7 rungs (superseded by P7a; its `Counterfactual` rung is counterfactual evidence) | obsolete → COMPAT |
| P7a `Meta`/`Transcendent` | refusal witness | REFUSE |
| conflict probe legacy class | refusal witness | REFUSE |

So of the six, 1 was a certification query, 1 was a value whose meaning was
never read, 2 were context or obsolete, and 2 are refusal witnesses.
Removed: 1 field (`EdgeRole::band`).

CE64 `TrustTexture` callers: 0 production readers or writers. Topology
queries: 0 (they already used `CausalTopology` / `Topology2`). MUL /
calibration: 0. Other subsystems: 0. Physical copy now joint: 1 (the V3
lift). Compatibility only: the nextstate two-writer comparison, the
`band_reading` `Trust` lens, `causal-edge`'s own tests.

Raw-code checks that became population queries: 4 call sites (`dismech`
test, conflict F3, revision `every_write_is_a_declared_joint_code`,
affordance reserved codes). Production callers of `facts_population`: 0.
It is a contract operator awaiting its first runtime fold.

Raw5 literals remaining: coordinates by intent only (fixtures and
code-arithmetic tests), each pinned to its decoded factors where it drives
behaviour.

Ordinal bridges between unrelated `TrustTexture` types: none, before and
after.

Legacy APIs still load-bearing: no production path reads or writes 59..63
through a legacy lens. What remains carries the `v1`-feature stubs, the
comparison and refusal witnesses, and two historical probes.

## 10. Disable runs (deprojection)

- `EdgeRole` reports a constant code instead of the cut edge's → the
  `dismech` coordinate test fails.
- The V3 lift swaps the two halves → 3 `edge_v3` tests fail (joint-code
  preservation, full field parity, tail placement).
- The P7a stamp ignores its topology → conflict F3 fails at the coordinate
  pin.

Gates: `causal-edge` 81 (v2) / 39 (v1); `lance-graph-contract`
`epistemic_state5` 12; planner `dismech` 24 and `probe_four_plane` builds;
`cognitive-shader-driver --examples` 26 suites, 0 failures, clippy clean.
Clippy `-D warnings` still fails on lints this branch does not touch
(`causal-edge` `edge.rs` / `tables.rs` / `module_inception`, planner
`nested_bands.rs`, `nars_engine.rs`, `probe_nxg_hist_1`, all `chunks_exact`
or older).
