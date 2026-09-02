# Post-teardown buildup survey v1 — the six families as ingredients

> **Status:** SURVEY, read-only, plan-only (no code, no tenant, no ClassView, no
> opcode, no axis vocabulary). **Baseline:** lance-graph `20eaf7f` (the #1134
> merge), OGAR `954fc52` (the #298 merge), ndarray current. **Date:** 2026-09-02.
> **Predecessor:** the semantic-family recovery
> (`E-SIX-SEMANTIC-FAMILIES-MUST-NOT-IMPERSONATE-EACH-OTHER-1`). **Successor
> gate:** PROBE-POP-READOUT-1 (§6, D-POP-1) — nothing in this plan becomes a
> carrier decision until that probe has run.

## 0. Framing

After the teardown the six semantic families are ingredients, not competing
owners of one lane. The buildup question is therefore not "what replaces the
removed September register" but "what can be composed from the six families
and the shipped mathematical mechanisms before anything new is invented".

```
families = state / evidence carriers
atoms    = cheap transformations / readouts over families
recipes  = compositions of atoms over families
styles   = policies selecting / admitting recipes
```

Philosophical labels (Kant, Hegel, Nietzsche, Wittgenstein) are working names
for molecules built from the ingredients — never a seventh, eighth or ninth
storage family. No opcode, field, tenant or ClassView is minted for them here.

Method: four parallel read-only traces over the merged baseline (logic /
information operators; geometry / palette primitives; temporal / causal /
revision mechanisms; qualia / recipe / style / loco machinery), each row cited
to a file. Rows the tracers could not confirm in source are marked
UNVERIFIED. Nothing here was measured by running code.

**Headline.** Family 3 (population geometry) can be tried as a MOLECULE
first: every ingredient for a "where am I relative to the population" readout
exists and is tested, none of it is wired into any selection path, and the only
shipped selector reads four scalars with zero population or qualia input. The
blocking gaps are a LABEL (no ground truth for a good recipe choice exists) and
a FETCH (the trained Cam96 artifacts are release-only), not a carrier.

## 1. Six-family inventory

| # | Family | Shipped carrier | Persisted | Production readers | Honest state |
|---|---|---|---|---|---|
| 1 | Episodic / Markov loci | `CausalWitnessFacet` tenant 14 (row bytes `[204,220)`), `WitnessLens` (borrow, non-Copy), `WitnessStream` | yes, 12 B | `recipe_loci`, `dispatch_guard`, `meta_basin`, `style_strategy::reliability_for` | live; `WitnessStream` is test-only by its own doc; `ChainResolution::out_of_horizon` is emitted by five functions and read by none |
| 2 | Epistemic qualia | `QualiaI4_16D` tenant 1 (`QualiaI4Column`, 8 B/row); producer `Qualia17D::from_convergence` behind `with-engine` | yes | `mul::i4_eval::gate_decision_i4` → `SigmaTierRouter`; `affective_temperature` → `ThoughtCtx.temperature` | live but narrow: one routing decision (gate Block ⇒ Rest) and four of 34 kernels; the 11-D proprioception surface (`qualia_to_state`, `StateClassifier`, `hydrate`) has no production consumer |
| 3 | Population geometry | none | no | none | ratified vacancy; nearby primitives in §2 rows G1–G9 |
| 4 | Causality trajectory | `TrajectorySignature`, `RevisionTrajectory`, `belief_runs` / `suggest_reopening` / `foresight_*` (`witness_fabric.rs`); `meta_basin::grade_rows` | derived | `meta_basin` only; the version-axis family has zero in-repo callers | shipped, tested, unwired |
| 5 | Causal graph | `CausalEdge64` v2 (`CausalTopology` 59–60, `ReasoningBand` 61–63, W-slot 53–59), `band_reading` `(classid, rail)`, `dismech_replay` / `dismech_counterfactual` | yes (edge) | replay, `EdgeRole` | live; the provenance chain W-slot → `WitnessTable` → `mailbox_ref` → `spo_fact_ref` has no producer for its middle link (`witness_table.rs:38-41`) |
| 6 | Knowledge graph | AriGraph `TripletGraph` (String-keyed, in-memory), `SpoFacet` (A1 reading of the same 12 B), `BasinRow` tenant 15, three `TruthValue` types | partly | `BeliefArena` (in-memory), `GraphSensorium` | live; `BasinRow.self_code` is written as zeros by `rail_rows` |

Duplications to carry into any design: three `TruthValue` types; two
`InferenceType` enums (the contract's `from_mantissa` collapses ±6/±7 to
Synthesis, so a Counterfactual edge loses identity when round-tripped).

## 2. Atom / operator inventory

"F" = families consumed. "Wired" = has a production caller.

| id | Operator | F | Output | Wired | Evidence |
|---|---|---|---|---|---|
| S1 | `normalized_entropy`, `expectation`, `mean_confidence` (`thought_atoms.rs`) | none | f32 | no (0 callers; `expectation` re-implemented at `nars/truth.rs:35`, `exploration.rs:123`) | tests |
| S2 | `GraphSensorium::compute` → `truth_entropy`, `deduction_yield` → `suggested_bias` → `TacticChoice` | 6 | `GraphBias` → one of 5 recipe ids | yes (dialectic loop) | tests + `tactic_select_confusion` probe |
| S3 | `FrontierEdge::curiosity` = novelty × uncertainty; `curiosity_gestalt` (MUL + 17-D texture) | 6 (+2-shaped) | f32 / `TextureGestalt` | `MassExplorer::next_frontier_edge` (examples; `from_graph` leaves the frontier empty, TD-EXPLORATION-1) | g_cm_1..5 |
| S4 | `FreeEnergy::compose`, `Resolution::from_ranked` | none | struct | referenced by kernels; no live producer | tests |
| S5 | `SubstrateView::{logical_confidence, logical_surprise, logical_dissonance, logical_rung, logical_candidates}` | 1, 6 (qualia excluded by lock) | `ThoughtCtx` | examples only | `E-RECIPE-SUBSTRATE-WIRING-1`: 34/34 kernels read only the scalar proxy |
| S6 | `materialize::select_tactic` (free_energy, dissonance, sd, rung → argmax over 34) | none directly | u8 | no | `E-RECIPE-SELECTOR-REACHABILITY-1`: 8/34 reachable |
| S7 | `quorum_mantissa`, `elect_peers`, `opinion_strength` | 1 | u8 / `PeerElection` | `meta_basin` (mantissa only) | outcome-blindness pinned |
| T1 | `BeliefArena::{revise_at, admit_derived, close_transitive}`, `Copula::transits` | 6 | arena mutation; `contradiction` = max\|f₁−f₂\| kept | yes (5 tactics) | four two-sided tests; in-memory only |
| T2 | `derive_depth_from_support`, `SignedTarskiWitnessView` | 6 / geometry of 1 | depth vector | probe only | withdrawn circular claim on record |
| T3 | `RungLevel`, `Recipe::min_rung` / `admissible_at`, `EpistemicMode::for_rung`, `rung_horizon::claim_admitted` | none (ordinal) | admission | `style_strategy`, alpha tunnel | tests |
| B1 | ndarray `U64x8::ternlog::<IMM>` (+ `U32x16`), named tables `AND3`/`MAJ3`/… | none | mask | no caller in lance-graph or OGAR | all-256-IMM parity test |
| B2 | OGAR `FnIndex::TERNLOG` 0x86 | none | mask | ABI slot, no in-tree body | census 96 |
| B3 | `BandReading::project_truth` / `project_band` (provenance before lens, refuses `Unknown`) | 5 | `Result<u8>` | `ClassView::band_reading` | G3′/G4′/G5b |
| B4 | shipped four-valued things: `CausalTopology` (2 b), `GestaltState{Crystallizing, Contested, Dissolving, Stable}` | 5 / 6 | enum | yes / UNVERIFIED tests | — |
| M1 | `FieldMask`, `FieldMask::inherit`, `WideFieldMask`, `CausalWitnessFacet::{project, elected}` (fail-closed EMPTY) | 1 | mask / `Option<i8>` | yes | `elected_but_unbound_locus_reads_none` |
| M2 | `StepMask` (selection, never control flow) | none | u64 | no wiring to `ElixirTemplate` found | tests |
| M3 | `ThoughtMask::covered_by`, `rung_schedule::schedule` (≤9 waves, unreachable named) | none | `Schedule` | `wave_dispatch`, `alpha_tunnel`, `rung_horizon` | `rung_waves` |
| M4 | `recipe_loci::{required_loci, loci_disqualifier, reachable, carried_awareness, active_after_prune}`, `dispatch_guard::guard` | 1 | admission | guard: probe only | four two-sided tests each |
| M5 | `standing_mask::fires`, `selection` rail walk, `wave_dispatch::touched_indices` | none | bool / view | subscription path | tests |
| R1 | `RevisionTrajectory`, `belief_runs`, `superseded_runs`, `suggest_reopening`, `foresight_calibration` | 1 → 4 | structs | none | 8 tests incl. can-fire / can-stay-silent pair |
| R2 | `TrajectorySignature`, `meta_basin::{grade_rows, meta_cluster, outlier_suggestions}` | 1 → 4 | clusters, suggestions | `meta_basin` | hand-tuned weights declared |
| C1 | `dismech_replay::replay_step` / `replay_chain` (34.7 ns), `counterfactual_replay` (cut arm tagged −6) | 5, 6 | trace rows | D-DCR-1 merged; D-DCR-2/3 in PR | disable-verified gates |
| C2 | `deposit_counterfactual` (4-bit), `CounterfactualMailbox` (`todo!()`), `ScenarioWorld` (trait only); `World::fork` does not exist | 5 | — | probe / none | documented-only |
| C3 | `local_trajectory_of` (owner, cast_seq) | 5, 6 | per-owner chain | replay tests | precondition, not a check |
| G1 | `Cam96Space::{encode, reconstruct, distance, rails}`, `PairPalette::{distance, similarity}`, `ScalarAdc` | none (content-blind) | f32 | `SemanticSpace` only; `SubstrateView` uses the L1-grid surrogate `pair_similarity`, not the trained table | `E-CAM96-REVIEW-CORRECTIONS-1` (cite 0.766 held-out, never 0.828) |
| G2 | `centroid_point`, `spread_about` (private), `basin_self_code` → `BasinCode{self_code, width, contradiction}` | episodic rail | struct | `bible_wave` | width falsified as semantic (`E-BASIN-WIDTH-IS-N-ARTIFACT-1`) |
| G3 | `spearman`, `average_ranks`, `partial_spearman`, `shuffle_beliefs_null`, `heldout_*_gate` | measurement | f32 | `evidence.rs`, `bible_wave` | the mandatory label-shuffle null |
| G4 | `helix::fisher_z::{fisher_z, hyperbolic_depth}`; `bgz_tensor::fisher_z::{FamilyGamma::from_cosines, FisherZTable}` | none | z / i8 table | `ResidueEncoder`; the L4 cosine canon | ρ ≥ 0.999 on 21 roles (recorded) |
| G5 | `helix::quantize::RollingFloor::{quantize, occupancy, drift_score}`, `DistanceLut` | none | bucket, `[u32;256]` | `continuous_field` | occupancy IS the empirical distribution, 1 KB |
| G6 | `helix::CurveRuler` (stride-4/17; phase generated, never stored), `HemispherePoint` | none | residues | `ResidueEncoder` | D-QUANTGATE walk |
| G7 | bgz17 `DistanceMatrix` (128 KB), `PaletteSemiring` compose table (64 KB), `Scent*` prune filters | none | u16 / u8 | palette graph | Scent is not a metric (`helix/distance.rs:7-10`) |
| G8 | `FamilyTrie::{build, is_ancestor_of, dn, hhtl_packable}` | 6 (lineage) | bool / DN | `bible_wave` | trie 74 pointers == 295-pair closure |
| G9 | `Nsm::word_similarity`, `triple_similarity`; tesseract-paperless `GraphEngine::decide_endorse` | none / 6 | f32 / endorse | the one shipped end-to-end population-relative readout (in tesseract-rs) | disable-verified |
| L1 | OGAR loco shared core (96 named), `Vocabulary` composed routing, `BasinCodebook::{intern, seal, resolve_operand}`, `ConstantPool` | none | bytecode | yes | 8 tests, two strengthened after vacuity |
| L2 | `recipe_vocab` (34 ops at 0x90..0xB1, `grounded_program`, `Refusal{unwilling, unable}`, `max_rung_admitted`) | 1, 2, 6 via `ThoughtCtx` | `Vec<FnIndex>` | probes | "first policy, not canon" |
| L3 | `ogar-r2il` 82 ops, `CallMask` masked lane projection | none | zero-copy re-read | consumers | zipper parity |

**Documented-only (no code):** `info_gain`; Belnap / FDE / K3 as a lattice;
the signed Mengenlehre tally (`thought_atoms.rs:10-13` says it is not yet
built); `quorum_project` / `resolve_contest` (`todo!()`);
`CounterfactualMailbox`; `awareness.revise` / `ParamTruths`; a needle /
sparse-salience primitive (only `Scent` and `RollingFloor::occupancy` come
close); `PackedQualia` (not found).

## 3. Family × operator legality matrix

**L** lawful today (types line up, no aliasing) · **U** unsupported (lawful in
principle; no seam or no data) · **X** meaningless or forbidden (would alias
families).

| Operator | 1 loci | 2 qualia | 3 population | 4 trajectory | 5 causal | 6 knowledge |
|---|---|---|---|---|---|---|
| Shannon entropy / info gain (S1, S2) | U | L (texture carries an entropy dim) | U (needs a distribution: `occupancy` or Cam96 codes) | U (entropy over flip series) | L (`GraphSensorium`) | L |
| Tarski closure / consequence (T1–T3) | X (pointers have no consequence relation) | X | X | U (rung ladder per revision step) | U (via arena only) | L |
| Belnap / ternlog over masks (B1–B4) | L (orientation stencils, `probe_witness_presence_2bit`) | U (no producer) | U (no basin) | U | L (`CausalTopology`, `BandReading` refusal) | L (`GestaltState`) |
| Fisher-z / Helix orientation (G4, G6) | X as value (a locus is an offset) | U (similarity-shaped dims only) | **L** (`FamilyGamma::from_cosines` is a population scale) | U | U (endpoint cosine) | U |
| palette256 distribution (G1, G2, G5) | X (tenant 14 bytes are pointers) | X | **L** (centroid + dispersion + rank exist; `self_code` is a Cam96) | X | U (L4 reading of endpoints) | L (`SpoFacet` A1 reading) |
| needle / salience (G5, G7) | X | U (top-k dims) | U (occupancy bucket; Scent as prune) | X | X | U |
| masks / frontier admission (M1–M5, S3) | L | L (`gate_decision_i4`) | U (no frontier reads population) | U (`suggest_reopening` has no caller) | L (`claim_admitted`, `band_reading`) | L (`next_frontier_edge`) |
| revision (T1, R1) | L (`revision_trajectory`) | X (stakes are never revised evidence) | X | L | L (`replay_step`) | L (`revise_at`) |
| trajectories (R1, R2) | L (source) | X | U (needs 3 first) | L | X (ruled: trajectory ≠ causal graph) | U |
| counterfactual replay (C1–C3) | X | X | X | U (factual vs cut trajectory) | L | L |
| BasinCodebook / loco / R2IL (L1–L3) | L | L via `ThoughtCtx.temperature` | U (a sealed `classid(4)+12` entry table is the natural home for a population code table; no mint) | U | L (`ogar-ro`, `ogar-dismech`) | L |

The two rows that keep the September collision from recurring: palette
distribution and Fisher-z are **X on families 1 and 2** and **L on family 3**;
masks / Belnap are **L on family 1** only as orientation stencils, never as
valence.

## 4. Candidate Recipe / Style molecules

**E** already expressible · **O** needs a new operator or readout (no new
state) · **G** requires genuinely missing information.

| Molecule (working label) | Composition over existing atoms | Class | Missing |
|---|---|---|---|
| Shannon/Tarski entropy-plateau selection | `close_transitive` passes → per-pass `normalized_entropy` over arena `expectation()` → stop when Δentropy < ε; rung from `RungLevel::for_pass` | O | ~20-line readout gluing S1 to T1; a label for the right stopping pass |
| Contradiction-driven revision | `elect_peers` → `Locus::Contradiction` + `BeliefArena::revise_at` → `RevisionTrajectory.flips` → `suggest_reopening` | O | one producer writing elected offsets back through `WitnessLens::write_register` |
| Population-distribution + needle routing | `Cam96Space::distance(member, BasinCode.self_code)` → `fisher_z` → `RollingFloor::quantize` + `occupancy` → admission mask → `ternlog` with a `required_loci` stencil | O | codebook seam into `SubstrateView`; data fetch; no needle primitive beyond occupancy / Scent |
| Causal counterfactual admission | `counterfactual_replay` → `Verdict` (frequency) → `RevisionVerdict::is_acceptable` → `KanbanColumn::advance_on_revision` | E for a claim | verdict → tactic-id attribution (`KanbanMove` carries no recipe id) |
| Provenance / genealogy traversal | `Belief.premises` + `rung` DAG; `local_trajectory_of(owner, cast_seq)`; `FamilyTrie::dn` | E in memory | the W-slot → `WitnessTable` → `spo_fact_ref` middle link |
| Kant-like admissibility | `ClassView` + `WideFieldMask::inherit` + `Recipe::admissible_at(rung)` + `rung_horizon::claim_admitted` → the reachable subspace Shannon may measure | E | nothing; it is `recipes_for_at` plus `schedule`'s `unreachable` list |
| Hegel-like contradiction / trajectory transformation | `GestaltState::Contested` or arena `contradiction` + Fisher-z orientation of the poles + `RevisionTrajectory` + closure → next-recipe bias | O | a (support, refute) join over two `WideFieldMask`s (~30 lines, no storage) |
| Nietzsche-like genealogy / distribution / salience | `Belief.premises` + `FamilyTrie::dn` + occupancy bucket of the subject in its population + `FrontierEdge::curiosity` | O / G | population code per subject (data); salience primitive |
| Wittgenstein-like contextual boundary | `WitnessStream::window_at` + `RollingFloor::occupancy` + `dispatch_mode::route` + `Refusal{unwilling, unable}` | O | a real `WitnessStream` producer |

None needs a tenant, ClassView, opcode or field. Two need a PRODUCER that
already has a consumer shape waiting (contradiction write-back; witness
stream from real rows).

## 5. Missing-information gaps

Ordered by how much each blocks §6, with the proof it cannot be cheaply derived:

1. **A label for recipe-choice quality.** `Outcome.delta_conf` is
   self-asserted; `Trace.rested` is guaranteed by the decay constants;
   `reliability_*` is documented as reliability, not validity;
   `RevisionVerdict` adjudicates a claim and is unattributable to a tactic.
   Genuinely missing; must be constructed.
2. **Population codes on disk.** `cam96_codebook.bin` / `cam96_codes.bin` /
   `bible_vocab.txt` are gitignored release assets (`v0.1.0-cam96-data`) and
   the KJV text is not committed; `academic_20k.csv` yields routing ranks
   only (`word_similarity` returns `None`). Derivable only by fetch.
3. **A real `WitnessStream` producer.** Every `WitnessStream::new()` is under
   `#[cfg(test)]`; no path from text or `NodeRow`.
4. **The contradiction write-back.** `elect_peers` computes it,
   `write_register` can store it, nothing connects them. Cheap.
5. **`SubstrateView` codebook seam.** `logical_confidence` / `logical_beliefs`
   call the L1-grid surrogate; `PairPalette` has no injection point. Cheap.
6. **Needle / salience primitive.** Nothing named; whether sparse salience
   over Cam96 rails is derivable from `FisherZTable` ranks is exactly what §6
   measures. Do not mint before that.
7. **Tactic-id attribution on Kanban moves.** A DTO change, deferred until a
   label exists to attach.

Not missing, despite appearances: a population CARRIER. §3 shows palette
distribution lawful on family 3 with `Cam96` (12 B) as the code and
`FamilyGamma` (8 B) as the population scale — both existing types.

## 6. PROBE-POP-READOUT-1 — is population position a molecule? (D-POP-1)

**Claim (pre-registered, falsifiable):** a readout composed only of shipped
atoms — `Cam96Space::distance(candidate, BasinCode.self_code)` → `fisher_z` →
`RollingFloor` bucket + `occupancy` — improves next-frontier-edge prediction
on the KJV stream over the shipped chooser (`FrontierEdge::curiosity`,
NARS-only) and over the qualia-texture chooser (`curiosity_gestalt`), by a
margin a size-preserving label-shuffle null does not reach.

**Why this target.** No production recipe chooser has labels, so "frontier
choice" is operationalised as `MassExplorer::next_frontier_edge` over a
`LiteralGraph` built from the first N versions of the KJV SPO stream; the
label is whether the picked edge is confirmed by later versions
(`process_results` confirm/deny — the engine's own shipped semantics). Real,
deterministic, and the same population deepnsm-v2 already measured.

**Arms** (same candidate frontier per step; metric precision@k of
later-confirmed edges):

| arm | ranker |
|---|---|
| A0 | `curiosity` only (shipped default) |
| A1 | `curiosity_gestalt` (qualia texture + MUL) — the "qualia / context alone" baseline |
| A2 | A0 + population readout (Fisher-z of candidate object's distance to the subject basin centroid; occupancy bucket as salience term) |
| A3 | A1 + population readout |
| N | size-preserving shuffle of the code↔word binding (`shuffle_beliefs_null` pattern), re-running A2/A3 — mandatory per `E-BASIN-WIDTH-IS-N-ARTIFACT-1` |

**PASS:** A2 − A0 and A3 − A1 both exceed the null's spread by the
pre-registered margin (Δprecision@10 ≥ 0.05 above the null's 95th
percentile). **KILL:** either delta lies inside the null band, or A3 ≤ A1 while
A2 > A0 (population helps only where qualia is absent — evidence for
projection into qualia rather than a separate readout). **Vacuity guards:**
frontier non-empty after `seed_frontier` (TD-EXPLORATION-1); the null must
actually destroy the binding (its ρ against A2 scores must drop); one
disable arm replaces Fisher-z by raw distance (if PASS survives, Fisher-z is
decoration).

**Ingredients:** `Cam96Space`, `basin_self_code`, `PaletteVocab`,
`TemporalStream` (deepnsm-v2); `fisher_z`, `RollingFloor` (helix);
`MassExplorer`, `FrontierEdge::curiosity{,_gestalt}` (contract
`exploration.rs`); `spearman`, `shuffle_beliefs_null` (deepnsm-v2
`evidence.rs`). **Data:** fetch `v0.1.0-cam96-data` and Gutenberg #10 exactly
as `bible_wave.rs` does. **Cost:** one example binary; no new types, no
tenant, no mint.

**What a PASS licenses, and only that:** treating family 3 as a readout
molecule feeding the frontier chooser, and opening the projection-into-qualia
question with data. **What it does not license:** a 12-byte carrier, a
ClassView, a fixed axis set, the EMPTY/−7..+7 nibble, or any philosophical
name as storage.

## 7. Provenance and limits

- Four parallel read-only traces (Opus), each returning file:line-cited rows;
  the orchestrating thread synthesised §§1–6. No code was run. Where a tracer
  reported a doc-asserted number (bgz17 layer ρ values, "10,000× faster") it
  is marked UNVERIFIED in §2 and not relied upon here.
- Known stale citations in the tree: `deepnsm-v2/src/lib.rs:14-15` still
  prints ρ 0.828 / 0.711 (in-sample); per `E-CAM96-REVIEW-CORRECTIONS-1` the
  held-out figures are 0.766 / 0.624.
- The tracers reported the working-tree HEADs as `f3eb2f6` (lance-graph) and
  `1eb2ddb` (OGAR); both are the trees the merges `20eaf7f` / `954fc52` carry.

---

## 6a. PROBE-POP-READOUT-1 — RESULT: **KILL** (measured 2026-09-02)

Shipped as `crates/deepnsm-v2/examples/pop_readout.rs`. Deterministic; 89 s on
the whole book. Inputs: `bible_wave --export` (70,393 triples over 31,102
verses) plus the trained `v0.1.0-cam96-data` codebook (12,543 words, 12 axes).
9 split points × 25 shuffles; 227,261 candidates pooled.

```
cargo run --manifest-path crates/deepnsm-v2/Cargo.toml --example bible_wave  -- pg10.txt --export spo.tsv
cargo run --manifest-path crates/deepnsm-v2/Cargo.toml --example pop_readout -- spo.tsv
```

| arm | mean p@10 | mean p@25 | mean p@100 | ρ vs label |
|---|---|---|---|---|
| A0 `curiosity` (shipped ranker) | 0.289 | 0.173 | 0.249 | ≈ −0.27 |
| A1 `curiosity_gestalt` (assessment A) | 0.289 | 0.173 | 0.249 | ≈ −0.27 |
| A1B `curiosity_gestalt` (assessment B) | 0.289 | 0.173 | 0.249 | ≈ −0.27 |
| **AF frequency (control)** | **0.756** | **0.751** | **0.674** | **+0.27** |
| AP population readout alone | 0.011 | 0.036 | 0.076 | +0.09 |
| A2 = A0 + AP (rank mean) | 0.011 | 0.022 | 0.069 | −0.06 |
| A3 = A1 + AP (rank mean) | 0.011 | 0.022 | 0.069 | −0.06 |

Decisive statistics:

| quantity | value |
|---|---|
| real partial ρ(AP, label \| freq) | **0.090** |
| null partial ρ — mean / 95th pct | −0.018 / 0.020 |
| mean(A2 p@10) − mean(A0 p@10) | **−0.278** |
| mean(A2 p@10) vs its null mean / 95th pct | 0.011 vs 0.031 / 0.133 |
| mean(AP p@10) vs its null mean / 95th pct | 0.011 vs 0.034 / 0.144 |

Pre-registered rule: PASS iff (a) real partial ρ > null p95 + 0.02 **AND**
(b) Δp@10 ≥ 0.05 and A2 p@10 above its null p95. **(a) passes, (b) fails
decisively → VERDICT KILL.**

### The four findings

1. **KILL on the pre-registered claim.** The population readout does not
   improve frontier ranking; it degrades it, 0.289 → 0.011 at p@10, and lands
   *below its own shuffle null* (0.011 vs a null p95 of 0.133). Combining it
   into the ranking is worse than not having it.
2. **The signal is nevertheless real and null-surviving.** Controlling for
   frequency, "typical for its subject" carries partial ρ = 0.090 against a
   null p95 of 0.020. A weak global monotone trend and a useless top-k coexist:
   the extreme of `−pop` is degenerate (objects sitting essentially *on* their
   centroid) while the overall ordering still leans the right way. Precision@k
   probes the tail; Spearman probes the trend; they disagree here, and the
   disagreement is the finding, not an error.
3. **Qualia is rank-inert at the frontier — measured, not argued.**
   `spearman(A0, A1) = 1.000000` and `spearman(A0, A1B) = 1.000000` pooled over
   227,261 candidates under two deliberately contrasting `MulAssessment`s.
   Reading `exploration.rs:180-215` says why: `magnitude = base · fw · dk ·
   flow · trust · staunen_boost · ground_gate`, and every factor except `base`
   is per-GRAPH, identical for every candidate. `curiosity_gestalt` can
   rescale a frontier; it can never reorder one. So "does population beat
   qualia/context alone" had an a-priori answer for any ranking task: qualia
   contributes exactly zero ranking information at the frontier today.
4. **Plain counting dominates every cognitive arm.** Prefix frequency reaches
   p@10 = 0.756 against the shipped ranker's 0.289 — 2.6× — and the shipped
   ranker is *anti*-correlated with recurrence (ρ ≈ −0.27). That is consistent
   with `curiosity` working as designed (it prefers the rare and unqueried, and
   rare things do not recur) rather than being broken; but it means the
   frontier ranker is not selecting for what the corpus goes on to confirm, and
   any future ranking claim must clear the frequency control first.

### Honest limits

- One corpus (KJV), one label (exact-triple recurrence), one basin definition
  (a subject's outgoing objects). This KILLs "a population readout improves
  frontier ranking on the recurrence label"; it does not show population
  geometry is useless, and recurrence is not what `curiosity` is built to
  maximise.
- The codebook is Bible-vocabulary and held-out ρ 0.766
  (`E-CAM96-REVIEW-CORRECTIONS-1`; never cite the crate doc's in-sample 0.828).
- The Fisher-z and `RollingFloor`-occupancy legs the plan named were **NOT
  RUN**, for a reason the survey missed: `helix` is not a dependency of
  `deepnsm-v2`, and adding one pulls the ndarray git fork into this crate's
  build. Under a rank-based combination Fisher-z is analytically inert anyway
  (a strictly monotone transform cannot change a rank), so the plan's
  Fisher-z disable arm is answered for a rank readout and remains open only
  for a magnitude-valued one.

### Consequence

Family 3 as a **molecule feeding frontier selection** is NOT licensed by this
measurement — and a carrier is licensed even less than before. The vacancy
stands. What the result does license is a narrower next question, stated as a
question and not a direction: the readout's honest home may be the global
trend (a basin-level prior) rather than a top-k selector, and any such probe
must carry the frequency control and the shuffle null from the start.
