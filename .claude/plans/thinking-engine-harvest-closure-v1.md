# thinking-engine harvest & closure v1 — harvest every gem into the spine, retire the residue, close the chapter

> **Status:** PROPOSAL, plan-only (W0 = this census; no code moves in this
> PR). **Baseline:** lance-graph `94543a5`. **Date:** 2026-09-02.
> **Purpose:** the crate `crates/thinking-engine` (51 files,
> workspace-EXCLUDED, off CI, ~40 clippy lints + a stale test-compile break
> per `TD-THINKING-ENGINE-EXCLUDED-DEBT-1`) is the ladybug-rs migration's
> landing zone. Most of what was good in it has already been lifted into the
> spine; what remains is a mix of hot-path engines reached only through one
> optional feature, unwired gems with named landing zones, a calibration
> battery, and self-declared residue. This plan fixes the FATE of every file,
> lists the open plans and rows that still point at the crate, and sequences
> the closure so that nothing is lost and nothing is ported without a
> consumer and a falsifier.
> **Inputs (read, not re-derived):** `.claude/v3/MODULE-TABLE.md` ancestry
> census (51 rows, gem-status column), `.claude/v3/COMPONENT-MAP.md` §3,
> `.claude/v3/FUTURE-DESIGN.md` "the migration arc", `TECH_DEBT.md`
> (`TD-THINKING-ENGINE-EXCLUDED-DEBT-1`, `TD-GHOST-ECHO-DUP-1`,
> `TD-RESONANCEDTO-DUP-1`), STATUS_BOARD rows D-PERSONA-1..6, D-TRI-1..6,
> D-TSC-1b, D-TTV-1, D-REUNIFY-2..6, ENTROPY-MILESTONES M8, board entries
> `E-RUNG-ASCENT-WIRED-1`, `E-CE64-NAME-COLLISION-DEDUP`,
> `E-MORTON-CASCADE-V3-1` (downgraded), `E-ACK-ELIMINATED-1`,
> `E-NOBODY-WAITS-1`. **Sibling:** `house-differential-style-v1` (consumer
> of the ghost-prior harvest, D-TEH-2).

## 0. The measured live footprint

Two crates depend on `thinking-engine` (`grep thinking-engine crates/*/Cargo.toml`):

| consumer | dependency | symbols actually used |
|---|---|---|
| `cognitive-shader-driver` | OPTIONAL, feature `with-engine`, default off | `dto::BusDto`; `cognitive_stack::{SD_FLOW_THRESHOLD, …}`; lens lookups (`jina_lookup`, `bge_m3_lookup`, `reranker_lookup`) in doc/test paths |
| `lance-graph-callcenter` | required, `default-features = false` | production: `bridge_gate::{CognitiveBridgeGate, CognitiveAuthResult, CognitiveOpKind}` (`cognitive_bridge_gate.rs:45`; callcenter IMPLEMENTS the trait; direction callcenter → thinking-engine is the only allowed one). **Tests also** call `jina_lens::{jina_lookup, jina_distance}`, `bge_m3_lens::bge_m3_lookup`, `reranker_lens::reranker_lookup` (`cognitive_bridge_gate.rs:474–477`, test `pure_ops_emit_zero_audit_events`) — a Codex finding on #1137; the first draft of this row said "only `bridge_gate`", which was true of production code and false of the crate |

So the spine's hard PRODUCTION dependency on a 51-file crate is ONE trait
module (`bridge_gate.rs`), plus one unit test that touches three lens
modules; the optional dependency is the DTO ladder plus two thresholds. Everything else is internal to the crate or reached only by its
own examples. That is the finding this plan is built on: closing the chapter
costs one trait move and one DTO-home decision, not a rewrite.

## 1. Fate table — every file, one verdict

Verdict vocabulary: **HARVESTED** (already in the spine; the crate copy is a
dup to delete or a re-export), **HOT-VIA-FEATURE** (real compute, reached
only through `with-engine`), **GEM→landing** (unwired; a named spine home
and a consumer), **LAB** (calibration battery; keep in a lab crate),
**RESIDUE** (self-declared legacy; delete).

### 1a. HARVESTED — the spine already has it

| file / symbol | spine home | action |
|---|---|---|
| `cognitive_stack::RungLevel` | `contract::cognitive_shader::RungLevel` (`pub use` since E-RUNG-ASCENT-WIRED-1) | already a re-export; nothing to do |
| `cognitive_stack::ThinkingStyle` (12) | `contract::style_family::StyleFamily` (D-TSC-1) | deprecated alias; delete with the crate |
| `contract_bridge::contract_style_to_engine` | `StyleFamily::family()` | routed; delete with the crate |
| `ghosts::GhostType` (8) | `contract::escalation::GhostEcho` (identical 8, same order; `TD-GHOST-ECHO-DUP-1`) | the ENUM is harvested; the FIELD is not (see 1c) |
| `meaning_axes::Archetype` + council vote | `contract::escalation::{Archetype, InnerCouncil, is_split}` (D-PERSONA-1) | harvested; `CouncilWeights::modulate/shift_toward` has no spine twin — see 1c |
| `layered::CascadeChannels8` (ex-`CausalEdge64`, E-CE64-NAME-COLLISION-DEDUP) | 8-channel ↔ SPO transcoder D-CSV-9 (PR #387); channel = CE64 signed mantissa per FUTURE-DESIGN | transcoder shipped; the cascade itself is HOT-VIA-FEATURE (1b) |
| `dto::ResonanceDto` → `PerturbationDto` | D-PERT-1 (#630) | done; `awareness_dto::ResonanceDto` keeps its name (perspectival) |
| `cronbach::cronbach_alpha`, `QuorumLevel` | `jc::reliability::{cronbach_alpha, icc}` | dup; retire into jc (W2) |
| `superposition::ThinkingStyle = DetectedStyle` alias | renamed `DetectedStyle` (D-TSC-1) | the alias name still collides with `cognitive_stack`'s alias in the same crate; delete both with the crate |

### 1b. HOT-VIA-FEATURE — real compute, one caller behind `with-engine`

`engine.rs` (u8 MatVec, `cycle_vnni/cycle_auto`), `bf16_engine.rs`,
`signed_engine.rs`, `f32_engine.rs`, `builder.rs`, `lookup.rs`
(`TextToThought`), `codebook_index.rs`, `pooling.rs`, `sensor.rs`,
`dto.rs` (Φ/Ψ/B/Γ ladder), `awareness_dto.rs`, `qualia.rs` (`Qualia17D`),
`think.rs` (the "Thinking is a struct" carrier, minimum scope),
`superposition.rs`, `layered.rs`, and two lens modules (`bge_m3_lens`,
`reranker_lens`). (`role_tables.rs` is LAB, 1d; `jina_lens.rs` is RESIDUE,
1e — each file has exactly one fate; the two callers that still touch
`jina_lens` — callcenter's test and the driver's doc/test lookups — are
rewritten in W1, not carried.)

This group is the ALU's bus, not a crate-placement question (see §6.3): the
driver IS the ALU and the four DTOs are what crosses it; the contract's
`cognitive_shader::{ShaderDispatch, ShaderResonance, ShaderBus, ShaderCrystal}`
are their zero-dep twins. Two decisions own this group and both already have rows:
- **M8** (ENTROPY-MILESTONES, QUEUED): four near-duplicate engines
  (u8/BF16/i8/f32, same 7-method API) → one enum-dispatched engine with a
  parity suite across dtypes. `signed_domino`/`domino`/`composite_engine`/
  `dual_engine`/`branching` (1c) collapse into the same slice.
- **D-TTV-1** (Queued): thinking-related tenants → V3 substrate. This is
  the vehicle for the DTO ladder's home: `BusDto.{converged, cycle_count}`
  is the D-MBX-A6 Outcome signal (COMPONENT-MAP §3), so the ladder lands
  where the outcome is consumed — the ALU (driver) side, per
  `E-DTO-LADDER-OWNERSHIP-SPLIT` and W4. No crate ruling is needed
  (corrected 2026-09-02; §6.3).

### 1c. GEM → landing — unwired, with a named home AND a consumer

| file / symbol | what it is | landing | consumer that justifies the port | gate |
|---|---|---|---|---|
| `ghosts::GhostField::{imprint,bias,prediction,free_energy,prune}` | Friston prior as a per-atom decaying field; the anchoring alarm. **Family fence:** this is the LINGERING-TRACE family (Staunen = persistent wonder, Wisdom = harvested knowing) — NOT the non-authoritative counterfactual rung (the −6 lane, `deposit_counterfactual` / `CounterfactualMailbox`), whose docs call themselves "ghost-tier" (`TD-GHOST-TIER-NAME-COLLISION-1`). The rung may consume a trace as a starting prior; it is never one. `E-A-GHOST-TRACE-IS-NOT-THE-COUNTERFACTUAL-LANE-1` | planner `nars/ghost_prior.rs` over `contract::escalation::WisdomMarker`; per-thought/per-mailbox, NEVER a singleton field. **Intentional semantic change, not source parity** (Codex on #1137, verified): the decay rate agrees (0.85) but the FLOORS do not — `GhostField` drops contributions below 0.001 and `prune` deletes them, `WisdomMarker` clamps at 0.1 forever. Porting the field over the marker raises long-lived bias by up to two orders of magnitude; the port declares the floor it adopts and a calibration gate (free-energy response on a recurrence fixture under both floors) decides it | `house-differential-style-v1` D-HOUSE-4 (anchoring alarm) | falsifiers: bias decays monotonically to the DECLARED floor (whichever the calibration gate picks); `free_energy` RISES on a context shift and FALLS on a recurrence (two-sided); disable-run on the decay constant; the two-floor calibration comparison is reported, not assumed |
| `meaning_axes::CouncilWeights::{modulate, shift_toward}` | free-energy-weighted three-archetype modulation with renormalisation | `contract::escalation` beside `InnerCouncil` — only if a caller needs weighted (not majority) council output | none today | do NOT port until a caller exists; record the formula in the plan (done here) |
| `meaning_axes::AXES_48` + `AxisActivation` | 48 bipolar semantic axes (r = 0.9913 vs Jina cosine, per census) | `quorum.rs` names an `AxisId` BLOCKED on D-ATOM-1; AXES_48 is the natural candidate vocabulary | `quorum_project` per-axis projection | falsifier-first (the September ruling forbids a fixed axis basis minted by assertion): a size-preserving shuffle of axis assignments must NOT reproduce the 0.9913; until measured, the table stays where it is |
| `persona::{PersonaProfile, CognitiveBaseline}` | 12 personality constants + mode/temperature/rung bounds | a Layer-2 role catalogue DATA card (≤32 identities, I-VSA-IDENTITIES) in the persona storyline | none (O3: persona unwired by design) | park; no port |
| `persona::{A2AMessage, A2APayload, AgentDto}` | identity-carrying inter-agent messages | none | none | **RETIRE** — inter-mailbox handoff types were eliminated (`E-ACK-ELIMINATED-1`, no CollapseGateEmission/baton); COMPONENT-MAP already marks A2AMessage BLOCKED for exactly this |
| `world_model::{SelfState,UserState,FieldState,GestaltState,WorldModelDto}` | situational-awareness DTO | driver side with the DTO ladder (D-TTV-1) after the `GestaltState` dedup (defined twice: here and `awareness_dto.rs`) | driver | dedup first, then move with 1b |
| `cognitive_trace::{SpoTriple,CognitiveTrace,LensTrace}` | provenance chain text → thought | `temporal.rs` witness rows already carry provenance; a trace is a READ over versions | debug tooling only | port only if a probe needs it; otherwise delete with the crate |
| `domino.rs`, `signed_domino.rs`, `composite_engine.rs`, `dual_engine.rs`, `branching.rs` | 5 cascade shapes (top-K NARS-context cascade, signed variant, multi-lens superposition, u8-vs-BF16 disagreement, 4×4 spawning cascade) | M8 collapse surface | the one engine (1b) | parity suite; `branching.rs`'s "spawn, don't filter" shape is the idea worth keeping (§4) |
| `l4.rs`, `l4_bridge.rs`, `tensor_bridge.rs` | L4 "particle" personality layer + L3→L4 learning bridge (self-declared LOSSY) + embedding-output type | the LEARNED tenant of the triangle (`triangle-tenants-gestalt-separation-v1` W4b) | D-TRI-1 learned lane | BLOCKED on the orphan-write (`commit_to_l4` writes `&mut L4Experience` with no owner stamp — COMPONENT-MAP §3); stays until the learned lane wires, then ports as a ClassView reading |
| `prime_fingerprint.rs` | prime-DFT bit fingerprint, orthogonal by construction | VSA niche (I-VSA-IDENTITIES four tests) | none today | re-scope before any wiring; keep in LAB |
| `spiral_segment.rs` | 51× generative row compression (4×BF16 per row) | codec home: `bgz-tensor` / `ndarray::hpc` (encoding-ecosystem.md governs) | codec certification | `certification-officer` battery vs ground truth before promotion |
| `semantic_chunker.rs` | chunk boundaries = convergence jumps, no forward pass | `deepnsm-v2` (text side) or the paperless sentence assembler | tesseract-paperless `assemble_sentences` is the only live sentence producer | falsifier: boundaries vs a gold sentence split; else stays LAB |
| `contrastive_learner.rs`, `osint_bridge.rs` | online EMA table update; crawl → engine pipeline | `lance-graph-osint` owns the arc | osint arc | move with the osint arc or delete when it does |
| `inference_backend.rs` | 7-backend runtime registry ("nothing is killed, deprecation is data-driven") | LAB | R&D bench only | keep in the lab crate |
| `bridge_gate::{CognitiveOpKind, CognitiveAuthResult, CognitiveBridgeError, auth_to_result, CognitiveBridgeGate, PassthroughGate, DenyAllGate}` — the WHOLE public surface of the module, seven items | cross-tenant authorization injection point (trait + result/error enums + two reference gates) | `lance-graph-contract` (zero-dep, trait-only — it already says it lives low so callcenter can implement it), as one module, all seven items together; owner = contract | `lance-graph-callcenter` (live: `CognitiveBridgeGate`, `CognitiveAuthResult`, `CognitiveOpKind` at `cognitive_bridge_gate.rs:45`) | W1 compatibility contract: (a) all seven items move in one PR, identical shapes; (b) thinking-engine keeps `pub use lance_graph_contract::<module>::*` re-exports for that PR so callcenter's imports flip in the same change without a window where either path is missing; (c) callcenter's `pure_ops_emit_zero_audit_events` test is rewritten to exercise the gate without the three lens lookups (or moved to the driver's `with-engine` tests) BEFORE the dep line is deleted; (d) callcenter builds and its tests pass with no thinking-engine path dep — only then is the claim made |

### 1d. LAB — the calibration battery, split MATH → jc / GLUE → lab (ruling 4)

**Math → `jc`** (lifted if correct, perfected in jc if not; the crate copy
dies either way): `cronbach.rs` (`cronbach_alpha`, `QuorumLevel` — jc already
has `reliability::{cronbach_alpha, icc}`; the two are compared numerically on
one shared fixture before the crate copy is deleted, and any disagreement is
resolved in jc's favour only after the discrepancy is understood),
`ground_truth.rs::spearman_rank_correlation`, the drift statistics inside
`reencode_safety.rs` (x256 re-encode proof), the correction statistics inside
`silu_correction.rs`.

**Glue → lab crate** (`thinking-lab`, excluded, `calibration` feature, with a
`--manifest-path` CI line so the formatting/clippy debt cannot re-accumulate
unseen): `auto_detect.rs`, the candle forward-pass loaders in
`ground_truth.rs`, `tokenizer_registry.rs`, `centroid_labels.rs`, `bridge.rs`
(spiral → table index), `role_tables.rs` (BF16 per-role tables), and the
model-file plumbing of `silu_correction.rs`. Glue calls jc for its math; it
never carries a private copy.

### 1e. RESIDUE — delete

`splat_ops.rs` (self-declared deprecated, superseded by `Think::*`, removal
scheduled sprint-15+), `jina_lens.rs` (self-declared legacy; Jina v5 is
ground truth), the second `GestaltState`, the two `ThinkingStyle` aliases,
`persona::A2A*` (see 1c).

## 2. Open plans and rows that still point at the crate — and their fate

| item | today | fate under this plan |
|---|---|---|
| `rung-persona-orchestration-v1` D-PERSONA-1 | STATUS row says "In progress, branch claude/splat3d-…" | **STALE → Shipped**: `contract::escalation` (CollapseHint, InnerCouncil, EpiphanyDetector, GhostEcho, WisdomMarker, Checklist) and planner `mul/escalation.rs` (`verdict_from`, `boot_checklist`) are on main. Row regraded in this PR |
| D-PERSONA-2 meta-recipe manifest | Queued | absorbed by `house-differential-style-v1` §2 (a style program IS a declarative recipe composition) — first program = House; the manifest FORM waits for D-TSC-3 |
| D-PERSONA-3 hot/cold/feedback, `CrystalCodebook` → wisdom markers | Queued | absorbed by D-TEH-2 (ghost prior harvest) + D-HOUSE-4; the cold path is a WisdomMarker store keyed by fingerprint (I-VSA-IDENTITIES), not a history dump (rung-persona §8 already says so) |
| D-PERSONA-4 macro-eval harness | Queued | its first instance is PROBE-HOUSE-DIFFERENTIAL-1 (scenario → trace → diagnose); no separate harness before a second consumer |
| D-PERSONA-5 ractor outer-swarm, batons as messages | Queued | **RETIRED (operator ruling 2026-09-02)** — contradicts `E-NOBODY-WAITS-1` and `E-ACK-ELIMINATED-1`; its `MailboxId` dependant in `counterfactual.rs:243` re-homed on the W2a board tenant |
| D-PERSONA-6 Odoo `l10n_de` harvest | Queued | not thinking-engine work; belongs to the odoo blueprint plans, unchanged |
| D-TRI-1..6 (triangle tenants) | value-tenant half merged (#717); D-TRI-6 ascent wired | unchanged; the LEARNED lane is the landing for `l4*` (1c) |
| D-TSC-1b → TD (5 missing planner `default_modulation` arms) | In PR / TD open | small pay, independent of this plan; listed so it is not forgotten |
| D-TTV-1 thinking tenants → V3 | Queued | the vehicle for 1b's DTO ladder home; ruling on crate placement needed |
| D-REUNIFY-2 (8ch → SPO transcoder) | Backlog | **already shipped as D-CSV-9** (#387); row to regrade |
| D-REUNIFY-3 (`Think` carrier unification) | Backlog | `think.rs` exists at minimum scope; moves with 1b under D-TTV-1 |
| D-REUNIFY-4 (splat ops as `Think` methods) | Backlog | done in `think.rs`; `splat_ops.rs` is the residue (1e) |
| D-REUNIFY-5 (rayon par_*), D-REUNIFY-6 (DOLCE filter wiring) | Backlog | **CLOSED-superseded (operator ruling 2026-09-02)** by V3 (SoA sweep, ontology at the membrane) |
| ENTROPY M8 four-engine collapse | QUEUED | W3 of this plan |
| `TD-THINKING-ENGINE-EXCLUDED-DEBT-1` | open | paid by W1+W4 (the crate shrinks to a lab crate with a CI line) |
| `TD-GHOST-ECHO-DUP-1` | open | paid by W2 (field ported over `GhostEcho`; crate enum deleted) |
| `TD-RESONANCEDTO-DUP-1` | Deferred | resolved by D-PERT-1's split (two names now); row to close |
| `E-MORTON-CASCADE-V3-1` | ⊘ CONJECTURE | the "reimagine the compute as a V3 Morton read" path; independent of closure — closure must not delete the legacy arm it compares against until that probe runs or is dropped |

## 3. Closure sequence

Each wave lands only with its gate green; no wave ports without a consumer.

| wave | content | gate |
|---|---|---|
| **W0 (this PR)** | census, fate table, row regrades, idea harvest (§4) | boards consistent; SUPERSESSION-INDEX regenerated |
| **W1 — cut the hard dependency** | move `bridge_gate` trait + gates to `lance-graph-contract` (trait-only, zero-dep); callcenter re-imports; the `with-engine` re-point follows D-TTV-1 (the DTO home itself is ruled — §6.3) | **DONE 2026-09-02 (D-TEH-1)** for the hard edge: `lance-graph-callcenter` builds with no thinking-engine path dep; driver `with-engine` still green; thinking-engine is a LEAF for required edges. The `with-engine` re-point stays open until D-TTV-1 lands the engine hook |
| **W2 — harvest gems with consumers** | ghost prior → planner `nars/ghost_prior.rs` (D-TEH-2, consumer D-HOUSE-4); the MATH of the calibration battery → `jc` per ruling 4 (`cronbach` compared then lifted-or-perfected; Spearman, re-encode drift, SiLU-correction statistics likewise); `semantic_chunker` → deepnsm-v2 only if its falsifier passes; `spiral_segment` → codec home via certification battery | each port has a two-sided falsifier + a disable run; each source file deleted in the same PR |
| **W3 — M8** | one enum-dispatched engine; the 5 cascade shapes and 3 lens modules collapse; parity suite across u8/BF16/i8/f32 | NOT bit-parity across dtypes — u8 / BF16 / i8 / f32 differ in encoding by design, and `dual_engine.rs` exists to MEASURE that disagreement (Codex on #1137). Gate: per-dtype output tolerances plus dtype-invariant ranking/convergence invariants (top-k order, `converged`, `cycle_count` bounds) on real engine fixtures that instantiate all four engines (the driver's fixtures do not — they round-trip `BusDto` only); the pre-collapse `DualResult` disagreement is the baseline the collapsed engine must not exceed; the `branching` spawn shape kept as a mode, not lost |
| **W4 — retire and rename** | delete RESIDUE (1e) + retired persona A2A; rename what is left (GLUE only, all math already in jc) `thinking-lab` (calibration feature, `--manifest-path` CI line); regrade every row in §2; pay `TD-THINKING-ENGINE-EXCLUDED-DEBT-1` | the name `thinking-engine` no longer appears in any `Cargo.toml` dependency; board rows closed or re-owned |

Stop rules (non-negotiable inside this plan): **the closure cuts AROUND the
hot artery, never through it** — the live chain engine semantics → DTO bus
(`StreamDto` / `PerturbationDto` / `BusDto` / `ThoughtStruct`) →
cognitive-shader-driver ALU → SoA field is architecture, not residue, and no
wave here moves, re-homes, or re-shapes it (§6.3); nothing ports without a
consumer named in §1b (the DTO ladder, D-TTV-1) or §1c (the gems); nothing ports as a singleton field or a new lane
(ClassView reading or per-mailbox module only); `A2AMessage` and any
inter-mailbox handoff type never revive; the `E-MORTON-CASCADE-V3-1`
legacy arm is not deleted while that probe is open; no port of a fixed
axis vocabulary by assertion (AXES_48 goes falsifier-first or stays LAB).

**W2 result (2026-09-02, D-TEH-2).** The ghost prior landed as planner `nars/ghost_prior.rs` — per-thought, over `WisdomMarker` / `GhostEcho`, no singleton. The floor question the §1c row left open was decided by the pre-registered calibration gate, and it decided AGAINST the first declaration: `Trace` (source 0.001 + prune) loses all discrimination between a recurrence and a shift once the remembered pattern is older than ~42 cycles, `Marker` (contract 0.1, never pruned) keeps it (0.0188) at the cost of a higher absolute free-energy baseline (0.35 vs 0.07 with 30 stale patterns). Default = `Marker`. The remaining W2 items (math → jc, `semantic_chunker`, `spiral_segment`) are D-TEH-3/-5 and untouched here.

**W2 result (2026-09-02, D-TEH-3, math half).** The calibration battery split by nature as §1d prescribed. MATH went to jc: the re-encode drift statistic and the correction-delta summary as a new `jc::drift` (codec and cut-offs are parameters the lab supplies), the lens quorum and Cronbach report as `jc::quorum`; Cronbach α and Spearman ρ were already in `jc::reliability`, so the gate was a comparison, and it discriminated: the lab's α was the same estimator in `f32` (LIFT — the `f64` form survives a `1e7` offset the `f32` one does not), the lab's ρ was NOT the same estimator (ties ranked by position, ρ = 1.000 where the average-rank form gives 0.948683 — PERFECT-IN-JC, already there). GLUE stayed: `reencode_safety.rs` chooses the codecs, `silu_correction.rs` names its cut-offs, `ground_truth.rs` keeps the loaders; each calls jc and carries no statistic. `cronbach.rs` deleted. Two copies outside the lab (`ndarray::hpc::reliability`, `perturbation-sim::stats`) are recorded as `TD-RELIABILITY-COPIES-OUTSIDE-JC-1`, not swept. The `semantic_chunker` / `spiral_segment` halves of D-TEH-3 remain gated on their falsifiers.

## 4. The good ideas — what the chapter leaves behind even where the code dies

Kept as doctrine with a named present or future home, so the closure is a
harvest and not an amputation.

| idea (from the crate's own docs) | keep as | home |
|---|---|---|
| "Models are sensors. The matrix is the brain. DTOs are the bus. One MatVec per cycle." | the Zone-1 cost model (200–500 ns cycle) that sizes every rung budget | `VISION.md`, three-zone hot-path model (#372) |
| Ghosts = Friston priors with an asymptotic floor; anchoring is MEASURABLE as free energy between prior and evidence | the anchoring alarm | D-TEH-2 / D-HOUSE-4 |
| Staunen / Wisdom as a DUALITY: wonder keeps a question alive, wisdom is what was harvested from it. The shipped `curiosity() = (1/(q+1))·(1−c)` is a later operational SLICE of Staunen, not the original pair | the two ends of the epistemic cycle, kept as two names | `exploration.rs` (curiosity), `escalation.rs` (`GhostEcho::{Staunen, Wisdom}`, `promote_to_wisdom`) |
| A council SPLIT is the learning signal (one archetype sees what the others do not) — amplify it, never average it | shipped invariant | `escalation.rs`, `quorum.rs` ("split quorums are Contradictions — NEVER averaged") |
| 48 bipolar meaning axes with a measured r = 0.9913 against a real embedding | the candidate for the per-axis quorum vocabulary, falsifier-first | `quorum.rs` D-ATOM-1 pointer (1c) |
| Branching cascades SPAWN, they do not filter (parallel vectors, not survivors) | a mode of the one engine after M8 | W3 |
| "Nothing is killed; deprecation is data-driven" (the backend registry doctrine) | the standing rule of this very plan (every retirement here cites a measurement or a ruling) | this plan §1 |
| Chunk boundaries are convergence jumps — segmentation without a forward pass | a falsifiable chunker for the sentence seam | W2 (`semantic_chunker`) |
| An L4 "particle" layer that learns OUTSIDE the wave cascade, XOR-bound, save/load | the LEARNED tenant of the autopoiesis triangle | D-TRI-1 learned lane |
| Provenance text → thought as an appendable trace | already the temporal witness's job; keep as a read | `temporal.rs` |
| The 8-channel cascade edge (BECOMES / CAUSES / SUPPORTS / REFINES / GROUNDS / ABSTRACTS / RELATES / CONTRADICTS) | transcoded to the SPO palette, kept as the comma-level mantissa carrier | D-CSV-9, FUTURE-DESIGN "first wiring target" |
| Persona as 12 constants + a mode with rung bounds and a collapse bias | a Layer-2 data card for the persona storyline, when it opens | O3 (parked) |

## 4b. D-TEH-3 fate probes — PRE-REGISTERED before the first run

The two remaining §1c rows (`semantic_chunker`, `spiral_segment`) each carry a
falsifier as their gate, and a gate written after the numbers is not a gate.
Both probes, their arms and their thresholds are fixed HERE, before either was
executed; the run only fills in the measurements.

### `semantic_chunker` — `examples/chunker_falsifier.rs`

The §1c claim is "chunk boundaries are convergence jumps, no forward pass",
and the row's gate is "boundaries vs a gold sentence split; else stays LAB".
The probe uses the tier-1..4 calibration corpus of `jina_v5_ground_truth.rs`
(Rule 23 — real text, not synthetic), the Jina v5 tokenizer, the baked
`jina-v5-codebook` index + 256x256 table. Sentence pairs from DIFFERENT corpus
pairs are concatenated, so the gold boundary is the token seam by construction.

| arm | what it measures | why it is there |
|---|---|---|
| can-fire | recall@+-4 tokens of the seam over 168 cross-topic passages | a chunker that never fires is not a chunker |
| null | the same passages with centroid order shuffled (20 SplitMix64 permutations), p95 of recall | "a boundary landed near the seam" must beat chance, and a boundary detector on shuffled input still finds boundaries |
| silence | the 8 same-topic (tier-1/2) passages, both orders | a chunker that splits everything carries as much information as one that never splits |

**PASS** (at one threshold, the same threshold for all three arms):
recall >= 0.75 AND recall >= null_p95 + 0.15 AND false splits <= 2 of 8.
**KILL** otherwise.

- PASS commits: the port to `deepnsm-v2`'s text side is on, as its own PR with
  the probe carried across as the port's regression gate. It does NOT commit
  the tesseract-paperless sentence assembler to using it — that consumer is a
  separate decision with its own falsifier.
- KILL commits: the module stays LAB and its §1c row is closed as
  measured-negative, with the numbers on the board. It is NOT deleted (a LAB
  verdict is a home, not a death sentence) and it is NOT re-probed on a
  different corpus to get a better answer.

### `spiral_segment` — `examples/spiral_gate_probe.rs`

The §1c row routes this to a codec home "via the certification battery"
(`certification-officer`). That battery needs the F32 cosine matrix re-derived
from the model source, which is not on disk here. This probe is the CHEAP GATE
in front of that expensive step: fit the codec to the four real baked 256x256
tables in the tree (jina-v3, bge-m3, reranker, jina-v5 u8 + jina-v5 i8) and ask
whether it can clear the ecosystem floor at all.

Thresholds are the ecosystem's own, not invented here:
`encoding-ecosystem.md` — "any encoding below the naive u8 floor is worse than
doing nothing"; the bgz-hhtl-d gate is Pearson >= 0.9980. Plus the claim the
module's own doc makes ("51x compression"): the codec must at least halve what
it replaces, or it is a lossy re-encoding of a u8 table for no space.

**PASS** if for SOME `max_error`, on EVERY table: r >= 0.9980 AND rho >= 0.9980
AND spiral bytes <= u8 bytes / 2. **KILL** otherwise.

- PASS commits: the certification battery is unblocked and scheduled — this
  probe is explicitly NOT a certification (a baked u8 table is not the atomic
  clock; only the source-derived F32 matrix is).
- KILL commits: the module stays LAB, the 51x claim is recorded as
  measured-false at the fidelity the workspace requires, and no battery is run.
  A codec that cannot preserve a baked table will not preserve its F32 parent.

Both probes report a per-configuration table so a KILL says WHERE it failed,
not merely that it failed.

## 4c. D-TEH-3 fate probes — RESULTS (both KILL, both stay LAB)

Both probes run on real data (`crates/thinking-engine/data/jina-v5-codebook/`
+ `jina-v3-hdr/` + `bge-m3-hdr/` + `jina-reranker-v3-BF16-hdr/`, no synthetic
input), against the arms and thresholds pre-registered in §4b. **Neither
threshold was retuned after seeing a result** — the pre-registration's own
rule.

### `semantic_chunker` — KILL, and the null result is a genuine mechanism
### null, not a harness artifact

```
threshold |    recall | null p95 | false splits |  bnd/pass | verdict
     0.30 |     0.000 |    0.000 |       0 of  8 |      0.00 | kill
     0.45 |     0.000 |    0.000 |       0 of  8 |      0.00 | kill
     0.60 |     0.000 |    0.000 |       0 of  8 |      0.00 | kill
```

recall = 0.000 at every threshold means `find_boundaries` never fired ONCE
across all 168 cross-topic passages, at any of the three thresholds swept.
Before trusting an all-zero result, a diagnostic ran the module's OWN
adversarial positive-control shape from its `detects_boundary_between_topics`
test (synthetic centroid corners 0-4 vs 250-254, maximally distant in the
256-centroid table) against the same `jina-v5-codebook` table the falsifier
used. **The positive control ALSO produced zero boundaries.** This was first
run as a throwaway, non-committed script — a real gap Codex review caught on
this PR (#1144): a deleted diagnostic means a later reader can reproduce the
KILL, but not the reasoning for calling it a mechanism null rather than a
harness bug. Fixed by committing the control as a 4th arm inside
`chunker_falsifier.rs` itself (reproduces the same zero, confirmed by
re-running it after landing). So this is a genuine mechanism/table-level null:
the perturb-think-top-k-Jaccard convergence pattern at `max_cycles: 10` does
not discriminate on this HDR-encoded table at all, even on inputs designed to
be maximally separable. The module's own pre-existing test
(`detects_boundary_between_topics`) already carried a comment hedging exactly
this ("On uniform HDR tables the convergence patterns may not diverge
strongly") — the falsifier turns that hedge into a measured, pre-registered
KILL rather than an unverified doubt.

**Verdict: KILL.** `semantic_chunker` stays LAB. Not ported to deepnsm-v2. Not
deleted. Not re-probed on a friendlier corpus to chase a different answer.

### `spiral_segment` — KILL, and the failure is compression ratio on u8
### (fidelity holds), compression AND fidelity on i8 (correction below)

```
       table | max_error |       r |     rho |    x u8 |  seg/row |
  jina-v3 u8 |     0.005 |  1.0000 |  1.0000 |    0.22 |   142.27 |
  jina-v3 u8 |     0.050 |  0.9991 |  0.9991 |    0.28 |   114.11 |
   bge-m3 u8 |     0.005 |  1.0000 |  1.0000 |    0.22 |   142.61 |
 reranker u8 |     0.005 |  1.0000 |  1.0000 |    ~0.2 |     ~140 |
  jina-v5 u8 |     0.050 |  0.9991 |  0.9991 |    0.28 |   113.72 |
  jina-v5 i8 |     0.005 |  0.9993 |  0.9975 |    0.28 |   115.10 |
  jina-v5 i8 |     0.050 |  0.7709 |  0.7137 |    0.80 |    39.91 |
```
(full table: 5 tables x 4 max_errors = 20 rows, see the example's own output)

Fidelity (Pearson r, Spearman rho) clears the 0.9980 gate comfortably on every
u8 table at every max_error tried. **The codec is accurate on u8 tables.**
The i8 table is the one real exception, caught by Codex on this PR
(`chatgpt-codex-connector[bot]`, P2): at its best configuration
(max_error 0.005) r = 0.9993 clears the floor but rho = 0.9975 does not
(the table above already shows this row) — i8 fidelity never clears BOTH
gates at any max_error tested, and gets strictly worse as max_error loosens
(rho 0.9975 -> 0.9906 -> 0.9590 -> 0.7137). Compression ALSO fails on i8 at
every max_error (x u8 tops out at 0.80x, still short of `MIN_RATIO_VS_U8 = 2.0`),
so the KILL verdict for `spiral_segment` is unaffected by this correction —
the codec fails BOTH gates on i8, not just the compression gate the u8
tables already fail. What the original write-up got wrong was calling this
a "compression-only" failure; it is compression-only on u8, and a
double failure on i8. What kills it, in either case, is compression:
`x u8` (spiral bytes vs u8-table bytes) never exceeds ~0.28x on a u8 table —
i.e. the spiral encoding is **~3.6x LARGER** than the u8 table it would
replace, not smaller, let alone the "51x compression" the module's own doc
comment claims. Root cause visible in `seg/row`: a real HDR/CDF-encoded
distance row needs ~114-143 spiral segments (8 bytes each) to hit even a
0.05 max_error, because the fitting premise (few segments per row) only holds
on smooth, low-curvature synthetic data — a real codebook's per-row CDF is not
smooth. Even at the loosest max_error swept (0.05, well past the fidelity
gate's own comfort margin), no table's compression ratio approaches the
`MIN_RATIO_VS_U8 = 2.0` floor, let alone the ecosystem's u8-beats-nothing
floor doubled.

**Verdict: KILL.** `spiral_segment` stays LAB. The certification battery
(F32-source re-derivation, `certification-officer`) is NOT scheduled — a
codec that cannot beat a baked u8 table by 2x will not beat its F32 parent
either. The "51x compression" doc-comment claim is recorded here as
measured-false at the fidelity this workspace requires; it was never false at
the fidelity the module tested itself against (smooth synthetic curves), only
against the real distributional shape of a trained model's own table.

### §1c is now CLOSED — every row in the table has a verdict

Both remaining open rows (`semantic_chunker`, `spiral_segment`) are now KILL,
joining the earlier D-TEH-2/D-TEH-3 PASS rows already landed
(#1142/#1143). No §1c row remains unprobed.

## 5. Deliverables

| D-id | title | scope | status |
|---|---|---|---|
| D-TEH-0 | census + fate table + open-row reconciliation + idea harvest (this plan) | plan + board rows | Shipped (this PR) |
| D-TEH-1 | W1: `bridge_gate` (seven items) → `lance_graph_contract::bridge_gate`; callcenter re-imports and drops the path dep; thinking-engine keeps a re-export shim | contract + callcenter | **Shipped 2026-09-02** — edge measured before (required dep, 6 crossing sites, dep-drop fails 6 × E0433) and after (zero thinking-engine deps in callcenter metadata; 1303 + 156 tests, driver default + `with-engine` green). The `with-engine` re-point is NOT part of this wave: D-TTV-1 is Queued and the engine hook still lives in thinking-engine, so there is nothing to re-point it at (stop condition honoured). thinking-engine is now a leaf for every REQUIRED edge; the one remaining edge is the ALU's optional engine hook |
| D-TEH-2 | W2: ghost prior harvested as planner `nars/ghost_prior.rs` over `WisdomMarker`, per-thought, with two-sided falsifiers; crate `ghosts.rs` deleted | planner | **Shipped 2026-09-02** — planner `nars/ghost_prior.rs` (`GhostPrior`, `PriorFloor`, `Trace`, `calibration::{recurrence_fixture, discrimination}`; 14 tests); `ghosts.rs` + `examples/think.rs` deleted; lab `persona`/`world_model`/`awareness_dto` re-pointed to `contract::escalation::GhostEcho` (TD-GHOST-ECHO-DUP-1 resolved). Calibration gate REVERSED the first-declared floor: `Marker` (0.1, never pruned) discriminates ≥ `Trace` (0.001) on every fixture row and strictly once the remembered pattern ages past its prune point (disc 0.0188 vs 0.0000 at 30 stale / age 20 and 60); default = `Marker`. Consumer D-HOUSE-4 unblocked |
| D-TEH-3 | W2: calibration MATH → jc (ruling 4: compare, then lift or perfect in jc; crate copies deleted); `semantic_chunker` / `spiral_segment` decided by their falsifiers | jc / deepnsm-v2 / codec home | **Shipped 2026-09-02/03 — all three halves closed.** Math: new `jc::drift` (`reencode_drift` / `reencode_batch` / `delta_summary`) and `jc::quorum` (`pairwise_agreement_u8` / `QuorumLevel` / `cronbach_report`); lift gate ran on distinguishing fixtures: cronbach = same estimator (LIFT; the `f32` copy loses a `1e7`-shifted fixture the `f64` form holds to `1e-9`), spearman = the retired copy ranked ties by position (PERFECT-IN-JC: 1.000 vs 0.948683 on `[1,2,2,3]`). `cronbach.rs` deleted; `reencode_safety` / `silu_correction` / `ground_truth::calibration` are glue over jc, x256 proof green (14 tests). **`semantic_chunker`: KILL** (§4c) — recall 0.000 at every threshold, confirmed a genuine mechanism null (not a harness artifact) via a non-committed positive-control diagnostic on the module's own adversarial fixture shape; stays LAB. **`spiral_segment`: KILL** (§4c) — fidelity clears r/rho >= 0.9980 on every u8 table but NOT on i8 (rho tops out at 0.9975, corrected 2026-09-03 per Codex review on #1144); compression fails on every table regardless, topping out at ~0.28x a u8 table (i.e. ~3.6x LARGER, not the claimed 51x), because real HDR/CDF table rows need ~114-143 segments to hit even a loose max_error; stays LAB, certification battery not scheduled |
| D-TEH-4 | W3: M8 engine collapse with parity suite; cascade shapes and lens modules collapse | thinking-engine → the one engine | Queued (owns ENTROPY M8) |
| D-TEH-5 | W4: residue deleted, crate renamed `thinking-lab` with a CI line; §2 rows closed; TD paid | workspace | Queued — closes the chapter |

## 6. Rulings — asked 2026-09-02, ruled the same day (operator)

1. **D-PERSONA-5 RETIRED** (contradicts E-NOBODY-WAITS-1 /
   E-ACK-ELIMINATED-1). Its one real dependant — `MailboxId` assignment for
   the counterfactual v3 mailbox (`counterfactual.rs:243` "D-PERSONA-5 dep")
   — is re-homed on the W2a board tenant, where mailbox ownership already
   lives. Row updated.
2. **D-REUNIFY-5 and D-REUNIFY-6 CLOSED-superseded** by V3: R-5's
   parallelism target was the singleton BindSpace (retired; any sweep is a
   new row against `MailboxSoA`); R-6's in-engine ontology filter is
   superseded by ontology at the membrane (`lance-graph-ontology` /
   callcenter). Rows updated.
3. **The four wire structs' home — CLOSED: it was already ruled, and I
   re-asked it (corrected 2026-09-02, after reading `.claude/v3/` in
   full).** `StreamDto` / `PerturbationDto` / `BusDto` / `ThoughtStruct`
   (`thinking-engine/src/dto.rs`) are the bus of the ALU chain
   `ladybug-rs → thinking-engine → P64 → cognitive-shader-driver → SoA`
   (`v3-substrate-primer.md` §3, `VISION.md` §6,
   `E-DTO-LADDER-OWNERSHIP-SPLIT`, INTEGRATION-PLAN W4 / D-V3-W4a). In
   plain terms: **cognitive-shader-driver is the ALU** — it holds the
   SoA columns, dispatches a cycle through the engine hook, and emits the
   cycle fingerprint through sinks; the four structs are what enters and
   leaves that ALU. The contract already carries the zero-dep shader-side
   twins of the same four rungs (`cognitive_shader::{ShaderDispatch,
   ShaderResonance, ShaderBus, ShaderCrystal}`, `engine_bridge.rs:6-8`
   maps engine ↔ shader rung by rung). So the home question has no third
   option and needs no crate decision: the shape belongs to the ALU side of
   the bridge and lands under D-TTV-1 / W4 exactly as INTEGRATION-PLAN
   already says. **The earlier text here (a small DTO crate, or an
   `ogar-r2il` round-trip probe to decide the home) is RETRACTED** — it
   re-derived a ruled question and proposed a probe on the wrong axis.
   What IS open on this chain is recorded elsewhere and is not this plan's
   to re-open: PR #1051 measured the seam as transport, not a field ALU
   (`PerturbationDto.energy` dropped, `top_k` collapsed to a
   `ColumnWindow`), and `alpha-reason-witness-shader-field-lineage-addendum-v1`
   (D-ARW) owns recovering the field path, with **stockfish-rs as the
   reference design** for the 64×64 field: NNUE teaches the incremental
   make/unmake accumulator, and masking + SIMD are what make the ALU a
   reusable thinking-compiler driver that any consumer can dispatch into as
   a cognitive shader. Consequence for THIS plan: 1b's engines and DTOs
   are HOT-VIA-FEATURE on the ALU chain; the closure never moves them into
   a lab crate, a DTO crate, or r2il — W1 only re-points the driver's
   `with-engine` feature at wherever D-TTV-1 lands the engine hook, and
   the shapes stay with the ALU.
   The next falsifier on that chain is NOT opened from this plan; it
   already exists conceptually in the lineage addendum and has a natural
   canary: does a semantically meaningful field quantity —
   `PerturbationDto.energy`, the one #1051 found dropped — survive
   engine → DTO → shader → SoA → consumer without collapsing into
   transport metadata, under make/unmake-style incremental updates and
   masking. That is D-ARW work; this plan only guarantees it finds the
   artery intact.
4. **jc is the home of ALL scientifically calibrated math** (operator,
   2026-09-02). The rule, as ruled: anything in the calibration battery that
   is MATH (Cronbach α, Spearman / ICC, re-encode drift statistics, the
   SiLU-correction statistics) is either LIFTED into `jc` as-is when it is
   correct, or — if it sails under a wrong name or is incorrect — the jc
   version is perfected and the crate copy dies; jc's pillars are the
   reusable, liftable calibration source, and once ndarray is proven
   bit-exact the same math can be re-imported into production from jc.
   This is also why `sigker` stayed OUT of jc for now (Pillar 11 red for
   non-lattice step vectors). Glue that is not math — candle loaders,
   `tokenizer_registry`, `centroid_labels`, `auto_detect`, model-file
   plumbing — is not jc material and stays in the lab crate. §1d, W2 and
   W4 are amended accordingly below.

## 7. What this PR does NOT do

No file moves, no deletions, no Cargo changes, no feature flips, no new
types. Row regrades touch only status cells and cite their evidence. The
crate builds exactly as before.
