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

Two decisions own this group and both already have rows:
- **M8** (ENTROPY-MILESTONES, QUEUED): four near-duplicate engines
  (u8/BF16/i8/f32, same 7-method API) → one enum-dispatched engine with a
  parity suite across dtypes. `signed_domino`/`domino`/`composite_engine`/
  `dual_engine`/`branching` (1c) collapse into the same slice.
- **D-TTV-1** (Queued): thinking-related tenants → V3 substrate. This is
  the vehicle for the DTO ladder's home: `BusDto.{converged, cycle_count}`
  is the D-MBX-A6 Outcome signal (COMPONENT-MAP §3), so the ladder lands
  where the outcome is consumed — the driver side — never in the zero-dep
  contract. Ruling needed on WHICH crate (driver vs a small `thinking-dto`
  crate); this plan does not decide it.

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

### 1d. LAB — the calibration battery (keep, rename, gate)

`auto_detect.rs`, `ground_truth.rs` (candle-gated), `reencode_safety.rs`
(x256 re-encode proof), `silu_correction.rs`, `tokenizer_registry.rs`,
`centroid_labels.rs`, `bridge.rs` (spiral → table index), `role_tables.rs`
(BF16 per-role tables). These feed D-MTS-2's certification gate and are
calibration-only by design. Fate: they survive as a lab crate
(`thinking-lab`, excluded, `calibration` feature) with a `--manifest-path`
CI line so the formatting/clippy debt cannot re-accumulate unseen.

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
| D-PERSONA-5 ractor outer-swarm, batons as messages | Queued | **proposed RETIRE** — contradicts `E-NOBODY-WAITS-1` (ractor = compile-time ownership only; no messages, no actors) and `E-ACK-ELIMINATED-1`; ruling needed, noted on the row |
| D-PERSONA-6 Odoo `l10n_de` harvest | Queued | not thinking-engine work; belongs to the odoo blueprint plans, unchanged |
| D-TRI-1..6 (triangle tenants) | value-tenant half merged (#717); D-TRI-6 ascent wired | unchanged; the LEARNED lane is the landing for `l4*` (1c) |
| D-TSC-1b → TD (5 missing planner `default_modulation` arms) | In PR / TD open | small pay, independent of this plan; listed so it is not forgotten |
| D-TTV-1 thinking tenants → V3 | Queued | the vehicle for 1b's DTO ladder home; ruling on crate placement needed |
| D-REUNIFY-2 (8ch → SPO transcoder) | Backlog | **already shipped as D-CSV-9** (#387); row to regrade |
| D-REUNIFY-3 (`Think` carrier unification) | Backlog | `think.rs` exists at minimum scope; moves with 1b under D-TTV-1 |
| D-REUNIFY-4 (splat ops as `Think` methods) | Backlog | done in `think.rs`; `splat_ops.rs` is the residue (1e) |
| D-REUNIFY-5 (rayon par_*), D-REUNIFY-6 (DOLCE filter wiring) | Backlog | superseded by V3 (SoA sweep, ontology at the membrane); propose CLOSE-superseded, ruling needed |
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
| **W1 — cut the hard dependency** | move `bridge_gate` trait + gates to `lance-graph-contract` (trait-only, zero-dep); callcenter re-imports; decide the DTO-ladder home (D-TTV-1 ruling) so the driver's `with-engine` feature can point at it | `lance-graph-callcenter` builds with no thinking-engine path dep; driver `with-engine` still green; thinking-engine becomes a LEAF |
| **W2 — harvest gems with consumers** | ghost prior → planner `nars/ghost_prior.rs` (D-TEH-2, consumer D-HOUSE-4); `cronbach` → retire onto `jc::reliability`; `semantic_chunker` → deepnsm-v2 only if its falsifier passes; `spiral_segment` → codec home via certification battery | each port has a two-sided falsifier + a disable run; each source file deleted in the same PR |
| **W3 — M8** | one enum-dispatched engine; the 5 cascade shapes and 3 lens modules collapse; parity suite across u8/BF16/i8/f32 | NOT bit-parity across dtypes — u8 / BF16 / i8 / f32 differ in encoding by design, and `dual_engine.rs` exists to MEASURE that disagreement (Codex on #1137). Gate: per-dtype output tolerances plus dtype-invariant ranking/convergence invariants (top-k order, `converged`, `cycle_count` bounds) on real engine fixtures that instantiate all four engines (the driver's fixtures do not — they round-trip `BusDto` only); the pre-collapse `DualResult` disagreement is the baseline the collapsed engine must not exceed; the `branching` spawn shape kept as a mode, not lost |
| **W4 — retire and rename** | delete RESIDUE (1e) + retired persona A2A; rename what is left `thinking-lab` (calibration feature, `--manifest-path` CI line); regrade every row in §2; pay `TD-THINKING-ENGINE-EXCLUDED-DEBT-1` | the name `thinking-engine` no longer appears in any `Cargo.toml` dependency; board rows closed or re-owned |

Stop rules (non-negotiable inside this plan): nothing ports without a
consumer named in §1b (the DTO ladder, D-TTV-1) or §1c (the gems); nothing ports as a singleton field or a new lane
(ClassView reading or per-mailbox module only); `A2AMessage` and any
inter-mailbox handoff type never revive; the `E-MORTON-CASCADE-V3-1`
legacy arm is not deleted while that probe is open; no port of a fixed
axis vocabulary by assertion (AXES_48 goes falsifier-first or stays LAB).

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

## 5. Deliverables

| D-id | title | scope | status |
|---|---|---|---|
| D-TEH-0 | census + fate table + open-row reconciliation + idea harvest (this plan) | plan + board rows | Shipped (this PR) |
| D-TEH-1 | W1: `bridge_gate` trait → contract; DTO-ladder home ruling recorded; thinking-engine becomes a leaf | contract + callcenter (+ driver feature pointer) | Queued — first code wave |
| D-TEH-2 | W2: ghost prior harvested as planner `nars/ghost_prior.rs` over `WisdomMarker`, per-thought, with two-sided falsifiers; crate `ghosts.rs` deleted | planner | Queued — consumer = D-HOUSE-4 |
| D-TEH-3 | W2: `cronbach` → jc, `semantic_chunker` / `spiral_segment` decided by their falsifiers | jc / deepnsm-v2 / codec home | Queued |
| D-TEH-4 | W3: M8 engine collapse with parity suite; cascade shapes and lens modules collapse | thinking-engine → the one engine | Queued (owns ENTROPY M8) |
| D-TEH-5 | W4: residue deleted, crate renamed `thinking-lab` with a CI line; §2 rows closed; TD paid | workspace | Queued — closes the chapter |

## 6. Rulings this plan asks for (not assumed)

1. D-PERSONA-5 RETIRE (contradicts E-NOBODY-WAITS-1 / E-ACK-ELIMINATED-1).
2. D-REUNIFY-5/6 CLOSE-superseded by V3.
3. The DTO ladder's crate home under D-TTV-1 (driver vs a small DTO crate).
4. Whether `thinking-lab` survives as a named excluded crate or the
   calibration battery moves into `jc` outright.

## 7. What this PR does NOT do

No file moves, no deletions, no Cargo changes, no feature flips, no new
types. Row regrades touch only status cells and cite their evidence. The
crate builds exactly as before.
