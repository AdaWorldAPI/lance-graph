## classid-canon-custom-flip-v1 — the TRIGGERED §2.3 atomic flip

Plan: `.claude/plans/classid-canon-custom-flip-v1.md`. Operator trigger 2026-07-02.

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-CCF-0 | compose_classid/split_classid/CLASSID_CANON_HIGH + route all sites (zero behavior) | lance-graph-contract | Shipped (fd9bf6b) | plan §3/§4 P0 |
| D-CCF-1 | Flip + mint new-form classids (0x0701_1000 / 0x0A01_1000 / 0x0E01_1000) coexisting | lance-graph-contract | In PR (#628) | gated on P0 probes |
| D-CCF-2 | OGAR#95 hi-u16 app-prefix reconciliation | contract + OGAR | In progress (resolved: prefix = custom half; OGAR flips in lockstep per operator) | plan §2 row / §4 P2 |
| D-CCF-3 | q2 re-mints (osint-bake + cpic via contract pull; dissolves ISS-Q2-CPIC-MIRROR) | q2 (gate WAIVED) | Queued | plan §4 P3 |
| D-CCF-4 | 0x1000 marker retirement | all | Blocked (operator checkpoint) | plan §4 P4 |

## v3-convergence-wiring-v1 — wire, don't invent (the seam list)

Plan: `.claude/plans/v3-convergence-wiring-v1.md`. Sonnet-grindwork/Fable-decisions split.

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-VCW-1a | RungLevel arithmetic + RungElevator (sustained-BLOCK policy over certified mask algebra; converged with escalation::rung_delta) | lance-graph-contract | **Shipped** | 755 lib tests green incl. 6 new; clippy clean |
| D-VCW-1b | Driver wiring: elevator through cycle loop, ctx.rung proxy retired, wire/grpc from_u8 dedup | cognitive-shader-driver | **Shipped** | driver 100/100 green (2 new tests: sustained-BLOCK elevation across dispatches + rung load-bearing in tactic selection); driver-persistent RwLock elevator, base-change reset |
| D-VCW-2 | P6 wave-convergence probe (wave dist == certified palette read) | lance-graph core (arigraph) | **Shipped** | markov_soa 6/6 green (2 new P6 tests) |
| D-VCW-3 | P7 render probe (bitmask → askama; fields == masked tenants) | q2 (**gate WAIVED 2026-07-02**) | Queued (unblocked) | spec ready (plan §3) |
| D-VCW-4 | One-row registry + read-mode parity fuse | contract (+OGAR Phase B) | Queued | plan §4; Phase B operator-gated |
| D-VCW-5 | cascade3 nibble-ancestry falsifier | q2 (**gate WAIVED 2026-07-02**) | Queued (unblocked) | ISS-Q2-CASCADE3-NIBBLE-ANCESTRY |
| D-VCW-6 | Rule 7: negative-existence claims need exhaustive-search declaration | knowledge doc | **Shipped** | autoattended-multiagent-pattern.md §5 Rule 7 |
| D-VCW-7 | rig/rs-graph-llm FailureTicket loop | rs-graph-llm (sibling) | Deferred | plan §6; probe-first when opened |

## cognitive-compilation-v1 — Elixir-template stack (LLM teaches, Lance-Graph runs)

Plan: `.claude/plans/cognitive-compilation-v1.md`. The new idea is the
Elixir-shaped template; the rest of the loop reuses existing organs.

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D-CC-RUNTIME-1 | elixir-template: representation + parser + `source_ranking_v1` slice | elixir-template | **Scaffolded** | 6 tests green (parse, version split, custom atom, round-trip, 7-step slice + guardrail); clippy clean |
| D-CC-RUNTIME-2 | template-runtime: deterministic OGAR-action dispatch (reflex executor) | template-runtime | **Scaffolded** | 4 tests green (threaded dispatch, unknown-action, empty, unimplemented-bubbles); action bodies deferred |
| D-CC-EQUIV-1 | template-equivalence: replay grading | template-equivalence | **Scaffolded** | 4 tests green (Exact, RankOrder-within-tolerance, new-claim-fail, confidence-drift-fail); Semantic deferred |
| D-CC-COMPILER-1 | cognitive-compiler: trace→template synthesis surface | cognitive-compiler | **Scaffolded** | 3 tests green (NotImplemented contract, non-Execution reject, unsourced-claim reject); synthesis = first probe |
| D-CC-RIG-1 | rig-surrealdb pointed at AdaWorldAPI kv-lance fork | rig (sibling) | Queued | additive Cargo wiring |
| D-CC-RUBICON-1 | graph-flow Task for templates (isolated, cherry-pickable) | rs-graph-llm (sibling) | Queued | local copy + branch push as recovery paths |
| D-CC-OGAR-1 | OGAR canonical classes for the loop | OGAR ogar-ontology/ogar-from-elixir | **Exists** | reused, not rebuilt |
| D-CC-INDEX/REVIEW/PROMOTE/LEDGER/FENCE | basin match / reviewers / PR automation / provenance / ownership fence | planner / agents / surreal kv-lance / ractor | **Exists or DEFERRED** | not this PR |

---

## symbiont-golden-image-harness — the living all-in-one substrate binary + the first runtime edges

The golden image (`crates/symbiont`, workspace-`exclude`d): the full Ada stack in ONE binary, then real cross-crate edges onto the canonical SoA. Plan: `crates/symbiont/INTEGRATION_PLAN.md` (PR #555, merged `37cc21b2`).

| D-id | Title | Crate(s) | Status | Evidence |
|---|---|---|---|---|
| D0 | Golden image compiles+links (lockstep lance-7) | symbiont | **Shipped** | git-deps build `CARGO_EXIT=0`, unified `lance 7.0.0 / lancedb 0.30.0 / df 53.1 / arrow 58`, binary runs |
| D1 | Grid→NodeRow bridge — each bus = 1 SoA board, f64 → `Energy` tenant | symbiont/bridge.rs | **Shipped** | 2 probes green; 64 buses→64 NodeRows, perturbation in the Energy(f32) tenant, all finite |
| E2 | Parallel SoA sweep at scale (16k boards = 8 MiB, zero-copy) | symbiont/bridge.rs | **Shipped** | `run_scale_demo(16384)` → 8 MiB, all 16384 Energy tenants finite |
| D3-AMX | Domino POC — 16-board AMX 16×16 BF16 Morton-tile cascade + NaN-projection | symbiont/domino.rs | **Shipped** | 3/3 tests green; 256 boards × 16 AMX-16×16 batches × 3-stage BF16 Morton-tile Domino cascade, NaN-clean via the projection surface. Polyfill-only (`ndarray::simd::bf16_tile_gemm_16x16` re-export `05bfea7a` jirak; `f32_to_bf16_batch_rne`; only `morton4` consumer-side). **Ran AVX-512 fallback** — AMX genuinely OFF on this guest (functional probe `/tmp/amxcheck`: XCR0 tile bits 17/18 = 0, `arch_prctl(158)` XTILEDATA = **-95 -EOPNOTSUPP** kernel refuses; CPUID also masked). NOT merely CPUID-masked → cannot be enabled here; a forced byte-encoded TDPBF16PS would fault. AMX dispatch correct + arch_prctl-158 gotcha-safe; fires `[AMX TDPBF16PS]` on an AMX-granted guest. |
| D2 | Kanban loop — pure-SoA slice (version-tick → `NextPhaseScheduler` → `try_advance_phase`) | symbiont/kanban_loop.rs | **Shipped (slice)** | 2/2 tests green; `SymbiontBoard` impls `MailboxSoaView`+`MailboxSoaOwner` over the `Vec<NodeRow>`, a `u32` tick stands in for the Lance subscription; drove `Planning→CognitiveWork[BF16 Domino sweep]→Evaluation→Commit`, Libet anchor on the Σ-crossing, halted absorbing in 3 cycles, NaN-clean. Reuses the COMPLETE contract kanban surface (`KanbanColumn`/`KanbanMove`/`NextPhaseScheduler`/`MailboxSoa{View,Owner}`) — zero new types. **ractor = ownership guarantee** (no messages, no tokio; E-CE64-MB-4 / #477 "nothing transmitted between mailboxes") — already embodied by `SymbiontBoard`'s single `&mut` owner, NOT a deferred message actor. **Trigger is SYNCHRONOUS — the writer fires it:** `VersionScheduler::on_version(&view, DatasetVersion(u64), exec)` is a sync pure function; a batch writer knows the version it committed and fires the kanban update inline (`on_version`→`try_advance_phase`, no async). `surreal_container/tests/scheduler_seam.rs` drives the whole Rubicon arc with plain `#[test]` feeding `DatasetVersion(i)`; `cognitive-shader-driver` `MailboxSoA` test 11 runs the same in-RAM loop (`mailbox_soa.rs:700` "no surreal / ractor message bus needed"). This loop's `u32` tick IS that pattern. Async is ONLY the Lance write I/O + the SUBSCRIPTION variant `LanceVersionScheduler::drive_once` (async because it READS a version it didn't write; shipped, 5 tokio tests). Only `surreal_container::read_via_kv_lance` is a stub. |
| E1 | Spain-grid acceptance gate (real fixture, NaN-free, clippy+machete clean) | symbiont | Queued | the north star — first N *real* nodes on the SoA in parallel |
| BT | Battle-test plan (probes A1–E3, gated behind singleton-BindSpace→SoA) | workspace | **Shipped (doc)** | `crates/symbiont/BATTLE_TEST_PLAN.md`; A1 partial-green + D1 green; A2–E3 specced |

---

## entropy-ladder-spo-rung-v1 — Staunen↔Wisdom entropy coordinate unifies SPO rungs + NARS reliability (R1 shipped; R2–R6 roadmap)

Plan path: `.claude/plans/entropy-ladder-spo-rung-v1.md`. Foundation: `ndarray::hpc::{reliability, edge_codec, entropy_ladder}`. Selector: `lance-graph-contract::EdgeCodecFlavor`.

| D-id | Title | Crate(s) / repo | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|
| D-EL-1 | Entropy-ladder foundation (reliability + edge_codec + entropy_ladder + EdgeCodecFlavor + bgz17 fix) | ndarray + lance-graph-contract + bgz17 | LOW | **In PR** | `d3b608f`,`83be7c3`,`920671d`,`6d48ced`; ρ=−0.78; ICC 0.97–0.99 |
| D-EL-2 | `entropy_class` → CausalEdge64 spare bits [63:61] | `causal-edge` | MED | **Queued** | version-gated + field-isolation (I-LEGACY-API-FEATURE-GATED) |
| D-EL-3 | CAM-PQ AMX centroid assignment (GEMM + 2×2/4×4 grid) | `ndarray` | MED | **Queued** | bit-exact + GMAC/s probe |
| D-EL-4 | HHTL+helix basin attraction | `lance-graph` + `helix` | MED | **Probe queued** | +15% recall vs HHTL-alone gate |
| D-EL-5 | Markov SPO rung-ladder → episodic context | `deepnsm` / `lance-graph` | MED | **Probe queued** | prune-without-recall-loss gate |
| D-EL-6 | Energy axis / particle↔wave | `lance-graph` MailboxSoA | MED | **Blocked** | gated on Mailbox-SoA map |
| D-EL-COCA | Superposition 2/3 pruning (cluster-identity layer) | `deepnsm` | HIGH | **Design** | I-VSA-IDENTITIES design-gate |

---

## singleton-to-snapshot-nudge-v1 — every shared-mutable singleton → per-owner MailboxSoA + Arc-swap snapshot (7 deliverables; codebooks left as-is)

Plan path: `.claude/plans/singleton-to-snapshot-nudge-v1.md`. Companions: `bindspace-singleton-to-mailbox-soa-v1` (BindSpace dissolution), `cycle-coherent-soa-snapshot-v1` (snapshot mechanism). Debt: TD-UNBUNDLE-FROM-1.

| D-id | Title | Crate(s) / repo | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|
| D-SNGL-1 | Workspace-wide singleton census (codebook vs shared-mutable) | docs/architecture | LOW | **Queued** | audit only; gates classification |
| D-SNGL-2 | Classification gate — "mutated-after-init?" decision procedure | docs/architecture | LOW | **Queued** | gates on D-SNGL-1 |
| D-SNGL-3 | `AttentionMatrix.gestalt` correctness (raw-sum+count or rebuild) | `lance-graph-planner::cache::kv_bundle` | MED | **In progress** | `unbundle_from` deprecated this session (branch `claude/stoic-turing-M0Eiq`); full fix pending |
| D-SNGL-4 | `ndarray/crates/burn` ATTENTION_CACHE / LINEAR_CACHE audit | `ndarray` | LOW | **Queued** | classify JIT-cache vs runtime-belief |
| D-SNGL-5 | `SnapshotProvider` adoption checklist per nudged crate | workspace | LOW | **Queued** | gates on D-SOA-SNAP-1/2 |
| D-SNGL-6 | No-cross-cycle-lag falsification per nudged crate | workspace | MED | **Queued** | reuses D-SOA-SNAP-5 shape |
| D-SNGL-7 | Board hygiene + E-SINGLETON-IS-CODEBOOK-OR-SOA | `.claude/board` | LOW | **In progress** | this entry + INTEGRATION_PLANS prepend |

---

## cesium-osm-substrate-v1 — OpenStreetMap as 6th Cesium ingest source class (7 deliverables; substrate-reuse with splat-native)

Plan path: `.claude/plans/cesium-osm-substrate-v1.md`. Parent: `3DGS-ArcGIS-Cesium-ingestion-plan.md` (structural). Sibling: `splat-native-ultrasound-v1.md` (Gaussian3D carrier reuse). OGAR coordination 2026-06-05 locked Q1/Q2/Q3 rulings. OGAR-side docs PR (DOMAIN-INSTANCES §2.6 + RDF-OWL-ALIGNMENT §10 Phase 2c) queued behind this PR.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Sprint | Status | PR / Evidence |
|---|---|---|---|---|---|---|---|
| D-OSM-1 | `crates/cesium/src/osm_pbf.rs` stub (mirrors `arcgis_pbf.rs` shape; OsmNode/OsmWay/OsmRelation/OsmPbfBlock + OSM-XYZ → TMS Y-flip helper; no osmpbf dep yet) | `ndarray` | 400 | LOW | P1 sprint 1 | **Queued** | foundation; gates nothing upstream |
| D-OSM-2 | osmpbf v0.4 consumer + Arrow RecordBatch emitter → Lance datasets `osm_nodes` / `osm_ways` / `osm_relations` (tags as Q1 v1 fallback `List<Struct<key,value>>`; qk_tms_path per Q2) | `lance-graph` | 600 | MED | P1 sprint 1-2 | **Queued** | gates on D-OSM-1 |
| D-OSM-3 | OSM tag → SPO triple lift (`(Way#123, ogar:hasTag, "building=yes")`); **OGAR-crossing contract** that `ogar-from-osm-pbf` Phase 2c consumes | `lance-graph-ontology` | 200 | LOW | P2 sprint 3 | **Queued** | gates on D-OSM-2 + OGAR readiness signal |
| D-OSM-4 | `ndarray::simd::dem::batched_sample_height` W1c primitive (bilinear interp; all three backends AVX-512/NEON/scalar) | `ndarray` | 300 | MED | P2 sprint 3 | **Queued** | foundation; sibling to D-SPLAT-2 |
| D-OSM-5 | Geospatial splat-fit: OSM footprint × DEM → extruded `Gaussian3D` batch (consumes D-SPLAT-1 carrier + D-SPLAT-3 SoA verbatim — substrate-reuse payoff) | new `crates/splat-fit-geo` OR `splat-fit` `geo` feature | 800 | MED-HIGH | P3 sprint 4-5 | **Queued** | gates on D-OSM-1 + D-OSM-2 + D-OSM-4 + D-SPLAT-1 + D-SPLAT-3 |
| D-OSM-6 | `cesium-3dtiles-writer` crate — b3dm/cmpt/tileset.json emitter (**the genuine Rust gap; first-of-its-kind**); MVP scope, gltf-crate-based | `ndarray` (new `crates/cesium-3dtiles-writer` or `writer` feature on existing `cesium` crate) | 500 | HIGH | P3 sprint 4-5 | **Queued** | gates on D-OSM-5 + D-SPLAT-3 |
| D-OSM-7 | Nominatim sidecar HTTP adapter (UX-edge optional; geocoding/reverse-geocoding via reqwest); response → D-OSM-2 primary path | `lance-graph` or new `crates/nominatim-client` | 150 | LOW | P4 sprint 6+ (optional; ship on UX-edge demand only) | **Queued** | independent path |

---

## splat-native-ultrasound-v1 — CPU-only Gaussian-splat ultrasound SaMD (14 deliverables across ndarray/lance-graph/MedCare-rs/OGAR + new standalone crates)

Plan path: `.claude/plans/splat-native-ultrasound-v1.md`. Companions: ndarray `.claude/plans/splat-native-ultrasound-simd-substrate-v1.md`; OGAR `docs/SPLAT-NATIVE-CUSTOMER.md`; MedCare-rs `.claude/handovers/2026-06-05-splat-native-medcare-hipaa-wire.md`. Customer of OGAR PR #30 §6 FMA bones-rendering litmus + ADR-022 SaMD audit-controls evidence base.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Sprint | Status | PR / Evidence |
|---|---|---|---|---|---|---|---|
| D-SPLAT-1 | `Gaussian3D` carrier (`mu`/`sigma_packed`/`amplitude`/`opacity`/`sh[16]`/`frame_idx`/`class_id`; 80 B/row) | `lance-graph-contract::splat` | 120 | LOW | P1 sprint 1-2 | **Queued** | gates on `MailboxSoAHeader` (D-MBX-10) or own feature flag |
| D-SPLAT-2 | `ndarray::simd::splat` batch ops — `batched_cholesky_3x3` / `batched_mahalanobis` / `batched_opacity_blend` / `batched_sh_eval_l3` / `batched_se3_transform`; all three backends (AVX-512/NEON/scalar) | `ndarray::src/simd_splat.rs` | 600 | MED | P1 sprint 1-2 | **Queued** | foundation; none |
| D-SPLAT-3 | `SplatBatch<N>` SoA carrier (per-column slices for SIMD sweep; inherits MailboxSoAHeader versioning) | `lance-graph-contract::splat` | 150 | LOW | P1 sprint 1-2 | **Queued** | gates on D-SPLAT-1 |
| D-SPLAT-4 | SH-aware palette extension in `crates/bgz17` (256×256×2B compose table; SH-basis-id per centroid) | `bgz17::sh_palette` | 250 | MED | P3 sprint 4-5 | **Queued** | gates on D-SPLAT-1 |
| D-SPLAT-5 | Splat-to-splat registration math — Σ-sandwich Mahalanobis ICP + SE(3) Levenberg-Marquardt | `lance-graph::splat::registration` | 400 | HIGH | P4 sprint 6-7 | **Queued** | gates on D-SPLAT-2 + D-SPLAT-3 |
| D-SPLAT-6 | `crates/splat-fit` engine — RF/IQ → beamformed → local-maxima → PSF estimate → SH projection → emit Gaussian3D batch | `crates/splat-fit` (new standalone, 0-dep, ndarray-hpc feature) | 1500 | HIGH | P2 sprint 3 | **Queued** | gates on D-SPLAT-1 + D-SPLAT-2 + OQ-SPLAT-3 |
| D-SPLAT-7 | Splat actors — `SplatFitActor`/`PoseAccumulatorActor`/`RegistrationActor`, each owns one `MailboxSoA<Gaussian3D>`; consumes bardioc #17 Rubicon kanban verbatim | `crates/splat-actors` (or `ractor_actors`) | 500 | MED | P3 sprint 4-5 | **Queued** | gates on D-SPLAT-3 + D-SPLAT-6 + bardioc #17 (shipped) |
| D-SPLAT-8 | FMA atlas hydrator — TTL → `fma_class.lance` + `fma_relation.lance` + `fma_atlas_splat.lance` (~150M Gaussians full body) | `lance-graph-ontology` + `crates/fma-hydrator` | 800 | HIGH | P4 sprint 7-8 | **Queued** | gates on OGAR PR #30 Phase 8 + D-SPLAT-3 + ndarray PR #189 (shipped) |
| D-SPLAT-9 | `fma_blueprint::style_recipe` D-Atom catalogue (AnatomicalRegion, OrganSystem, Innervation, Vasculature, Joint, Muscle, Bone, OrganParenchyma, Tract); mirrors PR #433 Odoo pattern | `lance-graph-ontology::fma_blueprint` | 400 | LOW | P4 sprint 7-8 | **Queued** | gates on D-SPLAT-8 |
| D-SPLAT-10 | `memory.ultrasound_frame.lance` + `memory.ultrasound_splat.lance` datasets via `soa_mapping.rs`; new `SensitivityReason::UltrasoundRawPHI`/`UltrasoundAnonymized` variants in `column_mask_bridge` | MedCare-rs `crates/medcare-analytics` | 250 | MED | P5 sprint 9-10 | **Queued** | gates on D-SPLAT-3 + MedCare PR #162 (shipped) |
| D-SPLAT-11 | `commit_event` audit chain for splat ingest via `LanceMembrane::commit_event` (callcenter PR #467, sole-writer membrane); `KnowableFromStore::register("ogit-medcare/ultrasound_ingest", Some(ddl_hint))` | MedCare-rs `crates/medcare-analytics` | 100 | LOW | P5 sprint 9-10 | **Queued** | gates on D-SPLAT-10 + PR #467 (shipped) + OGAR #25/#31 (shipped) |
| D-SPLAT-12 | AR splat renderer — HoloLens OpenXR (clinical AR target) + Cesium ion + Three.js (browser fallback) + headless PNG (regression); CPU does math, GPU only paints | `crates/splat-render` (new) | 1200 | HIGH | P6 sprint 11-13 | **Queued** | gates on D-SPLAT-2 + D-SPLAT-3 + D-SPLAT-5 |
| D-SPLAT-13 | IMU/POSE 4D accumulator — VIO against splat features at IMU rate (~200 Hz); splat-corrected pose at frame rate (~30 Hz); Planning-column readiness at t = −550ms | `splat-actors::PoseAccumulatorActor` | 200 | MED | P3 sprint 4-5 | **Queued** | gates on D-SPLAT-7 |
| D-SPLAT-14 | SaMD documentation track — research-tool → clinical-study → Class IIa (IEC 62366 / IEC 80001 / ISO 14971 / MDR Annex VIII Rule 11). ADR-022 firewall IS the audit-controls evidence base | `q2`/`quarto` or `docs/` | 600 | LOW | P7 sprint 14+ (parallel through P4-P6) | **Queued** | gates on none architecturally; v1/v2/v3 phased |

---

# Status Board — Cross-Deliverable View

> Deliverable-level status across all active integration plans.
> **Status** and **PR / Evidence** columns are the only mutable
> fields — title, plan-version, and scope are immutable.
>
> For plan-level status see `INTEGRATION_PLANS.md`.
> For per-PR decision history see `PR_ARC_INVENTORY.md`.
> For current contract inventory see `LATEST_STATE.md`.

---

## D-HELIX-1 — `crates/helix` golden-spiral Place/Residue codec (zero-dep + optional ndarray-hpc)

**Status:** Shipped (branch `claude/gallant-rubin-Y9pQd`; **61 unit + 6 doctests green** on the default zero-dep build AND under `--features ndarray-hpc`; clippy -D warnings + fmt clean). New standalone crate (empty `[workspace]`, root `exclude`) realising the user's `KNOWLEDGE.md`: `HemispherePoint` (√u equal-area placement) → `CurveRuler` (stride-4-over-17) → `Similarity` (Fisher-Z/arctanh) → `RollingFloor` (256-palette; occupancy-drift + version stamp) → `ResidueEdge` (3-byte endpoint pair) + `DistanceLut` (metric-safe 256×256 L1; `distance_adaptive` vs non-metric `distance_heuristic`) + `prove()` (2-D discrepancy companion to `jc::weyl`). Optional `ndarray-hpc` = batch Fisher-Z via `simd_ln_f32`. ~80% clean-room overlap with CERTIFIED primitives (E-HELIX-OVERLAP / TD-HELIX-OVERLAP-1); consolidation path in `KNOWLEDGE.md`. Process: autoattended — 5 research agents + 4 parallel Sonnet leaf workers + central consolidation. Next (owed): fidelity-vs-ground-truth probe (naive-u8 floor gate ≥0.9980 Pearson, CONJECTURE). **Update (post-#460):** ndarray is now a MANDATORY non-optional **git** dep (codex P2 + directive "ndarray is mandatory for lance-graph"); `simd.rs` always uses `ndarray::simd`; `ndarray-hpc` feature removed. 63 unit + 6 doctests green; clippy/fmt clean. See E-HELIX-NDARRAY-MANDATORY.

## D-A3 — I4x32/I4x64 signed-i4 CAM codec (carrier `pack`/`unpack` + the 256-bit wide carrier)

**Status:** Shipped (branch `claude/jolly-cori-clnf9`; contract lib **562 green**, offline). `I4x32::pack`/`unpack` (two's-complement signed-i4 nibble; even→low/odd→high; saturate `[−8,7]`; sign-agnostic) + new `I4x64` (256-bit / 64 signed dims) + private `sext4`. The carrier is a deterministic 32×/64× **CAM address** + sparse-intensity "smell" — NOT a similarity vector (no vector search, no float; the `{instance,reference}` dual REJECTED, "64" = 64 poles). 33 atoms → dims 0..32. Resolved the 3 stale BLOCKED notes. Plan `.claude/plans/a3-carrier-v1.md` (5-research + 3-brutal sandwich). Next: A4 (CAM-address resolver + `is_signed` + `AtomLane`/`LaneMask` newtypes).

## D-EW64-2 — EpisodicEdges64 MRU promote (Hebbian hot-tier "stronger immediate edges")

| D-id | deliverable | status | PR / evidence |
|---|---|---|---|
| D-EW64-2 | `EpisodicEdges64::{promote, strongest}` — MRU slot-order = strength; fire→slot 0, evict coldest (`E-EW64-STRENGTH-IS-CE64-PLASTICITY`) | In PR (claude/jolly-cori-clnf9) | contract lib 533 green (+5); default clippy clean |

## Status Legend

| Status | Meaning |
|---|---|
| **Shipped** | Merged to main. PR column cites the merge commit. |
| **In PR** | PR open, under review. Not yet merged. |
| **In progress** | Active branch, code in flight, not yet PR. |
| **Queued** | Next up; spec is clear; work not started. |
| **Backlog** | Future; still in scope but not yet queued for a phase. |
| **Deferred** | Explicitly parked. Rationale recorded. Will be revisited. |
| **Abandoned** | Removed from scope. Rationale recorded. Will not be revisited. |

Rules:
- New rows APPEND (at the bottom of the relevant section).
- Status field is the ONLY field that gets edited in place.
- When a deliverable ships, record the PR number — never delete the
  row.
- When a deliverable is superseded by a different design, keep the
  row with Status = Abandoned and cite the replacement.

---

## normalized-entity-holy-grail-v1 — typed unified normalization + Op chain

Stage 1 contract surface scaffold. Typed consumer pipeline grammar that
unifies OGIT/OWL/DOLCE/Odoo inheritance + cognitive shader + JIT +
MailboxSoA into one surface. Plan path:
`.claude/plans/normalized-entity-holy-grail-v1.md`.

### Stage 1 deliverables (D-NEH-1a..g)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| **D-NEH-1a** | `cognition::{NormalizedEntity, stages, Op, OpKind, MailboxRow, Output}` typed surface | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1b** | `transaction::{Interactive, Bulk, Periodisch, Context, OgitCtx/OwlCtx/DolceCtx/FibuCtx}` context shapes | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1c** | 5-verb advancement methods on `NormalizedEntity<S>` | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1d** | `CascadeKind` + `TraversalMode` + `CascadeWalker` trait | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1e** | Compile-fail tests + 7 positive typestate tests | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1f** | Crate doc + example chain + `docs/COGNITION_HOLY_GRAIL.md` | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1g** | Board hygiene (AGENT_LOG + STATUS_BOARD) | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |

### Stage 2..7 deliverables (future plans)

| D-id | Title | Status |
|---|---|---|
| D-NEH-2a..z | ~50 Op kernel bodies + shader dispatch wiring | **Backlog** |
| D-NEH-3a..c | Consumer DSL macros (medcare/woa/smb) | **Backlog** |
| D-NEH-4a..b | Stream + GenServer integration | **Backlog** |
| D-NEH-5 | Jahresabrechnung kernel + fiscal-close JIT | **Backlog** |
| D-NEH-6 | palantir-foundry parity audit | **Backlog** |
| D-NEH-7 | elixir-OTP parity audit | **Backlog** |

---

## codec-sweep-via-lab-infra-v1 — JIT-first codec sweep

Active integration plan. 7 Phase 0 deliverables (D0.1–D0.7) + Phases
1–5 queued. One upfront Wire-surface rebuild; every candidate
afterwards is a JIT kernel, not a rebuild. Plan path:
`.claude/plans/codec-sweep-via-lab-infra-v1.md`.

### CI Gate — JC Substrate Proof

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| CI-JC | `.github/workflows/jc-proof.yml` — runs prove_it on every PR touching `crates/jc/` or `cam.rs` | **In PR** | 5-min timeout, exits 0 = substrate sound |

### Phase 0 — API hardening (partial in PR #225; remainder queued)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0.1 | Extend `WireCalibrate` + `WireTensorView` (64-byte-aligned decode, object-oriented methods) | **Shipped** | #227 — 55/55 tests passing |
| D0.2 | `WireTokenAgreement` endpoint stub — I11 cert gate (Phase 0 surface, Phase 2 harness) | **In PR** | branch — `WireTokenAgreement` + `WireTokenAgreementResult` + `WireBaseline` DTOs + 3 round-trip tests. Stub handler returns `stub:true` / `backend:"stub"` until D2.1–D2.3 wire real decode-and-compare. |
| D0.3 | `WireSweep` streaming endpoint + Lance append stub | **In PR** | branch — `WireSweepGrid` + `cardinality()` + `enumerate()` → `Vec<WireCodecParams>` + `WireMeasure` enum + `WireSweepRequest` / `WireSweepResult` / `WireSweepResponse` DTOs + 5 tests. Streaming handler + Lance writer defer to Phase 3 D3.1. |
| D0.4 | Surface freeze (commit + rebuild) | **Ready** | D0.1–D0.7 all Shipped / In PR; freeze fires on merge of this PR. |
| D0.5 | `auto_detect.rs` — `ModelFingerprint` from `config.json` | **In PR** | branch — `auto_detect::{detect, ModelFingerprint, DetectError}` + HF config.json parser + per-architecture lane/distance heuristics (llama/qwen3/bert/modernbert/xlm-roberta/generic) + 8 tests. CODING_PRACTICES gap 1 remediated. |
| D0.6 | `CodecParamsBuilder` fluent API | **Shipped** | #225 — `contract::cam` +290 LOC of codec-params types, 14 tests (CODING_PRACTICES gap 3) |
| D0.7 | Precision-ladder validation (OPQ↔BF16x32, Hadamard pow2, overfit guard) | **Shipped** | #225 — `CodecParamsError` at `.build()` BEFORE JIT compile |

### Phase 1 — JIT codec kernels

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D1.1 | `CodecKernelCache` — structural cache layer (generic over handle) | **In PR** | branch — `CodecKernelCache<H>` + `StubKernel` + `get_or_compile` / `try_get_or_compile` with RwLock concurrent-safe double-check + compile/hit/ratio counters + 9 tests. Scaffold ships NOW; D1.1b Cranelift IR emission follows. |
| D1.1b | Adapter: `CodecKernelEngine` wrapping `ndarray::hpc::jitson_cranelift::JitEngine` with two-phase BUILD/RUN lifecycle (Arc-freeze). CodecParams → CodecScanParams adapter + codec-specific IR emission in jitson_cranelift/scan_jit analog | **Queued** | target ~250 LOC; `JitEngine` already ships (`/home/user/ndarray/src/hpc/jitson_cranelift/engine.rs`); the work is the CodecParams adapter + codec-specific JITSON template |
| D1.2 | Rotation primitives: Identity / Hadamard / OPQ as `RotationKernel` impls | **In PR** | branch — `RotationKernel` trait (Send+Sync+Debug, object-safe) + `IdentityRotation` (no-op) + `HadamardRotation` (real Sylvester butterfly, O(N log N) in-place, norm²-scaling verified) + `OpqRotationStub` (matrix-blob-id placeholder for D1.1b) + `build(&Rotation, dim)` factory + `RotationError` typed errors + 15 tests. Hadamard stays at Tier-3 F32x16 (add/sub, not matmul → no AMX benefit per Rule C). |
| D1.3 | Residual PQ via decode-kernel composition | **In PR** | branch — `DecodeKernel` trait (Send+Sync+Debug, object-safe, encode/decode/signature/bytes_per_row/dim/backend) + `StubDecodeKernel` (byte-exact round-trip for testing) + `ResidualComposer` (base + residual with subtract/add; nests recursively for depth >1) + `DecodeError` typed errors + 9 tests. Scope clarified: hydration/calibration path, NOT cascade inference (cascade uses `p64_bridge::CognitiveShader` per `cognitive-shader-architecture.md` line 582). |

### Phase 2 — Token-agreement harness (I11 cert gate) — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2.1 | Token-agreement harness scaffold (reference model stub + top-k comparator + stub result) | **In PR** | branch — `ReferenceModel::{load, stub}` + `TokenAgreementError` + `TopKAgreement::{compare, top1_rate, top5_rate, meets_cert_gate, aggregate}` + `TokenAgreementHarness::{measure_stub, measure_full}` + 13 tests. Real safetensors load + decode loop defer to D2.2. |
| D2.2 | Decode-and-compare loop (top-k, per-layer MSE) | **Queued** | target ~220 LOC |
| D2.3 | Handler wiring for `/v1/shader/token-agreement` | **In PR** | branch — `token_agreement_handler` routes `WireTokenAgreement` → TryFrom(CodecParams) at ingress (precision-ladder + overfit guard fire here) → `ReferenceModel::load` or stub fallback on nonexistent paths → `TokenAgreementHarness::measure_stub()` → `WireTokenAgreementResult { stub:true }`. Route added: `POST /v1/shader/token-agreement`. Phase 0 Wire + Phase 2 harness now round-trip end-to-end. |

### Phase 3 — Sweep driver + Lance logger — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D3.1 | Server-side sweep handler + Lance fragment append | **In PR** | branch — `sweep_handler` batch mode: enumerates `WireSweepGrid::enumerate()`, validates each via TryFrom(CodecParams) at ingress, returns `WireSweepResponse { results: [WireSweepResult { kernel_hash, stub:true }], cardinality, elapsed_ms }`. SSE streaming + real calibrate/token-agreement per point deferred to D3.1b. Route: `POST /v1/shader/sweep`. |
| D3.2 | Client-side driver + config files | **In PR** | branch — 3 starter YAML configs (`configs/codec/{00_pr220_baseline, 10_wider_codebook, 12_hadamard_pre_rotation}.yaml`), `scripts/codec_sweep.sh` curl wrapper, `configs/codec/README.md`, YAML-shape spec-drift guard test. 118/118 tests pass. |

### Phase 4 — Frontier analysis — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D4.1 | DataFusion SQL over `sweep_results` Lance | **Queued** | target ~80 LOC |
| D4.2 | Pareto frontier notebook | **Queued** | target ~120 LOC |

### Phase 5 — Graduation — Fires per-candidate

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D5  | Graduation to canonical `OrchestrationBridge` (per winner) | **Queued** | target ~120 LOC per graduation; gate: ICC ≥ 0.99 held-out + token-agreement top1 ≥ 0.99 |

---

## elegant-herding-rocket-v1 — Phase-structured

Active integration plan, 12 deliverables D0 + D2–D11 (D1 dropped
early — CausalityFlow extension deferred). Plan path:
`.claude/plans/elegant-herding-rocket-v1.md`.

### Phase 1 — Shipped (PR #210, merged 2026-04-19)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0  | grammar-landscape.md + linguistic-epiphanies + fractal-codec knowledge docs | **Shipped** | #210 — 3 docs, 1151 LOC |
| D4  | ContextChain reasoning ops (coherence / replay / disambiguate / WeightingKernel) | **Shipped** | #210 — 396 LOC, 8 tests |
| D6  | Role-key catalogue with contiguous `[start:stop]` slice addressing | **Shipped** | #210 — 404 LOC, 7 tests |

### Phase 2 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2  | DeepNSM emits `FailureTicket` on low coverage (wiring step 4) | **Queued** | — |
| D3  | Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` | **Queued** | — |
| D5  | Markov ±5 bundler + Trajectory + content_fp (wiring steps 1-3) | **Shipped** | PR #243 — `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). 63 deepnsm tests pass. |
| D7  | Thinking styles + free-energy + RoleKey-as-operator | **Shipped** | PR #243 — `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery_margin (295 LOC added, 14 tests). 175 contract tests pass. |

### Phase 3 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D8  | Story-context bridge: AriGraph commit + global_context + contradiction (wiring steps 5-6) | **Queued** | — |
| D10 | Forward-validation harness (Animal Farm: chapter-10 > chapter-1 accuracy = AGI test) | **Queued** | — |

### Phase 4 — Backlog

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D9  | ONNX story-arc export + ArcPressure / ArcDerivative awareness hook | **Backlog** | — |
| D11 | Bundle-perturb emergence interface (transformer-free generative stack) | **Backlog** | — |

### Dropped / Deferred from the plan itself

| D-id | Title | Status | Notes |
|---|---|---|---|
| D1  | CausalityFlow 3→9 slot extension (modal/local/instrument + beneficiary/goal/source) | **Deferred** | User decision; follow-up PR after Phase 2 |

---

## Infrastructure / governance (not in elegant-herding-rocket)

Workspace-level bootstrap work. Tracked here rather than PR_ARC
because it's process, not architecture.

| Item | Status | PR / Evidence |
|---|---|---|
| CLAUDE.md §Session Start — three mandatory reads | **Shipped** | #211 |
| CLAUDE.md §A2A Orchestration — two layers (runtime + session) | **Shipped** | #211 |
| CLAUDE.md §Model Policy — grindwork vs accumulation + never Haiku | **Shipped** | #211 |
| CLAUDE.md §GitHub Access Policy — zipball-for-reads | **Shipped** | #211 |
| `.claude/BOOT.md` session entry + prior-art links | **Shipped** | #211 |
| `.claude/agents/BOOT.md` orchestration spec (renamed from README) | **Shipped** | #211 |
| `.claude/agents/README.md` function inventory | **Shipped** | #211 |
| `.claude/board/LATEST_STATE.md` current-state snapshot | **Shipped** | #211 |
| `.claude/board/PR_ARC_INVENTORY.md` append-only decision arc | **Shipped** | #211 |
| `.claude/board/INTEGRATION_PLANS.md` versioned plan index | **Shipped** | #211 |
| `.claude/board/STATUS_BOARD.md` this file | **Shipped** | #211 |
| `.claude/settings.json` team-shared governance (ask/deny + hooks) | **Shipped** | #211 |
| `.claude/hooks/session-start.sh` + `post-compact.sh` | **Shipped** | #211 |
| `.claude/skills/cca2a/` pattern-explanation skill | **Shipped** | #211 |
| `.claude/plans/elegant-herding-rocket-v1.md` plan in workspace | **Shipped** | #211 |

## Infrastructure — queued

| Item | Status | Notes |
|---|---|---|
| `.claude/rules/` with `paths:` frontmatter | **Backlog** | Audit rec 2; replace / complement `READ BY:` headers with path-scoped loading |
| Skill `context: fork` + `agent:` field | **Backlog** | Audit rec 4; read-only isolation for search-only skill variants |
| Auto memory (`~/.claude/projects/<proj>/memory/`) | **Backlog** | Audit rec; unstructured addition to curated LATEST_STATE |

---

## Cross-cutting research threads (orthogonal to grammar work)

Separate research thread — not entangled with grammar/crystal/A2A.
Tracked here so it doesn't get lost.

| Item | Status | Notes |
|---|---|---|
| Named-Entity pre-pass (NER) — biggest OSINT blocker | **Deferred** | Dedicated PR after Phase 2 |
| FP_WORDS = 160 migration (currently 157) | **Deferred** | Needs coordinated ndarray change |
| Crystal4K 41:1 persistence compression | **Deferred** | ladybug-rs owns it; would port later |
| 200–500 YAML TEKAMOLO templates per language | **Deferred** | Training pipeline; future |
| Cross-linguistic active parsers (EN+FI+RU+TR) | **Deferred** | Role keys exist; parsers later |
| Fractal-descriptor leaf codec (MFDFA on Hadamard) | **Research** | `.claude/knowledge/fractal-codec-argmax-regime.md`. 30-min probe first. |
| UK Biobank cardiac MRI benchmark | **Research** | Downstream of fractal-codec probe |
| Chess vertical (ruci + lichess-bot integration) | **Deferred** | Capstone Tier 0, parallel stream |
| Wikidata ingest (1.2 B triples → 14.4 GB) | **Deferred** | `.claude/knowledge/wikidata-spo-nars-at-scale.md` |
| OSINT pipeline (spider + reader-lm + DeepNSM) | **Deferred** | `.claude/knowledge/osint-pipeline-openclaw.md` |
| Python/TypeScript grammar-stack convergence | **Deferred** | `.claude/knowledge/grammar-landscape.md` §7 |

---

## Prior-art audit (61 + 41 = 102 existing docs)

Before this session, the workspace accumulated 61 `.claude/*.md`
top-level docs + 41 `.claude/prompts/*.md` files across prior
sessions. They are indexed in `.claude/BOOT.md §Existing content`
and `CLAUDE.md §Prior art`, but their individual **status** (still
active / superseded / archival) has not been audited.

Status rows per bucket, not per file (102 rows would drown the
board — use filesystem + INTEGRATION_PLANS + PR_ARC for per-file
history):

| Bucket | Count | Status | Notes |
|---|---|---|---|
| `.claude/*.md` top-level calibration reports / handovers / audits / snapshots | 61 | **Indexed** | Pointed at from BOOT.md + CLAUDE.md. Per-file active/superseded status: **Backlog** (needs one-pass audit). |
| `.claude/prompts/*.md` scoped session / probe / handover prompts | 41 | **Indexed** | Pointed at from BOOT.md via `SCOPED_PROMPTS.md` index. Per-file status: **Backlog**. |
| `.claude/knowledge/*.md` structured knowledge | 12 | **Active** | Current; each has `READ BY:` header; used by Knowledge Activation triggers. |
| `.claude/agents/*.md` specialist + meta-agent cards | 24 | **Active** | Current; used by spawning + Knowledge Activation. |
| `.claude/hooks/*.sh` | 2 | **Active** | Wired via settings.json. |
| `.claude/skills/cca2a/*.md` | 3 | **Active** | Current. |
| `.claude/plans/*.md` integration plans | 1 (v1) | **Active** | Elegant herding rocket v1, Phase 1 shipped. |

**Backlog item — prior-art audit.** One-pass sweep across the
61+41 files. Per file: label as active / superseded / archival
with a one-line note. Deliverable = an `ARCHIVE_INDEX.md` that
splits the 102 into current vs historical, plus rename/move of
superseded files into an `archive/` subdirectory. Estimate ~200
LOC of meta work, ~2 hours of reading. **Not urgent**; useful
before the next major planning session.

---

## ADR 0001 — Archetype transcode + Lance/DataFusion stack + Persona 16^32

Three-decision architectural lock, accepted 2026-04-24. First ADR in the
workspace. Path: `.claude/adr/0001-archetype-transcode-stack.md`.

| Decision | Status | Mutability |
|---|---|---|
| **D1 — Archetype is TRANSCODED, not bridged** | **Accepted** | Immutable (unlocking requires new ADR) |
| **D2 — Stack lock** (Lance + DataFusion + Supabase-shape scheduler + Arrow temporal; Polars rejected; Ballista deferred to 1s-P99) | **Accepted** | Ballista threshold mutable; rest immutable |
| **D3 — Persona 16^32 is THE identity space** (56-bit PersonaSignature; atom vector BBB-banned) | **Accepted** | Immutable; shared-DTO unification OPEN for future ADRs |

**Follow-up items tracked** (per ADR implications):

| Item | Priority | Location |
|---|---|---|
| DU-2 clarification (rename "bridge" → "transcode") | P2 | `unified-integration-v1.md` DU-2 |
| First `lance-graph-archetype` skeleton crate | P1 (when deliverable lands) | — |
| Grok gRPC A2A expert adapter | P2 | `TECH_DEBT.md` 2026-04-24 |
| Enrichment-shape follow-up ADR | P2 | `TECH_DEBT.md` 2026-04-24 |
| Ballista threshold tuning (post-benchmark amend) | P3 | `TECH_DEBT.md` 2026-04-24 |

Merged via PR #249 (2026-04-24).

---

## callcenter-membrane-v1 — Supabase-shape over Lance + DataFusion

External callcenter membrane crate. BBB enforced by Arrow type system at
compile time. Plan: `.claude/plans/callcenter-membrane-v1.md`. **Validated
by ADR 0001 Decision 2** (DM-4 `LanceVersionWatcher` + DM-6 `DrainTask`
pattern IS the Supabase-shape transcode approach).

### DM-0 / DM-1 — Shipped in this session

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-0 | `ExternalMembrane` trait + `CommitFilter` in `lance-graph-contract/src/external_membrane.rs` | **Shipped** | session 2026-04-22 — `pub mod external_membrane` added to contract lib.rs |
| DM-1 | `lance-graph-callcenter` crate skeleton: `Cargo.toml` (feature gates) + `src/lib.rs` (stub + UNKNOWN markers) | **Shipped** | session 2026-04-22 — added to workspace members |

### DM-2 through DM-9 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-2 | `LanceMembrane: ExternalMembrane` impl with `project()` + compile-time BBB leak test | **In progress** | Phase A shipped `9a8d6a0` — `LanceMembrane` struct + `project()` + `ingest()` + `subscribe()` stub. Phase B: full Lance append + version counter pending DM-4. |
| DM-3 | `CommitFilter` → DataFusion `Expr` translator (`[query]` feature) | **Queued** | — |
| DM-4 | `LanceVersionWatcher` — tails Lance version counter, emits Phoenix `postgres_changes` (`[realtime]`) | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-4a/b/c: `version_watcher.rs` (117 LOC, 4 tests), `lib.rs` `pub mod version_watcher`, `LanceMembrane::watcher` field + `project()` calls `bump()`, `subscribe()` returns `watch::Receiver<CognitiveEventRow>`. |
| DM-5 | `PhoenixServer` — minimal WS server, Phoenix channel subset (`[realtime]`) | **Queued** | Resolve UNKNOWN-2 (which consumers need Phoenix wire?) first |
| DM-6 | `DrainTask` — `steering_intent` Lance read → `UnifiedStep` → `OrchestrationBridge::route()` | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-6a/b scaffold: `drain.rs` (89 LOC, 2 tests), `lib.rs` `pub mod drain`, `Poll::Pending` until follow-up PR wires real drain loop. |
| DM-7 | `JwtMiddleware` + `ActorContext` → `LogicalPlan` RLS rewriter (`[auth]`) | **Queued** | Resolve UNKNOWN-3 (pgwire?) + UNKNOWN-4 (actor_id type) first |
| DM-8 | `PostgRestHandler` — query-string → DataFusion SQL → Lance scan → Arrow response (`[serve]`) | **Queued** | Confirm PostgREST compat needed (§ 8 stop point 4) before building |
| DM-9 | End-to-end test: shader fires → `LanceMembrane::project()` → Lance append → Phoenix subscriber receives event | **Queued** | Depends on DM-2 through DM-6 |

---

## grammar-foundry-followup-v1 — Wire stubs to existing tissue

Plan: `.claude/plans/grammar-foundry-followup-v1.md`. Session 2026-04-29.
Six explicit stubs in PRs #275-#283 + 1 keystone (LF-12 Pipeline DAG). 13 PRs total in 3 waves.

### Wave 1 — no deps (parallel)

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-S1 | LF-12 Pipeline DAG: `UnifiedStep.depends_on` + topological executor | **Queued** | Keystone. Unblocks F4, G2, G6 |
| PR-F1 | PolicyRewriter UDF wrap: `RedactionMode` executors (closes `policy.rs:122`) | **Queued** | Unblocks F2, F5 |
| PR-F3 | Audit log Lance-backed writer (closes `lib.rs:100`) | **Queued** | |
| PR-F6 | `dn_path.rs` real scent via CAM-PQ (closes `dn_path.rs:53`) | **Queued** | Risk: bgz-tensor dep |
| PR-G1 | Triangle bridge real Causality footprint (closes `triangle_bridge.rs:90,221`) | **Queued** | |
| PR-G3 | ContextChain real `Binary16K` fingerprint (closes `context_chain.rs:345`) | **Queued** | |
| PR-G4 | verb_table seed 10/12 families (closes empty `default_table()` rows) | **Queued** | |
| PR-G5 | AriGraph episodic unbundle/rebundle (per `integration-plan-grammar-crystal-arigraph.md`) | **Queued** | |

### Wave 2 — depends on Wave 1

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-F2 | RowEncryption + DifferentialPrivacy executors (closes `policy.rs:147,181`) | **Queued** | After F1; needs key-mgmt ADR |
| PR-F4 | PostgREST → DataFusion dispatch (closes `EchoHandler` stub) | **Queued** | After S1 |
| PR-F5 | `audit_from_plan()` helper (closes `orchestration.rs:202` `unimplemented!`) | **Queued** | After F1 |
| PR-G2 | Disambiguator wiring at parser boundary + FailureTicket emission | **Queued** | After S1 |

### Wave 3 — depends on Waves 1+2

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-G6 | Animal Farm harness real run (D10 from PR #243) | **Queued** | After G1+G2+G3; text licensing needed |

---

## unified-integration-v1 — PersonaHub × ONNX × Archetype × MM-CoT × RoleDB

Plan: `.claude/plans/unified-integration-v1.md`. Session 2026-04-23.

| D-id | Title | Status | Notes |
|---|---|---|---|
| DU-0 | PersonaHub 56-bit compression: `(atom_bitset: u32, palette_weight: u8, template_id: u16)` offline extraction from 370M HF parquet rows | **Queued** | Runs offline; no code deps. Output: `personas.bin` + `sigs_dedup.bin` + `templates/*.yaml` |
| DU-1 | ONNX persona classifier @ L4/L5 — 288-class `(ExternalRole × ThinkingStyle)` product prediction; `style_oracle: Option<&OnnxPersonaClassifier>` in Think struct | **Queued** | Needs ~10K labeled cycles from Lance internal_cold (DM-2 must ship first); replaces Chronos proposal |
| DU-2 | Archetype ECS bridge crate `lance-graph-archetype-bridge` — `ArchetypeWorld → Blackboard`, `ArchetypeTick → UnifiedStep`, `project() → DataFrame component` adapters | **Queued** | Needs DM-2 (ExternalMembrane impl) before adapter can be built |
| DU-3 | RoleDB DataFusion VSA UDFs: `unbind`, `bundle`, `hamming_dist`, `braid_at`, `top_k` — registers in DataFusion session | **Queued** | Fingerprint column type decision needed first (FixedSizeBinary vs FixedSizeList); see open question in plan § 5 |
| DU-4 | MM-CoT stage split: add `rationale_phase: bool` to `CognitiveEventRow`; surface `FacultyDescriptor.is_asymmetric()` in projected RecordBatch | **Shipped** (Phase A: 2026-04-23 `a05979e`; Phase B: 2026-04-24) | Phase A: field exists. Phase B: `set_faculty_context()` on `LanceMembrane` wires `rationale_phase` from `AtomicBool`; orchestration layer calls it with `FacultyDescriptor::is_asymmetric()` + stage. Column is live, not ghost. |
| DU-5 | Board hygiene: DU-0 through DU-4 registered; INTEGRATION_PLANS.md + LATEST_STATE.md updated | **Shipped** (2026-04-23, commit `a05979e`) | Plan corrections + precision-tier §18 + father-grandfather concept committed in follow-up. |

## splat-osint-ingestion-v1 — Splat contract + EWA OSINT bridge

Active plan, 7 deliverables (D-SPLAT-1..7) staged across 6 PRs of the
`gaussian-splat-cam-plane-workaround.md` doc-sequence. PR 1+2 in flight
on branch `claude/splat-osint-ingestion`.
Plan path: `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`.

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-SPLAT-1 | `crates/lance-graph-contract/src/splat.rs` — `SplatChannel`, `CamPlaneSplat`, `SplatPlaneSet`, `AwarenessPlane16K`, `CamSplatCertificate`, `SplatDecision`, `TriadicProjection`, `ReasoningWitness64` + 10 unit tests | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-2 | `crates/jc/examples/osint_edge_traversal.rs` — EWA-Sandwich Σ-push-forward demo for OSINT 5-hop chain, side-by-side vs naive convolution | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-3 | `witness_to_splat()` deterministic conversion (PR 2 of doc-sequence) | **In PR** | branch `claude/phase-3b-witness-to-splat` |
| D-SPLAT-4 | Splat deposition into BindSpace columns via `MergeMode::AlphaFrontToBack` lanes (PR 3 of doc-sequence) | **Queued** | — |
| D-SPLAT-5 | `PlanarSplatBundle4096` with local/short/medium/long bands (PR 4 of doc-sequence) | **Queued** | — |
| D-SPLAT-6 | Semantic-CAM-distance integration — survivor tile selection vs splatted pressure planes (PR 5 of doc-sequence) | **Queued** | — |
| D-SPLAT-7 | Replay fallback — exact 4096-cycle ThoughtCycleSoA replay slice when certificate insufficient (PR 6 of doc-sequence) | **Queued** | — |

Cross-ref: SPLAT-1 row in `ARCHITECTURE_ENTROPY_LEDGER.md` (Aspirational → Wired stage 1, entropy 4 → 2).

---


## causaledge64-mailbox-rename-soa-v1 — sprint-10 spec corpus + sprint-11 impl queue

Active integration plan. Specs shipped via PR #372 (merged 2026-05-14, governance-only).
Plan path: `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`.

### Sprint-10 — spec sprint (12 CCA2A workers + Opus meta) — Shipped

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CE64-MB-1 | par-tile crate apex + Mailbox<T> + 3 backings + AttentionMask SoA + BindSpaceView | **Spec shipped** | #372 — `pr-ce64-mb-1-par-tile-crate.md` (W1) |
| D-CE64-MB-2 | CausalEdge64 v2 layout proposal + OQ-LAYOUT-1 BLOCKER finding | **Spec shipped** | #372 — `pr-ce64-mb-2-causaledge64-v2.md` (W2) |
| D-CE64-MB-2-regress | PAL8 / NARS regression tests (accessor-based, post-OQ-LAYOUT-1) | **Spec shipped** | #372 — `pr-ce64-mb-2-pal8-nars-regression.md` (W3) |
| D-CE64-MB-3 | BindSpace E/F/G/H column extension | **Spec shipped** | #372 — `pr-ce64-mb-3-bindspace-efgh.md` (W4) |
| D-CE64-MB-4 | AriGraph SPO-G + ghost edges + SpoWitnessChain + SCHEMA_VERSION 2→3 | **Spec shipped** | #372 — `pr-ce64-mb-4-arigraph-spo-g.md` (W5) |
| D-CE64-MB-5 | MailboxSoA<N> + AttentionMaskActor (single tick per cycle) | **Spec shipped** | #372 — `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) |
| D-CE64-MB-6 | SigmaTierRouter + banding + INT4-32D cold-start + Hebbian plasticity + KernelHandle cache + Σ9-10 escalation | **Spec shipped** | #372 — `pr-ce64-mb-6-sigma-tier-router.md` (W7) |
| D-CE64-MB-7 | bevy 0.14 cull plugin proof-PR | **Spec shipped** | #372 — `pr-ce64-mb-7-bevy-cull-plugin.md` (W9) |
| D-NDARRAY-MIRI-COMPLETE | Miri coverage ~760 → ~1550 | **Spec shipped** | #372 — `pr-ndarray-miri-complete.md` (W8) |
| D-SPRINT-10-DEPGRAPH | 8 PRs × 6 waves + parallel-landability + cross-spec consistency checks | **Spec shipped** | #372 — `sprint-10-pr-dep-graph.md` (W10) |
| D-SPRINT-10-TESTPLAN | Unified test plan + Miri growth target + proptest Miri runtime | **Spec shipped** | #372 — `sprint-10-test-plan.md` (W11) |
| D-SPRINT-10-EXECPLAN | Sprint-11 fleet definition + post-merge governance + worker prompt template | **Spec shipped** | #372 — `sprint-10-execution-plan.md` (W12) |
| D-SPRINT-10-META | Opus meta-review (CSI-1..6 + E-META-1..5 + sprint-11 gate decision) | **Shipped** | #372 — `.claude/board/sprint-log-10/meta-review.md` |

### Sprint-11 — implementation wave — Queued (blocked)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CE64-MB-1-impl | par-tile crate impl (W1 → code) | **Queued** | blocked on OQ-5 user ratification (rayon vendor) |
| D-CE64-MB-2-impl | CausalEdge64 v2 layout impl (W2 → code) | **Queued** | blocked on CSI-1 user ratification (which Option A/B/C/D/E for bit reclaim) |
| D-CE64-MB-2-regress-impl | PAL8 / NARS regression test impl (W3 → code) | **Queued** | blocked on D-CE64-MB-2-impl |
| D-CE64-MB-3-impl | BindSpace E/F/G/H impl (W4 → code) | **Queued** | blocked on D-CE64-MB-1-impl |
| D-CE64-MB-4-impl | AriGraph SPO-G + ghosts impl (W5 → code) | **Queued** | blocked on D-CE64-MB-2-impl |
| D-CE64-MB-5-impl | MailboxSoA + AttentionMaskActor impl (W6 → code) | **Queued** | blocked on OQ-3 user ratification (plasticity granularity) + CSI-2 spec patch (g_slot_at_drop field) |
| D-CE64-MB-6-impl | SigmaTierRouter impl (W7 → code) | **Queued** | blocked on OQ-1 user ratification (Σ4-Σ5 banding) + CSI-3 spec patch (PR-J1 Wave 0.5 prerequisite) |
| D-CE64-MB-7-impl | bevy cull plugin impl (W9 → code) | **Queued** | blocked on D-CE64-MB-1-impl + CSI-4 spec patch (BindSpaceView::empty_static() ctor in W1) |
| D-NDARRAY-MIRI-COMPLETE-impl | Miri coverage impl (W8 → code) | **Queued** | independent; can spawn first |
| D-PR-J1-INT4-32D-ATOMS | INT4-32D codebook for SigmaTierRouter cold-start | **Queued** | new Wave 0.5 prerequisite; not in original W10 dep graph |
| D-CSI-2 | W6 CompartmentReport `g_slot_at_drop: u8` field patch | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-3 | W10 dep graph PR-J1 Wave 0.5 row patch | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-4 | W1 spec `BindSpaceView::empty_static()` + `from_arc()` constructors | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-5 | W1 spec move `SigmaTier` to `lance-graph-contract::orchestration` | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-6 | W11 test-count drift reconciliation | **Queued** | small spec edit; pre-sprint-11 |

### User-ratification gates (block sprint-11 spawn)

| Gate | Wave blocked | Resolution path |
|---|---|---|
| **CSI-1** — CausalEdge64 bit-reclaim Option (A/B/C/D/E) | Wave 2 (D-CE64-MB-2-impl) | User picks; meta-review recommends Option C-conservative (drop temporal + G-slot, allocate W-slot + lens) |
| **OQ-1** — Σ4-Σ5 banding (Tokio reflex vs InMem cycle-speed) | Wave 5 (D-CE64-MB-6-impl) | Default Tokio is safe-to-ship; ratification only PROMOTES |
| **OQ-3** — Plasticity update granularity (bit-counter per emission + NARS revise at AriGraph commit) | Wave 4 (D-CE64-MB-5-impl) | Tentative resolution recorded; user formal-acknowledge |
| **OQ-5** — Rayon vendor decision (std::thread::scope first vs vendored-rayon) | Wave 1 (D-CE64-MB-1-impl) | Tentative defer; user formal-acknowledge |

### Reunification track (sprint-12+)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-REUNIFY-1 | Acknowledge dual `CausalEdge64` types in TYPE_DUPLICATION_MAP + LATEST_STATE + EPIPHANIES | **Shipped** | this commit (post-merge #372 board-hygiene tail) |
| D-REUNIFY-2 | 8-channel → SPO transcoder spec at thinking-engine L3 commit boundary | **Backlog** | per Option R-3; sprint-12+ |
| D-REUNIFY-3 | `Think` carrier struct prototype unifying thinking-engine cascade + cognitive-shader-driver SoA | **Backlog** | per `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` sprint-12 |
| D-REUNIFY-4 | Splat op fleet (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`) as methods on `Think` | **Backlog** | sprint-13+ |
| D-REUNIFY-5 | rayon work-stealing par_* method variants | **Backlog** | sprint-14+ |
| D-REUNIFY-6 | OWL DOLCE / OntologyFilter wiring into `emit_causal_edges_filtered` | **Backlog** | sprint-15+ |

---

## cognitive-substrate-convergence-v1 — i4 mantissa + gapless baton + active inference

Active integration plan. Authored 2026-05-15 (cross-session A2A discussion).
Plan path: `.claude/plans/cognitive-substrate-convergence-v1.md`.
Consolidates sprint-10 architectural decisions before context dilution.

### Phase A — Substrate primitives (sprint-11)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-1 | `causal-edge` crate v2 layout (signed mantissa, W-slot, lens, drop temporal) | **Shipped** | PR #383 merge `03bd175`; OQ-CSV-2 ratified to 6 bits (default) |
| D-CSV-2 | `QualiaI4_16D` type in `lance-graph-contract::qualia` + f32↔i4 migration helpers | **Shipped** | PR #384 merge `0751a8b`; OQ-CSV-1 ratified to Option α (canonical convergence-observable vocab; drop dim 16 "integration") |
| D-CSV-3 | InferenceType signed-mantissa expansion (absorbs PR-LL-1 Intervention/Counterfactual into canonical edge enum) | **Shipped** | PR #383 merge `03bd175`, paired with D-CSV-1 in same crate |
| D-CSV-4 | `CollapseGateEmission` wire format spec + impl per plan §8 | **Shipped** | PR #383 merge `03bd175`, contract crate (Vec instead of SmallVec to preserve zero-dep — TD-COLLAPSE-GATE-SMALLVEC-1) |

### Phase B — Storage & dispatch path (sprint-11)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-5a | QualiaColumn migration phase 5a — sibling `QualiaI4Column` add + double-write (no read-side change) | **Shipped** | PR #385 merge `6f58418`; OQ-CSV-4 ratified to sibling-cutover (default); 5b cutover follows in separate PR |
| D-CSV-5b | QualiaColumn migration phase 5b — flip readers to i4, drop f32 column, drop f32 push arg | **In PR (#390 W-G1)** | sprint-12 Wave G fleet; depends on D-CSV-5a (merged) + downstream reader audit |
| D-CSV-6a | `WitnessCorpus` partial (W-slot anchor + chain invariant; sorted by emission cycle, drop-oldest truncation) | **Shipped** | PR #386 merge `33110c8` (paired with D-CSV-7) |
| D-CSV-6b | `WitnessCorpus` full (CAM-PQ-indexed, unbounded, salience decay) | **In PR (#390 W-G2)** | sprint-12 Wave G fleet; depends on D-CSV-6a (merged) |
| D-CSV-7 | MailboxSoA integration: W-slot referencing + per-row plasticity accumulator + apply_edges | **Shipped** | PR #386 merge `33110c8` (paired with D-CSV-6a) |

### Phase C — Reasoning path (sprint-12)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-8 | MUL evaluation in integer SIMD: DK/TrustTexture/FlowState/GateDecision consume i4 qualia + signed mantissa | **Shipped** | PR #387 merge `e042c70` (scalar i4 path; AVX-512/NEON deferred → D-CSV-13/13b sprint-13 per TD-D-CSV-8-SIMD-1) |
| D-CSV-9 | 8-channel ↔ SPO-palette transcoder (Option R-3) at thinking-engine L3 commit boundary | **Shipped** | PR #387 merge `e042c70` (paired with D-CSV-8) |
| D-CSV-10 | Σ-tier Rubicon-resonance dispatch in `SigmaTierRouter`: ΔF + resonance threshold → Σ10 commit | **In PR (#388 W-F1)** | sprint-12 Wave F; sigma-tier-router crate present in workspace post-Wave G #390 cargo metadata (hand-tuned threshold per OQ-CSV-6; Jirak-derived → D-CSV-15 sprint-13+) |

### Phase D — Streaming infrastructure (sprint-12 productization)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-11a | Vertical streaming structs in ndarray: `QualiaStream` / `QualiaI4Row` | **Shipped** | ndarray PR #147 merge `d867b1c` |
| D-CSV-11b | Vertical streaming structs in ndarray: `InferenceStream` / `InferenceRow` | **Shipped** | ndarray PR #147 merge `d867b1c` |
| D-CSV-11c | Vertical streaming structs in ndarray: `SplatFieldStream` (+ `par_*` rayon variants deferred to sprint-14+ behind `parallel` feature) | **Shipped** | ndarray PR #147 merge `d867b1c`; `par_*` rayon variants deferred (Queued sprint-14+) |
| D-CSV-12 | Splat shader op fleet (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`) — scalar standalone ops | **Shipped** | PR #388 merge `77f2d26` (W-F7 scalar; on-Think method migration → D-CSV-14 sprint-13) |

### Phase E — Sprint-12/13 new entries (NEW in v2 + sprint-13 preflight)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-13 | Batch i4 scalar MUL (paired with D-CSV-8 SIMD-readiness) | **Shipped** | PR #388 merge `77f2d26` (W-G3 batch i4 scalar) |
| D-CSV-13b | SIMD vectorization of D-CSV-8 i4 MUL evaluation (AVX-512 + NEON intrinsics) | **In PR (sprint-13/W-I1 salvage)** | branch `claude/sprint-13-w-i1-salvage`; AVX-512F+BW dispatch via `simd_caps()`; bench on Skylake-AVX512 host = 8.7× dk / 7.4× trust / 5.2× flow / 10.2× gate_disc / 3.1× mul_assess at batch 1024 — all SHIP gates met; 5 SIMD-vs-scalar parity tests over 10 sizes green |
| D-CSV-14 | On-Think method migration for D-CSV-12 splat ops (struct-method surface per L-20) | **Queued (PP-4 spec drafting)** | sprint-13; depends on D-CSV-11 streaming substrate (shipped via ndarray #147) |
| D-CSV-15 | Σ10 Jirak-derived threshold (TD-SIGMA-TIER-THRESHOLDS-1 resolution) | **In PR (#390 W-G4 Jirak threshold)** | sprint-12 Wave G partial; full VAMPE coupled-revival deferred sprint-13+ |
| D-CSV-16 | NEW sprint-13 entry | **Queued (PP-5 spec drafting)** | sprint-13 preflight |
| D-CSV-17 | NEW sprint-13 entry | **Queued (PP-3 spec drafting)** | sprint-13 preflight |

### Open-question gates (block specific D-CSV-* spawns)

| Gate | Blocks | Recommendation |
|---|---|---|
| **OQ-CSV-1** Qualia 16D per-dim assignment | D-CSV-2, D-CSV-5 | Ratify proposed §7.2 layout with `qualia-engineer` agent cross-check against `thinking-engine/src/qualia.rs` |
| **OQ-CSV-2** W-slot width 6 vs 8 bits | D-CSV-1 | Default 6 (= 64 active corpora); promote to 8 if multi-tenant SaaS demands |
| **OQ-CSV-4** QualiaColumn migration phasing | D-CSV-5 | Default sibling-column-then-cutover (lower risk; 1 extra PR worth it) |
| **OQ-CSV-6** Σ10 Rubicon threshold derivation | D-CSV-10 (sprint-12) | Hand-tuned acceptable for sprint-11/12 with TECH_DEBT note per `I-NOISE-FLOOR-JIRAK`; principled Jirak derivation deferred to VAMPE coupled-revival sprint-13+ |

### Cross-spec patches (one bundled prep PR pre-sprint-11) — **SHIPPED via PR #381 (merged 2026-05-16, commit `a7c0545`)**

| Spec | Patch | LOC | Status |
|---|---|---|---|
| `pr-ce64-mb-2-causaledge64-v2.md` (W2) | §3 bit layout → plan §6; OQ-LAYOUT-1 resolved; signed-mantissa rationale; G-slot API stripped from test plan + risk matrix (codex P1) | ~160 actual | **Shipped** |
| `pr-ce64-mb-2-pal8-nars-regression.md` (W3) | Tests parameterized on v2 layout; mantissa roundtrip + lens 4-state; v1-temporal=0 safe-migration fix + version-gate test (codex P1) | ~370 actual | **Shipped** |
| `pr-ce64-mb-3-bindspace-efgh.md` (W4) | QualiaColumn migration step (D-CSV-5) cross-ref | ~40 actual | **Shipped** |
| `pr-ce64-mb-4-arigraph-spo-g.md` (W5) | `SpoWitnessChain<32>` → `WitnessCorpus`; `W5-INV-CHAIN-ORDER` invariant; W-slot semantics | ~316 actual | **Shipped** |
| `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) | `g_slot_at_drop` field (CSI-2); spatial-temporal accumulator semantics | ~50 actual | **Shipped** |
| `pr-ce64-mb-6-sigma-tier-router.md` (W7) | Σ10 Rubicon-resonance threshold; integer-SIMD MUL path | ~120 actual | **Shipped** |
| `sprint-10-pr-dep-graph.md` (W10) | PR-J1-INT4-32D-ATOMS + CAM-PQ wiring elevated to Wave 3 hard dep | ~50 actual | **Shipped** |
| `sprint-10-test-plan.md` (W11) | Refresh test counts for v2; i4-roundtrip + signed-mantissa-product tests | ~87 actual | **Shipped** |

**Total spec-patch LOC:** ~1,200 actual across 5 commits (`9bd66d9`, `f730528`, `5253c79`, `e4d15a3`, `33509ab`) merged 2026-05-16 in PR #381. Original ~870 estimate undershot W3 (codex P1 fix added ~280 LOC) and W5 (full WitnessCorpus section added ~16 LOC over estimate). All 8 workers complete. Sprint-11 spawn now unblocked on the spec-patch dimension; remaining gates: OQ-CSV-1, OQ-CSV-2, OQ-CSV-4 user ratifications.

---

## rung-persona-orchestration-v1 — time-bound persona orchestration (checklist → meta-recipe → hot/cold/feedback anneal)

Active proposal. Authored 2026-05-26. Plan path:
`.claude/plans/rung-persona-orchestration-v1.md`. Sibling/time-bound
composition layer over `rung-mul-grounding-v1`. Grounds ladybug's
hot/cold/feedback loop onto our contract types + SoA floor
(restore-on-SoA, not port). Epiphany: `E-RIGID-RULES-OPEN-DOORS`.

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-PERSONA-1 | escalation+epiphany loop = the checklist (`felt_parse` collapse-hint + `InnerCouncil`/`HdrResonance` split + `EpiphanyDetector`; green-flip = Epiphany/Wisdom ghost) — NOT a bespoke verifier | contract + planner | 160 | LOW | **In progress** | branch `claude/splat3d-cpu-simd-renderer-MAOO0` |
| D-PERSONA-2 | meta-recipe manifest (declarative child-spec, recipe-as-data, macro-evaluable) | contract | 150 | MED | **Queued** | — |
| D-PERSONA-3 | hot/cold/feedback wiring — anneal + `CrystalCodebook`→wisdom-marker cold path + Preload hydrate | planner + Lance | 240 | MED | **Queued** | — |
| D-PERSONA-4 | macro-eval harness (scenario→trace→discover→diagnose; suspect-bridge = blasgraph betweenness; 5 rubrics from D-RUNG-MUL) | planner + Lance | 280 | HIGH | **Queued** | — |
| D-PERSONA-5 | ractor outer-swarm runtime under `OrchestrationBridge` (batons as messages, async only at boundary) | planner | 200 | MED | **Queued** | — |
| D-PERSONA-6 | `odoo_scanner` + `OdooBridge` — harvest Odoo `l10n_de` → Finance-ns `MappingProposal`s; bind existing `TaxEngine`; GoBD by construction | ontology + contract + planner | 280 | MED | **Queued** | — |

---

## unified-soa-convergence-v1 — ONE LE SoA end-to-end across 9 consumers + version gate + Lance 6.0.1 stack + 4-phase Rubicon kanban

> **Plan P0 status:** SHIPPED in PR #434 (merged 2026-05-29). Deliverable rows below remain Queued; they ship in follow-up PRs per phase sequencing.

Plan path: `.claude/plans/unified-soa-convergence-v1.md`. Handover `.claude/handovers/2026-05-29-1825-soa-convergence-author-to-impl.md`. Review addendum `.claude/plans/unified-soa-convergence-v1-addendum-2026-05-29-review.md`. Epiphany `E-SOA-IS-THE-ONLY` (+ §11.3/4/6 refinements).

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-MBX-A1 | migrated thoughtspace columns landed on `MailboxSoA<N>` (`edges`/`qualia`/`meta`/`entity_type`) | cognitive-shader-driver | 60 | LOW | **Shipped** | between #418 and #433 (verified `mailbox_soa.rs` 2026-05-29) |
| D-MBX-A2 | close BindSpace expressivity gaps in `MailboxSoA<N>` (`content_ref`, S/P/O role slices, temporal/expert fold per OQ-2) | cognitive-shader-driver + contract | 140 | MED | **Shipped (carrier)** | columns landed post-2026-06-13 via W1 `22f5120a` (temporal/expert/sigma) + W1b `707360dc` (dense content/topic/angle planes) + W1c + W4a shim, with accessors + parity + field-isolation tests in `mailbox_soa.rs`. OQ-1 resolved (dense planes hot, per reconciliation-doc supersession of the ≤6B-ref framing). S/P/O role slices = **NON-GAP** (roles are VSA-unbind vs `contract::grammar::role_keys`, not a per-row column — `I-VSA-IDENTITIES`). Residual: OQ-2 fold-vs-standalone (landed standalone; deferrable). See `E-DMBXA2-SHIPPED-RECONCILE`. |
| D-MBX-A3 | `witness_arc: [u32; W]` per-row column (the belief-state arc handle into AriGraph episodic Markov chain) | cognitive-shader-driver | 100 | MED | **Queued** | gates on D-MBX-A2 + OQ-11.2 |
| D-MBX-A4 | Staunen × Wisdom counterfactual plasticity spreader (Hebbian, hot-path-only, Planning-gated) | cognitive-shader-driver | 80 | LOW | **Queued — design** | gates on D-MBX-A3 + OQ-11.1 + `phase` field |
| D-MBX-A5 | SPO-W witness pointer dual-residency (SoA / kanban / mailbox index); SoA decides commit modality (chain pointer vs cold fact) | cognitive-shader-driver + AriGraph SPO-G | 150 | HIGH | **Queued** | gates on D-MBX-A3 + D-MBX-4 |
| D-MBX-A6 | `lance-graph-planner` DTO surface overhaul: DTOs as SoA-row-lenses; planner output = `KanbanMove`s; 5-phase feature-gated cutover (OQ-11.7) | lance-graph-planner + contract | 600 | HIGH | **Queued** | gates on D-MBX-10 + D-MBX-8 + OQ-11.7 |
| D-MBX-7 | `lance-graph` containers ≡ `MailboxSoA` layout ≡ `ndarray::simd_soa.rs`-aligned (1.4–4.2× SIMD payoff; hard prereq for SurrealDB transparent view) | lance-graph + ndarray | 300 | HIGH | **Queued** | gates on D-MBX-A2 + D-MBX-10 + D-MBX-11 + PR-NDARRAY-MIRI-COMPLETE |
| D-MBX-8 | Σ10 commit stamps **t = −550 ms** wall-clock (Libet anchor) in `SigmaTierRouter`; downstream ractor START fires | sigma-tier-router + shader-driver | 120 | MED | **Queued** | gates on D-MBX-A4 + D-MBX-A6 Phase 1 |
| D-MBX-9 | Rubicon kanban view in `surrealkv`-on-lance (4 columns: Planning · Cognitive work · Evaluation · Commit·Plan·Prune); ractor lifecycle hooks = kanban moves | surreal_container + ractor | 250 | HIGH | **Queued** | gates on D-MBX-7 + D-MBX-8 + surreal_container BLOCKED(B/C/D) resolved (OQ-11.6) + D-PERSONA-5 |
| D-MBX-9-IN | contract slice of D-MBX-9 IN-direction (`E-SUBSTRATE-IS-THE-SCHEDULER`): `scheduler::{DatasetVersion, VersionScheduler, NextPhaseScheduler}` — Lance `versions()` tick → next legal `KanbanMove`, zero-dep, read-only-over-view (propose-not-dispose) | lance-graph-contract | 130 | LOW | **Shipped (contract)** | 509 lib tests (+6); clippy pedantic-clean; CI-gated twin `D-MBX-9-IN-impl` (LanceVersionScheduler over `VersionedGraph::versions()`) named not written |
| D-H2H-1 | head2head superposition winner-select (item 4, Go infight-vs-Raumgewinn): `head2head::{Head2Head, WinnerCriterion, CompetitionOutcome}` — `select(&Blackboard)` arg-extremum over existing bids, select-not-duplicate | lance-graph-contract | 130 | LOW | **Shipped (contract)** | 516 lib tests (+7); clippy pedantic+nursery clean; parallel-mailbox executor = CI-gated consumer side |
| D-EW64-1 | `episodic_edges::{EpisodicEdges64, EdgeRef}` — AriGraph episodic edges (4x[4b family|12b local]); intra=inherited (~98.6%), cross=4-bit nibble->OGIT-class palette (~1.4%) | lance-graph-contract | 120 | LOW | **Shipped (contract)** | 527 lib tests; clippy clean; SoA columns = D-EW64-2 (CI-gated) |
| D-VIEW-1 | `view_angle::ViewAngle` — 4-bit view-schema selector; presence-bitmask-as-attention (inherited) | lance-graph-contract | 40 | LOW | **Shipped (contract)** | 527 lib tests; clippy clean |
| D-MBX-10 | SoA version byte at layout root (`MailboxSoAHeader`); refuse v(N>M) bytes on v(M) reader; field-isolation matrix tests on every column op (`I-LEGACY-API-FEATURE-GATED` discipline) | lance-graph-contract | 100 | HIGH | **Queued** | foundation — should land early in P2; gates on OQ-11.5 |
| D-MBX-11 | Lance bump (5 Cargo.toml) — **OBE: main jumped `=6.0.0 → =7.0.0`, not `=6.0.1`** | workspace Cargo.toml | 10 | LOW | **Abandoned (superseded by #445, 2026-06-14)** | done by PR #445 (lance/lance-linalg `=7.0.0`, lancedb `=0.30.0`, object_store 0.13.2); `=6.0.1` never existed on the lancedb path. Residual: TD-SURREALDB-KVLANCE-LANCE7 (fork still pins 6) |
| D-MBX-12 | 8-PR workspace-wide consumer alignment: 12.1 AriGraph · 12.2 Vsa16k audit · 12.4 lance-graph · 12.5 planner · 12.6 shader-driver · 12.7 callcenter · 12.8 ontology audit · 12.9 thinking-styles | per-crate | 800 | per-PR | **Queued (multi-PR)** | sequencing per OQ-11.8: 12.4 → 12.5 → 12.6 → 12.7 → 12.1 → 12.9 → 12.2 → 12.8 |
| D-MBX-A6-P1 | contract slice of D-MBX-A6: `kanban::{KanbanColumn, KanbanMove}` + `soa_view::{MailboxSoaView, MailboxSoaOwner}` + `StepDomain::Kanban` — the planner⟷ractor⟷surreal seam, zero-dep, no parallel DTO family | lance-graph-contract | 340 | HIGH | **Shipped** | #437 (merged 9161bd7); + `class_id` N1 hook ride-along |
| D-MBX-A6-P2 | Rubicon lifecycle enforcement + exec-target tag: `KanbanColumn::{next_phases, can_transition_to, is_absorbing}` (the lifecycle DAG) + `MailboxSoaOwner::try_advance_phase` (checked, `RubiconTransitionError`) + `ExecTarget{Native,Jit,SurrealQl,Elixir}` on `KanbanMove` | lance-graph-contract | 120 | LOW | **In PR** | builds on P1; 489 lib tests (+4); downstream cargo-check clean; gates the ractor owner-impl + planner emit (P3) |
| D-MBX-A6-P3a | StyleStrategy: thinking-style -> cluster -> mechanism -> recipe_kernels Tactic selection (planning substrate; carries tau JIT addr) | lance-graph-planner | 130 | LOW | **In PR** | #439; first cut of A6-P3 consumer wiring; planner now consumes contract recipes/styles; deferred: i4-32D decode, Outcome->Candidate, tau->JIT, membrane commit |
| D-MBX-A6-P3-M1 | `Tactic::requires() -> ThoughtMask` + `ThoughtField`/`ThoughtMask` (checklist-as-data keystone): 34 tactics declare their ThoughtCtx field-reads; `covered_by` = reliability-coverage gate | lance-graph-contract | 120 | LOW | **In PR** | #439; the panel-recalibrated keystone (extraction not construction); makes P1/P7/P11 derived; teeth-test asserts masks varied not stub |
| D-CLS-FM | `class_view`: FieldMask(u64 presence) + ClassView meta-DTO resolver trait + ClassProjection (the class flies ABOVE the SoA; labels resolved late from OGIT cache, zero in the bytes) — extends ObjectView, reuses class_id | lance-graph-contract | 270 | LOW | **Shipped** | #441 D-CLS contract foundation; OD-gates ratified; presence!=semantics (C2); N3 stable positions; 3 teeth-tests |
| D-CLS-RES | `class_resolver`: `RegistryClassView` impls `ClassView` over the live OntologyRegistry — the ontology-side 'parser' (class_id -> shape, DOLCE resolved LATE via classify_odoo from the cache URI, memoized over the O(n) registry scan) | lance-graph-ontology | 200 | LOW | **Shipped** | #441 D-CLS; makes the contract trait live; field-set supplied (D-CLS audit deferred); 4 teeth-tests |
| D-CLS-SIG | `class_signature`: deterministic structural-signature audit of curated OdooEntity consts (FNV-1a over kind+field-hist+method-hist+state-machine) -> shape-family group-by + `object_view()` derives the real ObjectView bit-basis (fills the D-CLS-RES placeholder) | lance-graph-ontology | 230 | LOW | **Shipped** | #441 D-CLS; the HONEST D-CLS-3 (group-by-on-structural-hash, NOT aerial-cluster vaporware, classes.md:43); 4 teeth-tests over real l1 data |
| D-CLS-AUDIT | `class_signature` corpus audit: `curated_entities()` (all 15 l-lanes, 64 consts) + `corpus_summary()` + falsifiable test that the real curated corpus collapses entities->fewer shape-families (classes.md:42 CONFIRMED on real data, not asserted) | lance-graph-ontology | 90 | LOW | **Shipped** | #441 D-CLS Wave-2 input; +clippy fix (unused FieldMask import in class_resolver) |
| D-CLS-RENDER | `ClassView::render_rows` + `RenderRow{label,predicate}` — the off-bits-skipped render surface (C2 presence-only; template-agnostic, askama engine deferred to its own crate-Wave) | lance-graph-contract | 50 | LOW | **Shipped** | #441 D-CLS; the render LOGIC (classes.md:49), not the engine; +doc-lint fix |
| D-WIKI-HHTL-1 | `contract::hhtl::NiblePath`: the 16ⁿ Abstammung bucket router (subClassOf nibble path, bit-shift O(1), `root`/`child`/`basin`/`parent`/`is_ancestor_of`) + `FieldMask::inherit` (mask-inherits-as-delta). DOLCE-agnostic (`basin: u8` = dolce_id 0..3, resolved through the cache — NO enum); multi-parent = facet bit in the same mask, NOT a 2nd path. The downstream router #438 names. | lance-graph-contract | 155 | LOW | **In PR** | Wikidata-HHTL slice 1 = the 16ⁿ router (hub-side); reuses #441 FieldMask; convergent with D-ARM-14 (firewall preserved). 4 teeth-tests; 501 contract lib green. See FINDING D-CLS↔D-ARM-14 (EPIPHANIES). |
| D-WIKI-HHTL-2 | `wikidata_hhtl`: the N4 second-domain falsifier — `WikidataClass` (curated real QIDs: human/person/city/film/tv-series/event) routed through the SAME class-meta-DTO: `nibble_path()` (basin=cache dolce_id, NO enum), `presence_mask()`=FieldMask, `signature()`=StructuralSignature over the canonical property-set, `dcls_triple()`=(ClassId,StructuralSignature,FieldMask), + `WikidataClassView` impls the #441 `ClassView`. | lance-graph-ontology | 290 | LOW | **In PR** | Wikidata-HHTL slice 2; classes.md N4 CONFIRMED on data: corpus collapses to fewer shape-families (film≡tv-series twin), triple shape domain-independent, ClassView resolves Wikidata unchanged, subclass inherits path+mask-as-delta. Reuses contract::hash::fnv1a. 5 teeth-tests; 245 ontology lib green. Deferred: the 115M streaming load (separate plan). |

---

## bindspace-singleton-to-mailbox-soa-v1 — dissolve `Arc<BindSpace>` into per-mailbox `MailboxSoA<N>`

Plan path: `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`. Epiphany `E-MAILBOX-IS-BINDSPACE`. Migration of the shared singleton address space into mailbox-owned ephemeral thoughtspace (LE-contract SoA columns); drops the 64 KB `Vsa16kF32` `cycle` plane.

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-MBX-1 | add migrated columns (`edges`/`qualia`/`meta`/`entity_type`) to `MailboxSoA<N>` behind `mailbox-thoughtspace` feature | cognitive-shader-driver | 120 | MED | **Queued** | gated on D-CE64-MB-1-impl + PR-NDARRAY-MIRI-COMPLETE |
| W3+W4a | atomic read/write shim (`backing::{BackingStore,BackingStoreWrite}`) — `driver.run()` keeps ONE body, `mailbox-thoughtspace` (default-OFF) flips substrate singleton→`MailboxSoA`; 6 reads re-pointed; W2 differential proves bit-identity; firewall lint + field-isolation + footprint gates | cognitive-shader-driver | 600 | MED | **In PR** | branch `claude/bindspace-mailbox-soa-w3-w4a`; default 97+2+2 tests, feature 98+2+2+4; clippy/fmt clean; `unbind_busdto` C5 downgrade feature-gated (cycle plane never migrated). Plan `bindspace-mailbox-soa-w3-w4a-impl-v1.md` |
| D-MBX-2 | move `engine_bridge` per-row read/write surface onto mailbox rows; `cycle` plane becomes a transient local | cognitive-shader-driver | 180 | MED | **Queued** | blocked on D-MBX-1 + OQ-1 (content-ref shape) |
| D-MBX-3 | `ShaderDriver` holds a sea-star of mailboxes; kill the `BindSpace::zeros(4096)` singleton in `serve.rs` | cognitive-shader-driver | 160 | HIGH | **Queued** | blocked on D-MBX-2 + OQ-2 (temporal/expert fold) |
| D-MBX-4 | death → SPO-G quad + Lance tombstone-witness (link-integrity back-pointer) | cognitive-shader-driver + Lance | 200 | HIGH | **Queued** | blocked on D-MBX-3 + Zone-2 persistence |
| D-MBX-5 | delete `BindSpace` singleton + `Vsa16kF32` `cycle` plane; remove feature gate | cognitive-shader-driver | 80 | MED | **Queued** | blocked on D-MBX-4 + OQ-4 (CLAUDE.md "The Click" doctrinal update) |
| D-MBX-6 | `ThoughtStruct` = transparent hot/cold view over SurrealDB container table(s) (same SoA both tiers; ~64k–256k hot ceiling, ~6 KB/thought) | cognitive-shader-driver + surreal_container | 220 | HIGH | **Queued** | blocked on D-MBX-3 + surreal_container unblock (BLOCKED A/B/C/D) or callcenter Zone-2 |
| TD-RESONANCEDTO-DUP-1 | dedup the two `ResonanceDto` (thinking-engine) | thinking-engine | 60 | LOW | **Deferred** | user 2026-05-27 — fold into D-MBX-2 |

---

## odoo-savant-reasoners-v2 — reshape: `Reasoner` trait → typed composition over `CausalEdge64` + `Tactic` + `callcenter/role_keys`

Reshape of v1 (shipped PR #420). v1's `Reasoner` trait surface fails CLAUDE.md "P-1 The Click" + "P0 AGI-as-glove" litmus tests; v2 routes the canonical path through the agnostic substrate that already exists (CausalEdge64 + Tactic + 33-TSV atoms + role-key catalogues). v1 stays under `legacy-reasoner` feature with `#[deprecated]` until woa-rs migrates. Plan path: `.claude/plans/odoo-savant-reasoners-v2.md`. Driver epiphany: `E-SAVANT-COMPOSITION-1`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ODOO-SAV-5a | `SavantPattern` + `TacticInvocation` + `EdgeEmissionSpec` + `AtomTouchMask` primitives (Group D, zero-dep, in contract) | lance-graph-contract | 200 | HIGH | **Queued** | additive — ships with this plan + INTEGRATION_PLANS prepend + this STATUS_BOARD section + EPIPHANIES entry (board hygiene) |
| D-ODOO-SAV-5b | `callcenter/role_keys.rs` with 25 disjoint Vsa16kF32 slices + lookup-by-enum + slice-allocation manifest (Group E) | lance-graph-callcenter | 250 | HIGH | **Queued** | parallel with 5a — independent; coordinate disjoint slice range vs `grammar/role_keys.rs` |
| D-ODOO-SAV-5c | 25 `SavantPattern` consts drawn from `.claude/odoo/savants/<N>.md` slot 1/4 + `.claude/odoo/L*.md` business semantics (Group F) | lance-graph-callcenter | 600 | MED | **Queued** | blocked on 5a + 5b; likely one D-id per savant in a Wave if translation is large; 14 NEEDS-INPUT savants ship pattern + emission spec only |
| D-ODOO-SAV-5d | `#[deprecated]` + `legacy-reasoner` feature gate + migration pointers on v1 `Reasoner` trait + 4 `*Reasoner` impls + `SavantConclusion` + `SavantSuggestion` + `build_conclusion` (Group G) | lance-graph-contract + lance-graph-callcenter | 120 | HIGH | **Queued** | blocked on 5c (so the migration pointer names a real target); removal in a follow-up after woa-rs migrates |
| D-ODOO-SAV-5e | End-to-end test: FiscalPositionResolver `SavantPattern` over a synthetic ontology fixture → expected `CausalEdge64` row (SPO + NARS truth + v2 signed mantissa); the proof the reshape works | lance-graph-callcenter tests | 150 | MED | **Queued** | ships with 5c completion as the round-trip proof; uses `CausalEdge64::pack_v2` per `I-LEGACY-API-FEATURE-GATED` |

---

## odoo-business-logic-blueprint-v1 — typed Odoo entity DTOs as the substrate for OGIT → OWL → DOLCE → FIBU/FIBO normalization + JITson / recipe codegen

PREREQUISITE for `odoo-savant-reasoners-v2` Group F (per `E-SAVANT-COMPOSITION-1`). Establishes the typed `OdooEntity` + sub-types layer that the inheritance chain operates on — replaces today's ad-hoc string-keyed maps against `model_name`. Both passes (L-docs first as curated filter, Odoo source extraction second as exhaustive backing). All 15 lanes (L1–L15). Plan path: `.claude/plans/odoo-business-logic-blueprint-v1.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ODOO-BP-1a | `OdooEntity` + sub-types (`OdooField`/`OdooMethod`/`OdooDecorator`/`OdooStateMachine`/`OdooConstraint`/`OdooProvenance`) — zero-dep, const-only, no serde | lance-graph-ontology | 300 | HIGH | **Queued** | ships with plan + INTEGRATION_PLANS prepend + this STATUS_BOARD section (board hygiene); additive — zero churn to existing call sites |
| D-ODOO-BP-1b | L-doc projection: one `OdooEntity` const per entity, 15 lanes, per-lane module `odoo_blueprint::l{1..15}`, provenance=Curated with line-range citations | lance-graph-ontology | 2500 | HIGH | **Queued** | blocked on 1a; ships in Waves (L1-L5, L6-L10, L11-L15), one subagent per lane (Sonnet, mechanical prose→const projection); ~5 entities/lane average × 15 lanes ≈ 75-200 consts |
| D-ODOO-BP-1c | Wire OGIT classifier to take `&OdooEntity` (replaces string-keyed `resolve_odoo`); uses field/method semantics for richer dispatch; covers 0x63/0x90 from PR #414 | lance-graph-ontology + lance-graph-callcenter::family_table | 250 | HIGH | **Queued** | blocked on 1b; parallel with 1d/1e |
| D-ODOO-BP-1d | Wire OWL hydrator to take `&OdooEntity`: relational fields → edges, computed fields → SHACL-equivalent constraints, decorators → axioms | lance-graph-ontology | 350 | MED | **Queued** | blocked on 1b; parallel with 1c/1e |
| D-ODOO-BP-1e | Wire DOLCE classifier + FIBU/FIBO alignment to take `&OdooEntity`; closes D-ODOO-SAV-2's `None`-class alignment for stock.* / analytic.distribution.model / account.account.tag over typed input | lance-graph-ontology | 200 | HIGH | **Queued** | blocked on 1b; parallel with 1c/1d |
| D-ODOO-BP-1f | Odoo source extraction tool: tree-sitter Python AST → candidate `OdooEntity` consts with Confidence=Extracted; validates + extends 1b's curated set | tools/odoo-blueprint-extractor/ | 800 | MED | **Queued** | blocked on 1b/c/d/e; conflicts (curated vs extracted) flag for ratification, default to curated |
| D-ODOO-BP-1g | Wire JITson → recipes: `jit::JitCompiler` compiles `Tactic` kernels parameterized by `(&OdooEntity, AtomTouchMask)`; produces DTO-ish NARS that lands in shader-driver | lance-graph-contract::jit + thinking-engine | 400 | MED | **Queued** | blocked on 1c/d/e; proof-of-concept on FiscalPositionResolver, the rest follow in `odoo-savant-reasoners-v2` Group F |
| D-ODOO-STYLE-1 | `style_recipe.rs` — Phase 1 D-Atom interpretation step: typed Odoo SoA → `OdooStyleRecipe` cognitive fingerprints (12 DAtom basis, 7-rule cascade, FNV-1a recipe_id, never stored back as triples) | lance-graph-ontology::odoo_blueprint | 746 | HIGH | **Shipped** | commit `feb8be54` (PR #433 merged); 13/13 tests; DAtom::ALL discriminant-order pinned; OdooStyleRecipe != contract::recipe::StyleRecipe (documented) |
| D-ODOO-OP-1 | `op_emitter.rs` — Phase 2 bucket-dispatch codegen: `bucket_corpus` groups OdooStyleRecipe by OdooMethodKind; `emit_op_dispatch` emits compilable Rust (RECIPE_* consts + per-kind Op structs + static Op slices); deterministic, recipe_id dedup collapses identical DAtom profiles | lance-graph-ontology::odoo_blueprint | 400 | HIGH | **Shipped** | commit `63f3e2ca`; 12/12 tests; zero-dep emitted output; 230/230 existing tests green |

---

## streaming-arm-nars-discovery-v1 — upstream proposer leg into the SPO substrate (20K-200K rows/window pair-stats + optional Aerial+ → NARS-truth → SpoStore hypothesis test → council ratification → op_emitter codegen)

The missing UPSTREAM discovery leg. Today's proposers (curated L-docs + AST-extracted Odoo source) are bounded by the literal artifact; this plan adds runtime-tabular-data ARM discovery, gated through the epiphany-brainstorm-council before reaching the deterministic codegen path. Plan: `.claude/plans/streaming-arm-nars-discovery-v1.md`. Handover: `.claude/handovers/2026-05-29-2030-arm-discovery-author-to-impl.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ARM-1 | `ProvenanceTier::{Curated,Extracted,ArmDiscovered,Ratified,Conjecture}` enum + ordering | lance-graph-contract | 50 | HIGH | **Queued** | blocks all other D-ARM-*; additive |
| D-ARM-2 | `Proposer` trait + `CandidateRule` carrier + `WindowMetadata` | lance-graph-contract | 100 | HIGH | **Queued** | blocks D-ARM-3, D-ARM-9. D-ARM-13 shipped **local mirrors** (`rule::{CandidateRule, Proposer, Item}`) ahead of this — see **TD-ARM-CARRIER-FORK**: re-export via `pub use` when this lands (firewall allows path-dep on zero-dep contract). Field set diverges — local carries bare `n: u32`, this plans `WindowMetadata`; reconcile (recommend `n: u32`) so the shape matches. |
| D-ARM-3 | Pair-stats proposer (default trunk, deterministic, k² pair counters per window) | lance-graph-arm-discovery::proposer::pair_stats | 400 | HIGH | **Queued** | depends on D-ARM-1/2/7; blocks D-ARM-12 |
| D-ARM-4 | ARM-truth → NARS-truth translator + Odoo `FeedProjector` impl | lance-graph-arm-discovery::translator | 200 | HIGH | **Partially shipped (branch)** | The translator substance landed early inside D-ARM-13: `translator::{arm_to_nars, NarsTruth, CandidateTriple, FeedProjector}` (verbatim paper §2/§3.3 mapping, 35/35 tests). REMAINING: the real Odoo `FeedProjector` (currently a `DebugProjector` stub emitting `implies`) + contract homing on D-ARM-1/2. Depends on D-ARM-1/2. |
| D-ARM-5 | Hypothesis test: SpoStore round-trip, NARS revision, contradiction commit per The Click | lance-graph-arm-discovery::hypothesis | 350 | MED | **Queued** | depends on D-ARM-4; verifies `spo::truth::Contradiction` primitive exists |
| D-ARM-6 | `RatificationQueue` ring buffer + corrections-to-#434 spec PR (`discovery_arc D=8`, `discovery_origin u8`) | lance-graph-arm-discovery::queue + #434 spec follow-up | 200 + spec | MED | **Queued** | depends on PR #434 D-MBX-A3 landing |
| D-ARM-7 | Jirak-2016 weak-dependence significance thresholds (mandatory Stage A floor) | lance-graph-arm-discovery::jirak | 150 | HIGH | **Queued — HARD PREREQUISITE** | blocks D-ARM-3; cites I-NOISE-FLOOR-JIRAK. **ISSUE ARM-JIRAK-FLOOR (2026-05-30, 3-savant review):** D-ARM-13 ships the Aerial proposer with NO Jirak floor (classical `min_support`/`min_confidence` only). MUST land before D-ARM-5 wires the proposer to a live `SpoStore`, else the substrate calcifies on thin-but-frequent noise (plan §11.1). **ENGINE EXISTS:** `jc::jirak` (Jirak-Cartan Pillar 5) is the weak-dependence Berry-Esseen rate (`n^(p/2-1)`); this deliverable is the *gate function* (rule → significant?) that derives its threshold from it — NOT a from-scratch Jirak impl. See E-ARM-JC-RESOLVES-BOTH-SEAMS + `splat-codebook-aerial-wikidata-compression.md`. |
| D-ARM-8 | `Feed` + `FeedProjector` + window-size config + Odoo `account.move` projector example | lance-graph-arm-discovery::feed | 250 | MED | **Queued** | depends on D-ARM-2 |
| D-ARM-9 | Aerial+ IPC client (feature-gated `arm-aerial`, NDJSON over Unix socket) | lance-graph-arm-discovery::proposer::aerial_ipc | 200 | MED | **Superseded by D-ARM-13** | The native in-process Aerial+ transcode (D-ARM-13, branch `claude/jolly-cori-clnf9`) replaces the need for the Python IPC client. The determinism-boundary rationale the IPC was designed for (keep the nondeterministic autoencoder out of the Rust path) is now met in-process via seed (`aerial::Rng`) + `aerial` feature gate + workspace `exclude`. Keep this row ONLY if a Python-only Aerial variant is later required; otherwise close as Abandoned-by-replacement. |
| D-ARM-10 | `op_emitter::bucket_corpus` ratification filter (`confidence ≥ Ratified`) + 2 tests | lance-graph-ontology::op_emitter | 30 | HIGH | **Queued** | depends on D-ARM-1 |
| D-ARM-11 | `style_recipe.rs` rule 8 — ArmDiscovered backing adds `DAtom::Compute` weight 2 (provisional) | lance-graph-ontology::style_recipe | 80 | MED | **Queued** | depends on D-ARM-1 |
| D-ARM-12 | End-to-end pipeline test + bench (synthetic Odoo feed → all 5 stages → council micro-batch) | lance-graph-arm-discovery::tests + benches | 400 | MED | **Queued** | depends on Waves 1-6; informs OQ-ARM-2 + OQ-ARM-7 |
| D-ARM-13 | **Aerial+ Rust transcode — deterministic codebook-probe backend** (float-free). The paper's `f32` denoising autoencoder is REPLACED by an integer `CodebookDistance` oracle (palette256 distance, ρ=0.9973 vs cosine): the reconstruction probe is a codebook top-k, not a softmax over float weights. Integer evidence counts + ppm gates + `TruthU8` (= CausalEdge64 wire). `AerialProposer` impl of `Proposer`. Count loop is a row-bitset SoA (`RowMasks`) → AND+popcount, routed through `ndarray::simd::U64x8` under the `ndarray-simd` feature. | lance-graph-arm-discovery::aerial | ~1.1K | HIGH | **Shipped (branch)** | branch `claude/jolly-cori-clnf9`; standalone zero-dep crate (excluded); **33/33** tests + clippy `-D warnings` clean on BOTH default (scalar) and `--features ndarray-simd`; **zero f32 in the discovery path** (audit), float only at the `TruthValue`/`Triple` serialization edge. Bitwise-deterministic ⇒ joins the trunk; the nondeterminism firewall + D-ARM-9 IPC rationale are moot. SIMD target-cpu caveat: real AVX-512/AMX kernels need `-C target-cpu=native`/`x86-64-v4`. v1 (autoencoder) superseded per the user's no-float directive. |
| D-ARM-14 | **Splat-codebook oracle + Wikidata skeleton discovery** — wire the certified jc splat codebook into aerial as the `CodebookDistance` oracle, discover OWL/DOLCE+ SPO HHTL classes + basins, drive the `wikidata-hhtl-load.md` deterministic compression (skeleton + basins + CAM-dedup + thin rows). | lance-graph-arm-discovery::aerial + crates/jc + wikidata loader | ~? | MED | **In progress** | **Phase 1 (branch `claude/jolly-cori-clnf9-darm14`):** the two aerial-side seams — `aerial::TopKDistance` (the sparse splat-top-k `CodebookDistance` the 10000² BLASGraph splat actually emits; top-k per node, not a dense table) + `aerial::ontology::{DolceCategory, OntologyProjector}` (DOLCE 4-facet skeleton → `rdfs:subClassOf`/`rdf:type` SPO). End-to-end test: splat top-k → aerial discovers `occupation→DOLCE-class` → projects the skeleton triple. 41/41 + clippy clean (default + `ndarray-simd`), zero-dep. Float still OFFLINE in jc only (`ewa_sandwich`+`sigma_codebook_probe` ρ=0.9973+`pflug` Lε); aerial online path integer. **Phase 2 (branch `claude/jolly-cori-clnf9-darm14-p2`):** the proposer→hub landing. (a) `OntologyProjector::dolce_id()` — emits the stable `dolce_id` u8 (= basin nibble) the hub routes by, NOT a hardcoded IRI (the OD-DOLCE alignment #442 deferred to this lane; basin ordering already matches `dolce_id::{ENDURANT=0..}`). (b) gated worked example `tests/wikidata_landing.rs` (`--features landing`, opt-in `dev-dep lance-graph-contract` à la jc): splat top-k → aerial recovers all 6 DOLCE basins → lands each on the REAL merged `contract::hhtl::NiblePath` (16ⁿ router, #442) + `class_view::FieldMask` (+`inherit`) + `hash::fnv1a_str` signature. CONFIRMED on data: corpus collapses 6→5 families (film≡tv-series twin), human⊂person inherits path + mask-as-delta, basin preserved. 42/42 default (zero-dep) + landing test green, clippy clean both. Rebased onto post-#442 main; the inline-nibble stand-in swapped for the real `NiblePath`. **Remaining:** real jc/blasgraph splat producing the lists; the ndjson→`WikidataClass` loader; gated on D-ARM-7 (`jc::jirak`). Map: `splat-codebook-aerial-wikidata-compression.md`; E-ARM-JC-RESOLVES-BOTH-SEAMS. |
| D-ARM-SYN-1 | Add `Implies`/`CoOccursWith` to `ruff_spo_triplet::Predicate` closed vocabulary (+ `Provenance` tier) so ARM rules load through the same `parse_triples` ndjson path as the static extractor | ruff/ruff_spo_triplet | 40 | MED | **Queued** | council-gated (deliberate ontology change); blocks SYN-2; see `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md` §1 |
| D-ARM-SYN-2 | `CandidateRule → ruff_spo_triplet::ModelGraph` adapter so the Aerial runtime-data leg joins the `ruff_python_dto_check` static-AST leg in one graph before `expand()` | lance-graph-arm-discovery + ruff_spo_triplet | 120 | MED | **Queued** | depends on SYN-1; synergy doc §2 |
| D-ARM-SYN-3 | Calibrate `ProvenanceTier::ArmDiscovered` `(f,c)` below the `op_emitter` ratification gate + below static `Inferred (0.85,0.75)` so un-ratified ARM truth is council-visible but codegen-filtered | lance-graph-contract + lance-graph-ontology::op_emitter | 30 | MED | **Queued** | depends on D-ARM-1 + SYN-1; synergy doc §3/§4 |
| **D-CHESS-BRINGUP-1** | **Chess-into-OWL falsification slice** — encode 3-5 opening positions as OWL/ttl (meaning in CONTENT, no chess-special SoA field), run `lance-graph-arm-discovery::AerialProposer` (the now-shipped #436 Rust transcode) over it, see whether GM-flavored *legal* candidates fall out, AND whether chess needs columns Odoo's SoA didn't have. The cognitive-risc-core/classes spec's emphatic **N4 freeze-time non-negotiable** — falsifies/confirms "one SoA serves all" cheaply on a board, before the WAL freezes. Council R2+R4 verdict 2026-05-30: this is FIRST, not a peer option. | lance-graph-arm-discovery (read) + new crate `chess-bringup-test` | ~300 | HIGH | **Queued** | NEW (council recalibration 2026-05-30). NOT in scope for branch `claude/activate-lance-graph-att-k2pHI` (per R1 — needs its own branch + freeze-decision authority). Unblocked by #436's Aerial+ shipping in Rust (user-flagged 2026-05-30 "aerial+ has been transcoded and is now a lance-graph-* crate"). Cross-ref `cognitive-risc-core.md` §"The bring-up test"; `cognitive-risc-classes.md:66` N4; `post-438-integration-options-v1.md` §1 Option G. |

---

## odoo-classes-bitmask-render-v1 — bounded-weekend ClassId + FieldMask + per-class askama templates (Aerial+ discovers ~10-15 shape-families from 66 OdooEntities; presence-bitmask render path)

The bounded-weekend fix `cognitive-risc-classes.md:56-57` prescribes (discriminator + parent-pointer + parent-walking; full machinery deferred). 4-way `DolceCategory` consolidation + `ClassId(u16)` hook + per-class `FieldPositionTable` + `FieldMask(u64)` + per-class askama templates. **All deliverables `Blocked-on-OD` until spec owner ratifies OD-DOLCE-CANONICAL, OD-CLASSID-WIDTH, OD-CLASSID-VS-ENTITYKIND, OD-TEMPLATE-ENGINE.** Plan: `.claude/plans/odoo-classes-bitmask-render-v1.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| **D-CLS-1** | Canonical `DolceCategory` in `lance-graph-contract` + re-exports from 3 sites + `From<DolceCategory> for DolceMarker` | lance-graph-contract + 3 modified | 80 | HIGH | **Blocked-on-OD** | Wave 1A, Sonnet. Additive per C6. Arm-discovery uses local newtype + TryFrom (zero-dep stance preserved) |
| **D-CLS-2** | Structural-signature audit of 66 OdooEntities → `.claude/knowledge/odoo-66-structural-signatures.psv` | lance-graph-ontology (read only) | 230 | HIGH | **Blocked-on-OD** | Wave 1B, Sonnet. Read-only emit; BLAKE3-128 truncated u64 per-entity hash |
| **D-CLS-3** | Aerial+ structural-hash → ~10-15 candidate shape-families + ratified `CANONICAL_CLASS_TABLE` | lance-graph-arm-discovery (example) | 350 | MED | **Blocked-on-OD** | Wave 2A, **Opus**. SPEC-OWNER GATE after output: names + ratifies clusters |
| **D-CLS-4** | New `lance-graph-ontology-render` crate skeleton + askama dep + `exclude=` workspace entry | new crate | 70 | HIGH | **Blocked-on-OD** | Wave 1C, Sonnet. Standalone like bgz17/deepnsm |
| **D-CLS-5** | `ClassId(u16)` newtype + `UNCLASSIFIED` const in `lance-graph-contract::cognition::entity` | lance-graph-contract | 40 | HIGH | **Blocked-on-OD** | Wave 3A, Sonnet. The N1 hook |
| **D-CLS-6** | `class_id: ClassId` field on `OdooEntity` + back-fill 66 consts via ratified CANONICAL_CLASS_TABLE | lance-graph-ontology (mod + 15 lanes + new CLASS_TABLE.rs) | 260 | HIGH | **Blocked-on-OD** | Wave 3B, Sonnet. Mechanical edit across 15 lane files |
| **D-CLS-7** | `FieldMask(u64)` + `FieldPositionTable` (N3 append-only positions) + per-class width audit | lance-graph-contract (new field_mask.rs) + lance-graph-ontology (CLASS_TABLE extend + class_audit.rs) | 250 | HIGH | **Blocked-on-OD** | Waves 3C + 3D + 3E split (5 Sonnet agents in Wave 3 total) |
| **D-CLS-8** | Per-class askama templates (~10-15 .txt.j2) + `render(entity, mask) -> String` + per-class smoke tests | lance-graph-ontology-render (lib + templates + tests) | 510 | MED | **Blocked-on-OD** | Wave 4 (3 Sonnet agents — templates, dispatch, tests) |
| **D-CLS-9** | Integration test 66 entities × class templates + C2 mutant-mask test + mask-density audit report | lance-graph-ontology-render (tests + audit + snapshots) | 2,310 | HIGH | **Blocked-on-OD** | Wave 5A, **Opus**. Bulk LOC is 66 generated snapshots + per-class density report |

---

## wikidata-lazy-spine-hydration-v1 — the NiblePath-keyed tiered hydration manager + addressing (the "agnostic lazy world-spine" runtime)

The one missing runtime piece behind the converged delta-card / world-spine vision (`delta-card-addressing-integration-map.md`, `agnostic-lazy-world-spine.md`). Plan: `.claude/plans/wikidata-lazy-spine-hydration-v1.md` (9 D-ids, authored by the W1 wave worker). All gated on D-ARM-7 (Jirak floor) before any hydrated rule writes a live store; firewall (aerial = zero-dep proposer, hub owns contract/ontology) preserved.

| D-id | Deliverable | Crate(s) | LOC | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-LWS-1 | Sparse radix range-delegation register (path-compressed trie over the frozen ontology; occupied branch points only; reuses `NiblePath` as the address — never re-encodes identity) | lance-graph-contract / -ontology | ~? | MED | **Queued** | partition-as-address; 27-bit floor with ~0-bit row |
| D-LWS-2 | Delta-card value model (`reconstruct = deck ⊗ delta`; per-entity surprise as a `FieldMask` delta over the inherited archetype; modal member = empty card) | lance-graph-contract | ~? | MED | **Queued** | built on `FieldMask::inherit` |
| D-LWS-3 | RISC compose-cache + per-predicate composability flag (store generators, compose ≤7-hop closure via `ComposeTable`/`mxm`; dissolves the hub problem) | lance-graph + bgz-tensor | ~? | MED | **Queued** | generators=continuant/cold, composed=occurrent/evictable |
| D-LWS-4 | I/P/B frame model over Lance versioning (I=frozen radix+base, P=append, B=compose-cache, GOP=compaction) | lance-graph | ~? | MED | **Queued (spike)** | R2: repo wires dataset-level `VersionedGraph`, not fragment-level — fragment GOP is a NEW spike |
| D-LWS-5 | **The `NiblePath`-keyed tiered hydration manager** (THE missing piece): hot `MailboxSoaView` ↔ cold `VersionedGraph`, address-not-join, agnostic SoA, carries CE64+witness arc; write-refusal until D-ARM-7 | lance-graph | ~? | MED | **Queued** | centerpiece; D-ARM-7 write-refusal acceptance test |
| D-LWS-6 | Foveated prefetch cascade (`HhtlCache::route` Skip/Attend/Compose/Escalate decides periphery prefetch into the 256K envelope) | lance-graph + bgz-tensor | ~? | MED | **Queued** | the Google-Maps tile prefetch |
| D-LWS-7 | Eviction on the DOLCE continuant/occurrent 1-bit (`dolce_id==PERDURANT` ⇒ occurrent ⇒ evictable; 4-facet axis preserved, residence bit derived) | lance-graph | ~? | MED | **Queued** | the perm/temp residence policy |
| D-LWS-8 | Probe harness — runs the 3 falsifiers (Louvain-CLAM locality, delta-card residual, compose hit-rate) on real `data/ontologies/*.ttl` + fixtures; PRODUCES the gates | crates/jc + lance-graph | ~941 | HIGH | **Probe-1 SHIPPED** | `jc/examples/ontology_locality_probe.rs` RUN: **locality 98.6%, max fan-out 3 (≤16), Q=0.325 → PASS** on real ontologies (not yet Wikidata). Probes 2-3 queued. |
| D-LWS-9 | DEFERRED full Wikidata 115M load (skeleton+basins+CAM-dedup+thin rows) | wikidata loader | ~? | LOW | **Deferred** | gated on all 3 probes PASSED + D-ARM-7; CONJECTURE (no dump on disk) |

## Markov substrate clarification (markov_soa / EW64) — three-Markovs taxonomy

| D-id | Deliverable | Crate(s) | LOC | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-MKV-SOA | `arigraph::markov_soa` — the Markov *wave* (AriGraph cold-path chain promoted to hot-path SoA); vocabulary-agnostic `SpoRanks{u16}` + `SoaWavePrimer` + `WaveProjection::best_guess_match(injected dist)`; the "hybrid+ autocomplete" #2 proposer (dark-horse) | lance-graph::graph::arigraph | ~230 | MED | **Shipped (branch, unverified-offline)** | moved out of deepnsm (SoC fix); match = AriGraph's own cam_pq, language stays upstream; 4 tests written, core doesn't build in sandbox → verify on full checkout. Findings: three-Markovs, markov_soa-IS-AriGraph |
| D-EW64-NOTE | `MailboxSoaView` doc note: `EpisodicWitness64` = AriGraph in the mailbox SoA view (the particle; cold→hot); deferred accessor (qualia-pattern) | lance-graph-contract::soa_view | ~20 | HIGH | **Shipped (branch)** | verified (contract builds, 3/3 soa_view tests); EW64 not yet a code symbol — P2 of three-Markovs ordering |

---

## Update protocol

When a deliverable ships:
1. Edit this file's Status column in place for the row → **Shipped**.
2. Fill in PR / Evidence column with the merge commit or PR #.
3. Append a new section to `PR_ARC_INVENTORY.md` (Added / Locked /
   Deferred / Docs / Confidence).
4. Update `LATEST_STATE.md` (Recently Shipped PRs + Current Inventory
   if types change).

When a deliverable moves phase (e.g. Queued → In progress → In PR):
1. Edit Status column in place. Don't reorder rows.
2. If the move reflects scope correction, also update
   `INTEGRATION_PLANS.md` Status line for the parent plan.

When a new deliverable is added to a plan:
1. Append a new row at the bottom of the plan's section.
2. D-id is sequential in the plan (D12, D13, etc.).
3. Original scope becomes immutable once committed.

When a deliverable is abandoned:
1. Edit Status → **Abandoned**. Don't remove the row.
2. Cite the replacement in Notes.

## D-EW64-3 / D-EW64-4 (2026-06-01, autoattended)

| D-id | deliverable | status | evidence |
|---|---|---|---|
| D-EW64-3 | `EpisodicEdges64::{coldest, contains}` — MRU cold-tier read surface | In PR | contract lib 545 green; clippy clean |
| D-EW64-4 | `DemotionSink` trait + `promote_into` — hot→cold exit seam (impls gated OQ-11.6) | In PR | contract lib 545 green; clippy clean |

---

## identity-architecture-exists-vs-needs-v1 — structured NodeGuid + frugal north-star OGAR mint

Plan path: `.claude/plans/identity-architecture-exists-vs-needs-v1.md`. Epiphanies: E-IDENTITY-WHITEBOX-1, E-OGAR-NORTHSTAR-1. Rides in the open identity PR on `claude/nice-edison-g4rhhl`.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-IDENTITY-1 | `identity::NodeGuid` (UUIDv8) + `NiblePath::from_packed` — byte layout, version/variant gates, field-isolation matrix | `lance-graph-contract` | ~250 | LOW | **Shipped** | Phase A; +15 contract tests, clippy-D clean |
| D-IDENTITY-2 | Frugal north-star mint: dedup-by-URI global template id + `entity_type↔NiblePath` bijection pair table + round-trip tests (moves 1+2+3) | `lance-graph-ontology` | ~250 | LOW | **In PR** | dedup + `register_class_path`/`niblepath_of`/`entity_type_of`/`rows_with_entity_type`; +5 tests, 14 registry green |
| D-IDENTITY-3 | Gate legacy positional `contract/ontology.rs:85 entity_type_id` per I-LEGACY-API-FEATURE-GATED (move 4) | `lance-graph-contract` / -ontology | ~80 | MED | **Queued** | needs consumer audit first |
| D-IDENTITY-4 | Pair-table Lance persistence (re-register-on-hydration → persisted) | `lance-graph-ontology` | ~60 | LOW | **Queued** | TECH_DEBT TD-PAIRTABLE-1 |

---

## polyglot-container-query-membrane-v1 — three dialects, one HHTL address space, mailbox as cold path

Plan path: `.claude/plans/polyglot-container-query-membrane-v1.md`. Research grounded 2026-06-09; rides on `claude/nice-edison-g4rhhl`.

| D-id | Title | Crate(s) / repo | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-PG-1 | `addr64` left-aligned HHTL codec + order-preservation property test (subtree ⇔ contiguous range) | `lance-graph-contract` | ~120 | LOW | **Queued** | first brick; everything stands on it |
| D-PG-2 | `SoaEnvelope` impl for `MailboxSoA<N>` (= identity N3, confirmed live) + LE parity test | `cognitive-shader-driver` | ~150 | LOW | **Queued** | gap re-verified 2026-06-09 (§2.4 of plan) |
| D-PG-3 | Read-only mailbox `Transactable` adapter (5 methods, phase-pinned) + hot==cold differential test | shader-driver + fork contract | ~250 | MED | **Queued** | gated on D-PG-1,2 |
| D-PG-4 | `SurrealqlParse` strategy → ArenaIR (SELECT point/range) + selector rule | `lance-graph-planner` | ~300 | MED | **Queued** | slot proven by sparql_parse |
| D-PG-5 | DDL ⇄ registry bridge (DEFINE walker → mint; reverse via C16b `ToSql`) | `lance-graph-ontology` | ~250 | MED | **Queued** | gated on fork C16c |
| D-PG-6 | (optional) `surreal_container` unblock → kanban view over LanceDB | `surreal_container` | ~200 | LOW | **Queued** | ruling-compliant; OQ-PG1 open |
| D-PG-7 | Deterministic foveated tree-builder (CLAM-style 16-way bootstrap + append-stable insertion → `register_class_path`) | `lance-graph-ontology` + ndarray CLAM | ~300 | MED | **Queued** | plan §8 addendum; gated on D-PG-1; determinism + append-stability property tests mandatory |
