# LATEST_STATE — What Just Shipped (read this FIRST)

> **Auto-injected at session start via SessionStart hook.**
> Updated after every merged PR.
> **Last updated:** 2026-05-13 (sprint-5 cross-repo landing complete: lance-graph PR #364 + MedCare-rs#112 + smb-office-rs#31 + ndarray#142 all merged the same day. lance-graph #364 ships D-SDR-3/4/5 + sprint-log-4 governance + sprint-5-9 roadmap + codex P1/P2 surgical fixes (OwlIdentity 3-byte canonical, UnifiedAuditEvent 26 bytes, OgitFamilyTable sparse `HashMap<u16, FamilyEntry>`, audit super_domain via AuditChain). MedCare-rs#112 (PR-B) wires `UnifiedBridge<MedcareBridge>` + medcare-rbac + medcare-realtime substrate (+2963 LOC, 17 files, §73 SGB V + BMV-Ä §57 + BtM regulatory tests). smb-office-rs#31 (PR-C) wires `UnifiedBridge<OgitBridge>` (+111 LOC). ndarray#142 ships VBMI gate for `permute_bytes` (P0 SIGILL fix on Skylake-X / Cascade Lake / Ice Lake-SP) + Inf clamp for `simd_exp_f32`. D-SDR-5 `UnifiedBridge` surface is now consumed end-to-end across MedCare + smb-office. Prior: 2026-05-07 (PR #354). Prior: 2026-05-07 (PR #353). Prior: 2026-05-07 (PR #352). Prior: 2026-05-06 (splat-osint-ingestion-v1 PR 1+2 of 6 in flight). Prior: 2026-04-21 post PR #243.
>
> Purpose: prevent new sessions from hallucinating structure that
> already exists or proposing features already shipped. Read this
> BEFORE proposing any grammar/crystal/contract changes.

---

## Recently Shipped PRs (reverse chronological)

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#364** | 2026-05-13 | D-SDR-3/4/5 + sprint-log-4 governance + sprint 5-9 roadmap + codex P1/P2 | Tier-A substrate close: **D-SDR-3** OgitFamilyTable + FamilyEntry codebook (~300 LOC), **D-SDR-4** merkle-chained UnifiedAuditEvent (~460 LOC, AuditMerkleRoot = u64 FNV-1a), **D-SDR-5** authorize_* through Policy::evaluate with audit emission (~300 LOC). **Codex P1 fix** (`3208743`): OwlIdentity widened u8→u16 slot → 3-byte canonical `[family, slot_lo, slot_hi]`; OgitFamilyTable → sparse `HashMap<u16, FamilyEntry>`; UnifiedAuditEvent canonical_bytes 25→26. **Codex P2 fix** (`e23ce89`): emit_audit uses AuditChain.super_domain() instead of static FAMILY_TO_SUPER_DOMAIN. **CI fix** (`a3c753f`): ndarray/hpc-extras opt-in for blake3. Sprint-log-4 governance corpus (12 worker specs + 2 meta reviews) + sprint-5-through-9 roadmap (70 agents = 60W + 10M across 5 sprints, mandatory 12-step plan-read-order in worker prompts). 97/97 callcenter lib tests pass. All 5 CI checks green on `c8176cb`. Adjacent: ndarray#142 (VBMI gate + Inf clamp) merged same day. |
| **#354** | 2026-05-07 | gov: #353 post-merge + cross-repo adjacent-landings | Pure governance close-out. PR_ARC entry for #353 + LATEST_STATE row. Documents the 5-PR coordinated landing across 4 repos: lance-graph #352/#353/#354 + OGIT #2 (woa+medcare bridges unblocked for OGIT-O(1)) + woa-rs #2 (cross-repo `--features ontology` integration) + MedCare-rs #109 (`?source=lance` exercising Zone 2 → Zone 3 rewriter chain). Locks: append-only board hygiene durability across 4 sequential prepends; cross-repo coordinated-landing recipe. |
| **#353** | 2026-05-07 | plan: palantir-parity-cascade v2 + SoA DTO entropy ledger + #352 post-merge governance | Three artifacts. **v2 capstone** (262 lines): integrates 4 prior Foundry parity docs. Pillar 0 carry-forward: Foundry parity IS SoA-as-canon parity. Column H (PR #272 SHIPPED) is already the Foundry Object Type bridge. 15 D-PARITY-V2 deliverables. **SoA DTO entropy ledger** (210 lines, append-only knowledge): 22 DTOs classified across 4 tiers (sensor → engine → contract → callcenter). Buckets: 9 bare-metal / 7 SoA-glue / 6 bridge-projection (3 OPEN). `ResonanceDto` IS the SoA. Codec cascade columns all OPEN today. **#352 post-merge governance**: PR_ARC + LATEST_STATE updates. |
| **#352** | 2026-05-07 | plan: lance-graph-ontology v5 + ogit-cascade v1 | Two-plan PR. **v5** (177 lines): 15 deliverables for ontology crate post-merge follow-on (D-1 dcterms:source, D-2 SpoBridge::promote_to_spo, D-9 ontology-aware MUL thresholds). 4 ratifications (smb-ontology export-only, D-9 above D-2, MulThresholdProfile in lance-graph-contract, OGIT-fork upstream non-PR). **v1 cascade** (209 lines): 15 D-CASCADE deliverables for SoA-as-canon + Zone 1/2/3 + BioPortal arsenal + bridge collapse. **Pillar 0**: OntologyRegistry IS the SoA, schema IS the DTO + name→row index. **Codec cascade per row** (target state, NOT YET WIRED — D-CASCADE-V1-7): identity Vsa16kF32 → CAM-PQ 6 B → Base17 34 B → palette key 4 B → Scent 1 B + qualia 18×f32 + meta 8 B + edge 8 B, every step O(1). |
| **#243** | *(open)* | D5+D7 categorical-algebraic inference | `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery (295 LOC, 14 tests), `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). Plans: `categorical-algebraic-inference-v1.md` (496 lines). Knowledge: `paper-landscape-grammar-parsing.md`, `session-2026-04-21-categorical-click.md`. CLAUDE.md § The Click (P-1). 12 epiphanies. |
| **#225** | *(open)* | Codec-sweep plan + D0.6/D0.7 CodecParams | 9-commit plan (`codec-sweep-via-lab-infra-v1.md`, Rules A-F, 9 starter YAMLs, CODING_PRACTICES audit) + `lance-graph-contract::cam` CodecParams/Builder/precision-ladder validation (14 tests). 147/147 contract suite |
| **#224** | 2026-04-20 | lab = API+Planner+JIT, thinking harvest, I11 measurability | `lab-vs-canonical-surface.md` extended: three-part lab stack (API + Planner + JIT), thinking-harvest subsection (REST/Cypher → `{rows, thinking_trace}` = the AGI magic bullet), I11 invariant (every layer L0→L4 emits harvest-ready trace; no black-box short-circuits) |
| **#223** | 2026-04-20 | LAB-ONLY firewall + AGI-as-SoA + I1-I10 | `lab-vs-canonical-surface.md` initial doc: canonical consumer = `UnifiedStep`/`OrchestrationBridge`, Wire DTOs are lab quarantine. AGI = (topic, angle, thinking, planner) = struct-of-arrays consuming cognitive-shader-driver. 10 cross-cutting invariants I1-I10 (BindSpace read-only, canonical `simd::*` import, temporal budgets, temperature hierarchy, thinking IS AdjacencyStore, weights are seeds, per-cycle cascade, 4096 surface, three DTO families, HEEL/HIP/BRANCH/TWIG/LEAF) |
| **#210** | 2026-04-19 | Phase 1 grammar + knowledge docs | ContextChain reasoning ops, role_keys slice catalogue, 3 knowledge docs (grammar-landscape, linguistic-epiphanies E13-E27, fractal-codec) |
| **#209** | 2026-04-19 | sandwich layout + bipolar cells | Crystal fingerprint sandwich, VSA_permute reference, lossless bundling corrections |
| **#208** | 2026-04-19 | grammar + crystal + AriGraph unbundle | Contract grammar/ + crystal/ modules, AriGraph episodic unbundle hooks with SIMD dispatch |
| **#206** | 2026-04-18 | state classification pillars | qualia.rs (17D), proprioception.rs (7 anchors), world_map.rs, sigma_rosetta 64 glyphs + 144 verbs, Pumpkin NPC example |
| **#205** | 2026-04-18 | engine bridge + CMYK/RGB qualia | engine_bridge.rs, 12-style unified mapping, 17D vs 18D qualia distinction |
| **#204** | 2026-04-18 | cognitive-shader-driver | New crate, ShaderDispatch/Resonance/Bus/Crystal DTOs, BindSpace struct-of-arrays, full ladybug-rs import |

## Current Contract Inventory (lance-graph-contract)

Types that EXIST — do NOT re-propose them:

**`grammar/`**: `FailureTicket`, `PartialParse`, `CausalAmbiguity`, `TekamoloSlots`, `TekamoloSlot`, `WechselAmbiguity`, `WechselRole`, `FinnishCase`, `finnish_case_for_suffix`, `NarsInference`, `inference_to_style_cluster`, `ContextChain` (with coherence_at / total_coherence / replay_with_alternative / disambiguate / DisambiguationResult / WeightingKernel), `RoleKey` + 47 `LazyLock<RoleKey>` instances + `Tense` enum + `finnish_case_key / tense_key / nars_inference_key` lookups, **`RoleKey::bind/unbind/recovery_margin`** (slice-masked XOR), **`Vsa10k`** + `VSA_ZERO` + `vsa_xor` + `vsa_similarity`, **`GrammarStyleConfig`** + **`GrammarStyleAwareness`** + `revise_truth` + `ParseOutcome` + `divergence_from`, **`FreeEnergy`** + **`Hypothesis`** + **`Resolution`** (Commit / Epiphany / FailureTicket) + `from_ranked` + thresholds.

**`crystal/`**: `Crystal` trait, `CrystalKind`, `TruthValue`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`, `CrystalFingerprint` (Binary16K / Structured5x5 / Vsa10kI8 / Vsa10kF32 / **Vsa16kF32**), `Structured5x5`, `Quorum5D`, `SentenceCrystal`, `ContextCrystal`, `DocumentCrystal`, `CycleCrystal`, `SessionCrystal`, sandwich layout constants, **`vsa16k_zero` / `binary16k_to_vsa16k_bipolar` / `vsa16k_to_binary16k_threshold` / `vsa16k_bind` / `vsa16k_bundle` / `vsa16k_cosine`** (Click switchboard carrier + algebra, 64 KB, inside-BBB only).

**`cognitive_shader`**: `ShaderDispatch`, `ShaderResonance`, `ShaderBus`, `ShaderCrystal`, `MetaWord` (u32 packed), `MetaFilter`, `ColumnWindow`, `StyleSelector`, `RungLevel`, `EmitMode`, `ShaderSink` trait, `CognitiveShaderDriver` trait.

**`cognitive-shader-driver` BindSpace substrate (2026-04-24)**: `FingerprintColumns.cycle` is now `Box<[f32]>` (Vsa16kF32 carrier, 16_384 f32 per row = 64 KB) — migrated from `Box<[u64]>` (Binary16K). New constant `FLOATS_PER_VSA = 16_384`. New methods: `set_cycle(&[f32])`, `set_cycle_from_bits(&[u64; 256])` (adapter with `binary16k_to_vsa16k_bipolar` projection), `cycle_row() -> &[f32]`. `write_cycle_fingerprint()` API unchanged (takes `&[u64; 256]`), converts internally. `byte_footprint()` for 1 row = 71_774 bytes. Other three planes (content/topic/angle) remain `Box<[u64]>`.

## cognitive-shader-driver Wire Surface (lab-only, post D0.1)

Types live in `crates/cognitive-shader-driver/src/wire.rs` behind `--features serve`:

- **`WireCodecParams`** + `WireLaneWidth {F32x16, U8x64, F64x8, BF16x32}` + `WireDistance {AdcU8, AdcI8}` + `WireRotation {Identity, Hadamard{dim}, Opq{matrix_blob_id, dim}}` + `WireResidualSpec {depth, centroids}` — serde mirrors of the `contract::cam::*` types from PR #225. `TryFrom<WireCodecParams> for CodecParams` runs the precision-ladder validation (OPQ↔BF16x32, overfit guard, pow2 Hadamard) at ingress BEFORE any JIT compile.
- **`WireTensorView {shape, lane_width, bytes_base64}`** + methods `row(&AlignedBytes, usize)` / `subspace(&AlignedBytes, row, k, sub_bytes)` / `row_count()` / `col_count()` / `row_bytes()` / `element_bytes()` / `decode() -> AlignedBytes`. Per Rule E (Wire surface IS the SIMD surface, object-oriented) + Rule A (stdlib `slice::array_windows::<N>` + `ndarray::simd::*` loaders).
- **`AlignedBytes`** — heap-allocated, 64-byte-aligned owned buffer produced once by `WireTensorView::decode` per Rule F (decode at REST ingress, never inside). Safe Send/Sync; `Drop` dealloc with matching layout.
- **`WireCalibrateRequest`** extended with optional `params: Option<WireCodecParams>` + `tensor_view: Option<WireTensorView>` (new path) alongside legacy fields (`num_subspaces` / `num_centroids` / `kmeans_iterations` / `max_rows`) for back-compat.
- **`WireCalibrateResponse`** extended with `kernel_hash: u64` (= `CodecParams::kernel_signature()` of the executed kernel) + `compile_time_us: u64` + `backend: String` ("amx" | "vnni" | "avx512" | "avx2" | "legacy"; **never "scalar"** — iron rule).
- **`WireTensorViewError {Base64, SizeMismatch, ZeroShape}`** — typed decode errors.

**`proprioception`**: 7 `StateAnchor` (Intake/Focused/Rest/Flow/Observer/Balanced/Baseline), 11-D `ProprioceptionAxes`, `StateClassifier` trait, `DefaultClassifier`, `hydrate()` softmax-weighted blend.

**`qualia`**: 17-D `QualiaVector`, `qualia_to_state` projection (17→11).

**`world_map`**: `WorldMapDto`, `WorldMapRenderer` trait, `DefaultRenderer`.

**`world_model`**: `WorldModelDto` with `qualia`, `axes`, `proprioception`, `cycle_fingerprint`, `timestamp`, `cycle_index`, `is_self_recognised()` / `is_liminal()`.

**`container`**: `Container = [u64; 256]` (16Kbit = 2KB), `CogRecord`.

**`property`** (new, SMB domain): `PropertyKind` (Required / Optional / Free), `PropertySpec` (predicate + kind + `CodecRoute` + NARS floor), `PropertySchema` (`&'static`-based, const schemas), `Schema` + `SchemaBuilder` (runtime builder: `.required()` / `.optional()` / `.searchable()` / `.free()` / `.validate()`), `CUSTOMER_SCHEMA`, `INVOICE_SCHEMA`. Maps bardioc Required/Optional/Free to I1 Codec Regime Split (ADR-0002).

**`repository`** (new, SMB domain): `EntityStore` + `EntityWriter` + `Batch` + `EntityKey` — Arrow-agnostic row store contract.

**`mail`** (new, SMB domain): `MailParser` + `ThreadLinker` + `ParseHints` + `AttachmentRef` + `PartRef`.

**`ocr`** (new, SMB domain): `OcrProvider` + `PageImage` + `OcrOpts` + `Bbox` + `BlockKind` + `LayoutBlock`.

**`splat`** (new, 2026-05-06): `SplatChannel` (6 variants: Support / Contradiction / Forecast / Counterfactual / Style / Source), `CamPlaneSplat` (q8 amplitude / width / theta_accept + 16-byte witness identity + 8-byte `replay_ref`), `AwarenessPlane16K` (256 × u64 = 2 KB pressure tile), `SplatPlaneSet` (6 channel planes = 12 KB), `CamSplatCertificate` (q8 pressure measurements + replay decision), `SplatDecision` (Proceed / RequireExactReplay / PrefetchOnly / ScenarioOnly / Drop), `TriadicProjection`, `ReasoningWitness64`. Resolves SPLAT-1 row in entropy ledger (Aspirational → Wired stage 1, entropy 4 → 2). Per `.claude/knowledge/gaussian-splat-cam-plane-workaround.md` PR 1. Plan: `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`.

**`tax`** (new, SMB domain): `TaxEngine` + `TaxPeriod` + `PeriodKind` + `Jurisdiction` + `PostingBatchRef` + `RuleBundle`.

**`reasoning`** (new, SMB domain): `Reasoner` + `ReasoningKind` + `ReasoningContext` + `EvidenceRef` + `Budget`.

**`cam`** (extended by PR #225): `CodecRoute` + `route_tensor` (existing), `CamByte`, `CamStrategy`, `DistanceTableProvider` trait, `CamCodecContract` trait, `IvfContract` trait, plus codec-sweep parameter shape — `LaneWidth` (F32x16 / U8x64 / F64x8 / BF16x32), `Distance` (AdcU8 / AdcI8), `Rotation` (Identity / Hadamard{dim} / Opq{matrix_blob_id, dim}), `ResidualSpec {depth, centroids}`, `CodecParams {subspaces, centroids, residual, lane_width, pre_rotation, distance, calibration_rows, measurement_rows, seed}` with `kernel_signature() -> u64` + `is_matmul_heavy() -> bool`, `CodecParamsBuilder` fluent API, `CodecParamsError {ZeroDimension, OpqRequiresBf16, HadamardDimNotPow2, CalibrationEqualsMeasurement}` — **precision-ladder validation fires at `.build()` BEFORE any JIT compile**.

**`graph_render`** (new, q2 cockpit): `RenderNode`, `RenderEdge`, `InferredConnection`, `Contradiction`, `GraphSnapshot`, `GraphHealth`, `CypherResult`, `CypherValue`, `CypherError`, `EpisodicTrace`, `ShaderEvent`, traits `GraphSnapshotProvider`, `GraphInferenceProvider`, `CypherExecutor`, `EpisodicTraceProvider`, `ShaderEventStream`. Visual render surface for Neo4j/Palantir Gotham cockpit — q2 consumes, lance-graph arigraph produces.

**`a2a_blackboard`**, **`collapse_gate`**, **`exploration`**, **`literal_graph`**, **`orchestration_mode`**, **`jit`**, **`nars`**, **`plan`**, **`orchestration`**, **`thinking`** (36 styles, 6 clusters), **`mul`**, **`sensorium`**, **`high_heel`**.

## Current AriGraph Inventory (lance-graph/src/graph/arigraph/)

4696 LOC shipped, 7 modules:
- `episodic.rs` (210 LOC + unbundle hooks from #208) — `Episode`, `EpisodicMemory`, `unbundle_hardened`, `unbundle_targeted`, `rebundle_cold`, `UnbundleReport`, `RebundleReport`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`
- `triplet_graph.rs` (1064 LOC) — SPO graph, NARS truth, BFS, spatial paths
- `retrieval.rs` (447 LOC) — fingerprint retrieval policies
- `sensorium.rs` (539 LOC) — observation → triplets
- `orchestrator.rs` (1562 LOC) — AriGraph coordinator
- `xai_client.rs` (521 LOC) — xAI enrichment
- `language.rs` (339 LOC) — LM bridge

## Workspace Conventions (locked in CLAUDE.md)

1. **Model policy:** main thread Opus + deep thinking; subagent grindwork → Sonnet; accumulation → Opus; NEVER Haiku.
2. **GitHub reads:** zipball to `/tmp/sources/` + local grep for 3+ reads per repo. MCP only for writes (PR, comments) and single-path reads.
3. **Contract zero-dep invariant:** `lance-graph-contract` has no external crate deps. Do not add any.
4. **Read before Write:** always Read a file before overwriting. Write-over-self without Read is the documented failure mode.
5. **No JSON serialization in types.** Serde stays out of types (debug-only). Wire formats are explicit.
6. **Pumpkin framing** for externally-visible examples (clinical / game-AI disguise for the AGI primitives).

## Active Branches (local at /home/user/lance-graph)

- `claude/teleport-session-setup-wMZfb` — shipped PRs #223 (LAB-ONLY + AGI-as-SoA + I1-I10), #224 (three-part stack + thinking harvest + I11), #225 (codec-sweep plan + D0.6 CodecParamsBuilder + D0.7 precision-ladder validation). Next on this branch: board hygiene + CLAUDE.md tightening; then D0.1 / D0.2 / D0.3 / D0.5 Wire-surface code.
- `claude/deepnsm-grammar-phase1` — Phase 1 PR #210, merged into main.
- `main` — up-to-date post #225.

## Active Integration Plans

- **`elegant-herding-rocket-v1`** — grammar / NARS / crystal / AriGraph (Phase 1 shipped in #210; Phase 2 queued).
- **`codec-sweep-via-lab-infra-v1`** (NEW 2026-04-20) — JIT-first codec sweep through lab endpoint; 1 upfront rebuild, unlimited candidates afterwards. D0.6 + D0.7 shipped in #225.

## Immediate Next Work

**`codec-sweep-via-lab-infra-v1` Phase 0 remainder (next up):**

- **D0.1** Extend `WireCalibrate` + `WireTensorView` (object-oriented, 64-byte-aligned decode) (~180 LOC).
- **D0.2** `WireTokenAgreement` endpoint stub — the I11 cert gate (~160 LOC).
- **D0.3** `WireSweep` streaming endpoint + Lance append (~200 LOC).
- **D0.5** `auto_detect.rs` reading `config.json` for `ModelFingerprint` (~140 LOC).
- Four test gates: `kernel_contract_test`, `amx_dispatch_test` (x86_64), `wire_object_surface_test`, `no_internal_serialisation_test`.

Total Phase 0 remainder: ~680 LOC, one upfront rebuild, surface freezes after.

**`elegant-herding-rocket-v1` Phase 2 (still queued):**

Per `.claude/plans/elegant-herding-rocket-v1.md`:

- **D2** DeepNSM emits `FailureTicket` on low coverage (~150 LOC).
- **D3** Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` (~220 LOC).
- **D5** Markov ±5 bundler with role-indexed VSA (~300 LOC).
- **D7** NARS-tested grammar thinking styles as meta-inference policies (~260 LOC).

Total ~930 LOC, one PR when it ships.

## Deferred (do NOT propose these — they're explicitly parked)

- CausalityFlow TEKAMOLO extension (modal/local/instrument + beneficiary/goal/source, 9 total) — struct change deferred until after Phase 2.
- D8 story-context bridge, D9 ONNX arc export, D10 Animal Farm validation, D11 bundle-perturb emergence — Phase 3/4.
- Named Entity pre-pass (NER) — biggest OSINT blocker, separate PR.
- FP_WORDS = 160 migration (currently 157) — coordinated ndarray change.
- Crystal4K 41:1 persistence compression.
- 200-500 YAML TEKAMOLO templates per language — future training pipeline.
- Python/TypeScript grammar-stack convergence.

## If You're Tempted to Propose Something

Check this file first. Then check the KNOWLEDGE_INDEX.md for which
docs cover your domain. Then load only those docs. If you're still
uncertain whether something exists, grep the actual source before
proposing a new type.

The fastest way to waste 30 turns is to re-invent what's already in
the contract. This file exists to prevent that.

---

## 2026-05-05 — Recently Shipped backfill (PRs #244–#335)

> The "Recently Shipped PRs" table above stops at #243 (last refreshed 2026-04-21). Roughly 50 PRs have merged since. This section retrofits them.

| PR | Merged | Title | What it added (one-line) |
|---|---|---|---|
| **#335** | 2026-05-05 | Claude/thought cycle soa integration plan | Two new knowledge docs: gaussian-splat-cam-plane-workaround + entropy-budget-codebook-superposition |
| **#330** | 2026-05-01 | docs: add Cursor Cloud specific instructions to AGENTS.md | AGENTS.md section: ndarray path, CI commands, fmt-drift inventory, bgz-tensor known failures |
| **#329** | 2026-05-01 | style: apply rustfmt to contract lib.rs + python bindings | Tier-A rustfmt drift in contract lib.rs + python bindings (no semantic change) |
| **#328** | 2026-05-01 | ci(test): add lance-graph-contract unit tests to test gate | `cargo test -p lance-graph-contract --lib` added to CI rust-test.yml |
| **#327** | 2026-05-01 | style(shader-driver): drop double-space alignment in bindspace.rs | Two-line rustfmt drift fix in bindspace.rs introduced by #323 |
| **#326** | 2026-05-01 | fix(sigma-propagation): correct log_norm_growth_negative test seed | Fix broken test from #322: seed at 4·I not I so attenuation reduces log-norm |
| **#325** | 2026-04-30 | chore(toolchain): bump pin 1.94.0 → 1.94.1 | rust-toolchain.toml bumped to 1.94.1 to match sibling repos |
| **#324** | 2026-04-30 | feat(shader-driver): Pillar-7 α-front-to-back-merge sink mode (B5) | AlphaFrontToBack MergeMode + EWA Kerbl-2023 compositing in stage [7] |
| **#323** | 2026-04-30 | feat(cognitive-shader-driver): add Σ-codebook-index column to FingerprintColumns (B2) | FingerprintColumns.sigma u8 column (+1 byte/row, 0.02% overhead) |
| **#322** | 2026-04-30 | feat(contract): promote EWA-Sandwich Σ-propagation kernel to contract (B1) | sigma_propagation.rs: Spd2, ewa_sandwich, log_norm_growth, pillar_5plus_bound |
| **#321** | 2026-04-30 | fix: 10 pre-existing test failures (cosine_distance, arigraph, parse_triplets) | Fixed cosine inversion, Stagnant ordering, quality_window clear, SPO arg order |
| **#320** | 2026-04-30 | ci: declare rustfmt + clippy as pinned-toolchain components | rust-toolchain.toml gets components=[rustfmt,clippy]; fixes CI fmt failure |
| **#319** | 2026-04-30 | fix(transcode): per-month day-validity in parse_iso_date_to_days | Gregorian per-month + leap-year gate before civil_to_days |
| **#316** | 2026-04-30 | feat(transcode): round-3 typed-value resolver for triples_to_batch | triples_to_batch_with_resolver: Currency→f32, Date→Date32, Id→u64 |
| **#315** | 2026-04-30 | ci: revert ndarray-branch pin — PR #115 landed on master | Remove temp ndarray branch pin from rust-test.yml + style.yml |
| **#314** | 2026-04-30 | docs(vision): clear post-F1 staleness items in medcare-foundry-vision.md | §1–§4 DRAFT/forward-tense/PR-N placeholders replaced with real anchors |
| **#313** | 2026-04-30 | feat(transcode): Phase-2-B triples_to_batch (ExpandedTriple → RecordBatch) | ExpandedTriple stream → N-row RecordBatch, lenient-Utf8, 19 tests |
| **#312** | 2026-04-30 | feat(transcode): Phase-2-A pushdown classification (Inexact for recognised filters) | OntologyTableProvider classifies entity_type/predicate/nars filters as Inexact |
| **#311** | 2026-04-30 | docs(vision): mark F1 shipped, restate next deliverable as F2 | medcare-foundry-vision.md §7: F1 parity shipped; F2 RBAC is next posture |
| **#310** | 2026-04-30 | feat(transcode): r2 fixes — typed Arrow + codec_route + partial writes + CachedOntology | Currency/Date/Id→typed Arrow; CachedOntology; validate_route; from_columns_partial |
| **#309** | 2026-04-30 | feat(callcenter::transcode): outer ↔ inner ontology mapper + parallelbetrieb | transcode submodule: zerocopy, cam_pq_decode, spo_filter, ontology_table, parallelbetrieb |
| **#308** | 2026-04-30 | feat: bilingual ontology DTO surface + bgz-tensor workspace inclusion | OntologyDto locale projection; smb_ontology + medcare_ontology; bgz-tensor in workspace |
| **#307** | 2026-04-30 | refactor: dedup FNV-1a — one canonical hash::fnv1a in contract | contract::hash::fnv1a const fn; 8 call sites unified |
| **#306** | 2026-04-30 | feat(G4): verb_table tense modulation (Quirk CGEL grounded) | 12 VerbFamily priors + tense_modifier → 144 unique cell values |
| **#305** | 2026-04-30 | feat(G3): DisambiguateOpts builder + deepnsm caller wiring real fingerprint | DisambiguateOpts builder; sign_binarize_to_binary16k; disambiguator_glue.rs |
| **#304** | 2026-04-30 | feat(G1): Pearl 2³ causality footprint with PAD-model qualia mapping | compute_pearl_mask() 3-bit SPO→CausalMask; PAD qualia footprint replaces 0.5 |
| **#303** | 2026-04-30 | feat(F6): FNV-1a scent with scent_u64 accessor + birthday collision tests | scent() FNV-1a fold-to-u8; scent_u64() full 64-bit digest; 10 tests |
| **#302** | 2026-04-30 | feat(F3): LanceAuditSink with temporal timestamps + full schema round-trip | LanceAuditSink → Lance dataset append; temporal timestamp; O(1) scan_back |
| **#301** | 2026-04-30 | feat(F1): ColumnMaskRewriter full-tree expression walk + Hash UDF hard-fail | Full-tree OptimizerRule covering Filter/Aggregate/Join; NotYetWiredHashUdf |
| **#300** | 2026-04-30 | feat(LF-12): Pipeline DAG with StepId derivation + OrchestrationBridge adapter | PipelineDag Kahn's algorithm; FNV-1a StepId; execute_via_bridge; cycle detection |
| **#299** | 2026-04-29 | revert #294/#295/#296 + clean on top | Reverts #294–#296 confabulation; corrects probe routing (M1/P2-P4 → shader-lab) |
| **#296** | 2026-04-29 | ideas: COCA-Bundle vs Jina-CLAM bucket comparison (**REVERTED by #299**) | IDEAS.md Open entry for COCA/Jina probe (premise flawed; reverted) |
| **#295** | 2026-04-29 | docs: probe-queue data-available followup (**REVERTED by #299**) | bf16-hhtl-terrain.md data-available update (inherited bad routing; reverted) |
| **#294** | 2026-04-29 | docs(probe-queue): honest "needs production data" assessment (**REVERTED by #299**) | bf16-hhtl-terrain.md probe routing table (wrong routing; reverted) |
| **#293** | 2026-04-29 | jc: drain Probe P1 (γ-phase-offset ranking discrimination) → PASS | probe_p1_gamma_phase.rs; P1 PASS: min Spearman ρ=-0.963 (Dupain-Sós) |
| **#292** | 2026-04-29 | docs(board): posthoc-correct PRs #290 #291 via canonical board mechanism | CONJECTURE banners; 5 Open IDEAS.md entries; 2 EPIPHANIES.md entries |
| **#291** | 2026-04-29 | docs: idea journal — proposed application pillars 7/8/9 captured | IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md with Pillars 7/8/9 + PASS criteria |
| **#290** | 2026-04-29 | docs: idea journal — streaming-hydration + fractal-codec captured | IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md separating two ideas |
| **#289** | 2026-04-29 | jc: Pillar 6 — EWA-Sandwich Σ-push-forward | ewa_sandwich.rs; Pillar 6: 10000/10000 PSD-preserving hops; KS bound tightness 1.467× |
| **#288** | 2026-04-29 | jc: Σ-Codebook Viability Probe — rules out CausalEdge64 8→16B expansion | sigma_codebook_probe.rs; R²=0.9949 at k=256; CausalEdge64 stays 8 bytes |
| **#287** | 2026-04-29 | jc: Pillar 5++ — Düker-Zoubouloglou Hilbert-space CLT | dueker_zoubouloglou.rs; Pillar 5++: bundle-of-N in ℝ^16384 → Gaussian limit in ℓ² |
| **#286** | 2026-04-29 | jc: Pillar 5+ — Köstenberger-Stark concentration on Hadamard 2×2 SPD | koestenberger.rs; Pillar 5+: tightness 0.969× on SPD manifold |
| **#285** | 2026-04-29 | Re-land #283 unlocks (Quantum, Disambiguator, verb_table, animal-farm) | Quantum mode, Disambiguator trait, verb_table, animal-farm harness; PhaseTag overflow fix |
| **#284** | 2026-04-29 | Re-land #281 unlocks (PolicyRewriter, DomainProfile) | PolicyRewriter trait, ColumnMaskRewriter, DomainProfile HIPAA thresholds |
| **#282** | 2026-04-29 | fix: Grammar/Markov hardening — slice unification, kernel wiring | CRITICAL slice fix; rotate_right removed; coherence kernel wired; 363 tests |
| **#280** | 2026-04-29 | fix: Foundry hardening — sealed RLS, VecDeque audit, URL decode, Plugin handshake | Sealed RLS default; O(1) audit ring; FNV-1a; URL decode; Plugin handshake; 58 tests |
| **#279** | 2026-04-29 | feat: DeepNSM grammar parser — Markov ±5 bundler, role keys, thinking styles | D0/D2/D3/D4/D5/D6/D7: MarkovBundler, RoleKeySlice, GrammarStyleConfig, 12 YAML configs |
| **#278** | 2026-04-29 | feat: Foundry parity — RLS rewriter, audit log, PostgREST, with_registry | LF-3/DM-7 RLS; LF-90 audit; DM-8 PostgREST stub; LanceMembrane::with_registry; 35 tests |
| **#277** | 2026-04-28 | plan: unified Foundry roadmap for SMB + MedCare (corrects #276 framing) | foundry-roadmap-unified-v1.md; correct scale decisions per FormatBestPractices.md |
| **#276** | 2026-04-28 | plan: Foundry Consumer Parity — shared ontology + UNKNOWN resolutions | foundry-consumer-parity-v1.md; 5 callcenter UNKNOWNs resolved; DM-8 unblocked |
| **#275** | 2026-04-28 | feat: add lancedb 0.27.2 + pin lance =4.0.0 | lancedb=0.27.2 optional dep; lance exact-pinned =4.0.0 for compat |
| **#274** | 2026-04-27 | fix: F-01 identity-tear race + F-08 bounds check + F-09 poison recovery | Single ActorState RwLock; poison recovery; push bounds check |
| **#273** | 2026-04-27 | feat: bump lance 2→4 + datafusion 51→52 + deltalake 0.30→0.31 | Version bumps + API break fixes (invalid_input, DeltaTableProvider migration) |
| **#272** | 2026-04-27 | feat: Column H — EntityTypeId on BindSpace (Phase 1 of 4) | EntityTypeId u16 on BindSpace; push_typed(); 1-based index; 4 tests |
| **#271** | 2026-04-27 | plan: BindSpace Columns E/F/G/H — 4→8 SoA integration plan | bindspace-columns-v1.md; 24 deliverables; 7 SOUND / 7 CAUTION / 0 WRONG |
| **#270** | 2026-04-26 | ci: remove typos spell-check job (too many false positives) | Removed crate-ci/typos from style.yml; cargo fmt --check remains |
| **#269** | 2026-04-26 | feat: Distance trait + SIMD Hamming/cosine wiring + PaletteDistanceTable + Dockerfile docs | Distance trait; SIMD Hamming/cosine wiring; PaletteDistanceTable 128KB; Dockerfile.md |


---

## 2026-05-07 — Append: lance-graph-ontology shipped (commit 4cf9a26, branch claude/create-graph-ontology-crate-gkuJG)

(Per APPEND-ONLY rule: this dated annotation augments the "Recently Shipped PRs" table and "Current Contract Inventory" snapshot above. Treat the row below as the new top-of-table entry; treat the inventory paragraph below as a new top-of-inventory entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **(open / pending merge)** | *(open)* | feat(lance-graph-ontology): scaffold OGIT-canonical ontology spine | New workspace member `crates/lance-graph-ontology/` (~3000 LOC, 28 tests = 16 inline + 12 integration). Phases 3-5 of the v4 plan: scaffold + TTL hydration + tenant bridges. Public surface: `OntologyRegistry`, `NamespaceBridge` trait, `NamespaceId`, `OgitUri`, `SchemaPtr`, `SchemaKind`, `MappingProposal`, `MappingProposalKind`, `MappingRow`, `MappingHandle`, `HydrationReport`, `HydrationFailure`, `BridgeError`, `Error`, `SchemaSource` trait, `EntityRef`, `EdgeRef`, `OntologyAssembler`, `SemanticTypeMap`, `TtlSource`. Default tenant bridges: `bridges::WoaBridge`, `bridges::MedcareBridge`, `bridges::OgitBridge`. Feature-gated `lance_cache::LanceWriter` (under `lance-cache` feature, gated to keep zero-protoc compile path). Builds on prior commit `edef321` (recon + SPO-1 decision: federated two-layer cache, Option B). |

### Current Contract Inventory — new entry

**`lance-graph-ontology`** (new crate, 2026-05-07): consolidates per-tenant bridge multiplication into one ontology spine. OGIT becomes the canonical TTL ontology source; Lance is the (feature-gated) runtime dictionary cache; tenant bridges become thin scoped views over the shared registry. Public types: `OntologyRegistry`, `NamespaceBridge` trait, `NamespaceId`, `OgitUri`, `SchemaPtr`, `SchemaKind`, `MappingProposal`, `MappingProposalKind`, `MappingRow`, `MappingHandle`, `HydrationReport`, `HydrationFailure`, `BridgeError`, `Error`, `SchemaSource` trait, `EntityRef`, `EdgeRef`, `OntologyAssembler`, `SemanticTypeMap`, `TtlSource`. Default tenant bridges: `bridges::WoaBridge`, `bridges::MedcareBridge`, `bridges::OgitBridge`. 28 tests passing (16 inline + 12 integration). Feature-gated Lance persistence under `lance-cache` (kept off by default so the crate compiles without `protoc`, which `lance-encoding`'s build-script requires). Branch `claude/create-graph-ontology-crate-gkuJG`; commit `4cf9a26`; prior recon + decision in `edef321` (`.claude/RECON_ONTOLOGY_CRATE.md`, `.claude/DECISION_SPO_ARIGRAPH.md`).

---

## 2026-05-07 — Sprint-2: Unified OGIT Architecture synthesis (recently shipped — documentation tier)

> **APPEND-ONLY annotation.** Per the governance rule above, this section augments — does not edit — prior content. Treat as the new top-of-state. Branch: `claude/unified-ogit-architecture-synthesis`.
>
> Sprint-2 was a 12-agent + meta-review coordinated burst. **Zero code changes; documentation tier only.** It captures 16 turns of architectural conversation (2026-05-07) as a unified pattern-recognition framework over already-shipped substrate, plus three concrete next-PR sub-plans and one proof-of-vision plan. The dominant finding: ~80% of the "unified OGIT architecture" we were about to design is **already shipped**; recognising this drops architecture entropy by **−11** with no code written.

### Sprint-2 deliverables (12 workers + meta)

**New plan-docs (4)**

| File | Size | Worker | Purpose |
|---|---|---|---|
| `.claude/plans/unified-ogit-architecture-v1.md` | ~22 KB | W1 | Master synthesis: 15 patterns (A-O) + Tier 0-4 stack + proof-of-vision. Canonical reference for the unified OGIT architecture. |
| `.claude/plans/ogit-g-context-bundle-v1.md` | ~10 KB | W10 | Tier-1 sub-plan: G-overlay wiring; Patterns A (G-slot) + B (context-bundle) + C (per-cycle cascade). |
| `.claude/plans/compile-time-consumer-binding-v1.md` | ~10 KB | W11 | Tier-2 sub-plan: compile-time consumer binding + ractor; Patterns E (consumer-binding) + F (zero-overhead actor seam). |
| `.claude/plans/anatomy-realtime-v1.md` | ~12 KB | W12 | Proof-of-vision: north-star realtime anatomy demo end-to-end across the unified stack. |

**New knowledge doc (1)**

| File | Size | Worker | Purpose |
|---|---|---|---|
| `.claude/knowledge/tier-0-pattern-recognition.md` | ~13 KB | W2 | File→pattern map covering ~30 already-shipped files. Read this FIRST in any future session that touches OGIT architecture to avoid the Discovery-Loop anti-pattern. |

**Board appends (5, append-only governance preserved)**

| File | Worker | Append summary |
|---|---|---|
| `.claude/patterns.md` | W3 | Appended **Pattern Recognition Framework**: 15 patterns A-O catalogued + new Anti-Pattern **"Designing What's Already Built"**. |
| `.claude/board/EPIPHANIES.md` | W4 | Appended **17 architectural epiphanies**: E-OGIT-1 through E-RECOGNITION-OVER-DESIGN-17. |
| `.claude/board/TECH_DEBT.md` | W5 | Appended **11 TD entries**: TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11, each with effort estimate. |
| `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` | W6 | Appended **5 row reframes** (THINK-1 5→3, HEEL-1 4→2, ADJ-THINK-1 4→2, CRYSTAL-1 4→2, CAM-DIST-1 3→2) + 15-pattern absorption table. **Net entropy delta: −11**. |
| `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` | W7 | Appended RECOGNITION-1 meta-finding row + Anti-Pattern surfaced ("Designing What's Already Built"). |

**Index update (1)**

| File | Worker | Update |
|---|---|---|
| `.claude/board/INTEGRATION_PLANS.md` | W8 | Indexed the 4 new plan-docs (W1 master + W10 + W11 + W12). |

**Sprint coordination (CCA2A pattern, `/sprint-log-2`)**

- `.claude/board/sprint-log-2/SPRINT_LOG.md` — master coordination index.
- `.claude/board/sprint-log-2/agents/agent-W{1..12}.md` — per-agent append-only logs (12 files).
- `.claude/board/sprint-log-2/meta-1-review.md` — meta agent brutally-honest review.
- `.claude/board/sprint-log-2/agents/agent-W9.md` — this worker's handover log.

### Aggregate impact

- **15 architectural patterns (A-O)** named and catalogued.
- **~80% of the "unified OGIT architecture" is recognised as already shipped** — Patterns H, M, N, O at substrate level; Pattern F shape proven by gRPC.
- **~20% genuinely new wiring work** captured as TECH_DEBT entries with effort estimates (TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11).
- **Net entropy reduction from recognition alone: −11** (no code changes; 5 row reframes + 15-pattern absorption).
- **Totals shipped this sprint:** 4 new plan-docs + 1 knowledge doc + 5 board appends + 1 index update + sprint-log-2 scaffolding (1 master + 12 agent logs + 1 meta review).

### What this enables

Future sessions that read `.claude/knowledge/tier-0-pattern-recognition.md` first will avoid the **Discovery-Loop anti-pattern at architectural scale** — the same anti-pattern `.claude/patterns.md` warns about at cycle level (proposing concepts that already exist in workspace).

The master plan-doc `.claude/plans/unified-ogit-architecture-v1.md` provides the canonical reference for the unified OGIT architecture. The three sub-plans give concrete next-PR scope:

- **Tier 1 next PR** — `.claude/plans/ogit-g-context-bundle-v1.md` (G-overlay wiring).
- **Tier 2 next PR** — `.claude/plans/compile-time-consumer-binding-v1.md` (compile-time consumer binding + ractor).
- **Proof of vision** — `.claude/plans/anatomy-realtime-v1.md` (north-star demo).

### Cross-references

- All sister deliverables listed above (W1–W12 + meta).
- 16-turn architectural conversation (2026-05-07).
- Pre-existing plans absorbed into the unified framework: `lance-graph-ontology-v5` (PR #355), `palantir-parity-cascade-v2` (PR #353), `ogit-cascade-supabase-callcenter-v1` (PR #355).
- Substrate already shipped (Patterns H/M/N/O): see "Current Contract Inventory" and "Current AriGraph Inventory" sections above; especially `lance-graph-ontology` (commit `4cf9a26`), `cognitive-shader-driver` BindSpace SoA (PR #204+ thru #323), `crystal/` Vsa16kF32 sandwich (PR #208/#209), `cam/` codec cascade (PR #225).

### Brutally-honest self-review (W9)

- **In scope:** append-only update to `LATEST_STATE.md`. Did not edit any prior content. Verified the file's existing closing section (`2026-05-07 — Append: lance-graph-ontology shipped`) is preserved verbatim.
- **Risk:** the "~80% already shipped" claim is W1/W2's recognition assertion, not independently re-verified by W9. This section reports it as the synthesis output; the canonical evidence lives in `tier-0-pattern-recognition.md` (W2) and the entropy ledger reframe rows (W6).
- **Governance:** append-only preserved. No deletions. No edits to the prior `## 2026-05-07 — Append: lance-graph-ontology shipped` section. Section heading matches the spec exactly.
- **What this section does NOT do:** it does not edit the top-of-file "Last updated" line (would violate append-only); it does not edit the "Recently Shipped PRs" table (Sprint-2 shipped no PRs); it does not edit "Active Branches" (Sprint-2 is documentation tier on a branch that has not yet merged).

## 2026-05-12 — Sprint-3: Tier-1 Implementation Specs (PR #360 + #361 + post-#360 substrate sweep)

**PR #360** (sprint-3 main): 11 PR-X-1 specs covering 7 design-phase patterns A/B/C/D/E/F/J + 3 trivia closures + supporting docs. ~140 KB across `.claude/specs/`. Engineer can now execute Tier-1 in ~6 working days parallelized (per W10 sequencing).

**PR #361** (post-#360 corrections): PR-F-1 supervisor must skip inert bundles (DOLCE/FMA have consumer_pointer=None by design); PR-E-1 build script must emit data-only (no consumer crate refs) to avoid Cargo dependency cycle. Both fixed via append-only correction sections; inventory-crate self-registration recommended for actor binding.

**Post-#360 substrate-recognition sweep** (this PR): 3 of 11 specs reclassified PARTIALLY SHIPPED:
- Pattern A: SchemaPtr.ontology_context_id + NamespaceRegistry::seed_defaults already ship; PR-A-1 reduces to ~150 LOC / 1 day
- Pattern C: BridgeFromRegistry + 3 impls + woa-rs#2 + medcare-rs#110 consumer scaffolds already ship; PR-C-1 reduces to ~80 LOC / ½ day
- Pattern D: parse_ttl_directory_with_provenance + attach_provenance already ship; PR-D-1 reduces to ~250 LOC / 1-2 days (OWL/RDF-XML adapter only)

Compressed sprint-3 critical path: ~6 days → ~3-4 days parallelized. The genuinely-new ~5-pattern set is B (context bundle), E (manifest-modules), F (ractor port), G (inheritance protocol), J (INT4-32D atoms).

### New knowledge docs (sprint-3 substrate-sweep)

- `.claude/knowledge/pattern-recognition-cross-source.md` — A-O ↔ Pillars 0-4 ↔ `.grok/` ↔ shipped substrate matrix (4 parallel taxonomies cross-referenced)
- `.claude/knowledge/cca2a-sprint-prompt-template.md` — substrate-grep checklist + wrong-repo guardrail + pattern-letter discipline (mandatory pre-spawn template for future sprints)

### Anti-Pattern recurrence captured

The "Designing What's Already Built" anti-pattern (introduced PR #358) recurred in sprint-3's own design (PR-A-1/PR-C-1/PR-D-1 over-scoped because they didn't sweep post-#355 substrate). The correction PR formalizes the substrate-grep checklist as mandatory before any new spec.

### Recurring failure mode: wrong-repo error

Sprint-2 W7 → ndarray; sprint-3 W9 → ada-consciousness. Both corrected via main-thread pygithub recovery. Wrong-repo guardrail snippet now mandatory in every worker prompt (per `.claude/knowledge/cca2a-sprint-prompt-template.md`).

### Cross-references

- `.claude/specs/sprint-3-execution-plan.md` (W1 master)
- `.claude/specs/sprint-3-pr-graph.md` (W10 sequencing — to be updated for compressed timeline)
- `.claude/specs/pr-{a,b,c,d,e,f,j}-1-*.md` (11 PR-ready specs; A/C/D have appended CORRECTION sections)
- `.claude/specs/consumer-crate-template.md` (W8; re-target from hubspo-rs hypothetical to woa-rs/medcare-rs precedent)
- `.claude/specs/ogit-g-smoke-test.md` (W11 validation)
- `.claude/specs/trivia-prs-bundle.md` (W12 — 3 quick wins parallel-shippable)
- `.claude/board/sprint-log-3/{SPRINT_LOG.md,agents/agent-W1..W12.md,meta-1-review.md,sprint-summary.md}`

PR sequence: #360 → #361 → post-#360 substrate-sweep (this PR).
