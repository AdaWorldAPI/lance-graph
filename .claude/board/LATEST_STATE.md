# LATEST_STATE — What Just Shipped (read this FIRST)

> **Auto-injected at session start via SessionStart hook.**
> Updated after every merged PR.
> **Last updated:** 2026-04-21 post PR #243 (D5+D7 + categorical-algebraic inference architecture).
>
> Purpose: prevent new sessions from hallucinating structure that
> already exists or proposing features already shipped. Read this
> BEFORE proposing any grammar/crystal/contract changes.

---

## Recently Shipped PRs (reverse chronological)

| PR | Merged | Title | What it added |
|---|---|---|---|
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

**`cam`** (extended by PR #225): `CodecRoute` + `route_tensor` (existing), `CamByte`, `CamStrategy`, `DistanceTableProvider` trait, `CamCodecContract` trait, `IvfContract` trait, plus codec-sweep parameter shape — `LaneWidth` (F32x16 / U8x64 / F64x8 / BF16x32), `Distance` (AdcU8 / AdcI8), `Rotation` (Identity / Hadamard{dim} / Opq{matrix_blob_id, dim}), `ResidualSpec {depth, centroids}`, `CodecParams {subspaces, centroids, residual, lane_width, pre_rotation, distance, calibration_rows, measurement_rows, seed}` with `kernel_signature() -> u64` + `is_matmul_heavy() -> bool`, `CodecParamsBuilder` fluent API, `CodecParamsError {ZeroDimension, OpqRequiresBf16, HadamardDimNotPow2, CalibrationEqualsMeasurement}` — **precision-ladder validation fires at `.build()` BEFORE any JIT compile**.

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
