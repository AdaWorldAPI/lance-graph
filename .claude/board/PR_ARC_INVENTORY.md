# PR Arc — Architectural Decision History

> **Auto-loaded at session start.** Every merged PR, its meta, and
> the decisions it locked in. Read BEFORE proposing anything — a new
> proposal that contradicts a decision in this arc is a 30-turn
> rediscovery tax waiting to happen.
>
> ## APPEND-ONLY RULE (MANDATORY)
>
> 1. **New PRs PREPEND** a new section at the top (most-recent first).
> 2. **Old PR sections are IMMUTABLE HISTORY.** Never rewrite or
>    delete a past PR's Added / Locked / Deferred / Docs entries.
> 3. **The ONE exception: Confidence annotations.** Each PR section
>    may have a `**Confidence (YYYY-MM-DD):**` line that IS updatable.
>    Use it to record: "working", "partial", "superseded by PR #N",
>    "broken — see PR #N for fix". This is the only mutable field.
> 4. **Corrections append.** If a Locked claim turns out wrong,
>    append a `**Correction (YYYY-MM-DD from PR #N):**` line to the
>    same entry — do not edit the original Locked line. Both stay.
> 5. **Reversals are their own PR entry.** If a later PR explicitly
>    undoes a decision, the later entry documents the reversal; the
>    earlier entry's Confidence line references it. Both remain in
>    the arc.
>
> The arc is the historical record. Rewriting it destroys the
> "why was this decided that way" context that prevents future
> rediscovery. Every entry stays.
>
> **Format:** reverse chronological. Each PR carries:
> - **Added** — new types / modules / LOC (immutable)
> - **Locked** — conventions / invariants / patterns (immutable)
> - **Deferred** — explicit parks (immutable)
> - **Docs** — knowledge files produced (immutable)
> - **Confidence (YYYY-MM-DD):** — the ONLY mutable field

---

## #243 — D5+D7 categorical-algebraic inference architecture (2026-04-21)

**Confidence (2026-04-21):** Working. 175/175 contract, 63/63 deepnsm (grammar-10k).

**Added:**
- `contract::grammar::thinking_styles` — `GrammarStyleConfig`, `GrammarStyleAwareness` (NARS-revised `HashMap<ParamKey, TruthValue>`), `revise_truth`, `ParseOutcome` (5 polarities), `divergence_from(prior)` (KL term). 490 LOC, 12 tests.
- `contract::grammar::free_energy` — `FreeEnergy` (likelihood + KL → total), `Hypothesis` (role fillers + Pearl 2³ mask), `Resolution` (Commit / Epiphany / FailureTicket), `from_ranked` classifier, `HOMEOSTASIS_FLOOR` / `EPIPHANY_MARGIN` / `FAILURE_CEILING`. 347 LOC, 7 tests.
- `contract::grammar::role_keys` — `RoleKey::bind/unbind/recovery_margin` (slice-masked XOR), `Vsa10k` type alias, `VSA_ZERO`, `vsa_xor`, `vsa_similarity`, `word_slice_mask` helper. +295 LOC, +14 tests (5-role lossless superposition verified).
- `deepnsm::content_fp` — 10K-dim content fingerprints from COCA vocab ranks (SplitMix64). 98 LOC, 5 tests. Feature-gated: `grammar-10k`.
- `deepnsm::markov_bundle` — `MarkovBundler` (±5 ring buffer, role-key bind, braiding via `vsa_permute`, XOR-superpose, `WeightingKernel`). 250 LOC, 8 tests.
- `deepnsm::trajectory` — `Trajectory` (Think carrier): `role_bundle`, `mean_recovery_margin`, `ambient_similarity`, `free_energy`, `resolve`. 298 LOC, 4 tests.
- `CLAUDE.md` § The Click (P-1): top-of-file architecture diagram + 3 simplicity invariants + shader-cant-resist + thinking-is-a-struct + tissue-not-storage + grammar-of-awareness + 2 litmus tests.
- `.claude/plans/categorical-algebraic-inference-v1.md` (496 lines): meta-architecture proving 5 operations are 1 algebraic substrate, grounded in 8-paper proof chain.

**Locked:**
- `RoleKey::bind` is slice-masked XOR (categorically optimal per Shaw 2501.05368 Kan extension theorem). Not a design choice — a theorem consequence.
- `FreeEnergy = (1 - likelihood) + KL` where likelihood = mean role recovery margin, KL = `awareness.divergence_from(prior)`. Three thresholds: F<0.2 commit, ΔF<0.05 epiphany, F>0.8 escalate.
- NARS revision asymptotes at φ-1 ≈ 0.618 (golden ratio confidence ceiling). Feature, not bug. Permanent epistemic humility.
- Markov = XOR of braided sentence VSAs. No HMM. No transition matrix. No weights.
- Thinking is a struct (not a service, not a function). The DTO carries cognition as identity.
- AriGraph/episodic/CAM-PQ are thinking tissue (organs of Think), not storage services.
- Object-does-the-work test: free function on carrier's state = reject. Method on carrier = accept.
- Five-lens test: every new type serves Parsing / Free-Energy / NARS / Memory / Awareness or is drift.

**Deferred:**
- Steps 4-8 of the 8-step wiring sequence (pipeline, AriGraph commit, global context, awareness revision, KL feedback). Three PRs to close the loop.
- D10 Animal Farm benchmark (the AGI test: chapter-10 accuracy > chapter-1 accuracy).
- Cross-lingual bundling (needs parallel corpora).
- ONNX arc model (D9, D11).

**Docs:**
- `.claude/knowledge/paper-landscape-grammar-parsing.md` — 14 papers in 3 tiers.
- `.claude/knowledge/session-2026-04-21-categorical-click.md` — session handover with 12 critical insights + 7 anti-patterns.
- `.claude/board/EPIPHANIES.md` — 12 new epiphanies with "why this dilutes" warnings.
- `.claude/board/INTEGRATION_PLANS.md` — `categorical-algebraic-inference-v1` entry prepended.

---

## #225 — Codec-sweep plan + D0.6/D0.7 CodecParams types (merged 2026-04-20)

**Confidence (2026-04-20):** Working. 147/147 contract suite passing (133 prior + 14 new).

**Added:**
- `.claude/plans/codec-sweep-via-lab-infra-v1.md` (~1,800 lines) — JIT-first codec sweep plan operationalising PR #220's "What's Needed to Fix" list through the lab endpoint. One upfront Wire-surface rebuild, unlimited JIT-kernel candidates afterwards.
- 9 starter YAML configs under Appendix A (controls + four #220 fixes + composite + cross-product grid).
- `.claude/board/INTEGRATION_PLANS.md` — prepended `codec-sweep-via-lab-infra-v1` entry per APPEND-ONLY rule.
- `contract::cam::LaneWidth` {F32x16, U8x64, F64x8, BF16x32} — mirrors `ndarray::simd::*` lane types.
- `contract::cam::Distance` {AdcU8, AdcI8} — split per CODING_PRACTICES gap 5 (sign-handling / bipolar cancellation).
- `contract::cam::Rotation` {Identity, Hadamard{dim}, Opq{matrix_blob_id, dim}} + `is_matmul()`.
- `contract::cam::ResidualSpec` {depth, centroids}.
- `contract::cam::CodecParams` + `kernel_signature() -> u64` + `is_matmul_heavy() -> bool`.
- `contract::cam::CodecParamsBuilder` — fluent API (CODING_PRACTICES gap 3 remediation).
- `contract::cam::CodecParamsError` {ZeroDimension, OpqRequiresBf16, HadamardDimNotPow2, CalibrationEqualsMeasurement}.
- 14 new `codec_params_tests` covering builder defaults + each validation + kernel_signature stability + matmul-heavy detection.

**Locked:**
- **Six rules A-F bind every JIT-emitted kernel in the codec sweep:**
  - Rule A: tensor access via stdlib `slice::array_windows::<N>()` + `ndarray::simd::*` loaders
  - Rule B: SIMD exclusively via `ndarray::simd::*` / `simd_amx::*` / `hpc::amx_matmul::*` / `hpc::simd_caps::*`
  - Rule C: polyfill hierarchy AMX → AVX-512 VNNI → AVX-512 baseline → AVX-2, **no consumer-visible scalar tier**
  - Rule D: JSON / YAML / REST configuration only
  - Rule E: Wire surface IS the SIMD surface (object-oriented, `LaneWidth` explicit, methods not scalar bags, 64-byte-aligned decode)
  - Rule F: **Serialisation at REST edge only; never inside**
- **Iron rule:** SoA never scalarises without ndarray. If a kernel runs scalar on the SoA path, the SoA invariant is broken.
- **Intel AMX** (not Apple) — `ndarray::simd_amx::amx_available()` + `ndarray::hpc::amx_matmul::{tile_dpbusd, tile_dpbf16ps, vnni_pack_bf16}` on Sapphire Rapids+ via stable inline asm (rust-lang #126622 keeps AMX intrinsics nightly).
- **Precision-ladder validation fires BEFORE JIT compile.** OPQ rotation requires BF16x32 lane. Hadamard dim must be 2^k.
- **Overfit guard typed-error-rejects the PR #219 pattern.** `CalibrationEqualsMeasurement` refuses to emit ICC when `calibration_rows == measurement_rows`.
- **Kernel signature excludes seed.** Seed changes calibration sample but not IR — cached kernels stay hot across seeds.
- **Zero ndarray changes.** "Everything the sweep needs is already in ndarray" — user directive, enforced.
- **Zero serde in the contract.** YAML/JSON deserialisation belongs to the consumer crate.

**Deferred:**
- D0.1 (`WireCalibrate` extension), D0.2 (`WireTokenAgreement`), D0.3 (`WireSweep`), D0.5 (`auto_detect`) — next PR.
- D1.1-D1.3 (JIT codec kernels), D2.1-D2.3 (token-agreement harness), D3.1-D3.2 (sweep driver + Lance logger), D4.1-D4.2 (frontier analysis), D5 (graduation bridge) — later PRs.

**Docs:**
- Plan references: `.claude/knowledge/lab-vs-canonical-surface.md`, `cam-pq-unified-pipeline.md`, `codec-findings-2026-04-20.md`, `rotation_vs_error_correction.md`, `encoding-ecosystem.md`.

**Decisions for future PRs to respect:**
- When testing a codec candidate: reconstruction error → reconstruction ICC on held-out rows → **token agreement**. The cert gate is token agreement, not synthetic ICC (PR #219 lesson).
- Adding a new codec candidate is authoring a YAML file. Zero Rust changes. Zero rebuilds.
- `CodecParams::kernel_signature` is the JIT cache key. Adding unrelated fields to `CodecParams` must NOT change what goes into the signature.

---

## #224 — Lab = API+Planner+JIT, thinking harvest, I11 measurability (merged 2026-04-20)

**Confidence (2026-04-20):** Working. Docs-only PR, no build impact.

**Added:**
- `.claude/knowledge/lab-vs-canonical-surface.md` extended with three load-bearing sections:
  - "Why the Lab Surface Exists (positive purpose)" — three-part stack (API + Planner + JIT), not just quarantine scaffolding.
  - "The third purpose — thinking harvest (the AGI magic bullet)" — REST/Cypher → `{rows, thinking_trace}` externalises planner's 36-style / 13-verb / NARS trace for log/replay/revision.
  - I11 invariant: measurable stack, not a black box. Every layer L0→L4 emits harvest-ready trace.

**Locked:**
- **Codec cert is token agreement, not synthetic ICC.** PR #219's 0.9998 was overfit-on-training; PR #220's 0.195 was reconstruction-only. Real cert gate is decoded codec's top-k tokens matching Passthrough.
- **Three-part lab stack:** REST/gRPC API (curl entry, no rebuild) × Planner (real dispatch path, not toy bench) × JIT (runtime kernel swap, no relink). All three together = unlimited candidates measured via real dispatch.
- **Thinking harvest = AGI magic bullet.** An AGI that cannot observe its own reasoning cannot revise it. REST/Cypher injection + JIT + planner closes that loop outside the binary.
- **I11 — measurable stack.** Every layer's trace is harvest-ready through the lab surface. Proposed changes that shrink trace for perf/simplicity are rejected.
- **Two allowed edges** for serialisation: REST/gRPC ingress (JSON/protobuf in, once per request), REST/gRPC response (JSON/protobuf out, once per response). **No internal serde between layers.** Lance append is the one persistent egress.

**Deferred:** Actual ONNX story-arc training, actual token-agreement harness implementation — all after Phase 0 Wire surface hardens.

**Docs:**
- `lab-vs-canonical-surface.md` — now the canonical cross-cutting invariant doc (I1-I11 + six rules A-F in PR #225).

**Decisions for future PRs to respect:**
- Never propose a codec cert claim based on reconstruction ICC alone. Always measure token agreement.
- The three-part stack is the iteration testbed AND the observability port — both uses share the same binary.

---

## #223 — LAB-ONLY firewall + AGI-as-SoA + I1-I10 (merged 2026-04-20)

**Confidence (2026-04-20):** Working. Docs-only PR, no build impact.

**Added:**
- `.claude/knowledge/lab-vs-canonical-surface.md` (NEW) — MANDATORY pre-read for REST/gRPC/Wire DTO/OrchestrationBridge/codec-research work. Three sections:
  - The One-Line Rule: `cognitive-shader-driver` IS the unified API; Wire DTOs are lab quarantine.
  - AGI = (topic, angle, thinking, planner) = struct-of-arrays consuming `cognitive-shader-driver`. The four AGI axes map to the four BindSpace SoA columns.
  - 10 cross-cutting architecture invariants I1-I10 (below).
- `CLAUDE.md` P0 rule: read this doc BEFORE any REST/gRPC/Wire DTO/endpoint/shader-lab work.

**Locked (Invariants I1-I10):**
- **I1** — BindSpace read-only `Arc<[u64; 256 * N]>`; writes cross the CollapseGate airgap via `MergeMode::{Xor, Bundle}`.
- **I2** — Canonical SIMD import is `ndarray::simd::*`. Never `ndarray::hpc::simd_avxNNN::*` reach-through.
- **I3** — Layer temporal budgets: L0 sub-ns, L1 ns zero-copy, L2 ns, L3 µs, L4 ms.
- **I4** — Temperature hierarchy Hot (BindSpace HDR) → Warm (CAM-PQ) → Cold (DataFusion scalar joins) → Frozen (metadata). Cold narrows first; HDR semirings fire only on survivors.
- **I5** — Thinking IS an `AdjacencyStore`. 36 styles at τ-prefix 0x0D. One engine, two graphs.
- **I6** — Weights are seeds. GGUF hydrates into palette + `Fingerprint<256>` + FisherZTable + holographic residual + `CausalEdge64`. Inference = Hamming cascade + palette lookup, no matmul.
- **I7** — Per-cycle cascade budget ~2.3ms/1M rows with monotone narrowing (topic → angle → causality → qualia → exact).
- **I8** — 4096 address surface = 16 prefix × 256 slots. `Addr(u16)`. Prefix `0x0D` is thinking styles.
- **I9** — Three DTO families (doctrinal, not yet shipped): StreamDto (pre-parse) / ResonanceDto (active sweep) / BusDto (post-collapse). Field ≠ sweep ≠ bus.
- **I10** — HEEL / HIP / BRANCH / TWIG / LEAF progressive precision hierarchy. bgz17 IS HEEL — not LEAF identity.

**Locked (framing):**
- Claude Code sessions in this workspace **never** write a parallel `struct Agi { topic, angle, thinking, planner }`. Those ARE the BindSpace SoA columns. Wrapping them in a new struct breaks SIMD sweep.
- Extend by **column**, not by layer. New AGI capability = new BindSpace column, not a new trait / endpoint / DTO family.
- REST endpoints (`/v1/shader/*`) are LAB-ONLY. Adding `/v1/shader/<new>` is the Kahneman-Tversky System-1 easy path; extending `OrchestrationBridge` / adding a `StepDomain` variant is the System-2 correct move.

**Deferred:** Thinking harvest subsection, I11 measurability invariant, codec-sweep plan — PR #224 / #225.

**Docs:**
- `lab-vs-canonical-surface.md` (NEW, 429 lines).

**Decisions for future PRs to respect:**
- Never add a per-op REST endpoint as "the API." The canonical consumer surface is `UnifiedStep` via `OrchestrationBridge`.
- Never bypass `ndarray::simd::*` to reach `hpc::simd_avxNNN::*`. That's a private backend, not a consumer surface.
- AGI is NOT a new crate. AGI is the already-shipped BindSpace + ShaderDriver + OrchestrationBridge interpreted through the four-axis lens.

---

## #210 — Phase 1 grammar + knowledge docs (merged 2026-04-19)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `contract::grammar::context_chain` — coherence_at / total_coherence / replay_with_alternative / disambiguate / DisambiguationResult / WeightingKernel {Uniform, MexicanHat, Gaussian} (+396 LOC, 8 tests).
- `contract::grammar::role_keys` — 47 canonical role keys addressed as contiguous `[start:stop]` slices over 10,000 VSA dims. FNV-64 + per-dim LCG deterministic generation. `Tense` enum (12 variants). `finnish_case_key / tense_key / nars_inference_key` lookups (+404 LOC, 7 tests).

**Locked:**
- **Role-key VSA addressing uses contiguous slices**, not scattered bits. Subject=[0..2000), Predicate=[2000..4000), Object=[4000..6000), Modifier=[6000..7500), Context=[7500..9000), TEKAMOLO slots=[9000..9900), Finnish cases=[9840..9910), tenses=[9910..9970), NARS inferences=[9970..10000).
- **All role-key slices are disjoint**; binding into one slice does not contaminate another.
- **ContextChain coherence is Hamming-based** on the Binary16K variant, graceful zero-score on other variants (zero-dep constraint).
- **Mexican-hat weight:** `(1 - 2x^2) · exp(-2x^2)` where `x = d / MARKOV_RADIUS`. Monotone on d=0..5.
- **DISAMBIGUATION_MARGIN_THRESHOLD = 0.1** — below this the `escalate_to_llm` flag fires.

**Deferred:**
- CausalityFlow 3→9 slot extension (modal/local/instrument + beneficiary/goal/source).
- Phase 2 work: D2 FailureTicket emission, D3 Triangle bridge, D5 Markov bundler, D7 grammar thinking styles.
- All of Phase 3/4.

**Docs:**
- `grammar-landscape.md` (429 lines)
- `linguistic-epiphanies-2026-04-19.md` (466 lines, E13-E27)
- `fractal-codec-argmax-regime.md` (256 lines, orthogonal thread)

**Decisions for future PRs to respect:**
- Finnish object marking uses Nominative/Genitive/Partitive, NOT Latinate Accusative (except personal pronouns).
- Russian 6 cases include Instrumental (not omitted).
- Each language gets its native case terminology.
- Never spawn Haiku subagents.
- Explore subagents → Sonnet, `general-purpose` grindwork → Sonnet, accumulation → Opus.

---

## #209 — sandwich layout + bipolar cells (merged 2026-04-19)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `CrystalFingerprint::Structured5x5` uses sandwich layout: 3,125 cells in middle (dims 3437..6562), 5 quorum floats (6562..6567), quorum sentinel (6567), plus leading/trailing role-binding space.
- Bipolar cell encoding: `u8 0..=255 → f32 [-1, 1]` via `v/127.5 - 1.0`.
- Lossless bundle/unbundle between Structured5x5 ↔ Vsa10kF32 sandwich.
- Codex-review fixes: Binary16K aliasing, i8 /128 clamp, `quorum: None` sentinel.

**Locked:**
- **VSA operations stay in `ndarray::hpc::vsa`** (bind, unbind, bundle, permute, similarity, hamming, sequence, clean). DO NOT duplicate in contract.
- **10K f32 Vsa10kF32 (40 KB) is lossless under linear sum**, not a wire-only format; lancedb natively handles 10K VSA.
- **Signed 5^5 bipolar is lossless**; unsigned / bitpacked binary is lossy via saturation.
- **CAM-PQ projection is distance-preserving** (lossless across form transitions).
- **VSA convention is `[start:stop]` contiguous slices**, not scattered bits.
- `Structured5x5` is the native rich form; `Vsa10kF32` is native storage (not passthrough).

**Deferred:**
- PhaseTag types (ladybug-rs owns them).
- Crystal4K 41:1 compression persistence (ladybug-rs owns it).
- ladybug-rs quantum 9-op set port.

**Docs:**
- `crystal-quantum-blueprints.md` (existing, cross-referenced)
- Cross-repo-harvest H1-H14 (Born rule, phase tag, interference, Grammar Triangle ≡ ContextCrystal(w=1), NSM ≡ SPO axes, FP_WORDS=160, Mexican-hat, Int4State, Glyph5B, Crystal4K, teleport F=1, 144-verb, Three Mountains).

---

## #208 — grammar + crystal + AriGraph unbundle (merged 2026-04-19)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `contract::grammar/` module (6 files): FailureTicket, PartialParse, CausalAmbiguity, TekamoloSlots/TekamoloSlot, WechselAmbiguity/WechselRole, FinnishCase, NarsInference (7 variants), ContextChain (ring buffer), LOCAL_COVERAGE_THRESHOLD = 0.9, MARKOV_RADIUS = 5.
- `contract::crystal/` module (7 files): Crystal trait, CrystalKind, TruthValue, CrystalFingerprint (Binary16K / Structured5x5 / Vsa10kI8 / Vsa10kF32), SentenceCrystal / ContextCrystal / DocumentCrystal / CycleCrystal / SessionCrystal.
- `lance-graph::graph::arigraph::episodic`: unbundle_hardened / unbundle_targeted / rebundle_cold with ndarray::hpc::bitwise::hamming_batch_raw SIMD dispatch under `ndarray-hpc` feature.
- `UNBUNDLE_HARDNESS_THRESHOLD = 0.8` synchronized in contract + arigraph.

**Locked:**
- **AriGraph lives in-tree** at `lance-graph/src/graph/arigraph/` (not a standalone crate). 4696 LOC transcoded from Python AdaWorldAPI/AriGraph.
- **Crystals unbundle when hardness ≥ 0.8.** Rebundle for cold entries.
- **FailureTicket carries SPO × 2³ × TEKAMOLO × Wechsel decomposition** plus coverage + attempted_inference + recommended_next.
- **Finnish 15 cases, Russian 6 cases, Turkish 6 cases** + agglutinative chain, German 4 cases, Japanese particles — each in native terminology.

**Deferred:**
- DeepNSM emission of FailureTicket (D2, Phase 2).
- Grammar Triangle bridge into DeepNSM (D3, Phase 2).

**Docs:**
- `integration-plan-grammar-crystal-arigraph.md` (E1-E12 epiphanies).
- `crystal-quantum-blueprints.md` (Crystal vs Quantum modes).
- `endgame-holographic-agi.md` (5-layer stack).

---

## #207 — session capstone + Wikidata plan (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `session-capstone-2026-04-18.md` — 8 epiphanies (E1-E8), Sleeping Beauties (SB1-7), Missing Bridges (MB1-5), Known Brittle (KB1-5), priority map.
- `wikidata-spo-nars-at-scale.md` — 1.2B triples → 14.4 GB scale demo plan.

**Locked:**
- **§7 addendum correction:** AriGraph is SHIPPED, not deferred. Invalidates capstone's DD2 and MB3.
- **4-pillar inheritance** (NARS + thinking + qualia + proprioception) is compile-time contract.
- **CMYK (17D experienced) vs RGB (18D observed) qualia** — dim 17 = classification_distance.
- **Vocabulary IS semantics** — glyph names use pretraining-derived associations on purpose.

---

## #206 — state classification pillars (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `contract::qualia` — 17-D QualiaVector + 17→11 projection.
- `contract::proprioception` — 7 StateAnchor (Intake/Focused/Rest/Flow/Observer/Balanced/Baseline), 11-D state vector, ProprioceptionAxes, StateClassifier trait, DefaultClassifier, softmax `hydrate()`.
- `contract::world_map` — WorldMapDto + WorldMapRenderer trait + DefaultRenderer.
- `world_model.rs` extended: `qualia`, `axes`, `proprioception`, `cycle_fingerprint`, `timestamp`, `cycle_index`, `is_self_recognised()`, `is_liminal()`.
- `cognitive-shader-driver::sigma_rosetta` — 64 glyphs (Σ11) × 144 verbs (Σ12 = 12 roots × 12 tenses), SigmaState, TriangleGestalt, QuadTriangleGestalt, InteractionKinematic (Σ13 dome), QPL ↔ sigma 17D bridge.
- REST (`serve` feature) + gRPC (`grpc` feature) endpoints.
- `examples/villager_ai.rs` — Pumpkin NPC AI reference.

**Locked:**
- **Proprioception in the contract** = ontological self-recognition by compilation.
- **WorldMapRenderer pattern** — drop-in framing trait keeps contract vocabulary neutral.
- **Pumpkin / villager AI framing** for externally-visible examples (corporate disguise).
- **`proprioception::axes` named fields** (drive / arousal / dominance / …) — public API.
- **No code comments or PR descriptions** reveal the disguise scrubbing. Everything personal goes to ada-rs.

---

## #205 — engine_bridge + CMYK/RGB qualia (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `cognitive-shader-driver::engine_bridge` — ingest_codebook_indices, dispatch_from_top_k, EngineBusBridge, persist_cycle, UNIFIED_STYLES[12] 3-way mapping (ThinkingStyle ↔ StyleSelector ↔ p64 StyleParams).
- CMYK vs RGB qualia decomposition — 17D experienced vs 18D observed, classification_distance as dim 17.

**Locked:**
- **12 UNIFIED_STYLES** are the canonical style inventory (3-way mapping must stay aligned).
- **Named emotion archetypes** (fear/anger/sadness/joy/surprise/disgust) live in engine_bridge as classification references.

---

## #204 — cognitive-shader-driver crate + Shader DTOs (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- New crate `cognitive-shader-driver` (24 tests).
- `contract::cognitive_shader` — ShaderDispatch / ShaderResonance / ShaderBus / ShaderCrystal, MetaWord (u32 packed: thinking 6 + awareness 4 + nars_f 8 + nars_c 8 + free_e 6), CognitiveShaderDriver trait, ShaderSink commit-adapter.
- `auto_style` — 18D qualia → style ordinal.
- 630K LOC ladybug-rs import into `lance-graph-cognitive` (grammar, spo, learning, world, search, fabric, spectroscopy, container_bs, core_full).
- `crates/holograph` imported from RedisGraph, 10K→16K migration.
- `contract::container` — Container (16K fingerprint) + CogRecord (4KB = meta + content).
- `contract::collapse_gate` — GateDecision, MergeMode.

**Locked:**
- **Shader IS the driver** (role reversal from thinking-engine-first).
- **MetaWord packing layout** — thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6).
- **BindSpace struct-of-arrays** — FingerprintColumns (4 planes × 256 u64), EdgeColumn, QualiaColumn (18 f32), MetaColumn (u32).
- **`ShaderBus::cycle_fingerprint: [u64; 256]`** IS `Container` IS `CrystalFingerprint::Binary16K` (same 2 KB backing).
- **No serde in types** (debug-only); wire formats explicit.

**Docs:**
- `cognitive-shader-architecture.md` (canonical architecture reference).

---

## How to Use This File

1. **Opening a session on this workspace:** read the top 3 PRs
   (most recent). That covers ~90 % of what you need to know about
   current state.
2. **Before proposing a new type:** grep this file for the type
   name. If it's listed under Added, stop and read the source.
3. **Before proposing a convention:** grep for the topic. If it's
   listed under Locked, your proposal needs explicit justification
   to overturn it.
4. **When a PR merges:** prepend a new section at the top of this
   file. Old PRs stay — they are the arc.

This file is the fastest bootstrap available for a new session on
this workspace. Load it, then load 1-2 knowledge docs as the domain
triggers, then start working. Target: 3-5 turn cold start, not 30.
