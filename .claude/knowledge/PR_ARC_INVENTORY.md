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
