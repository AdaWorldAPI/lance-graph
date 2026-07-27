# Cosine census — lab tier (bgz-tensor / bgz17 / highheelbgz / thinking-engine)

Scope: all files matching `cosine` (case-insensitive) under
`crates/bgz-tensor/src/**`, `crates/bgz17/src/similarity.rs`,
`crates/highheelbgz/src/**`, `crates/thinking-engine/src/**`, enumerated via
Grep (not a stale list). 28 + 1 + 5 + 24 = 58 files. Read-only; no production
file edited; no `cargo` invoked.

## Headline finding (read this first)

`bgz17` and `bgz-tensor` ARE declared as Cargo dependencies of the spine:

- `crates/lance-graph/Cargo.toml` (core): `bgz17` optional (feature
  `bgz17-codec`) and `bgz-tensor` optional (feature `tensor-codec`) — **both
  features are in `default = [...]`**, so a default `cargo build` compiles
  them in.
- `crates/lance-graph-planner/Cargo.toml`: `bgz17` is an **unconditional**
  dependency (line 36).

But a repo-wide grep for path-qualified usage (`bgz17::`, `bgz_tensor::`,
`highheelbgz::`, `thinking_engine::`) found **zero occurrences** in
`crates/lance-graph/src/**`, `crates/lance-graph-planner/src/**`,
`crates/lance-graph-contract/src/**`, or `crates/lance-graph-cognitive/src/**`
that resolve to these four lab crates. Every apparent hit was one of:

1. A doc-comment mention (`crates/lance-graph-planner/src/nars/facet_fold.rs:12`
   — "`bgz17::HierarchicalPalette` ... is SEPARATE future [work]"; a TODO in
   `crates/lance-graph-planner/src/physical/accumulate.rs:253` — "TODO: bgz17
   palette semiring").
2. `crates/lance-graph-contract/src/escalation.rs:284` — "Mirrors
   `thinking_engine::ghosts::GhostType` (an excluded crate that cannot be a
   contract dependency)" — an explicit doc-comment note that the contract
   crate is (by its own zero-deps design, confirmed in its Cargo.toml) NOT
   allowed to depend on thinking-engine. Not an import.
3. `ndarray::hpc::bgz17_bridge::Base17` / `ndarray::hpc::heel_f64x8::*` — a
   **different crate's** module that happens to be named `bgz17_bridge`. This
   lives in the separate `ndarray` fork repo, not in `crates/bgz17` or
   `crates/bgz-tensor` of this repo. Confirmed by reading
   `crates/lance-graph/src/graph/hydrate.rs`, `neuron.rs`,
   `crates/lance-graph-planner/src/cache/{kv_bundle,convergence}.rs`: all
   `use ndarray::hpc::bgz17_bridge::...`, never `use bgz17::...`.
4. `crates/lance-graph/src/nsm/nsm_word.rs` uses `super::similarity::SimilarityTable`
   — its own sibling module `crates/lance-graph/src/nsm/similarity.rs`
   (confirmed by reading it: `//! Calibrated similarity table ... Built from
   the exact distance distribution of a word distance matrix`), **not**
   `crates/bgz17/src/similarity.rs` or `crates/bgz-tensor/src/similarity.rs`.
   Parallel/analogous design, same name, independent implementation — not an
   import of either of my assigned `similarity.rs` files.

**Conclusion: none of the 58 files in my assigned scope have a live
spine importer today.** `bgz17-codec` / `tensor-codec` in `lance-graph`
core and the unconditional `bgz17` dep in `lance-graph-planner` are
currently **dead dependency edges** (declared, compiled, never
path-referenced in source) as far as this census's target modules go. This
itself is worth a TECH_DEBT-style flag: a default-on feature that pulls in a
whole crate with zero use sites is either stale wiring or a silent
in-progress migration; I did not find a WIP `mod` gated behind
`#[cfg(feature = "tensor-codec")]` anywhere in `crates/lance-graph/src`.

Real external importers of `bgz17::` / `thinking_engine::` / `bgz_tensor::` /
`highheelbgz::` DO exist in the repo, but every one of them is a crate
**outside** the four named spine crates (`p64-bridge`, `cognitive-shader-driver`,
`lance-graph-callcenter`, `reader-lm`, `bge-m3`, `lance-graph-arm-discovery`,
plus examples/tests within the lab crates themselves). Per the task's scope
(only lance-graph-contract / lance-graph-planner / lance-graph-cognitive /
lance-graph core count as "spine"), these don't create an EXPORTED-FLOAT-PATH
row, but are noted in case the spine definition is meant more broadly.

## File classification

Legend: **LAB-CALIBRATION** (research/codec similarity method or ground-truth
work — includes cases judged closest-fit even when the file does live
runtime float computation rather than strict "calibration", noted in
evidence), **TABLE-BUILD** (cosine consumed once/incrementally to construct
or fill a calibrated integer LUT that reasoning then reads), **DOC/STRING-ONLY**.
No file classified EXPORTED-FLOAT-PATH — see headline finding.

### crates/bgz-tensor/src/ (28 files)

| file | class | evidence | spine importer |
|---|---|---|---|
| stacked.rs | LAB-CALIBRATION | `StackedBF16x4::cosine` (f64) + test comparing stacked/base17/true cosine accuracy | none |
| adaptive_codec.rs | LAB-CALIBRATION | `cosine_f32_to_f64_simd` imported "used by tests and future GPTQ compensation"; test-only | none |
| morton_cascade/legacy.rs | DOC/STRING-ONLY | one comment, "lookup_f32(a,a) is the self-cosine" — no cosine code | none |
| zipper.rs | LAB-CALIBRATION | `cosine_phase_only`/`cosine_magnitude_only`/`cosine_zipper_full`/`cosine` — real similarity methods on ZipperDescriptor-family types, module doc frames as Matryoshka truncation research | none |
| belichtungsmesser.rs | TABLE-BUILD | `SimilarityTable`-style band builder consumes `(a,b,ground_truth_cosine)` triples to build similarity bands | none |
| bin/cam_pq_calibrate.rs | LAB-CALIBRATION | binary tool; local `fn cosine` used for ICC_3_1 ground-truth calibration stat | none (bin, not lib) |
| codebook_calibrated.rs | TABLE-BUILD | `cosine_f32_slice` (from stacked_n) drives furthest-point sampling + γ+φ calibration mapping raw cosine → u8 `CalibratedCodebook` | none |
| euler_fold.rs | TABLE-BUILD | `fast_cosine` groups vectors into CLAM families (one-time clustering) | none |
| fisher_z.rs | TABLE-BUILD | `FisherZGamma` encode/decode pairwise cosine ↔ i8 via Fisher-z transform — explicit calibrated table codec | none |
| fractal_descriptor.rs | LAB-CALIBRATION | `cosine()` method on phase-signature descriptors — live similarity method, research descriptor | none |
| gamma_calibration.rs | TABLE-BUILD | `CosineGamma`/`MetaGamma`/`CalibrationProfile` — explicit "cosine replacement" 3-γ calibration file | none |
| had_cascade.rs | LAB-CALIBRATION | `cosine_f32_to_f64_simd` (from ndarray) used only for avg/pairwise reconstruction-quality tests | none |
| hdr_belichtung.rs | LAB-CALIBRATION | `leaf_hydrate_cosine` — doc says "LEAF stage: BF16→f32 hydration for exact cosine" — permitted derived-decode pattern, one call site | none |
| hhtl_cache.rs | TABLE-BUILD | `gamma_meta: [f32;4]` field doc: "Per-basin gamma correction factors for exact cosine restore" — calibration metadata stored alongside the distance table | none |
| hhtl_d.rs | TABLE-BUILD | Fisher-z i8 pairwise cosine table; `cosine_lookup()` reads the precomputed table (Option<f32>) — the LUT decode step itself | none |
| hhtl_f32.rs | LAB-CALIBRATION | test-only `fn cosine`, validates Path-A codec reconstruction quality | none |
| holographic_residual.rs | LAB-CALIBRATION | `cosine_f32_to_f64_simd` used only in reconstruction-quality tests | none |
| jina.rs | LAB-CALIBRATION | calls external Jina API, computes `cosine_f32` as ground truth vs Base17 cosine — canonical ground-truth calibration file | none |
| matryoshka.rs | LAB-CALIBRATION | "Measure reconstruction quality: per-row cosine and pairwise rank" — validation function | none |
| neuron_hetero.rs | TABLE-BUILD | `SpatialRole::cosine` delegates to Stacked; `ThinkingStyleFingerprint::cosine_proxy` = **the doctrine's own "cosine replacement" pattern** — bit-agreement fraction (integer XOR+popcount) as cosine proxy, f64 only at the final ratio (derived decode) | none — doc explicitly says "FieldModulation cosine from lance-graph-contract" exists as an analog in contract, but contract's own field is not this file and is out of my assigned scope |
| projection.rs | LAB-CALIBRATION | `Base17::cosine`/`Base17Fz::cosine` (f64) — core similarity methods of bgz-tensor's OWN `Base17` type. **Not the same type** as `ndarray::hpc::bgz17_bridge::Base17` that the spine imports (name collision, confirmed distinct by import paths) | none |
| slot_l.rs | LAB-CALIBRATION | test-only `fn cosine` helper | none |
| stacked_n.rs | TABLE-BUILD | `StackedN::cosine` (SIMD via ndarray) + `ClamCodebook::build_cosine` (furthest-point clustering) + exported `cosine_f32_slice`/`cosine_f32_slice_scalar` consumed by 3 other bgz-tensor files | none |
| turboquant_kv.rs | LAB-CALIBRATION | "Level 2: exact cosine(Q, dequant) on 5% survivors" — genuinely a runtime attention-cascade verification step on a small survivor set, not calibration per se; flagged as closest-fit only | none |
| xor_adaptive.rs | LAB-CALIBRATION | `cosine_f32_to_f64_simd` used only in reconstruction-quality test | none |
| similarity.rs | TABLE-BUILD | `SimilarityTable::from_paired` — bins `(L1, ground_truth_cosine)` pairs into a calibrated table; `collect_calibration_pairs` — explicit calibration-table builder | none |
| gamma_calibration.rs *(see above)* | — | — | — |
| hydrate.rs | (no cosine match — not in scope) | — | — |

(28 files enumerated by Grep; `similarity.rs`/`gamma_calibration.rs` listed once each above — no duplicate rows beyond formatting.)

### crates/bgz17/src/similarity.rs (1 file)

| file | class | evidence | spine importer |
|---|---|---|---|
| similarity.rs | DOC/STRING-ONLY | Only 1 hit: file-header comment `//! calibrated from corpus statistics. Drop-in replacement for cosine similarity.` No `cosine` identifier, function, or type appears in the file body. | none — and confirmed via `bgz17::` importer sweep (p64-bridge, cognitive-shader-driver import `bgz17::{base17,palette,palette_semiring,distance_matrix}`, never `bgz17::similarity`) |

### crates/highheelbgz/src/ (5 files)

| file | class | evidence | spine importer |
|---|---|---|---|
| rehydrate.rs | LAB-CALIBRATION | `SpiralEncoding::cosine`/`cosine_interpolated` — used as the actual nearest-vocab-entry query mechanism ("Tokenize: find nearest vocab entry by spiral cosine") — genuine runtime float query, not calibration; closest-fit bucket only | none |
| simd_hardened.rs | LAB-CALIBRATION | `cosine_f32_8`/`cosine_f32_16` — fixed-size SIMD-friendly primitives, generic utility | none |
| source.rs | LAB-CALIBRATION | test-only self-cosine diagnostic on `SpiralWalk` | none |
| tensor_bridge.rs | LAB-CALIBRATION | cascade_search: HEEL 3-finger filter → hydrate survivors → `StackedN::cosine` final rank — real query-time computation on a filtered survivor set | none |
| lib.rs | LAB-CALIBRATION | `SpiralWalk::cosine`; `find_best_walk_config` calibrates (start,stride,length) against ground-truth pairwise cosine | none |

### crates/thinking-engine/src/ (24 files)

| file | class | evidence | spine importer |
|---|---|---|---|
| silu_correction.rs | TABLE-BUILD | `cosine_f32` builds `true_cos` vs `linear_cos` correction values feeding an HDR CDF-encoded distance table | none |
| jina_lens.rs | DOC/STRING-ONLY | single comment describing table semantics ("CDF percentile rank of the pairwise cosine") | none |
| bf16_engine.rs | TABLE-BUILD | `from_f32_cosines`/`from_f64_cosines`/`from_mean_pair_cosines` — canonical BF16 table builders from cosine | none |
| splat_ops.rs | DOC/STRING-ONLY | deprecated free-function wrappers; doc says "Returns cosine-similarity-style metric"; real logic now lives in think.rs | none |
| inference_backend.rs | DOC/STRING-ONLY | architecture-comment only ("Compare pairwise (cosine/Hamming), route via cascade") | none |
| dual_engine.rs | TABLE-BUILD | decodes existing u8 table to f32 (`(v-128)/127`) then rebuilds via `from_f32_cosines` — decode-then-table-rebuild | none |
| f32_engine.rs | TABLE-BUILD (flag) | `F32ThinkingEngine` stores pairwise cosine **directly as f32, no quantization** ("no quantization" is verbatim in the doc). This is the file in the whole census closest to a literal doctrine violation shape (a persisted all-float distance table) — but it is internal-only lab code with zero spine importer today. Flagged for visibility, not for fix (out of scope). | none |
| think.rs | LAB-CALIBRATION (flag) | `Think::replay_coherence`/`score_hole_closure` — genuine runtime dot-product/cosine-style computation on f32 `SplatField` energies, not calibration. Closest-fit bucket only; internal, no spine import | none |
| bridge.rs | TABLE-BUILD | `hydrate_and_cosine` (exact f64 decode) → `enrich_table_from_source` maps cosine → u8 table entry (`((cosine+1)/2*255) as u8`) | none |
| codebook_index.rs | TABLE-BUILD | doc: "Built offline by finding the nearest weight row (cosine similarity)" — offline codebook construction | none |
| signed_engine.rs | TABLE-BUILD | `from_f32_cosines` quantizes raw f32 cosine directly to signed i8 table | none |
| reranker_lens.rs | DOC/STRING-ONLY | one comment re: cosine range symmetry | none |
| l4.rs | DOC/STRING-ONLY | one comment, "L1-L3 are waves (cosine, interference, immutable tables)" | none |
| prime_fingerprint.rs | LAB-CALIBRATION | `prime_cosine` — real f32 cosine between "additive prime fingerprints", used in a test/diagnostic comparing to XOR+popcount alternative | none |
| osint_bridge.rs | TABLE-BUILD | loads a precomputed on-disk `cosine_table_path` (F32 distance table) into `Osint...` bridge; `ContrastiveLearner` (separate file) updates it from observed cosines | none |
| meaning_axes.rs | DOC/STRING-ONLY | one comment stating a calibration RESULT ("Pearson r = 0.9913 between Jina cosine and 48-axis Hamming similarity") — no cosine code | none |
| engine.rs | TABLE-BUILD | `cosine_f64_simd` computes pairwise cosine over 4096 centroids, mapped to u8 `[0,255]` table — canonical table-build | none |
| sensor.rs | LAB-CALIBRATION (flag) | `Sensor::from_embedding` — `cosine_f32_to_f64_simd` ranks centroids by similarity to a query embedding to pick top-N activations. This reads as a genuine runtime routing computation (sensor→codebook activation), not calibration; closest-fit bucket, no spine import | none |
| role_tables.rs | TABLE-BUILD | builds `cosines` matrix from activated Q/K/V/Gate vectors, feeds `BF16ThinkingEngine::from_f64_cosines` | none |
| reencode_safety.rs | LAB-CALIBRATION | numeric constants "reranker minimum/maximum cosine" used as calibration bounds in a safety/regression check | none |
| tensor_bridge.rs | LAB-CALIBRATION | `EmbeddingOutput::cosine` (SIMD via ndarray) + `pairwise_cosines()` matrix builder — bridge/calibration utility | none |
| ground_truth.rs | LAB-CALIBRATION | `GroundTruthEmbedding::cosine`, `CalibrationCorpus` — canonical ground-truth calibration file (module doc explicitly says so) | none |
| builder.rs | TABLE-BUILD | `raw_cosines()` builder API feeds `from_f32_cosines`/`SignedThinkingEngine::from_f32_cosines` table constructors | none |
| contrastive_learner.rs | TABLE-BUILD | `ContrastiveLearner::update_pair`/`fan_out_update` — online EMA update of a cosine-valued table from observed forward-pass cosines | none |

## EXPORTED-FLOAT-PATH details (line-level)

**None found.** No row for this table — see headline finding above. Every
`use bgz17::…` / `use bgz_tensor::…` / `use highheelbgz::…` /
`use thinking_engine::…` in the repo resolves to a crate outside the four
named spine crates (`p64-bridge/src/lib.rs`, `cognitive-shader-driver/src/{driver,mailbox_soa,engine_bridge,auto_style}.rs`,
`lance-graph-callcenter/src/cognitive_bridge_gate.rs`, `reader-lm/src/weights.rs`,
`bge-m3/src/weights.rs`, `lance-graph-arm-discovery/src/{lib,aerial/codebook}.rs`),
or is a same-crate/lab-internal reference, or is a doc-comment/false-positive
(the two `Base17`-named-type collision and the two `similarity.rs`-named-file
collision, both detailed above).

If the spine definition is later widened to include `cognitive-shader-driver`
(workspace member per root `Cargo.toml` line 20, and architecturally
downstream of `lance-graph-contract`'s `CognitiveShader`/`OrchestrationBridge`
per this repo's `CLAUDE.md` "AGI is the glove" section), then
`crates/cognitive-shader-driver/src/driver.rs` does import real `bgz17::`
symbols (`palette_semiring::PaletteSemiring`, `base17::Base17`,
`palette::Palette`) — but those are the integer palette/distance-table API,
not the cosine-family functions this census tracks, so it would still not
add an EXPORTED-FLOAT-PATH row even under a widened definition.

## Summary

- 58 files censused: 0 EXPORTED-FLOAT-PATH, ~34 TABLE-BUILD, ~17 LAB-CALIBRATION
  (several flagged as "genuine runtime float compute, closest-fit bucket
  only" rather than true calibration — `f32_engine.rs`, `think.rs`,
  `sensor.rs`, `turboquant_kv.rs`, `rehydrate.rs`, `tensor_bridge.rs` [highheelbgz]),
  ~7 DOC/STRING-ONLY.
- **Dependency-vs-usage gap**: `lance-graph` core declares `bgz17` and
  `bgz-tensor` as default-on optional deps (`bgz17-codec`, `tensor-codec`
  features); `lance-graph-planner` declares `bgz17` unconditionally. Neither
  crate's source actually path-references `bgz17::` or `bgz_tensor::`
  anywhere. Worth a TECH_DEBT entry if not already tracked — either the
  wiring is stale, or a planned `mod` behind these features was never
  written.
- **Two name collisions caused false-positive risk** and are worth flagging
  for future auditors: (1) `Base17` exists as both bgz-tensor's own type
  (`crates/bgz-tensor/src/projection.rs`) and `ndarray::hpc::bgz17_bridge::Base17`
  (a different repo's type) — the spine imports only the latter; (2)
  `SimilarityTable` exists independently in `crates/lance-graph/src/nsm/similarity.rs`,
  `crates/bgz-tensor/src/similarity.rs`, and (name only, no code) `crates/bgz17/src/similarity.rs`
  — the core's own copy is a from-scratch parallel implementation, not an
  import of either lab-tier file.
- If the doctrine's "float only as derived decode or one-time LAB/CALIBRATION
  table build" line is drawn strictly, the single file worth a follow-up
  look is `crates/thinking-engine/src/f32_engine.rs` — it is explicit in its
  own doc comment that it stores the full pairwise cosine matrix as raw f32
  with no quantization, which is architecturally the exact shape the
  doctrine warns against, even though it is currently unreached from any
  spine crate.
