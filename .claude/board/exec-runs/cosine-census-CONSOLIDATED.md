# Cosine-replacement census — CONSOLIDATED (orchestrator synthesis, 2026-07-27)

> Operator directive: `lance-graph-contract::distance()` is the canonical
> integer dispatch (`fn distance(&self,&Self) -> u32`; impls today: `[u64;256]`
> Hamming, `[u8;6]` CamPq L1, `[u8;3]` Palette L1); grep every cosine site and
> classify against it. Doctrine: NO FLOAT EVER on reasoning paths; palette256
> (ρ 0.9973–0.9995) with 256×256 integer LUTs; float only as derived decode;
> `vsa_cosine` only inside the I-VSA-IDENTITIES niche.
>
> Inputs (Sonnet fleet, read-only): `cosine-census-contract.md` ·
> `cosine-census-planner-core.md` · `cosine-census-deepnsm-group.md` ·
> `cosine-census-lab-crates.md`. 103 production files censused.

## THE REPLACE LIST — everything that actually needs migration

| # | site | status | detail |
|---|---|---|---|
| 1 | `lance-graph-contract/src/cam.rs:260-271` — `AdcMetric::Cosine::cell()` (raw dot/norm) → `DistanceTableProvider::distance/distance_batch` (`:172,:175,:344-354`) | **HOT** — consumed per-candidate by planner `CamPqScanOp` | ADC tables + summed distance stay f32 end-to-end. Self-documented gap: `distance.rs:95-96` calls the integer `[u8;6]` impl an "L1 fallback", explicitly NOT the real ADC. **This is the cosine replacement.** |
| 2 | `deepnsm/src/trajectory.rs:78-96` — cosine in `role_candidates()` vs `Vec<Vec<f32>>` codebook | **HOT within deepnsm** (workspace-excluded standalone) | The documented "resonance vs codebook" step. No `Distance` impl exists for the variable-width carrier; score transient (`Candidate.score: f32`), not persisted. |
| 3 | `lance-graph-cognitive/src/grammar/nsm.rs:349-371` (`NSMField::dot/cosine_similarity`) → `triangle.rs:124` (`weighted_similarity`) | **DORMANT** — grep-verified unwired | Only external consumer (`deepnsm/triangle_bridge.rs`) bypasses it for binary fingerprints. Migrate or retire when the Grammar Triangle is wired. |

That is the entire list. Three sites, one of them dormant.

## VERIFIED NON-ISSUES (the doctrine already holds)

- **NARS insight/tactics hot path is float-cosine-free** — `insight.rs`'s only
  "cosine" is a doc comment about a REMOVED design.
- **deepnsm-v2 is already migrated**: `Cam96Space::distance` (`space.rs:253-272`)
  computes `AdcMetric::SquaredL2` through the contract's `PairPalette`; its own
  inner comment says "No cosine call". Two stale doc comments (`lib.rs:83`,
  `space.rs:158-163`) still say "6×cosine²" — label fix only.
- **deepnsm-v2 belief wiring**: `space.rs` similarity does NOT feed
  `BeliefArena` admission/revision (zero cross-references in `belief.rs`/`reason.rs`).
- **QUERY-SURFACE (9 sites, one chain)**: Cypher `vector_distance()`/
  `vector_similarity()` — `ast.rs → parser.rs → semantic.rs → udf.rs →
  vector_ops.rs → lance_vector_search.rs → python/graph.rs`. User-facing query
  language over embedding columns; the one real SIMD float cosine backing it is
  `vector_ops.rs` via `ndarray::hpc::heel_f64x8`. Separate migration question,
  deliberately NOT on the replace list.
- **VSA-NICHE (sanctioned)**: `crystal/fingerprint.rs` `vsa_cosine`/`vsa16k_cosine`;
  `witness_corpus.rs:736-793` (feature-gated test).
- **LAB tier (58 files: bgz-tensor 28, bgz17 1, highheelbgz 5, thinking-engine 24)**:
  ~34 TABLE-BUILD (cosine used once to construct palette LUTs — the replacement
  mechanism working) + ~17 LAB-CALIBRATION + 7 DOC-ONLY. **Zero exports into the
  spine** — repo-wide grep found no real `bgz17::`/`bgz_tensor::`/`highheelbgz::`/
  `thinking_engine::` import in contract/planner/cognitive/core; all apparent
  hits were doc comments or name collisions (`ndarray::hpc::bgz17_bridge::Base17`;
  core's independent `nsm/similarity.rs::SimilarityTable`).

## STORED-FLOAT-SIMILARITY FLAGS (letter-of-the-rule)

| site | verdict |
|---|---|
| `high_heel.rs` `LensProfile` curves (f32) | cold offline lens calibration; no in-crate runtime reader; certification-officer pattern |
| `scientific.rs` `StatisticalSimilarity.cosine_estimate` / `CrossValidation.cosine` | derived from integer Hamming; ZERO callers — dead scaffolding |
| `thinking-engine/f32_engine.rs` full pairwise f32 table, "no quantization" by its own doc | closest thing to the forbidden pattern found anywhere; LAB, unreached by spine |

## TECH-DEBT SPUN OFF
1. `bgz17`/`bgz-tensor` declared as deps of `lance-graph` (default-on features) and
   `lance-graph-planner` (unconditional) but **never imported** — dead dependency wiring.
2. Two stale "cosine²" doc comments in deepnsm-v2 (`lib.rs:83`, `space.rs:158-163`).
3. `f32_engine.rs` unquantized table (lab; note-only until something reaches it).
