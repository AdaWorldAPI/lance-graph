# Cosine census — deepnsm group (deepnsm, deepnsm-v2, helix, holograph, jc, sigker, template-equivalence, reader-lm)

Read-only census. Depth = full read on all 14 assigned files, plus targeted
greps (recorded below) to answer the deepnsm-v2 belief-wiring question the
brief flagged as highest priority.

## HEADLINE — deepnsm-v2 belief-admission wiring (the flagged highest-priority question)

**Finding: NOT WIRED.** `grep -n "similarity|cosine" crates/deepnsm-v2/src/*`
hits only `lib.rs` and `space.rs`. `crates/deepnsm-v2/src/belief.rs` and
`crates/deepnsm-v2/src/reason.rs` contain **zero** references to
`space`/`basin`/`Cam96Space`/`SemanticSpace`/`similarity`/`distance`. So as
currently written, `Nsm::word_similarity` / `Nsm::triple_similarity` /
`Cam96Space::similarity` do **not** feed `BeliefArena` admission or revision
in this crate — the two systems are wired to each other only through
`basin.rs` (`centroid_point`/`spread_about` consume `Cam96Space`, but that
path also does not touch `belief.rs`). Also note: **it isn't literally
cosine anyway** — see below, `Cam96Space`/`SemanticSpace` compute squared-L2,
not a trig cosine, despite doc-comments saying "cosine²".

## Sites

| file:line | expression | class | carrier | impl exists? | hot/cold |
|---|---|---|---|---|---|
| `deepnsm/src/fingerprint16k.rs:10` | doc: "Replaces cosine with popcount" | DOC-STRING-ONLY | — | — | — |
| `deepnsm/src/fingerprint16k.rs:82-84` | `similarity()` = `1.0 - hamming as f32 / DIM_BITS as f32` | DERIVED-READING (not stored) | `Fingerprint16K{words:[u64;256]}` — matches canonical `[u64;256]` Hamming shape | Contract impl exists (`lance_graph_contract::distance::Distance for [u64;256]`, `crates/lance-graph-contract/src/distance.rs:82`) but this crate has its **own** local `hamming()`/`similarity()` (line 72-84), not consuming the canonical one | HOT — core distance primitive of the crate, used by `hamming_early_exit` band-check path |
| `deepnsm/src/markov_bundle.rs:138-141` | doc: "cosine comparisons across kernel choices" | DOC-STRING-ONLY | — | — | — |
| `deepnsm/src/trajectory.rs:74-96` | `fn cosine(a:&[f32], b:&[f32]) -> f32` (dot/‖a‖‖b‖) | REPLACE-WITH-DISTANCE | `Vec<f32>` variable-length role-bundle slice (100-2000 dims, `role_keys.rs` widths) | No serving impl — canonical `Distance` trait only covers `[u64;256]`/`[u8;6]`/`[u8;3]`, none matches a variable-width f32 role slice | HOT — called from `role_candidates()` (line 49) scoring a codebook of `Vec<Vec<f32>>` per query; this is the crate's "resonance vs codebook" reasoning-lens step named in the repo `CLAUDE.md` §"The Click" |
| `deepnsm/src/trajectory.rs:99-103` | `pub struct Candidate { codebook_index, score: f32 }` | **stored-float field** | `score` holds the `cosine()` output | n/a | Collected into `Vec<Candidate>`, sorted/truncated in `role_candidates` — transient (not persisted to disk/Lance), but is a struct field carrying a cosine value, flagged per the "stored float field" ask |
| `deepnsm-v2/src/lib.rs:83` | doc: "`Cam96Space` (`6×cosine²` DISTRIBUTION...)" | DOC-STRING-ONLY / terminology mismatch | — | — | Actual code is squared-L2 (see `space.rs:253-272`), not a trig cosine — see note below |
| `deepnsm-v2/src/lib.rs:142-144` | `word_similarity()` → `self.space.similarity(...)` | DERIVED-READING (not stored; delegates to space.rs) | `Cam96 = [u8;12]` | No canonical `Distance` impl for `[u8;12]` (only `[u8;6]`/`[u8;3]`/`[u64;256]` exist) | Confirmed COLD w.r.t. belief/reason (see headline); hot/cold vs other callers UNDETERMINED — no callers found inside deepnsm-v2 crate outside its own tests |
| `deepnsm-v2/src/lib.rs:150-162` | `triple_similarity()` → 3× `self.space.similarity(...)` | same as above | same | same | same |
| `deepnsm-v2/src/space.rs:74-80` | `SemanticSpace::similarity`/`distance` → `self.palette.similarity(a,b)` (wraps `lance_graph_contract::recipe_substrate::PairPalette`) | ALREADY-CANONICAL-SERVING-IMPL, but **float**, not integer LUT | `(u8,u8)` pair, `PairPalette` | Yes — `PairPalette::distance/similarity` is the canonical contract-level palette pair impl (`crates/lance-graph-contract/src/recipe_substrate.rs:134-155`) | UNDETERMINED (no caller found in assigned files) |
| `deepnsm-v2/src/space.rs:158-163` | doc on `type Cam96`: "each a `palette256:palette256` pair (a cosine²)" | DOC-STRING-ONLY / terminology mismatch | — | — | See note below — actual op is squared-L2 |
| `deepnsm-v2/src/space.rs:165-171` | doc on `Cam96Space`: "**No cosine call**: the normalized `[x;y]` coordinate distance carries the ordering directly" | DOC-STRING-ONLY (self-correcting — this comment is accurate) | — | — | — |
| `deepnsm-v2/src/space.rs:253-272` | `Cam96Space::distance()` = Σ 12× axis-wise squared-L2 | ALREADY-SQUARED-L2 (not cosine, not integer) | `Cam96 = [u8;12]`, 12 `AxisCodebook = Vec<Vec<f32>>` centroids | No canonical `[u8;12]` `Distance` impl; live f32 centroid computation via `AdcMetric::cell` (`crates/lance-graph-contract/src/cam.rs:250-256`), not a precomputed integer LUT despite the palette256 doctrine's push toward LUT dispatch | Same caller-status as above (UNDETERMINED / cold within assigned scope) |
| `deepnsm-v2/src/space.rs:276-282` | `Cam96Space::similarity()` = `1 - distance/d_max`, clamped | DERIVED-READING (not stored) | same | same | same |
| `helix/src/fisher_z.rs:20-28` | `struct Similarity(pub f64)` — newtype "a cosine (or other) similarity value" | LAB-CALIBRATION / pipeline-stage carrier, not a cosine computation itself | `f64` scalar, transient input | n/a — consumes an already-computed similarity from elsewhere | Cold in this file (no producer of the input value lives here; this is Stage 3 of a pipeline, per file header) |
| `helix/src/fisher_z.rs:55-78` | `fisher_z()`/`hyperbolic_depth()` = arctanh transform of the wrapped value | DERIVED-READING | `f64` | n/a | Whole file is a pure transform stage; not itself a cosine site |
| `holograph/src/representation.rs:215-226` | `GradedVector::cosine_similarity()` = `dot/(‖a‖‖b‖)` | REPLACE-WITH-DISTANCE candidate | `GradedVector{values: Vec<i8>; 10_000 dims}` | No canonical Distance impl for this shape (10K i8, not the 256/6/3 canon carriers); dot products are integer (`i32`), only the final `sqrt`+divide is float | COLD — grep confirms no caller of `cosine_similarity` anywhere in the workspace outside this file's own `#[cfg(test)]` (`test_graded_bundle`, line 618-619); `GradedVector` itself is referenced only inside `holograph` |
| `holograph/src/representation.rs:206-213` | `dot()` (i32), `sign_distance()` (u32 Hamming-like), `l1_norm`/`l2_norm_sq` (i32) | ALREADY-INTEGER | same | n/a | These integer primitives ARE already available and could serve as the non-float replacement basis if `cosine_similarity` is ever wired to a real caller |
| `jc/src/substrate.rs:39-44` | `fn cosine_sim(a:&[f64], b:&[f64]) -> f64` | LAB-CALIBRATION | `Vec<f64>`, `D=10_000`, one-shot research probe | n/a — this is `E-SUBSTRATE-1`'s associativity probe (verifies VSA bundle math, not a runtime reasoning path) | COLD/one-time — `prove()` is a research probe run once per invocation, not a hot reasoning-path call; permitted per doctrine as "one-time table build or research probe" |
| `jc/src/probe_p1_gamma_phase.rs` | — | N/A — **no cosine call found** in this file | `Vec<f64>`/`f64` (toroidal distance + Spearman ρ) | n/a | File uses `toroidal_distance` + `spearman_rho`, not cosine; flagged for completeness since it was in-scope |
| `sigker/src/kernel.rs:55-65` | `signature_kernel_normalized()` — doc: "cosine in tensor-algebra feature space" | REPLACE-WITH-DISTANCE candidate — NO (out of scope; see note) | `Vec<Vec<f64>>` paths → scalar kernel values | No serving Distance impl (not a fingerprint/codebook carrier at all — genuine continuous path-signature kernel method) | UNDETERMINED — no caller found in assigned files; likely a standalone path-comparison numerical method (OSINT trajectory kernels per module doc), not part of the palette256 reasoning substrate |
| `sigker/src/log_signature.rs:167-172` | `LogSignature::cosine()` = `dot/(‖a‖‖b‖)` | REPLACE-WITH-DISTANCE candidate — NO (out of scope; same as above) | `Vec<f64>` (Lyndon-basis log-signature coeffs, ~70-1.9M dims depending on depth/dim) | No serving Distance impl | UNDETERMINED — no caller found; same math-library carve-out as kernel.rs |
| `sigker/src/randomized.rs:169-177` | `RandomizedSignature::cosine()` = `dot/(‖a‖‖b‖)` | REPLACE-WITH-DISTANCE candidate — NO (out of scope; same as above) | `Vec<f64>` (fixed-width `state_dim`, e.g. 4096) | No serving Distance impl; file's own doc explicitly frames this as "comparable to Vsa16k" and "NOT lossless" trade-off vs the codebase's bind+bundle path | UNDETERMINED — no caller found in assigned files |
| `template-equivalence/src/lib.rs:11-13` | doc: `EquivalenceClass::Semantic` "(embedding cosine) is the deferred piece and currently degrades to Failure" | DOC-STRING-ONLY (explicitly unimplemented) | — | — | Code (`StructuralChecker::compare`, line 148-150) fails closed for the Semantic case — **no cosine is computed anywhere in this file** |
| `reader-lm/src/weights.rs:149-150` | doc: "Returns full-precision f32 vectors for cosine/distance computation" | DOC-STRING-ONLY | — | — | `hydrate_role()` itself just dequantizes `StackedN` → `Vec<Vec<f32>>`; no cosine call in this file |
| `reader-lm/src/weights.rs:165` | doc: "DO NOT use for cosine comparison — OpenChat = ALL ZEROS at this resolution" | DOC-STRING-ONLY | — | — | Warning on the legacy i16 `Base17` path; no cosine call in this file |

## Stored-float fields (found)

| Field | File:line | Holds |
|---|---|---|
| `Candidate.score: f32` | `deepnsm/src/trajectory.rs:99-103` | The `cosine()` output, collected/sorted/truncated in `role_candidates()`. Transient (`Vec<Candidate>` return value, not persisted), but a struct field carrying a cosine value as requested. |
| `EquivalenceReport.score: f32` | `template-equivalence/src/lib.rs:56-61` | A structural (rank-position-based) score, **not** cosine — `Semantic`/cosine path is unimplemented and never reaches this field via cosine. Listed for completeness, not a violation. |

No other assigned file stores a float similarity as long-lived state (Lance
row / SoA column / persisted struct). `helix::Similarity(f64)`,
`RandomizedSignature{state}`, `LogSignature{coeffs}` hold the *carrier*
(input to a similarity function), not a *computed similarity result*.

## Summary

- **Actual cosine (dot/‖a‖‖b‖) implementations found:** 6 —
  `deepnsm/src/trajectory.rs::cosine` (HOT, feeds reasoning-lens role
  candidates, no serving Distance impl), `holograph/src/representation.rs::GradedVector::cosine_similarity`
  (COLD, no callers anywhere in the workspace), `jc/src/substrate.rs::cosine_sim`
  (LAB-CALIBRATION, one-shot probe), `sigker/src/kernel.rs::signature_kernel_normalized`,
  `sigker/src/log_signature.rs::LogSignature::cosine`, `sigker/src/randomized.rs::RandomizedSignature::cosine`
  (all three sigker sites: standalone path-signature kernel math, no callers
  found in assigned scope, arguably outside the palette256/BindSpace
  reasoning-substrate doctrine entirely).
- **"Cosine" in doc-comments but NOT actually cosine in code:** `deepnsm-v2`
  (`lib.rs:83`, `space.rs:158-163`) — both `Cam96Space` and `SemanticSpace`
  compute **squared-L2** (`AdcMetric::SquaredL2` / `PairPalette`), confirmed
  by reading `lance-graph-contract/src/cam.rs:250-256` and
  `recipe_substrate.rs:125-155`. `deepnsm-v2/src/space.rs:165-171`'s own doc
  correctly self-reports "No cosine call" for `Cam96Space` — the mismatch is
  only in the *type-level* doc comment (`lib.rs:83`, `space.rs:158-163`)
  using "cosine²" as decorative shorthand for the palette-pair distance
  concept.
- **Highest-priority item resolved:** the deepnsm-v2 `space.rs` similarity
  code (whether read as cosine or, correctly, squared-L2) does **not**
  currently feed `belief.rs` admission/revision — confirmed by direct grep
  of `belief.rs` and `reason.rs` for any `space`/`basin`/`similarity`/
  `distance` reference (zero hits in both). The wiring the brief worried
  about does not exist in the current tree.
- **No float-similarity fields found stored as persistent state** (Lance
  columns, SoA envelopes) in any of the 14 files — the one stored field
  (`Candidate.score`) is a transient in-memory ranking artifact.
- **REPLACE-WITH-DISTANCE candidates with no serving impl today:**
  `deepnsm::trajectory::cosine` (Vec<f32> role slices — the one with an
  active, hot caller) and `deepnsm-v2`'s `Cam96Space`/`SemanticSpace`
  (`[u8;12]`/`(u8,u8)` carriers — not literally cosine, but float-computed
  live rather than integer-LUT-dispatched, contrary to the "256×256 integer
  LUT" doctrine cited in the brief).
- **LAB-CALIBRATION (permitted):** `jc/src/substrate.rs` (E-SUBSTRATE-1
  associativity probe).
- **Out-of-scope math library (own carve-out, not fingerprint/codebook
  shaped):** all of `sigker/` (signature kernels over continuous paths) —
  flagged but not forced into REPLACE-WITH-DISTANCE since no
  `[u64;256]`/`[u8;6]`/`[u8;3]` carrier applies to variable-length path
  signatures.
