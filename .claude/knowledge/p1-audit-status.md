# P1 Audit Status — v2.x Certification + ndarray Foundation

## READ BY: ALL AGENTS
## STATUS: IN PROGRESS (2026-04-13)
## BRANCH: claude/risc-thought-engine-TCZw7

---

## Scope

This document tracks the verified/unverified status of every commit in
the v2.x certification cascade and the ndarray foundation commits that
sit under it. Per P1 spec: `git show`, `Read` every file touched,
`cargo check` + `cargo test`, flag any identifier untraceable to disk.

---

## ndarray Foundation Commits (3 commits, April 11)

### 17bfde3 — fix(hpc/gguf): F16 → F32 QNaN fix

| Item | Status |
|---|---|
| `f16_to_f32` exists at `src/hpc/gguf.rs:417` | VERIFIED |
| QNaN logic: `0x7fc00000 \| (mantissa << 13)` at line 460 | VERIFIED — correct per IEEE 754 |
| Infinity branch: `0x7f800000` at line 457 | VERIFIED |
| Subnormal branch: normalization loop at lines 428-438 | VERIFIED |
| `bf16_to_f32` at line 470 (simple shift) | VERIFIED |
| gguf tests: 5 passed (test_f16_conversion, test_bf16_conversion, etc.) | VERIFIED |
| **Commit verdict** | **VERIFIED** |

### c489d31 — feat(simd_avx512): AVX-512-F RNE path for f32 → bf16

| Item | Status |
|---|---|
| `f32_to_bf16_scalar_rne` at `src/simd_avx512.rs:1834` | VERIFIED |
| `f32_to_bf16_x16_rne` at `src/simd_avx512.rs:1859` | VERIFIED |
| `f32_to_bf16_batch_rne` at `src/simd_avx512.rs:1913` | VERIFIED |
| `convert_f32_to_bf16_avx512f_rne` at line 1938 (inner loop) | VERIFIED |
| NaN handling: `((bits >> 16) as u16) \| 0x0040` at line 1840 | VERIFIED — forced quiet bit |
| Subnormal flush: `& 0x8000` at line 1844 | VERIFIED — preserves sign only |
| RNE bias trick: `wrapping_add(0x7FFF).wrapping_add(lsb)` at line 1847 | VERIFIED — matches Intel SDM |
| SIMD: blend order (normal → subnormal → NaN) at lines 1901-1903 | VERIFIED |
| Test oracle: 15-entry reference table at lines 2203-2226 | VERIFIED — hand-derived from SDM |
| Tests couldn't run via `cargo test` (filtered to 0 by avx512 cfg) | BLOCKED — env issue, not code issue |
| `cargo check` passes for ndarray | VERIFIED |
| **Commit verdict** | **VERIFIED** (code + logic; test execution blocked by env) |

### 7caefe9 — feat(simd): re-export f32_to_bf16_batch_rne / scalar_rne

| Item | Status |
|---|---|
| Re-export at `src/simd.rs:118-121` | VERIFIED |
| `#[cfg(target_arch = "x86_64")]` guard | VERIFIED |
| Doc comment references correct workspace rule | VERIFIED |
| **Commit verdict** | **VERIFIED** |

---

## v2.x Certification Commits (7 commits, April 11, all lance-graph)

### Location

Certification harness: `crates/thinking-engine/examples/certify_jina_v5_7lane.rs`
(2360 LOC, behind `#[cfg(feature = "calibration")]`)

JSON output: `.claude/knowledge/certification/jina-v5-small_7lane.json` (33 KB)

### Crate compilation

| Check | Status |
|---|---|
| `cargo check` on thinking-engine | PASS |
| `cargo check` on bgz-tensor (dependency) | PASS (10 warnings) |
| `cargo test` on bgz-tensor | FAIL (type mismatch E0308 — pre-existing, unrelated to certification) |
| `cargo check` on lance-graph workspace | FAIL (missing protoc — env issue) |

### Import verification

| Import | Traceable to disk? |
|---|---|
| `bgz_tensor::quality` | YES — `crates/bgz-tensor/src/quality.rs`, `pub mod quality` in lib.rs:76 |
| `bgz_tensor::quality::pearson` | YES (used in harness at lines 2243, 2268) |
| `bgz_tensor::quality::spearman` | YES (used in harness at lines 2244, 2269) |

### Function verification (all in certify_jina_v5_7lane.rs)

| Function | Line | Status |
|---|---|---|
| `upper_triangular_f32` | 1873 | VERIFIED |
| `upper_triangular_u8` | 1884 | VERIFIED |
| `upper_triangular_i8` | 1895 | VERIFIED |
| `measure_lane` | 1906 | VERIFIED |
| `bootstrap_ci` | 2003 | VERIFIED |
| `bca_bootstrap_ci` | 2230 | VERIFIED — correct Efron BCa implementation |
| `z_score_normalize` | 2083 | VERIFIED |
| `round4` / `round6` | 2095 / 2100 | VERIFIED |
| `erf` / `phi` / `phi_inv` | 2116 / 2132 / 2139 | VERIFIED — standard implementations |
| SplitMix64 constant `0x9E3779B97F4A7C15` | 2251 | VERIFIED — standard golden-ratio hash |

### Structural integrity of the harness

| Property | Status |
|---|---|
| Loads real data from disk (f32 reference matrix) | VERIFIED (lines 77-87) |
| Extracts upper-triangular pairs (N=32,640 from 256×256) | VERIFIED (lines 97-102) |
| NaN scan at every stage | VERIFIED (lines 105-109) |
| Lane 6 atomic clock target ≥ 0.9999 | VERIFIED (line 58) |
| Compressed lane targets (Pearson ≥ 0.9980, Spearman ≥ 0.9990) | VERIFIED (lines 59-60) |
| JSON output structure matches code | VERIFIED (cross-referenced JSON fields) |

### Per-commit status

| Commit | Version | Verdict | Notes |
|---|---|---|---|
| `3232d8b` | v2.0 | VERIFIED (structure) | Initial harness, 532 LOC. All functions traceable. |
| `48b7387` | v2.0+ | VERIFIED (structure) | Extensions: bootstrap CIs, Belichtungsmesser bands. |
| `c777fce` | v2.1 | VERIFIED (structure) | σ lens correction (120 bands at 1/20 σ). |
| `dfeaab4` | v2.2 | VERIFIED (structure) | Fisher z-transform 3σ CI + jackknife. |
| `6e77fca` | v2.3 | VERIFIED (structure) | γ+φ ICC profile round-trip. |
| `d3a19c2` | v2.4 | VERIFIED (structure) | 4 math corrections + reality anchors. |
| `df19235` | v2.5 | VERIFIED (structure) | BCa bootstrap, CHAODA filter, naive u8 floor. |

**"VERIFIED (structure)"** means: every function, constant, import, and algorithm
in the commit is traceable to a file on disk, the code compiles, and the logic
follows the cited statistical methods. It does NOT mean the harness was executed
in this session — the data files (Jina v5 safetensors) are gitignored and not
present in this environment. Full verification requires running the harness with
data.

### JSON output cross-check

| Metric | JSON value | Knowledge doc value | Match? |
|---|---|---|---|
| Lane 1 Spearman ρ | 0.999992 | 0.999992 | YES |
| Lane 2 Pearson r | 0.999250 | 0.999250 | YES |
| Lane 3 Spearman ρ | 0.999992 | 0.999992 | YES |
| Lane 4 Spearman ρ | 0.999463 | 0.999463 | YES |
| Lane 6 (all three) | ≥ 0.9999 | 0.999978 | YES |

---

## Unverifiable Items (require data or runtime)

1. **JSON numbers vs actual harness output**: The JSON report contains
   6+ decimal precision numbers that look like real measurement output
   (non-round, consistent CIs). But without the Jina v5 data files,
   the harness can't be re-run to confirm the numbers were produced by
   the code and not fabricated. UNVERIFIABLE in this environment.

2. **seven_lane_encoder correctness**: The certification harness consumes
   output from the seven_lane_encoder. Whether the encoder correctly
   implements each lane's encoding is not verified here — that would
   require the safetensors source data.

3. **bgz-tensor test failure**: `cargo test` on bgz-tensor fails with
   E0308 (type mismatch). This is a pre-existing issue unrelated to
   the certification commits but means the bgz-tensor test suite
   can't validate the `quality` module's Pearson/Spearman implementations.

4. **ndarray simd_avx512 tests**: Exist in source (reference oracle
   table at line 2203) but filtered to 0 at test runtime — likely
   due to cfg gating or missing target-cpu flag in the test profile.
   The ndarray binary aborts on OOM in openchat::inference before
   reaching these tests when running unfiltered.

---

## Known Issues Found During Audit

1. **ndarray OOM**: `hpc::openchat::inference::tests::test_kv_cache_grows`
   allocates unrealistic weights, causing SIGABRT. Blocks full test suite.
   Not a fabrication — a resource issue.

2. **lance-graph workspace build**: Missing `protoc` in environment blocks
   `cargo check`/`cargo test` on workspace members. Standalone crates
   (bgz17, bgz-tensor, thinking-engine) compile independently.

3. **bgz-tensor test failure**: Type mismatch E0308 in test compilation.
   Pre-existing, needs investigation.

---

## Audit Methodology

1. `git show <hash> --stat` for each commit
2. `Read` every file touched by the commit on disk
3. `Grep` for every function name, constant, import
4. `Read` the implementation of key functions
5. `cargo check` on the containing crate
6. `cargo test` where possible (targeted by test name)
7. Cross-reference JSON output against knowledge docs
8. Verify statistical algorithms against cited papers (Efron 1987, etc.)

---

## Next Steps

- [ ] Investigate bgz-tensor E0308 test failure
- [ ] Install protoc and run full workspace check
- [ ] If data files become available: re-run certification harness
- [ ] Audit remaining ~50 Claude commits in lance-graph (non-certification)
- [ ] Audit ndarray: 50 Claude commits, prioritized by recency and load-bearing status

---

## Update Protocol

When updating this document:
1. Change the STATUS line at the top
2. Record new findings in the appropriate section
3. Commit with `docs(knowledge): p1 audit — [what changed]`
