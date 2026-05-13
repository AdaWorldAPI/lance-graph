
## W5 — sprint-log-4 run (2026-05-13)

### Task
Spec TD-SIMD-CALLCENTER-BATCH-PATHS-1 (P2): callcenter consumer-side batch paths still scalar-loop where ndarray::simd is canonical (§19.2).

### Research performed
1. Grepped `crates/lance-graph-callcenter/src/` for scalar loops — found all hot loops in `vsa_udfs.rs` only; other loop sites are non-compute (audit, zerocopy schema, postgrest).
2. Inspected `vsa_udfs.rs` in full — confirmed 5 scalar hot functions: `bundle_op`, `hamming_dist_op`, `bytes_to_words`, `words_to_bytes`, `braid_at_op`, `top_k_op`.
3. Explored `/home/user/ndarray/src/`: found `simd.rs`, `simd_avx2.rs`, `simd_ops.rs`, `hpc/bitwise.rs`, `hpc/vsa.rs`, `hpc/simd_caps.rs`, `hpc/simd_dispatch.rs`.
4. Confirmed canonical primitives: `hamming_distance_raw`, `hamming_batch_raw`, `hamming_top_k_raw` in `ndarray::hpc::bitwise`; `vsa_bind`, `vsa_bundle`, `vsa_similarity`, `vsa_hamming` in `ndarray::hpc::vsa`; `SimdCaps` / `simd_caps()` singleton in `ndarray::hpc::simd_caps`.
5. Noted: names in prompt spec (`vsa_cosine_batch`, `vsa_bundle_simd`, `vsa_bind_simd_inplace`, `Vsa16kF32`) do NOT exist — documented actual canonical names in spec §3.3.
6. Confirmed no `ndarray-hpc` feature flag exists yet in callcenter Cargo.toml — must be added.
7. Identified Arrow buffer contiguity question for `hamming_batch_raw` as open question.

### Output
- Spec written: `.claude/specs/td-simd-callcenter-batch.md` (15,926 bytes / 413 lines)
- Covers: scalar inventory, ndarray surface (actual names), 2 migration diff sketches, bench plan w/ 4x AVX2 assertion, feature-flag fallback, W6 coordination, 4 open questions.

### Key findings
- ALL hot scalar loops are in `vsa_udfs.rs` — no `batch.rs`, `similarity.rs`, or `bundle.rs` exist in this crate (prompt spec assumed files that don't exist).
- ndarray `VsaVector` = `[u64; 256]` exactly matches callcenter `[u64; FP_WORDS]` — zero-copy bridge possible via `from_bytes`.
- `hamming_batch_raw` enables true Arrow-columnar batch (one call per DataFusion batch, not N calls per row) — largest win.
- `simd_caps()` singleton shared with thinking-engine (W6) — no per-crate dispatch cache needed.

