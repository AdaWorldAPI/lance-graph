# turbovec ⇄ ndarray/lance-graph — synergy map & measured findings

> **READ BY:** anyone touching `lance-graph-turbovec`, the ndarray AMX/VNNI
> int8 GEMM (`hpc::amx_matmul`), or the bgz-tensor cascade primitives.
> **Status:** FINDING (benchmarked 2026-06-13 on AVX-512+VNNI, no AMX tiles).
> **Branch:** `claude/wonderful-hawking-lodtql` (turbovec + ndarray + lance-graph).

## 0. What turbovec is, and where it goes

`turbovec` = Google Research **TurboQuant** (arXiv 2504.19874, ICLR 2026): a
data-oblivious scalar-quantization ANN index. Pipeline: normalize → random
rotation (faer QR) → per-coord TQ+ calibration → Lloyd-Max 2/3/4-bit →
bit-pack → SIMD nibble-LUT scan (FAISS-FastScan-style `pshufb`/`vqtbl`,
32-vector blocks, u16 accumulators, in-register top-k prune + allowlist
early-exit).

**Placement (answers "does it belong in ndarray?"): NO — it belongs on the
lance-graph spine.** It is a *search index* (IO, id-map, deletes, filtered
search), not a hardware primitive. It sits beside `bgz17` / `deepnsm` /
`bgz-tensor` as an `exclude`d standalone codec/search crate. What belongs in
**ndarray** is the *kernel math* — and ndarray already owns that substrate:
`hpc::clam` + `clam_search` (CLAM neighborhood), `hpc::cam_pq` (PQ-ADC),
`hpc::cascade` (HDR Welford banding), `hpc::amx_matmul` (AMX/VNNI int8 GEMM),
`simd::*` (the typed-lane polyfill). Integration rule: **index here, every
wide op borrowed from `ndarray::simd`.**

P0 wiring: turbovec's `ndarray = "0.17"` (crates.io) is re-pointed to the
AdaWorldAPI fork (`path`), which *is* rust-ndarray 0.17.2 + HPC/SIMD — so the
core array API is unchanged AND `ndarray::simd` becomes reachable. The `blas`
feature is made opt-in (default uses the pure-Rust `matrixmultiply` fallback
for the one encode-time `.dot()`), so no OpenBLAS link is needed. Rust pinned
to 1.95 (`turbovec/rust-toolchain.toml`) to match the fork.

## 1. The polyfill / AMX architecture (the "ship AMX via dispatch" path)

turbovec writes **zero raw intrinsics** in the polyfill path. Scoring is
re-expressed as a batched int8 GEMM and handed to ndarray:

```
S[nq×n] = Q_i8[nq×dim] · X̂ᵀ_i8[dim×n]   via  ndarray::simd::matmul_i8_to_i32
```

`matmul_i8_to_i32` (re-exported through `simd.rs`; impl in `hpc::amx_matmul`)
runs a 4-tier ladder, **bit-identical across tiers**, selected at runtime:

| tier | instr | MAC/instr | when |
|---|---|---|---|
| 1 | AMX `TDPBUSD` tile (byte-asm) | 16 384 | `amx_available()` ∧ 16/16/64-aligned (Sapphire Rapids+) |
| 2 | AVX-512 `VPDPBUSD` zmm | 64 | `avx512vnni` |
| 3 | AVX-VNNI `VPDPBUSD` ymm | 32 | `avxvnni` (Arrow/Meteor Lake) |
| 4 | scalar i32 reference | 1 | otherwise |

So a turbovec built on a Sapphire-Rapids host **ships AMX for free** — no code
change, no `#[cfg]` in turbovec. ndarray "knows how to enable AMX via byte
asm" (`ldtilecfg` / `TDPBUSD` encodings + `amx_available()` CPUID/XCR0/prctl
gating live in `hpc::amx_matmul` + `simd_amx`). The carrier for the tiles is
`hpc::blocked_grid::AmxInt8Grid = BlockedGrid<u8,16,64>` (the 16×64 TDPBUSD
half-square — the "gridlake SoA tiles"). `simd_soa::MultiLaneColumn` is the
layout-only 64B-aligned SoA byte carrier; `simd_ops` is the f32 slice layer.

## 2. THE HEADLINE FINDING (measured)

`n=20 000, dim=512, nq=256, k=10, 4-bit`, AVX-512+VNNI host (`amx_available =
false`), `cargo run --release --example kernel_speed --features
ndarray-simd,bench-internals` in turbovec:

| kernel | ns/query | recall@10 | DB memory |
|---|---|---|---|
| native LUT-ADC (AVX-512BW) | **76 073** | 0.785 | 5 000 KB (4-bit) |
| polyfill GEMM (VPDPBUSD-zmm) | 866 899 | 0.764 | 10 000 KB (i8) |
| scalar reference | 6 267 279 | — | — |

- native vs scalar: **82.4× faster** (the hand-written nibble-LUT kernel earns
  its keep).
- polyfill GEMM vs native: **11.4× slower**.

**Why, and the architectural lesson:** TurboQuant's design *deliberately
trades the matmul away*. LUT-ADC is an O(1) table gather per coordinate; the
GEMM does the full `dim`-length dot per (query,vector) pair. **AMX accelerates
exactly the operation TurboQuant removed.** Even VPDPBUSD (64 MAC/instr)
can't close the algorithmic gap; the AMX tile (256 MAC/instr, ~4× VNNI) would
bring the polyfill from 11.4× → ~3× slower — still a loss. A gather is not a
matmul, and no tile engine makes it one.

**Decision: keep the native LUT kernel as the production path.** AMX is the
wrong tool for *this* index. The polyfill is retained because (a) it proves
the index is `ndarray::simd`-clean and AMX-ready should the workload ever
become matmul-shaped (e.g. an exact-rerank LEAF over a tiny survivor set —
see §3C), and (b) it is the honest measured baseline. Caveat on the 11.4×:
`matmul_i8_to_i32` re-allocates the u8 LHS + colsum-bias buffers per call;
a production AMX rerank would amortise those — but the work-ratio conclusion
(full dot ≫ LUT hit) dominates regardless.

## 3. Synergy map — the three primitives the request named

### (A) HDR popcount stacking early-exit
- **bgz-tensor home:** `stacked.rs` (`StackedBF16x4`, `SearchKey17`,
  `vedic_cascade` 3-stage L1 thresholds), `stacked_n.rs`, `hdr_belichtung.rs`
  (`PaletteCascade` over `ndarray::hpc::cascade`). NB the "popcount" name is
  **vestigial** — `stacked_n.rs:158` deleted sign-Hamming as "not a valid
  metric," replaced by palette-L1; true Hamming survives only in
  `turboquant_kv.rs`.
- **turbovec analogue:** its early-exit is the per-block `block_has_allowed`
  mask skip + the in-register `_mm256_cmp_ps(scores, heap_min)` threshold
  prune (`avx2_post_flush_heap_update`). Same *shape* as the vedic cascade's
  staged reject, but turbovec's thresholds are the running top-k min, not a
  pre-calibrated band.
- **Synergy:** wrap turbovec's nibble-LUT scan as the **TWIG/LEAF** of
  bgz-tensor's HEEL→HIP→TWIG→LEAF cascade — a `SearchKey17`/palette HEEL
  rejects ~85% before turbovec ever runs. turbovec contributes the fast exact
  ADC at the leaf; bgz-tensor contributes the coarse pre-reject it lacks.

### (B) Belichtungsmesser statistical confidence-interval thresholds
- **bgz-tensor home:** `belichtungsmesser.rs` — `calibrate(&[u32])` computes
  μ + σ of pairwise L1 distances → 12 half-σ bands → `classify`/`three_stroke`
  σ-relative reject. Online σ + recalibration (`ShiftAlert`) delegated to
  `ndarray::hpc::cascade`. **No Jirak grounding** — bands are hand-set indices
  (open obligation per I-NOISE-FLOOR-JIRAK; any "N σ above noise floor" claim
  on these palette-correlated distances needs a Jirak-2016 weak-dependence
  bound, not classical Berry-Esseen).
- **turbovec gap → synergy:** turbovec has **no statistical threshold** — it
  keeps a fixed top-k heap and prunes by the current heap-min only. A
  `Belichtungsmesser` calibrated on a query's score distribution could supply
  a **σ-gated block reject** (skip a 32-vector block whose best-possible score
  is below μ+kσ) *in addition to* the heap-min prune — turning fixed-k search
  into confidence-bounded early termination. This is the cleanest, lowest-risk
  turbovec⇄bgz-tensor wiring and the one most worth prototyping next.

### (C) preheating vs palette256 ranking / attention headers
- **palette256 ranking (bgz-tensor):** `WeightPalette::build(rows, k≤256)` —
  256-archetype CLAM codebook by *furthest-point sampling* (NOT k-means;
  metric-safe radius bound) + L1 ranking; `Codebook4096` is the 2-level 6+6
  variant; `AttentionTable`/`CompiledHead` freeze all k² pairs into a u16 LUT
  ("attention headers"). Ranking = nearest-archetype-by-L1.
- **preheating:** **absent in bgz-tensor** (tables are built once, eagerly; no
  lazy/warm path) — but turbovec *has* it: `TurboQuantIndex::prepare()` eagerly
  fills the `OnceLock` caches (rotation matrix, Lloyd-Max centroids, blocked
  SIMD layout) so the first query pays no init cost. The polyfill mirrors this
  with the lazily-built i8 reconstruction (`reconstruct_db_i8_transposed`,
  cached in `TurboVec`'s `OnceLock`).
- **Synergy / contrast:** turbovec's `prepare()` is the *codec/layout* warmup
  (data→SIMD-ready); palette256 ranking is a *coarse quantizer* (data→256
  archetypes). They compose: a palette256 HEEL ranks a query to a handful of
  archetype buckets, then turbovec — **preheated** — runs the exact ADC only
  within those buckets. "Preheating vs palette256" is not either/or: preheat
  the leaf scanner, palette-rank the coarse router.

## 4. Overlap / convergence (prior-art guard)

turbovec's scoring is ~80% conceptual overlap with two things already present:
- `bgz-tensor::turboquant_kv.rs` (`TurboQuantEntry`/`TurboQuantKvCache`) —
  literally "TurboQuant": gain-shape → i4 → sign-fingerprint → Hamming-cascade
  → cosine rerank. The KV-cache cousin; uses sign-Hamming (the metric
  `stacked_n` deleted — a `truth-architect`/`iron-rule-savant` call, not free).
- `ndarray::hpc::cam_pq::PackedDatabase::distance_batch_avx512` — already a
  SIMD-LUT PQ-ADC scan with a 99%-rejection HEEL/branch/full cascade.

**This crate does NOT re-implement those** — it brings the *upstream Google
index* in verbatim (the published recall/speed numbers are the value) and adds
only the bridge + the `ndarray::simd` polyfill. If a future session wants ONE
unified ADC, the convergence target is `cam_pq` (it already has the cascade);
turbovec's contribution there would be TQ+ calibration + the data-oblivious
rotation, not a third scan kernel.

## 5. Reproduce

```bash
# native vs polyfill vs scalar + recall (this host = VNNI):
cd /home/user/turbovec && cargo run --release --example kernel_speed \
    --features ndarray-simd,bench-internals
# on a Sapphire Rapids / AMX host, prepend RUSTFLAGS="-C target-cpu=native"
# and matmul_i8_to_i32 lights up the TDPBUSD tile path (amx_available → true).

# bridge tests:
cargo test --manifest-path crates/lance-graph-turbovec/Cargo.toml
# polyfill recall + reconstruct tests:
cd /home/user/turbovec && cargo test -p turbovec --features ndarray-simd search_polyfill
```
