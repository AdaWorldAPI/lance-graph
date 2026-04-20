# Plan — Codec Candidate Sweep via Lab Infra (JIT-first, no rebuilds)

> **Author note (2026-04-20):** Operationalises the #220 "What's
> Needed to Fix" list (wider codebook / residual PQ / Hadamard
> pre-rotation / OPQ) as a parameter sweep through the lab
> endpoint, not as four separate rebuild-heavy branches.

## Context & Prerequisites (read first)

- `.claude/knowledge/lab-vs-canonical-surface.md` — especially
  "Why the Lab Surface Exists" (three-part stack: API + Planner +
  JIT), "The third purpose — thinking harvest", and I11 (measurable
  stack, not a black box).
- PR #219 — lab-gated CAM-PQ candidates; ICC 0.9998 was
  **synthetic / overfit-on-training**, not tokens.
- PR #220 — honest negative result: reconstruction ICC 0.195 mean,
  0/234 ≥ 0.99 on real Qwen3-TTS-0.6B safetensors; lists 4 fixes
  (a, b, c, d) as the way forward.
- PR #221 — REST/gRPC scaffolding + `CodecResearchBridge` /
  `PlannerAwareness : OrchestrationBridge`. This is the surface we
  extend in Phase 0.
- `crates/lance-graph-contract/src/jit.rs` — `JitCompiler`,
  `StyleRegistry`, `KernelHandle`. Already in the contract; we
  consume it, don't invent it.

## Why JIT is the spine

Everything that changes between candidates is a JIT artefact:

- **Codec decode kernel** — `(centroid_count, subspace_count,
  residual_depth)` changes the kernel shape.
- **Pre-rotation** — Identity / Hadamard (Sylvester 2^k) / learned
  OPQ rotation — each is a different SIMD routine.
- **Distance table layout** — Hamming vs cosine vs ADC; u8 vs u16
  entries; 256² vs 1024² size.
- **Token-agreement comparator** — top-k match, per-position
  divergence, latency measurement.

One long-running `shader-lab` binary + `JitCompiler` = the sweep
runs thousands of candidates without a single `cargo` invocation
after Phase 0. `KernelHandle`s are cached by `CodecParams` hash
and reused across calls.

## Phase 0 — API hardening (one-time rebuild; everything else is
rebuild-free)

**Rule:** commit all surface changes in Phase 0, rebuild
`shader-lab` exactly once, then freeze the Wire contract for the
duration of the sweep. Any mid-sweep endpoint tweak forfeits the
JIT benefit.

### D0.1 — Extend `WireCalibrate` with full `CodecParams`

`crates/cognitive-shader-driver/src/wire.rs` — add:

```rust
pub struct CodecParams {
    pub subspaces: u32,          // e.g. 6
    pub centroids: u32,          // 256 / 512 / 1024 / 2048
    pub residual_depth: u8,      // 0 = no residual, 1+ = residual PQ
    pub pre_rotation: Rotation,  // Identity / Hadamard / Opq(learned)
    pub distance: Distance,      // Hamming / Cosine / Adc
    pub calibration_rows: u32,   // held-out row selection
    pub seed: u64,               // reproducibility
}

pub enum Rotation { Identity, Hadamard, Opq { matrix_blob_id: u64 } }
pub enum Distance { Hamming, Cosine, Adc }

pub struct WireCalibrate {
    pub tensor_path: String,
    pub params: CodecParams,
    pub measure: MeasureSet,     // { reconstruction, icc, token_agreement }
}
```

Handler stays the same shape; only the payload grows.
~120 LOC (DTO + serde glue + handler wiring).

### D0.2 — New endpoint `WireTokenAgreement`

```rust
pub struct WireTokenAgreement {
    pub model_path: String,         // safetensors root
    pub reference: Baseline,        // Passthrough by default
    pub candidate: CodecParams,     // from D0.1
    pub prompt_set_blob_id: u64,    // pre-uploaded prompt blobs
    pub n_tokens: u32,              // how far to decode
}
pub struct WireTokenAgreementResult {
    pub top1_rate: f32,
    pub top5_rate: f32,
    pub divergence_positions: Vec<u32>,     // which tokens differ
    pub per_layer_mse: Vec<f32>,            // pinpoint where error grows
    pub candidate_latency_us: u64,
    pub reference_latency_us: u64,
}
```

This is the **I11 cert gate** from `lab-vs-canonical-surface.md`.
~160 LOC (DTO + handler + ref-model load glue).

### D0.3 — Sweep streaming endpoint `WireSweep`

```rust
pub struct WireSweepRequest {
    pub tensor_path: String,
    pub grid: SweepGrid,           // declarative param grid
    pub measure: MeasureSet,
    pub log_to_lance: Option<String>, // path where per-candidate row appends
}
pub struct SweepGrid {
    pub subspaces: Vec<u32>,
    pub centroids: Vec<u32>,
    pub residual_depths: Vec<u8>,
    pub rotations: Vec<Rotation>,
    pub distances: Vec<Distance>,
}
```

Streams one `WireCalibrateResult` + `WireTokenAgreementResult`
pair per grid point via Server-Sent Events or gRPC stream. Server
holds the shader-lab process; no per-candidate curl spin-up.
~200 LOC (streaming handler + Lance append writer).

### D0.4 — Freeze the surface

- Commit D0.1 + D0.2 + D0.3 as one PR.
- Rebuild `shader-lab` binary.
- `cargo test -p lance-graph-contract` must still pass (Wire
  additions are additive, no contract-level changes).
- No further Wire changes allowed during Phase 1-5.

**Total Phase 0: ~480 LOC, one rebuild, one PR.**

## JIT Kernel Contract (non-negotiable; binds every kernel in Phases 1-3)

Every JIT-emitted kernel in this plan obeys four hard rules. Any
kernel that violates one is rejected.

### Rule A — Tensor access via stdlib `slice::array_windows::<N>()` + `ndarray::simd::*` loaders

Per `ndarray/.claude/rules/data-flow.md` Pattern 1: SIMD reads are
zero-copy `&[u8]` borrows from the backing store (PackedDatabase
/ Arrow buffer / BindSpace column). Fixed-size windowing uses
the **stdlib** const-generic primitive `slice::array_windows::<N>()`
(stable since Rust 1.77), which yields `&[T; N]` tuples with
bounds handled by the compiler. No manual index math, no raw
pointer reach, no per-kernel slicing arithmetic.

```rust
use ndarray::simd::F32x16;

let row_bytes: &[u8] = column.row_slice(row_idx);        // zero-copy borrow, 64-byte aligned

// Stdlib array_windows (const generic, stable 1.77) — one F32x16 lane per window:
for w in row_bytes.array_windows::<64>() {
    // w: &[u8; 64] — bounds guaranteed by the type
    let lane = F32x16::from_slice(bytemuck::cast_slice(w));
    // …SIMD accumulate via ndarray::simd::* ops…
}

// For non-overlapping subspace reads use slice::array_chunks::<N>() (stable 1.88):
for chunk in row_bytes.array_chunks::<SUBSPACE_BYTES>() {
    // chunk: &[u8; SUBSPACE_BYTES]
    …
}
```

Why `array_windows` specifically: the const-generic type
guarantees each window has exactly the lane width the SIMD type
expects, so `from_slice` on it never panics and LLVM can elide
the bounds check. Hand-rolled windowing is rejected.

**SoA source of the `&[u8]` slice.** The row bytes come from a
`BindSpace` column — `FingerprintColumns`, `QualiaColumn`,
`MetaColumn`, or `EdgeColumn` per the struct-of-arrays identity in
`lab-vs-canonical-surface.md`. The codec JIT reads from the same
columns the shader sweeps:

```rust
use cognitive_shader_driver::{BindSpace, FingerprintColumns};

let fp_col: &FingerprintColumns = bindspace.fingerprints();
let row_bytes: &[u8] = fp_col.row_bytes(row_idx);   // zero-copy into SoA column
for w in row_bytes.array_windows::<64>() { /* …SIMD accumulate… */ }
```

No new data structures. The SoA column IS the input surface.

### Rule B — SIMD exclusively via `ndarray::simd::*` and its AMX sibling modules

All primitives already exist in ndarray. The codec JIT consumes
them as-is; **no ndarray changes**:

```rust
// Canonical lane types (ndarray::simd re-exports):
use ndarray::simd::{F32x16, U8x64, Fingerprint, hamming_distance_raw, popcount_raw};

// AMX + VNNI (sibling top-level module, canonical AMX surface):
use ndarray::simd_amx::{amx_available, vnni_dot_u8_i8, vnni_matvec, matvec_dispatch};

// AMX tile primitives (inline-asm stable path; Rust-lang #126622 keeps
// intrinsics nightly, so ndarray ships stable inline asm):
use ndarray::hpc::amx_matmul::{
    tile_loadconfig, tile_zero, tile_load, tile_store, tile_release,
    tile_dpbusd, tile_dpbf16ps, vnni_pack_bf16,
};

// Runtime caps (at hpc::simd_caps — use the existing path, do not propose
// a re-export; "don't touch ndarray"):
use ndarray::hpc::simd_caps::{simd_caps, SimdCaps};

// Wrong (violates I2):
use ndarray::hpc::simd_avx512::F32x16;        // private backend reach
use std::arch::x86_64::_mm512_loadu_ps;        // hand-rolled intrinsic
```

Everything the sweep needs is already in ndarray. This plan wires
the existing surface into the lab infra (REST handlers +
`CodecKernelCache` + `CodecResearchBridge`); it adds nothing to
ndarray.

### Rule C — Polyfill hierarchy: Intel AMX → AVX-512 VNNI → AVX-512 baseline → AVX-2 → scalar

The SIMD tier each JIT-emitted kernel lands on follows this
strict polyfill chain — tier 1 is tried first, each tier falls
through to the next when unavailable:

**Iron rule — SoA never scalarises without ndarray.** If a kernel
runs scalar, the SoA invariant is broken. Every tier in the chain
below calls `ndarray::simd::*` or `ndarray::simd_amx::*` or
`ndarray::hpc::amx_matmul::*` — these modules handle their own
internal scalar fallback for exotic targets; the consumer never
hand-rolls a scalar loop.

| Tier | Primitive | Source | When selected | MACs / instr |
|---|---|---|---|---|
| **1 — Intel AMX tiles** (preferred for matmul-heavy paths: OPQ, distance-table build) | `tile_dpbusd` (u8×i8→i32) / `tile_dpbf16ps` (bf16×bf16→f32) | `ndarray::hpc::amx_matmul::*` | `ndarray::simd_amx::amx_available() == true` (Sapphire Rapids+, OS has enabled XCR0 tile bits 17/18, Linux `prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` succeeded) | **256** |
| 2 — AVX-512 VNNI | `vnni_dot_u8_i8`, `vnni_matvec`, `matvec_dispatch` | `ndarray::simd_amx::*` (VNNI lives one tier down from AMX, stable intrinsics) | AVX-512 VNNI subset | 64 |
| 3 — AVX-512 baseline | `F32x16`, `U8x64`, `F64x8` | `ndarray::simd::*` (mandatory default: ndarray's `.cargo/config.toml` sets `target-cpu=x86-64-v4`) | Always on canonical build targets | 16 |
| 4 — AVX-2 fallback | `F32x8`, `F64x4` | `ndarray::simd::*` (cfg-gated; triggers only when build cfg drops to `x86-64-v3`) | Compile-time cfg | 8 |

Note the absence of a consumer-visible "scalar" tier. Scalar
fallback — when it exists at all — lives inside `ndarray::simd`
for non-x86 correctness; the codec JIT never emits it directly
and never short-circuits out of an ndarray call into a hand-
written loop. Any such short-circuit on a SoA path is a contract
violation.

**Dispatch shape the JIT emits (real primitive names only):**

```rust
use ndarray::simd_amx::amx_available;
use ndarray::hpc::amx_matmul::{tile_dpbusd, tile_dpbf16ps};
use ndarray::simd::F32x16;

if amx_available() && kernel_params.is_matmul_heavy() {
    // Tier 1: Intel AMX tile matmul. Codebook distance-table build
    // drops from 24-48h (non-AMX) to ~1:20h at this tier per
    // simd_amx.rs top-of-module measurement.
    unsafe { tile_dpbf16ps(); }    // or tile_dpbusd for u8×i8 accumulators
} else {
    // Tiers 2-4 via ndarray::simd::* — target-cpu=x86-64-v4 keeps
    // Tier 3 (F32x16) as the always-available floor on canonical
    // builds. cfg resolves the specific lane type at compile time.
    // No hand-rolled scalar "else" branch — if ndarray::simd were
    // unavailable the SoA path itself would not be the right one.
    let lane = F32x16::from_slice(bytemuck::cast_slice(window));
    /* …accumulate via F32x16 ops… */
}
```

**Why Tier 1 matters for this sweep specifically.** The plan
exercises ~200 codec candidates across (centroids × subspaces
× residual depth × rotation × distance). On Sapphire Rapids
hardware, AMX drops codebook distance-table build from 24-48 h
to ~1 h 20 min (measured; cited in `simd_amx.rs` header). For
the four #220 fixes in particular:

- (a) **wider codebook (1024+ centroids)** — bigger distance
  table, so AMX matters more.
- (b) **residual PQ** — two distance-table lookups per row, AMX
  helps both.
- (c) **Hadamard pre-rotation** — add/sub butterfly, NOT matmul:
  stays at Tier 3 F32x16 (already fast; AMX adds no value here).
- (d) **OPQ** — learned rotation matrix applied as matmul → Tier 1
  AMX is the dominant speedup path.

**The JIT does NOT emit AMX inline assembly.** It emits IR that
calls `ndarray::hpc::amx_matmul::tile_*` primitives, which are
themselves stable-Rust-1.94 inline asm (verified on real
Sapphire Rapids hardware per the `simd_amx.rs` module header:
LDTILECFG / TILEZERO / TDPBUSD / TDPBF16PS / TILERELEASE all
tested on kernel 6.18.5 with XCR0 bits 17+18 set). Rust-lang
issue #126622 tracks AMX intrinsic stabilization; until it
lands, inline asm is the canonical stable path and the codec
JIT consumes it through `ndarray::hpc::amx_matmul::*`, never
directly.

### Reality-check against existing codec-findings (do NOT re-derive)

Per `.claude/knowledge/codec-findings-2026-04-20.md`:

- **Had-Q5×D-R** (shared codebook) — already ICC ≈ 0.99 at
  ~0 per-row bytes on q_proj / k_proj / gate_proj. **Argmax
  compression with shared codebook is solved.**
- **I8-Hadamard** (per-row only) — ICC ≈ 0.9 at 9 B/row. Leader
  for no-shared-codebook constraint.
- **Zipper family** — tops at ICC ≈ 0.2, serves bundling /
  progressive / anti-moiré axis, NOT argmax ICC.
- **Fractal leaf descriptors** — sign-flip invariant (ICC
  ≈ −0.999); **DEAD** without breaking the invariance.

The sweep here does NOT re-explore what's measured. It focuses
on the #220 candidates (wider codebook, residual PQ, Hadamard
pre-rotation with trained codebook, OPQ) and measures their
**token agreement** — the missing axis that reconstruction ICC
alone doesn't close.

### Rule D — Configuration is JSON / YAML / REST only

No codec candidate is defined in Rust. Every kernel shape is
fully expressed as declarative config. Three equivalent surfaces,
one schema (`CodecParams`):

**YAML** (human-authored sweeps, under `configs/codec/*.yaml`):

```yaml
# configs/codec/cam_pq_wide_residual_hadamard.yaml
name: cam_pq_wide_residual_hadamard
subspaces: 6
centroids: 1024
residual_depth: 1
pre_rotation:
  kind: hadamard
  dim: 4096
distance: adc
calibration_rows: 2048
seed: 42
```

**JSON** (REST payload for sweeps, e.g. `curl -d @file.json`):

```json
{
  "name": "cam_pq_wide_residual_hadamard",
  "subspaces": 6,
  "centroids": 1024,
  "residual_depth": 1,
  "pre_rotation": { "kind": "hadamard", "dim": 4096 },
  "distance": "adc",
  "calibration_rows": 2048,
  "seed": 42
}
```

**REST endpoint** (identical schema, SSE-streamed results):

```
POST /v1/shader/calibrate
Content-Type: application/json
Body: <the JSON above>
```

Adding a new codec candidate means authoring a YAML file or
constructing a JSON body. **Zero Rust changes. Zero rebuilds.**
The JIT kernel cache hashes `CodecParams` and compiles once per
unique shape; everything after is cache hits.

### Rule enforcement — test gate in Phase 0

Phase 0's verification gate adds:

- `kernel_contract_test` — iterates a list of `CodecParams` (from
  `configs/codec/*.yaml`), compiles each, scans emitted IR for
  uses of `ndarray::simd::array_window` and `ndarray::simd::*`
  symbols, fails if any kernel reaches `std::arch::*` or
  `ndarray::hpc::*`.
- `amx_dispatch_test` (aarch64-apple-darwin only, gated with
  `#[cfg(all(target_arch = "aarch64", target_os = "macos"))]`) —
  verifies `simd_caps().has_amx() == true` on M-series and that a
  rotation kernel's trace records `backend = "amx"` for that
  call.

These tests fire as part of `cargo test -p cognitive-shader-driver
--features lab` in Phase 0 CI; any Phase 1+ commit that breaks
them is rejected.

## Phase 1 — JIT codec kernels (rebuild-free from here on)

### D1.1 — `CodecParams → KernelHandle` via `JitCompiler`

`crates/cognitive-shader-driver/src/codec_research.rs` — add:

```rust
use ndarray::simd::{array_window, simd_caps, F32x16, U8x64};

struct CodecKernelCache {
    handles: HashMap<u64 /* CodecParams hash */, KernelHandle>,
    compiler: JitCompiler,   // Cranelift via jitson
    caps: SimdCaps,          // from ndarray::simd::simd_caps()
}

impl CodecKernelCache {
    fn kernel_for(&mut self, params: &CodecParams) -> &KernelHandle {
        let key = hash_codec_params(params);
        self.handles.entry(key).or_insert_with(|| {
            // codec_ir emits calls to ndarray::simd::* only:
            //   - array_window(tensor, row, cnt) for tensor access
            //   - U8x64::from_lanes(...) for centroid index reads
            //   - F32x16 arithmetic for ADC distance accumulation
            // Zero std::arch, zero ndarray::hpc reach.
            self.compiler.compile(codec_ir(params, &self.caps))
        })
    }
}

fn codec_ir(params: &CodecParams, caps: &SimdCaps) -> KernelIr {
    // Emits IR that:
    //   for each subspace s:
    //     let w = array_window(input, s * sub_dim, sub_dim);
    //     let d = adc_distances_simd::<F32x16>(w, codebook[s]);
    //     accumulate into row_distance via caps-aware reduction
    //   if params.residual_depth > 0: recurse on residual
    ...
}
```

The JIT never emits raw intrinsics; it emits IR calls to
`ndarray::simd::*`. Those resolve to AMX / AVX-512 / NEON /
scalar at link time via `simd_caps()`. Compile time: ~5–20 ms
per unique `CodecParams` shape; cached forever after. ~180 LOC.

### D1.2 — Rotation primitives as JIT kernels (AMX-backed on Apple)

- **Identity** — no-op. Kernel returns the input window
  unchanged. 0 LOC runtime.
- **Hadamard** — Sylvester construction at dim = 2^k. The JIT
  emits calls to `ndarray::simd::hadamard_butterfly(window,
  caps)`; that primitive dispatches to AMX tile butterflies when
  `caps.has_amx()`, AVX-512 permute-add on x86_64+AVX512, NEON
  SWAR otherwise. Window iteration uses `array_window` over the
  row. ~90 LOC.
- **Opq(matrix_blob_id)** — load the learned rotation matrix from
  a Lance blob column (one-time per matrix_blob_id). JIT emits
  calls to `ndarray::simd::matmul_tiled(window, rot_matrix,
  caps)`; that primitive dispatches to **AMX tile-matmul when
  available** (best path on M-series), AVX-512 VNNI / FMA
  otherwise. Matrix is learned offline via a separate training
  pipeline; blob ID is part of the YAML/JSON config. ~100 LOC.

Rotation is a separate `KernelHandle` composed with the decode
kernel at call time (see D1.3 for composition). ~190 LOC total.

### D1.3 — Residual PQ via JIT composition

Encode residuals after first-pass decode; second-pass PQ on the
residual. All three stages (first-decode, subtract, second-decode,
add) are `array_window`-driven and SIMD via `ndarray::simd::*`:

```
candidate_kernel = jit.compose(&[
    first_pass_decode(CodecParams { residual_depth: 0, .. }),
    //   reads via array_window, accumulates via F32x16
    ndarray::simd::sub_tiled,     // SIMD subtract, AMX-backed on Apple
    second_pass_decode(CodecParams::residual_shape(params)),
    ndarray::simd::add_tiled,     // SIMD add
]);
```

`jit.compose` emits a straight-line Cranelift function, inlining
each stage; no runtime function-call overhead. Every stage still
obeys Rules A-D of the kernel contract. ~150 LOC.

**Total Phase 1: ~520 LOC; no canonical-surface changes; all
behind `--features lab`.**

## Phase 2 — Token-agreement harness (the I11 cert gate)

### D2.1 — Reference model loader

`crates/cognitive-shader-driver/src/token_agreement.rs` — new
module. Loads a reference model via ndarray:

- safetensors → `ndarray::hpc::fingerprint` tables + KV cache
- Passthrough baseline uses the untouched weights.
- Candidate path routes the weights through the JIT codec kernel
  from D1.1.

~180 LOC.

### D2.2 — Decode-and-compare loop

For each prompt in the prompt set:

```
reference_logits = decode(model, prompt, n_tokens, Passthrough)
candidate_logits = decode(model, prompt, n_tokens, candidate_kernel)
top1_match       = argmax(ref) == argmax(cand)
top5_match       = argmax(ref) in topk(cand, 5)
per_layer_mse    = [mse(ref_layer_k, cand_layer_k) for k in 0..n_layers]
```

Aggregate across prompts and tokens → `WireTokenAgreementResult`.
~220 LOC.

### D2.3 — Handler wiring

`src/serve.rs::token_agreement_handler` — reads
`WireTokenAgreement`, dispatches through D2.1/D2.2, returns
`WireTokenAgreementResult`. ~60 LOC.

**Total Phase 2: ~460 LOC. Measures the gate that actually
certifies a codec.**

## Phase 3 — Sweep driver + Lance logger

### D3.1 — Server-side sweep handler

`src/serve.rs::sweep_handler` — reads `WireSweepRequest`,
enumerates the grid, calls D0.1 + D0.2 per grid point, appends
each result row to Lance via `lance::write_fragment`. Streams
progress to the client via SSE.

~200 LOC.

### D3.2 — Client-side driver (no rebuild; just curl)

`scripts/codec_sweep.sh` — bash script:

```bash
curl -N -X POST http://localhost:8080/v1/shader/sweep \
  -d @configs/phase1-centroid-sweep.json
```

Each config is a small JSON file declaring the grid. Configs live
under `configs/`; adding a new one is a text edit.

~20 LOC script + N config files.

**Total Phase 3: ~220 LOC + configs. The sweep runs without
touching cargo for the rest of the project's lifetime.**

## Phase 4 — Frontier analysis

### D4.1 — DataFusion queries over the Lance log

```sql
SELECT centroids, residual_depth, rotation,
       AVG(token_agreement_top1) AS t1,
       AVG(reconstruction_icc)   AS icc,
       AVG(bytes_per_row)        AS br
FROM sweep_results
GROUP BY centroids, residual_depth, rotation
ORDER BY t1 DESC
LIMIT 20
```

Answers "which of the four #220 fixes actually move token
agreement?" empirically, not speculatively.

### D4.2 — Pareto-frontier notebook

`.claude/analyses/codec_frontier.py` — reads the Lance log,
plots `(bytes_per_row × token_agreement_top1)` Pareto frontier,
highlights which `CodecParams` live on it.

~120 LOC notebook / script.

**Total Phase 4: ~120 LOC. The empirical answer to "which fix
wins" falls out of the data, not a hand-argued hypothesis.**

## Phase 5 — Graduation

Only a candidate that passes these gates graduates:

1. `reconstruction_icc ≥ 0.99` on held-out rows.
2. `token_agreement_top1 ≥ 0.99` on the prompt set.
3. `token_agreement_top5 ≥ 0.999`.
4. `bytes_per_row ≤ 16` (i.e. still a real compression).

Graduation means: add a `StepDomain::Codec(CodecParams)` variant
(or the right existing variant) with an `OrchestrationBridge`
impl on the **planner side**, not on the lab side. The lab
endpoint stays live for continued sweep iteration. The canonical
consumer (the real production pipeline) walks `UnifiedStep`, not
`WireCalibrate`.

~120 LOC for the graduation bridge impl (when a candidate
actually earns it).

## Totals

| Phase | LOC | Rebuilds | What it buys |
|---|---|---|---|
| 0 | ~480 | 1 | Hardened Wire surface; no mid-sweep churn |
| 1 | ~520 | 0 | JIT codec + rotation + residual kernels |
| 2 | ~460 | 0 | Token-agreement cert gate (I11) |
| 3 | ~220 | 0 | Sweep driver + Lance append logger |
| 4 | ~120 | 0 | Frontier analysis / winner selection |
| 5 | ~120 | 1 per winner | Graduation to canonical surface |

**~1,920 LOC, 1 upfront rebuild, unlimited candidates for free
afterwards.** Compare to the naive path: 4 fixes × 8–17 min
rebuild × N parameter tweaks per fix = hundreds of hours.

## Measurability (I11 enforcement)

Every JIT kernel emits trace fields through the existing
`thinking_trace` / `ShaderHit` contract:

- `kernel_hash` — which exact kernel ran.
- `compile_time_us` — JIT compile cost (cached on second hit).
- `reconstruction_per_row` — full histogram, not just mean.
- `token_divergence_positions` — exact indices where candidate
  differs from reference.
- `per_layer_mse` — where in the stack error accumulates.

Any proposal to drop fields from this trace for "perf" violates
I11 and is rejected. The lab surface is the observation port.

## Explicit Non-Scope

- No changes to `UnifiedStep` / `OrchestrationBridge` until a
  candidate graduates (Phase 5).
- No new `StepDomain` variants during Phases 0-4.
- No changes to the canonical re-exports from
  `cognitive-shader-driver::lib`.
- No OpenAI-compatible endpoints for codec sweep — this is
  research transport, not production.
- No CAM-PQ-specific assumptions baked into the Wire surface —
  `CodecParams` is codec-agnostic so future candidates (non-PQ,
  e.g. lattice quantization) plug in without Wire churn.

## Verification

- `cargo check -p cognitive-shader-driver --features lab` after
  each phase.
- `cargo test -p lance-graph-contract` — 133/133 must pass after
  Phase 0; Wire DTO additions are additive.
- **Sweep self-test:** a single-point grid with `Identity`
  rotation + `centroids=256` + `residual_depth=0` must reproduce
  PR #220's reconstruction ICC numbers (≈ 0.195 mean) —
  demonstrates the pipeline doesn't silently "fix" the prior
  measurement.
- **JIT round-trip self-test:** compile → execute → recompile
  same params → handle hit cache; verify identical output and
  `compile_time_us == 0` on second call.
- **Token-agreement regression gate:** Passthrough ↔ Passthrough
  must return top1_rate = 1.0 exactly. Any drift indicates
  non-determinism in the harness.

## Branch / PR Shape

- **PR A** (Phase 0): hardens the Wire surface. One rebuild;
  should merge quickly since it's additive.
- **PR B** (Phases 1 + 2): JIT kernels + token-agreement. No
  further Wire changes. Depends on PR A.
- **PR C** (Phases 3 + 4): Sweep driver + analysis. Pure lab
  tooling. Depends on PR B.
- **PR D** (Phase 5): fires only when a candidate graduates —
  narrow graduation bridge impl.

Total 4 PRs over ~1,920 LOC. Each PR has a crisp deliverable and
a clean verification step.

## What this plan is NOT

- It is not a commitment to any specific codec "winning." The
  four #220 fixes might all fail the token-agreement gate. That's
  a valid outcome — the plan is measurement infrastructure, not a
  codec proposal. Winners emerge from D4 frontier analysis, not
  from the plan author's prior.
- It is not coupled to CAM-PQ. `CodecParams` generalises to
  lattice quantization, residual vector quantization, neural
  codecs — any codec where decoding is parameterised by a small
  struct fits the sweep driver unchanged.
