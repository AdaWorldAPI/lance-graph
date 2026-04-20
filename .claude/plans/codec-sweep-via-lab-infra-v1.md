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

## Phase 1 — JIT codec kernels (rebuild-free from here on)

### D1.1 — `CodecParams → KernelHandle` via `JitCompiler`

`crates/cognitive-shader-driver/src/codec_research.rs` — add:

```rust
struct CodecKernelCache {
    handles: HashMap<u64 /* CodecParams hash */, KernelHandle>,
    compiler: JitCompiler,   // Cranelift via jitson
}

impl CodecKernelCache {
    fn kernel_for(&mut self, params: &CodecParams) -> &KernelHandle {
        let key = hash_codec_params(params);
        self.handles.entry(key).or_insert_with(|| {
            self.compiler.compile(codec_ir(params))
        })
    }
}
```

Cranelift emits a decode function specialised to
`(subspaces, centroids, residual_depth, distance)`. Typical
compile time on our hardware: ~5–20 ms per unique shape; cached
forever after. ~180 LOC.

### D1.2 — Rotation primitives as JIT kernels

- **Identity** — no-op, 0 LOC runtime.
- **Hadamard** — Sylvester construction at dim = 2^k. JIT emits
  XOR / add-subtract butterfly with SIMD vector width
  specialisation. ~90 LOC.
- **Opq(matrix_blob_id)** — load learned rotation matrix from
  blob store (Lance column), JIT emits unrolled matmul over the
  matrix. Matrix is learned offline; blob ID points to it.
  ~100 LOC.

Rotation is a separate KernelHandle composed with the decode
kernel at call time. ~190 LOC total.

### D1.3 — Residual PQ via JIT composition

Encode residuals after first-pass decode; second-pass PQ on the
residual. In JIT terms:

```
candidate_kernel = compose(
    first_pass_decode(CodecParams { residual_depth: 0, .. }),
    subtract,
    second_pass_decode(CodecParams::residual_shape(params)),
    add,
)
```

`compose` is a Cranelift function that emits the straight-line
sequence — no runtime function-call overhead. ~150 LOC.

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
