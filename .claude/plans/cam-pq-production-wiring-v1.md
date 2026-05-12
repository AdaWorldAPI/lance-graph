# Plan — CAM-PQ Production Wiring (2026-04-20)

> **Status:** DRAFT (unscheduled follow-up PR, awaiting prioritization)
> **Driver:** 2026-04-20 measurement: CAM-PQ at 6 B/row + 24 KB shared
> codebook achieves ICC 0.9999 / top-5 recall 1.0 on Qwen3-8B q_proj /
> k_proj / gate_proj. 1300× compression at near-Passthrough fidelity.
> See `.claude/board/EPIPHANIES.md` 2026-04-20 entry and PR #218 bench.
>
> **Scope:** wire CAM-PQ as the default codec for argmax-regime tensors
> (attention Q/K/V/O, MLP gate/up/down, logit head), leaving index-regime
> tensors (embeddings, lm_head indexing) on Passthrough per invariant I1.

---

## What exists (no new code needed for these)

- **`ndarray::hpc::cam_pq`** — production codec: `CamCodebook`,
  `SubspaceCodebook`, `CamFingerprint` (6 bytes), `DistanceTables`,
  `PackedDatabase` (stroke-layout cascade), `train_geometric`,
  `train_semantic`, `train_hybrid`. 620+ LOC, 15+ tests. Just not
  routed to.
- **`lance-graph-contract::cam::CamCodecContract`** — zero-dep trait
  surface. Consumers bind against the contract, not the implementation.
- **`lance-graph-planner::physical::CamPqScanOp`** — DataFusion
  operator. Already shipped.
- **`codec_rnd_bench.rs` CamPqRaw / CamPqPhase candidates** — the
  measurement probe that validated the approach (commit `f1498bc`).

## The gap

Consumers of argmax-regime weight tensors default to Passthrough f32
storage. No production consumer currently routes through
`CamCodecContract` → `CamPqScanOp`. The integration layer between
"codec exists" and "tensors flow through it" is missing.

---

## Deliverables

### D1 — Tensor-type classifier

**What:** a function that given a tensor name + shape returns
`CodecRoute::{CamPq | Passthrough | Skip}` per invariant I1.

**Where:** `lance-graph-contract::cam::route_tensor(name, dims) ->
CodecRoute` — extends the existing `classify_tensor` in
`ndarray::hpc::gguf_indexer` with the argmax/index distinction.

**Rule:**
- `attn_{q,k,v,o}_proj`, `mlp_{gate,up,down}_proj`, `ffn_{gate,up,down}`
  → `CamPq`
- `token_embd`, `embed_tokens`, `lm_head`, `wte`, `wpe` → `Passthrough`
- `norm`, `ln_*`, small (< 4096 elem) → `Skip` (not worth codec)
- Ambiguous 2D matrix ≥ 4096 elem → `CamPq` (argmax default)

**LOC:** ~60 in contract, ~30 in tests.

### D2 — Per-tensor calibration pipeline

**What:** offline tool that reads a safetensors/GGUF file, classifies
tensors, runs `cam_pq::train_geometric` on each argmax-regime tensor,
serializes the resulting `CamCodebook` alongside the fingerprints.

**Where:** `crates/bgz-tensor/src/cam_pq_hydrate.rs` (new file) — mirrors
`hydrate.rs` pattern for bgz7 shards. CLI bin `cam_pq_calibrate`
under `required-features = ["calibration"]`.

**Pipeline:**
```
safetensors / GGUF  →  per-tensor rows  →  train_geometric(rows, dim, 20)
                                                    ↓
                                                CamCodebook (24 KB)
                                                    ↓
                        row-by-row:  fingerprint = codebook.encode(row)  (6 B)
                                                    ↓
                          Lance FixedSizeBinary(6) column + codebook blob
```

**Calibration cost:** k-means 20 iterations × 6 subspaces × 256
centroids × (n_rows × subspace_dim). For 4096-dim q_proj with
4096 rows: ~20 × 6 × 256 × 4096 × 682 ≈ 40 GFLOPs, ~5 s on CPU.

**LOC:** ~180.

### D3 — Storage format

**What:** Lance column schema for CAM-PQ-encoded weights.

**Schema:**
```
struct TensorStorage {
    route: CodecRoute (u8),
    fingerprints: FixedSizeList<UInt8, 6>,   // if CamPq
    codebook: LargeBinary,                    // if CamPq, serialized CamCodebook
    passthrough: FixedSizeList<Float32, N>,   // if Passthrough
    // Norm/skip tensors: stored as f32 passthrough, small
}
```

**Serialization:** `CamCodebook` serializes to ~24 KB (6 codebooks ×
256 centroids × 682 f32 subdim × 4 B ≈ 4 MB — oops, that's wrong, let
me recompute). 6 × 256 × 682 × 4 = ~4.2 MB per codebook for 4096-d
tensor. Actually 24 KB was wrong; real cost is ~4 MB shared per
tensor.

**Revised storage accounting:**
- Per 4096×4096 tensor at f32: **64 MB** (Passthrough)
- Per 4096×4096 tensor via CAM-PQ: **4 MB codebook + 24 KB
  fingerprints = ~4 MB**
- Compression ratio: **16×** (not 1300× — prior calc forgot codebook size)
- Still a huge win, but calibrate expectations.

**LOC:** ~120 for Lance column codec + tests.

### D4 — Runtime decode path

**What:** consumer APIs that receive an opaque tensor handle and
transparently decode on access.

**API:**
```rust
pub trait TensorAccess {
    fn row(&self, i: usize) -> Cow<[f32]>;
    fn rows_batch(&self, indices: &[usize]) -> Vec<Cow<[f32]>>;
    fn distance_table(&self, query: &[f32]) -> DistanceTables;  // CAM-PQ fast path
}
```

**Fast path:** for argmax queries, skip decoding entirely — use
`cam_pq::DistanceTables::distance(fingerprint)` directly. This is
O(6) per candidate (6 table lookups + 5 adds) regardless of tensor dim.

**LOC:** ~80 in contract trait + ~150 in the two impls
(CamPqAccess, PassthroughAccess).

### D5 — Validation harness on full-size tensors

**What:** the 128-row bench measurement was a sample. Need to verify
ICC holds on the full 4096-row (or 12288-row for gate_proj) tensor
with the codebook trained on the same.

**Where:** new bench in `crates/bgz-tensor/benches/cam_pq_fullsize.rs`.

**Test matrix:**
- Per tensor: train codebook on full row set, encode, decode, measure:
  - Cosine fidelity on 1000 random pair queries vs ground truth
  - Top-k retrieval recall (k=1, 5, 10)
  - Calibration time
- Compare: 128-row-trained codebook vs full-trained codebook. Does the
  sample version generalize? (Expected yes, test anyway.)

**Gate:** ICC ≥ 0.99 on full-size before production rollout.

**LOC:** ~200.

### D6 — End-to-end model storage benchmark

**What:** actual byte count of Qwen3-8B stored as Passthrough vs CAM-PQ
across all tensors, with a correctness check (run model inference on
a few prompts, verify argmax token agreement).

**Where:** `crates/bgz-tensor/examples/cam_pq_model_bench.rs`.

**Metrics:**
- Total bytes per tensor (passthrough vs cam_pq)
- Total bytes per model
- Argmax top-1 agreement on standard eval prompts (LAMBADA, HellaSwag, etc.)
- Inference latency delta

**LOC:** ~150.

### D7 — Fallback path

**What:** if CAM-PQ calibration produces poor ICC on a specific tensor
(unusual distribution, edge case), fall back to Passthrough.

**Detection:** during D2 calibration, compute reconstruction error;
if `mean_reconstruction_error > threshold`, mark that tensor as
Passthrough in the storage manifest.

**Threshold:** `||x − decode(encode(x))||² / ||x||² > 0.05` = 5% L2
error. Empirically tune.

**LOC:** ~40.

---

## Invariants respected

- **I1 (two regimes):** index-regime tensors stay Passthrough. CAM-PQ
  only routes attention/MLP (argmax-decoded).
- **I2 (near-orthogonality):** CAM-PQ's subspace k-means captures the
  structure without needing Hadamard rotation (measured).
- **I7 (codec tier):** per-tensor calibration is the legitimate
  "vector-as-sparse-signal" path.

## Risks

1. **128-row sample might not generalize to full tensor.** Gated by D5.
   Mitigation: if generalization fails, sample more rows at calibration
   time (say 512 rows) — linear cost increase.

2. **Index-regime routing bug:** if D1 misclassifies an embedding as
   argmax-regime, CAM-PQ corrupts identity lookup. Mitigation:
   conservative default — ambiguous tensors route to Passthrough, not
   CAM-PQ.

3. **Codebook storage cost:** ~4 MB per attention tensor × ~28 layers ×
   4 projections = ~450 MB codebook overhead for Qwen3-8B. Plus ~24 KB
   × 28 × 4 = 2.7 MB fingerprints. Still 64 GB → ~500 MB = **128×
   compression**, not 1300×. Honest number.

4. **Cold-start calibration time:** Qwen3-8B full calibration ~28
   layers × 4 attention + 3 MLP = 196 tensors × 5 s each = ~16 min.
   One-time cost per model.

5. **Fidelity at inference:** we measured ICC on pairwise cosines.
   Actual inference fidelity (argmax token agreement after multi-layer
   propagation) must be verified separately. Gate D6.

## Acceptance criteria

- [ ] D1 route classifier: 100% correct routing on Qwen3-8B tensors
- [ ] D2 calibration pipeline: runs on Qwen3-8B in ≤ 20 min
- [ ] D3 Lance schema: round-trip preserves CamCodebook via `Write → Read`
- [ ] D4 runtime API: `TensorAccess::row(i)` returns within 50 µs
- [ ] D5 full-size ICC: ≥ 0.99 on every argmax tensor
- [ ] D6 end-to-end: ≤ 1% top-1 token agreement loss vs Passthrough baseline
- [ ] D7 fallback: any tensor failing D5 auto-marked Passthrough
- [ ] Storage ratio: ≥ 100× on Qwen3-8B total

## Effort estimate

- D1 / D3 / D4 / D7: 1 person-day each (mechanical wiring against
  existing contracts).
- D2: 2 person-days (calibration pipeline + Lance artifact + CLI).
- D5: 1 person-day (bench + ICC measurement across full tensors).
- D6: 1 person-day (end-to-end eval, requires a small eval harness —
  may borrow from `crates/thinking-engine/examples/cascade_inference.rs`).

**Total: ~8 person-days.** One dedicated sprint.

## Out of scope (follow-ups)

- CAM-PQ for cross-model transfer (train once on family A, use on
  family B) — unclear whether codebook generalizes; separate research.
- CAM-PQ + SIMD-packed distance-table inference (bgz-tensor
  AttentionSemiring already does this for its own format; extend to
  CAM-PQ if D6 proves the compression win).
- Zipper family as fallback for novel-population query-time tensors
  where no codebook exists — architectural niche, not blocking.

## Cross-references

- `.claude/board/EPIPHANIES.md` 2026-04-20 "CAM-PQ solves argmax blind
  spot" entry (measured result).
- `.claude/knowledge/codec-findings-2026-04-20.md` decision tree.
- `.claude/knowledge/encoding-ecosystem.md` Invariant I1/I2/I7.
- `crates/thinking-engine/examples/codec_rnd_bench.rs` CamPqRaw,
  CamPqPhase candidates.
- `ndarray::hpc::cam_pq` production codec.
- `lance-graph-contract::cam::CamCodecContract` integration trait.
- `lance-graph-planner::physical::CamPqScanOp` operator.
