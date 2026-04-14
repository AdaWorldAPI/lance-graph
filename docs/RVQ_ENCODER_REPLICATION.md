# RVQ Encoder End-to-End Replication

> How to run the RVQ (Residual Vector Quantization) encoder pipeline against any
> Hugging Face safetensors BF16 model and decide whether its codebooks are
> shippable.

Target reader: an engineer with a fresh context window. Everything here is
runnable on a vanilla `lance-graph` checkout.

---

## 1. Purpose

The pipeline lives in:

```
crates/thinking-engine/examples/tts_rvq_e2e.rs
```

It does two full passes over a safetensors model:

1. **Pass 1 — raw BF16.** Load every tensor as BF16 → F32, run TTS inference,
   capture the 15 codec-token streams as the reference.
2. **Pass 2 — RVQ compressed.** Re-load the same tensors, but compress every 2D
   weight matrix with ≥128 rows and ≥128 cols (skipping `norm` / `bias`) via
   CLAM furthest-point sampling + progressive residual quantization. Reconstruct
   from `(codebooks, indices)`, run TTS again, and diff the codec tokens
   against pass 1.

Goal: answer "if we ship only the RVQ codebooks + index tables for this model,
do we get the same codec tokens out?" — i.e. is RVQ a viable delivery format
for this checkpoint?

Exit code of `cargo run` is 0 when the run completes regardless of match
quality; the verdict is printed, not returned.

---

## 2. Prerequisites

| Requirement  | Value                                                         |
| ------------ | ------------------------------------------------------------- |
| Toolchain    | Rust 1.94 stable                                              |
| Build flags  | `.cargo/config.toml` with `rustflags = ["-C", "target-cpu=x86-64-v4"]` (local, not committed) |
| CPU          | x86-64 with AVX-512 (F32x16 hot paths in `l2_dist_sq`)        |
| AMX          | Optional; `ndarray::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16` dispatches a polyfill automatically when AMX is absent |
| RAM          | ~4× model size (streaming two-pass: raw + compressed held simultaneously during phase [3]/[4]) |
| Model        | Hugging Face safetensors in BF16 (tested: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, ~1.8 GB) |

Create the local cargo config if you don't already have one:

```toml
# .cargo/config.toml (do not commit)
[build]
rustflags = ["-C", "target-cpu=x86-64-v4"]
```

Without AVX-512 the F32x16 paths still compile (ndarray falls back via
`PREFERRED_F32_LANES`), but runtime will be substantially slower than the
baseline in section 9.

---

## 3. Download the model

Either approach works. The path you end up with is what you pass to the
binary in section 5.

### 3a. Hugging Face CLI

```sh
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  model.safetensors \
  --local-dir /home/user/models/qwen3-tts-0.6b
```

### 3b. Raw curl

```sh
mkdir -p /home/user/models/qwen3-tts-0.6b
curl -L \
  -o /home/user/models/qwen3-tts-0.6b/model.safetensors \
  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/model.safetensors
```

The example defaults to `/home/user/models/qwen3-tts-0.6b/model.safetensors`
when no argument is given (see `main()` in `tts_rvq_e2e.rs`).

---

## 4. Build

```sh
cargo build --release --example tts_rvq_e2e \
  --manifest-path crates/thinking-engine/Cargo.toml
```

Always `--release`. Debug builds of the RVQ loop are not usable for a model
of this size.

---

## 5. Run

```sh
cargo run --release --example tts_rvq_e2e \
  --manifest-path crates/thinking-engine/Cargo.toml \
  -- /home/user/models/qwen3-tts-0.6b/model.safetensors
```

Omit the trailing path to fall back to the default above.

---

## 6. Output anatomy

The run prints five phases. Each is marked with `[N]`.

### [1] Raw load

```
[1] Loading raw weights (pass 1: raw inference)...
  <N> tensors loaded in <duration>
```

Reads the safetensors header, iterates every tensor, converts BF16/F16/F32 →
F32, and stores each as a flat `Vec<f32>`. No RVQ on this pass — this is the
reference.

### [2] Raw TTS

```
[2] Running TTS (raw weights)...
  <duration>, <tokens> tokens × 15 codebooks
```

Runs the 33-layer TTS stack (28 talker + 5 code_predictor) and produces 15
codec-token streams. These are cached as the gold reference.

### [3] RVQ compress + pass 2 load

```
[3] Loading + RVQ compressing (pass 2)...
    [idx] tensor_name                                       [rows×cols] cos=X.XXXX k=[...] <duration>
    ...
  Compressed in <duration>
  Codebook: X.X MB, Indices: X.X MB, Total: X.X MB
```

Re-seeks the file to offset 0 and re-reads every tensor, but this time
qualifying 2D tensors with `rows ≥ 128 && cols ≥ 128 && !name.contains("norm")
&& !name.contains("bias")` go through `build_rvq`. K-ladder depends on role:

- `k_proj`, `v_proj`, `down_proj` → `[256, 512, 1024]` (3 levels)
- everything else qualifying → `[256, 512, 1024, 4096]` (4 levels)

Per-tensor line format:

```
[  1] model.layers.0.self_attn.q_proj.weight         [1024x1024] cos=0.9998 k=[256, 512, 1024, 4096] 12.3s
```

`cos` is the cosine similarity of the first row of the original against the
first row of the RVQ reconstruction (quick quality probe, not a full tensor
metric). `k=` is the RVQ ladder that was used. Trailing duration is wall time
for that tensor's RVQ build.

### [4] RVQ TTS

```
[4] Running TTS (RVQ weights)...
  <duration>, <tokens> tokens × 15 codebooks
```

Same inference path as [2], but with the reconstructed weights.

### [5] Comparison

```
[5] Comparison:
  Codec token match: M/N (PP.P%)
  Original weights: XXX.X MB
  RVQ compressed:   YYY.Y MB (Z:1)
  ★ SUCCESS: RVQ codebook preserves >90% codec tokens
  (or ◐ PARTIAL / ✗ FAIL)

  First 5 tokens, codebook 0:
    RAW: ...
    RVQ: ...
```

Compares the 15 codec streams token-by-token and prints storage accounting.
The ratio is `orig_bytes : (cb_bytes + idx_bytes)` — a ratio < 1 means RVQ
storage is *larger* than the original weights. See section 8.

---

## 7. Adapting to a new model

The current example is hard-coded to Qwen3-TTS-0.6B. Every constant below
lives at the top of `tts_rvq_e2e.rs` or is inlined in `main()`. Treat this as
a checklist — skipping any item will produce silent shape mismatches or NaN
outputs.

### 7a. Tokenizer / special tokens

In `main()`:

```rust
let tokens: Vec<usize> = std::iter::once(151672)
    .chain(text.bytes().map(|b| b as usize))
    .chain(std::iter::once(151673))
    .collect();
```

`151672` is the BOS for Qwen3 and `151673` is the EOS. Replace both with
your model's equivalents. The byte-level fallback
(`text.bytes().map(|b| b as usize)`) is a stand-in, not a real tokenizer —
if your model expects proper BPE, plug a tokenizer in here.

### 7b. Layer count constants

```rust
const TALKER_LAYERS: usize = 28;
const CP_LAYERS: usize = 5;
```

Current target has 28 talker layers + 5 code_predictor layers = 33 total.
Update both to match your architecture.

### 7c. Shape constants

| Constant          | Current | Meaning                               |
| ----------------- | ------- | ------------------------------------- |
| `TALKER_HIDDEN`   | 1024    | Talker hidden dim                     |
| `TALKER_HEADS`    | 16      | Attention heads                       |
| `TALKER_KV_HEADS` | 8       | KV heads (GQA)                        |
| `TALKER_HEAD_DIM` | 64      | `TALKER_HIDDEN / TALKER_HEADS`        |
| `TALKER_INTER`    | 3072    | MLP intermediate dim                  |
| `CP_HIDDEN`       | 1024    | Code predictor hidden dim             |
| `CP_HEADS`        | 16      | CP attention heads                    |
| `CP_KV_HEADS`     | 8       | CP KV heads                           |
| `CP_HEAD_DIM`     | 64      | `CP_HIDDEN / CP_HEADS`                |
| `CP_INTER`        | 3072    | CP MLP intermediate dim               |
| `SAMPLE_RATE`     | 24000   | Used by the WAV writer                |

`HEAD_DIM × HEADS == HIDDEN` must hold. `HEADS` must be a multiple of
`KV_HEADS` (GQA grouping is `n_heads / n_kv_heads`).

### 7d. Tensor name prefixes

The inference helpers use the literal prefix `talker.model.` and
`talker.code_predictor.model.`. If your checkpoint uses different names
(e.g. `model.` without the `talker.` prefix), grep for `talker.` in
`tts_rvq_e2e.rs` and rename — there are several call sites.

### 7e. RVQ qualification predicate

If your model has important ≤128-row matrices or tensors whose names contain
`norm`/`bias` but should still be compressed, loosen the guard in
`load_weights` (`n_rows >= 128 && n_cols >= 128 && !name.contains("norm")`).

Checklist:

- [ ] BOS/EOS IDs swapped
- [ ] `TALKER_LAYERS`, `CP_LAYERS` match architecture
- [ ] All 10 shape constants updated
- [ ] Tensor-name prefix strings match checkpoint keys
- [ ] K-ladder still sensible for the new shapes (see
      `RVQ_K_LADDER_TUNING.md`)

---

## 8. Success criteria

Use these as the go/no-go gate for declaring a new model RVQ-ready.

**Per-tensor quality**

- `cos ≥ 0.999` on all attention projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`). This
  is the first-row probe printed on each `[idx]` line.

**End-to-end preservation**

- Codec-token match ≥ 90% between raw and RVQ runs (line: `Codec token
  match: M/N (PP.P%)`). The example prints a ★ SUCCESS banner at this
  threshold.

**Storage**

- `cb_bytes + idx_bytes < orig_bytes` (i.e. ratio > 1:1).
- On the current Qwen3-TTS-0.6B baseline this is NOT yet met — see
  `RVQ_K_LADDER_TUNING.md` for the plan to get there.

---

## 9. Known-good baseline and known issues

First successful end-to-end run, Qwen3-TTS-0.6B:

| Metric               | Value                                              |
| -------------------- | -------------------------------------------------- |
| Tensors at cos=1.000 | 477 / 478                                          |
| Tensors below 0.999  | 1 — `text_embedding` at cos=0.054                  |
| Codec token match    | 80.4%                                              |
| Storage ratio        | 1:1.24 (RVQ is **larger** than original — see §8)  |
| Pass 2 wall time     | ~24 min on 16-core AVX-512, single-threaded        |
| `text_embedding`     | ~15 min of that pass 2 time                        |

The `text_embedding` collapse and the 1:1.24 storage inversion are the two
headline issues. Remediation paths are tracked in `RVQ_K_LADDER_TUNING.md`
and `RVQ_ALTERNATIVES.md`.

Verification status for this session:

- RVQ e2e binary compiled and ran to exit 0.
- Companion ndarray commit on branch `claude/teleport-session-setup-wMZfb`
  merged: additive `TDPBF16PS` + `vnni_pack_bf16` + `bf16_tile_gemm`
  polyfill. 1616 ndarray tests passed.

Commits landed in `AdaWorldAPI/lance-graph#176` (merged):

| Commit    | Subject                                                        |
| --------- | -------------------------------------------------------------- |
| `b7db84f` | AudioNode + HHTL cascade bridge (recovered)                    |
| `1bd4e98` | Fused alloc-free `l2_dist_sq` + O(k²) `assign_nearest` fix     |
| `d5daa28` | AVX-512 F32x16 FMA in `l2_dist_sq`                             |
| `cfed5b9` | AMX BF16 probe (initial)                                       |
| `6c2e97b` | Probe refactored to use ndarray polyfill                       |

No other stats, runs, or commits beyond the above have been verified in
this session.

---

## 10. Cross-reference

| Topic                           | Location                                                 |
| ------------------------------- | -------------------------------------------------------- |
| K-ladder tuning (storage fix)   | `docs/RVQ_K_LADDER_TUNING.md` (sibling in this PR)       |
| Alternatives to RVQ             | `docs/RVQ_ALTERNATIVES.md` (sibling in this PR)          |
| AMX BF16 GEMM polyfill          | `ndarray::hpc::bf16_tile_gemm` (`bf16_tile_gemm_16x16`)  |
| Example source                  | `crates/thinking-engine/examples/tts_rvq_e2e.rs`         |
| Main PR                         | `AdaWorldAPI/lance-graph#176` (merged)                   |

---
