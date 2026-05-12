# Compression Mindset Shifts

> Session-end design reflection after iterating RVQ → HCLAM → passthrough on Qwen3-TTS-0.6B (PRs #176, #177, #178). What we learned, where we went narrow, and four expansions of scope that would matter more than the next local fix.

## What this session actually established

- The Rust inference pipeline (28 talker + 5 code-predictor + codec_head + RVQ dequant + conv decoder) works end-to-end on Qwen3-TTS-0.6B: `text → 128 frames of codec tokens → 1.37 s 24 kHz WAV`, validated via varied codec tokens across positions and realistic WAV statistics (RMS -21 dB, dynamic envelope, non-constant output).
- The SIMD and AMX wiring is correct: F32x16 FMA encoder, TDPBF16PS polyfill with AVX-512 fallback, bf16_to_f32 batch decode, gemm_f32 via matrixmultiply. All additive to ndarray (`hpc::bf16_tile_gemm`).
- Codec-token preservation across a compression pass is 225 / 225 = 100 %.

## What this session did NOT establish

- **Storage ratio is a net loss: 5 096 MB compressed vs 3 657 MB original = 1 : 1.39.** Per-tensor RVQ codebooks at `k = [256, 512, 1024, 4096]` are individually *larger* than the BF16 tensors they compress.
- **Correctness proof does not equal product proof.** The 225 / 225 match says the pipeline is self-consistent; it does not say the output is a shippable codec.
- **The HCLAM path we tried collapsed worse than the RVQ it was supposed to fix** (cos 0.0046 vs 0.0544 on the vocab embedding). The `docs/RVQ_K_LADDER_TUNING.md` § 3 claim "2.4:1 at cos ≈ 1 via hierarchical CLAM 256×256" is REFUTED for near-orthogonal high-dim rows and should be struck.

## The insight that reframes the rest

BPE + argmax decoding puts the network in **two different regimes** with opposite compression requirements:

| Regime | Where it lives | What it requires | Error behavior |
|---|---|---|---|
| **Argmax-decoded** | All 33 transformer layers, codec head, lm_head | **Top-1 argmax stability** under `hidden @ W.T` | Robust — small reconstruction noise leaves argmax unchanged |
| **Index-lookup** | `text_embedding`, codec-token re-feed, any `token_id → row[token_id]` | **Per-row identity** (row `i` decodes ≡ row `i`) | Cascading — any row-identity loss corrupts all downstream |

This is why the 477 compressed attention/MLP tensors survived at cos ≈ 0.95 – 1.0 (argmax regime: ρ = 0.95 is *plenty*), and why the one vocab embedding at cos = 0.05 destroyed the whole pipeline (index regime: there is no argmax downstream to rescue the error).

**The implication is structural, not tuning:** the two regimes want different codecs. The current `tts_rvq_e2e.rs` uses one codec for everything and is surprised when it fails on the one tensor that isn't argmax-downstream.

## Four mindset shifts, ranked by blast radius

### 1. From "compress weights" to "index a cognitive surface"

Current framing: weights are *data to shrink*. RVQ produces anonymous codebook indices — the compressed form has no semantic structure beyond "row `i` maps to codebook entry `c_i`".

Architecture framing (per `CLAUDE.md`): `ndarray = hardware`, `lance-graph = query + codec + semantics`, `ladybug-rs = BindSpace + 4096 surface + SPO`. In this framing every compressed tensor is a *node in a semantic graph*, not a blob. `bgz-tensor::HhtlDTensor` already does this — the 4-byte encoding is `HEEL (basin) + HIP (family) + TWIG (palette idx) + polarity`, which is a *graph address*, not just an index.

**Shift:** compression as indexing, not as squeezing. Every row gets a semantic address.

### 2. From "f32 GEMM on reconstructed weights" to "inference in codec space"

The entire RVQ pipeline reconstructs f32 weights to run f32 GEMM. The BPE + argmax observation says this is algorithmically wasteful — nothing downstream needs f32 precision, only argmax stability.

Already in the codebase:

| Project | Inference model |
|---|---|
| `bgz-tensor` HHTL cascade | 95 % of attention pairs resolved by table lookup, no weight access |
| `deepnsm` | 680 GB transformer → 16.5 MB, 50 ms/token → <10 μs/sentence via lookup + VSA |
| `bgz-tensor::AttentionSemiring` | Q · Kᵀ / √d = `table[q_idx][k_idx]` in O(1) |

**Shift:** inference is table-walking, not matrix multiplication. f32 GEMM is legacy from the training regime.

### 3. From "Qwen3-TTS, specifically" to "any BPE-argmax transformer, generically"

`tts_rvq_e2e.rs` hardcodes 28 + 5 layers, Qwen3-TTS tensor names, 2048 hidden. The Model Registry in `CLAUDE.md` lists 7 in-scope models (Jina v5, Reranker v3, ModernBERT, BGE-M3, Jina v3, Qwopus, Reader-LM). Plus Gemma 4, Qwen3-VL, Qwen3-SLM. All fit the BPE-vocab + argmax-decode pattern.

The two-regime dispatch is a *universal rule* for the family, not a Qwen3-TTS idiosyncrasy.

**Shift:** build `fn encode_model(safetensors) -> HhtlDPack` that dispatches per tensor by role, not a model-specific pipeline.

### 4. From "build it all" to "integrate what exists"

In one session we wrote: HCLAM 256×256 (wrong), F32x16 FMA distance kernel (works but slower than matrixmultiply's AVX-512 baseline on large matrices), bf16 tile GEMM polyfill (correct but untestable here), passthrough dispatch (sidesteps the problem).

Meanwhile, already in the repo:

| Component | Status |
|---|---|
| `bgz-tensor::HhtlDTensor` + `SharedPaletteGroup` + `FisherZTable` | 343 : 1 on Qwen3-TTS-1.7B, documented, tested |
| `bgz-tensor::matryoshka` | 4 : 1 at per-layer ε < 0.001 through 33 layers |
| `ndarray::hpc::clam` | Full CLAM build + search + ρ-NN, 46 tests |
| `deepnsm` | 4096² u8 distance matrix + 512-bit VSA, 4096 COCA vocab |
| `bgz17` | 121 tests, palette semiring, Base17 VSA |

And Lance 4.0 / 5.0-rc.1 ships IVF_RQ, multi-split, distributed segment builds, CacheBackend + CacheCodec.

**Shift:** stop building codecs, start wiring the ones that exist.

## Concrete next-session proposal (combined shifts 3 + 4)

One new example, not a multi-week refactor: `crates/thinking-engine/examples/universal_hhtld_encode.rs`.

### Contract

- Input: any BPE-vocab safetensors model path
- Output: a single `.hhtld` pack file
- Validation: argmax-parity (not cos) against raw inference on a held-out prompt

### Dispatch

```text
for tensor in safetensors_header:
    role = bgz-tensor::shared_palette::classify_role(tensor.name)
    match role:
        "embed" | "lm_head":
            encode with HhtlDTensor + Slot L (Matryoshka SVD band 0, 8 i8 on shared basis)
            → 12 B/row, ρ > 0.98 per row, preserves row identity for index lookups
        "qko" | "v" | "gate" | "up" | "down":
            encode with HhtlDTensor Slot D only
            → 4 B/row, ρ ≈ 0.95 per row, sufficient for argmax-stable matmul
        "norm" | "bias" | tiny_shape:
            passthrough BF16
```

### Storage estimate on Qwen3-TTS-0.6B

| Class | Tensor count | Bytes/row | Total |
|---|---|---|---|
| Argmax regime (attention + MLP + code predictor) | 477 | 4 | ~3 MB entries + ~26 × 206 KB shared palettes = ~8 MB |
| Index regime (text_embedding + lm_heads) | ~16 | 12 | ~2 MB entries + 16 × SVD basis ≈ ~18 MB |
| Passthrough (norms, biases) | ~50 | — | ~3 MB |
| **Total** | | | **~29 MB** vs original 3 657 MB = **126 : 1** |

For Qwen3-TTS-1.7B (where the published `BGZ_HHTL_D.md` number of 343 : 1 already applies): consistent with the existing claim plus a quality-preserving Slot L on the 1.7 B vocab embedding.

### Work breakdown

1. Port `matryoshka::SvdBasis` "band 0" (8-dim i8 on shared SVD basis) as an optional Slot L on `HhtlDEntry` — backwards-compat via magic byte on serialize/deserialize
2. Add `classify_role` dispatch in the encoder
3. Wire into the `run_tts` decode path so inference uses `HhtlDTensor::reconstruct_row` instead of RVQ codebook sum
4. Validate via argmax-parity: run inference on prompt, compare codec tokens to reference — target 225 / 225 at ≥ 50 : 1 ratio
5. Optionally: produce WAV and verify RMS / ZC / envelope within 5 % of reference

### What this replaces / deprecates

- `crates/thinking-engine/examples/tts_rvq_e2e.rs` — demonstrates the RVQ approach; kept as reference, not the product path
- `docs/RVQ_K_LADDER_TUNING.md` § 3 — already REFUTED by this session's HCLAM collapse; needs a strikethrough note pointing readers at the universal encoder
- Custom `build_rvq` / `reconstruct_rvq` / `build_hclam_256x256` / `reconstruct_hclam` in the e2e example — all superseded by `HhtlDTensor::encode/reconstruct`

## Alternative mindset expansion (shift 2 only)

Accept that we are not preserving f32 weights; we are preserving the argmax behaviour of the network. Inference becomes distance-table lookups gated by the 2-bit HEEL basin. The HHTL cascade Skip / Attend / Compose / Escalate routing (already in `bgz-tensor::hhtl_cache`) is the computation model.

This is further from anything we have done but closer to where the codebase is pointing (see `bgz-tensor::AttentionSemiring`, `deepnsm` lookup-table inference, `ladybug-rs` BindSpace 4096 surface).

Cost: a multi-session architecture pivot. Benefit: order-of-magnitude speedups on top of the compression ratio, and alignment with the `ndarray = hardware / lance-graph = spine / ladybug-rs = brain` contract.

## Open questions for the next session

1. Does `matryoshka::SvdBasis::build` need to run per palette group, per tensor, or globally? (Tradeoff: fidelity vs SVD basis storage overhead.)
2. What is the actual argmax-parity threshold we will accept? 225 / 225 is the current bar; is 223 / 225 acceptable if it means 100 : 1 better ratio?
3. If we go with shift 2 (inference in codec space), what is the first tensor to migrate? Probably the code predictor's five layers — smaller than the talker, lower blast radius.
4. Does `bgz-tensor::FisherZTable` already give us the pairwise cosine lookup we need for argmax-space inference, or is there a missing piece?
5. Lance IVF_RQ + multi-split (`docs/LANCE_UPGRADE_ROADMAP.md`) vs `bgz-tensor::HhtlDTensor` with Slot L — which is the *shipping* codec and which stays a research curiosity?

## Cross-references

- `AdaWorldAPI/lance-graph#176` (merged) — AVX-512 F32x16 FMA encoder + AMX TDPBF16PS polyfill
- `AdaWorldAPI/lance-graph#177` (merged) — HCLAM dispatch (REFUTED) + F32x16 rms_norm + 3 RVQ docs
- `AdaWorldAPI/lance-graph#178` (open) — passthrough BF16 for vocab tensors + Lance upgrade roadmap + WAV validity test
- `docs/RVQ_ENCODER_REPLICATION.md` — runnable pipeline for any BF16 safetensors model
- `docs/RVQ_K_LADDER_TUNING.md` — § 3 REFUTED; see this doc for the corrected approach
- `docs/RVQ_ALTERNATIVES.md` — codec-family comparison
- `docs/LANCE_UPGRADE_ROADMAP.md` — Lance 2 → 4 / 5 migration plan
- `crates/bgz-tensor/BGZ_HHTL_D.md` — the 343 : 1 lookup-grade encoding we should be using
- `.claude/prompts/fisher-z-wiring/` — 12-step HhtlDTensor integration plan
- `CLAUDE.md` § Architecture Notes — the codec stack and thinking pipeline this all plugs into

https://claude.ai/code/session_01NYGrxVopyszZYgLBxe4hgj
