# RISC Thought Engine — AGI Roadmap & Ground Truth Documentation

> **Date**: April 6, 2026
> **Branch**: `claude/risc-thought-engine-TCZw7`
> **Status**: Forward pass PROVEN, 7-lane encoding BUILT, quorum pipeline READY

---

## 1. GROUND TRUTH RESULTS (empirically proven)

### Forward Pass Discrimination

| Model | Params | Hidden | Rumi-Rumi | Rumi-TCP | Gap | Verdict |
|-------|--------|--------|-----------|----------|-----|---------|
| Jina v5 | 0.6B | 1024D | 0.512 | 0.384 | 0.128 | DISCRIMINATES |
| Qwen3-VL-Embedding-2B | 2B | 2048D | 0.454 | 0.322 | **0.132** | BETTER |
| ModernBERT-large | 395M | 1024D | 0.959/0.824 | 0.964/0.869 | ~0 | FAILS (MLM) |

**Verdict**: 2B > 0.6B. Forward pass required. Token embeddings alone max rho=0.54.

### 7-Lane Encoding Results

| Metric | Qwen3-VL (2048D) | Jina v5 (1024D) |
|--------|------------------|-----------------|
| Cosine range | [-0.846, 0.539] | [-0.187, 0.675] |
| Cosine mean | 0.163 | 0.225 |
| E/I ratio | 99.2% | 99.3% |
| gamma (auto) | 0.168 | 0.226 |
| phi_scale | 0.846 | 0.675 |
| BF16 max err | 0.0038 | 0.0039 |
| Spiral drift avg | 0.057 | 0.096 |
| Spiral drift max | 0.331 | 0.434 |

### Previous Session Findings (validated)

```
u8 CDF encoding = rho=1.000 vs f32 (PERFECT, encoding irrelevant)
Bucket count >> bucket precision (K=256->4096 = 3x improvement)
Mean-pair table 2.9x better than centroid cosine
Gate L3 alone: rho=0.839 (beats ALL table methods)
Token + 0.5*gate_delta: rho=0.951 (sweet spot)
x256 re-encode safety PROVEN (idempotent after iteration 1)
Cronbach alpha on 3 baked lenses: 71.5% disagreement (superposition needed)
Cross-model calibration IMPOSSIBLE (rho=0.029)
```

---

## 2. THE 7-LANE ENCODING

### Lane Definitions

| Lane | Type | Size | Purpose | Decode Metadata |
|------|------|------|---------|-----------------|
| 1 | u8 CDF | 64 KB | Percentile rank (legacy HDR) | None |
| 2 | i8 direct | 64 KB | round(cos*127), signs preserved | scale=127 |
| 3 | u8 gamma+phi | 64 KB | Golden ratio redistribution | role_gamma, phi_scale |
| 4 | i8 gamma+phi | 64 KB | Signed gamma+phi | role_gamma, phi_scale |
| 5 | f32 SiLU | 256 KB | cos(silu(gate)*up) - cos(raw) | None (zeros for embed) |
| 6 | BF16 direct | 128 KB | StackedN source precision | None |
| 7 | u8 spiral | 64 KB | HighHeelBGZ reconstruction drift | stride, avg_drift, max_drift |

### Output Files Per Model

```
{model}-7lane/
  distance_table_256x256.u8           Lane 1: u8 CDF
  distance_table_256x256.i8           Lane 2: i8 direct
  distance_table_256x256.gamma_phi.u8 Lane 3: u8 gamma+phi
  distance_table_256x256.gamma_phi.i8 Lane 4: i8 gamma+phi signed
  silu_deltas_256x256.f32             Lane 5: SiLU correction
  distance_table_256x256.bf16         Lane 6: BF16 direct
  spiral_drift_256x256.u8             Lane 7: spiral reconstruction drift
  cosine_matrix_256x256.f32           Raw f32 (for experiments)
  codebook_index.u16                  Token -> centroid assignments
  encoding_metadata.json              All calibration params
```

Total per model: ~770 KB (7 lanes + raw + index + metadata)

---

## 3. MODELS & SOURCES

### On Disk (safetensors, proven)

| Model | Path | Size | Tensor Prefix |
|-------|------|------|---------------|
| Qwen3-VL-Embedding-2B | `data/qwen3-vl-embedding/model.safetensors` | 4.0 GB | `model.language_model.` |
| Jina v5 0.6B | `data/jina-v5-onnx/model.safetensors` | 1.2 GB | none (root) |
| ModernBERT-large | `data/modernbert-onnx/model.safetensors` | 1.5 GB | none |

### GGUF (for quorum comparison)

| Model | HuggingFace Repo | Format | Size |
|-------|-----------------|--------|------|
| Qwen3-VL-Embedding-2B | DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF | Q8_0/F16 | 2-3.5 GB |
| Jina v5 0.6B | jinaai/jina-embeddings-v5-text-small-text-matching | F16 | 1.2 GB |
| Jina Reranker v3 | jinaai/jina-reranker-v3-GGUF | BF16 | ~1 GB |

### NOT used (buggy, documented)

```
BUGGY OLD LENSES (DO NOT USE for quorum):
  jina-v3-hdr:      Q8_0 contamination, XLM-RoBERTa tokenizer (250K != 151K)
  bge-m3-hdr:       F16 != BF16 (different exponent range)
  reranker-hdr:     Built from Qwen2 tokens, needs Qwen3 rebuild
```

### Tokenizer Sharing (critical for quorum)

All 3 quorum models share **Qwen3 BPE (151K vocab)** -> codebook assignments directly comparable.

---

## 4. CODE LOCATIONS

### Examples (the runnable pipeline)

| File | Purpose | Command |
|------|---------|---------|
| `examples/seven_lane_encoder.rs` | 7-lane from safetensors | `cargo run --release --features calibration --example seven_lane_encoder -- qwen3-vl-embedding` |
| `examples/qwen3_vl_forward.rs` | Qwen3-VL 2048D forward pass | `cargo run --release --features calibration --example qwen3_vl_forward` |
| `examples/forward_pass.rs` | Jina v5 1024D forward pass | `cargo run --release --features calibration --example forward_pass` |
| `examples/modernbert_forward.rs` | ModernBERT (MLM, doesn't discriminate) | `cargo run --release --features calibration --example modernbert_forward` |
| `examples/stream_signed_lens.rs` | 5-lane from GGUF (streaming) | `cargo run --release --example stream_signed_lens -- repo filename.gguf` |
| `examples/playground.rs` | Interactive thinking engine | `cargo run --release --example playground` |
| `examples/calibrate_lenses.rs` | Spearman rho + ICC | `cargo run --release --features calibration --example calibrate_lenses` |

### Core Modules

| File | LOC | Purpose |
|------|-----|---------|
| `src/engine.rs` | ~600 | u8 ThinkingEngine (MatVec cycle) |
| `src/bf16_engine.rs` | ~300 | BF16 distance table engine |
| `src/signed_engine.rs` | ~250 | i8 signed table with inhibition |
| `src/role_tables.rs` | ~300 | Per-role gate modulation (silu(gate)*Up) |
| `src/builder.rs` | ~500 | Fluent API, BuiltEngine, Temperature |
| `src/pooling.rs` | ~200 | ArgMax/Mean/TopK/Nucleus |
| `src/cronbach.rs` | ~310 | Cronbach alpha, variance_agreement_scores |
| `src/composite_engine.rs` | ~220 | Multi-lens superposition |
| `src/dual_engine.rs` | ~200 | Two-engine comparison |
| `src/semantic_chunker.rs` | ~250 | Convergence-based boundary detection (NAIVE, not late) |
| `src/prime_fingerprint.rs` | ~200 | VSA bundle perturbation |
| `src/reencode_safety.rs` | ~300 | x256 idempotency proof |
| `src/spiral_segment.rs` | ~200 | (anfang, ende, stride, gamma) compression |

### HighHeelBGZ Crate

| File | Purpose |
|------|---------|
| `crates/highheelbgz/src/lib.rs` | SpiralAddress, SpiralWalk, NeuronPrint, HEEL 3-finger |
| `crates/highheelbgz/src/source.rs` | GGUF lazy hydration |
| `crates/highheelbgz/src/rehydrate.rs` | BF16 anchor reconstruction |
| `crates/highheelbgz/src/simd_hardened.rs` | SIMD SpiralAddr (6-byte, integer HEEL) |
| `crates/highheelbgz/src/tensor_bridge.rs` | SpiralWalk -> StackedN, cascade search |

### Configuration Files

| File | Purpose |
|------|---------|
| `data/qwen3-vl-embedding/config.json` | Qwen3-VL model config |
| `data/jina-v5-onnx/config_candle.json` | Jina v5 candle config (with rope_theta) |
| `data/modernbert-onnx/config.json` | ModernBERT config |
| `{model}-7lane/encoding_metadata.json` | Per-model calibration params |

---

## 5. QUORUM PIPELINE (next step)

### Architecture

```
3 Models x 2 Sources = 6 Encodings
  Qwen3-VL safetensors  ----+
  Qwen3-VL GGUF (F16)   ----|
  Jina v5 safetensors    ----|----> 7 lanes each
  Jina v5 GGUF (F16)    ----|      = 42 distance tables
  Reranker v3 BF16       ----|
  Reranker v3 GGUF (BF16)----|

Per-lane Cronbach alpha:
  alpha(lane_k) = agreement across 6 encodings for lane k
  High alpha -> lane is format-insensitive (good)
  Low alpha -> lane captures format-specific signal (interesting)

Per-model Spearman rho:
  rho(model, safetensors vs GGUF) per lane
  Should be > 0.99 for all lanes (encoding irrelevant)
  If not -> GGUF quantization matters for that lane
```

### What We Expect

Based on proven findings:
- **Lanes 1-4** (all derived from same cosine matrix): alpha ~1.0 between safetensors/GGUF
- **Lane 5** (SiLU): zeros for token_embd, needs role-specific encoding for real comparison
- **Lane 6** (BF16): max_err 0.004 -> alpha ~1.0
- **Lane 7** (spiral drift): MAY differ between safetensors/GGUF (sensitive to weight precision)

---

## 6. AGI ARCHITECTURE (the cascade)

```
Input: text / image / audio
  |
  v
[Tokenize] Qwen3 BPE (151K vocab, shared across all 3 models)
  |
  v
[Forward Pass] 28 layers, NO KV cache (embedding model, not autoregressive)
  |   Per-layer: gate accumulation = EKG of thinking
  |   L3 gate as thermostat (Temperature per layer)
  |   L20-L22 = epiphany jump (E/I ratio diverges)
  v
[Embedding] 1024D (Jina v5) or 2048D (Qwen3-VL)
  |
  v
[CLAM] Greedy centroid selection -> 256/4096 codebook
  |
  v
[7-Lane Encoding] u8/i8/gamma+phi/BF16/spiral
  |
  v
[4096x16 Branch Graph]
  |   For EACH centroid: forward pass -> top-16 nearest = branches
  |   65,536 sparse edges, 256 KB
  |   Replaces 32 MB dense table (128x compression)
  v
[ThinkingEngine] L1(64^2) -> L2(256^2) -> L3(4096^2) -> L4(16384)
  |   MatVec cycle, 10 cycles, ~7ms
  |   Temperature + Pooling (ArgMax/TopK/Nucleus)
  v
[NARS] per-branch truth (frequency, confidence)
  |   Bundle = majority_vote(branch fingerprints)
  |   L4 learn(bundle, reward) -> holographic memory
  v
[L4 Cognitive Markers]
  |   WISDOM (+i8):   exploit known good -> Analytical
  |   STAUNEN (0):    explore unknown -> Creative
  |   BLOCKED (-i8):  suppress redundant -> break attractor collapse
  v
[CausalEdge64] 7+1 channels
  |   CAUSES, SUPPORTS, REFINES, GROUNDS, ABSTRACTS, RELATES, BECOMES
  |   + CONTRADICTS (subtracts)
  v
[Cross-Domain Meta-Awareness] 20 KB ONNX
  |   Learns per-pair bridge layers (where L14 converge, L3 diverge)
  v
[Output]
  ThoughtStruct (committed, with provenance)
  QualiaDto (17D feeling from convergence)
  CognitiveTrace (SPO triples)

OSINT Loop (external branching):
  User query -> Google Search -> spider-rs -> Reader-LM 1.5B -> clean text
  -> Jina v5 / Qwen3-VL embedding -> Branch Graph -> NARS reasoning
  -> NARS identifies GAPS (low confidence) -> new queries -> loop
  -> When confidence saturates -> commit -> answer
```

---

## 7. SEMANTIC CHUNKING STATUS

### Current: Naive Chunking

`semantic_chunker.rs` does:
1. Slide window over pre-tokenized centroids
2. Think per window INDEPENDENTLY (fresh engine.reset())
3. Compare convergence patterns (Jaccard on top-k atoms)
4. Mark boundaries where pattern JUMPS

**Problem**: Each chunk is re-thought independently. No cross-paragraph context.

### Needed: Late Chunking (Jina-style)

```
1. Forward pass on FULL document (Qwen3-VL, 28 layers)
2. Get contextualized token embeddings (2048D per token)
3. Find paragraph boundaries (by embedding similarity jumps)
4. Average embeddings per chunk (preserves cross-paragraph context)
```

Since we have a working forward pass, late chunking is ~50 lines of code.
The forward pass ALREADY gives us contextualized embeddings.
We just need to chunk the output, not re-compute per chunk.

---

## 8. GITHUB RELEASES

### Existing

| Release | Contents | Size |
|---------|----------|------|
| v0.1.2-qwopus-layers | 305 u8 tables + codebooks + tokenizer | 28.7 MB |
| v0.1.1-tokenizers | 4 tokenizer files | 38.7 MB |
| v0.1.0-qwopus-4096 | 4096-centroid codebook | 100.7 MB |
| v0.1.0-bgz-data | bgz7 indexes | 735.2 MB |

### Needed (this session)

```
v0.2.0-7lane-codebooks (proposed):
  qwen3-vl-embedding-7lane/    ~770 KB (7 tables + raw + index + metadata)
  jina-v5-7lane/               ~770 KB
  jina-reranker-v3-7lane/      ~770 KB (when built)
  README.md                    (this document, condensed)

Files per model:
  distance_table_256x256.u8    Lane 1
  distance_table_256x256.i8    Lane 2
  distance_table_256x256.gamma_phi.u8  Lane 3
  distance_table_256x256.gamma_phi.i8  Lane 4
  silu_deltas_256x256.f32      Lane 5
  distance_table_256x256.bf16  Lane 6
  spiral_drift_256x256.u8      Lane 7
  cosine_matrix_256x256.f32    Raw ground truth
  codebook_index.u16           Token assignments
  encoding_metadata.json       Config + calibration params
```

---

## 9. KNOWN ISSUES & TECHNICAL DEBT

```
BUGS (from previous session, still unfixed):
  1. reranker-hdr codebook_index.u16 built from Qwen2 tokens -> needs Qwen3 rebuild
  2. signed_domino.rs is dead code (never called)
  3. l4_bridge.rs uses table rows as centroid proxy (documented limitation)

ARCHITECTURE GAPS:
  4. semantic_chunker is naive (not late chunking)
  5. SiLU correction (Lane 5) always zeros for token_embd
     -> needs per-role encoding with gate tensor for real values
  6. 4096x16 branch graph not built yet (needs 4096-centroid CLAM)
  7. L4 holographic memory not wired to forward pass
  8. ONNX cross-domain meta-awareness not trained
  9. No vision test yet (Qwen3-VL can embed images, untested)

FORMAT SENSITIVITY (to verify with quorum):
  10. Spiral drift higher for Jina v5 (0.096) than Qwen3-VL (0.057)
      -> wider hidden dim = more interpolation accuracy?
  11. BF16 roundtrip error ~0.004 for both -> negligible
  12. GGUF vs safetensors comparison not yet run
```

---

## 10. TODO (priority order)

```
[x] Qwen3-VL-Embedding-2B forward pass (DONE, discriminates)
[x] 7-lane encoder from safetensors (DONE, both models)
[ ] Download GGUF for quorum (Qwen3-VL F16, Jina v5 F16, Reranker BF16)
[ ] Run 7-lane from GGUF (extend stream_signed_lens to 7 lanes)
[ ] Cronbach alpha quorum: 3 models x 7 lanes x 2 sources
[ ] Late chunking mode in semantic_chunker
[ ] 4096-centroid CLAM (16x more centroids, proven 3x better rho)
[ ] 4096x16 branch graph from forward pass
[ ] NARS + bundle on the spot per forward pass
[ ] L4 cognitive markers (Wisdom/Staunen/Blocked)
[ ] Vision test (embed image + text, cross-modal cosine)
[ ] LoRA fine-tuning (4 domain adapters)
[ ] Evaluate Gemma 4 31B GGUF (bartowski/google_gemma-4-31B-it-GGUF)
[ ] OSINT pipeline wiring (spider-rs + Reader-LM + embedding + NARS)
```

---

## 11. MODELS TO EVALUATE

| Model | Params | Source | Purpose |
|-------|--------|--------|---------|
| Qwen3-VL-Embedding-2B | 2B | safetensors + GGUF | Primary anchor (multimodal) |
| Jina v5 | 0.6B | safetensors + GGUF | Proven anchor (text-only) |
| Jina Reranker v3 | 0.6B | BF16 + GGUF | Cross-encoder (listwise) |
| Gemma 4 31B-it | 31B | GGUF | Large model evaluation |
| Reader-LM 1.5B | 1.5B | BF16 safetensors | HTML->Markdown for OSINT |
| Qwopus 27B | 27B | BF16 GGUF | 305 tables in Release |
