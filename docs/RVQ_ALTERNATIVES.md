# RVQ Alternatives and Multi-Modal Adaptation

Scope: a decision guide for engineers who have run the progressive-residual
RVQ pipeline (`crates/thinking-engine/examples/tts_rvq_e2e.rs`, PR
`AdaWorldAPI/lance-graph#176`) and need to know when to reach for a
different codec, how to extend the encoder to multi-modal models such as
Qwen3-VL, and how RVQ relates to the other compression stacks already in
this repo.

Companion documents (do not duplicate their content):

- `docs/RVQ_ENCODER_REPLICATION.md` — how to run the pipeline end to end
- `docs/RVQ_K_LADDER_TUNING.md` — the hierarchical CLAM 256x256 remediation
  for `n_rows > 8192`

---

## 1. When RVQ is the right codec

Progressive-residual RVQ with the default ladder `k = [256, 512, 1024, 4096]`
is well-matched to:

- Dense transformer projection matrices: attention `K`, `Q`, `V`, `O` and
  MLP `gate`, `up`, `down` weights with row count up to and including 8192.
- Tensors where cos ~ 1.0 reconstruction is achievable at `k >= rows/4`.
- Weight-storage compression where full-precision `f32` GEMM is still run
  at inference time (weights are reconstructed on load, not replaced).

The Qwen3-TTS-12Hz-0.6B-Base run confirms this boundary: 477 of 478 tensors
reconstruct at cos = 1.000 with the default ladder. The single failure is
`model.text_embedding.weight [151936, 2048]` at cos = 0.054 — a vocab-sized
tensor that needs a different code path. Overall codec token match is
80.4% and storage ratio is 1 : 1.24 (worse than original), both driven by
that one tensor.

---

## 2. When RVQ is the wrong codec

For `n_rows > 8192` the primary fix is the hierarchical CLAM 256x256
path documented in `RVQ_K_LADDER_TUNING.md`. The cases below are
structural mismatches where even a tuned RVQ will lose to a different
codec family.

| Mismatch | Symptom | Better codec |
|---|---|---|
| Vocab-sized tensor, `n_rows` >> `k_final` | cos << 0.99, codebook dominates storage | Hierarchical CLAM 256x256 (`RVQ_K_LADDER_TUNING.md`) |
| Attention hot path, `Q Kᵀ / sqrt(d)` dominates runtime | Compression did not change inference cost | bgz-tensor palette + compose table |
| Retrieval encoder, goal is runtime embedding streaming | RVQ compresses storage but encoder still loads as dense | Jina v5 / Jina-Reranker-v3 5-lane BF16 |
| Small fixed vocabulary, inference can be replaced by lookup | RVQ preserves GEMM; GEMM itself is the waste | DeepNSM COCA 4096 x 512-bit VSA |

Each alternative is discussed in section 4.

---

## 3. Multi-modal adaptation (Qwen3-VL and similar ViT + LLM models)

Qwen3-VL is a vision-language model: an image/video ViT encoder whose
patch embeddings are merged into the LLM's hidden state. For compression
purposes the tensor inventory splits into familiar LLM-block shapes plus a
small ViT front-end.

### 3.1 Tensor-class decision table

| Tensor class | Default RVQ path | Needs hierarchical CLAM? | Notes |
|---|---|---|---|
| ViT `patch_embed` | yes | no | small |
| ViT attention / MLP blocks | yes | no | same shape as LLM block at equivalent hidden dim |
| Vision merger / projection | yes | no | borderline (~9216 rows on some variants — check per model) |
| LLM transformer blocks | yes | no | same as Qwen3 base |
| `model.text_embedding.weight` | no | YES | `n_rows = vocab_size >= 100K` |
| `lm_head.weight` (if untied) | no | YES | same shape as `text_embedding` |
| Video 3D-patch temporal | yes | no | adds inference-time tokens, no new tensor shapes |
| MLP `[intermediate, hidden]` at 72 B scale | no | YES | `intermediate ~ 29568` crosses the 8192 row threshold |

Hidden-dim progression across model sizes (for sizing the MLP row count):
2048 (~0.5 B) -> 4096 (7 B) -> 8192 (72 B). At 72 B the MLP inverse-shape
tensors push past 8192 rows and move onto the CLAM path even though the
tensor is not vocab-adjacent.

### 3.2 Decision rule

The decision rule does not change from the base LLM pipeline:

```
if n_rows > 8192:
    hierarchical CLAM 256x256
else:
    default RVQ ladder k = [256, 512, 1024, 4096]
```

This is a one-liner scan over the safetensors header; no per-model tuning
required for the common case.

### 3.3 What a broken `text_embedding` actually breaks in a VL model

Image tokens enter the LLM via `ViT encode -> merger -> hidden-state
injection` at the post-merger layer. They do NOT go through the text
`text_embedding` lookup. Consequences if `text_embedding` is miscompressed
but the merger and LLM blocks are clean:

- Pure-text prompts degrade (bad cos on text token embeddings).
- Generated captions and any text tokens emitted by the LLM degrade
  (same table is used on the output side if `lm_head` is tied).
- Vision-conditional hidden state entering at the merger boundary is
  preserved.

This is why the single 151K-row failure in the TTS run matters: the
symptom at generation time is worse than the per-tensor headline, because
it hits both the input embedding lookup and (when tied) the output
projection. Fix it with the CLAM path before evaluating end-to-end
codec / logits match.

---

## 4. Comparison with existing in-repo compression stacks

All four codecs exist for different reasons. RVQ is for "I have a fixed
model architecture, I do not want to change inference, I just want the
weights smaller." The others replace parts of inference itself.

| Codec | Target | Ratio | Storage form | Runtime | When to use |
|---|---|---|---|---|---|
| RVQ (this PR) | full-precision weight matrices | 2 - 2.4x (with CLAM fix) | codebook + indices | unchanged (full GEMM on reconstructed weights) | generative TTS / LLM weight storage |
| Jina 5-lane BF16 | retrieval encoder inference | ~1000x | u8 / i8 / gamma+phi lanes | 5-lane streaming | run a text-embedding encoder on device |
| DeepNSM COCA | inference replacement | ~40000x | 16.5 MB distance matrix + VSA | < 10 us / sentence lookup | fixed 4096-word English semantic tasks |
| bgz-tensor palette | attention compute | ~500x per attention matrix | palette + compose table | O(1) per `Q Kᵀ` lookup | precomputable attention heads |

### 4.1 Jina v5 / Jina-Reranker-v3 five-lane BF16

Per `lance-graph/CLAUDE.md` Model Registry: `jina-reranker-v3-BF16-5lane/`,
five-lane encoding reduces ~1.2 GB of model to ~1.1 MB (~1000x ratio).
The lanes, distilled from `CLAUDE.md` and `.claude/knowledge/phi-spiral-reconstruction.md`:

- u8 lane: distance / quantized lookup
- i8 lane: energy / signed quantized
- gamma + phi lane: spiral encoding (gamma+phi family, see knowledge doc)

Domain: Jina v5 is a RETRIEVAL encoder (text -> 1024-d vector). The
five-lane encoding targets inference-time embedding streaming, not
weight-matrix reconstruction for a generative model. RVQ and Jina 5-lane
do not compete — Jina wants to RUN the encoder tiny; RVQ wants to STORE a
generative model's weights compactly while preserving full-precision
inference.

### 4.2 DeepNSM COCA 4096 x 512-bit VSA lookup

Per `lance-graph/CLAUDE.md` `deepnsm/` crate: 4,096-word COCA vocabulary,
4096² u8 distance matrix, 512-bit VSA encoder (XOR bind + majority
bundle), 6-state PoS FSM producing SPO triples. Headline: "680 GB
transformer -> 16.5 MB, 50 ms/token -> <10 us/sentence."

This is a re-architecture, not a compression. Vocabulary is 4096 English
words and is incompatible with a 151936-entry BPE vocabulary like
Qwen3-TTS's as a drop-in replacement. Use DeepNSM when the task fits a
fixed small-vocabulary semantic surface; do not use it when you need
subword coverage or non-English.

### 4.3 bgz-tensor attention-as-table-lookup

Per `crates/bgz-tensor/`: Base17 palette, 256 archetypes,
`AttentionSemiring = distance table + compose table`. Headline progression:

```
weight matrix 64 MB
  -> Base17        136 KB
  -> 256 archetypes  8.5 KB
  -> distance table 128 KB
```

Different compression family from RVQ: palette / semiring, not vector
quantization. Best for attention where `Q Kᵀ / sqrt(d)` can become a
precomputed compose-table lookup. Complementary to RVQ — RVQ shrinks the
stored weights, bgz-tensor shrinks the work at inference time.

---

## 5. Practical workflow for a new model

1. List tensor shapes from the safetensors header.
2. Bucket by row count:
   - `n_rows < 128`  -> skip RVQ (too small, codebook dominates)
   - `128 <= n_rows <= 8192` -> default RVQ ladder
   - `n_rows > 8192` -> hierarchical CLAM 256x256 (see `RVQ_K_LADDER_TUNING.md`)
3. Run `tts_rvq_e2e` (or its VL / LLM cousin) end to end. Confirm per-tensor
   cos and the end-to-end codec / logits match.
4. If any tensor has cos < 0.99 and is not on the `> 8192` path, raise the
   k-ladder locally for that tensor.
5. If total storage > original (as happened at 1 : 1.24 on the TTS run),
   find which tensor's codebook exceeds its own weights. Either raise its
   threshold onto the CLAM path or skip RVQ for that tensor entirely.
6. If token match is still below 90% after steps 1-5, the problem is not
   the codec. Check tokenizer, BOS / EOS handling, and architecture
   constants (rope base, head dim, tied vs untied `lm_head`).

---

## 6. What this document does not cover

- Jina 5-lane implementation details — see `ndarray::simd` and
  `.claude/knowledge/phi-spiral-reconstruction.md`.
- DeepNSM implementation — see `crates/deepnsm/`.
- bgz-tensor implementation — see `crates/bgz-tensor/`.
- Retraining or distillation across model families (e.g. replacing
  Qwen3-TTS's `text_embedding` with a Jina v5 embedding). That is a model
  architecture change, not a compression-pipeline change, and is out of
  scope.
