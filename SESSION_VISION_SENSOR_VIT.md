# SESSION: Vision Sensor — ViT-Huge-14 for Medical Imaging + Multimodal

## THE ARCHITECTURE

```
Text sensor (current):
  text → tokenizer → token_ids → codebook_index → centroids → distance table → think

Vision sensor (planned):
  image → ViT patches (14×14 px) → patch embeddings → codebook_index → centroids → distance table → think

Same engine. Different sensor. Same MatVec. Same domino cascade.
Models are SENSORS. The matrix is the BRAIN.
```

## GROUND TRUTH MODELS

### Text (Jina v5 — Qwen3-0.6B)

```
Model:    jinaai/jina-embeddings-v5-text-small-text-matching
Base:     Qwen3-0.6B
Format:   safetensors (1.19 GB) + ONNX f32 (2.39 GB) + GGUF F16 (1.2 GB)
Tokenizer: Qwen3 BPE (151K vocab, 11.4 MB)
Dim:      1024
Pooling:  last-token
Tool:     candle (loads safetensors directly, no ONNX needed)
Status:   tokenizer downloaded, candle wired, forward pass TODO
```

### Vision (ViT-Huge-14 from CLIP)

```
FP32 ground truth:
  Repo:   Kijai/WanVideo_comfy
  File:   open-clip-xlm-roberta-large-vit-huge-14_visual_fp32.safetensors
  Size:   2.53 GB
  Precision: FP32 (24-bit mantissa, NO BF16 truncation)
  Tool:   candle (loads safetensors) OR rten (after ONNX conversion)

BF16 production:
  Repo:   DeepBeepMeep/Wan2.1
  File:   models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors
  Size:   2.39 GB
  Precision: BF16 (7-bit mantissa, ±0.008 rank flips at boundaries)
  Includes: BOTH text encoder (XLM-RoBERTa) + visual encoder (ViT-Huge-14)

Architecture:
  ViT-Huge-14:
    Patch size: 14×14 pixels
    Each patch = one "token" (like BPE subword for text)
    ~630M parameters
    Trained contrastively with XLM-RoBERTa (CLIP objective)
```

### Cross-Modal (CLIP — text ↔ image in same space)

```
The CLIP training objective:
  For (text, image) pairs:
    text_emb = XLM-RoBERTa(text)
    image_emb = ViT-Huge-14(image)
    loss = contrastive(text_emb, image_emb)

After training:
  cos(text_emb, image_emb) = semantic similarity across modalities
  "amyloid plaque in temporal lobe" ↔ brain MRI = high cosine
  "amyloid plaque in temporal lobe" ↔ chest X-ray = low cosine

For our architecture:
  Text codebook (XLM-RoBERTa) and vision codebook (ViT) share embedding space
  Cross-modal distance table: text_centroid × image_centroid → similarity
  One CompositeEngine with text lens + vision lens → superposition
```

## MEDICAL IMAGING PIPELINE

```
Phase 1: Image input
  DICOM → PNG/TIFF → resize to ViT resolution
  OR: direct from PACS/radiology viewer

Phase 2: ViT forward pass (rten, pure Rust)
  Image → 14×14 patches → ViT encoder → f32 embedding per patch
  Global: mean pool patch embeddings → 1024D image embedding
  Local: per-patch embeddings for segmentation

Phase 3: Codebook + distance table
  CLAM 256 centroids from ViT patch embeddings (same as text pipeline)
  256×256 distance table (same HDR CDF or i8 signed encoding)
  codebook_index: patch_embedding → centroid_id

Phase 4: ThinkingEngine
  perturb(patch_centroid_ids) → think(10 cycles) → commit()
  Same engine, same MatVec, same domino cascade
  Qualia from convergence = visual gestalt of the image

Phase 5: SPO extraction
  Dominant atoms → centroid labels → SPO triples
  (lesion, ADJACENT_TO, ventricle)
  (tumor, LARGER_THAN, 2cm)
  NARS truth values from convergence confidence

Phase 6: Cross-modal query
  Text: "Show me cases with amyloid plaques near the hippocampus"
  → Jina v5 tokenize → codebook → text_centroids
  → CLIP cross-modal similarity with image_centroids
  → Ranked retrieval from image database
```

## WHALE SONOGRAPHY (SESSION_WHALE_SONOGRAPHY.md)

```
Same pipeline applied to:
  Ultrasound images → ViT patches → codebook → think
  Age-cohort stratification via L4 experience
  Longitudinal tracking via trajectory (trajectory-cartographer agent)

The ViT sensor treats ultrasound frames as images.
No special medical preprocessing — the codebook learns the topology.
```

## OSINT INTEGRATION

```
WikiLeaks documents often contain:
  Text (cables, reports) → Jina/BGE-M3 text sensor
  Images (maps, photos, diagrams) → ViT vision sensor
  OCR'd text from images → text sensor (after ocrs/rten OCR)

Cross-modal CLIP similarity:
  "drone strike coordinates" (text) ↔ satellite image (vision)
  Both in same embedding space → one distance table query
```

## CALIBRATION (same pattern as text)

```
Vision ground truth:
  FP32 safetensors → candle forward pass → f32 patch embeddings
  Calibrate against: baked u8 CDF, i8 signed, γ+φ encoded tables
  Same 5-lane encoder, same Spearman ρ, same ICC profiles

Text ground truth:
  Jina v5 safetensors → candle forward pass → f32 text embeddings
  Same calibration pipeline

Cross-modal ground truth:
  CLIP FP32 → both encoders → cross-modal cosine
  Calibrate: cross-modal distance table vs CLIP cosine
```

## THREE TOOLS FOR THREE SENSOR TYPES

```
Tool      Text sensor              Vision sensor            Cross-modal
────      ───────────              ─────────────            ───────────
candle    Jina v5 forward pass     ViT-Huge-14 forward      CLIP joint
ort       Reranker cross-encoder   —                        —
rten      —                        Medical ViT segmentation  —

candle loads safetensors (text + vision).
ort loads ONNX (reranker only, cross-encoder architecture).
rten loads ONNX (medical imaging, pure Rust, AdaWorldAPI fork).
```

## IMPLEMENTATION ORDER

```
1. [NOW]  Jina v5 text ground truth (candle + Qwen3 tokenizer)
2. [NEXT] Cross-model text calibration (Jina v3 ↔ v5 ↔ Reranker ↔ BGE-M3)
3. [NEXT] 5-lane encoding + Spearman ρ + ICC profiles
4. [THEN] ViT-Huge-14 vision ground truth (candle + FP32 safetensors)
5. [THEN] Medical imaging codebook (CLAM on ViT patch embeddings)
6. [THEN] Cross-modal CLIP distance table
7. [THEN] OSINT multimodal query (text + image in same search)
```

## FILES

```
Ground truth models:
  jinaai/jina-embeddings-v5-text-small-text-matching  (text, Qwen3)
  Kijai/WanVideo_comfy/..._visual_fp32.safetensors    (vision, ViT-Huge-14, FP32)
  DeepBeepMeep/Wan2.1/..._bf16.safetensors            (combined CLIP, BF16)

Tokenizer:
  data/jina-v5-tokenizer.json          (Qwen3 BPE, 151K vocab, 11.4 MB)
  data/jina-v3-hdr/tokenizer.json      (XLM-RoBERTa, 250K vocab, 8.7 MB)

Code:
  src/tokenizer_registry.rs   (6 models, cross-model tokenization)
  src/ground_truth.rs         (calibration DTOs, Spearman ρ)
  src/composite_engine.rs     (multi-lens including future vision lens)
  src/tensor_bridge.rs        (F32/I8/U8/Tensor bridge for candle output)
  examples/stream_signed_lens.rs (5-lane encoder with γ+φ metadata)

Agents:
  .claude/agents/family-codec-smith.md  (HEEL/HIP/BRANCH/TWIG/LEAF encoding)
  ndarray/.claude/agents/truth-architect.md (BF16 truth, causality)
  ndarray/.claude/agents/cascade-architect.md (3-stroke search)
```
