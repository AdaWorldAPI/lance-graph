# HANDOVER: Next Session — Forward Pass Branch Graph + Vision + LoRA

## Date: April 6, 2026 (end of signed session, 20 PRs, 294 tests)

---

## WHAT WORKS (proven, tested, committed)

```
Forward Pass:
  ✓ Jina v5 candle Qwen3 → 1024D f32 → DISCRIMINATES
    Rumi↔Rumi cos=0.512 > Rumi↔TCP cos=0.384
    safetensors 1.2 GB on disk, candle loads natively
    VarBuilder rename (strip "model." prefix)
    Last-token pooling + L2 normalize

  ✗ ModernBERT → does NOT discriminate (MLM ≠ embedding)
    CLS: 0.959 ≈ 0.964. Mean: 0.824 ≈ 0.869. Neither works.
    MLM foundation model, not trained for embeddings.

Engine:
  ✓ BF16ThinkingEngine (from_codebook, from_f32, from_mean_pair)
  ✓ role_tables.rs (silu(gate)×Up, per-role gate modulation)
  ✓ pooling.rs (ArgMax, Mean, TopK, Nucleus with temperature)
  ✓ builder.rs (fluent API, Temperature, CommitSinks)
  ✓ dual_engine.rs (u8/i8/BF16 comparison via BuiltEngine)
  ✓ composite_engine.rs (multi-lens via BuiltEngine)
  ✓ x256 re-encode safety (idempotent after 1 iteration)
  ✓ Cronbach α (71.5% disagreement = superposition not redundant)

Calibration:
  ✓ u8 CDF encoding = ρ=1.000 (PERFECT, encoding irrelevant)
  ✓ i8 ≈ f32 for all methods (confirmed 3×)
  ✓ Bucket count >> bucket precision (K=256→4096 = 3×)
  ✓ Mean-pair table 2.9× better than centroid cosine
  ✓ Gate L3 alone: ρ=0.839 (beats ALL table methods)
  ✓ Token + 0.5×gate_delta: ρ=0.951 (sweet spot, pairwise)
  ✓ Gate correction at K=256: does NOT help (buckets too large)
  ✓ Cross-model: ρ=0.029 (IMPOSSIBLE, each model needs own codebook)

Infrastructure:
  ✓ HF API key works (streaming + download proven)
  ✓ stream_signed_lens.rs (5-lane from GGUF, tested on Reranker)
  ✓ tokenizer_registry.rs (8 models, real BPE)
  ✓ auto_detect.rs (config.json → Architecture routing)
  ✓ playground.rs (LM Studio for thinking engine)
  ✓ prime_fingerprint.rs (VSA bundle perturbation, 33 atoms vs 2)
  ✓ spiral_segment.rs ((anfang,ende,stride,gamma) 15,000× compression)
  ✓ reencode_safety.rs (14 tests, all safe)
```

## WHAT TO TEST NEXT

### 1. Qwen3-VL-Embedding-2B (the upgrade)

```
Repo: Qwen/Qwen3-VL-Embedding-2B
Size: 4.7 GB safetensors (BF16, fits on disk — 19 GB free)
Arch: Qwen3 (same as Jina v5), 28 layers, 2048D embeddings
Features: MULTIMODAL (text + image + video + screenshots)
Tokenizer: Qwen3 BPE (same as Jina v5 and Reranker v3)
Matryoshka: 64-2048 dimensions
Companion: Qwen3-VL-Reranker-2B (listwise, same base)

Why: 2B > 0.6B (Jina v5) → better discrimination expected
     2048D > 1024D → more information in embeddings
     Vision → the ImageGen sensor we planned
     Same candle Qwen3 architecture → same forward pass code

Test plan:
  1. Download safetensors (4.7 GB)
  2. candle forward pass (same code as Jina v5, adjust dims)
  3. Measure: does it discriminate BETTER than Jina v5?
  4. If yes → use as primary ground truth anchor
  5. Vision test: embed image + text → cross-modal cosine
```

### 2. 4096×16 Branch Graph

```
For EACH of 4096 centroids:
  1. Forward pass (standalone, NO KV cache, ~3ms per centroid)
  2. Top-16 nearest centroids in output space = branches
  3. NARS truth per branch (frequency + confidence)
  4. Bundle = majority_vote(top-16 fingerprints)
  5. L4 learn(bundle, reward)

Total: 4096 × 3ms = ~13 seconds (batch 32)
Result: 65,536 edges = sparse graph
  16 × (u16 index + BF16 value) = 64 bytes/centroid
  4096 × 64 = 256 KB sparse table

No KV cache needed:
  Each centroid = independent forward pass
  No autoregressive dependency
  O(1) memory per pass

The 256 KB sparse table replaces the 32 MB dense table:
  128× compression
  AND: semantic edges (from forward pass) instead of syntactic (from token embeddings)
```

### 3. LoRA Fine-Tuning (domain-specific embeddings)

```
Source: https://blog.gopenai.com/fine-tuning-qwen-qwen3-embedding-0-6b-with-lora-9de023fd6b66
Base: Qwen3-Embedding-0.6B (= Jina v5 architecture)
Tool: candle autograd (already in deps, --features calibration)

4 LoRA adapters (like Jina v5 paper):
  Ada Chat:     trained on 5700+ Ada conversation pairs
  Medical:      trained on clinical/research pairs (SESSION_WHALE_SONOGRAPHY)
  OSINT:        trained on WikiLeaks/Snowden/surveillance pairs
  Cross-domain: trained on Rumi↔Alzheimer type bridge pairs

= ThinkingPreset AS LoRA adapter:
  Analytical → OSINT adapter
  Creative → Cross-domain adapter
  Balanced → base model (no adapter)
  Focused → Medical adapter (precise, narrow)

Size: ~20M params per adapter × 4 = 80M total (vs 600M base)
Train: candle, minutes per adapter
Inference: base + adapter, no extra cost
```

### 4. Multilayered Cascade with Forward Pass Branching

```
THE ARCHITECTURE (everything connects):

  Text input
    │
    ▼
  Tokenize (Qwen3 BPE, 151K vocab)
    │
    ▼
  Forward Pass (Jina v5 / Qwen3-VL-Embedding, 28 layers)
    │   ← NO KV cache, standalone per text
    │   ← Each layer: gate accumulation (EKG of thinking)
    │   ← L3 gate as thermostat (Temperature per layer)
    ▼
  1024D/2048D Output Embedding
    │
    ▼
  Top-16 Branches (nearest centroids in output space)
    │
    ├──► NARS truth (freq, conf per branch)
    ├──► Bundle = majority_vote(branch fingerprints)
    ├──► L4 learn(bundle, reward) → holographic memory
    │
    ▼
  Sparse Graph (65K edges, 256 KB)
    │
    ▼
  ThinkingEngine cycle:
    │  L1 (64²):    HEEL routing       → CoarseBand
    │  L2 (256²):   HIP refinement     → Basin assignment
    │  L3 (4096²):  Branch propagation  → Sparse graph edges
    │  L4 (16384):  VSA superposition   → Holographic memory
    │
    ├──► Pooling (ArgMax/Mean/TopK/Nucleus)
    ├──► Temperature (per-role, gate as thermostat)
    ├──► Commit → BusDto
    │
    ▼
  L4 Feedback Loop:
    │  recognize(bundle) → Wisdom / Staunen / Blocked
    │  bias_sensor() → modify next perturbation
    │  
    │  WISDOM (+i8):   exploit known good → Analytical
    │  STAUNEN (0):    explore unknown → Creative
    │  BLOCKED (-i8):  avoid redundant → breaks attractor collapse
    │
    ▼
  CausalEdge64 (7+1 channels):
    │  CAUSES, SUPPORTS, REFINES, GROUNDS, ABSTRACTS, RELATES, BECOMES
    │  + CONTRADICTS (subtracts)
    │
    ▼
  NARS Reasoning:
    │  freq(A→B) = accumulated evidence
    │  conf(A→B) = diversity of evidence
    │  Revision rule: evidence-weighted merge
    │
    ▼
  Cross-Domain Meta-Awareness (20 KB ONNX):
    │  Input: (centroid_a, centroid_b, basin_a, basin_b)
    │  Output: which layer-combination bridges these domains?
    │  = where L14 says "converge" while L3 says "diverge"
    │  = the ONNX learns the TRANSITIONS, not the domains
    │
    ▼
  Output:
    ThoughtStruct (committed, with provenance)
    QualiaDto (17D feeling from convergence)
    CognitiveTrace (SPO triples)
    PersonaState (identity, style, gate openness)
```

### 5. What NOT to do

```
✗ Don't use ModernBERT for embeddings (MLM, doesn't discriminate)
✗ Don't use u8 CDF for table VALUES (encoding irrelevant, use mean-pair)
✗ Don't compare cross-model tables (ρ=0.029, impossible)
✗ Don't apply StackedN to centroids (centroids stay 1:1 full resolution)
✗ Don't use IDF weighting on centroids (IDF hurts at token level)
✗ Don't expect token embeddings to have semantics (needs forward pass)
✗ Don't use KV cache for embedding (standalone per text, no cache)
✗ Don't prescribe tools (CC decides candle vs ort vs rten)
```

## MODELS ON DISK

```
jina-v5-onnx/              3.4 GB  safetensors + ONNX + tokenizer + config
modernbert-onnx/            1.5 GB  safetensors + tokenizer + config (MLM, not useful for embeddings)
jina-v5-codebook/            18 MB  256 + 4096 centroid tables
jina-reranker-v3-BF16-5lane/ 1.1 MB  5-lane from GGUF stream (E/I 51/49%)
jina-v3-hdr/                560 KB  baked u8 lens (LEGACY)
bge-m3-hdr/                 560 KB  baked u8 lens (LEGACY)
jina-reranker-v3-BF16-hdr/  368 KB  baked u8 lens (Qwen2 tokens — needs rebuild)

GitHub Releases:
  v0.1.2-qwopus-layers:  28.7 MB  305 u8 tables + codebooks + tokenizer
  v0.1.1-tokenizers:     38.7 MB  4 tokenizer files
  v0.1.0-qwopus-4096:   100.7 MB  4096-centroid codebook
  v0.1.0-bgz-data:      735.2 MB  bgz7 indexes

HuggingFace (streamable):
  Jina v5 GGUF F16:     1.2 GB   v5-small-text-matching-F16.gguf
  Jina v5 GGUF Q8_0:    639 MB   + 12 more quantizations
  Qwopus 27B GGUF BF16: 53.8 GB  Jackrong/Qwopus3.5-27B-v3-GGUF
  
TO DOWNLOAD (next session):
  Qwen3-VL-Embedding-2B: 4.7 GB safetensors (multimodal, 2048D)
  Qwen3-VL-Reranker-2B:  companion reranker
```

## KEY INSIGHTS FROM THIS SESSION

```
1. Encoding precision is IRRELEVANT (u8 CDF = f32 = ρ=1.000)
2. Bucket count is EVERYTHING (more centroids = better ρ)
3. Mean-pair table 2.9× better than centroid cosine (same size)
4. Forward pass is REQUIRED for semantic discrimination
5. Token embeddings alone: max ρ=0.54 (syntax, not semantics)
6. Gate L3 is a BETTER representation space than tokens (ρ=0.839)
7. x256 re-encode safety PROVEN (idempotent after 1 iteration)
8. Cronbach α 71.5% disagreement (multi-lens superposition needed)
9. Reranker v3 = Qwen3 (NOT v2 XLM-RoBERTa — CRITICAL fix)
10. Cross-model calibration IMPOSSIBLE (ρ=0.029)
11. Gate E/I ratio across layers = EKG of thinking (L20-L22 = epiphany)
12. Spiral segment compression 15,000× (34.5 KB for 16384-level)
13. LanceDB for sorting (IVF_PQ), CLAM for reconstruction
14. Unresolved Δ = intelligence signal (where to explore)
15. i8 cognitive markers: Wisdom (+), Staunen (0), Blocked (-)
```

## CRITICAL FIXES STILL NEEDED

```
1. Reranker lens codebook_index.u16 built from Qwen2 tokens → needs Qwen3 rebuild
2. HANDOVER_SIGNED_SESSION.md has StackedN error (corrected in CALIBRATION_STATUS)
3. signed_engine.rs from_unsigned() is CDF rank relabeling (WARNING documented)
4. l4_bridge.rs uses table rows as centroid proxy (LIMITATION documented)
5. signed_domino.rs is dead code (never called)
```
