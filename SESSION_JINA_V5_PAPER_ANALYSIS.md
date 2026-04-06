# SESSION: Jina v5 Paper Analysis — Task-Targeted Embedding Distillation

Source: arXiv:2602.15547v1, Feb 2026
Authors: Akram, Sturua, Havriushenko, Herreros, Günther, Werk, Xiao (Jina by Elastic)

---

## ARCHITECTURE

```
Base:      Qwen3-0.6B-Base (596M params)
LoRA:      4 × 20.2M params (one per task)
RoPE θ:    3.5M (train) → inference at 32K tokens
Pooling:   last-token (end-of-sequence token)
Dim:       1024 (Matryoshka: truncatable to 32, 64, 128, 256, 512, 768)
Teacher:   Qwen3-Embedding-4B (distilled FROM, 7× larger)
Training:  50K steps general-purpose + long-context fine-tune
Languages: 119 (from Qwen3-0.6B-Base)

Nano variant:
  Base:    EuroBERT-210M (212M params)
  LoRA:    4 × 6.7M
  Dim:     768
```

## 4 TASK-SPECIFIC LoRA ADAPTERS

```
Adapter         Loss Function              Prefix        Our ThinkingPreset
───────         ─────────────              ──────        ──────────────────
Retrieval       InfoNCE + Distill + GOR    Query:/Doc:   Analytical (T_gate=0.1)
STS             CoSENT ranking             Document:     Balanced (T_gate=0.7)
Clustering      Distill (cluster instr.)   Document:     Creative (T_gate=1.5)
Classification  Bi-directional NCE + RKD   Document:     Focused (T_gate=0.05)

Key: adapters are FROZEN base + trainable LoRA.
Same base model, different task-specific adaptation.
Like our ThinkingPresets: same engine, different T_gate.
```

## TRAINING STAGES

```
Stage 1: Embedding Distillation
  Teacher: Qwen3-Embedding-4B
  Student: Qwen3-0.6B-Base
  Loss: cosine distance in projected space (student→teacher dim)
  Data: 300+ datasets, 30+ languages, 50K steps
  Projection: Linear(1024 → 4096) to match teacher dim

Stage 2: Task-Specific Adapter Training
  Freeze base weights
  Train LoRA adapters per task
  Different loss per adapter (InfoNCE, CoSENT, NCE+RKD)
  Each adapter: 20.2M trainable params

For our calibration:
  Stage 1 = our Pass 3 (SiLU-ONNX MLP: distill gate correction)
  Stage 2 = our Pass 4 (LoRA per thinking style: task adaptation)
  Both in candle. No Python needed.
```

## LOSS FUNCTIONS (relevant for our training)

### InfoNCE (retrieval)
```
L_NCE = -log(exp(cos(q,d+)/τ) / (exp(cos(q,d+)/τ) + Σ exp(cos(q,d-)/τ)))
τ = learnable temperature
Hard negatives mined during training
```

### CoSENT (STS — directly optimizes Spearman ρ)
```
L_co = ln(1 + Σ exp((cos(xj,yj) - cos(xi,yi)) / τ'))
      for all pairs where s_i > s_j (ground truth ordering)

THIS IS EXACTLY WHAT OUR CALIBRATION MEASURES.
CoSENT loss = "make the ranking match the ground truth ranking."
Spearman ρ = "how well does the ranking match?"
Same thing, different notation.
```

### GOR — Global Orthogonal Regularizer (binary quantization robustness)
```
L_GOR = (1/B²) Σ (q_i · q_j)² + (1/B²) Σ (p_i · p_j)²

Penalizes high pairwise similarity between non-matching embeddings.
Drives embeddings to be uniformly distributed on the unit sphere.
→ Robust to quantization (u8/i8 lose less information)
→ Robust to truncation (Matryoshka dims)
→ Better ANN retrieval

FOR US: models trained WITH GOR have embeddings that survive
our 8-bit quantization better. Jina v5 was trained with GOR.
Jina v3 was NOT (older training). Expect: v5 i8 tables lose
LESS information than v3 i8 tables.
```

### RKD — Relational Knowledge Distillation (classification)
```
L_r = (1/M²) Σ ((1-cos(s_i,s_j))/μ_S - (1-cos(t_i,t_j))/μ_T)²

Preserves RELATIONAL structure (pairwise distances) not just embeddings.
Teacher = base model without adapter. Student = adapter output.
Prevents feature collapse during classification training.

FOR US: our distance tables ARE relational structure.
RKD loss = "preserve pairwise cosines after transformation."
Our ICC profile = "correct pairwise distances after encoding."
Same goal, different stage.
```

## BINARY QUANTIZATION ROBUSTNESS

```
Paper claims: "embeddings that remain robust under truncation
and binary quantization."

GOR regularizer is the mechanism:
  uniform distribution on unit sphere
  → no cluster of embeddings near each other
  → quantization doesn't collapse distinct embeddings to same bucket
  → u8/i8 encoding preserves more rank order

PREDICTION for our calibration:
  Jina v5 (GOR trained): Spearman ρ of i8 table vs f32 ground truth > 0.95
  Jina v3 (no GOR):      Spearman ρ of i8 table vs f32 ground truth < 0.90
  The GOR-trained model IS calibration-friendly. v3 is not.
```

## MATRYOSHKA — TRUNCATION ROBUSTNESS

```
Matryoshka dims: [32, 64, 128, 256, 512, 768, 1024]

During training: random truncation of embedding dim
→ first N dimensions carry the most information
→ cos(truncated_128D, full_1024D) ≈ 0.95

FOR US: Matryoshka means we can use LOWER dimensions for
faster distance table computation:
  1024D centroids: accurate but slow CLAM (O(N×K×D))
  256D centroids:  4× faster, ~95% of accuracy
  64D centroids:   16× faster, ~85% of accuracy

CLAM on 256D Matryoshka slice vs full 1024D:
  Same centroids but computed 4× faster.
  Table should be nearly identical.
  Test this: Spearman ρ(table_256D, table_1024D).
```

## WHAT THIS MEANS FOR OUR STACK

```
1. Jina v5 as ground truth:
   Qwen3-0.6B base = candle loads safetensors directly
   Last-token pooling = take embedding at seq_len-1
   1024D embedding = same dim as Jina v3 codebook
   GOR-trained = robust to our quantization

2. Per-task LoRA = per-style ThinkingPreset:
   We don't need to retrain the base model (Pass 4 nuclear option).
   We need LoRA adapters per thinking style, just like Jina v5 has.
   candle can train these.

3. CoSENT loss for calibration:
   If our i8 tables don't correlate with ground truth (ρ < 0.998),
   we can TRAIN a correction using CoSENT loss directly.
   Not SiLU-ONNX MLP. Not ICC linear fit.
   CoSENT: directly optimize "make the ranking match."
   candle has autograd. CoSENT is 5 lines of loss computation.

4. GOR as encoding quality predictor:
   Models trained with GOR → better i8 tables (less rank flip).
   Models without GOR → worse i8 tables (more rank flip).
   This predicts H4 outcome: ICC correction will be SMALLER for v5.

5. Matryoshka for fast CLAM:
   Build codebook from 256D slice → 4× faster.
   Verify: Spearman ρ(table_256D, table_1024D) > 0.99.
   If yes: always use 256D for CLAM, keep 1024D for ground truth only.

6. Projection layer (1024→4096):
   Jina v5 projects student embeddings to teacher space.
   We could project our 256-centroid space to Jina v5's 1024D.
   This would let us compute cos(our_centroid, jina_embedding).
   Direct calibration of codebook centroids against model output.
```

## THE CALIBRATION CHAIN (UPDATED)

```
Models trained WITH GOR (Jina v5, Gemma-300M):
  Pass 1: encode i8 from f32 cosines → table
  Pass 2: Spearman ρ vs ground truth → expect > 0.95 (GOR helps)
  Pass 3: CoSENT fine-tune if ρ < 0.998 → candle, 5 lines of loss
  Pass 4: LoRA adapter per style → candle, same as Jina v5's approach

Models trained WITHOUT GOR (Jina v3, BGE-M3, older models):
  Pass 1: encode i8 from f32 cosines → table
  Pass 2: Spearman ρ vs ground truth → expect < 0.90 (no GOR)
  Pass 3: ICC profile → linear correction
  Pass 4: SiLU-ONNX MLP if ICC insufficient → candle, 270K params
  Pass 5: LoRA adapter if model is the problem → candle

The order depends on the model. GOR-trained models need less correction.
```
