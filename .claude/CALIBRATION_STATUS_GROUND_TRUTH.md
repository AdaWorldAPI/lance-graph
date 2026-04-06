# CALIBRATION STATUS — Ground Truth, April 5 2026

This document corrects several prompts that overstated readiness.
Read this BEFORE any of the SESSION_*.md prompts.

---

## TABLE SOURCE FORMATS (known, not all clean)

```
Lens                           Source Format   Clean?
────                           ─────────────   ──────
jina-v3-hdr                    CONTRADICTS: CODEBOOKS.md=Q8_0, jina_lens.rs=F16   UNCERTAIN
bge-m3-hdr                     F16 (CompendiumLabs)                                CAUTION (F16≠BF16)
CompendiumLabs_bge-m3-hdr      F16 (same source, different CLAM run?)              CAUTION
bartowski_reader-lm-1.5b-hdr   Q8_0                                               CONTAMINATED
Qwen3-5-4B-BF16-hdr            dirname=BF16                                        LIKELY CLEAN
Qwen3-5-9B-BF16-hdr            CONTRADICTS: dirname=BF16, CODEBOOKS.md=Q8_0       UNCERTAIN
jina-reranker-v3-BF16-hdr      BF16 confirmed (commit c88f51d)                     CLEAN

Qwopus 27B (305 files):        BF16 confirmed (streamed, verified)                 CLEAN

F16 ≠ BF16:
  BF16: 8-bit exponent (same as F32), 7-bit mantissa. BF16→F32 = lossless shift.
  F16:  5-bit exponent (DIFFERENT), 10-bit mantissa. F16→F32 = lossy conversion.
  Tables from F16 have a DIFFERENT exponent range than tables from BF16.
  Comparing F16-sourced vs BF16-sourced tables = comparing different scales.
```

## CALIBRATION PIPELINE — WHAT EXISTS vs WHAT'S WIRED

```
Component                    Code exists    Applied to table build
─────────                    ──────────     ──────────────────────
CLAM centroid building       YES            YES (all tables)
CDF percentile u8 encoding  YES            YES (all 256×256 HDR tables)
γ+φ encoding (gamma_phi.rs) YES            NO — CODEBOOKS.md: "Not yet calibrated"
highheelbgz spiral encoding  YES            NO — spiral is for LOOKUP, not BUILD
ICC profile correction       YES (DTO)      NO — LensProfile::build() never called
Per-role scale factors       DESIGNED       NO — nowhere stored, nowhere applied
HEEL three-finger rejection  YES            PARTIALLY — bridge.rs maps bands, not tested
calibrate_roles.rs output    YES (24 tables) UNCERTAIN — RoleGamma printed but consumed?

The 7 HDR tables are: raw CLAM cosine → CDF percentile → u8.
No γ. No spiral. No ICC. No per-role scaling.
```

## ATTRACTOR COLLAPSE — ROOT CAUSE (known)

```
MatVec repeated N times = power iteration → dominant eigenvector.
This is math, not a bug. Flat topology + no inhibition = guaranteed collapse.

Mitigations built:
  Residual connections (energy = energy + α × layer_output): PARTIALLY WORKS
  RMSNorm: WIRED in Qwopus forward pass
  Temperature/nucleus sampling: DESIGNED, deferred for i8 inhibition
  Signed i8 tables: DESIGNED in old session, BUILT in new session

Mitigations NOT wired:
  γ calibration → sharper topology → less flat → less collapse
  Per-role scaling → different ranges get different resolution
  SiLU correction: 33% of cells change on REAL Qwopus data (not cosmetic)
  ICC profile: corrects systematic encoding bias
```

## WHAT'S ACTUALLY MEASURED vs ASSUMED

```
MEASURED (in old session):
  BF16→f32 shift: lossless ✓
  CLAM 256/4096 centroids: balanced, stable ✓
  CDF encoding: uniform u8 distribution ✓
  Qwopus BF16 streaming: 64 layers, 305 files, verified ✓
  SiLU real impact: 99% cells, Δ=84-85 u8 (33% of scale) ✓
  Gate near-zero: 57-69% per layer ✓
  Cross-role independence: cos≈0.000 (orthogonal subspaces) ✓
  Input diversity: different prompts → different centroid indices ✓
  
CLAIMED BUT NOT RE-MEASURED:
  StackedBF16 SPD=32 Pearson 0.996 — from a previous session
  HEEL three-finger 72% rejection — from a previous session
  Euler-gamma fold 0.958 at n=6 — from a previous session

NOT MEASURED:
  Spearman ρ with real tokenizer (hash gave 0.13 = garbage)
  F32 ONNX ground truth (not downloaded)
  γ+φ encoding quality on any table
  F16 vs BF16 side-by-side Spearman ρ
  Codebook size knee (64/128/256/512/1024/2048/4096 sweep)
  i8 signed table quality (designed, not measured)
  Cronbach α across lenses
```

## WHAT TO DO FIRST

```
1. Determine which HDR tables are Q8_0 contaminated.
   Check the actual GGUF files used by stream_hdr_lens.rs / jina_hdr_table.rs.
   If Q8_0: rebuild from BF16 source.
   Known clean: jina-reranker (BF16), Qwopus 27B (BF16).
   Known contaminated: reader-lm (Q8_0).

2. Run codebook_pearson.rs on CLEAN tables only.
   Measure Pearson between 256-centroid table and 1:1 BF16 ground truth.

3. Apply γ+φ encoding to ONE clean table (reranker).
   Does γ sharpen the topology? Does Pearson improve?

4. Build ONE i8 signed table from reranker BF16.
   Compare: u8 CDF vs i8 signed vs γ+φ on the SAME BF16 source.

5. Wire calibrate_roles.rs output into table build.

6. Jina v5 ONNX = ground truth anchor.
   Jina v2 dropped. Jina Bert XLM or ModernBERT ONNX large (BF16/F32,
   whatever is available). CC decides the tool. Don't prescribe.
```

## KNOWN CLEAN DATA TO BUILD ON

```
Qwopus 27B BF16:
  64 layers × 5 roles = 305 tables
  Token embeddings: 248,320 × 5120 dims
  4096-centroid codebook (16 MB table)
  SiLU correction: 33% measured impact
  Gate near-zero: 57-69%
  All BF16 confirmed. THIS is the foundation.

Jina Reranker v3 BF16:
  cos[-0.886, +0.826] — widest symmetric range
  256-centroid HDR table, BF16 confirmed
  reranker_lens.rs wired, 9 tests
  Best candidate for encoding comparison experiments.

Jina v5 ONNX:
  Ground truth anchor. Replaces Jina v2 (dropped).
  Jina Bert XLM or ModernBERT ONNX large — use whatever format available.
  Not downloaded yet. Header verified.
```

## SESSION PROMPT CORRECTIONS

```
Several SESSION_*.md prompts written during this session contain errors:

"7 HDR lenses baked"       → 3 wired. Source format mixed, some possibly Q8_0.
"F16 required"             → BF16 or F32 required. F16 has different exponent range.
"24 F16 role tables"       → Source format uncertain for some models.
"Pearson 0.996 proven"     → Claimed from previous session. Not re-measured.
"SiLU correction ~1 MB"    → On REAL data: 33% of cells change. Not cosmetic.
"No embedding needed"      → CORRECT for runtime. Calibration needs ground truth 
                              forward pass (Jina v5 ONNX).

DO NOT prescribe tools (rten/ort/candle) in prompts. CC decides.
Use this document as the OVERRIDE for any contradictions.
```

## ADDED BY THIS SESSION (April 6 2026)

```
MEASURED (new):
  x256 re-encode safety: PROVEN, idempotent after iteration 1
  Cronbach α on 3 baked lenses: 71.5% disagreement (superposition NOT redundant)
  Prime stride analysis: stride=11 (GOLDEN_STEP) optimal for calibration
  Family zipper: 4 families × stride=4 = 100% coverage
  Bundle perturbation: 33 atoms (16× more than point perturbation)
  BF16 engine: signed, inhibition works, temperature differentiates

BUILT (new modules, 290 tests):
  bf16_engine.rs       — BF16 distance table engine (from StackedN)
  role_tables.rs       — per-role gate modulation (silu(gate)×Up)
  prime_fingerprint.rs — centroid as prime-DFT + VSA bundle perturbation
  reencode_safety.rs   — x256 idempotency proof
  spiral_segment.rs    — (anfang, ende, stride, gamma) = 8 bytes/row
  cronbach.rs          — Cronbach α + variance agreement scores
  pooling.rs           — ArgMax/Mean/TopK/Nucleus/Weighted
  builder.rs           — fluent API + Temperature + CommitSinks
  + 8 more modules

CONFIRMED (architecture):
  BF16 table format (not i8, not u8 CDF) — matches StackedN source precision
  Only Up gets silu(gate) modulation (not K, not V — different subspace)
  L4 is holographic VSA memory (16 KB superposition of all learned bundles)
  Cognitive Shader maps to GPU shared memory (50μs/thought)
```

## CRITICAL FIX: Reranker v3 = Qwen3 (discovered April 6 2026)

```
WRONG (in all previous session docs):
  Reranker v3 → XLM-RoBERTa (250K vocab, 12 layers, 768 hidden)
  This was Reranker v2. v3 is completely different.

CORRECT (confirmed from jinaai/jina-reranker-v3/config.json):
  Reranker v3 → Qwen3 (151K vocab, 28 layers, 1024 hidden, silu)
  = SAME architecture as Jina v5
  = SAME tokenizer (Qwen3 BPE)
  = SAME gate modulation (silu = GeGLU)

Impact:
  1. Baked reranker lens codebook_index.u16 maps Qwen2 token IDs.
     Reranker v3 uses Qwen3 tokens. → Needs rebuild.

  2. Reranker v3 and Jina v5 share the same Qwen3 tokenizer.
     → Cronbach α between them = architecture diff, not tokenizer diff.

  3. has_gate_modulation = true (silu). The 33% SiLU correction applies.

  4. ρ=-0.64 was tokenizer mismatch (Qwen3→Qwen2 codebook). Not real.
```

## CALIBRATION ERROR BUDGET (measured April 6 2026, CORRECTED)

```
Jina v5, 256 centroids, pairwise CENTROID cosine Spearman ρ:

  Pipeline stage          ρ vs raw 1024D    Δρ         Verdict
  ──────────────          ──────────────    ──         ───────
  Raw 1024D cosine        1.0000            —          ground truth
  u8 CDF encoding         1.0000            0.000      PERFECT
  i8 direct encoding      0.9973            0.003      near-perfect
  BF16 truncation          —                0.000      irrelevant

CENTROID PAIRS ARE ALWAYS 1:1 FULL RESOLUTION (1024D f32).
  The centroids are NEVER compressed. Only the TABLE VALUE is quantized.
  cos(centroid_i, centroid_j) → u8 CDF = ρ=1.000 = PERFECT.

CORRECTION: StackedN ρ=0.73 was a TEST ERROR.
  StackedN is for WEIGHT ROW streaming, NOT for centroid encoding.
  Centroids are always full 1024D. Never folded to 544D.
  The 27% "loss" was from folding centroids through StackedN,
  which nobody does. The centroids stay 1:1.

The REAL error is CLAM bucket assignment (151K tokens → K buckets):
  K=256:    ρ=0.14  (~591 tokens/bucket, many collisions)
  K=4096:   ρ=0.44  (~37 tokens/bucket, fewer collisions)
  K=16384:  ρ=0.54  (~9 tokens/bucket, near 1:1)
  
  Within-bucket: distance=0 (wrong for non-centroid tokens)
  Between-bucket: distance=cos(c_i, c_j) = EXACT
  More buckets = fewer within-bucket collisions = better ρ

Conclusions:
  1. u8 CDF is PERFECT for centroid pair encoding (ρ=1.000).
  2. i8/BF16/γ+φ debate was unnecessary for encoding precision.
  3. CLAM bucket count is the ONLY knob that matters.
  4. Bucket count >> bucket precision (proven: u8≈f32, K=256≠K=4096).
  5. StackedN is for streaming/compression of weight rows, not for centroids.
```
