# KNOWLEDGE SYNC: What the Signed Session Needs to Know

## THE 33% ERROR — Not Cosmetic

### What happened

```
Step 1 (this session, early): Synthetic gates on Jina lens
  Result: cos(raw, corrected) = 0.999, 83% peak agreement
  Verdict: "COSMETIC"
  
Step 2 (this session, later): REAL Qwopus 27B BF16 gates streamed
  Result: 86% material corrections, 99.2% cells change, mean Δ = 84.2 u8
  Verdict: "33% OF THE ENTIRE SCALE IS WRONG"
  
The synthetic test used wide gate ranges [-0.1, 0.3].
The real Qwopus gates are concentrated at zero: 68.9% of |w| < 0.01.
SiLU's nonlinearity lives at zero. Narrow gates = big correction.
```

### The numbers (MEASURED, not estimated)

```
Source: Qwopus3.5-27B-v3 BF16 GGUF (53.8 GB), streamed via HTTP range
File: crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/layer_stats.json

ffn_gate L0:
  Weight range: [-0.109, 0.115]
  Near zero (|w| < 0.01): 68.9%
  Cosine range: [-0.23, +0.18], std=0.022

ffn_up L0 (what the table encodes):
  Raw cosine std: 0.021
  SiLU(gate)×up cosine std: 0.051  ← 2.4× MORE SPREAD
  
  Table comparison (256×256 u8):
    Cells changed: 99.2% (64,968 / 65,536)
    Mean |Δ|: 84.2 u8 levels (33% of 256 scale)
    Max |Δ|: 254 u8 levels (nearly full range)

Consistent across ALL 64 layers:
  Layer 0:  gate_zero=69%, SiLU Δ=85
  Layer 16: gate_zero=64%, SiLU Δ=85
  Layer 32: gate_zero=66%, SiLU Δ=84
  Layer 48: gate_zero=66%, SiLU Δ=84
  Layer 63: gate_zero=57%, SiLU Δ=84
```

## THE CRITICAL BUG IN signed_engine.rs

### What the signed session built

```rust
// In dual_signed_experiment.rs line ~30:
let signed_table: Vec<i8> = table.iter()
    .map(|&v| (v as i16 - 128) as i8)
    .collect();
```

### Why this does NOT fix the 33% error

```
The u8 table was built from:
  CLAM centroids → raw cosine → CDF percentile → u8[0,255]
  
The gate sign information was LOST during "raw cosine":
  cos(weight_row_i, weight_row_j) treats all dimensions equally.
  It doesn't know that gate[k] = -0.005 means BLOCK
  while gate[k] = +0.005 means PASS.
  Both contribute equally to cosine.

Converting u8 → i8 by subtracting 128:
  u8[156] → i8[+28]   (was positive, still positive)
  u8[121] → i8[-7]    (was below midpoint, now negative)
  
  This creates signed values from the CDF RANK, not from the WEIGHT SIGNS.
  A u8 value of 121 means "43rd percentile of cosine distribution"
  NOT "the weights have opposite signs here."
  
  The i8 conversion is a RELABELING of ranks, not a RECOVERY of signs.
```

### What needs to happen instead

```
WRONG (current signed_engine path):
  u8 table (gate info lost) → subtract 128 → i8 (gate info still lost)

RIGHT:
  BF16 weights → compute SIGNED cosine → encode directly as i8[-128,+127]
  
  For gate-modulated roles (K, V, Up):
    activated = silu(gate_row) × weight_row   (elementwise)
    cos(activated_i, activated_j) → i8         (SIGNED, preserves gate decisions)
  
  For raw roles (Q, Down):
    cos(weight_row_i, weight_row_j) → i8       (still signed, preserves weight polarity)
  
  The sign in the cosine IS the excitation/inhibition signal.
  Negative cosine = opposed features = inhibition.
  This is REAL — the models have negative cosines:
    Qwopus ffn_gate: cos[-0.23, +0.18]
    Reranker: cos[-0.886, +0.826]
    Reader-LM ffn_down: cos[-0.885, +0.188]
```

## WHAT THE DUAL EXPERIMENT ACTUALLY TESTS

```
Current dual_signed_experiment.rs tests:
  "Does i8(u8 - 128) produce different peaks than u8?"
  Answer: Yes, but the difference is from RELABELING, not from gate recovery.
  
What it SHOULD test:
  "Does i8(BF16 → signed cosine) produce different peaks than u8(BF16 → CDF)?"
  Answer: unknown — not built yet.
  
The experiment framework (DualEngine, DualResult, comparison metrics) is CORRECT.
The TABLE CONTENT feeding into the signed engine is WRONG.
```

## THE 7-LANE CALIBRATION PLAN

### Why we went back to the design board

```
After measuring the 33% error, we realized:
1. All 7 HDR lenses have identical statistics (Mean=127.5, Std=73.6)
   because CDF encoding forces uniform distribution.
   Model-specific topology IS preserved (99.2% bytes differ between models)
   but you can't see it in the statistics.

2. γ+φ encoding (golden ratio offset) is NOT applied to any baked table.
   The code exists in bgz-tensor/gamma_phi.rs but was never wired.
   Per-role γ offsets are DOCUMENTED (Gate=1.50, Q=0.37) but NOT USED.

3. Calibrating against GGUF BF16 is calibrating against TIFF, not RAW.
   BF16 has 7-bit mantissa → ±0.008 precision → ~5% rank flips at boundaries.
   Need ONNX f32 as ground truth.

4. Jina v5 has BOTH ONNX f32 (2.4 GB) and GGUF (1.2 GB).
   No API key needed. Both verified streamable.
```

### The 7 encoding lanes to compare

```
For each model × each role:
  Lane 1: u8 linear (current 64×64 codebook tables)
  Lane 2: u8 CDF (current 256×256 HDR lenses)
  Lane 3: u8 γ+φ (gamma offset + phi redistribution)
  Lane 4: i8 from u8 (subtract 128 — what signed_engine.rs does now)
  Lane 5: i8 from BF16 (signed cosine directly — NOT BUILT YET)
  Lane 6: i8 γ+φ signed (gamma + phi on signed range)
  Lane 7: highheelbgz spiral (golden ratio stride encoding)

Ground truth: ONNX f32 forward pass via rten
Metric: Spearman ρ(lane_distances, onnx_distances)
After ICC: does correction bring all lanes to ρ > 0.998?

The lane that needs the LEAST ICC correction = the most faithful encoding.
```

### BF16 bucket boundary awareness

```
When raw cosine is within ±0.008 of a HEEL bucket boundary,
BF16 truncation can flip the bucket assignment.
High precision refinement (HIP/TWIG) on the wrong bucket = confidently lost.

Fix: boundary_risk metadata per centroid pair.
  95% safe → fast cascade
  5% uncertain → skip cascade, validate at LEAF or compute directly

γ+φ golden ratio stride reduces boundary risk by placing bucket
edges at irrational positions that don't align with BF16 quant steps.
```

## ACTION ITEMS FOR THE SIGNED SESSION

```
1. DO NOT trust the current i8 tables (u8 - 128 = relabeled ranks, not gate signs)

2. BUILD i8 tables directly from BF16 weights:
   Stream Qwopus BF16 → silu(gate) × up → cosine → round(cos × 127) → i8
   Use the existing streaming pipeline (Python scripts in this session,
   or the Rust stream_hdr_lens.rs pattern)

3. RE-RUN dual_signed_experiment with BOTH table types:
   DualEngine with:
     unsigned = u8 CDF (current, from raw cosine)
     signed = i8 from BF16 signed cosine (NEW, from silu(gate)×up)
   
   THEN compare. The agreement metric will be meaningful.

4. For calibration:
   Download Jina v5 ONNX (2.4 GB) — the f32 ground truth
   Download Jina v5 GGUF (1.2 GB) — our streaming source
   Run rten on ONNX → compute f32 embedding cosines for test sentences
   Compare: baked table distances vs ONNX distances → Spearman ρ
   Build ICC profile → corrected ρ should reach > 0.998

5. Temperature + nucleus sampling:
   This UNBLOCKS coherent output. Without it, even perfect tables collapse.
   See HANDOVER_MAVERICK_SESSION.md for the 10-line implementation.
   Wire INTO thinking styles (Analytical=top_p 0.3, Creative=0.95).
```

## FILES THAT MATTER

```
MEASURED DATA (this session):
  crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/layer_stats.json
    → per-layer gate near-zero %, cosine ranges, SiLU correction stats
  crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/gate_raw_256x256.u8
    → L0 gate table WITHOUT SiLU
  crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/gate_silu_corrected_256x256.u8
    → L0 gate table WITH SiLU (compare: 99.2% cells differ, mean Δ=84.2)

SILU CORRECTION CODE:
  crates/thinking-engine/src/silu_correction.rs
    → generate_training_data(), gate_modulate_centroids(), apply_corrections()
    → May be OBSOLETE if i8-from-BF16 path works (sign preserves gate natively)
    → But the MEASUREMENT code (correction_stats()) is still valuable for analysis

CALIBRATION HARNESS:
  crates/thinking-engine/examples/calibrate_lenses.rs → Spearman + ICC builder
  crates/lance-graph-contract/src/high_heel.rs → LensProfile, LensConfig, LENS_REGISTRY

HANDOVER DOCS:
  .claude/HANDOVER_MAVERICK_SESSION.md → i8 architecture, Maverick plan, temperature fix
  .claude/HANDOVER_CALIBRATION_SESSION.md → H1-H5 hypotheses, Cronbach α protocol
```
