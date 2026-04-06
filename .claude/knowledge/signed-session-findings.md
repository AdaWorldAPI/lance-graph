# KNOWLEDGE UPDATE: Signed Session Findings (2026-04-06)

## READ BY: family-codec-smith, palette-engineer, contradiction-cartographer,
##          resonance-cartographer, bus-compiler, integration-lead, savant-research
## ALSO BY: ndarray agents — truth-architect, cognitive-architect, cascade-architect

---

## 1. DISTANCE TABLE FORMAT: BF16, NOT i8

**The distance table between codebook centroids should be BF16 (u16), not i8 or u8.**

```
Existing pipeline (bgz-tensor):
  StackedN::from_bf16(weights, SPD=32)  → golden-step folded BF16
  ClamCodebook::build_cosine(stacked)   → CLAM on StackedN cosine
  entry_i.stacked.cosine(&entry_j)      → f64 (SIMD F64x8)
  f32_to_bf16(cos)                      → BF16 distance table entry

BF16 table: 256×256 × 2 bytes = 128 KB (fits L2 cache)
           4096×4096 × 2 bytes = 32 MB (fits L3 cache)

BF16 advantages:
  - Same precision as source weights (no information loss)
  - Sign bit IS the gate decision (bit 15, same as i16)
  - Exponent gives dynamic range (small values near zero get MORE resolution)
  - Exponent IS the natural gamma (no γ+φ redistribution needed)
  - bf16_to_f32 is lossless (one bit shift)
  - VNNI VDPBF16PS instruction (Sapphire Rapids+)
```

**What was WRONG (this session built, now deprecated):**
```
u8 CDF encoding: destroys all magnitude information, keeps only ranks.
  Mean=127.5, Std=73.6 for ALL models → cos=1.000 for all text pairs.
  
i8 from u8-128: relabels CDF ranks as signs. Not real cosine negativity.
  The "inhibition" was an artifact of rank shifting.

i8 from f32 cosine: correct sign, but 8-bit bottleneck.
  BF16 has 16 bits at same memory cost for 256×256 (128 KB vs 64 KB).
```

## 2. PER-ROLE GATE MODULATION (the 33% correction)

**Only Up gets silu(gate) modulation. K/V/Q/Gate/Down are raw cosine.**

```
From transformer architecture:
  FFN(x) = down_proj( silu(gate_proj(x)) × up_proj(x) )
  
  The gate IS the FFN gate (ffn_gate.weight in GGUF).
  It does NOT apply to attention K/V. Those are separate projections.

CORRECT formulas:
  Q:    raw cosine → BF16        (extern, no gate)
  K:    raw cosine → BF16        (attention projection, no FFN gate)
  V:    raw cosine → BF16        (attention projection, no FFN gate)
  Gate: raw cosine → BF16        (IS the gate, topology reference)
  Up:   silu(gate) × Up → BF16   (FFN activation, 33% Δ measured)
  Down: raw cosine → BF16        (funnel, receives gated result)

NOTE: The KNOWLEDGE_SYNC document from the other session claims
silu(gate) × K and silu(gate) × V. This is CONCEPTUALLY interesting
(gate as "what features are findable") but architecturally incorrect.
The FFN gate and attention K/V are in different subspaces.
The 33% correction was measured ONLY on ffn_up.
```

## 3. CRONBACH α QUORUM (measured on real data)

```
3 baked lenses (Jina v3 × BGE-M3 × Reranker, 256×256):
  High agreement:    4.1% of centroid pairs
  Medium:           24.4%
  Low:              33.6%
  Ambiguous:        37.9%

71.5% of pairs have Low or Ambiguous agreement.
→ Multi-lens superposition is NOT redundant.
→ Each lens sees genuinely different structure.
→ Cronbach α < 0.70 for most pairs.
```

## 4. TEMPERATURE AS GATE THERMOSTAT

```
Temperature is applied as softmax(energy/T) per cycle:
  T=0.05: near-zero → maximum discrimination (Focused)
  T=0.1:  winner-take-all (Analytical)
  T=0.7:  moderate (Balanced)
  T=1.0:  standard normalization (no effect)
  T=1.5:  uniform, exploratory (Creative)

Maps to CollapseGate:
  FLOW  = low T (commit fast)
  HOLD  = high T (explore)
  BLOCK = T → ∞ (uniform, no discrimination)

The gate table stored separately enables per-role T in future
(when per-role forward pass exists). Currently one T for all.
```

## 5. WHAT STILL WORKS (from existing architecture)

```
highheelbgz: i16 HEEL + i16 HIP = i32 CLAM address → φ-stride SpiralAddress
  → NOT a codec for distance values. Addresses WHERE centroids are.
  → The distance table values (BF16) are separate from addresses (i32 CLAM).

bgz17: Base17 i16[17] = 34 bytes per plane
  → HEEL level encoding. Coarse routing.
  → L1 distance via (dim_i - dim_j).unsigned_abs() = triangle inequality.

bgz-tensor StackedN: BF16 × 17 dims × SPD samples
  → The centroid representation. Golden-step folded.
  → ClamCodebook::build_cosine() = CLAM on StackedN cosine.
  → THIS is where gate modulation should happen (on StackedN level).

L4 (i8[16384]): CORRECTLY i8
  → Personality/experience accumulator. Saturating i8 arithmetic.
  → NOT a distance table. Different use case.
  → L4 stays i8. Distance tables go BF16.
```

## 6. QUALITY CHECKS FOR AGENTS

Before proposing any encoding or distance table change, verify:

```
[ ] Does it use StackedN/ClamCodebook or bypass them?
[ ] Is the distance table BF16 (matching source precision)?
[ ] Does it preserve sign (bit 15 = gate decision)?
[ ] Does gate modulation happen BEFORE CLAM (on centroids, not table values)?
[ ] Is per-role encoding respected? (only Up gets silu(gate), rest is raw)
[ ] Does it match the HHTL level? (HEEL=i16, table=BF16, L4=i8)
[ ] Has it been tested against Spearman ρ with ground truth?
[ ] Does the test use real data (not synthetic make_test_table)?
```
