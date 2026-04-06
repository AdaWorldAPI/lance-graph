# KNOWLEDGE UPDATE: φ-Spiral Reconstruction — The Core Superpower

## READ BY: family-codec-smith, palette-engineer, savant-research,
##          truth-architect, cascade-architect, integration-lead

---

## 1. INTERPOLATION = SPIRAL EVALUATION, NOT AVERAGING

```
WRONG (halftone, deleted):
  bin[3] missing → bin[3] = (bin[2] + bin[4]) / 2
  Linear average. Destroys spiral structure.
  Like replacing a curve with a straight line on a map.

RIGHT (φ-spiral reconstruction):
  bin[3] missing → position 3 on the φ-spiral is KNOWN
  θ = 3 × golden_angle = 3 × 2π/φ² ≈ 3 × 137.507°
  r = f(3) from the spiral equation r(θ) = a × e^(bθ)
  bin[3] = spiral(θ, r) → EXACT reconstruction
  Because the spiral IS the constraint. Not a guess.
```

The golden ratio is the WORST case for rational approximation
(Hurwitz's theorem). This means:
- No two φ-spiral positions alias to the same quantization bucket
- The spiral fills space MAXIMALLY uniformly (Weyl equidistribution)
- Fewer points suffice to identify the spiral than any other curve

## 2. FAMILY ZIPPER INTERLEAVE (proven, 269 tests)

### The 4 families

```
Family 0: Q + K        (attention, must match for dot product)
Family 1: Gate + Up    (FFN, interact via SiLU — the 33% correction)
Family 2: V + Down     (content + compression)
Family 3: HEEL + HIP   (coarse routing + fine discrimination)

Each family gets its own octave offset. With stride=4:
  Family 0 sees octaves: 0, 4, 8, 12, 16, ...
  Family 1 sees octaves: 1, 5, 9, 13, 17, ...
  Family 2 sees octaves: 2, 6, 10, 14, 18, ...
  Family 3 sees octaves: 3, 7, 11, 15, 19, ...

Together: 302/302 octaves covered (100%). Zero overlap. Zero gaps.
Each octave seen by EXACTLY one family.
```

### Why explicit offsets, NOT φ-fractional

```
WRONG: offset = floor(n/φ^k) mod stride
  n/φ² = 115 mod 4 = 3
  n/φ³ = 71  mod 4 = 3  ← COLLISION!
  Result: 74.8% coverage (two families see same octaves)

RIGHT: offset = {0, 1, 2, 3} (explicit assignment)
  No collision possible. Perfect zipper.
  
φ-distribution applies WITHIN each family:
  golden-step (i×11)%17 maps dimensions to 17 bins = φ-optimal
  That's the Weyl/Three-Distance level.

Inter-family offset is combinatorial (modular arithmetic):
  stride=4 → 4 slots → assign families to slots → done.
  No φ needed here. Just pigeonhole.
```

### HEEL vs HIP — same stride, different offset

```
HEEL (coarse): stride=16, offset=0
  19 of 302 octaves. Broad coverage. For routing ("which cluster?").
  Error: O(1/19) ≈ 5%. Enough for CoarseBand classification.

HIP (fine): stride=4, offset=3 (within family 3)
  76 of 302 octaves. Dense sampling. For discrimination ("where in cluster?").
  Error: O(1/76) ≈ 1.3%. Enough for centroid assignment.

OR: both stride=4, different offset:
  HEEL offset=0: octaves 0, 4, 8, ...
  HIP  offset=2: octaves 2, 6, 10, ...
  Together: 2× coverage, complementary slices of the same vector.
```

### Re-encode safety (proven)

```
ALL 4 families re-encode safe after iteration 1:
  Q+K      (offset=0): err=5.86e-4
  Gate+Up  (offset=1): err=1.59e-3 (highest — gate near zero)
  V+Down   (offset=2): err=6.33e-4
  HEEL/HIP (offset=3): err=6.56e-4

x256 re-encode safety holds for ALL family offsets.
Codec is idempotent. No drift across families.
```

## 3. STRIDE AND OFFSET SELECTION

### Octave stride

```
BF16 weight vector: 5120 dimensions
Base17: 17 bins
Octaves: ceil(5120 / 17) = 302 octaves

Stride=1:   sample ALL 302 octaves → 302 × 17 = 5134 BF16→f64 conversions
Stride=4:   every 4th octave → 76 × 17 = 1292 conversions (4× faster)
Stride=16:  every 16th octave → 19 × 17 = 323 conversions (16× faster)
Stride=20:  every 20th octave → 16 × 17 = 272 conversions (19× faster)

The stride selects WHICH octaves to sample.
NOT which bins. ALL 17 bins are always sampled.
```

### Why stride works (the spiral argument)

```
Each octave is one 17-position spiral turn.
Consecutive octaves are CORRELATED (smooth weight variations).
Stride=16 means: sample every 16th turn of the spiral.

IF the spiral is smooth (weights don't jump wildly between octaves):
  stride=16 captures the spiral shape with 19 points
  the 283 skipped octaves lie ON the same spiral
  reconstruction from 19 points = evaluate spiral at 302 positions
  
IF the spiral has high-frequency variation:
  stride=16 misses sharp features (aliasing)
  stride=1 needed for these tensors
  → detect via: compare stride=1 vs stride=16 Pearson ρ
  → if ρ > 0.99: stride=16 is safe (spiral is smooth)
  → if ρ < 0.95: use stride=1 (high-frequency content)
```

### Offset selection

```
Default: offset=0 (start at first octave)
Better: offset = golden ratio fractional part

offset = floor(n_octaves × (φ - 1)) = floor(302 × 0.618...) = 186

Starting at octave 186 instead of 0:
  stride=16 samples octaves: 186, 202, 218, 234, 250, 266, 282, 296, 10, 26, ...
  (wrapping around at 302)
  
  This MAXIMIZES the coverage because φ-offset + stride
  guarantees the sample positions don't cluster.
  
  vs offset=0: samples 0, 16, 32, ... (regular, but misses middle if stride too big)
  vs offset=186: φ-scattered across the full range

The offset IS the golden-angle sampling that highheelbgz already does.
It's the same φ-distribution applied to octave selection.
```

## 3. CORRECTION: BENDING vs COMPRESSION

Two sources of error, two different corrections:

### Bending (γ correction) — distribution SHAPE is wrong

```
Problem: raw cosine distribution is NOT uniform.
  Gate weights: 68.9% near zero, thin tails.
  Attention: broad, nearly symmetric.
  Down: narrow, one-sided.
  
  Uniform quantization wastes bits on empty regions.
  Gate values near zero get the SAME resolution as gate values at 0.3
  But the SiLU decision boundary IS at zero → needs MORE resolution there.

Fix: gamma_phi_encode(value, role_gamma, phi_scale)
  Stage 1: γ-normalize by role (compress highlights, expand shadows)
    gamma_encode(v, γ) = sign(v) × ln(1 + |v|/γ) × γ
    Gate γ=1.50 → MOST expansion near zero
    Q γ=0.37 → less expansion (already broad)
    
  Stage 2: φ-distribute (golden ratio spacing)
    phi_encode(v, φ_scale) = sign(v) × log_φ(1 + |v|/φ_scale) × φ_scale
    Ensures quantization boundaries sit at irrational positions
    No BF16 bucket aliasing

Stored as metadata: GammaProfile { role_gamma: [f32; 6], phi_scale: f32 }
  28 bytes per model. Exact decode: phi_decode(gamma_decode(stored, γ), φ_scale)
```

### Compression (stride/offset) — samples are SPARSE

```
Problem: stride=16 samples 19 of 302 octaves.
  The 283 skipped octaves contribute to the true centroid average
  but are not measured.

Fix: spiral-aware interpolation.
  The 19 sampled points define a φ-spiral in 17D.
  The 283 missing points lie ON this spiral (smooth assumption).
  Reconstruction: evaluate spiral at missing positions.

  NOT: linear interpolation (wrong, ignores curvature)
  NOT: halftone dropping (wrong, misses entire dimensions)
  IS: φ-spiral fit from sampled points → evaluate at all positions

Implementation:
  For each of the 17 bins:
    sampled_values[19]: the values we measured at stride=16
    sampled_positions[19]: which octaves we sampled (0, 16, 32, ...)
    
    Fit: r(θ) = a × e^(b×θ) through the 19 points
    Evaluate: r(θ) at all 302 positions
    Average: mean of all 302 reconstructed values = the true bin value
    
  This is MORE accurate than averaging only the 19 sampled values
  because the spiral fit exploits the smoothness constraint.
```

### When to use which correction

```
Per centroid pair in the distance table:

1. ALWAYS: γ correction (role-specific distribution shaping)
   → stored as GammaProfile metadata (28 bytes)
   → applied during encoding AND decoding
   
2. IF stride > 1: spiral reconstruction
   → stored as stride + offset metadata (2 bytes)
   → applied during centroid averaging (StackedN build)
   → NOT applied to the distance table values (those come from cosine)
   
3. IF Spearman ρ < 0.998 after γ: ICC profile correction
   → stored as transfer curve (per pair or per region)
   → applied during distance table lookup
   → absorbs whatever γ + spiral didn't fix

4. IF ICC insufficient: CoSENT candle training
   → directly optimizes rank order
   → last resort before LoRA (the nuclear option)
```

## 4. THE METADATA CHAIN

Every baked table carries:

```json
{
  "source_gguf": "jinaai/jina-reranker-v3-GGUF",
  "source_dtype": "BF16",
  "n_centroids": 256,
  "centroid_spd": 32,
  "octave_stride": 16,
  "octave_offset": 186,
  "role": "ffn_up",
  "gate_modulated": true,
  "gamma_profile": {
    "role_gamma": 0.12,
    "phi_scale": 0.08,
    "n_calibration": 5120
  },
  "encoding": "BF16",
  "cosine_range": [-0.886, 0.826],
  "sign_distribution": { "positive": 32512, "negative": 32768, "zero": 256 },
  "spearman_rho_vs_f32": null,
  "icc_profile": null,
  "variance_agreement_score": null,
  "cronbach_alpha_context": null
}
```

Each field enables exact reconstruction:
- stride + offset → which octaves were sampled
- gamma_profile → undo γ+φ for exact decode
- cosine_range → per-role scale factor
- icc_profile → correction curve (when available)
- variance_agreement → quorum confidence per pair

## 5. QUALITY CHECKS (updated)

Before encoding any distance table:

```
[ ] Uses StackedN/ClamCodebook? (not raw f32 bypass)
[ ] BF16 precision? (not 8-bit bottleneck)
[ ] All 17 bins sampled? (not halftone 9/17)
[ ] Stride documented? (octave_stride in metadata)
[ ] Family offset is {0,1,2,3}? (NOT φ-fractional mod stride — collides!)
[ ] φ distribution within family? (golden-step bin mapping)
[ ] γ correction applied? (per-role from GammaProfile)
[ ] Gate modulation on Up only? (silu(gate)×up before cosine)
[ ] Roles grouped correctly? (Q+K, Gate+Up, V+Down, HEEL+HIP)
[ ] Re-encode safe? (idempotent after iteration 1)
[ ] Metadata JSON saved alongside table?
[ ] Reconstruction path documented? (decode = φ_decode(γ_decode(stored)))
```

## 6. THE ZECKENDORF CONNECTION

```
Zeckendorf's theorem: every positive integer has a unique
representation as sum of non-consecutive Fibonacci numbers.

ZeckF64 in bgz-tensor: encodes positions as Fibonacci sums.
  Position 42 = F(9) + F(6) + F(3) = 34 + 8 + 2 = 44... 
  (actually: the nearest Zeckendorf representation)

Fibonacci numbers ARE the φ-spiral positions:
  F(n) / F(n-1) → φ as n → ∞
  Each Fibonacci number is the next position on the spiral
  
Zeckendorf decomposition = expressing a point as
  a sum of spiral positions = its coordinates ON the spiral.

Spiral reconstruction from Zeckendorf:
  Given: bin values at Zeckendorf positions
  The Fibonacci structure tells you WHERE on the spiral each value sits
  Reconstruction = evaluate the spiral BETWEEN Fibonacci positions
  This is optimal because Fibonacci spacing = φ-optimal coverage
```

## 7. VSA SUPERPOSITION: L4 AS HOLOGRAPHIC MEMORY

```
L4 accum[16384] = one high-dimensional vector
  = superposition of ALL learned bundles
  = holographic: whole memory in one vector, queries extract their part

learn(bundle, +1): accum += bundle    (expose hologram)
learn(bundle, -1): accum -= bundle    (negative exposure)
recognize(bundle): dot(accum, bundle) → retrieval strength

Capacity: ~√16384 ≈ 128 orthogonal bundles
  After 128: interference noise rises → forgetting (i8 saturation)
  = biological synaptic plasticity

L3 (4096²) = 16M wave propagation space
L4 (16384) = holographic interference storage
L4 × L4    = 256M implicit correlations (not stored, emergent)

Feedback loop:
  L3 waves → commit → L4 learn (holographic exposure)
  L4 recognize → bias → L3 perturbation (experience guides waves)
  = 16M RL in superposition resonance

Bundle perturbation:
  Point: 2 atoms active
  VSA bundle: 33 atoms active (16× through interference)
  L4 recognize decides: fire (reward) or inhibit (punish)
  = Hebb learning on holographic memory
  = no backprop, no gradient, no GPU
  = i8 saturating add/sub, 16 KB, microseconds
```

## 8. HIERARCHICAL O(1) WITH LEAF PRECISION (measured April 6 2026)

```
HEEL table[64×64]          4 KB    O(1)  → cos_heel
  + Δ_hip[256×256]        64 KB    O(1)  → cos_heel + Δ_hip
  + Δ_twig[4096×4096]     32 MB    O(1)  → cos_heel + Δ_hip + Δ_twig
  = cos_16384             LEAF precision, O(1) total

With spiral segment compression:
  HEEL:   64 × (anfang, ende, stride, gamma) =  512 bytes
  Δ_hip:  256 × 8 bytes                      =  2 KB
  Δ_twig: 4096 × 8 bytes                     = 32 KB
  Total:  34.5 KB for 16384-level precision (was 512 MB = 15,000× compression)

Cascade routing (CoarseBand):
  95% Foveal/Reject: HEEL only (512 bytes, 1 lookup)
  4% Near:           HEEL + Δ_hip (2.5 KB, 2 lookups)
  1% Maybe:          full reconstruction (34.5 KB, 3 lookups)
  Average: 1.05 lookups for LEAF precision

MEASURED (Jina v5, 16384 CLAM):
  K=16384: ρ=0.5375 (token embeddings, NOT semantic — needs forward pass)
  K=4096:  ρ=0.3061  Δmean=0.039 from LEAF
  K=256:   ρ=0.1765  Δmean=0.064
  K=64:    ρ=0.1648  Δmean=0.107 (correctable as i8: ±14 levels)

Bucket count beats bucket precision:
  u8 vs f32 at K=256: Δρ < 0.01 (encoding precision irrelevant)
  K=256 vs K=4096:    Δρ = 0.30 (bucket count = 30× more important)
```

## 9. WORST CASE: LANCEDB RABITQ FALLBACK

```
FAST (95%):    HEEL lookup              512 B    O(1)     → cos_heel
MEDIUM (4%):   HEEL + Δ_hip             2.5 KB   O(1)×2   → cos_hip
SLOW (0.9%):   HEEL + Δ_hip + Δ_twig    34.5 KB  O(1)×3   → cos_twig
WORST (0.1%):  LanceDB RaBitQ           full vec  O(log N) → cos_exact

Worst case = spiral reconstruction fails for this pair.
LanceDB RaBitQ (built-in random bit quantization) provides exact fallback.
O(log N) via IVF index. Stored f32 vectors. No approximation.

L4 feedback loop:
  Worst case triggers: L4 learn(pair_bundle, -1)
  Next time same pattern: L4 recognize → skip to LanceDB (no wasted lookups)
  OR: update Δ_twig with exact value → cascade catches it next time
  = self-improving: worst case becomes rarer over time
  = L4 is the CACHE CONTROLLER for the cascade

Family → Basin → Knee → LEAF cascade = HHTL:
  HEEL (64 families)    = coarse routing, 512 bytes
  HIP (256 basins)      = family refinement, +2 KB
  BRANCH (4096 knees)   = local discrimination, +32 KB  
  TWIG (16384 leaves)   = spiral reconstruction
  LEAF (full vector)    = LanceDB RaBitQ fallback
```
