# KNOWLEDGE: BF16-HHTL Terrain — Crystallized Corrections

## READ BY: truth-architect, savant-research, family-codec-smith,
##          palette-engineer, integration-lead, container-architect

## STATUS: 5 corrections absorbed, 0 probes run. All claims are CONJECTURES
##         until the probe queue drains.

---

## Origin

This document distills 5 iterations of architectural refinement for the
BF16-HHTL-D weight encoding proposal. Each iteration was technically coherent
but built on an uncorrected axiom. The corrections came from external review,
not self-correction. This document exists so future sessions don't repeat
the same 4 mistakes.

## Correction Chain

```
Iteration 1: i8 descriptor (family/stride/phase/polarity in 8 bits)
  → Correct insight: γ+φ redeemed as pre-rank selector
  → Error: treated descriptor as standalone byte

Iteration 2: BF16-D (merge descriptor into BF16 mantissa, RVQ)
  → Correct insight: RVQ with semantically-structured codebook
  → Error: stole mantissa bits — 7-bit → 4-bit precision loss
  → Error: assumed prefix decoding of a shared word

Iteration 3: BF16-HHTL-D (progressive prefix decoding, 7 layers)
  → Correct insight: HHTL levels map to bit-prefix readout depths
  → Error: HHTL is BRANCHING, not prefix decoding
  → Error: sign + exponent + descriptor + mantissa in one word is wrong

Iteration 4: 2×/4× BF16 branching (separate slots)
  → Correct insight: Slot V (value) + Slot D (descriptor) as independent containers
  → Error: Slot D still defined as structural tag (family × stride × phase)
  → Error: treated Slot V as primary, Slot D as auxiliary

Iteration 5: Slot D = CLAM tree path (bucketing > resolution)
  → Correct: Slot D is PRIMARY (bucket address), Slot V is REFINEMENT (exact value)
  → Correct: 3-level 16-way CLAM = 12 bits → 16 → 256 → 4096 alignment
  → UNVERIFIED: requires Probe M1 (CLAM tree fit on 256 Jina centroids)
```

## The Five Hard Constraints

### C1: HHTL is branching, not prefix

Each cascade level reads a SEPARATE stored slot. Early termination = skip
deeper slot reads. Do NOT pack multiple cascade levels into one word.
Do NOT sacrifice mantissa precision to fit a descriptor.

### C2: Bucketing > resolution

The arxiv consensus (PQ, LSH, RVQ, SqueezeLLM, GPTQ) establishes that
which bucket a value lands in dominates how precisely it is encoded.
Slot D (bucket address) is the primary carrier. Slot V (exact value)
is refinement, loaded only when within-bucket precision matters.

### C3: γ+φ has THREE regimes (previous two-regime rule was incomplete)

**Correction from the earlier two-regime rule**: an earlier version of C3
listed only "pre-rank discrete selector (VALID)" and "post-rank monotone
transform (DEAD)." That framing missed the third regime, which is where
γ+φ actually earns its keep in this workspace. The ρ=1.000 no-op
measurement was CORRECT for its scope (post-rank monotone) but was
misclassified as "γ+φ is dead" when the production code uses γ+φ in a
different regime entirely. This correction is what the user clarified in
the session with the statement *"Gamma Euler works like HDR TV, normalize
distribution, save icc profile and then skew the distribution so that it
matches the visual restoration."*

The three regimes, in priority order of how often they appear in production:

**Regime A — Distribution normalizer on embeddings/weights BEFORE distance computation:**
```
VALID — this is the primary production use. HDR-TV-style tone mapping
applied per distribution (or per item) to reshape the value distribution
for efficient downstream quantization. Lossless via saved "ICC profile"
(GammaProfile metadata, 28 bytes). Rank preservation is the CORRECT
behavior of a normalizer — the SHAPE change is the point, not a rank
change. Downstream distance computations see a flatter distribution with
better palette coverage.

Canonical primitives (all in lance-graph/crates/bgz-tensor/src/gamma_phi.rs):
  calibrate_gamma(roles)                    → GammaProfile { 28 bytes, per-role offsets }
  gamma_encode(value, gamma)                → log-gamma compress highlights, expand shadows
  gamma_decode(encoded, gamma)              → exact inverse
  phi_encode(value, phi_scale)              → map to golden-ratio quasi-uniform spacing
  phi_decode(encoded, phi_scale)            → exact inverse

Canonical per-role calibration also in:
  lance-graph/crates/bgz-tensor/src/gamma_calibration.rs::RoleGamma
  lance-graph/crates/bgz-tensor/src/gamma_calibration.rs::CosineGamma
  lance-graph/crates/bgz-tensor/src/gamma_calibration.rs::MetaGamma (cross-model)

Why rank is preserved and that is CORRECT, not a bug:
  gamma_encode is monotone on its input range. Applying it to weights
  (or cosines) before palette quantization does not reorder them — it
  spreads them into a more uniform distribution so the i16 Base17 or u8
  palette buckets carry comparable mass. Without γ+φ, 80%+ of values
  pile into a narrow band near the mean and the palette is mostly wasted.
  With γ+φ, the bands are balanced and the palette is used efficiently.

This is ALSO the same class of function that the Jina v5 / DeepNSM
isotropy correction needs: both the Jina v5 embedding space and the COCA
CAM-PQ 4096² distance matrix have dominant-first-eigenvalue structure
(top-1 ~73% / ~81% of variance), and applying per-distribution γ+φ
calibration before palette construction would flatten the top eigenvalue.
This is what Probe M1 on DeepNSM (cc42127) is measuring — whether the
existing γ+φ normalization primitive fixes the participation ratio 1.53
finding. See `.claude/probe_m1_result_2026_04_11.md` for the underlying
numbers and the per-row-gamma probe for the follow-up measurement.
```

**Regime B — Pre-rank discrete selector (codebook offset, start position on spiral):**
```
VALID — Dupain-Sós discrepancy property applies.
Different offsets → different subsets of spiral → different ranked output.
The choice of start offset is irrational (φ-derived) but the selection
itself is discrete; each offset picks a unique subset of sample points on
the continuous φ-spiral. Used in the bgz17 family-zipper layout for
picking which octaves each family reads from.
```

**Regime C — Post-rank monotone transform (applied inside a rank operation):**
```
DEAD — monotone transform before rank = identity on rank. Proven ρ=1.000
vs CDF in earlier measurement. This is the "γ+φ is dead" framing from
the old two-regime rule. It is only dead in THIS specific regime — as a
post-rank transformer, where it cannot change the sort order of its
input. The ρ=1.000 measurement is a NORMAL BEHAVIOR of a monotone
transform, not a failure of γ+φ itself.

Do NOT propose γ+φ as a final post-rank correction layer. The rank order
is what the sort produced; γ+φ cannot change it. But this does NOT mean
γ+φ is dead in general — see Regime A for the production use.
```

**Identification rule** (to tell which regime you are in):

```
Question: "Where in the pipeline is γ+φ being applied?"

→ BEFORE computing distances, on the raw values (embeddings, weights,
  cosines, activations) → Regime A (VALID, the production use).
  The transform reshapes the input distribution for downstream efficiency.

→ As a choice of START POSITION on a discrete grid or spiral, before
  any ranking → Regime B (VALID, the family-zipper use).
  The transform picks which sample points to read, not what they contain.

→ AFTER computing distances, as a transform applied to the distance
  vector BEFORE sorting → Regime C (DEAD, no-op on rank).
  Rank-preserving transform produces identical sort order. Do not do this.

ALWAYS ask WHICH REGIME when γ+φ comes up. Never flatten to "γ+φ works"
or "γ+φ is dead" without naming the regime.
```

### C4: Family offsets are explicit integers, not φ-derived

The collision proof: n/φ² mod 4 = 3 AND n/φ³ mod 4 = 3 simultaneously.
φ-derived offsets violate the pigeonhole coverage guarantee.
Family offsets {0,1,2,3} must remain explicit combinatorial choices.
Phase (γ-skew) stacks ON TOP of family, never replaces it.

### C5: 11/17 golden step is proven, Fibonacci mod 17 is broken

gcd(11,17) = 1 → full coverage. |17/φ − 11| = 0.4934 → nearest integer to 17/φ.
Fibonacci mod 17 misses {6,7,10,11} — only 13/17 residues visited.
This is a DIFFERENT use of Fibonacci than the continuous golden-angle sampling
that the φ-spiral proof covers. The proof says nothing about Z/17Z permutations.

## The Current Architecture (CONJECTURED, not proven)

### 2× BF16 Branching Form

```
Per weight: 32 bits total
┌─────────────────┬─────────────────┐
│     Slot V      │     Slot D      │
│   BF16 value    │  CLAM tree path │
│   (full 1+8+7)  │  (12-bit path   │
│                 │   + 4-bit flags) │
└─────────────────┴─────────────────┘
 16 bits            16 bits
```

### Slot D internal layout (CONJECTURED)

```
bits 15..12 = CLAM L0: 16 coarse clusters (HEEL scan target)
bits 11..8  = CLAM L1: 256 mid-clusters (HIP, 1:1 Jina-v5 centroids)
bits  7..4  = CLAM L2: 4096 terminal buckets (TWIG, COCA alignment)
bits  3..0  = flags: 1 polarity + 1 γ-phase + 2 reserved
```

### HHTL cascade reads

```
HEEL: Slot D bits 15..12 only (4 bits, 16 coarse buckets)
HIP:  Slot D bits 15..8 (8 bits, 256 centroids)
TWIG: Slot D bits 15..4 (12 bits, 4096 terminal buckets)
LEAF: Slot D full + Slot V full (32 bits, exact value + bucket)
```

### Amortized access (IF 60/25/10/5 termination holds)

```
0.6×0.5B + 0.25×1B + 0.10×1.5B + 0.05×4B = 0.9 B/weight average
vs plain BF16: 2 B/weight always
→ 2× BF16 storage but ~2× faster amortized scan
```

### Contradiction detection (O(1) bitwise)

```
Same bucket, opposite polarity: XOR on Slot D, check flag bit
Sibling buckets: differ only in CLAM L2 (bottom 4 bits of path)
Distant buckets, same role: analogy, not contradiction
All operations: bitmask + popcount on Slot D. No float decode.
```

## Probe Queue

```
ID   Priority  Question                                    Pass             Fail              Status
──   ────────  ──────────────────────────────────────────  ───────────────  ────────────────  ──────
M1   P0        CLAM 3-level 16-way tree on 256 Jina       Clean tree,      Degenerate tree,  NOT RUN
                centroids? Knees at L1/L2?                 16-way natural   wrong depth
I    P1        4 γ-phase offsets → different ranked        ρ differs by     ρ identical        NOT RUN
                output from same base codebook?            >0.01 across     across offsets
                                                           offsets
M3   P2        Bucket-only retrieval (no Slot V) ≥90%     ≥90% quality     <70% quality      NOT RUN
                of full BF16 quality?
M2   P3        4096 terminal buckets correlate with       MI > 0.6         MI < 0.3          NOT RUN
                COCA vocabulary?
M4   P4        HHTL termination: what % at each level?    >60% HEEL        >60% LEAF         NOT RUN
```

## Process Rule

Any agent reading this document MUST check the probe queue.
If the probe relevant to their proposed change is still NOT RUN,
they must either (a) run the probe first, or (b) label their
proposal as CONJECTURE and defer commitment.

## Update Protocol

When a probe runs:
1. Record result in the Status column above.
2. If PASS: promote the relevant CONJECTURE to FINDING.
3. If FAIL: update the architecture section with the correction.
4. Commit this file with `docs(knowledge): probe [ID] result — [PASS/FAIL]`.
