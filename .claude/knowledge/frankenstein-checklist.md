# KNOWLEDGE: Composition Discipline — Mathematical Frankenstein

## READ BY: ALL AGENTS. truth-architect enforces.
## INSPIRED BY: Xu et al. 2026 "VibeTensor" (arXiv:2601.16238) §7
## ADAPTED: their runtime composition failures (mutexes, uninitialized buffers,
##          batch alignment) are Python/C++ problems that Rust prevents at
##          compile time. Our Frankenstein risk is MATHEMATICAL: encodings that
##          each preserve their own invariant but lose information silently at
##          representation boundaries.

---

## Our Frankenstein: Invariant Leakage at Representation Boundaries

The lance-graph codec stack chains multiple representations:

```
f32 → BF16 → i16[17] → palette index → scent byte → pairwise cosine
```

Each step preserves ONE invariant (rank, magnitude, basin, sign).
The composition risk: **information dies at the boundary between two
representations, and neither representation's tests detect it** because
each tests its own invariant in isolation.

### Example (real, measured):
```
γ+φ as post-rank monotone: ρ = 0.999992 vs 0.999992 (Lane 3 = Lane 1).
The transform PRESERVES rank (its own invariant) but ADDS NOTHING.
It composes correctly but wastes a pipeline stage.
The Frankenstein here isn't a crash — it's a zombie: alive, correct,
contributing zero.
```

### Example (real, measured):
```
Naive u8 floor: Spearman 0.999749.
Full γ+φ+CDF: Spearman 0.999992. Benefit: +0.000244.
If the next stage quantizes to scent bytes (ρ=0.937), that 0.02%
improvement is destroyed at the boundary. Wasted work.
```

---

## The Three Composition Failures That Matter

### 1. Precision Cliff at Representation Boundary

```
Stage A outputs at fidelity ρ_A. Stage B outputs at fidelity ρ_B.
If ρ_B << ρ_A, everything A does above ρ_B is wasted work.
The pipeline's fidelity = min(ρ) across all stages, not max.

Example: BF16 (ρ=0.999978) → scent byte (ρ=0.937).
         The 0.062 gap is information destroyed at the boundary.

Test: For every A→B boundary, measure ρ_A and ρ_B.
      If ρ_A − ρ_B > 0.01, the boundary is a precision cliff.
      Optimizing A above ρ_B is pointless.

RULE: THE WEAKEST LINK SETS THE CEILING.
```

### 2. Semantic Basin Mismatch

```
Stage A operates in the distribution basin (rank, shape, wave).
Stage B operates in the semantic basin (identity, address, concept).
The boundary conflates the two and neither notices.

Example: BGZ17 palette index (distribution: which wave shape) fed
         into a COCA lookup (semantic: which concept).
         Palette index ≠ semantic address. If code treats one as
         the other, the composition is silently wrong.

Test: At every boundary, label the semantic type:
      rank → rank: OK (monotone transforms preserve)
      value → value: OK (precision bounded)
      rank → address: WRONG (rank order ≠ identity)
      address → rank: WRONG (identity ≠ distributional order)
      bucket → address: ONLY IF bucket = CLAM path AND address = centroid ID
                         AND the mapping is 1:1 (Probe M1 validates this)

RULE: LABEL EVERY BOUNDARY with its semantic type.
```

### 3. The Zombie Stage

```
A pipeline stage is correct, passes its own tests, contributes
measurably in isolation — but its contribution is destroyed by
the next stage, making it zero-value overhead.

Test: For every stage S, measure:
      ρ(pipeline_with_S) vs ρ(pipeline_without_S).
      If |Δ| < measurement noise, S is a zombie. Remove it.

RULE: EVERY STAGE MUST PAY RENT.
      "Pays rent" = measurable improvement in END-TO-END fidelity,
      not just the stage's own metric.
```

---

## The Pipeline Fidelity Chain

The ONLY number that matters is end-to-end Spearman ρ between the
pipeline's output distances and the f32 reference distances.
Individual stage ρ values are diagnostics, not verdicts.

```
Boundary                        ρ vs f32      Status
──────────────────────────────  ────────────  ──────────────
f32 → BF16 RNE                 0.999978      FINDING (v2.4)
f32 → u8 CDF                   0.999992      FINDING (v2.4)
f32 → naive u8                 0.999749      FINDING (v2.5)
f32 → scent byte (ZeckF64[0])  0.937         FINDING
f32 → i16[17] Base17           ?             NOT MEASURED
f32 → ZeckF64 full (8 bytes)   ?             NOT MEASURED
f32 → palette index            ?             NOT MEASURED
f32 → NeuronPrint 6D           ?             NOT MEASURED
f32 → BGZ-HHTL-D Slot D only   ?             NOT MEASURED
```

The "?" rows are where Frankenstein hides.

**Endgame gate (v2.5):** naive u8 Pearson = 0.999860. Any encoding
below this floor is worse than doing nothing. bgz-hhtl-d needs
Pearson ≥ 0.9980 to justify cascade overhead.

---

## Mandatory Process

### Before adding a new pipeline stage:

```
1. Measure end-to-end ρ WITHOUT the stage.
2. Measure end-to-end ρ WITH the stage.
3. Is (2) > (1) by more than measurement noise?
4. If yes: stage pays rent. Proceed.
5. If no: zombie. Do not add.
```

### Before optimizing an existing stage:

```
1. What is the NEXT stage's input fidelity floor?
2. Is this stage already above that floor?
3. If yes: optimization wasted (precision cliff). Stop.
4. If no: optimize, then re-measure end-to-end.
```

### Before connecting two subsystems:

```
1. What semantic type does upstream output? (rank / value / bucket / address)
2. What does downstream expect?
3. Match? → proceed.
4. Mismatch? → insert explicit conversion, test round-trip.
```

---

## Citation

Xu et al., "VibeTensor: System Software for Deep Learning, Fully Generated
by AI Agents," arXiv:2601.16238, Jan 2026. §7: Frankenstein composition effect.
Adapted: their failure modes (runtime, memory, concurrency) are prevented by
Rust's type system. Ours are mathematical: invariant leakage, precision cliffs,
basin mismatches, zombie stages.
