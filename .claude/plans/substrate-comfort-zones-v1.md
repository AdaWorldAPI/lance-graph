# substrate-comfort-zones-v1 — where does each substrate formula feel at home?

> **Status:** ACTIVE, exploratory tier. Bars author-written and unaudited —
> fine for exploratory, said out loud (the
> `E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1` rule).
> No verdict-tier claim may be promoted out of this plan without an
> independent adversarial spec audit.
>
> **Operator framing (2026-08-12), three messages:**
> 1. *hold different situations constant — over water vs flatland vs storm
>    with high velocity differences / turbulence*
> 2. *good geometry vs badly calibrated*
> 3. *then we find out where the substrate formulas etc. feel at home*
>
> **The hypothesis to falsify:** a badly-calibrated substrate that maps
> DYNAMICALLY performs BETTER in strong storms than a well-calibrated
> absolute one — i.e. miscalibration is not uniformly a defect; in a
> high-variance regime an anchor-free adaptive encoding may win precisely
> because the fixed one saturates.

---

## §0 Why this plan is not a fresh idea but a convergence of three measured findings

This is not new speculation. Three already-measured results from this arc
point at the same seam, and this plan is the test that joins them:

1. **Fisher-z is per-read, not universal** (#926, `[H]`): **8.3× tighter**
   than plain rank in the storm tail on the raw field, **4.7× worse** than
   uniform on ring means. One encoding, opposite verdicts, depending on
   what it is asked to represent. *That is a comfort-zone finding already —
   this plan generalizes it to a map.*
2. **There is no absolute anchor** (`three_register_probe`, #926):
   rank-normalised palette256 is the ONLY frame in which cross-variable
   distance is defined at all. So "badly calibrated in absolute terms" and
   "the only frame that works across variables" are the same property seen
   from two sides.
3. **The golden index floor** (operator-ruled, #932): golden structure is
   usable only from convergent index ≈17 up; below that it resonates. That
   is the **geometry** axis's own pre-existing good/bad split, already
   measured and already ruled — this plan reuses it rather than inventing
   a geometry quality scale.

---

## §1 PREFLIGHT — ALREADY RUN, and it corrected the plan before any bar existed

Per the W6 lesson (`E-THE-DISPLACEMENT-FILTER-ATE-THE-STRANDED-STRATUM-1`:
*check the sample's own arithmetic against the new question's discriminating
variable BEFORE the first fetch*), the regime definitions were measured
first. Two corrections resulted, both load-bearing:

**Measured at `t0=54358`, 16°×16° boxes, `land_sea_mask` +
`geopotential_at_surface` + MSLP + 10 m winds:**

| candidate | lsm | elev σ (m) | spd σ (m/s) | **\|∇p\| mean** |
|---|---|---|---|---|
| Amazon basin | 0.98 | 121 | 0.63 | **10.23** |
| OCEAN (S Pacific gyre) | 0.00 | 0.2 | 2.05 | **14.96** |
| Australian outback | 0.95 | 175 | 1.61 | 19.84 |
| Sahara (Libyan erg) | 0.86 | 299 | 2.36 | 24.95 |
| Argentine pampas | 0.79 | 792 | 2.51 | 33.07 |
| US Great Plains | 0.99 | 698 | 2.23 | 37.41 |
| W Siberian lowland (Ob) | 0.95 | **65** | 2.77 | **43.78** |
| N European plain | 0.79 | 319 | 1.94 | 46.25 |
| **STORM** (CT-F14 storm-1 centre) | 0.00 | 7 | 5.48 | **95.59** |

**Correction 1 — "flatland" is NOT a single regime.** It spans from
*calmer than open ocean* (Amazon, `|∇p|`=10.2) to *3× ocean* (W Siberia,
43.8). Defining the middle regime by its surface-type LABEL would have
scored two physically opposite fields as one condition. **The regime axis
is therefore defined by the MEASURED field character (`|∇p|`), and the
surface-type labels are only the a-priori strategy for FINDING boxes at
different points on that axis.**

**Correction 2 — elevation confounds MSLP.** Mean-sea-level pressure over
elevated terrain is an *extrapolated* quantity, so its gradient is partly
a reduction-formula artifact, not physics. The first flatland candidate
(57 N/75 E) measured elev σ = 143 m; US Great Plains 698 m; Argentine
pampas 792 m. **Only boxes with elev σ ≤ 150 m are admissible as land
regimes** — which admits W Siberian lowland (65 m) and Amazon (121 m) and
excludes the other four land candidates on evidence, not taste.

**Correction 3 — wind speed is NOT the regime discriminator.** Ocean-calm
and the first flatland candidate measured 5.33 vs 5.47 m/s mean speed —
indistinguishable. `|∇p|` separates them 14.96 vs 39.90. **The plan scores
on `|∇p|` and speed *variance* (σ), never mean speed.**

### The regime ladder, as adopted

| tier | box | centre | `\|∇p\|` (preflight) | why this one |
|---|---|---|---|---|
| **R1 CALM** | Amazon basin | 4 S, 296 E | 10.2 | flattest *dynamics*; elev σ 121 m (admissible) |
| **R2 OCEAN** | S Pacific gyre | 25 S, 220 E | 14.9 | the operator's "constant over water"; elev σ ≈ 0 |
| **R3 ACTIVE** | W Siberian lowland | 60 N, 72 E | 43.8 | the operator's "flatland"; flattest admissible land (65 m) |
| **R4 STORM** | CT-F14 storm centres | (19 stored) | 95.6 | high velocity differentials + turbulence |

Dynamic range R1→R4 ≈ **9.3×** on `|∇p|` — enough to expect a crossover if
one exists.

---

## §2 THE TWO AXES (the operator's "good geometry vs badly calibrated")

The two axes are **orthogonal by construction** and are varied
independently, so a result can be attributed to one or the other rather
than to their blend.

### Axis A — GEOMETRY (where the samples sit)

| arm | construction | a-priori quality |
|---|---|---|
| `GEO-GOLDEN-HI` | Vogel lattice, N = F(17)² = 2 550 409, emergent pair at the index floor | **good** (operator-ruled) |
| `GEO-GOLDEN-LO` | Vogel lattice, N = F(10)² = 3 025, emergent pair F(9)/F(10) | **bad** — the sub-floor resonance zone |
| `GEO-TEMPERED` | coprime integer walk, modulus 17, stride 4 (`CurveRuler`) | **good at small q** (measured, #932) |
| `GEO-GRID` | regular square lattice, same budget | the naive baseline |

**Budget is held EXACTLY equal across arms** (the W2s-a lesson: an unequal
budget silently advantages the arm with more samples — measured there as
64 vs 80, 256 vs 293). Each arm draws exactly `n` samples per box.

### Axis B — CALIBRATION (how the 256 palette levels are placed)

| arm | construction | absolute anchor? | dynamic? |
|---|---|---|---|
| `CAL-ABS-OWN` | 256 uniform levels over THIS box's own min/max | yes | no |
| `CAL-ABS-FOREIGN` | 256 uniform levels over a DIFFERENT regime's min/max | yes, **wrong one** | no |
| `CAL-RANK-DYN` | rank-normalised within the window, re-derived per box | **no** | **yes** |
| `CAL-FISHERZ-DYN` | Fisher-z on within-window ranks (the arc's analytic codebook) | **no** | **yes** |

`CAL-ABS-FOREIGN` is the literal reading of "badly calibrated";
`CAL-RANK-DYN` / `CAL-FISHERZ-DYN` are "badly calibrated in absolute terms
BUT dynamically mapping" — the operator's actual candidate.

**Metric:** reconstruction RMSE in **Pa** (the physical unit, per the
`E-R²-IS-NEAR-BLIND` lesson — never R² alone), plus Spearman ρ of the
reconstructed vs true field, plus mean **bias** in Pa.

---

## §3 BARS (pre-registered; commit before running; controls FIRST)

- **C0 CONTROLS FIRST — and they must be able to LOSE.** Two controls per
  cell, both through the identical pipeline: (i) `CAL-SHUFFLE` — the 256
  codebook levels randomly permuted (destroys the ordering the encoding
  depends on); (ii) `GEO-DEGENERATE` — all samples drawn from one small
  sub-patch instead of spread over the box. **Both must be WORSE than every
  real arm in every regime.** If either matches a real arm anywhere, that
  cell measures nothing and is reported VOID. *(The
  `E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1` lesson: a control that
  cannot lose is as vacuous as a test that cannot fail — and, per W5, one
  that cannot DIFFER is the same defect. Both controls are cheap-smoke-
  tested for losability BEFORE the full run.)*
- **C1 REGIME LADDER HOLDS (anti-cherry-pick):** the `|∇p|` ordering
  R1 < R2 < R3 < R4 must hold on **≥3 independent timesteps**, not just the
  preflight's one. If the ladder inverts on any timestep, the regime axis
  is not stable and every downstream cell is reported with that caveat.
- **C2 THE CROSSOVER — the operator's hypothesis, two-sided:**
  `Δ = RMSE(CAL-RANK-DYN) − RMSE(CAL-ABS-OWN)` must be **> 0 in R1/R2
  (calm: dynamic loses) AND < 0 in R4 (storm: dynamic wins)** — a genuine
  sign flip. **Both failure directions are reportable results, not
  disappointments:** no flip = the hypothesis is refuted on this data and
  says so; flip in the *opposite* direction = dynamic encoding is a
  calm-regime tool, which would be a real and surprising finding.
- **C3 THE MISCALIBRATION PENALTY SHRINKS WITH TURBULENCE:** the ratio
  `RMSE(CAL-ABS-FOREIGN) / RMSE(CAL-ABS-OWN)` must be **strictly smaller in
  R4 than in R1** — the direct statement of "storms are more forgiving of
  bad calibration." Reported with the ratio at every tier, so a monotone
  trend (or its absence) is visible rather than inferred from two endpoints.
- **C4 GEOMETRY FLOOR BITES HERE (or it does not):** `GEO-GOLDEN-LO` must
  be worse than `GEO-GOLDEN-HI` at equal budget. **Pre-registered honest
  reading:** W5's B4 already found the floor to be a *safety margin, not a
  mechanism* on a smoothing metric — so a NULL here is expected-plausible
  and must be reported plainly, not buried. What would be genuinely
  informative is the floor biting on a *sampling-fidelity* metric where it
  did not bite on a *smoothing* one.
- **C5 THE COMFORT MATRIX (descriptive, the deliverable):** the full
  `regime × (geometry × calibration)` RMSE table, plus each cell normalized
  by its regime's best arm — so "where does this formula feel at home" is
  read directly off the matrix rather than argued.

---

## §4 OUTPUT CONTRACT (the artifact-completeness lesson)

Per the repeated finding that a first artifact ships summaries and omits
the operands its headline rests on (W6's per-storm predictors; W5's
family-B histogram; the chat-only 99.38 %), the JSON **must** carry:

- every cell's **raw** RMSE / bias / Spearman ρ, in Pa where dimensional
- the per-regime **codebook edges actually used** (so a miscalibration
  claim is auditable without a re-fetch)
- the **measured** `|∇p|`, spd σ, elev σ and lsm per box per timestep
- the **sample count actually drawn** per arm (the equal-budget proof, not
  the intent)
- **units on every dimensional field name**, per the `c_bow`-is-km⁻¹ lesson

Checkpoint one JSONL row per `(regime, timestep)` with resume-skip;
tag-file heartbeat in `exec-runs/`.

**Cost:** surface chunks only (`f"{t}.0.0"`, ~4 MB each). 4 regimes × 3
timesteps × 3 variables ≈ 36 chunks ≈ **150 MB, a few minutes**. The
`land_sea_mask` / `geopotential_at_surface` statics are one chunk each,
already fetched in preflight.

---

## §5 WHAT THIS PLAN DELIBERATELY DOES NOT CLAIM

- **Not a verdict on any encoding's general superiority.** It maps comfort
  zones on ONE variable (MSLP) over FOUR boxes. A comfort zone is a
  measured local fact, not a ranking.
- **Not a physics claim about storms.** "Turbulence" here is operationalized
  as `|∇p|` + speed σ. Whether that is the right proxy for the operator's
  intended *velocity-differential* notion is itself open — an alternative
  operationalization (local shear, `|∇ × v|`) is a named follow-up, not a
  silent substitution.
- **Not a substitute for CT-F17.** Nothing here touches the directional
  claim; it is a substrate-fidelity map, a different question entirely.
