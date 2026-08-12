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
> **Operator correction (2026-08-12), two further messages that reshaped
> the design — v1's §2 was rebuilt around them, see §2's correction note:**
> 4. the method is **cross-swap and hypothesis testing, under the premise
>    that the model captures the phenomenon but is not calibrated**
> 5. in science you hold variables constant to test the others; **constancy
>    is relative**, so the design deliberately manufactures strong
>    correlation differences — **on the assumption that those differences
>    are fit to evaluate the hypothesis** (that assumption gets its own
>    falsifier, C1c)
>
> **The hypothesis to falsify:** a badly-calibrated substrate that maps
> DYNAMICALLY preserves MORE STRUCTURE in strong storms than a
> well-calibrated absolute one — i.e. miscalibration is not uniformly a
> defect; in a high-variance regime an anchor-free adaptive encoding may
> win precisely because the fixed one saturates.
>
> **Stated as the instrument** (§2): under cross-swap, the absolute
> encoding's transfer loss `L` should shrink as turbulence rises, while the
> dynamic encoding's is zero by construction — so the crossover is a
> statement about `ρ` on the diagonal, not about RMSE anywhere.

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

## §2 THE INSTRUMENT — CROSS-SWAP (Kreuztausch), not a horse race

> **Correction of this plan's first draft (operator, 2026-08-12).** v1 §2
> was built as a *comparison of formulas* — four calibration arms scored
> against each other on RMSE. That reads the premise backwards. The premise
> is: **assume the model DOES capture the phenomenon, but is NOT
> calibrated.** Under that premise miscalibration is the *condition of
> measurement*, not one arm in a race — and RMSE under a deliberately wrong
> calibration is bad **by definition**, so scoring it answers nothing. The
> informative quantity is how much of the captured **structure** survives
> the swap. The regime ladder, the budget discipline and the control gate
> from v1 are unchanged; only the question is inverted.

### §2.1 What is held constant, what is swapped

The operator's methodological frame: *you hold variables constant to test
the others; constancy is relative, so the design deliberately manufactures
strong correlation differences — on the assumption that those differences
are fit to evaluate the hypothesis.* Both halves are load-bearing here, and
the assumption in the second half is given its own falsifier (C1c).

Each cell holds everything fixed except one thing:

| held constant | varied |
|---|---|
| the box (regime), the timestep, the geometry arm, the sample count | **only the calibration's donor regime** |

Notation: `M[D][T]` = read regime **T**'s field through a codebook derived
from regime **D**. The diagonal `D = T` is own-calibration; every
off-diagonal cell is a swap. One full matrix per geometry arm, so geometry
never blends into the calibration answer.

**Constancy is operationalized, never assumed** (C1b): the within-box
spread of the discriminator must be small relative to the between-box
spread. A box that fails that is not a control condition — it is just
another data point wearing the label.

### §2.2 The matrix and its metrics

The 4×4 `M[D][T]` over `{R1 CALM, R2 OCEAN, R3 ACTIVE, R4 STORM}`. Per
cell, measured and stored raw:

| quantity | what it answers |
|---|---|
| **`ρ` Spearman(reconstructed, true)** | **PRIMARY** — how much ordering survived |
| `occupancy` = fraction of the 256 levels actually used | the mechanism: a foreign codebook collapses the field onto few levels |
| `saturation` = fraction clipped at level 0 or 255 | the other half of the mechanism: the field runs off the donor's range |
| RMSE / bias in Pa | **secondary** — evidence the swap genuinely hurts in absolute terms, never the verdict |

**The derived quantity the whole plan turns on:**

```
transfer loss   L[D][T] = ρ[T][T] − ρ[D][T]
```

— how much structure regime `T` loses when read through `D`'s calibration.
`L` is the cross-swap statement of "badly calibrated": *low* `L` means the
substrate carried the structure even though the numbers were wrong.

**`L < 0` is possible and is pre-registered as informative, not as a bug.**
Nothing forces the diagonal to be the best cell: a box whose own min/max is
set by a single outlier spreads its 256 levels badly, while a donor with a
wider range may spread them better — so a *foreign* codebook can beat the
own one. If that occurs it is reported as measured, with `occupancy` and
`saturation` beside it (which is where the mechanism would show), and it
carries a specific consequence: **`ρ[T][T]` is then not a valid reference
point for that box's `L̄[T]`, so C3's trend must be re-read against the
best-available cell and the substitution stated.** Naming this in advance is
the point — an unexpected sign discovered mid-analysis is exactly the kind
of result that gets explained away.

### §2.3 The dynamic row is degenerate — and that IS the point

A rank-normalised or Fisher-z encoding re-derived **inside the window** has
no donor at all. Its matrix row is therefore identical across `D` and
`L ≡ 0` **by construction**. That is not a defect to hide; it is precisely
the property under test — a dynamically-mapped substrate cannot be
mis-calibrated, because it carries no absolute anchor to get wrong.

Two consequences, both mandatory:

1. **The degeneracy must be VERIFIED, not asserted.** A nonzero
   off-diagonal on a dynamic arm means a donor parameter leaked into the
   window — a bug, and the run is void until it is found.
2. **…and the pipeline must be proven able to produce a NON-degenerate
   row** through the identical code path (the absolute arms must show real
   off-diagonal degradation). A row that is constant because the harness
   cannot vary anything is the `E-A-CONTROL-THAT-CANNOT-LOSE` defect in its
   can-it-DIFFER form, measured in W5.

So the real comparison is not "which formula wins" but:

> In which regime does `ρ(dynamic, own window)` exceed
> `ρ(absolute, foreign donor)` — and does the margin grow with turbulence?

That is the operator's hypothesis stated as a cross-swap, and it is the
only form in which a mis-calibrated arm can be scored fairly.

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

Three encodings, each run through the **full 4×4 donor matrix**:

| arm | construction | absolute anchor? | matrix shape |
|---|---|---|---|
| `CAL-ABS` | 256 uniform levels over donor `D`'s min/max | yes | **full** — diagonal = own, off-diagonal = swap |
| `CAL-RANK` | rank-normalised within the target window | **no** | **degenerate** — `L ≡ 0` by construction |
| `CAL-FISHERZ` | Fisher-z on within-window ranks (the arc's analytic codebook) | **no** | **degenerate** — same |

v1 listed `CAL-ABS-OWN` and `CAL-ABS-FOREIGN` as two separate arms. They
are not two arms — they are the **diagonal and the off-diagonal of one
arm's matrix**, and splitting them was the tell that the design was still a
race. `CAL-ABS` with `D = T` is own-calibration; `CAL-ABS` with `D ≠ T` is
the literal "badly calibrated"; the two dynamic arms are "badly calibrated
in absolute terms BUT dynamically mapping" — the operator's actual
candidate, and the ones whose rows are flat by construction.

**Metrics:** primary `ρ` and the derived transfer loss `L` (§2.2);
secondary RMSE and mean **bias** in **Pa** (the physical unit, per the
`E-R²-IS-NEAR-BLIND` lesson — never R² alone), reported for every cell but
never used as the verdict.

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
- **C1b CONSTANCY IS RELATIVE — so it is MEASURED, not claimed.** For the
  discriminator (`|∇p|`), the **within-box** spread must be small relative
  to the **between-box** spread: report
  `separation = (between-box range) / (mean within-box σ)` and require
  **≥ 3**. Below that, "holding the regime constant" is a label rather than
  a condition, and every downstream cell inherits that caveat explicitly.
  *(This is the operationalization of the operator's point that constancy
  is relative — a box is a control condition only insofar as its internal
  variation is dominated by the spread the design manufactured.)*
- **C1c THE REGIMES MUST DIFFER IN CORRELATION STRUCTURE, not merely in
  `|∇p|` — the suitability ASSUMPTION, made falsifiable.** The ladder is
  built on a pressure-gradient discriminator, but the hypothesis is about
  **structure**. So before any swap runs: measure each box's own
  **autocorrelation decay length** and **rank-distribution shape** (Gini /
  tail ratio of `|∇p|`). If R1 and R4 are indistinguishable on those, they
  are ONE regime for this question no matter how far apart their gradients
  are, and the ladder measures four copies of the same condition.
  **Pre-registered honest reading:** a null here VOIDS the cross-swap
  interpretation rather than weakening it — it would mean the manufactured
  spread was manufactured on the wrong axis. Reported first, not last.
- **C2 THE DEGENERATE ROW — verified, and proven capable of being
  non-degenerate.** Two halves, both required:
  (i) every dynamic arm (`CAL-RANK`, `CAL-FISHERZ`) must show `L[D][T] = 0`
  for all `D` **exactly**; any nonzero off-diagonal means a donor parameter
  leaked into the window and the run is VOID until it is found;
  (ii) through the **identical code path**, `CAL-ABS` must show a
  **non-zero** off-diagonal in at least one regime. Half (i) alone is the
  can-it-DIFFER defect measured in W5 — a row that is flat because the
  harness cannot vary anything proves nothing about the encoding.
- **C3 TRANSFER LOSS SHRINKS WITH TURBULENCE:** for `CAL-ABS`, the mean
  off-diagonal transfer loss `L̄[T] = mean_{D ≠ T} L[D][T]` must be
  **strictly smaller in R4 than in R1** — the cross-swap statement of
  "storms are more forgiving of bad calibration." Reported at every tier so
  a monotone trend (or its absence) is visible rather than inferred from
  two endpoints, and reported **alongside `occupancy` and `saturation`**, so
  a shrinking loss can be attributed to a mechanism rather than asserted.
- **C4 THE CROSSOVER — the operator's hypothesis, two-sided:**
  `Δ[T] = ρ(CAL-RANK, T) − ρ(CAL-ABS, D=T, T)` must be **< 0 in R1/R2
  (calm: own-calibration absolute wins) AND > 0 in R4 (storm: dynamic
  wins)** — a genuine sign flip against the *diagonal*, which is the
  hardest available opponent. **Both failure directions are reportable
  results, not disappointments:** no flip = the strong hypothesis is
  refuted on this data and says so; flip the *other* way = dynamic encoding
  is a calm-regime tool, which would be real and surprising.
  **The weak form is reported separately and never conflated with it:**
  `ρ(CAL-RANK, T) > ρ(CAL-ABS, D ≠ T, T)` — dynamic beats a *mis-calibrated*
  absolute. That one is nearly guaranteed by C2(i) and is therefore
  evidence of wiring, not of merit.
- **C5 GEOMETRY FLOOR BITES HERE (or it does not):** `GEO-GOLDEN-LO` must
  be worse than `GEO-GOLDEN-HI` at equal budget. **Pre-registered honest
  reading:** W5's B4 already found the floor to be a *safety margin, not a
  mechanism* on a smoothing metric — so a NULL here is expected-plausible
  and must be reported plainly, not buried. What would be genuinely
  informative is the floor biting on a *sampling-fidelity* metric where it
  did not bite on a *smoothing* one.
- **C6 THE TRANSFER MATRIX (descriptive, the deliverable):** the full
  `geometry × (donor × target)` table of `ρ`, `L`, `occupancy`,
  `saturation`, RMSE and bias — every cell raw, plus the derived `L̄[T]`
  column. "Where does this formula feel at home" is then **read off the
  diagonal**, and "how badly does it travel" **off the off-diagonal** —
  neither argued.

---

## §4 OUTPUT CONTRACT (the artifact-completeness lesson)

Per the repeated finding that a first artifact ships summaries and omits
the operands its headline rests on (W6's per-storm predictors; W5's
family-B histogram; the chat-only 99.38 %), the JSON **must** carry:

- every cell's **raw** `ρ` / `occupancy` / `saturation` / RMSE / bias,
  keyed by `(geometry, donor D, target T)` — the full matrix, not the
  diagonal plus a summary. `L[D][T]` is DERIVED in the report from stored
  `ρ`, never stored alone (per the W6 lesson: store the operands, so a
  headline can be re-derived without a re-fetch)
- the per-regime **codebook edges actually used**, for every donor — a
  miscalibration claim is auditable only if the wrong codebook is on disk
- the **measured** `|∇p|`, spd σ, elev σ and lsm per box per timestep,
  plus C1b's `separation` ratio and C1c's decay length + tail ratio
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
- **Not a claim that the four boxes are the same condition minus one
  knob.** Real regimes differ in more than the discriminator. C1b bounds
  how far the "held constant" label is earned, C1c bounds whether the
  manufactured spread lies on the axis the hypothesis is about — and
  whatever those two report travels with every downstream number rather
  than being dropped once the matrix is filled.
- **Not a claim that transfer loss isolates calibration alone.** `L` is
  measured with `occupancy` and `saturation` beside it precisely because a
  shrinking `L` could also mean the target's field happens to sit inside
  the donor's range by luck of that timestep. Three timesteps bound that;
  they do not eliminate it.
