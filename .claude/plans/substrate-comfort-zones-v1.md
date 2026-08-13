# substrate-comfort-zones-v1 — where does each substrate formula feel at home?

> **Status:** RUN, §7 — the pre-registered hypothesis is **REFUTED** on this
> data. Bars author-written and unaudited — fine for exploratory, said out
> loud (the `E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1`
> rule). No verdict-tier claim may be promoted out of this plan without an
> independent adversarial spec audit.
>
> **⚠ HEADLINE, READ FIRST — see §7, and §7.9 for the confound.**
> **The hypothesis is NOT SUPPORTED:** there is no sign flip in either
> pre-registered measure, normalised or not — `CAL-ABS` wins its own
> diagonal in all four regimes. That much is solid.
>
> **But the stronger "cleanly reversed, monotonic" reading is WITHDRAWN**
> (§7.9, triggered by the operator asking whether storm modelling is sound
> given that vortices are not modelled at all). `C3`'s monotone `L̄` rise
> is **rank-correlated ρ = 1.000 with each regime's own value range**, and
> `L` tracks `saturation` at Pearson **+0.917** — so it largely restates
> *wide-range boxes are hard to cover with a foreign codebook*, which is
> arithmetic, not meteorology. `C4`'s "+10.78 Pa margin" is range-inflated
> for the same reason; normalised as a ratio it reads 3.96 / 1.85 / 1.14 /
> **2.35** — not monotone, and R1 is the extreme, not R4.
>
> **Scope, stated bluntly:** every bar here runs on a **scalar pressure
> field**. No wind, no vorticity, no rotation enters any metric. This plan
> answered *"absolute vs window-local encoding of a scalar field whose
> range varies by regime"* — a real question — but **not** the operator's
> original question about turbulence.
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
> dynamic encoding's is zero by construction.
>
> **⚠ AMENDED by the D-CZ-1 run — see §6.4.** This block first read *"so the
> crossover is a statement about `ρ` on the diagonal, not about RMSE
> anywhere."* **Measured: `ρ` is SATURATED on the diagonal** — the spread
> between the two real arms is 3×10⁻⁶…4.7×10⁻⁵, so C4 as pre-registered
> could not have fired. `L` (off-diagonal) keeps `ρ`, where it has four
> orders of magnitude of range; **C4 moves to RMSE in Pa** with `ρ` as a
> floor check. Corrected here rather than left standing in the header while
> §6 says otherwise — a summary sentence surviving a revision that
> contradicts it is this arc's most repeated defect.

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

> **⚠ THIS TABLE IS NOW REPRODUCED, AND PARTLY CORRECTED — see §6.1/§6.2.**
> When written, none of these figures had a committed script or JSON behind
> them. Four rows now reproduce (ratios 1.004 / 1.022 / 0.994 / 0.931); the
> five EXCLUDED land candidates remain **unreproducible** because their box
> centres were never recorded anywhere. The `|∇p|` definition is identified
> as **Pa per grid cell with NO cos(lat) metric**, which understates high
> latitude — R3 is ~40 % low. Metric-corrected the ladder is 10.3 / 15.5 /
> 61.2 / 100.9: the **ORDER survives** and the range widens to ≈**9.8×**.
> The regime axis stands; these magnitudes do not.

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
> **`ρ(absolute, OWN donor — the diagonal)`** — and does the margin grow
> with turbulence?

**Against the DIAGONAL, deliberately: it is the hardest available
opponent.** The comparison against a *foreign* donor is the weak form. It
is reported separately (C4) and labelled as evidence of **wiring, not of
merit**, because C2(i) makes the dynamic arm win it almost by construction
— an encoding with no anchor cannot be out-transferred by one whose anchor
is deliberately wrong. Scoring the hypothesis on the swapped cell would
re-introduce exactly the tautology this whole section removes.

That is the operator's hypothesis stated as a cross-swap, and it is the
only form in which a mis-calibrated arm can be scored fairly.

> **Caught by review, and it is the same defect one level up (2026-08-12,
> CodeRabbit on #944).** The first version of this passage named the
> foreign-donor comparison as "the operator's hypothesis" — contradicting
> C4, `STATUS_BOARD`, and the `INTEGRATION_PLANS` entry, all of which
> correctly score against the diagonal. The passage was written BEFORE C4
> was sharpened, and the sharpening was not propagated back. Left standing,
> the plan would have carried two incompatible verdict criteria with the
> weaker one holding the headline — the swapped cell establishing merit,
> which is the precise thing §2 exists to prevent. Recorded rather than
> silently patched: it is the fifth instance of a claim that is consistent
> with its own operands but inconsistent with a sibling claim (cf. #930's
> fused relation, #927's decimal claim, #928's audit figures, #941's
> dropped qualifier), and it is a REVIEWER who caught it, not the author.

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
- **C4 THE CROSSOVER — the operator's hypothesis, two-sided.**
  **⚠ METRIC AMENDED by D-CZ-1 (§6.4): the primary quantity below is RMSE in
  Pa, not `ρ`.** `ρ` measured saturated on the diagonal (real-arm spread
  3×10⁻⁶…4.7×10⁻⁵) and is retained only as a floor check (< 0.999 = broken,
  not merely lost). The sign convention flips with the metric: RMSE is
  lower-is-better, so the inequalities below invert. Stated as the amended
  bar, with the pre-registered form kept beneath it:
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


---

## §6 RUN — D-CZ-0 (reproduced) + D-CZ-1 (the gate), 2026-08-12

Script `probes/weather-p1/substrate_comfort_d_cz_0_1.py`, results
`…_d_cz_0_1.json`, tag-file `exec-runs/…txt`. One timestep, ~96 MB.

### §6.1 D-CZ-0 was recorded DONE with NO committed artifact

**The finding that forced this run.** §1's nine-row preflight table is on the
board as **DONE**, and no script and no JSON producing any of its figures had
ever been committed. That is the chat-only-figure defect of #936, and it
survived three subsequent PRs — including an explicit self-audit on #945 that
called the ladder numbers "verified". That audit compared the arc entry to the
PLAN. Both are prose. **A figure cited in two documents is cited twice, not
confirmed.**

Now reproduced and committed:

| regime | recorded | reproduced (identified defn) | ratio |
|---|---|---|---|
| R1 CALM Amazon | 10.23 | 10.27 | 1.004 |
| R2 OCEAN S-Pacific | 14.96 | 15.29 | 1.022 |
| R3 ACTIVE W-Siberia | 43.78 | 43.53 | 0.994 |
| R4 STORM (19, each at its OWN `t0`) | 95.59 | 88.99 | 0.931 |

**Still NOT reproducible:** the five EXCLUDED land candidates (Australian
outback, Sahara, Argentine pampas, US Great Plains, N European plain). Their
box centres were never written down anywhere. No coordinates were invented to
fake those rows; the gap is reported instead.

### §6.2 The recorded `|∇p|` ignores the cos(lat) metric — order survives, magnitudes do not

The definition was never committed either, only values — so four candidates
were computed and the winner decided from the data:

| candidate | max \|ratio−1\| |
|---|---|
| Pa per 100 km, cos(lat) | 4.028 |
| Pa per cell, cos(lat) | 0.398 |
| Pa per 100 km, flat | 2.678 |
| **Pa per cell, FLAT (no cos)** | **0.069** ✅ |

So §1's figures are a plain `np.gradient` over the raw lat/lon array. That
understates the ZONAL gradient by `1/cos(lat)`, i.e. **R3 at 60 N is ~40 %
low**. Metric-corrected the ladder reads 10.3 / 15.5 / 61.2 / 100.9 — the
**ORDER survives and the dynamic range WIDENS from 9.3× to ≈9.8×**. The
regime axis stands; the recorded magnitudes do not.

### §6.3 D-CZ-1 — the C0 gate PASSES

Both controls lose to both real arms, on both metrics, in **all four**
regimes. The mechanism is visible rather than assumed: `GEO-DEGENERATE`
saturates **72–97 %** of the box because its donor patch's range is far too
narrow — which is exactly the failure mode the transfer matrix is built to
measure.

> **⚠ FIGURE CORRECTED (2026-08-12, self-audit on #947).** This read
> **92–97 %** when first written, and that was TRUE of the numbers then on
> disk (R1 0.9496 / R2 0.9174 / R3 0.9718 / R4 0.9179). It went stale in
> the SAME PR: fixing R4 to measure each storm at its own `t0` moved R4's
> saturation to **0.7224**, and the range with it. Per-regime as committed:
> **R1 0.9496 · R2 0.9174 · R3 0.9718 · R4 0.7224**; across all 19 storms
> min **0.693** / median **0.843** / max **0.974**.
>
> **The verdict is unaffected** — 72 % is still overwhelming saturation and
> the gate passes 19/19 — but the figure was wrong in four files and two PR
> bodies. Seventh instance of the arc's recurring defect: a number true when
> written, stale once the artifact beneath it changed, and carried forward
> because the check compared prose to prose. Caught only by re-verifying
> against the JSON.
>
> **And the corrected number is more interesting than the wrong one.** The
> storm regime saturates *least* (0.72 vs 0.92–0.97 in the calmer tiers) —
> i.e. the degenerate donor hurts **less** where the field is strongest,
> which is independently the direction §6.6's exploratory correlation
> measured (ρ = +0.444). Two measurements that were never connected agree.
> Recorded as a coherence, NOT as evidence: §6.6 is still p = 0.0578.

| regime | CAL-ABS ρ | CAL-RANK ρ | SHUFFLE ρ | DEGENERATE ρ (satur.) |
|---|---|---|---|---|
| R1 | 0.999960 | 0.999993 | 0.058 | 0.380 (0.950) |
| R2 | 0.999981 | 0.999992 | 0.159 | 0.478 (0.917) |
| R3 | 0.999989 | 0.999992 | 0.003 | 0.290 (0.972) |
| R4 | 0.999962 | 0.999992 | 0.006 | 0.477 (0.918) |

**C1b also passes: `separation` = 6.28** against the ≥ 3 bar.

**R4 re-run across ALL 19 storms, not one representative.** The first pass
scored a single median-`|∇p|` storm — defensible for a smoke test, but the
19 fields were already in memory, and reporting one when 19 are available is
the sample-composition weakness this arc has paid for twice (W6's stranded
stratum, W5's subsampled control). Re-run:

| arm | ρ min | ρ median | ρ max | RMSE median (Pa) |
|---|---|---|---|---|
| `CAL-ABS` | 0.999944 | 0.999977 | 0.999987 | 4.6 |
| `CAL-RANK` | 0.999992 | 0.999992 | 0.999992 | 7.6 |
| `CAL-SHUFFLE` | −0.078 | 0.005 | 0.154 | 1498.6 |
| `GEO-DEGENERATE` | 0.277 | 0.773 | 0.943 | 936.1 |

**The gate holds for EVERY storm** — controls lose on ρ and on RMSE in
19/19, not merely at the median. That is a materially stronger statement
than the single-storm version supported.

**And the spread exposes something the single storm hid.**
`GEO-DEGENERATE`'s ρ ranges **0.277 → 0.943**: on some storms a degenerate
donor is nearly adequate. It still loses everywhere (the real arms sit at
0.99999), but *how badly* miscalibration hurts is strongly storm-dependent
— which is the plan's own hypothesis appearing in the control rather than
in an arm. Note also that the storm which is median by `|∇p|` is NOT median
by ρ (0.477 vs the true median 0.773), so the first pass's representative
was unrepresentative on the axis that mattered.

### §6.4 ⚠ AMENDMENT to C4 — ρ is saturated on the DIAGONAL

**Measured, not argued:** the ρ spread between the two REAL arms is
**3×10⁻⁶ … 4.7×10⁻⁵**. At 256 levels on a smooth pressure field both real
arms reconstruct the ordering essentially perfectly, so **ρ cannot separate
`CAL-ABS` from `CAL-RANK` at all** — and C4 as pre-registered compares
exactly those two, on the diagonal. **C4 could not have fired.**

The same run shows ρ is *excellent* where the plan actually needs it: real
vs degraded is 0.99999 vs 0.29–0.48, four orders of magnitude of separation.
So the honest split is **not** "ρ is the wrong metric" — it is:

- **`L` (transfer loss, off-diagonal) keeps ρ.** Huge dynamic range there.
- **C4 (two real arms, both on their own diagonal) moves to RMSE in Pa**,
  with ρ retained as a floor check (any arm dropping below ρ ≈ 0.999 has
  broken, not merely lost). RMSE *does* discriminate: real-arm ratios 3.96 /
  1.85 / 1.14 / 1.49 across R1–R4.

**Why this amendment is legitimate and what would make it not.** D-CZ-1's
stated purpose is to test the apparatus BEFORE the expensive cells, and no
C4 measurement exists — this is a metric found blind in preflight, which is
what preflight is for. It would be illegitimate the moment any C4 cell had
been scored. Recorded as an amendment with its trigger rather than edited
into C4 silently.

### §6.6 EXPLORATORY — the penalty may shrink with field strength WITHIN R4

`GEO-DEGENERATE`'s ρ spread across the 19 storms invites an obvious
question: does the miscalibration penalty shrink as the field gets stronger
— C3's direction, but inside ONE tier and on a CONTROL arm?

| quantity | value |
|---|---|
| Spearman ρ(`GEO-DEGENERATE` ρ, storm `\|∇p\|`) | **+0.444** |
| n | 19 |
| permutation p, two-sided, 200 000 perms | **0.0578** |
| null \|ρ\| 95th percentile | 0.456 |

**Above 0.05. NOT significant.** The direction matches the hypothesis and
the magnitude is right at the threshold, which is exactly the shape of
result that gets over-claimed.

**The obvious dismissal was checked and does NOT work.** The tautology would
be: a stronger gradient just means a wider pressure range across the box, so
a fixed narrow donor patch covers proportionally less of it and ρ falls
mechanically. Measured:

| confound | Spearman ρ |
|---|---|
| `GEO-DEGENERATE` ρ vs box pressure range | **−0.035** |
| storm `\|∇p\|` vs box pressure range | +0.253 |
| `GEO-DEGENERATE` ρ vs donor/box range fraction | −0.081 |

So `|∇p|` and pressure range are **not** proxies here, and the penalty does
not track how much of the box the donor covers. The +0.444 survives all
three — which means neither the confirmation nor the easy dismissal is
available on this data.

**Status: EXPLORATORY, not a result.** One tier, a control arm, n = 19,
unpre-registered, p above 0.05. Its only legitimate use is as a reason the
full cross-regime run is **worth doing** — never as evidence that it will
succeed. Committed to the JSON (`exploratory_within_r4`) so a later run
cannot quietly restate it as a confirmation.

### §6.5 A HINT that is explicitly NOT a result

`CAL-ABS` beats `CAL-RANK` on RMSE in every regime, and its margin **shrinks
as the field gets more active**: R1 3.96× → R2 1.85× → R3 1.14×. That is the
shape of the operator's hypothesis — the absolute encoding's advantage
eroding as turbulence rises.

**It is not evidence.** One timestep; R4 breaks the monotone (1.49); no
control was run on this particular comparison; and the whole point of the
cross-swap design is that a diagonal-vs-diagonal difference is not what the
hypothesis is about. Recorded so a later run cannot present it as a
confirmation that was there all along.


---

## §7 RUN — D-CZ-2..7, the cross-swap matrix (2026-08-12)

Script `probes/weather-p1/substrate_comfort_d_cz_2_7.py`, results
`…_d_cz_2_7.json`, tag-file `exec-runs/…txt`. Run in the order the plan
requires: **C1c first**, because a null there would VOID the interpretation
before anything downstream is worth trusting. C0's construction had a real
bug, found and disable-verified fixed before any of C2–C6 was reported.

### §7.0 A construction bug in C0 itself, caught by C0's own gate

The first run of this probe **failed** C0 in two of four regimes:
`GEO-DEGENERATE` did not lose to the real arms in R1 or R4. Per §3's own
rule — *"if either matches a real arm anywhere, that cell measures
nothing and is reported VOID"* — nothing downstream could have been
trusted as written.

**Cause:** the "degenerate donor" was built from `truth[:len//64]` — the
first slice of an array already subsampled by `rng.choice` for equal
budget. `rng.choice` returns no spatial order, so that slice is an
**ordinary random subsample**, not a narrow spatial patch — a materially
different construction from D-CZ-1's correct one
(`p[si,sj][:n_i,:n_i]`, a genuine 2-D corner). A random subsample of a
flat array is not reliably narrower in range than the whole array; by
chance it can nearly match it.

**Fixed** by keeping each regime's full 2-D box alongside the equal-budget
flat evaluation sample, and building the degenerate donor from an actual
`n_i × n_i` corner of that box (`n_i = side // 8`), matching D-CZ-1
exactly. **Disable-verified**: reverting to the flat-slice construction
reproduces the *exact* original failure (R1 and R4 fail, R2/R3 pass);
the fix reproduces the same real-arm numbers unchanged (C2–C6 identical
between the two runs) while making **C0 pass cleanly in all four
regimes**. The bug was isolated to the control; the real arms were never
wrong.

### §7.1 C1c PASSES — the cross-swap interpretation is licensed

| regime | decay length (cells) | Gini(\|∇p\|) | tail ratio (p99/p50) |
|---|---|---|---|
| R1 CALM | 25.5 | 0.4321 | **7.65** |
| R2 OCEAN | 32.0 | 0.2758 | 3.72 |
| R3 ACTIVE | 32.0 | 0.2262 | 1.94 |
| R4 STORM (median of 19) | 22.5 | 0.3062 | **2.95** |

R4/R1 ratios: decay **0.88**, Gini **0.71**, tail ratio **0.385** — the
storm regime's gradient field is far *less* concentrated (a smaller
handful of cells carrying most of the gradient in calm regions; a more
uniformly elevated field in storms) than the calm regime's, a genuinely
different **shape**, not just a different mean. All three deviate from 1
by ≥ 20 %. **The regimes differ in structure, not merely in `\|∇p\|`** —
the assumption behind the whole regime ladder holds.

*(The 20 % threshold is this run's own operationalization of §3's
qualitative bar — the plan did not pre-register a number. Flagged as an
author choice; the raw ratios are reported above so a reader can apply a
stricter one and reach the same conclusion — the smallest deviation,
12 %, is on decay length, but tail ratio's 61 % deviation alone clears
any reasonable bar.)*

**C1 (ladder) and C1b (separation) both PASS too**: the `\|∇p\|` order
R1 < R2 < R3 < R4 holds at all three tested timesteps, and separation is
**5.87–8.24** at every one — comfortably above the ≥ 3 bar.

### §7.2 C2 PASSES both halves

Dynamic arms (`CAL-RANK`, `CAL-FISHERZ`) show `L[D][T] = 0` **exactly**
for every donor in every regime — no leak. Through the *identical* code
path, `CAL-ABS`'s off-diagonal `L` ranges **0.011 (R1) → 0.947 (R4)** —
demonstrably non-degenerate. The can-it-DIFFER gate is satisfied.

### §7.3 ⚠ C3 FAILS — and the failure is a clean, monotonic REVERSAL

`L̄[T]`, the mean off-diagonal transfer loss for `CAL-ABS`:

| R1 CALM | R2 OCEAN | R3 ACTIVE | R4 STORM |
|---|---|---|---|
| **0.011** | 0.309 | 0.671 | **0.690** |

The bar required `L̄[R4] < L̄[R1]`. **Measured: `L̄[R4]` is 62× larger than
`L̄[R1]`, and the increase is monotonic across all four tiers.** This is
not "no shrinkage" — it is the pre-registered relationship holding in
**reverse**: transfer loss under a foreign absolute calibration grows,
not shrinks, as turbulence rises. Storms are **less** forgiving of bad
calibration on this data, not more.

### §7.4 ⚠ C4 FAILS — same direction, same shape

`Δ[T] = RMSE(CAL-RANK, own window) − RMSE(CAL-ABS, own diagonal)`, in Pa
(positive = absolute wins):

| R1 CALM | R2 OCEAN | R3 ACTIVE | R4 STORM |
|---|---|---|---|
| +1.29 | +1.11 | +0.46 | **+10.78** |

**No sign flip anywhere** — absolute wins its own diagonal in all four
regimes, and its margin of victory is **largest in the storm regime**,
not smallest. The strong form of the operator's hypothesis is refuted on
this data: a well-calibrated absolute encoding does not lose its edge as
turbulence rises here — it gains one.

The **weak form** (`ρ(CAL-RANK) > ρ(CAL-ABS, D ≠ T)`) passes trivially
everywhere, exactly as pre-registered it would — dynamic beats a
genuinely *mis-calibrated* absolute by **329–2508 Pa** across regimes.
Per §3: *"nearly guaranteed by C2(i)… evidence of wiring, not of merit."*
Reported, not conflated with C4's real bar.

### §7.5 C5 — NOT RUN, structural blocker

`GEO-GOLDEN-HI` needs `N ≥ F(17)² = 2 550 409` before the golden lattice
behaves like the irrational it approximates (§1's own index-floor rule).
A 16° box at 0.25° resolution holds `65 × 65 = 4225` cells — **three
orders of magnitude below the floor**. `GEO-GOLDEN-HI` cannot be
constructed at box scale on this grid; the comparison has no admissible
high arm. Reported as a structural gap, not skipped silently, and not
faked with an interpolated sub-grid lattice (which would sample the
interpolator, not the field).

### §7.6 C6 — the transfer matrix, delivered

Full `donor × target` table for all five arms (`CAL-ABS`, `CAL-RANK`,
`CAL-FISHERZ`, `CAL-SHUFFLE`, `GEO-DEGENERATE`) × four regimes, every
`ρ` / `L` / `occupancy` / `saturation` / RMSE / bias cell raw, in
`substrate_comfort_d_cz_2_7.json` → `C6_matrix`.

### §7.7 ⚠ This CONTRADICTS §6.6's exploratory hint — stated plainly

§6.6 measured, *within* R4 only, ρ(`GEO-DEGENERATE` ρ, storm `\|∇p\|`) =
+0.444 (p = 0.058, not significant) — the *direction* of that correlation
matched the operator's hypothesis, and was recorded as a coherence with
the D-CZ-1 saturation numbers.

**The properly-powered, cross-regime test says the opposite.** §6.6 was a
single-regime, single-arm, unpre-registered correlation at n=19. §7.3/§7.4
are the pre-registered, cross-swap, equal-budget bars the whole plan was
built to produce. Where they disagree, the pre-registered cross-regime
result governs. §6.6 is retained as what it always was — exploratory, not
a result — and this section exists so a later read does not average the
two into a false middle. **The hypothesis is refuted on this data.**

### §7.8 What this plan concludes

The operator's hypothesis — *a badly-calibrated substrate that maps
DYNAMICALLY preserves MORE structure in strong storms than a
well-calibrated absolute one* — is **refuted, cleanly, on this box, this
variable, these three timesteps plus 19 storms**. Both pre-registered
measures of it (C3's transfer loss, C4's diagonal crossover) point the
same direction, monotonically, not merely "no effect."

**What is NOT concluded:** that miscalibration is never more forgivable
under turbulence, anywhere, on any variable, at any scale — this is one
box size, one variable (MSLP), one gradient definition, and C5's absence
means the geometry axis (the *other* half of "good geometry vs badly
calibrated") was never tested at all. The regime axis and the C0/C1/C1b/
C1c apparatus all hold; the specific calibration hypothesis they were
built to test does not.


### §7.9 ⚠ CONFOUND FOUND POST-RUN — C3's monotonicity is largely RANGE, not turbulence

**Trigger (operator, 2026-08-12):** *do we have a problem with the storm
modelling, since we do not actively model vortices?* The question is
correct, and chasing it surfaced a confound that qualifies §7.3 and §7.4.

**Fact first:** `substrate_comfort_d_cz_2_7.py` loads **only**
`mean_sea_level_pressure`. No wind, no vorticity, no rotation. D-CZ-0/1
loaded 10 m winds solely to *report* `spd_sigma`; no bar ever consumed
them. R4 "STORM" is therefore characterised by a **scalar pressure-gradient
magnitude**, not by the rotating structure that makes a storm a storm.

**The confound that follows, measured:**

| relationship | statistic |
|---|---|
| `L` vs the cell's `saturation` | Pearson **+0.917**, Spearman **+0.833** (n=8) |
| `L̄[T]` vs the regime's OWN value range | **Spearman +1.000 — perfectly monotone** |

`L` is essentially a function of **how much of the target's distribution
falls outside the donor's codebook range**. A regime's own range is what
makes it hard to cover: R4's implied range is ~**18×** R1's (≈7075 Pa vs
≈386 Pa). So `L̄` rising monotonically R1→R4 substantially **restates the
regimes' width ordering**, which the `|∇p|` ladder itself produced (a
deeper low means a steeper gradient *and* a wider box range).

Width alone is not the whole mechanism — `R3 → R2` has a *wider* donor
(log₂ ratio +1.32) yet saturation 0.949, because the two boxes sit at
different absolute pressure levels. The true driver is **coverage**
(width **and** offset), which `saturation` captures directly and which the
+0.917 correlation measures.

**C4 is range-inflated the same way.** `Δ` was amended to RMSE **in Pa**
(§6.4), and RMSE scales with the field's range, so an absolute-Pa margin is
not comparable across regimes differing 18× in range. Normalised as a
ratio, C4 reads **3.96 / 1.85 / 1.14 / 2.35** (R1→R4) — **not monotone**,
and R1, not R4, is where absolute wins by the largest factor.

**What survives, and what does not:**

- **SURVIVES — the hypothesis is NOT SUPPORTED.** There is no sign flip in
  either metric, normalised or not; `CAL-ABS` wins its own diagonal in all
  four regimes. That conclusion is unaffected by the confound.
- **DOES NOT SURVIVE — "cleanly reversed, monotonic."** §7.3's monotone
  `L̄` increase is confounded with range (ρ = 1.000), and §7.4's "margin
  grows to 10.78 Pa" is range-inflated. **"Storms are LESS forgiving of bad
  calibration" is withdrawn as a physical claim** — as measured it says
  *wide-range boxes are harder to cover with a foreign codebook*, which is
  arithmetic.
- **C1c still passes** and still licenses the design — but note what it
  measured: structure of the **`|∇p|` field**, not rotational structure.
  It cannot have caught this, because the confound is between the
  ladder's own discriminator and the codebook's coverage, not between two
  regimes.

**The deeper gap this exposes.** The hypothesis was about *"high velocity
differences / turbulence"* — a **dynamical** property. Every bar in this
plan is computed on a **scalar pressure field**. A vortex is not
represented anywhere: not in the regime definition, not in the codebook,
not in any metric. So this plan has tested *"badly-calibrated absolute vs
window-local encoding of a scalar field whose range varies by regime"* —
which is a real question, and answered — but it is **not** the operator's
original question about turbulence, and should stop being described as if
it were.

**Consequence for the next run:** a genuine turbulence regime needs a
**rotational** discriminator (relative vorticity ζ = ∂v/∂x − ∂u/∂y, or
Okubo–Weiss), and a range-matched or range-normalised transfer metric so
coverage cannot masquerade as the finding. Both are pre-conditions for
re-asking C3/C4 as a turbulence question rather than a width question.
