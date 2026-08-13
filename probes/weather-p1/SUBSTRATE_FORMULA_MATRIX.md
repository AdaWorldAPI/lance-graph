# Substrate Formula Matrix — the weather-physics arc, rated

> **What this is.** Every formula, encoding, sampling geometry, physical
> model and statistical instrument this arc actually put under a
> pre-registered bar, with its measured verdict, its comfort zone, and the
> known-effect-vs-discovered-explanation pairing side by side.
>
> **Scope.** PRs #920–#946, probes in this directory, plans
> `weather-substrate-{evaluation,poc-v1,poc-v2}-v1`, `weather-w-probes-v1`,
> `golden-vs-tempered-stride-v1`, `substrate-comfort-zones-v1`, and the
> `COMET_TAIL_REPORT.md`. Built 2026-08-12 by re-extracting from the
> committed artifacts — **not** from session memory, per
> `E-A-FIGURE-CITED-TWICE-IS-NOT-CONFIRMED-ONCE-1`.
>
> **Provenance rule.** Every figure traces to a committed JSON, plan RUN
> section, or merged PR. Figures with no committed source are marked
> UNBACKED and carry **no rating**.

---

## §0 UPDATE 2026-08-13 — what changed, and what it costs the ratings

> The document below was built 2026-08-12 against PRs #920–#946. Since then
> PRs **#947–#950** landed and moved four things. This block is the delta;
> the sections below carry the detail. Same provenance rule throughout —
> every figure here was **recomputed from the committed JSON in the same
> command that printed it**, not transcribed (see the new apparatus lesson
> A10, which exists because that rule was broken once on this very update).

**1. The cross-swap matrix is no longer a gap — it RAN, and the hypothesis
lost.** §5 listed *"no off-diagonal cell has ever been computed"* as the
arc's largest hole. D-CZ-2..7 (PR #947) computed them.

- **C1c PASSES** — the regimes differ in correlation *structure*, not only in
  gradient magnitude. R4/R1 ratios: decay length **0.882**, Gini **0.709**,
  tail ratio **0.385** (all ≥ 20 % from 1, as pre-registered).
- **C3 and C4 both FAIL, and reverse.** `L̄` per regime is
  **0.011 / 0.309 / 0.671 / 0.690** (R1→R4), monotone increasing, **62.5×**
  from calm to storm — the bar required storm to be *lower*. C4's RMSE delta
  is **+1.29 / +1.11 / +0.46 / +10.78 Pa**: no sign flip, and the absolute
  encoding's margin *grows* under turbulence.
- **⚠ but the ladder is confounded** — see item 2.

**2. The regime ladder measured RANGE, not turbulence — the arc's most
consequential correction.** `L̄` tracks each regime's own value width and
saturation almost perfectly (regime-level Pearson **+0.921**, and see plan
`substrate-comfort-zones-v1.md` §7.9 for the *cell*-level figures the
confound argument actually rests on — a different and larger population than
the four regime means recomputed here). R4's off-diagonal saturation is
**0.969** against R1's **0.667**. **Consequence for the ratings below:** the
regime ladder (G-tier rows) is demoted from "a turbulence axis" to "a range
axis that correlates with turbulence." Any row whose evidence is *"it behaved
differently across R1→R4"* now carries that qualifier. The withdrawn claim is
"monotone reversal proves storms are less forgiving"; what survives is "the
hypothesis is not supported."

**3. A real substrate exists now, and its codec passed the gate.**
`crates/weather-poc` (workspace-EXCLUDED, zero-dep) shipped key codec, floor,
manifest and L4 lane; **`D-WXS-7` (bar B6) passed 12/12** across three real
seasons on live-fetched grid data, computed with `jc::reliability::spearman`.
The shuffled control collapses to **0.020–0.024**. This is the first
**A-tier, grid-scale, real-data** rating the arc has for the quantiser.
`D-WXS-8` is mixed — its KILL-gated control holds **19/19**, its strict
primary fails **10 of 19** cross-unit pairs (see §5).

**4. Two silent-corruption paths were found in the shipped codec, by review
and by following review into its class.** Neither is a rating change; both
are apparatus (A9) and both were live on real data, not hypothetical.

### Rating movements

| row / claim | was | now |
|---|---|---|
| the off-diagonal cross-swap matrix | **NOT RUN** (§5's largest gap) | **RUN** — C1c pass, C2 pass, C3+C4 fail-and-reverse |
| the regime ladder as a *turbulence* axis | implicit `[H]` | **demoted** — range-confounded, `[G]` only as a *range* axis |
| the 256-level linear floor, grid-scale fidelity | untested at scale | **A `[G]`** — 12/12, 3 real seasons, real data |
| shared canonical floor beats per-variable, cross-unit | `[H]` (1 timestep, 3 vars) | **`[G]` directionally** — 19/19 control, 3 seasons, 19 cross-unit pairs; *exact* thresholds NOT universal |
| Morton vs the shipped whole-byte key layout | not a question yet | **closed by KILL** — no unambiguous win in key space; no migration |

---

## How to read the ratings

Two scales, deliberately **not** merged into one score.

**Fitness verdict — what the measurement said about *using* it:**

| tier | meaning |
|---|---|
| **A — proven in zone** | cleared its pre-registered bar, with a control demonstrably able to lose |
| **B — works, bounded** | reliable inside named limits |
| **C — comfort-zone mapped** | measured excellent in one regime and bad in another; the map *is* the result |
| **D — refuted at test point** | failed its own pre-registered bar |
| **V — VOID** | the apparatus could not measure it — says nothing about truth |

**Evidence grade** (workspace convention): `[G]` measured/proven ·
`[H]` bounded but open · `[S]` speculative/analogy-only.

**A "C" is not a bad grade.** The founding result of this arc is that most
substrate formulas are neither good nor bad — they have *homes*. Fisher-z
alone carries three different verdicts depending on what it is asked to do.
This matrix exists to read the homes off, not to crown a winner.

**A "D" and a "V" are different in kind.** D means the claim was tested and
lost. V means the test could not distinguish anything — a control matched
the real arm, or an upstream gate failed. **V is not a weak D**; it carries
no information about the hypothesis at all.

---

## §1 MASTER MATRIX

### 1a. Physical / meteorological models

| # | Formula / model | Verdict | Grade | Measured | Comfort zone |
|---|---|---|---|---|---|
| P1 | **wn-1 dipole spine fit** — ring-mean profile + single wavenumber-1 residual mode; constrained 2-param form = 14 logical values | **A** | `[G]` | R² **0.943 / 0.909** (2 storms) vs ring-only **0.635 / 0.294**; ~89–92 % of azimuthal residual is one mode; replicated over 41+ storms, 3 samples, 1980–2021, four seasons | MSLP in-disk structure, R=1200 km, extratropical |
| P2 | **Geostrophic steering → low pole 90° left of motion (NH)** | **D** | `[S]` | n=2 pilot PASS (−42.0°, −40.2°); **n=10 blind: 6/10 = 0.60, indistinguishable from a coin flip**; CT-F14 n=19: 13/19 = 0.684, p=0.0835, below its own n≥20 floor | structure yes, **sign no** at scale |
| P3 | **Systematic offset is a rotation, not noise** (circular resultant estimate) | **A** | `[G]` | μ = **−30.2° ± 36.5°**, R̄ 0.516, Rayleigh **p = 0.0050** (n=19, same sample, post-hoc) | estimating the offset, not proving direction |
| P4 | **Ekman/surface-friction cross-isobar inflow as the offset's cause** | **D** (bounded contributor) | `[G]` measurement, `[H]` attribution | median α **+14.7°** (ocean) / **+22.0°** (46 % land), ocean-only +13.0° — vs the ~40° needed | can own **at most ~⅓** of the offset |
| P5 | **Baroclinic tilt → steering level should score better** | **D** | `[G]` | F16a **0.579** (worse than surface 0.684), p=0.324; F16b spread **68.29° → 87.71°, +28.4 % WIDER**; level sweep monotone toward the **surface**, best 850 hPa | the leading mechanistic rescue is dead |
| P6 | **W6 vector-sum dipole** `D = c_geo·P_geo + c_bow·P_bow` | **V** | — | single-geo R² **−0.104**; permuted control **−0.071**, rotated-90° **−0.062** — both beat the real predictor's ceiling (−0.074). Joint margin +0.018 vs +0.10 required | **VOID** — no signal for a joint fit to identify |
| P7 | **Rankine vortex profile + hemispheric rotation asymmetry, on bitboards** | **A** | `[G]` | NH lows CCW **0.636**, SH lows **0.402** (inversion control fires), NH highs **0.151** vs SH highs **0.719** (mirrored); Rankine rise **2.17 m/s** → peak ring 3 → decay **11.20 m/s** | popcount physics over threshold masks |
| P8 | **Go-territory: overlapping influence tessellation predicts fronts** | **D** (both halves, incl. controls) | — | A-E1 **0.523** (bar 0.55); random-center control **0.406** — not storm-specific; B-E1 **inverted**: contested \|∇T\| **lower** than secured (ratio 0.77–0.83 vs ≥1.4 required) | refuted as specified |

### 1b. Encodings / codecs

| # | Encoding | Verdict | Grade | Measured (Pa unless noted) | Comfort zone |
|---|---|---|---|---|---|
| E1 | **Register A — affine against a fixed reference** (absolute anchor) | **C** | `[G]` | flat by construction: **10.71 / 10.76 / 10.79 / 10.71** across storm-tail / shoulder / bulk / high-tail | the control; wins the tail, never the bulk |
| E2 | **Register B — percentile rank → palette256** (equal probability mass) | **C** | `[G]` | bulk **2.33** (4.6× tighter than A) but storm tail **204.54** (**19× worse** than A) | bulk/typical values only |
| E3 | **Register C — Fisher-z of rank** (resolution moved into the tails) | **C** | `[G]` | storm tail **24.74** = **8.3× tighter than B**; high tail **6.86**; bulk **16.49** (7× worse than B) | **tail reads** — "a storm IS a tail event" |
| E4 | **Fisher-z as an L4 ring/level codebook** | **D** | `[G]` | pooled ring RMSE **18.07 vs uniform 3.84** → **4.7× worse**; post-hoc rank-vs-field rescue **19.00** — also fails | refuted in the interpolate/level role |
| E5 | **Fisher-z buckets in the CI-vs-floor frame (2 m temperature)** | **D** (not a win) | `[G]` | saturation 0.820 % vs linear 0.848 % (indistinguishable); at a 0.25 K floor **95.65 %** of interior points sit in buckets whose CI exceeds it; linear CI constant **0.0941 K** | tail resolution bought with wide bulk buckets |
| E6 | **Shared canonical floor vs per-variable floors** (cross-variable comparability) | **A** | `[G]` | t2m×u10 **0.9997** shared vs **0.875** per-variable; td2m×u10 **0.9997 vs 0.857**; t2m×td2m 0.99956 (misses the 0.9996 target) | rank-normalisation is what makes cross-variable distance **exist** |
| E7 | **Same u8 = same rarity across variables** (R5 bar) | **A** | `[G]` | max spread across all 256 bytes **0.000434** vs one bucket **0.003906** (MSLP / t2m / wind10m); for the absolute register the comparison is **undefined** (Pa, K, m/s share no unit) | the "statistical gold" property |
| E8 | **palette256 LINEAR bucket layout vs uniform-SD** | **B** | `[G]` | MAE **0.068 vs 0.217 K**; effective buckets **115.7 vs 28.1**; empty buckets **0 vs 76** | layout choice is not cosmetic |
| E9 | **bf16 on anomaly vs bf16 on raw Kelvin** | **B** | `[G]` | MAE **0.00609 vs 0.456 K** → **74.9×**; anomaly σ 6.75 K vs raw σ 22.65 K | subtract the climatology before quantising |
| E10 | **12-byte V3 facet carve D** (dipole rail + 10 ring bytes, 2 interpolated) | **B** | `[G]` | R² **0.94340 / 0.90903** vs f64 **0.94344 / 0.90905**; RMSE 241.71 vs 241.64; bias **+1.59 Pa** — recovery, **not lossless** | capacity was the miss, precision was free |
| E11 | **Carve A** (dipole rail + rings 0–9, outer rings held) | **D** | `[G]` | R² **0.9129 / 0.9018**, bias **+92.76 Pa**, loss 0.0306 vs the 0.02 bar | dropping outer rings costs real Pa |
| E12 | **Carve B** (all 12 rings, no dipole rail) | **D** | `[G]` | R² **0.635 / 0.294** — collapses to the axisymmetric model | the dipole rail is load-bearing |
| E13 | **Global vs per-storm codebook** | **A** | `[G]` | storm-2's codebook on storm 1: **620.79 Pa** vs shared **4.48 Pa** → **139× penalty** | one table read, globally — the carrier's whole point |
| E14 | **CAL-ABS (own-calibration absolute) vs CAL-RANK (window-local)** | **C** | `[G]` | ρ both **> 0.99996** (indistinguishable); RMSE ratio **3.96 / 1.85 / 1.14 / 1.49** across calm→storm | absolute wins RMSE; margin **shrinks** as the field activates |
| E15 | **256-level linear floor** (`weather-poc::floor`, percentile-trimmed 0.4–99.6, re-expressed from `helix::RollingFloor`) — **grid-scale fidelity** | **A** | `[G]` | bar B6, **12/12** across 3 real seasons on live-fetched ERA5: ρ(L256) **0.999909 / 0.999895 / 0.999684**; ladder strictly monotone (L16→L64→L256) every season; shuffled-decode control collapses to **0.020–0.024** | the arc's first A-tier real-data rating for the quantiser. **Supersedes** the fixture-scale near-miss (K×K 0.999556) — that did **not** replicate at grid scale |
| E16 | **Shared canonical floor vs per-variable floors, cross-unit** | **A** (direction) / **C** (exact threshold) | `[G]` | control **19/19** — per-variable loses on *every* cross-unit pair, ρ_pervar **0.245–0.939** vs ρ_shared **0.9987–0.9999**. But the strict ρ ≥ 0.9996 primary passes only **9/19** (winter 2/9, spring 4/5, summer 3/5) | the KILL does **not** fire ⇒ the shared-floor design stands. The *exact* bar does not hold universally — failures cluster in winter, the only season carrying `mean_sea_level_pressure` |
| E17 | **"Zero empty buckets" under a shared floor** (carried verbatim from a 1-timestep/3-variable fixture) | **D** | `[G]` | fails at **all three** seasons: **38 / 39 / 45** of 256 buckets empty (15–18 %) | a percentile-trimmed *pooled* window necessarily leaves slack for any one variable's narrower spread. Direction unsurprising; the literal "zero" had simply never been re-verified at scale |
| E18 | **L4 `6×(8:8)` lane** — pack/unpack one 16-byte facet against a ClassView-side manifest (`weather-poc::lane`) | **A** (as built) | `[G]` | 34 crate tests; 4 disable-verified bars (lo/hi swap, hard-coded slot, version-guard bypass, unmapped-slot emission) each kill exactly the expected tests | slot purity as *code*: the lane names no ERA5 variable in its own source. **Not** a fidelity claim — that is E15/E16 |
| E19 | **`quantize` on non-finite input** (the codec's own total-function behaviour) | **D — hazard confirmed** | `[G]` | `NaN` → bucket **0**, `-inf` → **0**, `+inf` → **255**; `f64::clamp` *propagates* NaN, then the float→int cast saturates. All three are legitimate buckets, indistinguishable from a real reading | **live, not hypothetical**: ARCO-ERA5 404 = all-NaN = valid store semantics, and 5 W1-set variables 404 at the arc's own fixture timestep. Guard at the boundary, never in the hot primitive (A9) |

### 1c. Sampling geometries / address generators

| # | Geometry | Verdict | Grade | Measured | Comfort zone |
|---|---|---|---|---|---|
| G1 | **Golden-ratio index floor** — φ-behaviour only from convergent index ≈ 17–21 | **A** | `[G]` | convergent error: n=10 **1.5e-4**, n=13 **8.2e-6**, n=17 **1.8e-7**, n=21 **3.7e-9**. Binds the **emergent parastichy stride** (≈ √N) ⇒ needs **N ≳ F(17)² = 2 550 409** | a design gate, not a preference |
| G2 | **Tempered coprime stride** (closes exactly at m=q) | **A** | `[G]` | fills **q/q always** (coprimality ⇒ bijection, proof); competitive inside the budget | **bounded budget**, m ≤ q — byte rails, palette indices |
| G3 | **Golden angle** (irrational, never repeats) | **A** | `[G]` | golden fills only **124–127 / 140** at q=140 (13–16 empty cells); but at m=200q it is **68.2–106.4× ahead** at all 10 tested q | **unbounded budget**, m ≫ q |
| G4 | **The crossover between them** | **A** | `[G]` | verified-permanent m\* = **1.9–2.7 × q** (corrected twice, both times *away* from q; the superseded table said 1.0–1.4×) | golden needs ≈ **two tempered cycles** to win |
| G5 | **Naive golden stride** `s = round(frac·q)` without a coprimality check | **D** (hazard confirmed) | `[G]` | **114 / 292 = 39.0 %** of q ∈ [8,300) collapse (gcd > 1 ⇒ only q/g cells reached); bar was 25 % | never do this — a coprime search cannot collapse |
| G6 | **"The golden step is the best step" at small q** | **D** | `[G]` | at q=17, stride **4** beats the φ-derived stride **11**: star discrepancy **0.2000 / 0.1111 / 0.0769** vs **0.2000 / 0.1503 / 0.0905** at m = 5/9/13. `17/11 = 1.5455`, error 7.3e-2 vs φ — an order of magnitude worse than 13/8 | the shipped stride-4 walk was right; only its *rationale* was wrong |
| G7 | **Temperament reading** — a coprime stride IS a circle of fifths | **A** (mechanism) | `[G]` | 12 pure fifths miss closure by the **Pythagorean comma +23.46 ct**; the 17-TET fifth closes exactly, **+3.93 ct/fifth** spread. The distributed comma **is** D-QUANTGATE's anti-moiré dither | explains *why* closure beats goldenness below the floor |
| G8 | **Golden two-lattice pairing has even pair distances** | **D** | `[G]` | cv_golden **0.368** vs cv_grid **1.6e-12** at N=2 550 409 — the grid is even to twelve decimals and golden is not | refuted on **projected lat/lon**; the disk property did not transfer |
| G9 | **…and its no-ties half** | **V** | — | ties = **0 for both** constructions ⇒ the test cannot discriminate | VOID — a control that cannot lose |
| G10 | **Two Fibonacci-stride ADI sweeps ≈ one isotropic 2-D diffusion** | **D** | `[G]` | anisotropy **1.5251** vs isotropic baseline **1.0046** (bar 1.25); operator contributes ≈ **0.52** | refuted at N = 3·F(17)² = 7 651 227 |
| G11 | **…with its distance-matched shuffled control** | **V** | — | control anisotropy **1.5657**, ratio control/fib **1.027** — the control smooths **as isotropically as Fibonacci** | VOID — the Fibonacci structure adds nothing measurable *here* |
| G12 | **On a golden lattice, locality ⟺ Fibonacci membership** | **A** | `[G]` | **99.68 %** (famA) / **99.56 %** (famB) of qualifying control links land on pure Fibonacci offsets; dominant offsets **2584 = F(18)** (4 745 846 links) and **4181 = F(19)** (4 732 643), out of N = 7 651 227 | the three-distance theorem, made operational |
| G13 | **Golden-spiral sampling of a storm disk at fixed budget** | **B** | `[G]` | RMSE spiral / grid / random = **234.5 / 269.0 / 319.4** (N=64), **119.1 / 123.3 / 164.2** (N=256), **58.9 / 59.9 / 84.2** (N=1024) — spiral ≤ grid ≤ random at every budget | low-discrepancy sampling earns its keep |
| G14 | **…and its "ripple" claim** (spiral order gives a low-entropy 1-D signal) | **D** (inverted) | `[G]` | spiral-order Δ-entropy **7.128 bits** vs raster-order **5.947** — spiral order is **higher** entropy, the opposite of the claim | the sampling wins; the *ordering* story does not |
| G15 | **…and its axisymmetry premise** | **D** | `[G]` | E1 **0.639** vs the 0.70 bar; off-center control **0.005** (index can fail) | the 36 % azimuthal residual is exactly what P1's dipole then explains |
| G16 | **HEEL/HIP whole-byte-per-axis key** (`weather-poc::key`) — a lat/lon grid as the literal-x/y case of the 3×4 cascade | **A** | `[G]` | exhaustive **1 038 240**-cell round-trip, collision-free; a tile-aligned box is **exactly 1** contiguous range; ragged last tiles (721 = 11·64+17; 1440/64 = 22.5) round-trip without padding; seam-crossing box returns a range **set**, never one range | the arc's hand-picked 16° boxes **are** HEEL values — a box read is a prefix scan, zero value decode. Needs 5–6 of 12 available quaternary levels/axis |
| G17 | **Morton (nibble-interleaved) vs the shipped whole-byte layout** | **C — no unambiguous win** | `[G]` | Morton wins neighbour locality **16 vs 32** (2×) but loses range count **212.50 vs 140.00** on the median non-tile-aligned box. Bar required *both* ⇒ **KILL fires, no migration** | the shipped order is tile-row-major then row-major *within* a tile — already blocked, not flat row-major. Morton's win is on exactly the metric a ζ stencil would spend; that is a **prior for half B**, not a result |
| G18 | **CONTROL-BAD (axis bytes reversed)** — the locality control | **A** (as a control) | `[G]` | **3100** ranges and neighbour distance **15862**, vs 140/212 and 32/16 for the real arms | a control that can lose, and does, by ~20×. Without it the G17 comparison would be unreadable |

### 1d. Statistical instruments (the apparatus itself)

| # | Instrument | Verdict | Grade | Measured | Note |
|---|---|---|---|---|---|
| S1 | **Binomial sign-consistency fraction** | **D — RETIRED** | `[G]` | at n=19 only **14/19 (p=0.0318)** separates from chance; the arc's headline **0.684** was also scored by a deliberately **90°-rotated** referent; on a distribution centred at −103° it reports 0.833 | "the plateau was a property of the STATISTIC, not the data" |
| S2 | **Circular resultant** (R̄, μ, Rayleigh p) | **A** | `[G]` | clean hierarchy: real **R̄ 0.516 / p 0.0050** > structured-but-wrong **0.343 / p 0.107** > permuted **0.142 / p 0.688**; uniform floor R̄ ≈ 0.203 at n=19 | resolves rows the sign test saturated on |
| S3 | **Dual controls — permuted AND rotated** | **A** | `[G]` | fired exactly as designed in W6: both controls **out-predicted** the real predictor ⇒ VOID rather than a weak pass | the single highest-yield rule in the arc |
| S4 | **R² as the loss metric for a small offset** | **D** | `[G]` | computed via `var()` at 11 sites in 8 probes; **+92.76 Pa** moved R² by 0.0083 (3rd decimal), **+1.59 Pa** by 2.4e-06 (6th) | "lossless" was claimed from the one statistic that could not see the loss |
| S5 | **Spearman ρ as a diagonal discriminator** | **D** (blind here) | `[G]` | real-arm spread **3×10⁻⁶ … 4.7×10⁻⁵** — cannot separate CAL-ABS from CAL-RANK; but spans **0.99999 → 0.28** real-vs-degraded | right for transfer loss, blind for the crossover |
| S6 | **Star discrepancy over useful prefixes** | **A** | `[G]` | the only metric that separated stride 4 from stride 11 at q=17; three metrics pick **three different champions** — worst-case-over-all-m is dominated by the degenerate m=2 | always name the prefix range |
| S7 | **Author-written falsifiers** | **D** | `[G]` | 13-agent audit: 24 claims → **22 CONFIRMED, 2 PARTIAL**; but 11 specs → **10 VACUOUS, 1 UNDERSPECIFIED, zero SOUND**. The author went **0 for 11** | `E-ZERO-FOR-ELEVEN` — the spec author cannot audit his own falsifiers |
| S8 | **Curve shape as evidence** | **D** | `[G]` | EV-4's Fisher-z non-monotone CI curve moved with ε and interpolation choice, **method_max/min ratio 64.14** ⇒ APPARATUS-DOMINATED, retracted as a data finding | a shape is a claim about the apparatus until proven otherwise |
| S9 | **Append-only ledger audit by zero-deletions** | **D → replaced** | `[G]` | zero deletions proves only ADDITIVE, not PREPEND-ONLY; replaced by a **suffix check** (`new.endswith(old)`); an end-append scores `zero-del=True, suffix=False` | the audit must terminate at the property you actually want |

---

## §2 KNOWN EFFECT vs DISCOVERED EXPLANATION

The left column is prior art the arc did not invent and does not claim.
The right column is what measuring it **on this substrate, on real ERA5**
added — including where the measurement went **against** the prior.

| # | Known effect (prior art) | What the arc measured — and what changed |
|---|---|---|
| K1 | **Azimuthal wavenumber decomposition**; a linear background field has zero ring-mean and projects entirely onto cos(θ−θ₀) with amplitude ∝ r | The residual is not merely wn-1-*dominated*, it is **89–92 % one mode**, and **14 logical values** lift a storm from R² 0.29–0.63 to **0.91–0.94**. Amplitude-vs-radius correlation **0.800 / 0.998** confirms the linear-background signature. *New: the compression ratio, and that it survives 41+ storms across four seasons.* |
| K2 | **Geostrophy + steering flow** ⇒ background gradient ⊥ motion ⇒ low pole 90° left (NH) | **Holds as structure, fails as sign.** n=10 blind: **6/10**. The prediction's *direction* is not established at scale — but the failure is not noise: the offset is a **systematic −30.2° ± 36.5° rotation** (Rayleigh p **0.0050**). *New: the residual has a shape, and the shape was invisible until the instrument changed.* |
| K3 | **Ekman layer**: cross-isobar inflow ~20–45° over land, less over ocean | Measured **+14.7° / +22.0°** (ocean-only +13.0°) against the ~40° the offset would need. **Bounded to ≤ ⅓** of the effect. *New: a textbook mechanism quantitatively excluded as the primary cause, and later re-scoped — friction rotates the wind, not the pressure dipole, except second-order.* |
| K4 | **Baroclinic tilt**: extratropical cyclones are steered by mid-tropospheric flow, so a steering-level reference should score better than the surface | **The opposite, monotonically.** Level sweep 400→850 hPa gives sign fraction 0.579 → 0.684 and spread 89.5° → 77.0° — **best at the surface end**. The steering rescue made the claim **worse** (+28.4 % wider spread). *New: the most physically motivated fix available was measured and killed.* |
| K5 | **Weyl equidistribution**: irrational rotation has prefix discrepancy O(log m / m), improving without bound; a rational stride freezes at its closure value | Both halves confirmed — **and the crossover located**: golden does not win immediately, it needs **1.9–2.7 × q**. By m = 200q the gap is **68–106×**. *New: a number where there was a slogan; and the first draft's 1.0–1.4× was wrong in the flattering direction.* |
| K6 | **Coprimality ⇒ full permutation of Z/qZ** (cyclic-group order) | Reframed as **musical temperament**: the coprime stride *is* a circle of fifths — it closes exactly and distributes the incommensurability error uniformly. **12 pure fifths miss by +23.46 ct** (Pythagorean comma); 17-TET closes with **+3.93 ct/fifth**. *New: the distributed comma **is** D-QUANTGATE's anti-moiré dither — two workspace doctrines turned out to be one mechanism.* |
| K7 | **Three-distance theorem**: azimuthal gaps on a golden lattice take ≤ 3 values | Measured on a 7.65 M-point lattice: **locality ⟺ Fibonacci membership**. **99.68 % / 99.56 %** of qualifying links land on pure Fibonacci offsets, dominated by **F(18)=2584** and **F(19)=4181**. *New: "near" on a golden lattice is not a distance predicate, it is a **membership** predicate.* |
| K8 | **Continued-fraction convergents of φ** converge geometrically | Turned into a **binding design gate**: φ-behaviour needs index ≈ 17–21 (error 1.5e-4 → 1.8e-7 → 3.7e-9), and because the floor binds the *emergent parastichy stride* (≈ √N), a probe needs **N ≳ F(17)² = 2 550 409**. *New: two already-merged probe specs (W5 at N=4096, W2s-a at N=2048) were **six orders of magnitude sub-floor** and had to be re-specced.* |
| K9 | **Low-discrepancy sampling beats Monte-Carlo** (O(log N/N) vs O(1/√N)) | Confirmed on a real storm disk at equal budget: RMSE spiral ≤ grid ≤ random at **N=64, 256 and 1024**. *New: it transfers to a physical field, not just to integration.* |
| K10 | **"Golden is the most irrational number, so the golden step is the best step"** (folklore, widely repeated) | **Refuted in the quantized regime.** At q=17 the φ-derived stride 11 is **measurably the worse one**: discrepancy 0.1111 / 0.0769 (stride 4) vs 0.1503 / 0.0905 (stride 11) at m=9/13. *New: the shipped `CurveRuler` stride-4 walk was correct all along — only its in-tree **label** was wrong (filed as a doc defect, code untouched).* |
| K11 | **Fisher-z (arctanh) is the variance-stabilizing transform for correlation-like quantities** | **Three verdicts, one transform** — the arc's founding comfort-zone result. As a **rank-register**: **8.3× tighter** than plain rank in the storm tail (24.74 vs 204.54 Pa). As a **level/ring codebook**: **4.7× worse** than uniform (18.07 vs 3.84 Pa). In the **CI-vs-floor frame** on temperature: **not a win** (0.820 % vs 0.848 % saturation; 95.65 % interior-CI exceedance at a 0.25 K floor). *New: "which codec is better" is the wrong question — **what the read is for** is the discriminator.* |
| K12 | **Rank/percentile normalisation** puts unlike quantities on a common scale | Measured as the **licence for cross-variable distance to exist at all**: shared floor **0.9997** vs per-variable **0.857–0.875** on cross-variable pairs, and the same u8 denotes the same rarity across MSLP/t2m/wind10m to **0.000434 < 1/256**. For the absolute register the comparison is **undefined** — no shared unit. *New: "there is no absolute anchor" stops being a slogan and becomes a measured property.* |
| K13 | **Rankine vortex** (solid-body core, decaying outside) and hemispheric rotation asymmetry | Reproduced by **popcounts over threshold-mask bitboards** — and **survives u8 quantisation**: max popcount-fraction deviation raw→palette **0.00475**, every raw verdict reproduced. *New: the substrate claim, not the physics claim, is what this probe actually established.* |
| K14 | **Stockfish/NNUE**: `popcount(attacks & targets)` as the evaluation primitive; deterministic address + stored magnitude | Transfers as a **frame** (the E4 result above). Explicitly **fenced as rhymes, not proven**: the Walsh-Hadamard bipolar sign pyramid has **no NNUE analog**; Morton 2×2 tiling is **not** NNUE's king-bucket×piece addressing; palette256² is **not** how NNUE stores weights. *New: a graded transfer ledger instead of an analogy.* |

---

## §3 THE THREE LOAD-BEARING CARDS

Everything above is a row. These three are the ones a product decision
would actually turn on.

### Card 1 — The storm spine (P1 / E10): **the arc's one durable win**

**What it is.** A surface low's in-disk MSLP field compresses to a centre
address plus **14 logical values** — ~12 ring-profile means and a 2-value
wavenumber-1 dipole (amplitude slope + bearing).

**Measured.** R² **0.943 / 0.909** against **0.635 / 0.294** for the
axisymmetric model alone. Replicated across three independent samples,
41+ storms, 1980–2021, four seasons — *"never shaken once."*

**Carrier fit.** It lands in a **12-byte V3 facet** as `6×(8:8)`
palette256 pairs: carve D reproduces f64 to R² **0.94340 vs 0.94344**,
with a **+1.59 Pa** bias — *recovery, not lossless*. Dropping the outer
rings (carve A) costs **+92.76 Pa**; dropping the dipole rail (carve B)
collapses to **0.635 / 0.294**.

**The correction that matters.** The published compression figure was
**measured on a model nobody claimed** — `decompose()` fit a dipole
*per ring* (36 params), not the 2 claimed. Caught by external review
(Codex P1 on #926): *"every 93–97 % in this document is an overstatement
of ~2.5× in parameter count."* The corrected 90.9–94.3 % is the number of
record. **The structural finding survived the correction; the headline did
not.**

**Honest limits.** MSLP only; structure, **not forecast skill**; planar
cos(lat) geometry degrades toward the pole; rings 10–11 do not fit the
12-byte facet (*reported, not engineered away*).

### Card 2 — Fisher-z (E3/E4/E5): **the comfort-zone archetype**

One transform, **three measured verdicts**, and they disagree:

| role | result | figure |
|---|---|---|
| rank register, **tail** read | **wins decisively** | 24.74 vs 204.54 Pa (**8.3×**) |
| ring/level codebook, **interpolate** read | **loses decisively** | 18.07 vs 3.84 Pa (**4.7× worse**) |
| bucket-CI vs noise-floor frame (temperature) | **not a win** | 0.820 % vs 0.848 % saturation; **95.65 %** interior-CI exceedance at 0.25 K |

**The design rule this produces:** *the discriminator is what the read is
for, not which codec is better.* A per-class analytic codebook chosen **by
measurement** — never one axis declared canonical.

**Two corrections rode along.** The "5× worse" first written was
**4.71×** — a 6 % overstatement in the favourable direction, nearly frozen
into an append-only ledger. And a post-hoc rescue (ranks against the field
rather than encoded values) was measured at **19.00 Pa** and **also
failed** — recorded rather than dropped.

### Card 3 — Two regimes, two generators (G2/G3/G4/G6/G7)

**The rule, now measured rather than asserted:**

| regime | generator | why |
|---|---|---|
| **bounded budget**, m ≤ q — byte rails, palette indices, facet slots | **tempered coprime stride** | fills **q/q by proof**; competitive inside the budget; a coprime search **cannot** collapse |
| **unbounded budget**, m ≫ q — continuum lattices, real phyllotaxis | **golden angle** | no ceiling: **68–106× ahead** by m = 200q |

**The crossover is at 1.9–2.7 × q** — golden needs roughly **two tempered
cycles** to overtake. Below that, closure beats irrationality.

**Three traps, all measured:**
1. **Naive rounding collapses 39.0 % of the time** (114/292 moduli) — `s = round(frac·q)` without a gcd check reaches only q/gcd cells.
2. **At q=17 the φ-derived stride is the worse one** (stride 11 loses to stride 4 at m=9 and m=13).
3. **Below the index floor a Fibonacci ratio shows the *opposite* of incommensurability** — resonance/moiré, not evenness. The floor is ≈ index 17–21, i.e. **N ≳ 2 550 409**.

---

## §4 THE APPARATUS LESSONS (why these ratings are trustworthy at all)

Thirteen rules — nine from the probe half, four added 2026-08-13 — each
bought with a measured failure in this arc. They are the reason the D's and
V's above can be believed alongside the A's.

1. **A control that cannot lose is no control** — and one that cannot
   *differ* is the same defect. W2s-a's no-ties test: ties = 0 for **both**
   constructions → VOID. W5's shuffled control smoothed **as isotropically
   as Fibonacci** (ratio 1.027) → VOID.
2. **Dual controls, permuted AND rotated.** In W6 both controls
   **out-predicted** the real predictor (−0.071 and −0.062 vs −0.104),
   turning a weak-looking result into an honest VOID.
3. **A statistic must be able to fail.** A deliberately **90°-rotated**
   referent scored **0.684** — numerically identical to the arc's headline.
   The sign test was retired as verdict-grade on that evidence.
4. **The metric that separates one contrast is blind to another.** ρ spans
   0.99999 → 0.28 for real-vs-degraded and **3×10⁻⁶** for real-vs-real.
   Pick the metric per **contrast**, never per plan.
5. **A shape is a claim about the apparatus until proven otherwise.**
   EV-4's non-monotone curve moved with ε and interpolation choice
   (method max/min ratio **64.14**) → retracted as a data finding.
6. **Rounding hides failures.** 0.999556 fell below a 0.9996 bar and
   rounding hid it; "5×" was really 4.71×. Gate constants full-precision;
   3-sig-fig figures are prose.
7. **The author cannot audit his own falsifiers.** 13 agents: source claims
   **22/24 CONFIRMED**, but specs **0 of 11 SOUND** — written by the same
   author who had just been corrected for exactly those failure modes,
   citing the falsifiability rule throughout.
8. **An audit must terminate at an artifact.** Comparing a summary to a
   plan compares prose to prose. Zero-deletions proves only *additive*;
   the property wanted was *prepend-only*, which needs a **suffix check**.
9. **Check the sample's arithmetic against the new question first.** W6's
   stranded stratum was **structurally empty**: `displacement ≥ 250 km/6 h`
   implies `|v| ≥ 11.57 m/s`, so a `|v| < 8 m/s` stratum could never exist.

**Added 2026-08-13 — four more, each bought the same way:**

10. **A figure you tallied yourself is a DERIVED figure.** Every number the
    D-WXS-8 runner *computed and printed* was carried correctly; the only two
    wrong ones were tallied by **counting rows in terminal output by eye**
    (`10/16` and `16/16`, actually **9/19** and **19/19** — and the direction
    mattered: the strict bar fails on a *majority* of pairs, not a minority).
    Rule now in force: **any count, rate or ratio in a writeup is computed
    from the artifact in the same command that prints it.** This is the
    narrow form of lesson 8 — that one is about *citing* a number, this one
    is about *producing* one.
11. **A disable probe can itself be vacuous.** Three verification probes in
    one session did not probe: a wrong symbol name (the substitution aborted
    and the run reported green on **unmodified** code — the dangerous one), a
    dead-code insert that bound nothing, and a probe aimed at a path the loop
    could never reach. **Signature: a disable that kills ZERO tests is more
    likely a broken probe than a missing guard.** Every substitution now
    asserts it applied.
12. **A total function that cannot refuse is a corruption path — and it is
    worse in an INSTRUMENT than in a store.** `quantize` maps every input to a
    valid-looking bucket (E19). The store-side hole (`pack_facet`) writes one
    wrong row; the instrument-side hole (`saturation_of`, bar B2's own
    measuring device) would have scored an all-NaN field as **1.0 = "fully
    saturated"** when the truth is "no data". A bad value is one wrong row; a
    bad instrument is every conclusion drawn with it. **When a finding lands
    on a total function, check its measurement call sites before its storage
    call sites.** Silently *dropping* invalid input is the other half of the
    same mistake — report it (`SaturationScore {fraction, finite, non_finite}`).
13. **A blocker is an artifact-checkable claim like any other.** `D-WXS-7`
    was reported as gated behind the classid mint **four times** without once
    being checked against its own bar. It never was — bar B6 needs real
    values, the shipped quantiser, a control and a Spearman; no Lance dataset,
    no classid. "On the substrate" described *scale and source*, not
    persistence. **An unverified blocker is more expensive than an unverified
    figure, because it prevents work rather than merely describing it wrongly.**

---

## §5 HONEST GAPS — what is NOT rated, and why

| gap | status |
|---|---|
| **CT-F17** — the fresh-sample directional verdict | **NOT RUN.** Gated on an independent adversarial spec audit. The directional claim (P2) stays *not established*. |
| ~~**The full cross-swap matrix** (comfort zones C2–C6)~~ | ✅ **CLOSED 2026-08-13 (PR #947).** C1c/C2 pass, C3+C4 fail-and-reverse, C6 matrix delivered. **But** see §0 item 2 — the ladder is range-confounded, so the matrix measures transfer under *width* mismatch, which is not the same question as transfer under *turbulence*. |
| ~~**CAL-FISHERZ arm**~~ | ✅ **CLOSED** — run in D-CZ-2..7 as a degenerate row (`L ≡ 0` by construction, same as CAL-RANK: levels re-derived in-window). Measured, and measured to be structurally uninformative for this comparison. |
| **Geometry axis** (GEO-GOLDEN-HI/LO, TEMPERED, GRID) | **STILL NOT RUN, and now known to be structurally unreachable at box scale.** C5's floor needs N ≥ F(17)² = **2 550 409**; a 16° box holds **4 225**. Even the *full global grid* (1 038 240) is short by **~2.5×**. Needs a different construction or a finer grid — not more budget. |
| **`D-WXS-8`'s strict primary bar** | **PARTIALLY MET, and reported as such.** 9/19 cross-unit pairs. The KILL-gated control (19/19) is what licenses the shared-floor design; the exact ρ ≥ 0.9996 threshold is **not** established universally. Anything needing the number *per pair* must treat this as open. |
| **Bar B6's ladder for cross-unit pairs** | **NOT RUN** — the 16/64/256 resolution ladder was computed for the K×K pair only. |
| **A 4th+ season / the wind-pressure skew** | **NOT RUN / OBSERVED-NOT-TESTED.** The primary failures cluster in winter and toward wind+pressure; that is a pattern seen, not a hypothesis tested. |
| **`D-WXS-2a` half B** (Morton vs shipped under the ζ stencil) | **NOT RUN**, gated on `D-WXS-9` → `D-WXS-0`. Half A closed by KILL; half B carries a stated prior (Morton ≈ 2× on locality) that is **a prior, not a result**. |
| **EV-1 … EV-10** | **ALL TEN NOT RUN.** Every v1 spec was ruled not-sound; the current specs are v2 rewrites awaiting execution. EV-5 is **blocked** — its fixture cannot exist at the pinned timestep. |
| **Five excluded land candidates** in the regime preflight | **UNREPRODUCIBLE.** Their box centres were never recorded anywhere. No coordinates were invented to fake the rows. |
| **W5 at n_idx = 19/21** | **NOT RUN** — iteration count scales ~2V/h², orders of magnitude more expensive. Mechanism stated, not silently dropped. |
| **`ValueTenant::HelixResidue`** | **Zero writers, zero decoders** tree-wide (exhaustive grep, verified twice). The helix-in-substrate story is unwired at the contract boundary. |
| **helix CI** | **No CI gate anywhere** — workspace-excluded, in no workflow. Every helix `[G]` rests on hand-run tests. |
| **BF16 in a prognostic path** | **Untested risk.** No conservation falsifier exists; no error budget quantified. |
| **`E-R²-IS-NEAR-BLIND`** | **Cited twice by id, never written as an entry.** The underlying finding is real (S4 above); the id is not. |
| **Claim C4** (beat a learned model) | **Explicitly out of scope**, quarters of cost. Not attempted, not planned. |
| Out-of-tree performance priors (`~125 ms compute / 233 ms disk`) | **Operator-reported, not in-tree.** Usable as priors, **never as citations**. |

---

## §6 READ-OFF: what to build on, what to stop building on

**Build on these** — measured, controlled, replicated:

- the **wn-1 spine** as a compression target (14 values, R² 0.91–0.94);
- **rank-normalisation onto a shared palette256** as the cross-variable
  frame (it is the only frame in which the distance is defined);
- **per-class analytic codebooks chosen by measurement**, with Fisher-z
  used for **tail reads only**;
- a **global** codebook, never per-storm (139× penalty);
- **tempered coprime strides for bounded budgets**, golden angle for
  unbounded ones, with the index floor treated as a hard gate;
- **circular statistics with dual controls** as the directional instrument.

**Stop building on these** — tested and lost:

- the **signed** left-of-motion prediction as a scoring referent (P2, and
  its steering-level rescue P5);
- the **W6 vector-sum dipole in its current form** (VOID by its own control);
- **Fisher-z as a general L4 axis** (4.7× worse on level reads);
- the **Go-territory tessellation** as specified (both halves, incl. controls);
- **binomial sign fractions** as verdict-grade statistics;
- **naive `round(frac·q)`** strides anywhere.

**The single most valuable thing this arc produced** is not on either list.
It is the **apparatus**: the rules in §4 — nine at first writing, **thirteen**
as of 2026-08-13 — each bought with a measured failure, which is why a "D" in
this document can be trusted as much as an "A". Most of the entries here are
negative results, and they were expensive to get right — an arc that only
reported its wins would have shipped the 36-parameter compression figure,
Fisher-z as the universal axis, the sign-test headline, a stride-11 "golden"
walk, a codec that writes missing data as plausible measurements, and an
instrument that scores an absent field as fully saturated.

---

## §7 NEXT STEPS (2026-08-13) — what to do, in order, and why

> Ordered by **what unblocks the most**, not by effort. Every item names its
> gate and its kill condition, because an item without one is a wish.

### The situation in one paragraph

The arc has turned from *probing a hypothesis in Python* to *a substrate with
a measured codec*. `crates/weather-poc` exists, is zero-dep, and its quantiser
passed the real gate at grid scale on real data (E15, 12/12). The
comfort-zone hypothesis that motivated the earlier half **lost** — cleanly,
with its own controls holding — and the ladder it lost on turned out to be
range-confounded, which is the more useful finding. What remains is not
"more measurement of the same thing"; it is **one blocked step that gates
everything downstream**, and a small number of genuinely open questions.

### Tier 1 — the one real blocker

**N1. `D-WXS-0` — mint the weather-cell + statics classids.** OGAR-side,
operator-gated; **cannot be resolved from this repo.**

- **Blocks:** `D-WXS-4` (the bake), `D-WXS-5` (statics), and transitively
  `D-WXS-6` (version-range read), `D-WXS-9`/`D-WXS-10` (ζ, the vorticity
  work), and `D-WXS-2a` half B.
- **Why it cannot be worked around:** the bake writes rows; a row needs a
  routable classid. Writing under `0x0000_0000` produces a dataset
  indistinguishable from a bootstrap row — the zero-fallback ladder owns that
  value. The bake is coded to **refuse to write** until this resolves, and
  that refusal is correct, not a gap.
- **What it needs:** a `domain:appid` assignment (`0x0F = Geo` exists; the
  appid/classview half is open). One decision, then W1 finishes and W2/W4
  open.
- **Cost of leaving it:** everything below Tier 2 stays unreachable
  indefinitely. This is the highest-leverage item in the document.

### Tier 2 — runnable now, no mint needed

**N2. Close bar B6's ladder for the cross-unit pairs.** The 16/64/256
resolution ladder was computed for the K×K pair only; the 19 cross-unit pairs
have a single-resolution number each.

- **Why it matters:** the ladder is the *can-it-differ* half. Without it, a
  cross-unit ρ of 0.9995 vs 0.9996 is a number with no scale attached, and
  the 10 primary failures cannot be read as "close" with confidence.
- **Cost:** small — the prep stage already emits the arrays; it is one more
  loop in `fidelity_probe_prep.py` and its reader.
- **Kill:** if 16 levels is indistinguishable from 256 on cross-unit pairs, ρ
  is decorative *for that comparison* and bar B7 must move to a physical-unit
  metric before any per-pair claim is made. (This is bar B6(c)'s own kill,
  applied to the pairs it was never run on.)

**N3. Test the wind/pressure skew as a hypothesis, or drop it.** The 10
primary failures concentrate in winter (2/9) and toward wind + pressure.
Right now that is **an observation, not a finding** — and this document's own
rules say an unlabelled pattern is a liability.

- **Cheapest honest form:** add a 4th and 5th season (the HEAD-sweep script
  already finds them) and check whether the skew survives. If it does, the
  next question is whether it is a *variable* property or a *season* property
  — winter is the only season carrying `mean_sea_level_pressure`, so the two
  are currently confounded, exactly like the regime ladder was.
- **Kill:** if the skew does not replicate across more seasons, delete the
  observation rather than leaving it as folklore.

**N4. Decide what to do about E17 (the empty-bucket gap).** 38–45 of 256
buckets empty under the shared floor, at every season. The direction is
unsurprising; the question is whether it *costs* anything.

- **The measurement that would settle it:** does the empty-bucket fraction
  degrade cross-unit ρ, or is it free? Correlate per-season empty count
  against per-season primary pass rate. n = 3 is too small to conclude — so
  this rides on N3's extra seasons rather than being run alone.
- **Design option if it does cost:** a per-variable *offset* with a shared
  *width* (the coverage lesson from §7.9 — width alone was never the driver).

### Tier 3 — real, but should wait

**N5. `D-WXS-2a` half B — Morton vs shipped under the ζ stencil.** Gated on
N1. Carries a stated prior (Morton ≈ 2× on neighbour locality) that is
explicitly **not** a result. Do not migrate on the prior.

**N6. Re-ask C3/C4 as a turbulence question.** The original comfort-zone
hypothesis is not dead — it was never properly *asked*, because the ladder
measured range. Doing it properly needs ζ (hence N1) **and** coverage-matched
donor selection, both of which the full grid makes possible and four hand-
picked boxes did not.

**N7. The geometry axis (C5).** Structurally unreachable at any scale this
arc currently has — the golden index floor needs 2 550 409 points and the
global 0.25° grid has 1 038 240. Either a finer grid (0.1° would clear it) or
a different construction. **Not** a budget problem; do not queue it as one.

### Explicitly NOT next

- **A second product-lead document.** This one is the surface; a roadmap file
  that drifts from it is worse than no roadmap.
- **Migrating the key layout to Morton.** G17's kill fired. The prior is
  recorded; acting on it before half B would be exactly the "re-reading a
  split as a win" failure the arc has a rule against.
- **Any claim resting on `D-WXS-8`'s exact ρ ≥ 0.9996 threshold.** The
  directional claim is established; the per-pair threshold is not.
- **Beating a learned model (claim C4).** Out of scope, unchanged, quarters
  of cost.

### The one-line read for a product decision

> **The codec works at grid scale and is measured. The substrate is built and
> gated. Exactly one operator-side decision — the classid mint — separates it
> from a running bake, and nothing downstream of that can start without it.**
