# B4 — Statistics mapping: `shared/lab.rs` vs the canonical owners (jc, ndarray)

Read-only research, 2026-10-10. Checkouts: `/home/user/_ro/lance-graph` (main),
`/home/user/_ro/ndarray` (master). Grades: **VERIFIED-IN-CODE** (signature quoted
from the file at the cited line), **CLAIMED-IN-DOC** (a doc/board sentence only),
**ABSENT** (searched; search space named).

Files read in full: `crates/cognitive-shader-driver/examples/shared/lab.rs` (377 l),
`crates/jc/src/lib.rs` (199), `jirak.rs` (190), `reliability.rs` non-test body,
`stats.rs` lines 1-1447 (all non-test code; test names enumerated),
`drift.rs` 1-260, `quorum.rs` 1-190; ndarray `src/hpc/statistics.rs` 360-500,
`src/hpc/reliability.rs` 1-300, `rolling_floor.rs` module doc + signatures,
`simd_masking_ops.rs` PowerSums/CrossPowerSums structs; probes' lab stages
(`moore_nars_gomoku_probe.rs` 1397-1730, 860-900, 1165-1182;
`crossword_crossing_care_probe.rs` 1455-1580); board entries
`2026-10-10-d-self-calibrating-lab-{0,1}.md`, `2026-10-10-d-lab-p1-certified-argmax-gomoku.md`.

Governing ruling (board index, `.claude/board/EPIPHANIES.md:194`, CLAIMED-IN-DOC):
**E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1** — "jc is the single certified home for
scientifically calibrated math and reliability statistics; other crates and probes
consume it and never extend it ad hoc." `lab.rs` is a probe-local statistics module
that extends the surface (sign-flip, Holm, verdict, CI, ECE, Brier, Beta/Gamma
sampling) — on its face the shape that ruling forbids.

Packaging fact (VERIFIED-IN-CODE): `crates/jc` is in the root `Cargo.toml`
**exclude** list; `crates/cognitive-shader-driver` is a **member**.
`crates/cognitive-shader-driver/Cargo.toml` has no `jc` dependency (grep for `jc`
returned none). So today lab.rs *cannot* call jc without adding a path dev-dep.

---

## Deliverable 1 — inventory of jc + ndarray statistical facilities

### 1a. jc (lance-graph `crates/jc/src`)

| facility | file:line | signature (quoted) | assumptions | tests |
|---|---|---|---|---|
| Pearson r | `reliability.rs:94` | `pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64>` | point estimate; `None` on n<2 / zero var / non-finite | `pearson_perfect_and_anti`, `pearson_textbook_value`, `pearson_degenerate_returns_none` |
| Spearman ρ | `reliability.rs:180` | `pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64>` | average-rank ties; point estimate | `spearman_monotone_nonlinear_is_one`, `spearman_tie_correction` |
| Cronbach α | `reliability.rs:225` | `pub fn cronbach_alpha(items: &[Vec<f64>]) -> Option<f64>` | tau-equivalence; population divisor | `cronbach_identical_items_is_one`, `cronbach_known_value` |
| ICC | `reliability.rs:268,292` | `pub enum IccForm { Icc2_1, Icc3_1 }`; `pub fn icc(ratings: &[Vec<f64>], form: IccForm) -> Option<f64>` | two-way ANOVA (Shrout–Fleiss) | Shrout–Fleiss worked example in tests |
| Cohen κ | `stats.rs:286` | `pub fn cohen_kappa(a: &[usize], b: &[usize]) -> Option<f64>` | nominal labels | `kappa_*` (5 tests) |
| McDonald ω_t | `stats.rs:404` | `pub fn omega_total(items: &[Vec<f64>]) -> Option<f64>` | congeneric 1-factor | `omega_*` (~12 tests) |
| φ / 2×2 | `stats.rs:603,620,660` | `pub fn phi(x: &[bool], y: &[bool]) -> Option<f64>`; `pub struct BinaryAssociation`; `pub fn binary_association(...)` | | `phi_*`, `binary_association_*` |
| KR-20 | `stats.rs:724` | `pub fn kr20(items: &[Vec<bool>]) -> Option<f64>` | = α on dichotomous | `kr20_*` |
| R, R², η² | `stats.rs:809,906,960` | `multiple_r_squared`, `multiple_r`, `eta_squared(groups: &[Vec<f64>]) -> Option<f64>` | effect-size family is **r**; d-family deliberately out of scope (`stats.rs:35-42`) | `r_squared_*`, `eta_squared_*` |
| t-tests | `stats.rs:1017,1045,1075,1115` | `pub fn t_test_one_sample(x: &[f64], mu0: f64) -> Option<TTest>`; `pub fn t_test_paired(x: &[f64], y: &[f64]) -> Option<TTest>`; `t_test_welch`; `t_test_student` | **classical, IID-normal**; `t_test_paired` = one-sample on differences; returns `None` on zero-variance differences | `one_sample_t_textbook_value`, `paired_t_is_one_sample_on_differences` (stats.rs ~2319), `welch_t_reference_value`, `t_tests_degenerate_return_none` (asserts `t_test_paired(&[1.0,2.0],&[0.0,1.0]) == None`, constant difference) |
| `TTest` result | `stats.rs:980` | `pub struct TTest { pub t: f64, pub df: f64, pub p_two_tailed: f64 }` | **no confidence interval field** | — |
| one-way ANOVA | `stats.rs:1153,1282,1291` | `anova_one_way`, `anova_from_power_sums(groups: &[PowerSums]) -> Option<Anova>`, `eta_squared_from_power_sums` | classical F | `two_group_anova_f_equals_pooled_t_squared_and_eta_matches`, power-sums equivalence module |
| cross-moment projections | `stats.rs:1342,1352,1380,1398` | `pearson_from_cross_power_sums(m: &CrossPowerSums)`, `sample_covariance_from_cross_power_sums`, `simple_regression_from_cross_power_sums`, `r_squared_from_cross_power_sums` | exact i128 centring | `cross_power_sums_equivalence` module |
| Fisher 2z | `stats.rs:1434,1443` | `pub fn fisher_2z(r: f64) -> f64`, `fisher_2z_inv` | variance-stabilising transform (×2) | `fisher_2z_*` |
| t / F tail p | `stats.rs:239,248` | `fn t_two_tailed_p(t: f64, df: f64) -> Option<f64>` (**private**), `fn f_upper_p` (private) | regularised incomplete beta (NR §6.4) | `t_distribution_tail_matches_reference`, `incomplete_beta_matches_reference_values` |
| t **quantile** / CI | — | **ABSENT** (no inverse t, no interval anywhere in jc; searched `quantile|interval|t_quantile|inv_t` in `crates/` ) | | |
| Jirak weak dependence | `jirak.rs:64,127` | `fn berry_esseen_sup_error(samples: &[f64]) -> f64` (**private**); `pub fn prove() -> PillarResult` | see 1c | `dependent_error_exceeds_iid_error`, `iid_data_converges_to_normal` |
| drift summaries | `drift.rs:96,147,228` | `reencode_drift`, `reencode_batch`, `delta_summary(deltas, material_threshold, large_threshold) -> Option<DeltaSummary>` | descriptive only | module tests |
| lens quorum | `quorum.rs:44,70,105,169` | `max_u8_variance`, `pairwise_agreement_u8`, `enum QuorumLevel` (230/179/128 bands), `cronbach_report` | descriptive dispersion | module tests |
| multiple testing (Holm/Bonferroni/BH) | — | **ABSENT** in jc (and in all of `lance-graph/crates/**/*.rs` outside lab.rs; grep `holm|bonferroni|benjamini`) | | |
| sign-flip / permutation / bootstrap | — | **ABSENT** in jc (only hit: a comment at `stats.rs:1689` "global sign flip" test about ω loadings, unrelated) | | |
| pre-registration struct / verdict | — | **ABSENT** in jc | | |
| ECE / Brier / reliability diagram | — | **ABSENT** in jc and lance-graph crates outside lab.rs | | |
| Gamma / Beta sampler | — | **ABSENT** in jc. `jc::stats::ln_gamma` (`stats.rs:126`, private) is a log-Γ *function*, not a sampler. | | |

### 1b. ndarray (`/home/user/_ro/ndarray`)

| facility | file:line | signature | assumptions | tests |
|---|---|---|---|---|
| `Statistics` trait (median, population variance, std, percentile, top_k, …) | `src/hpc/statistics.rs:22-62` | `pub trait Statistics<A> { fn median(&self)->A; fn variance(&self)->A; … fn percentile(&self, p: A)->A; … }` | population divisor; descriptive | `test_median_*`, `test_variance`, `test_percentile` … |
| exact mergeable moments (u32) | `statistics.rs:378,391,401,410,426,470` | `pub struct MomentsU32 { pub n: u64, pub sum: u128, pub sum_sq: u128 }`; `observe`, `merge`, `mean`, `variance` (population); `pub fn moments_u32(values: &[u32]) -> MomentsU32` | exact integers; descriptive | `moments_u32_is_exact`, `merge_is_exact_and_order_independent`, `mean_and_variance_match_two_pass` |
| grouped power sums | `src/simd_masking_ops.rs:2118,2372,2169,2545` | `pub struct PowerSums { pub n: u64, pub sum: i64, pub sum_sq: u128 }`; `pub struct CrossPowerSums {…}`; `pub fn masked_group_power_sums_i32(mask_words: &[u64], keys: &[u32], values: &[i32], out: &mut [PowerSums])`; `masked_group_cross_power_sums_i32` | i32 values only | consumed by jc equivalence tests |
| rolling floor (HDR) | `src/hpc/rolling_floor.rs:506,567,583,743` | `pub struct RollingFloor`; `calibrate(sample: &[u32]) -> Self`; `observe(&mut self, distance: u32) -> Option<FloorShift>`; `threshold(&self, level: SigmaLevel) -> u32` | empirical σ-lattice thresholds, drift rule |Δμ|>σ/2, |Δσ|>σ/4 (module doc) | module tests |
| quantiles | `rolling_floor.rs:140,159` | `pub fn rank_per_10000(len: usize, per_10000: u32) -> usize`; `pub fn quantile_of_sorted(sorted: &[u32], per_10000: u32) -> u32` | empirical | doctests |
| reliability (fidelity) | `src/hpc/reliability.rs:40,98,119,164,219` | `pub fn pearson(a:&[f64], b:&[f64]) -> f64`; `spearman`; `cronbach_alpha(items: &[&[f64]]) -> f64`; `icc_a1(ratings: &[&[f64]]) -> f64`; `pub struct FidelityReport` | **degenerate → 0.0** (jc returns `None`); ICC(2,1) only | module tests |
| normal sampler | `randomized_signature.rs:365`, `compression_curves.rs:35`, `clam.rs:2813` | all **private** (`fn normal(&mut self)`, `fn prng_normal`, `fn next_gaussian`) Box–Muller | | — |
| t-test / CI / Holm / sign-flip / ECE / Brier / Gamma / Beta | — | **ABSENT** (grep over `src` and `crates` for `holm|bonferroni|brier|thompson|gamma_sample|beta_sample|marsaglia|t_quantile|confidence interval`; only false hits on "student" in crystal_encoder distillation) | | |

### 1c. Jirak — what it is and is not (VERIFIED-IN-CODE, `jirak.rs`)

- **What it computes:** `berry_esseen_sup_error` = `sup_x |F_n(x) − Φ((x−mean)/std)|`
  (Kolmogorov distance of a standardised empirical CDF to the normal), on 5,000
  Hamming distances at d = 16,384.
- **Inputs:** synthetic fingerprint pairs, one set with deliberate weak dependence
  (25 % shared prefix + 10 % XOR-blended overlap), one IID set.
- **Returns:** `PillarResult` with `pass = dep_error > iid_error && iid_error < 0.1 && dep_error < 0.5`
  (`jirak.rs:150`). It is a **qualitative pillar** (dependence inflates the
  Berry–Esseen error), not a reusable calibration function: no public API takes a
  sample and returns a weak-dependence-corrected threshold, variance, or p-value.
- **Assumptions:** physical-dependence (Wu) weak dependence among the *bits* of a
  fingerprint; the rate `n^(p/2−1)` is cited in the doc (`jirak.rs:12-15`), not
  computed (`jirak.rs:138-142`: "we check the rate, not the constant").
- So "use Jirak for significance" (iron rule, `stats.rs:44-53`, `reliability.rs:32-40`)
  currently names a *citation discipline*, not a callable — CLAIMED-IN-DOC.

---

## Deliverable 2 — every pub item in `lab.rs`, classified

Class codes: (1) equivalent in jc · (2) equivalent in ndarray · (3) exists but differs
in assumptions/semantics · (4) genuinely missing. Kind: **P** point estimation,
**C** empirical calibration, **S** significance testing / decision.

| lab item (lab.rs:line) | kind | class | evidence / difference | smallest canonical addition (owner) |
|---|---|---|---|---|
| `enum CertKind {Deterministic, Statistical, Empirical}` (:23) | meta | **4** | No certificate-kind vocabulary in jc (`PillarResult` at `lib.rs:96` has `pass/measured/predicted/detail` only). | `jc`: a `CertKind` enum beside `PillarResult` (or a field on it). Vocabulary, not math. |
| `struct Prereg {id, hypothesis, baseline, treatment, endpoint, higher_is_better, min_effect, blocks}` (:34) | S | **4** | No pre-registration type in jc. Note: `verdict` never reads `pr.higher_is_better` (only `blocks`, `min_effect`, :150-158); direction is whatever the caller passed to `paired(…, higher_is_better)`. All 13 call sites pass a literal matching the prereg (checked: H2-H7, T1-T4, C1-C4), but nothing enforces it. | `jc::lab` (or `jc::prereg`): `Prereg` + a `paired_for(&Prereg, b, t)` that takes the direction FROM the prereg. |
| `enum Verdict {Supported, Falsified, Inconclusive}` (:49) | S | **4** | none in jc | `jc` with `verdict`. |
| `struct Paired {n, mean, sd, p, lo, hi}` (:57) | P+S | **3** | Closest: `jc::stats::TTest {t, df, p_two_tailed}` (`stats.rs:980`). Differences: (a) jc has **no interval** (`lo`,`hi`); (b) jc's p is a **parametric t** p, lab's is an **exact sign-flip** p; (c) jc returns `None` on zero-variance differences, lab returns `sd = 0` and a zero-width interval. | extend `jc::stats` with `t_interval(x, level) -> Option<(lo, hi)>` (needs a public t-quantile; see T975). |
| `const T975: [f64; 30]` (:70) | S | **4** | jc has `t_two_tailed_p` (`stats.rs:239`, private) — the forward CDF, not the inverse. No quantile anywhere. Table values checked against the standard table (df1 12.706 … df15 2.131 … df30 2.042) — correct. **Semantic defect:** df > 30 uses 1.96, i.e. z, which is anti-conservative for df 31-∞ (true df 40 = 2.021, df 60 = 2.000). Indexing `T975[n-2]` for df = n-1 is right. | `jc::stats::t_quantile(p, df)` by bisection on the existing `reg_inc_beta` — one function, no table, no df cut-off. |
| `fn sign_flip_p(d: &[f64]) -> f64` (:77) | S | **4** | No permutation/randomisation test in jc or ndarray. Exact (2^n enumeration), `assert!(n ≤ 20)`. Tie tolerance `obs − 1e-9` is **absolute**, so it is scale-dependent (fine for scores ~1e-2 and folds ~1e2; wrong for endpoints near 1e-9). n = 0 → 1 mask, p = 1. | `jc::stats::sign_flip_p(d) -> Option<f64>` (exact ≤ ~24, Monte-Carlo above with a stated seed), relative tolerance. |
| `fn paired(baseline, treatment, higher_is_better) -> Paired` (:99) | P+S | **3** | = `jc::stats::t_test_paired` for the point estimate (mean of d, unbiased sd: lab :110 divides by n-1, same as jc `sample_var` `stats.rs:89`), but different **test** (sign-flip vs t) and adds a CI jc lacks. Panics (`assert_eq!`) on ragged input where jc returns `None`. | Compose in jc: `PairedResult { mean, sd, t: TTest, sign_flip_p, ci }`. |
| `fn holm(p: &[f64], alpha: f64) -> Vec<bool>` (:132) | S | **4** | No multiple-testing correction in jc, ndarray, or any lance-graph crate. Implementation is textbook step-down, `≤` threshold, stops at first failure; ties sorted by `total_cmp` (stable). Correct (`holm_matches_the_textbook_step_down`). | `jc::stats::holm(p, alpha) -> Vec<bool>` verbatim; optionally `holm_adjusted(p) -> Vec<f64>`. |
| `fn verdict(pr, r, rejected) -> Verdict` (:149) | S | **4** | Decision rule: SUPPORTED ⇐ Holm-reject ∧ mean ≥ min_effect; FALSIFIED ⇐ CI upper < min_effect; wrong block count ⇒ INCONCLUSIVE. Two notes: (i) the FALSIFIED arm is a **one-sided 97.5 % non-superiority bound, not multiplicity-corrected** (Holm governs only support), so a family of k falsifications has per-claim, not family-wise, error control; (ii) the CI is the t-interval, so SUPPORTED and FALSIFIED rest on **different** tests (sign-flip vs t-normality) — they can disagree at the margin. | `jc`, beside `Prereg`; state both error-control facts in its doc. |
| `struct Bin {lo, hi, n, mean_pred, observed}` (:164) | C | **4** | none | `jc::calibration::Bin`. |
| `fn reliability(binned: &[(u64, f64, u64)]) -> (Vec<Bin>, f64)` (:176) | C | **4** (name collides with `jc::reliability`, a different concept) | ECE (count-weighted |mean_pred − observed|) over caller-binned equal-width bins. Docstring (:172-175) claims it also returns "the Brier score lower bound computable from bins" — **it does not** (returns `(bins, ece)` only): doc/code mismatch. Empirical point estimate, no interval; bins pool 16 blocks whose predictions are sequentially dependent (same learner), so any interval on ECE would need a dependence-aware method (block bootstrap over seeds), not IID. | `jc::calibration::{ece, reliability_bins}`; name it `calibration`, not `reliability`. |
| `fn brier(pairs: &[(f64, bool)]) -> f64` (:202) | C | **4** | none; trivially correct; empty input returns 0.0 (divides by `len().max(1)`) — a silent value where jc's contract would return `None`. | `jc::calibration::brier -> Option<f64>`. |
| `fn report(pr, r, rejected) -> String` (:211) | — | **4** (presentation) | Labels every line "(statistical)". | stays in the probe (formatting is not math). |
| `fn normal(u) -> f64` (:228, private) | sampling | **3** | Same Box–Muller exists three times in ndarray, all **private** (`randomized_signature.rs:365`, `compression_curves.rs:35`, `clam.rs:2813`). Clamp differs: lab `f64::MIN_POSITIVE`, ndarray 1e-300 / 1e-15 / 1e-12. | ndarray `hpc`: one public `standard_normal(u: impl FnMut()->f64)` (draws belong to the compute substrate; ndarray already owns SplitMix64). |
| `fn gamma_sample(shape, u) -> f64` (:235) | sampling | **4** | Marsaglia–Tsang + `U^{1/a}` boost for shape < 1. jc has only `ln_gamma` (private, `stats.rs:126`). | ndarray (sampler) or jc (if treated as calibrated math); tests move with it. |
| `fn beta_sample(a, b, u) -> f64` (:258) | sampling | **4** | Gamma ratio. Semantically linked to `causal-edge/tests/d_rpf_p5_beta.rs` (NARS expectation = Jeffreys Beta(½,½) posterior mean, test-only helpers `posterior_mean`, `posterior_var`, lines 46-55) — that file has **no sampler**. | same owner as `gamma_sample`; Thompson uses it with `pos+½, neg+½` (`moore_nars_gomoku_probe.rs:887`). |
| tests (:264-376) | — | — | `beta_sample_has_the_beta_moments`, `sign_flip_p_is_exact_on_a_small_case`, `holm_matches_the_textbook_step_down`, `a_planted_effect_is_found_and_a_null_is_not`, `reliability_sees_calibration_and_miscalibration` — move with their functions. | |

**Net:** nothing in lab.rs is class (1) or (2). Two items are class (3) (`paired`
vs `jc::stats::t_test_paired`; private Box–Muller vs ndarray's three private
copies). Everything else is class (4): jc has point estimates and classical
p-values but no exact/permutation test, no CI, no t-quantile, no multiple-testing
correction, no pre-registration/verdict, no calibration (ECE/Brier), no samplers.

### Where each dependence model applies

- **Jirak (weak dependence)** applies where the *unit of inference is a sum over
  weakly dependent bits or codebook-shared components*: Hamming distances on 16,384-bit
  fingerprints, role-key-overlapping slices, CAM-PQ-quantised codes, XOR-bundle
  accumulation (`jirak.rs:7-10`). A σ-threshold or p-value **on those sums** must not use
  IID Berry–Esseen or the classical t (`stats.rs:44-53`).
- **Matched-block paired design over independent seeds** — the lab — is a different
  regime. The unit of inference is the **block** (one seed / one puzzle), and the
  statistic is the per-block difference `d_i`. Dependence *inside* a block (300 games of
  one sequentially-learning agent; bit-level dependence of any fingerprints the agent
  uses) is absorbed into `d_i` and never enters the test. What the exact sign-flip p
  needs is: (a) `d_1..d_16` mutually independent — met by construction (distinct
  seeds 2001–2016 / 4001–4016; distinct puzzles from seed 0x1AB5EED), and (b) under H0
  each `d_i` is **symmetric about 0** (sign-exchangeable). (b) is an *assumption*, not
  a consequence of randomisation: arms were not randomly assigned within a block, both
  arms run the same seed deterministically. It holds under the sharp null "the config
  change has no effect on the outcome distribution of a block", which is what H0 says.
  Given (a)+(b) the p-value is exact; Jirak is irrelevant at this level.
- **The t-interval** (`lo`,`hi`) additionally assumes approximately normal `d_i`; with
  n = 16 that is a real assumption (game-score windows are bounded averages, folds/step
  is heavy-tailed). The lab itself says so (lab.rs:7-9).
- **ECE** pools predictions across steps of one learner: those are serially dependent,
  so ECE is correctly labelled *empirical*; no interval may be attached using IID
  formulas. A block bootstrap over seeds is the dependence-honest interval.

---

## Deliverable 3 — every verdict the lab produced, and whether the canonical owner changes it

Sources: board entries (CLAIMED-IN-DOC numbers; the probes print them, they are not
asserted in tests) and the call sites (VERIFIED-IN-CODE). Every family: n = 16 matched
blocks, Holm α = 0.05, decided by `lab::paired` → `lab::holm` → `lab::verdict`
(`moore_nars_gomoku_probe.rs:1570-1584`, `1689-1702`; `crossword_crossing_care_probe.rs:1555-1568`).

**Counterfactual method.** The canonical owner for a paired significance test is
`jc::stats::t_test_paired` (`stats.rs:1045`). It returns no interval, so the
FALSIFIED arm (CI upper < min_effect) can only come from lab's t-interval in either
world; only the p fed to Holm changes. I reconstructed the paired t from each reported
CI: `se = (hi − lo) / (2 · 2.131)` (df 15, `T975[14]`), `t = mean / se`, two-tailed p by
numerical integration of the t₁₅ density (scratch python, not repo code). Rounding of
the 4-dp CIs moves t by < 0.3 %, p by < ±0.0005.

### Family v0 — Gomoku (`--lab`, seeds 2001–2016, 300 games, last-100 window)

| id | hypothesis | effect (+ = better) | 95 % CI | sign-flip p | Holm (lab) | lab verdict | t-test p (jc) | Holm thr. at rank | verdict with jc |
|---|---|---|---|---|---|---|---|---|---|
| H4 | certified minimax cuts folds/step (cost 0) | +28.9 | [+20.3, +37.6] | < 0.0001 | reject | SUPPORTED (+ deterministic: identical games all 16 blocks) | ≈ 0 | 0.0100 | unchanged |
| H7 | fold-cost term cuts folds/step | +142.7 | [+118.3, +167.1] | < 0.0001 | reject | SUPPORTED | ≈ 0 | 0.0125 | unchanged |
| H2 | learned chooser beats fixed HPM, score | +0.0238 | [+0.0060, +0.0415] | 0.0156 | reject | SUPPORTED | 0.0120 | 0.0167 | unchanged |
| H3 | delayed credit beats immediate, score | +0.0225 | [+0.0036, +0.0414] | **0.0267** | **keep** | **INCONCLUSIVE** | **0.0228** | **0.0250** | **CHANGES → SUPPORTED** (t-p 0.0228 ≤ 0.025 and mean 0.0225 ≥ min 0.02) |
| H5 | decay 0.95 adapts faster post-switch | +0.0072 | [−0.0051, +0.0195] | 0.2322 | keep | FALSIFIED (CI hi 0.0195 < 0.02) | 0.2314 | 0.0500 | unchanged (falsification comes from the CI, identical in both) |

Deterministic side-checks (not statistics): H4 identical games on every block (printed
`identical`, probe :1563-1569); pinned by `the_certificate_changes_no_game_for_a_cost_blind_learner`
(probe :2059). Calibration (empirical, base arm): ECE 0.317 over all outcomes, 0.050 over
non-neutral — `lab::reliability`; no canonical owner exists, so nothing to compare.

### Family v1 — Gomoku (`--lab-v1`, seeds 4001–4016)

| id | hypothesis | effect | 95 % CI | sign-flip p | Holm | lab verdict | t-test p (jc) | thr. | with jc |
|---|---|---|---|---|---|---|---|---|---|
| T1 | Thompson beats argmax e, score | −0.0500 | [−0.0620, −0.0380] | < 0.0001 | reject | FALSIFIED | ≈ 0 | 0.0125 | unchanged |
| T2 | Thompson adapts faster post-switch | −0.0153 | [−0.0244, −0.0062] | 0.0040 | reject | FALSIFIED | 0.0027 | 0.0167 | unchanged |
| T3 | structural context beats one global key | +0.0247 | [+0.0050, +0.0444] | 0.0186 | reject | SUPPORTED | 0.0174 | 0.0250 | unchanged |
| T4 | Thompson reduces folds/step | −18.5 | [−38.4, +1.4] | 0.0596 | keep | FALSIFIED (hi 1.4 < 5.0) | 0.0662 | 0.0500 | unchanged |

Note T1/T2: Holm *rejects* and the verdict is FALSIFIED — the two-sided test rejects
in the wrong direction; `verdict` reads direction only through `mean ≥ min_effect`, then
falsifies through the CI. Correct, but a reader must not read "Holm reject" as "support".

### Family C — crossword (`crossword_crossing_care_probe lab`, 16 puzzles, seed 0x1AB5EED, ln endpoints)

| id | hypothesis | effect (ln) | 95 % CI | Holm | lab verdict | with jc |
|---|---|---|---|---|---|---|
| C1 | Support cuts nodes vs Baseline | +3.59 | [+3.00, +4.17] | reject | SUPPORTED | unchanged (t ≫ 10) |
| C2 | Support cuts ops vs Baseline | −2.57 | [−2.90, −2.24] | reject | FALSIFIED | unchanged |
| C3 | Single cuts ops vs Baseline | −1.07 | [−1.71, −0.43] | reject | FALSIFIED | unchanged |
| C4 | Pair cuts ops beyond Single | −0.78 | [−1.13, −0.43] | reject | FALSIFIED | unchanged |

(Board does not print the C-family p-values; all four CIs exclude 0 by > 2 se, so any
paired test rejects at every Holm step.) Deterministic side-check: all arms sound,
16/16 solved, same fills where both solved.

### D-LAB-P1 (not a lab statistic)

Certified argmax is a **deterministic** certificate (`certified_minimax_picks_what_minimax_picks_with_fewer_folds`):
0 changed picks over 400 positions, 36.2 / 44.8 / 61.0 % fewer folds at k 4/5/8. No
significance test; no canonical-owner question.

### What the one flip means

H3 is the only verdict that depends on which owner computes the p-value: exact
sign-flip 0.0267 vs classical t 0.0228, against a Holm step threshold of 0.025. The
pre-registration (board lab-0 table, lab.rs header) named the sign-flip test, so the
recorded INCONCLUSIVE is the correct verdict *for the registered analysis*; switching
to `jc::stats::t_test_paired` after the fact would be an analysis change, and would
also swap an exact distribution-free test for one that assumes normal block
differences. The flip is evidence that (a) H3 sits on the decision boundary
(board's own OPEN: "H3 needs more blocks"), and (b) a migration of lab.rs onto jc
must **carry the sign-flip test into jc**, not substitute jc's t-test — otherwise a
recorded verdict silently changes.

---

## Findings worth acting on (ranked)

1. **lab.rs is an ad-hoc statistics module in a probe**, against
   E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1. Ten of its pieces have no owner anywhere
   (Holm, sign-flip, t-quantile/CI, Prereg, Verdict, ECE/Bin, Brier, Gamma, Beta,
   CertKind). Canonical additions, smallest first: `jc::stats::{holm, sign_flip_p,
   t_quantile, t_interval}`, `jc::calibration::{ece, brier}`, `jc::{Prereg, Verdict,
   verdict}`; samplers → ndarray `hpc` (one public `standard_normal` replacing three
   private Box–Mullers, plus `gamma`/`beta`). jc is workspace-*excluded*, so wiring needs
   a path dev-dep from cognitive-shader-driver (not verified to build — do not assume).
2. **Migration hazard:** replacing `lab::paired` with `jc::stats::t_test_paired` flips
   H3 INCONCLUSIVE → SUPPORTED and turns any zero-variance block-difference into `None`.
3. **`T975` df > 30 → 1.96** is anti-conservative; harmless today (n = 16) but wrong
   for the first 32+ block run. A computed quantile from jc's incomplete beta fixes it.
4. **Doc/code mismatch** lab.rs:172-175: says `reliability` returns a Brier lower bound;
   it returns `(Vec<Bin>, f64)` only.
5. **`verdict` ignores `Prereg::higher_is_better`** (lab.rs:149-160); direction is the
   caller's literal. All 13 current call sites agree with their preregs; nothing enforces it.
6. **Falsification is not multiplicity-controlled** (one-sided 97.5 % per hypothesis);
   support is Holm-controlled. State this in the verdict doc, or put the CI on a
   Bonferroni-/Holm-adjusted level.
7. `sign_flip_p` tie tolerance is absolute (1e-9); make it relative before reuse on
   small-magnitude endpoints.
8. ndarray and jc both ship Pearson/Spearman/α/ICC with **different degeneracy
   contracts** (ndarray → 0.0, jc → `None`) — a separate duplication, outside lab.rs.
