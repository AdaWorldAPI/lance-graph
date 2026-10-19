# JC-R v1: an R-shaped statistical API over the canonical SoA substrate

> **Status:** Proposal / dependency audit, 2026-10-19. Documentation only.
> **Mathematical owner:** crates/jc (JC, Jirak-Cartan).
> **Proposed frontend/bridge:** a separate thin jc-r-formula / lance-graph-jc-r adapter, with an optional external R package.
> **Execution:** Existing relalg / R2IL / Quack → mask-risc → ndarray::simd; no second evaluator.
> **Related consumer:** [CerebroCortex-RS cognitive orchestration plan](https://github.com/AdaWorldAPI/CerebroCortex-RS/pull/1).
> **Precedence:** [sql-polyfill-r2il-v1](sql-polyfill-r2il-v1.md), [relalg-v1](relalg-v1.md), [souffle-ram-64-thought-stack-v1](souffle-ram-64-thought-stack-v1.md) and the D-ASM council's join restrictions. This proposal does not override any of them.

## 0. The decision in one paragraph

Let **R specify statistical intent**, while JC remains the sole owner of
validated statistical estimators and uncertainty conventions. Adopt two
separate **optional** client surfaces:

1. A restricted R-flavored **formula / statistical-call syntax**, parsed and
   resolved **outside the hot path** to typed procedure invocations over
   canonical SoA columns, presence/validity masks, and witnessed datasets.
2. A genuine **R package/client binding** that can call supported JC metrics
   from R, either on explicit input vectors (reference mode) or on a canonical
   dataset handle (zero-copy / handle-based mode where proven).

These are *frontends*, **not a request to embed the GNU R interpreter inside
the cognition engine**. In ACT-R, the letter R stands for Rational; the
statistical language R is a separate concern. Its value is researcher ergonomics,
cross-validation, and parameter estimation for cognitive experiments.

## 1. Existing implementation we must reuse (checked 2026-10-19)

| Already in JC | Ownership / existing function | Immediate API opportunity |
| --- | --- | --- |
| Pearson, Spearman, Cronbach alpha, ICC | jc::reliability | Reference vector APIs; rank/reliability outputs |
| Cohen kappa, McDonald omega, phi, KR-20, eta², R² | jc::stats | Agreement/reliability / effect-size calls |
| One-sample, paired, Welch, Student t tests; one-way ANOVA | jc::stats | R-compatible statistical entry points with declared assumptions |
| Pearson from cross moments | jc::stats::pearson_from_cross_power_sums | Resident masked two-column correlation |
| Sample covariance from cross moments | jc::stats::sample_covariance_from_cross_power_sums | Resident covariance |
| Simple OLS and R² from cross moments | jc::stats::simple_regression_from_cross_power_sums / r_squared_from_cross_power_sums | Small model-spec subset |
| One-way ANOVA and eta² from per-group moments | jc::stats::anova_from_power_sums / eta_squared_from_power_sums | Grouped response ~ categorical predictor |
| Weak-dependence calibration | jc::jirak | Explicit noise-floor claims, not ordinary independent-sample p-values |

The last four groups of calls consume ndarray::simd::CrossPowerSums or
PowerSums rather than materialized row-group vectors. The grouped moment
terminals exist in lance-graph-mask-risc / ndarray; they are a particularly
strong precedent for the intended pipeline.

**Current limitation:** Quack / mask-risc currently have integer/code lane
capabilities and explicit validity planes, **not** a complete native f64
statistical relation engine. Do not pretend that arbitrary R numeric vectors,
factors, weights, formulas or GLMs already lower to one admitted mask fold.
Admit the existing typed lanes only; float-valued, general model and dynamic
formula work must pass separate expansion gates.

JC's direct metric implementations and moment projections have distinct
conditioning, size limits and degeneracy contracts. They are independent
reference paths, not permission to implement a duplicate scalar evaluator
inside a frontend.

## 2. Two public-facing surfaces, one mathematical authority

### Surface A: restricted formula and function calls (typed compilation)

Indicative user syntax only, not shipped calls:

~~~r
jc_cor(x, y, data = cohort, use = "complete.obs", method = "pearson")
jc_aov(score ~ group, data = cohort)
jc_lm(score ~ marker, data = cohort)
~~~

Compile each accepted expression through a **cold, typed** boundary:

~~~text
R-like call or formula
    ↓ parse / whitelist / type-check (cold; source text discarded)
resolved column IDs + statistical operator + policies + dataset witness
    ↓ reuse existing procedure/binder owner; no parallel general query AST
visibility mask ∧ validity masks ∧ explicit selection mask
    ↓ relalg / R2IL / Quack legal lowering
single Mask-RISC evaluator over canonical resident SoA / ndarray::simd
    ↓ integer/code sufficient statistics, when supported
JC estimator / calibration / undefined-result policy
    ↓
typed statistical report (estimate, assumptions, sample size, witness)
~~~

The formula representation is a **typed statistical model specification**
(e.g. response column, predictor column, group codebook, estimator kind,
null policy, provenance). It **must not** recreate a general relational AST,
R interpreter, parser-as-executor, or a second SQL procedure vocabulary.
Binding occurs once: column names and factor labels resolve via codebooks to
canonical IDs; no hot String / JSON / schema inference.

For the first milestone accept only:
- jc_cor(x, y, method = "pearson", use = "complete.obs") over supported
  integer numeric lanes with explicit pairwise input validity.
- jc_aov(y ~ group) with one categorical codebook and i32-valued response,
  backed by grouped PowerSums.
- jc_lm(y ~ x) for a **single** numeric predictor, backed by CrossPowerSums;
  require explicit intercept convention, degrees of freedom and degeneracy
  handling, and do **not** promise full base-R lm summary parity.

Not admitted initially: general R expression evaluation, arbitrary function
calls inside formulas, implicit data-frame recycling, unverified factor
contrasts, multi-predictor or interaction formulas, missing join provenance,
R object attribute semantics, and silently coerced types.

### Surface B: genuine R package / bindings (outside core)

A separate package (provisional name **lancegraphjc**) can expose idiomatic R
functions that correspond to JC operations. Begin with deterministic copied
numeric vectors and report data, then evolve toward a typed opaque
dataset/reference handle if performance requires it.

Candidate transport choices to evaluate **before adopting one**:
- A narrow native C ABI / .Call or extendr-based wrapper as an optional
  separately built binding package; audit ABI stability, R API correctness,
  garbage collection, thread-safety, license and distribution constraints.
- Apache Arrow / Lance-compatible batch interchange when ownership,
  schema, validity and zero-copy properties are demonstrated.
- Explicit offline R↔Rust fixture exchange in CI, independent of a resident
  R runtime and entirely adequate for the initial differential oracle.

Do not link the core jc crate against libR, make R a transitive production
dependency, or claim that Arrow automatically preserves a thought's
identity/version without an explicit witness binding. The real R package
and the compiler frontend may share **one typed statistical request
contract**, not duplicate mathematical formulas.

## 3. Mathematical and data semantics: fail closed

### 3.1 NA, NaN, infinity, invalid estimates

R distinguishes NA from NaN and supports multiple missing-value policies;
JC presently rejects non-finite inputs and reports undefined/degenerate
estimators as None. A boolean mask alone is not a license to equate
a missing observation with zero.

A typed request must declare:
- Value lane data type and a separately resident **validity plane**.
- A complete-case policy, by default intersection of validity for **every**
  required variable, executed **before** moment folding.
- Treatment of NA, NaN, ±Inf, and skipped/unpaired samples. Unsupported
  combinations return a typed refusal, never silently reinterpret.
- Distinction between zero observations, undefined estimates (None), and
  valid estimates equal to 0.0.

First supported policy: **complete observations only**, with matched
row-universe/version and explicit counts. R compatibility policies such as
"everything", "pairwise.complete.obs", factor exclusion, na.action and
weights are out of scope until individually specified/tested. The R adapter
may report undefined results as an R NA with an accompanying diagnostic,
not as a valid 0.0 or false significance.

### 3.2 Population identity, factors, and samples

- All folded columns must refer to the **same sealed ordered population**
  or to a separately proven coordinate alignment. No foreign-key gathers,
  strided indexed hops, or arbitrary joins disguised as bitwise AND.
- Each statistical request carries an immutable dataset **version witness**
  and a scope/visibility mask. Scope is intersected before sample counts and
  group statistics, so inaccessible observations cannot influence aggregates.
- Categorical factors use a declared codebook, including valid level IDs,
  observed vs unused levels, stable ordering and an explicit treatment of
  empty groups. Do not treat unvalidated display strings as factor identity.
- Preserve grouping, observational unit, independent-replication identity
  and provenance: repeated thoughts, linked facts, and repeated simulations
  do not automatically create independent samples.
- Models fitted on time-correlated trajectories must explicitly record that
  dependence and either select a compatible calibration or decline IID claims.

### 3.3 Numerical contracts and calibration

- Keep **JC** responsible for variance-divisor conventions, p-value tails,
  effect-size definitions, undefined estimates, and conditioning.
- CrossPowerSums / PowerSums have bounded exact integer accumulation and
  checked conversions; specifically preserve the currently documented
  ~2^32 rows/group bound and checked i128 centering. Overflow is refusal.
- Use ndarray::simd for numeric execution; mask-risc remains the single
  bulk evaluator. A statistical frontend cannot introduce a scalar
  alternative, even as an unadvertised fallback.
- The significance path must distinguish classical IID t/ANOVA calculations
  from jc::jirak weak-dependence calibration for correlated Hamming bitfields.
  No single p-value label is valid for both without declaring the model.
- Retain precision and rounding conventions and test adversarial
  large offsets, constant series, sparse groups, NaNs and overflow.

### 3.4 Output contract

An illustrative typed result (not production layout):

~~~rust
struct StatisticalResult {
    statistic: StatisticId,
    estimate: Option<f64>,
    sample_count: u64,
    degrees_of_freedom: Option<f64>,
    calibration: CalibrationId,
    assumption_flags: AssumptionMask,
    dataset_witness: WitnessVersion,
    // Optional named model/parameter IDs resolved by the caller.
}
~~~

Results should distinguish no estimate, insufficient samples, invalid
calibration, stale witness, unavailable lane primitive and unsupported
formula. Emit R lists/data.frame-friendly shape **only at the R boundary**;
do not make arbitrary JSON a canonical machine representation.

## 4. Integration with cognition and experiment design

The statistical surface is *about evaluating a cognitive process*, not part
of ACT-R's production-firing cycle:

- Fit and compare ACT-R activation decay, retrieval probability and
  utility-selection parameters against independently labeled experimental
  behavior where supported.
- Evaluate CerebroCortex-RS activation/FSRS recall, spreading, skill
  selection, consolidation and longitudinal drift with confidence-qualified
  metrics.
- Provide R-compatible inputs for simulation-based inference **as an
  optional external experiment**; posterior inference is not the same thing
  as NAL evidential confidence and must not overwrite CE64 truth semantics.
- Analyze reproducible interventions or ablations over the same witnessed
  population; preserve the chosen experimental unit and causal assumptions.
- A generated hypothesis remains a hypothesis, even if its R-style fit looks
  good. Calibration does not establish causal identification or cognition.

No function in the statistical API may silently mutate thought state,
schedule cognitive productions, elevate confidence or select external
effects. These are distinct owners.

## 5. Proposed work packages and falsifiers

| Gate | Work package | Acceptance / negative control |
| --- | --- | --- |
| **D-JCR-0** | Freeze existing JC and Quack/Mask-RISC type inventory, documented estimator semantics, sample constraints | An executable map from each proposed R call to an existing **real** JC function and admitted terminal; unsupported types labeled REFUSE |
| **D-JCR-1** | Standalone typed StatisticalRequest, policy enums and result metadata (no parser yet) | Construct valid and invalid requests in Rust; no independent math or row evaluator; validity/scope/witness mandatory |
| **D-JCR-2** | Integer-lane correlation over CrossPowerSums via resident masks | Same sample set, moment fields and Pearson result as independent materialized-vector oracle; constant and large-offset controls |
| **D-JCR-3** | One-way ANOVA / eta² over grouped PowerSums; simple one-predictor OLS | Exact group codes and counts; numeric/error parity to JC slice methods; empty/degenerate group refusal |
| **D-JCR-4** | Whitelisted R-style formula parser and resolved-model lowering | Accepted calls become typed requests; arbitrary function evaluation, recycling, unknown factors, foreign joins and misaligned populations rejected |
| **D-JCR-5** | External R parity harness (reference mode) | Differential fixtures against R's base stats::cor/t.test/aov/lm for declared overlapping semantics; tolerances preregistered |
| **D-JCR-6** | Optional R package or native ABI, separate from core | R can call supported JC metrics; no R runtime dependency in jc; GC/thread/ABI/ownership tests; license reviewed |
| **D-JCR-7** | SoA-handle mode, broader numeric lanes and formulas (only after separate substrate gates) | Real masked-lane support proven, no duplicate evaluator, predeclared p95/p99/memory/move-count benefit vs reference |
| **D-JCR-8** | Cognitive research consumers / R experiment notebooks | Parameter-sensitivity and held-out ablation comparison reproducible with witnessed datasets, independent samples and negative results |

First priority is **D-JCR-0 through D-JCR-2**. A small honest
correlation-from-resident-moments experiment is more valuable than promising
a large R language implementation.

### Minimal differential / falsification matrix

1. Numeric pairs: perfect ±correlation, uncorrelated, one-element,
   repeated constant, large shared offset and high cancellation risk.
2. Scope/validity: differing missingness masks, all-invalid, sparse
   selected population, visibility denied, unaligned column versions.
3. Groups: two balanced, highly unbalanced, empty advertised factor,
   zero within-group variance, one group only, unused codebook level.
4. Model: constant predictor, constant response, insufficient residual
   degrees of freedom, intentionally wrong intercept assumption.
5. Arithmetic: overflow guard, oversized moment count, conversion
   bounds, deterministic fold-partition and tile-order variations.
6. Inference: an IID control versus a correlated dependent-bit control,
   ensuring the *wrong* significance route is refused.
7. API: injected R functions/calls, dynamic formula evaluation, unbound
   symbols, implicit recycling, unauthorized attribute mutation.
8. Replay: same request + same witness + same policy → same result;
   a different witness is not reused from a stale cache.

The tests must compare to R as an **external oracle** where compatible;
they do not justify silently adopting all of R's language semantics.
Report `estimate`, denominators, degrees of freedom, missing-case count,
accepted assumption set, and numeric error rather than just PASS/FAIL.

## 6. Dependency / implementation constraints

- The **jc** crate retains its independent mathematical ownership and
  existing ndarray dependency. It does not acquire a general SQL engine,
  language runtime, or frontend parser dependency.
- Use a **separate adapter crate** for statistical request construction,
  formula parsing and linkage to lance-graph's planner. A potential public
  `jc::r` namespace is a convenience re-export **only if** the dependency
  graph is proven acyclic; it is not a reason to move the adapter into jc.
- The SQL-polyfill plan makes SQL operator names procedures, not new opcodes.
  Likewise R statistical calls are *procedures* over existing semantics,
  not a new R2IL opcode range unless a missing primitive first earns one.
- Preserve D-ASM's prohibition on gathers and foreign-index hops disguised
  as aligned mask operations, and the H14c/H14d thought-address ordering.
- Do not represent every nested formula operation as another thought; the
  0+63 resident thought overlay budget is unrelated to the formula AST depth
  or to independent statistical replicates.
- Format conversions (R objects, C ABI, Arrow, JSON for fixtures) terminate at
  declared protocol/lab boundaries. Production SoA lanes are typed/codebook
  backed, not text/JSON.
- Copy no implementation code from R, its packages, Soufflé or proprietary
  sources without an explicit license and dependency review. The Rust
  implementation is an independent, tested compatibility subset.

## 7. Explicit stop conditions and next action

**STOP** if any prototype requires: a second numeric evaluator; hidden
row-materializing backend; lossy NA/NaN conflation; silent sample-count
change; speculative float-lane support; a second canonical model/relational
AST; meaning-changing group mapping; unauthorized population read; or a
significance claim that ignores dependence.

**Next implementation PR:** D-JCR-0/D-JCR-1 as a narrow contract and
differential fixture inventory. Do not add R as a runtime dependency.
Follow with D-JCR-2 as a separate measured probe invoking the existing
CrossPowerSums → jc::stats::pearson_from_cross_power_sums path.

This is a plan, not evidence that these functions are already reachable
through R, and not approval to change the JC arithmetic contract.
