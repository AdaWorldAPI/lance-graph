# D-JCSBI-v1 — preregistered SBI/JC falsification matrix

> **Status:** SPECIFICATION ONLY, 2026-10-19. The rows below are proposed experiments and disabled-guard controls. **Zero new tests were executed** as part of this documentation PR. Companion: [integration plan](../../.claude/plans/jc-sbi-counterfactual-integration-v1.md).

For each fixture, record simulator/model/prior source+revision, observational unit/root, base version/seal, RNG seed, summary/discrepancy, method, budget, oracle distribution, expected and measured confidence/coverage and whether the negative control fires. A passing statistical diagnostic is *conditional on the generative model*, not a proof of scientific adequacy.

| ID | Experiment | Required assertion / falsifier |
| --- | --- | --- |
| SB-01 | Beta-Binomial small exact posterior | Analytic posterior mass and credible intervals match independently computed reference, incl. tails |
| SB-02a | **Genuinely zero posterior support:** every permitted prior × likelihood mass is zero / all log weights = -∞ | Refuse with a typed undefined-posterior status; never return an all-zero normalized "posterior" or fabricate support |
| SB-02b | **Numerically underflow-prone but finite log weights**, e.g. [-1000, -1001, -1200] or equally shifted by a huge negative constant | Stable log-sum-exp/softmax yields finite, normalized weights matching a high-precision oracle and invariant under constant shifts; no premature exp/zero-support refusal |
| SB-03 | One fixed seed and identical model+data witness | Bitwise/declared-tolerance reproducibility; changing model revision invalidates cache |
| SB-04 | Identifiable 1D synthetic retrieval-decay generator | Correct-model parameter recovery and empirical interval coverage over independent replicated datasets |
| SB-05 | Two ACT-R configurations with almost identical observational distribution | Report nonidentifiability/wide or multimodal posterior, not arbitrary single winner |
| SB-06 | ABC epsilon far too wide / far too narrow | Empirical approximation bias vs zero acceptance/ESS collapse surfaced; no silent posterior claim |
| SB-07 | ABC proposal replaces prior without importance correction | Explicit biased arm fails correctness against exact reference |
| SB-08 | SBC correct-family generator with randomized discrete rank ties | Rank distribution nominal within preregistered power tolerance; tie handling differs from always-first |
| SB-09 | Deliberately biased posterior sampler in SBC | Rank/coverage control flags wrong inference at prespecified sample budget |
| SB-10 | Missing-not-at-random retrieval: only successful recalls logged | Held-out miscalibration or explicit unsupported-selection refusal; cannot claim IID fit |
| SB-11 | Correlated repeats from same subject/chain | Group-aware uncertainty differs from treating each trial independently; no false independent-root counts |
| SB-12 | Posterior predictive out-of-family simulator | Detect discrepancy or mark inadequate power, not a green "model validated" receipt |
| SB-13 | One model near-1 observed posterior preference but unstable across replicated data | Distinguish PMP from predictive distribution of future PMPs, per #1611 |
| SB-14 | Duplicate simulated traces and duplicated source-root labels | No increase of independent empirical evidence / NAL confidence |
| SB-15 | Root and access scope exclusion | Denied observations change neither sufficient-statistic counts nor scores nor posterior |
| SB-16 | Mixed-version or different baked coordinate axes | Refusal, even if ordinal indices happen to equal |
| SB-17 | `dismech_candidates` Support, Refute, Partial, NoEvidence | Only first two narrow; no implicit numerical likelihood mapping from stance |
| SB-18 | Known-intermediate label with empty or ungrounded mediators | Excluded from identity gold; no artificial mediator "recovery" success |
| SB-19 | IndirectUnknownIntermediates vs Unknown topology | Former allows role-seeking only; latter does not assert mediator role; no false ground truth identity |
| SB-20 | One truly load-bearing and one redundant cut edge | Existing factual and counterfactual `chain_replay` pathways reused; threshold verdict and typed provenance preserved |
| SB-21 | Counterfactual cut arm treated as `InterventionBacked` | Guard refuses escalation; stays SimulationOnly/counterfactual (not observed SPO) |
| SB-22 | Confounded synthetic SCM where deleting an edge differs from `do(X=x)` | The edge-cut proxy must fail the intervention-equivalence assertion |
| SB-23 | OGAR search `FORK` of same hypothesis with differently numbered branches | Branch count not counted as posterior weight or as independent empirical evidence |
| SB-24a | EIG two-experiment finite discrete oracle | Analytic/nested-constant EIG ranking matches JC estimate within its declared MC error |
| SB-24b | Two hypotheses indistinguishable in current data, but distinguished by one **permitted future** design | Conditional EIG of informative design > 0; model-adequacy-gated design proposal succeeds rather than blanket abstention |
| SB-24c | Two hypotheses observationally equivalent under **every permitted design** | EIG is zero within numeric/MC error for distinguishing them; explicit design-relative nonidentifiability and abstention from identification claim, not a false positive design |
| SB-25 | EIG vs DAV/EWA disagreement heuristic at matched budgets | Report held-out identification, calibration, and all compute costs; permit a negative result |
| SB-26 | Entropy reduction while posterior model remains misspecified | EIG output marked conditional; real-world "informative" claim refused |
| SB-27 | Numerical degenerate statistics (no data, constant, overflow, Inf/NaN) | Typed refusal; no silent 0 or valid-looking significance |
| SB-28 | New ndarray masked reduction candidate vs incumbent | Scalar mathematical oracle + SIMD parity across tiers, copies/bytes/cost recorded; no new op solely for elegance |
| SB-29 | R frontend injected function / implicit recycling / arbitrary eval | Cold typed refusal, never execution in hot engine |
| SB-30 | 1000 independent simulations in one thought / 64 live hypotheses | Simulations do not exhaust 63 speculative slots; 64th simultaneous alternative returns typed BranchesFull |
| SB-31 | Malicious foreign-column index used in SBI summary | Refuse foreign gather; no d′ or hash-join fallback masquerading as mask-AND |
| SB-32 | Deliberately disable an optimizer (rather than semantic guard) | Semantics unchanged and cost/plan shape change recorded; correctness must NOT require optimization |

Mandatory mutational controls: disable scope mask (SB-15), collapse topology enum (SB-19), promote simulation-only receipt (SB-21), conflate cut with intervention (SB-22), double-count independent roots (SB-14/23), allow foreign gather (SB-31), and remove prior correction from SMC (SB-07). Each must make a targeted semantic/authorization test red. Merely removing a legitimate optimization is **not** a semantic falsifier (SB-32).

Minimum first runnable experiment: SB-01, **SB-02a and SB-02b**, SB-03, SB-06, SB-14 and SB-27. Before D-JCSBI-7, additionally run **SB-24a/b/c**. No empirical accuracy or performance claim before raw results and disabled-guard controls are published.


## Supplemental dynamic-experiment, Sudoku and ACT-R falsifiers

| ID | Experiment | Required assertion / falsifier |
| --- | --- | --- |
| SB-33 | Same grid/model revision, params, design and RNG stream | Identical typed output (declared tolerance); different seed alone does not create variation in a deterministic model |
| SB-34 | Identical baseline, change uniform susceptance scale | Preserve documented relative DC flow/cascade/connectivity-loss invariants; absolute bus angles may vary |
| SB-35 | Identical grid, sweep uniform thermal derate | May change overload thresholds/trip counts; do not apply the susceptance-scaling invariance to this case |
| SB-36 | Input `AgeModel::Uniform` vs `DensityProxy` vs `ModernizationSpend` | Distinct model revision/hypothesis identity and capability assumptions; hidden heterogeneity is not silently inferred |
| SB-37 | Fixed-topology covariance via `angle_cov` then a line trip | Recompute changed Laplacian/covariance or refuse; do not reuse `L⁺ Σ L⁺` from the old topology |
| SB-38 | `acflow` solver nonconvergence | Preserve solver-limitation status; do not automatically classify it as observed physical voltage collapse |
| SB-39 | OGAR `FOLD_XREF`, `NAKED_SINGLE`, `HIDDEN_SINGLE` vocabulary declarations | Test separate executable owner and unit/codebook constraints; unresolved operator must refuse, not masquerade as shipped method |
| SB-40 | Different question scopes over the same candidate field | Narrowing is reversible by asking a new question; no global fact removal or evidence-root generation |
| SB-41 | `FORK` 2 alternative hypotheses + 200 simulator evaluations | Exactly 2 stable alternatives, 200 simulation receipts, **zero** minted independent empirical observations; slot count depends on concurrent hypotheses |
| SB-42 | ACT-R activation, procedural utility, NAL confidence, Bayesian EIG | Force contradictory orderings; show they remain different typed axes and one cannot silently overwrite another |
| SB-43 | ACT-R conflicting productions with same eligibility | Explicit deterministic tie-break at same timestamp/version/policy; scope/version change invalidates result |
| SB-44 | ACT-R learning only from its own simulated reward | No independent observed success receipt, so cannot claim empirical procedural utility update |
| SB-45 | Three grid hypotheses, two permitted perturbation designs | FIFO, planner heuristic, ACT-R-only and ACT-R+JC-EIG match sample root/simulator-call budget; identification assessed held-out |
| SB-46 | "Mediators" in `perturbation-sim` vs DisMech biological intermediates | Distinct domain/model types; reject accidental physical-to-biological parameter/ontology substitution |
| SB-47 | Power-grid example uses composite score containing the target outcome | Validity oracle detects target leakage; cannot claim independent prediction from outcome-derived predictor |
| SB-48 | Domain simulator attempts an unauthorized real external intervention | Simulation is pure proposal; no effect without existing planner/policy action gate |

These supplement SB-01..32, not replace their causal and evidential safeguards. No test above has been executed as part of this docs-only change.
