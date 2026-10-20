# D-JCSBI-v1 — simulation-based inference as calibrated mathematics over witnessed cognitive and causal experiments

> **Status:** PROPOSAL / source inventory / preregistered falsifiers, 2026-10-19. Documentation only; no runtime, model-fit, benchmark or pillar result is claimed.
>
> **Mathematical owner:** `crates/jc` for inference, posterior diagnostics, uncertainty and experimental-design estimators; **numeric execution:** existing `ndarray::simd` methods; **data selection:** existing same-coordinate RelAlg/Quack → R2IL → mask-risc; **effectful experiment orchestration:** planner / existing external graph-flow adapter.
>
> **Predecessors:** [#1613 JC-R](https://github.com/AdaWorldAPI/lance-graph/pull/1613) (OPEN, separate plan), [#1612 D-SRAM](https://github.com/AdaWorldAPI/lance-graph/pull/1612) (merged PLAN), [#1611 meta-uncertainty](https://github.com/AdaWorldAPI/lance-graph/pull/1611), [#1543 cognitive fabric](https://github.com/AdaWorldAPI/lance-graph/pull/1543), [CerebroCortex-RS #1](https://github.com/AdaWorldAPI/CerebroCortex-RS/pull/1).
>
> **Negative authority:** a simulator output is not a clinical observation; a cut edge is not automatically a Pearl `do()` intervention; calibration does not establish causal identification or prove a model family contains reality. No hot JSON/String, second relational engine, foreign gather, new CE64 code, runtime R interpreter, unauthorized clinical action, or automatic promotion of simulated outputs to NAL evidence.

## 0. Decision

Introduce **`jc::sbi` only after narrow mathematical parity**, ideally an additive Rust module under JC rather than a new orchestration crate. The optional R formula/package surfaces proposed in #1613 *invoke* a typed SBI request at a cold boundary; they never own inference formulas. Do **not** create a dedicated `lance-graph-sbi` crate unless two independent callers need reusable bindings absent from JC and the dependency proof says it avoids duplication.

SBI contributes inference over simulators, finite-simulation uncertainty, model diagnostics and expected information gain (EIG). It **does not** reimplement ACT-R, OGAR vocabulary, DisMech's upstream oracle, `chain_replay`, the canonical SoA, NAL revision, planner authorization or Mask-RISC.

Three separate scientific questions must remain separately answerable:
1. **Computational calibration:** does the inference algorithm recover known generating parameters when the simulator family is correct?
2. **Model adequacy:** do simulator outputs resemble held-out independent *real observations*, accounting for selection and dependence?
3. **Causal/interventional validity:** is a proposed operation a well-defined intervention on an independently justified structural causal model (SCM), or merely a chain deletion / counterfactual sensitivity probe?

A pass on 1 cannot prove 2 or 3.

## 1. Verified owners and what they actually ship

| Source | Verified role | Explicit non-role |
| --- | --- | --- |
| `crates/jc/src/stats.rs`, `reliability.rs`, `jirak.rs` | Pearson/ANOVA/OLS from checked PowerSums/CrossPowerSums, reliability, and separately scoped weak-dependence noise-floor methods | No shipped ABC/SMC, simulation-based calibration, posterior predictive inference, general likelihood-free posterior or EIG numerical method identified |
| `crates/jc/examples/dav_active_observation_probe.rs` | A real diagnostic comparison of route disagreement, EWA propagation, Quack/mask-RISC observation filtering and independently witnessed Gadamer revision; explicitly **not** production policy | Disagreement is not a calibrated Bayesian EIG |
| `crates/lance-graph-planner/src/dismech_candidates.rs` | Question-scoped `EvidenceMask` candidate-set intersection/subtraction: only `Supports::Support` and `Refute` narrow; `Partial` / `NoEvidence` are inert | Not model likelihood or belief update; excluded candidates are not globally disproved |
| `crates/lance-graph-planner/src/chain_counterfactual.rs` | D-DCR-3, **renamed from** `dismech_counterfactual`, now domain-agnostic: one edge cut; same `chain_replay::replay_chain` for both arms; thresholded `Verdict`; `InferenceType::Counterfactual` for cut arm | **Not general SCM or Pearl do-calculus**, not stochastic simulator or independent empirical evidence; the separate `base_seq` ranges identify distinct traces, not independent real observations |
| `crates/lance-graph-contract/src/dismech_evidence.rs` | Source-authoritative 4-state topology (Direct / IndirectKnownIntermediates / IndirectUnknownIntermediates / Unknown), source stances and evidence semantics | Topology codes or `Supports` categories are not numerical probabilities |
| `crates/lance-graph-contract/src/causal_audit.rs` | `SupportBasis::SimulationOnly`, `InterventionBacked`, evidence source IDs and source-aware receipts | Cannot transform simulated findings into directly observed or intervention-backed support |
| `AdaWorldAPI/OGAR/crates/ogar-dismech` | `DISMECH_CONCEPT_ID=0x0333`, 19 public `dismech:` causal predicates `0x90..0xA2`, 7 separate LOCAL search operations `0xA3..0xA9` including FORK, candidate-set operations, sparse residue overlay | Does **not** implement a stochastic dynamics simulator; the search overlay **is not evidence** and may not be fed into the graph as source truth |
| `AdaWorldAPI/dismech-rs` | Independent zero-dependency semantic reference for Monarch DisMech | Integration/SoA/classid implementation must **not** be moved into this oracle |
| `AdaWorldAPI/CerebroCortex-RS` | Deterministic ACT-R-like activation/FSRS/spreading/procedure baseline to replay | Neither a complete ACT-R cognitive controller nor a calibrated probabilistic simulator today |
| `#1613` | Proposed R-styled formula parser + optional external R client over typed statistics requests | Not merged/shipped, not a general R interpreter; admits only narrow Pearson/ANOVA/simple OLS first |
| `#1612` | Planned Soufflé AST-to-RAM-inspired *cold* optimizer; 0 givens + 63 live alternative thoughts, not a 64-iteration recursion cap | **D-RA-4/d′ is currently bannered a gather VIOLATION**; never use it as approved recursion path |

Current `dismech_counterfactual.rs` name is historical, **not** a missing file to recreate. The general implementation is `chain_counterfactual.rs`. Also: `ogar-dismech` is an OGAR crate, not a top-level `AdaWorldAPI/ogar-dismech` repository.

### Source-qualified DisMech oracle population

`dismech_evidence.rs` documents distinct populations. The source's label `INDIRECT_KNOWN_INTERMEDIATES` **alone is not a ground-truth mediator identity**. Many such records provide no `intermediate_mechanisms` and many named intermediates are ungrounded prose. A hidden-mediator experiment requires a nonempty, actually grounded identity and explicit provenance; its eligible count must be measured after the filter. `INDIRECT_UNKNOWN_INTERMEDIATES` supports the existence of a mediator role but *not* its identity; `UNKNOWN` does not even assert a mediator role. Reject any SBI experiment that scores guessed mediator IDs as observed truth in these latter categories.

## 2. Data flow: no second engine

```text
  CerebroCortex ACT-R simulator           dismech-rs corpus reference
         |                                      |
         +---- explicit typed generative model -+--- optional future approved SCM
                             |
                    versioned ModelSpec
            parameters/prior, seed, data-unit, model revision,
            sampling rule, observation/visibility selection model
                             |
 R-shaped request (optional #1613) -> typed SBIRequest (cold, whitelisted)
                             |
              witnessed SoA selection through existing
                RelAlg / Quack / R2IL / Mask-RISC
                 (same seal, same ordinal, validity)
                             |
           **jc::sbi** calibration + inference + diagnostics
                     ndarray::simd arithmetic
                             |
        typed SBIReport (model posterior/interval, predictive
        discrepancy, sample unit, estimator uncertainty, refusal)
                             |
       planner decides what to investigate / graph-flow runs
      *authorized* external experiment / independent empirical
         receipt returns through canonical evidence gates
```

SBI's simulation loop is statistical control of a **declared** `simulate(theta, design, seed)` callback, not a graph executor. The simulator lives with its domain owner. JC never pulls in `dismech-rs`, OGAR, planner, R, Rig or graph-flow as production dependencies. Consumer-side or dev-only harnesses connect them.

A candidate query may be specialized by #1612's future cold analysis, but SBI does **not** require an unapproved Datalog fixpoint engine. Same-coordinate columns may use existing masks. Foreign-coordinate joins are typed refusals until a legal, separately proven layout exists. The old D-RA-4 preimage/gather path remains STOPPED.

## 3. Minimal typed protocol (illustrative, not a shipped Rust API)

```rust
struct SimulationSpec {
    model_id: ModelId,                 // stable model family identity, not a thought slot
    model_revision: ModelRevision,     // simulator code + parameters/prior contract revision
    observation_unit: UnitId,          // person, episode, independent chain, etc.
    prior_id: PriorId,
    observation_policy: ObservationPolicyId,
    seed: u64,
    budget: SimulationBudget,
}
struct SBIRequest {
    spec: SimulationSpec,
    dataset: WitnessedPopulation,     // canonical base/version + scope + validity
    statistic: StatisticId,           // resolved typed summary/discrepancy
    method: InferenceMethod,          // ABC-grid, rejection ABC, optional SMC later
}
enum SBIStatus { Admitted, InsufficientCalibration, Misspecified, Unidentified, Refused }
struct SBIReport {
    status: SBIStatus,
    posterior: ModelResultHandle,     // metadata + bounded summaries, not hot serialized samples
    predictive_diagnostics: DiagnosticsHandle,
    simulation_count: u64,
    effective_sample_size: Option<f64>,
    model_revision: ModelRevision,
    dataset_witness: WitnessVersion,
    // independent empirical receipts vs simulated run ids must remain separate
}
```

Types are *conceptual placeholders*. Reuse existing canonical identity/witness/model fields where available; do not mint a second ID authority, new CE64 bits or a new hot layout. A model's parameter-set identities are not thought-slot ordinals. For independent examples, supply stable observation-root/group IDs. Drawing another simulated trace with a different RNG seed does **not** mint an independent empirical support root. The 0+63 thought overlay budget counts concurrently live *epistemic alternatives*, not posterior draws or Monte Carlo steps.

One candidate R-like research syntax **after** #1613 gates (not admitted in v1):
```r
jc_sbi_fit(recall_success ~ recency + association,
           simulator = actr_model, method = "abc_rejection",
           observations = witnessed_cohort, design = recall_trials)
jc_sbi_check(model = fitted, kind = "posterior_predictive")
jc_sbi_design(model = fitted, candidate_experiments = allowed_probes,
              objective = "expected_information_gain")
```
All strings terminate at the cold R boundary. The formula owner resolves identifiers into the same typed contract as native callers. Never claim these R calls exist today.

## 4. Mathematical method ladder and validation

### M0: exact reference / grid before neural SBI

Start with an analytically tractable small generative model: Beta-Binomial posterior or equivalent finite enumerated latent model. Implement a bounded discrete grid / exact small-case reference in **test code**, with JC's mathematical operations as the production candidate. Prior, observed data, likelihood where available, **stable log-domain posterior normalization**, and true zero-support refusal must match independent high-precision R/Python reference fixtures. A finite vector of log weights with very negative magnitudes is not zero support: subtract its finite maximum and apply log-sum-exp/softmax without first exponentiating the unshifted values. Refuse only when posterior normalization is mathematically undefined (e.g. every permissible prior×likelihood contribution is exactly zero, represented by all -∞ log weights, or invalid/non-finite inputs), not when an ordinary floating-point exp would underflow. Explicitly test both cases as separate fixtures. If exact likelihood is available at comparable cost, prefer conventional inference; SBI earns inclusion only when intractable likelihood or simulator behavior justifies it.

### M1: likelihood-free rejection ABC

For bounded low-dimensional `theta`: sample prior, run seeded simulator, compute declared summary/discrepancy under validity/scope policy, accept simulations at preregistered epsilon. Report acceptance, error, sensitivity to epsilon and summaries, effective sample size where meaningful. A posterior approximation conditioned on *chosen summary* is not automatically the true posterior for the raw dataset. No fake likelihood or non-finite inverse weighting.

### M2: sequential Monte Carlo ABC (gated)

Only after M1 parity, explore sequential populations, adaptive epsilon schedule, weight corrections and ESS/resampling diagnostics. Include known-target normalizing checks, degeneracy and history-dependent proposals. Reject biased empirical coverage or collapse. This is **not** `#1612`'s recursive Datalog fixpoint: no implicit `d′` path.

### M3: validation, not an optional pretty plot

- **Simulation-based calibration (SBC):** independently draw `theta ~ prior`, `y ~ simulator(theta)`, run inference, rank generating theta among posterior draws with randomized/tie-aware discrete handling; check rank uniformity/miscoverage with preregistered statistical tests and calibration uncertainty.
- **Posterior predictive checks:** independent replicated observations vs held-out data; report discrepancy, rare-event behavior and systematic error, including a deliberately out-of-family generator.
- **Identifiability:** test observationally indistinguishable parameter pairs, wide posteriors and sensitivity to prior/summary; no fabricated point-identification from multimodal fit.
- **Selection model:** log attempted/failed retrieval and non-observed cases; correct or refuse missing-not-at-random contexts, visibility/sampling bias and temporal dependence; do not treat `n` dependent trials as IID.
- **Model comparison meta-uncertainty:** replicate dataset-level model probabilities where explicitly defined and estimate their variability; follow #1611's separation of PMP levels from CE64/NAL F/C. No blanket claim that near-1 PMP means stable future preference.

### M4: Bayesian optimal experimental design (BOED/EIG), last

EIG is expected posterior information gain under the model and design distribution, not entropy, candidate-mask narrowing or EWA disagreement. Use a well-defined estimator (nested Monte Carlo or a vetted likelihood/ratio method) and an independent small discrete-oracle fixture. Separate aleatoric uncertainty, posterior/parameter uncertainty, estimator Monte Carlo error and model-misspecification warning.
 
 **Nonidentifiability is design-relative, not a blanket refusal.** When current observations cannot distinguish hypotheses but at least one *allowed future design* predicts different outcomes, use conditional EIG to rank those designs. That is precisely the identification objective of BOED. Only when *all* permitted designs are observationally equivalent under the declared model may the controller report nonidentifiability unresolvable within the available design space; it then abstains from claims requiring identification. If the simulator is inadequately calibrated or seriously misspecified, refuse **authoritative or real-world identification claims** and flag EIG as model-conditional; this is a distinct validity gate, not a reason to ban exploration in otherwise admissible, clearly labeled research simulations. Estimate MC uncertainty and allow a no-op/abstention if even the best permissible design is not sufficiently informative for its cost. Include experiment cost and authorized design constraints. Planner proposes next observation; it cannot bypass authority gates or create empirical evidence before executing the experiment.

**No neural density estimator or GFlowNet in the first implementation.** Compare them only if finite-model and ABC controls demonstrate a measurable gap with bounded data and compute. No borrowed source code or unreviewed license transplant.

## 5. DisMech-specific bridge and the counterfactual distinction

This is an **evaluation harness**, not a new production simulator:

1. `dismech-rs` is the independent source/semantic oracle; `ogar-dismech` remains the one canonical `dismech:` predicate vocabulary and 7 LOCAL search operators. No new classids/predicates to "encode SBI".
2. Existing `dismech_candidates::evaluate` performs **question-scoped** deterministic candidate masking. `Partial` and `NoEvidence` must not be recast as automatically decisive likelihood updates. The candidate mask and 3-bit OGAR residue are not probabilistic posterior weights or epistemic certification.
3. Existing `chain_counterfactual::counterfactual_replay` compares factual against **one edge deleted** using the same replay implementation. Tag every cut result `SimulationOnly` / counterfactual for downstream interpretation; do not write as observed SPO. Distinct trace sequence ranges are not distinct empirical sources.
4. **D-SBI-CF-0 (sensitivity only):** choose a grounded, source-qualified known-mediator sample; cut one asserted chain edge, replay and compare the threshold verdict, root trace and prediction. Compare to an independent recorded oracle for *edge load-bearing classification*, not a causal intervention effect.
5. **D-SBI-CF-1 (stochastic model, only when specified):** introduce a separately declared and validated generative model with observation noise, intervention semantics, confounder policy, boundary conditions and data-generating assumptions. Calibrate against held-out observed outcomes; require a positive/negative known-SCM synthetic control. Cut replay never silently substitutes for `do(X=x)`.
6. Distinguish **IndirectKnownIntermediates with grounded nonempty list** (possible hidden-mediator oracle), **known label without explicit identity**, **IndirectUnknownIntermediates** (mediator role but missing identity), and **Unknown** (even mediator role unsupported). Negative controls must prove hallucinated ID "recovery" never scores as success.
7. Source and patient boundaries: only public/reference or separately authorized de-identified evaluation material; observational validity and intervention recommendations are independent gated decisions. No health advice or automated treatment actions from a model comparison.

The connection to `OGAR::FORK` is architectural only: its bounded alternative-preservation semantics are compatible with speculative design arms; a FORK is neither a stochastic draw nor a posterior mass assignment. SBI can *score* such alternatives via an explicitly modeled simulator without redefining OGAR ops.

## 6. ndarray vs JC: promotion rules

**JC owns:** posterior weights and stable log-sum-exp normalization (with a true all-zero-support refusal); ABC accept/reject laws; ESS formula/finite-sample diagnostics; SBC rank and empirical coverage; posterior predictive discrepancy; model comparison and meta-uncertainty accounting; EIG estimator/MC error; typed statistical refusal semantics.

**ndarray owns only genuinely reusable compute:** feature/summary masks, SIMD masked distance kernels, logsumexp/normalized positive reductions if missing and independently proved necessary, reductions and batched finite numerical primitives used by multiple consumers. First inventory `ndarray::simd`; do not create a duplicate vector backend. A new `ndarray` primitive needs exact scalar mathematical oracle + SIMD backend parity and a measured speedup against existing operations with masks, gather prohibition, bytes, allocations and hardware tier recorded.

**JC pillar policy:** experimental SBI procedures and simulation calibrations do not become a new "pillar" merely by being computationally nontrivial. Keep them as `jc::sbi` research methods with tests and uncertainty reports. Promote *specific mathematically proved/validated numerical invariants*, such as stable normalized weights, finite-bound accumulation, or coverage under a declared reference generative family, via JC's established proof/measurement review. Always distinguish theorem assumptions from empirical Monte Carlo evidence and from general simulator adequacy.

**Existing moment terminals:** use `PowerSums`/`CrossPowerSums` for sufficient statistics where the chosen model actually admits them; they are **not sufficient for arbitrary SBI summaries, multimodal posteriors, trajectory dependence or neural density inference**. Float-valued or foreign-coordinate payloads require separately proven gates.

## 7. Phases (each follow-up its own separately authorized implementation PR)

| Gate | Deliverable | Acceptance / stop condition |
| --- | --- | --- |
| **D-JCSBI-0** | Source+API+sample-unit inventory over JC/ndarray, #1613, #1612, OGAR vocabulary, DisMech replay, CerebroCortex P0 | List actual method/terminal versus proposed; concrete simulator contract and independent oracles; reject false "SCM" claim |
| **D-JCSBI-1** | Cold typed `SimulationSpec`/`SBIRequest`/`SBIReport` contracts and refusal matrix, unit/test fixture only | No JC production dependencies on planner, R, OGAR; scope/version/model-revision necessary; seed and policy replay deterministic |
| **D-JCSBI-2** | Small exact discrete posterior fixture, log-domain normalized weights and degenerate bounds | Independent analytical + R/Python oracle parity; all -∞ zero-support refuses while finite extreme-negative log weights normalize correctly; no new SIMD op unless missing proven |
| **D-JCSBI-3** | Bounded ABC rejection and diagnostics on synthetic ACT-R/FSRS-like retrieval simulator | Parameter coverage/recovery and hold-out prediction vs fixed config and conventional calibration at matched budget; distinct empirical/sample grouping |
| **D-JCSBI-4** | SBC + posterior predictive + misspecification and missingness diagnostics | Uniform rank/nominal coverage **under correct generator**, deliberately wrong model and biased observation pipeline flagged, explicit abstention |
| **D-JCSBI-5** | DisMech cut replay harness (not SCM), known/unknown mediator strata, source receipt checks | Same replay source for both arms; observed-vs-simulated never conflated; guessing unknown mediators not scored; oracle-matched load-bearing classification |
| **D-JCSBI-6** | Model probability replication variability and ABC-SMC *if* earlier probes justify | Correct weighting/ESS and reproducible held-out stability, simulation-budget error surfaced |
| **D-JCSBI-7** | EIG over permitted experiment choices, optionally compare DAV/EWA heuristic | Exact discrete EIG oracle; current-design ambiguity recoverable by one permitted experiment is **ranked, not preemptively refused**, while ambiguity shared by all designs remains unresolved; better held-out identification or calibrated utility **at equal total simulation+selection budget**, not only lower entropy |
| **D-JCSBI-8** | #1613 R-style typed SBI adapter and optional R oracle harness | No eval of arbitrary R code, no hot strings, no R core dependency, same typed reports/witness, safety and parity |
| **D-JCSBI-9** | ndarray SIMD candidate and conditional JC numerical-pillar audit | Multi-consumer insufficiency proof, matched scalar and SIMD errors, hardware tiers, cold+warm throughput and read bytes; negative or null wins retain baseline |

**First implementation recommendation:** D-JCSBI-0/1/2 only, then a DisMech replay cross-check and an ACT-R recovery fixture. EIG is downstream of calibrated prediction. Avoid making D-JCSBI-3 rely on the forbidden D-RA-4 d′ driver; #1612 D-SRAM-4 requires an approved co-addressed replacement.

## 8. Evaluation ledger and stop conditions

Every measured comparison records: Git revisions; model/prior/simulator code revision; random seed stream identity; dataset seal/version; observation unit and root groups; missingness/visibility policy; oracle and baseline; method, epsilon/prior and selection constraints; simulator calls; compiler/cold time; warm execution; arithmetic/backend tier; MC standard errors and effective sample size; type of result (posterior distribution, predictive outcome, evidence receipt); p50/p95/p99; memory and bytes; rate of refusals; which semantic guard was disabled in negative control.

**STOP** if: source metadata, missingness or visibility silently alters the denominator; repeated simulation is counted as independent real evidence; simulated support is promoted to direct/intervention basis; `UNKNOWN` topology is treated as a known mediator; an ungrounded mediator becomes a gold identity; a model is declared causal merely from edge-deletion replay; prior or summary is hidden from report; an inference is overconfident under synthetic or held-out misspecification; R/JSON/string is made internal production representation; an unproven foreign gather or second SQL/Datalog/LLM runtime is introduced; a new opcode or CE64 reuse appears; or optimizer disabled changes valid statistical answers.

**Scientific precedence / external reading** (research citations, no source copied):
- Talts et al., *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*: https://arxiv.org/abs/1804.06788 .
- Schmitt, Radev, Bürkner, *Meta-Uncertainty in Bayesian Model Comparison*: https://arxiv.org/abs/2210.07278 ; existing #1611 note.
- Klein et al., *Supercharging Simulation-Based Inference for Bayesian Optimal Experimental Design*: https://arxiv.org/abs/2602.06900 .
- `dismech-rs` ownership correction: https://github.com/AdaWorldAPI/dismech-rs/blob/main/.claude/plans/causal-graph-soa-integration-v1.md .
- OGAR predicate and search palette: https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dismech/src/lib.rs .
- Current planner counterfactual: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-planner/src/chain_counterfactual.rs .


## 9. Shared dynamic experiment interface: OGAR mechanical Sudoku + ACT-R + domain simulations

> **Addition motivated by perturbation-sim and CerebroCortex, 2026-10-19. Proposal only.** The current crate APIs and examples were inspected. This section does NOT rename `dismech-rs`, change OGAR's vocabulary, turn the grid simulator into a biomedical model, implement an ACT-R controller, or claim the existence of a generic experiment API.

### 9.1 Current executable and symbolic building blocks

- `crates/perturbation-sim` is an executable, **parameterized** deterministic power-grid research simulator, not merely static scenarios: `simulate_outage(&Grid, &[f64], seed_line, CascadeConfig)`; its `model.rs` provides `AgeModel::{Uniform, DensityProxy, ModernizationSpend}`, susceptance scaling and uniform derating. `CascadeResult` exposes the perturbation shape and islanding. The crate is standalone and has its own scalar numerical reference paths; `ndarray::simd` is optional for Walsh-Hadamard and `angle_cov` under `pillar` explicitly calls ndarray Pillar-9 `CovHighD::sandwich` on a **fixed topology**. It is not an interchangeable biomedical simulator.
- `perturbation-sim/src/acflow.rs` contains a limited AC Newton–Raphson prototype. Solver nonconvergence is not ground-truth voltage collapse, and the prototype lacks some operational controls; label its outputs accordingly.
- `perturbation-sim/examples/validate_mediators.rs` checks **electrical/spectral mediators**, not the DisMech ontology's biological mediator identities. It explicitly warns about outcome reuse/circular validation in its composite collapse-number test. Never call that an independent prediction oracle.
- `OGAR/crates/ogar-dismech` declares seven LOCAL **mechanical Sudoku search verbs** at `0xA3..0xA9`: `CANDIDATES`, `FOLD_XREF`, `NAKED_SINGLE`, `HIDDEN_SINGLE`, `ELIMINATE`, `FORK`, `RESIDUE_BAND`. The symbolic vocabulary and conformance tests are **not evidence that a full generic constraint-solving execution engine is already shipped**. `FORK` keeps all alternatives, not an if/else or a posterior sampling step. The sparse 3-bit residue denotes search state / where to look, not evidence.
- `lance-graph-planner/src/dismech_candidates.rs` has an executable but narrower **question-scoped** candidate mask fold. `Supports::Partial` and `NoEvidence` do not eliminate candidates, and a question mask must not rewrite the canonical field.
- `CerebroCortex-RS` PR #1 proposes ACT-R production eligibility, conflict resolution, witness-checked transitions, learned procedural utility, and differential parity with existing activation, spreading, and FSRS. Those are currently **proposed**, not a completed ACT-R controller.
- `dismech-rs` is explicitly the upstream semantic oracle and must **stay independent/zero-dependency**; OGAR owns its integration vocabulary. Agnosticize the Sudoku-style **operations at their common contract / planner binding**, not by importing grid physics into `dismech-rs` or embedding an application-specific ontology into `perturbation-sim`.

### 9.2 Abstraction: one typed experiment proposal with domain-specific physics

Use a small, optional **cold-bound generic experiment surface** above existing simulators, *not* a second scheduler, graph evaluator, universal SCM, LLM agent, mutable Graph/JSON AST, or new R2IL opcode.

```rust
// Architecture sketch ONLY. Reuse canonical identity/witness types.
// The owning generic method contract must be dependency-acyclic.
struct ExperimentRequest<P, D> {
    model: ModelHandle,                // domain, revision, capability declarations
    population: WitnessedPopulation,  // address/row universe + visibility + version
    parameters: P,                     // declared typed/scaled parameter bundle
    design: D,                         // domain-owned, authorized intervention/probe kind
    random_stream: RandomStreamId,     // fixed replay seed or deterministic sentinel
    budget: SimulationBudget,
}
enum ExperimentOutcome<S> {
    Completed { summary: S, provenance: SimulationReceipt },
    Refused(ExperimentRefusal),
    Inconclusive { reason: ModelLimitation, provenance: SimulationReceipt },
}
// Domain adapter, not a JC core dependency:
trait ExperimentModel {
    type Parameters;
    type Design;
    type Summary;
    fn simulate(&self,
        request: &ExperimentRequest<Self::Parameters, Self::Design>
    ) -> ExperimentOutcome<Self::Summary>;
}
```

The domain adapter binds a model revision, coordinate system, resolved parameters, finite summary shape and observational-unit root. A domain may be fully deterministic; changing RNG seeds is **not** a source of independent outcomes if the model does not use randomness. A bounded sequence of experiments can vary inputs/deliberate perturbations freely while preserving replay and cost accounting. Materialized `Vec` outputs in today's standalone physics oracle are acceptable as **oracle behavior**, not a precedent for an implicit row materializer in the canonical substrate. Production later needs typed handles and measured terminal reductions under legal co-addressed masks.

**Power-grid implementation adapter** (initial oracle-side fixture, no existing code moved): `Grid` revision + balanced injection design + seed edge + `CascadeConfig` + optional `AgeModel`/derating/susceptance. Typed outcomes include `tripped`, `trip_round`, islanding/fragmentation, spectrum, voltage/flow fields and any nonconvergence / capability limits. A parameter sweep is a sequence of requests, not a set of new storage thoughts.

**DisMech implementation adapter:** existing source-qualified causal chain/predicate IDs, allowed candidate population and `chain_counterfactual` single-edge cut **for sensitivity only**; not physically replaceable by `simulate_outage`, not a probability distribution and not do-calculus. An actual stochastic biological dynamics / SCM simulator is **not shipped**. Add it only if a separate falsifiable, provenance-controlled generative model is written and independently validated; do not patch `dismech-rs` (the oracle) to turn it into the simulator.

**CerebroCortex adapter:** frozen scoped memory and access history, retrieval-context intervention, ACT-R/FSRS parameters, deterministic seeded outcome summary or declared retrieval stochasticity, and actual attempted-vs-failed observation logging. The existing behavior remains the reference oracle.

### 9.3 Agnosticize *search semantics*, not the scientific domain

Treat the mechanical Sudoku operations as method vocabulary / typed small algorithms with a proof of admissible coordinate layout:
1. `CANDIDATES`: derive the **question-specific** eligible set over one named population.
2. `FOLD_XREF`: normalize co-referent IDs through a declared and previously resolved equivalence/codebook mapping, **not** a forbidden foreign gather.
3. `NAKED_SINGLE` and `HIDDEN_SINGLE`: constraints can force a unique address/placement given a complete declared unit, but **"forced" means conditional on constraints**, not independently verified truth. Hidden-single needs a declared unit relation, not an unproved join.
4. `ELIMINATE`: propagation can narrow one question's candidates, never silently erase global ontology facts.
5. `FORK`: preserve bounded alternatives; each arm has a stable logical model/hypothesis ID and separate thought version witness. 0 givens + 63 concurrent hypothesis slots; Monte Carlo simulation rounds do not each occupy a slot. Overflow typed-refuses instead of aliasing.
6. `RESIDUE_BAND`: compact trace of the attention/search pattern over existing node addresses; **not an evidence root, probability, Bayesian posterior, CausalEdge64 class or new bit allocation**.

If only `dismech_candidates` implements part of the operations, begin with a differential fixture/owner audit; do not report the other operations as executable until measured. Keep one canonical binding and refuse unsupported xref/foreign joins.

### 9.4 'A little ACT-R': a policy for choosing the next experiment

ACT-R-inspired control is **orthogonal to constraint solving**. It does not replace JC Bayesian methods:
- **Match:** evaluate eligible productions (`retrieve_context`, `narrow_candidates`, `probe_perturbation`, `compare_models`, `wait_for_observation`, `abstain`) from witnessed goal/buffer state and an authorized, question-scoped candidate set. The productions are **planned**; do not claim they are already in CerebroCortex.
- **Select:** arbitrate competing eligible steps with independently declared *activation*, *learned procedural utility*, *execution cost* and, **optionally after calibration**, JC's estimated **expected information gain**. These units cannot be compared by adding unnormalized scores; specify a cold policy and verify ranking stability/ablation. NAL confidence and CE64 topology are not selection utility or model probability.
- **Fire:** use the existing planner + domain experiment adapter; check source version, scope, budget and any effect authorization. A simulated edge trip is a calculation, not authorization to act on a real grid or a patient's care. Simulation outputs are tagged `SimulationOnly`.
- **Learn:** only after an external, independent validation receipt may the policy update appropriately witnessed procedural utility. A simulator prediction or OGAR search residue by itself is not independent ground-truth reward. Track expected vs observed efficacy and retain failures.

The ACT-R layer must *not* cause the experiment engine to launch an autonomous agent, create a second `FlowRunner`, or use Graph-flow for recursive pure numerical loops. For long-running external experiments, use existing `graph-flow-lance` handles and an explicit human/authorization gate.

### 9.5 First vertical experiment: three models, two probes, one witnessed question

**Grid test:** take a fixed small grid and compare three declared hypotheses about a later cascade: **H1** topology/bridge weakness, **H2** threshold derating, **H3** heterogeneous asset assumptions. Present two admissible designs: a bounded line-trip stress test and a bounded injection-pattern variation. The Sudoku method restricts legal candidate/design combinations; the ACT-R method chooses a cost-bounded probe; the domain simulator predicts `fraction_tripped`, `islanded` and `node_field`; `jc::sbi` infers which parameterizations remain plausible and optionally estimates EIG. Compare against a held-out *external* outage observation if available; synthetic test data alone only validates the inference method conditional on the grid simulator.

**DisMech analogue:** choose one actually grounded known-mediator fixture plus an `IndirectUnknownIntermediates` restraint control. Treat edge cuts as *sensitivity probes* only; never identify an unknown mediator by the fact a cut influenced a replay verdict. A future independently specified biological perturbation model is a separate domain adapter, not the present `perturbation-sim` physics.

**Behavioral analogue:** test ACT-R recency vs associative spreading on witnessed memory retrieval attempts, where the discriminative experiment is a cue or timing change, not a graph-edge trip.

For all three, compare four schedulers with **matched total computation and identical authorized candidate set**: FIFO, current planner/Moore/Staunen/DAV heuristic, ACT-R utility alone, and ACT-R + calibrated JC-EIG. Record held-out calibration/identification gains, compute cost, failure/abstention rate and independent observation roots. An ACT-R or EIG improvement is **hypothesis**, not an earned result. Negative-control: a high simulator fit from circular, reused outcomes must not pass a held-out predictive test.

### 9.6 Additional staged work

- **D-JCSBI-EX0:** dynamic experiment input/output contract + actual OGAR Sudoku operation-vs-implementation inventory, both in docs/tests. Prove no new scheduler or second math owner.
- **D-JCSBI-EX1:** implement a consumer-side adapter around the existing `perturbation_sim::simulate_outage` oracle; change `AgeModel`, `CascadeConfig` or design deterministically with replay. Return bounded typed results and model limitations.
- **D-JCSBI-EX2:** implement only justified *co-addressed* candidate/method matching with old `dismech_candidates` oracle. Validate `FOLD_XREF`, hidden-single, elimination, fork and residue separately before claiming executable coverage.
- **D-JCSBI-EX3:** introduce **optional ACT-R conflict-selection policy** with independent activation/utility/EIG and explicit witness/budget gate; differential against CerebroCortex P0. No ontology rewrite.
- **D-JCSBI-EX4:** calibrate competing grid models through JC-ABC; check against exact synthetic reference, then independent held-out empirical data where available. Compare to ACT-R retrieval-domain calibration and DisMech edge-cut restraint controls.
- **D-JCSBI-EX5:** measured, dependency-reviewed SIMD/SoA admission if demand is real; existing standalone `perturbation-sim` should remain usable as an independent oracle.

**STOP** on: a generic power-grid algorithm masquerading as biology; the `dismech-rs` oracle acquiring OGAR/planner dependencies; a declared vocabulary operation assumed to be implemented; optimizer or coordinate layout bypass; unconstrained parameter and design dispatch; a stochastic posterior minted from deterministic reseeding; circular outcome validation; ACT-R utility/activation confused with NAL evidence; and simulator-derived results promoted to empirical observation.

## Connected sources from #1601–#1617 (2026-10-20)

Docs and data merged in #1601–#1617 that this plan owns. They are reading notes or reference data, not implementation; each now links back here. Board: `.claude/board/entries/2026-10-20-prs-1601-1617-plan-links.md`.

- [`.arxiv/2210.07278_meta_uncertainty_bayesian_model_comparison_vs_ce64_mul.md`](../../.arxiv/2210.07278_meta_uncertainty_bayesian_model_comparison_vs_ce64_mul.md) (#1611): Meta-uncertainty in Bayesian model comparison: distinct from CE64 MUL
- [`crates/lance-graph-act-r/README.md`](../../crates/lance-graph-act-r/README.md) (#1615): lance-graph-act-r
