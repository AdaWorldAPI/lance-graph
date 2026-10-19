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

Start with an analytically tractable small generative model: Beta-Binomial posterior or equivalent finite enumerated latent model. Implement a bounded discrete grid / exact small-case reference in **test code**, with JC's mathematical operations as the production candidate. Prior, observed data, likelihood where available, posterior normalization, and zero-support refusal must match independent high-precision R/Python reference fixtures. If exact likelihood is available at comparable cost, prefer conventional inference; SBI earns inclusion only when intractable likelihood or simulator behavior justifies it.

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

EIG is expected posterior information gain under the model and design distribution, not entropy, candidate-mask narrowing or EWA disagreement. Use a well-defined estimator (nested Monte Carlo or a vetted likelihood/ratio method) and an independent small discrete-oracle fixture. Separate aleatoric uncertainty, posterior/parameter uncertainty, estimator Monte Carlo error and model-misspecification warning. Only allow objectives after sufficient calibration; when misspecified or unidentifiable, abstain. Include experiment cost and authorized design constraints. Planner proposes next observation; it cannot bypass authority gates or create empirical evidence before executing the experiment.

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

**JC owns:** posterior weights and log normalization; ABC accept/reject laws; ESS formula/finite-sample diagnostics; SBC rank and empirical coverage; posterior predictive discrepancy; model comparison and meta-uncertainty accounting; EIG estimator/MC error; typed statistical refusal semantics.

**ndarray owns only genuinely reusable compute:** feature/summary masks, SIMD masked distance kernels, logsumexp/normalized positive reductions if missing and independently proved necessary, reductions and batched finite numerical primitives used by multiple consumers. First inventory `ndarray::simd`; do not create a duplicate vector backend. A new `ndarray` primitive needs exact scalar mathematical oracle + SIMD backend parity and a measured speedup against existing operations with masks, gather prohibition, bytes, allocations and hardware tier recorded.

**JC pillar policy:** experimental SBI procedures and simulation calibrations do not become a new "pillar" merely by being computationally nontrivial. Keep them as `jc::sbi` research methods with tests and uncertainty reports. Promote *specific mathematically proved/validated numerical invariants*, such as stable normalized weights, finite-bound accumulation, or coverage under a declared reference generative family, via JC's established proof/measurement review. Always distinguish theorem assumptions from empirical Monte Carlo evidence and from general simulator adequacy.

**Existing moment terminals:** use `PowerSums`/`CrossPowerSums` for sufficient statistics where the chosen model actually admits them; they are **not sufficient for arbitrary SBI summaries, multimodal posteriors, trajectory dependence or neural density inference**. Float-valued or foreign-coordinate payloads require separately proven gates.

## 7. Phases (each follow-up its own separately authorized implementation PR)

| Gate | Deliverable | Acceptance / stop condition |
| --- | --- | --- |
| **D-JCSBI-0** | Source+API+sample-unit inventory over JC/ndarray, #1613, #1612, OGAR vocabulary, DisMech replay, CerebroCortex P0 | List actual method/terminal versus proposed; concrete simulator contract and independent oracles; reject false "SCM" claim |
| **D-JCSBI-1** | Cold typed `SimulationSpec`/`SBIRequest`/`SBIReport` contracts and refusal matrix, unit/test fixture only | No JC production dependencies on planner, R, OGAR; scope/version/model-revision necessary; seed and policy replay deterministic |
| **D-JCSBI-2** | Small exact discrete posterior fixture, normalized weights and degenerate bounds | Independent analytical + R/Python oracle parity, negative controls red; no new SIMD op unless missing proven |
| **D-JCSBI-3** | Bounded ABC rejection and diagnostics on synthetic ACT-R/FSRS-like retrieval simulator | Parameter coverage/recovery and hold-out prediction vs fixed config and conventional calibration at matched budget; distinct empirical/sample grouping |
| **D-JCSBI-4** | SBC + posterior predictive + misspecification and missingness diagnostics | Uniform rank/nominal coverage **under correct generator**, deliberately wrong model and biased observation pipeline flagged, explicit abstention |
| **D-JCSBI-5** | DisMech cut replay harness (not SCM), known/unknown mediator strata, source receipt checks | Same replay source for both arms; observed-vs-simulated never conflated; guessing unknown mediators not scored; oracle-matched load-bearing classification |
| **D-JCSBI-6** | Model probability replication variability and ABC-SMC *if* earlier probes justify | Correct weighting/ESS and reproducible held-out stability, simulation-budget error surfaced |
| **D-JCSBI-7** | EIG over permitted experiment choices, optionally compare DAV/EWA heuristic | Exact discrete EIG oracle; better held-out identification or calibrated utility **at equal total simulation+selection budget**, not only lower entropy |
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
