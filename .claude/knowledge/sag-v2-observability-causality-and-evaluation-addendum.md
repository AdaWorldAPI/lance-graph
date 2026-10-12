# D-SAG-2 supplemental contract: operational observability, causal feedback, metamorphic validation, and resource bounds

**Status:** research and falsifier expansion after #1510's initial documentation commit. **Non-authoritative.** No production behavior, CE64 bit, ISA layout, method registry, policy grant or external action is changed or approved here.

**READ BY:** Claude Code implementing D-SAG-2-1..5; SAG SoftwareGraph, Ruff AST/SPO harvest, Quack/R2IL, OGAR/dir-sim/HotPlug, and any ONA-inspired feedback or evaluation operator.
**Parent:** [D-SAG-2 plan](../plans/software-awareness-graph-v2-ruff-ona-closed-loop.md). **Execution instructions:** [D-SAG-2 prompt](../prompts/software-awareness-graph-v2-claude-code.md). **Sources:** [.arxiv annotated follow-up](../../.arxiv/2026-10_software_awareness_ona_observability_metamorphic_holdout_excerpts.md).

## 1. Not-yet-shipped questions and why they change the experiment

The original D-SAG-2 deliverables defined predictions and independent observations, but did not fully formalize when **silence** is evidence, how repeated adaptive tests are prevented from overfitting, whether a feedback update is *causally* responsible for later decisions, how non-operation can be a reference action, and what it costs to introspect the introspector itself.

**Six missing contracts**, all proposed:
1. A negative observation requires a covered, complete, current and functioning observation window.
2. A source-code prediction is not a causal effect estimate merely because it precedes an edit.
3. Feedback credit goes to the originating evidence source, not every derivative graph edge or repeated harvest.
4. The agent must be compared with identical no-feedback, shuffled-feedback and no-op arms on new, held-out edits.
5. Semantic equivalence of programs requires relations over input/output, including ordering and multiplicity, not only triple/set counts.
6. Introspection consumes a fixed, measured compute and memory budget and cannot recursively certify itself.

## 2. The absence/unknown contract: **not observed** is not **observed false**

ONA's 2020 architecture describes anticipation and evidence adjustment through *Assumption of Failure* (see paper §3, PDF printed p. 5). That is a mechanism in ONA, **not** a license to assume closed-world completeness for SAG. Static harvesters may miss unsupported syntactic constructs; traces may sample events; the time window may not have closed. Therefore define the proposed test model:

```text
CoverageReceipt {
  collector_id, version, source_revision, snapshot_id, language,
  predicate, scope, feature_support: Supported | Partial | Unsupported,
  scan_completed, scanned_units, excluded_units, errors,
  observation_window_start, observation_window_end
}

PredictionStatus {
  Confirmed, Refuted, Pending, NotObservable, Incomplete, Stale
}
```

**Admission rule** for negative evidence: predicate in declared supported set AND exact population covered AND observation end/window satisfied AND scan completed without errors/exclusions AND pinned snapshot valid AND oracle independent enough for the claim. Otherwise return `NotObservable`, `Incomplete`, `Pending`, or `Stale`, and do not change prediction confidence based on absence.

| Source B outcome | Collector contract | Legitimate inference |
|---|---|---|
| positive hit with matching source witness | supported, fresh, pinned | Confirmed, subject to semantic precision |
| no hit | supported + complete + closed window | Refuted *for that scoped proposition* |
| no hit | unsupported C free-function call edge in Ruff SPO | NotObservable; **not** proof of no call |
| no hit | Odoo XPath predicate unsupported / parent outside tree | Incomplete; not proof of no control |
| no hit | sampled runtime trace misses occurrence | Incomplete unless trace oracle establishes completeness |
| no hit | edit/forecast still in progress | Pending |
| no hit | B reading/seal/version differs | Stale or schema incompatibility, never evidence for current version |

A predicted *negative* (no impact) is also a real prediction. Use a no-op edit and a semantically inert edit to establish the baseline false-positive rate.

**First red tests:** deliberately drop one source file after recording a prediction; switch a supported predicate to unsupported without changing code; abort a harvest midway; feed a late event after the observation window; and reorder arrival of equivalent receipts. No case may silently become a failed prediction or a confidence decrease unless its negative-evidence preconditions are satisfied.

## 3. Time, provenance, and causality are different axes

Keep **source revision**, **observation/event time**, **ingestion time**, and **decision time** distinct. They are not always equal: a journal or reharvest can arrive late; a branch can rewind; receipt ingestion can happen after a newer prediction. Use a total source snapshot identity for replay and a monotonic append index only for append order.

Separate these questions:

- *Who said it?* Collector/provenance and reproducible source artifact.
- *What derived it?* Rule version, input evidence IDs and derivation receipt.
- *When was it true?* Source or device state revision, observation interval, as-of version.
- *When did we learn it?* Ingest/revision sequence.
- *Who was allowed to act?* OGAR action receipt at commit time, independently checked.
- *Would the result differ without the action?* Controlled intervention/no-op twin, NOT observational co-occurrence.

Minimum proposed `EvidenceKey` for de-dup: originating input SHA/record ID, collector, source revision, observation scope, predicate, asserted event ID. **Do not** use graph-edge ID or layer ID alone; downstream copies are not independent observations.

A **causal DAG of receipts** may reference sources and derivations without inventing graph semantic causality for the software being analyzed. Two paths from one Ruff AST token via the same ModelGraph are one evidential base, not two independent votes. Two different parsers of the same commit may be separate implementation oracles for correctness *but are not independent evidence of two world events*.

**Prediction-before-intervention** requires a committed artifact containing (a) exact A revision, (b) frozen candidate/rule set, (c) expected observable delta, (d) source coverage class, (e) scoring rule and (f) machine timestamp or immutable commit order prior to B. Branch A and B names are not enough without commit SHA/hash receipts.

## 4. A stronger experiment than "we predicted an edited dependency"

Four controlled interventions over the **same fixture family**, each preregistered:

| Arm | Mutation | Expected graph property | Why useful |
|---|---|---|---|
| M0 | no source change, rerun harvest | zero semantic delta | oracle, cache and identity stability |
| M1 | whitespace/comment formatting or reorder independent declarations | structural equivalence under declared canonicalization, spans may change | metamorphic normalization |
| M2 | one parent Odoo view / Rails callback / C++ overload change | affected subgraph changes, unaffected subgraph invariant | factual change-impact attribution |
| M3 | misleading new comment, unresolved XPath, omitted module, disabled harvester | abstention/unknown rather than invented witness | epistemic stress test |

Where semantics allow, run **A→B→A'** (revert) and require the declared semantic graph to return to A while historical receipts remain addressed and immutable. A method's display name can change without changing its authorized caller-issued MethodKey **only** if an explicit versioned correspondence proves continuity; never auto-merge by fuzzy name.

Metamorphic relation examples to pin as tests:
- Adding irrelevant whitespace must not change the semantic SPO projection except source spans/hashes.
- Permuting independent declarations preserves graph content while source order metadata may differ.
- Swapping Ruby `write; raise` and `raise; write` can have identical unordered W/R/X/C but **must not** be asserted behaviorally equivalent.
- Duplicate source inclusion with the same evidence ID cannot inflate independent confidence.
- A separately repeated *new* independent observation retains independent weight.
- Under supported semantics, full reharvest(A→B) equals incremental reharvest(A→B); if not, caching is not a proof.
- A→B→A' restores the current semantic projection without erasing historical A/B receipts.
- C linkage `static` and C++ static member remain distinct even if labels match.

For source-only behavior claims, ask for an independent executable test or return `NotObservable`. An AST assertion is not proof of exception ordering, database side effects, or deployment effects.

## 5. Operational loop versus ONA research vocabulary

ONA 2020 keeps semantic and sensorimotor inference apart, uses bounded attention and evidence stamps. For a SAG probe we propose the following **behavioral observation loop**, not a literal ONA port:

```text
read frozen A
  -> propose hypotheses H[0..n]
  -> choose a test within budget (no side effect by default)
  -> commit prediction and observability contract
  -> authorized fixture intervention OR select explicit NoOp
  -> independently obtain source/runtime result at pinned B
  -> classify Confirmed | Refuted | Pending | NotObservable | Incomplete | Stale
  -> revise ONLY eligible hypothesis evidence
  -> choose the next experiment from a frozen candidate pool
```

Observe three classes of operation separately:
- **NoOp**: no source or external change; tests attention and measurement drift. An ONA issue discusses ambiguity around a "do nothing operator" as an implementation question, not a settled semantic analogue for SAG.
- **Probe**: read-only, potentially costly harvest/measurement. Its side effect is **knowledge intake**, not target-program modification.
- **Proposal**: inert suggested code/transformation/action that cannot execute externally before OGAR authorization.

No unbounded action loops, no automatic source rewriting. A decision to probe is not the same as a commitment to mutate. A failure prediction becomes negative evidence **only** if the collector could have witnessed its expected outcome.

**Introspection depth budget:** a meta-query about the current SAG must read immutable snapshot k; telemetry about its own execution may enter k+1, never the same query epoch. Terminate cycles by (snapshot, method key, query signature, recursion depth/remaining budget), and return a typed budget/recursion refusal. No new runtime engine is necessary to articulate or test this rule.

## 6. Adaptive testing can make a self-optimizer hallucinate progress

Dwork et al. (2015) show why repeated access to an adaptively reused holdout can bias apparent generalization. Our finite source-change test sets have the same conceptual risk. A system that learns which Ruff mutations the benchmark contains can score better without transferable software reasoning.

Preregister three disjoint buckets, grouped by **repository + edit family**, not random lines:
- **Development:** visible failures may be iterated on; all changes logged.
- **Validation:** limited, counted selection of policies/thresholds; freeze the chosen candidate and feedback law afterward.
- **Protected test:** never exposed during training/selection; one run of final frozen policy and its ablations.

At protected test time, compare under identical precommitted budget and candidate pool:
- A0: no observation→belief feedback (logging only).
- A1: source-grounded revision enabled.
- A2: shuffled or deliberately stale receipts (negative control; should not improve).
- A3: no-op or heuristic scheduler only (distinguishes information from exploration strategy).
- A4: Oracle-aware abstention enabled versus blindly treating silence as false.
- Optional A5: attention policies FIFO / confidence / naive entropy / calibrated EIG / Staunen under unchanged inference law.

Report **paired decision-quality difference on unseen edits**, uncertainty intervals where sample size permits, coverage/abstention, false-positive/negative rates under complete supported predicates, calibration for any probabilistic score, discovery per decoded byte, and instrumentation overhead. "Different next decision" is a minimal causal check; "better" requires predeclared utility and held-out evidence.

If learning works only when evaluation fixtures are visible, or with shuffled evidence, claim **selection leakage** or a weak control, not a cognitive advance. A null result is useful.

## 7. Attention and CARE: measure the *value of information*, not its aesthetic

A candidate set of maskable unknown cells is useful as a *search agenda*, not ground truth. `popcount(unknown_mask)` measures how many unknowns are available; it does not measure probability, causal relevance or human-like curiosity. Any prospective 50–52 CE64 CARE/motivation reading must remain a **versioned, typed semantic interpretation**, not an assignment of meaning to the ISA bytes; #1496 records unresolved existing interpretations of those bits. No new semantics are shipped here.

A falsifiable resource allocator could propose: `score = E[oracle-supported improvement or error reduction | probe] / expected total cost`, estimated only from calibrated historical observations with no future leakage. Compare with FIFO and random, and explicitly count costs for:
- AST/SPO parsing, projection, binding and query lowering;
- mask/lane byte decode, allocations, branch fan-out and materialization;
- oracle invocations, negative controls, replay, merge and retention;
- wasted probes, abstentions and cache misses.

High Shannon entropy on irrelevant facts must not dominate. Structural redundancy, epistemic uncertainty and execution cost remain distinct axes; hard correctness/security gates dominate any score. If computing exact EIG is unjustified, report that instead of renaming a heuristic "EIG."

## 8. Action boundaries: hard separation of belief, intent and permission

The Rubicon research (#1500) describes human commitment/implementation intentions, not software authorization. Define a *proposed* software boundary:

```text
Hypothesis -> CandidateProbe -> PrecommittedPrediction -> ReadOnlyProbe
          -> OutcomeReceipt -> RevisedEvidence -> CandidateProposal
          -> OperatorCommitIntent (revocable) -> OGAR Action Gate
          -> ExternalActionReceipt -> PostconditionObservation
```

A committed intent can become invalid at any point due to role loss, authorization revocation, schema/reading mismatch, stale test receipt or expired universe. Commit checks must be atomic with authorization or otherwise refuse and re-evaluate. #1503's `Undetermined` is not a grant. The operator's approval of a *research PR* is not authorization to perform a production source rewrite.

Adversarial dry runs: revoke permission between intention and dispatch; switch user's role path to alternate still-valid grant; leave a security group unread; change method seal after commitment; delayed outcome arrives after cancellation. Refuse external mutation without current policy and witnessed capability; do not convert source graph edges into executable actions merely because the same class defines both.

## 9. Versioned hot-plug and graph custody

Avoid a parallel method-mint registry. Caller-issued MethodKey is the identity under #1509. Class/field/readings resolve through OGAR's existing ownership; Quack only binds what the active glove/schema declares. Preserve the difference between:
- logical schema shape;
- hotplug activation/version and current permission;
- source-program revision and compiler build;
- read snapshot/persistent Alpha overlay;
- source code harvest provenance.

**Cross-repository evolution** is a compatibility proof obligation. When Ruff exposes a new predicate, it is not automatically a valid OGAR ontology value. When OGAR adds a method key, it is not a rule to rewrite historic SAG identifiers. Typed unknown/residual is better than casual string fallback.

Proposed minimal artifact manifest (no new crate):
```text
FixtureManifest {
  sag_base_sha, ruff_sha, model_or_method_key, classid, schema_seal, reading_id,
  harvester_features, supported_predicates, excluded_sources, fixture_digest,
  source_A_sha, prediction_receipt_sha, source_B_sha, oracle_receipt_sha,
  collector_version, evaluator_version, cpu_tier, budget
}
```
If any artifact cannot be pinned, report non-reproducible, not verified.

## 10. Probe matrix supplement: P21–P40 (not executed)

| ID | Test/falsifier | Must reject or distinguish |
|---|---|---|
| P21 | complete supported negative observation at B | scoped Refuted allowed |
| P22 | unsupported predicate omitted by Ruff | NotObservable, no negative revision |
| P23 | truncated / failed extraction with omitted module | Incomplete, not Refuted |
| P24 | premature or late event beyond window | Pending/Stale versus negative evidence |
| P25 | no-op source edit plus repeat harvest | zero semantic delta; no fabricated change |
| P26 | formatting/alpha-renaming with explicit identity map | no false semantic diff; no name-based method key |
| P27 | A→B→A' revert, historic receipts retained | current graph restored, history immutable |
| P28 | full versus incremental reharvest at pinned snapshot | same supported predicates/edges |
| P29 | same source via Ruff AST → ModelGraph → SAG | one evidence contributor, no double confidence |
| P30 | same originating event repeated across hypothesis layers | one contributor, not two |
| P31 | two independent sources of same proposition | preserve both contributors |
| P32 | read k while own telemetry appends to k+1 | no recursive contamination or time drift |
| P33 | self-introspection infinite feedback possibility | bounded recursion/typed refusal |
| P34 | no-op operator versus probe operator | separate effect and observational cost |
| P35 | visible-dev success, unseen protected holdout | no hidden adaptive benchmark selection |
| P36 | shuffled/stale feedback control | cannot claim source-grounded learning advantage |
| P37 | mismatch in behavior with equal Ruby unordered sets | behavioral equivalence disallowed |
| P38 | high entropy irrelevant facts vs useful low entropy | report discovery per total budget |
| P39 | role revoked between Rubicon commit and action gate | refusal; intent alone no authority |
| P40 | observation feedback decreases accuracy on new edit family | negative result preserved, not relabeled improvement |

**Kill conditions:** false Verified, phantom negative evidence, provenance overlap counted twice, unbounded introspective recursion, hidden second evaluator, unpinned test reuse presented as validation, unsupported Ruff fact treated as false, or action past a revoked authorization. Ship a narrow failing fixture and stop broadening until repaired.

## 11. References to consult before writing code

- [Hammer & Lofthouse (2020), ONA Architecture and Control, PDF](https://agi-conf.org/2020/wp-content/uploads/2020/06/AGI-20_paper_52.pdf), §§2–3 (queue, stamps, sensorimotor predictions and *Assumption of Failure*).
- [ONA implementation](https://github.com/opennars/OpenNARS-for-Applications) and [do-nothing operator discussion #286](https://github.com/opennars/OpenNARS-for-Applications/issues/286).
- [Dwork et al. (2015), Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629).
- [Li et al. (2024), Metamorphic Relation Generation](https://arxiv.org/abs/2406.05397) and [Duque-Torres et al. (2023), Bug or not Bug?](https://arxiv.org/abs/2305.09640).
- [SAG #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509); [method composition #1508](https://github.com/AdaWorldAPI/lance-graph/pull/1508); [identity evidence #1503](https://github.com/AdaWorldAPI/lance-graph/pull/1503); [NARS Self #1499](https://github.com/AdaWorldAPI/lance-graph/pull/1499); [PLN/Shannon #1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497); [Rubicon #1500](https://github.com/AdaWorldAPI/lance-graph/pull/1500).
- [AdaWorldAPI/ruff #117](https://github.com/AdaWorldAPI/ruff/pull/117), [#118](https://github.com/AdaWorldAPI/ruff/pull/118), [#120](https://github.com/AdaWorldAPI/ruff/pull/120).

**No-code constraint:** this file asks future implementers to falsify before changing behavior. No passage is an instruction to rename classes, allocate CE64 bits, widen the ISA, construct a new Quack primitive, execute production changes or assert ONA implementation parity without a pinned source check.
