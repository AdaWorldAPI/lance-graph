# D-SAG-2 — Ruff × ONA software-awareness closed-loop validation

**Status:** RESEARCH / IMPLEMENTATION PLAN, not an implementation. **Base:** merged lance-graph #1509, main merge 42fa2a347cda22eaaa327bba514f2029cc9daa82 (2026-10-12).
**Owner of authority:** existing OGAR action gate. **First executor:** existing Quack → mask-risc; R2IL FoldDialect is the alternate frontend. **Independent source oracle:** AdaWorldAPI/ruff.
**Prompt:** [software-awareness-graph-v2-claude-code.md](../prompts/software-awareness-graph-v2-claude-code.md).
**Research shelf:** [2026-10_software_awareness_ona_ruff_research_excerpts.md](../../.arxiv/2026-10_software_awareness_ona_ruff_research_excerpts.md).
**Evidence contract:** [software-awareness-provenance-parity.md](../knowledge/software-awareness-provenance-parity.md).

## 0. Research question, answer threshold, and limits

Can a software graph that already queries its own compiled methods: (1) preserve independent, versioned observations; (2) make a **registered prediction before a code change**; (3) compare that prediction with an independent Ruff harvest or runtime oracle; and (4) change a subsequent prediction or experiment selection **because of** the observed mismatch?

**Success is a measured closed loop, not self-awareness as consciousness:** a prediction is committed before the intervention; blind reharvest supplies observations; a controlled update affects the next decision, with an ablated no-feedback twin as the causal control. If recording an event does not alter any subsequent choice, it is instrumentation only.

**Non-goals:** no new graph engine, second method registry, automatic self-rewriting, new NARS semantics or CE64 bit allocation, inferred permission grants, unpinned external runtimes, or performance claims from theoretical equivalence.

## 1. Ground truth: #1496–#1509 in dependency order

| PR | Actual shipped state | Role for D-SAG-2 |
|---|---|---|
| #1496 | docs: CE64 hypothesis layers/NAL polyfill; unresolved physical-vs-semantic reading conflicts | keep byte-level ISA immutable; readings versioned |
| #1497 | docs: PLN/Shannon/EIG and attention falsifiers | candidate schedulers; **not** truth laws |
| #1498 | docs: Hyperon/MeTTa fold parity | preserve multiplicity, binding, nested values and snapshot semantics |
| #1499 | docs: NARS self, evidence provenance/fission and snapshot | independent contributors, revision/ablation semantics |
| #1500 | docs: Heckhausen/Gollwitzer Rubicon | revocable commitment distinct from hypothesis |
| #1501 | docs: Class × Schema × Methods HotPlug | existing inventory/binding seam; no second registry |
| #1502 | docs: GPO/Intune/Linux policy baseline; no policy reader implemented | configured ≠ assigned ≠ observed |
| #1503 | code: dir-sim membership-path witnesses; Held/NotHeld/Undetermined/NotDeclared roles | missing evidence must not become negative authorization |
| #1504 | code: class-qualified MethodTable → Quack/R2IL; NarsRevision law | execution contract and independent row oracle |
| #1505 | code: native CPU default and portable v3 config | pinned build tier/performance fairness |
| #1506 | code: compute DAG Kahn over all 256 field positions | dependency cycles and wide-field gate |
| #1507 | code: runtime text declarations → same MethodTable prepare path | source change without recompiling executor |
| #1508 | code: Gremlin V/has/where/values/dedup/count/sum; predicates inlined at prepare | caller/callee relationships; relation hops refused |
| #1509 | code: SoftwareGraph, typed tiers, compiled programs, Quack one-hop queries, H1 candidate | direct base, **not** a general introspective reasoner |

Confirmed seams:
- glove-parity/src/method.rs: MethodKey is caller-issued; schema seal is local FNV over ordered basis; ReadingId and Universe gate calls; Action is named but pure dispatch refuses it.
- glove-parity/src/gremlin.rs: NoRelationCarrier for in/out/both; effects have no executable Gremlin steps.
- glove-parity/src/awareness.rs: SoftwareGraph projects MethodTable; edge table Binder → Draft → ScatterOrU32; provenance tiers Declared/Derived/Compiled/Verified/Hypothesis.
- glove-parity/examples/awareness_loop.rs: H1 is a **candidate**. Existing tiled masked sum was reported ~1.59–1.78× scalar reference; fused SIMD terminal and post-fusion improvement are NOT measured.
- #1509 plan acknowledges generic multi-hop, canonical cross-build digest, and durable source revisions as unfinished.

## 2. Proposed contract: typed observations without competing authority

Conceptual records only, **not** an ABI or new CE64 reading:

~~~text
ObservationReceipt {
  id, source_kind: RuffHarvest | Compiler | RuntimeOracle | Benchmark | Policy,
  source_revision, snapshot_revision, operation_or_method_key,
  schema_seal, reading_id, universe, evidence_source_id, derivation_id,
  observed_relation_or_outcome, observation_grade,
  evidence_independence_key, collector_version, environment_fingerprint
}
PredictionReceipt {
  id, preregistered_at_revision, frozen_input_snapshot, claim,
  expected_delta_or_result, eligible_candidate_set, method_and_rule_version,
  independent_oracle, abstention_policy, budget, decision_log
}
ResultReceipt {
  prediction_id, oracle_revision, matched, unexpected, missing, unknown,
  truth_or_semantic_parity, costs, control_arm, observed_counterexamples
}
~~~

Do NOT map collector verdict directly to compile-tier or verified-tier. Existing source proof and independent oracle evidence are different authorities. Keep "source declared", "derived from AST", "compiled", "oracle verified" and "hypothesis" separately. An external Ruff extraction cannot certify a runtime behavior without the independent runtime test.

Identity must be scoped: (repository, revision, language, source-span-or-symbol, concept, seal, reading) → stable correspondence to method/class identity; **do not infer MethodKey from method label**. Preserve overload signatures, C linkage, Odoo qualified identities, and Ruby event order as appropriate. Introduce a mapping **adapter** only after proving exactness of all supported predicates and an explicit residual category for unsupported facts.

Avoid version-blind provenance merges. Full graph snapshot and delta semantics should be decided by a falsifier before changing storage. When using hypothesis overlays, identical evidence seen in multiple layers must not add independent confidence.

## 3. Audit first: three probable failure modes, all UNVERIFIED

**A. Shared-edge provenance.** In awareness.rs, link() deduplicates on (src,rel,dst,tier), excluding ctx/prov. If two method contexts share a field→claim edge, reprojecting one method can remove a witness or leave ownership missing. Reproduce with two contexts, two revisions, one shared edge. Prove the precise invariant; **do not fix preemptively**.

**B. Compute DAG invalidation.** project_compute() appends ComputesFrom and ComputeCycle verdict edges but does not mirror project()'s per-method cleanup. Reproject cyclic→acyclic and dependency A→B to A→C under the same class. Inspect stale edges and refuse the wrong version. Distinguish historical evidence from current projection.

**C. Oracle trust/certificates.** certify() checks method digest/seal, while caller supplies the test identifier and proof. It does not authenticate the execution receipt. An untrusted caller could submit a certificate with current digest. Mark accepted metadata versus independently verified test execution; require a registered oracle runner to upgrade to Verified. Avoid changing existing API until compatibility tested.

Additional audits: caller-issued MethodKey; FNV(Debug(MethodSpec)) cross-build stability; caller-supplied SAG_REV; temporal snapshot consistency; edge source ownership; duplicate evidence; memory-only graph; execution and graph projection cost (avoid hot-path instrumentation).

## 4. Implementation waves, authorization, and deliverables

### D-SAG-2-0: frozen oracle inventory (no code)

- Pin baseline #1509 merge SHA and Ruff revision, relevant ONA/OpenNARS/Hyperon reference commits before execution.
- Record supported Ruff IR predicates versus residuals by language, with explicit Odoo/Rails/C++/C# limitations.
- Establish independent scalar/original runtime oracles. Record adversarial fixtures before designing answers.
- Run unchanged #1509 tests and benchmarks only in the actual toolchain; do not claim CI from documentation.

Gate: reproducible source ledger and no ungrounded "supported" predicates.

### D-SAG-2-1: provenance and revocation hardening (tests first)

- Falsify or clear issues A–C, including identical shared relations with separate contexts.
- Define authoritative evidence source, derivation, observation, revision, schema/reading and test certificate roles, with no evidence duplication.
- Check snapshot pinning (version k replay remains identical while k+1 appends). Separate historical receipt preservation from current projection deletion.
- Write a negative control: forged Verified, stale seal/body, reinterpreted reading and wrong CPU tier must not be current proof.

Gate: every mutation triggers a red test before a minimal fix; no new engine or broad schema redesign.

### D-SAG-2-2: Ruff → SAG independent read-only observation

- Compare Ruff ModelGraph and emitted Triples for **supported projection only** against SAG mapping; align identity with explicit correspondence map.
- Python/Odoo: inheritance, compute dependencies and view inherit_id/anchors; Ruby/Rails: ordered MethodOre, private helpers, guard/order; C++: overloads, signatures, no overloaded C file-scope static semantics; C#: preserve out-of-proc Roslyn provenance.
- Keep Ruff's reassemble() restriction visible: currently inverse only for the C++ machine-plane **projected form**, not core-7 and OpenProject.
- Never fabricate free-function dispatches_to: Ruff #118 explicitly keeps C free-function calls frontend-local.
- Collect accepted/rejected/residual facts, missing edges, extra edges, identity collision counts, and time/memory cost.

Gate: structural parity only within declared supported subset; unsupported remains Unknown, not False.

### D-SAG-2-3: prediction → independent intervention → revision (the main proof)

- Snapshot revision A. Before a change, emit PredictionReceipt for impact on affected methods, classes, edges or authorization.
- Apply controlled edit in a fixture branch, separately reharvest revision B with Ruff; independently run semantic oracle where needed.
- Compare delta with the prediction; stratify matched/missing/extra/unknown and evidence class; count precision/recall only on complete labeled universe.
- Update a *contextual hypothesis* or experiment priority from the negative/positive receipts, never rewrite compiler- or source-authoritative truth.
- Repeat on a new seed with **identical model, budget, candidate order and fixtures** except feedback enabled versus disabled.

Gate: preregistered prediction changed by evidence; a paired ablation demonstrates an effect on subsequent decision and reports failures/no-effect without inventing improvement.

### D-SAG-2-4: ONA, NARS, PLN, EIG and Rubicon controls

- ONA-inspired operation cycle = context → precondition → expected event → operation proposal → observed event → revision. Trace it with receipts, not consciousness claims. NAL-9 introspective research is not proof ONA implements it completely.
- Preserve NARS evidence independence, with explicit repeated-source/independent-source controls. Jøsang fission is a *separate control law*, never a drop-in NARS replacement.
- Scheduler-only comparison: FIFO, confidence priority, naive Shannon entropy, calibrated EIG and Staunen/Moore. Freeze candidate pool, inference rules, oracle and resource budget. Then freeze scheduler and compare reasoning laws. High-entropy irrelevant distractors must not win by default.
- Rubicon phases: deliberate → commit revocable if-then plan → check authority **at execution** → observe outcome → revise/disengage. A committed plan grants no privilege; OGAR is only DO authority. An Undetermined directory role fails closed.
- Snapshot consistency and branch multiplicity are hard correctness preconditions for Hyperon-like folds/rewrites.

Gate: if no benefit with paired controls, keep scheduling hypothesis falsified and do not add runtime machinery.

### D-SAG-2-5: generic multi-hop and optimization proposal (separate, gated)

- First show the exact two-hop query that #1509 cannot perform with one legal Quack Program, then evaluate #1508 D-CMH-G1-4 **relation carrier** or existing EqU32Via; do not feed an intermediate population mask between programs illegally.
- Compare Odoo/Ruby structural relation paths with dir-sim's existing witnessed BFS as an **independent oracle**, not a replacement engine.
- H1 fusion: output parity across NULL/UNKNOWN, 1/64/65/4097+ rows, bags/sets, no false primitive claim; benchmark at named native and v3 tiers, medians, allocations, decoded bytes and control cost. Show separate performance results for direct current Program, scalar reference and actual compiled fused candidate (if authorized later).
- Candidate life cycle: hypothesized → structurally feasible → semantically oracle verified → multi-workload cost measured → operator-approved or rejected. No self-change application in this plan.

Gate: performance claim only after a real fused program exists and passes the parity oracle.

## 5. Falsification matrix (pre-register before implementing)

| ID | Deliberately adversarial case | Fails if |
|---|---|---|
| SAG2-P01 | two methods share field→claim; reproject one | other owner's witness/provenance lost |
| SAG2-P02 | compute DAG A→B becomes A→C, cyclic becomes acyclic | old current edges/refusal survive |
| SAG2-P03 | fake test name with matching digest | verified-tier accepted as independent proof |
| SAG2-P04 | same method label across class/seal/reading | identities collapse |
| SAG2-P05 | duplicate evidence in two hypothesis layers | belief confidence rises twice |
| SAG2-P06 | replay pinned k while k+1 appends | result changes or physical snapshot drifts |
| SAG2-P07 | Ruff Ruby equal (W,R,X,C) sets, different event order | SAG asserts behavioral equivalence |
| SAG2-P08 | C file-scope static vs C++ static method | semantic conflation |
| SAG2-P09 | Ruff reassemble(core-7) | unsupported roundtrip silently called lossless |
| SAG2-P10 | Odoo view parent edit/extension anchors | delta impact guessed without source witness |
| SAG2-P11 | unknown or unsupported relation | absence reported as proven False |
| SAG2-P12 | multi-hop call dependency | hidden second evaluator or invalid population intermediate |
| SAG2-P13 | 63/64/255 compute field cycle | DAG check misses edge |
| SAG2-P14 | prediction registered only after edit | leaked result counted as forecast |
| SAG2-P15 | feedback disabled twin versus revision-enabled | same choices misreported as learning |
| SAG2-P16 | correlated evidence and irrelevant high-entropy distractors | attention metric rewarded without useful oracle findings |
| SAG2-P17 | staged action with Undetermined role/revoked commit | action permitted by attention/prediction |
| SAG2-P18 | H1 mixed workloads/CPU tier and NULL | speed claim despite mixed/parity failure |
| SAG2-P19 | Hyperon 4-result nondeterminism | 2-result mask incorrectly called equivalent |
| SAG2-P20 | Ruff harvest versus original program behavior | static fact treated as runtime proof |

Use positive/negative twins and disable one guard at a time. Report test names, red-before-fix result, measured environment and orphaned concerns. Unknown is an acceptable **typed answer**.

## 6. Report schema and success metrics

For each experiment publish: exact base SHA, Ruff SHA, fixture/license, source spans, source/oracle independence, proposition, forecast timestamp/revision, affected identities, emitted evidence count, unsupported residues, known truth universe, TP/FP/FN/Unknown, replay check, authorization verdict, budget, CPU tier, allocations, wall/CPU time, number of mask/lane decodes, observation/learning/control ablations, and explicit pass/fail/undetermined.

Never collapse structural correctness, epistemic uncertainty and execution cost into a scalar "awareness score." A score cannot override authority, independent provenance or semantic equivalence. If using MDL, preserve #1509's result: single-condition shared reuse need not improve description length.

## 7. Architectural decisions needing explicit sign-off

1. Source-of-truth for durable method identities, currently caller issued, and schema/readings versioning; OGAR #357 is still a proposal at the #1509 base.
2. Who may issue and authenticate independent test receipts (not arbitrary claim()).
3. Provenance deduplication semantics: (src,rel,dst,tier,ctx,prov) or separate witnesses per logical edge; preserve shared edges, not duplicate facts.
4. Graph persistence and snapshot pinning: memory-only v1 vs Lance receipts; no write-back to compiler authority.
5. Explicit unsupported/residual contract for polyglot AST harvest and #1508 relation carrier.
6. Permission-gated action proposal semantics and rollback, without conflating NARS evidence, Rubicon commitment and OGAR authority.

**Stop conditions:** any incorrectly upgraded authority, double-counted source, non-replayable snapshot, nondeterministic oracle mismatch presented as parity, hidden second engine, or false H1 speedup claim. Repair first; do not widen scope.

## 8. Sources and exact entry points

Internal: [#1496](https://github.com/AdaWorldAPI/lance-graph/pull/1496), [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497), [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498), [#1499](https://github.com/AdaWorldAPI/lance-graph/pull/1499), [#1500](https://github.com/AdaWorldAPI/lance-graph/pull/1500), [#1501](https://github.com/AdaWorldAPI/lance-graph/pull/1501), [#1502](https://github.com/AdaWorldAPI/lance-graph/pull/1502), [#1503](https://github.com/AdaWorldAPI/lance-graph/pull/1503), [#1504](https://github.com/AdaWorldAPI/lance-graph/pull/1504), [#1505](https://github.com/AdaWorldAPI/lance-graph/pull/1505), [#1506](https://github.com/AdaWorldAPI/lance-graph/pull/1506), [#1507](https://github.com/AdaWorldAPI/lance-graph/pull/1507), [#1508](https://github.com/AdaWorldAPI/lance-graph/pull/1508), [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509).

Ruff: [AdaWorldAPI/ruff](https://github.com/AdaWorldAPI/ruff), [polyglot RFC](https://github.com/AdaWorldAPI/ruff/blob/main/.claude/OGAR-POLYGLOT-AST-INTEGRATION.md), [ModelGraph](https://github.com/AdaWorldAPI/ruff/blob/main/crates/ruff_spo_triplet/src/ir.rs), [C++-only reassembler](https://github.com/AdaWorldAPI/ruff/blob/main/crates/ruff_spo_triplet/src/reassemble.rs), [#117 order](https://github.com/AdaWorldAPI/ruff/pull/117), [#118 C linkage](https://github.com/AdaWorldAPI/ruff/pull/118), [#120 Odoo views](https://github.com/AdaWorldAPI/ruff/pull/120).

Primary literature and independent runtimes: see the .arxiv shelf. All external implementation versions are **not pinned for execution yet**.
