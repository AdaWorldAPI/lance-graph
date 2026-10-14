# D-ORACLE-0 — Original-oracle fidelity, auditable thoughts, and algebraic normalization

**Date:** 2026-10-14  
**Status:** RESEARCH PROTOCOL / OPEN ENDS. No oracle parity or end-to-end speedup is claimed.  
**Boundary:** research documentation only; does not modify CE64, DeepNSM-v2, NARS, parser, mask-risc, or ABI.

## 0. Operator intent (governing)

The goal is **not** one blended NAL/ONA/Hyperon/Pearl/Gadamer reasoning model, and not a fresh implementation masquerading as an oracle. Run and freeze the **original implementations one-to-one as independent reference oracles**; keep their native outputs, traces, and non-equivalent semantics distinct. Compare mathematical operations, logic, statistics, arithmetics, abstractions, and operational decisions across roughly **100–150 pre-registered fixtures**, preserving alternative hypotheses in up to **64 auditable Thought alpha channels per evaluation window**.

A separate experiment then asks whether the existing, semantics-agnostic register/ISA, Quack/Datalog, class methods and mask-risc can represent and evaluate each original's *eligible operations* without changing results. Only after fidelity gates pass can H11 test end-to-end solution materialization and scaling. Hyperon-experimental is **already Rust**; first evaluate a native-Rust adapter, never assume a port or replacement is needed.

The three continuing architecture lines:
1. **Logic**: Datalog Logic Objects, reusable Class/Schema/Methods over mathematical primitives; algebraically transform $var **before** $var.think(). Do not resurrect the former object-per-row, pointer-chasing/session scatter-gather execution path; the removal of that path was **not** a rejection of Class Methods.
2. **Intrinsic addressing**: query[$address], Morton coordinates, composed aperture/prefix restrictions, zipper-fold joins. Intrinsic coordinates are not merely a new database index. Do not materialize intermediate extensional relations unless the queried consumer requires them.
3. **Normalization boundary**: many original semantics lower to a small mathematical instruction vocabulary **where mathematically valid**. It is operational-shape normalization, **not** necessarily reduced Shannon entropy of the evidence, and not a license to erase distinctions.

## 1. Original-reference ledger: versions MUST be frozen before measurements

| Arm | Original implementation | Runtime | Native output / primary question | Avoid |
|---|---|---|---|---|
| JAVA-NARS | https://github.com/opennars/opennars | Java | NAL term inference, belief revision, stamps, tasks, expectation | Equating every NARS historical version, or counting stamps only after the fact |
| ONA | https://github.com/opennars/OpenNARS-for-Applications | C | application-oriented NARS, operations, temporal events, procedural predictions | Treating ONA as merely the Java code ported to C |
| OPENPSI | https://github.com/opencog/openpsi | OpenCog/GNU Scheme integration; verify precise runner | needs/drives, activation, action selection; Rubikon comparison is an **explicit hypothesis** | Pretending OpenPsi is a NAL or Heckhausen identity |
| HYPERON | https://github.com/trueagi-io/hyperon-experimental | **Rust** core, MeTTa/AtomSpace | unification, bindings, metagraph rewriting, evaluation | Inventing a universal Hyperon truth or NARS operator |
| LG-CE64 | https://github.com/AdaWorldAPI/lance-graph/tree/main/crates/causal-edge | Rust | existing CE64 truth operators and field contracts | Calling packed Pearl projections proof of intervention |
| LG-DEEPNSM | https://github.com/AdaWorldAPI/lance-graph/tree/main/crates/deepnsm-v2 | Rust | competing grammatical readings, inference pointers, BeliefArena, TEKAMOLO | Quoting PR #1314 exclusions as permanent current state |
| INDEPENDENT-MATH | independent reference formulas and small executable models | language independent | Boolean algebra, interval closure, distinct-source counting, trace equality | Using an implementation under test as its own only oracle |

Capture per oracle: repository URL, immutable commit SHA, tag (if any), license/reuse constraints, compiler/interpreter version, dependency lockfile, hardware, invocation, configuration, seed, full stdout/event trace, structured oracle-specific output, and checksum. Confirm OpenPsi dependencies and output observability before scheduling its fixtures. Do not vendor third-party code without checking licensing.

## 2. One fixture schema; distinct model-native semantics

Minimal conceptual metadata, **not** a mandated serialized ABI:

- Fixture: id, family, task, corpus/synthetic seed, input events with timestamps, context and operation bindings, expected native observables, numerical tolerances.
- Oracle run: implementation + frozen SHA, adapter revision, configuration, deterministic seed, resource budget, original full trace, normalized **comparison view** with an explicit mapping contract, abstention/unsupported reason.
- Auditable thought: slot 0..63 **plus generation/epoch**, hypothesis identity, observation-event IDs, evidence-source attribution IDs, direct/derived distinction, premises and source transitive closure, inference rule + version, original f/c or native truth values, separate interpretative admission, causal proof/witness level, state transitions, address/prefix, time.
- Provenance must distinguish **source** (paper/sensor/submitter), **individual observation receipt** (one dated event), **derivation path**, **shared-root dependence**, **reader-relative echo**, and **interpretive admission**. A source-bit stamp is only a bounded fast filter. Its modulo collisions must conservatively refuse pooling rather than silently assert independence.
- 64 channels = a bounded *concurrent hypothesis window*, **not** only 64 total tests, not 64 universal truth functions, not a proof that distinct evidence roots fit in six bits. Compare more hypotheses through separate windows with stable IDs and no aliasing.
- Keep the original native output alongside any normalized view; normalize only when a declared homomorphism/correctness mapping supports it. Unsupported/non-comparable is a valid recorded result, not a failure to hide.

## 3. Non-interchangeable operations and semantic firewalls

| Concept | Correct boundary | Specific falsifier |
|---|---|---|
| Literal F/C counting | f = w+/w, c = w/(w+k) subject to evidence definition and k | Repeat one source through a second inference path |
| NAL deduction/induction/abduction/revision | Specific Pei Wang inference formula, evidence dependency, stamps, choice | Double-counting or loss of premise lineage |
| ONA native task/action processing | ONA output/state transitions under same native inputs | Treat ONA runtime scheduling as identical to Java NARS |
| Shannon entropy | H = -sum p log2 p of **declared random variable** | Substituting c or f for unspecified Shannon distribution |
| Gadamer-inspired interpretation | Reader/context-specific admission or changed horizon; local engineering policy, **not** an original numeric Gadamer theorem | Rewriting original evidence weight when only interpretation changed |
| OpenPsi motivation | Needs, utility/control and action selection | Calling a Rubikon stage an existing OpenPsi API without mapping evidence |
| Hyperon rewriting/unification | Original MeTTa binding/unification/evaluation semantics | Replacing rewrite multiplicity or bindings with CE64 f/c |
| Pearl causal ladder | Association vs intervention vs counterfactual require distinct **causal identification and proof conditions** | Label SO/PO/SPO bits alone as L1/L2/L3 proof |
| Grammar case / TEKAMOLO | Linguistic role and direction (LOCATION/GOAL), not causal intervention by syntax alone | Akkusativ automatically dispatches do() |
| Class method / primitive | Method can compose $var symbolically, before .think(), and lower to existing ISA | Pointer-chasing object-method runtime quietly reinstated |
| ISA carrier / contextual reading | Stable bytes, explicit law/reader, physical kernel semantics-neutral | Changed meaning silently changes opcode or violates RegisterLaw |

Distinct comparison kinds: **exact formula parity**, **logical conclusion equivalence**, **trace/operational parity** where meaningful, and **structural expressibility**. Never average heterogeneous results into a single accuracy number.

## 4. Current local evidence and urgent loose ends (grounded 2026-10-14)

1. **X17 unresolved in visible code path, regression needed (BLOCKING).**
   - PR #1319 measured derived A→C from sources 0 and 1 at c=0.6561 and **empty stamp**; offering source 0 again produced c=0.9160 with insufficient provenance.
   - Current crates/deepnsm-v2/src/belief.rs has an appropriate disjoint-stamp guard for *nonempty* stamps, but close_transitive still constructs pure-derived entries with Stamp::default() while recording premises. Thus a direct observation can be pooled with already-used underlying evidence. Test red first, follow derived provenance through premises, distinguish empty/unknown from proven disjoint, and do not modify NARS without an authorized later code PR.
   - Study planner copy, ndarray nars_revision, deepnsm BeliefArena separately.
2. **CE64 saturation defect (BLOCKING for affected truth parity).** crates/causal-edge/src/isa.rs truth::evidence_weight saturates at f32::MAX; combining two saturated revision operands can overflow to infinity and pack NaN as zero. Existing code comments explicitly pin the defect. Pre-register the reference expected result and avoid silent clamp equivalence claims.
3. **Grammar-vs-Pearl mask confusion (BLOCKING).** crates/deepnsm-v2/src/fsm.rs answered_questions returns causal_edge::pearl::CausalMask to represent syntactic S/P/O question completion; crates/causal-edge/src/pearl.rs names SO/PO/SPO projections as levels of causal reasoning. Shared 3-bit **shape** does not grant an executed Pearl intervention or counterfactual. Require distinct typed readings and proof-witness gate; CE64 intervention/counterfactual intentionally refuse.
4. **Sim transitivity (PROBE).** belief.rs permits Copula::Sim blanket transitive closure; generic similarity is not transitive. Counterexample A~B, B~C, A not~C, then check the **declared** weaker notion actually intended. Arbitrary Rel verbs appropriately do not auto-transit.
5. **DeepNSM status freshness (FIX DOCUMENTATION ASSUMPTIONS).** PR #1314 explicitly excluded morph/TEKAMOLO/NARS at that 2026-10-03 wave. CURRENT main already exports belief, reason, tekamolo, hydrate etc. tekamolo::read_clause holds LaneHypothesis to right corner but ClauseRead exposes committed lanes plus counters, not a complete durable hypothesis ledger. Differentiate built, prototype, contract-only, candidate and absent.
6. **Reasoning trace/provenance shape (PROBE).** reason.rs carries premises and rungs; hydrate.rs preserves addressed triples and non-deduplicated revisit scanpaths. Determine whether each trace can round-trip to auditable Thought identity across transformations and method calls without mixing observation roots.
7. **144-verb palette (HOLD).** PR #1227 records 12x12 roots/tenses, 12x12 slot priors, and 6x24 actual DnVerb addresses with incompatible membership. Predicate palette still missing in #1294 (P=0). Resolve authority by explicit compatibility mapping, not silent remap; 12x12 factors only represent rank-one 144-cell fields unless enlarged.
8. **Register laws / anaphora (PROBE).** #1410 has signed i4 and RelativeOffset vs AxisPosition vs Support law binding, refuses overflow; i4 local window is not a confidence value or global graph pointer. Hold i4 anaphora and predicate factors as separate typed readings, compatible shared arithmetic kernels where legal.
9. **Dialectic engine and old class-method runtime (AUDIT).** .claude/plans/dialectic-engine-v1.md has S1: Boolean reachability first, NARS truth second pass; S2 unique-statement dedup; S4 stamps. Verify these exact invariants with original oracles. Removing prior session pointer-chasing scatter/gather class executor (#1524 lineage) did not abolish class methods as an abstraction; method-on-$var should be symbolic and lower through existing Quack/R2IL/mask-risc.
10. **H8/H9/H11 sequencing.** H8 semantics-neutral shared primitives, H9 shared-coordinate zipper-fold, H11 end-to-end materialization with logic, addresses and normalization. None of these benchmarks proves oracle semantics. H11 needs oracle/fidelity gates first and all costs (preprocessing, maintained inverses, updates, proof witnesses, scratch masks, terminal payload) charged.

Primary workspace references: #1314, #1319, #1321, #1407, #1408, #1410, #1227, #1294, #981; above paths are evidence **to be rechecked at frozen SHA**. Old PR-body findings are not proof of current absence.

## 5. Proposed 144-fixture matrix: candidate allocation, NOT completed tests

| Group | Count | Gold oracle / invariant | Adversarial emphasis |
|---|---:|---|---|
| A. NAL truth / syllogisms | 26 | OpenNARS Java + independent formula oracle | deduction, induction, abduction, choice, revision, boundary f/c, quantization |
| B. Evidence roots / audit | 20 | native stamped traces + independent provenance oracle | X17, diamond dependence, duplicate citation, modulo-64 collision, derived→observed |
| C. ONA operational reasoning | 18 | ONA native execution | task scheduling, events, procedural steps, prediction under budget |
| D. OpenPsi / Rubikon mapping | 16 | OpenPsi native traces + separately specified Heckhausen/Gollwitzer mapping | need conflict, action gating, volition, feedback, no false equivalence |
| E. Hyperon/MeTTa semantics | 20 | hyperon-experimental Rust interpreter | bindings, matching, rewrite order, nondeterminism/alternatives, multiplicities |
| F. Grammar / interpretive hypotheses | 16 | DeepNSM-v2 + independently annotated linguistic cases | PosSet alternatives, TEKAMOLO abstention, i4 window, German case, no automatic do() |
| G. Pearl vs native projections | 16 | independently specified causal models / formal counterexamples | confounding, intervention, counterfactual, observational equivalence |
| H. Class, Datalog, address algebra | 12 | reference relational interpreter + equivalence proof/test | $var symbolic composition, joins, interval closure, thought channel independence |
| **TOTAL** | **144** |  |  |

Each row needs actual input, expected output, exact source implementation SHA, executable invocation, provenance/output trace, tolerance, and a kill condition. These counts are proposed budgets, not fabricated executed cases. If an original system cannot express a cross-system fixture natively, record NON-COMPARABLE and put it in a genuine native-domain fixture instead. Oracle alignment is not forced.

## 6. Pre-registered kill probes

- **F1/X17**: same source enters as premise and fresh event; c MUST NOT increase by reusing that evidential root. Empty stamp != independently disjoint. Contrast genuinely new independent source.
- **F2**: two structurally different derivations share one observation; no double count. Record ancestry even if fast stamp is folded.
- **F3**: source IDs 0 and 64 collide in the 64-bit filter; may conservatively refuse, never falsely pool.
- **F4**: high c1,c2 CE64 revision under near-1 confidence produces no NaN-to-zero; if it does, mark FAIL and halt parity claims.
- **F5**: complete syntax S/P/O does not infer do(X) or counterfactual. Causal counterexample with confounder.
- **F6**: similarity nontransitivity counterexample must not become a generally valid rule.
- **F7**: syntactic alternatives all survive until an admitted criterion resolves them; ledger shows rejected and abstained alternatives as well as winner.
- **F8**: reproducing a Hyperon binding/rewrite MUST preserve native result multiplicity and ordering when observable; no forced NAL truth object.
- **F9**: ONA and Java NARS differences reported by version and event trace; non-parity does not automatically mean one is wrong.
- **F10**: Rubikon mapping does not declare motivation equivalent to F/C; test counterexample with high-confidence knowledge but low need/drive.
- **F11**: symbolic $var restrict/intersect/zip can precede think without object-per-row allocations; trace result independent of physical representation; account for any auxiliary indexes.
- **F12**: 64 simultaneous thought slots independent with versioned stable IDs; channel reuse cannot conflate provenance.
- **F13**: preservation of NAL f/c through normalization depends on ordering and evidence ancestry, not blanket associative mxm; check S1 two-pass rule.
- **F14**: end-to-end output boundary: many valid answers must incur corresponding serialization/payload cost; sparse answer case should show avoidable intermediate materialization.

## 7. Sequencing and explicit non-goals

**Gate O0:** Inventory and frozen originals; reproduce each original's own upstream smoke tests; record external build/install/license costs, skip with reasons where unavailable. Do not silently substitute clones.

**Gate O1:** Produce fixture ledger and independent gold outcomes in native oracles. Start with F1, F4, F5, F6, F8, F10; keep critical failures visible.

**Gate O2:** Map native semantics through typed adapters to existing Class/Schema/Methods/Datalog/Quack/CE64. Lower only legally equivalent operations, record refused or unmappable semantics. No new "universal truth value" or model-specific opcode inserted into the immutable byte ISA.

**Gate O3:** 64 parallel auditable Thought channels with observable event/provenance identities; test collisions, repeated sources, branch differences and replay. All 144 fixture outcomes reproducible with single commanded run documented (multiple batched windows fine).

**Gate O4:** Only then the H8/H9/H11 optimizations and an 8-arm 2^3 E2E factorial ablation: logic/methods, intrinsic address math, normalization. Include independent row-at-time reference, baseline engines where practical, parse/ingest, normalization, seal, maintained inverses, scratch masks, updates and final materialization. Report both latency and physical work, as well as semantic correctness. A speedup without fidelity is FAIL.

**Not authorized in this PR:** implementing/fixing X17, altering CE64 truth/ISA/PEARL bits, changing the 512-byte row envelope, adding a new class method executor, claiming human-level intelligence, or claiming universal entropy/performance invariance.

## 8. Research questions / open decisions

- Which OpenNARS Java commit best represents Pei Wang's chosen canonical truth rules, versus separate revision/version variants? Freeze each compared lineage.
- Which ONA C runtime flags, event scheduler, NAL level and reproducibility seed can be pinned?
- Can OpenPsi's OpenCog/Scheme environment be reproduced with an archived dependency image? Which actions/needs are externally observable?
- Which subset of MeTTa evaluation/bindings is stable enough for native Rust FFI/crate linkage versus process protocol? License/cross-version checks first.
- What exact audit identity persists when a derived belief becomes directly observed? Keep original trace while preserving uniqueness of conclusion storage.
- Is a 64-bit source mask enough only as fast conservative gating while exact provenance lives in evidence receipts/graph? How do we track source vs event vs shared underlying root?
- Which operations and outputs are genuinely comparable cross-oracle? Where does abstention/non-comparable become mandatory?
- Is Gadamer here an *admission policy* or genuinely a *revision of contextual understanding*? Treat separately from numeric evidence and specify falsifiers.
- How to prevent syntactic answered_questions() and Pearl causal proofs from sharing an implicit semantic enum?
- Can Datalog represent method logic and retained competing alternatives with $var manipulated symbolically before $var.think(), lowering without new pointer-chasing executor?
- Does algebraic normalization preserve model observables under high/low entropy distributions, without collapsing model-specific information?
- On what narrow conditions can the 12×12 predicate field be compact factorization vs requiring arbitrary higher rank?
- What is the correct source of truth for OpenPsi / Rubikon correspondence: program behavior, published models, or a new explicit adapter?

## 9. Handoff

See the accompanying .claude/knowledge/oracle-fidelity-reference-ledger-v1.md for source URLs/status and .claude/prompts/d-oracle-0-claude-handoff.md for the executable read-first prompt. This PR only registers the research and unresolved decisions; no measurements were performed by committing this document.
