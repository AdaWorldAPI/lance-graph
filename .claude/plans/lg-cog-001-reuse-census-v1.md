# LG-COG-001 reuse census: capability and admission matrix (v1, with v2 §10)

**Status:** CENSUS, docs only. **Date:** 2026-11-24. **Pinned at:** lance-graph `main` 855d051, ndarray `master` (`src/hpc/codec9.rs`, `src/hpc/abakus/`), lance-context `main` (after #3/#4), rs-graph-llm `main` 71450cc.
**Basis:** #2077 is the canonical LG-COG-001 plan. #2078 is harvested here, not merged as a second plan (§5).
**Claim:** EXO-389. **Widens:** D-LFA-0 (`d-lfa-lance-graph-flow-rubicon-assimilation-v1.md`).

This file answers one question per component:
- what exists as code;
- what has been measured in a lab;
- which interface the pilot needs but does not have;
- what blocks production admission.

"Exists" means read at the pinned commit. It does not mean admitted. Nothing here builds a runtime, evaluator, codec, truth store, temporal engine, trajectory store, receipt model or workflow engine.

## 1. Matrix

Grades:
- **CODE**: read in source at the pin.
- **LAB-MEASURED**: a test or probe ran and its result is recorded.
- **DOC**: a plan or research note only.
- **ABSENT**: searched in a named space and not found.

| Component | Code (path) | Lab evidence | Missing interface for the pilot | Production blocker |
|---|---|---|---|---|
| Thought-git slots | `crates/thought-git-probe/src/slots.rs` (`SlotPool`, generations, `Stale`) | `tests/slots.rs` 7/7 pass at 855d051: FIFO and least-used eviction, reuse without pressure, candidate retention on branching | none for a lab pilot | D-TGB-7 crate admission |
| Thought-git squash | `squash.rs` | `tests/squash_gate.rs` 10/10 and `tests/protected_readers.rs` 7/7: squash equals landing every final state; divergence refused and recorded; bare squash loses contradictions, protected squash keeps them | none for a lab pilot | D-TGB-3 landing |
| Thought-git oracle | `oracle.rs` (`next_version`, `Givens`) | `tests/oracle.rs` 9/9: unwritten lane reads absent, not zero; agreement with a given is not stored | none | lab only by design |
| Thought-git receipts (D-TGB-4) | `receipt.rs`: `Receipt`, `capture`, `replay`, `replay_live`, `MetadataTable`, `CommitWitness`, `VersionStore::commit_cycle` | `tests/receipt.rs` 15/15: seed/config/source changes refuse; torn witness detected; a sealed row cannot be replaced; `replay_cannot_raise_evidence_but_a_new_root_can` | `BaseRef` stands in for `DurableRef` | **EXO-148**: `DurableRef` / `ThoughtCommitId` do not exist. No atomicity across data and metadata writes (stated in the module doc). Wiring into `LanceCycleWriter` is D-TGB-7. |
| Replay evidence rule | `receipt.rs:550-597`: `Replayed { inherited_roots: receipt.roots, independent_roots: 0 }` | `tests/signals.rs` 10/10, including `replayed_material_cannot_raise_evidence` and `the_second_landing_of_the_same_thought_earns_nothing` | the pilot must carry `independent_roots` into NAL stamps (§3) | none beyond the receipt row above |
| Temporal horizon | `crates/lance-graph-planner/src/temporal.rs`: `QueryReference::at(v, rung)` (:188), `EpistemicMode::for_rung`: 0–4 Strict, 5–8 Aware, 9+ Retro; `deinterlace`, `classify` | planner tests (not re-run here) | none | `QueryReference::default()` uses `ref_version = u64::MAX`, which admits every row (F-QREF-STRICT, doc at :163-174). **The pilot must use `at(v, 0)`, never `default()`.** |
| Persist sink | `persist_sink.rs`: one WAL write per 64k cycle, write-side stable order, sealed read horizon (`frame.base_version`) | planner tests (not re-run here) | none for a read-only pilot | the pilot writes nothing through it |
| Lifecycle contract | `.claude/plans/thought-git-lifecycle-contract-v1.md` (D-TGL-0…G12) | DOC: "CONTRACT, docs only. Nothing here is wired" | n/a | not wired |
| EAW-7 / EAW-9 | `docs/research/epistemic-awareness/eaw-7-…`, `eaw-9-…` | DOC: EAW-7 "No test executed"; EAW-9 R4 defines teacher withdrawal, BL-14 "withdrawal destroys measured gain ⇒ no rediscovery claim" | the pilot reuses the R4 protocol instead of inventing one | docs only |
| NAL revision | `crates/lance-graph-nal/src/nal.rs`: `Revision` / `PreparedRevision` / `Revised`. `commit` refuses `OVERLAP` (`DependentEvidence`) and `NO_PROVENANCE` (`MissingProvenance`). `Evidence { belief, stamp: &[u64] }` (lib.rs:250) | crate tests (not re-run here) | stamps from thought-git receipts: no adapter yet (§3, seam S1) | NAL plan admission (`nal-polyfill-v1.md`) |
| NSL selection and loss | `crates/lance-graph-nsl`: `FeatureTable`, `Adjacency`, `select`, `FocusLease` (index-seal check on reads), `loss::{graph_loss, Distance}`; `eval.rs` against frozen TensorFlow goldens (`data/nsl_golden.txt`, `data/nsl_loss_golden.txt`) | **MEASURED** (`docs/research/nsl/production-path-evaluation.md`): A label codebook ADOPT (18/19, NaN refused); B1 quantized selection REJECT; C4 residual coding PROBE (strongest) | a pilot ranking signal is optional (D-CFA-4) | selection keeps exact f64 order; quantized order needs a tolerance contract |
| ONNX vocabulary | `crates/lance-graph-onnx/src/lib.rs`: ALIAS / INLINE over native functions, no graph or interpreter, depends on `ndarray` only | crate tests (not re-run here) | none: the pilot needs no ONNX operator | operators outside the crosswalk are NATIVE-TODO or REFUSE |
| D-TFLP | `.claude/plans/d-tflp-tflite-palette-polyfill-capstone-v2.md` (PROPOSAL); lab PRs #1800 (softmax certification, 529 + 27 replay witnesses) and #1801 (8-bit admission, `Receipt`/`Expect`/`bind`, 17 tests) | LAB-MEASURED in those PRs, per the plan's table | none for the pilot | docs/lab only |
| D-CFA-4 | `docs/research/counterfactual-behavior/cfa-4-nsl-onnx.md` | DOC: "cold research option / falsifier preregistration only" | optional | no trained model; no learned evidence authority |
| RLVR curriculum | `.claude/knowledge/neurosymbolic-rlvr-causal-curriculum-v1.md` | DOC: "PROPOSAL … no code yet; ratify §6 before fan-out" | supplies the L-level vocabulary (§2) | unratified |
| Codec9 | ndarray `src/hpc/codec9.rs`: `EpiLane9` (P bit + u8 magnitude, −0 = never touched); `Codec9<R, C>` (compile-time scale); `Codec9Runtime<R>` (runtime scale, checked once); `encode_touched_batch`, `decode_batch` | ndarray tests (#35 in this series: decode-vs-native, `encode_touched_checked`) | none | the module doc says: "Epistemic presence is NOT a source-write or alpha-claim authorization" |
| Abakus pm | ndarray `src/hpc/abakus/pm.rs`: `add_pm`, `neg_pm`, `sub_pm`; a carry lane is `PmLane::Escaped`, never a silently saturated value | exhaustive oracle (task #33) | none | exact counts must stay on the escape path, never saturated |
| Shared-anchor residual coding | ABSENT in `codec9.rs` and `abakus/` (searched for `anchor` and `residual`) | the closest measured mechanism is NSL C4 ("source codes + int8 residual on its own scale", PROBE strongest) | if wanted, C4 is the probe to extend; no new codec | not admitted |
| RolloutStore | lance-context `rollout.rs`: `RolloutRecord` with `rollout_id`, `problem_id`, `sequence_order`, `role`, `reward`, `raw_reward`, `grader_id`, `score`, `advantage`, `policy_version`, `include_in_training`, `relationships`; `rollout_store.rs`: `add(&[RolloutRecord])`, `checkout(version)`, `is_version_pinned`, `refresh_latest`, `flush` | lance-context tests | no typed witness batch API (#2077 Q4); the pilot uses `add` through a cold adapter | **EXO-287 / EXO-300**: not admitted as a durable store (newest write wins; non-atomic existence check; tenant is a filter only). The pilot treats it as experiment memory, never as authority. |
| lance-context id lookups | `id_lookup.rs` + Quack arm (lance-context #3/#4, merged) | 13 tests plus disable runs (this session) | none | same as RolloutStore |
| graph-flow + Rig | rs-graph-llm `graph-flow/src/task.rs` `NextAction` (Continue, ContinueAndExecute, WaitForInput, End, GoTo); `context.rs` `get_rig_messages` behind the `rig` feature; rig-core 0.39 fork as a workspace dependency, aliased `rig` | rs-graph-llm tests (not re-run here) | the Claude call goes through Rig only, at an explicit LLM boundary (D-LFA) | sessions and JSON storage are not truth (#2077 table); storage backends there (Postgres, Surreal) are not used |
| lance-graph-flow | ABSENT (`crates/` at 855d051) | — | nothing: the pilot does not need it | D-LFA-3 stays gated on D-LFA-1/2 |
| Serve surface | `serve.rs` (lab only) | — | the pilot does not use it | the canonical surface is `UnifiedStep` / `OrchestrationBridge` (EXO-323) |

## 2. Learning levels and how the pilot measures each

These are the requester's levels, mapped onto oracles that already exist. A level counts only if its gain survives teacher withdrawal, using the EAW-9 R4 protocol: freeze, remove teacher feedback on held-out items, compare against frozen and reset baselines. If withdrawal erases the gain (BL-14), the pilot reports dependence, not learning.

| Level | What is learned | Existing oracle | Evidence kind |
|---|---|---|---|
| L1 declarative | which method, law or fact applies | Codegraph source signatures; kanban transition table (#2077 §4.6) | historical observation |
| L2 procedural | a method plan that binds and executes | Quack / R2IL / mask-risc reference arms; thought-git `next_version` | hypothetical derivation, replayed |
| L3 evidential | when evidence is independent | NAL `Revision` refusals (`OVERLAP`, `NO_PROVENANCE`); receipt `independent_roots = 0` on replay | provenance |
| L4 strategic | which plan to try first (ordering) | NSL exact f64 selection order; D-CFA-4 ranking falsifiers | retrospective teacher feedback |
| L5 autonomous discovery | gain that persists with the teacher removed | EAW-9 R4 / BL-14 | withdrawal |

The three evidence kinds are kept in separate fields and never pooled:
- **historical observations**, read under `QueryReference::at(v, 0)`;
- **hypothetical derivations**, in thought-git slots 1..63, never landed;
- **retrospective teacher feedback**, revealed only after the learner commits its answer, per EAW-9.

A replayed derivation carries `independent_roots = 0`. It cannot count as new NAL evidence. This holds by construction (`receipt.rs:595`) and is pinned by test.

## 3. Missing seams, ranked

| Seam | What is missing | Smallest honest step | Owner of the blocker |
|---|---|---|---|
| S1 receipt → NAL stamp | `Receipt::roots` is a **64-bit bit set** (`receipt.rs:381-383`: "Provenance roots of those sources, as a bit set"). NAL's `Evidence::stamp` is a slice of root **identities** (`&[u64]`, lib.rs:250). The mask is a **local coordinate**, not an evidence identity: two bases can use the same bit for different roots. Its meaning has to come from a witnessed map. | First check whether the existing DR addressing already expresses this, before designing any new table. `lance-graph-dt64::dr::LocalMap` (dr.rs:576) is "the prepared global → local pointer table for one tile under one frame … identity references and local positions only, never values". It is issued under a sealed `Frame` (dr.rs:489, no public constructor) and versioned by `MapTable` (dr.rs:874). Falsifiers: (a) partially overlapping root sets refuse as `OVERLAP` on the shared roots only; (b) a duplicate replay contributes nothing; (c) independent new evidence pools; (d) identical masks under different frames are **resolved, not assumed**: they may denote the same evidence or different evidence; (e) a stale root map refuses; (f) ABA reuse of a local position does not alias; (g) a permutation of root order does not change the verdict; (h) more than 64 roots refuses, never truncates, until the existing tiled, wide or sparse addressing is audited. | lab; no contract type |
| S2 rollout witness | there is no typed row for (fixture, arm, stage S0–S5, oracle verdict, receipt identity, `QueryReference`) | an **isolated, single-writer experimental dataset**, written through a cold adapter to `RolloutRecord` (`metadata` / `relationships`), as #2077 and #2078 §5 propose. Test before relying on it: conflicting event ids, flush and read-back, deterministic replay at a pinned `checkout(version)`. If any of these cannot be shown, fall back to temporary append-only artifacts. | lab use and production admission are separate questions; EXO-300 governs only the second |
| S3 durable base | `BaseRef` stands in for `DurableRef` | none: the mock experiment uses the existing lab `BaseRef` and the sealed-version witnesses (`VersionStore::seal`, `CommitWitness`) | EXO-148 blocks **production only**; it is not a prerequisite for the mock-teacher experiment |
| S4 Rig caller | the pilot needs a Claude call with a strict output schema and a hard cost cap | a cold caller through the graph-flow `rig` feature or rig-core directly, **only after** the mock teacher and the oracles are accepted | operator: $5 cap, key requested via `request_credential` |

## 4. Pilot order (unchanged from the operator decisions)

1. The mock-teacher harness and the independent oracles run with zero API calls. The fixture answers come from the oracles in §2.
2. The disable controls turn each named oracle red.
3. Only then: an A/B pilot through Rig under a **$5 hard cap**, with lance-context read-only, `QueryReference::at(v, 0)` pinned, and results recorded as lab evidence only.

## 5. Harvest of #2078 into #2077

#2078 adds no conflicting decision. These points are folded in:
- **Hypotheses H1–H5.** H5 (replay after teardown) is now testable on existing code: `tests/receipt.rs::a_released_thought_replays_exactly`.
- **The S0–S5 ladder with a full denominator.** No optimistic filtering; executable passes and correct refusals are counted separately.
- **Fatal falsifiers §4.8.** In particular "WAL add ack misrepresented as canonical version change" and "`UNKNOWN` K3 mapped to `TRUE` for commit".
- **Smallest first probe.** A Rubicon `IncreaseEligible` + `Counterfactual::NotRun` fixture whose correct answer is `Plan`, not `Commit`.

Superseded by operator decision:
- the $200 budget, by the $5 pilot cap;
- the 288-call main run, which waits on pilot results.

Corrected by this census:
- #2078 §3 says `thought-git-probe` has no "replay/persistence". It does have receipt-based replay (`receipt.rs`), in a lab. What it lacks is production persistence.

## 6. Correction carried into D-LFA

`d-lfa-lance-graph-flow-rubicon-assimilation-v1.md:89` says the D-TGB-4 receipt "does not exist in code yet". That is false at 855d051: `crates/thought-git-probe/src/receipt.rs` implements it in a lab, with 15 passing tests. The note is appended to D-LFA rather than rewritten.

## 7. Not searched

These are open, not absent:
- the planner, NAL, NSL and ONNX test suites were read, not re-run;
- Codegraph coverage of the fixture domains;
- lance-context `ContextStore` retrieval fixtures;
- the #1800/#1801 branches themselves (only the D-TFLP table was read).

## 8. Three sources of acquired knowledge (proposed comparison)

The census adds a non-LLM rule-discovery baseline to the pilot design. XLOG `xlog-induce` does bounded exact induction over the chain, star, fan-out and fan-in topologies, plus n-ary induction, with scoring and provenance. Its module doc says it is behaviourally equivalent to the Python reference `pyxlog/python/pyxlog/ilp/exact_induce.py`.

| Source | Produces | Verified by |
|---|---|---|
| Claude (teacher) | method plans | the same thought-git / independent-oracle conditions |
| `xlog-induce` | rules discovered from relational observations | the same |
| native composition | recombinations of already verified methods | the same |

All three run under identical verification. Comparing them under teacher withdrawal on held-out problems separates three things: imitation (Claude only), induction (`xlog-induce`) and compositional discovery (recombination). PROPOSAL only: nothing here is wired, and the production path of `xlog-induce` is a GPU kernel, so a lab baseline would use its CPU reference.

## 9. Independent gates

These three stay independently gated:
- the full XLOG Reactor Swap (seven reasoning tracks, EXO-377);
- the cognitive-architecture census (EXO-390, Track A);
- the small teacher pilot (§4).

None is a prerequisite for another.

## 11. Shared addressing across provenance consumers (follow-up seam, not a blocker)

Several consumers need stable identities, versioned references and dependencies that can be reconstructed. They may share **addressing** infrastructure (DR `Frame` / `LocalMap` / `MapTable`; DT64 binding witnesses). They must **not** share a truth model or a meaning: a provenance root, a probabilistic choice variable, a possible world, a thought slot and an observation episode are different identities.

| Consumer | What it must identify | Existing code |
|---|---|---|
| thought-git receipts | original evidence behind a derivation | `receipt.rs` (`roots` bit set) |
| NAL | whether two evidence sets are independent | `nal.rs` `Revision`, `Evidence::stamp` |
| XLOG PIR | which choice variables reach an outcome | `xlog-prob/src/provenance.rs`, `pir.rs` |
| XLOG EIR | which world and assumptions a statement holds under | `xlog-prob/src/epistemic.rs`, `wfs.rs` |
| DRed | which derivations survive withdrawal of a premise | none; research only |
| `xlog-induce` | which observations support an induced rule | `xlog-induce/src/provenance.rs` |
| RolloutStore | which experience produced an outcome | `rollout.rs` `relationships` |
| `temporal.rs` | whether evidence was knowable at decision time | `QueryReference::at` |

Record only; nothing is unified here.

## 12. `xlog-induce` on native Cube operations (follow-up seam)

Hypothesis, to be tested before any lowering is built:

| Topology | Possible native realisation |
|---|---|
| chain | relation composition |
| star | intersections of masks over a shared coordinate |
| fan-out | forward folds over the pair plane |
| fan-in | reverse folds over the pair plane |
| n-ary | factorised conjunctive body evaluation (`Cube::body_answer`) |

The open question: can candidate rules be scored over resident relations without building candidate tuple populations? A candidate oracle is the Python reference `pyxlog/python/pyxlog/ilp/exact_induce.py`, provided its independence from the native path is checked first.

## 10. Cognitive-architecture reconciliation (v2)

Grades: CODE (in the tree, with tests), LAB-MEASURED (probe with a recorded measurement), DOC (a plan or design only), ABSENT (nothing found). Paths are under the lance-graph repository root unless marked external. Only D-SAG-3 and `rubicon-volition-v1.md` name OpenPsi. No Rust file under `crates/` mentions it (checked 2026-11-24).

### 10.1 Who owns what

| Concern | Question it answers | Existing code | Grade | Existing doc | Missing | Must NOT be read as |
|---|---|---|---|---|---|---|
| OpenPsi (demand, intrinsic priority) | Which open question deserves scarce budget now? | none | ABSENT | `.claude/plans/sag-wisdom-frontier-ona-hyperon-v1.md` §3 (:77-93), §4 EIG priority (:95-114); marked "RESEARCH / FALSIFIABLE DESIGN ONLY" (:5) | Everything: demand signals, calibrated EIG, a pre-registered scheduler | Evidence (:79-88 keeps demand separate); a commitment; a permit (:93) |
| Heckhausen/Gollwitzer Rubicon | Is thinking broad (deliberative) or narrow and shielded (implemental)? Is an intention committed? Is there a cue-to-action plan? | `crates/lance-graph-contract/src/rubicon_witness.rs` (`read_crossing` :228, `breadth_drop` :209, `persistence_gain` :211, `RubiconVerdict` :183); flow probe `crates/gkat-probe/src/rubicon.rs` (`decide_reference` :74, no tests in file) | CODE for the attention reading; LAB for the decision flow; ABSENT for commitment state and conditional intentions | `.claude/plans/rubicon-volition-v1.md` (state machine D-RV-1 :24-63, `IfThenPolicy` D-RV-2 :73-75, "PROPOSAL ONLY" :4); `docs/architecture/FLOW_COG_RUBICON_ONA_CONVERGENCE.md` (:75-88) | Commitment record, revocation, cue-matched plan, persisted traces | A scheduler or command: the witness only reads (`rubicon_witness.rs` :26-32); `RequestCommit` is an intent (`gkat-probe/src/rubicon.rs` :15) |
| ACT-R (procedural arbitration) | Of the eligible rules, which one is proposed? | `crates/lance-graph-act-r/src/lib.rs` (`arbitrate` :66, `witness_matches` :103) | CODE (own workspace, local experimental types :11-12) | none beyond the crate docs; `FLOW_COG_...md` does not mention it | Canonical contract types, utility learning, buffer timing (:9), a real eligibility source | Truth (`utility` :27) or evidence or permission (`activation` :29); `Decision` is not an authorized effect (:36) |
| Hyperon / MeTTa (method vocabulary) | How are compositional queries and rules declared? | `crates/lance-graph-metta/examples/{join_probe,join_range_probe,wcoj_probe}.rs`; `src/lib.rs` has no items ("Nothing is shipped here yet", :2) | LAB-MEASURED (probes vs datafrog, `.claude/plans/metta-polyfill-v1.md` §6b, §6d) | `metta-polyfill-v1.md` (§4 :109-132, D-MTP table :386-395); `FLOW_COG_...md` crate gate (:41) | `Space` adapter D-MTP-2, recursion D-MTP-3, any binder from MeTTa text | An execution engine: MeTTa is a front end; mask-risc is the only engine (`metta-polyfill-v1.md` :123-125) |
| NAL (evidence revision) | How strongly is a statement supported, by independent roots? | `crates/lance-graph-nal/src/lib.rs` (W3 revision with evidence classes :52-64) | CODE for W3; W4 truth laws, W5 syllogisms, W6 text front end are "Not here yet" (:66-67) | `.claude/plans/nal-polyfill-v1.md` (named in `metta-polyfill-v1.md` :9; not read for this section) | Per-row truth laws, syllogisms, temporal inference | Utility, calibrated probability, or authorization (`FLOW_COG_...md` :83; D-SAG-3 :34) |
| NSL (learned ranking) | Which neighbors does a seed use, and what graph loss applies? | `crates/lance-graph-nsl/src/lib.rs` (neighbor selection, :1-3), `loss.rs`, `eval.rs` | CODE, checked against NSL's own golden file (`lib.rs` :16-17; README) | `crates/lance-graph-nsl/README.md`; plan lives in the external NSL repo | Any link from NSL output to goals or priorities. The crate reproduces `pack_nbrs` and the loss; it does not learn a ranking itself | Evidence, utility, or a goal source |
| thought-git (counterfactual worlds, receipts) | What would this thought change, on a branch, before landing? | `crates/thought-git-probe/` (`oracle`, `slots`, `squash`, `receipt` modules; `lib.rs` :1-9 "no production land, rebase or persistence pipeline") | LAB-MEASURED: 58/58 tests pass at 855d051 (§1); D-TGB-4 receipts exist in `receipt.rs` | `.claude/plans/thought-git-basics-v1.md`, `thought-git-api-v1.md`, `thought-git-lifecycle-contract-v1.md` | Land, rebase, persistence, a durable receipt store (EXO-148) | A committed state or an observed outcome |
| temporal.rs (hindsight-blind horizon) | What may a reader at this rung see? | `crates/lance-graph-planner/src/temporal.rs` (`EpistemicMode::for_rung` :87-97: 0..=4 Strict, 5..=8 Aware, 9 Retro) | CODE | `.claude/plans/rubicon-loco-rung-cognitive-fabric-v1.md` §B (:67-108), gate `F-RLR-1` (:583) | A demonstrated non-rung-4 horizon end to end (:106-108) | Attention breadth, or the right to think at a rung |
| RolloutStore (experiential feedback) | What actually happened after acting? | external: lance-context `crates/lance-context-core/src/rollout.rs` (`RolloutRecord`: `reward`, `grader_id`, `policy_version`, `relationships`) and `rollout_store.rs` (§1); no consumer in lance-graph `crates/` | CODE (external repo) | `.claude/plans/ontology-jit-compile-v1.md` (:344-346, :472; "Reward and rollout events grant no authority" :546) | Any in-repo consumer of outcome receipts; D-SAG-3 needs one (:89); production admission (EXO-300) | Authority, or predicted self-confirmation (D-SAG-3 :89) |
| Codec9 (compact numerical state) | How is a magnitude plus a presence bit stored? | ndarray `src/hpc/codec9.rs` (`EpiLane9`, `Codec9`, `Codec9Runtime`, batch encode/decode, §1); layout spec `crates/lance-graph-contract/src/lane_codec.rs` (:55-62); a 9-bit mirror in `crates/lance-graph-nsl/src/eval.rs` ("Not a production codec") | CODE | `.claude/plans/d-codegraph-0-binary-lanes-v1.md` (:55-72) | Nothing for cognition; it is storage | Truth, confidence, or priority. Presence 0 means untouched, not false; codec9.rs says presence is "NOT a source-write or alpha-claim authorization" |

### 10.2 Separations

- **Utility** is owned by ACT-R's `Candidate.utility` (`lance-graph-act-r/src/lib.rs` :27) and, for goals, a proposed `utility_basis` (D-RV-2). It must not be read as truth.
- **Motivation** would be owned by OpenPsi demand (D-SAG-3 §3). Nothing implements it. It must not be read as utility or as a commitment.
- **Confidence** is owned by NAL `f/c` (`lance-graph-nal`). It must not be read as calibrated probability or as the `p_i` in EIG (D-SAG-3 :34, :103).
- **Epistemic certification** is the CE64 59..63 reading (D-SAG-3 :33), outside this section's code. It must not carry desire, entropy or a commit bit.
- **Numerical presence** is the lane presence plane (`lane_codec.rs` :55-62). It must not be read as truth: unobserved is not false (D-SAG-3 :67-70).
- **Execution authority** is the action gate and a live permit (`FLOW_COG_...md` :27, :80; D-RV-1 :50-62). It must not be read from a Rubicon verdict, an ACT-R `Decision`, a NAL confidence or a model reply.

### 10.3 Extended ladder: L6 and L7 (PROPOSAL, no wired code)

Neither level exists in code. The oracles below exist; the loop that would use them does not.

- **L6, identify worthwhile unresolved questions.**
  - Oracle to reuse: D-SAG-3 EIG, `H(Theta|D)` minus expected posterior entropy (:97-103), and its priority hypothesis (:105-108). It applies only over a pre-registered calibrated distribution; otherwise it abstains (:103).
  - Falsifiers to carry unchanged: noisy-TV vs learnable cause (ID 7, :146), refuted vs unknown conflation (ID 5, :144), whole-work accounting (ID 15, :154).
  - Withdrawal test: remove the external question feed. The chosen set must still beat FIFO and smallest-popcount-debt (:161-162) on unique oracle-supported discoveries per total work (:165), on the same snapshot and budget (ID 0, :139), and must not promote the noisy-TV item.
- **L7, regulate exploration, commitment and disengagement.**
  - Oracle to reuse: `rubicon_witness::read_crossing` on a pre-commitment and a post-commitment `FocusTrace`, giving `breadth_drop`, `persistence_gain` and a verdict (`rubicon_witness.rs` :209, :211, :228). Tests already pin forced-single-candidate silence and the Inverted case (:373, :397).
  - What the witness cannot judge: whether a release was correct. For that, reuse D-RV-4 fixture RV-E4 (justified vs premature release, :130) and RV-E2 (zero actions after revoke, :127).
  - Withdrawal test: remove the externally supplied commitment. The system's own policy must still read `Crossed` on a deliberated task and `Indistinguishable` on a forced single candidate, with no action after revocation, and the verdict must never itself drive a transition (:26-32).
  - Gap: D-ACR-8 consumers were four probe examples when last audited (`rubicon-loco-rung-cognitive-fabric-v1.md` :278-282). I did not re-check that count.

### 10.4 Comparison arm (PROPOSAL)

Question: does bounded OpenPsi-inspired selection keep working when the teacher stops supplying goals?

- Arm A: externally supplied goals, fixed list. Arm B: demand plus calibrated EIG over the same candidate set (D-SAG-3 :105-108). Arm C: deterministic reference scheduler (:161-162).
- Same snapshot, seeds, budget and oracle for all arms (ID 0, ID 15). At step T the goal feed to A is cut and B and C continue alone.
- Report discoveries per total work, stale or post-revoke actions, false-causal rate, and original-root inflation, which must be 0 (:165).
- Kill condition: if B does not beat C after T, the OpenPsi layer is not needed. This follows the D-RV-5 pattern (:147-148), and per D-SAG-3 :178 a post-outcome peek ends the run.

### 10.5 Not built, and not to be built here

- **No scheduler.** Progression is existence, not command (`rubicon_witness.rs` :28-32; `rubicon-loco-rung-cognitive-fabric-v1.md` §K).
- **No AtomSpace.** No second OpenCog store (`FLOW_COG_...md` :39); Classic AtomSpace is AGPL (`metta-polyfill-v1.md` §8).
- **No interpreter.** Hyperon's interpreter stays in Hyperon and is not modified (`metta-polyfill-v1.md` §8).
- **No execution engine.** ACT-R never executes effects (`lance-graph-act-r/src/lib.rs` :6-8); D-RV-5 item 10 names a second scheduler or runtime as scope creep (:156).
- **No truth store.** Evidence lives in NAL with original roots (`lance-graph-nal` :52-64); D-SAG-3 forbids a shadow registry or new index (:37).

## 13. The 64-bit root limit is an addressing law, already in code (research note)

The 64-bit root limit is the width of a local operation, not the size of the evidence universe. The decomposition already exists in DR. It does not need a separate implementation in thought-git, NAL or XLOG:
- `lance-graph-dt64::dr::Width` (dr.rs:35) has two widths, `W64` and `W256`.
- `Axis::tile_of` (dr.rs:89) computes `position / tile.positions()`.
- `Axis::for_survivors` (dr.rs:66-85) picks one DR64 tile for up to 64 positions, one DR256 tile for up to 256, and `ceil(n / 256)` DR256 tiles beyond that.
- `LocalMap` (dr.rs:576) is the witnessed global → local pointer table for one tile under one `Frame`.

So a receipt's `u64` root mask is the 64-wide tile of this law. A larger root set means more tiles under the same `Frame`, never truncation. Within a tile, the local bit position carries no identity; `LocalMap` supplies it. Modular arithmetic cannot decide whether two roots are the same or independent. That stays with the identity map and NAL.

**Observation.** `Axis::for_survivors` saturates `tiles` at `u32::MAX` (dr.rs:78-82). That is a reservation count, so a saturating value cannot alias evidence. An evidence path that reuses this law must refuse at that bound, not saturate.

**Hypothesis, PROPOSAL only.** The law could be stated once, declaratively (tile = ⌊i / w⌋, local = i mod w for a power-of-two w), and lowered by the existing compiler donors to shifts and masks. Two things to check first: whether XLOG's existing RIR, `crepe_plan!` and the Soufflé-harvested passes already lower such a rule; and whether DT64's method binding can carry it. No new configuration language is to be introduced. The open question for LG-COG-001: can a teacher or `xlog-induce` contribute *configurations* of existing laws (grammar → typed law → method configuration → verified composition → native execution) instead of new runtime code? It is gated like every other PROPOSAL here: oracle, disable control, no hot-path change.

### 13.1 Correction: evidence identity is invariant under lawful coordinate transformations

S1 falsifier (d) once said identical masks under different frames resolve to *different* identities. That is wrong. The same evidence root can survive across versions and coordinate assignments. The invariant to test:

| Transformation | Expected |
|---|---|
| same identity, different local position | same evidence |
| same position, different identity | different evidence |
| same identity, newer valid frame | same evidence |
| stale or invalid frame | refusal |
| permuted coordinates with the matching mask | same verdict |
| more than 64 roots | tiled representation, or explicit refusal |

### 13.2 Width-parametric addressing (follow-up research seam)

- **Dense ordinals.** `i = W·t + l` with `l = i mod W` is the addressing law. DR implements W = 64 and W = 256 today (`Width`, dr.rs:35). Other widths are conceptual, not admitted.
- **Sparse identities.** Modulo cannot translate them. DR's witnessed `LocalMap` / `Frame` does.
- **Codec9 is not a width.** Its 512 codewords are value codes for one lane: 511 values plus the never-touched code. They are not an address width.
- **The law could be configuration, not code.** It could be stated once as typed configuration. Datafrog would be the relational reference, Ascent BYODS the abstract-relation formalism, and Crepe / Soufflé the compiler donors.
- **Semantics stay with the consumers.** One law, several physical configurations, and independent semantic consumers: NAL, thought-git, PIR, EIR, DRed, induce.
- **Scope.** No new grammar engine, and no change to implementation scope.

### 13.3 Scope of the addressing finding

DR supports multiple tiles. The receipt format does not yet: `Receipt::roots` is one `u64`. A receipt can use DR's multi-tile law only after a witnessed binding from its root tiles to DR `Axis` / `LocalMap`. Until then, more than 64 roots refuses (§13.1).

## 14. Measuring how far the existing algebra reaches (PROPOSAL metric for LG-COG-001)

Each new problem the pilot solves is classed by what it took:

| Level | What is acquired | Runtime change |
|---|---|---|
| configuration | parameters and bindings of existing laws | none |
| composition | new combinations of existing verified methods | none, where already expressible |
| extension | an operation the algebra cannot express yet | yes; needs its own admission gate |

Report the three counts separately, per knowledge source (Claude, `xlog-induce`, native composition, §8). Most solved problems landing in configuration or composition would be evidence that capability grows without a matching growth in code. A configuration is valid only if every field resolves to an existing typed contract. No new schema or configuration language.
