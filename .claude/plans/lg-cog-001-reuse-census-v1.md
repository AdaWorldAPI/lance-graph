# LG-COG-001 reuse census: capability and admission matrix (v1)

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
| S1 receipt → NAL stamp | `Receipt::roots` is a **64-bit bit set** (`receipt.rs:381-383`: "Provenance roots of those sources, as a bit set"). NAL's `Evidence::stamp` is a slice of root **identities** (`&[u64]`, lib.rs:250). A bit position is not an identity: two bases can reuse the same bit for different roots, and a set holds at most 64 roots. So the translation needs a root-identity table per base, not a numeric cast. | a lab-only adapter in a probe crate, mapping bit to root id through a per-`BaseRef` table. Falsifiers: (a) partially overlapping root sets refuse as `OVERLAP` on the shared roots only; (b) a duplicate replay contributes nothing (`independent_roots = 0`); (c) independent new evidence pools; (d) the same bit under two bases with different root ids does **not** collide; (e) a root that overflows 64 bits refuses instead of aliasing. | lab; no contract type |
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
