# D-DTM-1 — Distributed Thought-Delta Membrane: exchanging computational understanding

**Date:** 2026-10-25. **Status:** S0 DOCUMENTATION/PR-MAP ONLY, not an implementation authorization. This plan introduces no Rust, ABI, serialization format, schema, storage, executor, Raft group, protocol service, build dependency, certified law, or benchmark result.
**Coordination:** [EXO-50 append-only agent inbox](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id). **Execution brief:** [D-DTM-1 handoff prompt](../prompts/d-dtm-1-reasoning-interchange-claude-code.md).
**Question:** Can independent workers reuse a *sealed focus + versioned method + evidence/dependency witness* while exchanging fewer bytes and performing less repeated compilation than document-centric RAG or full-data RPC, with no silent loss of source semantics, mathematical validity, numerical fidelity or historical knowledge horizons?

## 0. The paradigm shift and its limit

Scientific citation or RAG exchanges a **reference** and asks another interpreter to rebuild meaning. Wikidata exchanges addressable **typed relationships**. This proposal tests the next step: exchange a **reusable, executable, evidence-bearing derivation** that another worker may validate, run, refuse, and extend. “Understanding” is a research claim to operationalize, not a synonym for either a language-model answer or a compiled program.

Exchange *three separately versioned objects*:
- **Knowledge**: canonical source identity, typed relation facts and values, revision/branch, source/index/scope witness; storage may be Lance or a compatible read adapter.
- **Attention/focus**: a DR64/DR256 mapping, mask or relation-support descriptor, coordinate domain and focus generation; may live independently of derived value facets.
- **Method**: typed premises and dependencies, operator/law identities, a guarded egglog proof/extraction or explicitly uncertified candidate, compiled existing R2IL/Quack/ndarray recipe, target and arithmetic contract.

The substrate is *not* a global neural consciousness, generic AGI engine, Kafka-native AST, second VM, common mutable graph, complete scientific proof machine, or new physical storage owner. Demonstrable benefit is reduced repeated interpretation, compilation and transfer on admitted sparse/reusable workloads. Dense training, causality from correlation, missing science and global cross-region consistency are not “solved” by compact envelopes.

## 1. Immutable ownership and prerequisite crosswalk (inspect before edits)

| Concern | Authority / source | Current evidence / limit | D-DTM must NOT do |
| --- | --- | --- | --- |
| Semantic classes, capabilities and typed DTO resolution | OGAR ClassView/DTO-IR; [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md), [EXO-89](https://linear.app/exo-red/issue/EXO-89/d-rmap-1-1a-implicit-relationship-register-and-separate-thought) | RelationshipMapStore/MapPalette proposal; no production map owner | Invent a competing class registry, RelationOrder store or globally valid u16 map |
| Native read mathematics | [Quack portability ruling](../board/entries/2026-10-05-quack-storage-portability.md), existing relalg/contract/R2IL/mask-risc/ndarray | Bind once, native numeric operations, no hidden relational gather | Route every fold through RPC, introduce another interpreter, or change Bag/Distinct/K3 semantics |
| Focus and values | [D-DRF-1](d-dr64-focus-lease-foreign-cache-v1.md), [D-DRAV-1 draft #1775](https://github.com/AdaWorldAPI/lance-graph/pull/1775), [EXO-151](https://linear.app/exo-red/issue/EXO-151/d-drf-immutable-dr64-focus-leased-foreign-cache-sparse-writeback) | D-DRAV AV1–AV12 unrun; separate MAP/BORROW/GATHER/COMPILED/STORAGE are candidates | Claim map-only workloads generally value-free; retain stale map/support or borrow after revoke |
| Source and thought identities | Thought-Git/D-TGA, [EXO-148](https://linear.app/exo-red/issue/EXO-148/durableref-thoughtcommitid-do-not-exist-four-plans-and-d-tgb-3-wait-on) | DurableRef and ThoughtCommitId production definitions unresolved | Invent a second durable ID or identify Kafka offset / Raft index with ThoughtCommitId |
| Derived mathematical methods | [D-DRMC](dr-mathematical-compiler-v1.md), [Math Foundry #1693](https://github.com/AdaWorldAPI/lance-graph/pull/1693), [#1694](https://github.com/AdaWorldAPI/lance-graph/pull/1694), [EXO-127](https://linear.app/exo-red/issue/EXO-127/d-drmc-structural-dr-to-reusable-mathematical-compilation) | Merged E1 exact-integer lab proof/replay capsule and guarded in-memory catalogue; no independent verifier, no general wire capsule | Promote replay/self-check to independent proof, numerical conformance or scientific validity |
| Symbolic inputs | [D-MTP](metta-polyfill-v1.md), [D-MPS](mork-pathmap-schnellverschluss-v1.md), [EXO-79](https://linear.app/exo-red/issue/EXO-79/d-mps-mork-pathmap-schnellverschluss-08), [EXO-143](https://linear.app/exo-red/issue/EXO-143/metta-four-binding-case-one-fold-compared-test-against-the-d-mps-9) | MORK/PathMap external differential oracle; set/bag distinctions, allocator/licensing upstream gates | Link MORK into production, leak PathMap symbols/pointers or call oracle proof authority |
| Learned graph/model import | [D-NFC](d-native-first-dependency-correction-v1.md), [D-ONNX CAP](d-onnx-relalg-cap-v1.md), #1754 / #1758 | Native-first ONNX facade shipped for subset; ONNX/TFLite are external compatibility leaves, DR64³ binding missing | Make ONNX interpreter the native graph or import unsupported operators by approximation |
| Source-code reasoning tracert | [D-SAG-6](d-sag-6-rust-codegraph-math-logic-tracert-v1.md), merged [#1760](https://github.com/AdaWorldAPI/lance-graph/pull/1760) | Proposal for read-only Rust Codegraph → math → logic → traceroute, not implemented | Build a separate codegraph, AST engine, law registry or automatic source mutation |
| Per-owner / epistemic time | [temporal.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-planner/src/temporal.rs) | local trajectories + Strict/Aware/Retro query-time projection; HLC/dependency bodies incomplete | Reorder committed state-machine apply by HLC or treat default Strict(u64::MAX) as bounded |
| Durable publication | Existing CycleWriter/DetachedCycleBatch, D-TGB, [EXO-150](https://linear.app/exo-red/issue/EXO-150/idea-a-certificate-replaces-the-deltas-lane-0-keeps-delta-receiptseal) | Existing cycle seal and real/lab boundaries; “certificate replaces operations” remains IDEA | Replace owned dirty state with unverified proof reference or treat queueing as commit |

**Conservative precedence:** D-NFC native-first correction controls ONNX assumptions; D-RMAP-1A controls map ownership; D-DRAV-1 and D-DRF own focus eligibility; D-SAG-6 owns codegraph; D-DRMC owns law candidate/replay; Quack owns query semantics; Thought-Git and CycleWriter own historical identity and durable publication. D-DTM defines an **external exchange seam only after those owners admit it**.

## 2. Data geometry, four objects that must not be conflated

- Canonical example: 65,536 rows × 512 bytes = **32 MiB** source image (not assumed network payload).
- DR64 **u16 address map**: 4,096 × 2 bytes = **8 KiB**. A DR64² Boolean relationship is instead 4,096 bits = **512 bytes**.
- Proposed DR64³ Boolean relation/support volume: 64³ bits = **32 KiB**; this is **not** 64³ numerical cells and is not an implemented representation. DR64³ of u16 values would be 512 KiB.
- A method reference (content digest + versioned manifest) might be far smaller than any of these **only when both peers already possess, admit, and can bind the immutable artifact**.

Transport only what a typed operation needs. A Boolean relation probe may need no new value facet. An ONNX numerical operation generally needs real operands or a mathematically validated compiled equivalent. Neither a DR mask nor a proof digest magically materializes missing values.

## 3. Proposed logical exchange contract (NOT an ABI declaration)

~~~text
ReasoningExchange {
  payload_kind: FocusRef | FocusDelta | TypedOperand | MethodRef | MethodArtifact | Result | Refusal
  source: source_id + branch + witnessed snapshot + retention/read lease
  coordinates: canonical domain + map/index seal + generation + ACL/scope
  thought: existing ThoughtCommitId / DurableRef when actually defined
  method: recipe/capsule digest + semantic key + rule/opset/codebook version
  execution: target/ABI/numerical contract + operation/semiring + result demand
  knowledge: QueryReference observed head + epistemic mode + dependency closure
  replay: application idempotency key + causal parents + result digest
  bytes: typed, length-bounded canonical LE fields or valid Arrow IPC payload
}
~~~

These field names are design vocabulary only. **No new global ID, serializer, wire tag or Rust type is authorized.** If a required witness lacks an authoritative representation, return typed refusal and block transport admission. No raw Rust memory layout, unchecked pointers or hidden JSON serialization. Validate source/scope/version before any borrowed value access; preserve slot mapping, row ordering, Set/Bag/NULL, signed zeros/NaNs and effects.

A proof/equality witness, finite empirical observation, learned model confidence and scientific evidence are distinct credentials. The E1 replay capsule checks its own retained derivation under stated premises, not arbitrary proof validity; “compiled” alone is not “proven”. Results carry provenance, including refusals and numerical bounds.

## 4. Transport, persistence and consensus are orthogonal

| Mechanism | Correct function | Not provided |
| --- | --- | --- |
| **Kafka** native binary record values / batches | Durable per-topic-partition ordered delta/reference publication, consumer progress and replay | Global order across partitions, cross-store atomicity, native DR semantics |
| **Arrow Flight/gRPC** DoExchange / DoGet | Remote operand/results streaming and application metadata; optional large Arrow batches | Persistent agreed commit log, exactly-once replay, arbitrary bytes masquerading as Arrow IPC |
| **TiKV** | MVCC + Raft-replicated key-range transactional storage as **one optional backend** | Shared Thought-Git identity, indefinitely retained historical snapshots, a second Raft requirement |
| **raft-rs / OpenRaft** | Consensus for a purpose-built state machine only when a new replicated owner is justified | Automatic cross-partition global causality or proof checking |
| **Temporal planner** | Local owner-chain reconstruction and reader-horizon/depends projection | Permission to reorder committed application |

**Do not run OpenRaft around every TiKV transaction.** Do not tunnel the full Kafka protocol through Flight merely to avoid JSON: both support binary payloads at their boundaries. Compare native Kafka payload, native Flight operation, and Flight→Kafka gateway only as distinct arms. RSKafka is a minimal partition/WAL lab candidate (no transactions/groups); rust-rdkafka is a mature transactional baseline; Confluent's pure-Rust client is preview, not an admitted production foundation. Upstream `lance-fly` currently has DoExchange/DoGet unimplemented.

Multiple Kafka partitions or TiKV ranges have no single globally valid source head without explicit coordination. HLC is useful for ordering observations, **not** a consistent snapshot or a causal proof. Use a witnessed vector of source positions / snapshot closure where appropriate; never compare unrelated offsets numerically.

## 5. Proposed PR slices, separate approval per slice

**PR-0 D-DTM-S0 (THIS PR, docs only):** capstone, owner/status/references map, frozen experiment protocol, Claude Code handoff and board registration; no runtime changes. EXO-50 linked; wait for review/ACK, never auto-merge.

**PR-1 D-DTM-P1 (read-only feasibility and independent oracle):** fixture: two workers with the same typed 64K-row synthetic SoA, identity/coordinate revisions and a real independent scalar/mathematical oracle. Collect cold/warm baselines for row, compact lane and mask-only reads; prohibit hidden layout copies. Preregister missing-source and Set/Bag/K3 failures. No network or new API. Depends on admission owner signoff, not a license to redefine AV1/AV2 of #1775.

**PR-2 D-DTM-P2 (focus/demand binding probe):** a sealed focus descriptor references existing D-RMAP/BIND + D-DRAV; optional local value demand and changing source/index/scope or derived-support invalidation. Count bind/facet reads and independent repairs; alias-safe dirty positions. Requires AV1/AV2 and EXO-46/89 decisions as appropriate; preserve original DRF and DRAV falsifier numbering.

**PR-3 D-DTM-P3 (cold method artifact lookup and replay):** reuse E1 #1693/#1694 `ReplayCapsule` and `Catalogue`; propose only an external digest-bearing manifest and capability check. Compare hot cache hit vs fresh extraction vs trusted baseline; reject semantic-key, target/ABI, codebook, scope and source mismatch. A replayable recipe is **not** a general independently verified mathematical certificate. General ONNX/float/MeTTa methods blocked on their original proof/conformance owners.

**PR-4 D-DTM-P4 (binary Kafka delta/reference transport, lab only):** feed existing frozen cycle outputs or externally owned synthetic deltas via an adapter; compare RSKafka, rust-rdkafka and optionally Confluent preview with identical byte payloads. Measure key/record overhead, local allocations/copies, broker ack vs application receipt, replay, duplicate suppression, partition ordering, leader failure and crash recovery. Never change CycleWriter or claim atomic Kafka+Lance/TiKV commit.

**PR-5 D-DTM-P5 (Flight DoExchange compute transport, lab only):** Arrow Flight metadata carries typed request/reference; valid IPC carries actual operands/results; compare direct Flight with Kafka and a deliberately expensive Flight→Kafka gateway arm. Real remote fold, error/refusal and cancellation, no remote RPC in each nanosecond fold, no new query executor. Existing lance-fly is an upstream skeleton/reference, not a relied-on complete service.

**PR-6 D-DTM-P6 (temporal cross-worker receipt):** resolve EXO-148 / stable source+thought IDs, read horizons and dependency closure; test cross-partition stagger, retention/GC, delayed evidence, reconnect, duplicate delivery and refusal on an inconsistent snapshot. Strict at *concrete* observed head excludes future rows; Aware may view hindsight but never retroactively rewrites committed state. No global HLC-order replay.

**PR-7 D-DTM-P7 (optional symbolic/model import):** ONNX/TFLite graph/operator/weights/quantization as **external** compatibility; MeTTa atoms interpreted through existing D-MTP, MORK/PathMap oracle through D-MPS, egglog/DialEgg-inspired cold transformation via D-DRMC. One narrow end-to-end case with exact integer/Boolean semantics first; general floating-point laws, MORK license, proof checker and model import remain separately gated. Never make foreign engines native executor dependencies.

**PR-8 D-DTM-P8 (two-node system falsifier):** compare (A) baseline document/RAG replay or text mediation *if reproducible*, (B) full rows, (C) narrow values, (D) full focus, (E) cached focus+method digest+delta; cold/warm and 1/8/64/256 repeats. Independent correctness oracle on both nodes and distinct physical read layouts. Report end-to-end CPU+network+broker+I/O+replay cost, bytes, p50/p95/p99, retries, rejected stale artifacts and proof/conformance category separately. A win on a synthetic fixture does NOT imply representative AGI savings.

**Dependencies:** PR-1 can proceed independently of import/proof work; P2 must not claim D-DRAV admission prematurely; P3 needs proof/status reconciliation; P4 and P5 can be independent transport micro-labs; P6 is required before historical or cross-partition correctness claims; P7 is optional; P8 depends on whichever subset it measures. Never create one giant stack or retroactively call proposals “shipped”.

## 6. Falsifiers and decision gates (pre-registered, NOT RUN)

| ID | Kill / refusal test | Required result |
| --- | --- | --- |
| F01 | Same `u16` slot in different coordinate domains | Identity mismatch REFUSE, never silent alias |
| F02 | Source changes between focus receipt and execution; value-derived support changes | Stale binding REFUSE or separately sealed successor |
| F03 | Missing dependency outside local DR focus | Expand through witnessed mapping or REFUSE, never absent = FALSE |
| F04 | Bag vs set duplicates; SQL unknown; three-valued masks | Oracle parity or typed semantic refusal |
| F05 | Different F32 reduction order, NaN, FMA, quantization/codebook | Explicit conformance envelope; no false bit-exact claim |
| F06 | Forged/incomplete egglog proof, wrong premise or target, warmed mismatched catalogue | Reject or trusted fallback; self-check not promoted |
| F07 | Replay after duplicate Kafka delivery, offset reset, crashed worker | Same admitted result; no duplicate accepted effect |
| F08 | Mixed Kafka partition offsets, changed TiKV/Lance source and GC-expired history | Refuse inconsistent/expired snapshot |
| F09 | Strict reader at concrete head receives a future-row claim | No hindsight leak; Aware/Retro clearly labelled |
| F10 | Flight stream cancelled midway, unsupported Arrow encoding or malicious lengths | No partial accepted result; parser rejects safely |
| F11 | Complete transport receipt without durable applied state | Distinct receipt types; never report committed reasoning |
| F12 | Cold method artifact not cached, many low-value single folds | Baseline wins where overhead dominates; proposal rejected |
| F13 | MORK oracle deduplicates MeTTa four-binding Bag result | Preserve multiplicity or REFUSE; do not reinterpret silently |
| F14 | Learned ONNX graph op outside admitted native subset | Refuse at import; no invented float algebraic proof |
| F15 | Compiled method discovered from correlated population | Observation cannot promote itself to theorem or causal law |

Measure **complete** cost, not isolated warmed masks; count materialized source bytes and physical pages for cold reads. Compare against a trusted direct local primitive, not only against JSON. A network speedup without the same accepted result and provenance is a failure.

## 7. Minimal test fixture and cost ledger

Proposed fixture: 65,536 canonical rows with deterministic typed values and relationships, two logical owners, duplicate facts, NULLs, stale generations and two independent physical read layouts. Seal revision V0. At V1 change a non-topological value; at V2 change relation support; at V3 change scope; at V4 introduce late evidence. Source fixtures must be frozen *before* benchmark observations (EXO-133). An external oracle must not call the same lowering as the implementation (EXO-109).

Account separately: `T_parse + T_resolve + T_bind + T_compile + T_network + T_broker_ack + T_source_io + T_native_exec + T_verify + T_apply + T_replay`. Do not sum overlapped pipeline phases as though sequential when reporting wall time. Compare 1/8/64/256 uses, clustered/random foci, compact and full values, warm/cold artifact, and lost/duplicated/delayed delivery. Report compression, allocation and payload bytes; payload ratio is not a speedup.

## 8. Reference inventory and open decisions

**Local primary references**: [D-SAG-6](d-sag-6-rust-codegraph-math-logic-tracert-v1.md); [D-NFC](d-native-first-dependency-correction-v1.md), [D-ONNX CAP](d-onnx-relalg-cap-v1.md); [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md); [D-DRMC](dr-mathematical-compiler-v1.md); [D-MPS](mork-pathmap-schnellverschluss-v1.md); [D-DRF](d-dr64-focus-lease-foreign-cache-v1.md); [D-DRAV draft #1775](https://github.com/AdaWorldAPI/lance-graph/pull/1775); [temporal.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-planner/src/temporal.rs); [#1693](https://github.com/AdaWorldAPI/lance-graph/pull/1693), [#1694](https://github.com/AdaWorldAPI/lance-graph/pull/1694), [#1760](https://github.com/AdaWorldAPI/lance-graph/pull/1760).

**Linear coordination/evidence**: [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id) (message inbox only), [EXO-46](https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production), [EXO-79](https://linear.app/exo-red/issue/EXO-79/d-mps-mork-pathmap-schnellverschluss-08), [EXO-84](https://linear.app/exo-red/issue/EXO-84/d-mtp-metta-as-a-front-end-over-the-masking-polyfill-06), [EXO-89](https://linear.app/exo-red/issue/EXO-89/d-rmap-1-1a-implicit-relationship-register-and-separate-thought), [EXO-127](https://linear.app/exo-red/issue/EXO-127/d-drmc-structural-dr-to-reusable-mathematical-compilation), [EXO-135](https://linear.app/exo-red/issue/EXO-135/crumb-trail-what-may-be-built-before-execution-under-what-witness-and), [EXO-143](https://linear.app/exo-red/issue/EXO-143/metta-four-binding-case-one-fold-compared-test-against-the-d-mps-9), [EXO-148](https://linear.app/exo-red/issue/EXO-148/durableref-thoughtcommitid-do-not-exist-four-plans-and-d-tgb-3-wait-on), [EXO-150](https://linear.app/exo-red/issue/EXO-150/idea-a-certificate-replaces-the-deltas-lane-0-keeps-delta-receiptseal), [EXO-151](https://linear.app/exo-red/issue/EXO-151/d-drf-immutable-dr64-focus-leased-foreign-cache-sparse-writeback).

**External API/spec references**: [Arrow Flight specification](https://arrow.apache.org/docs/format/Flight.html), [Kafka protocol](https://kafka.apache.org/protocol.html), [TiKV documentation](https://tikv.org/docs/), [Confluent native Rust client preview](https://github.com/confluentinc/kafka-clients/tree/master/rust), [rust-rdkafka](https://github.com/fede1024/rust-rdkafka), [RSKafka](https://github.com/influxdata/rskafka), [lance-fly server skeleton](https://github.com/lance-format/lance-fly), [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html), [TFLite schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs), [trueagi MORK](https://github.com/trueagi-io/MORK), [DialEgg CGO 2025](https://pldb.kirancodes.me/p/conf/cgo/ZayedD25).

**Open decisions (do not decide in S0):** durable identity/type owner (EXO-148), multi-source snapshot boundary/retention, independent proof admission format, MORK licence/allocator gate, native Kafka client admission, first model and opset validation, authorization across worker boundaries, and whether the synthetic savings justify a complete system experiment.

## 9. Review discipline

- Every future stage opens a **separate** small PR only after its owner/gate is acknowledged. Keep experimental source edits in the existing lab family or separate feature branch; no auto-merge.
- S0 registers proposed work but creates no unapproved Linear subtasks, no parallel competing owner and no board history rewrite. Only send a new append-only handoff comment in EXO-50.
- A later status update must cite exact commit, PR, test/oracle results and changed-file inventory. Source evidence beats promising narrative.
