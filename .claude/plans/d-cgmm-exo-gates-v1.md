# D-CGMM-EXO-0: Linear decision ledger and staged PR ownership crosswalk

> **2026-11-04 · DOCS / DECISION REQUESTS ONLY.** This ledger records named issue owners and **decisions to request**, not permissions granted. The source of truth for implementation is [D-CGMM #1958](https://github.com/AdaWorldAPI/lance-graph/pull/1958), its [implementation blueprint](d-cgmm-implementation-blueprint-v1.md) and [staged charter](d-cgmm-staged-pr-charter-v1.md). [EXO-155](https://linear.app/exo-red/issue/EXO-155) tracks this program, [EXO-50](https://linear.app/exo-red/issue/EXO-50) is an append-only message bus. Linked issues are **not** implicit approvals.

## 1. Critical path owner decisions: answer with a source and a test

| Priority / stages | Existing Linear issue | Exact decision owner must make | Required evidence / what to refuse before ruling |
| --- | --- | --- | --- |
| **P0, CGMM-1** | [EXO-155](https://linear.app/exo-red/issue/EXO-155) (program); source producer owner still to be named | May Ruff/Charon writer extend typed source fact lanes for read/write/receiver/dispatch without breaking versioned `.rows.bin` consumer? Which source identity is stable? | Writer+reader contract/fixture and independent Kythe-like identity oracle. If producer lacks effects, keep UNKNOWN and stop automatic method equivalence. |
| **P0, CGMM-2** | [EXO-85](https://linear.app/exo-red/issue/EXO-85) MeTTa/Datalog assimilation; [EXO-84](https://linear.app/exo-red/issue/EXO-84) native polyfill | Is demand/binding metadata a read-only compiler artifact consumed by Codegraph and Quack, with no parallel interpreter? | Compare current Codegraph bound joins to demand-scoped masks and `crepe_plan!` D1/D2 against independent oracle. Do not import Soufflé/Crepe runtime. |
| **P0, CGMM-3** | [EXO-121](https://linear.app/exo-red/issue/EXO-121) ClassView capstone | Is `FieldViewDef` an immutable offer anchored to existing `FieldRef` and full behavior lens, or should an existing type be reused? Does it belong above zero-dep `ClassView`? | Same label different ClassId/schema, 109-field wide basis, full_classid behavior lens, scratch mask vs presence distinction. No trait mutation before decision. |
| **P0, CGMM-4** | [EXO-95](https://linear.app/exo-red/issue/EXO-95) business parity; ontology/OGAR owners | Which OGIT `LinkSpec` and OGAR `ActionDef` fields are authoritative for role, M:N cardinality, ordered effects and app-specific behavior? | Tenant A/B, duplicate edges, SameConcept/DifferentAction, unknown Rails/SAP evidence. A LinkSpec does not authorize edge instances or action execution. |
| **P0, CGMM-5/6** | [EXO-152](https://linear.app/exo-red/issue/EXO-152) normalized method DTO/native aliases | Can an eligible pure FieldViewDef map to an existing `RelAlg::Rel` and then an admitted DT64 MethodBinding without creating a second method identity? What semantics demand REFUSE? | Exact single-population read vs M:N refusal; guard-free/K3/WHEN refusal; pinned method version and native independent oracle. |
| **P0, CGMM-3/4/6** | [EXO-46](https://linear.app/exo-red/issue/EXO-46) counted global→focus binding | Who attests cold symbol→DR global/local and scope binding? What lookup costs are legal before borrowed fold? | No Codegraph `Pos` = DR ordinal or ClassId shortcut. Refuse unowned foreign bind; count preparation once and require source/frame witness. |
| **P0, CGMM-4/5** | [EXO-140](https://linear.app/exo-red/issue/EXO-140) admission B call-site classification | Which `Semijoin/EqU32Via/Gather` or foreign-hop shapes are lawful; who classifies each call site? | No value-derived cross-source gather masquerading as FieldViewDef; classify each with reader and source-order witnesses. |
| **P0, CGMM-4/5/8** | [EXO-108](https://linear.app/exo-red/issue/EXO-108) Set-vs-Bag terminal | How is count-distinct over projection distinguished from Bag fanout product and Boolean support? | Duplicate-edge fixture with different Set and Bag answers; no novel opcode absent insufficiency proof. |
| **P1, CGMM-5/8** | [EXO-73](https://linear.app/exo-red/issue/EXO-73) recursion driver | Where does a legal recursive driver and bounded/unbounded termination contract live? | CGMM first explicitly refuses recursion. An α-acyclic shape classifier and a bounded external Datalog oracle do not create an admitted runtime. |
| **P1, CGMM-8** | [EXO-79](https://linear.app/exo-red/issue/EXO-79) MORK PathMap/license, [EXO-143](https://linear.app/exo-red/issue/EXO-143) MeTTa four-binding case | Under which license/version may MORK run externally as differential oracle, and how is Set vs multiset binding checked? | External oracle pinned, 4-binding counterexample, no MORK code vendoring; no runtime authority. |
| **P1, CGMM-7/8** | [EXO-154](https://linear.app/exo-red/issue/EXO-154) zero-copy virtual driver, [EXO-153](https://linear.app/exo-red/issue/EXO-153) DR registers | May prepared plan/route receipts borrow canonical SoA without copying business values? What source/layout/permission epochs are checked? | #1955 local borrowed proof only; #1956 ALU speed loss; no disk/remote zero-copy inference, no new index. |
| **P1, CGMM-8** | [EXO-104](https://linear.app/exo-red/issue/EXO-104) tuple materialization, [EXO-105](https://linear.app/exo-red/issue/EXO-105) reference executor, [EXO-109](https://linear.app/exo-red/issue/EXO-109) oracle independence | Which functions are admissible only as independent test oracles, and how do we stop reference tuples reaching production? | Candidate and oracle cannot share the same execution core; compile/CI root-scoped bypass counters; no broad symbol grep presented as a full call-graph proof. |
| **P1, CGMM-6** | [EXO-155](https://linear.app/exo-red/issue/EXO-155) + DT64/R2IL execution owner to be explicitly named | Who issues production execution capability, validates WHEN/DO, body revision and tenant permission? | `ExecPermit` is lab-only; current `GuardNotExecutable` and `ThreeValuedNotExecutable` remain active. Do not treat `BindStore::admit` as an execute permit. |
| **P2, CGMM-8** | [EXO-101](https://linear.app/exo-red/issue/EXO-101) datafrog/semijoin parity | Which external Datalog fixture may be used as a credible separate reference for recursion/bag? | No unproven gather driver, no semantic cross-contamination or production import. |
| **P2, CGMM-8** | [EXO-50](https://linear.app/exo-red/issue/EXO-50) | Preserve append-only review, cross-agent disagreement and PR/source references | EXO-50 posts are communications, not admission or merge authorization. |

**Note:** Issue names/statuses represent the 2026-11-04 source check, not a frozen guarantee that the owners have already approved a decision. A named issue is a decision queue; **the actual accepted ruling, commit, tests and scope must be recorded separately**.

## 2. Stage-to-Linear traceability

| Proposed PR | Main Linear tracker | Cross-owner gates | Definition of done at lab level |
| --- | --- | --- | --- |
| D-CGMM-0 (this docs PR) | EXO-155 | EXO-50, 121, 152 | Source maps, owner ledger, independent fixture matrix, no runtime change |
| D-CGMM-1 source facts | EXO-155 | Ruff/Charon producer owner, EXO-95 | Writer/reader contract, stable identity under revisions, detected effect-loss mutation |
| D-CGMM-2 lazy demand | EXO-155 | EXO-84, 85, 73 | Same graded mask as eager/oracle, query Q cost/copy evidence, stale plan refusal |
| D-CGMM-3 FieldViewDef scratch | EXO-155 | EXO-121, 152 | Correct ClassView basis + typed OFFER/REFUSE/ABSTAIN, no field-codebook mutation |
| D-CGMM-4 ontology/action/tenants | EXO-155 | EXO-95, 108, 46, 140 | M:N cardinality and Bag distinct, tenant and effect-order kills, independent references |
| D-CGMM-5 RelAlg polyfill | EXO-155 | EXO-152, 73, 108, Quack/R2IL owner | One legal pure fold and explicit unsupported foreign/action/WHEN refusals |
| D-CGMM-6 DT64 native bind | EXO-155 | EXO-121, 152, execution permit owner | Immutable method identity, native parity, stale witness/guard/K3 refusals |
| D-CGMM-7 memo economy | EXO-155 | EXO-153, 154, 46 | No copied payload, Q amortization + whole-query cost, correct cache invalidation |
| D-CGMM-8 evidence/callgraph | EXO-155 | EXO-79, 143, 101, 73, 104, 105, 109, 154 | Independent oracles, root-scoped coverage, uncertainty and external licensing tracked |

**Approval dependency:** CGMM-1/2 may be performed as workspace-excluded *labs* only after the applicable source/editor access is established. CGMM-3/4 may draft schema offers without admitting them. **CGMM-5/6 require named RelAlg/DT64/OGAR owners before altering any production vocabulary or executing a generated MethodBinding.** CGMM-7/8 must not be used to backdoor an unadmitted performance optimization.

## 3. Exact questions for EXO-155 owner review

1. **Fact identity:** Which persistent source-node key actually survives Ruff/Charon export and schema refresh? Is it enough for cross-language provenance? If not, how is UNKNOWN retained without a new graph registry?
2. **Demand:** Is Codegraph permitted to keep demanded compile metadata in an excluded/lab struct, and should prior XLOG magic/polarity loss be patched *only* after a disable test proves observable semantic impact?
3. **FieldViewDef home:** Existing ClassView field/ontology object references versus a new small sibling typed offer; prohibit mutable scratch in zero-dependency ClassView trait.
4. **Tenant edges:** Do we propose OGIT LinkSpec and OGAR ActionDef independently rather than a giant normalized class schema? Which application policy owns intertenant permission?
5. **RelAlg boundary:** Does single-row RelAlg handle the exact pure offer, and when is Quack/XLOG/DT64 necessary? How do unsupported join/WHEN/recursion/Bag shapes refuse rather than degrade?
6. **Prepared native memo:** Which authority attests signature, semantic law, value revision, source, layout, ClassView, scope and permissions? Which of these can change independently?
7. **Negative evidence:** What coverage witness would permit closed-world conclusions about source effects? Under PARTIAL harvest the default remains ABSTAIN.
8. **Oracle independence:** What separate source fixture and external implementation can verify actual answers without calling candidate graph functions?
9. **CI gate:** Which declared migrated Codegraph/RelAlg/DT64 call roots can be sealed first so a deliberate value-copy or bypass turns CI red? Do not claim all historical calls.
10. **Product intent:** When is a candidate semantic DTO an application-specific class offer rather than a domain-common concept? Same vocabulary is insufficient to equate transaction effects.

## 4. Evidence and status protocol

~~~text
EXO-RULING-REQUEST:
 issue: EXO-<id>
 from: D-CGMM-<stage> / GitHub PR / exact commit
 current_source: path:line and tool/commit digest
 proposed_semantic_contract: typed inputs, outputs, refusal and owner
 counterexample: independent witness or mutation that fails
 evidence_grade: SOURCE_VERIFIED | LAB_MEASURED | PROPOSED | GATED | NOT_COMPARABLE
 decision: ACCEPT_NARROWLY | REFUSE | DEFER | REQUEST_PROBE
 authorization: explicit owner/ruling reference, *not* an EXO-50 comment
 open_dependencies: linked EXO IDs and affected future PRs
~~~

**A permitted proposal is not an admitted executor.** In particular, an inferred ON/WHEN/DO tuple cannot become executable simply because its source syntax was accepted, and a joined M:N view cannot become a source-owned foreign lookup because a path planner has an index hint.
