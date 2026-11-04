# D-CGMM-IMPLEMENTATION-0: executable research blueprint, from source evidence to lawful native method

> **2026-11-04 · DETAILED PLAN · DOCS ONLY · NOT AN IMPLEMENTATION.** This is an expansion of [D-CGMM-0](d-cgmm-codegraph-metta-method-builder-v1.md) in draft [PR #1958](https://github.com/AdaWorldAPI/lance-graph/pull/1958), not a new source graph or general-purpose MeTTa runtime. Tracking: [EXO-155](https://linear.app/exo-red/issue/EXO-155); [EXO-50](https://linear.app/exo-red/issue/EXO-50) is an append-only communications bus, not permission. Owner decisions and their exact exit artifacts are in the [EXO gate ledger](d-cgmm-exo-gates-v1.md); the [fixture matrix](d-cgmm-executable-fixtures-v1.md) defines independent tests. The existing [staged charter](d-cgmm-staged-pr-charter-v1.md) remains canonical for CGMM-0…8.
>
> **Operator objective:** make MeTTa-shaped logical schema constraints a **cold class-method builder** attached to witnessed ClassView/FieldViewDef roles. Separate **durable source facts** (Ruff/Charon/Kythe) from **cold Datalog demand analysis** (Crepe/Soufflé/XLOG compile artifacts) and **lazy Codegraph query masks**. Convert only semantically and permission-eligible, owner-approved declarations to immutable OGIT/OGAR/DT64 contracts, and reuse existing RelAlg/R2IL/mask-risc/ndarray execution. Never reconstruct canonical business objects or introduce a second execution engine.

## 0. Status and unnegotiable source facts

| Surface | What current source establishes | What this plan MUST NOT call shipped |
| --- | --- | --- |
| `crates/lance-graph-codegraph/src/artifact_bin.rs` | A `CrateFile` owns one validated `Vec<u8>`, accessors borrow decoded lanes/labels, FUNDECL/PROV section tags exist; writer is `ruff_charon::store_bin` | All read/write effects, receiver typing, Rails/SAP AST, or zero-copy across every producer |
| `crates/lance-graph-codegraph/src/graph.rs` | `Codegraph` owns lanes, name interner, local `Pos` table, `MapTable`, `Cube` of six relations, masks; staged refresh; `rconf`/`rposs` `OnceCell` | Stable Kythe source identity equaling `Pos`, a completely demand-lazy store, or append-only historical truth for arbitrary source updates |
| `crates/lance-graph-codegraph/src/query.rs` | `Graded { confirmed, possible, unknown }` masks, bound `partners` joins, `or_partners` loops over positions | Closed-world negative evidence from `MAY_DISPATCH`, implicit joins on foreign identity, zero-cost repeated binding |
| `crates/lance-graph-contract/src/class_view.rs` | `ClassView` above SoA, field-position basis, `FieldMask` pure presence, `WideFieldMask` and `lens_fields(full_classid)` | A mutable MeTTa scratch field inside the trait or a new ClassId issuer |
| `crates/lance-graph-ontology/src/proposal.rs` | `MappingProposalKind::Edge { LinkSpec }` and explicit ontology source context | A true tenant permission, an actual edge instance, or M:N multiplicity |
| `crates/lance-graph-contract/src/property.rs` | `LinkSpec`, `Cardinality::ManyToMany`, `ActionSpec` | OGAR `ActionDef` equivalence, a join engine, write permission |
| `crates/lance-graph-relalg/src/ir.rs` | Single-population `Rel`/`Query` mask/atom/selection/reduction vocabulary | General M:N join, recursive fixpoint, arbitrary method effects |
| `crates/lance-graph-relalg/src/join_body.rs` | GYO α-acyclicity analysis over variable sets | Executed Yannakakis/WCOJ plan or a proof of foreign-hop admission |
| `crates/lance-graph-r2il/src/binding.rs` and `binding_native.rs` | Validates method/binding, guard/truth restrictions; can prepare/run a subset of native mask-risc folds using borrowed planes | General executable WHEN, K3, fold-fed-fold, or production `ExecPermit` |
| [Crepe/Soufflé/XLOG source archaeology](../board/entries/2026-11-02-coresearch-xlog-souffle-crepe-dr3-archaeology.md) | Crepe bound/free/index information, Soufflé-like demand, XLOG compiler metadata and losses, discarded polarity/magic report, `crepe_plan!` D1/D2 drift | A proven compatible compiler plan or license to import external runtime |
| [#1944](https://github.com/AdaWorldAPI/lance-graph/pull/1944) | Bounded, ground relational common subset from separate frontends | General MeTTa unification/rewrite or a new Datalog engine |
| [#1955](https://github.com/AdaWorldAPI/lance-graph/pull/1955), [#1956](https://github.com/AdaWorldAPI/lance-graph/pull/1956) | Local borrowed register proof; correct but slower-than-native carry ALU in measured cases | Remote/storage source admission or universally superior Abakus |

**Design target, not code:** `SourceFactLease → DemandEvidence → LazyCodeView → FieldViewDefOffer → ConstraintScratch → {OntologyOffer, ActionOffer, MethodOffer} → owner admission → PreparedNativePlan`. All names here are *descriptive*, not public Rust API. Preserve existing functions/types and request ownership before any API change.

## 1. Source intake: fact identity must survive a lazy query

A source harvester should produce *atomic, provenance-grounded facts*, rather than a monolithic serialized AST object. The minimal **conceptual** source receipt is:

~~~text
SourceFactReceipt {
  producer: (language, tool, version, artifact schema version),
  source: (repository identity, commit/revision, file identity, span),
  symbol: stable producer-scoped node key + source owner,
  predicate: kind (defines / calls / may-dispatch / reads / writes / ...),
  object: stable producer-scoped key OR typed scalar/role,
  confidence: compiler-confirmed | possible | unresolved | missing-producer,
  extraction: COMPLETE | PARTIAL + explicit residual reasons
}
~~~

**No new durable ID authority:** Link source-producer identities to Codegraph's local interner/`Pos` only under a per-artifact validated table. Never equate Kythe VName or Charon `FunDecl` with global DR identity, MethodId, ClassId or local ordinal.

**Binary-lane policy:** Existing `.rows.bin` is versioned/validated by an independently implemented reader. To add effect/receiver/tenant facts, first prove that the producer actually emits them and identify how its writer/reader will evolve without breaking older must-understand sections. An optional sidecar in a lab may serve as a negative-control input; no production sidecar format is silently admitted.

**Atomic refresh:** On parser refusal or broken provenance, `Codegraph::prepare_refresh`/commit semantics must leave the previously committed graph and source witness unchanged. Newly observed absence is not `FALSE` under a PARTIAL producer.

**First falsifier:** two same-named methods in different source versions, one resolved and one dynamic call, one documented write whose producer drops it. An incorrect label-only interner or loss of the write fact must prevent a method-equivalence OFFER.

## 2. Demand compiler: use Crepe/Soufflé insight but not their executor

Demand planning is a *pure*, cold analysis of exactly which relations/columns need querying for a method-building question. Conceptual contract:

~~~text
CodeDemandV0 {
  roots: validated source-node IDs or admitted ClassView role refs,
  relation_atoms: typed (predicate, bound/free coordinates, direction),
  grade_floor: confirmed | include_possible | include_unknown,
  output: graded_mask | source_span_mask | field_role_offer | method_offer,
  semiring: B | N, unknown_mode: OpenWorld | CompleteWitnessed,
  source_artifact_generation, producer_schema, ontology_class_revision,
  scope/tenant/principal, recursion_policy, max_hops/max_atoms/max_bytes
}
CodeDemandPlanV0 {
  demanded fact lanes, bound columns, legal ordering, proof obligations,
  required coverage/strata/polarity witnesses, identity mapping, budgets
}
~~~

**Do not turn this into a new runtime planner**. Crepe bound/free metadata and Soufflé magic demand are *candidate compile records* only. XLOG's magic-set/polarity/stratum records must be source-confirmed and tested for loss before using them as admission facts. `crepe_plan!` vs `crepe!` D1/D2 drift explicitly prevents trusting a plan without a comparator.

**Current baseline:** Codegraph already executes `Cube::scan→bind→join→support` lazily for a bound symbol but eagerly builds its six relations at intake and performs per-set-bit repeated `partners` inside `or_partners`. First benchmark this exact baseline. A demand-scoped version must not be credited for a lower cold cost that it merely moved into the caller.

**Conservative first implementation:** start with a small *excluded lab or test fixture* that constructs a demand descriptor, resolves one bound symbol using the existing Codegraph query API, and measures source facts touched. If the data structure lacks selective construction, do not pretend fewer fact lanes were built. Measure masks and binding counts; introduce demand pruning only after its independent semantic oracle passes.

**Logical errors are mandatory refusals:** partial source negation, non-stratified negation, unsupported recursive SCC, unsupported indexed foreign gather, stale source revision, mismatch of Bag vs Boolean Set or wrong binding direction. A demand compiler cannot turn UNKNOWN into FALSE to simplify a plan.

## 3. FieldViewDef: exact semantic boundary

**Proposed immutable offer, never a mutable ClassView trait extension:**

~~~text
FieldViewDefV0 {
  owner_class: ClassId + ClassView schema/behavior-lens revision,
  role: ScalarField | EdgeSubject | EdgeObject | ActionOperand,
  referent: FieldRef | OGIT LinkSpec role | OGAR ActionDef operand,
  type: field dtype, signedness, null/absence, presence basis,
  relation: optional (predicate, orientation, cardinality,
                      Set|Bag, edge-identity, source/tenant scopes),
  admissible_reads: pure method candidates with source evidence,
  evidence: source facts + artifact generation + confirmed/unknown masks
}
FieldConstraintScratchV0 {
  declarations: bounded, borrowed FieldViewDef references,
  candidate_masks: logical MATCH/CONTRADICTION/UNKNOWN over a distinct basis,
  relational_constraints: typed Codegraph/Quack relation references,
  rule_version, source_version, class_version, permission scope,
  result: OFFER | REFUSE(reason) | ABSTAIN(missing_evidence)
}
~~~

**Strict basis rule:** `FieldMask` and `WideFieldMask` continue to encode instance presence ONLY. Candidate logical masks must use a distinct, typed logical basis, not borrow bit 0 and reinterpret it as "matching". For behavior lenses, require `lens_fields(full_classid)`; do not silently fall back to the u16 concept half. `FieldViewDef` is **not** OGAR's rendered `FieldView`.

**Logical rule universe:** finite, typed, bounded MeTTa-shaped facts and equality constraints over source roles; no general MeTTa interpreter, MORK trie, executable rewrite, automatic transitive closure, AST copy or source-driven script execution. Scratch can be memoized with full source/ontology/class/rule/scope witness but remains disposable, *never an admitted MethodId*.

**Two-way workflow:** Codegraph may provide source evidence supporting or contradicting an existing DT64 method; any mismatch produces a **diagnostic candidate** for review, not direct mutation of DT64 or recompiled production methods.

## 4. Entity, edge, action: three distinct offers

The same temporary constraint workspace supports three **typed destinations**:

| Offer | Existing canonical owner | Minimum witness | Common refusal |
| --- | --- | --- | --- |
| Entity/field mapping | OGIT ontology `OntologyRegistry`, `ClassView` owner | URI/context + source type + source/source-revision + field identity | Field label match with type/layout conflict |
| Relationship edge | `MappingProposalKind::Edge { LinkSpec }` / RelationshipMap owner | subject/object ClassIds, predicate URI, orientation, cardinality, edge instance/multiplicity when needed, tenant authorization | M:N mistaken for Bag; cross-tenant link without permit |
| Action/method | OGAR `ActionDef` and DT64 `MethodBinding` owners, separately | signature/class/ON, ordered WHEN/DO/effect/transaction witnesses and scope | Same-name method with different ordered effects or missing source writes |

A single ontology URI is **not** sufficient tenant authorization. An M:N declared association is **not** an observed edge population. Bag multiplicity depends on actual derivations, not `Cardinality::ManyToMany`. A reversed edge isn't automatically an inverse or symmetric relation. An event can have executable effects even if its source syntax resembles a read-only query.

For Rails/Odoo/SAP examples, use synthetic labelled profiles unless independently harvested source and transaction traces exist. Specifically, an Odoo ORM callback and SAP approval transaction are not interchangeable because their user-facing names match.

## 5. Native method lowering and refusal map

After an owner separately approves a **pure, finite, typed** offer, a cold lowerer may reuse existing RelAlg IR for *one witnessed population*:

| Schema/read shape | Candidate mapping to existing algebra | Admission outcome |
| --- | --- | --- |
| `FieldRef` U32 == known constant | Typed `Rel::Atom` U32-equality over an existing lane | OFFER after field, source, class and null witnesses |
| I32 amount ≥ bound | Existing I32 compare + `Rel::And` with allowed population | OFFER if exact signedness and absence handling |
| Keep support or count | Existing `Query::Measure` / native terminal | OFFER under one-population, Set/Bag interpretation |
| Binary same-source mask combination | `Rel::{And,Or,Not}` where supported | OFFER only for same coordinate universe and admitted truth law |
| OGIT M:N role traversal | Quack/XLOG relation planning outside single-row RelAlg | **REFUSE this lowerer;** owner-specific relation OFFER only |
| Cross-source foreign join | Admission-B/EXO-46/140 | **BLOCKED** until owned lookup/index and scope established |
| Conditional WHEN / K3 truth | DT64 can store, current R2IL prepare explicitly refuses | **GuardNotExecutable / ThreeValuedNotExecutable** |
| OGAR action with effects, writes or rollback | OGAR behavior owner and transaction owner | Declaration OFFER, **not executable read** |
| General MeTTa rewrite / recursion / eglog congruence | No admitted runtime analogue | **REFUSE**; external oracle only |

**No hidden fallback** to a tuple interpreter, DataFusion, Crepe/Soufflé/datafrog evaluator or implicit value gather. `RelAlg::join_body::is_alpha_acyclic` is a structural classifier, not a many-to-many executor. `ExecPermit` is lab permission only.

## 6. Three distinct memoization levels, with invalidation law

1. **Compile/demand-plan memo:** recipe keyed by source artifact generation, producer version, typed query roots, bound/free pattern, stratification/polarity, semiring, compiler-rule version and permission scope. It may retain plan *metadata*, not copied source objects.
2. **Semantic constraint scratch memo:** optional cache of offers/rejections keyed by source fact digest, ClassView/ontology/OGAR revisions, tenant/permission/scope, rule basis and relevant evidence completeness. Its eviction/recompute changes no previously admitted truth.
3. **Admitted native method binding:** only after real ownership approval, reuse `MethodRef`/`SigRev`/`ClassView`/OGAR body revision/IR/semiring/source-frame-layout/scope witness, checking any value-derived registers against value revision. Hot execution borrows canonical SoA rather than re-parsing AST or constructing `PSCustomObject` rows.

**No cache-by-name** and **no cache-by-FNV-identity alone**. Any source revision, codebook shift, method change, semantic law change, permission revocation or lease expiry must trigger revalidation or refusal.

**Abakus is *not* the schema solver.** Its existing half/full-adder words may execute numeric/threshold operations under ndarray, but #1956's measured carry implementation lost to native words in eight tested shapes. Compare only where arithmetic semantics are required. No public CPU-layout emulator or learned-codebooks as automatic plan substitutes.

## 7. Proof sequence, data collection and STOP rules

**CGMM-1 (fact intake)** first checks the writer→reader edge and source identity; no producer-generated effects means **UNKNOWN**, not fabricated writes. Independent oracle is the original source fixture. Deliberately drop a WRITE fact and verify the converter refuses calling that method pure.

**CGMM-2 (demand/lazy)** compares baseline Codegraph eager six relations and current `partners/or_partners` against one bounded demand query. Measure ingestion, cold plan, query Q=1/2/8/16/64/256, retained bytes, masks, graph refresh, physical CPU time, query-site count and traversal. Kill a `MAY_DISPATCH` truth upgrade and stale cache. Report NO BENEFIT if demand loses.

**CGMM-3 (FieldViewDef)** uses source-backed ClassView positions and one 109-field wide class; prove different schemas with identical labels don't alias and behavior-lens full class ID required. Candidate masks differ from instance-presence masks, and scratch eviction changes no admitted schema. Kill a class schema basis mismatch.

**CGMM-4 (OGIT/OGAR)** uses M:N tenant fixture plus duplicate edge and two same-name actions with different effect order. Independent oracle enumerates global IDs by tenant and ordered effects. Kill tenant filter, Bag support collapse, inverse assumption, an action effect; all must be detected.

**CGMM-5 (RelAlg polyfill)** runs an actual same-universe scalar/measure query on borrowed source, with independent scalar reference. It must reject an M:N join, K3, effectful or guarded method. Compiler-produced scalar/mask values are not identical to ClassView metadata.

**CGMM-6 (DT64 admission)** only when owners authorize: create immutable MethodBinding with correct method and ClassView basis; prepare through current R2IL native path. Compare exact result and refuse stale schema/source/scope and guarded WHEN. No new R2IL interpreter.

**CGMM-7 (memo)** reports Q and cold vs hot economics including invalidation/eviction; compare native Boolean/byte and optional Abakus only where their semantics match. Never claim speedup from one benchmark without whole-query denominator.

**CGMM-8 (audit)** uses independent oracle and Codegraph reachability to identify access-capability bypasses in chosen migrated roots, preserving dynamic unknown paths and scope. EXO-104/105/109 prohibit tuple-shaped oracle leakage into production. NNUE only proposes candidate order.

**Common STOP:** Source facts not independently verifiable; missing source-effect harvest; proven incompatible class or tenant scope; emitted Codegraph local Pos treated as durable class identity; implicit primary-field or row reconstruction; unowned lookup/foreign gather; unsupported general rewriting or recursive evaluator; method admitted from name/NNUE score; CI falsely asserting global source completeness.

## 8. Publication packet per future staged PR

Each stage must include:
- **START:** base main SHA, relevant upstream tool/artifact revs, exact EXO issue and owner.
- **CONTRACT:** precise input/outputs/witnesses, obligations, refusal and fallback policy.
- **FIXTURE:** independent fact and value oracle, one positive/one adversarial instance; seeded reproducibility.
- **DISABLE:** deliberate mutation which turns at least one specific nonvacuous test red.
- **NUMBERS:** ingestion cost, cold bind, warm query, Q, bytes retained/copied, source/route accesses, p50/p95/p99 and variance; do not compare incomparable hot/cold arms.
- **VERDICT:** PASS | FAIL | NOT_COMPARABLE | NOT_RUN | BLOCKED_OWNER with source links and bounded claim.
- **HANDOFF:** PR + commit hash, board cross-links, append-only EXO-50 message; separate operator merge/admission decision.

**First recommended executable PR:** CGMM-1 source fact loss and identity census in an excluded test workspace, *followed* by CGMM-2 demand-projection economics. The larger FieldViewDef DSL and method-binding program must not race ahead of demonstrably reliable compiler source facts.
