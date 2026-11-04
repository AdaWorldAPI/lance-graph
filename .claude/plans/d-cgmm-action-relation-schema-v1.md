# D-CGMM-EDGE: OGIT link + OGAR ActionDef + ONNX-style alias + RelAlg evidence, in one logical schema scratch

> **2026-11-04 · ADDENDUM to [D-CGMM-0](d-cgmm-codegraph-metta-method-builder-v1.md) · PROPOSED.** This is a *crosswalk of existing owned declarations*, **not** a new persistent schema registry, source parser, graph-store, join engine, ClassId minting algorithm, ONNX graph runtime, ActionDef implementation, or tenant authorization. [EXO-155](https://linear.app/exo-red/issue/EXO-155), [EXO-95](https://linear.app/exo-red/issue/EXO-95), [EXO-152](https://linear.app/exo-red/issue/EXO-152), [EXO-121](https://linear.app/exo-red/issue/EXO-121), [EXO-46](https://linear.app/exo-red/issue/EXO-46) remain owner gates.

## 1. Operator's sharpened design

**The logical schema scratch should be shaped like a *combination* of OGIT's object/edge meaning, OGAR's static ActionDef and the ONNX-native polyfill contract.** It should tell us not only "Rails, Odoo and SAP contain a time entry," but **which typed relation connects which roles, whether an edge is many-to-many, in whose tenant/context it is meaningful, what action is permitted on that edge, and which existing algebra answers a read.**

The schema's *language* may be MeTTa-shaped facts/constraints because those are convenient for describing heterogeneous models and candidate rules, but the execution model is **not MeTTa**. The *scratch* temporarily composes already-harvested code and ontology facts; it does not own the resulting ontology, a source row, an admitted method or an execution permit.

~~~text
OGAR Class / Association / ActionDef (static declared behavior)
              + OGIT Entity / LinkSpec / MappingProposal (typed meaning)
              + Codegraph source AST facts (source-grounded effects)
              + external DTO rules (ONNX-like ALIAS / INLINE / REFUSE)
                                   |
               ephemeral MeTTa-shaped logical schema constraint scratch
                 [Role, Edge, Cardinality, TenantScope, Guard, Effect]
                                   |
                     OFFER, REFUSE or ABSTAIN with evidence
                      /                        \
           ontology MappingProposal       DT64 MethodBinding
              [owner admits]               [owner admits]
                      \                        /
             one coordinate/relationship identity; no copied values
                                   |
               Quack/XLOG/mask-risc/ndarray as admitted read execution
                       permitted effects through distinct Rubicon
~~~

**Ontological membership, relation traversal, method behavior, and physical execution are distinct axes.** They may share one conceptual declaration vocabulary, but must not become a second, catch-all engine.

## 2. Exact existing owner crosswalk

| Owner | Concrete verified surface | What it can legally contribute to scratch |
| --- | --- | --- |
| `lance-graph-ontology/src/lib.rs` | `OntologyRegistry` canonical OGIT TTL dictionary, bridges/contexts, `MappingProposal` intake | OGIT URI and namespace, bridge/public label, source/provenance/marking, ontology schema pointer; **not a new identity** |
| `lance-graph-ontology/src/proposal.rs` | `MappingProposalKind::{Entity{Schema}, Edge{LinkSpec}, Attribute}`; `MappingRow` stores `subject_type`/`object_type` and ontology context id | a versioned candidate typed edge, distinguished from entity/property; mapping confidence is not proof of semantics |
| `lance-graph-contract/src/property.rs` | `LinkSpec{subject_type,predicate,object_type,cardinality,codec_route}`; `Cardinality::{OneToOne,OneToMany,ManyToMany}` | typed schema-level subject/predicate/object and many-to-many *constraint*; **not an actual edge population**, unique instance or SQL multiplicity |
| same contract file | `ActionSpec{name,entity_type,target_predicate,trigger}` with Manual/Auto/Suggested | coarse UI/application action declaration, **not equivalent to OGAR ActionDef** |
| OGAR `ogar-vocab::Class` | cross-language structural model, associations, callback, computed fields, methods, validations, source language | static language-independent class shape, source-action candidates |
| OGAR `ogar-vocab::ActionDef` | action identity/predicate/object_class, default subject/temporal/modal, causal state guard, ordered workflow intent, optional state transition / timeout | richer **declared** behavioral semantics, not an executed action |
| OGAR `ActionInvocation` | dynamic action instance, tenant/company local context, provenance and lifecycle | **never** silently collapse a dynamic instance into a static ActionDef or execute it from scratch |
| `lance-graph-onnx/src/lib.rs` | aliases re-export native functions; INLINE adapters validate shape/dtype; unsupported ops REFUSE, no new runtime | pattern for *native method equivalence admission*, **not** a source for business effects |
| `lance-graph-relalg/src/ir.rs` | `Rel` and `Query` describe *one table's row selection* and `Measure`, no raw names or owned rows | normalize a supported single-row-population predicate/fold; **not** a many-to-many join/recursive relation executor |
| `lance-graph-xlog` / Quack | already-held Boolean/Bag relationship methods, constrained folds through mask-risc | eligible relationship query plan **only under existing coordinate, multiplicity and foreign-hop rules** |
| DT64 `MethodBinding` / OGAR vocabulary | immutable class/method/revision/ON×WHEN×ordered DO/ValueDemand | admitted method destination for a scratch *offer*, **not** source AST or new codebook |
| Codegraph `Graded` / residuals | source calls / possible dispatch / unknown edges | compiler-grade evidence; effect/order/transaction constraints stay unknown where harvest incomplete |

**Source verified:** [ontology](../../crates/lance-graph-ontology/src/lib.rs), [proposal](../../crates/lance-graph-ontology/src/proposal.rs), [LinkSpec and ActionSpec](../../crates/lance-graph-contract/src/property.rs), [ONNX facade](../../crates/lance-graph-onnx/src/lib.rs), [RelAlg IR](../../crates/lance-graph-relalg/src/ir.rs), [OGAR IR source](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-vocab/src/lib.rs), [OGAR ActionDef mapping](https://github.com/AdaWorldAPI/OGAR/blob/main/docs/OGAR-AST-CONTRACT.md).

## 3. Relation-level scratch, without minting relation truth

**Conceptual scratch fields (not a new Rust wire struct):**

~~~text
RelationOffer {
  predicate: proposed OGIT/OGAR predicate identity + ontology context
  roles: [subject_role: typed ClassRef, object_role: typed ClassRef]  # minimum binary
  direction: declared and independently observed
  cardinality: 1:1 | 1:N | M:N | unknown
  multiplicity: Set | Bag | unknown   # DISTINCT from cardinality
  reflexive / symmetric / transitive / inverse: only if explicitly proved
  edge_instance_identity: unknown | witnessed  # for association attributes/duplicates
  key_constraints: explicit uniqueness / foreign-key / deletion policy if observed
  scope: tenant owner + named graph + principal/permission + inter-tenant permit
  provenance: source artifact/line + ontology URI/rev + coverage/grade
  candidate_behavior: ActionDef reference + guards/ordered effects/transaction policy
  status: OFFER | REFUSE(reason) | ABSTAIN(unknown cause)
}
~~~

**Hard distinctions:**
- **Cardinality** (M:N) constrains allowable endpoint associations; **Bag multiplicity** counts derivations or repeated edges; **relation existence** is a Boolean support mask. One M:N edge can be a Set even if its association table permits multiple rows, or a Bag if duplicates matter. Never infer multiplicity from `Cardinality::ManyToMany`.
- **Edge identity** matters when the association itself has attributes, revisions, permissions or distinct repeated instances; a pure pair relation and a reified association entity are not interchangeable.
- **Tenant** is not just an ontology `bridge_id`. The existing registry indexes `(bridge_id,public_name)` and OGIT URI; `SchemaPtr::ontology_context_id` describes an ontology context. Neither alone is a per-row tenant permission. A cross-tenant link needs explicit two-endpoint authorization and a witnessed relation scope. Missing tenant policy => ABSTAIN or REFUSE, not global graph adjacency.
- **Direction** (A→B) does not imply inverse (B→A) or symmetric role exchange. `XX/XY/XZ` virtual bus names are *operand projections*, not relation schema semantics.
- **Many-to-many does not imply a Cartesian product.** The actual edge population determines support, and joining two populations still costs and requires admissible shared identity routing.
- **Recursion** is an external driver gate (EXO-73); `LinkSpec` alone does not license transitive closure.
- **Write effects** require a separate action-invocation/transaction/Rubicon admission. `MappingProposalKind::Edge` is not a command to insert an edge.

## 4. Example: Billable work entry × project × approval × tenant

**Illustrative semantics, not observed Rails/SAP source or runnable MeTTa syntax:**

~~~text
OGIT entity: BillableWorkEntry; application-local: Odoo.account.analytic.line,
 SAP.CATS, Rails.TimeEntry (Rails parity unverified).
Link candidate: (Worker, assigned_to, Project) [M:N]
Scoped role: a Worker in tenant A may be linked to a Project in tenant B
  ONLY with explicit policy and source/relationship witness; otherwise REFUSE.

Event: Approve(BillableWorkEntry)
OGAR ActionDef candidates can differ:
  Odoo [dependency/callback, company scope; if harvested]
  SAP [BAPI/CATS transaction; source trace required]
  Rails [ActiveRecord callback/transaction; Ruby harvester required]

DT64 offer: ClassId concepts, app-specific method refs, versioned ClassView,
  ON / WHEN / ordered DO, declared truth/multiplicity, no executable effects yet.

Read example: Count approved entries by project under one tenant/permission;
  only if source/method/relationship witnesses and Bag semantics agree.
~~~

**Counterexample:** Tenant A and B share an OGIT URI and both use the label "approves", but A's method changes approval state, while B's triggers an accounting posting and rollback. The schema scratch must say **SameConcept / DifferentAction**, not normalize both into one executable method.

## 5. Admission mapping: OGIT vs OGAR vs DT64 vs ONNX

**No new universal schema owner.** Scratch should emit distinct *offers* to existing owners:
1. **Ontology OFFER:** a `MappingProposalKind::Edge{LinkSpec}` or entity/attribute mapping, with observed source and explicit tenant/context metadata. Only ontology's canonical registry may admit it.
2. **Behavior OFFER:** OGAR `ActionDef` or existing OGAR Class association/callback declarations, with provenance and effect/temporal/modal semantics. The OGAR codebook owner admits identity.
3. **Method OFFER:** DT64 `MethodBinding` using an existing ClassView basis, MethodId/SigRev, ON/WHEN/ordered effects. DT64 stores it; `ExecPermit`/execution ownership separately admits a limited native program.
4. **Native ALIAS offer:** like ONNX, exact semantic compatibility may select an existing native operation. A narrowly specified adapter may be needed (and separately approved), or REFUSE; no new handler invented to rescue a missing vocabulary.
5. **Relational query OFFER:** a non-effectful Set or Bag query on witnessed relationship planes under Quack/XLOG/RelAlg scope. `RelAlg::Rel` alone cannot represent arbitrary M:N joins, so it is not extended into another Datalog AST.

## 6. Independent many-to-many falsifier ("edge tenant" test)

**Independent fixture:** Tenant A has workers W1,W2 and projects P1,P2; tenant B has W3/P3. Relation edges:
- A: (W1,P1), (W1,P2), (W2,P2).
- B: (W3,P3).
- A duplicate (W1,P2) for Bag-control (same Boolean support, higher multiplicity).
- A deliberately forged cross-tenant (W2,P3) that is **present in a raw table** but not admitted by the tenant-permission graph.
- An "approval" method with the same label in A and B, but B has an additional side effect.

**Independent oracles:** source rows keyed by actual global identity and tenant scope, enumerated outside Codegraph/Quack; OGIT type oracle; per-application method-effect and auth fixtures. Compare support query, Bag count, cardinality constraint, tenant authorization, inverse traversal and approval-method parity separately.

**Six deliberate kills:**
1. Drop tenant check: cross-tenant link appears.
2. Replace Bag with Boolean: duplicate is lost.
3. Join on local position instead of global identity: wrong worker/project.
4. Reverse edge direction automatically: inverse false-positive.
5. Erase an ordered side effect: same-name methods falsely equivalent.
6. Turn a source effect UNKNOWN into absent: unsafe method admitted.

**Expected result at CGMM-0:** only a test specification. Nothing executed. Future lab uses XLOG/Quack's existing relation orientation and source identities, never invents a relation engine. Attach exact source rows, expected counts, mutation results and owner gates to the PR.

## 7. Why this unifies the architecture without creating a giant DTO

The reusable abstraction is a **typed offer + constraint scratch + owner-specific admission**, rather than one gigantic persistent catch-all record for application classes, graph edges and effects.

- The *same* scratch vocabulary can describe an entity, binary/many-to-many relationship or action behavior, but each output goes to its rightful owner.
- The *same* native compiled method may be reused across application-specific DTOs **only if** a full semantic witness (types, null/Bag, effects, permissions, source layout and versions) proves equivalence.
- NNUE may rank candidate structural equivalences offline, but a high score never asserts identity, permits cross-tenant access or makes an ActionDef executable.
- The hot loop remains borrowed SoA + mask-risc/ndarray; no MeTTa interpreter, ontology TTL walk or source AST decoding is in that loop.
- Because role/cardinality/tenant semantics are *explicit*, the same scratch engine may describe non-business graph tenants later without special casing Odoo, Rails or SAP.

**The next discriminating step is not NNUE training.** Build one bounded lab where Codegraph's graded source facts plus ontology's known `LinkSpec::many_to_many` can generate a role-typed relation offer, and demonstrate a deliberate unknown effect or tenant policy blocks its conversion to executable DT64 behavior. That is the causal firewall between *description* and *authorization*.
