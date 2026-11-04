# D-CGMM-LAZY-0: Kythe facts × Datalog demand × lazy Codegraph × FieldViewDef × DT64

> **2026-11-04 · ARCHITECTURE ADDENDUM · PROPOSED / NOT IMPLEMENTED.** The central task is to **separate durable source identity and facts from on-demand relational evaluation and the class/method semantic lens**. This does not adopt Crepe, Soufflé or MORK as a production executor; it does not introduce a second graph store, general-purpose rewrite engine, new F###, promoted WORLD facts, output row cache or remotely dereferenceable register. [D-CGMM master](d-cgmm-codegraph-metta-method-builder-v1.md) · [Action/Edge schema](d-cgmm-action-relation-schema-v1.md) · [staged PR charter](d-cgmm-staged-pr-charter-v1.md) · [falsifiers](d-cgmm-falsifier-matrix-v1.md) · [EXO-155](https://linear.app/exo-red/issue/EXO-155).

## 0. The architectural inversion

**Kythe-shaped facts are evidence, not a query engine. Datalog-shaped analysis is a compiler/demand vocabulary, not a second evaluator. Codegraph is a lazily resolved *view* of the requested evidence, not a duplicate AST object graph. FieldViewDef is a typed, ClassView-based semantic offer, not a new ontology. DT64 admits immutable method contracts, and only current native algebra executes them.**

~~~text
Ruff / Charon / other admitted source harvesters
      |
      | source-specific nodes and edges + stable producer identity
      v
  Kythe-shaped compiler evidence and existing .rows.bin/FUNDECL/PROV lanes
      |  source identity, revision, provenance, coverage grades
      |  no guessed missing facts, no revived source AST objects
      v
  COLD DATALOG COMPILE / DEMAND ANALYSIS
  Crepe-style bound/free index requirements, Soufflé-style magic demand
  stratification/negation rules and XLOG compile metadata, *as DATA*
      |  required relations, bindings, polarity, semiring, termination
      |  explicit unsupported/refused shapes, no imported runtime
      v
  LAZY CODEGRAPH SELECTOR
  bound relation/symbol queries -> typed support and evidence masks
  incremental/epoch-scope result cache only if validated; otherwise recompute
      |  confirmed / possible / unknown / residual preserved
      v
  ClassView + FieldViewDef OFFER / disposable MeTTa-shaped scratch
  finite source-to-concept mapping, roles, action/effect/tenant constraints
      |          |
      |          +---- OGIT LinkSpec M:N + OGAR ActionDef owner offers
      v
  DT64 MethodBinding: immutable ON × WHEN × ordered DO
      | owner admit, source/class/method/scope/exec witness
      v
  Existing RelAlg / Quack / R2IL -> mask-risc -> ndarray
      | source-owned borrowed SoA, no mandatory canonical value gather
      v
  scalar / relation mask / typed refusal, no automatic WORLD write
~~~

**What is actually lazy today:** `lance-graph-codegraph/src/graph.rs` already interns symbols into local `Pos`, has DR `MapTable` and XLOG `Cube` relations plus `rconf` and `rposs` `OnceCell` caches. `query.rs` performs bound joins on demand, returning graded masks rather than tuple objects. Yet intake eagerly constructs six core relations and row/vector masks; `or_partners` iterates input positions and invokes repeated bound joins. Claim *partial laziness*, not an already-completed lazy Codegraph engine. The first optimization may only need **demand metadata and fewer repeated bindings**, not a rewrite of the graph.

## 1. Three stores must NOT be confused

| Layer | Existing source/owner | Durable content | What is only a view |
| --- | --- | --- | --- |
| Source artifacts / Kythe-like evidence | Ruff, Charon, source producer, `.rows.bin` and provenance | Symbol identity, AST span, resolved/may/dynamic calls, typed effects where actually harvested, producer revision | Codegraph interned `Pos` and masks |
| Canonical relational register | DR `MapTable` / `Frame` / source-owning SoA and XLOG/Quack execution | Source/coordinate identities, admitted relationship planes and source values | A demand-scoped relation projection or semantic field lens |
| Semantic declarations | OGIT ontology/`LinkSpec`, OGAR class/action codebook, DT64 `BindStore` | Admitted ClassId, MethodId, schema and method revisions, `ActionDef` | `FieldViewDef` proposals, MeTTa constraint scratch, compiled temporary query plans |

**Kythe VNames or equivalent stable symbol identity need their own namespace/producer/source revision.** Do not treat local Codegraph `Pos`, Charon function ID, human symbol name, DR local ordinal or DT64 ClassId as identical IDs. If a source producer does not emit stable identity or needed effect evidence, record `UNKNOWN_SOURCE_ID` / `UNKNOWN_EFFECT` rather than inventing a link.

## 2. Take the *compile artifacts* from Crepe/Soufflé/XLOG, not their runtime

[2026-11-02 source archaeology](../board/entries/2026-11-02-coresearch-xlog-souffle-crepe-dr3-archaeology.md) documents the traps:

- **Crepe**: binding analysis (which relation arguments are bound or free) chooses index needs; `crepe_plan!` emits planning data but its recorded D1/D2 drifts from `crepe!` need a falsifier before trusting it.
- **Soufflé**: demand/magic rewriting constrains which facts/rules matter to a query; stratification, negation and recursion are substantial semantic constraints. Do not import its runtime, indexes or automatic recursion by analogy.
- **XLOG compiler**: source archaeology identifies already-derived SCC/stratum, magic-set reports, negation polarity, join ordering and selectivity metadata; some are dropped at compile boundaries. *Recover as named admission/planning evidence only if an independent probe demonstrates loss matters.* XLOG execution is not admitted by an AST import.
- **Kythe**: node and edge facts provide source-grounded anchors and cross-references. Facts that are only `MAY_DISPATCH` or unresolved are not negative knowledge.
- **AST**: syntax structure belongs to a producer; Codegraph consumes typed facts, and only fetches a bounded source AST fragment when a declared evidence gap requires it. Never rehydrate full AST trees as a standard query operation.
- **datafrog**: an independent finite Datalog/fixpoint oracle for a bounded case; EXO-73 owns recursion execution policy. Crepe, Soufflé and datafrog are *oracles/comparators*, not runtime owners.
- **MORK**: a set-oriented external structural oracle; differences from MeTTa multiset binding and Datalog Bag semantics must remain typed and measured. EXO-79 license gate remains.
- **NNUE**: optional offline ranking of candidate patterns; does not issue evidence, ClassIds or execution permits.

**No universal closed-world negation:** absence of a `CALLS` fact under partial coverage cannot prove a method has no writes or caller. If a program needs stratified negation on complete facts, its required completeness witness must be explicit; otherwise the compiler returns `ABSTAIN(UNKNOWN_COVERAGE)`.

## 3. Demand descriptor (conceptual, NOT Rust API)

~~~text
CodeDemand {
  roots: source symbol ids / owner-validated ClassView role ids
  required_edges: [CALLS, MAY, IMPLEMENTS, READS, WRITES, HAS_FIELD,
                   EXTENDS, DECLARES_ACTION, LINK_ROLE, ...]
  bind_pattern: per relation argument bound/free, direction and depth
  evidence_floor: confirmed | allow_possible | include_unknown
  semiring: BooleanSet | CountingBag (with overflow law)
  logic: two-valued | Kleene (with complete-world declaration)
  source: repository/language/artifact digest/generation/producer schema
  ontology: OGIT/OGAR class and method revision (if relevant)
  scope: owner/tenant/permission and cross-source admission
  budget: max relations, symbols, hops, rounds, scratch bytes
  terminal: support mask | graded mask | relation offer | method offer
}
DemandPlan {
  required fact lanes, proven safe binding/index/order metadata
  snapshot and identity witnesses, compilation evidence and refusals
  no arbitrary executable native code and no new persisted index
}
~~~

**Lazy materialization policy:** a demand may prepare an *ephemeral, bounded support mask* or a few existing XLOG/DR relation views, but not source AST objects, newly owned source-value arrays, a separate always-on call graph or all tenant edges. The same semantic plan may be reused if every source, method, scope and compiler-version witness matches. Different roots can share an immutable admitted plan, while bound source positions and resulting masks remain **request scoped**.

Performance must measure **cold demanded-fact binding + eager baseline + query cost**, not only a prepared fold that hides initialization. When query cost is dominated by index lookup, proposal is a *demand memo* on existing Plan/DR support, not automatic global hash-table construction.

## 4. ClassView plus FieldViewDef: the narrow schema glue

`lance-graph-contract/src/class_view.rs` explicitly defines the class as a meta lookup flying above SoA; `fields(class_id)` resolves a stable ordered field basis, while `lens_fields(full_classid)` resolves a behavior-lens basis. `FieldMask`/`WideFieldMask` mean **instance presence**, not semantic candidate truth, permissions or logical negation.

**FieldViewDef** is a proposed *cold, immutable, typed offer referring to those existing fields*, not the existing `ogar-render-askama::field_view::FieldView`, which is a render-only addressed UI carrier. The MeTTa-shaped **FieldViewConstraintScratch** is a short-lived matcher of supported/contradicted/unresolved class, field, edge and ActionDef roles; it does not modify ClassView or create a new method codebook.

### 4.1 One declaration, several owner-specific views

~~~text
FieldViewDef (proposal only):
  class: existing ClassId / full behavior lens + schema basis
  role: scalar | relation_subject | relation_object | action_operand
  reference: FieldRef or LinkSpec role or ActionDef operand (source-witnessed)
  types/truth: dtype, absence, cardinality, semiring, tenant scope
  provenance: compiler/ontology/source receipts and unresolved evidence
  read_intent: optional pure selection/fold, no action executed

MeTTa-shaped constraints (illustrative, not parser syntax):
  HasField(Worker, project_id)
  HasLink(Worker, assigned_to, Project, M:N, tenant=A)
  HasAction(ApproveEntry, ordered_effects=[...], when=guard)
  Writes(method, field) [SOURCE_CONFIRMED | UNKNOWN]
  EquivalentUnder(methodA, methodB, typed-source-witness) [CANDIDATE]

Scratch: OFFER | REFUSE(reason) | ABSTAIN(missing evidence)
~~~

A spaCy-like `Doc/Token/Span/Matcher` pipeline is a **useful cold annotation interface analogy**, not a second crate or text-sequence representation. A Codegraph edge or M:N link is not a contiguous `Span`. A source span can ground the edge's declaration, never stand in for its relationship population.

### 4.2 Native polyfill boundary

If a `FieldViewDef`'s read is confined to **one witnessed row universe** and existing numeric lanes, it may lower to existing `RelAlg::Rel::{All,Empty,Plane,Atom,And,Or,Not}` and `Query::Measure::{Count,Sum,Min,Max,Keep}`, then `RelAlg::run` → R2IL/mask-risc. RelAlg's IR has no strings, owns no source rows, and does not contain a general join or recursive execution program.

**Examples:**
- `status == 2` as verified U32 lane -> `Rel::eq_u32(lane,2)`, with absence/truth checked.
- `amount >= 100` on a verified I32 lane -> `Rel::i32(lane,Cmp::Ge,100)`.
- typed Boolean intersections -> `Rel::And` / `Or` / `Not` on the *same* frame and truth law.
- M:N ontology link -> **relation OFFER only** to Quack/XLOG owners; `RelAlg::join_body::is_alpha_acyclic` is a structural classifier, not a join executor.
- OGAR effect/ActionDef or DT64 guarded WHEN -> declaration OFFER / execution REFUSE pending owner and vocabulary; `binding_native` does not generally execute these.
- source AST rewrite, arbitrary MeTTa unification, recursion, tenant-forbidden foreign hop -> typed REFUSE/ABSTAIN, never an alternative code path.

## 5. Demand and memoization laws

- **L1 Identity**: source Kythe/node identity ≠ Codegraph interned `Pos` ≠ ClassId/MethodId ≠ DR local ordinal.
- **L2 Evidence**: confirmed, possible and unknown stay distinct under projection, negation and derivation; partial extraction never proves absence.
- **L3 Lazy**: only requested facts/roles are projected, with bounded work; do not manufacture complete AST objects or an independent always-on graph.
- **L4 Semantics**: Set/Bag support, M:N cardinality, SQL Kleene and effects never collapse by name or mask width.
- **L5 Ownership**: Source owns AST/facts; Codegraph owns graded query view; ClassView owns basis; OGIT/OGAR/DT64 admit meaning; existing algebra executes.
- **L6 Cache**: snapshot/producer/ontology/class/method/semiring/scope versions form a complete invalidation witness. Scratch eviction cannot change truth.
- **L7 Economics**: track cold intake, eager-vs-demand preparation, allocations, retained memory, query latency, partial coverage and cache hit/miss separately.
- **L8 Failure**: no silently empty results on unsupported recursive, foreign, missing source, unknown effects or unauthorized tenant tasks.

## 6. Smallest runnable falsifier

**Fixture:** two crates with 20 source symbols. Include five confirmed `CALLS` edges, two `MAY_DISPATCH` edges, one unresolved dynamic edge, source spans, one field read and one write, one OGIT `LinkSpec(Worker,assigned_to,Project,ManyToMany)` declaration, two application-local method candidates with same name but different effects. Use independent global-ID source fixture and a separately specified scalar/Datalog oracle.

**Demand D1:** From one known method, ask only confirmed callees and fields read. The candidate must **not** build unrelated M:N/ActionDef views, and return the exact confirmed mask and source spans.

**Demand D2:** Request tenant-scoped Worker→Project M:N candidates and effect-sensitive method equivalence. The candidate must preserve duplicate-edge Bag vs Boolean support and **ABSTAIN** when required Rails/SAP source effects or tenant authorization is absent.

**Demand D3:** Change only the source's write-effect fact or a dynamic target, without changing ClassView field positions. Derived constraint cache invalidates; admitted ClassView basis stays valid; previously optimistically matched methods must now refuse/abstain.

**Independent oracles:** (a) simple Rust scalar reference enumerating immutable source facts by stable source ID; (b) optional Crepe/datafrog/Soufflé bounded query comparator with pinned version, license/semantic gates, and independent semantics; (c) Codegraph eager baseline over same fact corpus. A comparator using candidate's own `Cube::partners` is **not independent**.

**Deliberate disables:** force eager full-graph materialization, silently mark `MAY` confirmed, equate two local symbol positions across source revisions, remove tenant guard, drop one ordered effect, and let a stale cache survive a source mutation. Each must fail a result, provenance or representation assertion. Measure all relevant costs and explicitly report when lazy demand **loses**.

**Acceptance:** clear separation between durable facts, demand plan and temporary Codegraph view, exact graded semantics, at least one legitimate RelAlg-native read and one correctly refused effectful/M:N request. **Do not promote source or method authority or promise production latency from a small fixture.**

## 7. Why this beats an eager universal schema DTO

Codegraph never has to *become* Rails, Odoo, SAP, MeTTa, MORK, Soufflé or spaCy. It asks the smallest witnessed question about each source; ClassView/FieldViewDef turns evidence into candidate meaning; native relational algebra executes admitted computation. This is **a reusable demand-driven semantic boundary** instead of a second object world.

## Sources and owners

- [Codegraph graph source](../../crates/lance-graph-codegraph/src/graph.rs), [graded query source](../../crates/lance-graph-codegraph/src/query.rs), [binary lanes](../../crates/lance-graph-codegraph/src/artifact_bin.rs).
- [Crepe/Soufflé/XLOG archaeology](../board/entries/2026-11-02-coresearch-xlog-souffle-crepe-dr3-archaeology.md), [RelAlg IR](../../crates/lance-graph-relalg/src/ir.rs), [RelAlg compiler/executor](../../crates/lance-graph-relalg/src/lib.rs).
- [ClassView](../../crates/lance-graph-contract/src/class_view.rs), [OGIT LinkSpec](../../crates/lance-graph-contract/src/property.rs), [OGAR ActionDef](https://github.com/AdaWorldAPI/OGAR/blob/main/docs/OGAR-AST-CONTRACT.md), [DT64](../../crates/lance-graph-dt64/src/lib.rs), [R2IL native binding](../../crates/lance-graph-r2il/src/binding_native.rs).
- Existing decisions: [EXO-84](https://linear.app/exo-red/issue/EXO-84), [EXO-85](https://linear.app/exo-red/issue/EXO-85), [EXO-79](https://linear.app/exo-red/issue/EXO-79), [EXO-143](https://linear.app/exo-red/issue/EXO-143), [EXO-73](https://linear.app/exo-red/issue/EXO-73), [EXO-108](https://linear.app/exo-red/issue/EXO-108), [EXO-121](https://linear.app/exo-red/issue/EXO-121), [EXO-152](https://linear.app/exo-red/issue/EXO-152), [EXO-153](https://linear.app/exo-red/issue/EXO-153), [EXO-154](https://linear.app/exo-red/issue/EXO-154), [EXO-155](https://linear.app/exo-red/issue/EXO-155).
