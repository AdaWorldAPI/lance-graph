# D-CGMM-0 · Codegraph × MeTTa logical schema scratch × DT64 MethodBinding

> **2026-11-04 · Architecture / research pre-registration · DOCS ONLY.** This does not implement a parser, new codegraph store, class registry, compiler, generic rewrite engine, NNUE scorer, new F### word, production authorization, or execute an unguarded/guarded DT64 method. [EXO-155](https://linear.app/exo-red/issue/EXO-155) tracks the program; [EXO-50](https://linear.app/exo-red/issue/EXO-50) is append-only coordination. [Staged PRs](d-cgmm-staged-pr-charter-v1.md), [test matrix](d-cgmm-falsifier-matrix-v1.md), [board](../board/entries/2026-11-04-d-cgmm-0-metta-class-method-builder.md).
>
> **Core idea:** MeTTa becomes the **logical DTO and class-method builder**, while **schema-constraint scratch** is the disposable staging area where source evidence from Codegraph is matched against MeTTa-style declarations, normalized and evaluated for *eligibility*. Only independently admitted declarations become immutable DT64 `MethodBinding`s. Admitted execution uses existing Quack/R2IL/mask-risc/ndarray; **no MeTTa evaluator**, duplicate code store, object-reconstructing hot path or unearned SIMD/Abakus speedup.
>
> The operator also wants a domain comparison between **Rails, Odoo, SAP**, not false cross-platform equivalence: same concept may have different callback timing, transaction semantics, authorization, field layouts and effects. Codegraph must represent those *differences* first.

## 1. One pipeline, two directions, four authority levels

~~~text
INPUTS (source evidence; not method authority)
    Rust/Charon + Ruff subset      Odoo/Python      Rails/Ruby?    SAP/ABAP?
             |                          |                |             |
     compiler-grounded artifact     bounded harvest   ADAPTER GAP   ADAPTER GAP
      (symbols, calls, spans,       with provenance,   until real    until real
       maybe-dispatch, residuals)   not guessed types   evidence      evidence
             +-------------+------------+----------------+-----------+
                           |
                  CODEGRAPH FACT PLANES
           (confirmed / possible / unknown / residual)
                   source + schema generations
                           |
                  [LOGICAL SCHEMA SCRATCH]
           bounded temporary constraints / role candidates
             identity, unification, compatibility, refusal
             (MeTTa-shaped atoms, scoped typed relations)
                           |
       candidate class/method declaration + evidence/refusal
                           |
                 DT64 / OGAR OWNER ADMISSION
              ClassId / MethodId / SigRev / ClassView
              ON × WHEN × ordered DO / ValueDemand
                 codebook + provenance + permit
                           |
          COLD method preparation / bind / typed refusal
          Quack/RelAlg OR optional R2IL native reading
                           |
               mask-risc -> ndarray/Abakus if useful
                  borrowed canonical SoA/DR lens
                           |
                 witnessed result or refusal
~~~

**Backwards direction (audit):** For an already admitted DT64 `MethodRef`, Codegraph may locate and grade its *supporting source evidence*, map the declared behavior to source call/effect sites and identify stale or absent evidence. It **must not** synthesize compiler facts from the admitted method, infer that call absence proves no effect, or rewrite a source program because the schemas look similar.

**Four different objects** must never be conflated:
- **Codegraph fact:** compiler or harvester observation with source/source-revision/span/grade/coverage; neither domain truth nor an execution permit.
- **Logical schema scratch candidate:** per-prepare, bounded, provisional constraint assignment; can be evicted with *no* change to admitted truth.
- **DT64 binding:** immutable, owner-validated class/schema/method identity and ON/WHEN/DO method contract.
- **Prepared native execution:** validated method/lens/opcode/route and executable operations under live source/frame/scope/permission, optionally reusable after all relevant witnesses are checked.

## 2. Source-grounded compatibility and known missing data

| Existing component | Verified/source-reported role | What must not be inferred |
| --- | --- | --- |
| `lance-graph-codegraph/src/artifact_bin.rs` | Versioned, validated `.rows.bin`; one owned bytes buffer; borrowed binary lanes including FUNDECL/PROV | Not all language AST, types, effects or borrows have been harvested |
| `lance-graph-codegraph/src/graph.rs` | Symbol interning to positions, DR `MapTable`, XLOG cube relations `CALLS/NODE/MAY/IMPL/RCONF/RPOSS` | Codegraph `Pos` is not ClassId, MethodId, OGAR codebook id or permanent DR global id |
| `lance-graph-codegraph/src/query.rs` | `confirmed/possible/unknown` masks; no tuple decode in hot queries | `possible` is not confirmed; `unknown` is not a negative proof |
| `lance-graph-codegraph/src/diag.rs` | PARTIAL, NOT_RUN, residual coverage; typed diagnosis | "No known gap" is not proof of complete harvest |
| Ruff/Charon source harvest | Compiler LLBC facts and limited Python intake; reports source effects and dispatch in some paths | Effects/type/borrow/control flow may be dropped before Codegraph; cannot certify from prose labels |
| [#1944 cognitive polyfill](https://github.com/AdaWorldAPI/lance-graph/pull/1944) | Ground, bounded conjunctive common subset, Set vs Bag and NAL differ; 12/12 lab tests | General MeTTa rewriting, unification, recursion, Atomese or symbol bind are **not implemented** |
| [MeTTa assimilation](metta-datalog-assimilation-v1.md) and [council v3](metta-datalog-assimilation-v1-council-v3.md) | MeTTa AST as declaration, symbols bound once, no value gather, Quack native semantics | Council forbids value-derived cross-lane gather; old proposals superseded |
| `lance-graph-dt64/src/lib.rs` | `MethodBinding {id,sig_rev,name,on,when,reads,effects,population,truth,demand}`; ordered effects and immutable per-class schema | Registration does not execute or authorize behavior |
| `lance-graph-r2il/src/binding.rs` | `prepare` binds eligible effects, `run` checks source/frame/scope and explicit `ExecPermit` | It explicitly refuses `GuardNotExecutable` and `ThreeValuedNotExecutable` |
| `lance-graph-r2il/src/binding_native.rs` | `prepare_native` compiles eligible fold programs; `run_native` rechecks and runs borrowed planes without allocating | Data-dependent fold chaining, changing shape or unadmitted method are not general-purpose execution |
| [#1955](https://github.com/AdaWorldAPI/lance-graph/pull/1955) | Lab address reuse and canonical in-memory SoA borrow | No source-backed production lease or remote/disk zero-copy |
| [#1956](https://github.com/AdaWorldAPI/lance-graph/pull/1956) | Abakus k-of-n exact but carry words 1.5–4.8× slower on tested shapes | No reason to choose carry words as default |
| [EXO-152](https://linear.app/exo-red/issue/EXO-152) | Native algebra/external DTO normalized-method capstone | No new normalized method registry may be unilaterally minted |
| [EXO-95](https://linear.app/exo-red/issue/EXO-95) | Existing SAP/Odoo cross-glove and business parity evidence | Rails semantics or SAP executable trace have **not** been established |

The #1944 W-1 cold symbol→DR global→local binder and W-2 `Guards::check`/`FrameStore` owner gap remain open; those are not "solved" by a new scratch object. Existing `ClassView` and `AlgebraRef` positions are authoritative: string method names are labels, not IDs.

## 3. The new central object: *logical schema-constraint scratch*

**Proposed name:** `SchemaConstraintScratch`, purely conceptual, **NOT** an existing Rust type or API. A short-lived, bounded, typed **working set of hypotheses about class and method identities**. This should be built from Codegraph's already interned identities, compiler facts, external schema declarations and a versioned ClassView/OGAR vocabulary, ideally as bit masks over existing Codegraph/DR address coordinates.

**It is not:**
- mask-risc `Scratch` (runtime register slots, reset by a program/epoch),
- MeTTa/Hyperon's `GroundingSpace` (another persistent trie/interpreter),
- MORK `PathMap` (external Set oracle/index, may materialize copies),
- an owned value dataset or an alternate ClassId/MethodId codebook,
- a cache of source ASTs, business objects or executable function pointers,
- an automatically inferred equivalence relation from names/NNUE similarity.

### 3.1 Minimal conceptual constraint basis

| Constraint atom | Cold scratch meaning | Can it be admitted from source alone? |
| --- | --- | --- |
| `HasField(ClassCandidate, FieldRole, Type, LayoutEvidence)` | Codegraph-observed or externally declared role/field candidate | Only with source + versioned field basis/owner decision |
| `HasMethod(ClassCandidate, MethodCandidate, Arity, Signature)` | Source decl and signature candidate, separate from MethodId | Only with unambiguous identity/shape |
| `Reads(Method, Field)` / `Writes(Method, Field)` | observed read/write effect with provenance and coverage grade | Missing effect evidence => UNKNOWN, not pure |
| `OnChange(Method, Field)` | callback or trigger from source framework | Framework/source-specific, cannot guess from call edge |
| `When(Method, Expr, Logic)` | guard candidate over typed field roles | DT64 can **store**, R2IL cannot generally **execute** |
| `Do(Method, OrderedEffects)` | sequence-preserving effect description | Executable only for supported native words/arity/effects |
| `EquivalentUnder(A,B,Semantics,Witness)` | proposed equivalence scoped by explicit assumptions | Never on names or NNUE alone; must survive oracle |
| `Incompatible(A,B,Reason)` | negative evidence (effect, type, auth, ordering etc) | Retained even if candidate ranking is high |
| `UnknownCause(Candidate,Gap)` | absent language harvest, unresolved dispatch, missing definition/effect | No promotion while decisive gap remains |

Scratch construction may be memoized **only as disposable pre-admission optimization** keyed by immutable artifact generation, source snapshot, producer version, ClassView/schema/codebook revision, scope and semantic rule version. Cached inference is **never** execution authority. Eviction changes latency, not truth.

### 3.2 Constraint processing: source → scratch → admission

~~~text
FACTS:   F = (codegraph generation, source artifact digests, coverage)
SCOPE:   U = (candidate domain, ClassView basis, method lens, permissions)
SCRATCH: H = {field mappings, method mappings, guard constraints,
              effect order, allowed algebra, negatives, abstentions}
RULES:   H' = NormalizeAndConstrain(F, U, H)  # COL D / PROPOSED
CHECK:   admissibility(H') = ADMIT | REFUSE(kind) | ABSTAIN(cause)
OUTPUT:  at most an immutable method OFFER and source witness, no execution
~~~

An **ABSTAIN** preserves uncertainty when a source producer is incomplete, a dispatch target is dynamic, an AST variant is unsupported or an NNUE proposal lacks proof. **REFUSE** means an independently verified incompatibility or policy violation. Neither is `COUNT=0` nor "method absent". Unresolved sources must remain visible in Codegraph's graded masks and residuals.

Scope and lifetime are two layers: Codegraph scratch is **cold pre-admission**; mask-risc scratch is **hot transient evaluation**. Do not let scratch identities survive a ClassView rev, source revision, permission revocation or frame replacement. At the admission Rubicon, only *immutable DT64-certified binding metadata* may persist. Any cached compiled program has a separate source/layout/witness lifetime, and is not a durable schema fact.

### 3.3 Example: same concept ≠ same method behavior

This is **illustrative pseudo-declaration**, *not MeTTa grammar accepted by any existing parser*:

~~~text
Concept: BillableWorkEntry
Candidate adapters: Odoo.WorkLog, SAP.CATS, Rails.TimeEntry
Logical fields: actor, project, interval, approval, billable
Method intent: ApproveEntry(entry)

Cross-source constraints:
  same_concept ≠ same_transaction_policy
  Odoo: observed ORM dependency/callback evidence (versioned, if harvested)
  SAP: observed BAPI/CATS semantics (requires actual source/trace)
  Rails: ActiveRecord callback/transaction semantics (requires Ruby harvest)
  match only with explicit field types, effect order, authorization
  absent Rails/SAP AST/effect trace => ABSTAIN, never assume parity

Proposed DT64 offer: ON(field positions), WHEN(approved/preconditions),
  READS, DO(ordered allowed effects), Truth, ValueDemand, SigRev
Native executable result: only if admitted semantics exactly match allowed R2IL
~~~

The target is not "convert SAP into Odoo." It is **one normalized concept with distinct application-local renderings and lawful method bindings**, able to share a native fold *only when the algebraic behavior is proven identical*.

## 4. MeTTa-shaped declaration semantics, MORK and datafrog boundaries

**Use MeTTa as a declarative syntax / pattern vocabulary.** The admissible subset is finite, typed, bounded facts, constraints, structural joins and verified normalizations. No unconstrained `superpose` / `collapse` execution, arbitrary grounded callbacks, impure rewrite, recursive fixpoint, unification over unbounded nested terms, or term construction in the hot path.

**Rewrite distinction:** A candidate rewrite is not a proven equivalence. Codegraph `CALLS` is a fact; MeTTa-like inference may derive *candidate patterns*, but `A=B` becomes an allowable method substitution only after algebra/type/effect/permission/source witnesses satisfy a separate law. An egglog congruence is not a Codegraph edge; a learned NNUE score is not an algebraic law.

**MORK:** an external set-oriented structural oracle; its `PathMap` is not a second Lance identity authority. [EXO-79](https://linear.app/exo-red/issue/EXO-79) records license/usage gate. **MeTTa `BindingsSet` is a multiset**; MORK space is Set. Do not silently drop duplicates, and don't ask a Boolean mask to answer multiplicity. [EXO-143](https://linear.app/exo-red/issue/EXO-143) four-binding case and [EXO-108](https://linear.app/exo-red/issue/EXO-108) bag terminal remain gates.

**datafrog:** independent oracle for applicable Datalog/join/fixpoint semantics. No automatic recursion lowering into mask-risc; [EXO-73](https://linear.app/exo-red/issue/EXO-73) owns a lawful recursive driver, and [EXO-101](https://linear.app/exo-red/issue/EXO-101) explicitly rules out prior gather-based semijoin results as a production basis.

**NNUE:** an optional, *offline* approximate candidate prioritizer over codegraph-derived features (types, roles, call/effect provenance, structural positions). It may rank which cross-app method candidates the exact constraint solver tests first. It **cannot create facts, approve an absent source effect, override a refusal, grant execution, populate WORLD, or cause a learned representation to be treated as canonical DR addresses**. Report calibration, novelty, abstention, negative controls. A perfect NNUE score is not authority.

## 5. DT64 binding contract, exactly where the execution seam stops

A successful scratch result describes an OFFER to the owner of `ClassId`/`ClassView`/`BindStore`. On admission:

- Class identity is issued by its existing codebook owner; application-local class identity and shared concept half are distinct.
- Resolve symbol labels to **DR global identities once, cold**, then source-specific local field positions. Never equate Codegraph `Pos`, Charon `FunDecl` or a symbolic atom with `MethodId` or a physical row address.
- Bind `ClassSchema {schema_rev,field_count,extent}` and `MethodBinding` under immutable signature revisions. ON and READS are masks; WHEN stores a guarded expression and truth mode; DO retains **effect order**. Do not reduce method semantics to a bag of calls.
- Reject unproven effect order, unknown field type, changed source schema, unclear permissions, missing guard semantics or unauthorized effect. **Prepared plan caching** must use its full binding and source/layout/scope witness, not the fact that a method name happens to match.
- The real `r2il::binding::prepare` explicitly refuses methods whose WHEN is present and three-valued truth. `binding_native::prepare_native` inherits eligibility conditions and refuses fold-result-dependent future folds, data-derived scalar after fold, shape changes and excess outputs. **These are honest stop conditions.**
- `ExecPermit` is lab-level only, not an authorization issuance mechanism. The source owning executor and the named operator must approve production admission. DT64's own `admit` does not grant execution.

**Important inversion:** Do not try to "make Codegraph execute." Codegraph learns source structure; DT64 binds meaning; existing native executor executes eligible semantics. The connection is **source evidence → immutable admissible method**, not Codegraph → arbitrary AST interpreter.

## 6. Memoization through the existing ndarray / Abakus fabric

There are **three separate memoization levels**, with different validity:

| Level | May be cached cold? | Keys and invalidations | Runtime meaning |
| --- | --- | --- | --- |
| Candidate constraint scratch result | optionally; *disposable* | artifact digest/generation, harvester version, provenance gaps, class/method schema, rule revision, scope | Proposal only; cache hit never grants permission |
| Admitted method recipe / native prepared plan | if owner permits | `MethodRef` + `SigRev` + `GuardWitness` + `ClassView` basis + OGAR vocabulary rev + IR/semantics version + source/layout/permission guards | Reuse compiled algebra with hot rechecks; no copied values |
| Local DR route / optional source locator receipt | D-DRISA / D-DRIVER-ZC gated | frame/map issue, source incarnation, layout/row-index seal, field scope, source lease | Data-local address reuse only, no business-object cache |

**Abakus:** a potential bit-sliced arithmetic *execution consumer* for k-of-n or later significance-plane accumulation. #1956's measured carry path loses on eight tested shapes. The memoization gain must be independently measured as **compile/bind/re-route amortization or reusable significance-plane calculations**. Never promise that Abakus changes semantic-schema parsing cost or outperforms native arithmetic without an experiment. Use existing `ndarray::simd` API; no new ISA, cache backend, F### or CPU emulator.

**Cold/hot split:** parse/harvest/constraint solve/DT64 registration/prepare all **cold**; hot path performs witness checks, scoped borrow, existing native program and result. Profiling must not hide cold preparation in a benchmark claiming zero allocation.

## 7. What this can and cannot establish

The *smallest meaningful demonstration* is not SAP ingestion or trained NNUE.

1. Harvest one **compiler-grounded** source method and one independently defined synthetic application profile with differing method effects.
2. Derive a bounded set of candidate field/method constraints from Codegraph into ephemeral masks, including a **negative incompatible candidate**.
3. Produce one lawful, guard-free two-valued DT64 `MethodBinding` proposal and one refused guarded/dynamic/effectful proposal.
4. Owner-approved lab admission binds the valid proposal via the existing R2IL native seam, runs an independent scalar oracle, then executes 256 times without reconstructing values or reparsing source.
5. Change schema/source/scope/effect witness and prove the prepared binding or candidate scratch is invalidated appropriately. A wrong inferred ClassId or field ordinal must be refused.
6. Count scratch footprint, cold harvest and prepare, allocations, hot path, copied canonical value bytes, and status per evidence class.

This proves a **class-method builder contract** in one constrained case, not general executable MeTTa, a Rails/SAP parser or semantic equivalence of enterprise systems.

## 8. Source and owner crosswalk

**Repository source:**
- [Codegraph](../../crates/lance-graph-codegraph/src/lib.rs), [graph](../../crates/lance-graph-codegraph/src/graph.rs), [query grades](../../crates/lance-graph-codegraph/src/query.rs), [coverage](../../crates/lance-graph-codegraph/src/diag.rs), [binary lanes](../../crates/lance-graph-codegraph/src/artifact_bin.rs).
- [DT64 MethodBinding](../../crates/lance-graph-dt64/src/lib.rs), [R2IL prepare/run](../../crates/lance-graph-r2il/src/binding.rs), [native prepare/run](../../crates/lance-graph-r2il/src/binding_native.rs).
- [MeTTa frontend](../../crates/lance-graph-metta/src/lib.rs), [#1944 cross-frontend probe](../probes/2026-11-03-d-cognitive-polyfill/REPORT.md), [MeTTa plan](metta-polyfill-v1.md), [council correction](metta-datalog-assimilation-v1-council-v3.md), [MORK Schnellverschluss](mork-pathmap-schnellverschluss-v1.md).
- [Semantic grammar recovery](../board/entries/2026-11-02-coresearch-semantic-grammar-recovery.md) particularly B1–B9 AST/effect/provenance harvest losses, source FACT → GUARD → TRANSFORMATION distinction.

**Linear existing decision owners:**
[EXO-84](https://linear.app/exo-red/issue/EXO-84) MeTTa; [EXO-85](https://linear.app/exo-red/issue/EXO-85) assimilation; [EXO-79](https://linear.app/exo-red/issue/EXO-79) MORK; [EXO-143](https://linear.app/exo-red/issue/EXO-143) binding bag; [EXO-101](https://linear.app/exo-red/issue/EXO-101) datafrog; [EXO-73](https://linear.app/exo-red/issue/EXO-73) recursion; [EXO-108](https://linear.app/exo-red/issue/EXO-108) Bag; [EXO-121](https://linear.app/exo-red/issue/EXO-121) ClassView; [EXO-46](https://linear.app/exo-red/issue/EXO-46) cold symbol/DR and focus; [EXO-140](https://linear.app/exo-red/issue/EXO-140) foreign lookup; [EXO-152](https://linear.app/exo-red/issue/EXO-152) normalized method external dialects; [EXO-95](https://linear.app/exo-red/issue/EXO-95) enterprise business parity; [EXO-153](https://linear.app/exo-red/issue/EXO-153) native register; [EXO-154](https://linear.app/exo-red/issue/EXO-154) zero-copy driver.

**Non-goals:** no new MeTTa execution, MORK code vendoring, SAP or Rails harvester invented from text, Codegraph → DT64 automatic promotion, neural authority, production WHEN interpreter, e-graph equality from graph adjacency, new ClassId allocator, second role/tuple store, or new cognitive-shader driver.

## Update (2026-11-04): lazy Codegraph and FieldViewDef / Datalog separation

The operator refined the class-method builder: **facts and identity remain Kythe-shaped source evidence, Crepe/Soufflé/XLOG supply only cold demand/binding/stratification plan evidence, and Codegraph becomes a demand-scoped graded view rather than a second owned AST object tree.** Its existing `Cube` / `Pos` / `OnceCell` paths are partially lazy already, not a complete demand-driven implementation. See the source-backed [D-CGMM-LAZY-0 addendum](d-cgmm-lazy-codegraph-fact-demand-v1.md) and the [staged plan](d-cgmm-staged-pr-charter-v1.md).

**FieldViewDef** is a **proposed, immutable semantic offer on the existing ClassView bit basis** (including `lens_fields(full_classid)` when behavior is requested). A disposable MeTTa-shaped constraint scratch can produce typed OGIT `LinkSpec` / OGAR `ActionDef` / DT64 `MethodBinding` offers, subject to independent owner admission. **Pure, same-population** views may polyfill through existing RelAlg, but arbitrary M:N joins, guarded WHEN, effectful action bodies, recursive rules and unknown source effects remain typed refusals/ABSTAIN. `FieldMask` bits continue to mean instance presence only; no new ClassView method, codebook or hot-path MeTTa evaluator is authorized by this plan.
