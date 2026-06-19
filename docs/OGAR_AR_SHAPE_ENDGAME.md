# OGAR AR-Shape Endgame — the ontology compiler, not an Odoo transcode

> **Audience:** every future session that touches Odoo / OpenProject / WoA /
> SMB-Office / SAP transcode, or proposes a new ERP curator.
> **Status:** doctrine (operator-ratified 2026-06-19; closes the
> 5+3 council on `E-AR-PROJECTION-CORRECTION-1`).
>
> **The headline:**
>
> > **OGAR IS the AR-shaped THINK/DO compiler.**
> > **Foundry / Gotham / OpenProject / Odoo are just schema + class
> > inheritance FED INTO it — not external systems OGAR adapts to.**
>
> Every "platform" worth competing with (Palantir Foundry, Gotham, Odoo as
> an ERP-platform, Rails AR / OpenProject as an app-platform, future SAP)
> reduces, at the architecture level, to "an ontology of classes with
> inheritance + actions + policies." OGAR is that ontology compiler. The
> ERPs become INPUTS — schema dumps fed to the compiler — not foundations
> the compiler depends on.
>
> **One-line read:** flat triples are corpse-scan; OGAR classes are living
> anatomy; ERP curators teach the system what business software keeps
> reinventing; the deliverable is an **executable ontology compiler**, not
> a port.
>
> **The spine (memorize this; everything else fits under it):**
>
> > **Curators teach. OGAR compiles. LanceGraph thinks.
> > SurrealAST + Kanban orchestrate. Adapters obey.**
>
> **The litmus failure (name it when you see it):**
>
> > The same `OgarAst::Do(PostInvoice, …)` MUST execute semantically
> > identically across NativeLance / SurrealAST / Odoo adapter / Rails
> > adapter. **If one backend leaks its syntax into the semantic result,
> > that curator has started wearing the crown — that is the bug.**
>
> _Tiny brass thunderbolt: Foundry maps the enterprise. OGAR metabolizes it. ⚙️🫀_

---

## 0. The full ladder

```
Rails/OpenProject AR   +   Odoo ORM   +   WoA ERP   +   SAP (later)
                              │ teach
                              ▼
              OGAR inherited classes + ontologies
                              │ fill
                              ▼
                    agnostic AR-shape
                              │ compiles into
                              ▼
                 DO / THING / THINK AST
                              │ routes through
                              ▼
              Rubicon / Heckhausen Kanban
                              │ executes via
                              ▼
   ractor-proven LanceGraph SoA  +  SurrealDB / Lance KVS  +  callcenter membrane
```

The earlier "flat Odoo triples → generated DDL" path is **rejected as the
endgame** — kept as a negative calibration artifact (see
`E-AR-PROJECTION-CORRECTION-1`). It preserves syntax, not meaning. AR-shape
is **not flat triples**. AR-shape is the **living adapter grammar** learned
from Rails/OpenProject + Odoo, purified through OGAR/DOLCE until it is
backend-agnostic.

---

## 1. Why flat triples are rejected as the deliverable

Flat triples are the **harvest substrate**, not the ontology. They are
correct as the polyglot carrier (`lance_graph_contract::codegen_spine::Triple`,
the cross-language wire shape Rails / Odoo / C++ extractors all emit into).
They are wrong as the destination.

| Flat-triple endgame | OGAR-class endgame |
|---|---|
| `odoo:model → table` | `Invoice <: LegalDocument <: EconomicCommitment <: SocialObject` |
| `odoo:field → column` | `posting_date :: GoBD-immutable-after-Posted` |
| `validates_constraint` lifted to `ASSERT` | `PostInvoice DO` gated by `GoBD-compliant THINK policy` |
| One SurrealQL DDL emission per ERP | One executable ontology consumed by any backend |
| Curator's syntax preserved | Curator's INVARIANT promoted |

The flat path reaches a SurrealQL schema that **looks like a translated Odoo
schema**. The OGAR path reaches an ontology that **outlives the curator** —
when the curator is gone, the meaning stays, because the meaning was never
the curator's.

This is the difference between **transcoding** and **metabolizing**.

---

## 2. Rails / OpenProject / Odoo / WoA-rs / SAP as curators, not foundations — and Foundry/Gotham reduces to the same shape

These ERPs are **fossils in the cliff-face** — what we learn from, not what
we depend on. Every "Foundry-class platform" reduces to the same shape:
**class inheritance + actions + policies over enterprise objects**. Palantir
Foundry's ontology layer, Gotham's case ontology, OpenProject's AR models,
Odoo's ORM — they all converge on this. OGAR is the compiler that takes
the schema-shape of any of them as INPUT and emits an executable ontology
as output.

They teach the canonical primitives that every business software keeps
reinventing:

```
class            field            association
scope            validation       callback
constraint       state machine    posting
audit chain      tenant           document persona
workflow         permission       transaction boundary
```

The job is to **observe these across curators**, factor out the curator's
syntactic accidents, and promote the surviving invariant into OGAR.

### Per-curator role

- **Rails / OpenProject** (`AdaWorldAPI/openproject-nexgen-rs`) — already
  proved the polyglot extractor pattern: AST → `Triple` carrier → typed
  projection. 27+ predicates surveyed. **Role:** Rails-shaped AR projection
  prototype, NOT the ontology home. (See `E-AR-PROJECTION-CONVERGED` +
  `E-AR-PROJECTION-CORRECTION-1`.)
- **Odoo** (`tools/odoo-blueprint-extractor` + `crates/lance-graph-ontology/src/odoo_blueprint/`)
  — mature ERP with regulatory anchors (GoBD, restrictive audit trail),
  composition (`_inherit`), value domains (`fields.Selection`), MRO
  derivation. **Role:** dense regulatory + behavioural teaching corpus,
  NOT the source of truth for class identity.
- **WoA-rs** (`/home/user/woa-rs`) — sea-orm/MySQL/axum German ERP. Has
  GoBD audit_chain, sammelrechnung, mahnstufe state machine, 7-doc-type
  `Vorgang`. **Role:** German ERP sanity witness — *does our promoted
  ontology recognize what WoA already has?*  NOT the first SurrealQL
  consumer (it doesn't use SurrealDB).
- **SMB-Office** (`/home/user/smb-office-rs`) — C# WinForms → Rust port,
  MongoDB with German field names. **Role:** parity-validation curator
  for German legacy ERP behaviour (`db_*.cs` schemas, `Crypt.cs` quirks).
- **SAP** (future) — when the SAP frontend arrives, it is the third
  independent ERP confirmation that the curated OGAR ontology is real.

The doctrine: **the curator teaches by example. The ontology survives by
being more abstract than any one curator.** A primitive is only Core-grade
when ≥2 independent curators surface it under different syntactic forms.

### Correction (2026-06-19, operator) — the curator distinction is one regex

The per-curator role list above (Rails / Odoo / WoA / SMB / SAP each with a
distinct "Role:" sentence) read this doc as if each curator demanded a
distinct architecture — that overstates the mechanical reality. **The
curator distinction at the harvest→AST seam is one tiny regex.**
OpenProject is a project-management domain; Odoo is an ERP domain. At the
extractor surface they emit the SAME AR-shape predicate vocabulary
(`rdf:type` / `has_attribute` / `declares_association` / …) on the SAME
`codegen_spine::Triple` carrier; they differ only in their namespace
prefix (`openproject:` vs `odoo:`). `from_triples::strip_namespace` needs
to recognise both (and any future ERP prefix); that's the entire
mechanical difference. Domain identity rides the namespace; the compiler
treats curators uniformly.

The per-curator "Role:" sentences ABOVE still read accurately as
*what each curator teaches the ontology* (Odoo's regulatory anchors, WoA's
GoBD audit chain, OpenProject's project-management primitives, SMB-Office's
legacy German ERP behaviour). They do NOT mean the compiler architecture
varies by curator — the variation is the namespace, full stop.

### Correction follow-on (2026-06-19, operator) — once domains are namespaced, the work is synergy wiring

The regex correction above closes one question and opens the next: **what
DOES the compiler do, once the namespaces are distinguished?** Answer:
**wire synergies between namespace-tagged inputs and the OGAR Core**.
Synergies are bidirectional:

- **Input synergy (curator promotion):** multiple namespace-prefixed
  identities resolve INTO one OGAR class identity. The ≥2-curator
  promotion rule is the resolver:
  `{ odoo:account_move, openproject:invoice, woa:vorgang(doc_type=invoice),
  sap:bkpf+bseg } → ogar:Invoice <: LegalDocument`.
  Same shape for primitives that aren't classes but predicates / kinds:
  `{ odoo:fields.Selection('state'), woa:WoStatusAction(enum),
  openproject:status_id, rails:acts_as_state_machine } → ogar:StateMachine`.
- **Output synergy (adapter dispatch):** one OGAR class projects OUT to
  multiple namespace-prefixed targets via the §3 `adapter_targets` slot +
  the ARM `Executor::Adapter(&'static str)` discriminator (post §11
  remediations). `ogar:Invoice → { odoo:account_move,
  rails:invoice_ar, woa:vorgang, sap:fi_document }`.

**The synergy registry is the work.** Inc 4's "curator promotion table"
is exactly this registry mechanised — the §11.1 framing should read it
that way (synergy table, not just a promotion log). Each row of the
registry is a synergy: `(ogar_class, [namespace-tagged inputs],
[namespace-tagged outputs])`. Inc 4's F4 gate (≥4 primitives surface
under ≥2 curators) is the falsifier that the registry actually has
SYNERGIES, not just single-curator entries dressed up as "promoted."

This sharpens §3 (the class shape `adapter_targets` is the output-synergy
slot — already named, just unlabelled as synergy), §11.1 Inc 4 (the
promotion table IS the synergy table), and §10's Invoice example (the
four `adapter_targets` listed there ARE the output synergies of
`ogar:Invoice`).

### Correction punchline (2026-06-19, operator) — _"tadaa"_: WoA-rs consumes ERP through the synergy registry, not through SurrealQL

The synergy framing above closes the loop with
`E-AR-PROJECTION-CORRECTION-1` (the 5+3 council that retracted the
"WoA-rs as first SurrealQL consumer" claim because WoA is sea-orm /
MySQL / axum, not SurrealDB). The retraction stands for **SurrealQL
specifically** — WoA never consumed the SurrealQL DDL adapter target,
and never will under its locked stack. What WoA-rs DOES consume — and
what makes it a first downstream consumer of the OGAR Core IN A WAY
THAT IS NOW MECHANICALLY CORRECT — is the **synergy registry**.

The flow (with WoA on its native stack, no SurrealQL anywhere):

```
   curators                synergy registry           consumer
   ─────────                ─────────────────         ────────
   odoo:account_move ─┐                              ┌─ woa:vorgang
   openproject:invoice ─┼─►  ogar:Invoice           ─┼─►  sea-orm Entity (MySQL)
   sap:bkpf+bseg ─────┘    (cross-curator             │
   (read into OGAR via     promoted; carries          └─ codegen against
   namespace prefix +       regulatory anchor +          OGAR class shape
   ≥2-curator rule)         state machine +              instead of
                            audit chain +                hand-rolling
                            adapter_targets)             per-curator)
```

WoA-rs's first cross-curator value isn't "render SurrealQL" — it's
**inherit the cross-curator definition of `Invoice` (Odoo's regulatory
GoBD anchors + OpenProject's PM linkage + future SAP's FI mapping) and
project it onto its own sea-orm/MySQL adapter target**. The SurrealQL
adapter target stays one of N output synergies (per `E-AR-PROJECTION-
CORRECTION-1` Phase 1/2 placement); sea-orm is another; both project
from the same `ogar:Invoice` class shape.

This is the _tadaa_: once domains are namespaced (regex) and synergies
are wired (registry), every consumer — WoA-rs (sea-orm/MySQL),
smb-office-rs (MongoDB), future SAP NetWeaver, future SurrealQL — projects
from ONE OGAR class shape via its own `Executor::Adapter` target. No
consumer needs the SurrealQL adapter to benefit; SurrealQL is just one
of the projection lanes. WoA-rs becomes a first consumer of the
**synergy registry**, not the SurrealQL DDL — and that distinction is
what makes the "first downstream consumer" framing finally honest.

---

## 3. OGAR inherited class model

The Core (`crates/lance-graph-ontology`) holds the ontology as **inherited
classes**, not flat tables. Inheritance carries:

```
Invoice <: LegalDocument <: EconomicCommitment <: SocialObject
PostingAction <: IrreversibleTransition <: GoBDRegulatedAction
DunningStage <: EscalatingObligationState <: TenantScopedState
TenantScope <: InstitutionalBoundary
```

Each class has:

1. **Identity** — `classid` (the canonical address; see `E-GUID-IS-THE-GRAPH`)
2. **State schema** — value tenants on SoA columns (the **THING** arm)
3. **Action set** — `ActionDef`s with gates (the **DO** arm)
4. **Policy set** — `ThinkSpec`s (the **THINK** arm, new — promotes
   `validation_kind` + `@api.constrains` + GoBD checks + RBAC into a typed
   first-class slot beside actions)
5. **Regulatory anchor** — links the class to a regulation IRI (GoBD §146,
   GDPR Art. 17, GAAP ASC 606, …) so audit + legal review is structural.
6. **Adapter targets** — known concrete backends (Odoo `account.move`,
   Rails `Invoice`, WoA `Vorgang(doc_type=invoice)`, SAP FI document).
7. **Inheritance** — `classid → ClassView` resolves the parent chain;
   child shadows parent by predicate (see `mro::resolve_overrides`).

Promotion rule (the curator-to-Core gate):

> A primitive is promoted into OGAR Core ONLY when ≥2 independent curators
> surface it under different syntactic forms. The first curator names it;
> the second confirms it; the third makes it canonical.

This is why WoA's `audit_chain.rs::chain_hash` is interesting **not** because
it equals Odoo's `_post` chain (it doesn't — `E-AR-PROJECTION-CORRECTION-1`
retracts that claim), but because both Odoo and WoA independently surfaced
the same GoBD I9 hash-chain INVARIANT. The promoted primitive is
`AuditChain <: GoBD-immutability`, not "the chain_hash function."

---

## 4. DOLCE THING / DO / THINK split

The three-way split is the user's refinement of the prior two-arm DOLCE
mapping (`E-AR-DO-WIRING`'s Endurant/Perdurant). The third arm —
**THINK** — is promoted from "validation lifted to `ASSERT`" into a typed
policy slot.

### THING — what exists (Endurant)

The state schema. Persisted on SoA value tenants. Read by every consumer.

```
Customer        Invoice         WorkOrder       Project
Obligation      Payment         Tenant          Role
Resource        AuditEvent      Sequence        JournalEntry
```

Backend home: `MailboxSoA<N>` columns + `ValueSchema::Cognitive` /
`Compressed` preset (per `classid` read-mode).

### DO — what mutates / acts / commits (Perdurant)

Discrete commits through the gate. Each is an `ActionInvocation` that has
been **adjudicated** by `commit(def, actor, impact, guard, now) → ActionState`
(`def-match → RBAC → state-guard → MUL`).

```
post invoice            approve work package      reserve stock
assign worker           send dunning notice       reverse journal entry
close project phase     escalate dunning stage    finalize sammelrechnung
```

Backend home: `ActionDef` registry + `ActionInvocation` envelopes + the
mailbox-cycle commit gate.

### THINK — what plans / chooses / evaluates (NEW slot)

Continuous policies, never committing themselves but gating every DO. Each
is a typed query against the current world.

```
should this be posted?                 is the tenant boundary satisfied?
is this action allowed under GoBD?     what is the next Kanban state?
is the actor authorized?               what adapter can execute this?
what risk/benefit/impact does Rubicon see?
```

Backend home: a new typed slot on `ClassView` (CONJECTURE; council-gated
before code) — `ClassView::policies: &[ThinkSpec]` alongside the existing
`ClassView::compute_dag` and `ClassActions`. THINK reads the SoA state
(THING) + the proposed action (DO) + the actor context, and returns a
typed `Verdict { ProceedAs(plan) | RouteToHuman | Reject(reason) | Defer(condition) }`.
THINK never mutates; DO mutates iff THINK yields `ProceedAs`.

### The triad invariant

> **THING is read. DO writes (gated). THINK never writes.**
> A DO without a THINK passes the gate by default-allow.
> A THINK without a DO is a query, not a decision.
> A THING without either is dead inventory.

---

## 5. AST operation grammar

The portable semantic operation tree — what the curator's syntax compiles
INTO, what every backend (Lance / SurrealQL / Odoo adapter / Rails adapter)
compiles FROM.

```rust
enum OgarAst {
    // THING — addressable state
    Thing(ClassRef),                    // "an Invoice"
    Slot(ClassRef, FieldRef),            // "an Invoice's posting_date"
    Relation(ClassRef, RelRef, ClassRef),// "Invoice has_many JournalLines"

    // DO — proposed mutation
    Do(ActionRef, BindingSet),           // "PostInvoice(this=I42)"
    Transition(ClassRef, StateRef, StateRef), // "Invoice: Draft → Posted"

    // THINK — typed query
    Think(PolicyRef, BindingSet),        // "GoBdImmutabilityCheck(I42)"
    Verdict(VerdictRef),                 // the result of a THINK

    // glue
    Constraint(ConstraintRef),           // structural invariant
    AdapterCall(AdapterTarget, BindingSet), // route this op out
}
```

The AST is **the inter-lingua**. It is what makes the system polyglot at
the OPERATION level (not just at the harvest level). Today's harvest is
already polyglot via `codegen_spine::Triple`; this AST extends the same
contract upward — from "what classes/fields/methods exist" to "what
operations they support."

Falsifiable scope (CONJECTURE until probe): the AST is intended to be the
**canonical form** that:
- Cypher patterns lower to (per `E-CYPHER-IS-THE-KANBAN-AST`),
- SurrealQL queries lower to (per `E-AR-DO-WIRING`'s `ExecTarget::SurrealQl`),
- ractor commands lower to (the mutation channel),
- Adapter calls lower from (when the ontology dispatches OUT).

The probe that hardens this: the same `OgarAst::Do(PostInvoice, …)` must
execute correctly via the Lance-native `compute_dag` path AND via an Odoo
adapter call AND via a SurrealQL transaction. If those three diverge in
SEMANTICS (not just rendering), the AST is leaking syntax and needs
refactoring.

---

## 6. ARM adapter-routing grammar

The ARM is the **routing layer over the AST**. AST is portable
("PostInvoice for I42"); ARM is the dispatch that picks which executor
realizes the operation.

```rust
struct ArmDecision {
    op: OgarAst,
    executor: Executor,
    fallback: Option<Executor>,
}

enum Executor {
    NativeLance,              // run as compute_dag + ractor mutation
    SurrealAst,               // emit DEFINE TABLE / CREATE / UPDATE / RELATE
    HumanKanban,              // surface as a Kanban card needing operator decision
    Adapter(AdapterTargetId), // route to a callcenter-registered backend
}

type AdapterTargetId = &'static str;
// Concrete adapters — Odoo, Rails/OpenProject, SAP, external HTTP — live in the
// callcenter registry, NOT in lance-graph-contract: the contract names only the
// routing slot. Drops pre-council ExternalHttp(Url) (zero-dep violation) and
// Dll(CapabilityId) (phantom); OdooAdapter/RailsAdapter collapse into Adapter.
```

The ARM consults THINK before picking an executor:

```
Verdict::ProceedAs(plan) → executor = plan.preferred_executor
Verdict::RouteToHuman   → executor = HumanKanban
Verdict::Defer(cond)    → wait; re-route when cond satisfied
Verdict::Reject(reason) → no executor; emit audit event
```

The ARM is **not business logic**. It is the **routing skin** between the
ontology and the world. It does not decide whether an invoice should be
posted (that's THINK). It decides whether the post happens locally,
remotely, or by a human.

This is what `lance-graph-callcenter` becomes: the outer membrane carrying
ARM decisions out.

### Adapter — callcenter-registered backend capability

The third leg of AST/ARM/Adapter: an `Adapter(AdapterTargetId)` is a
registered, runtime-resolvable executor that the ARM can pick. Today's
compile-time executors are `NativeLance` and `SurrealAst`. Tomorrow's
adapters (an SAP adapter, a customer-specific tax engine, Odoo,
Rails/OpenProject) are registered at runtime through `callcenter`'s adapter
registry — never named in `lance-graph-contract`.

The adapter slot keeps the ontology **open** without forcing every adapter
into the compile graph.

---

## 7. Ownership boundaries — ractor compile-time, LanceGraph thinks, SurrealAST + Kanban orchestrate

The crisp split (operator-ratified; **corrects 2026-06-19 first draft** that
conflated ractor's compile-time guarantee with runtime mutation authority):

```
ractor                       COMPILE-TIME ownership guarantee
                             mailbox-as-owner; Rust move/borrow semantics
                             prove no aliasing / no data race / no UAF;
                             UB becomes a compile error (canonical §9
                             E-CE64-MB-4). NOT runtime mutation authority —
                             the mailbox itself IS the owner; ractor makes
                             the pattern type-safe at the type system.

LanceGraph (lance-graph)     THE THINKING — where it actually happens
                             SoA cognitive substrate (MailboxSoA), compute_dag,
                             cycle-aware writes (write_row + wrap-aware u32 gate),
                             zero-copy lifecycle, tombstones. The recompute
                             organ. THINK / DO compute on these columns.

SurrealAST + Kanban          THE ORCHESTRATION — the decision plane
                             SurrealAST drives query + mutation lowering
                             (sister to Cypher AST per E-CYPHER-IS-THE-KANBAN-AST);
                             KanbanColumn (Planning → CognitiveWork →
                             Evaluation → Commit) sequences which cognitive
                             cycle fires when; ARM verdicts route execution.
                             NOT the thinking itself — the dispatch over the
                             thinking.

OGAR (lance-graph-ontology)  Meaning
                             inherited classes, regulatory anchors, the
                             living-anatomy ontology

lance-graph-contract         Interface promises
                             trait surfaces, carriers (Triple, ActionDef,
                             ClassView, MailboxSoA contracts)

lance-graph-callcenter       Outer execution membrane
                             adapter dispatch, adapter registry, human-task
                             Kanban queue, external system integration
```

This is the answer to the implicit question buried in
`E-CYPHER-IS-THE-KANBAN-AST`: **where does the thinking happen and where does the
orchestration happen?**

> Thinking happens in LanceGraph's SoA (cycle-aware writes recompute
> dependents via compute_dag; the mailbox-as-owner pattern is the
> type-safe ownership argument). Orchestration happens in the
> SurrealAST + Kanban plane (which cycle fires when, in which phase,
> with which state transition). ractor is the COMPILE-TIME proof that
> the orchestration→thinking handoff cannot race; it is NOT a runtime
> mutation authority that sits between them.

Concretely:

```
┌─────────────────────────────────────────────────────────────────┐
│  OGAR (ontology — meaning)                                       │
├─────────────────────────────────────────────────────────────────┤
│  lance-graph-contract (interface promises)                       │
├─────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION plane (decides which cycle, when, in what phase)  │
│   ├─ SurrealAST (query + mutation lowering — sister to Cypher)   │
│   └─ Kanban     (Rubicon phases: Plan → Cog → Eval → Commit)     │
│                          │                                       │
│                          │ dispatches a cycle into…              │
│                          ▼                                       │
│  THINKING plane (where the recompute actually fires)             │
│   └─ LanceGraph SoA (MailboxSoA, compute_dag, write_row(cycle))  │
│                          ▲                                       │
│                          │ proved race-free at compile time by…  │
│                          │                                       │
│  ractor (compile-time mailbox-as-owner ownership guarantee)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  lance-graph-callcenter
                  (outer adapter membrane — tools, ERPs, humans)
```

The single binary holds all of this. The layers are **logical**, not
process boundaries. But they MUST NOT collapse into soup — the
ownership rules above are the discipline that keeps the layers
distinct in code review. The most common drift to watch for: collapsing
"orchestration" and "thinking" into "ractor runs the business logic" —
that's the misframe this section corrects.

---

## 8. Rubicon / Heckhausen Kanban execution loop

The execution-time decision pattern is **the orchestration plane** (per §7
correction). SurrealAST handles the query/mutation lowering side;
Kanban-with-Rubicon-phases handles the sequencing side. Together they
DECIDE which cycle fires when. They do NOT do the thinking — the thinking
fires in LanceGraph's SoA inside the Commit phase. (Named in
`E-CYPHER-IS-THE-KANBAN-AST` — extended here with the Rubicon-Heckhausen
phase model and the AST/ARM/thinking-plane connection):

```
   ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
   │ Planning │ →  │ Cognitive    │ →  │ Evaluation│ →  │  Commit  │
   │ (intent) │    │  Work        │    │ (Rubicon: │    │ (gate +  │
   │          │    │ (THINK)      │    │  cross?)  │    │  DO)     │
   └──────────┘    └──────────────┘    └───────────┘    └──────────┘
        │                                                     │
        └────────────────────── failure / defer ──────────────┘
                                       │
                                       ▼
                              [ HumanKanban card ]
```

Mapping to Heckhausen's Rubicon model:

- **Pre-decisional (Planning)** — the actor weighs alternatives. An AST
  `OgarAst::Think(WhichInvoiceShouldPostNext)` returns ranked candidates.
- **Pre-actional (Cognitive Work)** — the AST is built; THINK is evaluated;
  ARM resolves executor. The Rubicon is the point of decision: cross OR
  defer.
- **Actional (Evaluation + Commit)** — DO fires through the mailbox-cycle
  commit gate; cycle-aware `write_row` lands the mutation.
- **Post-actional (next planning cycle)** — the result feeds the next
  planning round; failures become Kanban cards.

The Kanban board IS the graph (per `E-CYPHER-IS-THE-KANBAN-AST`):
column = state, card = node, move = edge rewrite. The Rubicon phase is
the lifecycle of ONE mailbox; the BOARD is the lifecycle of business
objects (invoices in Draft → Posted → Reversed). Same algebra, two
scales.

---

## 9. Callcenter outer membrane

`lance-graph-callcenter` is the **hands** of the system — the routing
skin between the ontology and the world. NOT business logic; routing.

What callcenter holds:

- Adapter target registry (Odoo XML-RPC endpoint, Rails HTTP API,
  WoA REST, SAP RFC, …).
- Runtime-resolved `Adapter(AdapterTargetId)` executors (registered in the table above).
- Human-task queue (Kanban cards that need operator decisions).
- Tool-call surface for external integrations (LLM tools, MCP servers).

What callcenter does NOT hold:

- Class definitions (those live in OGAR).
- Action gates (those live in `ActionDef.commit`).
- State (that lives in Lance SoA columns).
- Policies (those live in `ClassView::policies` THINK slot).

This is the discipline that keeps `callcenter` thin. It's a **membrane**,
not an organ.

---

## 10. Minimal example — Invoice + PostInvoice

The whole stack lit up on one operation:

### THING — Invoice class in OGAR

```
class Invoice
    is_a: LegalDocument <: EconomicCommitment <: SocialObject
    regulatory_anchor:
      - GoBD §146-148 (German tax law: hash chain, immutability after post)
      - HGB §238 (commercial bookkeeping)
    has_state_machine:
      Draft → Posted → Reversed
    fields:
      number          :: Sequence<Invoice>
      posting_date    :: Date (GoBD-immutable-after-Posted)
      tenant_id       :: TenantScope
      lines           :: has_many(InvoiceLine)
      audit_chain     :: AuditChain
    adapter_targets:
      - Odoo: account.move (move_type=out_invoice)
      - Rails: Invoice ActiveRecord
      - WoA: Vorgang (doc_type=invoice)
      - SAP: FI Document (later)
```

### DO — PostInvoice action

```
action PostInvoice(this: Invoice)
    requires_role: 'buchhaltung'
    state_guard: this.state == Draft
    effects:
      reserve_sequence(this.number)
      compute_posting_lines(this.lines) → JournalEntries
      write_immutable_event(audit_chain, this)
      transition(this.state, Draft → Posted)
    commit_gate:
      def-match → RBAC(actor.role ⊇ 'buchhaltung')
        → state-guard(this.state == Draft)
        → MUL(impact ≤ homeostasis floor)
```

### THINK — policies that gate PostInvoice

```
policy GoBdImmutabilityCheck(this: Invoice)
    when: about_to_transition(this.state, _ → Posted)
    asserts:
      audit_chain_prev_hash_matches(this)
      sequence_gapless(this.number)
      posting_date_within_open_period(this.posting_date, this.tenant_id)
    on_fail: Verdict::Reject('GoBD violation: <detail>')

policy ActorAuthorized(action: ActionRef, actor: ActorContext)
    when: any action proposed
    asserts:
      actor.roles ⊇ action.required_role
      action.tenant == actor.session_tenant
    on_fail: Verdict::Reject('unauthorized for this tenant')

policy RubiconImpactCheck(this: Invoice, action: PostInvoice)
    when: about_to_commit
    asserts:
      MUL.impact(action) ≤ MUL.homeostasis_floor
    on_fail: Verdict::RouteToHuman('high-impact post — needs operator')
```

### AST — what the operation compiles to

```rust
OgarAst::Do(
    PostInvoice,
    BindingSet { this: I42 },
)
```

### ARM — what the routing decides

```
Verdict::ProceedAs {
    preferred_executor: Executor::NativeLance,
    fallback:          Some(Executor::SurrealAst),
}
```

(THINK passed; PostInvoice runs locally; if Lance-native path errors,
SurrealQL transaction fallback.)

### Execution path

```
PostInvoice AST
  → ORCHESTRATION plane: ARM decides Executor::NativeLance
  → ORCHESTRATION plane: Kanban advances Planning → CognitiveWork
  → THINK policies evaluated (Cognitive Work phase)
  → ORCHESTRATION plane: Kanban advances CognitiveWork → Evaluation
       Rubicon decision: cross?
  → THINKING plane fires:
        ActionDef.commit(def, actor, impact, guard='Draft', now)
            → ActionState::Committed
        MailboxSoA::write_row(I42, cycle, WriteCell{state=Posted, audit=new_hash})
        compute_dag dirties: lines (recompute JournalEntries), summary fields
        cycle-aware write lands; tombstones prior version
            — ractor's compile-time mailbox-as-owner guarantee proves
              this write cannot race the read that THINK just did,
              cannot alias any other mailbox, cannot UAF — and that
              proof is at the type system, not at runtime
  → ORCHESTRATION plane: Kanban advances Evaluation → Commit phase complete
  → ORCHESTRATION plane: SurrealAST projection
        `UPDATE Invoice:I42 SET state='Posted', ...`
        (the query side of the orchestration plane — readers see the change)
  → callcenter (outer membrane): emits audit event to external GoBD
        compliance system (via Executor::Adapter("gobd-compliance"))
```

This is **one operation**. Six layers participate (ontology, contract,
orchestration, thinking, ractor's compile-time proof, membrane).
None of them is the "Odoo transcode." All of them are doing the work the
ontology specifies. The thinking is in LanceGraph; the orchestration
is in SurrealAST + Kanban; ractor proves the handoff is safe at compile
time; the membrane reaches out.

That is the OGAR endgame.

---

## 11. What this doc supersedes / leaves intact

**Supersedes the framing of:**
- Any prior framing that called the deliverable "an Odoo SurrealQL DDL
  emitter" (the DDL emission is one *adapter target*, not the deliverable).
- The "WoA-rs as first SurrealQL consumer" framing
  (`E-AR-PROJECTION-CORRECTION-1` retraction made explicit).
- The "AR-shape is flat triples" framing (rejected as syntax-preservation).

**Leaves intact:**
- The polyglot extractor pattern (`codegen_spine::Triple` carrier; per-ERP
  extractors emit Triples with backend-specific predicate extensions).
- The OGAR Core's existing typed slots (`ClassView`, `ClassActions`,
  `ActionDef`, `MailboxSoA`, `compute_dag`, etc.).
- The 5+3 council verdict on the **typed AST placement** question
  (Phase 1 Option A in nexgen; Phase 2 Option D in surrealdb fork via
  C16b/C16c). That decision is one *adapter target* of this larger
  ontology — important but not the doctrine.
- `E-OGAR-IS-FOUNDRY`, `E-AR-DO-WIRING`, `E-CYPHER-IS-THE-KANBAN-AST`,
  `E-GUID-IS-THE-GRAPH`, `E-ODOO-CORE-FIRST-STRUCTURAL` — this doc is the
  capstone that names what those entries point to together.

## 12. Conjecture / Finding status

Per workspace discipline:

| Section | Status |
|---|---|
| §1 flat-triples rejected | FINDING (operator-ratified; supersedes prior framing) |
| §2 curators-not-foundations | FINDING (operator-ratified; ≥2-curator promotion gate doctrinal) |
| §3 inherited-class model | FINDING (the shape) + CONJECTURE (the §4 THINK slot extension to ClassView) |
| §4 THING / DO / THINK | FINDING (the trichotomy) + CONJECTURE (THINK as new typed slot beside actions) |
| §5 AST grammar | CONJECTURE (named, not yet shipped — probe: same `OgarAst::Do` runs identically on Lance + SurrealQL + Odoo adapter) |
| §6 ARM grammar | CONJECTURE (named, not yet shipped — overlaps `OrchestrationBridge` today; the Adapter slot is new) |
| §7 ownership boundaries | FINDING (operator-ratified, codifies prior architecture) |
| §8 Rubicon Kanban loop | FINDING (Rubicon phases already in `KanbanColumn`) + CONJECTURE (AST/ARM connection) |
| §9 callcenter as membrane | FINDING (matches existing callcenter scope) |
| §10 Invoice example | CONJECTURE (illustrative; the named policies + ARM verdicts are not yet code) |

The next session that wants to harden a CONJECTURE picks one row and
ships the probe. Don't ship more synthesis without a measurement.

---

## 13. Glossary

| Term | Means |
|---|---|
| **OGAR** | Ontology + Graph + Action + Reasoning. The deliberate typed Core. |
| **AR-shape** | The cross-curator adapter grammar (Active Record-inspired but ERP-agnostic). |
| **AST** | `OgarAst` — portable semantic operation tree. |
| **ARM** | The routing decision layer over the AST. Picks an executor. |
| **Adapter** | Callcenter-registered backend capability (runtime-resolved via `Adapter(AdapterTargetId)`). |
| **THING** | DOLCE Endurant. State that exists. SoA columns. |
| **DO** | DOLCE Perdurant. Mutations that commit through the gate. |
| **THINK** | New typed slot. Continuous policies, never commit, gate every DO. |
| **Curator** | An ERP (Rails/Odoo/WoA/SAP) used as a teaching corpus, not a foundation. |
| **Promotion rule** | A primitive enters OGAR Core when ≥2 independent curators surface it under different syntactic forms. |
| **Rubicon** | Heckhausen's pre-/actional decision threshold. The phase boundary in the orchestration plane. |
| **Orchestration plane** | SurrealAST + Kanban. The decision plane that sequences which cycle fires when. NOT the thinking. |
| **Thinking plane** | LanceGraph's SoA — MailboxSoA, compute_dag, cycle-aware writes. Where recompute actually happens. |
| **ractor** | The compile-time mailbox-as-owner ownership guarantee. Rust move/borrow semantics prove no aliasing / no race / no UAF. NOT a runtime mutation authority — UB becomes a compile error (canonical §9 E-CE64-MB-4). |
| **Membrane** | callcenter's role — routing skin between ontology and world. |

---

_"flat triples = corpse scan. OGAR classes = living anatomy."_
