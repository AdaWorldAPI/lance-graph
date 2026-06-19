# OGAR AR-Shape Endgame — the ontology compiler, not an Odoo transcode

> **Audience:** every future session that touches Odoo / OpenProject / WoA /
> SMB-Office / SAP transcode, or proposes a new ERP curator.
> **Status:** doctrine (operator-ratified 2026-06-19; closes the
> 5+3 council on `E-AR-PROJECTION-CORRECTION-1`).
> **One-line read:** flat triples are corpse-scan; OGAR classes are living
> anatomy; ERP curators teach the system what business software keeps
> reinventing; the deliverable is an **executable ontology**, not a port.
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
   ractor-owned LanceGraph SoA  +  SurrealDB / Lance KVS  +  callcenter membrane
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

## 2. Rails / OpenProject / Odoo / WoA-rs / SAP as curators, not foundations

These ERPs are **fossils in the cliff-face** — what we learn from, not what
we depend on. They teach the canonical primitives that every business
software keeps reinventing:

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
    NativeLance,       // run as compute_dag + ractor mutation
    SurrealQl,         // emit DEFINE TABLE / CREATE / UPDATE / RELATE
    OdooAdapter,       // route to a live Odoo instance via XML-RPC / RPC
    RailsAdapter,      // route to OpenProject/Rails AR over HTTP
    HumanKanban,       // surface as a Kanban card needing operator decision
    ExternalHttp(Url), // call out
    Dll(CapabilityId), // dynamically loaded backend capability
}
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

### DLL — dynamically loaded backend capability

The third leg of AST/ARM/DLL: a `DLL` is a registered, dynamically
loadable executor that the ARM can pick. Today's executors
(NativeLance, SurrealQl) are compile-time. Tomorrow's DLLs (an SAP
adapter, a customer-specific tax engine) are registered at runtime
through `callcenter`'s capability table.

The DLL slot keeps the ontology **open** without forcing every adapter
into the compile graph.

---

## 7. SurrealDB / Lance / ractor ownership boundaries

The crisp split (operator-ratified):

```
ractor                  owns mutation authority + actor mailboxes
LanceGraph (lance-graph) owns physical SoA memory + zero-copy lifecycle + tombstones
SurrealDB / SurrealQL    owns query/control plane + schema projection + live queries
OGAR (lance-graph-ontology) owns meaning
lance-graph-contract     owns interface promises (trait surfaces, carriers)
lance-graph-callcenter   owns outer execution membrane (adapter dispatch)
```

This is the answer to the implicit question buried in
`E-CYPHER-IS-THE-KANBAN-AST`: **who owns business state?**

> Not SurrealDB. SurrealDB is the Rubicon/Kanban/query plane sitting OVER
> Lance-backed KVS.

Concretely:

```
┌─────────────────────────────────────────────────────────────────┐
│  OGAR (ontology — meaning)                                       │
├─────────────────────────────────────────────────────────────────┤
│  lance-graph-contract (interface promises)                       │
├─────────────────────────────────────────────────────────────────┤
│  SurrealQL projection (query / live query / DDL view)            │
│       ▲                                                          │
│       │ reads / writes through controlled KVS path               │
│       │                                                          │
│  ractor (mutation authority, transactional choreography)         │
│       │                                                          │
│       ▼ owns                                                     │
│  LanceGraph SoA (physical columns, zero-copy, versioned)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  lance-graph-callcenter
                  (outer adapter membrane — tools, ERPs, humans)
```

The single binary holds all of this. The layers are **logical**, not
process boundaries. But they MUST NOT collapse into soup — the
ownership rules above are the discipline that keeps the layers
distinct in code review.

---

## 8. Rubicon / Heckhausen Kanban execution loop

The execution-time decision pattern (already named in
`E-CYPHER-IS-THE-KANBAN-AST` — extended here with the Rubicon-Heckhausen
phase model and the AST/ARM connection):

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
- Capability table for DLL executors (dynamically loaded).
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
    fallback:          Some(Executor::SurrealQl),
}
```

(THINK passed; PostInvoice runs locally; if Lance-native path errors,
SurrealQL transaction fallback.)

### Execution path

```
PostInvoice AST
  → ARM decides NativeLance
  → ractor mailbox receives mutation request
  → THINK policies re-checked at commit-time
  → ActionDef.commit(def, actor, impact, guard='Draft', now) → ActionState::Committed
  → MailboxSoA::write_row(I42, cycle, WriteCell{state=Posted, audit=new_hash})
  → compute_dag dirties: lines (recompute JournalEntries), summary fields
  → cycle-aware write lands; tombstones prior version
  → SurrealQL projection: `UPDATE Invoice:I42 SET state='Posted', ...`
       (the QUERY plane sees the change; not the mutation authority)
  → callcenter: emits audit event to external GoBD compliance system
       (via Executor::ExternalHttp)
```

This is **one operation**. Five layers participate. None of them is the
"Odoo transcode." All of them are doing the work the ontology specifies.

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
| §6 ARM grammar | CONJECTURE (named, not yet shipped — overlaps `OrchestrationBridge` today; the DLL slot is new) |
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
| **DLL** | Dynamically loaded backend capability (registered at runtime). |
| **THING** | DOLCE Endurant. State that exists. SoA columns. |
| **DO** | DOLCE Perdurant. Mutations that commit through the gate. |
| **THINK** | New typed slot. Continuous policies, never commit, gate every DO. |
| **Curator** | An ERP (Rails/Odoo/WoA/SAP) used as a teaching corpus, not a foundation. |
| **Promotion rule** | A primitive enters OGAR Core when ≥2 independent curators surface it under different syntactic forms. |
| **Rubicon** | Heckhausen's pre-/actional decision threshold. The commit gate. |
| **Membrane** | callcenter's role — routing skin between ontology and world. |

---

_"flat triples = corpse scan. OGAR classes = living anatomy."_
