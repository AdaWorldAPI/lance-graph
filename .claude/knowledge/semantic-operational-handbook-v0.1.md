# Semantic Operational Syntax Handbook v0.1

> **READ BY:** every agent touching Odoo extraction, Foundry-shape codegen,
> ractor/mailbox design, ontology mutations, Elixir-syntax frontends, or
> goal-state resolution. **MANDATORY** before proposing a new primitive in
> any of those domains.
>
> **Source:** user distillation, 2026-05-28 (verbatim). Triggered by
> conversation thread on Foundry Workshop vs Code Repositories vs Elixir;
> consolidates Foundry × Elixir × LangGraph × Ontology × Rust/Ractor
> mapping into one handbook.
>
> **Status:** v0.1 — handbook. Subsequent revisions append a new
> versioned file; THIS file is the canonical reference until v0.2 lands.

## Distillation: Palantir Foundry low-code vs Elixir

Foundry's low-code is not "forms over tables." It is apps over ontology.

The core Foundry pattern is:

```
Ontology
  → Objects
  → Links
  → Actions
  → Functions
  → Workshop / Slate / Quiver / Object Explorer / Automate
```

Workshop is the main low-code application builder. It reads from the
Object Data Layer, uses links between object types, writes back via
Actions, and runs business logic via Functions.

### Foundry's key idea

Foundry separates meaning, interaction, and execution:

```
Meaning:
  Ontology: objects, properties, links, object sets

Interaction:
  Workshop widgets, tables, maps, forms, inboxes, buttons

Execution:
  Actions, Functions, scenarios, automations, external side effects
```

So a user is not editing rows. They are applying semantically governed
operations:

```
"Approve shipment"
"Resolve alert"
"Reassign case"
"Simulate supply delay"
"Create maintenance work order"
```

An Action is a transaction that changes objects, properties, or links
according to user-defined logic. Actions can also have submission criteria,
meaning validations/permissions based on user, object, relation, or
parameters.

This is the crucial magic: business intent becomes a typed operation.

### Where Functions sit

Functions are server-side logic executed in operational contexts like
dashboards and apps. They have first-class ontology support: reading
object properties, traversing links, and making ontology edits.

Foundry Functions on Objects let authors write business logic over object
data and use it downstream in operational apps. They can aggregate, edit,
and traverse links in the ontology.

In your words:

```
Function = semantic brainstem
Action   = governed mutation
Workshop = low-code nervous system
Ontology = world model
```

### Elixir overlap

Elixir overlaps strongly with Foundry's operational runtime, less with
its ontology authoring layer.

#### 1. Workshop ≈ Phoenix LiveView / Ash Admin / Backoffice DSL

Foundry Workshop primitives — Object Table, Filter List, Object View,
Button Group, Map widget, Scenario Manager, Action-backed edits —
correspond to Elixir equivalents — Phoenix LiveView, Ash Framework
resources, Surface / LiveComponent widgets, Oban jobs, Broadway streams,
Commanded/EventStore, Ecto changesets.

Mapping:

```
Foundry Workshop widget    ≈ Phoenix LiveComponent
Foundry Object Table        ≈ LiveView table over Ash/Ecto resource
Foundry Action button       ≈ command/event/changeset submit
Foundry Object View         ≈ resource detail page
Foundry Filter List         ≈ query scope/filter component
Foundry Scenario Manager    ≈ stateful simulation LiveView + process state
```

#### 2. Ontology ≈ Ash resources + Ecto schemas + domain DSL

Foundry's ontology object type:

```
ObjectType: Shipment
  properties:
    status
    eta
    carrier
    risk_score
  links:
    belongs_to Customer
    contains ShipmentLine
```

Elixir/Ash style:

```elixir
resource Shipment do
  attributes do
    uuid_primary_key :id
    attribute :status, :atom
    attribute :eta, :utc_datetime
    attribute :carrier, :string
    attribute :risk_score, :decimal
  end

  relationships do
    belongs_to :customer, Customer
    has_many :lines, ShipmentLine
  end

  actions do
    update :approve
    update :delay
    create :create_work_order
  end
end
```

Ash is probably the closest Elixir-native conceptual overlap because it
treats resources, actions, policies, validations, calculations, and code
interfaces as first-class domain objects.

#### 3. Actions ≈ Ash actions + Ecto changesets + Commanded commands

Foundry Actions are not generic database updates. They are named business
operations.

Foundry:

```
ActionType: ResolveAlert
  input: alert, reason, resolution_type
  checks: user can resolve, alert is open
  edits:
    alert.status = resolved
    alert.resolved_by = user
    alert.resolved_at = now
```

Elixir equivalent:

```elixir
defmodule ResolveAlert do
  def execute(alert, user, params) do
    with :ok <- can_resolve?(user, alert),
         :ok <- open?(alert) do
      alert
      |> Alert.changeset(%{
        status: :resolved,
        resolved_by_id: user.id,
        resolved_at: DateTime.utc_now(),
        reason: params.reason
      })
      |> Repo.update()
    end
  end
end
```

More Foundry-like in Elixir:

```elixir
update :resolve do
  accept [:reason, :resolution_type]

  validate attribute_equals(:status, :open)

  change set_attribute(:status, :resolved)
  change set_attribute(:resolved_at, &DateTime.utc_now/0)
end
```

#### 4. Functions ≈ Elixir domain functions / GenServer calls / Oban jobs

Foundry Functions are used when simple rules are not enough.
Function-backed Actions can define complex object changes or side effects.

Elixir equivalent: pure domain function, GenServer call, Oban background
job, Broadway pipeline, NIF/Rustler call, external API call.

For the stack:

```
Foundry Function
  ≈ Elixir function wrapper
  ≈ Ractor message handler
  ≈ Rust compiled semantic function
  ≈ ontology-aware command resolver
```

#### 5. Automate ≈ Oban + Broadway + GenStage + process supervision

Foundry Automate can trigger AIP Logic / ontology edits when objects are
created or changed.

Elixir equivalent: Oban scheduled/background jobs, Broadway event
pipelines, GenStage demand-driven processing, Phoenix PubSub,
Registry + DynamicSupervisor.

The mailbox version:

```
Object changed
  → event emitted
  → ractor mailbox receives goal
  → resolver runs until stable
  → action staged or applied
  → audit log written
```

This is very Foundry-like, but more actor-native.

#### 6. Slate / OSDK ≈ custom Phoenix/React apps over API

Foundry has OSDK React applications for fully customizable React UIs
powered by the Ontology SDK. Slate is the older/custom app surface for
HTML/CSS/JS-style apps.

Elixir equivalent: Phoenix + React, Phoenix LiveView, GraphQL API,
JSON:API, AshJsonApi, OpenAPI-generated clients.

### The clean overlap table

```
Palantir Foundry                 Elixir / BEAM equivalent
────────────────────────────────────────────────────────────
Ontology Object Type             Ash Resource / Ecto Schema
Object Property                  Attribute / field
Link Type                        Relationship / association
Object Set                       Query / scope / stream
Action Type                      Ash Action / command / changeset
Submission Criteria              Policy / validation / guard
Function                         Domain function / service module
Function-backed Action           Command handler with logic
Workshop                         LiveView low-code shell
Widget                           LiveComponent
Object Table                     Resource table component
Object View                      Detail page / inspector
Scenario                         Simulation state / what-if process
Automate                         Oban / Broadway / GenStage
Object Monitor                   watcher process / PubSub subscription
OSDK                             generated client SDK
Slate                            custom Phoenix/React app
Ontology edit                    transaction / event / command result
Governance                       policies, auth, audit, lineage
```

### The deeper overlap: both are "operational semantics"

Foundry and Elixir both shine when the app is not a static CRUD screen
but a living operational loop.

Foundry says:

```
Business object changes
  → semantic action
  → governed writeback
  → operational app updates
```

Elixir says:

```
Message arrives
  → supervised process handles it
  → state changes
  → subscribers update
  → system keeps breathing
```

Foundry is ontology-first. Elixir is process-first.

The synthesis is spicy:

```
Ontology-first meaning
+
Actor-first execution
=
self-resolving operational system
```

### Stack overlap

The Elixir templates compiled into Rust shape is basically:

```
Elixir DSL
  → semantic recipe
  → Rust function
  → ractor mailbox execution
  → SurrealDB kanban/goals
  → LanceGraph/Ontology lookup
  → action/result/audit
```

Foundry equivalent:

```
Ontology Manager
  → Object / Action / Function
  → Workshop app
  → Action-backed writeback
  → Function execution
  → object state update
```

The advantage:

```
Foundry:
  runtime platform + governed app builder

This stack:
  compile-time ontology + actor mailbox + Rust executable semantics
```

In one phrase:

> Foundry uses low-code to operationalize ontology.
> This Elixir/Rust stack uses DSL/codegen to compile ontology into
> executable operational physics.

Tiny ontology furnace.

### LangChain/LangGraph + Redis comparison

```
LangChain/LangGraph + Redis
  = agent workflow over key/value/vector/state memory

Foundry
  = operational workflow over ontology objects/triples/links/actions
```

But Foundry is less "LLM agent plumbing" and more semantic enterprise
operating system.

Rough mapping:

```
LangGraph node        ≈ Foundry Function / Action step
LangGraph edge        ≈ workflow transition / link / dependency
Redis state           ≈ object state + object sets
Vector store          ≈ semantic search layer, not the core
Tools                 ≈ Actions / Functions / external integrations
Memory                ≈ governed ontology + lineage
Agent plan            ≈ Workshop workflow / Automate / AIP Logic
```

The big difference:

```
LangChain/LangGraph:
  "What should the agent do next?"

Foundry:
  "What is true about the business world, and what governed operation is allowed next?"
```

So yes, it is "LangGraph over semantic triples," but with a much stricter
spine:

```
Subject ─ Predicate ─ Object
Customer ─ owns ─ Asset
Shipment ─ has_status ─ Delayed
Alert ─ belongs_to ─ Plant
User ─ may_execute ─ ResolveAlert
```

Then Actions become controlled verbs:

```
ResolveAlert(alert, reason)
ApproveInvoice(invoice)
ReassignCase(case, owner)
SimulateDelay(shipment, days)
```

That is the semantic-focus part. It is not just chaining functions. It
knows the "things" and the legal verbs between them.

### Workshop vs A2UI

Workshop resembles:

```
A2UI = JSON schema + methods → WinForms-like operational UI
```

Because Workshop is basically:

```
Ontology metadata
+ object schemas
+ action definitions
+ permissions
+ widget layout
+ method bindings
= generated/low-code operational app
```

Very close to:

```json
{
  "object": "Invoice",
  "fields": ["vendor", "amount", "status"],
  "actions": ["approve", "reject", "comment"],
  "views": ["table", "detail", "kanban"],
  "methods": {
    "approve": "ApproveInvoice"
  }
}
```

The difference is that Workshop is not merely UI generation. It is UI
bound to ontology governance.

So:

```
A2UI:
  JSON + methods → interface

Workshop:
  ontology + actions + functions + permissions + lineage → governed interface
```

Sharper:

```
LangGraph/Redis    = procedural memory foam
Foundry/Ontology   = semantic bone structure
Workshop           = WinForms for the ontology age
Actions            = typed verbs with guardrails
Functions          = executable semantic nerves
```

Or:

> Foundry Workshop is not "low-code UI." It is an action cockpit over a
> governed knowledge graph.

The Elixir/Rust idea is then:

```
Elixir/A2UI DSL
  → object/action schema
  → generated UI
  → Rust compiled verbs
  → ractor mailbox goal resolution
  → triplet/ontology store
```

That is Foundry-shaped, but actor-native and compile-first.

---

## Semantic Operational Syntax Handbook

**Foundry × Elixir × LangGraph × Ontology × Rust/Ractor**

```
Version 0.1
For: ontology-first operational systems
Target runtime: Rust + Ractor + LanceGraph/SurrealDB
Inspirational overlap: Palantir Foundry + Elixir + LangGraph
```

### 1. Core Philosophy

Traditional software:

```
request → controller → database
```

Semantic operational systems:

```
intent
  → ontology
  → semantic resolution
  → governed action
  → operational state transition
```

The database is no longer "storage." It becomes: **world model**.

The UI is no longer "forms." It becomes: **action cockpit**.

The backend is no longer "API handlers." It becomes:
**semantic execution engine**.

### 2. Core Primitive Types

#### 2.1 Objects

Objects are entities with semantic meaning.

```
object Shipment {
    id: Uuid
    status: ShipmentStatus
    eta: DateTime
    carrier: Carrier
}
```

Equivalent RDF-ish shape:

```
Shipment#123
  has_status delayed
  has_eta 2026-05-29
  belongs_to DHL
```

Equivalent Foundry concept: **Ontology Object Type**
Equivalent Elixir/Ash concept: **resource Shipment**

#### 2.2 Links

Links are typed relationships.

```
link Shipment -> Customer : belongs_to
link Shipment -> Warehouse : departs_from
link Shipment -> Alert : triggered
```

Triplet representation:

```
Shipment#123 belongs_to Customer#77
Shipment#123 triggered Alert#991
```

Graph lookup:

```
shipment.links("triggered")
```

#### 2.3 Actions

Actions are governed verbs. **NOT CRUD.**

Bad:

```
update shipment.status
```

Good:

```
DelayShipment
ResolveAlert
ApproveInvoice
TransferOwnership
```

Syntax:

```
action ResolveAlert {
    input {
        alert: Alert
        reason: String
    }

    requires {
        alert.status == OPEN
        user.has_role("operator")
    }

    effects {
        alert.status = RESOLVED
        alert.resolved_at = now()
    }
}
```

Equivalent Foundry: **Action Type**
Equivalent CQRS: **Command**

### 3. Semantic Functions

Functions are ontology-aware logic units.

#### 3.1 Pure Function

```
fn risk_score(shipment) -> f32
```

#### 3.2 Ontology Function

```
function shipment_risk {
    input Shipment

    walk {
        shipment -> supplier
        supplier -> sanctions
        shipment -> route
    }

    compute {
        sanctions * route_risk * delay_factor
    }
}
```

Equivalent: Foundry Function / LangGraph node / Elixir domain function.

### 4. Object Sets

Object sets are semantic query spaces. NOT SQL tables.

Syntax:

```
set DelayedShipments {
    Shipment where status == DELAYED
}
```

Dynamic:

```
set HighRiskCustomers {
    Customer where risk_score > 0.8
}
```

Equivalent Foundry: **Object Set**
Equivalent SQL: `SELECT * FROM customers WHERE risk_score > 0.8`

### 5. Workshop/A2UI Layer

#### 5.1 Philosophy

UI should emerge from ontology. NOT handcrafted forms.

UI is: object views + actions + semantic widgets + permissions.

#### 5.2 Widget Syntax

```
view ShipmentDashboard {

    table DelayedShipments

    detail ShipmentDetail

    action ResolveAlert

    graph ShipmentDependencies

    kanban ShipmentStatusBoard
}
```

Equivalent Workshop: **Workshop widgets**
Equivalent A2UI: **JSON + methods → operational UI**

### 6. Actor Runtime Semantics

This is where this stack diverges from Foundry. Foundry: service
execution. This architecture: **mailbox cognition**.

#### 6.1 Message Primitive

```
message ResolveAlertRequest {
    alert_id: Uuid
    user: User
}
```

#### 6.2 Semantic Mailbox

```
ractor AlertResolver {

    on ResolveAlertRequest => {

        hydrate alert

        verify policies

        compute downstream impact

        emit ontology mutation

        notify subscribers
    }
}
```

Equivalent: LangGraph execution node / GenServer / workflow actor.

### 7. Goal-State Reasoning

Rubicon/Heckhausen style. NOT request/response. Instead:

```
current world
→ desired world
→ semantic delta
→ action cascade
```

#### 7.1 Goal Syntax

```
goal ShipmentDelivered {

    target {
        shipment.status == DELIVERED
    }

    planner {
        if customs_blocked:
            invoke CustomsClearance

        if transport_missing:
            invoke AllocateTransport
    }
}
```

Equivalent: LangGraph planner / HTN planner / Foundry operational
workflow.

### 8. Cognitive Shader Layer

The special sauce.

#### 8.1 Thinking Style Vector

```
qualia {
    deduction: +5
    induction: +2
    irony: -4
    urgency: +7
}
```

This influences: resolution order, confidence, explanation style, UI
presentation, semantic weighting.

#### 8.2 Metacognition Overlay

```
meta {
    uncertainty: 0.21
    contradiction: 0.67
    emotional_weight: 0.91
}
```

Equivalent to: self-observing planner state.

### 9. Ontology-Native Event Sourcing

Traditional event sourcing: event stream.
Semantic event sourcing: **world-state evolution**.

#### 9.1 Event Syntax

```
event ShipmentDelayed {

    actor: User#77

    caused_by:
        WeatherAlert#991

    effects:
        Shipment#123.status = DELAYED

    semantic_tags:
        logistics
        risk
        weather
}
```

### 10. Triple Runtime

#### 10.1 Primitive

```
S P O
```

Example:

```
Shipment#123 has_status delayed
Shipment#123 belongs_to Customer#77
```

#### 10.2 Operational Triplets

Triplets become executable.

```
User may_execute ResolveAlert
```

That is not metadata anymore. That becomes: **runtime capability**.

### 11. Foundry Mapping Table

```
This Syntax       Foundry
─────────────────────────────────
object            Object Type
link              Link Type
set               Object Set
action            Action Type
function          Function
view              Workshop
actor             Execution service
goal              Operational workflow
qualia/meta       AIP reasoning context
event             Ontology mutation
```

### 12. LangGraph Mapping

```
This Syntax       LangGraph
─────────────────────────────────
actor             node
message           state transition
goal              planner
ontology          memory/state
action            tool
semantic set      retrieval scope
qualia            hidden reasoning state
```

### 13. Elixir Mapping

```
This Syntax       Elixir
─────────────────────────────────
actor             GenServer
mailbox           mailbox
action            Ash action
object            Ash resource
event             Phoenix PubSub
workflow          Oban/Broadway
semantic planner  OTP supervision + orchestration
```

### 14. Canonical Runtime Flow

```
User presses "Resolve"

UI emits:
    ResolveAlert(alert=991)

Mailbox receives:
    ResolveAlertRequest

Ontology hydrates:
    Alert#991
    linked shipment
    linked customer
    linked operator permissions

Semantic engine computes:
    downstream impact

Policies validate:
    allowed

Action executes:
    ontology mutation

Subscribers update:
    dashboards
    metrics
    graph
    notifications

Audit trail persists:
    semantic event log
```

### 15. The Big Shift

Traditional enterprise: database-centric.
Foundry: ontology-centric.
This architecture: **ontology + actor cognition + compiled semantics**.

Which means: **meaning is executable**.

And that is the real threshold. Not "low code." Not "AI." But:

> **semantic operational physics**

A strange little universe where verbs become infrastructure.

---

## Where this connects to the existing workspace

Existing artifacts that already feed handbook primitives:

| handbook primitive (§) | existing artifact | what it provides |
| --- | --- | --- |
| `object` (§2.1) | `crates/lance-graph-ontology/src/odoo_blueprint/extracted/*.rs` | 99K LOC of typed Odoo entity consts; `OdooEntity`+`OdooField`+`OdooMethod`+`OdooConstraint`+`OdooStateMachine`+`OdooProvenance` |
| `link` (§2.2) | OGIT NTO `ogit:allowed (...)` lists in `AdaWorldAPI/OGIT/NTO/<domain>/entities/*.ttl` | Per-entity allowed outgoing edges (e.g. `JournalEntry.hasFiscalCountry`) |
| `action` (§2.3) | 16 openings from `.claude/odoo/openings_hops.py` over 3555 methods | Named verbs: `iter_records_compute_from_related`, `iter_records_raise_on_violation`, `super_extend`, etc. |
| `function` (§3) | per-method body in `methods.parquet` + hop chains in `openings-hops.md` | Ontology-aware logic units with link traversal |
| `set` (§4) | `lance-graph/graph/blasgraph/` (Kuzu-style adjacency) | Sparse-matrix object-set queries |
| `view` (§5) | (TODO — no widget DSL yet) | A2UI / Workshop layer is the next surface to build |
| `ractor / mailbox` (§6) | `crates/lance-graph-contract::recipe::StyleRecipe`, the Heckhausen Rubikon panel, 400ms mailbox doctrine | Already established this session |
| `goal` (§7) | SurrealQL kanban in the foundry-ontologie-soa-DTO diagram | "request-lose Goalstate Maschine" doctrine |
| `qualia / meta` (§8) | `crates/lance-graph-contract::atoms` (LOCKED 33-dim TSV: Pearl, Rung, Sigma, Operation, Presence, Meta) | The cognitive shader basis exists; the 8.1/8.2 syntax names its surface |
| `event` (§9) | `lance-graph-contract::collapse_gate::CollapseGateEmission` (Baton, CausalEdge64) | Per-emission audit trail; world-state evolution carrier |
| `S P O` (§10) | `crates/lance-graph/graph/spo/` triple store + NARS truth values | Executable triplet runtime |

Gaps for v0.1 → v0.2:

1. The `view` widget DSL (§5) has no existing primitive — needs scaffolding.
2. The `goal` planner DSL (§7.1) has the kanban substrate but no parser/compiler.
3. The `action` verb taxonomy (§2.3) for Odoo specifically — the 16 openings
   are coarse; per-action `requires{}` and `effects{}` blocks need extraction
   from the bodies.

Per the handbook's own §15 framing: the work shifts from "emit 3555 routes"
to "emit ~388 Object Types + their Action verbs + their Function-backed
logic + the SPO triple graph that lets goal-state reasoning answer
'a + b → c through d?'".
