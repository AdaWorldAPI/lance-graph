# OGAR API — the consumer contract (THINK + DO)

> **Audience:** the consumer repos that transcode an Active-Record domain onto
> the OGAR Core — `odoo-rs`, `openproject-nexgen-rs`, `woa-rs`, `tesseract-rs`
> (and any future `customer-<name>` crate).
> **Source of truth:** the types in `lance-graph-contract`. This doc is the
> map; the `///` docs on each type are authoritative.
> **Doctrine:** OGAR Core-First — a generated adapter is only as clean as the
> Core it targets. Emit **thin, `classid`-keyed adapters that ASSUME the Core**;
> never a parallel object model, never an adapter that carries its own state.

---

## 0. The one-paragraph model

An Active Record (an Odoo `sale.order`, an OpenProject work-package, a WoA work
order, a Tesseract C++ class) is **fields + methods**. OGAR splits it along
DOLCE: **fields → Endurant → THINK state** (SoA value tenants), **methods →
Perdurant → DO actions** (`ActionInvocation` via `UnifiedStep`), **inheritance →
`classid → ClassView`**. The SPO harvest (`has_function` / `reads_field` /
`inherits_from` / `target` / `validation_kind`) **is** the manifest the consumer
generates from; the consumer never hand-authors the object model.

```
 your domain source ──harvest──► SPO {s,p,o,f,c} ──generate──► const ClassMethods + ClassActions
        │                          (has_function,                       │ (classid-keyed tables)
        │                           reads_field, …)                     ▼
        └────────────────────────────────────────────► the OGAR Core (lance-graph-contract)
                                                         classid → ClassView → ValueTenant (THINK)
                                                         classid → ClassActions → ActionDef (DO)
                                                                    │
                                            ActionInvocation::commit (RBAC + MUL gate)
                                                                    │
                                              UnifiedStep → ExecTarget (Native/Jit/SurrealQl/Elixir)
```

---

## 1. THINK arm — state, identity, methods (already converged + merged)

| Concern | OGAR Core type (`lance-graph-contract`) | Sourced from (harvest) |
|---|---|---|
| **Identity** | `canonical_node::NodeGuid` (`classid` u32 + GUID tail) | the class itself; `classid` bound OGAR-side, never minted by the manifest |
| **State** (Endurant) | `canonical_node::{ValueTenant, ValueSchema, VALUE_TENANTS}`; presence delta = `class_view::FieldMask` | `reads_field` |
| **Method signatures** | `codegen_manifest::{MethodSig, ClassMethods}` + `methods_for(registry, classid)` | `has_function` |
| **Composition / inheritance** | `class_view::ClassView` (`fields()` / `inherit()` / `value_schema()`); `FieldMask::inherit` | `inherits_from` / `virtually_overrides` |
| **Relations** | `canonical_node::EdgeBlock` (12 in-family + 4 out-of-family) | `target` / `inverse_name` |

**Rule:** the `MethodSig`/`ClassMethods` tables and the `ValueSchema` presets are
generated **in your crate** as `const … : &[…]` (every field is `&'static`, so
they compile as `const`). The Core owns the *type + the lookup*; you own the
*data*. `methods_for` is zero-fallback — an unregistered `classid` resolves to an
empty slice, never a panic.

---

## 2. DO arm — actions (`lance-graph-contract::action`)

The Perdurant complement. Generated from `has_function`, gated at commit by RBAC
+ MUL, routed via `UnifiedStep`.

### 2.1 `ActionDef` — static action declaration (`const`-constructible)

```rust
pub struct ActionDef {
    pub predicate:     &'static str,          // the has_function method, e.g. "action_confirm"
    pub object_class:  u32,                    // OGAR classid
    pub exec:          ExecTarget,             // Native | Jit | SurrealQl | Elixir
    pub guard:         Option<StateGuard>,     // KausalSpec: fire only when field == value
    pub required_role: Option<&'static str>,   // RBAC role required (None = unguarded)
    pub overrides:     Option<&'static str>,   // parent-class action this supersedes (inheritance)
}
```

Generate one `const ACTIONS: &[ActionDef]` per class, register as
`ClassActions { classid, actions }`, resolve with
`actions_for(registry, classid)` (zero-fallback, the action-axis sibling of
`methods_for`).

### 2.2 OGAR inheritance — `effective_actions(parent, child)`

A class's DO surface is **its parents' actions + its own, child overrides parent
by `predicate`**. This is `classid → ClassView` inheritance on the action axis —
the same mechanism the field-set uses. You do **not** flatten a parent's actions
into the child; you compose them:

```rust
let eff = effective_actions(parent_class_actions, child_class_actions);
// parent actions, with any same-`predicate` child action substituted, then child net-new appended.
```

### 2.3 `ActionInvocation` — dynamic fire (one per call)

```rust
pub struct ActionInvocation {
    pub object_class: u32, pub predicate: &'static str,  // → the ActionDef realized
    pub object_instance: u32,                            // the GUID identity tail acted on
    pub state: ActionState,                              // Pending → Committed | Failed | Cancelled
    pub cycle: u32,                                      // S2.5 SoA cycle-ownership stamp
    pub idempotency_key: u64, pub trace_id: u64,
    pub emitted_at_millis: Option<u64>,                  // HLC stamp, set on commit
}
```

### 2.4 The commit gate — RBAC then MUL (the egress)

A DO action mutates an external domain system, so it does **not** fire freely.
`ActionInvocation::commit` is the "commit to the external consumer after the
cycle decides the result sound" egress:

```rust
let outcome = inv.commit(&def, &actor /* auth::ActorContext */, &impact /* mul::GateDecision */, now_millis);
```

- **RBAC first** (`auth::ActorContext`): actor must hold `def.required_role`
  (or be admin) → else `ActionState::Failed` (never reaches the impact gate).
- **MUL impact** (`mul::GateDecision`): `Flow → Committed` (HLC-stamped,
  dispatched); `Hold → ` stays `Pending` (escalate / re-assess next cycle);
  `Block → Cancelled`.
- Terminal states are **sticky** (a committed/failed/cancelled action is never
  re-adjudicated).

### 2.5 Dispatch — `UnifiedStep`, never a per-crate endpoint

A committed action runs via `orchestration::{UnifiedStep, OrchestrationBridge}`
routed by `step_type` prefix to a `StepDomain`, at the `ExecTarget` the
`ActionDef` names. `ExecTarget::SurrealQl` lowers the action to SurrealQL and
runs it in the substrate (the AR-shaped API surface). **Do not** add a
`/v1/<crate>` REST endpoint — that is the System-1 trap; extend the canonical
bridge.

---

## 3. Per-consumer wiring (the recipe)

For each consumer crate:

1. **Harvest** your domain into SPO `{s,p,o,f,c}` (`has_function`, `reads_field`,
   `inherits_from`, `target`, `validation_kind`). For Rails/Python ORMs use the
   `ruff`/OGAR producer bridges; for C++ use `ruff_cpp_spo`.
2. **Generate** `const` tables: `ClassMethods` (from `has_function`),
   `ClassActions` (from `has_function` → `ActionDef`, with `required_role` from
   your RBAC map and `exec` = your target, typically `SurrealQl`),
   `ValueSchema`/`FieldMask` (from `reads_field`), inheritance edges (from
   `inherits_from`).
3. **Bind classids** OGAR-side (a `to_node_row(classid, …)`-style entry; the
   manifest never mints a classid).
4. **Body adapters** — thin, `classid`-keyed, ASSUME the Core. A leaf method
   reads value tenants / edges, transforms, writes back through the gated path.
   It carries **no state of its own**.
5. **Route intrusive methods to hand-port** — a method that mutates a child
   collection from transient state (`_apply_grid`, `@api.onchange` buffers,
   matrix configurators) does **not** fit the adapter mold; raw hand-port it.
   Forcing it in is the Adapter-State-Leak the doctrine forbids.
6. **Commit through the gate** — never write to the external system directly;
   build an `ActionInvocation`, `commit` it with the actor + the MUL gate, and
   dispatch the `Committed` ones via `UnifiedStep`.

---

## 4. Iron rules for consumers

1. **Thin adapters that ASSUME the Core.** Identity = `classid`; state = value
   tenants; relations = `EdgeBlock`; composition = `classid → ClassView`;
   invocation = `UnifiedStep`. An adapter that needs state the SoA can't carry
   is a **Core gap → file a `ClassView` extension**, never an adapter hack.
2. **The harvest IS the manifest.** Don't hand-author the object model; generate
   the `const` tables from SPO. Don't let an adapter keep its own `@api.depends`
   table (that reinvents the ORM) — recompute dispatch is a `ClassView`
   capability (`compute_dag`, in progress), not adapter state.
3. **No parallel object model.** One OGAR Core; consumers are classid adapters
   into it.
4. **No model identifier** in any committed artifact; **no PII labels** leaking
   from the domain (leaf-rename at the adapter).
5. **Egress only through the commit gate** (RBAC + MUL) and `UnifiedStep`.

---

## 5. Status (what's CODED vs in-progress)

- **CODED:** `NodeGuid`/`classid`/`EdgeBlock`, `ClassView` trait + `RegistryClassView`,
  `ValueTenant`/`ValueSchema`, `codegen_manifest::{MethodSig, ClassMethods, methods_for}`,
  `orchestration::{UnifiedStep, OrchestrationBridge, StepDomain}`, and the DO arm
  `action::{ActionDef, ActionInvocation, ClassActions, actions_for, effective_actions}`
  with the RBAC+MUL commit gate (this PR).
- **In progress / named gaps:** the D-CLS field enumeration that auto-populates a
  `ClassView` field-set from a harvested model (lance-graph #534 landed the
  resolution keystone); the `ClassView::{compute_dag, constraints}` extension for
  computed-field recompute + validation dispatch; `PROBE-OGAR-ADAPTER-UNICHARSET`
  byte-parity (the licence to scale the adapter approach).

---

## 6. Minimal end-to-end (the AR→DO existence proof)

```rust
use lance_graph_contract::action::*;
use lance_graph_contract::kanban::ExecTarget;

// generated from sale_order's has_function row:
const SALE_ORDER: &[ActionDef] = &[ActionDef {
    predicate: "action_confirm", object_class: 0x0A1E_0001,
    exec: ExecTarget::SurrealQl, guard: Some(StateGuard { field: "state", value: "draft" }),
    required_role: Some("sales_manager"), overrides: None,
}];

let mut inv = ActionInvocation::pending(0x0A1E_0001, "action_confirm", /*instance*/ 42, /*cycle*/ 7, 1, 1);
let outcome = inv.commit(&SALE_ORDER[0], &actor /* holds sales_manager */, &GateDecision::Flow, now);
// outcome == ActionState::Committed → dispatch via UnifiedStep at ExecTarget::SurrealQl.
```
