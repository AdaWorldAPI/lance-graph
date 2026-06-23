# Integration Plan — ActionHandler ⟷ RBAC ⟷ Orchestration spine (v1)

> **Created:** 2026-06-23. **Status:** HARDENING (5+3 in progress).
> **Branch:** `claude/medcare-bridge-lance-graph-wmx76z`. **Target PR:** one, against `main`.
> **Workflow:** each file below is implemented by ONE Sonnet agent writing its new
> code **commented-out** (inert — the crate still compiles). The Opus orchestrator
> then reviews every draft, **uncomments + reconciles**, compiles/tests **once**
> centrally (cargo-hygiene: shared checkout, no per-agent worktrees, no per-agent
> builds), and commits the activated code to the PR.

## Why (the audit gaps this closes)

From the chain audit (`ActionHandler transcode ⟷ MARS ⟷ RBAC ⟷ orchestration`),
the lance-graph-side load-bearing gaps:

1. **RBAC §4 trait is partial** — `contract::rbac::ClassRbac` has only
   `actor_roles` + `grant_permits`; the keystone §4 `roles_reaching` /
   `row_scope` / `field_mask` (axes 2/3/4) are absent.
2. **`authorize()` is positive∧op-gate only** — no §5 stage-2 row-scope
   predicate, no projecting `Allow{scope, mask}`.
3. **commit ↔ authorize disconnected** — `ActionInvocation::commit` does an
   *inline* `actor.roles.contains(required_role)`; it never consults `ClassRbac`.
   Two RBAC surfaces, unconverged.
4. **No `impl ClassRbac for OgarClassView`** — keystone Q5's intended
   active-record impl is missing (only the reference `ClassGrants` exists).
5. **Orchestration not wired** — `OgarActionProvider` (DO surface) +
   `graph-flow-action` (executor) + `graph-flow-kanban` (lifecycle) compose only
   *in principle*; nothing runs `classid → effective_actions → ActionInvocation →
   Kanban cycle → authorized dispatch` end to end.

**Out of scope for THIS PR (deferred follow-up, OGAR repo):** minting the MARS
class family in `ogar_vocab::class_ids`, and implementing the `gen_statem →
ActionDef` behavioral lift (`ogar-from-elixir`'s three `todo!()`s). Those are a
parser implementation in a different repo under strict canon/shell discipline;
they gate the *MARS DO surface* but not the RBAC/orchestration spine, which this
PR completes generically (provider-agnostic) so MARS plugs in unchanged later.

## Non-negotiables (carried from the workspaces)

- **Backward compatibility:** extending `ClassRbac` MUST use **default methods**
  so the existing `ClassGrants` impl + the `PROBE-OGAR-RBAC-AUTHORIZE` gate keep
  compiling and passing **unchanged**. No breaking the green probe.
- **Contract stays zero-dep.** New types in `contract::rbac` use only contract
  types (`PrefetchDepth`, `FieldMask` if reused, no heap-heavy deps in hot types).
- **kgV invariant** (`I-ACTIONHANDLER-IS-KGV-NOT-CHOKEPOINT`): the orchestration
  integrator stays generic over a *provided* action manifest + a *provided*
  `ClassRbac` — it must NOT import a thinking style or wrap the SoA columns.
- **commit's existing semantics are sticky/tested** — the convergence is
  *additive* (a new `commit_*` entry point), never a silent change to `commit`.

## Files (one Sonnet agent each — code COMMENTED)

### F1 — `crates/lance-graph-contract/src/rbac.rs` (extend the §4 trait)
- Add types: `RoleSet` (a small owned set of `RoleId` — `Vec<RoleId>` newtype or
  `&[RoleId]`), `ScopeSpec` (compiled row-scope predicate handle: an opaque
  `{ tenant: Option<u64>, predicate_key: u32 }`-shape carrier — NOT a runtime
  domain; §3 axis-3), and reuse the existing `OpMask`/`ClassGrant`.
- Extend `trait ClassRbac` with **DEFAULT** methods:
  - `fn roles_reaching(&self, class: ClassId) -> RoleSet` (default: empty — the
    role-hierarchy fold is impl-provided; default keeps it positive-only).
  - `fn row_scope(&self, role: RoleId, class: ClassId) -> Option<ScopeSpec>`
    (default `None` — global, no row restriction).
  - `fn field_mask(&self, role: RoleId, class: ClassId) -> u64` (default
    `u64::MAX` — all fields; axis-4). (u64 mask to avoid pulling canonical_node's
    FieldMask if it widens the dep; agent confirms the lightest correct type.)
- Tests: defaults preserve today's behaviour; a typed impl exercises all four.

### F2 — `crates/lance-graph-rbac/src/authorize.rs` (§5 two-stage)
- Add `ScopedDecision { decision: AccessDecision, scope: Option<ScopeSpec>,
  field_mask: u64 }` and `authorize_scoped(rbac, actor, class, op) ->
  ScopedDecision`: stage-1 = the existing positive∧op-gate (REUSE `authorize`);
  stage-2 = AND the granting roles' `row_scope` (restrictive default-deny) and
  union their `field_mask`. `authorize()` stays as the collapsed compat path
  (unchanged signature, unchanged probe).
- Tests: a scoped impl yields scope+mask on Allow; deny short-circuits scope.

### F3 — `crates/lance-graph-contract/src/action.rs` (commit ↔ authorize)
- Add `ActionInvocation::commit_via<R: ClassRbac>(&mut self, def, rbac, actor_id,
  impact, guard_field_value, now) -> ActionState`: identical lifecycle to
  `commit`, but the **RBAC step resolves through `rbac.grant_permits(role, class,
  &Operation::Act{action: predicate})`** for each `rbac.actor_roles(actor_id)`
  instead of the inline `ActorContext.roles.contains`. `commit` is untouched
  (documented as the coarse ActorContext gate; `commit_via` is the ClassRbac
  convergence). Reuse `def.required_role` semantics: if `required_role` is set,
  the actor must hold a role whose grant permits the act on `def.object_class`.
- Tests: `commit_via` accepts an authorized actor, rejects an ungranted one,
  parity with `commit` on the ActorContext-equivalent case.

### F4 — `crates/lance-graph-ogar/src/rbac_impl.rs` (Q5 — `impl ClassRbac for OgarClassView`)
- New module `pub mod rbac_impl;` + `impl ClassRbac for OgarClassView` resolving:
  `actor_roles` (from a provided membership table — keep the impl table-backed,
  same shape as `ClassGrants`, since `project_role.granted` isn't in OGAR Core
  yet — documented as the bridge until §6 lands), `grant_permits` (via
  `contract::rbac::grants_permit` over a per-role `&[ClassGrant]`), §4 defaults.
- Verify in isolation (contract-only scratch like the actions module — the lib
  builds against fresh OGAR main now, so an in-crate test is fine).

### F5 — `crates/graph-flow-kanban/src/orchestrate.rs` (rs-graph-llm — the end-to-end spine)
- New module: `run_cycle(actions: &[ActionDef], rbac: &impl ClassRbac, actor_id,
  classid, predicate, gate, guard_value, now) -> CycleOutcome` that:
  resolve the `ActionDef` by `(classid, predicate)` from the provided manifest →
  build a `KanbanPlanEnvelope` → advance Planning→CognitiveWork on `Flow` →
  build an `ActionInvocation` → `commit_via(rbac, …)` at the CognitiveWork→commit
  boundary → map the `ActionState` onto a Kanban terminal (Committed→Commit,
  Pending→Plan, Cancelled/Failed→Prune) → return `{ outcome, envelope }`.
  Generic over the manifest + rbac (kgV: no provider/thinking import).
- Tests: authorized act drives the cycle to `Commit`; unauthorized → `Prune`;
  MUL `Hold` → `Plan` (re-deliberate).

## Sequencing / dependencies
- F1 is the type root (F2, F3, F4, F5 reference its §4 surface / commit_via).
- F3 depends on F1 (uses `ClassRbac`). F5 depends on F1+F3 (uses `commit_via`).
- F2 depends on F1. F4 depends on F1.
- Because all drafts are **commented**, agents may run fully in parallel; the
  Opus review reconciles cross-file type names during uncomment.

## HARDENING OUTCOME — corrected specs (v2, AUTHORITATIVE for the fleet)

> Supersedes the F1–F5 blocks above where they conflict. From 5+3 review
> (integration-lead, preflight-drift, convergence-architect, dto-soa-savant,
> core-first-architect + 3 reviewers pending). Every impl agent obeys THIS.

**F1 — `contract/src/rbac.rs`:** Reuse `contract::class_view::FieldMask` (NOT a
raw `u64` — it already exists, is zero-dep, and rbac's `PermissionSpec.projection`
already uses it). **Keep** the `ScopeSpec` newtype as the axis-3 row-scope token:
`pub struct ScopeSpec { tenant: Option<u64>, predicate_key: u32, deny: bool }`
(`Copy` POD, no interpreting methods; `tenant: None` = global, `deny: true` = the
empty scope) — a dedicated token rather than a bare `Option<TenantId>` because the
restrictive-AND fold needs a sound *conflict* value (two distinct tenants → empty,
not "self wins") and a reserved `predicate_key`. Drop the `RoleSet` newtype —
`roles_reaching(&self, class) -> &[RoleId]` (default `&[]`). All three are
**DEFAULT** methods (default `field_mask` = `FieldMask::FULL`, `row_scope` =
`None`, `roles_reaching` = `&[]`) so `ClassGrants` + the green
`PROBE-OGAR-RBAC-AUTHORIZE` compile/pass unchanged.

**F2 — `rbac/src/authorize.rs`:** `ScopedDecision { decision: AccessDecision,
scope: Option<ScopeSpec>, field_mask: FieldMask }` + `authorize_scoped(...)`. Stage-1
REUSES `authorize()`. `AccessDecision` has **THREE** variants — `Allow`,
`Deny{reason}`, `Escalate{reason}` — the stage-2 match MUST be exhaustive and
short-circuit scope on ANY non-`Allow`. The stage-2 fold intersects only
*concrete* `Some` row-scopes (a `None`/global role never narrows the fold; it
leaves the `None` sentinel intact), and `ScopeSpec::intersect` returns the empty
scope (`deny`) on a two-tenant conflict. `authorize()` stays byte-unchanged.

**F3 — `contract/src/action.rs`:** Add `commit_via<R: ClassRbac>(&mut self, def,
rbac: &R, actor_id: ActorId<'_>, impact, guard_field_value, now) -> ActionState`.
RBAC step: if `def.required_role.is_some()`, require `rbac.actor_roles(actor_id)`
to contain ≥1 role where `rbac.grant_permits(role, def.object_class,
&Operation::Act{action: def.predicate})`; if `required_role.is_none()` → proceed
(parity with `commit`). **PINNED semantics (integration-lead R1):** `commit_via`
has **NO `is_admin()` bypass** — admin must be a granted role (more auditable;
documented divergence from `commit`). `commit` is **untouched**. (OQ-CSV deferred:
convergence-architect's "make `commit` a forwarder over an `ActorRoleRbac`
adapter" — elegant but the adapter would need `def`; revisit post-PR.)

**F4 — `lance-graph-ogar/src/rbac_impl.rs`:** **DO NOT** `impl ClassRbac for
OgarClassView` — that is an **orphan-rule violation** (both trait + type foreign
to the crate) AND a Core-state-leak. Instead a **local newtype** with an
**injected grant source** (the §6 evaporation seam):
```rust
pub trait GrantSource {
    fn roles_of(&self, actor: ActorId<'_>) -> &[RoleId];
    fn grants_of(&self, role: RoleId) -> &[ClassGrant];
}
pub struct OgarRbac<S: GrantSource> { source: S }      // owns NO grant data
impl<S: GrantSource> ClassRbac for OgarRbac<S> { /* actor_roles/grant_permits via source; §4 defaults */ }
```
Body reads only from `source`, so when §6 `project_role.granted` lands the source
flips from a fixture to the tenant resolver with **zero body change** (the
"evaporation test"). File the §6 tenant work as a core-gap ticket in the PR body.
Verify in-crate (`lance-graph-ogar` builds against fresh OGAR main — confirmed).

**F5 — `/home/user/rs-graph-llm/graph-flow-kanban/src/orchestrate.rs`** (rs-graph-llm
workspace, cwd `/home/user/rs-graph-llm`, NOT lance-graph). **CONSUME** the
EXISTING `KanbanPlanEnvelope` (`::new`/`.advance`/`.try_transition` — do NOT
re-author it). Net-new: `run_cycle(actions: &[ActionDef], rbac: &impl ClassRbac,
actor_id, classid, predicate, gate, guard_value, now) -> CycleOutcome { outcome,
envelope }`. Resolve `ActionDef` by `(classid, predicate)` → drive the envelope
Planning→CognitiveWork on `Flow` → call **F6 `dispatch_via`** (not `commit_via`
directly — compose the executor, don't fork it) → map `ActionState` onto a Kanban
terminal. Stay generic (kgV: no thinking-style / no provider / no SoA column). One
test must feed a **provider-shaped** manifest (so the `def.object_class` match is
real, per truth-architect/integration-lead R3).

**F6 — `/home/user/rs-graph-llm/graph-flow-action/src/lib.rs`** (NEW, added by
hardening — M1): add `dispatch_via<H: ActionHandler, R: ClassRbac>(handler, rbac,
actor_id, gate, action, inv, guard_value, now) -> HandlerOutcome` mirroring
`dispatch` but routing the cold-floor through `inv.commit_via(action, rbac,
actor_id, gate, guard_value, now)` instead of `commit`. Closes the "executor still
on the old RBAC surface" half of gap #3. `dispatch` stays untouched.

### Canonical signatures (FINAL — 5+3 synthesized; every agent obeys verbatim)

```rust
// REUSE (never redefine): ClassId=u32, ActorId<'a>=&'a str, RoleId=&'static str,
//   Operation<'a>, OpMask, ClassGrant, grants_permit, ClassRbac (contract::rbac);
//   contract::class_view::FieldMask (FieldMask(pub u64); ::FULL, ::EMPTY, .has, .intersect, .inherit);
//   AccessDecision { Allow, Deny{reason:&'static str}, Escalate{reason:&'static str} } — FROZEN;
//   KanbanPlanEnvelope, KanbanColumn, ExecTarget (graph-flow-kanban + contract::kanban);
//   GateDecision (contract::mul), ActionDef, ActionInvocation, NodeGuid.

// F1 contract/src/rbac.rs — extend trait with DEFAULT methods (zero existing impl edited) + ScopeSpec
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScopeSpec { pub tenant: Option<u64>, pub predicate_key: u32, pub deny: bool } // Copy POD, NO interpreting methods; tenant None=global, deny true=empty scope; predicate_key reserved (0 = tenant-only). intersect: distinct tenants => DENY (never self-wins); deny absorbing.
// added to `trait ClassRbac` (ALL with default bodies):
//   fn roles_reaching(&self, _class: ClassId) -> &[RoleId] { &[] }                 // axis-2 hook, default empty (CONJECTURE: not impl'd this PR)
//   fn row_scope(&self, _role: RoleId, _class: ClassId) -> Option<ScopeSpec> { None } // axis-3, default global
//   fn field_mask(&self, _role: RoleId, _class: ClassId) -> FieldMask { FieldMask::FULL } // axis-4, reuse FieldMask

// F2 lance-graph-rbac/src/authorize.rs
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedDecision { pub decision: AccessDecision, pub scope: Option<ScopeSpec>, pub field_mask: FieldMask }
pub fn authorize_scoped(rbac:&impl ClassRbac, actor:ActorId<'_>, class:ClassId, op:Operation<'_>) -> ScopedDecision;
//   stage1 = authorize(...) UNCHANGED. non-Allow (Deny OR Escalate) => {decision, None, FieldMask::FULL}.
//   stage2 subset = actor_roles(actor).iter().filter(|r| grant_permits(r,class,&op))  [SAME predicate, NOT roles_reaching]
//     scope = fold row_scope by restrictive-AND; field_mask = fold field_mask by union (.inherit/.intersect-union).
//   AccessDecision FROZEN. authorize() byte-unchanged. probe stays green.

// F3 contract/src/action.rs — `commit` UNTOUCHED; add:
impl ActionInvocation { pub fn commit_via<R: ClassRbac>(&mut self, def:&ActionDef, rbac:&R,
    actor_id:ActorId<'_>, impact:&GateDecision, guard_field_value:Option<&str>, now_millis:u64) -> ActionState; }
//   order: def-match -> RBAC -> guard -> MUL (mirror commit). NO is_admin bypass.
//   RBAC: if let Some(_)=def.required_role { let ok = rbac.actor_roles(actor_id).iter()
//          .any(|&r| rbac.grant_permits(r, def.object_class, &Operation::Act{action: def.predicate}));
//          if !ok { self.state=Failed; return self.state; } }  // None required_role => proceed (parity)
//   resolve `ok` to a bool BEFORE mutating self.state. No unwrap/panic.

// F4 lance-graph-ogar/src/rbac_impl.rs — LOCAL newtype (NOT impl for OgarClassView; orphan E0117)
pub trait GrantSource { fn roles_of(&self, actor:ActorId<'_>) -> &[RoleId]; fn grants_of(&self, role:RoleId) -> &[ClassGrant]; }
pub struct OgarRbac<S: GrantSource> { pub source: S }   // owns NO grant data; source is the §6 evaporation seam
// impl ClassRbac for OgarRbac<S>: actor_roles->source.roles_of; grant_permits->grants_permit(source.grants_of(role),class,op); §4 defaults.
// ADD `pub mod rbac_impl;` to lance-graph-ogar/src/lib.rs.

// F5 rs-graph-llm/graph-flow-kanban/src/orchestrate.rs  (cwd /home/user/rs-graph-llm; Cargo.toml UNCHANGED, contract-only)
#[derive(Debug, Clone)] pub struct CycleOutcome { pub outcome: KanbanColumn, pub envelope: KanbanPlanEnvelope }
#[allow(clippy::too_many_arguments)]
pub fn run_cycle(actions:&[ActionDef], rbac:&impl ClassRbac, actor_id:ActorId<'_>, classid:u32,
   predicate:&'static str, gate:&GateDecision, object_instance:NodeGuid, guard_value:Option<&str>, now_millis:u64) -> CycleOutcome;
//   def = actions.iter().find(|a| a.object_class==classid && a.predicate==predicate) -> Option; None => outcome Prune (no unwrap).
//   env=KanbanPlanEnvelope::new(classid as MailboxId, def.exec); env.advance(gate) x2 (Planning->CognitiveWork->Evaluation);
//   inv=ActionInvocation::pending(classid,predicate,object_instance,env.cycle,0,0); st=inv.commit_via(def,rbac,actor_id,gate,guard_value,now_millis);
//   terminal = match st { Committed=>Commit, Pending=>Plan, Cancelled|Failed=>Prune }; env.try_transition(terminal); CycleOutcome{terminal, env}.
//   ADD `pub mod orchestrate;` to graph-flow-kanban/src/lib.rs.

// F6 rs-graph-llm/graph-flow-action/src/lib.rs — `dispatch` UNTOUCHED; add (executor-side convergence):
pub fn dispatch_via<H: ActionHandler, R: ClassRbac>(handler:&H, rbac:&R, actor_id:ActorId<'_>, gate:&GateDecision,
   action:&ActionDef, inv:&mut ActionInvocation, guard_field_value:Option<&str>, now_millis:u64) -> HandlerOutcome;
//   mirror `dispatch` but cold-floor via inv.commit_via(action, rbac, actor_id, gate, guard_field_value, now_millis).
//   needs `use lance_graph_contract::rbac::{ClassRbac, ActorId};` — contract-only, no new dep.
```

## Activation (Opus, after all agents)
1. Read each commented draft; reconcile type names + signatures across F1–F5.
2. Uncomment, `cargo fmt` + `cargo clippy` + `cargo test` per crate (contract,
   rbac, ogar via manifest, graph-flow-kanban) — once, centrally.
3. Board-hygiene: prepend INTEGRATION_PLANS.md + a PR_ARC entry in the SAME commit.
4. Open ONE PR; note the deferred OGAR MARS follow-up.
