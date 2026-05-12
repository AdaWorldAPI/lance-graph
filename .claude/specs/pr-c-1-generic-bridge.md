# PR-C-1: GenericBridge dispatching per-G ConsumerPointer

> **Pattern:** C — Generic Bridge dispatching ConsumerPointer
> **Tech debt:** TD-GENERIC-BRIDGE-3
> **Phase:** DESIGN — engineer-ready spec, no code yet
> **Worker:** W4 of Sprint-3 (12+meta CCA2A)
> **Predecessors required:** PR-B-1 (W3, ContextBundle + ConsumerPointer slot), PR-A-1 (W2, G-aware quad store)

## Goal

Replace the per-consumer newtype `MembraneGate` implementations (PR #29 `SmbMembraneGate`, PR #98 `MedCareMembraneGate`) with **one canonical `GenericBridge` impl** that reads a per-G `ConsumerPointer` from OGIT and dispatches to the correct policy. Per-consumer behavior becomes **data on `ConsumerPointer`** rather than a new type per consumer.

This closes TD-GENERIC-BRIDGE-3 and operationalizes the LOC-reduction promise of the unified OGIT-G architecture: adding the Nth consumer drops from ~800 LOC scaffolding (`MEDCARE_POLICY_GAP.md` baseline) to ~30 LOC of `ConsumerPointer` data — a ~25× reduction validated by W8's consumer template dry-run.

## Why this works (orphan rule dissolved)

The original justification for per-consumer newtype gates (PR #29 commentary) was the Rust **orphan rule**: `impl MembraneGate for X` had to live in a crate that owned either the trait or the type. By making `ConsumerPointer` a single struct in `lance-graph-contract` and `GenericBridge` a single impl in `lance-graph-callcenter`, the orphan rule is satisfied **once**. Per-consumer specialization moves to:

- A `ConsumerPointer` value (data, registered into OGIT)
- Optional `Arc<dyn RbacPolicy>` for policy logic that doesn't fit declarative slots
- Backwards-compat newtype wrappers for ergonomics + test surface preservation

## Files to touch

### NEW
- `crates/lance-graph-contract/src/consumer.rs` — `ConsumerPointer`, `ActionCap`, `GateDecision`, `DomainProfile`, `RbacPolicy` trait
- `crates/lance-graph-callcenter/src/generic_bridge.rs` — `GenericBridge` + `MembraneGate` impl

### MODIFY
- `crates/lance-graph-contract/src/lib.rs` — `pub mod consumer;` + re-exports
- `crates/lance-graph-callcenter/src/lib.rs` — `pub mod generic_bridge;` + re-export `GenericBridge`
- `crates/medcare-rs/crates/medcare-realtime/src/gate.rs` — `MedCareMembraneGate` → wrapper around `GenericBridge::for_g(MEDCARE_G)`
- `crates/smb-office-rs/crates/smb-realtime/src/gate.rs` — `SmbMembraneGate` → wrapper around `GenericBridge::for_g(SMB_G)`

## API sketch (~120 LOC contract + dispatcher; ~80 LOC wrappers)

### ConsumerPointer (in `lance-graph-contract::consumer`)

```rust
use std::sync::Arc;
use smallvec::SmallVec;
use smol_str::SmolStr;

/// Per-G data record describing how to gate, route, and transcode for a consumer.
/// Lives in the OGIT ContextBundle (see PR-B-1 / W3).
#[derive(Clone)]
pub struct ConsumerPointer {
    /// G-id (canonical OGIT G u32 from PR-A-1)
    pub g: u32,
    /// Human-readable domain label (e.g. "medcare", "smb_office")
    pub domain_name: SmolStr,
    /// Entity types this consumer claims (small inline set)
    pub entity_types: SmallVec<[u16; 16]>,
    /// Optional dynamic RBAC policy (escape hatch for non-declarative logic)
    pub rbac_policy_ref: Option<Arc<dyn RbacPolicy>>,
    /// Audit retention, fail-closed posture, escalation rules (declarative)
    pub stack_profile: DomainProfile,
    /// Optional schema overlay (G-local terminology mapping)
    pub schema_overlay: Option<SchemaPtr>,
    /// Action surface (closes Meta-3 HIGH #1 from medcare sprint)
    pub action_capabilities: SmallVec<[ActionCap; 8]>,
    /// Transcode kernels available to this consumer
    pub transcode_kernels: SmallVec<[KernelRef; 4]>,
}

#[derive(Clone)]
pub struct ActionCap {
    pub name: SmolStr,
    pub gate_decision: GateDecision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateDecision { Allow, Escalate, Deny }

#[derive(Clone, Debug)]
pub struct DomainProfile {
    pub fail_closed: bool,
    pub audit_retention_days: u32,
    pub escalation_path: Option<SmolStr>,
}

pub trait RbacPolicy: Send + Sync {
    fn should_emit(&self, commit: &Commit, ctx: &RequestContext) -> bool;
}
```

### GenericBridge (in `lance-graph-callcenter::generic_bridge`)

```rust
use std::sync::Arc;
use lance_graph_contract::consumer::{ConsumerPointer, GateDecision};

pub struct GenericBridge {
    registry: Arc<OntologyRegistry>,
    g: u32,
}

impl GenericBridge {
    pub fn for_g(registry: Arc<OntologyRegistry>, g: u32) -> Self {
        Self { registry, g }
    }

    fn pointer(&self) -> Option<ConsumerPointer> {
        self.registry.resolve(self.g)
            .and_then(|b| b.consumer_pointer.clone())
    }
}

impl MembraneGate for GenericBridge {
    fn should_emit(&self, commit: &Commit, ctx: &RequestContext) -> bool {
        let Some(p) = self.pointer() else {
            // Inert G (no consumer registered) → deny by default
            return false;
        };

        // 1. Fail-closed posture honored even before policy
        if p.stack_profile.fail_closed && ctx.is_degraded() {
            return false;
        }

        // 2. Action capability check (declarative path)
        if let Some(action) = ctx.action() {
            if let Some(cap) = p.action_capabilities.iter().find(|a| a.name == action) {
                match cap.gate_decision {
                    GateDecision::Deny     => return false,
                    GateDecision::Escalate => { /* fall through to policy */ }
                    GateDecision::Allow    => return true,
                }
            }
        }

        // 3. Dynamic policy hook (escape hatch)
        if let Some(policy) = &p.rbac_policy_ref {
            return policy.should_emit(commit, ctx);
        }

        // 4. No policy + no Allow capability → deny
        false
    }
}
```

### Backwards-compat wrappers

```rust
// crates/smb-office-rs/crates/smb-realtime/src/gate.rs
use lance_graph_callcenter::generic_bridge::GenericBridge;
use lance_graph_contract::ogit::SMB_G;

pub struct SmbMembraneGate(GenericBridge);

impl SmbMembraneGate {
    pub fn new(registry: Arc<OntologyRegistry>) -> Self {
        Self(GenericBridge::for_g(registry, SMB_G))
    }
}

impl MembraneGate for SmbMembraneGate {
    fn should_emit(&self, commit: &Commit, ctx: &RequestContext) -> bool {
        self.0.should_emit(commit, ctx)
    }
}
```

`MedCareMembraneGate` follows the identical pattern with `MEDCARE_G`.

## Test plan

| Test | File | Scope |
|---|---|---|
| Dispatch routes to correct policy | `tests/generic_bridge_dispatch.rs` | Register DOLCE bundle + Healthcare bundle; verify `for_g(DOLCE_G)` and `for_g(HEALTHCARE_G)` produce policy-distinct decisions |
| Inert G denies | `tests/generic_bridge_inert_g_denies.rs` | Bundle without `consumer_pointer` → `should_emit` returns `false` (fail-closed by default for unregistered G) |
| Fail-closed posture honored | `tests/generic_bridge_fail_closed.rs` | `DomainProfile { fail_closed: true }` + degraded `RequestContext` → deny without invoking policy |
| Action capability dispatch | `tests/generic_bridge_action_caps.rs` | `ActionCap { Allow }` → emit; `ActionCap { Deny }` → deny; `ActionCap { Escalate }` → falls through to `rbac_policy_ref` |
| SMB wrapper compat | `tests/smb_membrane_gate_compat.rs` | All 13 PR #29 SMB tests still green via wrapper |
| MedCare wrapper compat | `tests/medcare_membrane_gate_compat.rs` | All 33 PR #98 medcare regulatory + integration tests still green via wrapper |

**Net new:** 4 GenericBridge tests + 2 wrapper compat suites that re-run existing test bodies through the wrapper.

## Dependencies

- **PR-B-1 (W3)** must land first — `ContextBundle` carries the `consumer_pointer: Option<ConsumerPointer>` slot
- **PR-A-1 (W2)** must land first — `OntologyRegistry::resolve(g: u32)` requires the G-aware quad store
- Optional: PR-E-1 (W5) manifest can register `ConsumerPointer` from declarative YAML; this PR works without it

## Acceptance criteria

- [ ] `ConsumerPointer`, `ActionCap`, `GateDecision`, `DomainProfile`, `RbacPolicy` exported from `lance_graph_contract::consumer`
- [ ] `GenericBridge` impls `MembraneGate` and lives in `lance_graph_callcenter::generic_bridge`
- [ ] `SmbMembraneGate` is a thin wrapper around `GenericBridge::for_g(_, SMB_G)`
- [ ] `MedCareMembraneGate` is a thin wrapper around `GenericBridge::for_g(_, MEDCARE_G)`
- [ ] All 13 SMB tests stay green
- [ ] All 33 medcare regulatory + integration tests stay green
- [ ] 4 new GenericBridge tests added and green
- [ ] No new dependencies on per-consumer crates from `lance-graph-callcenter` (one-way: callcenter → contract only)

## Effort

Medium. ~200 LOC (120 contract + dispatcher, 80 wrappers + tests skeleton) + 4 new tests + 2 wrapper module rewrites. **~1–2 engineer-days**, predominantly mechanical once PR-A-1 + PR-B-1 are in.

## Open questions for engineer

1. **Wrappers — keep or deprecate?** The orphan rule was the original justification (PR #29). With `ConsumerPointer`-as-data + one canonical impl, the problem dissolves. But keeping wrappers as ergonomic aliases preserves existing call sites and test surfaces with zero churn. **Recommended: keep wrappers in this PR; mark `#[deprecated(note = "use GenericBridge::for_g")]` in a follow-up after one release cycle.**

2. **Action capabilities surface (Meta-3 HIGH #1 from medcare sprint).** `ConsumerPointer.action_capabilities` slot closes the "actions unreachable via gate" gap. The slot is data; the dispatcher reads it. This PR ships the slot + dispatch arm; populating it for medcare/smb is downstream work covered by their respective consumer specs.

3. **`RbacPolicy` trait location.** Two options:
   - `lance-graph-contract` — every consumer pulls the trait transitively via `ConsumerPointer`
   - `lance-graph-rbac` (new crate) — keeps contract minimal, requires consumers to take an extra dep

   **Recommended:** `lance-graph-contract::consumer` for this PR (one fewer crate to scaffold, trait is small). Split into `lance-graph-rbac` only if contract bloat becomes a measured problem.

4. **`SchemaPtr` and `KernelRef` types** — keep as opaque `Arc<dyn …>` or use concrete IDs into a registry? **Recommended:** `Arc<dyn …>` for symmetry with `RbacPolicy`; cheap to clone.

5. **Async policy.** `RbacPolicy::should_emit` is sync today (matches `MembraneGate`). If a downstream consumer needs async (e.g. external policy decision point), the trait needs a sibling `async_should_emit` or `MembraneGate` itself goes async. **Out of scope for this PR; flag for a follow-up RFC.**

## LOC reduction validation

Per `MEDCARE_POLICY_GAP.md`: a new consumer (medcare) cost ~800 LOC of scaffolding under the per-newtype pattern (gate impl + RBAC bindings + bridge wiring + tests for each).

Under `GenericBridge`:
- **~30 LOC** to register a `ConsumerPointer` for a new G
- **0 LOC** of new gate impl (the canonical one already exists)
- **Optional ~50 LOC** of `RbacPolicy` impl if declarative slots aren't enough

**Total: ~30–80 LOC vs ~800 LOC. ~10–25× reduction.** W8's consumer template spec dry-runs this against `hubspo-rs` and is the empirical validation.

## Cross-references

- `.claude/plans/ogit-g-context-bundle-v1.md` — D-OGIT-G-3 (architectural anchor)
- `.claude/board/TECH_DEBT.md` — TD-GENERIC-BRIDGE-3 (this PR closes it)
- `.claude/board/MEDCARE_POLICY_GAP.md` — the 800-LOC baseline this PR collapses
- `.claude/specs/pr-b-1-context-bundle.md` (W3) — required predecessor
- `.claude/specs/pr-a-1-spo-g-u32-slot.md` (W2) — required predecessor
- `.claude/specs/consumer-crate-template.md` (W8) — validates the LOC reduction empirically
- `.claude/specs/sprint-3-execution-plan.md` (W1) — master sequencing
- PR #29 (SMB membrane gate) — original newtype pattern this PR generalizes
- PR #98 (MedCare membrane gate) — second instance that motivated generalization

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wrapper compat suites miss a behavioral diff | Medium | High (regulatory tests) | Run wrapper test suites before deleting per-consumer impl bodies; keep both for one release |
| `Arc<dyn RbacPolicy>` clones cost too much in hot path | Low | Medium | Resolve `ConsumerPointer` once per request, not per commit; if still hot, intern by G-id |
| `OntologyRegistry::resolve` blocks the gate path | Medium | Medium | Resolution must be lock-free read (PR-A-1 acceptance covers this); add `criterion` benchmark in this PR |
| Inert-G "deny by default" surprises an existing caller | Low | Low | Document in `CHANGELOG.md`; the only path to inert G today is misconfiguration, which should fail closed anyway |
