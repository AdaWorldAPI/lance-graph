# PR-CE64-MB-6 — `SigmaTierRouter` + Σ-tier banding + plasticity + pruning + JIT KernelHandle

> **Sprint:** sprint-log-10 W7 (sigma-tier-router)
> **PR target:** PR-CE64-MB-6 (Wave 5 — highest-risk PR in the sprint)
> **Crate:** `crates/lance-graph-supervisor/` (extends shipped Tokio supervisor from PR #366 S7-W3)
> **LOC envelope:** ~1500 LOC + ~600 LOC tests/benches (per parent plan §7 — highest in sprint)
> **Risk:** **High** — new dispatcher replaces several ad-hoc paths; load-bearing for Σ10 Rubicon runtime
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 + §7 PR-CE64-MB-6 + §11 OQ-1 + §11 OQ-4
> **Composes:**
> - `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (W1) — `Mailbox<T>` trait + `InMemoryMailbox` + `TokioMailbox` + `SupabaseSubMailbox` backings (cycle-speed vs reflex-speed split)
> - `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) — `MailboxSoA<N>` + `SigmaTier` enum + `dispatch_cycle` + `CompartmentReport` (`drop_row` return)
> - `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` (W2) — `CausalEdge64` G/W/truth accessors emitted from compartments
> - `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` (W5) — ghost-edge persistence + `GhostReactivationEvent` (Σ9 EPIPHANY path)
> - Σ10 Rubicon doctrine (`.claude/knowledge/linguistic-epiphanies-2026-04-19.md` E21) — runtime dispatcher for the named Σ1-Σ10 tier architecture
> - `THINKING_ORCHESTRATION_WIRING.md` Gap 4 (Elevation Not Connected) — closed by this PR; Gap 3 (JIT pipeline) closes naturally at compartment-spawn site
> **Delta vs parent plan:** §6 names this dispatcher and §7 lists the PR scope ("SigmaTierRouter + Σ-tier banding policy + plasticity + pruning triggers + budget consumption"). This spec resolves it into: (a) `SigmaTierRouter` struct + state; (b) per-Σ-tier dispatch table + backing selection rules; (c) INT4-32D K-NN cold-start fallback wiring (OQ-4 resolution); (d) plasticity feedback into spawn-prior bias (Hebbian, per E-CE64-MB-10); (e) 3-trigger pruning policy (budget / XOR-cancel / outcome-sufficient); (f) `KernelHandle` consumption at compartment-spawn (closes JIT pipeline Gap 3); (g) Σ9-Σ10 EPIPHANY escalation to `CallcenterSupervisor` parent actor.

---

## §1 Scope statement

`SigmaTierRouter` is the **runtime dispatcher** that maps each compartment-spawn request to the correct mailbox backing by Σ-tier band, consumes `KernelHandle`s from the planner JIT cache, biases spawn priors via per-`(role, G)` plasticity counters, and prunes compartments via three triggers (budget / XOR-cancel / outcome-sufficient). It is the **single point of policy** for the four interlocking mechanisms named by parent plan §6:

1. **Σ-tier band → mailbox backing** (mechanism named in §6, no prior dispatcher).
2. **Plasticity-as-Hebbian-prior** (E-CE64-MB-10, no prior consumer).
3. **Pruning under budget pressure** (§6 names the 3 triggers, no prior policy).
4. **JIT KernelHandle pipeline closure** (Gap 3 in THINKING_ORCHESTRATION_WIRING.md, prior end was un-wired).

### Forbidden scope (explicitly out)

- **Replacing the existing `CallcenterSupervisor` ractor tree.** `SigmaTierRouter` is a child actor of `CallcenterSupervisor`. Existing one-for-one supervision + exponential backoff is untouched.
- **Inventing new `SigmaTier` enum variants** beyond what W6 defines. The 4-variant `SigmaTier` enum (`StaticReflex` / `Emergent` / `TwigBranching` / `EpiphanyEscalate`) owns the canonical taxonomy; W7 dispatches on those variants.
- **Modifying `MailboxSoA<N>`** layout. W7 calls the W6 surface (`push_row`, `dispatch_cycle`, `drop_row`); any layout drift escalates to W6 spec, not patched here.
- **Defining the INT4-32D codebook** — that's `.claude/plans/pr-j-1-int4-32d-atoms.md` scope. W7 only consumes the K-NN result via the documented `p64-bridge::STYLES` lookup.
- **Persisting plasticity counters cross-session.** Per-session ephemeral; cross-session aggregation deferred to sprint-12+ (one-line "TODO sprint-12: Lance dataset write").
- **Σ-tier banding policy ratification.** Banding is **proposed** here (Σ1-5 → Tokio, Σ6 → InMem reflex, Σ7-8 → InMem cycle-speed, Σ9-10 → escalate). User ratification is OQ-1 in parent plan §11 — required before sprint-11 Wave 5 spawn.

### In-scope deliverables

1. `crates/lance-graph-supervisor/src/sigma_tier_router.rs` — `SigmaTierRouter` actor + per-tier dispatch + spawn / prune / escalation.
2. `crates/lance-graph-supervisor/src/banding.rs` — Σ-tier → backing rules table + accessor.
3. `crates/lance-graph-supervisor/src/plasticity_aggregator.rs` — `(role, G)` plasticity rollup from `CompartmentReport` (W6 `drop_row` return).
4. `crates/lance-graph-supervisor/src/spawn_prior.rs` — Hebbian spawn-prior bias (read by router on `push_row`).
5. `crates/lance-graph-supervisor/src/pruning.rs` — three triggers (`BudgetExhausted` / `XorCancel` / `OutcomeSufficient`).
6. `crates/lance-graph-supervisor/src/escalation.rs` — Σ9-Σ10 → `CallcenterSupervisor` parent message route.
7. `crates/lance-graph-supervisor/src/coldstart.rs` — INT4-32D K-NN fallback path (OQ-4 wiring).
8. `crates/lance-graph-supervisor/src/lib.rs` — re-exports + child-spawn wiring under `CallcenterSupervisor`.
9. Tests: 10 banding-policy + 5 cold-start + 4 pruning + 3 escalation + 5 plasticity + 3 KernelHandle = **30 tests**.
10. Benches: 4 criterion (sigma_router_dispatch_latency, plasticity_aggregator_throughput, jit_kernelhandle_hit, kernel_handle_miss_then_compile).
11. CI: `lance-graph-supervisor-tests` job in `rust-test.yml` (path-filtered).

---

## §2 Crate layout

```
crates/lance-graph-supervisor/                        # extends shipped crate (PR #366)
├── Cargo.toml                                        # EDIT: +ractor (workspace) +par-tile (workspace)
├── src/
│   ├── lib.rs                                        # EDIT: re-export SigmaTierRouter
│   ├── callcenter_supervisor.rs                      # EXISTING (PR #366 S7-W3) — gains one child spawn
│   ├── sigma_tier_router.rs                          # NEW (~450 LOC) — actor + dispatch
│   ├── banding.rs                                    # NEW (~80  LOC) — Σ-tier→backing
│   ├── plasticity_aggregator.rs                      # NEW (~200 LOC) — (role,G) rollup
│   ├── spawn_prior.rs                                # NEW (~150 LOC) — Hebbian bias
│   ├── pruning.rs                                    # NEW (~180 LOC) — 3 triggers
│   ├── escalation.rs                                 # NEW (~150 LOC) — Σ9-10 → supervisor msg
│   └── coldstart.rs                                  # NEW (~120 LOC) — INT4-32D K-NN
├── tests/
│   ├── banding_policy.rs                             # NEW (~250 LOC, 10 tests)
│   ├── cold_start_k_nn.rs                            # NEW (~150 LOC, 5 tests)
│   ├── pruning_triggers.rs                           # NEW (~180 LOC, 4 tests)
│   ├── escalation.rs                                 # NEW (~120 LOC, 3 tests)
│   ├── plasticity_feedback.rs                        # NEW (~180 LOC, 5 tests)
│   └── kernel_handle_pipeline.rs                     # NEW (~150 LOC, 3 tests)
└── benches/
    └── sigma_router_bench.rs                         # NEW (~200 LOC, 4 criterion benches)
```


---

## §3 `SigmaTierRouter` — actor + state

`SigmaTierRouter` is a **ractor child actor** spawned by `CallcenterSupervisor` (one per tenant scope, matching the supervisor's tenant-scoped tree shape). It owns no `MailboxSoA<N>` directly — it dispatches **to** instances supplied by callers. This separates **policy** (router) from **mechanism** (SoA), per parent plan §6.

### §3.1 Type definition

```rust
// crates/lance-graph-supervisor/src/sigma_tier_router.rs

use ractor::{Actor, ActorRef, ActorProcessingErr};
use par_tile::mailbox_soa::{MailboxSoA, MailboxId, SigmaTier, RoleId, TemporalWindow, CompartmentReport};
use par_tile::{AttentionMaskActor, BindSpaceView, CausalEdge64};
use lance_graph_contract::jit::KernelHandle;
use crate::banding::Banding;
use crate::plasticity_aggregator::PlasticityAggregator;
use crate::spawn_prior::SpawnPriorBias;
use crate::pruning::PruningPolicy;
use crate::escalation::EscalationSink;
use crate::coldstart::ColdStartFallback;

pub struct SigmaTierRouter;

pub struct SigmaTierRouterState {
    /// Reference to the AttentionMaskActor (shared rename table for this tenant).
    pub mask:          ActorRef<par_tile::AttentionMaskActorMsg>,

    /// Parent supervisor for Σ9-Σ10 EPIPHANY escalation.
    pub supervisor:    ActorRef<crate::callcenter_supervisor::CallcenterSupervisorMsg>,

    /// Banding rules (mutable for runtime adjustment post-OQ-1 ratification).
    pub banding:       Banding,

    /// Per-(role, G_slot) plasticity rollup, drains on each `drop_row` via CompartmentReport.
    pub plasticity:    PlasticityAggregator,

    /// Hebbian spawn-prior bias (reads plasticity, writes spawn-prior table).
    pub spawn_prior:   SpawnPriorBias,

    /// Pruning policy (3 triggers).
    pub pruning:       PruningPolicy,

    /// Cold-start K-NN fallback (INT4-32D codebook lookup).
    pub cold_start:    ColdStartFallback,

    /// JIT KernelHandle cache (mirror of planner's cache; populated lazily on first spawn).
    pub kernel_cache:  KernelHandleCache,

    /// Tenant-scoped MailboxSoA capacity (one cap per tier band; informational, MailboxSoA is owned by caller).
    pub capacity:      TierCapacities,
}

#[derive(Copy, Clone, Debug)]
pub struct TierCapacities {
    pub static_reflex:     usize,   // Σ1-5: typically ≤ 32  (Tokio backing, ms-scale)
    pub emergent:          usize,   // Σ6:   typically ≤ 64  (InMem, Pearl 2-3)
    pub twig_branching:    usize,   // Σ7-8: typically ≤ 256 (InMem cycle-speed, 200 ns)
    pub epiphany_escalate: usize,   // Σ9-10: typically ≤ 16 (escalation queue depth)
}

impl Default for TierCapacities {
    fn default() -> Self {
        Self { static_reflex: 32, emergent: 64, twig_branching: 256, epiphany_escalate: 16 }
    }
}
```

### §3.2 Message protocol

```rust
pub enum SigmaTierRouterMsg {
    /// External spawn request from Zone-2 (callcenter) or Zone-1 (in-cycle epiphany).
    Spawn {
        role:            RoleId,
        sigma:           SigmaTier,
        temporal_window: TemporalWindow,
        bindspace_view:  BindSpaceView<'static>,
        intent:          Option<par_tile::ConsumerHandle>,
        style_id:        Option<par_tile::StyleId>,    // None → cold-start K-NN
        reply:           tokio::sync::oneshot::Sender<Result<SpawnReply, SpawnError>>,
    },

    /// Per-cycle dispatch — drives all active compartments under this router.
    DispatchCycle {
        soa_ref:         ActorRef<MailboxSoAMsg>,
        cycle:           u32,
    },

    /// Plasticity report (from MailboxSoA `drop_row` flowing back to router).
    PlasticityReport {
        report:          CompartmentReport,
    },

    /// EPIPHANY-tier witness escalation (incoming from compartment emission at Σ7-8 that
    /// crossed the Σ9 surprise threshold).
    EscalateEpiphany {
        edge:            CausalEdge64,
        from_mailbox:    MailboxId,
    },

    /// Manual prune (force-drop a compartment, e.g. on XorCancel detection upstream).
    Prune {
        id:              MailboxId,
        reason:          crate::pruning::PruneReason,
    },

    /// OQ-1 ratification hook: replace the banding table at runtime.
    SetBanding { banding: Banding },
}

pub struct SpawnReply {
    pub mailbox_id:      MailboxId,
    pub kernel_handle:   Option<KernelHandle>,
    pub resolved_style:  par_tile::StyleId,    // populated even when caller passed None
    pub backing:         BackingKind,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BackingKind { Tokio, InMemEmergent, InMemCycleSpeed, EscalateOnly }

#[derive(Debug)]
pub enum SpawnError {
    CapacityExhausted { tier: SigmaTier },
    ColdStartFailed,
    KernelCompileFailed { reason: String },
    BindFailure { reason: String },
}
```

### §3.3 Actor impl skeleton

```rust
#[ractor::async_trait]
impl Actor for SigmaTierRouter {
    type Msg   = SigmaTierRouterMsg;
    type State = SigmaTierRouterState;
    type Arguments = SigmaTierRouterArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args:    Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(SigmaTierRouterState {
            mask:         args.mask,
            supervisor:   args.supervisor,
            banding:      Banding::default(),    // Σ1-5 Tokio / Σ6 InMem / Σ7-8 cycle / Σ9-10 escalate
            plasticity:   PlasticityAggregator::new(),
            spawn_prior:  SpawnPriorBias::new(),
            pruning:      PruningPolicy::default(),
            cold_start:   ColdStartFallback::new()?,
            kernel_cache: KernelHandleCache::new(),
            capacity:     TierCapacities::default(),
        })
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        msg:    Self::Msg,
        state:  &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            SigmaTierRouterMsg::Spawn { role, sigma, temporal_window, bindspace_view, intent, style_id, reply } => {
                let res = handle_spawn(state, role, sigma, temporal_window, bindspace_view, intent, style_id).await;
                let _ = reply.send(res);
            }
            SigmaTierRouterMsg::DispatchCycle { soa_ref, cycle } => {
                drive_dispatch_cycle(state, soa_ref, cycle).await?;
            }
            SigmaTierRouterMsg::PlasticityReport { report } => {
                state.plasticity.absorb(report);
                state.spawn_prior.refresh_from(&state.plasticity);
            }
            SigmaTierRouterMsg::EscalateEpiphany { edge, from_mailbox } => {
                crate::escalation::escalate(state, edge, from_mailbox).await?;
            }
            SigmaTierRouterMsg::Prune { id, reason } => {
                state.pruning.queue_drop(id, reason);
            }
            SigmaTierRouterMsg::SetBanding { banding } => {
                state.banding = banding;   // OQ-1 ratification path
            }
        }
        Ok(())
    }
}
```


---

## §4 Banding policy — Σ1-Σ10 → mailbox backing

**Per parent plan §6 + §11 OQ-1 tentative resolution.** User ratification required before sprint-11 Wave 5 (see §13 OQs below).

| Σ tier | Tier name (E21) | Cycle budget | Mailbox backing | Rationale |
|---|---|---|---|---|
| Σ1 | STATIC | ms-scale | `TokioMailbox` (Zone-2 reflex) | Already shipped reflex shape; PR #366 supervises |
| Σ2 | repair-reflex | ms-scale | `TokioMailbox` | Same family as Σ1 |
| Σ3 | repair-emergent | ms-scale | `TokioMailbox` | — |
| Σ4 | minor-reflex | ms-scale | `TokioMailbox` | OQ-1 candidate for promotion to cycle-speed |
| Σ5 | minor-emergent | ms-scale | `TokioMailbox` | OQ-1 candidate for promotion to cycle-speed |
| Σ6 | EMERGENT (Pearl 2-3) | 1-10 µs | `InMemoryMailbox` | Crossbeam MPSC; not cycle-speed (Pearl 2-3 latency) |
| Σ7 | TWIG branching A | 200 ns | `InMemoryMailbox` cycle-speed | std::sync::mpsc; 200 ns p99 W6 spec target |
| Σ8 | TWIG branching B | 200 ns | `InMemoryMailbox` cycle-speed | Same |
| Σ9 | EPIPHANY (Pearl 5) | escalate | `EscalateOnly` (no own backing) | Routes to `CallcenterSupervisor` parent; Σ9 messages travel via supervisor msg, not router-owned mailbox |
| Σ10 | RUBICON | escalate | `EscalateOnly` | Same; Σ10 hits external commit + AriGraph + Wire DTO egress |

### §4.1 `Banding` struct

```rust
// crates/lance-graph-supervisor/src/banding.rs

#[derive(Clone, Debug)]
pub struct Banding {
    table: [BackingKind; 10],   // index = Σ-tier (1..=10) - 1
}

impl Banding {
    /// Default per parent plan §11 OQ-1 tentative resolution.
    pub fn default_tentative() -> Self {
        Self {
            table: [
                BackingKind::Tokio,            // Σ1
                BackingKind::Tokio,            // Σ2
                BackingKind::Tokio,            // Σ3
                BackingKind::Tokio,            // Σ4 (OQ-1 candidate to promote)
                BackingKind::Tokio,            // Σ5 (OQ-1 candidate to promote)
                BackingKind::InMemEmergent,    // Σ6
                BackingKind::InMemCycleSpeed,  // Σ7
                BackingKind::InMemCycleSpeed,  // Σ8
                BackingKind::EscalateOnly,     // Σ9
                BackingKind::EscalateOnly,     // Σ10
            ],
        }
    }

    /// Post-ratification alternative: Σ4-Σ5 promoted to cycle-speed reflex.
    pub fn alternative_fast_reflex() -> Self {
        let mut t = Self::default_tentative();
        t.table[3] = BackingKind::InMemCycleSpeed;   // Σ4
        t.table[4] = BackingKind::InMemCycleSpeed;   // Σ5
        t
    }

    pub fn lookup(&self, tier: u8) -> BackingKind {
        debug_assert!((1..=10).contains(&tier));
        self.table[(tier - 1) as usize]
    }
}

impl Default for Banding {
    fn default() -> Self { Self::default_tentative() }
}
```

### §4.2 `SigmaTier` (W6) → `u8` mapping helper

W6 ships a 4-variant enum (`StaticReflex` / `Emergent` / `TwigBranching` / `EpiphanyEscalate`) — coarser than the 10-band table. The router needs an explicit numeric Σ-tier on `Spawn`; the resolver:

```rust
// crates/lance-graph-supervisor/src/banding.rs

impl Banding {
    /// W6's coarse SigmaTier → fine-grained u8 tier.
    /// Caller passes the numeric tier (1..=10) via SpawnArgs; this helper validates compat
    /// with the W6 coarse band:
    ///   StaticReflex     ↔ 1..=5
    ///   Emergent         ↔ 6
    ///   TwigBranching    ↔ 7..=8
    ///   EpiphanyEscalate ↔ 9..=10
    pub fn validate_compat(coarse: SigmaTier, fine: u8) -> bool {
        match (coarse, fine) {
            (SigmaTier::StaticReflex,     1..=5)  => true,
            (SigmaTier::Emergent,         6)      => true,
            (SigmaTier::TwigBranching,    7..=8)  => true,
            (SigmaTier::EpiphanyEscalate, 9..=10) => true,
            _ => false,
        }
    }
}
```

**Note:** the actual numeric Σ-tier travels in `SpawnArgs.sigma_fine: u8`; W6's `SigmaTier` enum stays on the `MailboxSoA` row as the coarse band. The router stores both; assertions fire on coarse-fine drift (config error → SpawnError::BindFailure).

---

## §5 Cold-start K-NN fallback (INT4-32D)

**Resolves parent plan OQ-4** (INT4-32D wiring point). When a `Spawn` request arrives with `style_id = None` (no architectural style binding known for this `(role, G_slot)` pair), the router falls back to K-NN over `p64-bridge::STYLES` codebook (16 bytes per `ThinkingAtom32x4`). This is the cold-start safety net Pattern G was deferred for.

### §5.1 `ColdStartFallback`

```rust
// crates/lance-graph-supervisor/src/coldstart.rs

use lance_graph_contract::thinking::ThinkingStyle;
use p64_bridge::STYLES;

pub struct ColdStartFallback {
    /// K-NN codebook (loaded lazily; INT4-32D atoms = 16 B each).
    codebook:    Vec<p64_bridge::ThinkingAtom32x4>,

    /// Cache: (RoleId, G_slot) → resolved StyleId (avoids re-running K-NN every spawn).
    cache:       std::collections::HashMap<(RoleId, u8), par_tile::StyleId>,

    /// K parameter for K-NN; default 1 (nearest neighbor).
    k:           usize,
}

impl ColdStartFallback {
    pub fn new() -> Result<Self, SpawnError> {
        let codebook = p64_bridge::STYLES.to_vec();   // ~16 B × style count
        Ok(Self { codebook, cache: Default::default(), k: 1 })
    }

    /// K-NN lookup. Bias = situation features (role embedding + G_slot fingerprint).
    pub fn resolve(
        &mut self,
        role:    RoleId,
        g_slot:  u8,
        features: &SituationFeatures,
    ) -> Result<par_tile::StyleId, SpawnError> {
        if let Some(cached) = self.cache.get(&(role, g_slot)) {
            return Ok(*cached);
        }
        let query = features.to_int4_32d();   // pack situation to 16 B
        let nearest = self.codebook.iter()
            .min_by_key(|atom| atom.hamming_distance(&query))
            .ok_or(SpawnError::ColdStartFailed)?;
        let resolved = par_tile::StyleId(nearest.style_id);
        self.cache.insert((role, g_slot), resolved);
        Ok(resolved)
    }
}

pub struct SituationFeatures {
    pub role_id:        RoleId,
    pub g_slot:         u8,
    pub bindspace_hash: u64,    // hash of relevant BindSpace columns at spawn time
}

impl SituationFeatures {
    fn to_int4_32d(&self) -> p64_bridge::ThinkingAtom32x4 {
        // Pack (role_id, g_slot, bindspace_hash) into 16 B INT4-32D atom.
        // Bit-layout per pr-j-1-int4-32d-atoms.md §3.
        p64_bridge::ThinkingAtom32x4::pack(
            self.role_id.0 as u32,
            self.g_slot as u32,
            self.bindspace_hash,
        )
    }
}
```

**Cache invalidation:** the `(role, g_slot)` cache is **session-ephemeral**. On `AttentionMaskActor` eviction of a `G_slot`, the cache entries keyed on that slot become stale. The router subscribes to `AttentionMask` eviction broadcasts and drops stale cache rows on receipt.

### §5.2 Spawn integration

In `handle_spawn`:

```rust
async fn handle_spawn(
    state:           &mut SigmaTierRouterState,
    role:            RoleId,
    sigma_coarse:    SigmaTier,
    temporal_window: TemporalWindow,
    bindspace_view:  BindSpaceView<'static>,
    intent:          Option<par_tile::ConsumerHandle>,
    style_id_opt:    Option<par_tile::StyleId>,
) -> Result<SpawnReply, SpawnError> {
    // Step 1: G_slot via AttentionMaskActor.
    let g_slot = bind_g_via_mask(&state.mask, &bindspace_view).await?;

    // Step 2: resolve style (cold-start K-NN if None).
    let style_id = match style_id_opt {
        Some(sid) => sid,
        None => {
            let features = SituationFeatures {
                role_id: role,
                g_slot,
                bindspace_hash: bindspace_view.content_hash(),
            };
            state.cold_start.resolve(role, g_slot, &features)?
        }
    };

    // Step 3: get-or-compile KernelHandle (closes Gap 3).
    let kernel = state.kernel_cache.get_or_compile(style_id).await?;

    // Step 4: backing selection via banding table (numeric tier = coarse-resolve).
    let fine_tier = coarse_to_fine_default(sigma_coarse);   // helper
    let backing = state.banding.lookup(fine_tier);

    // Step 5: capacity check.
    if state.capacity_exceeded(sigma_coarse) {
        return Err(SpawnError::CapacityExhausted { tier: sigma_coarse });
    }

    // Step 6: spawn-prior bias check (Hebbian); may upgrade SigmaTier on hot (role,G).
    let _bias = state.spawn_prior.lookup(role, g_slot);
    // (Bias is informational here; full upgrade-on-bias logic lives in §6.)

    // Step 7: push to caller-owned MailboxSoA (router dispatches; doesn't own SoA).
    let mailbox_id = push_row_via_caller_soa(role, sigma_coarse, temporal_window, bindspace_view, intent).await?;

    Ok(SpawnReply { mailbox_id, kernel_handle: Some(kernel), resolved_style: style_id, backing })
}
```


---

## §6 Plasticity feedback — Hebbian spawn-prior bias (E-CE64-MB-10)

**Closes E-CE64-MB-10** ("Plasticity emerges naturally from the bit-counter on `MailboxSoA::plasticity_counters`"). On every `drop_row`, the `CompartmentReport` flows back to the router; the router aggregates plasticity counters by `(role, G_slot)` and biases next-spawn priors toward high-count pairings ("fired together, wired together"). Counterfactual ghosts emit at low-count slots (synaptic pruning).

### §6.1 `PlasticityAggregator`

```rust
// crates/lance-graph-supervisor/src/plasticity_aggregator.rs

use std::collections::HashMap;
use par_tile::mailbox_soa::{CompartmentReport, RoleId, PlasticityCounter};

pub struct PlasticityAggregator {
    /// (role, G_slot) → running plasticity total (saturating u64).
    counters: HashMap<(RoleId, u8), u64>,

    /// Total reports absorbed (telemetry).
    n_reports: u64,
}

impl PlasticityAggregator {
    pub fn new() -> Self { Self { counters: Default::default(), n_reports: 0 } }

    /// Called on every PlasticityReport.
    pub fn absorb(&mut self, report: CompartmentReport) {
        let key = (report.role, report.g_slot_at_drop);   // requires W6 to expose g_slot_at_drop
        let total = self.counters.entry(key).or_insert(0);
        *total = total.saturating_add(report.plasticity.0);
        self.n_reports += 1;
    }

    pub fn lookup(&self, role: RoleId, g_slot: u8) -> u64 {
        self.counters.get(&(role, g_slot)).copied().unwrap_or(0)
    }

    /// Top-K hot (role, G) pairs — used by spawn-prior bias.
    pub fn top_k(&self, k: usize) -> Vec<((RoleId, u8), u64)> {
        let mut v: Vec<_> = self.counters.iter().map(|(k, v)| (*k, *v)).collect();
        v.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        v.truncate(k);
        v
    }
}
```

### §6.2 `SpawnPriorBias`

```rust
// crates/lance-graph-supervisor/src/spawn_prior.rs

pub struct SpawnPriorBias {
    /// (role, G_slot) → bias multiplier (0.0 = avoid, 1.0 = neutral, > 1.0 = prefer).
    bias: HashMap<(RoleId, u8), f32>,

    /// Refresh epoch (incremented every N PlasticityReports).
    refresh_epoch: u64,
}

impl SpawnPriorBias {
    pub fn new() -> Self { Self { bias: Default::default(), refresh_epoch: 0 } }

    /// Recompute bias table from plasticity rollup. Cheap (O(K) for top-K).
    pub fn refresh_from(&mut self, agg: &PlasticityAggregator) {
        let top = agg.top_k(64);   // top-64 hot pairs get prefer-bias
        self.bias.clear();
        let max_count = top.first().map(|(_, c)| *c).unwrap_or(1) as f32;
        for ((role, g), count) in top {
            let normalized = (count as f32) / max_count;       // 0.0..1.0
            let bias = 1.0 + normalized;                       // 1.0..2.0
            self.bias.insert((role, g), bias);
        }
        self.refresh_epoch += 1;
    }

    pub fn lookup(&self, role: RoleId, g_slot: u8) -> f32 {
        self.bias.get(&(role, g_slot)).copied().unwrap_or(1.0)   // neutral default
    }
}
```

**Bias consumption** (called from `handle_spawn` step 6):

If `bias > 1.5` AND the caller's `sigma_coarse == Emergent`, the router **may upgrade** to `TwigBranching` (cycle-speed) — proven hot pair, justified for faster mailbox backing. Threshold (1.5) is hand-tuned per `I-NOISE-FLOOR-JIRAK`; principled Jirak-derived threshold deferred to sprint-12+ (see W7-OQ-3).

---

## §7 Pruning policy — 3 triggers

**Per parent plan §6 + §11.** The router does NOT prune **mid-cycle**; it queues drop-IDs and the next `DispatchCycle` consumer calls `MailboxSoA::drop_row` on each queued ID. This keeps the dispatch-cycle hot path branchless.

### §7.1 `PruningPolicy`

```rust
// crates/lance-graph-supervisor/src/pruning.rs

use par_tile::mailbox_soa::MailboxId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PruneReason {
    /// Trigger 1: budget exhausted (`Budget == 0`). Detected by MailboxSoA itself in
    /// dispatch_cycle (W6 spec §4.2); router stays informed via PlasticityReport on drop.
    BudgetExhausted,

    /// Trigger 2: XOR-cancel — compartment's CausalEdge64 emission cancels
    /// against an existing AriGraph edge (incoming-and-outgoing pair, opposite signs).
    /// Detected post-emission in CollapseGate; router receives Prune msg.
    XorCancel,

    /// Trigger 3: outcome-sufficient — the compartment achieved its temporal-window goal
    /// (style-specific success metric, e.g. `FreeEnergy < homeostasis_floor`). Detected
    /// by the dispatcher_fn in dispatch_cycle; router receives Prune msg.
    OutcomeSufficient,
}

pub struct PruningPolicy {
    /// Queued IDs to drop on next dispatch cycle (router does not call drop_row directly).
    queue: Vec<(MailboxId, PruneReason)>,

    /// Telemetry: count of prunes by reason.
    counts: [u64; 3],
}

impl PruningPolicy {
    pub fn queue_drop(&mut self, id: MailboxId, reason: PruneReason) {
        self.queue.push((id, reason));
        self.counts[reason as usize] += 1;
    }

    pub fn drain(&mut self) -> Vec<(MailboxId, PruneReason)> {
        std::mem::take(&mut self.queue)
    }
}

impl Default for PruningPolicy {
    fn default() -> Self { Self { queue: Vec::new(), counts: [0; 3] } }
}
```

### §7.2 Consumption in `DispatchCycle`

```rust
async fn drive_dispatch_cycle(
    state:   &mut SigmaTierRouterState,
    soa_ref: ActorRef<MailboxSoAMsg>,
    cycle:   u32,
) -> Result<(), ActorProcessingErr> {
    // Drain pruning queue first.
    let to_drop = state.pruning.drain();
    for (id, _reason) in to_drop {
        soa_ref.cast(MailboxSoAMsg::DropRow { id }).map_err(actor_err)?;
    }

    // Tick AttentionMaskActor once per cycle (W6 spec §2 invariant).
    state.mask.cast(par_tile::AttentionMaskMsg::Tick).map_err(actor_err)?;

    // DispatchCycle inner call routes to MailboxSoA::dispatch_cycle.
    soa_ref.cast(MailboxSoAMsg::DispatchCycle { cycle }).map_err(actor_err)?;

    Ok(())
}
```

---

## §8 KernelHandle JIT pipeline — closes Gap 3

**Per `THINKING_ORCHESTRATION_WIRING.md` Gap 3** ("JIT pipeline end-to-end"). The router's `kernel_cache` is the missing consumer of `lance-graph-planner::strategy::jit_compile::JitCompiler`. On `Spawn`, the router fetches-or-compiles a `KernelHandle` for the resolved `StyleId`; subsequent spawns for the same style return the cached handle.

### §8.1 `KernelHandleCache`

```rust
// crates/lance-graph-supervisor/src/sigma_tier_router.rs

use lance_graph_contract::jit::{JitCompiler, KernelHandle};

pub struct KernelHandleCache {
    cache: HashMap<par_tile::StyleId, KernelHandle>,
    /// Hits / misses for telemetry.
    hits:    u64,
    misses:  u64,
}

impl KernelHandleCache {
    pub fn new() -> Self { Self { cache: Default::default(), hits: 0, misses: 0 } }

    pub async fn get_or_compile(
        &mut self,
        style_id: par_tile::StyleId,
    ) -> Result<KernelHandle, SpawnError> {
        if let Some(handle) = self.cache.get(&style_id) {
            self.hits += 1;
            return Ok(handle.clone());
        }
        self.misses += 1;
        // Defer compilation to lance-graph-planner's JitCompiler trait.
        // The compiler reads YAML descriptor at `crewai-rust/agents/<style>.yaml`,
        // produces FieldModulation → ScanParams → JitTemplate → Cranelift IR → KernelHandle.
        let compiler = lance_graph_planner::strategy::jit_compile::get_jit_compiler();
        let handle = compiler.compile_for_style(style_id.into())
            .await
            .map_err(|e| SpawnError::KernelCompileFailed { reason: e.to_string() })?;
        self.cache.insert(style_id, handle.clone());
        Ok(handle)
    }
}
```

**Cross-crate dep:** `lance-graph-supervisor` gains a build-dep on `lance-graph-planner` for `get_jit_compiler()`. This is the **first time supervisor → planner is a hard build dep**; previously planner depended on supervisor through trait imports only. If the dep direction creates a cycle with the existing planner-uses-supervisor-for-Hebbian-feedback path (unlikely; that goes through Arrow-serialized Lance log), the router falls back to **JitCompiler trait object** stored in `SigmaTierRouterState` and injected at `pre_start`.

### §8.2 Compile telemetry

`KernelHandleCache::stats()` returns `(hits, misses, hit_rate)`; surfaced via `SigmaTierRouterMsg::Stats` for runtime observation. Bench `jit_kernelhandle_hit` validates hit-rate > 95% after 1K spawns over 10 distinct styles (10% miss = cold start only).

---

## §9 Σ9-Σ10 EPIPHANY escalation

**Per parent plan §10 + §13 OQ-9.** When a compartment at Σ7-Σ8 emits a witness whose Pearl rung crosses 5 (EPIPHANY-tier), the router routes the `CausalEdge64` to its parent `CallcenterSupervisor` actor — NOT through the cycle-speed mailbox, but through the **ractor supervisor message channel** (Tokio shape). This is the bridge from Zone-1 (in-cycle) to Zone-2 (callcenter), where cross-tenant MUL gate + AriGraph commit + optional Wire DTO egress happens.

### §9.1 `EscalationSink`

```rust
// crates/lance-graph-supervisor/src/escalation.rs

use ractor::ActorRef;
use par_tile::CausalEdge64;
use par_tile::mailbox_soa::MailboxId;
use crate::callcenter_supervisor::CallcenterSupervisorMsg;

pub async fn escalate(
    state:        &SigmaTierRouterState,
    edge:         CausalEdge64,
    from_mailbox: MailboxId,
) -> Result<(), ActorProcessingErr> {
    // Σ9 vs Σ10 split:
    //   Σ9: send EpiphanyWitness to supervisor; supervisor commits to AriGraph SPO-G.
    //   Σ10: send RubiconWitness to supervisor; supervisor additionally fires Wire DTO egress.
    let pearl = edge.pearl_rung();
    let msg = if pearl >= 7 {
        CallcenterSupervisorMsg::RubiconWitness { edge, from_mailbox }
    } else {
        CallcenterSupervisorMsg::EpiphanyWitness { edge, from_mailbox }
    };
    state.supervisor.cast(msg).map_err(actor_err)?;
    Ok(())
}
```

**Note on `CallcenterSupervisorMsg` extension:** PR #366 shipped the supervisor without `EpiphanyWitness` / `RubiconWitness` variants. **This PR adds them** (~15 LOC in `callcenter_supervisor.rs`). The receive-side handler is a small fan-out to AriGraph commit (existing) + (rubicon-only) Wire DTO egress (existing). No new supervisor logic — just two new message variants and the dispatch.

### §9.2 Backpressure under EPIPHANY flood

If Σ9-Σ10 messages saturate the supervisor mailbox, the router **drops to backpressure mode**: queues escalations with a 1024-entry bounded `VecDeque<EpiphanyQueueEntry>`; if the queue is full, the oldest is dropped and a `TECH_DEBT.md` entry is appended via `AGENT_LOG`. This is intentional **load-shedding** — better to drop a Σ9 escalation than to lock the cycle-speed dispatch path.


---

## §10 Test plan

Source: `.claude/specs/sprint-10-test-plan.md` §3 PR-CE64-MB-6 row + §7.5 (sigma_router × arigraph).

### §10.1 Banding policy (10 tests, `tests/banding_policy.rs`)

One test per Σ-tier dispatch decision:

| Test | Assertion |
|---|---|
| `sigma1_routes_to_tokio` | `Banding::lookup(1) == BackingKind::Tokio` |
| `sigma2_routes_to_tokio` | `Banding::lookup(2) == BackingKind::Tokio` |
| `sigma3_routes_to_tokio` | … |
| `sigma4_routes_to_tokio_default_OQ1_unratified` | Pre-ratification: Σ4 → Tokio |
| `sigma4_routes_to_cycle_speed_OQ1_ratified` | Post-ratification (alternative banding): Σ4 → InMemCycleSpeed |
| `sigma5_routes_to_tokio_default` | — |
| `sigma6_routes_to_inmem_emergent` | `Banding::lookup(6) == BackingKind::InMemEmergent` |
| `sigma7_routes_to_inmem_cycle_speed` | `Banding::lookup(7) == BackingKind::InMemCycleSpeed` |
| `sigma8_routes_to_inmem_cycle_speed` | — |
| `sigma9_routes_to_escalate_only` | `Banding::lookup(9) == BackingKind::EscalateOnly` |
| `sigma10_routes_to_escalate_only` | — |
| `coarse_fine_validate_compat` | `Banding::validate_compat(SigmaTier::TwigBranching, 7) == true`; `validate_compat(Emergent, 9) == false` |

### §10.2 Cold-start K-NN (5 tests, `tests/cold_start_k_nn.rs`)

| Test | Assertion |
|---|---|
| `cold_start_returns_some_style_for_unknown_pair` | Unknown `(role, G_slot)` resolves to a non-None StyleId |
| `cold_start_caches_after_first_call` | 2nd call for same (role, G_slot) hits cache (no codebook scan) |
| `cold_start_cache_invalidates_on_g_eviction` | Subscribed EvictionMsg drops cache row for that G_slot |
| `cold_start_returns_nearest_atom_int4_32d` | Hand-crafted INT4-32D query has known nearest atom in codebook; resolver returns its StyleId |
| `cold_start_fails_on_empty_codebook` | Codebook = []; `resolve` returns `SpawnError::ColdStartFailed` |

### §10.3 Pruning triggers (4 tests, `tests/pruning_triggers.rs`)

| Test | Assertion |
|---|---|
| `budget_exhausted_queues_drop` | `Budget == 0` after `dispatch_cycle`; `queue.drain()` includes that ID with reason `BudgetExhausted` |
| `xor_cancel_queues_drop` | `Prune` msg with `XorCancel` arrives; queue accumulates; drain returns it |
| `outcome_sufficient_queues_drop` | Dispatcher_fn returns sufficiency flag; `Prune` msg with `OutcomeSufficient` arrives |
| `pruning_drain_clears_queue` | After `drain`, `queue.is_empty()` |

### §10.4 Σ9-Σ10 escalation (3 tests, `tests/escalation.rs`)

| Test | Assertion |
|---|---|
| `sigma9_routes_to_epiphany_witness_msg` | Pearl rung = 5; supervisor receives `EpiphanyWitness { edge, from_mailbox }` |
| `sigma10_routes_to_rubicon_witness_msg` | Pearl rung = 7; supervisor receives `RubiconWitness { edge, from_mailbox }` |
| `escalation_load_sheds_oldest_when_queue_full` | 1025 escalations queued; oldest (FIFO drop) is shed; TECH_DEBT entry appended |

### §10.5 Plasticity feedback (5 tests, `tests/plasticity_feedback.rs`)

| Test | Assertion |
|---|---|
| `plasticity_report_absorbed_into_aggregator` | After 10 `PlasticityReport` msgs, `aggregator.lookup(role, G)` returns the sum |
| `top_k_returns_hottest_pairs` | After 100 reports with biased distribution, `top_k(10)` returns the 10 hottest in count order |
| `spawn_prior_bias_refresh_normalizes_to_2_0_max` | After refresh, max bias = 2.0 (1.0 + normalized 1.0) |
| `spawn_prior_bias_neutral_for_unknown_pair` | Unknown (role, G) returns 1.0 from `bias.lookup` |
| `hot_pair_upgrades_emergent_to_twig_at_threshold` | `bias > 1.5` AND incoming Σ6 → router upgrades dispatch to Σ7 (logged for replay) |

### §10.6 KernelHandle JIT pipeline (3 tests, `tests/kernel_handle_pipeline.rs`)

| Test | Assertion |
|---|---|
| `first_spawn_for_style_compiles` | `kernel_cache.misses` increments; cache size = 1; KernelHandle returned |
| `second_spawn_for_same_style_hits_cache` | `kernel_cache.hits` increments; no JitCompiler call observed |
| `kernel_compile_failure_propagates` | Mock JitCompiler returns Err; spawn returns `SpawnError::KernelCompileFailed` |

**Total: 30 tests** (matches sprint-10-test-plan.md §3 PR-CE64-MB-6 row target).

---

## §11 Bench plan

Per `sprint-10-test-plan.md` §8.7 (added for this PR — extension to existing bench table).

```rust
// crates/lance-graph-supervisor/benches/sigma_router_bench.rs
use criterion::{Criterion, criterion_group, criterion_main, black_box};

fn sigma_router_dispatch_latency(c: &mut Criterion) { /* < 1 µs target — spawn msg → reply */ }
fn plasticity_aggregator_throughput(c: &mut Criterion) { /* > 10M reports/sec — pure HashMap absorb */ }
fn jit_kernelhandle_hit(c: &mut Criterion) { /* < 100 ns — cached lookup */ }
fn kernel_handle_miss_then_compile(c: &mut Criterion) { /* < 50 ms first compile (mocked JitCompiler) */ }

criterion_group!(benches, sigma_router_dispatch_latency, plasticity_aggregator_throughput, jit_kernelhandle_hit, kernel_handle_miss_then_compile);
criterion_main!(benches);
```

Targets calibrated for ubuntu-24.04 github-hosted CI runner (2-core, 7 GB RAM).

---

## §12 Risk matrix

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| OQ-1 banding policy unratified at sprint-11 Wave 5 spawn (Σ4-Σ5 promotion question) | HIGH | Med | Pre-sprint-11 gate (W10 spec OQ-table): W7-impl pauses Wave 5 until user "go" on OQ-1. Default banding is the conservative path (all reflexes → Tokio); ratification can only PROMOTE to cycle-speed, never demote — so default is safe-to-ship. |
| Cross-crate dep cycle (supervisor → planner for JitCompiler) | HIGH | Low | Mitigation already specified §8.1: fallback to JitCompiler trait object injected at `pre_start`. Probe on first compile attempt. |
| Σ9-Σ10 escalation flood saturates supervisor mailbox | MED | Low | §9.2 backpressure load-shedding (1024-entry queue + TECH_DEBT on drop). EPIPHANY-tier emission is rare-by-construction (Pearl rung 5 threshold). |
| `(role, G_slot)` plasticity cache grows unbounded | MED | Med | `top_k(64)` truncates working set; old entries naturally evict at refresh epoch. Per-session ephemeral; no cross-session persistence in this PR (deferred to sprint-12+). |
| INT4-32D codebook unavailable (`p64-bridge::STYLES` not yet built/loaded) | HIGH | Low | `ColdStartFallback::new()` returns Err on empty codebook; tests catch (§10.2 last row). Sprint-11 dep graph: `pr-j-1-int4-32d-atoms` must merge before W7-impl (Wave 5). Verify in dep-graph W10 spec — not currently listed as a hard prerequisite; ESCALATE. |
| W6 `CompartmentReport` missing `g_slot_at_drop` field (§6.1 assumes it) | HIGH | High | **Cross-spec touchpoint:** W6 spec §4.2 returns `CompartmentReport { role, plasticity, sigma_tier, final_budget }` — no `g_slot_at_drop`. W7 needs this. **Action:** W7-impl coordinates with W6-impl to add the field (~3 LOC patch on W6). Meta-review must flag. |
| Banding `validate_compat` panics in debug builds on bad caller input | LOW | Med | Use `debug_assert` not `assert`; return `SpawnError::BindFailure` on coarse-fine mismatch in release. |
| KernelHandle cache stale after style hot-reload (e.g. YAML edit at runtime) | LOW | Low | Out of scope sprint-11 (no hot-reload supported). Cache invalidation on hot-reload is sprint-12+. |
| 1500 LOC envelope blown — actor + 6 sub-modules may push to 2000 LOC | MED | Med | Per `Bound your scope` rule: if heading > 2× envelope, append BLOCKER and stop. Likely culprit: §5 cold-start, §6 plasticity. Tight code review pre-merge. |

---

## §13 Open Questions

| OQ | Question | Tentative resolution | Resolver | Gating |
|---|---|---|---|---|
| **W7-OQ-1 (= parent plan OQ-1)** | Σ4-Σ5 banding (Tokio-reflex or InMem-cycle-speed?) | Keep Σ4-Σ5 → Tokio (conservative); upgrade to cycle-speed only after profiling shows reflex micro-decisions need <1 µs | **User must ratify before sprint-11 Wave 5** | BLOCKS Wave 5 |
| **W7-OQ-2 (cross-spec)** | W6 `CompartmentReport` missing `g_slot_at_drop` | W6 spec patch (~3 LOC: add `pub g_slot_at_drop: u8` field) | Meta-review verifies W6 spec; if absent, W6-impl adds before sprint-11 Wave 4 | BLOCKS Wave 4 → 5 |
| **W7-OQ-3 (Jirak threshold)** | Hand-tuned spawn-prior upgrade threshold (1.5) is not Jirak-derived | Hand-tune for sprint-11; CLAUDE.md `I-NOISE-FLOOR-JIRAK` mandates Jirak-derived bound when principled threshold is needed; defer derivation to sprint-12+ via VAMPE+Jirak coupled-revival track | Meta-review accepts hand-tuned with TECH_DEBT note | Non-blocking |
| **W7-OQ-4 (cross-spec)** | INT4-32D codebook (`p64-bridge::STYLES`) — when does it land? Must precede Wave 5 | `.claude/plans/pr-j-1-int4-32d-atoms.md` must be DONE before W7-impl. Currently NOT in W10 dep-graph (W10 spec missed this) | Meta-review flags W10 graph; main thread adds pre-sprint-11 dep | BLOCKS Wave 5 |
| **W7-OQ-5 (cross-crate dep direction)** | Supervisor → Planner as hard build dep (for `get_jit_compiler()`) — does it create a cycle? | Probably not; planner depends on supervisor through trait imports only. If cycle, fallback to trait object injection | W7-impl on first compile | Non-blocking if mitigation works |
| **W7-OQ-6 (parent plan OQ-3)** | Compartment plasticity update granularity (bit-counter per emission + NARS at AriGraph commit?) | bit-counter per emission (W6 owns) + NARS truth-refine at AriGraph commit (W5 owns); router only does Hebbian rollup of bit-counters at drop_row. Aligns with parent OQ-3. | User ratifies before sprint-11 Wave 4 (W6) | BLOCKS Wave 4 (not W7 directly, but coupled) |

---

## §14 Files-to-touch summary

| File | Action | LOC | Owner |
|---|---|---|---|
| `crates/lance-graph-supervisor/Cargo.toml` | EDIT | +5 | supervisor crate |
| `crates/lance-graph-supervisor/src/lib.rs` | EDIT | +10 (re-exports) | supervisor crate |
| `crates/lance-graph-supervisor/src/callcenter_supervisor.rs` | EDIT | +30 (EpiphanyWitness + RubiconWitness handlers + child spawn) | supervisor crate (existing PR #366) |
| `crates/lance-graph-supervisor/src/sigma_tier_router.rs` | NEW | ~450 | supervisor crate |
| `crates/lance-graph-supervisor/src/banding.rs` | NEW | ~80 | supervisor crate |
| `crates/lance-graph-supervisor/src/plasticity_aggregator.rs` | NEW | ~200 | supervisor crate |
| `crates/lance-graph-supervisor/src/spawn_prior.rs` | NEW | ~150 | supervisor crate |
| `crates/lance-graph-supervisor/src/pruning.rs` | NEW | ~180 | supervisor crate |
| `crates/lance-graph-supervisor/src/escalation.rs` | NEW | ~150 | supervisor crate |
| `crates/lance-graph-supervisor/src/coldstart.rs` | NEW | ~120 | supervisor crate |
| `crates/par-tile/src/mailbox_soa.rs` | EDIT (W6 cross-touch) | +3 (`g_slot_at_drop` field in `CompartmentReport`) | par-tile (W6) |
| `crates/lance-graph-supervisor/tests/*.rs` | NEW | ~1030 (6 test files) | supervisor crate |
| `crates/lance-graph-supervisor/benches/sigma_router_bench.rs` | NEW | ~200 | supervisor crate |
| `.github/workflows/rust-test.yml` | EDIT | +12 (new job) | workspace |
| `.claude/board/STATUS_BOARD.md` | EDIT | +1 row (D-CE64-MB-6) | board hygiene |
| `.claude/board/AGENT_LOG.md` | APPEND | +1 entry | board hygiene |

**Total: ~1500 LOC implementation + ~1230 LOC tests/benches + cross-spec touch on W6 (~3 LOC).** Matches parent plan §7 envelope.

---

## §15 Cross-references

**Plans / specs this composes:**
- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 (mechanism naming) + §7 PR-CE64-MB-6 (LOC + scope) + §10 (Zone-2 ractor topology) + §11 OQ-1/OQ-3/OQ-4 (gating ratifications) + E-CE64-MB-8/9/10 (Σ10 Rubicon dispatcher + JIT pipeline closure + plasticity emergence)
- `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (W1) — `Mailbox<T>` + 3 backings + `AttentionMaskActor`
- `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) — `MailboxSoA<N>` + `SigmaTier` + `CompartmentReport` (cross-spec touchpoint: `g_slot_at_drop` field add)
- `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` (W2) — `CausalEdge64` v2 layout (G/W/truth accessors emitted from compartments)
- `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` (W5) — Σ9 EPIPHANY → AriGraph SPO-G commit path
- `.claude/specs/sprint-10-pr-dep-graph.md` (W10) — Wave 5 placement (depends on Waves 1-4)
- `.claude/specs/sprint-10-test-plan.md` (W11) — §3 PR-CE64-MB-6 row, §8 bench targets
- `.claude/plans/pr-j-1-int4-32d-atoms.md` — INT4-32D codebook (cold-start dep; flagged W7-OQ-4)
- `.claude/knowledge/linguistic-epiphanies-2026-04-19.md` E21 — Σ10 Rubicon doctrine (runtime dispatcher closes the named tier doctrine)
- `THINKING_ORCHESTRATION_WIRING.md` Gap 4 (Elevation Not Connected — closed) + Gap 3 (JIT pipeline — closed via §8 KernelHandleCache)

**This spec does NOT:**
- Define `Mailbox<T>` trait shape (W1 owns)
- Define `MailboxSoA<N>` layout or `SigmaTier` enum variants (W6 owns)
- Define `CausalEdge64` accessors (W2 owns)
- Define `ThinkingAtom32x4` codebook contents (`pr-j-1-int4-32d-atoms` owns)
- Propose new ractor tree shape (extends PR #366's `CallcenterSupervisor`; no replacement)
- Persist plasticity counters cross-session (deferred to sprint-12+ Lance dataset writer)

**Board files this spec triggers** (per CLAUDE.md Mandatory Board-Hygiene Rule, when PR-CE64-MB-6 opens in sprint-11):
- `.claude/board/STATUS_BOARD.md` — append D-CE64-MB-6 row (Queued → In progress → In PR → Shipped)
- `.claude/board/AGENT_LOG.md` — one-liner per W7-impl commit
- `.claude/board/LATEST_STATE.md` — Contract Inventory append (SigmaTierRouter, Banding, PlasticityAggregator) post-merge
- `.claude/board/PR_ARC_INVENTORY.md` — PREPEND entry post-merge
- `.claude/board/EPIPHANIES.md` — already prepended E-CE64-MB-8/9/10 (no additional epiphanies in this spec)

---

*End of pr-ce64-mb-6-sigma-tier-router.md — W7 deliverable, sprint-log-10 sigma-tier-router worker.*
