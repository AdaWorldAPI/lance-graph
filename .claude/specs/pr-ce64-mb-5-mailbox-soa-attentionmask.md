# PR-CE64-MB-5 — MailboxSoA<N> + AttentionMask SoA + AttentionMaskActor

> **Status:** Spec draft (sprint-log-10 W6, 2026-05-14)
> **Worker:** W6 (mailbox-soa-attentionmask)
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §4 + §5 + §7
> **PR sequence position:** 5 of 7 (depends on PR-CE64-MB-1, PR-CE64-MB-2, PR-CE64-MB-3, PR-CE64-MB-4)
> **LOC estimate:** ~1200 (cross-crate: par-tile + lance-graph-supervisor)
> **Closes:** THINKING_ORCHESTRATION_WIRING Gap 1 (Contract Not Consumed)
> **Deliverables in scope:** D-CE64-MB-4, D-CE64-MB-5, D-CE64-MB-7, D-CE64-MB-8, D-CE64-MB-9

---

## §1 Statement of Scope

This PR is where the sparse-rename abstraction (§2 of parent plan) meets the runtime. Two design
ideas converge:

**Compartment-as-SoA-row** — A reasoning compartment is not a standalone ractor actor; it is a
typed row in `MailboxSoA<N>`. The entire compartment population for one supervisor scope is one
contiguous SoA, cache-friendly, dispatched per-cycle by the Sigma-tier router (PR-CE64-MB-6, W7).
This collapses the "10K mailboxes = 10K actor spawns" concern.

**Rename-as-register-file** — Architectural identities (OGIT domain G u32, witness palette W,
thinking style, truth qualia) are too numerous to fit in a 5/6/8-bit hot field. Per §2 of the
parent plan, `AttentionMask` plays the role of CPU register renaming: a session-ephemeral
register file that maps physical slots to architectural IDs and back.

`AttentionMaskActor` makes this register file safe in the async/multi-threaded world by
wrapping it in a ractor singleton (Tokio-backed this PR; InMemoryMailbox cycle-speed promotion
deferred to sprint-11+). Every compartment Bind/Lookup/Resolve call goes through the actor
message queue, adding microseconds of latency that is acceptable for the Tokio-shape Sigma-2-8
tier but will require a per-thread shadow table approach for true cycle-speed (OQ deferred;
see §9 Risk Matrix).

`MailboxSoA<N>` is the compartment topology: per-cycle dispatch, per-row delta buffers, zero-copy
`BindSpaceView` borrows into shared `Arc<BindSpace>`, and a per-row `PlasticityCounter` that
implements Hebbian "fired together wired together" (E-CE64-MB-10) naturally.

The `BindSpaceView<'_>` type (D-CE64-MB-8) is recommended to live in `par-tile` as the
diamond-apex crate visible to both the bevy session and lance-graph-supervisor internals,
rather than in `cognitive-shader-driver` which would invert the intended dep direction.

---

## §2 AttentionMask SoA — Full Implementation Spec

**Extends parent plan §4 verbatim skeleton; adds LRU policy detail, concurrent access model,
message shapes, and eviction broadcast.**

### §2.1 Struct fields (from parent plan §4, annotated with implementation detail)

```rust
// crates/par-tile/src/attention_mask.rs

/// Fixed slot counts (L1-cache-friendly: total struct size ~2 KB).
pub const G_SLOTS: usize     = 32;   // 5-bit physical OGIT domain
pub const W_SLOTS: usize     = 64;   // 6-bit witness palette
pub const STYLE_SLOTS: usize = 256;  // 8-bit thinking style

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum SlotKind { G, W, Style }

/// Session-ephemeral rename register file.
///
/// # Concurrency model
/// `AttentionMask` is `!Send` — interior mutable u32 LRU counters and
/// Option slot fields are mutated non-atomically by bind_* / touch_* / tick.
/// Wrap in tokio::sync::Mutex inside AttentionMaskActor; never share raw.
///
/// # Memory layout
/// repr(C, align(64)) pins struct to one cache-line boundary.
/// Total: G_SLOTS*(8+4) + W_SLOTS*(8+4) + STYLE_SLOTS*(8+4) + 4 + 4 ≈ 2 KB.
#[repr(C, align(64))]
pub struct AttentionMask {
    // Rename tables (architectural -> physical slot)
    pub g_slots:     [Option<OgitDomainId>; G_SLOTS],
    pub w_slots:     [Option<WitnessId>;    W_SLOTS],
    pub style_slots: [Option<StyleId>;      STYLE_SLOTS],

    // Active grammar selector
    pub active_grammar: GrammarAlphabet,

    // LRU clocks per slot table
    // Per-slot last-access cycle. Eviction picks argmin(*_lru) over occupied slots.
    // Initialized to 0; touched on every Bind / Resolve that hits the slot.
    pub g_lru:     [u32; G_SLOTS],
    pub w_lru:     [u32; W_SLOTS],
    pub style_lru: [u32; STYLE_SLOTS],

    // Monotonic session cycle counter (drives LRU)
    // Wraps at u32::MAX; harmless because LRU comparison is relative order.
    pub cycle: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct OgitDomainId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct WitnessId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct StyleId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum GrammarAlphabet {
    PlannerClusters12,       // 12 ThinkingStyle clusters (lance-graph-planner)
    CognitivePrimitives34,   // 34 hpc/styles entries (ndarray::hpc::styles)
    VerbsTekamolo144,        // 12x12 verb compositions (German cognitive grammar)
    FullStyle36Plus,         // 36 contract::thinking + YAML extensions (up to 256)
}

/// Stale-ok snapshot for read-only consumers who can tolerate staleness.
/// Avoids locking AttentionMaskActor for every lookup in snapshot-refresh mode.
#[derive(Clone, Debug)]
pub struct AttentionMaskSnapshot {
    pub g_slots:     [Option<OgitDomainId>; G_SLOTS],
    pub w_slots:     [Option<WitnessId>;    W_SLOTS],
    pub style_slots: [Option<StyleId>;      STYLE_SLOTS],
    pub cycle:       u32,
}
```

### §2.2 LRU policy detail (delta from parent plan §4)

Parent plan §4 states "evict slot with smallest *_lru[i] value." The full implementation
detail:

- `tick()` is called **once per dispatcher cycle** by SigmaTierRouter (PR-CE64-MB-6);
  in this PR it is also called by `MailboxSoA::dispatch_cycle` for test coverage.
- `touch_*` sets `*_lru[slot] = self.cycle` on every bind or resolve that hits the slot.
- Eviction candidate = `argmin(*_lru)` over slots where `*_slots[i].is_some()`.
- If a free slot exists, it is claimed first (no eviction needed).
- If the id being bound is already in a slot, that slot is returned (no eviction; LRU bumped).

```rust
impl AttentionMask {
    pub fn tick(&mut self) { self.cycle = self.cycle.wrapping_add(1); }

    pub fn touch_g(&mut self, slot: usize) {
        debug_assert!(slot < G_SLOTS);
        self.g_lru[slot] = self.cycle;
    }
    // Identical for touch_w / touch_style

    fn evict_candidate_g(&self) -> usize {
        self.g_slots.iter().enumerate()
            .filter(|(_, s)| s.is_some())
            .min_by_key(|(i, _)| self.g_lru[*i])
            .map(|(i, _)| i)
            .expect("evict_candidate_g called with no occupied slots")
    }

    pub fn bind_g(&mut self, id: OgitDomainId) -> (u8, Option<OgitDomainId>) {
        // Already bound? Return existing slot (no eviction).
        for (i, s) in self.g_slots.iter().enumerate() {
            if *s == Some(id) { self.touch_g(i); return (i as u8, None); }
        }
        // Free slot available?
        if let Some(i) = self.g_slots.iter().position(|s| s.is_none()) {
            self.g_slots[i] = Some(id);
            self.touch_g(i);
            return (i as u8, None);
        }
        // LRU eviction.
        let victim = self.evict_candidate_g();
        let evicted = self.g_slots[victim].take();
        self.g_slots[victim] = Some(id);
        self.touch_g(victim);
        (victim as u8, evicted)
    }

    pub fn lookup_g(&self, id: OgitDomainId) -> Option<u8> {
        self.g_slots.iter().position(|s| *s == Some(id)).map(|i| i as u8)
    }

    pub fn resolve_g(&self, slot: u8) -> Option<OgitDomainId> {
        self.g_slots.get(slot as usize).copied().flatten()
    }
    // Identical shapes for bind_w/lookup_w/resolve_w and bind_style/lookup_style/resolve_style
}
```

### §2.3 Concurrent access model

**`AttentionMask` is `!Send`** because the interior u32 LRU counters and Option slot fields
are mutated non-atomically. The sole solution for this PR: `AttentionMask` lives inside
`AttentionMaskActor` behind a `tokio::sync::Mutex<AttentionMask>`.

**Latency tax:** Every compartment Bind crosses a tokio channel + mutex boundary. Estimated
cost: 1-5 µs per round trip under moderate contention. Acceptable for Sigma-2-8 Tokio-shape
compartments (ms-scale dispatch cycles). Breaks the 200 ns cycle-speed Sigma-6-8 budget.

**Sprint-11+ optimization (deferred, ratify via OQ-SHADOW):** Per-thread shadow
`AttentionMaskSnapshot` refreshed once per dispatcher tick. Compartments read from shadow
without locking. Actor remains authoritative for writes and eviction broadcast.

### §2.4 BindRequest / BindReply message shapes

```rust
// crates/par-tile/src/attention_actor.rs — message type section

#[derive(Clone, Debug)]
pub enum ArchitecturalId {
    G(OgitDomainId),
    W(WitnessId),
    Style(StyleId),
}

pub enum AttentionMaskMsg {
    /// Bind an architectural identity to a physical slot. Reply: BindReply.
    Bind {
        id:    ArchitecturalId,
        reply: tokio::sync::oneshot::Sender<BindReply>,
    },
    /// Touch (bump LRU) for a known slot. Fire-and-forget; no reply.
    Touch { kind: SlotKind, slot: u8 },
    /// Look up physical slot. Reply: Option<u8>.
    Lookup {
        id:    ArchitecturalId,
        reply: tokio::sync::oneshot::Sender<Option<u8>>,
    },
    /// Reverse: physical slot -> architectural id. Reply: Option<ArchitecturalId>.
    Resolve {
        kind:  SlotKind,
        slot:  u8,
        reply: tokio::sync::oneshot::Sender<Option<ArchitecturalId>>,
    },
    /// Advance session cycle counter (one per dispatcher cycle). No reply.
    Tick,
    /// Return stale-ok snapshot. Reply: AttentionMaskSnapshot.
    Snapshot {
        reply: tokio::sync::oneshot::Sender<AttentionMaskSnapshot>,
    },
}

#[derive(Debug)]
pub struct BindReply {
    /// Physical slot index assigned (0..G_SLOTS, W_SLOTS, or STYLE_SLOTS).
    pub slot:    u8,
    /// Architectural id displaced by LRU eviction, if any.
    pub evicted: Option<ArchitecturalId>,
}
```

### §2.5 EvictionMsg broadcast

When `bind_*` evicts a slot, `AttentionMaskActor` broadcasts an `EvictionMsg` over a
`tokio::sync::broadcast::Sender<EvictionMsg>` (channel capacity: 256, see OQ-BCAST-SIZE).

```rust
/// Broadcast when AttentionMask evicts a physical slot for kind K.
/// Subscribers: compartments holding live slot refs, AriGraph for ghost-edge updates.
#[derive(Clone, Debug)]
pub struct EvictionMsg {
    pub kind:        SlotKind,
    pub slot:        u8,
    pub was:         ArchitecturalId,   // displaced id
    pub replaced_by: ArchitecturalId,   // incoming id
}
```

**Defensive re-resolve pattern (mandatory for all compartment code):**

Any compartment caching a `(slot, kind)` pair MUST:
1. Subscribe to a `broadcast::Receiver<EvictionMsg>` on startup.
2. Before emitting any `CausalEdge64` using a cached slot, drain pending eviction
   notices via `try_recv()`.
3. If an eviction notice matches the cached slot, call `AttentionMaskMsg::Resolve` to
   confirm the slot is still valid, or re-`Bind` if needed.

This pattern is mandatory because `broadcast::Receiver` may lag (`RecvError::Lagged`) or
disconnect (`RecvError::Closed`) under load (see §9 Risk Matrix, HIGH item 2).

---

## §3 AttentionMaskActor Ractor Singleton

**Extends parent plan §4 D-CE64-MB-5.**

### §3.1 Spawn point and supervision

`AttentionMaskActor` spawns as a named child of `CallcenterSupervisor` (the existing Tokio-backed
ractor supervisor from PR #366 S7-W3). This gives it one-for-one supervision (restart-on-panic,
exponential backoff 100ms to 30s, bounded mailbox default 1024).

```rust
// Addition to crates/lance-graph-supervisor/src/supervisor.rs (~80 LOC)
// Inside CallcenterSupervisor::pre_start or an Init message handler:

let mask_cell = AttentionMaskActor::spawn_linked(
    myself.get_cell(),
    AttentionMaskActorState::default(),
    (),
).await?.get_cell();
state.attention_mask_ref = Some(mask_cell);
```

One global instance per supervisor scope. Per-tenant scoping deferred (one actor per supervisor
is the correct default for single-tenant deployment).

### §3.2 Actor implementation skeleton

```rust
// crates/par-tile/src/attention_actor.rs

use ractor::{Actor, ActorProcessingErr, ActorRef};

pub struct AttentionMaskActor;

pub struct AttentionMaskActorState {
    mask:        AttentionMask,
    eviction_tx: tokio::sync::broadcast::Sender<EvictionMsg>,
}

impl Default for AttentionMaskActorState {
    fn default() -> Self {
        let (tx, _) = tokio::sync::broadcast::channel(256);
        Self { mask: AttentionMask::default(), eviction_tx: tx }
    }
}

impl Actor for AttentionMaskActor {
    type Msg       = AttentionMaskMsg;
    type State     = AttentionMaskActorState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args:   (),
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(AttentionMaskActorState::default())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg:     Self::Msg,
        state:   &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            AttentionMaskMsg::Bind { id, reply } => {
                let (slot, evicted_opt) = match &id {
                    ArchitecturalId::G(g)     => {
                        let (s, e) = state.mask.bind_g(*g);
                        (s, e.map(ArchitecturalId::G))
                    }
                    ArchitecturalId::W(w)     => {
                        let (s, e) = state.mask.bind_w(*w);
                        (s, e.map(ArchitecturalId::W))
                    }
                    ArchitecturalId::Style(st) => {
                        let (s, e) = state.mask.bind_style(*st);
                        (s, e.map(ArchitecturalId::Style))
                    }
                };
                if let Some(was) = &evicted_opt {
                    let kind = match &id {
                        ArchitecturalId::G(_)     => SlotKind::G,
                        ArchitecturalId::W(_)     => SlotKind::W,
                        ArchitecturalId::Style(_) => SlotKind::Style,
                    };
                    // Ignore send error — subscriber may have disconnected.
                    let _ = state.eviction_tx.send(EvictionMsg {
                        kind,
                        slot,
                        was:         was.clone(),
                        replaced_by: id.clone(),
                    });
                }
                let _ = reply.send(BindReply { slot, evicted: evicted_opt });
            }
            AttentionMaskMsg::Touch { kind, slot } => {
                match kind {
                    SlotKind::G     => state.mask.touch_g(slot as usize),
                    SlotKind::W     => state.mask.touch_w(slot as usize),
                    SlotKind::Style => state.mask.touch_style(slot as usize),
                }
            }
            AttentionMaskMsg::Lookup { id, reply } => {
                let r = match &id {
                    ArchitecturalId::G(g)     => state.mask.lookup_g(*g),
                    ArchitecturalId::W(w)     => state.mask.lookup_w(*w),
                    ArchitecturalId::Style(s) => state.mask.lookup_style(*s),
                };
                let _ = reply.send(r);
            }
            AttentionMaskMsg::Resolve { kind, slot, reply } => {
                let aid = match kind {
                    SlotKind::G     => state.mask.resolve_g(slot).map(ArchitecturalId::G),
                    SlotKind::W     => state.mask.resolve_w(slot).map(ArchitecturalId::W),
                    SlotKind::Style => state.mask.resolve_style(slot).map(ArchitecturalId::Style),
                };
                let _ = reply.send(aid);
            }
            AttentionMaskMsg::Tick => { state.mask.tick(); }
            AttentionMaskMsg::Snapshot { reply } => {
                let snap = AttentionMaskSnapshot {
                    g_slots:     state.mask.g_slots,
                    w_slots:     state.mask.w_slots,
                    style_slots: state.mask.style_slots,
                    cycle:       state.mask.cycle,
                };
                let _ = reply.send(snap);
            }
        }
        Ok(())
    }
}

impl AttentionMaskActor {
    /// Get a broadcast receiver for eviction events.
    /// Compartments must poll this to invalidate cached slot references.
    pub fn subscribe_evictions(
        state: &AttentionMaskActorState,
    ) -> tokio::sync::broadcast::Receiver<EvictionMsg> {
        state.eviction_tx.subscribe()
    }
}
```

**Tokio-backed (this PR).** InMemoryMailbox cycle-speed promotion in sprint-11+ introduces
a per-thread shadow snapshot refreshed once per dispatcher tick. Actor interface unchanged.

---

## §4 MailboxSoA<N> — Full Implementation Spec

**Extends parent plan §5 verbatim skeleton; adds lifecycle detail, dispatch sequence,
plasticity counter, and BindSpaceView integration.**

### §4.1 Type definition

```rust
// crates/par-tile/src/mailbox_soa.rs

use std::ops::Range;
use std::time::SystemTime;

/// Const-generic N = maximum concurrent compartment count.
/// Default N = 1024 (4 x current BindSpace row count, see OQ-N).
/// Consumers pick: N=512 for bevy, N=4096 for Sigma8-branching.
/// Type alias provided: pub type DefaultMailboxSoA = MailboxSoA<1024>;
pub struct MailboxSoA<const N: usize> {
    // Identity
    pub ids:    [MailboxId; N],
    pub active: [bool; N],
    pub count:  usize,

    // Per-compartment classification
    pub roles:       [RoleId;          N],
    pub temporals:   [TemporalWindow;  N],
    pub sigma_tiers: [SigmaTier;       N],

    // Shared BindSpace access (zero-copy, Arc-backed)
    pub bindspace_views: [BindSpaceView<'static>; N],

    // Ephemeral per-row state
    pub deltas:  [DeltaBuffer; N],  // ~1 KB; dropped on drop_row
    pub budgets: [Budget;      N],  // countdown; zero = force-prune

    // Emission channels (per-compartment CausalEdge64 sender)
    pub witness_outs: [tokio::sync::mpsc::Sender<CausalEdge64>; N],

    // External intent gate
    // None = pure-internal (~95% of Sigma-6-8); never reaches Zone 3.
    // Some(handle) = emission serializes to Zone 3 via the handle.
    pub intents: [Option<ConsumerHandle>; N],

    // Supervision chain
    pub parents: [Option<MailboxId>; N],

    // Plasticity bit-counters (E-CE64-MB-10)
    // Increments on every CausalEdge64 emission.
    // Per-row, not global (role, G) pair — aggregated by supervisor on drop_row.
    pub plasticity_counters: [PlasticityCounter; N],
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MailboxId(pub u64);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct RoleId(pub u32);

#[derive(Copy, Clone, Debug)]
pub struct TemporalWindow {
    pub start_cycle: u32,
    pub end_cycle:   u32,
    pub flags:       u32,
    pub _pad:        u32,
}

/// ~1 KB per-cycle scratchpad. Allocated on push_row; dropped on drop_row.
pub struct DeltaBuffer {
    pub bytes: Box<[u8; 1024]>,
}

#[derive(Copy, Clone, Debug)]
pub struct Budget(pub u64);

pub const DEFAULT_BUDGET: u64 = 1024;

/// Hebbian co-occurrence counter. u64 to avoid overflow (saturating_add used).
#[derive(Copy, Clone, Debug, Default)]
pub struct PlasticityCounter(pub u64);

impl PlasticityCounter {
    #[inline]
    pub fn increment(&mut self) { self.0 = self.0.saturating_add(1); }
}

#[derive(Copy, Clone, Debug)]
pub enum SigmaTier {
    StaticReflex,    // Sigma-1 to Sigma-5: STATIC / Pearl rung 1 / Tokio-backed
    Emergent,        // Sigma-6: EMERGENT / Pearl 2-3 / InMemoryMailbox
    TwigBranching,   // Sigma-7 to Sigma-8: TWIG / InMemoryMailbox cycle-speed
    EpiphanyEscalate, // Sigma-9 to Sigma-10: escalate to L4 planner
}

#[derive(Copy, Clone, Debug)]
pub enum ConsumerHandle {
    Postgrest(EndpointId),
    DrainWs(SubscriberId),
    SupabaseChannel(ChannelId),
    MysqlSink(SinkId),
    GrpcService(ServiceId),
}
```

### §4.2 Core lifecycle methods

```rust
impl<const N: usize> MailboxSoA<N> {
    /// Spawn: claim a free slot and initialize a compartment row.
    /// Returns MailboxId; caller must call AttentionMaskActor::Bind for
    /// each architectural identity (G, W, style) before dispatch_cycle.
    pub fn push_row(
        &mut self,
        role:            RoleId,
        temporal_window: TemporalWindow,
        sigma_tier:      SigmaTier,
        bindspace_view:  BindSpaceView<'static>,
        intent:          Option<ConsumerHandle>,
    ) -> Result<MailboxId, MailboxSoaError> {
        let slot = self.active.iter().position(|a| !*a)
            .ok_or(MailboxSoaError::CapacityExhausted)?;
        let id = MailboxId(next_mailbox_id());
        self.ids[slot]                 = id;
        self.active[slot]              = true;
        self.roles[slot]               = role;
        self.temporals[slot]           = temporal_window;
        self.sigma_tiers[slot]         = sigma_tier;
        self.bindspace_views[slot]     = bindspace_view;
        self.intents[slot]             = intent;
        self.parents[slot]             = None;
        self.budgets[slot]             = Budget(DEFAULT_BUDGET);
        self.plasticity_counters[slot] = PlasticityCounter::default();
        self.count += 1;
        Ok(id)
    }

    /// Drop: reclaim a compartment row and return its plasticity report.
    /// DeltaBuffer drops here. Ghost-edge emission (if unresolved) is the
    /// caller's responsibility (W5 AriGraph spec handles ghost-edge protocol).
    pub fn drop_row(&mut self, id: MailboxId) -> Option<CompartmentReport> {
        let slot = self.ids.iter().position(|i| *i == id)?;
        if !self.active[slot] { return None; }
        self.active[slot] = false;
        self.count -= 1;
        let report = CompartmentReport {
            id,
            role:         self.roles[slot],
            plasticity:   self.plasticity_counters[slot],
            sigma_tier:   self.sigma_tiers[slot],
            final_budget: self.budgets[slot],
        };
        // Reclaim DeltaBuffer allocation.
        self.deltas[slot] = DeltaBuffer { bytes: Box::new([0u8; 1024]) };
        Some(report)
    }

    /// Dispatch: run one cycle over all temporally-active compartments.
    ///
    /// The dispatcher callback receives (slot_index, &BindSpaceView, &DeltaBuffer)
    /// and returns Option<CausalEdge64>. Returning Some(edge):
    ///   1. Pushes edge to witness_outs[slot] channel.
    ///   2. Increments plasticity_counters[slot].
    ///   3. Decrements budgets[slot]; if zero, schedules drop_row.
    ///
    /// Caller is responsible for:
    ///   a. Sending returned edges through CollapseGate -> EdgeColumn write.
    ///   b. For sigma_tier >= TwigBranching OR intent.is_some(), also routing
    ///      to AriGraph SPO-G insert (G resolved via AttentionMaskActor::Resolve).
    ///   c. Calling drop_row on compartments whose XOR-cancel fires.
    pub fn dispatch_cycle(
        &mut self,
        current_cycle: u32,
        dispatcher: &dyn Fn(usize, &BindSpaceView<'_>, &DeltaBuffer) -> Option<CausalEdge64>,
    ) -> Vec<CausalEdge64> {
        let mut emissions = Vec::new();
        let mut prune_ids = Vec::new();

        for slot in 0..N {
            if !self.active[slot] { continue; }
            let tw = self.temporals[slot];
            if current_cycle < tw.start_cycle || current_cycle >= tw.end_cycle { continue; }

            if let Some(edge) = dispatcher(slot, &self.bindspace_views[slot], &self.deltas[slot]) {
                let _ = self.witness_outs[slot].try_send(edge);
                self.plasticity_counters[slot].increment();
                emissions.push(edge);
                if self.budgets[slot].0 > 0 { self.budgets[slot].0 -= 1; }
            }
            if self.budgets[slot].0 == 0 {
                prune_ids.push(self.ids[slot]);
            }
        }
        for id in prune_ids { self.drop_row(id); }
        emissions
    }
}

/// Returned by drop_row; carries per-compartment plasticity stats to supervisor.
#[derive(Debug)]
pub struct CompartmentReport {
    pub id:           MailboxId,
    pub role:         RoleId,
    pub plasticity:   PlasticityCounter,
    pub sigma_tier:   SigmaTier,
    pub final_budget: Budget,
}

#[derive(Debug)]
pub enum MailboxSoaError {
    CapacityExhausted,
}
```

### §4.3 Lifecycle sequence (5 steps from parent plan §5)

1. **Spawn**: Sigma-router (W7) -> `push_row(role, temporal, sigma_tier, bsv, intent)` -> `MailboxId`
2. **Bind**: caller -> `AttentionMaskActor::Bind(G, W, style)` -> hot slots stored in per-row metadata for use in dispatch
3. **Dispatch**: `dispatch_cycle(cycle, dispatcher_fn)` -> `Vec<CausalEdge64>` (each edge carries g_slot/w_slot from binding step)
4. **Emit**: edges -> `CollapseGate` (MergeMode per W4 spec) -> EdgeColumn write via `BindSpaceView::write_delta + commit_with_token`
5. **AriGraph**: edges where `sigma_tier >= TwigBranching || intent.is_some()` -> `AttentionMaskActor::Resolve(G, slot)` -> `AriGraph::insert_spog(S, P, O, G=full_u32)` (W5 spec)
6. **Drop**: `drop_row(id)` -> `CompartmentReport` -> plasticity log in Lance; ghost edge if unresolved (W5)

### §4.4 Plasticity bit-counter (E-CE64-MB-10)

`PlasticityCounter` increments on every emission in `dispatch_cycle`. The counter is per-row
(not a global `(role, G)` map in MailboxSoA). On `drop_row`, `CompartmentReport.plasticity`
carries the counter to the supervisor, which aggregates it into a per-tenant Lance dataset
(Zone 2 cross-cycle accumulation, outside MailboxSoA's scope).

Spawn priors: `SigmaTierRouter` (PR-CE64-MB-6, W7) reads the aggregate plasticity log to bias
compartment spawn decisions toward high-count `(role, G)` pairings.

Pruning heuristic: lives in SigmaTierRouter (not MailboxSoA — separation of mechanism vs policy).
When budget pressure rises (count approaching N), SigmaTierRouter may issue early `drop_row` for
low-plasticity compartments.

---

## §5 BindSpaceView<'_> Implementation

**Implements D-CE64-MB-8. Crate placement: par-tile (diamond apex).**

### §5.1 Rationale for par-tile placement

`BindSpaceView` lives in `crates/par-tile/src/bindspace_view.rs`, not in `cognitive-shader-driver`.
Placing it in the heavy `cognitive-shader-driver` crate would drag that dependency into par-tile,
inverting the intended dep direction (par-tile is the apex). W4 (BindSpace spec, PR-CE64-MB-3)
owns `Arc<BindSpaceColumns>`; this spec owns `BindSpaceView` as the consumer/borrow type that
takes an `Arc<BindSpaceColumns>` ref. Coordinate: W1 (par-tile apex) should include a stub;
this spec defines the authoritative shape since W1's spec does not exist at draft time.

### §5.2 Type definition

```rust
// crates/par-tile/src/bindspace_view.rs

use std::ops::Range;
use std::sync::Arc;

/// Zero-copy borrow into shared BindSpace column storage.
///
/// Read operations: direct slice access, no clone.
/// Write operations: return WriteToken (#[must_use]); actual mutation happens
/// on commit_with_token via CollapseGate. This is the I1 single-mutation-point
/// enforcement at the type level.
#[derive(Clone)]
pub struct BindSpaceView<'a> {
    /// Shared column storage (Arc keeps it alive for 'static use in MailboxSoA).
    pub columns:     Arc<BindSpaceColumns>,
    /// Row range this compartment is responsible for.
    pub rows:        Range<usize>,
    /// u32 bitfield: bit 0 = Column A, ..., bit 7 = Column H.
    /// u32 (not u8) for future-proof extensibility to Columns I-Z.
    pub column_mask: ColumnMask,
    _phantom: std::marker::PhantomData<&'a ()>,
}

/// Column selector bitfield. Bit position = column index (A=0, B=1, ..., H=7).
/// u32 provides 32 column slots; current design uses bits 0-7 only.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ColumnMask(pub u32);

impl ColumnMask {
    pub const COL_A: usize = 0;  // FingerprintColumns
    pub const COL_B: usize = 1;  // QualiaColumn
    pub const COL_C: usize = 2;  // MetaColumn
    pub const COL_D: usize = 3;  // EdgeColumn (CausalEdge64)
    pub const COL_E: usize = 4;  // OntologyDelta (PR-CE64-MB-3)
    pub const COL_F: usize = 5;  // AwarenessColumn
    pub const COL_G: usize = 6;  // ModelBindingColumn
    pub const COL_H: usize = 7;  // EntityTypeId u16

    pub const ALL_ABCD: Self = Self(0b0000_1111);
    pub const ALL_EFGH: Self = Self(0b1111_0000);
    pub const ALL:      Self = Self(0xFF);

    pub fn contains(self, col: usize) -> bool {
        col < 32 && (self.0 >> col) & 1 == 1
    }
}

/// Opaque write token. #[must_use] ensures it cannot be silently discarded.
/// Carry to commit_with_token; dropping without commit = discarding the write.
#[must_use = "WriteToken must be passed to commit_with_token; dropping it discards the write"]
pub struct WriteToken {
    pub(crate) row:   usize,
    pub(crate) col:   usize,
    pub(crate) delta: ColumnDelta,
}

/// Column-typed write delta value.
#[derive(Debug)]
pub enum ColumnDelta {
    Edge(CausalEdge64),      // Column D — EdgeColumn
    Ontology(OntologyDelta), // Column E (PR-CE64-MB-3)
    Awareness(u64),          // Column F — awareness bit-pattern delta
    ModelBinding(u32),       // Column G — model binding id
    EntityType(u16),         // Column H — EntityTypeId
}

impl<'a> BindSpaceView<'a> {
    /// Read 8 CausalEdge64 edges for a row (Column D). Zero-copy.
    pub fn read_edges(&self, row_offset: usize) -> &[CausalEdge64; 8] {
        debug_assert!(self.column_mask.contains(ColumnMask::COL_D));
        let row = self.rows.start + row_offset;
        self.columns.edge_column.row(row)
    }

    // Similar read_fingerprint, read_qualia, read_meta accessors (all &self, zero-copy).

    /// Prepare a write. Returns WriteToken; no mutation yet.
    /// Actual write happens on commit_with_token via CollapseGate (I1 compliance).
    pub fn write_delta(&self, row_offset: usize, delta: ColumnDelta) -> WriteToken {
        WriteToken {
            row:   self.rows.start + row_offset,
            col:   column_index_for_delta(&delta),
            delta,
        }
    }

    /// Commit token through CollapseGate.
    /// Gate applies MergeMode (Xor / Bundle / Superposition per W4 spec).
    /// Returns Ok(()) on Flow; Err(token) on Block/Hold (state unchanged, token returned).
    pub fn commit_with_token(
        &self,
        token: WriteToken,
        gate:  &CollapseGate,
    ) -> Result<(), WriteToken> {
        gate.apply(token, &self.columns)
    }
}
```

---

## §6 Integration Sequence Diagram (ASCII)

One compartment dispatch cycle from Sigma-router to AriGraph commit.

```
Sigma-ROUTER       MAILBOXSOA<N>      ATTENTIONMASK ACTOR    COLLAPSE GATE     ARIGRAPH
    |                   |                    |                    |                |
    |-- push_row ------->|                   |                    |                |
    |   (role=R, G,      |                   |                    |                |
    |    sigma=Twig,      |                   |                    |                |
    |    bsv, intent)    |                   |                    |                |
    |<-- MailboxId=42 ---|                   |                    |                |
    |                   |                    |                    |                |
    |                   |-- Bind(G=0x42) ---->|                   |                |
    |                   |                    |-- lock+bind_g      |                |
    |                   |                    |   -> slot=7         |                |
    |                   |<- BindReply{s=7} --|                   |                |
    |                   |-- Bind(W=0x1F) ---->|                   |                |
    |                   |<- BindReply{s=3} --|                   |                |
    |                   |-- Bind(style) ------>|                   |                |
    |                   |<- BindReply{s=12} -|                   |                |
    |                   |  store g=7,w=3,st=12 in row metadata   |                |
    |                   |                    |                    |                |
    |-- dispatch_cycle ->|                   |                    |                |
    |   (cycle=N,        |                   |                    |                |
    |    dispatcher_fn)  |                   |                    |                |
    |                   |  for each active row:                   |                |
    |                   |  call dispatcher_fn(slot, bsv, delta)  |                |
    |                   |  -> Some(CausalEdge64{                  |                |
    |                   |      g_slot=7, w_slot=3,               |                |
    |                   |      truth=Solid, ...})                 |                |
    |                   |  Touch(G, slot=7) --> actor (async)    |                |
    |                   |  plasticity_counters[slot]++           |                |
    |<-- Vec<CE64> ------|                   |                    |                |
    |                   |                    |                    |                |
    |-- write_delta ------------------------------------------>  |                |
    |   (row_off=0,      |                   |                    |                |
    |    ColumnDelta::Edge(edge))            |                    |                |
    |<-- WriteToken -------------------------------------------|                |
    |                   |                    |                    |                |
    |-- commit_with_token ---------------------------------------->|              |
    |   (token, gate)    |                   |             MergeMode::Bundle      |
    |                   |                    |             -> GateDecision::Flow  |
    |<-- Ok(()) ------------------------------------------ EdgeColumn write      |
    |                   |                    |                    |                |
    |   [sigma=Twig >= TwigBranching]        |                    |                |
    |                   |-- Resolve(G, slot=7) ------------------>|               |
    |                   |<- OgitDomainId(0x42) ------------------|               |
    |                   |                    |                    |                |
    |-- AriGraph SPO-G insert -------------------------------------------------------->|
    |   (S=subj, P=pred, O=obj, G=0x42,     |                    |  insert quad  |
    |    W=witness: resolve W slot=3)        |                    |  + W edge     |
    |                   |                    |                    |                |
    |-- drop_row ------->|                   |                    |                |
    |   (id=42, on       |                   |                    |                |
    |    window-end)     |                   |                    |                |
    |                   |-- Touch(G, s=7) --> actor (LRU update) |                |
    |<-- CompartmentReport                   |                    |                |
    |   (plasticity=N,   |                   |                    |                |
    |    role=R, sigma=Twig)                 |                    |                |
    |-- plasticity log write (Lance, Zone 2) |                    |                |
```

---

## §7 Files-to-Touch Table

| File | Action | LOC Delta | Notes |
|---|---|---|---|
| `crates/par-tile/src/attention_mask.rs` | Create (extend W1 stub if present) | +400 | `AttentionMask` struct + all bind/lookup/resolve/touch/tick/evict_candidate methods + snapshot + `!Send` marker impl |
| `crates/par-tile/src/attention_actor.rs` | Create (extend W1 stub if present) | +250 | `AttentionMaskActor` ractor impl + `AttentionMaskMsg` enum + `EvictionMsg` + broadcast sender + `subscribe_evictions` |
| `crates/par-tile/src/mailbox_soa.rs` | Create (extend W1 stub if present) | +400 | `MailboxSoA<N>` struct + `push_row` + `drop_row` + `dispatch_cycle` + all associated types |
| `crates/par-tile/src/bindspace_view.rs` | Create (extend W1 stub if present) | +200 | `BindSpaceView<'_>` + `ColumnMask` (u32) + `WriteToken` (#[must_use]) + `ColumnDelta` + read/write/commit accessors |
| `crates/par-tile/src/lib.rs` | Edit | +20 | `pub mod` declarations + re-exports for public surface |
| `crates/lance-graph-supervisor/src/lib.rs` | Edit | +30 | `pub use` for `AttentionMaskActor` + `AttentionMaskMsg` + `EvictionMsg`; `attention_mask_ref` on `SupervisorState` |
| `crates/lance-graph-supervisor/src/supervisor.rs` | Edit | +80 | Spawn `AttentionMaskActor` as child of `CallcenterSupervisor` in `pre_start`; forward Tick messages; expose `subscribe_evictions` |
| `crates/par-tile/Cargo.toml` | Edit | +5 | Add `ractor`, `tokio` (features = ["sync","rt","rt-multi-thread"]) to `[dependencies]` |
| `crates/par-tile/tests/integration.rs` | Create | +300 | All 9 integration tests from §8 |

**Total LOC estimate: ~1285.** Within plan's ~1200 target (rounding up for doc comments).

**Coordination with W1:** W1 defines `Mailbox<T>` trait + three backings and creates the par-tile
crate scaffold. This spec's four `src/*.rs` files extend whatever stubs W1 provides. If W1 ships
first and includes stubs, all "Create" actions become "Extend." If this spec ships first, W1
must not conflict on module names or re-exports.

---

## §8 Test Plan

All tests in `crates/par-tile/tests/integration.rs`. Async tests use `#[tokio::test]`.

### §8.1 AttentionMask unit tests

**`attention_mask_bind_lookup_resolve_round_trip`**
Bind 32 distinct `OgitDomainId(0..32)` via `bind_g`. Assert all 32 slots filled (no evictions).
For each id: `lookup_g` returns Some(slot); `resolve_g(slot)` == Some(id). Assert identity
preserved across the table.

**`attention_mask_lru_eviction_order`**
Fill all 32 G-slots. Advance cycle 100x via `tick()`. Touch slot 0 (making it MRU).
Bind G(32) — triggers LRU eviction. Assert evicted id is NOT G(0) (most-recently touched).
Assert G(32) now bound. Assert evicted id is from the set of untouched slots (g_lru == 0).

**`attention_mask_broadcast_subscribers_see_evictions`**
Create 3 broadcast receivers. Fill all 32 G-slots. Bind G(32) -> one eviction.
Assert all 3 receivers receive exactly 1 `EvictionMsg`.
Assert `EvictionMsg.replaced_by == ArchitecturalId::G(OgitDomainId(32))`.
Assert `EvictionMsg.kind == SlotKind::G`.

### §8.2 AttentionMaskActor async tests

**`attention_mask_actor_bind_lookup_resolve_async`**
Spawn `AttentionMaskActor` via `ractor::Actor::spawn`.
Send `Bind(G(42))` -> assert `BindReply.slot` in 0..32.
Send `Lookup(G(42))` -> assert `Some(slot)` matching the BindReply.
Send `Resolve(G, slot)` -> assert `Some(ArchitecturalId::G(OgitDomainId(42)))`.
Send `Tick` -> no reply. Send `Snapshot` -> assert snapshot.cycle == 1.

### §8.3 MailboxSoA tests

**`mailbox_soa_lifecycle_push_dispatch_drop`**
Construct `MailboxSoA::<128>`. push_row x100 (distinct roles, valid temporal windows).
Assert `count == 100`. Run `dispatch_cycle` with dispatcher returning Some(CausalEdge64::zero()).
Assert returned Vec length == 100. drop_row x50 (first 50 MailboxIds). Assert `count == 50`.
Spot-check: `active[0]` false; `active[50]` true.

**`mailbox_soa_xor_cancel_prunes_both`**
push_row x2. Both emit edges. Mock CollapseGate signals XOR-cancel when complementary edges
detected. Assert caller invokes drop_row on both. Assert each CompartmentReport.plasticity == 1.
(XOR-cancel detection lives in caller/SigmaTierRouter, not MailboxSoA; this test validates
the drop_row path is correctly called on cancel signal.)

**`mailbox_soa_intent_gate_strict`**
push_row with `intent = None`, `sigma_tier = TwigBranching`. Dispatcher returns Some(edge).
Assert edge in Vec<CausalEdge64>. Assert no ConsumerHandle callback fires (spy asserts zero calls).
This validates: intent=None compartments never reach Zone 3 regardless of sigma_tier.

**`mailbox_soa_plasticity_counter_monotonic`**
push_row x1. Run dispatch_cycle x100 with dispatcher always returning Some edge.
drop_row; check CompartmentReport.plasticity.0 == 100. Assert monotonic: plasticity never
decreases; saturating_add prevents overflow panic.

### §8.4 BindSpaceView tests

**`bindspace_view_zero_copy_borrow`**
Construct `Arc<BindSpaceColumns>` with 1000 rows.
Create `BindSpaceView` for rows 0..100, `ColumnMask::ALL_ABCD`.
Register a `GlobalAlloc` counter. Call `read_edges(0)` x1000.
Assert allocation count unchanged (zero heap allocs during read loop).

**`bindspace_view_write_token_gates_collapsegate`**
Construct `BindSpaceView` with a mock `CollapseGate` in Block mode.
Call `write_delta(0, ColumnDelta::Edge(CausalEdge64::zero()))` -> WriteToken.
Call `commit_with_token(token, &blocked_gate)` -> assert `Err(token_returned)`.
Call `read_edges(0)` -> assert still original value (no mutation occurred).

**`bindspace_view_must_use_token_clippy_gate`**
Static assertion: `#[must_use]` on WriteToken is enforced by Clippy via
`-D unused_must_use` in the CI lint gate. Verified in `build.rs` or `clippy.toml`.
(Test is a CI configuration check, not a runtime test.)

---

## §9 Risk Matrix

### HIGH — AttentionMaskActor Tokio latency vs cycle-speed budget

Tokio-backed actor adds 1-5 µs per Bind/Lookup/Resolve round trip. Acceptable for Tokio-shape
Sigma-2-8 (ms-scale dispatch). Breaks 200 ns cycle-speed budget for InMemoryMailbox Sigma-6-8
compartments (PR-CE64-MB-6).

**Mitigation (deferred, ratify via OQ-SHADOW before PR-CE64-MB-6):** Per-thread shadow
`AttentionMaskSnapshot` refreshed once per dispatcher tick (one Snapshot message, zero additional
Bind calls per compartment per cycle). Shadow read: zero async overhead. Actor write path
unchanged. Estimate: +150 LOC in SigmaTierRouter; zero interface change to AttentionMaskActor.

Benchmark before sprint-11+: N concurrent Bind calls under load; measure P99 latency;
confirm 1-5 µs bound or tighten the OQ.

### HIGH — Eviction broadcast subscriber-disconnect handling

`broadcast::Receiver` may return `RecvError::Lagged(n)` (insufficient polling) or
`RecvError::Closed` (actor restart). Compartments using stale slot references after eviction
will emit `CausalEdge64` with incorrect g_slot/w_slot values, causing silent corruption in
AriGraph SPO-G indexing.

**Mitigation:** Defensive re-resolve pattern is **mandatory** (see §2.5). Integration test
`attention_mask_broadcast_subscribers_see_evictions` validates happy path. Sprint-11+ adds
`attention_mask_broadcast_lagged_subscriber_recovers` stress test (high-throughput bind/unbind,
assert zero stale slot escapes). Broadcast channel capacity 256 (see OQ-BCAST-SIZE).

### MED — MailboxSoA\<N\> const-generic sizing

N is fixed at compile time. Mismatched N across crates creates incompatible types.

**Mitigation:** Type alias `pub type DefaultMailboxSoA = MailboxSoA<1024>` in par-tile lib.rs.
Document recommended N values: N=512 bevy; N=1024 supervisor default; N=4096 Sigma8-branching.
Sizing ratified via OQ-N (new OQ, see §11).

### MED — BindSpaceView column_mask future-proofing

8-bit u8 would cover current Columns A-H but not future I-Z.

**Mitigation:** Using u32 from day one (24 reserved bits, zero cost). If column count exceeds
32, upgrade to u64 via a single non-breaking change. Deferred to OQ-spec-naming.

### LOW — PlasticityCounter overflow at u64

Using `saturating_add`: counter stops at u64::MAX rather than wrapping or panicking.
At 10^9 emissions/second, MAX takes ~585 years. No action required.

---

## §10 Iron Rule Compliance

| Iron Rule | Compliance |
|---|---|
| **I-SUBSTRATE-MARKOV** | Preserved. AttentionMask rename tables are register files (identity -> slot), not transition kernels. No XOR-merge of identity fingerprints. CollapseGate MergeMode::Bundle on EdgeColumn preserves Markov guarantee. MergeMode::Xor used only for XOR-cancel of complementary emissions (not state transitions). |
| **I-NOISE-FLOOR-JIRAK** | PlasticityCounter thresholds and spawn priors in SigmaTierRouter (W7 spec) must cite Jirak-derived bounds when claiming statistical significance. This spec reports raw u64 counts only; significance testing deferred to W7. |
| **I-VSA-IDENTITIES** | Preserved. CausalEdge64 carries palette indices (architectural identity pointers), not bitpacked content. AttentionMask renames identity IDs to slot indices (still identities). BindSpaceView::write_delta emits typed ColumnDelta (not raw VSA operations). |
| **I1** (single mutation point) | Preserved. All compartment writes: write_delta -> WriteToken -> commit_with_token(gate). #[must_use] on WriteToken makes bypassing the gate a compile-time warning. BindSpaceView read accessors are &self only. Delta buffers are per-compartment-owned; never shared between rows. |
| **Method-on-carrier discipline** | Preserved. MailboxSoA::push_row / dispatch_cycle / drop_row are methods. AttentionMask::bind_g / lookup_g / resolve_g / touch_g / tick are methods. BindSpaceView::read_edges / write_delta / commit_with_token are methods. No free functions introduced. |
| **AGI-as-glove SoA invariant** | Preserved. Topic/Angle reads via BindSpaceView read accessors. Thinking writes via gated delta to MetaColumn. Planner writes via CausalEdge64 emission to EdgeColumn. Each MailboxSoA row is the runtime instantiation of one (role, G) at one cycle. |

---

## §11 Open Questions Surfaced for Meta-Review

**OQ-N (NEW — sizing):** Default N for `MailboxSoA<N>` across deployment contexts:
bevy (N=512?), supervisor Tokio default (N=1024?), Sigma8-branching (N=4096?).
Ratify at PR-CE64-MB-6 spec review when SigmaTierRouter scopes its requirements.

**OQ-SHADOW (NEW — cycle-speed):** Per-thread shadow `AttentionMaskSnapshot` refreshed
once per dispatcher tick: sufficient for correctness? Eviction events during a single tick
that invalidate in-flight slot references need analysis. Ratify before InMemoryMailbox
backing in PR-CE64-MB-6.

**OQ-BCAST-SIZE (NEW — channel capacity):** Broadcast channel default 256: sufficient under
Sigma8-branching load? Measure lagged-receiver frequency under load tests in PR-CE64-MB-6;
resize to 1024 if needed before merge.

**Parent plan OQ-2 (ghost edge NARS decay, unresolved):** This spec assumes fixed Pearl rung 3
for ghost persistence. W5 (AriGraph spec) owns the decision on whether NARS confidence drift
applies to ghosts. Must be ratified at meta-review to avoid W5/W6 spec inconsistency.

**Parent plan OQ-3 (plasticity granularity, partially resolved):** This spec implements
bit-counter per emission (high-frequency, MailboxSoA-side) plus NARS truth-refine at AriGraph
commit (low-frequency, W5 / lance-graph-supervisor responsibility). Confirm or refine at
meta-review.

---

## §12 Deliverable Mapping

| D-id | Description | Spec section |
|---|---|---|
| D-CE64-MB-4 | `AttentionMask` struct + accessors + LRU + broadcast | §2 full |
| D-CE64-MB-5 | `AttentionMaskActor` ractor singleton | §3 full |
| D-CE64-MB-6 | Tests: round-trip, LRU order, broadcast subscribers | §8.1 + §8.2 |
| D-CE64-MB-7 | `MailboxSoA<N>` struct + lifecycle methods | §4 full |
| D-CE64-MB-8 | `BindSpaceView<'_>` + ColumnMask + WriteToken | §5 full |
| D-CE64-MB-9 | Tests: lifecycle, XOR-cancel, plasticity, intent gate, zero-copy | §8.3 + §8.4 |

D-CE64-MB-1/2/3 are owned by W1/W2/W4 and are merge gates for this PR.
D-CE64-MB-8 is shared (W4 owns `Arc<BindSpaceColumns>`; this spec owns `BindSpaceView` as consumer).

---

*End of spec — PR-CE64-MB-5. Worker W6, sprint-log-10, 2026-05-14.*
