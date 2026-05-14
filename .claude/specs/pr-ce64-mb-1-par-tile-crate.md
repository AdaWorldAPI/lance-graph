# PR-CE64-MB-1 — `crates/par-tile/` NEW Crate Spec

> **Status:** Draft (2026-05-14, sprint-log-10, W1)
> **Deliverable IDs:** D-CE64-MB-4 (AttentionMask + LRU), D-CE64-MB-5 (AttentionMaskActor), D-CE64-MB-7 (MailboxSoA<N> lifecycle), D-CE64-MB-8 (BindSpaceView)
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 (par-tile NEW) + §7 (PR-CE64-MB-1)
> **Worker:** W1 (Sonnet, sprint-log-10)
> **LOC estimate:** ~1500 (per parent plan §7 PR-CE64-MB-1 row)
> **Risk:** Low — pure new crate, no consumers yet (per parent plan §7 table)
> **Delta discipline:** Every architectural decision cites parent plan §X.Y. New material is confined to §9 test plan, §10 risk matrix, and §3 module decomposition detail.

---

## §1 Crate Purpose Statement

`crates/par-tile/` is the **diamond dep-graph apex** of the Ada cognitive
substrate. ndarray, lance-graph, and bevy all depend on par-tile; par-tile
itself depends only on `std`, `ractor` (feature-gated), and optionally
`crossbeam` for alternate channel backings. Zero transitive exposure to
BLAS, MKL, lance, arrow, or datafusion. This constraint is structural and
enforced by the `[features]` design: the default build has no async runtime,
no columnar-storage dependency, and no SIMD requirement — making par-tile
safe to import from embedded contexts, Wasm, and bevy's plugin tree without
pulling in the full lance-graph stack.

The crate provides three things:

1. **`Mailbox<T>` trait** — the Zone-1-to-Zone-2 communication abstraction
   with three concrete backings (in-memory lock-free, Tokio channel, Supabase
   egress wrapper) per parent plan §6 column "this plan adds".
2. **`AttentionMask` SoA** — the session-ephemeral rename register file that
   maps unbounded architectural identities (OGIT domain u32, WitnessId u32,
   StyleId u32) to hot-path physical slots (5-bit G, 6-bit W, 8-bit style)
   per parent plan §4.
3. **`MailboxSoA<N>`** — the typed SoA whose rows ARE the reasoning
   compartments, each owning a delta buffer (~1 KB), a zero-copy
   `BindSpaceView<'_>` into the shared BindSpace, and an outbound
   `CausalEdge64` emission channel per parent plan §5.

---

## §2 Cargo.toml Shape

    [package]
    name        = "par-tile"
    version     = "0.1.0"
    edition     = "2021"
    rust-version = "1.95"
    description  = "Diamond-apex mailbox + attention-mask + compartment SoA crate (PR-CE64-MB-1)"
    # Implementation target: sprint-11+ execution (spec ratified sprint-10)
    # Parent plan: .claude/plans/causaledge64-mailbox-rename-soa-v1.md §6 + §7

    [dependencies]
    # Zero mandatory runtime deps — everything feature-gated.

    # ractor feature-gating: Zone-2 TokioMailbox backing + AttentionMaskActor
    ractor = { version = "0.14", optional = true, default-features = false, features = ["tokio_runtime"] }

    # crossbeam: alternate MPSC channel for InMemoryMailbox (lock-free, Miri-friendly)
    crossbeam-channel = { version = "0.5", optional = true }

    # tokio: needed at TokioMailbox boundary only; NOT imported in Zone-1 cycle paths.
    # Per lance-graph-supervisor I-2 rule: tokio::spawn ONLY at outbound supervisor boundary.
    tokio = { version = "1", features = ["rt", "sync", "time", "macros"], optional = true }

    # thiserror: typed errors
    thiserror = "1"

    # tracing: structured spans; optional so embedded targets compile without it
    tracing = { version = "0.1", optional = true }

    # causal-edge: CausalEdge64 u64 packed type (zero runtime deps — pure bit manipulation).
    # Does NOT violate diamond invariant: causal-edge has no lance/arrow/datafusion deps.
    causal-edge = { path = "../causal-edge" }

    [build-dependencies]
    # Used by build.rs dep guard to validate no forbidden transitive deps.
    cargo_metadata = { version = "0.18", optional = true }

    [features]
    default = []

    # tokio-backing: enables TokioMailbox<T> (Zone-2 callcenter-shape)
    # and AttentionMaskActor ractor singleton (wraps AttentionMask, receives BindRequest).
    # Requires: ractor + tokio.
    tokio-backing = ["dep:ractor", "dep:tokio"]

    # crossbeam-backing: enables InMemoryMailbox<T> backed by crossbeam-channel MPSC.
    # Default InMemoryMailbox uses std::sync::mpsc; this feature upgrades to lock-free crossbeam.
    crossbeam-backing = ["dep:crossbeam-channel"]

    # vendored-rayon: deferred per parent plan §11 OQ-5. DO NOT enable until profiling
    # reveals throughput cliff on std::thread::scope dispatch.
    vendored-rayon = []

    # tracing-spans: enables span instrumentation in dispatch_cycle + bind/evict paths.
    tracing-spans = ["dep:tracing"]

    # dep-guard: enables build.rs forbidden-dep check (cargo_metadata required).
    dep-guard = ["dep:cargo_metadata"]

    [dev-dependencies]
    tokio     = { version = "1", features = ["rt-multi-thread", "macros"] }
    proptest  = "1"
    crossbeam-channel = "0.5"

**Rationale for rust-version = "1.95":** matches the workspace pin
(post-#325 bump to 1.94.1 + anticipated 1.95 release timeline).

**No BLAS/MKL/lance/arrow/datafusion deps — ever.** Any PR introducing
such a dep automatically breaks the diamond-apex guarantee.

**ractor version pin = "0.14":** matches `lance-graph-supervisor/Cargo.toml`
(confirmed by reading that file). Same version in both crates ensures Cargo
resolves a single ractor instance, preserving message type unification.

---

## §3 Module Layout

    crates/par-tile/
    ├── Cargo.toml
    ├── build.rs               # dep guard: panic if forbidden transitive dep detected
    └── src/
        ├── lib.rs              # re-exports; feature-gated public surface
        ├── error.rs            # MailboxError, AttentionError
        ├── mailbox.rs          # Mailbox<T> trait + 3 backings
        ├── attention_mask.rs   # AttentionMask struct + LRU + bind/lookup/resolve + EvictionMsg
        ├── attention_actor.rs  # AttentionMaskActor ractor singleton (tokio-backing only)
        ├── mailbox_soa.rs      # MailboxSoA<N> + lifecycle: push_row, dispatch_cycle, drop_row
        └── bindspace_view.rs   # BindSpaceView<'_>: row range + column mask zero-copy borrow

### 3.1 `src/lib.rs`

    pub mod error;
    pub mod mailbox;
    pub mod attention_mask;
    pub mod mailbox_soa;
    pub mod bindspace_view;

    #[cfg(feature = "tokio-backing")]
    pub mod attention_actor;

    pub use error::{AttentionError, MailboxError};
    pub use mailbox::{InMemoryMailbox, Mailbox, Receiver, SupabaseSubMailbox};
    pub use attention_mask::{
        AttentionMask, EvictionMsg, GrammarAlphabet, OgitDomainId, StyleId, WitnessId,
        G_SLOTS, W_SLOTS, STYLE_SLOTS,
    };
    pub use mailbox_soa::{
        Budget, ConsumerHandle, DeltaBuffer, MailboxId, MailboxSoA,
        PlasticityCounter, RoleId, SigmaTier, TemporalWindow,
    };
    pub use bindspace_view::ColumnMask;
    pub use causal_edge::CausalEdge64;

    #[cfg(feature = "tokio-backing")]
    pub use attention_actor::{AttentionMaskActor, BindReply, BindRequest, BindKind};

    #[cfg(feature = "tokio-backing")]
    pub use mailbox::TokioMailbox;

### 3.2–3.6 — module role assignments

Per parent plan §6 + §4 + §5:

- `mailbox.rs` — Mailbox<T> trait + 3 backings (parent plan §6)
- `attention_mask.rs` — AttentionMask + LRU + bind/lookup/resolve (parent plan §4)
- `attention_actor.rs` — ractor singleton; tokio-backing feature only (parent plan §4 D-CE64-MB-5)
- `mailbox_soa.rs` — MailboxSoA<N> + lifecycle (parent plan §5 D-CE64-MB-7)
- `bindspace_view.rs` — BindSpaceView<'_> + ColumnMask (parent plan §5 D-CE64-MB-8)

---

## §4 `Mailbox<T>` Trait — Per Parent Plan §6

Three backings map to three zones per parent plan §0 Zone framing.

    // src/mailbox.rs

    use crate::error::MailboxError;

    /// Zone-agnostic mailbox abstraction.
    /// Zone-1 (20-200 ns): InMemoryMailbox
    /// Zone-2 (us-ms):     TokioMailbox
    /// Zone-3 (2-200 ms):  SupabaseSubMailbox
    pub trait Mailbox<T: Send + 'static>: Send + Sync {
        /// Send a message; non-blocking Zone-1, async-ready Zone-2.
        /// Returns Err if backing is at capacity.
        fn send(&self, msg: T) -> Result<(), MailboxError>;

        /// Non-blocking receive. Returns None if backing is empty.
        fn try_recv(&self) -> Option<T>;

        /// Subscribe a new receiver to this mailbox's broadcast stream.
        fn subscribe(&self) -> Receiver<T>;

        /// Approximate message count (for budget + pruning decisions).
        fn len(&self) -> usize;

        /// Returns true if no messages are queued.
        fn is_empty(&self) -> bool { self.len() == 0 }
    }

    /// Receiver handle — wraps whichever backing channel is in use.
    pub enum Receiver<T> {
        Std(std::sync::mpsc::Receiver<T>),
        #[cfg(feature = "crossbeam-backing")]
        Crossbeam(crossbeam_channel::Receiver<T>),
        #[cfg(feature = "tokio-backing")]
        Tokio(tokio::sync::broadcast::Receiver<T>),
    }

    // --- Backing 1: InMemoryMailbox (Zone-1, 20-200 ns) -------------------
    // Per parent plan §6: "InMemoryMailbox cycle-speed".

    pub struct InMemoryMailbox<T: Send + 'static> {
        #[cfg(not(feature = "crossbeam-backing"))]
        sender:   std::sync::Arc<std::sync::Mutex<std::sync::mpsc::SyncSender<T>>>,
        #[cfg(feature = "crossbeam-backing")]
        sender:   crossbeam_channel::Sender<T>,
        capacity: usize,
    }

    impl<T: Send + 'static> InMemoryMailbox<T> {
        /// Construct with given channel capacity (0 = unbounded).
        pub fn new(capacity: usize) -> (Self, Receiver<T>) { todo!() }
    }

    // --- Backing 2: TokioMailbox (Zone-2, us-ms) -------------------------
    // Per parent plan §6: wraps existing CallcenterSupervisor channels (#366 S7-W3).
    // Only compiled with tokio-backing feature.

    #[cfg(feature = "tokio-backing")]
    pub struct TokioMailbox<T: Send + 'static + Clone> {
        sender: tokio::sync::broadcast::Sender<T>,
    }

    #[cfg(feature = "tokio-backing")]
    impl<T: Send + 'static + Clone> TokioMailbox<T> {
        /// Construct; capacity is the broadcast ring buffer depth.
        pub fn new(capacity: usize) -> (Self, Receiver<T>) { todo!() }
    }

    // --- Backing 3: SupabaseSubMailbox (Zone-3 egress wrapper) -----------
    // Per parent plan §6: "calls drain.rs / supabase realtime publish".
    // Per parent plan §10: Zone-3 surface completely unchanged.
    // Holds Arc<dyn Zone3EgressSink<T>> to stay dep-free from lance-graph-callcenter.

    pub trait Zone3EgressSink<T: Send>: Send + Sync {
        fn emit(&self, msg: T) -> Result<(), MailboxError>;
    }

    pub struct SupabaseSubMailbox<T: Send + 'static> {
        sink:  std::sync::Arc<dyn Zone3EgressSink<T>>,
        queue: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<T>>>,
    }

**Key invariant**: `TokioMailbox` wraps the existing broadcast sender shape
from `lance-graph-supervisor`. Dependency arrow: `lance-graph-supervisor`
depends on `par-tile` (not the reverse) once the supervisor gains its
SigmaTierRouter in PR-CE64-MB-6.

---

## §5 `AttentionMask` SoA — Per Parent Plan §4

Extends parent plan §4 struct with `evict_notifications` field shape detail.

    // src/attention_mask.rs

    pub const G_SLOTS: usize = 32;
    pub const W_SLOTS: usize = 64;
    pub const STYLE_SLOTS: usize = 256;

    /// Session-ephemeral rename register file.
    /// Total struct size: ~2 KB (fits L1 dcache).
    /// Per parent plan §4 struct verbatim.
    #[repr(C, align(64))]
    pub struct AttentionMask {
        /// 5-bit slot -> architectural OGIT domain u32.
        pub g_slots:        [Option<OgitDomainId>; G_SLOTS],
        /// 6-bit slot -> architectural witness palette id.
        pub w_slots:        [Option<WitnessId>; W_SLOTS],
        /// 8-bit slot -> architectural ThinkingStyle / cognitive primitive / verb id.
        pub style_slots:    [Option<StyleId>; STYLE_SLOTS],
        /// Active grammar — selects which alphabet style_slots index into.
        pub active_grammar: GrammarAlphabet,
        /// LRU clock per slot table.
        pub g_lru:          [u32; G_SLOTS],
        pub w_lru:          [u32; W_SLOTS],
        pub style_lru:      [u32; STYLE_SLOTS],
        /// Monotonic session cycle counter (wraps at u32::MAX).
        pub cycle:          u32,
        /// Broadcast sender for eviction notifications.
        /// Per parent plan §4: "broadcast::Sender<EvictionMsg>".
        pub evict_notifications: EvictionSender,
    }

    /// Eviction message broadcast on slot reclaim.
    #[derive(Clone, Debug)]
    pub struct EvictionMsg {
        pub kind:          SlotKind,
        pub slot:          u8,
        pub evicted_g:     Option<OgitDomainId>,
        pub evicted_w:     Option<WitnessId>,
        pub evicted_style: Option<StyleId>,
        pub at_cycle:      u32,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum SlotKind { G, W, Style }

    /// Broadcast sender — keeps AttentionMask dep-free from tokio
    /// unless tokio-backing feature is active.
    pub enum EvictionSender {
        Std(std::sync::Arc<std::sync::Mutex<Vec<std::sync::mpsc::SyncSender<EvictionMsg>>>>),
        #[cfg(feature = "tokio-backing")]
        Tokio(tokio::sync::broadcast::Sender<EvictionMsg>),
    }

    #[derive(Copy, Clone, Eq, PartialEq)]
    pub enum GrammarAlphabet {
        PlannerClusters12,      // 12 ThinkingStyle clusters (lance-graph-planner)
        CognitivePrimitives34,  // 34 hpc/styles entries (ndarray)
        VerbsTekamolo144,       // 12 x 12 verb compositions (German cognitive grammar)
        FullStyle36Plus,        // 36 contract::thinking enum + YAML extensions
    }

    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)] pub struct OgitDomainId(pub u32);
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)] pub struct WitnessId(pub u32);
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)] pub struct StyleId(pub u32);

### 5.1 API Sketch — `impl AttentionMask`

Per parent plan §4 rename protocol:

    impl AttentionMask {
        /// Construct with empty slot tables.
        pub fn new(eviction_capacity: usize) -> Self { todo!() }

        // --- G-slot (OGIT domain, 5-bit) ---------------------------------

        /// Look up physical G-slot for an architectural domain.
        /// Returns Some(slot) if bound, None otherwise.
        /// Per parent plan §4: lookup_g.
        pub fn lookup_g(&self, id: OgitDomainId) -> Option<u8> { todo!() }

        /// Bind architectural OGIT domain to next free G-slot.
        /// Evicts LRU on pressure; broadcasts eviction notification.
        /// Per parent plan §4: bind_g -> (slot, Option<evicted>).
        pub fn bind_g(&mut self, id: OgitDomainId) -> (u8, Option<OgitDomainId>) { todo!() }

        /// Resolve physical G-slot -> architectural domain id.
        /// Per parent plan §4: resolve_g.
        pub fn resolve_g(&self, slot: u8) -> Option<OgitDomainId> { todo!() }

        /// Bump LRU clock for slot (call on every successful access).
        /// Per parent plan §4: touch_g.
        pub fn touch_g(&mut self, slot: u8) { self.g_lru[slot as usize] = self.cycle; }

        // --- W-slot (Witness palette, 6-bit) -----------------------------
        pub fn lookup_w(&self, id: WitnessId) -> Option<u8> { todo!() }
        pub fn bind_w(&mut self, id: WitnessId) -> (u8, Option<WitnessId>) { todo!() }
        pub fn resolve_w(&self, slot: u8) -> Option<WitnessId> { todo!() }
        pub fn touch_w(&mut self, slot: u8) { self.w_lru[slot as usize] = self.cycle; }

        // --- Style-slot (ThinkingStyle / cognitive primitive, 8-bit) -----
        pub fn lookup_style(&self, id: StyleId) -> Option<u8> { todo!() }
        pub fn bind_style(&mut self, id: StyleId) -> (u8, Option<StyleId>) { todo!() }
        pub fn resolve_style(&self, slot: u8) -> Option<StyleId> { todo!() }
        pub fn touch_style(&mut self, slot: u8) { self.style_lru[slot as usize] = self.cycle; }

        // --- Session lifecycle -------------------------------------------

        /// Advance session cycle counter. Per parent plan §4: tick.
        pub fn tick(&mut self) { self.cycle = self.cycle.wrapping_add(1); }

        /// Subscribe to eviction notifications.
        pub fn subscribe_evictions(&self) -> EvictionReceiver { todo!() }

        // --- Internal helpers --------------------------------------------

        /// Find the LRU slot (smallest lru value = oldest access).
        fn lru_victim(lru: &[u32]) -> usize {
            lru.iter().enumerate().min_by_key(|(_, &t)| t).map(|(i, _)| i).unwrap_or(0)
        }

        /// Broadcast eviction notification to all subscribers.
        fn broadcast_eviction(&self, msg: EvictionMsg) { todo!() }
    }

**LRU eviction policy** (per parent plan §4):
- Evict slot with smallest `*_lru[i]` when all slots occupied.
- `touch_*` on every successful lookup.
- Evicted slot's architectural identity persists in AriGraph forever (hibernation policy per parent plan §4).
- Wrap-around protection: on `tick()` when cycle wraps to 0, renormalize all LRU timestamps by subtracting `u32::MAX/2`.

---

## §6 `MailboxSoA<N>` — Per Parent Plan §5

    // src/mailbox_soa.rs

    use std::ops::Range;
    use crate::bindspace_view::{BindSpaceView, ColumnMask};
    use crate::attention_mask::AttentionMask;
    use causal_edge::CausalEdge64;

    /// Reasoning compartments as SoA rows.
    /// NOT individually-spawned ractor actors — rows dispatched per-cycle.
    /// Per parent plan §5.
    pub struct MailboxSoA<const N: usize> {
        pub ids:                 [MailboxId; N],
        pub roles:               [RoleId; N],
        pub temporals:           [TemporalWindow; N],
        pub sigma_tiers:         [SigmaTier; N],
        pub bindspace_views:     [BindSpaceView<'static>; N],
        pub deltas:              [DeltaBuffer; N],
        pub witness_outs:        [Sender<CausalEdge64>; N],
        pub intents:             [Option<ConsumerHandle>; N],
        pub parents:             [Option<MailboxId>; N],
        pub budgets:             [Budget; N],
        pub plasticity_counters: [PlasticityCounter; N],
        active: usize,
    }

    impl<const N: usize> MailboxSoA<N> {
        /// Initialise with all slots vacant.
        pub fn new() -> Self { todo!() }

        /// Spawn a new compartment row.
        /// Per parent plan §5 lifecycle step 1.
        /// Returns Err if all N slots are occupied.
        pub fn push_row(
            &mut self,
            role:           RoleId,
            temporal:       TemporalWindow,
            sigma_tier:     SigmaTier,
            bindspace_view: BindSpaceView<'static>,
            intent:         Option<ConsumerHandle>,
        ) -> Result<MailboxId, MailboxError> { todo!() }

        /// Run one dispatch cycle.
        /// Per parent plan §5 lifecycle step 2: scans sigma_tiers[], picks active
        /// compartments, calls dispatch per compartment, yields (MailboxId, CausalEdge64).
        pub fn dispatch_cycle(
            &mut self,
            current_cycle: u32,
            attn: &mut AttentionMask,
        ) -> impl Iterator<Item = (MailboxId, CausalEdge64)> + '_ { todo!() }

        /// Prune a compartment row.
        /// Per parent plan §5 lifecycle step 5: reclaims slot, delta buffer dropped,
        /// plasticity counter returned for caller to persist.
        /// If unresolved, emit ghost-edge at Pearl rung 3 to AriGraph.
        pub fn drop_row(&mut self, id: MailboxId) -> Result<PlasticityCounter, MailboxError> { todo!() }

        /// Emit for a single compartment after dispatch.
        /// Per parent plan §5 lifecycle step 3: writes to witness_outs channel.
        pub fn emit_one(&self, idx: usize, edge: CausalEdge64) -> Result<(), MailboxError> { todo!() }

        /// Increment plasticity counter for (row, G) co-occurrence.
        /// Per parent plan §9 E-CE64-MB-10: Hebbian "fired together wired together".
        pub fn increment_plasticity(&mut self, idx: usize) { todo!() }

        /// Check whether a compartment's temporal window is active.
        pub fn is_active(&self, idx: usize, current_cycle: u32) -> bool { todo!() }

        /// Check whether a compartment's budget is exhausted.
        pub fn is_budget_exhausted(&self, idx: usize) -> bool { todo!() }
    }

**Per-compartment memory footprint** (per parent plan §5 table):

| Field | Size |
|---|---|
| MailboxId | 8 B |
| RoleId | 4 B |
| TemporalWindow | 16 B |
| SigmaTier | 1 B |
| BindSpaceView | 24 B |
| DeltaBuffer | ~1024 B |
| witness_outs Sender | 24 B |
| intent | 16 B |
| parent | 8 B |
| Budget | 8 B |
| PlasticityCounter | 8 B |
| **Total** | **~1.2 KB** |

Per parent plan §5: 10,000 concurrent compartments = ~12 MB. Not re-derived.

---

## §7 `BindSpaceView<'_>` — Per Parent Plan §5 D-CE64-MB-8

    // src/bindspace_view.rs

    use std::ops::Range;

    /// Zero-copy borrow into shared BindSpace columns.
    /// Per parent plan §5 D-CE64-MB-8: row-range + column-mask + Arc<BindSpace>.
    ///
    /// par-tile does NOT depend on cognitive-shader-driver; borrow is via
    /// raw pointer + lifetime. Callers hold Arc<BindSpace>.
    /// Type is Copy so SoA fields can be trivially stack-cloned.
    #[derive(Copy, Clone)]
    pub struct BindSpaceView<'a> {
        /// Raw pointer into caller Arc<BindSpace> memory (aligned).
        columns_ptr: std::ptr::NonNull<u8>,
        /// Row range this compartment is responsible for.
        /// Per parent plan §5: `rows: Range<usize>`.
        pub rows:        Range<usize>,
        /// Column-subset filter — which of Columns A-H this compartment accesses.
        /// Per parent plan §5: `column_mask: ColumnMask`.
        pub column_mask: ColumnMask,
        _marker: std::marker::PhantomData<&'a ()>,
    }

    /// Bit-mask over BindSpace Columns A-H (per `bindspace-columns-v1.md`).
    ///
    /// Bit mapping:
    ///   A = FingerprintColumns   (bit 0)
    ///   B = QualiaColumn         (bit 1)
    ///   C = MetaColumn           (bit 2)
    ///   D = EdgeColumn           (bit 3)  CausalEdge64 x 8 per row
    ///   E = OntologyDelta        (bit 4)
    ///   F = AwarenessColumn      (bit 5)
    ///   G = ModelBindingColumn   (bit 6)
    ///   H = TypeColumn           (bit 7)  EntityTypeId u16 per row
    #[derive(Copy, Clone, Default)]
    pub struct ColumnMask(pub u8);

    impl ColumnMask {
        pub const A: Self = Self(0b0000_0001);
        pub const B: Self = Self(0b0000_0010);
        pub const C: Self = Self(0b0000_0100);
        pub const D: Self = Self(0b0000_1000);
        pub const E: Self = Self(0b0001_0000);
        pub const F: Self = Self(0b0010_0000);
        pub const G: Self = Self(0b0100_0000);
        pub const H: Self = Self(0b1000_0000);
        pub const ALL: Self = Self(0b1111_1111);

        /// Returns true if any column in `other` overlaps with this mask.
        pub fn overlaps(self, other: Self) -> bool { self.0 & other.0 != 0 }
    }

    impl<'a> BindSpaceView<'a> {
        /// Construct from caller-managed pointer + row range + column mask.
        /// Safety: caller guarantees pointer alive for 'a (Arc<BindSpace> keeps it).
        /// Per parent plan §5 D-CE64-MB-8: zero-copy borrow into Arc<BindSpace>.
        pub unsafe fn from_raw(
            ptr: std::ptr::NonNull<u8>,
            rows: Range<usize>,
            column_mask: ColumnMask,
        ) -> Self { todo!() }

        /// Row count for this view.
        pub fn row_count(&self) -> usize { self.rows.len() }

        /// True if this view's column mask overlaps another view's mask on the
        /// same row range (indicates conflicting mutable borrow).
        pub fn conflicts_with(&self, other: &BindSpaceView<'_>) -> bool {
            let row_overlap = self.rows.start < other.rows.end
                && other.rows.start < self.rows.end;
            row_overlap && self.column_mask.overlaps(other.column_mask)
        }
    }

**Column type cross-reference**: `cognitive-shader-driver::bindspace` provides
concrete column layouts (Columns A-H per `bindspace-columns-v1.md`). par-tile
does not import that crate; the borrower (lance-graph-supervisor, bevy plugin)
holds the Arc and passes raw pointer + layout offsets.

---

## §8 Dependency Graph Diagram

Per parent plan §6, §7:

    std + ractor (optional) + causal-edge (zero-dep u64 type)
           |
           v
     +-----------+
     | par-tile  |   <- pure apex: no BLAS/MKL/lance/arrow/datafusion
     +-----+-----+
           |
     +-----+-----------+
     |     |           |
     v     v           v
  ndarray  lance-graph  bevy
  (CLAM,   (AriGraph,   (NdarrayCullPlugin
   CAM-PQ)  planner,     frustum cull,
             callcenter)  PR-CE64-MB-7)

  Zone-1 (20-200 ns):
    InMemoryMailbox, MailboxSoA, AttentionMask
    [no async, no tokio in this path]

  Zone-2 (us-ms):
    TokioMailbox -> CallcenterSupervisor (#366 S7-W3)
    [tokio-backing feature required]

  Zone-3 (2-200 ms):
    SupabaseSubMailbox -> Zone3EgressSink<T> -> drain.rs/postgrest.rs/grpc.rs
    [all serialization at Zone-3 boundary only; per parent plan §0 iron rule]

  Explicit non-deps:
    par-tile NEVER imports: lance, lance-linalg, arrow, datafusion,
      lance-graph-callcenter, lance-graph, cognitive-shader-driver
    Enforced by build.rs dep guard (§9.7).

---

## §9 Test Plan

Target: **~30 tests** across property tests, unit tests, and integration smoke tests.

### 9.1 Property Tests — `Mailbox<T>` ordering

| Test | Strategy | Pass criterion |
|---|---|---|
| `prop_inmemory_send_recv_ordering` | Vec<u64> messages; send all, recv all | recv order == send order (FIFO) |
| `prop_inmemory_capacity_backpressure` | fill to capacity, one more send | Err(MailboxError::Full) |
| `prop_inmemory_concurrent_senders` | 4 threads x N sends | all N*4 messages received exactly once |
| `prop_supabase_sub_fire_and_forget` | send M messages; mock sink | sink.count() == M |

### 9.2 Property Tests — LRU eviction ordering

| Test | Strategy | Pass criterion |
|---|---|---|
| `prop_attention_mask_lru_evicts_oldest` | bind 32 G-slots; bind one more; random touch order | evicted slot has min g_lru value |
| `prop_attention_mask_resolve_after_eviction` | bind, touch, evict; resolve evicted slot | returns None |
| `prop_attention_mask_no_duplicate_slots` | bind 32 distinct domains | each -> unique slot |
| `prop_lru_clock_monotonic` | tick N times | cycle == N % u32::MAX |

### 9.3 Property Tests — rename round-trip

| Test | Strategy | Pass criterion |
|---|---|---|
| `prop_rename_round_trip_g` | random OgitDomainId; bind then resolve | resolve(bind(id)) == Some(id) |
| `prop_rename_round_trip_w` | same for WitnessId | same |
| `prop_rename_round_trip_style` | same for StyleId | same |
| `prop_bind_rebind_same_slot` | bind id, touch, bind same id again | same slot returned (idempotent) |
| `prop_eviction_broadcast_received` | bind 32 G-slots + 1 more; subscribe first | EvictionMsg with correct evicted id |

### 9.4 Unit Tests — `MailboxSoA<N>` lifecycle

| Test | Scenario | Pass criterion |
|---|---|---|
| `test_push_row_returns_unique_id` | push N rows | all MailboxIds distinct |
| `test_drop_row_reclaims_slot` | push, drop, push | slot reused |
| `test_dispatch_cycle_active_only` | 3 rows; expire 1 temporal window | only 2 emit |
| `test_budget_exhausted_prune` | budget=1; dispatch once | row pruned |
| `test_xor_cancel_complementary` | two rows with identical emission | net = 0 (cancelled) |
| `test_plasticity_counter_monotonic` | dispatch 5 times | plasticity_counter >= 5 |
| `test_intent_gate_none_never_zone3` | intent=None; dispatch | Zone3EgressSink never called |
| `test_intent_gate_some_emits_zone3` | intent=Some(_); dispatch | Zone3EgressSink called once per emission |
| `test_push_row_at_capacity_err` | fill all N slots; push one more | Err(MailboxError::Full) |

### 9.5 Unit Tests — `BindSpaceView`

| Test | Scenario | Pass criterion |
|---|---|---|
| `test_column_mask_overlap_detection` | overlapping masks, same rows | conflicts_with() == true |
| `test_column_mask_no_overlap` | disjoint column masks | conflicts_with() == false |
| `test_column_mask_disjoint_rows` | overlapping masks, disjoint rows | conflicts_with() == false |
| `test_row_count` | range 10..20 | row_count() == 10 |

### 9.6 Integration Smoke Tests

| Test | Scenario | Pass criterion |
|---|---|---|
| `test_attention_mask_actor_round_trip` | (tokio-backing) spawn actor; BindRequest; BindReply | slot in 0..32; no panic |
| `test_mailbox_soa_full_lifecycle` | push->dispatch->emit->drop x5 | all emissions captured; active == 0 |

### 9.7 Build-time Dep Guard

    // build.rs
    #[cfg(feature = "dep-guard")]
    fn main() {
        let meta = cargo_metadata::MetadataCommand::new().exec().unwrap();
        let forbidden = ["lance", "arrow", "datafusion", "blas", "mkl"];
        for pkg in &meta.packages {
            for dep in &pkg.dependencies {
                let n = dep.name.to_lowercase();
                for f in &forbidden {
                    assert!(!n.starts_with(f),
                        "par-tile dep '{}' violates diamond-apex invariant", dep.name);
                }
            }
        }
    }
    #[cfg(not(feature = "dep-guard"))]
    fn main() {}

Alternatively: `cargo deny` with the same deny-list in `.cargo/deny.toml`.

---

## §10 Risk Matrix

| # | Risk | Score | Mitigation |
|---|---|---|---|
| R-1 | **vendored-rayon scope creep** — attempting to vendor rayon inflates crate to 3500+ LOC and introduces profiled-unsafe surface. | **High if attempted; Low if deferred per OQ-5** | Start with `std::thread::scope` + crossbeam-backing. `vendored-rayon` feature = no-op panic stub until profiling shows throughput cliff (<5M rows/sec). |
| R-2 | **BindSpaceView raw-pointer safety** — `NonNull<u8>` into caller memory dangles if Arc dropped before view's lifetime. | **Medium** | `from_raw` is `unsafe`; add `#[must_use] ArcBindSpaceGuard` that bundles Arc + view. Miri integration test catches use-after-free. |
| R-3 | **ractor version pin mismatch** — different ractor versions in par-tile vs lance-graph-supervisor breaks message type unification. | **Medium** | Pin `ractor = "0.14"` (confirmed match with lance-graph-supervisor/Cargo.toml). `cargo tree -p par-tile | grep ractor` must show single version in CI. |
| R-4 | **`AttentionMask` u32 cycle counter wrap** — at 200ns/tick, wrap occurs at ~860s; stale LRU timestamps cause incorrect eviction. | **Low** (long sessions uncommon) | On wrap (cycle == 0 post-increment): renormalize all LRU arrays by subtracting `u32::MAX/2`. Property test `prop_lru_clock_monotonic` exercises wrap. |
| R-5 | **EpiphanyEscalate bypassing Zone-3 intent gate** — S9-S10 compartment with intent=None could reach L4 planner directly, serializing across zone boundary. | **Low** (architectural) | `dispatch_cycle`: if `EpiphanyEscalate && intent.is_none()` emit CausalEdge64 with Pearl rung=5 to `witness_outs` only. Zone-2 ractor supervisor routes to L4. Tests: `test_intent_gate_none_never_zone3` + `test_epiphany_stays_zone2`. |

---

## §11 Files-to-Touch Table

| File | Action | LOC est | Notes |
|---|---|---|---|
| `crates/par-tile/Cargo.toml` | CREATE | ~55 | §2 feature-gated deps |
| `crates/par-tile/build.rs` | CREATE | ~30 | §9.7 dep guard |
| `crates/par-tile/src/lib.rs` | CREATE | ~40 | §3.1 re-exports |
| `crates/par-tile/src/error.rs` | CREATE | ~30 | MailboxError, AttentionError |
| `crates/par-tile/src/mailbox.rs` | CREATE | ~300 | §4 Mailbox<T> + 3 backings |
| `crates/par-tile/src/attention_mask.rs` | CREATE | ~350 | §5 AttentionMask + LRU + EvictionMsg |
| `crates/par-tile/src/attention_actor.rs` | CREATE | ~150 | §5 ractor singleton (tokio-backing) |
| `crates/par-tile/src/mailbox_soa.rs` | CREATE | ~350 | §6 MailboxSoA<N> + lifecycle |
| `crates/par-tile/src/bindspace_view.rs` | CREATE | ~120 | §7 BindSpaceView + ColumnMask |
| `crates/par-tile/tests/mailbox_prop.rs` | CREATE | ~150 | §9.1 property tests |
| `crates/par-tile/tests/attention_mask_prop.rs` | CREATE | ~150 | §9.2+§9.3 property tests |
| `crates/par-tile/tests/mailbox_soa_unit.rs` | CREATE | ~120 | §9.4 unit tests |
| `crates/par-tile/tests/bindspace_view_unit.rs` | CREATE | ~60 | §9.5 unit tests |
| `crates/par-tile/tests/integration_smoke.rs` | CREATE | ~60 | §9.6 integration tests |
| `Cargo.toml` (workspace root) | EDIT | +2 | Add `"crates/par-tile"` to members |
| **Total** | | **~1965** | Source LOC ~1425 (within parent plan §7 1500 LOC); test LOC ~540 separate |

---

## §12 Per-Method API Sketch with Doc Comments

Full signatures + 1-line semantics for every public method not already given
in §4-§7.

### `error.rs`

    /// Mailbox-layer errors.
    #[derive(Debug, thiserror::Error)]
    pub enum MailboxError {
        /// Backing channel at capacity; caller should back-off or prune budget.
        #[error("mailbox full (capacity {0})")] Full(usize),
        /// Receiver disconnected; backing channel was dropped.
        #[error("mailbox disconnected")] Disconnected,
        /// SoA row index not found.
        #[error("compartment not found: {0:?}")] NotFound(MailboxId),
    }

    /// AttentionMask rename errors.
    #[derive(Debug, thiserror::Error)]
    pub enum AttentionError {
        /// All slots occupied; LRU eviction failed (should not occur).
        #[error("attention mask slot table full")] SlotsFull,
        /// Slot index out of range (caller bug).
        #[error("slot {0} out of range for {1}")] SlotOutOfRange(u8, &'static str),
    }

### `mailbox_soa.rs` — Supporting types

    /// Stable per-compartment identifier (monotonic u64; never reused in a session).
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
    pub struct MailboxId(pub u64);

    /// Architectural role (u32; hot path uses 8-bit slot via AttentionMask rename).
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
    pub struct RoleId(pub u32);

    /// Temporal window for a compartment's lifetime.
    #[derive(Copy, Clone, Debug)]
    pub struct TemporalWindow {
        pub start_cycle: u32,
        pub end_cycle:   u32,
        /// If true, row is pruned at end_cycle even if still emitting.
        pub hard_end: bool,
    }

    /// Sigma-tier classification per parent plan §5.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum SigmaTier {
        StaticReflex,      // S1-S5: reflexes, Tokio-backed (Zone-2)
        Emergent,          // S6: options appear, InMemory (Zone-1)
        TwigBranching,     // S7-S8: micro-choices, InMemory (Zone-1)
        EpiphanyEscalate,  // S9-S10: escalate to L4 planner
    }

    /// Per-compartment budget countdown.
    #[derive(Copy, Clone, Debug)]
    pub struct Budget(pub u64);

    impl Budget {
        /// Decrement by 1 (saturating at 0).
        pub fn decrement(&mut self) { self.0 = self.0.saturating_sub(1); }
        /// True if budget reached zero (prune trigger).
        pub fn is_exhausted(&self) -> bool { self.0 == 0 }
    }

    /// Hebbian co-occurrence counter (per parent plan §9 E-CE64-MB-10).
    #[derive(Copy, Clone, Debug, Default)]
    pub struct PlasticityCounter(pub u64);

    /// Per-cycle delta scratchpad (~1 KB). Dropped at temporal window end.
    pub struct DeltaBuffer { pub data: [u8; 1024], pub len: usize }

    /// Zone-3 consumer handle (per parent plan §5 ConsumerHandle enum).
    #[derive(Copy, Clone, Debug)]
    pub enum ConsumerHandle {
        Postgrest(EndpointId),
        DrainWs(SubscriberId),
        SupabaseChannel(ChannelId),
        MysqlSink(SinkId),
        GrpcService(ServiceId),
    }

    #[derive(Copy, Clone, Debug)] pub struct EndpointId(pub u32);
    #[derive(Copy, Clone, Debug)] pub struct SubscriberId(pub u32);
    #[derive(Copy, Clone, Debug)] pub struct ChannelId(pub u32);
    #[derive(Copy, Clone, Debug)] pub struct SinkId(pub u32);
    #[derive(Copy, Clone, Debug)] pub struct ServiceId(pub u32);

    /// Outbound channel for CausalEdge64 emissions.
    pub struct Sender<T>(std::sync::mpsc::SyncSender<T>);

### `attention_actor.rs` (tokio-backing feature)

    /// BindRequest message to AttentionMaskActor ractor singleton.
    /// Per parent plan §4 D-CE64-MB-5: "receives BindRequest{kind, id}".
    pub struct BindRequest { pub kind: BindKind, pub id: u32 }

    /// BindReply returned by AttentionMaskActor.
    /// Per parent plan §4 D-CE64-MB-5: "returns BindReply{slot, evicted}".
    pub struct BindReply { pub slot: u8, pub evicted: Option<u32> }

    #[derive(Copy, Clone, Debug)]
    pub enum BindKind { G, W, Style }

    /// ractor singleton wrapping AttentionMask.
    /// One per supervisor scope; all compartments message this actor
    /// instead of calling AttentionMask directly (no shared mutable ref).
    /// Per parent plan §4 D-CE64-MB-5.
    pub struct AttentionMaskActor { /* inner: AttentionMask */ }

