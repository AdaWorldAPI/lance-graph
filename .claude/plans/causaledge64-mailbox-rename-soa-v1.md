# CausalEdge64 Mailbox + Rename SoA — Integration Plan v1

> **Status:** Active (draft, 2026-05-14)
> **Author:** main thread (Opus 4.7 1M), session `claude/resolve-pr-369-conflicts-ozMXd`
> **Scope (immutable):** Compose 5 already-authored plans + Σ10 Rubicon doctrine + 1 genuine ndarray-side gap (`par-tile`) into a single substrate where ractor mailboxes carry **CausalEdge64** emissions, share **BindSpace** via zero-copy views, communicate cross-compartment via **SPOW witnesses** persisted to **AriGraph SPO-G quads**, and rename architectural identities (OGIT domain / witness palette / thinking style / truth) into 5-8 bit physical slots via session-ephemeral **AttentionMask** rename SoA. Result: ownership-typed reasoning compartments with compile-time UB-impossibility, ~1.5 KB per compartment, supporting ~24K parallel thoughts at 200ns cycle speed across ≤32 active OGIT domains.
> **Supersedes:** nothing (this composes; it does not replace).
> **Depends on:** `bindspace-columns-v1` · `oxigraph-arigraph-cognitive-shader-soa-merge-v1` · `ogit-g-context-bundle-v1` · `pr-g2-ractor-supervisor` (shipped Tokio shape via PR #366 S7-W3) · `pr-j-1-int4-32d-atoms` · Σ10 Rubicon doctrine (`linguistic-epiphanies-2026-04-19.md` E21).
> **Confidence (2026-05-14, pre-execution):** High. Architecture is composition of named-and-reviewed pieces, not new invention. Risk concentrates in §3 CausalEdge64 bit-layout reclaim (must not break PAL8 serialization or NarsTables LUT layout) and §8 AttentionMask LRU eviction protocol.
> **Cross-PR refs:** PR #355 (Pillar 0 + cascade columns SHIPPED), PR #366 (sprint-7 + Tokio ractor shape SHIPPED), PR #369 (Tier-A close + lance_cache schema bump SHIPPED), PR #370 (schema versioning + cfg(miri) bypasses + Miri sweep — in-flight on this branch).

---

## §0 Zone framing — corrected and load-bearing

Frequent miscommunication source. The zones from `ogit-cascade-supabase-callcenter-v1.md` Pillar 2, restated authoritatively:

| Layer | Position | Role | Speed | Substrate examples | Serialization? |
|---|---|---|---|---|---|
| **Zone 1** | Innermost | Inner ontology + cognitive-shader cycles + reasoning compartments. Holds zero-copy borrows into Zone 2. | 20-200 ns | CausalEdge64 microcopies, single-cycle Vsa16kF32 Markov bundle, par-tile InMemoryMailbox, MailboxSoA rows | NO. Wire DTO forbidden. |
| **BindSpace** | Boundary substrate | Read-only fingerprint substrate. Thinking/blackboard agents read here; writers commit via CollapseGate. The SoA itself. | 20-200 ns (read), µs (gated commit) | `cognitive-shader-driver` SoA Columns A-H per `bindspace-columns-v1.md` | NO. Internal carrier only. |
| **Zone 2** | Membrane | Outer ontology + lance-graph-callcenter. OGIT registry, AriGraph triplet graph, MUL gate, thinking-engine encode/decode, supervision dispatch. | µs-ms | Ractor Tokio shape (`CallcenterSupervisor` shipped #366 S7-W3), `lance-graph-ontology`, MUL, `thinking-engine` | NO except at egress traits. |
| **Zone 3** | Outer boundary | Consumer-facing serialization. The only emission point. | 2-200 ms | Wire DTO, postgrest, `drain.rs` WebSocket, Supabase realtime transcode, MySQL sinks, gRPC `crates/cognitive-shader-driver/src/grpc.rs` | YES. Wire DTOs allowed and required here only. |

**Critical rule (from `lab-vs-canonical-surface.md`):** serialization MUST happen at Zone 3 only. Zone 1/BindSpace/Zone 2 never serialize. CausalEdge64-as-routing-key + SpoWitness-in-AriGraph is the cumulative carrier across Zones 1/BindSpace/2; transcode-to-JSON only fires at Zone 3 egress.

**Ractor inhabits Zone 1 and Zone 2 simultaneously (the two-shape ractor) — but never Zone 3.** Zone 3 stays Wire DTO + `drain.rs`/postgrest/Supabase. Ractor's Tokio-shape supervisor emits Zone-3-bound deltas through the existing egress traits; it does not replace them. See §10 Blast Radius.

---

## §1 Executive summary

`CausalEdge64` (already shipped in `crates/causal-edge/` + `lance-graph-planner/src/cache/`) gains a 5-bit OGIT-domain slot (G), a 6-bit witness palette slot (W), and a 2-bit truth band by reclaiming the existing ~13 reserved bits — **in-place extension, no type bump, downstream PAL8/NARS LUT layouts preserved**. A session-ephemeral `AttentionMask` SoA carries the rename tables: 32-slot G ↔ u32 OGIT domain pointer, 64-slot W ↔ witness palette identity, 256-slot style ↔ {12-style cluster | 34 cognitive primitives | 144 verbs} per active grammar, 4-level truth read through 4 consumer lenses (TrustTexture / Wisdom / Staunen / MUL gate). Ractor reasoning compartments are **rows of `MailboxSoA`**, each owning a tiny delta-buffer + bindspace view + outbound CausalEdge64 channel; ~1.5 KB per compartment; ≤24K active concurrent compartments across the 32-slot active domain set. Cross-compartment cumulative state lives in AriGraph SPO-G quads (Christmas-tree decoration). Compile-time UB-impossibility via Rust ownership: cross-compartment communication can only flow as CausalEdge64 emissions (Copy, 8 bytes) — the borrow checker rejects any code that aliases BindSpace columns between compartments.

The Σ10 Rubicon tier architecture (`linguistic-epiphanies-2026-04-19.md` E21) becomes the runtime dispatcher: tier band → ractor mailbox backing (in-memory at Σ6-Σ8, Tokio at Σ1-Σ5 reflexes, escalate-to-L4-planner at Σ9-Σ10 EPIPHANY). Wiring Gap 3 from `THINKING_ORCHESTRATION_WIRING.md` (JIT pipeline end-to-end) closes naturally because compartment-spawn is the call site that consumes a `KernelHandle` for a thinking-style.

---

## §2 The universal sparse-rename pattern (load-bearing insight)

Every cognitive-state category has **two forms**: an architectural (cold, unbounded) form in AriGraph/OGIT/contract, and a physical (hot, small-int) form in CausalEdge64. The translation is **session-ephemeral rename**, semantically identical to CPU register renaming or SSA register allocation.

| Category | Architectural form (cold) | Where it lives | Physical form (hot) | Rename table |
|---|---|---|---|---|
| OGIT domain (G) | `u32` per `ogit-g-context-bundle-v1.md` D-OGIT-G-1 — 128+ domains, growing | AriGraph SPO-G quads + `lance-graph-ontology::NamespaceRegistry::seed_defaults()` | **5-bit slot** (32 entries) | `AttentionMask::g_slots: [Option<u32>; 32]` LRU |
| Witness role (W) | full witness palette: predicate IRI + provenance + W-question-type per SPOW tetrahedron §8 | AriGraph SPO-W edges + `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §1 witness lane | **6-bit slot** (64 entries) | `AttentionMask::w_slots: [Option<WitnessId>; 64]` LRU |
| Thinking style | 36 ThinkingStyle enum / 34 hpc/styles cognitive primitives / 144 (12×12) verbs | `lance-graph-contract::thinking` + `ndarray::hpc::styles` + grammar/role_keys verb catalogue | **8-bit slot** (256 entries) | `AttentionMask::style_slots: [Option<StyleId>; 256]` LRU + per-session active grammar |
| Truth qualia | TrustTexture(5) + Wisdom markers + Staunen depth + MUL GateDecision (3) | `lance-graph-contract::mul::TrustTexture` + planner MUL + `thinking-engine::qualia` | **2-bit band** | None — fixed lens collapse; 4 levels, 4 consumer-lens projections |
| Entity type (Column H) | `EntityTypeId u16` per `bindspace-columns-v1` §3 Column H | shared BindSpace Column H | `(u16 → 5-bit G + content-row index)` | implicit via row addressing |

**The truth-band lens collapse** — the critical type-deduplication:

| 2-bit value | TrustTexture | Wisdom marker | Staunen depth | MUL GateDecision |
|---|---|---|---|---|
| `00` | Crystalline | Mastered | Quiet (no surprise) | Proceed |
| `01` | Solid | Calibrated | Mild | Proceed |
| `10` | Fuzzy | Uncertain | Active (mild surprise) | Sandbox |
| `11` | Murky/Dissonant | Contradiction depth high | Loud (strong surprise) | Compass (veto) |

One physical field. Four consumer lenses. `lance-graph-contract::mul::TrustTexture` becomes the canonical 2-bit form; planner MUL `GateDecision` is a projection; Wisdom/Staunen are projections; `thinking-engine::qualia` Staunen depth maps to the same 2 bits.

**Closes**: `LATEST_STATE.md` "ThinkingStyle: 4 copies (contract canonical, not yet adopted)" — by reclaiming the canonical form via 8-bit-slot rename. Also closes `THINKING_ORCHESTRATION_WIRING.md` Gap 1 (Contract Not Consumed).

**Why this is CPU-shaped:** the architectural ID space is unbounded; the physical slot space is bounded by attention working-set size. Eviction = "this domain/witness/style fell out of attention this session." Rebinding = "new domain entered the conversation; claim a free slot, evict LRU if needed." Per-session different rename tables = per-session different focus-of-attention. The session IS the binding. Two sessions with different rename tables interpret the same 5-bit G value as different domains — exactly how human cognitive context-switching works neurologically (re-binding attention to different active populations).

**Per-session capacity:** 32 active domains × ~1000 typical compartments per domain ≈ **~24K parallel thoughts at any given cycle** with attention-mask filtering selecting which ~32 domains dispatch this 200ns tick.

---

## §3 CausalEdge64 layout extension — in-place reclaim of reserved 13 bits

Current `CausalEdge64` layout (per `crates/causal-edge/` + `crates/lance-graph-planner/src/cache/nars_engine.rs`):

```
bits  field            count       notes
─────  ──────           ─────       ─────
0-2    Pearl rung        3 bits     observational(0), interventional(1-2), counterfactual(3-6), full-cf(7) per Pearl 2³
3-10   S palette index   8 bits     bgz17 PaletteSemiring archetype ID for subject
11-18  P palette index   8 bits     same for predicate
19-26  O palette index   8 bits     same for object
27-42  temporal          16 bits    cycle index or temporal window mark
43-50  style ord         8 bits     ThinkingStyle ordinal (current 12-base, future 256 hot-slot)
51-63  reserved          13 bits    UNUSED today — reclaim target
```

**Proposed v2 layout (in-place, no type bump):**

```
bits  field            count       reclaimed?  notes
─────  ──────           ─────       ─────        ─────
0-2    Pearl rung        3 bits     no          unchanged
3-10   S palette index   8 bits     no          unchanged (bgz17 archetype, 256 entries)
11-18  P palette index   8 bits     no          unchanged
19-26  O palette index   8 bits     no          unchanged
27-42  temporal          16 bits    no          unchanged
43-50  style ord         8 bits     no          unchanged BUT semantics shift to "rename-table slot index" (§4 AttentionMask)
51-55  G slot            5 bits     YES         hot-path OGIT domain slot (32 active); full u32 G in AriGraph SPO-G quad
56-61  W slot            6 bits     YES         witness role slot (64 active); full witness in AriGraph SPO-W edge
62-63  truth band        2 bits     YES         TrustTexture/Wisdom/Staunen/MUL collapsed (§2 lens table)
```

**Total = 64 bits, no overflow, no extension.**

**Compatibility constraints (must hold for in-place extension to be safe):**

1. **PAL8 serialization** (per `crates/causal-edge/` 4101-byte PAL8 form): the layout is byte-packed; the reclaimed bits were zero in v1. PAL8 deserializers reading v1 PAL8 see G=0, W=0, truth=00 (Crystalline) — which is the **correct default** ("unrouted, no witness, fully trusted"). Existing PAL8 files round-trip without re-encoding. **Mandatory test:** deserialize-v1-encode-v2 produces byte-identical output when the new fields are zero (this is `causal-edge`'s round-trip-binary test extended).
2. **NarsTables LUT layout** (`lance-graph-planner/src/cache/nars_engine.rs`): the Pearl×style×palette LUT shape is keyed on bits 0-50 (rung + S/P/O + temporal + style). New bits 51-63 are **not** LUT-key-bearing — they're consumer-routing fields. LUT unchanged.
3. **`EdgeColumn` (BindSpace Column D)**: 8 × CausalEdge64 = 64 B/row, unchanged. The 8 edges per row stay 8 u64 entries with the new field semantics.
4. **`p64-bridge::STYLES` codebook**: extends from 12 → 36 → up-to-256 style entries; the 8-bit style ord slot was always sized for this growth. No layout change in p64-bridge itself; the rename table in AttentionMask becomes the runtime expansion mechanism.

**Deliverables (this section):**
- **D-CE64-MB-1** — Extend `CausalEdge64` accessors with `.g_slot() -> u8`, `.w_slot() -> u8`, `.truth() -> TrustTexture` (and setters). Methods only; bit layout updates the constants. Add feature flag `causal-edge-v2-layout` so consumers can opt in incrementally; v1 callers get `g_slot=0, w_slot=0, truth=Crystalline` by default.
- **D-CE64-MB-2** — PAL8 round-trip regression test confirming v1 ↔ v2 binary compatibility on zero-default fields.
- **D-CE64-MB-3** — NarsTables LUT regression test confirming LUT-key invariants hold across layout extension.

---

## §4 AttentionMask SoA — the session-ephemeral rename register file

Lives in Zone 1 (cycle-speed). One global instance per session (or per supervisor scope). Owned by a singleton ractor actor `AttentionMaskActor` that mediates rename requests.

```rust
// crates/par-tile/src/attention_mask.rs (new file)
// All slot tables fixed-size for predictable cache footprint.
// Total struct size: ~2 KB (fits L1 dcache).

pub const G_SLOTS: usize = 32;
pub const W_SLOTS: usize = 64;
pub const STYLE_SLOTS: usize = 256;

#[repr(C, align(64))]
pub struct AttentionMask {
    /// 5-bit slot → architectural OGIT domain u32 (0 = unallocated)
    pub g_slots: [Option<OgitDomainId>; G_SLOTS],
    /// 6-bit slot → architectural witness palette id
    pub w_slots: [Option<WitnessId>; W_SLOTS],
    /// 8-bit slot → architectural ThinkingStyle / cognitive primitive / verb id
    pub style_slots: [Option<StyleId>; STYLE_SLOTS],
    /// Active grammar — selects which architectural alphabet style_slots index into
    pub active_grammar: GrammarAlphabet,
    /// LRU clock per slot table (used for eviction policy)
    pub g_lru: [u32; G_SLOTS],
    pub w_lru: [u32; W_SLOTS],
    pub style_lru: [u32; STYLE_SLOTS],
    /// Monotonic session cycle counter (driving LRU)
    pub cycle: u32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum GrammarAlphabet {
    PlannerClusters12,  // 12 ThinkingStyle clusters per lance-graph-planner
    CognitivePrimitives34,  // 34 hpc/styles entries per ndarray
    VerbsTekamolo144,  // 12 × 12 verb compositions per German cognitive grammar
    FullStyle36Plus,  // 36 contract::thinking enum + YAML extensions
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct OgitDomainId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct WitnessId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct StyleId(pub u32);
```

**Rename protocol (the API surface every compartment uses):**

```rust
impl AttentionMask {
    /// Look up the physical slot for an architectural domain.
    /// Returns Some(slot) if currently bound, None otherwise.
    pub fn lookup_g(&self, id: OgitDomainId) -> Option<u8> { ... }

    /// Bind an architectural domain to a physical slot.
    /// Evicts LRU on slot pressure; returns evicted Option.
    /// Notifies subscribers of evictions via `bind_notifications` channel.
    pub fn bind_g(&mut self, id: OgitDomainId) -> (u8, Option<OgitDomainId>) { ... }

    /// Reverse — given a physical slot, return architectural id.
    pub fn resolve_g(&self, slot: u8) -> Option<OgitDomainId> { ... }

    // Same shape for w_slots and style_slots ...

    /// Touch — bump LRU clock for slot (called on every successful access).
    pub fn touch_g(&mut self, slot: u8) { self.g_lru[slot as usize] = self.cycle; }
    /// Tick — advance session cycle counter (called once per cycle by dispatcher).
    pub fn tick(&mut self) { self.cycle = self.cycle.wrapping_add(1); }
}
```

**LRU eviction policy:**
- On bind request when all slots occupied: evict slot with smallest `*_lru[i]` value.
- On successful lookup: `touch_*` updates the lru entry.
- Evictions are **broadcast** to consumers via `evict_notifications: broadcast::Sender<EvictionMsg>` so any cached references can be invalidated.
- **Hibernation policy** (resolves ghost-edge lifetime OQ): evicted slots' architectural identity persists in AriGraph forever; only the hot-path slot binding is dropped. When new evidence arrives on an evicted domain, a fresh slot is bound (potentially re-evicting another).

**Deliverables (this section):**
- **D-CE64-MB-4** — `AttentionMask` struct + accessors + LRU eviction + broadcast notifications. Pure Rust, no async deps. Lives in `crates/par-tile/src/attention_mask.rs`.
- **D-CE64-MB-5** — `AttentionMaskActor` ractor singleton wrapping `AttentionMask`; receives `BindRequest{kind, id}` messages, returns `BindReply{slot, evicted}`. Lives in `crates/par-tile/src/attention_actor.rs`.
- **D-CE64-MB-6** — Property tests: rename round-trip (bind → resolve → bind-again returns same slot until eviction); LRU eviction order; eviction broadcast received by all subscribers.

---

## §5 MailboxSoA — the compartment topology

Compartments are NOT individually-spawned ractor actors. They are **rows of a typed SoA** dispatched per-cycle by the Σ-tier dispatcher. This collapses the "10K mailboxes = 10K actor spawns" concern.

```rust
// crates/par-tile/src/mailbox_soa.rs

pub struct MailboxSoA<const N: usize> {
    /// Stable per-compartment identifier.
    pub ids: [MailboxId; N],
    /// Architectural role this compartment was spawned to (looks up via rename for hot path).
    pub roles: [RoleId; N],
    /// Per-compartment temporal window (cycle range to live).
    pub temporals: [TemporalWindow; N],
    /// Σ-tier classification — determines mailbox backing and budget.
    pub sigma_tiers: [SigmaTier; N],
    /// Zero-copy view into shared BindSpace columns (Arc-backed).
    /// View carries row range + column subset filter.
    pub bindspace_views: [BindSpaceView<'static>; N],
    /// Per-cycle delta buffer (~1 KB). Dropped when temporal window closes.
    pub deltas: [DeltaBuffer; N],
    /// Outbound channel for this compartment's emissions.
    pub witness_outs: [Sender<CausalEdge64>; N],
    /// External-intent gate. None = pure-internal compartment (~95% of Σ6-Σ8).
    /// Some(handle) = this compartment serializes to Zone 3 via the handle.
    pub intents: [Option<ConsumerHandle>; N],
    /// Parent supervisor for escalation (Σ9-Σ10 EPIPHANY escalates here).
    pub parents: [Option<MailboxId>; N],
    /// Per-compartment thinking-budget (countdown).
    pub budgets: [Budget; N],
    /// Plasticity bit-counter per (role, G) seen — Hebbian "fired together wired together".
    pub plasticity_counters: [PlasticityCounter; N],
}

#[derive(Copy, Clone)]
pub struct BindSpaceView<'a> {
    /// Borrowed reference to shared bindspace columns (Arc<BindSpace>).
    pub columns: &'a BindSpaceColumns,
    /// Row range this compartment is responsible for (typically a small window).
    pub rows: Range<usize>,
    /// Column-subset filter — which columns this compartment reads/writes.
    pub column_mask: ColumnMask,
}

#[derive(Copy, Clone)]
pub enum SigmaTier {
    /// Σ1-Σ5: STATIC, repair, Pearl rung 1 — reflexes. Tokio-backed mailbox.
    StaticReflex,
    /// Σ6: EMERGENT, Pearl 2-3 — options appear. InMemoryMailbox cycle-speed.
    Emergent,
    /// Σ7-Σ8: TWIG, branching micro-choices. InMemoryMailbox cycle-speed.
    TwigBranching,
    /// Σ9-Σ10: EPIPHANY, Pearl 5 — escalate to L4 planner (lance-graph-planner).
    EpiphanyEscalate,
}

#[derive(Copy, Clone)]
pub enum ConsumerHandle {
    /// Postgrest endpoint reachable via Zone 3 egress.
    Postgrest(EndpointId),
    /// Drain WebSocket subscription via lance-graph-callcenter::drain.
    DrainWs(SubscriberId),
    /// Supabase realtime channel via the transcode.
    SupabaseChannel(ChannelId),
    /// MySQL sink via the legacy bridge.
    MysqlSink(SinkId),
    /// gRPC service endpoint (cognitive-shader-driver::grpc).
    GrpcService(ServiceId),
}
```

**Lifecycle:**

1. **Spawn**: Σ-tier dispatcher reads incoming context (BindSpace row delta, external request, ghost-edge reactivation) → decides which compartment(s) to spawn → calls `MailboxSoA::push_row(role, temporal, sigma_tier, bindspace_view, intent)` → returns `MailboxId`.
2. **Dispatch (per cycle)**: dispatcher scans `sigma_tiers[]` mask, picks compartments whose temporal window is active AND attention-mask permits, calls `compartment.dispatch(cycle_input)` returning Option<CausalEdge64>.
3. **Emit**: compartment writes to `witness_outs[i]` channel. Emission causes CollapseGate merge (Xor for complementary mailboxes; Bundle for compatible). Merged emission lands in EdgeColumn (BindSpace Column D) row.
4. **AriGraph commit**: if `intents[i].is_some()` OR Σ-tier ≥ Σ7-Σ8, the emission also commits to AriGraph as SPO-G quad with W = witness palette id (full architectural form looked up via AttentionMask rename).
5. **Prune**: at end-of-temporal-window OR budget-exhausted OR XOR-cancel-with-sibling, `MailboxSoA::drop_row(id)` reclaims the slot. Delta buffer drops; plasticity counter persists. If unresolved, emit ghost-edge at Pearl rung 3 to AriGraph for future reactivation.

**Per-compartment memory footprint:**

| Field | Size | Notes |
|---|---|---|
| MailboxId | 8 B | u64 |
| RoleId | 4 B | u32 architectural; hot path uses 8-bit slot |
| TemporalWindow | 16 B | start/end cycle u32 + flags |
| SigmaTier | 1 B | enum discriminant |
| BindSpaceView | 24 B | borrow + row range + column mask |
| DeltaBuffer | ~1024 B | per-cycle scratchpad; dropped at window end |
| witness_outs channel handle | 24 B | Sender<CausalEdge64> |
| intent | 16 B | Option<ConsumerHandle> |
| parent | 8 B | Option<MailboxId> |
| Budget | 8 B | countdown |
| PlasticityCounter | 8 B | u64 fired-together-counter |
| **Total** | **~1.2 KB / compartment** | ≪ 26 MB strawman from prior analysis |

10,000 concurrent compartments = ~12 MB. **Fits L2 on Sapphire Rapids; fits L1 with hot subset.**

**Deliverables (this section):**
- **D-CE64-MB-7** — `MailboxSoA<N>` struct + lifecycle methods (`push_row`, `drop_row`, `dispatch_cycle`). Lives in `crates/par-tile/src/mailbox_soa.rs`.
- **D-CE64-MB-8** — `BindSpaceView` type with row-range + column-mask filter; zero-copy borrow into shared `Arc<BindSpace>`.
- **D-CE64-MB-9** — Property tests: spawn-dispatch-prune lifecycle; XOR-cancel of complementary mailboxes; plasticity counter monotonic; intent gate strict (None compartments never reach Zone 3).

---

## §6 The 5 substrate crates — per-crate change inventory

| Crate | Existing state | This plan adds | Status |
|---|---|---|---|
| **`crates/par-tile/`** (NEW) | does not exist | `Mailbox<T>` trait + 3 backings (InMemoryMailbox cycle-speed, TokioMailbox callcenter-shape — wraps existing CallcenterSupervisor, SupabaseSubMailbox Zone-3 egress wrapper); `MailboxSoA<N>`; `AttentionMask` + `AttentionMaskActor`; vendored rayon-shape work-stealing OR std-thread fallback for `par_tile<K>` | Pure new crate, ~1500 LOC, no external deps beyond ractor + std |
| **`crates/causal-edge/`** | CausalEdge64 u64 packed + NARS LUT + PAL8 serialization + self-reinforcement LoRA (4101 bytes) | New bit accessors for G(5)/W(6)/truth(2); v2 layout feature flag; round-trip regression; LoRA training signal extended to bump plasticity counter | In-place extension; no breaking change for v1 consumers |
| **`crates/cognitive-shader-driver/`** | BindSpace SoA Columns A-D shipped (FingerprintColumns / QualiaColumn / MetaColumn / EdgeColumn) | Columns E (OntologyDelta) / F (AwarenessColumn) / G (ModelBindingColumn) / H (TypeColumn EntityTypeId u16) per `bindspace-columns-v1.md`; CollapseGate extended with `MergeMode::Superposition` (preserve both deltas when XOR-equal); BindSpaceView accessor for per-compartment row-range borrows | Implements `bindspace-columns-v1.md` Phase 2; closes PR 355 #6 (per-row context_ids) + FIX-5 (trust_below_floor wiring) |
| **`crates/lance-graph-supervisor/`** | `CallcenterSupervisor` ractor Tokio shape shipped #366 S7-W3 (one-for-one, exponential backoff, separate 18-byte `LifecycleAuditEvent`) | Σ-tier dispatcher (`SigmaTierRouter`) that maps Σ1-Σ10 → InMemoryMailbox/TokioMailbox/L4-escalate; plasticity bit-counter feedback into spawn priors; budget-exhaustion + XOR-cancel + outcome-sufficient pruning triggers | Two-shape ractor: cycle-speed shape via par-tile, Tokio shape preserved; coexists with existing `CallcenterSupervisor` |
| **`crates/lance-graph/src/graph/arigraph/`** | AriGraph 4696 LOC (episodic.rs / triplet_graph.rs / retrieval.rs / sensorium.rs / orchestrator.rs / xai_client.rs / language.rs); SPO triple shape + NARS truth + unbundle hooks | SPO-G quad mode (5th tuple position via `ogit-g-context-bundle-v1.md` D-OGIT-G-1); ghost-edge persistence at Pearl rung 3/7; witness-chain SpoWitnessChain<N> shape for parent-supervisor edges (SpoWitness64 packed for peer edges) | Implements `ogit-g-context-bundle-v1.md` D-OGIT-G-1; SPO → SPOW upgrade per `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §1-§8 |

**Sibling crates that consume but don't change:**
- `lance-graph-contract` — `TrustTexture` already canonical (gets the 2-bit collapse confirmation in spec); ThinkingStyle 36-variant enum unchanged (gets the 8-bit-slot rename context); no new contract types
- `bgz-tensor` — palette 256×256 table consumed by Gaussian splat in compartments; no change
- `bgz17` — PaletteSemiring consumed; no change
- `deepnsm` — 4096 COCA distance matrix consumed; no change
- `thinking-engine` — encode/decode + lens stack + `ghosts.rs` (counterfactual self carrier) consumed; the `ghosts.rs` file becomes the host for ghost-edge logic (already named, never wired)
- `lance-graph-planner` — L4 thinking-style dispatch reads architectural style IDs; rename happens at compartment-spawn time, not in planner; JIT pipeline finally closes Gap 3 because compartment-spawn is where KernelHandle is consumed

---

## §7 Composition order — sequenced PRs

7 PRs, ordered by dep graph. Each is independently mergeable + reviewable.

```
PR-CE64-MB-1: par-tile crate apex
       │
       ▼
PR-CE64-MB-2: CausalEdge64 v2 layout (in-place reclaim)
       │
       ▼
PR-CE64-MB-3: BindSpace Columns E/F/G/H (bindspace-columns-v1 Phase 2)
       │           │
       │           ▼
       ▼   PR-CE64-MB-4: SPO-G upgrade in AriGraph (ogit-g-context-bundle-v1 D-OGIT-G-1)
       │           │
       └─────┬─────┘
             ▼
PR-CE64-MB-5: MailboxSoA + AttentionMask actor wiring
       │
       ▼
PR-CE64-MB-6: Σ-tier dispatcher (SigmaTierRouter) + cycle-speed InMemoryMailbox backing
       │
       ▼
PR-CE64-MB-7: Bevy proof plugin (NdarrayCullPlugin first, per bevy session recommendation)
```

| PR | Scope | LOC estimate | Risk | Closes |
|---|---|---|---|---|
| **PR-CE64-MB-1** | `crates/par-tile/`: `Mailbox<T>` trait + InMemory backing + AttentionMask SoA + tests | ~1500 | Low (pure new crate, no consumers yet) | Diamond apex per bevy session |
| **PR-CE64-MB-2** | `crates/causal-edge/`: v2 layout feature gate + G/W/truth accessors + PAL8 round-trip + NarsTables regression | ~400 | Med (binary compatibility critical) | None directly; enables PR-CE64-MB-5 |
| **PR-CE64-MB-3** | `crates/cognitive-shader-driver/`: Columns E/F/G/H + CollapseGate MergeMode::Superposition + BindSpaceView accessor | ~800 | Med (BindSpace SoA layout change) | PR 355 #6 (per-row context_ids), FIX-5 (trust_below_floor wiring), `bindspace-columns-v1.md` Phase 2 |
| **PR-CE64-MB-4** | `crates/lance-graph/src/graph/arigraph/`: SPO-G quad mode + ghost-edge persistence + SpoWitnessChain | ~600 | Med (changes core triple shape) | `ogit-g-context-bundle-v1.md` D-OGIT-G-1, SPOW §1-§8 |
| **PR-CE64-MB-5** | `crates/par-tile/` + `crates/lance-graph-supervisor/`: MailboxSoA + AttentionMaskActor + ConsumerHandle plumbing | ~1200 | Med (cross-crate, ractor integration) | THINKING_ORCHESTRATION_WIRING Gap 1 (Contract Not Consumed) |
| **PR-CE64-MB-6** | `crates/lance-graph-supervisor/`: SigmaTierRouter + Σ-tier banding policy + plasticity + pruning triggers + budget consumption | ~1500 | High (new dispatcher, replaces several ad-hoc paths) | THINKING_ORCHESTRATION_WIRING Gap 4 (Elevation Not Connected); Σ10 Rubicon runtime |
| **PR-CE64-MB-7** | Bevy plugin `NdarrayCullPlugin` consuming MailboxSoA for frustum cull | ~500 (bevy fork side) | Low (proof-of-pattern only) | bevy session recommended starting point |

**Sequencing gates:**
- PR-CE64-MB-2 and PR-CE64-MB-3 can land **in parallel** (different crates; only PR-CE64-MB-5 depends on both).
- PR-CE64-MB-4 can land **in parallel** with PR-CE64-MB-2 + PR-CE64-MB-3 (orthogonal AriGraph change).
- PR-CE64-MB-1 is the apex; everything else depends on it.
- PR-CE64-MB-7 is the proof step; lands after MB-1 through MB-6 stabilize.

**Per-PR worker estimate:** 1-2 weeks each Sonnet worker + 1 day Opus meta-review per CCA2A pattern (`cca2a-sprint-prompt-template.md`). Full composition: ~3 sprints = 6-9 weeks.

---

## §8 ndarray-side prerequisites — already in flight on `claude/resolve-pr-369-conflicts-ozMXd`

These are the upstream pieces ndarray needs to ship before this plan can fully land. Some already done; rest are tracked for §9 follow-up PRs on the ndarray side.

| Item | Status | Where |
|---|---|---|
| `cfg(miri)` cpuid bypass in `SimdCaps::detect` | **SHIPPED** | `e0907cd` on ndarray branch |
| `scripts/miri-tests.sh` with constrained scope | **SHIPPED** | `6590b9e` |
| `simd.rs:212` "5 of 30 types" comment correction (PR #146 shipped 24/24 parity) | **SHIPPED** | `530ffaa` |
| **U16x32 / U32x16 / U64x8 method gaps** (`simd_eq` / `simd_ne` / `simd_ge` / `simd_gt` / `simd_le` / `simd_lt` / `simd_clamp` / `select` / `to_bitmask` / `from_u8x64_lo+hi` / `pack_saturate_u8` / `shl` / `shr` / explicit `zero()`) | **OPEN** — needs follow-up PR on ndarray | `simd_nightly/u_word_types.rs` + same on `i_word_types.rs` |
| `crate::simd::*` dispatch routing through `simd_nightly` under `cfg(miri)` (closes the AVX target-feature wall that aborts `hpc::*` tests under Miri) | **OPEN** — load-bearing for full Miri coverage of cognitive shader paths; ~50 LOC in `src/simd.rs` | `simd.rs:215+` |
| `crates/par-tile/` (lives in lance-graph side per diamond rationale) | **OPEN** — PR-CE64-MB-1 above | new crate |
| Rayon-vendor (work-stealing inside par-tile, Miri-friendly) | **OPEN** — large; can defer to sprint-after-this | inside par-tile |

**ndarray follow-up PR (single, focused, lands BEFORE par-tile crate):**
- **PR-NDARRAY-MIRI-COMPLETE** — close the u-word method gaps, route `crate::simd::*` through `simd_nightly` under `cfg(miri)`, delete `src/simd_nightly/_original_draft.rs` (dead 5-type sketch), ~150 LOC + ~50 LOC. Validates: full `hpc::activations::*` + `hpc::*` test suite Miri-clean after dispatch reroute. Unblocks par-tile development that wants the polyfill paths Miri-checkable.

---

## §9 Synergies and epiphanies — what this composition unlocks

**E-CE64-MB-1 — The universal-rename pattern (load-bearing).** Every architectural identity (G, W, style, truth) renames to a hot-path slot via `AttentionMask`. Same pattern as CPU register renaming. Per-session different rename tables = per-session different attention. The same 5-bit G means different domains in different sessions because the rename table differs. **Closes a class of "type duplication" debt** (TrustTexture 4 copies, ThinkingStyle 4 copies) by making them architecturally one type with multiple lens projections.

**E-CE64-MB-2 — Role-as-mailbox retires Vsa16kF32 as universal carrier.** The 47 `LazyLock<RoleKey>` slice catalogue allocations across Vsa16kF32 (~3 MB if all materialized) collapse to 47 typed mailbox kinds (~50 KB). Vsa16kF32 retreats to its honest role: single-cycle Markov-bundle carrier for grammar parsing, dropped at cycle end. **No cumulative state in Vsa16kF32 anywhere.** Cumulative state is AriGraph SPO-G quads + EdgeColumn EdgeColumn CausalEdge64 emissions.

**E-CE64-MB-3 — Christmas-tree decoration via AriGraph SPO-G + ghost edges.** Compartment epiphanies emit directly to AriGraph as SPO-G quads (G = OGIT domain pointer). Unresolved hole-forms from SPOW tetrahedron emit as ghost edges at Pearl rung 3 (counterfactual) or rung 7 (full-cf). Ghosts hibernate until evidence arrives. **AriGraph IS the long-term memory; the rename table is the working memory.** Eviction-from-working-memory ≠ deletion-from-long-term-memory. The mind always decorates; the tree never resets.

**E-CE64-MB-4 — Ownership-typed compartment compartmentalization makes UB a compile error.** Each MailboxSoA row owns its delta buffer; BindSpace columns are `Arc`-shared but written only through CollapseGate (single point of mutation). Cross-compartment communication can only flow as CausalEdge64 emissions (Copy, 8 bytes). The borrow checker **rejects** any code that tries to alias mutable BindSpace columns across compartments. **Race conditions at 200ns cycle speed become compile errors, not runtime bugs.**

**E-CE64-MB-5 — Particle/wave duality in Rust semantics.** Particle = the owned compartment row in MailboxSoA (discrete, type-safe, Drop-managed). Wave = the CausalEdge64 emission rippling through EdgeColumn and decorating AriGraph SPO-G quads (continuous influence, non-local, no shared mutable state). Both fall out of "compartments own, AriGraph aggregates, CausalEdge64 crosses." **Not a metaphor — a structural property of the type system.**

**E-CE64-MB-6 — The gRPC service shape IS the ractor message protocol.** `crates/cognitive-shader-driver/src/grpc.rs` `Dispatch(DispatchRequest) -> CrystalResponse` is already the ractor mailbox handler shape. Same Request/Response pair, same typed payload, same no-shared-state contract. The transport varies (tonic gRPC vs InMemoryMailbox channel vs TokioMailbox vs SupabaseSubMailbox); the contract is one. **Reuse, don't invent.** The lab-only gRPC service becomes the production ractor protocol simply by adding non-gRPC backings.

**E-CE64-MB-7 — Truth qualia is 2 bits with 4 lenses (TrustTexture + Wisdom + Staunen + MUL).** Same field, four consumer projections per §2 table. Consolidates 4 type duplications into one canonical field with documented projection rules.

**E-CE64-MB-8 — Σ10 Rubicon dispatching IS the substrate-tier router.** The named Σ1-Σ10 tier doctrine from `linguistic-epiphanies-2026-04-19.md` E21 finally gets a runtime dispatcher: `SigmaTierRouter` maps incoming compartment-spawn requests to the correct mailbox backing by tier band. **Wires what was previously documented-but-unwired.**

**E-CE64-MB-9 — JIT pipeline closes (Gap 3 from THINKING_ORCHESTRATION_WIRING).** Compartment-spawn is where `KernelHandle` gets consumed: spawn message includes style-slot index; AttentionMask resolves to architectural ThinkingStyle; if `KernelHandle` cached, dispatch; if not, JIT-compile via `crates/lance-graph-planner/src/strategy/jit_compile.rs` from YAML descriptor and cache. **End-to-end FieldModulation → ScanParams → JitTemplate → Cranelift → KernelHandle finally fires.**

**E-CE64-MB-10 — Plasticity emerges naturally from the bit-counter on MailboxSoA::plasticity_counters.** Every successful emission increments `(role, G)` co-occurrence counter. Spawn priors next cycle bias toward high-count pairings (Hebbian "fired together wired together"). Counterfactual ghosts emit at low-counter slots (synaptic pruning). **No new mechanism — just SoA columns + LRU on AttentionMask + bit-counter increment on emission.**

---

## §10 Blast radius in Zone 2 — ractor inhabits without retiring surface area

The user's explicit ask: ractor in `lance-graph-callcenter` (Zone 2) must complement the Supabase realtime transcode logic, not retire its surface area.

**What ractor's Zone-2 Tokio shape ADDS:**
- `CallcenterSupervisor` (PR #366 S7-W3, **shipped**) — one-for-one supervision of per-tenant actor trees
- `SigmaTierRouter` (this plan, PR-CE64-MB-6) — Σ1-Σ5 reflexes dispatched here when external request arrives via Zone-3 boundary
- Σ9-Σ10 EPIPHANY escalation receiver — when an InMemoryMailbox at Σ7-Σ8 emits an EPIPHANY-tier witness, it routes to a Zone-2 actor for cross-tenant MUL gate + AriGraph commit + optional Wire DTO egress
- Plasticity feedback channel — `AttentionMaskActor` writes plasticity counters to a per-tenant Lance dataset for long-term Hebbian training; this happens at Zone 2 because it's cross-cycle accumulation

**What ractor's Zone-2 shape DOES NOT TOUCH:**
- `crates/lance-graph-callcenter/src/drain.rs` — WebSocket subscription/push remains unchanged; ractor emits *into* drain via existing `DrainSender` API
- `crates/lance-graph-callcenter/src/version_watcher.rs` — Lance tail-cursor remains unchanged; ractor reads tail events the same way `drain` does
- `crates/lance-graph-callcenter/src/postgrest.rs` — HTTP REST surface unchanged; ractor's compartments at intent=Postgrest emit through existing handler chain
- **Supabase realtime transcode logic** — completely untouched. The transcode is at Zone 3, beyond the BBB. Ractor's job is to *populate* AriGraph SPO-G quads + Lance row inserts at Zone 2; the Zone-3 transcode (which reads logical replication of Postgres + emits JSON over WebSocket to subscribers) reads those Lance commits the same way it always has. **Two completely orthogonal paths.**
- `crates/lance-graph-callcenter/src/auth.rs` + `policy.rs` + `rbac.rs` — RBAC + RLS + Policy chain unchanged; ractor's MUL gate fires *after* the policy chain accepts the request

**Topology diagram:**

```
                  ZONE 3 (consumer-facing, ms+, serialization OK)
   ┌─────────────────────────────────────────────────────────────┐
   │  postgrest.rs   drain.rs   grpc.rs   supabase-realtime     │  (unchanged)
   │       │            │          │              │              │
   │       ▼            ▼          ▼              ▼              │
   │  ┌────────────────────────────────────────────────────┐    │
   │  │           Wire DTOs (Zone-3 serialization)         │    │
   │  └────────────────────────────────────────────────────┘    │
   └──────────────────────────│──────────────────────────────────┘
                              │
                              ▼ ConsumerHandle dispatch
                              │
                  ZONE 2 (lance-graph-callcenter, µs-ms, no serialization)
   ┌─────────────────────────────────────────────────────────────┐
   │  ┌──────────────────────────────────────────────────┐      │
   │  │  CallcenterSupervisor (ractor, Tokio shape)      │      │
   │  │  ├── SigmaTierRouter (THIS PLAN)                 │      │
   │  │  ├── per-tenant actor tree (shipped #366 S7-W3)  │      │
   │  │  └── EPIPHANY escalation receiver                │      │
   │  └──────────────────────────────────────────────────┘      │
   │      │                                                       │
   │      ▼ ConsumerHandle escalation                            │
   │  ┌──────────────────────────────────────────────────┐      │
   │  │  AriGraph SPO-G quads + ghost edges              │      │
   │  │  OntologyRegistry + MUL gate + NARS truth       │      │
   │  └──────────────────────────────────────────────────┘      │
   └──────────────────────│───────────────────────────────────────┘
                          │
                          ▼ MailboxSoA push_row + AttentionMask bind
                          │
                BINDSPACE (substrate, 20-200ns reads, gated writes)
   ┌─────────────────────────────────────────────────────────────┐
   │  Arc<BindSpace> Columns A-H per bindspace-columns-v1        │
   │  CollapseGate (Flow/Block/Hold + MergeMode Xor/Bundle/Super)│
   └──────────────────────────│──────────────────────────────────┘
                              │
                              ▼ BindSpaceView<'_> zero-copy borrow
                              │
                ZONE 1 (inner ontology, 20-200ns, cycle-speed)
   ┌─────────────────────────────────────────────────────────────┐
   │  MailboxSoA<N> compartments (THIS PLAN)                     │
   │  AttentionMask rename SoA (THIS PLAN)                       │
   │  InMemoryMailbox via par-tile (THIS PLAN)                   │
   │  Vsa16kF32 single-cycle Markov bundle (cycle-temporary)     │
   │  CausalEdge64 microcopies (emissions)                       │
   └─────────────────────────────────────────────────────────────┘
```

**Blast radius summary:**
- New code lives in 5 crates (par-tile NEW, plus changes to causal-edge, cognitive-shader-driver, lance-graph-supervisor, lance-graph/arigraph)
- `lance-graph-callcenter` Zone-2 surface unchanged except `CallcenterSupervisor` gains the SigmaTierRouter sub-actor
- Zone 3 surface (postgrest / drain / grpc / supabase-realtime) **completely unchanged**
- Backward compatibility preserved on PAL8 + NarsTables + EdgeColumn binary layouts (§3 invariants)
- No existing consumer crate needs to change to keep working; new consumers opt in via the new ConsumerHandle ingress path

---

## §11 Open design questions (OQ) — ratify before / during execution

**OQ-1 — Σ-tier banding policy.** Sketched as:
- Σ1-Σ5 STATIC repair → Tokio shape (Zone-2 reflexes)
- Σ6 EMERGENT, Σ7-Σ8 TWIG → InMemoryMailbox (Zone-1 cycle-speed)
- Σ9-Σ10 EPIPHANY → escalate to L4 lance-graph-planner

Is this banding correct, or should Σ4-Σ5 also live cycle-speed (some "fast reflexes" are actually micro-decisions)? **Decision needed before PR-CE64-MB-6 SigmaTierRouter spec.**

**OQ-2 — Ghost edge persistence policy.** Resolved in §4 LRU framing: ghosts persist in AriGraph forever; only the hot rename slot evicts. But **should ghosts decay in AriGraph via NARS confidence drift**, or stay at fixed Pearl rung 3 until new evidence arrives? My read: NARS truth-revise on ghosts at AriGraph-commit boundaries (low-frequency, batched) — matches `causal-edge` LoRA training pattern.

**OQ-3 — Compartment plasticity update granularity.** Resolved tentatively: bit-counter per emission (high-frequency, AttentionMask-side) + NARS truth-refine at AriGraph commit (low-frequency, batched). Confirm or refine.

**OQ-4 — INT4-32D thinking atom integration point.** Per `pr-j-1-int4-32d-atoms.md` spec, `ThinkingAtom32x4` = 16 bytes per cognitive style fingerprint, K-NN over `p64-bridge::STYLES` codebook. When a Σ6-Σ8 compartment spawns and AttentionMask doesn't have a matching `(role, G)` style binding, **does the compartment compute an INT4-32D fingerprint from current situation features and K-NN-fallback to nearest known styles**? My read: yes — this is exactly the cold-start safety net Pattern G was deferred for. Wiring goes in PR-CE64-MB-6 SigmaTierRouter spawn path.

**OQ-5 — Rayon vendor decision.** Inside par-tile, work-stealing for compartment dispatch can use (a) vendored rayon-shape (~2 KLOC, Miri-friendly because we own the unsafe boundaries) or (b) `std::thread::scope` + crossbeam channels (~500 LOC, simpler, lower throughput). My read: defer to OQ-5 sprint-call; start with std::thread::scope; promote to vendored rayon-shape if profiling shows throughput cliff.

**OQ-6 — Vsa16kF32's final residence.** Stays in `crystal/fingerprint.rs` for within-cycle Markov bundle (grammar parsing role-binding), dropped at cycle end. Confirm or push back.

**OQ-7 — AwarenessColumn (Column F) sizing.** Spec says 256 B/row in `bindspace-columns-v1.md` §3. Stays 256 B/row because it's the SoA-wide awareness mantissa (per-word/per-u64 bit-purity, distribution shape, match strength, residual norm). Compartments write into it via gated delta; the column itself stays full BindSpace width.

**OQ-8 — Witness shape: SpoWitness64 packed vs SpoWitnessChain<N>.** Resolved tentatively: `SpoWitness64` (u64 packed, Copy, peer mailbox edges) + `SpoWitnessChain<N>` (Cow-shaped, parent-supervisor + AriGraph-commit edges). Both supported in protocol; sender picks by destination.

---

## §12 Iron rule compliance (CLAUDE.md invariants)

| Iron rule | Compliance |
|---|---|
| **I-SUBSTRATE-MARKOV** (VSA bundling guarantees Chapman-Kolmogorov semigroup) | Preserved — Vsa16kF32 within-cycle Markov bundle unchanged; cumulative state moves to AriGraph SPO-G + CausalEdge64, NEITHER of which use XOR-merge on identity fingerprints (the I1 violation). CollapseGate::Bundle on EdgeColumn aggregates compatible CausalEdge64 emissions; CollapseGate::Xor on EdgeColumn preserves complementary emissions per "opinions are committed contradictions." |
| **I-NOISE-FLOOR-JIRAK** (classical Berry-Esseen wrong under weak dependence) | Plasticity bit-counter thresholds + NARS truth-revise pulls from Jirak-derived bounds where principled — explicit OQ at spec sign-off. |
| **I-VSA-IDENTITIES** (VSA on IDENTITY fingerprints, never on content) | Preserved — CausalEdge64 carries palette indices (identities), not bitpacked content. AttentionMask renames identity IDs to slot indices (still identities). SPOW tetrahedron operates on identity W/S/P/O references. **Vsa16kF32 retreating from "universal carrier" to "single-cycle Markov bundle" actively strengthens this rule.** |
| **I1** (BindSpace read-only; CollapseGate bundles) | Preserved — all compartment writes go through CollapseGate; BindSpaceView is read-only borrow; delta-buffer in MailboxSoA is per-compartment-owned, never shared. |
| **AGI-as-glove SoA invariant** (Topic = FingerprintColumns read; Angle = QualiaColumn read; Thinking = MetaColumn write; Planner = EdgeColumn write) | Preserved — Topic/Angle reads via BindSpaceView; Thinking writes via gated delta to MetaColumn; Planner writes via CausalEdge64 emission to EdgeColumn. **Compartments are the runtime instantiation of "AGI = (topic, angle, thinking, planner) = SoA consuming cognitive-shader-driver."** Each compartment IS the glove fitting into the SoA columns for one (role, G) at one cycle. |
| **Method-on-carrier discipline** (free function = reject; method = accept) | Preserved — `MailboxSoA::push_row` / `dispatch_cycle` / `drop_row` are methods; `AttentionMask::bind_g` / `lookup_g` / `resolve_g` are methods; `CausalEdge64::g_slot` / `w_slot` / `truth` are accessors. **No new free functions introduced.** |

---

## §13 Cross-references — plans this composes + closes

**Composes (Active or Shipped plans this depends on / extends):**
- `.claude/plans/bindspace-columns-v1.md` — Columns E/F/G/H authored 2026-04-26, scientifically reviewed 7 SOUND / 7 CAUTION / 0 WRONG, **no PR yet** — PR-CE64-MB-3 ships Phase 2
- `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` — SPOW tetrahedron §8 + Gaussian splat §9 + 64×64/256×256/4096² planes §5/§6/§7 — PR-CE64-MB-4 ships SPO-G + SPOW + ghost edges
- `.claude/plans/ogit-g-context-bundle-v1.md` — D-OGIT-G-1 SPO-G u32 slot — PR-CE64-MB-4 implements
- `.claude/plans/pr-g2-ractor-supervisor.md` — Tokio shape shipped #366 S7-W3 — PR-CE64-MB-6 extends with SigmaTierRouter
- `.claude/specs/pr-j-1-int4-32d-atoms.md` — North-Star, OQ-4 above wires the K-NN cold-start path
- `.claude/plans/thought-cycle-soa-awareness-integration-v1.md` — AwarenessPlane16K + GrammarMarkovLens64 + ReasoningWitness64 + ThoughtCycleSoA — composed implicitly via MailboxSoA owning these as references
- `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md` — Gaussian splat integration consumed by compartment dispatch
- `.claude/plans/jc-pillars-runtime-wiring-v1.md` — JC Pillar 10/11 math kernels consumed by Gaussian-splat compartments

**Closes / unblocks:**
- **PR #355 deferred Tier B**: FIX-4 (codebook_index bit-collision edge — addressed via PaletteSemiring binning under 256×256 attention), FIX-5 (`trust_below_floor` wiring test — closed when Column H lands), per-row `BindSpace.context_ids` for `driver.rs:311` (closed when Column H lands)
- **`THINKING_ORCHESTRATION_WIRING.md` Gap 1** (Contract not consumed — 12 vs 36 styles) — closed via AttentionMask 8-bit-slot rename collapse
- **`THINKING_ORCHESTRATION_WIRING.md` Gap 3** (JIT pipeline never executed end-to-end) — closed via compartment-spawn consuming KernelHandle
- **`THINKING_ORCHESTRATION_WIRING.md` Gap 4** (Elevation not connected to execution) — closed via SigmaTierRouter as runtime elevation policy
- **TD-INT4-32D-ATOMS-6** — wired via OQ-4 cold-start path
- **TD-THINKING-ENGINE-UNWIRED-1** (582 KB cognitive substrate dormant) — wired via BindSpaceView references resolving thinking-engine encode/decode + lens stack on demand

**Doctrine anchors:**
- Σ10 Rubicon Tier Architecture: `.claude/knowledge/linguistic-epiphanies-2026-04-19.md` E21 — runtime dispatcher via SigmaTierRouter
- VSA switchboard three-layer architecture: `.claude/knowledge/vsa-switchboard-architecture.md` — Vsa16kF32 cycle-temporary discipline preserved
- Lab-vs-canonical surface: `.claude/knowledge/lab-vs-canonical-surface.md` — Wire DTO at Zone 3 only preserved
- Encoding ecosystem: `.claude/knowledge/encoding-ecosystem.md` — palette/CAM-PQ/HHTL cascade roles preserved within compartments

**Acknowledgments:**
- The fresh-eyes recursion sequence in this session: 3rd pair (bevy session — diamond dep graph + Slice↔Plane bridge + NdarrayCullPlugin proof-first); 4th pair (semantic naming over shape naming, MultiLaneColumn already named, 5-Layer Stack already named); 5th pair (Vsa16kF32 single-purpose correction, two-shape ractor framing, INT4-32D as North Star); 6th pair (ephemeral BindSpace + role-as-mailbox + space-time-collapse + external-intent gate + Ractor-SoA + Think-as-reference); 7th pair (CausalEdge64-as-emission-carrier + truth-collapse + 24K parallel thoughts via 32-slot session-ephemeral sparse rename + 12/34/144 hot-context pattern); zone-naming clarification (Zone 1 inner thinking / BindSpace substrate / Zone 2 callcenter / Zone 3 consumer boundary). The composition crystallizes from the recursion.

---

## §14 Status board entries to add post-spec-ratify

Per CLAUDE.md Mandatory Board-Hygiene Rule, when this plan lands and the first PR opens:

- `.claude/board/INTEGRATION_PLANS.md` — PREPEND new entry for this plan
- `.claude/board/LATEST_STATE.md` — when PR-CE64-MB-1 ships, add Contract Inventory row for par-tile types
- `.claude/board/STATUS_BOARD.md` — append D-CE64-MB-1 through D-CE64-MB-9 rows (Status = Queued initially)
- `.claude/board/EPIPHANIES.md` — PREPEND E-CE64-MB-1 through E-CE64-MB-10 entries (the 10 epiphanies in §9)
- `.claude/board/PR_ARC_INVENTORY.md` — append per merged PR in this series (one row each)

---

## §15 Final readiness checklist

Before spawning sprint workers against this spec:

- [ ] User ratifies OQ-1 (Σ-tier banding policy)
- [ ] User ratifies OQ-4 (INT4-32D cold-start wiring path)
- [ ] User ratifies OQ-5 (rayon vendor decision — start std::thread::scope, defer rayon-vendor)
- [ ] User confirms PR sequencing (§7) and per-PR ownership
- [ ] User confirms blast radius (§10) preserves Supabase realtime transcode untouched
- [ ] ndarray-side `PR-NDARRAY-MIRI-COMPLETE` lands first (closes u-word method gaps + `cfg(miri)` dispatch reroute)
- [ ] `.claude/board/INTEGRATION_PLANS.md` PREPEND entry drafted
- [ ] Sprint worker prompt template loaded with mandatory reads (this plan + 5 composed plans + Σ10 doctrine + zone framing §0)

When all checked, sprint-10 spec corpus is ready for CCA2A 12-worker fan-out.

---

**End of plan v1. This is the composition spec — no new architecture, just sequenced execution of named-and-reviewed pieces.**
