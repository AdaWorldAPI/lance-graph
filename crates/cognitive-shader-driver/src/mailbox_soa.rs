//! MailboxSoA<N>: spatial-temporal accumulator for per-row edge receipts.
//!
//! Per plan §9 — mailboxes are NOT message queues. Each row is a
//! single neuron with many incoming synapses; multi-source `CausalEdge64`
//! deliveries land via `apply_edges`; the row's energy integrates; rows at or
//! above `threshold` are "firing" (see `pending_count`) and are consumed by
//! the owner during its phase advance.
//!
//! D-CSV-7 scope (sprint-11 Phase B): core types + W-slot referencing +
//! per-row plasticity accumulator + apply_edges edge receipt. NO ractor
//! wrap, NO AttentionMask/LRU, NO cross-cycle rollup — those are W6's
//! orthogonal concerns / sprint-12 SigmaTierRouter integration.
//!
//! ## Zero-copy lifecycle (PR #477 three-tier model)
//!
//! This type is the per-mailbox, mailbox-owned Tier-1 SoA — the BindSpace
//! surrogate. The shared singleton `Arc<BindSpace>` dissolves *onto*
//! mailboxes (each owns its own LE-contract SoA columns:
//! edges/qualia/meta/entity_type — minus the deprecated `Vsa16kF32` plane),
//! it is NOT copied per mailbox. **There is no emission and no inter-mailbox
//! handoff type** — the SoA is zero-copy from creation to Lance tombstone;
//! Lance's columnar I/O writes LE bytes from this in-place store
//! (`SoaEnvelope` describes the geometry). The former `emit()` /
//! `CollapseGateEmission` path was removed per the ratified model
//! (`docs/architecture/soa-three-tier-model.md`). Ownership makes
//! no-alias/no-race a compile error (E-CE64-MB-4). Column map + gated steps:
//! `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`.

use causal_edge::CausalEdge64;
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::qualia::QualiaI4_16D;
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};

/// Canonical named-fingerprint plane width: 256 × u64 = 16,384 bits
/// (mirrors `bindspace::WORDS_PER_FP`; defined locally so the mailbox does NOT
/// depend on the singleton it is migrating off of — W7 deletes BindSpace, not this).
pub const WORDS_PER_FP: usize = 256;

/// Spatial-temporal accumulator for per-row edge receipts.
///
/// `N` is the maximum number of neuron rows this mailbox can serve.
/// Each row accumulates `energy` from incoming `CausalEdge64` deliveries
/// (via `apply_edges`); rows whose magnitude crosses `threshold` are
/// "firing" and are consumed in place by the owner (no emission).
///
/// # W-slot invariant
///
/// All accepted edges must have `edge.w_slot() == self.w_slot`.
/// Mismatched edges are silently dropped in `apply_edges`.
/// `w_slot` must be < 64 (6-bit limit per plan §6 L-6); the constructor
/// panics with a clear message if this is violated.
///
/// # Type alias
///
/// `DefaultMailboxSoA = MailboxSoA<1024>` — 4× current BindSpace row count.
pub struct MailboxSoA<const N: usize> {
    /// Corpus root handle (W-slot value lives here per plan §6 Option F).
    /// `MailboxId = u32` from contract is more than adequate for 64 corpora.
    pub mailbox_id: MailboxId,

    /// Per-row spatial-temporal energy accumulator.
    /// Energy integrates signed-mantissa × confidence contributions from
    /// incoming `CausalEdge64` batons until threshold crossing.
    pub energy: [f32; N],

    /// Per-row Hebbian integration counter (saturating u8 per W6 §4.4).
    /// Incremented once per accepted edge, never decremented within a cycle.
    pub plasticity_counter: [u8; N],

    /// Per-row last-active cycle stamp (renamed from `last_emission_cycle`
    /// per PR #477 — there is no emission; the stamp marks in-place
    /// consumption). Guards the "never consume the same firing row twice in
    /// one cycle" invariant: the owner stamps `last_active_cycle[row] =
    /// current_cycle` when it consumes a firing row during phase advance;
    /// rows already stamped this cycle are skipped.
    pub last_active_cycle: [u32; N],

    /// Per-row last-**write** cycle stamp (S2.5 cycle-aware write contract).
    /// Distinct from `last_active_cycle` (consumption): this records the cycle
    /// at which [`Self::write_row`] last accepted a write into the row. Kept
    /// separate so the write gate's *ordering* check never couples to
    /// `consume_firing`'s *exact-match* idempotency guard (`I-LEGACY-API-FEATURE-GATED`:
    /// same field, two semantics is forbidden). Sentinel `u32::MAX` = never written.
    pub last_write_cycle: [u32; N],

    // ── NEW: migrated thoughtspace columns (per-mailbox owned, D-MBX-A1) ──
    /// Per-row LE baton edge (`CausalEdge64`, 8 B/row).
    /// Migrated from `BindSpace.edges` (EdgeColumn).
    /// This IS the LE contract / baton edge for this mailbox row.
    pub edges: [CausalEdge64; N],

    /// Per-row affective role vector (`QualiaI4_16D`, 8 B/row).
    /// Migrated from `BindSpace.qualia` (QualiaI4Column).
    /// 16 signed i4 dimensions (arousal/valence/tension/…); 9× compression vs f32.
    pub qualia: [QualiaI4_16D; N],

    /// Per-row packed meta word (`MetaWord`, 4 B/row).
    /// Migrated from `BindSpace.meta` (MetaColumn).
    /// Layout: `thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6)`.
    pub meta: [MetaWord; N],

    /// Per-row OGIT entity-type index (`u16`, 2 B/row).
    /// Migrated from `BindSpace.entity_type`.
    /// 1-based index into the shared (immutable) ontology registry.
    /// The registry itself stays `Arc<OntologyRegistry>` (cold Zone-2, not owned here).
    pub entity_type: [u16; N],

    // ── NEW: D-MBX-A2 migrated columns (W1 — temporal / expert / sigma) ──
    /// Per-row temporal stamp (`u64`, 8 B/row). Migrated from `BindSpace.temporal`.
    ///
    /// Kept as a standalone column (OQ-2 fallback), NOT folded into the edge: the
    /// v2 `causal_edge::CausalEdge64` layout reclaimed the old temporal bits
    /// (`I-LEGACY-API-FEATURE-GATED`), so the edge cannot carry it. `current_cycle`
    /// is the mailbox-level clock; this is the per-row event stamp.
    pub temporal: [u64; N],

    /// Per-row expert/corpus id (`u16`, 2 B/row). Migrated from `BindSpace.expert`.
    ///
    /// Often subsumed by `mailbox_id` / `w_slot` (the mailbox *is* an expert), but
    /// kept per-row for multi-expert mailboxes during the migration.
    pub expert: [u16; N],

    /// Per-row Σ-codebook index (`u8`, 1 B/row). Migrated from
    /// `BindSpace.fingerprints.sigma`.
    ///
    /// A *reference* (1-byte index) into the 256-entry Σ codebook owned by
    /// `lance-graph-contract::sigma_propagation` — content, like the ontology and
    /// the CAM-PQ codebook, stays shared/cold and is NOT copied per row
    /// (`I-VSA-IDENTITIES`: indices, not content). The dense content/topic/angle
    /// identity planes are a separate W1b step.
    pub sigma: [u8; N],

    // ── NEW: D-MBX-A2 dense identity planes (W1b) ──
    // The content/topic/angle Hamming identity planes stay HOT in the mailbox
    // (~6 KB/thought; OQ-1 RESOLVED §2.7 — NOT reduced to a tiny ref). They are
    // HEAP `Box<[u64]>` of `N * WORDS_PER_FP` (a `[u64; N*256]` stack array is not
    // expressible on stable Rust and would be ~2 MB/plane at N=1024). The
    // deprecated `Vsa16kF32` `cycle` plane is NEVER migrated — compute it
    // transiently if a step needs it.
    /// Per-row content identity fingerprint (`WORDS_PER_FP` u64/row). Migrated from
    /// `BindSpace.fingerprints.content`. This is the heaviest column and the one the
    /// driver's resonance search reads (`content_row`).
    pub content: Box<[u64]>,

    /// Per-row topic identity plane (`WORDS_PER_FP` u64/row). Migrated from
    /// `BindSpace.fingerprints.topic`.
    pub topic: Box<[u64]>,

    /// Per-row angle identity plane (`WORDS_PER_FP` u64/row). Migrated from
    /// `BindSpace.fingerprints.angle`.
    pub angle: Box<[u64]>,

    /// Monotonic cycle stamp; advanced by `tick()`.
    pub current_cycle: u32,

    /// Count of writes rejected as stale by [`Self::write_row`] (telemetry).
    /// A late batch targeting a cycle the mailbox already advanced past is
    /// dropped, not applied — this counts those drops (drop-with-telemetry =
    /// the Strict `WriteDisposition`; an Aware local buffer is a future option).
    /// Mailbox-level (not per-row); [`Self::reset_row`] does NOT touch it.
    pub(crate) stale_write_count: u64,

    /// 6-bit W-slot value this mailbox represents.
    /// Incoming edges with `edge.w_slot() != self.w_slot` are rejected.
    /// Must be < 64 (plan §6 L-6).
    pub w_slot: u8,

    /// Firing threshold (default 1.0; tunable).
    /// A row is firing when `energy[row].abs() >= threshold`
    /// (see [`Self::pending_count`]).
    pub threshold: f32,

    /// **Declared populated-row count (W1c).** The logical row count this mailbox is
    /// using — the exact analogue of `BindSpace::len`, NOT the const capacity `N`.
    ///
    /// **Why it exists:** a zeroed `MetaWord` *passes* `MetaFilter::accepts`
    /// (`0 >= 0`, `thinking_mask == 0` accepts all). A prefilter sweep clamped to the
    /// capacity `N` (e.g. 1024) would therefore return `N − len` phantom rows for a
    /// mailbox that only uses `len` rows — diverging from a `BindSpace` window of
    /// `len`. Any row-bounded sweep (notably the migration read-shim's
    /// `meta_prefilter` analogue) MUST clamp to [`Self::populated`] (the logical
    /// row count), not the type-level capacity `N`. (Since W1c, `n_rows()` is
    /// bound to `populated`, so it is now safe to clamp to either — `N` is the
    /// only wrong choice.)
    ///
    /// **Semantics mirror `BindSpace::len`:** a *declared* size, set once at
    /// construction/population time via [`Self::set_populated`] (just as
    /// `BindSpace::zeros(len)` fixes `len`), NOT a per-write high-water-mark and NOT
    /// decremented by [`Self::reset_row`] (clearing a row's contents does not shrink
    /// the logical size, exactly as it does not change `BindSpace::len`). Defaults to
    /// `0` (an empty mailbox) until declared.
    pub(crate) populated: usize,

    /// The Rubicon lifecycle column this mailbox currently occupies — the
    /// **cognitive** FSM state (distinct from ractor's process-lifecycle
    /// `ActorStatus`; see `.claude/knowledge/orchestration-boundary-v1.md`).
    /// Mutated only via [`MailboxSoaOwner::advance_phase`] /
    /// [`MailboxSoaOwner::try_advance_phase`]; starts at
    /// [`KanbanColumn::Planning`]. Read it through the
    /// [`MailboxSoaView::phase`] getter. `pub(crate)` (not `pub`) so the
    /// "mutated only via the owner trait" invariant is compiler-enforced — a
    /// downstream crate cannot assign an arbitrary column directly and bypass
    /// the lifecycle DAG check + `KanbanMove` emission (PR #507 review).
    pub(crate) phase: KanbanColumn,
}

/// Default capacity: 1024 rows (4× current BindSpace row count).
pub type DefaultMailboxSoA = MailboxSoA<1024>;

/// Outcome of a cycle-aware row write through [`MailboxSoA::write_row`].
///
/// Infallible-with-outcome (NOT `Result`): ownership is compile-proven
/// (`&mut self`, E-CE64-MB-4), so "this write is for another cycle" is a valid
/// in-domain *outcome*, not an aliasing failure (council OQ-D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// `cycle == current_cycle` — the `cell` was applied and `last_write_cycle[row]`
    /// stamped to `cycle`.
    Accepted,
    /// `cycle` is (wrap-aware) strictly behind `current_cycle` — a late batch
    /// targeting a cycle the mailbox advanced past. Nothing mutated;
    /// `stale_write_count` incremented. This is the "nothing buffers a stale
    /// mailbox / no cycle-blind overwrite" guarantee.
    Stale,
    /// `cycle` is (wrap-aware) strictly ahead of `current_cycle`. Nothing
    /// mutated (Strict `WriteDisposition` default; an Aware buffer is a future
    /// option for the multi-producer interlace target).
    Future,
}

/// Field-presence staging cell for one cycle-aware row write.
///
/// Only `Some` fields are applied; `None` fields are left untouched. It is a
/// *view over what to write*, carrying no column data it does not need
/// (register-laziness, `I-VSA-IDENTITIES`). Slices borrow; scalars copy.
#[derive(Debug, Clone, Default)]
pub struct WriteCell<'a> {
    /// Content identity plane (`WORDS_PER_FP` u64) — borrowed.
    pub content: Option<&'a [u64]>,
    /// Topic identity plane (`WORDS_PER_FP` u64) — borrowed.
    pub topic: Option<&'a [u64]>,
    /// Angle identity plane (`WORDS_PER_FP` u64) — borrowed.
    pub angle: Option<&'a [u64]>,
    /// LE baton edge.
    pub edge: Option<CausalEdge64>,
    /// Affective i4-16D vector.
    pub qualia: Option<QualiaI4_16D>,
    /// Packed meta word.
    pub meta: Option<MetaWord>,
    /// OGIT entity-type index.
    pub entity_type: Option<u16>,
    /// Temporal stamp.
    pub temporal: Option<u64>,
    /// Expert/corpus id.
    pub expert: Option<u16>,
    /// Σ-codebook index.
    pub sigma: Option<u8>,
}

impl<const N: usize> MailboxSoA<N> {
    /// Construct a new `MailboxSoA` with all per-row state zero-initialised.
    ///
    /// # Panics
    ///
    /// Panics if `w_slot >= 64` — the W field is 6 bits (plan §6 L-6, 64 corpora max).
    /// This is a programming error, not a data error.
    pub fn new(mailbox_id: MailboxId, w_slot: u8, threshold: f32) -> Self {
        assert!(
            w_slot < 64,
            "w_slot must fit in 6 bits (0..=63 per plan §6 L-6), got {w_slot}"
        );
        Self {
            mailbox_id,
            energy: [0.0f32; N],
            plasticity_counter: [0u8; N],
            // u32::MAX is the "never consumed" sentinel. current_cycle starts at 0
            // and advances via wrapping_add(1); in practice it never reaches
            // u32::MAX during a session, so first consumption on any cycle is
            // always permitted (u32::MAX != any valid cycle stamp).
            last_active_cycle: [u32::MAX; N],
            // u32::MAX = "never written" sentinel (mirrors last_active_cycle).
            last_write_cycle: [u32::MAX; N],
            current_cycle: 0,
            stale_write_count: 0,
            w_slot,
            threshold,
            // ── NEW thoughtspace columns — zero-initialised (D-MBX-A1) ──
            edges: [CausalEdge64::ZERO; N],
            qualia: [QualiaI4_16D::ZERO; N],
            meta: [MetaWord(0); N],
            entity_type: [0u16; N],
            // ── NEW D-MBX-A2 columns — zero-initialised (W1) ──
            temporal: [0u64; N],
            expert: [0u16; N],
            sigma: [0u8; N],
            // ── NEW D-MBX-A2 dense identity planes — heap, zero-initialised (W1b) ──
            content: vec![0u64; N * WORDS_PER_FP].into_boxed_slice(),
            topic: vec![0u64; N * WORDS_PER_FP].into_boxed_slice(),
            angle: vec![0u64; N * WORDS_PER_FP].into_boxed_slice(),
            // ── W1c — empty mailbox: zero logical rows until `set_populated(...)`
            //          declares the size (no write path bumps this implicitly) ──
            populated: 0,
            // Pre-Rubicon: every mailbox starts in deliberation.
            phase: KanbanColumn::Planning,
        }
    }

    /// Accept a batch of `(target_row, CausalEdge64)` deliveries.
    ///
    /// For each delivery:
    /// - Rows out of `[0, N)` are silently dropped.
    /// - Edges whose `edge.w_slot()` differs from `self.w_slot` are silently
    ///   dropped (wrong corpus). Downstream telemetry can count rejections by
    ///   comparing the return value against `deliveries.len()`.
    /// - Accepted edges: `energy[row] += mantissa/8.0 * confidence`;
    ///   `plasticity_counter[row]` is incremented (saturating).
    ///
    /// Returns the count of accepted (non-dropped) deliveries.
    pub fn apply_edges(&mut self, deliveries: &[(u16, CausalEdge64)]) -> usize {
        let mut accepted = 0;
        for &(target, edge) in deliveries {
            let row = target as usize;
            if row >= N {
                continue;
            }
            if edge.w_slot() != self.w_slot {
                continue; // wrong corpus, skip
            }
            // Energy from edge: signed mantissa scaled by confidence.
            // mantissa range: -8..+7, divided by 8.0 → approximately ±1.0.
            let m = edge.inference_mantissa() as f32 / 8.0;
            let c = edge.confidence();
            self.energy[row] += m * c;
            self.plasticity_counter[row] = self.plasticity_counter[row].saturating_add(1);
            accepted += 1;
        }
        accepted
    }

    /// Consume one firing row in place: stamp `last_active_cycle[row]` to the
    /// current cycle and reset its energy, subject to the same-cycle
    /// idempotency guard.
    ///
    /// This is the zero-copy successor of the removed `emit()` path (PR #477
    /// three-tier model): nothing is packaged or transmitted — the owner
    /// reads the row's columns in place (edge/qualia/meta/entity_type) and
    /// then calls this to mark the row consumed for this cycle.
    ///
    /// Returns `true` if the row was consumed; `false` if the row is out of
    /// range, below threshold, or already consumed this cycle.
    pub fn consume_firing(&mut self, row: usize) -> bool {
        if row >= N {
            return false;
        }
        if self.energy[row].abs() < self.threshold {
            return false;
        }
        if self.last_active_cycle[row] == self.current_cycle {
            return false; // already consumed this cycle
        }
        self.last_active_cycle[row] = self.current_cycle;
        self.energy[row] = 0.0; // reset on consumption (single-shot per cycle)
        true
    }

    /// Advance to the next cycle.
    ///
    /// Wraps at `u32::MAX` (wrapping_add). This makes the
    /// same-cycle idempotency guard safe for long-running sessions.
    pub fn tick(&mut self) {
        self.current_cycle = self.current_cycle.wrapping_add(1);
    }

    /// Cycle-aware row write — the ONE deinterlacing mutator (S2.5).
    ///
    /// The gate is **wrap-aware** against `current_cycle` (a naive `<`/`>`
    /// misclassifies post-`u32`-wrap stragglers as `Future` across a long
    /// interlaced sweep):
    /// - `cycle == current_cycle` → apply the `Some` fields of `cell`, stamp
    ///   `last_write_cycle[row] = cycle`, return [`WriteOutcome::Accepted`].
    /// - `cycle` strictly behind (wrapping distance `< 2^31`) → mutate nothing,
    ///   increment `stale_write_count`, return [`WriteOutcome::Stale`].
    /// - `cycle` strictly ahead → mutate nothing, return [`WriteOutcome::Future`].
    ///
    /// Out-of-range `row` returns `Stale` without mutation (a row we do not own
    /// is never written). This is the cycle-blind-setter gap closed: no write
    /// lands without the owner's `current_cycle` agreeing.
    pub fn write_row(&mut self, row: usize, cycle: u32, cell: &WriteCell<'_>) -> WriteOutcome {
        if row >= N {
            return WriteOutcome::Stale;
        }
        // Wrap-aware: delta in [0, 2^31) ⇒ cycle is at-or-behind current
        // (0 = current, >0 = stale); [2^31, 2^32) ⇒ cycle is ahead (future).
        let delta = self.current_cycle.wrapping_sub(cycle);
        if delta == 0 {
            if let Some(w) = cell.content {
                self.set_content(row, w);
            }
            if let Some(w) = cell.topic {
                self.set_topic(row, w);
            }
            if let Some(w) = cell.angle {
                self.set_angle(row, w);
            }
            if let Some(e) = cell.edge {
                self.set_edge(row, e);
            }
            if let Some(q) = cell.qualia {
                self.set_qualia(row, q);
            }
            if let Some(m) = cell.meta {
                self.set_meta(row, m);
            }
            if let Some(t) = cell.entity_type {
                self.set_entity_type(row, t);
            }
            if let Some(t) = cell.temporal {
                self.set_temporal(row, t);
            }
            if let Some(x) = cell.expert {
                self.set_expert(row, x);
            }
            if let Some(s) = cell.sigma {
                self.set_sigma(row, s);
            }
            self.last_write_cycle[row] = cycle;
            WriteOutcome::Accepted
        } else if delta < 0x8000_0000 {
            self.stale_write_count = self.stale_write_count.saturating_add(1);
            WriteOutcome::Stale
        } else {
            WriteOutcome::Future
        }
    }

    /// Per-row last-**write** cycle stamp (`u32::MAX` = never written).
    #[inline]
    pub fn last_write_cycle_at(&self, row: usize) -> u32 {
        self.last_write_cycle[row]
    }

    /// Count of writes rejected as stale by [`Self::write_row`] (telemetry).
    #[inline]
    pub fn stale_write_count(&self) -> u64 {
        self.stale_write_count
    }

    /// Declared populated-row count (W1c) — the `BindSpace::len` analogue, NOT the
    /// type-level capacity `N`. Row-bounded sweeps (the migration read-shim's
    /// `meta_prefilter`) clamp to this logical size, so zeroed padding rows
    /// `populated..N` are not swept (a zeroed `MetaWord` would otherwise pass
    /// `MetaFilter::accepts`). Since W1c, [`MailboxSoaView::n_rows`] is bound to
    /// this field, so the two agree; only the const `N` is the wrong bound.
    /// This is a *declaration*, never an implicit per-write counter — callers
    /// manage it explicitly via [`Self::set_populated`].
    #[inline]
    pub fn populated(&self) -> usize {
        self.populated
    }

    /// Declare the populated-row count (clamped to the capacity `N`). Set this to the
    /// logical size the mailbox represents — e.g. when mirroring a `BindSpace` window
    /// of `len` rows, call `set_populated(len)`. Mirrors fixing `BindSpace::len` at
    /// construction; it is a declaration, not a per-write counter.
    #[inline]
    pub fn set_populated(&mut self, n: usize) {
        self.populated = n.min(N);
    }

    /// Reset one row to its zero-initialised state.
    ///
    /// Clears `energy`, `plasticity_counter`, and `last_active_cycle`
    /// for `row`. Out-of-range `row` values are silently ignored.
    /// Cross-cycle plasticity rollup is outside this scope (W6 §4.4 Zone 2).
    pub fn reset_row(&mut self, row: usize) {
        if row >= N {
            return;
        }
        self.energy[row] = 0.0;
        self.plasticity_counter[row] = 0;
        // Restore the "never consumed" sentinel so the row can fire immediately
        // on the next cycle without triggering the same-cycle guard.
        self.last_active_cycle[row] = u32::MAX;
        // Restore the "never written" sentinel (field-isolation: a new [u32; N]
        // that reset_row forgets is the exact leak the matrix test catches).
        self.last_write_cycle[row] = u32::MAX;
        // ── NEW thoughtspace columns reset (D-MBX-A1) ──
        self.edges[row] = CausalEdge64::ZERO;
        self.qualia[row] = QualiaI4_16D::ZERO;
        self.meta[row] = MetaWord(0);
        self.entity_type[row] = 0;
        // ── NEW D-MBX-A2 columns reset (W1) ──
        self.temporal[row] = 0;
        self.expert[row] = 0;
        self.sigma[row] = 0;
        // ── NEW D-MBX-A2 dense identity planes reset (W1b) ──
        let lo = row * WORDS_PER_FP;
        let hi = lo + WORDS_PER_FP;
        self.content[lo..hi].fill(0);
        self.topic[lo..hi].fill(0);
        self.angle[lo..hi].fill(0);
    }

    // ── Read-only inspectors ──────────────────────────────────────────────────

    /// Current energy accumulator for `row`. Panics (debug) or wraps (release)
    /// on out-of-bounds; callers should stay within `[0, N)`.
    #[inline]
    pub fn energy_at(&self, row: usize) -> f32 {
        self.energy[row]
    }

    /// Current plasticity counter for `row`.
    #[inline]
    pub fn plasticity_at(&self, row: usize) -> u8 {
        self.plasticity_counter[row]
    }

    /// Current monotonic cycle stamp.
    #[inline]
    pub fn cycle(&self) -> u32 {
        self.current_cycle
    }

    /// The 6-bit W-slot value this mailbox represents.
    #[inline]
    pub fn w_slot(&self) -> u8 {
        self.w_slot
    }

    /// Count of rows whose `|energy|` is at or above `threshold`.
    ///
    /// Useful for tests and telemetry without triggering emission side-effects.
    pub fn pending_count(&self) -> usize {
        self.energy
            .iter()
            .filter(|&&e| e.abs() >= self.threshold)
            .count()
    }

    // ── Thoughtspace column accessors (D-MBX-A1) ─────────────────────────────

    /// Return the `CausalEdge64` baton edge for `row`.
    ///
    /// Panics (debug) / wraps (release) on out-of-bounds; callers
    /// should stay within `[0, N)`.
    #[inline]
    pub fn edge(&self, row: usize) -> CausalEdge64 {
        self.edges[row]
    }

    /// Set the `CausalEdge64` baton edge for `row`.
    ///
    /// Panics (debug) / wraps (release) on out-of-bounds; callers
    /// should stay within `[0, N)`.
    #[inline]
    pub fn set_edge(&mut self, row: usize, e: CausalEdge64) {
        self.edges[row] = e;
    }

    /// Return the packed `QualiaI4_16D` affective vector for `row`.
    #[inline]
    pub fn qualia_at(&self, row: usize) -> QualiaI4_16D {
        self.qualia[row]
    }

    /// Set the packed `QualiaI4_16D` affective vector for `row`.
    #[inline]
    pub fn set_qualia(&mut self, row: usize, q: QualiaI4_16D) {
        self.qualia[row] = q;
    }

    /// Return the packed `MetaWord` for `row`.
    #[inline]
    pub fn meta_at(&self, row: usize) -> MetaWord {
        self.meta[row]
    }

    /// Set the packed `MetaWord` for `row`.
    #[inline]
    pub fn set_meta(&mut self, row: usize, m: MetaWord) {
        self.meta[row] = m;
    }

    /// Return the OGIT entity-type index for `row` (1-based, shared ontology).
    #[inline]
    pub fn entity_type_at(&self, row: usize) -> u16 {
        self.entity_type[row]
    }

    /// Set the OGIT entity-type index for `row`.
    #[inline]
    pub fn set_entity_type(&mut self, row: usize, t: u16) {
        self.entity_type[row] = t;
    }

    // ── D-MBX-A2 column accessors (W1: temporal / expert / sigma) ────────────

    /// Per-row temporal stamp for `row`.
    #[inline]
    pub fn temporal_at(&self, row: usize) -> u64 {
        self.temporal[row]
    }

    /// Set the per-row temporal stamp for `row`. (Distinct from the v2
    /// `CausalEdge64::set_temporal` no-op — this is the mailbox's standalone
    /// temporal column, the legitimate home per `I-LEGACY-API-FEATURE-GATED`.)
    #[inline]
    pub fn set_temporal(&mut self, row: usize, t: u64) {
        self.temporal[row] = t;
    }

    /// Per-row expert/corpus id for `row`.
    #[inline]
    pub fn expert_at(&self, row: usize) -> u16 {
        self.expert[row]
    }

    /// Set the per-row expert/corpus id for `row`.
    #[inline]
    pub fn set_expert(&mut self, row: usize, e: u16) {
        self.expert[row] = e;
    }

    /// Per-row Σ-codebook index for `row`.
    #[inline]
    pub fn sigma_at(&self, row: usize) -> u8 {
        self.sigma[row]
    }

    /// Set the per-row Σ-codebook index for `row`.
    #[inline]
    pub fn set_sigma(&mut self, row: usize, s: u8) {
        self.sigma[row] = s;
    }

    // ── D-MBX-A2 dense identity-plane accessors (W1b) ────────────────────────

    /// Zero-copy view of `row`'s content identity fingerprint (`WORDS_PER_FP`
    /// u64). This is the hot read the driver's resonance/Hamming search performs
    /// (the BindSpace equivalent is `FingerprintColumns::content_row`).
    #[inline]
    pub fn content_row(&self, row: usize) -> &[u64] {
        &self.content[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    /// Write `row`'s content identity fingerprint. Panics if `words.len() != WORDS_PER_FP`.
    #[inline]
    pub fn set_content(&mut self, row: usize, words: &[u64]) {
        assert_eq!(
            words.len(),
            WORDS_PER_FP,
            "content fingerprint must be WORDS_PER_FP u64"
        );
        self.content[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP].copy_from_slice(words);
    }

    /// Zero-copy view of `row`'s topic identity plane (`WORDS_PER_FP` u64).
    #[inline]
    pub fn topic_row(&self, row: usize) -> &[u64] {
        &self.topic[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    /// Write `row`'s topic identity plane. Panics if `words.len() != WORDS_PER_FP`.
    #[inline]
    pub fn set_topic(&mut self, row: usize, words: &[u64]) {
        assert_eq!(
            words.len(),
            WORDS_PER_FP,
            "topic plane must be WORDS_PER_FP u64"
        );
        self.topic[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP].copy_from_slice(words);
    }

    /// Zero-copy view of `row`'s angle identity plane (`WORDS_PER_FP` u64).
    #[inline]
    pub fn angle_row(&self, row: usize) -> &[u64] {
        &self.angle[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    /// Write `row`'s angle identity plane. Panics if `words.len() != WORDS_PER_FP`.
    #[inline]
    pub fn set_angle(&mut self, row: usize, words: &[u64]) {
        assert_eq!(
            words.len(),
            WORDS_PER_FP,
            "angle plane must be WORDS_PER_FP u64"
        );
        self.angle[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP].copy_from_slice(words);
    }
}

// ── Contract trait impls: MailboxSoA IS the in-RAM Rubicon owner ──────────────
//
// `MailboxSoaView` (read) + `MailboxSoaOwner` (write) make `MailboxSoA<N>` the
// in-RAM owner the contract names (`soa_view.rs` doc: "implements
// MailboxSoaOwner"). With these, a real `MailboxSoA` is BOTH the `view` a
// `VersionScheduler::on_version` reads AND the `owner` whose `try_advance_phase`
// applies the proposed `KanbanMove` — the in-process driving loop, no surreal /
// ractor message bus needed. The surreal-side external trigger (`surreal_container`
// view + `Notification → on_version`) is the fork-blocked follow-on (OQ-11.6).

impl<const N: usize> MailboxSoaView for MailboxSoA<N> {
    #[inline]
    fn mailbox_id(&self) -> MailboxId {
        self.mailbox_id
    }
    #[inline]
    fn n_rows(&self) -> usize {
        // Contract (`MailboxSoaView::n_rows`): "Number of POPULATED rows" — NOT the
        // const capacity `N`. Generic view consumers (e.g. `SoaWavePrimer::project`)
        // bound their row loop with `n_rows()`, so returning `N` would make them
        // scan the zeroed padding rows `populated..N` (a zeroed `MetaWord` passes
        // `MetaFilter::accepts`) — the exact phantom-row divergence W1c prevents.
        // Returns the W1c declared logical size; `N` (capacity) is a type-level const.
        self.populated
    }
    #[inline]
    fn w_slot(&self) -> u8 {
        self.w_slot
    }
    #[inline]
    fn current_cycle(&self) -> u32 {
        self.current_cycle
    }
    #[inline]
    fn phase(&self) -> KanbanColumn {
        self.phase
    }
    #[inline]
    fn energy(&self) -> &[f32] {
        &self.energy
    }
    #[inline]
    fn edges_raw(&self) -> &[u64] {
        // The `#[repr(transparent)]` on `CausalEdge64` (causal-edge `edge.rs`) is
        // the load-bearing layout guarantee for this cast; the const guards below
        // make a layout regression a COMPILE error (not silent UB) even if the
        // repr is ever removed upstream.
        const _: () = assert!(core::mem::size_of::<CausalEdge64>() == core::mem::size_of::<u64>());
        const _: () =
            assert!(core::mem::align_of::<CausalEdge64>() == core::mem::align_of::<u64>());
        // SAFETY: `CausalEdge64` is `#[repr(transparent)]` over `u64` (causal-edge
        // `edge.rs`), so `[CausalEdge64; N]` has identical size/alignment/layout to
        // `[u64; N]` (the two `const _` asserts above enforce it at compile time).
        // The pointer is non-null, aligned, and valid for `N` `u64` reads; the
        // returned slice borrows `&self`, so it cannot outlive the backing array.
        // Zero-copy — never clones the store (R1).
        unsafe { core::slice::from_raw_parts(self.edges.as_ptr() as *const u64, N) }
    }
    #[inline]
    fn meta_raw(&self) -> &[u32] {
        const _: () = assert!(core::mem::size_of::<MetaWord>() == core::mem::size_of::<u32>());
        const _: () = assert!(core::mem::align_of::<MetaWord>() == core::mem::align_of::<u32>());
        // SAFETY: `MetaWord` is `#[repr(transparent)]` over `u32`
        // (lance-graph-contract `cognitive_shader.rs`), so `[MetaWord; N]` has
        // identical layout to `[u32; N]` (const-asserted above). Same
        // validity/lifetime reasoning as `edges_raw`. Zero-copy.
        unsafe { core::slice::from_raw_parts(self.meta.as_ptr() as *const u32, N) }
    }
    #[inline]
    fn entity_type(&self) -> &[u16] {
        &self.entity_type
    }
}

impl<const N: usize> MailboxSoaOwner for MailboxSoA<N> {
    /// Advance the Rubicon column and emit the move. Prefer the trait's checked
    /// [`MailboxSoaOwner::try_advance_phase`] (it enforces the lifecycle DAG);
    /// this unchecked primitive is what it calls after the legality check passes.
    fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove {
        let from = self.phase;
        self.phase = to;
        KanbanMove {
            mailbox: self.mailbox_id,
            from,
            to,
            // Structural witness position (R4): the monotonic cycle stamp stands in
            // for the chain index until the witness_arc column lands — matching
            // `NextPhaseScheduler`'s convention. Read it as the SoA cycle-ownership
            // stamp via `KanbanMove::cycle()` (S2.5) — makes the move + planner +
            // SurrealQL exec cycle-aware off one source of truth. (No "emission":
            // the mailbox writes to itself in place; this is its own lifecycle
            // step recorded at its own current_cycle, per the #477 three-tier model.)
            witness_chain_position: self.current_cycle,
            libet_offset_us: if from == KanbanColumn::Planning && to == KanbanColumn::CognitiveWork
            {
                -550_000
            } else {
                0
            },
            exec: ExecTarget::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::CausalEdge64;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a CausalEdge64 with the given w_slot, signed mantissa, and
    /// confidence_u8 (0..=255, where 255 ≈ 1.0).
    fn make_edge(w_slot: u8, mantissa: i8, confidence_u8: u8) -> CausalEdge64 {
        let mut e = CausalEdge64::ZERO;
        e.set_confidence_u8(confidence_u8);
        e = e.with_w_slot(w_slot).with_inference_mantissa(mantissa);
        e
    }

    // ── test 1: new ───────────────────────────────────────────────────────────

    /// New mailbox must have all per-row state initialised correctly.
    ///
    /// energy and plasticity_counter are zero; last_active_cycle is u32::MAX
    /// (the "never consumed" sentinel so cycle 0 can consume without false guard).
    #[test]
    fn test_mailbox_soa_new_zero() {
        let mb: MailboxSoA<8> = MailboxSoA::new(7, 5, 1.0);
        assert_eq!(mb.mailbox_id, 7);
        assert_eq!(mb.w_slot, 5);
        assert_eq!(mb.threshold, 1.0);
        assert_eq!(mb.current_cycle, 0);
        for row in 0..8 {
            assert_eq!(mb.energy_at(row), 0.0, "energy[{row}] should be 0");
            assert_eq!(
                mb.plasticity_at(row),
                0,
                "plasticity_counter[{row}] should be 0"
            );
            assert_eq!(
                mb.last_active_cycle[row],
                u32::MAX,
                "last_active_cycle[{row}] should be u32::MAX (never-consumed sentinel)"
            );
        }
    }

    // ── test 2: w_slot panic ─────────────────────────────────────────────────

    /// w_slot >= 64 must panic with a clear message (programming error).
    #[test]
    #[should_panic(expected = "w_slot must fit in 6 bits (0..=63 per plan §6 L-6), got 64")]
    fn test_mailbox_soa_w_slot_panic_over_63() {
        let _mb: MailboxSoA<8> = MailboxSoA::new(0, 64, 1.0);
    }

    // ── test 3: accumulate ───────────────────────────────────────────────────

    /// Three batons targeting row 5 with mantissa=5 and full confidence.
    /// energy[5] should be 3 * (5/8.0 * confidence).
    #[test]
    fn test_mailbox_soa_apply_edges_accumulates() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(1, 3, 1.0);
        let confidence_u8 = 255u8; // ≈ 1.0
        let edge = make_edge(3, 5, confidence_u8);
        let batons = vec![(5u16, edge), (5u16, edge), (5u16, edge)];
        let accepted = mb.apply_edges(&batons);

        assert_eq!(accepted, 3, "all three batons should be accepted");
        assert_eq!(mb.plasticity_at(5), 3, "plasticity_counter[5] should be 3");

        let c = edge.confidence();
        let expected = 3.0 * (5.0 / 8.0) * c;
        let got = mb.energy_at(5);
        assert!(
            (got - expected).abs() < 1e-5,
            "energy[5] = {got}, expected ≈ {expected}"
        );

        // Other rows must be untouched
        for row in 0..8 {
            if row == 5 {
                continue;
            }
            assert_eq!(mb.energy_at(row), 0.0, "energy[{row}] should be 0");
        }
    }

    // ── test 4: wrong w_slot rejected ────────────────────────────────────────

    /// Baton with edge.w_slot() != self.w_slot must be silently dropped.
    #[test]
    fn test_mailbox_soa_apply_edges_rejects_wrong_w_slot() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(1, 3, 1.0);
        // Edge has w_slot = 4, mailbox expects w_slot = 3
        let edge = make_edge(4, 5, 200);
        let batons = vec![(2u16, edge)];
        let accepted = mb.apply_edges(&batons);

        assert_eq!(accepted, 0, "wrong-w_slot baton must be rejected");
        assert_eq!(mb.energy_at(2), 0.0, "energy[2] must be unchanged");
        assert_eq!(
            mb.plasticity_at(2),
            0,
            "plasticity_counter[2] must be unchanged"
        );
    }

    // ── test 5: out-of-range target rejected ─────────────────────────────────

    /// Targets >= N must be silently dropped.
    #[test]
    fn test_mailbox_soa_apply_edges_rejects_out_of_range_target() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(1, 2, 1.0);
        let edge = make_edge(2, 3, 200);
        // N = 8, so targets 8 and 13 are both out of range
        let batons = vec![(8u16, edge), (13u16, edge)];
        let accepted = mb.apply_edges(&batons);

        assert_eq!(accepted, 0, "out-of-range targets must be rejected");
        // All energy must remain zero
        for row in 0..8 {
            assert_eq!(mb.energy_at(row), 0.0);
        }
    }

    // ── test 6: consume firing row above threshold ────────────────────────────

    /// A row above threshold must be consumable in place: energy resets and
    /// the cycle stamp is set. No emission object exists (PR #477).
    #[test]
    fn test_mailbox_soa_consume_firing_above_threshold() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(10, 1, 1.0);
        // Directly set energy[3] to 1.5 (above 1.0 threshold)
        mb.energy[3] = 1.5;

        assert_eq!(mb.pending_count(), 1, "exactly one row should be firing");
        assert!(mb.consume_firing(3), "firing row must be consumable");
        assert_eq!(
            mb.energy_at(3),
            0.0,
            "energy[3] must reset to 0 after consumption"
        );
        assert_eq!(
            mb.last_active_cycle[3], mb.current_cycle,
            "last_active_cycle[3] must equal current_cycle"
        );
    }

    // ── test 7: below threshold — not consumable ─────────────────────────────

    /// Row energy below threshold must NOT be consumable.
    #[test]
    fn test_mailbox_soa_consume_below_threshold_rejected() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(5, 2, 1.0);
        mb.energy[3] = 0.5; // below 1.0 threshold

        assert!(
            !mb.consume_firing(3),
            "below-threshold row must not consume"
        );
        assert_eq!(mb.energy_at(3), 0.5, "energy[3] must be unchanged");
    }

    // ── test 8: double consume same cycle is idempotent ──────────────────────

    /// Second consume_firing() on the same cycle (no tick) must be blocked
    /// for rows already consumed, even if energy re-accumulated.
    #[test]
    fn test_mailbox_soa_consume_double_call_same_cycle_idempotent() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
        mb.energy[1] = 2.0;

        assert!(mb.consume_firing(1), "first consume must succeed");

        // Re-accumulate energy so energy[1] is above threshold again —
        // but the last_active_cycle guard must block re-consumption.
        mb.energy[1] = 2.0;
        assert!(
            !mb.consume_firing(1),
            "second consume same cycle must be blocked by last_active_cycle guard"
        );
    }

    // ── test 9: tick then re-accumulate then consume ──────────────────────────

    /// After tick(), a row can be consumed again in the new cycle.
    #[test]
    fn test_mailbox_soa_tick_then_consume_after_re_accumulation() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(3, 1, 1.0);
        mb.energy[4] = 1.5;

        // First consumption
        assert!(mb.consume_firing(4));
        assert_eq!(mb.energy_at(4), 0.0);

        // Advance cycle
        mb.tick();

        // Re-accumulate enough energy
        let edge = make_edge(1, 7, 255); // w_slot=1 matches mb.w_slot=1
        let deliveries = vec![(4u16, edge)];
        mb.apply_edges(&deliveries);
        // energy[4] = (7/8.0) * (255/255.0) ≈ 0.875 — below 1.0 threshold.
        // Let's set it directly to be safe.
        mb.energy[4] = 1.2;

        // Second consumption must succeed in the new cycle
        assert!(
            mb.consume_firing(4),
            "after tick, row 4 must be consumable again"
        );
    }

    // ── test 10: out-of-range row not consumable ──────────────────────────────

    /// consume_firing on a row >= N must return false and not panic.
    #[test]
    fn test_mailbox_soa_consume_out_of_range_rejected() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(1, 0, 1.0);
        assert!(!mb.consume_firing(8), "row == N must be rejected");
        assert!(!mb.consume_firing(100), "row > N must be rejected");
    }

    // ── test 11: the in-RAM driving loop (OUT+IN over a REAL MailboxSoA) ──────

    /// `VersionScheduler::on_version` proposes → `MailboxSoaOwner::try_advance_phase`
    /// applies, on a real `MailboxSoA` (not the contract's `FakeView`/`FakeSoa`).
    /// Drives the Rubicon forward arc Planning → CognitiveWork → Evaluation →
    /// Commit and halts at the absorbing column. This is the unblocked in-RAM
    /// loop — surreal's external version trigger is the fork-blocked follow-on.
    #[test]
    fn test_in_ram_driving_loop_walks_rubicon_to_commit() {
        use lance_graph_contract::kanban::ExecTarget;
        use lance_graph_contract::scheduler::{
            DatasetVersion, NextPhaseScheduler, VersionScheduler,
        };

        let mut mb: MailboxSoA<8> = MailboxSoA::new(42, 5, 1.0);
        assert_eq!(mb.phase(), KanbanColumn::Planning, "starts pre-Rubicon");

        let sched = NextPhaseScheduler;
        let mut steps = 0u32;
        let mut first_libet = 0i32;
        for v in 1..=10u64 {
            // IN-direction: the scheduler lowers a version tick to the next move…
            let Some(mv) = sched.on_version(&mb, DatasetVersion(v), ExecTarget::Native) else {
                break; // absorbing column reached — the cycle ended
            };
            // …proposed from the mailbox's CURRENT phase…
            assert_eq!(mv.from, mb.phase());
            // …and OUT-direction: the owner applies it through the checked airgap.
            let applied = mb
                .try_advance_phase(mv.to)
                .expect("the scheduler proposes only legal Rubicon edges");
            assert_eq!(applied.to, mv.to);
            if steps == 0 {
                first_libet = applied.libet_offset_us;
            }
            steps += 1;
        }

        assert_eq!(
            mb.phase(),
            KanbanColumn::Commit,
            "the forward arc calcifies at Commit"
        );
        assert!(mb.phase().is_absorbing());
        assert_eq!(
            steps, 3,
            "Planning→CognitiveWork→Evaluation→Commit = 3 advances"
        );
        assert_eq!(
            first_libet, -550_000,
            "the Planning→CognitiveWork crossing carries the Libet −550 ms anchor"
        );
    }

    // ── test 12: the unsafe zero-copy reinterpret casts round-trip ───────────

    /// Exercises the `repr(transparent)` reinterpret casts in `edges_raw()` /
    /// `meta_raw()` — the only genuinely `unsafe` code path in this impl, which
    /// the driving-loop test never touches. Writes known bit patterns into the
    /// `[CausalEdge64; N]` / `[MetaWord; N]` columns and asserts the `&[u64]` /
    /// `&[u32]` views read the SAME bytes back. A layout regression on either
    /// newtype would fail HERE (in addition to the `const _` size/align guards
    /// at the cast site failing to compile).
    #[test]
    fn test_edges_raw_meta_raw_reinterpret_round_trips() {
        let mut mb: MailboxSoA<4> = MailboxSoA::new(1, 0, 1.0);
        // Distinct, non-trivial bit patterns per row.
        for (i, raw) in [0xDEAD_BEEF_CAFE_0001u64, 2, 0xFFFF_FFFF_FFFF_FFFF, 0]
            .into_iter()
            .enumerate()
        {
            mb.edges[i] = CausalEdge64(raw);
            mb.meta[i] = MetaWord((raw & 0xFFFF_FFFF) as u32);
        }

        let edges_view: &[u64] = mb.edges_raw();
        let meta_view: &[u32] = mb.meta_raw();
        assert_eq!(edges_view.len(), 4);
        assert_eq!(meta_view.len(), 4);
        for i in 0..4 {
            // The reinterpret reads the EXACT u64/u32 backing the newtype.
            assert_eq!(edges_view[i], mb.edges[i].0, "edges_raw[{i}] bit-exact");
            assert_eq!(meta_view[i], mb.meta[i].0, "meta_raw[{i}] bit-exact");
        }
        // Pointer identity: the view borrows the column, never copies it (R1).
        assert_eq!(
            edges_view.as_ptr() as usize,
            mb.edges.as_ptr() as usize,
            "edges_raw is zero-copy (same backing pointer)"
        );
        assert_eq!(
            meta_view.as_ptr() as usize,
            mb.meta.as_ptr() as usize,
            "meta_raw is zero-copy (same backing pointer)"
        );
    }

    // ── test 13: W1 column parity — MailboxSoA carries what BindSpace carries ──

    /// **The "test the new before deleting the old" proof (W1).** Write distinct
    /// per-row values to a `BindSpace` window AND mirror them into a `MailboxSoA`,
    /// then assert every migrated LE-contract column reads back identically:
    /// `edges` / `qualia` / `meta` / `entity_type` (D-MBX-A1) plus the new
    /// `temporal` / `expert` / `sigma` (D-MBX-A2, W1). This proves the mailbox is a
    /// faithful carrier for these columns — it deletes nothing and touches no
    /// dispatch path. The dense `content`/`topic`/`angle` identity planes are the
    /// separate W1b step; the deprecated `cycle` (Vsa16kF32) plane is never migrated.
    #[test]
    fn test_mailbox_soa_column_parity_with_bindspace() {
        use crate::bindspace::BindSpace;
        use lance_graph_contract::cognitive_shader::MetaWord;
        use lance_graph_contract::qualia::QualiaI4_16D;

        const N: usize = 8;
        let mut bs = BindSpace::zeros(N);
        let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);

        for row in 0..N {
            let edge = 0xABCD_0000u64 | row as u64;
            let q = QualiaI4_16D::ZERO
                .with(0, (row % 7) as i8)
                .with(7, -((row % 8) as i8));
            let m = MetaWord::new(
                (row % 12) as u8,
                1,
                (row * 10) as u8,
                (row * 7) as u8,
                (row % 6) as u8,
            );
            let etype = (100 + row) as u16;
            let temporal = 0xDEAD_0000u64 | row as u64;
            let expert = (200 + row) as u16;
            let sigma = (row * 3) as u8;

            // BindSpace side (the singleton, source).
            bs.edges.set(row, edge);
            bs.qualia.set(row, q);
            bs.meta.set(row, m);
            bs.entity_type[row] = etype;
            bs.temporal[row] = temporal;
            bs.expert[row] = expert;
            bs.fingerprints.write_sigma(row, sigma);

            // MailboxSoA side (the migrated owner).
            mb.set_edge(row, CausalEdge64(edge));
            mb.set_qualia(row, q);
            mb.set_meta(row, m);
            mb.set_entity_type(row, etype);
            mb.set_temporal(row, temporal);
            mb.set_expert(row, expert);
            mb.set_sigma(row, sigma);
        }

        for row in 0..N {
            assert_eq!(mb.edge(row).0, bs.edges.get(row), "edges[{row}]");
            assert_eq!(mb.qualia_at(row), bs.qualia.row(row), "qualia[{row}]");
            assert_eq!(mb.meta_at(row).0, bs.meta.get(row).0, "meta[{row}]");
            assert_eq!(
                mb.entity_type_at(row),
                bs.entity_type[row],
                "entity_type[{row}]"
            );
            assert_eq!(mb.temporal_at(row), bs.temporal[row], "temporal[{row}]");
            assert_eq!(mb.expert_at(row), bs.expert[row], "expert[{row}]");
            assert_eq!(
                mb.sigma_at(row),
                bs.fingerprints.sigma_at(row),
                "sigma[{row}]"
            );
        }
    }

    // ── test 14: reset_row clears the W1 A2 columns ──────────────────────────

    /// `reset_row()` must clear the new `temporal` / `expert` / `sigma` columns
    /// (migration-invariant regression guard — a reset that forgot a new column
    /// would leak stale per-row state into a reused mailbox row).
    #[test]
    fn test_mailbox_soa_reset_row_clears_a2_columns() {
        let mut mb: MailboxSoA<4> = MailboxSoA::new(1, 0, 1.0);
        mb.set_temporal(2, 123);
        mb.set_expert(2, 77);
        mb.set_sigma(2, 9);

        mb.reset_row(2);

        assert_eq!(mb.temporal_at(2), 0, "temporal[2] must reset to 0");
        assert_eq!(mb.expert_at(2), 0, "expert[2] must reset to 0");
        assert_eq!(mb.sigma_at(2), 0, "sigma[2] must reset to 0");
    }

    // ── test 15: W1b dense identity planes — parity with BindSpace ───────────

    /// **The W1b "test the new" proof.** The content/topic/angle Hamming identity
    /// planes stay hot in the mailbox (OQ-1). For `content` — the migration-critical
    /// plane the driver's resonance search reads — assert byte parity against a
    /// `BindSpace` window written with the same words. For `topic`/`angle`, BindSpace
    /// exposes no public setter (they default zero there), so assert full round-trip
    /// correctness on the mailbox. The deprecated `cycle` (Vsa16kF32) plane is never
    /// migrated. Deletes nothing.
    #[test]
    fn test_mailbox_soa_dense_planes_parity_with_bindspace() {
        use crate::bindspace::BindSpace;

        const N: usize = 4;
        let mut bs = BindSpace::zeros(N);
        let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);

        // Distinct per-row, per-plane bit patterns so a cross-row or cross-plane
        // mixup fails. Set words across the full 256-word span.
        let mk = |row: usize, plane: u64| -> [u64; WORDS_PER_FP] {
            let mut w = [0u64; WORDS_PER_FP];
            w[0] = 0x1000_0000_0000_0000 | ((row as u64) << 8) | plane;
            w[row % WORDS_PER_FP] |= 1u64 << (row as u32 % 64);
            w[WORDS_PER_FP - 1] = plane.wrapping_mul(0x9E37_79B9) ^ row as u64;
            w
        };
        for row in 0..N {
            let (c, t, a) = (mk(row, 1), mk(row, 2), mk(row, 3));
            // content: write to BOTH (BindSpace exposes set_content) → true parity.
            bs.fingerprints.set_content(row, &c);
            mb.set_content(row, &c);
            // topic/angle: mailbox-only round-trip (no public BindSpace setter).
            mb.set_topic(row, &t);
            mb.set_angle(row, &a);
        }

        for row in 0..N {
            // content: byte-identical to the BindSpace plane (the hot read path).
            assert_eq!(
                mb.content_row(row),
                bs.fingerprints.content_row(row),
                "content[{row}] plane parity vs BindSpace"
            );
            // topic/angle: full-slice round-trip on the mailbox.
            assert_eq!(
                mb.topic_row(row),
                &mk(row, 2)[..],
                "topic[{row}] round-trip"
            );
            assert_eq!(
                mb.angle_row(row),
                &mk(row, 3)[..],
                "angle[{row}] round-trip"
            );
        }
    }

    // ── test 16: reset_row clears the W1b dense planes ───────────────────────

    /// `reset_row()` must zero the content/topic/angle plane spans for the row.
    #[test]
    fn test_mailbox_soa_reset_row_clears_dense_planes() {
        const N: usize = 4;
        let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
        let mut w = [0u64; WORDS_PER_FP];
        w[0] = 0xDEAD_BEEF;
        w[WORDS_PER_FP - 1] = 0xCAFE;
        mb.set_content(2, &w);
        mb.set_topic(2, &w);
        mb.set_angle(2, &w);

        mb.reset_row(2);

        assert!(
            mb.content_row(2).iter().all(|&x| x == 0),
            "content row cleared"
        );
        assert!(mb.topic_row(2).iter().all(|&x| x == 0), "topic row cleared");
        assert!(mb.angle_row(2).iter().all(|&x| x == 0), "angle row cleared");
        // A neighbouring row must be untouched by the reset (span isolation).
        mb.set_content(3, &w);
        mb.reset_row(2);
        assert_eq!(
            mb.content_row(3)[0],
            0xDEAD_BEEF,
            "row 3 content must survive row-2 reset"
        );
    }

    // ── test 17: W1c populated() — the prefilter-bound declaration ───────────

    /// `populated()` is the `BindSpace::len` analogue (declared logical size), NOT
    /// the const capacity `N`. It defaults to 0, is set via `set_populated` (clamped
    /// to `N`), and is NOT shrunk by `reset_row` — mirroring `BindSpace::len`, which
    /// is fixed at construction regardless of row contents. The contract trait method
    /// **`MailboxSoaView::n_rows()` reflects `populated()`** (its doc: "Number of
    /// populated rows") so generic view consumers (`SoaWavePrimer::project`) that
    /// bound `0..n_rows()` do NOT scan the zeroed padding rows `populated..N` (a
    /// zeroed `MetaWord` passes `MetaFilter::accepts`).
    #[test]
    fn test_mailbox_soa_populated_is_declared_len_not_capacity() {
        let mut mb: MailboxSoA<1024> = MailboxSoA::new(1, 0, 1.0);
        assert_eq!(mb.populated(), 0, "empty mailbox uses zero rows");
        // The trait surface mirrors the declared size, NOT the const capacity N=1024.
        assert_eq!(
            mb.n_rows(),
            0,
            "n_rows() (trait) reflects populated, not capacity N — the phantom-row guard"
        );

        mb.set_populated(4);
        assert_eq!(mb.populated(), 4, "declared logical size");
        assert_eq!(
            mb.n_rows(),
            4,
            "n_rows() (trait) tracks populated() so view sweeps clamp correctly"
        );

        // reset_row clears contents but does NOT shrink the declared size.
        mb.set_content(2, &[0u64; WORDS_PER_FP]);
        mb.reset_row(2);
        assert_eq!(
            mb.populated(),
            4,
            "reset_row must not change populated (mirrors BindSpace::len)"
        );

        // set_populated clamps to the capacity N — never exceeds the backing arrays.
        mb.set_populated(9999);
        assert_eq!(mb.populated(), 1024, "set_populated clamps to N");
        assert_eq!(mb.n_rows(), 1024, "n_rows() tracks the clamped populated");
    }

    // ── test 18: W4a field-isolation matrix — each column write is independent ─

    /// **The layout-bit-boundary regression guard (W4a, I-LEGACY-API-FEATURE-GATED).**
    /// Writing one migrated column on a row must leave EVERY other migrated column
    /// on that row byte-unchanged. This is the field-isolation matrix the iron rule
    /// mandates whenever a layout reclaims or co-locates per-row state. We seed a
    /// row with a known baseline across all columns, then mutate exactly one column
    /// at a time and assert the others are untouched.
    #[test]
    fn test_mailbox_soa_field_isolation_matrix() {
        const N: usize = 4;
        const R: usize = 2;

        // Baseline values (distinct, non-zero where the column allows).
        let base_edge = CausalEdge64(0x1111_2222_3333_4444);
        let base_qualia = QualiaI4_16D::ZERO.with(0, 3).with(5, -4);
        let base_meta = MetaWord::new(5, 2, 100, 120, 7);
        let base_etype = 42u16;
        let base_temporal = 0x9999_0000_0000_0001u64;
        let base_expert = 77u16;
        let base_sigma = 9u8;
        let base_content = {
            let mut w = [0u64; WORDS_PER_FP];
            w[0] = 0xCAFE;
            w[WORDS_PER_FP - 1] = 0xBEEF;
            w
        };

        let seed = |mb: &mut MailboxSoA<N>| {
            mb.set_edge(R, base_edge);
            mb.set_qualia(R, base_qualia);
            mb.set_meta(R, base_meta);
            mb.set_entity_type(R, base_etype);
            mb.set_temporal(R, base_temporal);
            mb.set_expert(R, base_expert);
            mb.set_sigma(R, base_sigma);
            mb.set_content(R, &base_content);
        };

        // Assert all columns EXCEPT `changed` match baseline. `changed` is a tag.
        let assert_others_unchanged = |mb: &MailboxSoA<N>, changed: &str| {
            if changed != "edge" {
                assert_eq!(mb.edge(R).0, base_edge.0, "edge changed by {changed}");
            }
            if changed != "qualia" {
                assert_eq!(mb.qualia_at(R), base_qualia, "qualia changed by {changed}");
            }
            if changed != "meta" {
                assert_eq!(mb.meta_at(R).0, base_meta.0, "meta changed by {changed}");
            }
            if changed != "entity_type" {
                assert_eq!(
                    mb.entity_type_at(R),
                    base_etype,
                    "entity_type changed by {changed}"
                );
            }
            if changed != "temporal" {
                assert_eq!(
                    mb.temporal_at(R),
                    base_temporal,
                    "temporal changed by {changed}"
                );
            }
            if changed != "expert" {
                assert_eq!(mb.expert_at(R), base_expert, "expert changed by {changed}");
            }
            if changed != "sigma" {
                assert_eq!(mb.sigma_at(R), base_sigma, "sigma changed by {changed}");
            }
            if changed != "content" {
                assert_eq!(
                    mb.content_row(R),
                    &base_content[..],
                    "content changed by {changed}"
                );
            }
        };

        // Mutate each column to a DISTINCT new value, one at a time, from a fresh
        // baseline each iteration, and assert isolation.
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            mb.set_edge(R, CausalEdge64(0xDEAD_BEEF_DEAD_BEEF));
            assert_others_unchanged(&mb, "edge");
        }
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            mb.set_qualia(R, QualiaI4_16D::ZERO.with(15, 7));
            assert_others_unchanged(&mb, "qualia");
        }
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            mb.set_meta(R, MetaWord::new(11, 3, 5, 6, 1));
            assert_others_unchanged(&mb, "meta");
        }
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            mb.set_entity_type(R, 999);
            assert_others_unchanged(&mb, "entity_type");
        }
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            mb.set_temporal(R, 0x1234_5678_9ABC_DEF0);
            assert_others_unchanged(&mb, "temporal");
        }
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            mb.set_expert(R, 12345);
            assert_others_unchanged(&mb, "expert");
        }
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            mb.set_sigma(R, 200);
            assert_others_unchanged(&mb, "sigma");
        }
        {
            let mut mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
            seed(&mut mb);
            let mut other = [0u64; WORDS_PER_FP];
            other[3] = 0xABCD;
            mb.set_content(R, &other);
            assert_others_unchanged(&mb, "content");
        }
    }

    // ── test 19: cycle-plane footprint — no cycle* symbol, ~6 KB/row hot ──────

    /// **The cycle-drop proof (cycle-plane is NEVER migrated).** The mailbox's
    /// hot per-row footprint is the three dense identity planes (content / topic
    /// / angle), each `WORDS_PER_FP` u64 = 2 KB, totalling ≈ 6 KB/row — NOT the
    /// ≈ 71.6 KB/row a BindSpace row costs (dominated by the 64 KB Vsa16kF32
    /// `cycle` plane). The absence of any `cycle*` storage on `MailboxSoA` is the
    /// structural guarantee; this test pins the dense-plane byte count.
    #[test]
    fn test_mailbox_soa_hot_footprint_excludes_cycle_plane() {
        const N: usize = 1024;
        // Per-row dense identity planes: 3 × WORDS_PER_FP × 8 bytes.
        let per_row_dense = 3 * WORDS_PER_FP * 8;
        assert_eq!(per_row_dense, 6144, "content+topic+angle = 6 KB/row");

        // The whole dense backing store is exactly N × per_row_dense bytes.
        let mb: MailboxSoA<N> = MailboxSoA::new(1, 0, 1.0);
        let dense_bytes = (mb.content.len() + mb.topic.len() + mb.angle.len()) * 8;
        assert_eq!(
            dense_bytes,
            N * per_row_dense,
            "dense planes total = N × 6 KB"
        );

        // The BindSpace per-row cost (incl. the 64 KB Vsa16kF32 cycle plane) is
        // ~71.6 KB — the mailbox is ~12× lighter per row because cycle is dropped.
        const BINDSPACE_PER_ROW: usize = 71_713; // see bindspace::tests footprint(1)
        assert!(
            per_row_dense * 11 < BINDSPACE_PER_ROW,
            "mailbox hot row ({per_row_dense} B) must be >10× lighter than a \
             BindSpace row ({BINDSPACE_PER_ROW} B) — the dropped 64 KB cycle plane"
        );
    }
}
