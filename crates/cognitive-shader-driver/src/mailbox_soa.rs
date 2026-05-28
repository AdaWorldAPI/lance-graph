//! MailboxSoA<N>: spatial-temporal accumulator for per-row baton receipts.
//!
//! Per plan §9 — mailboxes are NOT message queues. Each row is a
//! single neuron with many incoming synapses; multi-source batons land
//! via `apply_edges`; the row's energy integrates; when threshold crosses,
//! the row emits via the receiving CollapseGate.
//!
//! D-CSV-7 scope (sprint-11 Phase B): core types + W-slot referencing +
//! per-row plasticity accumulator + apply_edges baton receipt. NO ractor
//! wrap, NO AttentionMask/LRU, NO cross-cycle rollup — those are W6's
//! orthogonal concerns / sprint-12 SigmaTierRouter integration.
//!
//! Migration target (design, NOT yet wired): this type is the per-mailbox,
//! mailbox-owned, *ephemeral* "thoughtspace" — the BindSpace surrogate. The
//! shared singleton `Arc<BindSpace>` dissolves *onto* mailboxes (each owns its
//! own LE-contract SoA columns: edges/qualia/meta/entity_type — minus the
//! deprecated `Vsa16kF32` plane), it is NOT copied per mailbox. The only
//! cross-boundary state stays the LE baton `(u16, CausalEdge64)` (E-BATON-1);
//! ownership makes no-alias/no-race a compile error (E-CE64-MB-4). Column map +
//! gated steps: `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`.

use causal_edge::CausalEdge64;
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::collapse_gate::{CollapseGateEmission, MailboxId, MergeMode};
use lance_graph_contract::qualia::QualiaI4_16D;

/// Spatial-temporal accumulator for per-row baton receipts.
///
/// `N` is the maximum number of neuron rows this mailbox can serve.
/// Each row accumulates `energy` from incoming `CausalEdge64` batons
/// (via `apply_edges`) and emits when the magnitude crosses `threshold`.
///
/// # W-slot invariant
///
/// All accepted batons must have `edge.w_slot() == self.w_slot`.
/// Mismatched batons are silently dropped in `apply_edges`.
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
    /// Incremented once per accepted baton, never decremented within a cycle.
    pub plasticity_counter: [u8; N],

    /// Per-row last-emission cycle stamp.
    /// Guards the "never emit on the same cycle twice" invariant:
    /// if `last_emission_cycle[row] == current_cycle`, emission is suppressed.
    pub last_emission_cycle: [u32; N],

    // ── NEW: migrated thoughtspace columns (per-mailbox owned, D-MBX-A1) ──

    /// work
    /// Per-row LE baton edge (`CausalEdge64`, 8 B/row).
    /// Migrated from `BindSpace.edges` (EdgeColumn).
    /// This IS the LE contract / baton edge for this mailbox row.
    pub edges: [CausalEdge64; N],

    /// work
    /// Per-row affective role vector (`QualiaI4_16D`, 8 B/row).
    /// Migrated from `BindSpace.qualia` (QualiaI4Column).
    /// 16 signed i4 dimensions (arousal/valence/tension/…); 9× compression vs f32.
    pub qualia: [QualiaI4_16D; N],

    /// work
    /// Per-row packed meta word (`MetaWord`, 4 B/row).
    /// Migrated from `BindSpace.meta` (MetaColumn).
    /// Layout: `thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6)`.
    pub meta: [MetaWord; N],

    /// work
    /// Per-row OGIT entity-type index (`u16`, 2 B/row).
    /// Migrated from `BindSpace.entity_type`.
    /// 1-based index into the shared (immutable) ontology registry.
    /// The registry itself stays `Arc<OntologyRegistry>` (cold Zone-2, not owned here).
    pub entity_type: [u16; N],

    /// Monotonic cycle stamp; advanced by `tick()`.
    pub current_cycle: u32,

    /// 6-bit W-slot value this mailbox represents.
    /// Incoming batons with `edge.w_slot() != self.w_slot` are rejected.
    /// Must be < 64 (plan §6 L-6).
    pub w_slot: u8,

    /// Emission threshold (default 1.0; tunable).
    /// A row emits when `energy[row].abs() >= threshold`.
    pub threshold: f32,
}

/// Default capacity: 1024 rows (4× current BindSpace row count).
pub type DefaultMailboxSoA = MailboxSoA<1024>;

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
            // u32::MAX is the "never emitted" sentinel. current_cycle starts at 0
            // and advances via wrapping_add(1); in practice it never reaches
            // u32::MAX during a session, so the first emit on any cycle is always
            // permitted (u32::MAX != any valid cycle stamp).
            last_emission_cycle: [u32::MAX; N],
            current_cycle: 0,
            w_slot,
            threshold,
            // ── NEW thoughtspace columns — zero-initialised (D-MBX-A1) ──
            edges: [CausalEdge64::ZERO; N],
            qualia: [QualiaI4_16D::ZERO; N],
            meta: [MetaWord(0); N],
            entity_type: [0u16; N],
        }
    }

    /// Accept a batch of `(target_row, CausalEdge64)` batons.
    ///
    /// For each baton:
    /// - Rows out of `[0, N)` are silently dropped.
    /// - Batons whose `edge.w_slot()` differs from `self.w_slot` are silently
    ///   dropped (wrong corpus). Downstream telemetry can count rejections by
    ///   comparing the return value against `batons.len()`.
    /// - Accepted batons: `energy[row] += mantissa/8.0 * confidence`;
    ///   `plasticity_counter[row]` is incremented (saturating).
    ///
    /// Returns the count of accepted (non-dropped) batons.
    pub fn apply_edges(&mut self, batons: &[(u16, CausalEdge64)]) -> usize {
        let mut accepted = 0;
        for &(target, edge) in batons {
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
            self.plasticity_counter[row] =
                self.plasticity_counter[row].saturating_add(1);
            accepted += 1;
        }
        accepted
    }

    /// Scan all rows and emit batons for those whose `|energy|` meets or
    /// exceeds `threshold`, subject to the same-cycle idempotency guard.
    ///
    /// For each emitting row:
    /// - A baton is appended to `emission` with `target = row`.
    /// - The edge carries the clamped mantissa and the mailbox's `w_slot`.
    /// - `last_emission_cycle[row]` is stamped to `current_cycle`.
    /// - `energy[row]` is reset to `0.0` (single-shot per cycle).
    ///
    /// Rows that already emitted this cycle (`last_emission_cycle[row] == current_cycle`)
    /// are skipped regardless of current energy.
    pub fn emit(&mut self, source: MailboxId) -> CollapseGateEmission {
        let mut emission =
            CollapseGateEmission::new(source, self.current_cycle, MergeMode::Bundle);
        for row in 0..N {
            if self.energy[row].abs() < self.threshold {
                continue;
            }
            if self.last_emission_cycle[row] == self.current_cycle {
                continue; // already emitted this cycle
            }
            // Encode accumulated energy as a signed 4-bit mantissa (-8..+7).
            let mantissa =
                (self.energy[row].clamp(-1.0, 1.0) * 7.0).round() as i8;
            let edge = CausalEdge64::ZERO
                .with_inference_mantissa(mantissa)
                .with_w_slot(self.w_slot);
            emission.push_baton(row as u16, edge.0);
            self.last_emission_cycle[row] = self.current_cycle;
            self.energy[row] = 0.0; // reset on emission (single-shot per cycle)
        }
        emission
    }

    /// Advance to the next cycle.
    ///
    /// Wraps at `u32::MAX` (wrapping_add). This makes the
    /// same-cycle idempotency guard safe for long-running sessions.
    pub fn tick(&mut self) {
        self.current_cycle = self.current_cycle.wrapping_add(1);
    }

    /// Reset one row to its zero-initialised state.
    ///
    /// Clears `energy`, `plasticity_counter`, and `last_emission_cycle`
    /// for `row`. Out-of-range `row` values are silently ignored.
    /// Cross-cycle plasticity rollup is outside this scope (W6 §4.4 Zone 2).
    pub fn reset_row(&mut self, row: usize) {
        if row >= N {
            return;
        }
        self.energy[row] = 0.0;
        self.plasticity_counter[row] = 0;
        // Restore the "never emitted" sentinel so the row can emit immediately
        // on the next cycle without triggering the same-cycle guard.
        self.last_emission_cycle[row] = u32::MAX;
        // ── NEW thoughtspace columns reset (D-MBX-A1) ──
        self.edges[row] = CausalEdge64::ZERO;
        self.qualia[row] = QualiaI4_16D::ZERO;
        self.meta[row] = MetaWord(0);
        self.entity_type[row] = 0;
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

    /// work
    /// Return the `CausalEdge64` baton edge for `row`.
    ///
    /// Panics (debug) / wraps (release) on out-of-bounds; callers
    /// should stay within `[0, N)`.
    #[inline]
    pub fn edge(&self, row: usize) -> CausalEdge64 {
        self.edges[row]
    }

    /// work
    /// Set the `CausalEdge64` baton edge for `row`.
    ///
    /// Panics (debug) / wraps (release) on out-of-bounds; callers
    /// should stay within `[0, N)`.
    #[inline]
    pub fn set_edge(&mut self, row: usize, e: CausalEdge64) {
        self.edges[row] = e;
    }

    /// work
    /// Return the packed `QualiaI4_16D` affective vector for `row`.
    #[inline]
    pub fn qualia_at(&self, row: usize) -> QualiaI4_16D {
        self.qualia[row]
    }

    /// work
    /// Set the packed `QualiaI4_16D` affective vector for `row`.
    #[inline]
    pub fn set_qualia(&mut self, row: usize, q: QualiaI4_16D) {
        self.qualia[row] = q;
    }

    /// work
    /// Return the packed `MetaWord` for `row`.
    #[inline]
    pub fn meta_at(&self, row: usize) -> MetaWord {
        self.meta[row]
    }

    /// work
    /// Set the packed `MetaWord` for `row`.
    #[inline]
    pub fn set_meta(&mut self, row: usize, m: MetaWord) {
        self.meta[row] = m;
    }

    /// work
    /// Return the OGIT entity-type index for `row` (1-based, shared ontology).
    #[inline]
    pub fn entity_type_at(&self, row: usize) -> u16 {
        self.entity_type[row]
    }

    /// work
    /// Set the OGIT entity-type index for `row`.
    #[inline]
    pub fn set_entity_type(&mut self, row: usize, t: u16) {
        self.entity_type[row] = t;
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
    /// energy and plasticity_counter are zero; last_emission_cycle is u32::MAX
    /// (the "never emitted" sentinel so cycle 0 can emit without false guard).
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
                mb.last_emission_cycle[row],
                u32::MAX,
                "last_emission_cycle[{row}] should be u32::MAX (never-emitted sentinel)"
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
        assert_eq!(mb.plasticity_at(2), 0, "plasticity_counter[2] must be unchanged");
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

    // ── test 6: emit above threshold ────────────────────────────────────────

    /// A row above threshold must emit one baton; energy resets; cycle stamp set.
    #[test]
    fn test_mailbox_soa_emit_above_threshold() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(10, 1, 1.0);
        // Directly set energy[3] to 1.5 (above 1.0 threshold)
        mb.energy[3] = 1.5;

        let emission = mb.emit(10);

        assert_eq!(emission.baton_count(), 1, "should emit exactly 1 baton");
        let batons = emission.batons();
        assert_eq!(batons[0].0, 3u16, "baton target must be row 3");
        // mantissa = (1.5.clamp(-1,1) * 7.0).round() as i8 = (7.0).round() = 7
        let emitted_edge = CausalEdge64(batons[0].1);
        assert_ne!(
            emitted_edge.inference_mantissa(),
            0,
            "emitted edge mantissa must be non-zero"
        );
        assert_eq!(mb.energy_at(3), 0.0, "energy[3] must reset to 0 after emit");
        assert_eq!(
            mb.last_emission_cycle[3],
            mb.current_cycle,
            "last_emission_cycle[3] must equal current_cycle"
        );
    }

    // ── test 7: emit below threshold — no baton ──────────────────────────────

    /// Row energy below threshold must NOT produce a baton.
    #[test]
    fn test_mailbox_soa_emit_below_threshold_no_baton() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(5, 2, 1.0);
        mb.energy[3] = 0.5; // below 1.0 threshold

        let emission = mb.emit(5);

        assert_eq!(emission.baton_count(), 0, "below-threshold row must not emit");
        assert_eq!(mb.energy_at(3), 0.5, "energy[3] must be unchanged");
    }

    // ── test 8: double emit same cycle is idempotent ─────────────────────────

    /// Second emit() on the same cycle (no tick) must produce 0 batons for
    /// rows that already emitted.
    #[test]
    fn test_mailbox_soa_emit_double_call_same_cycle_idempotent() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
        mb.energy[1] = 2.0;

        let first = mb.emit(7);
        assert_eq!(first.baton_count(), 1, "first emit must produce 1 baton");

        // Re-accumulate energy so energy[1] is above threshold again —
        // but last_emission_cycle guard must block re-emission.
        mb.energy[1] = 2.0;
        let second = mb.emit(7);
        assert_eq!(
            second.baton_count(),
            0,
            "second emit same cycle must be blocked by last_emission_cycle guard"
        );
    }

    // ── test 9: tick then re-accumulate then emit ────────────────────────────

    /// After tick(), a row can re-emit in the new cycle.
    #[test]
    fn test_mailbox_soa_tick_then_emit_after_re_accumulation() {
        let mut mb: MailboxSoA<8> = MailboxSoA::new(3, 1, 1.0);
        mb.energy[4] = 1.5;

        // First emission
        let first = mb.emit(3);
        assert_eq!(first.baton_count(), 1);
        assert_eq!(mb.energy_at(4), 0.0);

        // Advance cycle
        mb.tick();

        // Re-accumulate enough energy
        let edge = make_edge(1, 7, 255); // w_slot=1 matches mb.w_slot=1
        let batons = vec![(4u16, edge)];
        mb.apply_edges(&batons);
        // energy[4] = (7/8.0) * (255/255.0) ≈ 0.875 — below 1.0 threshold.
        // Let's set it directly to be safe.
        mb.energy[4] = 1.2;

        // Second emission must succeed in the new cycle
        let second = mb.emit(3);
        assert_eq!(
            second.baton_count(),
            1,
            "after tick, row 4 must be able to emit again"
        );
    }
}
