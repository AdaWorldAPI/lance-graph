//! Message-free kanban **visibility + pure helpers** — what remains of the
//! retired actor surface.
//!
//! ## Tombstone (2026-08-05, operator-directed —
//! `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`)
//!
//! This module used to carry `KanbanActor` + `KanbanMsg::{Advance, MulAdvance,
//! Tick}` and the RPC drivers (`deliver_kanban_step` / `drive_mul_advance` /
//! `drive_version_tick` / `drive_scheduled_tick` / `run_to_absorbing`) — the
//! acknowledgement/pump/scheduler theater. All of it is DELETED, not
//! deprecated: a version tick was being read as *permission to advance*, and
//! an ack-shaped RPC was being read as *what makes the substrate progress*.
//! The substrate progresses because immutable versions exist:
//!
//! ```text
//! think → seal → publish Lance version → next cycle reads the published version
//! ```
//!
//! Nothing signals, acknowledges, or schedules that. The compile-time
//! exclusive-ownership guarantee the actor used to dramatize is native Rust:
//! `&mut MailboxSoaOwner` IS the serialization — a second mutator is a compile
//! error, no mailbox message required. Phase progression is enacted only by
//! the #879 sealed-cycle path (`cycle_driver`). Ack/SLA/retry vocabulary
//! survives ONLY in external consumer membranes (e.g. OGAR's arago-parity
//! ActionHandler runtime, `submitAction → Receipt::Acknowledged` — an
//! application wire protocol, never substrate mechanics).
//!
//! ## What lives here now
//!
//! - [`PhaseCensus`] — the read-only fleet visibility surface. Readers observe
//!   owners; a census is one `&self` pass, not 64k RPCs.
//! - [`mul_target`] — the pure MUL-gate lowering (`gate_decision_i4` → next
//!   DAG-legal phase), consumed by `cycle_driver`'s P4c gate.
//! - [`parse_kanban_step`] — the pure `kanban.<mailbox>.<phase>` step-string
//!   parser (the `UnifiedStep { step_type: "kanban.*" }` vocabulary,
//!   `lance_graph_contract::kanban` module doc).

use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::mul::i4_eval::gate_decision_i4;
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_contract::QualiaI4_16D;

// ─── The visibility surface: read-only fleet phase census ─────────────────────

/// A message-free, read-only census of Rubicon phases across a set of owners —
/// the visibility surface that replaced the actor RPC theater.
///
/// Observation is a plain borrow: [`PhaseCensus::observe`] walks any iterator
/// of [`MailboxSoaView`]s once and counts phases; nothing is mutated, nothing
/// is messaged, nothing is scheduled. For loops that already iterate owners,
/// [`PhaseCensus::record`] accumulates incrementally.
///
/// "Absorbing" is derived from the lifecycle DAG itself
/// (`KanbanColumn::next_phases()` empty), never hardcoded — if the Rubicon DAG
/// changes, the census follows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhaseCensus {
    counts: [usize; 6],
}

/// Stable census index per column. Exhaustive on purpose: a new
/// `KanbanColumn` variant is a compile error here, not a silently
/// uncounted phase.
fn census_idx(col: KanbanColumn) -> usize {
    match col {
        KanbanColumn::Planning => 0,
        KanbanColumn::CognitiveWork => 1,
        KanbanColumn::Evaluation => 2,
        KanbanColumn::Commit => 3,
        KanbanColumn::Plan => 4,
        KanbanColumn::Prune => 5,
    }
}

/// The column at a census index — inverse of `census_idx`.
fn census_col(idx: usize) -> KanbanColumn {
    match idx {
        0 => KanbanColumn::Planning,
        1 => KanbanColumn::CognitiveWork,
        2 => KanbanColumn::Evaluation,
        3 => KanbanColumn::Commit,
        4 => KanbanColumn::Plan,
        _ => KanbanColumn::Prune,
    }
}

impl PhaseCensus {
    /// Census an iterator of views in one read-only pass.
    #[must_use]
    pub fn observe<'a, V, I>(views: I) -> Self
    where
        V: MailboxSoaView + 'a,
        I: IntoIterator<Item = &'a V>,
    {
        let mut census = Self::default();
        for v in views {
            census.record(v.phase());
        }
        census
    }

    /// Record one observed phase (for callers already iterating owners).
    pub fn record(&mut self, phase: KanbanColumn) {
        self.counts[census_idx(phase)] += 1;
    }

    /// Owners observed in `col`.
    #[must_use]
    pub fn count(&self, col: KanbanColumn) -> usize {
        self.counts[census_idx(col)]
    }

    /// Total owners observed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Owners observed in an absorbing column (one with no forward arc —
    /// `next_phases()` empty, i.e. `Commit`/`Prune` under the current DAG).
    #[must_use]
    pub fn absorbing(&self) -> usize {
        self.counts
            .iter()
            .enumerate()
            .filter(|(i, _)| census_col(*i).next_phases().is_empty())
            .map(|(_, n)| n)
            .sum()
    }

    /// The fleet is at rest: at least one owner was observed and EVERY
    /// observed owner sits in an absorbing column. An empty census is NOT at
    /// rest — rest is an observation about owners, and observing nothing
    /// asserts nothing (the vacuous-truth reading would make an empty fleet
    /// indistinguishable from a finished one).
    #[must_use]
    pub fn at_rest(&self) -> bool {
        let total = self.total();
        total > 0 && self.absorbing() == total
    }
}

// ─── Pure MUL-gate lowering (consumed by cycle_driver's P4c gate) ─────────────

/// The MUL-gated target phase for `phase` given a node's `qualia` + inference
/// `mantissa`: run the i4 gate ([`gate_decision_i4`]) and lower it to the
/// DAG-legal next phase via [`KanbanColumn::advance_on_gate`] (Flow → forward,
/// Block → Prune-where-legal, Hold → `None`). Pure + integer-only (no f64/NaN).
pub fn mul_target(
    phase: KanbanColumn,
    qualia: &QualiaI4_16D,
    mantissa: i8,
) -> Option<KanbanColumn> {
    let gate = gate_decision_i4(qualia, mantissa);
    phase.advance_on_gate(&gate)
}

// ─── Pure step-string vocabulary parser ───────────────────────────────────────

/// Parse a `kanban.<mailbox>.<phase>` step type into `(mailbox, target_phase)`,
/// where `<phase>` is the snake_case [`KanbanColumn`] name (e.g. `cognitive_work`).
/// Returns `None` for anything that isn't a well-formed kanban step.
pub fn parse_kanban_step(step_type: &str) -> Option<(&str, KanbanColumn)> {
    let mut it = step_type.splitn(3, '.');
    match (it.next(), it.next(), it.next()) {
        (Some("kanban"), Some(mailbox), Some(phase)) if !mailbox.is_empty() => {
            phase_from_name(phase).map(|p| (mailbox, p))
        }
        _ => None,
    }
}

/// Snake_case phase name → [`KanbanColumn`] (the `kanban.*` routing vocabulary;
/// inverse of the canonical column names).
fn phase_from_name(name: &str) -> Option<KanbanColumn> {
    Some(match name {
        "planning" => KanbanColumn::Planning,
        "cognitive_work" => KanbanColumn::CognitiveWork,
        "evaluation" => KanbanColumn::Evaluation,
        "commit" => KanbanColumn::Commit,
        "plan" => KanbanColumn::Plan,
        "prune" => KanbanColumn::Prune,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId;

    /// Minimal read-only view for census tests — phases only, no owner
    /// surface needed (the census never mutates).
    struct ViewBoard {
        id: MailboxId,
        phase: KanbanColumn,
    }

    impl ViewBoard {
        fn new(id: MailboxId, phase: KanbanColumn) -> Self {
            Self { id, phase }
        }
    }

    impl MailboxSoaView for ViewBoard {
        fn mailbox_id(&self) -> MailboxId {
            self.id
        }
        fn n_rows(&self) -> usize {
            0
        }
        fn w_slot(&self) -> u8 {
            (self.id & 0x3F) as u8
        }
        fn current_cycle(&self) -> u32 {
            0
        }
        fn phase(&self) -> KanbanColumn {
            self.phase
        }
        fn energy(&self) -> &[f32] {
            &[]
        }
        fn edges_raw(&self) -> &[u64] {
            &[]
        }
        fn meta_raw(&self) -> &[u32] {
            &[]
        }
        fn entity_type(&self) -> &[u16] {
            &[]
        }
    }

    #[test]
    fn census_counts_a_mixed_fleet_and_is_not_at_rest() {
        // Can-fire half: a NON-TRIVIAL mixed fleet must be counted exactly and
        // must not read as at-rest while any owner is mid-arc.
        let fleet = [
            ViewBoard::new(1, KanbanColumn::Planning),
            ViewBoard::new(2, KanbanColumn::Planning),
            ViewBoard::new(3, KanbanColumn::CognitiveWork),
            ViewBoard::new(4, KanbanColumn::Commit),
        ];
        let census = PhaseCensus::observe(fleet.iter());
        assert_eq!(census.total(), 4);
        assert_eq!(census.count(KanbanColumn::Planning), 2);
        assert_eq!(census.count(KanbanColumn::CognitiveWork), 1);
        assert_eq!(census.count(KanbanColumn::Commit), 1);
        assert_eq!(census.count(KanbanColumn::Prune), 0);
        assert_eq!(census.absorbing(), 1, "only the Commit owner is absorbing");
        assert!(
            !census.at_rest(),
            "three owners are mid-arc — at_rest firing here would make the \
             signal as uninformative as one that never fires"
        );
    }

    #[test]
    fn census_at_rest_stays_silent_on_a_fully_absorbed_fleet() {
        // Can-stay-silent half, on a NON-TRIVIAL input: a fleet fully in
        // absorbing columns (both of them) reads at rest.
        let fleet = [
            ViewBoard::new(1, KanbanColumn::Commit),
            ViewBoard::new(2, KanbanColumn::Commit),
            ViewBoard::new(3, KanbanColumn::Prune),
        ];
        let census = PhaseCensus::observe(fleet.iter());
        assert_eq!(census.total(), 3);
        assert_eq!(census.absorbing(), 3);
        assert!(census.at_rest());
    }

    #[test]
    fn census_of_nothing_asserts_nothing() {
        // Emptiness handling (documented semantics), separate from the
        // discrimination halves above: observing zero owners is NOT rest.
        let census = PhaseCensus::observe(std::iter::empty::<&ViewBoard>());
        assert_eq!(census.total(), 0);
        assert!(!census.at_rest());
    }

    #[test]
    fn census_record_accumulates_like_observe() {
        let fleet = [
            ViewBoard::new(1, KanbanColumn::Evaluation),
            ViewBoard::new(2, KanbanColumn::Plan),
        ];
        let observed = PhaseCensus::observe(fleet.iter());
        let mut recorded = PhaseCensus::default();
        for v in &fleet {
            recorded.record(v.phase);
        }
        assert_eq!(observed, recorded);
    }

    #[test]
    fn mul_target_flow_advances_and_hold_holds() {
        // Flow qualia (warmth/groundedness high, low tension, calibrated) +
        // mantissa>0 → gate Flow → forward Planning → CognitiveWork. Same
        // construction cycle_driver's flow_qualia() helper uses.
        let flow_q = QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2);
        assert_eq!(
            mul_target(KanbanColumn::Planning, &flow_q, 4),
            Some(KanbanColumn::CognitiveWork)
        );
        // Neutral qualia + mantissa 0 → gate Hold → None.
        assert_eq!(
            mul_target(KanbanColumn::Planning, &QualiaI4_16D(0), 0),
            None
        );
    }

    #[test]
    fn parse_kanban_step_shapes() {
        assert_eq!(
            parse_kanban_step("kanban.mb42.cognitive_work"),
            Some(("mb42", KanbanColumn::CognitiveWork))
        );
        assert_eq!(parse_kanban_step("lg.foo"), None); // wrong domain
        assert_eq!(parse_kanban_step("kanban.mb42"), None); // no phase
        assert_eq!(parse_kanban_step("kanban..commit"), None); // empty mailbox
        assert_eq!(parse_kanban_step("kanban.mb42.bogus"), None); // unknown phase
    }
}
