//! Escalation+epiphany boot checklist — planner wiring (D-PERSONA-1).
//!
//! The escalation loop machinery (collapse-hint, [`InnerCouncil`],
//! [`EpiphanyDetector`], [`WisdomMarker`], [`Checklist`]) is the zero-dep
//! contract surface in `lance_graph_contract::escalation`. This module is the
//! planner-side wiring per `rung-persona-orchestration-v1` §2: the concrete
//! HARD/SOFT boot items and the adapter that turns a planner [`MulAssessment`]
//! into a [`CouncilVerdict`] (the `felt_parse` live-signal counterpart).

pub use lance_graph_contract::escalation::{
    fanout_width, is_split, noise_tolerance, rung_delta, Archetype, Checklist, ChecklistItem,
    CollapseHint, CouncilVerdict, Epiphany, EpiphanyDetector, GateKind, GhostEcho, InnerCouncil,
    WisdomMarker,
};

use super::MulAssessment;

/// Derive a [`CouncilVerdict`] from a planner MUL assessment — the per-turn
/// live signal that refines the inherited prior. Maps the four scalar
/// observables the council consumes from the assessment's own accessors.
pub fn verdict_from(assessment: &MulAssessment) -> CouncilVerdict {
    InnerCouncil::from_signals(
        assessment.trust.composite_score() as f32,
        assessment.dk_position.humility_factor() as f32,
        assessment.homeostasis.flow_factor() as f32,
        assessment.homeostasis.allostatic_load as f32,
    )
}

/// The boot checklist (§2): HARD items gate boot; SOFT items degrade
/// gracefully (the runtime routes around them — anytime / etiquette).
///
/// Each item is verified by the escalation+epiphany loop — not a bespoke
/// assert — and these items double as the continuous supervision health-checks
/// (a green item going red at runtime is a let-it-crash → [`Checklist::mark_red`]).
pub fn boot_checklist() -> Checklist {
    Checklist::new(vec![
        // ── HARD (boot gate) ──────────────────────────────────────────────
        ChecklistItem::new("contract_types_load", GateKind::Hard), // RungState=16B, SpoHead, MulAssessment Pod sizes
        ChecklistItem::new("soa_floor_up", GateKind::Hard),        // SoaColumns, i4-32 unpack
        ChecklistItem::new("operational_store", GateKind::Hard),   // Lance / SQLite (not surreal)
        ChecklistItem::new("nars_tables_loaded", GateKind::Hard),  // NarsTables lookup hot
        ChecklistItem::new("thresholds_loaded", GateKind::Hard), // MUL profile, SD_FLOW/BLOCK, rung thresholds
        ChecklistItem::new("free_energy_wired", GateKind::Hard), // FreeEnergy::compose available
        // ── SOFT (degrade if red) ─────────────────────────────────────────
        ChecklistItem::new("capabilities_registered", GateKind::Soft), // ExpertCapability / actor / MCP
        ChecklistItem::new("wisdom_marker_store", GateKind::Soft),     // cold start → foot of curve
        ChecklistItem::new("macro_eval_harness", GateKind::Soft), // run without offline updates if absent
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul::{assess, SituationInput};

    #[test]
    fn boot_checklist_has_six_hard_three_soft() {
        let cl = boot_checklist();
        assert_eq!(cl.items.len(), 9);
        assert_eq!(
            cl.items.iter().filter(|i| i.gate == GateKind::Hard).count(),
            6
        );
        assert_eq!(
            cl.items.iter().filter(|i| i.gate == GateKind::Soft).count(),
            3
        );
        // Fresh checklist: nothing green, not boot-ready.
        assert!(!cl.boot_ready());
    }

    #[test]
    fn high_trust_assessment_settles_to_flow() {
        let input = SituationInput {
            felt_competence: 0.8,
            demonstrated_competence: 0.85,
            source_reliability: 0.9,
            environment_stability: 0.9,
            calibration_accuracy: 0.85,
            challenge_level: 0.6,
            skill_level: 0.65,
            allostatic_load: 0.05,
            complexity_ratio: 0.7,
            ..Default::default()
        };
        let verdict = verdict_from(&assess(&input));
        assert_eq!(verdict.hint, CollapseHint::Flow);
    }

    #[test]
    fn boot_gate_clears_when_all_hard_green() {
        let mut cl = boot_checklist();
        let flow = CouncilVerdict {
            hint: CollapseHint::Flow,
            confidence: 1.0,
            split: false,
        };
        let e = Epiphany {
            similarity: 0.6,
            baseline: 0.3,
            samples: 5,
        };
        for item in [
            "contract_types_load",
            "soa_floor_up",
            "operational_store",
            "nars_tables_loaded",
            "thresholds_loaded",
            "free_energy_wired",
        ] {
            cl.step(item, &flow, Some(e));
        }
        // All HARD green → boot-ready, even though SOFT items are still red.
        assert!(cl.boot_ready());
        assert!(cl.degraded());
        assert!(!cl.all_flow());
    }
}
