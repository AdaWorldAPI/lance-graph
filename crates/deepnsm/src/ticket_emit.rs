//! META-AGENT: add `pub mod ticket_emit;` to lib.rs. Gate behind feature
//! `contract-ticket`. Also requires adding to Cargo.toml:
//!   lance-graph-contract = { path = "../lance-graph-contract", optional = true }
//!   [features] contract-ticket = ["dep:lance-graph-contract"]
//!
//! Emit FailureTicket from a partial DeepNSM parse for the LLM-tail router.
//!
//! See plan §D2 + grammar-tiered-routing.md "Combined failure ticket".
//!
//! Adapted to the actual `lance_graph_contract::grammar` surface:
//! - `PartialParse { resolved_tokens, unresolved_tokens, coverage }`
//! - `FailureTicket { partial_parse, attempted_inference, recommended_next,
//!    causal_ambiguity, tekamolo, wechsel, coverage, missing_required }`
//! - `TekamoloSlots { temporal, kausal, modal, lokal }` (no `has_unfillable`).
//!
//! `recommended_next` decision rules — the failure-mode IS the routing
//! signal:
//!
//! - `primes_found < 4`              → `Abduction`               (NSM-thin → LLM names primes)
//! - any TEKAMOLO slot unfillable    → `CounterfactualSynthesis` (slot must be hypothesised)
//! - `classification_distance > 0.7` → `Extrapolation`           (novel domain marker)
//! - else                            → `Revision`                (default refinement)

#![cfg(feature = "contract-ticket")]

use lance_graph_contract::grammar::{
    CausalAmbiguity, FailureTicket, NarsInference, PartialParse, TekamoloSlots,
    WechselAmbiguity,
};

/// Threshold above which `classification_distance` flags a novel domain.
pub const NOVEL_DOMAIN_THRESHOLD: f32 = 0.7;

/// Minimum NSM primes required before a parse is considered semantically
/// thick enough to NOT need LLM abduction.
pub const PRIMES_NEEDED: u8 = 4;

/// Decompose a parse coverage failure into the SPO × 2³ × TEKAMOLO ×
/// Wechsel fields the LLM router needs.
///
/// The caller checks `coverage_score >= LOCAL_COVERAGE_THRESHOLD` first —
/// if so, no ticket is needed. Once we are here, we already know the
/// parse failed coverage; we only need to choose the routing inference
/// and stash the partial fields.
///
/// `attempted_inference` is the NARS inference the local pipeline was
/// running when coverage tripped. The router uses it to avoid re-issuing
/// the same mode and to seed `recommended_next` with directional context.
#[allow(clippy::too_many_arguments)]
pub fn emit_ticket(
    partial: PartialParse,
    coverage_score: f32,
    classification_distance: f32,
    primes_found: u8,
    tekamolo: TekamoloSlots,
    wechsel: Vec<WechselAmbiguity>,
    causal_ambiguity: Option<CausalAmbiguity>,
    attempted_inference: NarsInference,
) -> FailureTicket {
    let recommended = if primes_found < PRIMES_NEEDED {
        NarsInference::Abduction
    } else if has_unfillable(&tekamolo) {
        NarsInference::CounterfactualSynthesis
    } else if classification_distance > NOVEL_DOMAIN_THRESHOLD {
        NarsInference::Extrapolation
    } else {
        NarsInference::Revision
    };

    FailureTicket {
        partial_parse: partial,
        attempted_inference,
        recommended_next: recommended,
        causal_ambiguity,
        tekamolo,
        wechsel,
        coverage: coverage_score,
        missing_required: Vec::new(),
    }
}

/// Backward-compat shim — assumes the local pipeline attempted Deduction.
/// Prefer `emit_ticket` with an explicit `attempted_inference`; this exists
/// so existing call sites compile while they are migrated.
#[deprecated(note = "use emit_ticket with explicit attempted_inference")]
pub fn emit_ticket_legacy(
    partial: PartialParse,
    coverage_score: f32,
    classification_distance: f32,
    primes_found: u8,
    tekamolo: TekamoloSlots,
    wechsel: Vec<WechselAmbiguity>,
    causal_ambiguity: Option<CausalAmbiguity>,
) -> FailureTicket {
    emit_ticket(
        partial,
        coverage_score,
        classification_distance,
        primes_found,
        tekamolo,
        wechsel,
        causal_ambiguity,
        NarsInference::Deduction,
    )
}

/// Returns true if ANY of the four TEKAMOLO slots failed to fill.
///
/// Replaces the previous all-None heuristic. The LLM router needs to know
/// when even one slot is missing because partial-TEKAMOLO is exactly the
/// case CounterfactualSynthesis was designed for: hypothesise the missing
/// slot from the surrounding ones rather than re-derive the whole frame.
fn has_unfillable(slots: &TekamoloSlots) -> bool {
    slots.temporal.is_none()
        || slots.kausal.is_none()
        || slots.modal.is_none()
        || slots.lokal.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::grammar::wechsel::WechselRole;

    fn empty_partial() -> PartialParse {
        PartialParse {
            resolved_tokens: vec![1, 2],
            unresolved_tokens: vec![3, 4],
            coverage: 0.5,
        }
    }

    fn filled_tekamolo() -> TekamoloSlots {
        TekamoloSlots {
            temporal: Some((0, 1)),
            kausal: Some((2, 3)),
            modal: Some((4, 5)),
            lokal: Some((6, 7)),
        }
    }

    #[test]
    fn low_primes_routes_to_abduction() {
        let t = emit_ticket(
            empty_partial(),
            0.6,
            0.1,
            2,
            filled_tekamolo(),
            Vec::new(),
            None,
            NarsInference::Deduction,
        );
        assert_eq!(t.recommended_next, NarsInference::Abduction);
        assert_eq!(t.coverage, 0.6);
        assert_eq!(t.attempted_inference, NarsInference::Deduction);
    }

    #[test]
    fn unfillable_tekamolo_routes_to_counterfactual_synthesis() {
        let t = emit_ticket(
            empty_partial(),
            0.6,
            0.1,
            5,
            TekamoloSlots::default(),
            Vec::new(),
            None,
            NarsInference::Deduction,
        );
        assert_eq!(t.recommended_next, NarsInference::CounterfactualSynthesis);
    }

    #[test]
    fn high_classification_distance_routes_to_extrapolation() {
        let t = emit_ticket(
            empty_partial(),
            0.7,
            0.85,
            5,
            filled_tekamolo(),
            Vec::new(),
            None,
            NarsInference::Deduction,
        );
        assert_eq!(t.recommended_next, NarsInference::Extrapolation);
    }

    #[test]
    fn default_path_is_revision() {
        let t = emit_ticket(
            empty_partial(),
            0.7,
            0.1,
            5,
            filled_tekamolo(),
            Vec::new(),
            None,
            NarsInference::Deduction,
        );
        assert_eq!(t.recommended_next, NarsInference::Revision);
    }

    #[test]
    fn wechsel_payload_passes_through() {
        let amb = WechselAmbiguity {
            token_index: 3,
            candidates: vec![WechselRole::PrepTemporal, WechselRole::PrepSpatial],
            local_ambiguity: 0.85,
        };
        let t = emit_ticket(
            empty_partial(),
            0.6,
            0.1,
            2,
            filled_tekamolo(),
            vec![amb],
            None,
            NarsInference::Deduction,
        );
        assert_eq!(t.wechsel.len(), 1);
        assert_eq!(t.wechsel[0].token_index, 3);
    }

    #[test]
    fn three_of_four_slots_filled_still_routes_to_counterfactual() {
        // Per-slot has_unfillable: even ONE missing slot must escalate to
        // CounterfactualSynthesis, not Revision (the previous heuristic
        // routed this case to Revision because `is_empty()` was false).
        let partially_filled = TekamoloSlots {
            temporal: Some((0, 1)),
            kausal: Some((2, 3)),
            modal: Some((4, 5)),
            lokal: None, // ← the unfillable slot
        };
        let t = emit_ticket(
            empty_partial(),
            0.7,
            0.1,
            5,
            partially_filled,
            Vec::new(),
            None,
            NarsInference::Deduction,
        );
        assert_eq!(t.recommended_next, NarsInference::CounterfactualSynthesis);
    }

    #[test]
    fn attempted_inference_threads_through() {
        let t = emit_ticket(
            empty_partial(),
            0.7,
            0.1,
            5,
            filled_tekamolo(),
            Vec::new(),
            None,
            NarsInference::Abduction,
        );
        assert_eq!(t.attempted_inference, NarsInference::Abduction);
    }
}
