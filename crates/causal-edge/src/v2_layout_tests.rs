//! V2 layout accessor round-trip tests for CausalEdge64.
//!
//! Category 1 tests from pr-ce64-mb-2-causaledge64-v2.md §9.
//! All tests gated on `#[cfg(feature = "causal-edge-v2-layout")]`.
//!
//! Tests verify:
//! - W slot (6-bit, bits 53-58) round-trips
//! - Truth-band lens (2-bit, bits 59-60) round-trips
//! - Signed inference mantissa (4-bit i4, bits 46-49) round-trips with correct sign-extension
//! - with_routing (W + truth combined set) round-trips
//! - V2 field writes do not disturb V1 fields
//! - Zero edge has correct v2 defaults
//! - Bit-boundary isolation (W max does not contaminate truth, truth max does not contaminate W)
//! - Spare isolation (bits 61-63 do not disturb W or truth)
//! - Mantissa does not contaminate plasticity (bits 50-52 are separate from bits 46-49)
//! - struct size unchanged at 8 bytes

#[cfg(test)]
#[cfg(feature = "causal-edge-v2-layout")]
mod v2_layout_tests {
    use crate::edge::{CausalEdge64, InferenceType};
    use crate::layout::TrustTexture;
    use crate::pearl::CausalMask;
    use crate::plasticity::PlasticityState;

    // ── test_w_slot_roundtrip ───────────────────────────────────────────────

    #[test]
    fn test_w_slot_roundtrip() {
        for w in [0u8, 1, 31, 63] {
            let edge = CausalEdge64::ZERO.with_w_slot(w);
            assert_eq!(
                edge.w_slot(), w,
                "w_slot round-trip failed for w={w}"
            );
        }
    }

    // ── test_truth_roundtrip ────────────────────────────────────────────────

    #[test]
    fn test_truth_roundtrip() {
        for t in [
            TrustTexture::Crystalline,
            TrustTexture::Solid,
            TrustTexture::Fuzzy,
            TrustTexture::Murky,
        ] {
            let edge = CausalEdge64::ZERO.with_truth(t);
            assert_eq!(
                edge.truth(), t,
                "truth round-trip failed for {t:?}"
            );
        }
    }

    // ── test_inference_mantissa_signed_roundtrip ────────────────────────────

    #[test]
    fn test_inference_mantissa_signed_roundtrip() {
        // Full i4 range coverage: min, min+1, -1, 0, 1, max
        for m in [-8i8, -7, -1, 0, 1, 7] {
            let edge = CausalEdge64::ZERO.with_inference_mantissa(m);
            assert_eq!(
                edge.inference_mantissa(), m,
                "inference_mantissa signed round-trip failed for m={m}"
            );
        }
    }

    // ── test_with_routing_roundtrip ─────────────────────────────────────────

    #[test]
    fn test_with_routing_roundtrip() {
        // v2 signature: with_routing(w: u8, t: TrustTexture) — no g parameter (L-3)
        let edge = CausalEdge64::ZERO.with_routing(42, TrustTexture::Fuzzy);
        assert_eq!(edge.w_slot(), 42, "with_routing: w_slot mismatch");
        assert_eq!(edge.truth(), TrustTexture::Fuzzy, "with_routing: truth mismatch");
    }

    // ── test_v2_fields_do_not_disturb_v1_fields ────────────────────────────

    #[test]
    fn test_v2_fields_do_not_disturb_v1_fields() {
        // Build a v1-style edge using the existing pack() (back-compat path).
        #[allow(deprecated)]
        let base = CausalEdge64::pack(
            143, 7, 201,            // S, P, O palette indices
            209, 181,               // NARS f=0.82, c=0.71
            CausalMask::PO,         // interventional level
            0b101,                  // direction triad
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            0,                      // temporal = 0 (v1 compat; bits 52-63 must be 0 for v2 clean read)
        );

        // Apply v2 routing and mantissa
        let v2 = base
            .with_routing(10, TrustTexture::Solid)
            .with_inference_mantissa(-3);

        // All v1 fields must be unchanged
        assert_eq!(v2.s_idx(), 143, "s_idx disturbed");
        assert_eq!(v2.p_idx(), 7, "p_idx disturbed");
        assert_eq!(v2.o_idx(), 201, "o_idx disturbed");
        assert_eq!(v2.frequency_u8(), 209, "frequency disturbed");
        assert_eq!(v2.confidence_u8(), 181, "confidence disturbed");
        assert_eq!(v2.causal_mask(), CausalMask::PO, "causal_mask disturbed");
        assert_eq!(v2.direction(), 0b101, "direction disturbed");

        // V2 fields must be what we set
        assert_eq!(v2.w_slot(), 10, "w_slot not set");
        assert_eq!(v2.truth(), TrustTexture::Solid, "truth not set");
        assert_eq!(v2.inference_mantissa(), -3, "inference_mantissa not set");
    }

    // ── test_zero_edge_v2_defaults ──────────────────────────────────────────

    #[test]
    fn test_zero_edge_v2_defaults() {
        let e = CausalEdge64::ZERO;
        assert_eq!(e.w_slot(), 0, "ZERO: w_slot must be 0");
        assert_eq!(e.truth(), TrustTexture::Crystalline, "ZERO: truth must be Crystalline");
        assert_eq!(e.inference_mantissa(), 0, "ZERO: mantissa must be 0");
        assert_eq!(e.spare(), 0, "ZERO: spare must be 0");
    }

    // ── test_w_slot_max_no_truth_contamination ──────────────────────────────

    #[test]
    fn test_w_slot_max_no_truth_contamination() {
        // W-slot max = 63 = 0b111111. Bits 53-58.
        // Truth-band = bits 59-60. Must be untouched.
        let e = CausalEdge64::ZERO.with_w_slot(63);
        assert_eq!(e.w_slot(), 63, "w_slot max round-trip failed");
        assert_eq!(
            e.truth(), TrustTexture::Crystalline,
            "w_slot=63 must not contaminate truth-band (bits 59-60)"
        );
    }

    // ── test_truth_max_no_w_contamination ───────────────────────────────────

    #[test]
    fn test_truth_max_no_w_contamination() {
        // Truth max = Murky = 0b11. Bits 59-60.
        // W-slot = bits 53-58. Must be untouched.
        let e = CausalEdge64::ZERO.with_truth(TrustTexture::Murky);
        assert_eq!(e.truth_raw(), 3, "truth_raw Murky must be 3");
        assert_eq!(
            e.w_slot(), 0,
            "truth=Murky must not contaminate W-slot (bits 53-58)"
        );
    }

    // ── test_spare_isolation ─────────────────────────────────────────────────

    #[test]
    fn test_spare_isolation() {
        // Spare = 0b111 (all 3 bits set). Bits 61-63.
        // W-slot and truth must remain 0.
        let e = CausalEdge64::ZERO.with_spare(0b111);
        assert_eq!(e.spare(), 0b111, "spare round-trip failed");
        assert_eq!(e.w_slot(), 0, "spare must not disturb W-slot");
        assert_eq!(
            e.truth(), TrustTexture::Crystalline,
            "spare must not disturb truth-band"
        );
    }

    // ── test_mantissa_no_plasticity_contamination ────────────────────────────

    #[test]
    fn test_mantissa_no_plasticity_contamination() {
        // Mantissa = -1 → bits 46-49 = 0b1111 (all 4 mantissa bits set).
        // Plasticity is bits 50-52 (shifted by +1 from v1 per L-4).
        // Bits 50-52 must be untouched (i.e., plasticity = ALL_FROZEN = 0).
        let e = CausalEdge64::ZERO.with_inference_mantissa(-1);
        assert_eq!(
            e.inference_mantissa(), -1,
            "mantissa -1 round-trip failed"
        );
        assert_eq!(
            e.plasticity(), PlasticityState::ALL_FROZEN,
            "mantissa=-1 (bits 46-49 all set) must not contaminate plasticity (bits 50-52)"
        );
    }

    // ── test_size_unchanged ──────────────────────────────────────────────────

    #[test]
    fn test_size_unchanged() {
        assert_eq!(
            std::mem::size_of::<CausalEdge64>(), 8,
            "CausalEdge64 must be exactly 8 bytes (one register)"
        );
        assert_eq!(
            8 * std::mem::size_of::<CausalEdge64>(), 64,
            "8 × CausalEdge64 must equal one cache line (64 bytes)"
        );
    }

    // ── test_const_assert_mask_coverage ─────────────────────────────────────
    // This is a compile-time assertion in layout.rs::_LAYOUT_COVERAGE.
    // If it compiles, the layout covers all 64 bits exactly once.
    // The test below just documents the intent:
    #[test]
    fn test_const_assert_mask_coverage_compiles() {
        // If the crate compiles with this feature enabled, the const assert passed.
        // layout::_LAYOUT_COVERAGE is evaluated at compile time.
        let _ = crate::layout::SPARE_MASK; // touch layout module to ensure it's linked
    }

    // ── Bonus: mantissa set/get for all 16 i4 values ────────────────────────

    #[test]
    fn test_mantissa_all_i4_values() {
        for m in -8i8..=7 {
            let e = CausalEdge64::ZERO.with_inference_mantissa(m);
            assert_eq!(
                e.inference_mantissa(), m,
                "inference_mantissa round-trip failed for m={m}"
            );
        }
    }

    // ── Bonus: with_routing idempotent on second call ────────────────────────

    #[test]
    fn test_with_routing_override() {
        let e = CausalEdge64::ZERO
            .with_routing(10, TrustTexture::Fuzzy)
            .with_routing(20, TrustTexture::Murky);
        assert_eq!(e.w_slot(), 20, "second with_routing should override W");
        assert_eq!(e.truth(), TrustTexture::Murky, "second with_routing should override truth");
    }

    // ── Bonus: InferenceType to_mantissa / from_mantissa round-trip ─────────

    #[test]
    fn test_intervention_counterfactual_mantissa_slots() {
        // PR-LL-1 absorbed at slots 6 and -6 per L-9
        assert_eq!(
            InferenceType::Intervention.to_mantissa(), 6,
            "Intervention must map to mantissa +6"
        );
        assert_eq!(
            InferenceType::Counterfactual.to_mantissa(), -6,
            "Counterfactual must map to mantissa -6"
        );
        // from_mantissa round-trip for PR-LL-1 slots
        assert_eq!(
            InferenceType::from_mantissa(6), InferenceType::Intervention,
            "from_mantissa(+6) must return Intervention"
        );
        assert_eq!(
            InferenceType::from_mantissa(-6), InferenceType::Counterfactual,
            "from_mantissa(-6) must return Counterfactual"
        );
    }

    // ── Bonus: pack_v2 defaults ──────────────────────────────────────────────

    #[test]
    fn test_pack_v2_v2_field_defaults() {
        let e = CausalEdge64::pack_v2(
            1, 2, 3,
            200, 200,
            CausalMask::None,
            0,
            PlasticityState::ALL_FROZEN,
        );
        assert_eq!(e.w_slot(), 0, "pack_v2: w_slot defaults to 0");
        assert_eq!(e.truth(), TrustTexture::Crystalline, "pack_v2: truth defaults to Crystalline");
        assert_eq!(e.inference_mantissa(), 0, "pack_v2: mantissa defaults to 0");
        assert_eq!(e.spare(), 0, "pack_v2: spare defaults to 0");
        // v1 fields must be set correctly
        assert_eq!(e.s_idx(), 1);
        assert_eq!(e.p_idx(), 2);
        assert_eq!(e.o_idx(), 3);
    }
}
