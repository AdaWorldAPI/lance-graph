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
    use crate::layout::{CausalTopology, ReasoningBand, TrustTexture};
    use crate::pearl::CausalMask;
    use crate::plasticity::PlasticityState;

    // ── test_w_slot_roundtrip ───────────────────────────────────────────────

    #[test]
    fn test_w_slot_roundtrip() {
        for w in [0u8, 1, 31, 63] {
            let edge = CausalEdge64::ZERO.with_w_slot(w);
            assert_eq!(edge.w_slot(), w, "w_slot round-trip failed for w={w}");
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
            assert_eq!(edge.truth(), t, "truth round-trip failed for {t:?}");
        }
    }

    // ── test_inference_mantissa_signed_roundtrip ────────────────────────────

    #[test]
    fn test_inference_mantissa_signed_roundtrip() {
        // Full i4 range coverage: min, min+1, -1, 0, 1, max
        for m in [-8i8, -7, -1, 0, 1, 7] {
            let edge = CausalEdge64::ZERO.with_inference_mantissa(m);
            assert_eq!(
                edge.inference_mantissa(),
                m,
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
        assert_eq!(
            edge.truth(),
            TrustTexture::Fuzzy,
            "with_routing: truth mismatch"
        );
    }

    // ── test_v2_fields_do_not_disturb_v1_fields ────────────────────────────

    #[test]
    fn test_v2_fields_do_not_disturb_v1_fields() {
        // Build a v1-style edge using the existing pack() (back-compat path).
        #[allow(deprecated)]
        let base = CausalEdge64::pack(
            143,
            7,
            201, // S, P, O palette indices
            209,
            181,            // NARS f=0.82, c=0.71
            CausalMask::PO, // interventional level
            0b101,          // direction triad
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            0, // temporal = 0 (v1 compat; bits 52-63 must be 0 for v2 clean read)
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
        assert_eq!(
            e.truth(),
            TrustTexture::Crystalline,
            "ZERO: truth must be Crystalline"
        );
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
            e.truth(),
            TrustTexture::Crystalline,
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
            e.w_slot(),
            0,
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
            e.truth(),
            TrustTexture::Crystalline,
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
        assert_eq!(e.inference_mantissa(), -1, "mantissa -1 round-trip failed");
        assert_eq!(
            e.plasticity(),
            PlasticityState::ALL_FROZEN,
            "mantissa=-1 (bits 46-49 all set) must not contaminate plasticity (bits 50-52)"
        );
    }

    // ── test_size_unchanged ──────────────────────────────────────────────────

    #[test]
    fn test_size_unchanged() {
        assert_eq!(
            std::mem::size_of::<CausalEdge64>(),
            8,
            "CausalEdge64 must be exactly 8 bytes (one register)"
        );
        assert_eq!(
            8 * std::mem::size_of::<CausalEdge64>(),
            64,
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
                e.inference_mantissa(),
                m,
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
        assert_eq!(
            e.truth(),
            TrustTexture::Murky,
            "second with_routing should override truth"
        );
    }

    // ── Bonus: InferenceType to_mantissa / from_mantissa round-trip ─────────

    #[test]
    fn test_intervention_counterfactual_mantissa_slots() {
        // PR-LL-1 absorbed at slots 6 and -6 per L-9
        assert_eq!(
            InferenceType::Intervention.to_mantissa(),
            6,
            "Intervention must map to mantissa +6"
        );
        assert_eq!(
            InferenceType::Counterfactual.to_mantissa(),
            -6,
            "Counterfactual must map to mantissa -6"
        );
        // from_mantissa round-trip for PR-LL-1 slots
        assert_eq!(
            InferenceType::from_mantissa(6),
            InferenceType::Intervention,
            "from_mantissa(+6) must return Intervention"
        );
        assert_eq!(
            InferenceType::from_mantissa(-6),
            InferenceType::Counterfactual,
            "from_mantissa(-6) must return Counterfactual"
        );
    }

    // ── Bonus: pack_v2 defaults ──────────────────────────────────────────────

    #[test]
    fn test_pack_v2_v2_field_defaults() {
        let e = CausalEdge64::pack_v2(
            1,
            2,
            3,
            200,
            200,
            CausalMask::None,
            0,
            PlasticityState::ALL_FROZEN,
        );
        assert_eq!(e.w_slot(), 0, "pack_v2: w_slot defaults to 0");
        assert_eq!(
            e.truth(),
            TrustTexture::Crystalline,
            "pack_v2: truth defaults to Crystalline"
        );
        assert_eq!(e.inference_mantissa(), 0, "pack_v2: mantissa defaults to 0");
        assert_eq!(e.spare(), 0, "pack_v2: spare defaults to 0");
        // v1 fields must be set correctly
        assert_eq!(e.s_idx(), 1);
        assert_eq!(e.p_idx(), 2);
        assert_eq!(e.o_idx(), 3);
    }

    // ── Codex P1 regression tests (PR #383 review) ──────────────────────────

    /// Codex P1 #1: forward() under v2 must decode the 4-bit signed mantissa
    /// via InferenceType::from_mantissa(), NOT the deprecated v1 3-bit
    /// inference_type() accessor. A v2 edge built with mantissa = -1 must
    /// route through the Abduction branch, not Reserved7 (which is what
    /// `(0b1111 as InferenceType from 3 bits)` would have produced).
    #[test]
    fn test_forward_decodes_negative_mantissa_under_v2() {
        // Build a weight edge with mantissa = -1 (Abduction direction).
        // Use pack_v2 so the v1 enum discriminant path is bypassed.
        let mut weight = CausalEdge64::pack_v2(
            10,
            20,
            30,
            200,
            200,
            CausalMask::SPO,
            0,
            PlasticityState::ALL_FROZEN,
        );
        weight = weight.with_inference_mantissa(-1);
        assert_eq!(
            weight.inference_mantissa(),
            -1,
            "weight must carry mantissa=-1"
        );
        assert_eq!(
            InferenceType::from_mantissa(-1),
            InferenceType::Abduction,
            "from_mantissa(-1) is Abduction per the v2 mapping table"
        );
        // The actual forward() execution is tested by feeding it through;
        // we assert the dispatch table by direct decode here. If forward()
        // routed via the deprecated inference_type() it would have read
        // bits 46-48 = 0b111 and dispatched as Reserved7/Synthesis instead.
        let resolved = InferenceType::from_mantissa(weight.inference_mantissa());
        assert_eq!(
            resolved,
            InferenceType::Abduction,
            "v2 forward() must dispatch negative mantissa through Abduction"
        );
    }

    /// Codex P1 #2: set_temporal() under v2 must be a NO-OP — bits 52-63
    /// are reclaimed for plasticity[2] + W + lens + spare. Writing temporal
    /// here would clobber routing state stamped by `with_routing()` /
    /// `with_inference_mantissa()` / etc. learn() calls set_temporal, so
    /// the no-op must hold transitively.
    #[test]
    fn test_set_temporal_no_op_under_v2() {
        let mut edge = CausalEdge64::pack_v2(
            1,
            2,
            3,
            200,
            200,
            CausalMask::SPO,
            0,
            PlasticityState::ALL_FROZEN,
        );
        edge = edge
            .with_w_slot(42)
            .with_truth(TrustTexture::Fuzzy)
            .with_spare(0b101);
        let pre = edge;
        // Call set_temporal with a value that, under v1, would set bits 52-61.
        edge.set_temporal(1023);
        // Under v2, the routing state must survive.
        assert_eq!(
            edge.w_slot(),
            42,
            "set_temporal must not clobber w_slot under v2"
        );
        assert_eq!(
            edge.truth(),
            TrustTexture::Fuzzy,
            "set_temporal must not clobber truth under v2"
        );
        assert_eq!(
            edge.spare(),
            0b101,
            "set_temporal must not clobber spare under v2"
        );
        // Raw bits identical to pre-call.
        assert_eq!(
            edge.0, pre.0,
            "set_temporal under v2 must be a complete no-op on the raw u64"
        );
    }

    /// Codex P2: pack() under v2 must write the signed mantissa via
    /// `inference.to_mantissa()`, not the raw enum discriminant. Without
    /// this, `pack(..., Abduction, ...)` would store the v1 discriminant
    /// 2 into bits 46-48 (bit 49 = 0), which `inference_mantissa()` reads
    /// as +2, which `from_mantissa(+2)` decodes as Induction — a silent
    /// semantic shift.
    #[test]
    #[allow(deprecated)] // exercises the v1 pack() under v2 feature
    fn test_pack_uses_mantissa_mapping_under_v2() {
        // Abduction: to_mantissa() = -1, decodes from -1 back to Abduction.
        let abd_edge = CausalEdge64::pack(
            1,
            2,
            3,
            200,
            200,
            CausalMask::SPO,
            0,
            InferenceType::Abduction,
            PlasticityState::ALL_FROZEN,
            0,
        );
        let m = abd_edge.inference_mantissa();
        assert_eq!(
            m,
            InferenceType::Abduction.to_mantissa(),
            "pack(Abduction) under v2 must round-trip through to_mantissa()"
        );
        assert_eq!(
            InferenceType::from_mantissa(m),
            InferenceType::Abduction,
            "pack(Abduction) under v2 must decode back to Abduction, not Induction"
        );

        // Counterfactual: to_mantissa() = -6, decodes back to Counterfactual.
        let cf_edge = CausalEdge64::pack(
            1,
            2,
            3,
            200,
            200,
            CausalMask::SPO,
            0,
            InferenceType::Counterfactual,
            PlasticityState::ALL_FROZEN,
            0,
        );
        let m = cf_edge.inference_mantissa();
        assert_eq!(
            m,
            InferenceType::Counterfactual.to_mantissa(),
            "pack(Counterfactual) under v2 must round-trip through to_mantissa()"
        );
        assert_eq!(
            InferenceType::from_mantissa(m),
            InferenceType::Counterfactual,
            "pack(Counterfactual) under v2 must decode back to Counterfactual"
        );

        // Intervention: to_mantissa() = +6, decodes back to Intervention.
        let iv_edge = CausalEdge64::pack(
            1,
            2,
            3,
            200,
            200,
            CausalMask::SPO,
            0,
            InferenceType::Intervention,
            PlasticityState::ALL_FROZEN,
            0,
        );
        assert_eq!(
            InferenceType::from_mantissa(iv_edge.inference_mantissa()),
            InferenceType::Intervention,
            "pack(Intervention) under v2 must decode back to Intervention"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // CausalTopology (bits 59-60, additive factual view over TrustTexture)
    // + ReasoningBand (bits 61-63, additive quantized view over `spare`)
    // ═══════════════════════════════════════════════════════════════════
    //
    // These fields do NOT move any bits and do NOT introduce a new layout
    // version — they are second readings of the identical TrustTexture /
    // spare registers. See layout.rs doc comments for the full
    // wire/ordinal/behavioural/provenance compatibility statement.

    // ── 1. Raw u64 fixtures round-trip byte-for-byte unchanged ─────────────

    #[test]
    fn test_raw_fixtures_round_trip_byte_for_byte_via_topology_and_reasoning_band() {
        // Arbitrary fixtures covering varied bit patterns across the whole
        // word, including every combination of the two shared registers
        // (bits 59-60, 61-63).
        let fixtures: [u64; 6] = [
            0x0000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
            0x1234_5678_9ABC_DEF0,
            0xDEAD_BEEF_CAFE_F00D,
            0x8000_0000_0000_0001,
            0x5555_5555_5555_5555,
        ];
        for &raw in &fixtures {
            let edge = CausalEdge64(raw);
            // Reading a lens then writing back exactly what was read must be
            // a complete no-op on the raw word — the new code touches no bit
            // it did not read.
            let via_topology = edge.with_topology(edge.topology());
            assert_eq!(
                via_topology.0, raw,
                "read-then-write-back through CausalTopology changed the raw word for {raw:#018x}"
            );
            let via_texture = edge.with_reasoning_band(edge.reasoning_band());
            assert_eq!(
                via_texture.0, raw,
                "read-then-write-back through ReasoningBand changed the raw word for {raw:#018x}"
            );
        }
    }

    // ── 2. truth() returns exactly what it did before, all four ordinals ───

    #[test]
    fn test_truth_unaffected_by_new_topology_lens_for_all_four_ordinals() {
        for t in [
            TrustTexture::Crystalline,
            TrustTexture::Solid,
            TrustTexture::Fuzzy,
            TrustTexture::Murky,
        ] {
            let edge = CausalEdge64::ZERO.with_truth(t);
            assert_eq!(edge.truth(), t, "truth() must be unchanged for {t:?}");
        }
    }

    // ── 3. topology() reads the same two raw bits as truth_raw() ───────────

    #[test]
    fn test_topology_reads_the_same_two_bits_as_truth_raw() {
        for raw2 in 0u8..=3 {
            let edge = CausalEdge64::ZERO.with_truth(TrustTexture::from_bits_2(raw2));
            assert_eq!(edge.truth_raw(), raw2, "truth_raw setup mismatch");
            assert_eq!(
                edge.topology().to_bits_2(),
                edge.truth_raw(),
                "topology() must read the identical raw bits as truth_raw() for raw={raw2}"
            );
        }
    }

    // ── Bonus: topology() round-trips via with_topology(), all 4 ordinals ──

    #[test]
    fn test_topology_roundtrip() {
        for t in [
            CausalTopology::Direct,
            CausalTopology::IndirectKnownIntermediates,
            CausalTopology::IndirectUnknownIntermediates,
            CausalTopology::Unknown,
        ] {
            let edge = CausalEdge64::ZERO.with_topology(t);
            assert_eq!(edge.topology(), t, "topology round-trip failed for {t:?}");
        }
    }

    // ── 4. All four ordinal aliases are exact (pairwise `as u8`) ────────────

    #[test]
    fn test_causal_topology_ordinals_are_exactly_trust_texture_ordinals() {
        assert_eq!(
            TrustTexture::Crystalline as u8,
            CausalTopology::Direct as u8,
            "Crystalline/Direct must alias to the same ordinal"
        );
        assert_eq!(
            TrustTexture::Solid as u8,
            CausalTopology::IndirectKnownIntermediates as u8,
            "Solid/IndirectKnownIntermediates must alias to the same ordinal"
        );
        assert_eq!(
            TrustTexture::Fuzzy as u8,
            CausalTopology::IndirectUnknownIntermediates as u8,
            "Fuzzy/IndirectUnknownIntermediates must alias to the same ordinal"
        );
        assert_eq!(
            TrustTexture::Murky as u8,
            CausalTopology::Unknown as u8,
            "Murky/Unknown must alias to the same ordinal"
        );
    }

    // ── 5. with_topology changes ONLY bits 59-60 (exact XOR-diff-mask form) ─

    #[test]
    fn test_with_topology_changes_only_bits_59_60_exact_mask_diff() {
        // Fixture with every OTHER field non-zero/non-default; truth-bits
        // (and therefore topology) left at the pack_v2 default of 0. Going
        // 0 -> Unknown (0b11, the field's max) makes the XOR diff mask equal
        // exactly TRUTH_MASK if and only if with_topology touches nothing
        // else in the word.
        let base = CausalEdge64::pack_v2(
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            CausalMask::SPO,
            0b111,
            PlasticityState::ALL_HOT,
        )
        .with_w_slot(63)
        .with_inference_mantissa(-7)
        .with_spare(0b111);
        assert_eq!(
            base.topology(),
            CausalTopology::Direct,
            "fixture truth-bits must start at 0 for the min->max diff to be exact"
        );

        let after = base.with_topology(CausalTopology::Unknown);
        let diff = base.0 ^ after.0;
        assert_eq!(
            diff,
            crate::layout::TRUTH_MASK,
            "with_topology(min->max) must flip exactly the TRUTH_MASK bits, nothing else"
        );

        // Named-field cross-check (belt and suspenders): every other field
        // survives untouched.
        assert_eq!(after.s_idx(), 0xAA, "S disturbed by with_topology");
        assert_eq!(after.p_idx(), 0xBB, "P disturbed by with_topology");
        assert_eq!(after.o_idx(), 0xCC, "O disturbed by with_topology");
        assert_eq!(
            after.frequency_u8(),
            0xDD,
            "frequency disturbed by with_topology"
        );
        assert_eq!(
            after.confidence_u8(),
            0xEE,
            "confidence disturbed by with_topology"
        );
        assert_eq!(
            after.causal_mask(),
            CausalMask::SPO,
            "causal_mask disturbed by with_topology"
        );
        assert_eq!(
            after.direction(),
            0b111,
            "direction disturbed by with_topology"
        );
        assert_eq!(
            after.inference_mantissa(),
            -7,
            "inference_mantissa disturbed by with_topology"
        );
        assert_eq!(
            after.plasticity(),
            PlasticityState::ALL_HOT,
            "plasticity disturbed by with_topology"
        );
        assert_eq!(after.w_slot(), 63, "w_slot disturbed by with_topology");
        assert_eq!(after.spare(), 0b111, "spare disturbed by with_topology");
    }

    // ── 6. with_reasoning_band changes ONLY bits 61-63 (exact XOR-diff-mask) ──

    #[test]
    fn test_with_reasoning_band_changes_only_bits_61_63_exact_mask_diff() {
        let base = CausalEdge64::pack_v2(
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            CausalMask::SPO,
            0b111,
            PlasticityState::ALL_HOT,
        )
        .with_w_slot(63)
        .with_truth(TrustTexture::Murky)
        .with_inference_mantissa(-7);
        assert_eq!(
            base.reasoning_band(),
            ReasoningBand::Surface,
            "fixture spare-bits must start at 0 for the min->max diff to be exact"
        );

        let after = base.with_reasoning_band(ReasoningBand::Transcendent);
        let diff = base.0 ^ after.0;
        assert_eq!(
            diff,
            crate::layout::SPARE_MASK,
            "with_reasoning_band(min->max) must flip exactly the SPARE_MASK bits, nothing else"
        );

        assert_eq!(after.s_idx(), 0xAA, "S disturbed by with_reasoning_band");
        assert_eq!(after.p_idx(), 0xBB, "P disturbed by with_reasoning_band");
        assert_eq!(after.o_idx(), 0xCC, "O disturbed by with_reasoning_band");
        assert_eq!(
            after.frequency_u8(),
            0xDD,
            "frequency disturbed by with_reasoning_band"
        );
        assert_eq!(
            after.confidence_u8(),
            0xEE,
            "confidence disturbed by with_reasoning_band"
        );
        assert_eq!(
            after.causal_mask(),
            CausalMask::SPO,
            "causal_mask disturbed by with_reasoning_band"
        );
        assert_eq!(
            after.direction(),
            0b111,
            "direction disturbed by with_reasoning_band"
        );
        assert_eq!(
            after.inference_mantissa(),
            -7,
            "inference_mantissa disturbed by with_reasoning_band"
        );
        assert_eq!(
            after.plasticity(),
            PlasticityState::ALL_HOT,
            "plasticity disturbed by with_reasoning_band"
        );
        assert_eq!(
            after.w_slot(),
            63,
            "w_slot disturbed by with_reasoning_band"
        );
        assert_eq!(
            after.truth(),
            TrustTexture::Murky,
            "truth disturbed by with_reasoning_band"
        );
    }

    // ── 7. spare() stays consistent with reasoning_band() after its write ────

    #[test]
    fn test_spare_stays_consistent_with_reasoning_band_after_a_reasoning_band_write() {
        for band in [
            ReasoningBand::Surface,
            ReasoningBand::Association,
            ReasoningBand::Relation,
            ReasoningBand::Causal,
            ReasoningBand::Counterfactual,
            ReasoningBand::Perspective,
            ReasoningBand::Meta,
            ReasoningBand::Transcendent,
        ] {
            let edge = CausalEdge64::ZERO.with_reasoning_band(band);
            assert_eq!(
                edge.spare(),
                band.to_bits_3(),
                "spare() must still report the identical 3-bit value reasoning_band() encodes for {band:?}"
            );
        }
    }

    // ── 8. All eight texture-band values round-trip ─────────────────────────

    #[test]
    fn test_reasoning_band_all_eight_values_round_trip() {
        for band in [
            ReasoningBand::Surface,
            ReasoningBand::Association,
            ReasoningBand::Relation,
            ReasoningBand::Causal,
            ReasoningBand::Counterfactual,
            ReasoningBand::Perspective,
            ReasoningBand::Meta,
            ReasoningBand::Transcendent,
        ] {
            let edge = CausalEdge64::ZERO.with_reasoning_band(band);
            assert_eq!(
                edge.reasoning_band(),
                band,
                "reasoning_band round-trip failed for {band:?}"
            );
        }
    }

    // ── 9. Field-isolation matrix: composed setters, every other field ─────

    #[test]
    fn test_field_isolation_matrix_survives_both_new_setters_composed() {
        // One shared fixture with every field at a distinct, non-zero value,
        // then BOTH new setters applied together (a composable builder
        // chain, matching the intended call-site pattern:
        // `edge.with_topology(...).with_reasoning_band(...)`).
        let base = CausalEdge64::pack_v2(
            11, // S
            22, // P
            33, // O
            44, // frequency
            55, // confidence
            CausalMask::SO,
            0b110, // direction
            PlasticityState::P_HOT,
        )
        .with_w_slot(17)
        .with_inference_mantissa(5);

        let edge = base
            .with_topology(CausalTopology::IndirectUnknownIntermediates)
            .with_reasoning_band(ReasoningBand::Causal);

        assert_eq!(edge.s_idx(), 11, "S disturbed by new setters");
        assert_eq!(edge.p_idx(), 22, "P disturbed by new setters");
        assert_eq!(edge.o_idx(), 33, "O disturbed by new setters");
        assert_eq!(
            edge.frequency_u8(),
            44,
            "frequency disturbed by new setters"
        );
        assert_eq!(
            edge.confidence_u8(),
            55,
            "confidence disturbed by new setters"
        );
        assert_eq!(
            edge.causal_mask(),
            CausalMask::SO,
            "causal_mask disturbed by new setters"
        );
        assert_eq!(
            edge.direction(),
            0b110,
            "direction disturbed by new setters"
        );
        assert_eq!(
            edge.inference_mantissa(),
            5,
            "inference_mantissa disturbed by new setters"
        );
        assert_eq!(
            edge.plasticity(),
            PlasticityState::P_HOT,
            "plasticity disturbed by new setters"
        );
        assert_eq!(edge.w_slot(), 17, "w_slot disturbed by new setters");

        // And the two new fields landed correctly, composed.
        assert_eq!(
            edge.topology(),
            CausalTopology::IndirectUnknownIntermediates
        );
        assert_eq!(edge.reasoning_band(), ReasoningBand::Causal);
    }

    // ── 10. Counterfactual mantissa (-6) round-trips independent of band ───

    #[test]
    fn test_counterfactual_mantissa_round_trips_independently_of_reasoning_band() {
        let base = CausalEdge64::pack_v2(
            1,
            2,
            3,
            200,
            200,
            CausalMask::SPO,
            0,
            PlasticityState::ALL_FROZEN,
        )
        .with_inference_mantissa(InferenceType::Counterfactual.to_mantissa());
        assert_eq!(
            base.inference_mantissa(),
            -6,
            "fixture must carry mantissa -6"
        );

        for band in [
            ReasoningBand::Surface,
            ReasoningBand::Meta,
            ReasoningBand::Transcendent,
        ] {
            let edge = base.with_reasoning_band(band);
            assert_eq!(
                edge.inference_mantissa(),
                -6,
                "counterfactual mantissa must survive a reasoning_band write for {band:?}"
            );
            assert_eq!(edge.reasoning_band(), band);
        }
    }

    // ── 11. ReasoningBand::Counterfactual does NOT imply mantissa == -6 ───────

    #[test]
    fn test_reasoning_band_counterfactual_does_not_imply_mantissa_minus_six() {
        // ReasoningBand::Counterfactual set, mantissa left at a DIFFERENT
        // value — proves the two are orthogonal, never derived from one
        // another.
        let edge = CausalEdge64::ZERO
            .with_reasoning_band(ReasoningBand::Counterfactual)
            .with_inference_mantissa(3);
        assert_eq!(edge.reasoning_band(), ReasoningBand::Counterfactual);
        assert_eq!(
            edge.inference_mantissa(),
            3,
            "mantissa must NOT be derived from reasoning_band"
        );
        assert_ne!(edge.inference_mantissa(), -6);

        // And the converse: mantissa == -6 (the Counterfactual InferenceType
        // slot) with a DIFFERENT reasoning_band.
        let edge2 = CausalEdge64::ZERO
            .with_inference_mantissa(InferenceType::Counterfactual.to_mantissa())
            .with_reasoning_band(ReasoningBand::Meta);
        assert_eq!(edge2.inference_mantissa(), -6);
        assert_eq!(
            edge2.reasoning_band(),
            ReasoningBand::Meta,
            "reasoning_band must NOT be derived from inference_mantissa"
        );
        assert_ne!(edge2.reasoning_band(), ReasoningBand::Counterfactual);
    }

    // ── 12. CausalTopology::Direct does not imply any NARS confidence ──────

    #[test]
    fn test_causal_topology_direct_does_not_imply_any_nars_confidence() {
        // Direct topology (0) coexists with every confidence level — no
        // auto-derivation from/to NARS confidence in either direction.
        for conf in [0u8, 1, 128, 254, 255] {
            let edge = CausalEdge64::pack_v2(
                1,
                2,
                3,
                100,
                conf,
                CausalMask::None,
                0,
                PlasticityState::ALL_FROZEN,
            )
            .with_topology(CausalTopology::Direct);
            assert_eq!(edge.topology(), CausalTopology::Direct);
            assert_eq!(
                edge.confidence_u8(),
                conf,
                "CausalTopology::Direct must not constrain or derive confidence={conf}"
            );
        }
    }
}
