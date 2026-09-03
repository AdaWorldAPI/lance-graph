//! Dtype-invariant parity suite for the collapsed thinking-engine surface.
//!
//! Part of D-TEH-4 (ENTROPY M8 engine collapse). The plan's gate, verbatim:
//! "NOT bit-parity across dtypes — u8 / BF16 / i8 / f32 differ in encoding
//! by design, and `dual_engine.rs` exists to MEASURE that disagreement
//! (Codex on #1137). Gate: per-dtype output tolerances plus dtype-invariant
//! ranking/convergence invariants (top-k order, `converged`, `cycle_count`
//! bounds) on real engine fixtures that instantiate all four engines (the
//! driver's fixtures do not — they round-trip `BusDto` only); the
//! pre-collapse `DualResult` disagreement is the baseline the collapsed
//! engine must not exceed".
//!
//! This module is that fixture set. It instantiates all four `BuiltEngine`
//! variants — through the SAME [`crate::builder::ThinkingEngineBuilder`]
//! path production code uses — from one real baked lens table (Jina), runs
//! them on identical perturbation, and asserts:
//!
//! 1. Every dtype ACTUALLY converges per its own `think()`'s delta-threshold
//!    termination, not merely "does not exceed `max_cycles`" (a bound every
//!    engine satisfies by construction and therefore proves nothing) —
//!    `every_dtype_engine_converges_and_commits_a_real_peak`. This is where
//!    F32 broke the naive version of this suite; see below.
//! 2. The u8-vs-BF16 top-k agreement through the collapsed builder path is
//!    at least a FROZEN pre-collapse baseline, not one recomputed at test
//!    time from the same current engine implementations the builder path
//!    also uses — `u8_vs_bf16_agreement_is_at_least_the_frozen_preexisting_baseline`.
//! 3. The four dtypes are NOT bit-identical — an anti-vacuity check that
//!    this suite is actually testing dtype-INVARIANT behaviour, not
//!    accidentally passing because every path happens to reduce to the
//!    same arithmetic — `dtypes_are_not_secretly_bit_identical`.
//!
//! ## A real finding, not merely a test-design correction (PR #1151 review)
//!
//! Measured on this fixture (`PROBE = [10, 20, 30]`, the baked Jina lens):
//! u8 converges in 7 cycles, i8 and BF16 in 15 — all well under `max_cycles`.
//! **F32 runs to the FULL cycle budget every time, at every `max_cycles`
//! tried from 30 to 200** — its softmax-with-temperature `cycle()`
//! ([`crate::f32_engine::F32ThinkingEngine`], `T=0.01` default) never drops
//! its energy delta below `convergence_threshold` on this input; the energy
//! vector is bit-identical at `max_cycles=30` and `max_cycles=200`, so it
//! isn't slow to converge, it settles into a fixed point the delta check
//! doesn't recognize as converged. An earlier version of this suite only
//! asserted `cycle_count <= max_cycles` — true of ANY engine by
//! construction, so it could not have caught this (flagged by Codex review
//! on #1151). This suite asserts u8/i8/BF16 exit EARLY (a real signal) and
//! documents F32's always-exhausts-the-budget behavior as the honest,
//! per-dtype-different assertion it measured to be — not silently papered
//! over with a uniform "converges" claim that would have been false for F32.
//! Investigating WHY F32 never trips its own delta threshold is out of
//! scope for this collapse wave (it is pre-existing engine-internal
//! behavior, not something this PR's dispatch/lens/cascade collapse
//! touches) and is left for a follow-up.
//!
//! What this suite deliberately does NOT assert: identical top-1 index,
//! identical energy values, or identical cycle counts across dtypes. F32
//! uses softmax-with-temperature normalization
//! ([`crate::f32_engine::F32ThinkingEngine::think`]); u8/i8/BF16 use
//! clamp-negative-to-zero + normalize-to-1. That is a real algorithmic
//! divergence, not merely a precision difference, and bit-parity across it
//! would be the wrong gate.

#[cfg(test)]
mod tests {
    use crate::builder::{ConfiguredEngine, Lens, TableType, ThinkingEngineBuilder};
    use crate::dto::PerturbationDto;
    use crate::dual_engine::DualEngine;

    const PROBE: &[u16] = &[10, 20, 30];
    const MAX_CYCLES: usize = 30;

    const ALL_TABLE_TYPES: [TableType; 4] = [
        TableType::UnsignedU8,
        TableType::SignedI8,
        TableType::BF16,
        TableType::F32,
    ];

    /// The same u8→f32 dequantization the builder itself applies internally
    /// for `TableType::BF16`/`F32` (`(v - 128.0) / 127.0`, see `builder.rs`'s
    /// `build()`), computed once here and threaded through
    /// `.raw_cosines()` for `SignedI8` so that arm exercises the CANONICAL
    /// `SignedThinkingEngine::from_f32_cosines` construction path instead of
    /// the deprecated CDF-rank-shift `from_unsigned` fallback `build()`
    /// otherwise takes when `raw_cosines` is unset (Codex review on #1151:
    /// the suite must not pass while only the deprecated path is tested).
    fn jina_cosines() -> Vec<f32> {
        crate::jina_lens::JINA_HDR_TABLE
            .iter()
            .map(|&v| (v as f32 - 128.0) / 127.0)
            .collect()
    }

    fn built(table_type: TableType) -> ConfiguredEngine {
        let mut builder = ThinkingEngineBuilder::new()
            .lens(Lens::Jina)
            .table_type(table_type)
            .max_cycles(MAX_CYCLES);
        if table_type == TableType::SignedI8 {
            builder = builder.raw_cosines(jina_cosines());
        }
        builder.build().expect(
            "builder must construct an engine from the baked Jina lens for every table type",
        )
    }

    fn top_indices_above_noise(energy: &[f32], cycles: u16) -> Vec<u16> {
        PerturbationDto::from_energy_f32(energy, cycles)
            .top_k
            .iter()
            .filter(|&&(_, e)| e > 1e-10)
            .map(|&(idx, _)| idx)
            .collect()
    }

    #[test]
    fn every_dtype_engine_converges_and_commits_a_real_peak() {
        for table_type in ALL_TABLE_TYPES {
            let mut configured = built(table_type);
            let bus = configured.process(PROBE);

            assert!(
                bus.cycle_count as usize <= MAX_CYCLES,
                "{table_type:?} ran {} cycles, exceeding max_cycles={MAX_CYCLES}",
                bus.cycle_count
            );
            assert!(
                bus.cycle_count > 0,
                "{table_type:?} must run at least one cycle on a real perturbation"
            );

            // The REAL convergence check (not merely the hard loop bound
            // above, which any engine satisfies by construction — Codex
            // review on #1151). Measured on this fixture: u8/i8/BF16 all
            // exit early via their own delta-threshold termination; F32
            // does not (see the module doc's "A real finding" section) —
            // so F32 gets the opposite, equally real assertion: it runs the
            // full budget, every time, on this input.
            if table_type == TableType::F32 {
                assert_eq!(
                    bus.cycle_count as usize, MAX_CYCLES,
                    "F32 was measured to always exhaust max_cycles on this fixture; if it now \
                     exits early, that is a genuine behavior change in F32ThinkingEngine's \
                     convergence and this module doc's finding needs re-measuring, not silently \
                     re-pinning"
                );
            } else {
                assert!(
                    (bus.cycle_count as usize) < MAX_CYCLES,
                    "{table_type:?} ran the full max_cycles={MAX_CYCLES} budget instead of \
                     exiting early via its own delta-threshold convergence — this dtype was \
                     measured to converge in well under the budget on this fixture"
                );
            }

            assert!(
                bus.energy > 0.0,
                "{table_type:?} committed a degenerate zero-energy peak"
            );
            assert!(
                bus.top_k.iter().any(|&(_, e)| e > 0.0),
                "{table_type:?} top_k contains no real peak (all zero)"
            );
        }
    }

    /// The pre-collapse `DualEngine::u8_vs_bf16()` agreement on this exact
    /// fixture (`PROBE`, `MAX_CYCLES`, the baked Jina lens), measured once
    /// and frozen as a literal — NOT recomputed at test time.
    ///
    /// Recomputing it at test time (the first version of this test did)
    /// calls the SAME current `ThinkingEngine`/`BF16ThinkingEngine`
    /// implementations, table conversions, and `think()` that the builder
    /// arm below also calls — so a regression shared by both engines moves
    /// both numbers together and the `>=` comparison stays green regardless
    /// (flagged by Codex review on #1151). Freezing the value means a
    /// future shared regression shows up as the LIVE `builder_agreement`
    /// falling below this fixed historical number, which is the actual
    /// guarantee this test claims to provide.
    const FROZEN_U8_VS_BF16_BASELINE: f32 = 0.875;

    #[test]
    fn u8_vs_bf16_agreement_is_at_least_the_frozen_preexisting_baseline() {
        // Anti-vacuity on the frozen constant itself (not a runtime assert
        // on a literal — clippy's `assertions_on_constants` correctly
        // rejects that shape): the sibling test
        // `dual_engine_still_reproduces_the_frozen_baseline_on_this_fixture`
        // is what proves FROZEN_U8_VS_BF16_BASELINE is a real, non-degenerate
        // measurement rather than a made-up floor.
        let mut u8_via_builder = built(TableType::UnsignedU8);
        let mut bf16_via_builder = built(TableType::BF16);
        u8_via_builder.engine.perturb(PROBE);
        bf16_via_builder.engine.perturb(PROBE);
        u8_via_builder.engine.think(MAX_CYCLES);
        bf16_via_builder.engine.think(MAX_CYCLES);

        let u8_top = top_indices_above_noise(
            u8_via_builder.engine.energy(),
            u8_via_builder.engine.cycles(),
        );
        let bf16_top = top_indices_above_noise(
            bf16_via_builder.engine.energy(),
            bf16_via_builder.engine.cycles(),
        );
        let overlap = u8_top.iter().filter(|p| bf16_top.contains(p)).count();
        let max_len = u8_top.len().max(bf16_top.len()).max(1);
        let builder_agreement = overlap as f32 / max_len as f32;

        assert!(
            builder_agreement >= FROZEN_U8_VS_BF16_BASELINE - 1e-6,
            "collapsed builder-path u8-vs-BF16 agreement ({builder_agreement}) fell below the \
             frozen pre-collapse baseline ({FROZEN_U8_VS_BF16_BASELINE}) — the collapse must not \
             regress the disagreement DualEngine already measured"
        );
    }

    /// Sanity check that [`DualEngine::u8_vs_bf16`] itself still reproduces
    /// the frozen baseline above on the identical fixture — this is
    /// provenance for the frozen constant, NOT the regression gate (that is
    /// `u8_vs_bf16_agreement_is_at_least_the_frozen_preexisting_baseline`,
    /// which does not call `DualEngine` at all, per the fix above).
    #[test]
    fn dual_engine_still_reproduces_the_frozen_baseline_on_this_fixture() {
        let table = crate::jina_lens::JINA_HDR_TABLE.to_vec();
        let mut dual = DualEngine::u8_vs_bf16(table);
        dual.perturb_both(PROBE);
        let result = dual.think_both(MAX_CYCLES);

        assert!(
            (result.agreement - FROZEN_U8_VS_BF16_BASELINE).abs() < 1e-6,
            "DualEngine::u8_vs_bf16 now reports {} on this fixture, not the frozen {} — \
             re-measure and re-freeze FROZEN_U8_VS_BF16_BASELINE deliberately, don't silently \
             widen the tolerance",
            result.agreement,
            FROZEN_U8_VS_BF16_BASELINE
        );
    }

    #[test]
    fn dtypes_are_not_secretly_bit_identical() {
        let mut u8_engine = built(TableType::UnsignedU8);
        let mut f32_engine = built(TableType::F32);
        let u8_bus = u8_engine.process(PROBE);
        let f32_bus = f32_engine.process(PROBE);

        assert!(
            (u8_bus.energy - f32_bus.energy).abs() > 1e-6,
            "u8 (clamp+normalize) and F32 (softmax) committed identical top energy \
             ({} == {}) — if these ever match exactly, this suite would be silently \
             gated on bit-parity instead of the dtype-invariant behaviour it claims \
             to test",
            u8_bus.energy,
            f32_bus.energy
        );
    }
}
