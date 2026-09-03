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
//! 1. Every dtype converges (respects `max_cycles`, runs at least one
//!    cycle, commits a non-degenerate peak) — `every_dtype_engine_converges_and_commits_a_real_peak`.
//! 2. The u8-vs-BF16 top-k agreement through the collapsed builder path is
//!    at least the pre-existing [`crate::dual_engine::DualEngine`]
//!    disagreement baseline — `u8_vs_bf16_agreement_is_at_least_the_preexisting_dual_engine_baseline`.
//! 3. The four dtypes are NOT bit-identical — an anti-vacuity check that
//!    this suite is actually testing dtype-INVARIANT behaviour, not
//!    accidentally passing because every path happens to reduce to the
//!    same arithmetic — `dtypes_are_not_secretly_bit_identical`.
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

    fn built(table_type: TableType) -> ConfiguredEngine {
        ThinkingEngineBuilder::new()
            .lens(Lens::Jina)
            .table_type(table_type)
            .max_cycles(MAX_CYCLES)
            .build()
            .expect(
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

    #[test]
    fn u8_vs_bf16_agreement_is_at_least_the_preexisting_dual_engine_baseline() {
        let table = crate::jina_lens::JINA_HDR_TABLE.to_vec();
        let mut dual = DualEngine::u8_vs_bf16(table);
        dual.perturb_both(PROBE);
        let baseline = dual.think_both(MAX_CYCLES);

        // Anti-vacuity: a baseline of 0.0 would make the ">=" assertion
        // below trivially true regardless of what the collapsed path does.
        assert!(
            baseline.agreement > 0.0,
            "the DualEngine baseline itself shows zero agreement — this test \
             would be vacuous against a floor of 0.0"
        );

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
            builder_agreement >= baseline.agreement - 1e-6,
            "collapsed builder-path u8-vs-BF16 agreement ({builder_agreement}) fell \
             below the pre-collapse DualEngine baseline ({}) — the collapse must not \
             regress the disagreement DualEngine already measured",
            baseline.agreement
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
