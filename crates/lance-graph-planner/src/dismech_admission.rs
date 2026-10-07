//! **DisMech admission — the one DisMech-specific piece of chain replay.**
//!
//! [`crate::chain_replay`] and [`crate::chain_counterfactual`] are
//! domain-agnostic: a step's predicate ordinal is an opaque `u8` they carry
//! as witness and never read. What IS DisMech-specific is the question *"is
//! this ordinal a minted DisMech causal predicate?"*, asked once when a chain
//! is first accepted. That check lives here, split out of the former
//! `dismech_replay` module so the replay core no longer names a domain.
//!
//! The palette mirror is the contract's
//! (`lance_graph_contract::dismech_evidence`); the fuse that proves the mirror
//! IS the palette lives in the armed tier
//! (`lance_graph_ogar::parity::assert_dismech_palette_parity`).

use crate::chain_replay::ChainStep;

/// Resolve a step's predicate ordinal to `(ordinal, name, curie)`, or `None`
/// when the byte names no minted DisMech predicate.
///
/// **Why this exists rather than a bare `u8`.** The replay arithmetic does not
/// read the ordinal in W1 — it is the ADDRESS the step travels under, not an
/// operand — so nothing in the hot path would ever notice a corrupt one. That
/// is exactly why the domain needs a name: a chain carrying `0xA3` is not a
/// causal chain at all (`0xA3` is the palette's SEARCH band), and without this
/// a replay would trace it to a byte-identical, entirely meaningless result.
///
/// The mirror is the contract's; the fuse that proves it IS the palette lives
/// in `lance_graph_ogar::parity::assert_dismech_palette_parity`, because
/// `ogar-dismech` is reachable only from that workspace-EXCLUDED armed tier.
/// Mirror here, authority there — the same split `ogar_codebook` already uses.
#[must_use]
pub fn chain_step_predicate(step: ChainStep) -> Option<&'static (u8, &'static str, &'static str)> {
    lance_graph_contract::dismech_evidence::dismech_predicate(step.0)
}

/// A step whose predicate ordinal names no minted DisMech predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnmintedOrdinal {
    /// Index of the offending step within the chain.
    pub step: usize,
    /// The ordinal that resolved to nothing.
    pub ordinal: u8,
}

/// Check that every step of a chain travels under a minted DisMech predicate.
///
/// # Why this is an ADMISSION check and NOT part of [`crate::chain_replay::replay_chain`]
///
/// Review on #1120 proposed validating inside the replay loop and returning an
/// error before emitting a trace row. The defect it names is real — a chain
/// carrying `0xA3` (the palette's SEARCH band) is not a causal chain, and
/// replaying it produces a byte-identical, meaningless trace. But the remedy
/// belongs one step earlier, for a reason that goes to the plan's keystone:
///
/// **Replay must not refuse history.** A recorded chain is a fact about what
/// was evaluated; the engine's job is to reproduce it, not to judge it. If the
/// palette ever drops or renumbers an ordinal, a replay that validated would
/// start returning `Err` for chains that were perfectly valid when recorded —
/// and "yesterday's evaluation replays today byte-for-byte" is precisely the
/// property the whole wave exists to hold.
///
/// So the judgement happens once, when a chain is FIRST accepted, and replay
/// stays total over everything already admitted.
///
/// # "Admission" means first acceptance — NOT loading a recording
///
/// The previous wording listed *"loading a recording"* as an admission site,
/// which **contradicted the paragraph above it** — reloading an older durable
/// recording and validating it against today's palette rejects exactly the
/// history the split exists to keep replayable. Codex caught this on #1122;
/// the argument was right and the instruction beneath it was wrong.
///
/// The rule, stated so the two cannot drift apart again:
///
/// - **A chain arriving from outside** (a producer, a boundary, a new
///   recording being made) is validated against the CURRENT palette, here.
/// - **A chain being re-read from the durable log** is already admitted. It is
///   not re-validated — the fact that it was recorded IS its admission, under
///   whatever palette was current then. Replay it.
/// - A caller that genuinely needs to check an old recording must check it
///   against the palette version it was admitted under, which this function
///   cannot do: it has one palette, today's. That is a versioned-palette
///   capability nothing in this wave has, and inventing one here would be
///   worse than declining.
///
/// Fails closed and reports WHICH step, so a rejection is actionable rather
/// than a boolean.
pub fn validate_chain(chain: &[ChainStep]) -> Result<(), UnmintedOrdinal> {
    for (step, &entry) in chain.iter().enumerate() {
        if chain_step_predicate(entry).is_none() {
            return Err(UnmintedOrdinal {
                step,
                ordinal: entry.0,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_replay::tests::{chain, compose_tables, edge, Lcg};
    use crate::chain_replay::{replay_chain, ComposeTables};
    use causal_edge::tables::NarsTables;

    #[test]
    fn a_chain_steps_ordinal_names_a_minted_predicate_and_a_search_op_does_not() {
        // The core cannot reach `ogar_dismech` (workspace boundary), so what
        // it CAN pin is that a step's ordinal resolves through the contract
        // mirror to the predicate the plan names — and that the byte one past
        // the band does not. The mirror-IS-the-palette half is fused in
        // `lance_graph_ogar::parity::assert_dismech_palette_parity`; neither
        // half alone is the claim.
        let mut rng = Lcg(0x51D3_7C4A_9B2E_6F08);
        let w = edge(&mut rng);
        assert_eq!(chain_step_predicate((0x90, w)).map(|p| p.1), Some("causes"),);
        assert_eq!(
            chain_step_predicate((0xA2, w)).map(|p| p.1),
            Some("variant_of"),
        );
        // Can-stay-silent, on a byte that is a REAL slot elsewhere in the
        // palette rather than an arbitrary one: 0xA3 is `CANDIDATES`, the
        // first search op. An empty-input silence case would prove nothing.
        assert!(chain_step_predicate((0xA3, w)).is_none());
        assert!(chain_step_predicate((0x8F, w)).is_none());
    }
    #[test]
    fn a_chain_carrying_a_search_op_is_refused_at_admission_and_still_replays() {
        // CodeRabbit #1120 asked that `replay_chain` itself reject an ordinal
        // outside the predicate band. Both halves are pinned here, because the
        // SPLIT is the decision, not either half alone.
        let mut rng = Lcg(0x5EA2_C401_9D3B_77E6);
        let mut c = chain(&mut rng, 5);
        c[2].0 = 0xA3; // CANDIDATES — a real slot, in the SEARCH band

        // (a) admission REFUSES it, and names the step so the rejection is
        //     actionable rather than a boolean.
        assert_eq!(
            validate_chain(&c),
            Err(UnmintedOrdinal {
                step: 2,
                ordinal: 0xA3
            }),
        );
        // Anti-vacuity: the same chain without that step must PASS, or the
        // check could be rejecting everything.
        let mut clean = c.clone();
        clean[2].0 = 0x90;
        assert_eq!(validate_chain(&clean), Ok(()));

        // (b) replay stays TOTAL over it. Replay must not refuse history: a
        //     recorded chain is a fact, and a replay that judged content would
        //     start returning Err for chains that were valid when recorded —
        //     against the keystone this whole wave exists to hold.
        let tables = NarsTables::build(1);
        let c_tabs = compose_tables();
        let tabs = ComposeTables {
            s: &c_tabs[0],
            p: &c_tabs[1],
            o: &c_tabs[2],
        };
        let seed = edge(&mut rng);
        let t = replay_chain(&c, seed, &tables, tabs, 1, 10).expect("reservation fits");
        assert_eq!(t.len(), 5);
        assert_eq!(
            t[2].predicate, 0xA3,
            "the witness records what was replayed"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn the_old_module_paths_still_name_the_same_items() {
        // The rename keeps `dismech_replay` / `dismech_counterfactual` as
        // deprecated aliases. Pin that the old paths resolve to the SAME
        // functions, not to copies that could drift.
        let old: fn(&[ChainStep]) -> Result<(), UnmintedOrdinal> =
            crate::dismech_replay::validate_chain;
        let new: fn(&[ChainStep]) -> Result<(), UnmintedOrdinal> = validate_chain;
        assert!(core::ptr::fn_addr_eq(old, new));
        let old_seq: fn(u64, usize) -> Option<u64> = crate::dismech_replay::next_base_seq;
        let new_seq: fn(u64, usize) -> Option<u64> = crate::chain_replay::next_base_seq;
        assert!(core::ptr::fn_addr_eq(old_seq, new_seq));
        assert_eq!(
            crate::dismech_counterfactual::DEFAULT_FREQUENCY_BAR,
            crate::chain_counterfactual::DEFAULT_FREQUENCY_BAR,
        );
    }
}
