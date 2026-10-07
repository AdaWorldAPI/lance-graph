//! **Chain admission — the palette a chain is checked against comes from its
//! classid.** The one domain-aware step in front of chain replay.
//!
//! [`crate::chain_replay`] and [`crate::chain_counterfactual`] are
//! domain-agnostic: a step's predicate ordinal is an opaque `u8` they carry as
//! witness and never read. The ordinal only means something inside a
//! vocabulary, and the vocabulary is selected by the chain's classid — the G of
//! its SPO-G quad. `0x90` is `causes` under the DisMech concept (`0x0333`), but
//! the loco floor is shared: the NARS recipes and the r2il ops also mint from
//! `0x90`. Checking a bare byte against one palette would read every other
//! vocabulary's `0x90` as `causes`.
//!
//! So admission takes `(classid, chain)`, routes by the concept half of the
//! classid ([`graph_of_classid`]), and checks the steps against that palette.
//! This mirrors OGAR's `VocabularyRegistry::resolve_classid` for the palettes
//! the zero-dependency contract mirrors; today that is DisMech's
//! (`lance_graph_contract::dismech_evidence`). The fuse that proves the mirror
//! IS the palette, concept id included, is
//! `lance_graph_ogar::parity::assert_dismech_palette_parity`.

use lance_graph_contract::dismech_evidence::{dismech_predicate, DISMECH_CONCEPT_ID};

use crate::chain_replay::ChainStep;

/// One minted predicate: `(ordinal, name, curie)`.
pub type PredicateRow = (u8, &'static str, &'static str);

/// A palette: ordinal to its minted row, `None` outside the band.
pub type Palette = fn(u8) -> Option<&'static PredicateRow>;

/// The palettes this crate can admit against, keyed by concept (G).
///
/// One entry per mirrored vocabulary. A concept with no entry is refused
/// ([`Unadmitted::UnknownPalette`]), never checked against another concept's
/// palette.
pub const PALETTES: &[(u16, Palette)] = &[(DISMECH_CONCEPT_ID, dismech_predicate)];

/// The DisMech classid under the bare concept, no app prefix. Any app prefix
/// routes the same way; G is the concept half alone.
pub const DISMECH_CLASSID: u32 = (DISMECH_CONCEPT_ID as u32) << 16;

/// The graph coordinate of a classid: its canon-high concept half
/// (`contract::spog_tenants::graph_of`, for a bare classid).
#[must_use]
pub const fn graph_of_classid(classid: u32) -> u16 {
    (classid >> 16) as u16
}

/// The palette a classid routes to, if this crate mirrors it.
#[must_use]
pub fn palette_of(classid: u32) -> Option<Palette> {
    let concept = graph_of_classid(classid);
    PALETTES
        .iter()
        .find(|(c, _)| *c == concept)
        .map(|(_, p)| *p)
}

/// A step whose predicate ordinal names no minted predicate of its palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnmintedOrdinal {
    /// Index of the offending step within the chain.
    pub step: usize,
    /// The ordinal that resolved to nothing.
    pub ordinal: u8,
}

/// Why a chain was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unadmitted {
    /// The classid's concept has no palette this crate mirrors.
    UnknownPalette {
        /// The concept half of the classid.
        concept: u16,
    },
    /// A step's ordinal is not minted in the classid's palette.
    Unminted(UnmintedOrdinal),
}

/// Resolve one step's predicate under `classid`.
///
/// # Errors
///
/// [`Unadmitted::UnknownPalette`] when the classid routes to no mirrored
/// palette; [`Unadmitted::Unminted`] (with `step: 0`) when the ordinal is
/// outside the palette's band. `0xA3` under DisMech is `CANDIDATES`, the first
/// SEARCH op: a real slot, but not a causal predicate.
pub fn chain_step_predicate(
    classid: u32,
    step: ChainStep,
) -> Result<&'static PredicateRow, Unadmitted> {
    let palette = palette_of(classid).ok_or(Unadmitted::UnknownPalette {
        concept: graph_of_classid(classid),
    })?;
    palette(step.0).ok_or(Unadmitted::Unminted(UnmintedOrdinal {
        step: 0,
        ordinal: step.0,
    }))
}

/// Check that every step of a chain travels under a minted predicate of the
/// palette its classid routes to.
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
/// # Errors
///
/// [`Unadmitted::UnknownPalette`] before any step is read when the classid
/// routes nowhere; otherwise the first unminted step, by index.
pub fn validate_chain(classid: u32, chain: &[ChainStep]) -> Result<(), Unadmitted> {
    let palette = palette_of(classid).ok_or(Unadmitted::UnknownPalette {
        concept: graph_of_classid(classid),
    })?;
    for (step, &entry) in chain.iter().enumerate() {
        if palette(entry.0).is_none() {
            return Err(Unadmitted::Unminted(UnmintedOrdinal {
                step,
                ordinal: entry.0,
            }));
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
        assert_eq!(
            chain_step_predicate(DISMECH_CLASSID, (0x90, w))
                .map(|p| p.1)
                .ok(),
            Some("causes"),
        );
        assert_eq!(
            chain_step_predicate(DISMECH_CLASSID, (0xA2, w))
                .map(|p| p.1)
                .ok(),
            Some("variant_of"),
        );
        // Can-stay-silent, on a byte that is a REAL slot elsewhere in the
        // palette rather than an arbitrary one: 0xA3 is `CANDIDATES`, the
        // first search op. An empty-input silence case would prove nothing.
        assert!(chain_step_predicate(DISMECH_CLASSID, (0xA3, w)).is_err());
        assert!(chain_step_predicate(DISMECH_CLASSID, (0x8F, w)).is_err());
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
            validate_chain(DISMECH_CLASSID, &c),
            Err(Unadmitted::Unminted(UnmintedOrdinal {
                step: 2,
                ordinal: 0xA3
            })),
        );
        // Anti-vacuity: the same chain without that step must PASS, or the
        // check could be rejecting everything.
        let mut clean = c.clone();
        clean[2].0 = 0x90;
        assert_eq!(validate_chain(DISMECH_CLASSID, &clean), Ok(()));

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
    fn the_same_byte_is_admitted_only_under_the_classid_whose_palette_mints_it() {
        // 0x90 is `causes` under DisMech and recipe #1 under the NARS recipe
        // vocabulary. A chain is checked against the palette its classid
        // routes to, never against DisMech by default.
        let mut rng = Lcg(0x0333_0306_0000_0001);
        let c = chain(&mut rng, 4);
        assert_eq!(validate_chain(DISMECH_CLASSID, &c), Ok(()));
        // RO relation bodies (0x0306) — a real concept with its own palette,
        // not mirrored here: refused, not read as DisMech.
        let ro_classid = 0x0306_0000;
        assert_eq!(
            validate_chain(ro_classid, &c),
            Err(Unadmitted::UnknownPalette { concept: 0x0306 }),
        );
        assert_eq!(
            chain_step_predicate(ro_classid, c[0]),
            Err(Unadmitted::UnknownPalette { concept: 0x0306 }),
        );
        // Refused before any step is read: an empty chain under an unknown
        // concept is refused too.
        assert_eq!(
            validate_chain(ro_classid, &[]),
            Err(Unadmitted::UnknownPalette { concept: 0x0306 }),
        );
    }

    #[test]
    fn only_the_concept_half_routes() {
        // G is the canon-high concept half. The app prefix (low half) selects a
        // render skin and must not change which palette admits a chain.
        let mut rng = Lcg(0x0333_1000_0005_0001);
        let c = chain(&mut rng, 3);
        for classid in [DISMECH_CLASSID, 0x0333_1000, 0x0333_0005, 0x0333_FFFF] {
            assert_eq!(validate_chain(classid, &c), Ok(()), "{classid:#010x}");
        }
        // Silence twin: the same low half under another concept routes nowhere.
        assert!(validate_chain(0x0334_1000, &c).is_err());
    }

    #[test]
    #[allow(deprecated)]
    fn the_old_module_paths_still_name_the_same_items() {
        // The rename keeps `dismech_replay` / `dismech_counterfactual` as
        // deprecated aliases. The replay items are the SAME functions; the old
        // DisMech-bound admission functions answer exactly as admission under
        // the DisMech classid does.
        let old_seq: fn(u64, usize) -> Option<u64> = crate::dismech_replay::next_base_seq;
        let new_seq: fn(u64, usize) -> Option<u64> = crate::chain_replay::next_base_seq;
        assert!(core::ptr::fn_addr_eq(old_seq, new_seq));
        assert_eq!(
            crate::dismech_counterfactual::DEFAULT_FREQUENCY_BAR,
            crate::chain_counterfactual::DEFAULT_FREQUENCY_BAR,
        );
        let mut rng = Lcg(0xA11A_5000_0000_0001);
        let mut c = chain(&mut rng, 4);
        assert_eq!(crate::dismech_replay::validate_chain(&c), Ok(()));
        c[1].0 = 0xA3;
        assert_eq!(
            crate::dismech_replay::validate_chain(&c),
            Err(UnmintedOrdinal {
                step: 1,
                ordinal: 0xA3
            }),
        );
        for b in [0x90u8, 0xA2, 0xA3, 0x8F] {
            assert_eq!(
                crate::dismech_replay::chain_step_predicate((b, c[0].1)),
                chain_step_predicate(DISMECH_CLASSID, (b, c[0].1)).ok(),
            );
        }
    }
}
