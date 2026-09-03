// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! D-POP-2 follow-up — the producer reaches its shipped consumers.
//!
//! `witness_fabric::elect_and_bind` shipped the PRODUCER: it elects a focal
//! row's social peers (Quorum / Contradiction) from a window of real content
//! loci and binds the election back into the row's own register. This file
//! proves the elected loci are actually READ by the two shipped consumers the
//! survey's §4 "contradiction-driven revision" molecule sits on:
//!
//! 1. **The recipe ladder** (`recipe_loci::reachable`) gates the
//!    consensus/revision recipes on Quorum / Contradiction being BOUND — so
//!    before the producer runs, a row whose content loci (SMeaning, Kausal)
//!    are bound but whose social loci are not cannot reach those recipes; once
//!    the producer runs, exactly the recipes whose OTHER required loci are
//!    already satisfied become newly reachable.
//! 2. **`SubstrateView::project`** — the logical markers the 34 kernels
//!    consume (`confidence`, `dissonance`, `free_energy`) move from
//!    ungrounded/zero to grounded/non-zero once the producer has bound the
//!    social loci.
//!
//! Each claim is two-sided: a recipe that also needs a still-unbound locus
//! (e.g. `MeaningLevel`, `PMeaning`/`OMeaning`) stays unreachable even after
//! the producer runs — the gate is the loci, not a blanket unlock.

use lance_graph_contract::awareness_facet::SpoFacet;
use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::qualia::QualiaI4_16D;
use lance_graph_contract::recipe_loci::{is_grounded, reachable, required_loci};
use lance_graph_contract::recipe_substrate::SubstrateView;
use lance_graph_contract::witness_fabric::{elect_and_bind, is_opinion, WitnessLens};

/// Same semantics as the private helper in `witness_fabric`'s own tests: rows
/// for positions `0..=max_pos`, blank rows are a default-class/default-basin
/// `NodeRow` with a zeroed value slab, and each named register is written
/// through [`WitnessLens::write_register`] — never a hand-poked byte offset.
fn rows_from(regs: &[(usize, CausalWitnessFacet)]) -> Vec<NodeRow> {
    let max_pos = regs.iter().map(|&(p, _)| p).max().unwrap_or(0);
    let mut rows: Vec<NodeRow> = (0..=max_pos)
        .map(|_| NodeRow {
            key: NodeGuid::local(1),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect();
    for &(pos, facet) in regs {
        WitnessLens::write_register(&mut rows[pos], &facet);
    }
    rows
}

/// The shared fixture: a focal row at position 5 whose content loci
/// (`SMeaning`, `Kausal`) are bound but whose social loci are not; an
/// agreeing peer at 4 that converges on BOTH content loci (same absolute
/// events: `SMeaning` at 7, `Kausal` at 2); and a dissenting peer at 6 that
/// agrees on `SMeaning` (event 7) but points its `Kausal` cause at a
/// different event (8 vs the focal's 2) — a preserved dissent, not mere
/// unrelatedness.
fn fixture() -> (CausalWitnessFacet, CausalWitnessFacet, CausalWitnessFacet) {
    let focal = CausalWitnessFacet::ZERO
        .with(Locus::SMeaning, 2) // 5 + 2 = event 7
        .with(Locus::Kausal, -3); // 5 - 3 = event 2
    let agree = CausalWitnessFacet::ZERO
        .with(Locus::SMeaning, 3) // 4 + 3 = event 7 (agrees)
        .with(Locus::Kausal, -2); // 4 - 2 = event 2 (agrees)
    let dissent = CausalWitnessFacet::ZERO
        .with(Locus::SMeaning, 1) // 6 + 1 = event 7 (agrees)
        .with(Locus::Kausal, 2); // 6 + 2 = event 8 (conflicts: focal's is 2)
    (focal, agree, dissent)
}

/// Run the producer over the fixture and return the focal row's ELECTED
/// register, re-read through the lens (never the input `focal` value — the
/// row's own bytes are the projection).
fn run_producer() -> (
    CausalWitnessFacet,
    lance_graph_contract::witness_fabric::ElectionReport,
) {
    let (focal, agree, dissent) = fixture();
    let mut rows = rows_from(&[(5, focal), (4, agree), (6, dissent)]);
    let report = elect_and_bind(&mut rows, |_| true);
    let f5 = *WitnessLens::new(&rows).at(5).unwrap();
    (f5, report)
}

#[test]
fn the_producer_makes_consensus_and_revision_recipes_reachable() {
    let (focal, _agree, _dissent) = fixture();
    let (f5, _report) = run_producer();

    // Anti-vacuity on the election itself: if the election changed, every
    // downstream assertion below would be measuring something else.
    assert_eq!(
        f5.quorum(),
        -1,
        "quorum must elect the agreeing peer at offset -1 (position 4)"
    );
    assert_eq!(
        f5.contradiction(),
        1,
        "contradiction must elect the dissenting peer at offset +1 (position 6)"
    );

    let before: Vec<u8> = reachable(&focal);
    let after: Vec<u8> = reachable(&f5);

    // `newly` computed from the required_loci TABLE itself, not by hand: a
    // recipe becomes newly reachable by the producer iff every locus it
    // requires is drawn from {SMeaning, Kausal, Quorum, Contradiction} (the
    // four loci ever bound across `before`/`after` in this fixture) AND it is
    // not already reachable pre-producer (i.e. it is not a subset of the
    // content-only loci {SMeaning, Kausal}).
    const CONTENT_AND_SOCIAL: [Locus; 4] = [
        Locus::SMeaning,
        Locus::Kausal,
        Locus::Quorum,
        Locus::Contradiction,
    ];
    const CONTENT_ONLY: [Locus; 2] = [Locus::SMeaning, Locus::Kausal];
    let newly: Vec<u8> = (1..=34u8)
        .filter(|&id| {
            let req = required_loci(id);
            let subset_of_content_and_social = req.iter().all(|l| CONTENT_AND_SOCIAL.contains(l));
            let subset_of_content_only = req.iter().all(|l| CONTENT_ONLY.contains(l));
            subset_of_content_and_social && !subset_of_content_only
        })
        .collect();
    // The table and this derived list must agree — if the table changes,
    // this fails loudly instead of silently drifting.
    assert_eq!(
        newly,
        vec![3, 7, 11, 17, 20, 27, 30],
        "the required_loci table's newly-reachable set drifted from the pinned list"
    );

    for &id in &newly {
        assert!(
            !before.contains(&id),
            "recipe {id} must NOT be reachable before the producer runs"
        );
        assert!(
            after.contains(&id),
            "recipe {id} must become reachable after the producer runs"
        );
    }

    // `after` as a set == `before` ∪ `newly` — nothing else appeared.
    use std::collections::HashSet;
    let after_set: HashSet<u8> = after.iter().copied().collect();
    let expected_set: HashSet<u8> = before
        .iter()
        .copied()
        .chain(newly.iter().copied())
        .collect();
    assert_eq!(
        after_set, expected_set,
        "after must be exactly before ∪ newly — the producer must not unlock anything else"
    );

    // Two-sided: a recipe needing a STILL-unbound locus stays unreachable
    // even after the producer runs — the gate is the loci, not a blanket
    // unlock. #21 needs MeaningLevel (unbound); #31 needs PMeaning/OMeaning
    // (unbound) in addition to Kausal/Contradiction/SMeaning.
    assert!(!before.contains(&21) && !after.contains(&21));
    assert!(!before.contains(&31) && !after.contains(&31));
    assert!(
        !is_grounded(&f5, 21),
        "#21 also needs MeaningLevel, still unbound"
    );
    assert!(
        !is_grounded(&f5, 31),
        "#31 also needs PMeaning/OMeaning, still unbound"
    );

    // Anti-vacuity on `before`/`after` sizing.
    //
    // NOTE (spec-ambiguity resolution): the required_loci table (verified by
    // exhaustive enumeration of all 34 entries) has NO recipe whose required
    // set is a subset of exactly {SMeaning, Kausal} — every entry that reads
    // either locus also reads at least one further, still-unbound locus (e.g.
    // #4 additionally needs OMeaning, #12 additionally needs Temporal, #29
    // additionally needs PMeaning). So for THIS fixture's focal (only
    // SMeaning + Kausal bound), `before` is provably EMPTY rather than
    // "non-empty via some unspecified recipe" — asserting non-emptiness here
    // would assert something the table does not support. The genuinely
    // falsifiable, non-vacuous claim this fixture supports is the exact
    // count relationship below, which still fails loudly if the producer
    // either fails to unlock the 7 newly-reachable recipes or unlocks
    // anything beyond them.
    assert!(
        before.is_empty(),
        "no recipe in required_loci requires a subset of exactly {{SMeaning, Kausal}} — \
         verified against the source table; a non-empty `before` here would mean the \
         table changed and this assumption needs re-deriving"
    );
    assert_eq!(
        after.len(),
        before.len() + 7,
        "after must gain exactly the 7 newly-reachable recipes"
    );
}

#[test]
fn the_producer_grounds_the_substrate_view_markers() {
    let (focal, _agree, _dissent) = fixture();
    let (f5, report) = run_producer();

    // Anti-vacuity on the election, mirrored from the first test (each test
    // must independently prove it is measuring the intended election).
    assert_eq!(f5.quorum(), -1, "quorum must elect the agreeing peer");
    assert_eq!(
        f5.contradiction(),
        1,
        "contradiction must elect the dissenting peer"
    );

    let before = SubstrateView::new(SpoFacet::default(), focal, QualiaI4_16D::ZERO);
    let after = SubstrateView::new(SpoFacet::default(), f5, QualiaI4_16D::ZERO);

    // Anti-vacuity: no SPO tenant is present, so `confidence` can ONLY be
    // grounded through the social loci this producer binds — not through SPO
    // agreement.
    assert!(
        !before.spo_present(),
        "the fixture carries no SPO tenant — confidence must be grounded socially or not at all"
    );

    let b = before.project();
    let a = after.project();

    // confidence: ungrounded (NaN) before any social locus is bound —
    // `SubstrateView::project`'s `conf_grounded` needs SPO present OR
    // Quorum/Contradiction bound, and none of those hold pre-producer.
    assert!(
        b.confidence.is_nan(),
        "confidence must be NaN before the producer binds a social locus"
    );
    assert!(
        a.confidence.is_finite(),
        "confidence must be grounded once Quorum/Contradiction are bound"
    );

    // dissonance: `logical_dissonance` is 0.0 while the contradiction edge is
    // unbound; once contradiction (+1) and quorum (-1) are both bound and
    // point at different offsets, it is `|quorum - contradiction| / 15`.
    assert_eq!(
        b.dissonance, 0.0,
        "dissonance must be exactly 0.0 while Contradiction is unbound"
    );
    let expected_dissonance = 2.0_f32 / 15.0; // |(-1) - 1| / 15, per logical_dissonance
    assert!(
        (a.dissonance - expected_dissonance).abs() < 1e-6,
        "dissonance must equal |quorum - contradiction| / 15 = {expected_dissonance} (logical_dissonance); got {}",
        a.dissonance
    );

    // free_energy (surprise): `logical_surprise` = 0.5 + contra − 0.4·situated,
    // contra = 0.4 iff Contradiction is bound, situated = bound_count / 16.
    // Before: bound_count = 2 (SMeaning, Kausal) → situated = 2/16, contra = 0.
    let expected_before_fe = 0.5_f32 - 0.4 * (2.0 / 16.0);
    // After: bound_count = 4 (SMeaning, Kausal, Quorum, Contradiction) →
    // situated = 4/16, contra = 0.4.
    let expected_after_fe = 0.5_f32 + 0.4 - 0.4 * (4.0 / 16.0);
    assert!(
        (b.free_energy - expected_before_fe).abs() < 1e-6,
        "before free_energy must equal logical_surprise's formula ({expected_before_fe}); got {}",
        b.free_energy
    );
    assert!(
        (a.free_energy - expected_after_fe).abs() < 1e-6,
        "after free_energy must equal logical_surprise's formula ({expected_after_fe}); got {}",
        a.free_energy
    );
    assert!(
        a.free_energy > b.free_energy,
        "the bound contradiction must raise surprise"
    );

    // opinion: a row whose Contradiction locus survives every revision in
    // its history is an opinion; an empty/never-bound history is not.
    assert!(
        is_opinion(&[f5]),
        "f5's Contradiction locus is bound in its one-revision history → opinion"
    );
    assert!(
        !is_opinion(&[focal]),
        "the pre-run focal has no bound Contradiction → not an opinion"
    );

    // The producer actually bound at least one contradiction this run.
    assert!(
        report.contradiction_bound >= 1,
        "elect_and_bind must have bound at least one contradiction across the fixture"
    );
}
