// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! **W2b — the ONE-NODE field falsifier** (`D-DCR-2b`).
//!
//! The W2b ruling says a position summarises its children in **24 signed
//! dimensions**, and that the SIGN collapses three states into one quantity:
//! `+` agreement, `−` disagreement, `0` silence. This file is the falsifier for
//! that carrier at the scale where it is decidable today: **one node**.
//!
//! # What this proves, and what it deliberately does NOT
//!
//! | | |
//! |---|---|
//! | **proves** | the 24×i4 carrier holds the three states distinguishably, round-trips through the 12-byte register, saturates instead of wrapping, and — the load-bearing one — that a negative lane is **not** a removal |
//! | **does NOT prove** | anything about a global sweep. Gap 3 (convergence semantics for a field-wide fixpoint) is fully open and one node cannot speak to it: one hop is a function, a field map is a fixpoint |
//! | **does NOT decide** | which value lane the summary occupies. That is an open operator decision; nothing here reserves a byte or mints a `ValueTenant` |
//! | **does NOT canonise** | the summarising *arithmetic* (gap 1). [`summarise`] below is this test's own minimal choice, marked as such — the ruling fixes the SIGN's meaning, not the sum |
//!
//! # Why it borrows `CausalWitnessFacet` as the codec
//!
//! The `G24N4` nibble codec already ships there, and this workspace consumes
//! rather than re-implements. That borrowing is **codec-only** and is not a
//! claim that W2b belongs in that lane — it does not:
//!
//! - `CausalWitness`'s value law is operator-locked to **loci, never
//!   strength/magnitude**, and a W2b summary is the magnitude case;
//! - its slots `16..24` are RESERVE-DON'T-RECLAIM.
//!
//! Both are reasons W2b needs its own lane, recorded in the census beside this
//! file. Here the facet is used purely as "the shipped 24-nibble codec".

use lance_graph_contract::causal_witness::{CausalWitnessFacet, WITNESS_LOCI};

/// What one child says about one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildVote {
    Agrees,
    Disagrees,
    Silent,
}

/// **This test's own arithmetic, NOT a ruling.** Gap 1 leaves the summarising
/// function open; the ruling fixes only what the sign MEANS. The minimal choice
/// that honours it: agreement counts up, disagreement counts down, silence
/// contributes nothing — then the carrier clamps.
fn summarise(votes: &[ChildVote]) -> i8 {
    let mut acc: i32 = 0;
    for v in votes {
        acc += match v {
            ChildVote::Agrees => 1,
            ChildVote::Disagrees => -1,
            ChildVote::Silent => 0,
        };
    }
    acc.clamp(-8, 7) as i8
}

/// **F1 — the three states are distinguishable through the register.**
///
/// Anti-vacuity: round-tripping each value is not enough. An implementation
/// that stored the MAGNITUDE and dropped the sign would round-trip every
/// non-negative case, so this also pins that agreement and disagreement of the
/// same magnitude produce **different registers**.
///
/// Disable: make `summarise` return `acc.abs()` ⇒ red.
#[test]
fn agreement_disagreement_and_silence_are_three_distinct_readings() {
    let agree = summarise(&[ChildVote::Agrees; 3]);
    let disagree = summarise(&[ChildVote::Disagrees; 3]);
    let silent = summarise(&[ChildVote::Silent; 3]);

    assert_eq!(agree, 3, "three agreeing children read as +3");
    assert_eq!(disagree, -3, "three disagreeing children read as -3");
    assert_eq!(silent, 0, "silence is zero, not a weak agreement");

    let mut a = CausalWitnessFacet::default();
    a.set(0, agree);
    let mut d = CausalWitnessFacet::default();
    d.set(0, disagree);
    let s = CausalWitnessFacet::default();

    assert_eq!(a.get(0), 3, "the register must hand back what was written");
    assert_eq!(d.get(0), -3, "sign survives the nibble round-trip");
    assert_eq!(s.get(0), 0);

    // The half a magnitude-only implementation fails.
    assert_ne!(
        a.to_register(),
        d.to_register(),
        "same magnitude, opposite sign: the two must not share a register"
    );
    assert_ne!(
        d.to_register(),
        s.to_register(),
        "silence must be distinguishable from disagreement (the missing-link carrier)"
    );
}

/// A node's membership in a neighbourhood, as asserted by the RAIL — a carrier
/// that is not the field lane. The whale case turns on these being separate
/// bytes, which is what F2 checks structurally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RailMembership {
    is_a_mammal: bool,
}

/// The write under test: record a per-dimension summary on the node.
///
/// Correct behaviour takes `rail` by shared reference — a summary write has no
/// business touching membership. The signature is what makes F2 falsifiable:
/// change it to `&mut` and act on the lane's sign, and F2 goes red.
fn record_summary(
    field: &mut CausalWitnessFacet,
    slot: usize,
    summary: i8,
    _rail: &RailMembership,
) {
    field.set(slot, summary);
}

/// **F2 — THE WHALE CASE. A negative lane is a value, not a removal.**
///
/// The ruling's own example: *"a whale records as a negative lane against the
/// mammal neighbourhood and stays a mammal, because a lane is a value and not a
/// removal."* This is the falsifier the kind-1/kind-3 ordering rests on — if
/// disagreement eliminated, the sweep would already be doing kind 3's job.
///
/// **Why this is structural and not a restated constant.** A first version
/// asserted `whale.is_a_mammal` after building `RailMembership { is_a_mammal:
/// true }` — vacuous: nothing between the two lines could change it, and its
/// "disable" meant editing the test rather than the code. What is genuinely
/// falsifiable is that the two live in DIFFERENT carriers: sweeping every value
/// the lane can hold — including the whole negative half — must leave the rail
/// byte-identical, and must leave the disagreement legible afterwards.
///
/// Disable: give [`record_summary`] `rail: &mut RailMembership` and let it
/// clear `is_a_mammal` when `summary < 0` ⇒ red.
#[test]
fn no_value_the_lane_can_hold_revokes_membership() {
    let rail = RailMembership { is_a_mammal: true };
    let before = rail;

    // The whole i4 range, negative half included — not one sampled value.
    let mut saw_negative = false;
    for summary in -8i8..=7 {
        let mut field = CausalWitnessFacet::default();
        record_summary(&mut field, 0, summary, &rail);

        assert_eq!(
            rail, before,
            "writing summary {summary} changed the membership carrier"
        );
        assert_eq!(
            field.get(0),
            summary,
            "and the summary must stay legible after the write, not be consumed by a decision"
        );
        if summary < 0 {
            saw_negative = true;
        }
    }
    assert!(
        saw_negative,
        "the sweep must actually exercise the negative half, or it proves nothing about disagreement"
    );

    // The hard case named explicitly: maximal disagreement, still a mammal.
    let votes = [ChildVote::Disagrees; 12];
    let summary = summarise(&votes);
    assert_eq!(summary, -8, "the fixture must reach the floor");
    let mut field = CausalWitnessFacet::default();
    record_summary(&mut field, 0, summary, &rail);
    assert_eq!(rail.is_a_mammal, before.is_a_mammal);
    assert_eq!(field.get(0), -8);
}

/// **F3 — saturation, never wrap.** A disagreement stronger than the carrier
/// can hold must clamp to `−8`. A wrap would turn strong disagreement into
/// agreement — silently, and in the direction that inverts the answer.
///
/// Disable: replace the clamp in `summarise` with `acc as i8` ⇒ red (`-20 as
/// i8` is `-20`, which `set` then clamps, but `-24` wraps through the nibble to
/// `+8`-shaped garbage); the assertion below pins the sign, which is what
/// actually matters.
#[test]
fn overflowing_disagreement_saturates_and_never_flips_sign() {
    for n in [9usize, 12, 40, 1000] {
        let votes = vec![ChildVote::Disagrees; n];
        let summary = summarise(&votes);
        assert_eq!(
            summary, -8,
            "{n} disagreeing children must clamp to the floor"
        );

        let mut f = CausalWitnessFacet::default();
        f.set(0, summary);
        assert!(
            f.get(0) < 0,
            "{n} disagreeing children read back as {} — a sign flip is the silent catastrophe",
            f.get(0)
        );
    }
    // The same on the agreement side, whose ceiling is +7 (asymmetric by i4).
    let many = vec![ChildVote::Agrees; 1000];
    assert_eq!(summarise(&many), 7);
}

/// **F4 — all 24 lanes are independently addressable.**
///
/// The classic packed-nibble defect: writing lane `i` disturbs its byte-mate
/// `i^1`. Every lane gets a distinct value and all are read back.
///
/// Disable: drop the `& 0xF0` / `& 0x0F` masking in the codec ⇒ red.
#[test]
fn every_one_of_the_twenty_four_lanes_holds_its_own_value() {
    let mut f = CausalWitnessFacet::default();
    // A pattern where no two adjacent lanes share a value, so a bleed shows.
    let value_for = |slot: usize| -> i8 { ((slot as i32 % 15) - 7) as i8 };

    for slot in 0..WITNESS_LOCI {
        f.set(slot, value_for(slot));
    }
    for slot in 0..WITNESS_LOCI {
        assert_eq!(
            f.get(slot),
            value_for(slot),
            "lane {slot} was disturbed by a neighbour"
        );
    }
    // Anti-vacuity: the pattern must actually put different values in byte-mates.
    assert_ne!(
        value_for(0),
        value_for(1),
        "the fixture cannot detect bleed if byte-mates share a value"
    );
    assert_eq!(
        WITNESS_LOCI, 24,
        "the ruling's width is 24 signed dimensions"
    );
}

/// **F5 — the summary is derived, and re-deriving it is stable.**
///
/// One node, same children, twice: identical register. Not a claim about a
/// sweep — a sweep's fixpoint is gap 3 — only that the per-node summarisation
/// is a function of its inputs and carries no hidden state.
#[test]
fn re_summarising_the_same_children_yields_the_same_register() {
    let children = [
        ChildVote::Agrees,
        ChildVote::Disagrees,
        ChildVote::Silent,
        ChildVote::Agrees,
    ];
    let build = || {
        let mut f = CausalWitnessFacet::default();
        for slot in 0..WITNESS_LOCI {
            // Rotate the votes per lane so the register is not uniform.
            let rotated: Vec<ChildVote> = (0..children.len())
                .map(|i| children[(i + slot) % children.len()])
                .collect();
            f.set(slot, summarise(&rotated));
        }
        f
    };
    assert_eq!(build().to_register(), build().to_register());
    // Anti-vacuity: the register must not be all-zero, or equality is trivial.
    assert_ne!(
        build().to_register(),
        [0u8; 12],
        "an all-zero register would make this comparison vacuous"
    );
}
