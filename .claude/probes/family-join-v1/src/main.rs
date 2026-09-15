//! ISS-FAMILY-IS-FOUR-WIDTHS-TWO-AT-OPPOSITE-ENDS — the falsifier.
//!
//! The issue claims: `family` denotes four different things in this tree, and the
//! two that share a width sit at OPPOSITE ENDS of the key — `CascadeKey::family`
//! IS HEEL (the root-most tier), while `NodeGuid`'s v2 `family` is the
//! second-finest field, after `leaf`. If that is real, then "the shared prefix by
//! family" is not one quantity, and a join computed under one naming is not the
//! join computed under the other.
//!
//! This links BOTH REAL TYPES — `lance_graph_contract::NodeGuid` (with
//! `guid-v2-tail`) and `perturbation_sim::CascadeKey` — rather than
//! reimplementing either. A probe that rebuilds the structure it is testing
//! tests nothing about the producer.
//!
//! Anti-vacuity, per the issue's own requirement: every pair below must differ
//! somewhere in bytes 4..14, or both sides return the same trivial answer and
//! the comparison proves nothing. Asserted, not assumed.

use lance_graph_contract::canonical_node::NodeGuid;
use perturbation_sim::CascadeKey;

/// The v2 tail read as a 3-tier cascade: `(leaf, family_v2, identity_v2)`.
/// This is the naive analog of `CascadeKey::shared_prefix_tiers` a caller would
/// write if they took "family" to mean the v2 tail's family.
fn v2_tail_tiers(a: &NodeGuid, b: &NodeGuid) -> u8 {
    if a.leaf() != b.leaf() {
        0
    } else if a.family_v2() != b.family_v2() {
        1
    } else if a.identity_v2() != b.identity_v2() {
        2
    } else {
        3
    }
}

/// The same GUID's HHTL triple, handed to the REAL `CascadeKey` — whose own
/// field names are `family`/`leaf`/`identity` but whose documented meanings are
/// HEEL/HIP/TWIG.
fn cascade_of(g: &NodeGuid) -> CascadeKey {
    CascadeKey { family: g.heel(), leaf: g.hip(), identity: g.twig() }
}

/// Root-first recomposition from DECODED field values — the only correct input
/// to a CLZ join (see E-THE-SEMIRING-IS-FREE-...-JOIN-IS-THE-SAME-XOR-1).
fn recompose(g: &NodeGuid) -> u128 {
    ((g.classid() as u128) << 96)
        | ((g.heel() as u128) << 80)
        | ((g.hip() as u128) << 64)
        | ((g.twig() as u128) << 48)
        | ((g.leaf() as u128) << 32)
        | ((g.family_v2() as u128) << 16)
        | (g.identity_v2() as u128)
}

/// Nibble level of first divergence, 0..=32. 32 == identical.
fn nibble_level(x: u128) -> u32 {
    x.leading_zeros() / 4
}

fn main() {
    // classid held constant except in the byte-order arm, so the tail comparisons
    // are not decided by the prefix.
    const C: u32 = 0x0701_1000;

    // ---- the two hazard pairs -------------------------------------------------
    // P1: IDENTICAL HHTL (heel/hip/twig), DIFFERENT v2-tail family.
    let p1a = NodeGuid::new_v2(C, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);
    let p1b = NodeGuid::new_v2(C, 0x1111, 0x2222, 0x3333, 0x4444, 0xAAAA, 0x6666);

    // P2: DIFFERENT HEEL, IDENTICAL v2 tail.
    let p2a = NodeGuid::new_v2(C, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);
    let p2b = NodeGuid::new_v2(C, 0x9999, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);

    // ---- controls -------------------------------------------------------------
    // P3: identical keys — both namings must say 3.
    let p3a = NodeGuid::new_v2(C, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);
    let p3b = p3a;

    // P4: differ in BOTH the coarse tier and the tail — both namings must say 0.
    let p4a = NodeGuid::new_v2(C, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);
    let p4b = NodeGuid::new_v2(C, 0x9999, 0x2222, 0x3333, 0xBBBB, 0x5555, 0x6666);

    // Each row carries its EXACT expected (cascade, v2tail). An aggregate or
    // relative assertion would pass after a regression that changes the very
    // mechanism this probe reports — which is the defect this probe's own
    // finding warns about, so it must not commit it. Review finding on 43dbfde.
    let pairs: [(&str, &NodeGuid, &NodeGuid, u8, u8); 4] = [
        ("P1 same HHTL, different v2 family", &p1a, &p1b, 3, 1),
        ("P2 different HEEL, same v2 tail  ", &p2a, &p2b, 0, 3),
        ("P3 identical (control)           ", &p3a, &p3b, 3, 3),
        ("P4 differ in both (control)      ", &p4a, &p4b, 0, 0),
    ];

    println!("== ISS-FAMILY falsifier: does \"shared prefix by family\" mean one thing? ==\n");
    println!("{:<34} {:>8} {:>8}   {}", "pair", "cascade", "v2tail", "verdict");
    println!("{}", "-".repeat(72));

    let mut disagreements = 0;
    for (name, a, b, want_cascade, want_v2tail) in pairs {
        // anti-vacuity: the pair must actually differ in bytes 4..14, else the
        // comparison is trivially satisfied and proves nothing.
        let differs_in_key = a.as_bytes()[4..14] != b.as_bytes()[4..14];
        let is_control_identical = name.starts_with("P3");
        assert!(
            differs_in_key || is_control_identical,
            "VACUOUS FIXTURE: {name} does not differ in bytes 4..14"
        );

        let cascade = cascade_of(a).shared_prefix_tiers(cascade_of(b));
        let v2tail = v2_tail_tiers(a, b);
        assert_eq!(
            (cascade, v2tail),
            (want_cascade, want_v2tail),
            "{name}: exact (cascade, v2tail) changed"
        );
        let agree = cascade == v2tail;
        if !agree {
            disagreements += 1;
        }
        println!(
            "{:<34} {:>8} {:>8}   {}",
            name,
            cascade,
            v2tail,
            if agree { "agree" } else { "** DISAGREE **" }
        );
    }

    println!("\n{} of 4 pairs disagree.", disagreements);
    // Exactness is asserted per pair above; this only records the count so a
    // reader sees it. `disagreements >= 2` alone was the original assertion and
    // it was too weak: P1 could drift to (2,1) and P2 to (1,3) and still pass.
    assert_eq!(disagreements, 2, "exactly P1 and P2 must disagree");

    // ---- the byte-order trap, from the sixth arc ------------------------------
    println!("\n== byte-order trap: CLZ over the raw 16 bytes vs recomposed ==\n");
    let t1a = NodeGuid::new_v2(0xA000_0000, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);
    let t1b = NodeGuid::new_v2(0x2000_0000, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);
    let t2a = NodeGuid::new_v2(C, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x0001);
    let t2b = NodeGuid::new_v2(C, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x0002);

    println!(
        "{:<38} {:>10} {:>10} {:>12}",
        "pair", "from_le", "from_be", "recomposed"
    );
    println!("{}", "-".repeat(74));
    for (name, a, b) in [
        ("T1 differs in classid's TOP nibble", &t1a, &t1b),
        ("T2 differs in identity's LAST nibble", &t2a, &t2b),
    ] {
        let le = nibble_level(
            u128::from_le_bytes(*a.as_bytes()) ^ u128::from_le_bytes(*b.as_bytes()),
        );
        let be = nibble_level(
            u128::from_be_bytes(*a.as_bytes()) ^ u128::from_be_bytes(*b.as_bytes()),
        );
        let rc = nibble_level(recompose(a) ^ recompose(b));
        println!("{name:<38} {le:>10} {be:>10} {rc:>12}");
    }

    let le_t1 = nibble_level(
        u128::from_le_bytes(*t1a.as_bytes()) ^ u128::from_le_bytes(*t1b.as_bytes()),
    );
    let le_t2 = nibble_level(
        u128::from_le_bytes(*t2a.as_bytes()) ^ u128::from_le_bytes(*t2b.as_bytes()),
    );
    let rc_t1 = nibble_level(recompose(&t1a) ^ recompose(&t1b));
    let rc_t2 = nibble_level(recompose(&t2a) ^ recompose(&t2b));

    println!(
        "\nrecomposed: root-nibble diff -> level {rc_t1}, leaf-nibble diff -> level {rc_t2} (expected 0 and 31)"
    );
    println!("from_le:    root-nibble diff -> level {le_t1}, leaf-nibble diff -> level {le_t2} (INVERTED if le_t1 > le_t2)");
    let be_t1 = nibble_level(
        u128::from_be_bytes(*t1a.as_bytes()) ^ u128::from_be_bytes(*t1b.as_bytes()),
    );
    let be_t2 = nibble_level(
        u128::from_be_bytes(*t2a.as_bytes()) ^ u128::from_be_bytes(*t2b.as_bytes()),
    );
    // EXACT LEVELS AT BOTH ENDS, for all three readings. The original assertion
    // here was `le_t1 > le_t2` — a RELATION — which is precisely what this
    // probe's own finding says is insufficient: "a test that checks the
    // ORDERING of two prefix lengths will pass the from_be implementation."
    // The probe stating that rule asserted a relation. Review finding, 43dbfde.
    assert_eq!((rc_t1, rc_t2), (0, 31), "recomposed levels changed");
    assert_eq!((le_t1, le_t2), (24, 3), "from_le levels changed");
    assert_eq!((be_t1, be_t2), (6, 29), "from_be levels changed");
    // And the discriminating property: from_be is MONOTONE (so an ordering
    // check passes it) while being wrong at both ends.
    assert!(be_t1 < be_t2, "from_be is monotone — this is why ordering is not enough");
    assert!(le_t1 > le_t2, "from_le inverts the two ends");

    // ── THE ARM THIS PROBE WAS MISSING (2026-09-15, operator correction) ──
    // `ISS-NODEGUID-HAS-NO-JOIN-SURFACE` was filed on a grep of canonical_node.rs
    // ALONE. The canonical join lives one module over: `hhtl::NiblePath`
    // (`from_guid_prefix_v2` + `common_prefix_depth`), and it is CALLED in
    // production — mailbox_scan.rs:149 (per row, in a scan), :263
    // (DistanceMeans::PrefixDepth), soa_graph.rs:408 (nearest_anchor). This arm
    // runs it on the same fixtures. It reads a DIFFERENT prefix than `recompose`:
    // the 16-nibble HEEL/HIP/TWIG/leaf path (the Abstammung axis) — NOT classid,
    // NOT the family/identity tail. So T1 (classid differs) and T2 (identity
    // differs) are invisible to it BY DESIGN; P1/P2 are what exercise it.
    use lance_graph_contract::hhtl::NiblePath;
    let cpd = |a: &NodeGuid, b: &NodeGuid| -> u8 {
        NiblePath::from_guid_prefix_v2(a).common_prefix_depth(NiblePath::from_guid_prefix_v2(b))
    };
    println!(
        "\n{:<38} {:>8} {:>10}   {}",
        "pair", "cascade", "canonical", "(cascade 0..=3 tiers; canonical 0..=16 nibbles)"
    );
    println!("{}", "-".repeat(74));
    for (name, a, b, _, _) in &pairs {
        let c = cascade_of(a).shared_prefix_tiers(cascade_of(b));
        println!("{name:<38} {c:>8} {:>10}", cpd(a, b));
    }
    for (name, a, b) in [("T1 classid top nibble (OUTSIDE path)", &t1a, &t1b),
                         ("T2 identity last nibble (OUTSIDE path)", &t2a, &t2b)] {
        println!("{name:<38} {:>8} {:>10}", "-", cpd(a, b));
    }

    // Pinned from the run above — EXACT, both ends, per the rule this probe states.
    // canonical == 16 exactly where cascade == 3, and 0 exactly where cascade == 0:
    // the shipped join agrees with CascadeKey on every pair, at 16-nibble
    // resolution instead of 3 tiers. T1/T2 read 16 because classid and identity
    // sit OUTSIDE the HEEL/HIP/TWIG/leaf path — the "two head axes" design.
    let canon: Vec<(u8, u8)> = pairs
        .iter()
        .map(|(_, a, b, _, _)| (cascade_of(a).shared_prefix_tiers(cascade_of(b)), cpd(a, b)))
        .collect();
    assert_eq!(canon, vec![(3, 16), (0, 0), (3, 16), (0, 0)], "canonical join moved");
    assert_eq!((cpd(&t1a, &t1b), cpd(&t2a, &t2b)), (16, 16), "path must be blind to classid/identity");
    // and the resolution ratio is not a constant someone typed: 16 nibbles / 3 tiers.
    assert_eq!(NiblePath::from_guid_prefix_v2(&p3a).depth(), 16, "MAX_DEPTH changed");

    println!("\nVERDICT: the two namings are NOT interchangeable. Filed hazard CONFIRMED.");
}
