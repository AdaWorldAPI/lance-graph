//! X-C2-1 anti-vacuity gate + kill floor.
//!
//! Charter rule: the harness must FIRST reproduce the known in-tree false
//! accepts as positive controls — "if the harness cannot detect these four,
//! it is measuring nothing and no other result from it may be reported."
//!
//! Status of the four, measured against the tree as it actually builds:
//!  (a) `MerkleTree` metadata blind zone — LIVE ndarray code, reproduced
//!      here against the real implementation (and the zone is WIDER than
//!      C2 recorded: words 48..56, 64..96 AND 112..256 are hashed by no
//!      branch).
//!  (b) `firefly_frame::verify_ecc` 100%-accept — NOT reproducible against
//!      real code, for the strongest possible reason: the `wip` feature
//!      that contains it fails to compile (104 errors, measured
//!      2026-08-19), so the defective path is unreachable by construction.
//!      Its fault CLASS (a verifier that cannot reject) is covered by the
//!      harness self-control below.
//!  (c) XOR-fold permutation acceptance — same situation (`container_bs`
//!      is wip-gated); the fault class is proven directly on the fold
//!      algebra below.
//!  (d) unbound digest wrong-slot acceptance — LIVE ndarray code
//!      (`hpc::seal::Plane`), reproduced against the real implementation.

use rp_seal_t0_probe::*;

// ── (a) POSITIVE CONTROL against REAL code: the MerkleTree blind zone ────────
#[test]
fn control_a_real_merkle_tree_is_blind_to_unhashed_metadata_words() {
    use ndarray::hpc::merkle_tree::MerkleTree;
    let content: [u64; 256] = [7u64; 256];
    let meta_a: [u64; 256] = [1u64; 256];
    let mut meta_b = meta_a;
    // All three words sit outside every BRANCH_REGION: 48..56 gap, 64..96
    // gap, and the 112..256 tail C2 recorded.
    meta_b[50] = 0xDEAD_BEEF;
    meta_b[70] = 0xFEED_FACE;
    meta_b[200] = 0xBAD0_CAFE;
    let ta = MerkleTree::from_cogrecord(&meta_a, &[&content]);
    let tb = MerkleTree::from_cogrecord(&meta_b, &[&content]);
    assert_eq!(
        ta.hamming(&tb),
        0,
        "REAL false accept: corruption in unhashed metadata words is \
         invisible to the multi-scale syndrome",
    );
    // Discriminance (the silence half): a word INSIDE a hashed region moves
    // the tree — the control is not vacuous.
    let mut meta_c = meta_a;
    meta_c[5] = 0x1234_5678;
    let tc = MerkleTree::from_cogrecord(&meta_c, &[&content]);
    assert!(
        ta.hamming(&tc) > 0,
        "a hashed-region word must change the tree, or this control proves nothing",
    );
}

// ── (d) POSITIVE CONTROL against REAL code: unbound digest, wrong slot ───────
#[test]
fn control_d_real_plane_seal_accepts_wrong_slot_substitution_of_identical_content() {
    use ndarray::hpc::plane::Plane;
    use ndarray::hpc::seal::Seal;
    // Two planes at DIFFERENT logical slots, identical content: the digest
    // binds no locus, so slot B's plane verifies against slot A's stored
    // root — the wrong-slot substitution is invisible ("certain for
    // identical/default content", C2 §B1).
    let mut slot_a = Plane::new();
    let mut slot_b = Plane::new();
    slot_a.encounter("the same content in two places");
    slot_b.encounter("the same content in two places");
    let stored_a = slot_a.merkle();
    assert_eq!(
        slot_b.verify(&stored_a),
        Seal::Wisdom,
        "REAL false accept: the unbound digest cannot see a wrong-slot \
         substitution of identical content",
    );
    // Discriminance: different content is rejected.
    let mut other = Plane::new();
    other.encounter("entirely different content");
    assert_eq!(
        other.verify(&stored_a),
        Seal::Staunen,
        "content change must be rejected, or this control proves nothing",
    );
}

// ── (b)+(c) fault-class controls on the fold algebra (real paths are
//    wip-gated AND non-compiling — unreachable by construction) ─────────────
#[test]
fn control_bc_xor_fold_class_accepts_permutations_and_paired_flips() {
    // The 4-way XOR fold (firefly's compute_ecc shape): ecc[i%4] ^= word.
    fn fold(words: &[u64]) -> [u64; 4] {
        let mut e = [0u64; 4];
        for (i, w) in words.iter().enumerate() {
            e[i % 4] ^= w;
        }
        e
    }
    let data: Vec<u64> = (0..16).map(|i| 0xA5A5_0000 + i as u64 * 977).collect();
    let clean = fold(&data);

    // Paired flip in the same fold class (words 0 and 4): 32 corrupted bits,
    // fold unchanged — the syndrome CANNOT fire.
    let mut flipped = data.clone();
    flipped[0] ^= 0xFFFF_FFFF;
    flipped[4] ^= 0xFFFF_FFFF;
    assert_ne!(data, flipped, "the corruption is real");
    assert_eq!(
        fold(&flipped),
        clean,
        "fault class: a same-class paired flip is invisible to the fold",
    );

    // Permutation within a fold class (swap words 1 and 5): content order
    // changed, fold identical — XOR is commutative, order is unprotected.
    let mut permuted = data.clone();
    permuted.swap(1, 5);
    assert_ne!(data, permuted, "the permutation is real");
    assert_eq!(
        fold(&permuted),
        clean,
        "fault class: the fold accepts a word permutation",
    );

    // Discriminance: a lone single-word flip DOES move the fold.
    let mut lone = data.clone();
    lone[3] ^= 1;
    assert_ne!(fold(&lone), clean, "a lone flip must move the fold");
}

// ── The kill floor + the S1U/S6 contrast, at multiplicity 1 ──────────────────
#[test]
fn kill_floor_s6_has_zero_false_accepts_on_i1_i2_i3() {
    let s6 = S6Bound;
    for inj in [
        Injection::I1 { pos: 17 },
        Injection::I2 { a: 3, b: 250 },
        Injection::I3 { pos: 99, bit: 1234 },
    ] {
        let (_, _, fa, alarms) = run_one(&s6, 256, &inj);
        assert!(
            fa.is_empty(),
            "S6 kill floor: {inj:?} produced false accepts {fa:?}"
        );
        assert!(
            alarms.is_empty(),
            "S6 flagged clean chunks on {inj:?}: {alarms:?}"
        );
    }
}

#[test]
fn s1_unbound_false_accepts_wrong_slot_stale_and_duplicate_where_s6_rejects() {
    let s1 = S1Unbound;
    let s6 = S6Bound;
    let cases = [
        (Injection::I4 { src: 5, dst: 9 }, 9usize),
        (
            Injection::I5 {
                pos: 40,
                old_version: 1,
            },
            40,
        ),
        (Injection::I6 { src: 12, dst: 200 }, 200),
    ];
    for (inj, victim) in cases {
        // Anti-vacuity: the substituted content genuinely differs from what
        // the victim slot should hold.
        if let Injection::I4 { src, dst } | Injection::I6 { src, dst } = &inj {
            assert_ne!(
                chunk_content(*src as u64, 3),
                chunk_content(*dst as u64, 3),
                "fixture must substitute DIFFERENT content",
            );
        }
        let (_, _, fa1, al1) = run_one(&s1, 256, &inj);
        assert_eq!(
            fa1,
            std::iter::once(victim).collect(),
            "S1U (the shipped unbound shape) must FALSE-ACCEPT {inj:?}",
        );
        assert!(al1.is_empty());
        let (_, _, fa6, al6) = run_one(&s6, 256, &inj);
        assert!(
            fa6.is_empty(),
            "S6 must reject {inj:?}; false accepts: {fa6:?}",
        );
        assert!(al6.is_empty());
    }
}

#[test]
fn both_schemes_detect_silent_corruption_and_correlated_patterns() {
    for scheme in [&S1Unbound as &dyn Scheme, &S6Bound as &dyn Scheme] {
        for inj in [
            Injection::I3 { pos: 7, bit: 9 },
            Injection::I7 {
                start: 64,
                len: 16,
                bit: 5,
            },
            Injection::I8 { group: 64, bit: 3 },
            Injection::I9 {
                stride: 16,
                phase: 2,
                bit: 40,
            },
        ] {
            let (affected, flagged, fa, alarms) = run_one(scheme, 256, &inj);
            assert!(!affected.is_empty(), "vacuous fixture for {inj:?}");
            assert_eq!(
                flagged,
                affected,
                "{}: content corruption must be exactly localized on {inj:?}",
                scheme.name(),
            );
            assert!(fa.is_empty());
            assert!(alarms.is_empty());
        }
    }
}

// ── The null control (charter: a scheme that flags clean cycles is as
//    broken as one that accepts dirty ones). 10⁵ here; the example runs the
//    full 10⁶ and the report records it. ─────────────────────────────────────
#[test]
fn null_control_no_scheme_flags_clean_chunks() {
    assert_eq!(null_control(&S1Unbound, 100_000), 0);
    assert_eq!(null_control(&S6Bound, 100_000), 0);
}
