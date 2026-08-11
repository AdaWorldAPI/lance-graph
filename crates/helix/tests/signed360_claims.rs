//! Tests for the three `Signed360` claims that `.claude/knowledge/`
//! documents but the crate's own suite did not exercise.
//!
//! Written 2026-08-11 after a session landed all three in a knowledge doc on the
//! strength of DOC COMMENTS alone. Per `CLAUDE.md`'s falsifiability rule — *"a
//! doc-comment claim is not a behaviour; a test must exercise the claim or the
//! claim must be labelled claimed, unverified"* — they are exercised here.
//!
//! NOTE: `helix` is excluded from the root workspace and appears in no CI
//! workflow, so these run only when invoked by hand in this crate.
use helix::placement::Sign;
use helix::residue::{ResidueEncoder, Signed360};

/// `azimuth` spans the FULL circle, not merely "varies".
///
/// The pre-existing `signed360_azimuth_varies_with_n` asserts only that two
/// consecutive `n` differ — which golden-angle stepping makes true of any sane
/// implementation, so it cannot detect a truncated field. Measured here over the
/// whole domain: min 0, max 65535, all 256 coarse buckets occupied.
///
/// Falsifier: an implementation packing the angle into the low 10 bits (or
/// scaling to anything short of the full u16) fails on `max` and on coverage.
#[test]
fn azimuth_spans_the_full_circle_not_merely_varies() {
    let enc = ResidueEncoder::new(65_536);
    let (mut lo, mut hi) = (u16::MAX, 0u16);
    let mut buckets = [false; 256];
    for n in 0..65_536usize {
        let a = enc.encode_signed(7, n, Sign::Pos).azimuth;
        lo = lo.min(a);
        hi = hi.max(a);
        buckets[(a >> 8) as usize] = true;
    }
    let covered = buckets.iter().filter(|b| **b).count();
    assert_eq!(lo, 0, "azimuth must reach the bottom of the u16 range");
    assert_eq!(hi, u16::MAX, "azimuth must reach the top of the u16 range");
    assert_eq!(covered, 256, "every 1/256 arc of the circle must be reachable");
}

/// The two hemispheres occupy their EXACT halves of the `polar` byte.
///
/// Broader than the rim-only `signed360_neg_sign_survives_near_rim_at_high_total`
/// regression (codex P2 #498): swept over the whole domain, `Pos` fills
/// `[128, 255]` and `Neg` fills `[0, 127]` with no overlap and no gap at the ends.
///
/// Falsifier: the #498 centred-at-128 round puts a vanishing negative lift at
/// polar 128, which lands in the `Pos` partition — `neg_max` would reach 128.
#[test]
fn polar_partitions_are_exactly_the_two_halves() {
    let enc = ResidueEncoder::new(65_536);
    let (mut pos_lo, mut pos_hi) = (255u8, 0u8);
    let (mut neg_lo, mut neg_hi) = (255u8, 0u8);
    for n in 0..65_536usize {
        let p = enc.encode_signed(7, n, Sign::Pos).polar;
        let q = enc.encode_signed(7, n, Sign::Neg).polar;
        pos_lo = pos_lo.min(p);
        pos_hi = pos_hi.max(p);
        neg_lo = neg_lo.min(q);
        neg_hi = neg_hi.max(q);
    }
    assert_eq!((pos_lo, pos_hi), (128, 255), "Pos must fill [128,255] exactly");
    assert_eq!((neg_lo, neg_hi), (0, 127), "Neg must fill [0,127] exactly");
}

/// ⚠ CHARACTERIZES A KNOWN DEFECT — pinned so a fix must break it deliberately.
///
/// An all-zero, never-written 6-byte lane decodes as a DEFINITE `Sign::Neg`
/// direction: `sign()` partitions on `polar >= 128`, and 0 < 128. So "unwritten"
/// and "genuinely lower-hemisphere" are indistinguishable — while the sibling
/// lanes `Tekamolo` / `CausalWitness` define all-zero as *unaddressed*, and the
/// CANON zero-fallback ladder reads a zero tier as "not consulted".
///
/// Harmless today only because the lane has zero writers and zero decoders; it
/// becomes a live hazard the moment one exists. Filed, not fixed.
///
/// When the defect IS fixed (e.g. `sign()` returning `Option<Sign>`, or a
/// reserved sentinel), this test MUST fail and be re-pinned deliberately —
/// never silently edited.
#[test]
fn dormant_all_zero_lane_decodes_as_a_definite_sign_known_defect() {
    let dormant = Signed360::from_bytes([0u8; 6]);
    assert_eq!(
        dormant.sign(),
        Sign::Neg,
        "DEFECT pinned: an unwritten lane reports a definite hemisphere"
    );
    assert_eq!(dormant.polar, 0);
    // The complement is unambiguous, which is what makes the zero case the odd one.
    assert_eq!(Signed360::from_bytes([0xFFu8; 6]).sign(), Sign::Pos);
}
