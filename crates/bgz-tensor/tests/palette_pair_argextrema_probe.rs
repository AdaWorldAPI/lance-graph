//! Palette256 points are addresses. Pair-LUT codes are signed relational scores.
use bgz_tensor::{FamilyGamma, FisherZTable};

fn fixture() -> FisherZTable {
    let mut entries = vec![0i8; 16];
    // Anchor 0: candidate 1 strong, 2 neutral, 3 negative.
    entries[1] = 90;
    entries[2] = 0;
    entries[3] = -90;
    FisherZTable {
        entries,
        k: 4,
        gamma: FamilyGamma { z_min: -2.0, z_range: 4.0 },
    }
}

fn argextrema(table: &FisherZTable, anchor: u8, candidates: &[u8]) -> Option<(u8, u8)> {
    let (&first, rest) = candidates.split_first()?;
    let mut biggest = first;
    let mut smallest = first;
    for &candidate in rest {
        if table.lookup_i8(anchor, candidate) > table.lookup_i8(anchor, biggest) {
            biggest = candidate;
        }
        if table.lookup_i8(anchor, candidate) < table.lookup_i8(anchor, smallest) {
            smallest = candidate;
        }
    }
    Some((biggest, smallest))
}

#[test]
fn extrema_are_invariant_under_a_single_family_monotonic_decode() {
    let table = fixture();
    assert_eq!(argextrema(&table, 0, &[3, 2, 1]), Some((1, 3)));
    let max = table.lookup_f32(0, 1);
    let neutral = table.lookup_f32(0, 2);
    let min = table.lookup_f32(0, 3);
    assert!(max > neutral && neutral > min);
    let original = table.entries.clone();
    for _ in 0..64 {
        assert_eq!(argextrema(&table, 0, &[1, 2, 3]), Some((1, 3)));
    }
    assert_eq!(table.entries, original);
}

#[test]
fn xor_sign_bit_indexing_requires_u8_before_usize() {
    for byte in 0..=255u8 {
        let signed = byte as i8;
        let index = ((signed as u8) ^ 0x80) as usize;
        assert_eq!(index, (byte ^ 0x80) as usize);
    }
    assert_eq!(((0i8 as u8) ^ 0x80) as usize, 128);
    assert_eq!(((127i8 as u8) ^ 0x80) as usize, 255);
}

fn unique_winner(best: (f32, f32), rival: (f32, f32)) -> bool {
    best.0 > rival.1
}

#[test]
fn overlapping_error_intervals_refuse_to_announce_an_argmax() {
    assert!(unique_winner((0.80, 0.90), (0.10, 0.20)));
    assert!(!unique_winner((0.49, 0.55), (0.52, 0.57)));
    assert!(!unique_winner((0.60, 0.70), (0.50, 0.60)));
}