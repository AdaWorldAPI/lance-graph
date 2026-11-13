//! H9 to H12: the 256^3 address space and three relation families, each
//! against an enumeration of all 16,777,216 addresses.

use virtual_address_lab::cube::*;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn masks() -> (AxisMask, AxisMask, AxisMask) {
    (
        axis(|i| mix(u64::from(i)).is_multiple_of(3)),
        axis(|i| mix(1_000 + u64::from(i)).is_multiple_of(2)),
        axis(|i| i % 7 < 3),
    )
}

/// Every address of the cube, once.
fn enumerate(mut member: impl FnMut(u8, u8, u8) -> bool) -> u64 {
    let mut n = 0u64;
    for x in 0..=255u8 {
        for y in 0..=255u8 {
            for z in 0..=255u8 {
                n += u64::from(member(x, y, z));
            }
        }
    }
    n
}

/// H12. FAILS IF: two cells share an address or one does not decode back.
#[test]
fn the_address_is_reversible() {
    let mut seen = vec![false; 1 << 16];
    for x in 0..=255u8 {
        for y in 0..=255u8 {
            for z in [0u8, 1, 37, 128, 255] {
                let a = address(x, y, z).unwrap();
                assert_eq!(decode(a), Some((x, y, z)));
                assert_eq!(a >> 16, u32::from(z));
            }
            let low = address(x, y, 0).unwrap() as usize;
            assert!(!seen[low], "({x},{y}) collides");
            seen[low] = true;
        }
    }
    assert!(seen.iter().all(|&s| s));
    assert_eq!(decode(1 << 24), None);
}

/// H9. FAILS IF: the product of three popcounts is not the enumeration.
#[test]
fn a_factorizable_relation_is_three_popcounts() {
    let (a, b, c) = masks();
    let want = enumerate(|x, y, z| has(&a, x) && has(&b, y) && has(&c, z));
    assert_eq!(count_factorized(&a, &b, &c), want);
    assert!(want > 100_000 && want < 16_000_000);
    assert_eq!(count_factorized(&FULL, &FULL, &FULL), 1 << 24);
}

/// H10. FAILS IF: solving for `z` miscounts, unrestricted or under masks.
/// The control shows three axis projections do NOT count it.
#[test]
fn a_coupled_relation_is_counted_by_solving_for_z() {
    let coupled = |x: u8, y: u8, z: u8| x.wrapping_add(y).wrapping_add(z) == 0;
    assert_eq!(count_coupled(&FULL, &FULL, &FULL), 65_536);
    assert_eq!(enumerate(coupled), 65_536);

    let (a, b, c) = masks();
    let want = enumerate(|x, y, z| has(&a, x) && has(&b, y) && has(&c, z) && coupled(x, y, z));
    assert_eq!(count_coupled(&a, &b, &c), want);
    assert!(want > 1_000 && want < 65_536);
    // An asymmetric pair of masks, so a wrong reflection or rotation shows.
    let low = axis(|i| i < 10);
    let one = axis(|i| i == 250);
    let want = enumerate(|x, y, z| has(&low, x) && has(&one, y) && coupled(x, y, z));
    assert_eq!(want, 10);
    assert_eq!(count_coupled(&low, &one, &FULL), 10);
    assert_eq!(count_coupled(&low, &one, &axis(|z| z == 6)), 1); // x = 0, y = 250

    // The axis projections of the unrestricted relation are all full, so
    // their product is the whole cube (the #2055 refutation).
    assert_eq!(count_factorized(&FULL, &FULL, &FULL), 16_777_216);
}

/// H11. FAILS IF: the unstructured relation happens to factorise (the
/// control would be void), or the candidate walk miscounts.
#[test]
fn an_unstructured_relation_has_no_such_form() {
    let (a, b, c) = masks();
    let want = enumerate(|x, y, z| has(&a, x) && has(&b, y) && has(&c, z) && unstructured(x, y, z));
    let (count, tests) = count_unstructured(&a, &b, &c);
    assert_eq!(count, want);
    // One membership test per candidate: the walk is the candidate product.
    assert_eq!(tests, count_factorized(&a, &b, &c));
    // Its axis projections over-count: every axis value occurs in a member.
    let (mut px, mut py, mut pz) = ([0u64; 4], [0u64; 4], [0u64; 4]);
    for x in (0..=255u8).filter(|&x| has(&a, x)) {
        for y in (0..=255u8).filter(|&y| has(&b, y)) {
            for z in (0..=255u8).filter(|&z| has(&c, z)) {
                if unstructured(x, y, z) {
                    px[usize::from(x) / 64] |= 1 << (x % 64);
                    py[usize::from(y) / 64] |= 1 << (y % 64);
                    pz[usize::from(z) / 64] |= 1 << (z % 64);
                }
            }
        }
    }
    assert!(count_factorized(&px, &py, &pz) > 4 * want);
    assert!(want > 100_000);
}
