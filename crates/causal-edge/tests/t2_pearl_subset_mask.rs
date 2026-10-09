//! T2 (CE64 coresearch, 2026-10-08): bits 40..42 are a SUBSET mask over the
//! S/P/O planes, not an ordinal ladder. Reading the field as a number and
//! testing `>= k` ranks the masks wrongly.

use causal_edge::CausalMask;

/// The rung a mask's own documentation assigns it, parsed from
/// `pearl_level()`, or `None` for masks the ladder does not name.
fn rung(m: CausalMask) -> Option<u8> {
    let s = m.pearl_level();
    s.strip_prefix("Level ")
        .and_then(|r| r.chars().next())
        .and_then(|c| c.to_digit(10))
        .map(|d| d as u8)
}

#[test]
fn po_is_numerically_below_so_but_ranks_above_it() {
    let (po, so) = (CausalMask::PO as u8, CausalMask::SO as u8);
    assert!(po < so, "PO={po:#05b} SO={so:#05b}");
    assert_eq!(rung(CausalMask::PO), Some(2));
    assert_eq!(rung(CausalMask::SO), Some(1));
    assert!(rung(CausalMask::PO) > rung(CausalMask::SO));
}

/// PO and SO are incomparable subsets: neither contains the other, so no
/// numeric order on the field can agree with the rung order for both.
#[test]
fn po_and_so_are_incomparable_subsets() {
    let (po, so) = (CausalMask::PO as u8, CausalMask::SO as u8);
    assert_ne!(po & so, po);
    assert_ne!(po & so, so);
    assert_eq!(CausalMask::SPO as u8, po | so);
}

/// Whether some `field >= k` admits exactly the named masks whose rank is at
/// least that of PO, under the ranking `rank`.
fn some_threshold_matches(rank: impl Fn(CausalMask) -> Option<u8>) -> bool {
    let named = [CausalMask::SO, CausalMask::PO, CausalMask::SPO];
    (0u8..=8).any(|k| {
        named
            .iter()
            .all(|&m| ((m as u8) >= k) == (rank(m) >= rank(CausalMask::PO)))
    })
}

/// No threshold on the raw field selects "Intervention or above".
#[test]
fn no_numeric_threshold_reproduces_the_ladder() {
    assert!(!some_threshold_matches(rung));
}

/// The check can pass: if the rung WERE the field's numeric value, a
/// threshold would work. Without this, the test above could not fail.
#[test]
fn the_threshold_check_passes_for_a_numeric_ranking() {
    assert!(some_threshold_matches(|m| Some(m as u8)));
}
