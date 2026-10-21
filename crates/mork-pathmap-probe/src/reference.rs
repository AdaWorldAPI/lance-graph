//! Independent oracle: a nested loop over the two relations' lanes. Names no
//! upstream type and shares no code with [`crate::join`] or
//! [`crate::boundary`]. It builds no row either: it counts.

use crate::JoinInput;

/// Derivations of `Q(x, y, z) :- R(x, y), S(y, z)` reading as `pattern`
/// (`None`: any value), by nested loop over `R`'s and `S`'s lanes.
pub fn matching(input: &JoinInput, pattern: [Option<u32>; 3]) -> u64 {
    let hit = |p: Option<u32>, v: u32| p.is_none_or(|w| w == v);
    let (rx, ry) = (&input.r.cols[0], &input.r.cols[1]);
    let (sy, sz) = (&input.s.cols[0], &input.s.cols[1]);
    let mut n = 0u64;
    for i in 0..rx.len() {
        if !hit(pattern[0], rx[i]) || !hit(pattern[1], ry[i]) {
            continue;
        }
        for j in 0..sy.len() {
            if sy[j] == ry[i] && hit(pattern[2], sz[j]) {
                n += 1;
            }
        }
    }
    n
}

/// Every derivation, counted.
pub fn bag_count(input: &JoinInput) -> u64 {
    matching(input, [None; 3])
}
