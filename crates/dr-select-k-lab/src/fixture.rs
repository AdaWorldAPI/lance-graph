//! Deterministic fixtures. No randomness source beyond SplitMix64.

use crate::source::{Source, Version};

/// Where the rows of one key range sit in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    /// Rows of a key range are contiguous in the source.
    Clustered,
    /// Rows of a key range are scattered over the whole source.
    Random,
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` rows at version 1.
///
/// - `key`: `slot / dup`, where `slot` is the row index (`Clustered`) or a
///   fixed permutation of `0..n` (`Random`). So each key value is shared by
///   `dup` rows, and the range `lo..=hi` selects `(hi - lo + 1) * dup` rows
///   while `hi < n / dup`.
/// - `a`: uniform in `0..100`, so `a >= t` keeps about `100 - t` percent.
/// - `b`: uniform in `0..1000`.
/// - `guid`: unique, and not monotone in the row or the key, so a tie on
///   the key is really broken by the guid.
pub fn source(n: usize, pattern: Pattern, dup: u32, seed: u64) -> Source {
    assert!(dup >= 1 && u32::try_from(n).is_ok());
    let mut state = seed;
    let mut slot: Vec<u32> = (0..n as u32).collect();
    if pattern == Pattern::Random {
        for i in (1..n).rev() {
            let j = (splitmix(&mut state) % (i as u64 + 1)) as usize;
            slot.swap(i, j);
        }
    }
    let key = slot.iter().map(|s| (s / dup) as i32).collect();
    let a = (0..n)
        .map(|_| (splitmix(&mut state) % 100) as i32)
        .collect();
    let b = (0..n)
        .map(|_| (splitmix(&mut state) % 1000) as i32)
        .collect();
    let guid = (0..n)
        .map(|i| (u128::from(splitmix(&mut state)) << 64) | i as u128)
        .collect();
    Source::new(Version(1), guid, key, a, b)
}
