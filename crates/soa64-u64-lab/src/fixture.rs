//! Deterministic fixture generator: the only source of truth for values,
//! presence and identities. Every arm ingests from here and the oracle reads
//! from here.

use std::cell::Cell;

/// Logical words per row.
pub const FIELDS: usize = 64;

/// Standard splitmix64 finalizer applied to `x` (state increment included).
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Global identity of row `i`: a random 64-bit value, not a dense ordinal.
pub fn node_guid(seed: u64, i: usize) -> u64 {
    splitmix64(seed ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93))
}

/// Declared width of a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldSpec {
    pub bits: u32,
}

/// Field 0 is the 4-bit predicate key, field 1 the 64-bit sum operand, field 2
/// the nullable 16-bit field; fields 3.. cycle through a width table.
pub fn field_bits(f: usize) -> u32 {
    const TABLE: [u32; 12] = [1, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64];
    match f {
        0 => 4,
        1 => 64,
        2 => 16,
        _ => TABLE[(f - 3) % 12],
    }
}

/// The spec of field `f`.
pub fn field_spec(f: usize) -> FieldSpec {
    FieldSpec {
        bits: field_bits(f),
    }
}

/// Keep the low `bits` bits of `v`.
#[inline(always)]
pub fn mask_to(v: u64, bits: u32) -> u64 {
    if bits == 64 {
        v
    } else {
        v & ((1u64 << bits) - 1)
    }
}

/// Value of field `f` of row `i`. Field 1 of every row with `i % 97 == 13` is
/// `u64::MAX` (exact-boundary rows). Defined even where the field is absent.
pub fn value(seed: u64, i: usize, f: usize) -> u64 {
    if f == 1 && i % 97 == 13 {
        return u64::MAX;
    }
    let v = splitmix64(node_guid(seed, i) ^ ((f as u64 + 1) << 56));
    mask_to(v, field_bits(f))
}

/// Presence of field `f` of row `i`. Only field 2 is nullable (~20% absent);
/// absent is not zero.
pub fn present(seed: u64, i: usize, f: usize) -> bool {
    if f == 2 {
        !splitmix64(node_guid(seed, i) ^ 0x00C0_FFEE).is_multiple_of(5)
    } else {
        true
    }
}

/// `count` pseudo-random ordinals in `0..n`, for point-access probes.
pub fn probes(seed: u64, n: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|j| {
            (splitmix64(seed ^ (j as u64).wrapping_mul(0xA24B_AED4_963E_E407)) % n as u64) as usize
        })
        .collect()
}

/// A fixture: `n` rows in canonical ordinal order (the source owner's sealed order).
#[derive(Debug, Clone)]
pub struct Fixture {
    pub seed: u64,
    pub n: usize,
    pub guids: Vec<u64>,
}

impl Fixture {
    pub fn new(seed: u64, n: usize) -> Fixture {
        Fixture {
            seed,
            n,
            guids: (0..n).map(|i| node_guid(seed, i)).collect(),
        }
    }
}

/// Tile coordinates of ordinal `i`: `(tile, slot)` with 256 slots per tile.
pub fn tile_slot(i: usize) -> (usize, usize) {
    (i >> 8, i & 255)
}

/// Global guid -> ordinal map, built once by sorting. Used for ingress and by
/// the identity-lookup null arm, never inside a positional sweep.
#[derive(Debug)]
pub struct G2L {
    sorted: Vec<(u64, u32)>,
    /// Number of `lookup` calls so far.
    pub lookups: Cell<u64>,
}

impl G2L {
    pub fn new(fx: &Fixture) -> G2L {
        let mut sorted: Vec<(u64, u32)> = fx
            .guids
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, i as u32))
            .collect();
        sorted.sort_unstable();
        G2L {
            sorted,
            lookups: Cell::new(0),
        }
    }

    pub fn lookup(&self, guid: u64) -> Option<u32> {
        self.lookups.set(self.lookups.get() + 1);
        self.sorted
            .binary_search_by_key(&guid, |&(g, _)| g)
            .ok()
            .map(|k| self.sorted[k].1)
    }
}
