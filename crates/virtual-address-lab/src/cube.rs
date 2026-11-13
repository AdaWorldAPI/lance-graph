//! The 256^3 address space: three axes of 256 identities, relations counted
//! without building a cube or a plane over two axes.

use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_dt64::dr::{address as tile_address, coordinates, MapRefusal, Width};

/// A subset of one axis: 256 bits.
pub type AxisMask = [u64; 4];

pub const FULL: AxisMask = [u64::MAX; 4];

pub fn axis(keep: impl Fn(u8) -> bool) -> AxisMask {
    let mut m = [0u64; 4];
    for i in (0..=255u8).filter(|&i| keep(i)) {
        m[usize::from(i) / 64] |= 1u64 << (i % 64);
    }
    m
}

pub fn has(m: &AxisMask, i: u8) -> bool {
    m[usize::from(i) / 64] >> (i % 64) & 1 == 1
}

pub fn pop(m: &AxisMask) -> u64 {
    m.iter().map(|w| u64::from(w.count_ones())).sum()
}

/// `(z << 16) | Morton8x8(x, y)`, the cell address of `dr::address` under an
/// outer byte.
pub fn address(x: u8, y: u8, z: u8) -> Result<u32, MapRefusal> {
    Ok(u32::from(z) << 16 | u32::from(tile_address(Width::W256, x, y)?.code()))
}

/// The inverse of [`address`]; `None` above 24 bits.
pub fn decode(address: u32) -> Option<(u8, u8, u8)> {
    if address >> 24 != 0 {
        return None;
    }
    let (x, y) = coordinates(Width::W256, Morton8x8::from_code(address as u16)).ok()?;
    Some((x, y, (address >> 16) as u8))
}

/// Family A, `A(x) ∧ B(y) ∧ C(z)`: three popcounts.
pub fn count_factorized(a: &AxisMask, b: &AxisMask, c: &AxisMask) -> u64 {
    pop(a) * pop(b) * pop(c)
}

/// `m` reflected: bit `i` of the result is bit `255 - i` of `m`.
pub fn reflect(m: &AxisMask) -> AxisMask {
    [
        m[3].reverse_bits(),
        m[2].reverse_bits(),
        m[1].reverse_bits(),
        m[0].reverse_bits(),
    ]
}

/// `m` rotated on the ring of 256: bit `(i + k) mod 256` of the result is bit
/// `i` of `m`.
pub fn rotate(m: &AxisMask, k: u8) -> AxisMask {
    let (words, bits) = (usize::from(k) / 64, u32::from(k) % 64);
    let mut out = [0u64; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = m[(i + 4 - words) % 4];
        let hi = m[(i + 4 - words + 3) % 4];
        *slot = if bits == 0 {
            lo
        } else {
            lo << bits | hi >> (64 - bits)
        };
    }
    out
}

/// Family B, `x + y + z ≡ 0 (mod 256)` under axis masks. `z` is determined
/// by `(x, y)`, so for each admitted `z` the pairs are those with
/// `y = -z - x`: `B` reflected and rotated onto the `x` axis, ANDed with `A`.
/// At most 256 steps of four words; nothing population-sized is built.
pub fn count_coupled(a: &AxisMask, b: &AxisMask, c: &AxisMask) -> u64 {
    let b_reflected = reflect(b);
    let mut total = 0u64;
    for z in (0..=255u8).filter(|&z| has(c, z)) {
        // y = s - x with s = -z; reflect gives B(255 - x), so rotate by s + 1.
        let s = z.wrapping_neg();
        let b_on_x = rotate(&b_reflected, s.wrapping_add(1));
        total += (0..4)
            .map(|i| u64::from((a[i] & b_on_x[i]).count_ones()))
            .sum::<u64>();
    }
    total
}

/// Family C, a relation with no structure: membership by a hash of the
/// address.
pub fn unstructured(x: u8, y: u8, z: u8) -> bool {
    let mut h = u64::from(address(x, y, z).expect("inside the tile"));
    h = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (h ^ (h >> 31)).is_multiple_of(5)
}

/// Family C counted the only way available: one membership test per
/// candidate of `A x B x C`. Returns `(count, tests)`.
pub fn count_unstructured(a: &AxisMask, b: &AxisMask, c: &AxisMask) -> (u64, u64) {
    let (mut count, mut tests) = (0u64, 0u64);
    for x in (0..=255u8).filter(|&x| has(a, x)) {
        for y in (0..=255u8).filter(|&y| has(b, y)) {
            for z in (0..=255u8).filter(|&z| has(c, z)) {
                tests += 1;
                count += u64::from(unstructured(x, y, z));
            }
        }
    }
    (count, tests)
}
