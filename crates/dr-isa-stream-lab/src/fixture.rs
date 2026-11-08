//! Deterministic values and WHERE masks. Nothing else defines them.

/// Number of WHERE masks, and the GEMM's N.
pub const MASKS: usize = 16;
/// Property lanes per GEMM row tile (the GEMM's M).
pub const ROWS: usize = 16;
/// Identities per GEMM K block.
pub const BLOCK: usize = 32;

/// The value set: an identity count, a property count and a seed.
#[derive(Debug, Clone, Copy)]
pub struct Fixture {
    pub n: usize,
    pub props: usize,
    pub seed: u64,
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

impl Fixture {
    /// Property `p` of identity `i`: an integer in [-8, 8], exact in bf16.
    pub fn value(&self, i: usize, p: usize) -> i32 {
        let h = mix(self.seed ^ ((p as u64) << 40) ^ i as u64);
        (h % 17) as i32 - 8
    }

    /// Whether identity `i` is selected by WHERE mask `m`.
    pub fn selected(&self, i: usize, m: usize) -> bool {
        // Mask m keeps roughly (m + 1) / 17 of the population, so the 16
        // masks differ in density as well as membership.
        mix(self.seed.rotate_left(17) ^ ((m as u64) << 48) ^ i as u64) % 17 <= m as u64
    }
}

/// bf16 encoding of `v`, or `None` when bf16 cannot hold it exactly.
pub fn bf16_exact(v: i32) -> Option<u16> {
    let f = v as f32;
    let bits = f.to_bits();
    // Compare in f64, which holds every i32 and every f32 exactly: casting
    // `f` back to i32 would saturate 2^31 to i32::MAX and pass it as exact.
    if bits & 0xFFFF != 0 || f64::from(f) != f64::from(v) {
        None
    } else {
        Some((bits >> 16) as u16)
    }
}

/// bf16 encoding of `v` by round-to-nearest-even; approximate by design.
pub fn bf16_round(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7FFF + lsb) >> 16) as u16
}

/// The f32 value of a bf16 encoding.
pub fn bf16_value(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Out-degree drawn per identity by [`Fixture::neighbors`].
pub const DEGREE: usize = 8;
/// One identity in this many gets a long-range edge in [`Fixture::neighbors`].
pub const FAR_EVERY: usize = 16;

impl Fixture {
    /// Identity `i`'s neighbors with the default long-range rate.
    pub fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.neighbors_with(i, FAR_EVERY)
    }

    /// Identity `i`'s `DEGREE` neighbors, within +-512 of `i` (its own and
    /// nearby cube tiles), except that one identity in `far_every` has its
    /// last edge go anywhere. Long-range edges are what make a block layout
    /// fill up, so the rate is a parameter. May repeat a neighbor.
    pub fn neighbors_with(&self, i: usize, far_every: usize) -> impl Iterator<Item = usize> + '_ {
        (0..DEGREE).map(move |k| {
            let h = mix(self.seed.rotate_left(29) ^ ((k as u64) << 56) ^ i as u64);
            if k + 1 == DEGREE && i.is_multiple_of(far_every) {
                (h % self.n as u64) as usize
            } else {
                let off = (h % 1025) as i64 - 512;
                (i as i64 + off).rem_euclid(self.n as i64) as usize
            }
        })
    }
}
