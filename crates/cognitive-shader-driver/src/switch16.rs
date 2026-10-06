//! Reversible switch fabric over one 16-byte tenant.
//!
//! The state is a [`Register128`]: sixteen independent byte lanes, already
//! resident in a value-slab rail. This module owns no second 16-byte type; it
//! is a reading of the existing one. Pairing the lanes into eight `8:8` rails,
//! four `Quad8`s or this fabric are all views of the same bytes.
//!
//! ```text
//! 16 byte lanes
//!      ↓
//! 7 stages, pair stride 1, 2, 4, 8, 4, 2, 1
//!      ↓
//! same 16 byte lanes
//! ```
//!
//! Each stage pairs all 16 lanes into 8 disjoint pairs. The pairing is closed
//! form: `(stage, pair)` are schedule ordinals, not coordinates, and no edge
//! list is stored. Every pair applies a bijective two-byte step, so running the
//! stages in reverse with the inverse step restores the input exactly.
//!
//! The layout is butterfly out, butterfly back (Beneš-shaped). This module
//! claims neither arbitrary-permutation routing nor cryptographic strength.
//! There is no key, no LUT, no matrix and no widened intermediate state.

use lance_graph_contract::register128::Register128;

const LANES: usize = 16;
const STAGES: usize = 7;
const PAIRS_PER_STAGE: usize = LANES / 2;

/// `log2` of the pair stride of `stage`: 0, 1, 2, 3, 2, 1, 0.
#[inline(always)]
const fn stride_log2(stage: usize) -> usize {
    let back = STAGES - 1 - stage;
    if stage < back {
        stage
    } else {
        back
    }
}

/// The two lanes of pair `pair` in stage `stage`, `left < right`.
///
/// The left lane is `pair` with a zero bit inserted at the stride position;
/// the right lane sets that bit. Pairs come out in ascending left-lane order.
#[inline(always)]
const fn pair_lanes(stage: usize, pair: usize) -> (usize, usize) {
    let k = stride_log2(stage);
    let low = pair & ((1 << k) - 1);
    let left = ((pair >> k) << (k + 1)) | low;
    (left, left | (1 << k))
}

/// Reversible switch fabric over a 16-byte tenant, applied in place.
pub trait Switch16 {
    /// Run the seven stages forward.
    fn switch_forward(&mut self);
    /// Run the seven stages in reverse with the inverse step; undoes
    /// [`Switch16::switch_forward`] exactly.
    fn switch_inverse(&mut self);
}

impl Switch16 for [u8; LANES] {
    fn switch_forward(&mut self) {
        for stage in 0..STAGES {
            for pair in 0..PAIRS_PER_STAGE {
                let (l, r) = pair_lanes(stage, pair);
                (self[l], self[r]) = step_forward(self[l], self[r], stage, pair);
            }
        }
    }

    fn switch_inverse(&mut self) {
        for stage in (0..STAGES).rev() {
            for pair in (0..PAIRS_PER_STAGE).rev() {
                let (l, r) = pair_lanes(stage, pair);
                (self[l], self[r]) = step_inverse(self[l], self[r], stage, pair);
            }
        }
    }
}

impl Switch16 for Register128 {
    fn switch_forward(&mut self) {
        self.0.switch_forward();
    }

    fn switch_inverse(&mut self) {
        self.0.switch_inverse();
    }
}

/// Bijective two-byte step; both outputs depend on both inputs.
///
/// ```text
/// x' = x + rotl(y, r1) + tweak   (mod 256)
/// y' = y XOR rotl(x', r2)
/// ```
#[inline(always)]
fn step_forward(x: u8, y: u8, stage: usize, pair: usize) -> (u8, u8) {
    let (r1, r2, tweak) = step_params(stage, pair);
    let x2 = x.wrapping_add(y.rotate_left(r1)).wrapping_add(tweak);
    (x2, y ^ x2.rotate_left(r2))
}

/// Undoes [`step_forward`]: the XOR first, then the modular add.
#[inline(always)]
fn step_inverse(x2: u8, y2: u8, stage: usize, pair: usize) -> (u8, u8) {
    let (r1, r2, tweak) = step_params(stage, pair);
    let y = y2 ^ x2.rotate_left(r2);
    (x2.wrapping_sub(y.rotate_left(r1)).wrapping_sub(tweak), y)
}

/// Fixed per-step constants derived from the schedule ordinals. Not a key.
#[inline(always)]
fn step_params(stage: usize, pair: usize) -> (u32, u32, u8) {
    let r1 = ((stage + pair) % 7 + 1) as u32;
    let r2 = ((stage * 3 + pair * 5) % 7 + 1) as u32;
    let tweak = (stage as u8)
        .wrapping_mul(29)
        .wrapping_add((pair as u8).wrapping_mul(17))
        .wrapping_add(0x5d);
    (r1, r2, tweak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_go_out_and_back() {
        let strides: Vec<usize> = (0..STAGES).map(|s| 1 << stride_log2(s)).collect();
        assert_eq!(strides, [1, 2, 4, 8, 4, 2, 1]);
    }

    #[test]
    fn every_stage_pairs_each_lane_exactly_once_at_its_stride() {
        for stage in 0..STAGES {
            let stride = 1 << stride_log2(stage);
            let mut touched = 0u16;
            for pair in 0..PAIRS_PER_STAGE {
                let (l, r) = pair_lanes(stage, pair);
                assert_eq!(r - l, stride);
                assert_eq!(l & stride, 0, "left lane has the stride bit clear");
                let bits = (1u16 << l) | (1u16 << r);
                assert_eq!(touched & bits, 0, "stage {stage} reuses a lane");
                touched |= bits;
            }
            assert_eq!(touched, u16::MAX, "stage {stage} misses a lane");
        }
    }

    #[test]
    fn every_output_lane_depends_on_every_input_lane() {
        let mut reach: [u16; LANES] = core::array::from_fn(|i| 1 << i);
        for stage in 0..STAGES {
            for pair in 0..PAIRS_PER_STAGE {
                let (l, r) = pair_lanes(stage, pair);
                let both = reach[l] | reach[r];
                reach[l] = both;
                reach[r] = both;
            }
        }
        assert!(reach.iter().all(|&m| m == u16::MAX));
    }

    /// Pins this closed-form fabric to the output of the #1338 prototype,
    /// which stored its stage edges in arrays.
    #[test]
    fn forward_matches_the_prototype_output() {
        let mut ramp: [u8; 16] = core::array::from_fn(|i| i as u8);
        ramp.switch_forward();
        assert_eq!(
            ramp,
            [190, 36, 1, 53, 228, 199, 177, 249, 189, 63, 199, 172, 228, 178, 228, 204]
        );
        let mut zero = [0u8; 16];
        zero.switch_forward();
        assert_eq!(
            zero,
            [108, 138, 44, 136, 234, 206, 42, 210, 121, 236, 197, 69, 83, 148, 30, 127]
        );
    }

    #[test]
    fn forward_then_inverse_restores_the_input() {
        let fixtures: [[u8; 16]; 4] = [
            [0; 16],
            [0xff; 16],
            core::array::from_fn(|i| i as u8),
            core::array::from_fn(|i| (i as u8).wrapping_mul(17)),
        ];
        for input in fixtures {
            let mut b = input;
            b.switch_forward();
            assert_ne!(b, input);
            b.switch_inverse();
            assert_eq!(b, input);
        }
    }

    #[test]
    fn inverse_then_forward_restores_the_input() {
        let input: [u8; 16] = core::array::from_fn(|i| 255u8.wrapping_sub((i as u8) * 13));
        let mut b = input;
        b.switch_inverse();
        b.switch_forward();
        assert_eq!(b, input);
    }

    #[test]
    fn a_register_switches_its_own_bytes() {
        let bytes: [u8; 16] = core::array::from_fn(|i| (i as u8) * 7);
        let mut reg = Register128(bytes);
        let mut raw = bytes;
        reg.switch_forward();
        raw.switch_forward();
        assert_eq!(reg.0, raw);
        reg.switch_inverse();
        assert_eq!(reg.0, bytes);
    }

    #[test]
    fn every_step_is_bijective_over_all_byte_pairs() {
        for stage in 0..STAGES {
            for pair in 0..PAIRS_PER_STAGE {
                for x in 0..=255u8 {
                    for y in 0..=255u8 {
                        let (x2, y2) = step_forward(x, y, stage, pair);
                        assert_eq!(step_inverse(x2, y2, stage, pair), (x, y));
                    }
                }
            }
        }
    }

    #[test]
    fn forward_is_not_a_byte_permutation() {
        let input: [u8; 16] = core::array::from_fn(|i| i as u8);
        let mut out = input;
        out.switch_forward();
        let (mut a, mut b) = (input, out);
        a.sort_unstable();
        b.sort_unstable();
        assert_ne!(a, b);
    }
}
