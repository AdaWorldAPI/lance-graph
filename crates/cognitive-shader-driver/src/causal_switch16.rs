//! Reversible 16-byte causal switch fabric.
//!
//! This is the non-reducing twin of the Palette256 / Quad8 work:
//!
//! ```text
//! 16 resident bytes
//!      ↓
//! seven fixed 2×2 switch stages
//!      ↓
//! same 16 resident bytes
//! ```
//!
//! Stage distances are `1,2,4,8,4,2,1`: a butterfly out, a butterfly back.
//! That is Beneš-shaped routing geometry, but this module deliberately does
//! not claim arbitrary-permutation routing or cryptographic security.
//!
//! Every stage is eight disjoint causal edges over the 16 byte positions.
//! Every edge applies a bijective two-byte lifting step. Running the same
//! topology in reverse stage order with the inverse lifting step reconstructs
//! the input exactly.
//!
//! No S-box table, no LUT, no matrix, and no widened intermediate state.

/// Number of resident byte wires.
pub const SWITCH16_WIRES: usize = 16;

/// Number of 2×2 edges per stage.
pub const SWITCH16_EDGES_PER_STAGE: usize = SWITCH16_WIRES / 2;

/// Beneš-shaped stage count for 16 wires: 2*log2(16)-1 = 7.
pub const SWITCH16_STAGES: usize = 7;

/// Stage strides: butterfly out to distance 8, then fold back.
pub const SWITCH16_STRIDES: [u8; SWITCH16_STAGES] = [1, 2, 4, 8, 4, 2, 1];

/// One causal interaction edge inside a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwitchEdge {
    pub left: u8,
    pub right: u8,
}

impl SwitchEdge {
    #[inline(always)]
    pub const fn new(left: u8, right: u8) -> Self {
        Self { left, right }
    }
}

/// The 16-byte resident state transformed by the switch fabric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SwitchState16([u8; SWITCH16_WIRES]);

impl SwitchState16 {
    #[inline(always)]
    pub const fn new(bytes: [u8; SWITCH16_WIRES]) -> Self {
        Self(bytes)
    }

    #[inline(always)]
    pub const fn bytes(self) -> [u8; SWITCH16_WIRES] {
        self.0
    }
}

/// Fixed reversible causal graph over sixteen byte wires.
#[derive(Clone, Copy, Debug, Default)]
pub struct CausalSwitch16;

impl CausalSwitch16 {
    /// Return the eight disjoint edges for one stage.
    ///
    /// Pairing is the canonical hypercube/butterfly relation
    /// `peer = wire XOR stride`, retaining only `wire < peer`.
    pub const fn stage_edges(stage: usize) -> Option<[SwitchEdge; SWITCH16_EDGES_PER_STAGE]> {
        if stage >= SWITCH16_STAGES {
            return None;
        }

        let stride = SWITCH16_STRIDES[stage];
        let mut out = [SwitchEdge::new(0, 0); SWITCH16_EDGES_PER_STAGE];
        let mut n = 0usize;
        let mut wire = 0u8;

        while wire < SWITCH16_WIRES as u8 {
            let peer = wire ^ stride;
            if wire < peer {
                out[n] = SwitchEdge::new(wire, peer);
                n += 1;
            }
            wire += 1;
        }

        Some(out)
    }

    /// Forward traversal of the seven-stage causal graph.
    #[inline]
    pub fn forward(self, state: SwitchState16) -> SwitchState16 {
        let mut bytes = state.0;

        for stage in 0..SWITCH16_STAGES {
            let edges = Self::stage_edges(stage).expect("stage is in range");
            for (edge_index, edge) in edges.into_iter().enumerate() {
                let left = bytes[edge.left as usize];
                let right = bytes[edge.right as usize];
                let (mixed_left, mixed_right) = pair_forward(left, right, stage, edge_index);
                bytes[edge.left as usize] = mixed_left;
                bytes[edge.right as usize] = mixed_right;
            }
        }

        SwitchState16(bytes)
    }

    /// Exact inverse traversal: stages and local pair transforms are reversed.
    #[inline]
    pub fn inverse(self, state: SwitchState16) -> SwitchState16 {
        let mut bytes = state.0;

        for stage in (0..SWITCH16_STAGES).rev() {
            let edges = Self::stage_edges(stage).expect("stage is in range");
            for (edge_index, edge) in edges.into_iter().enumerate().rev() {
                let left = bytes[edge.left as usize];
                let right = bytes[edge.right as usize];
                let (plain_left, plain_right) = pair_inverse(left, right, stage, edge_index);
                bytes[edge.left as usize] = plain_left;
                bytes[edge.right as usize] = plain_right;
            }
        }

        SwitchState16(bytes)
    }

    /// Structural reachability after all seven stages.
    ///
    /// This ignores byte values and tracks only the causal graph: when two
    /// wires meet at a 2×2 switch, both become reachable from the union of
    /// their prior source sets. The returned mask for each output wire is a
    /// 16-bit set of source wires that can causally influence it.
    pub fn reachability(self) -> [u16; SWITCH16_WIRES] {
        let mut reach = [0u16; SWITCH16_WIRES];
        let mut wire = 0usize;
        while wire < SWITCH16_WIRES {
            reach[wire] = 1u16 << wire;
            wire += 1;
        }

        for stage in 0..SWITCH16_STAGES {
            let edges = Self::stage_edges(stage).expect("stage is in range");
            for edge in edges {
                let union = reach[edge.left as usize] | reach[edge.right as usize];
                reach[edge.left as usize] = union;
                reach[edge.right as usize] = union;
            }
        }

        reach
    }
}

/// Reversible two-byte lifting step.
///
/// Both outputs depend on both inputs:
///
/// ```text
//! x' = x + rotl(y, r1) + tweak          (mod 256)
//! y' = y XOR rotl(x', r2)
//! ```
//!
//! The inverse undoes the XOR lift first, then the modular-add lift.
//! Stage/edge-derived constants are deterministic routing metadata, not keys.
#[inline(always)]
fn pair_forward(left: u8, right: u8, stage: usize, edge: usize) -> (u8, u8) {
    let (r1, r2, tweak) = pair_params(stage, edge);
    let mixed_left = left
        .wrapping_add(right.rotate_left(r1))
        .wrapping_add(tweak);
    let mixed_right = right ^ mixed_left.rotate_left(r2);
    (mixed_left, mixed_right)
}

#[inline(always)]
fn pair_inverse(left: u8, right: u8, stage: usize, edge: usize) -> (u8, u8) {
    let (r1, r2, tweak) = pair_params(stage, edge);
    let plain_right = right ^ left.rotate_left(r2);
    let plain_left = left
        .wrapping_sub(plain_right.rotate_left(r1))
        .wrapping_sub(tweak);
    (plain_left, plain_right)
}

#[inline(always)]
fn pair_params(stage: usize, edge: usize) -> (u32, u32, u8) {
    let r1 = ((stage + edge) % 7 + 1) as u32;
    let r2 = ((stage * 3 + edge * 5) % 7 + 1) as u32;
    let tweak = (stage as u8)
        .wrapping_mul(29)
        .wrapping_add((edge as u8).wrapping_mul(17))
        .wrapping_add(0x5d);
    (r1, r2, tweak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_is_exactly_sixteen_bytes() {
        assert_eq!(core::mem::size_of::<SwitchState16>(), 16);
    }

    #[test]
    fn every_stage_is_eight_disjoint_edges_at_its_stride() {
        for (stage, expected_stride) in SWITCH16_STRIDES.into_iter().enumerate() {
            let edges = CausalSwitch16::stage_edges(stage).unwrap();
            let mut touched = 0u16;

            for edge in edges {
                assert_eq!(edge.left ^ edge.right, expected_stride);
                let left_bit = 1u16 << edge.left;
                let right_bit = 1u16 << edge.right;
                assert_eq!(touched & (left_bit | right_bit), 0);
                touched |= left_bit | right_bit;
            }

            assert_eq!(touched, u16::MAX);
        }
    }

    #[test]
    fn graph_has_fifty_six_stage_edges() {
        let total = (0..SWITCH16_STAGES)
            .map(|stage| CausalSwitch16::stage_edges(stage).unwrap().len())
            .sum::<usize>();
        assert_eq!(total, 56);
    }

    #[test]
    fn causal_reachability_closes_over_all_sixteen_wires() {
        let reach = CausalSwitch16.reachability();
        assert!(reach.into_iter().all(|mask| mask == u16::MAX));
    }

    #[test]
    fn forward_inverse_round_trip_structured_fixtures() {
        let fabric = CausalSwitch16;
        let fixtures = [
            [0u8; 16],
            [0xffu8; 16],
            core::array::from_fn(|i| i as u8),
            core::array::from_fn(|i| (i as u8).wrapping_mul(17)),
            [
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
            ],
        ];

        for bytes in fixtures {
            let input = SwitchState16::new(bytes);
            let encoded = fabric.forward(input);
            let decoded = fabric.inverse(encoded);
            assert_eq!(decoded, input);
        }
    }

    #[test]
    fn inverse_forward_round_trip_also_holds() {
        let fabric = CausalSwitch16;
        let bytes = core::array::from_fn(|i| (255u8).wrapping_sub((i as u8).wrapping_mul(13)));
        let input = SwitchState16::new(bytes);
        assert_eq!(fabric.forward(fabric.inverse(input)), input);
    }

    #[test]
    fn local_pair_step_is_bijective_for_all_65536_inputs() {
        for left in 0u16..=255 {
            for right in 0u16..=255 {
                let plain = (left as u8, right as u8);
                let mixed = pair_forward(plain.0, plain.1, 3, 5);
                assert_eq!(pair_inverse(mixed.0, mixed.1, 3, 5), plain);
            }
        }
    }

    #[test]
    fn forward_is_not_merely_a_permutation_of_input_bytes() {
        let fabric = CausalSwitch16;
        let input = SwitchState16::new(core::array::from_fn(|i| i as u8));
        let output = fabric.forward(input).bytes();

        let mut in_sorted = input.bytes();
        let mut out_sorted = output;
        in_sorted.sort_unstable();
        out_sorted.sort_unstable();

        assert_ne!(out_sorted, in_sorted);
    }
}
