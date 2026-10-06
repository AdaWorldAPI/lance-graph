//! D-GSO-2 (P2): one-hop Moore local-plasticity probe.
//!
//! Plan: `.claude/plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`
//! §5 and §18 P2. Claim under test:
//!
//! ```text
//! neighbor mask -> neighbor ordinal -> 8:8 relation address -> Palette256 law
//!               -> immediate fold -> gated update of the resident byte
//! ```
//!
//! What it reuses, and what it adds:
//!
//! - **Resident state:** the 16 bytes of an existing [`Register128`], read as a
//!   4 x 4 grid (`lane = 4 * y + x`). This is a view, not a second owner.
//! - **Law:** the existing [`PalettePerturbation::hop`]
//!   (`perturb(current, relation(local, neighbor))`). No new law.
//! - **Schedule (the only new piece):** a closed-form Moore neighbourhood.
//!   `moore_mask(lane)` returns an 8-bit mask of the directions that stay on
//!   the grid; `neighbor(lane, dir)` turns a direction ordinal into a lane.
//!   No pair list is built to visit the eight neighbours.
//!
//! One hop only. A step reads from the register as it was before the step and
//! writes into the register after it, so the visiting order of lanes cannot
//! change the result. There is no "repeat until stable" loop here: a
//! recurrent relaxation would be a different, explicitly recurrent mechanism
//! (plan §5).
//!
//! The plasticity gate is a 16-bit lane mask: lane `i` may change only if bit
//! `i` is set. A frozen lane keeps its byte exactly.
//!
//! The 16-byte `before` copy is the one-step double buffer of the same
//! register (a `Copy` microcopy, read-only during the step). It is never
//! stored beside the register.
//!
//! Run: `cargo run -p cognitive-shader-driver --example moore_plasticity_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example moore_plasticity_probe`

use cognitive_shader_driver::palette_perturbation::{
    PaletteLut, PalettePerturbation, PaletteState, PALETTE_LUT_LEN,
};
use lance_graph_contract::register128::Register128;

/// Grid side. The register is 16 bytes, read as 4 x 4.
const SIDE: usize = 4;
/// Lanes in the register.
const LANES: usize = SIDE * SIDE;

/// The eight Moore directions as (dx, dy), ordinal 0..8:
/// NW, N, NE, W, E, SW, S, SE.
const DIRS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Bit `d` is set when direction `d` from `lane` stays on the grid.
///
/// Closed form: drop the west column of directions on the left edge, the east
/// column on the right edge, the north row on the top edge and the south row
/// on the bottom edge. No coordinates are tested per direction.
#[inline]
const fn moore_mask(lane: usize) -> u8 {
    let (x, y) = (lane % SIDE, lane / SIDE);
    let mut m = 0xFFu8;
    if x == 0 {
        m &= !0b0010_1001; // NW, W, SW
    }
    if x == SIDE - 1 {
        m &= !0b1001_0100; // NE, E, SE
    }
    if y == 0 {
        m &= !0b0000_0111; // NW, N, NE
    }
    if y == SIDE - 1 {
        m &= !0b1110_0000; // SW, S, SE
    }
    m
}

/// The lane reached from `lane` in direction ordinal `dir`.
///
/// Only called for directions whose bit is set in `moore_mask(lane)`, so the
/// result is always on the grid.
#[inline]
const fn neighbor(lane: usize, dir: usize) -> usize {
    let (dx, dy) = DIRS[dir];
    (lane as isize + dy * SIDE as isize + dx) as usize
}

/// One gated, synchronous Moore hop over all 16 lanes.
///
/// For each lane, the eight-neighbour fold starts at the lane's own byte and
/// applies the Palette law once per neighbour present in the mask, in
/// direction order. The relation byte is consumed the moment it is read; no
/// relation population exists.
fn moore_step(reg: &mut Register128, law: PalettePerturbation<'_>, plastic: u16) {
    let before = reg.0;
    for lane in 0..LANES {
        if plastic & (1 << lane) == 0 {
            continue;
        }
        reg.0[lane] = fold_lane(&before, lane, law);
    }
}

/// The fold for one lane, reading only `before`.
#[inline]
fn fold_lane(before: &[u8; LANES], lane: usize, law: PalettePerturbation<'_>) -> u8 {
    let local = PaletteState(before[lane]);
    let mut state = local;
    let mut mask = moore_mask(lane);
    while mask != 0 {
        let dir = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        state = law.hop(state, local, PaletteState(before[neighbor(lane, dir)]));
    }
    state.0
}

/// A Palette256 law table filled from a closed function (fixture only).
fn table_from(f: impl Fn(u8, u8) -> u8) -> Box<[u8; PALETTE_LUT_LEN]> {
    let mut v = vec![0u8; PALETTE_LUT_LEN];
    for (addr, slot) in v.iter_mut().enumerate() {
        *slot = f((addr >> 8) as u8, addr as u8);
    }
    v.into_boxed_slice()
        .try_into()
        .expect("Palette256 law is exactly 65,536 bytes")
}

/// Distinct bytes per lane, so a swapped or skipped neighbour changes the result.
fn fixture() -> Register128 {
    Register128(core::array::from_fn(|i| {
        (i as u8).wrapping_mul(37).wrapping_add(11)
    }))
}

fn main() {
    let relation = table_from(|a, b| a ^ b);
    let perturb = table_from(|s, r| s.wrapping_add(r).rotate_left(1));
    let law = PalettePerturbation::new(PaletteLut::new(&relation), PaletteLut::new(&perturb));

    let start = fixture();
    let mut all = start;
    moore_step(&mut all, law, u16::MAX);
    // Freeze the left column (lanes 0, 4, 8, 12).
    let frozen_left: u16 = !0x1111;
    let mut gated = start;
    moore_step(&mut gated, law, frozen_left);

    let neighbours: u32 = (0..LANES).map(|l| moore_mask(l).count_ones()).sum();
    println!("D-GSO-2 one-hop Moore probe over Register128 (4 x 4)");
    println!("  directed neighbour visits per hop: {neighbours} (corner 3, edge 5, interior 8)");
    println!("  start : {:?}", start.0);
    println!("  all   : {:?}", all.0);
    println!("  gated : {:?}  (left column frozen)", gated.0);
    let changed = (0..LANES).filter(|&l| all.0[l] != start.0[l]).count();
    println!("  lanes changed with every lane plastic: {changed} of {LANES}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_add_law<'a>(
        relation: &'a [u8; PALETTE_LUT_LEN],
        perturb: &'a [u8; PALETTE_LUT_LEN],
    ) -> PalettePerturbation<'a> {
        PalettePerturbation::new(PaletteLut::new(relation), PaletteLut::new(perturb))
    }

    /// FAILS IF: the closed-form mask disagrees with bounds-checked geometry
    /// for any lane or direction, or a neighbour lands off the grid.
    #[test]
    fn closed_form_mask_matches_geometry() {
        for lane in 0..LANES {
            let (x, y) = ((lane % SIDE) as isize, (lane / SIDE) as isize);
            for (dir, &(dx, dy)) in DIRS.iter().enumerate() {
                let (nx, ny) = (x + dx, y + dy);
                let on_grid = (0..SIDE as isize).contains(&nx) && (0..SIDE as isize).contains(&ny);
                assert_eq!(
                    moore_mask(lane) >> dir & 1 == 1,
                    on_grid,
                    "lane {lane} dir {dir}"
                );
                if on_grid {
                    assert_eq!(neighbor(lane, dir), (ny * SIDE as isize + nx) as usize);
                }
            }
        }
        // 4 corners x 3 + 8 edges x 5 + 4 interior x 8 = 84 directed visits.
        let total: u32 = (0..LANES).map(|l| moore_mask(l).count_ones()).sum();
        assert_eq!(total, 84);
        assert_eq!(moore_mask(0).count_ones(), 3);
        assert_eq!(moore_mask(1).count_ones(), 5);
        assert_eq!(moore_mask(5).count_ones(), 8);
    }

    /// FAILS IF: the mask-driven fold computes something other than the
    /// existing slice API fed the same eight pairs. The pair list exists here,
    /// in the test, only as the oracle.
    #[test]
    fn mask_fold_equals_the_pair_list_oracle() {
        let relation = table_from(|a, b| a ^ b);
        let perturb = table_from(|s, r| s.wrapping_add(r).rotate_left(1));
        let law = xor_add_law(&relation, &perturb);
        let before = fixture().0;
        for lane in 0..LANES {
            let pairs: Vec<(PaletteState, PaletteState)> = (0..8)
                .filter(|&d| moore_mask(lane) >> d & 1 == 1)
                .map(|d| {
                    (
                        PaletteState(before[lane]),
                        PaletteState(before[neighbor(lane, d)]),
                    )
                })
                .collect();
            let oracle = law.hops(PaletteState(before[lane]), &pairs);
            assert_eq!(fold_lane(&before, lane, law), oracle.0, "lane {lane}");
        }
    }

    /// FAILS IF: the step updates in place, so a lane reads a neighbour that
    /// was already rewritten in the same hop. Visiting lanes in reverse order
    /// must give the same register.
    #[test]
    fn the_hop_is_synchronous() {
        let relation = table_from(|a, b| a ^ b);
        let perturb = table_from(|s, r| s.wrapping_add(r).rotate_left(1));
        let law = xor_add_law(&relation, &perturb);
        let start = fixture();

        let mut forward = start;
        moore_step(&mut forward, law, u16::MAX);

        let mut reverse = start;
        let before = reverse.0;
        for lane in (0..LANES).rev() {
            reverse.0[lane] = fold_lane(&before, lane, law);
        }
        assert_eq!(forward, reverse);

        // Anti-vacuity: an in-place sweep really does differ on this fixture,
        // so equality above is evidence and not a property of the data.
        let mut in_place = start;
        for lane in 0..LANES {
            let snapshot = in_place.0;
            in_place.0[lane] = fold_lane(&snapshot, lane, law);
        }
        assert_ne!(in_place, forward);
    }

    /// FAILS IF: a frozen lane changes, or no plastic lane changes.
    #[test]
    fn the_gate_freezes_exactly_the_masked_lanes() {
        let relation = table_from(|a, b| a ^ b);
        let perturb = table_from(|s, r| s.wrapping_add(r).rotate_left(1));
        let law = xor_add_law(&relation, &perturb);
        let start = fixture();
        let plastic: u16 = !0x1111; // left column frozen
        let mut reg = start;
        moore_step(&mut reg, law, plastic);
        let mut changed = 0;
        for lane in 0..LANES {
            if plastic & (1 << lane) == 0 {
                assert_eq!(reg.0[lane], start.0[lane], "frozen lane {lane} moved");
            } else if reg.0[lane] != start.0[lane] {
                changed += 1;
            }
        }
        assert!(changed >= 10, "only {changed} of 12 plastic lanes changed");

        // Freezing a lane does not change what its neighbours compute: they
        // still read its byte from `before`.
        let mut all = start;
        moore_step(&mut all, law, u16::MAX);
        for lane in 0..LANES {
            if plastic & (1 << lane) != 0 {
                assert_eq!(reg.0[lane], all.0[lane], "lane {lane}");
            }
        }
    }

    /// FAILS IF: the step silently iterates (a second hop changes nothing
    /// because the first already relaxed to a fixed point), or an identity
    /// law moves any byte.
    #[test]
    fn one_hop_moves_and_an_identity_law_stays_silent() {
        let relation = table_from(|a, b| a ^ b);
        let perturb = table_from(|s, r| s.wrapping_add(r).rotate_left(1));
        let law = xor_add_law(&relation, &perturb);
        let mut once = fixture();
        moore_step(&mut once, law, u16::MAX);
        let mut twice = once;
        moore_step(&mut twice, law, u16::MAX);
        assert_ne!(once, fixture());
        assert_ne!(twice, once, "one call must be exactly one hop");

        let keep = table_from(|s, _| s);
        let identity = xor_add_law(&relation, &keep);
        let mut reg = fixture();
        moore_step(&mut reg, identity, u16::MAX);
        assert_eq!(reg, fixture());
    }

    /// FAILS IF: the same register and law replay to a different register.
    #[test]
    fn replay_is_deterministic() {
        let relation = table_from(|a, b| a ^ b);
        let perturb = table_from(|s, r| s.wrapping_add(r).rotate_left(1));
        let law = xor_add_law(&relation, &perturb);
        let mut a = fixture();
        let mut b = fixture();
        moore_step(&mut a, law, 0xA5A5);
        moore_step(&mut b, law, 0xA5A5);
        assert_eq!(a, b);
    }
}
