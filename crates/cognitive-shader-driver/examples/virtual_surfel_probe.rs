//! D-CTX-1: virtual surfel reading over a 16 x 16 Palette256 tile.
//!
//! Claim under test: a surfel needs no persistent identity and no population.
//! Everything the later rendering step needs from it can be *read* off
//!
//! ```text
//! pixel address (geometry) + PaletteState (material) + Fisher-Z pair law
//! ```
//!
//! at the moment it is consumed, and the read is identical to a surfel that was
//! materialized first.
//!
//! # The tile
//!
//! 256 resident bytes, one `PaletteState` per pixel. The lane index is the
//! [`Morton8x8`] code of `(x, y)`; a 16 x 16 grid is exactly the codes `0..256`,
//! so a Moore neighbour is `checked_offset(dx, dy)` kept when `code < 256`.
//! Position never lives in the byte and the byte never encodes position.
//!
//! # The reading
//!
//! For a center pixel `c` with neighbours `n` (decision A: the surfel pair is
//! `(center, local Moore neighbour)`, never a query):
//!
//! ```text
//! w_n   = R_local(c, n) as a non-negative weight   (i8 code + 127 ∈ 0..=254)
//! W     = Σ w_n
//! Σ_c   = Σ w_n · d_n d_nᵀ          d_n = Moore offset of n
//! amp   = activation[c]             (orthogonal reading; 0 = no surfel)
//! ```
//!
//! `code + 127` is the Fisher-Z code's position in the family's calibrated z
//! range (affine in z, scaled by 254), so it is a weight without any decode.
//! An *identity* neighbour (same palette byte) has no table entry; its weight is
//! the explicit argument `identity_weight`, passed by the caller, never assumed.
//! The tests show that this argument decides the surfel's orientation on a
//! stripe, which is why it is not a hidden constant.
//!
//! Integers throughout, so oracle and virtual path agree bit for bit.
//!
//! # Two implementations
//!
//! - **A, materialized oracle** (test shape): row-major geometry, a `Vec` of
//!   directed pairs, a `Vec` of relations through `FisherZTable::lookup_i8`, a
//!   per-pixel accumulator, then a `Vec<SurfelReading>`, then the consumer.
//! - **B, virtual:** for each activated lane, eight checked reads folded into one
//!   `SurfelReading` in registers and handed straight to the consumer.
//!
//! Run: `cargo run -p cognitive-shader-driver --example virtual_surfel_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example virtual_surfel_probe`

use std::mem::size_of;

use bgz_tensor::fisher_z::{FamilyGamma, FisherZTable};
use lance_graph_contract::morton8x8::Morton8x8;

#[path = "support/fisher_relation.rs"]
mod fisher_relation;
use fisher_relation::{allocations_during, representatives, PairwiseFisherZ, Relation, MOORE};

#[path = "support/virtual_surfel.rs"]
mod virtual_surfel;
use virtual_surfel::{
    activation_fixture, for_each_surfel, permutation_tile, repeated_tile, SurfelReading, Tile,
    IDENTITY_WEIGHT, PIXELS, SIDE,
};

/// A consumer that keeps one register: the amplitude-weighted tile moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TileMoment {
    surfels: u32,
    weight: u64,
    sxx: u64,
    sxy: i64,
    syy: u64,
}

impl TileMoment {
    #[inline]
    fn absorb(&mut self, s: SurfelReading) {
        let a = u64::from(s.amplitude);
        self.surfels += 1;
        self.weight += a * u64::from(s.weight);
        self.sxx += a * u64::from(s.sxx);
        self.sxy += a as i64 * i64::from(s.sxy);
        self.syy += a * u64::from(s.syy);
    }
}

/// B end to end: virtual surfels into one moment register.
fn virtual_moment(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>, iw: u16) -> TileMoment {
    let mut m = TileMoment::default();
    for_each_surfel(tile, act, law, iw, |s| m.absorb(s));
    m
}

// ── A: the materialized oracle ─────────────────────────────────────────────

/// What the oracle built, in bytes.
#[derive(Clone, Copy, Debug, Default)]
struct OracleInventory {
    pair_bytes: usize,
    relation_bytes: usize,
    surfel_bytes: usize,
}

/// Row-major geometry, a pair list, a relation population, per-pixel
/// accumulators, then a materialized `Vec<SurfelReading>` in Morton order.
fn oracle_surfels(
    tile: &Tile,
    act: &Tile,
    table: &FisherZTable,
    identity_weight: u16,
) -> (Vec<SurfelReading>, OracleInventory) {
    let side = i32::from(SIDE);
    let at = |x: i32, y: i32| tile[Morton8x8::from_xy(x as u8, y as u8).code() as usize];

    let mut pairs: Vec<((i32, i32), (i32, i32))> = Vec::new();
    for y in 0..side {
        for x in 0..side {
            for &(dx, dy) in &MOORE {
                let (nx, ny) = (x + i32::from(dx), y + i32::from(dy));
                if (0..side).contains(&nx) && (0..side).contains(&ny) {
                    pairs.push(((x, y), (nx, ny)));
                }
            }
        }
    }
    let relations: Vec<Relation> = pairs
        .iter()
        .map(|&((x, y), (nx, ny))| {
            let (a, b) = (at(x, y), at(nx, ny));
            if a == b {
                Relation::Identity
            } else {
                Relation::Pair(table.lookup_i8(a, b))
            }
        })
        .collect();

    // Per-pixel accumulators, row-major: (W, Σxx, Σxy, Σyy).
    let mut acc = vec![(0u32, 0u32, 0i32, 0u32); PIXELS];
    for (&((x, y), (nx, ny)), &rel) in pairs.iter().zip(&relations) {
        // Spelled out here rather than calling `weight_of`, so a defect in
        // the virtual path's weighting cannot cancel against the oracle.
        let w = match rel {
            Relation::Identity => u32::from(identity_weight),
            Relation::Pair(r) => (i32::from(r) + 127) as u32,
        };
        let (dx, dy) = (nx - x, ny - y);
        let e = &mut acc[(y * side + x) as usize];
        e.0 += w;
        e.1 += w * (dx * dx) as u32;
        e.2 += w as i32 * dx * dy;
        e.3 += w * (dy * dy) as u32;
    }

    let mut surfels = Vec::new();
    for code in 0..PIXELS as u16 {
        let amp = act[code as usize];
        if amp == 0 {
            continue;
        }
        let m = Morton8x8::from_code(code);
        let (w, sxx, sxy, syy) = acc[usize::from(m.y()) * side as usize + usize::from(m.x())];
        surfels.push(SurfelReading {
            center: m,
            amplitude: amp,
            weight: w,
            sxx,
            sxy,
            syy,
        });
    }
    let inv = OracleInventory {
        pair_bytes: pairs.len() * size_of::<((i32, i32), (i32, i32))>(),
        relation_bytes: relations.len() * size_of::<Relation>(),
        surfel_bytes: surfels.len() * size_of::<SurfelReading>(),
    };
    (surfels, inv)
}

/// A end to end: the materialized surfels into the same moment register.
fn oracle_moment(tile: &Tile, act: &Tile, table: &FisherZTable, iw: u16) -> TileMoment {
    let mut m = TileMoment::default();
    for s in oracle_surfels(tile, act, table, iw).0 {
        m.absorb(s);
    }
    m
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (tile, act) = (repeated_tile(), activation_fixture());

    let (b, n_alloc, n_bytes) =
        allocations_during(|| virtual_moment(&tile, &act, &law, IDENTITY_WEIGHT));
    let (a_surfels, inv) = oracle_surfels(&tile, &act, &table, IDENTITY_WEIGHT);
    let a = oracle_moment(&tile, &act, &table, IDENTITY_WEIGHT);

    // Every ordinal once: how many virtual surfels already carry a strictly
    // positive definite Σ for the rendering rounds.
    let mut spd = 0;
    for_each_surfel(&permutation_tile(), &[1; PIXELS], &law, 0, |r| {
        spd += usize::from(r.sigma().is_some());
    });

    println!("D-CTX-1 virtual surfel reading, 16 x 16 Morton tile");
    println!("  law generation             : {:#018x}", law.generation);
    println!("  surfels read (activated)   : {} of {PIXELS}", b.surfels);
    println!("  A == B (tile moment)       : {}", a == b);
    println!(
        "  tile moment                : W {}  Sxx {}  Sxy {}  Syy {}",
        b.weight, b.sxx, b.sxy, b.syy
    );
    println!("  permutation tile, SPD Σ    : {spd} of {PIXELS} readings");
    println!(
        "  resident tile / activation : {} B / {} B",
        size_of::<Tile>(),
        size_of::<Tile>()
    );
    println!(
        "  borrowed law               : {} B + {} B gamma",
        law.entries.len(),
        FamilyGamma::BYTE_SIZE
    );
    println!(
        "  virtual reading state      : SurfelReading {} B, moment register {} B",
        size_of::<SurfelReading>(),
        size_of::<TileMoment>()
    );
    println!(
        "  B heap while reading       : {n_alloc} allocations, {n_bytes} B (surfel population 0 B)"
    );
    println!(
        "  A materialized             : pairs {} B, relations {} B, surfels {} B ({} surfels)",
        inv.pair_bytes,
        inv.relation_bytes,
        inv.surfel_bytes,
        a_surfels.len()
    );
}

#[cfg(test)]
mod tests {
    use super::virtual_surfel::{neighbor, read_surfel};
    use super::*;
    use cognitive_shader_driver::palette_perturbation::PaletteState;

    fn table() -> FisherZTable {
        FisherZTable::build(&representatives(1), 256)
    }

    /// FAILS IF: any streamed virtual reading differs from the materialized
    /// surfel at the same position, or the two moments differ. Fixtures:
    /// the permutation tile, the repeated tile, and 32 random tiles, each at
    /// two identity weights.
    #[test]
    fn virtual_readings_equal_the_materialized_surfels() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let mut tiles = vec![permutation_tile(), repeated_tile()];
        let mut s: u32 = 0x9E37_79B9;
        for _ in 0..32 {
            tiles.push(core::array::from_fn(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8 % 6 * 40
            }));
        }
        let act = activation_fixture();
        for tile in &tiles {
            for iw in [0, IDENTITY_WEIGHT] {
                let (oracle, _) = oracle_surfels(tile, &act, &table, iw);
                let mut i = 0;
                for_each_surfel(tile, &act, &law, iw, |r| {
                    assert_eq!(r, oracle[i], "surfel {i}");
                    i += 1;
                });
                assert_eq!(i, oracle.len());
                assert_eq!(
                    virtual_moment(tile, &act, &law, iw),
                    oracle_moment(tile, &act, &table, iw)
                );
            }
        }
    }

    /// FAILS IF: reading surfels touches the heap, or the counter is blind
    /// (the oracle must register allocations).
    #[test]
    fn the_virtual_path_allocates_no_surfel() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (repeated_tile(), activation_fixture());
        let (_, n, bytes) =
            allocations_during(|| virtual_moment(&tile, &act, &law, IDENTITY_WEIGHT));
        assert_eq!((n, bytes), (0, 0));
        let (_, n_oracle, _) =
            allocations_during(|| oracle_surfels(&tile, &act, &table, IDENTITY_WEIGHT));
        assert!(n_oracle >= 4, "counter is blind");
    }

    /// FAILS IF: a checked Morton neighbour on the 16 x 16 tile differs from
    /// row-major geometry, or the directed visit count is not 1860.
    #[test]
    fn morton_neighbours_match_geometry_on_the_16x16_tile() {
        let mut visits = 0;
        for y in 0..SIDE {
            for x in 0..SIDE {
                let c = Morton8x8::from_xy(x, y);
                assert!(c.code() < PIXELS as u16);
                for &(dx, dy) in &MOORE {
                    let (nx, ny) = (i16::from(x) + i16::from(dx), i16::from(y) + i16::from(dy));
                    let on = (0..16).contains(&nx) && (0..16).contains(&ny);
                    assert_eq!(
                        neighbor(c, dx, dy),
                        on.then(|| Morton8x8::from_xy(nx as u8, ny as u8))
                    );
                    visits += usize::from(on);
                }
            }
        }
        // 4 corners × 3 + 56 edge pixels × 5 + 196 interior × 8.
        assert_eq!(visits, 1860);
    }

    /// FAILS IF: the permutation tile does not hold all 256 ordinals once, or
    /// a reading's position is anything but its own address.
    #[test]
    fn all_256_ordinals_fit_and_position_is_the_address() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let tile = permutation_tile();
        let mut seen = [false; 256];
        for &b in &tile {
            assert!(!seen[b as usize], "ordinal {b} twice");
            seen[b as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
        let all_on = [1u8; PIXELS];
        let mut next = 0u16;
        for_each_surfel(&tile, &all_on, &law, 0, |r| {
            assert_eq!(r.center.code(), next);
            next += 1;
        });
        assert_eq!(next, PIXELS as u16);
    }

    /// FAILS IF: moving a material patch changes what its interior pixels
    /// read (placement must move the surfel, not its material reading), or a
    /// reading at the old place survives unchanged.
    #[test]
    fn placement_moves_the_surfel_and_keeps_its_material_reading() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let patch = |ox: u8, oy: u8| -> Tile {
            let mut t = [5u8; PIXELS];
            for y in 0..4u8 {
                for x in 0..4u8 {
                    t[Morton8x8::from_xy(ox + x, oy + y).code() as usize] = 60 + 4 * y + x;
                }
            }
            t
        };
        let (here, there) = (patch(2, 3), patch(9, 8));
        for y in 0..4u8 {
            for x in 0..4u8 {
                let a = read_surfel(&here, Morton8x8::from_xy(2 + x, 3 + y), 1, &law, 0);
                let b = read_surfel(&there, Morton8x8::from_xy(9 + x, 8 + y), 1, &law, 0);
                assert_ne!(a.center, b.center);
                assert_eq!(
                    (a.weight, a.sxx, a.sxy, a.syy),
                    (b.weight, b.sxx, b.sxy, b.syy),
                    "patch ({x},{y})"
                );
            }
        }
        // Anti-vacuity: at the old place the moved tile now reads background.
        let old = read_surfel(&there, Morton8x8::from_xy(3, 4), 1, &law, 0);
        let new = read_surfel(&here, Morton8x8::from_xy(3, 4), 1, &law, 0);
        assert_ne!((old.weight, old.sxx), (new.weight, new.sxx));
    }

    /// FAILS IF: swapping two pixels' palette ordinals changes nothing.
    #[test]
    fn swapping_two_ordinals_changes_the_reading() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (permutation_tile(), activation_fixture());
        let mut swapped = tile;
        let (i, j) = (
            Morton8x8::from_xy(4, 4).code() as usize,
            Morton8x8::from_xy(11, 9).code() as usize,
        );
        swapped.swap(i, j);
        assert_ne!(
            virtual_moment(&tile, &act, &law, 0),
            virtual_moment(&swapped, &act, &law, 0)
        );
    }

    /// FAILS IF: a reading built from the table diagonal (a needle question)
    /// would match the pairwise reading on this fixture.
    #[test]
    fn a_diagonal_lookup_mutant_is_caught() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (permutation_tile(), activation_fixture());
        let mut mutant = TileMoment::default();
        for code in 0..PIXELS as u16 {
            let amp = act[code as usize];
            if amp == 0 {
                continue;
            }
            let center = Morton8x8::from_code(code);
            let c = tile[code as usize];
            let mut s = SurfelReading {
                center,
                amplitude: amp,
                weight: 0,
                sxx: 0,
                sxy: 0,
                syy: 0,
            };
            for &(dx, dy) in &MOORE {
                if neighbor(center, dx, dy).is_some() {
                    let w = (i32::from(table.lookup_i8(c, c)) + 127) as u32;
                    let (dx, dy) = (i32::from(dx), i32::from(dy));
                    s.weight += w;
                    s.sxx += w * (dx * dx) as u32;
                    s.sxy += w as i32 * dx * dy;
                    s.syy += w * (dy * dy) as u32;
                }
            }
            mutant.absorb(s);
        }
        assert_ne!(virtual_moment(&tile, &act, &law, 0), mutant);
    }

    /// A stripe of material `s` through background `b`, horizontal or not.
    fn stripe(s: u8, b: u8, horizontal: bool) -> Tile {
        let mut t = [b; PIXELS];
        for i in 0..SIDE {
            let (x, y) = if horizontal { (i, 8) } else { (8, i) };
            t[Morton8x8::from_xy(x, y).code() as usize] = s;
        }
        t
    }

    /// A material pair the law calls weakly related: code in -120..=-64, so its
    /// weight is small but never zero.
    fn weak_pair(law: &PairwiseFisherZ<'_>) -> (u8, u8) {
        for s in 0..=255u8 {
            for b in 0..=255u8 {
                if s != b
                    && matches!(law.relation(PaletteState(s), PaletteState(b)), Relation::Pair(r) if (-120..=-64).contains(&r))
                {
                    return (s, b);
                }
            }
        }
        panic!("no weak pair in the fixture law");
    }

    /// FAILS IF: the reading carries no orientation from the law. On a
    /// stripe of identical material, the surfel must elongate along the
    /// stripe when identity weighs at the ceiling, turn when the stripe turns,
    /// and flip when identity weighs nothing; a uniform field must stay
    /// isotropic. The last case shows `identity_weight` is load-bearing.
    #[test]
    fn orientation_is_read_from_the_law_and_the_identity_weight_decides_it() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (s, b) = weak_pair(&law);
        let on_stripe = Morton8x8::from_xy(8, 8);

        let h = read_surfel(&stripe(s, b, true), on_stripe, 1, &law, IDENTITY_WEIGHT);
        let v = read_surfel(&stripe(s, b, false), on_stripe, 1, &law, IDENTITY_WEIGHT);
        assert!(h.sxx > h.syy, "horizontal stripe: {h:?}");
        assert!(v.syy > v.sxx, "vertical stripe: {v:?}");
        assert_eq!(
            (h.sxx, h.syy),
            (v.syy, v.sxx),
            "a turned stripe turns the surfel"
        );

        let h0 = read_surfel(&stripe(s, b, true), on_stripe, 1, &law, 0);
        assert!(
            h0.sxx < h0.syy,
            "identity at 0 flips the orientation: {h0:?}"
        );

        let uniform = [s; PIXELS];
        let u = read_surfel(&uniform, on_stripe, 1, &law, IDENTITY_WEIGHT);
        assert_eq!((u.sxx, u.sxy), (u.syy, 0), "uniform field is isotropic");
    }

    /// FAILS IF: an activation-0 pixel yields a surfel, or an activated one
    /// does not.
    #[test]
    fn gated_pixels_produce_no_surfel() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (permutation_tile(), activation_fixture());
        let mut n = 0;
        for_each_surfel(&tile, &act, &law, 0, |r| {
            assert_ne!(act[r.center.code() as usize], 0);
            n += 1;
        });
        let active = act.iter().filter(|&&a| a != 0).count();
        assert!(active < PIXELS && active > 0);
        assert_eq!(n, active);
        assert_eq!(
            virtual_moment(&tile, &[0; PIXELS], &law, 0),
            TileMoment::default()
        );
    }

    /// FAILS IF: a reading's second moment is not positive semi-definite, or
    /// no interior reading reaches the contract's strict SPD check.
    #[test]
    fn every_reading_is_psd_and_interior_ones_are_spd() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let mut spd = 0;
        for tile in [permutation_tile(), repeated_tile()] {
            for_each_surfel(&tile, &[1; PIXELS], &law, IDENTITY_WEIGHT, |r| {
                let det = i64::from(r.sxx) * i64::from(r.syy) - i64::from(r.sxy).pow(2);
                assert!(det >= 0, "{r:?}");
                if r.sigma().is_some() {
                    spd += 1;
                }
            });
        }
        assert!(spd > 300, "only {spd} SPD readings");
    }
}
