//! D-LXC-29 end to end: a SPOG context resolves a `Register128` slab ONCE,
//! binds its rails ONCE, and bounded power sums fold tile by tile through
//! `ndarray::simd` into those rails — then widen losslessly to the exact
//! `PowerSums` / `CrossPowerSums` the wide `i32` kernels produce.
//!
//! jc is the one crate that already sees both sides (ndarray as a dependency,
//! the contract as a dev-dependency), so the cross-crate proof lives here and
//! no production crate gains a dependency.

use lance_graph_contract::canonical_node::{NodeGuid, NodeRow, ReadMode, ValueSchema};
use lance_graph_contract::hotplug::{Activation, ActivationDrift, SlabDeclaration, SlabReading};
use lance_graph_contract::register128::{Register128, RegisterLanes, RegisterRails};
use lance_graph_contract::soa_envelope::ENVELOPE_LAYOUT_VERSION;
use ndarray::simd::{
    fold_bounded_cross_power_sums_tiles, fold_bounded_power_sums_tiles,
    masked_group_bounded_cross_power_sums_u8, masked_group_bounded_power_sums_u8,
    masked_group_cross_power_sums_i32, masked_group_power_sums_i32, CrossPowerSums, PowerSums,
    BOUNDED_TILE_ROWS,
};

const CONCEPT: u16 = 0x0901;
const GROUPS: usize = 7;

fn activation() -> Activation {
    Activation::new(
        Vec::new(),
        Vec::new(),
        vec![(CONCEPT, ReadMode::PLUG_AND_PLAY_V3)],
    )
}

fn register_slab() -> SlabDeclaration {
    SlabDeclaration {
        reading: SlabReading::Register128,
        value_schema: ValueSchema::Full,
        layout_version: ENVELOPE_LAYOUT_VERSION,
    }
}

/// One group-result row, addressed under the context's concept. The key's
/// identity is the group; the payload carries no classid.
fn group_row(group: usize) -> NodeRow {
    NodeRow {
        key: NodeGuid::new(u32::from(CONCEPT) << 16, 1, 2, 3, 0x66, group as u32),
        edges: Default::default(),
        value: [0; 480],
    }
}

struct Population {
    n: usize,
    mask: Vec<u64>,
    keys: Vec<u32>,
    xs: Vec<u8>,
    ys: Vec<u8>,
}

fn population() -> Population {
    let n = 3 * BOUNDED_TILE_ROWS + 777;
    let mut mask = vec![0u64; n.div_ceil(64)];
    for i in (0..n).filter(|i| i % 11 != 4) {
        mask[i / 64] |= 1 << (i % 64);
    }
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 56) as u8
    };
    let xs: Vec<u8> = (0..n).map(|_| next()).collect();
    let ys: Vec<u8> = (0..n).map(|_| next()).collect();
    let keys = (0..n as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) % GROUPS as u32)
        .collect();
    Population {
        n,
        mask,
        keys,
        xs,
        ys,
    }
}

/// Resolution and binding run exactly once, before the population loop; the
/// loop sees only `RegisterLanes` (plain numbers) and byte lanes.
fn bind() -> RegisterLanes {
    activation()
        .resolve_for_context(CONCEPT, Some(&register_slab()))
        .expect("context resolves")
        .bind_register128(RegisterRails::Two)
        .expect("Register128 slab grants both rails")
}

/// FAILS IF: the tiled bounded fold, stored per tile and per group into the
/// value-slab rails and read back, does not widen and merge to exactly the
/// wide `i32` path over the whole population — univariate (rail 0) and
/// bivariate (rails 0+1) alike.
#[test]
fn register_rails_carry_bounded_stats_that_widen_to_the_wide_path() {
    let p = population();
    let lanes = bind();
    assert_eq!(lanes.concept(), CONCEPT, "concept comes from the context");

    // Univariate: one row per (tile, group); the register is written per
    // group, never per input row.
    let mut tile_rows: Vec<Vec<NodeRow>> = Vec::new();
    let mut regs = [[0u8; 16]; GROUPS];
    let mut out = [PowerSums::default(); GROUPS];
    fold_bounded_power_sums_tiles(p.n, &mut regs, &mut out, |t, regs| {
        masked_group_bounded_power_sums_u8(
            &p.mask[t.start / 64..],
            &p.keys[t.clone()],
            &p.xs[t],
            regs,
        )?;
        let rows = (0..GROUPS)
            .map(|g| {
                let mut row = group_row(g);
                assert!(lanes.set(&mut row, 0, Register128(regs[g])));
                row
            })
            .collect();
        tile_rows.push(rows);
        Ok(())
    })
    .unwrap();
    assert_eq!(tile_rows.len(), 4, "three full tiles and one partial");

    let wx: Vec<i32> = p.xs.iter().map(|&x| i32::from(x)).collect();
    let wy: Vec<i32> = p.ys.iter().map(|&y| i32::from(y)).collect();
    let mut want = [PowerSums::default(); GROUPS];
    masked_group_power_sums_i32(&p.mask, &p.keys, &wx, &mut want);
    assert_eq!(out, want, "tiled driver == whole");

    // Independently: re-read every stored rail, widen, merge.
    let mut reread = [PowerSums::default(); GROUPS];
    for rows in &tile_rows {
        for (g, row) in rows.iter().enumerate() {
            let reg = lanes.get(row, 0).unwrap();
            let w = ndarray::simd::widen_bounded_power_sums(&reg.0);
            reread[g] = reread[g].checked_merge(w).unwrap();
        }
    }
    assert_eq!(reread, want, "rails read back == whole");
    assert!(want.iter().map(|w| w.n).sum::<u64>() > BOUNDED_TILE_ROWS as u64);
    assert!(want.iter().all(|w| w.n > 0), "every group populated");

    // Bivariate: rail 0 = [n, Σx, Σx², ·], rail 1 = [Σy, Σy², Σxy, ·].
    let (mut r0, mut r1) = ([[0u8; 16]; GROUPS], [[0u8; 16]; GROUPS]);
    let mut cross = [CrossPowerSums::default(); GROUPS];
    let mut stored: Vec<NodeRow> = Vec::new();
    fold_bounded_cross_power_sums_tiles(p.n, &mut r0, &mut r1, &mut cross, |t, a, b| {
        masked_group_bounded_cross_power_sums_u8(
            &p.mask[t.start / 64..],
            &p.keys[t.clone()],
            &p.xs[t.clone()],
            &p.ys[t],
            a,
            b,
        )?;
        for g in 0..GROUPS {
            let mut row = group_row(g);
            assert!(lanes.set(&mut row, 0, Register128(a[g])));
            assert!(lanes.set(&mut row, 1, Register128(b[g])));
            stored.push(row);
        }
        Ok(())
    })
    .unwrap();
    let mut want_cross = [CrossPowerSums::default(); GROUPS];
    masked_group_cross_power_sums_i32(&p.mask, &p.keys, &wx, &wy, &mut want_cross);
    assert_eq!(cross, want_cross, "bivariate tiled driver == whole");

    let mut reread = [CrossPowerSums::default(); GROUPS];
    for (k, row) in stored.iter().enumerate() {
        let (a, b) = (lanes.get(row, 0).unwrap(), lanes.get(row, 1).unwrap());
        let g = k % GROUPS;
        let w = ndarray::simd::widen_bounded_cross_power_sums(&a.0, &b.0);
        reread[g] = reread[g].checked_merge(w).unwrap();
    }
    assert_eq!(reread, want_cross, "bivariate rails read back == whole");
}

/// FAILS IF: a population whose slab declares Facet96 (or nothing) can be
/// bound as registers. The fold never runs on such a slab.
#[test]
fn a_facet96_slab_is_never_bound_as_registers() {
    let a = activation();
    let facet = SlabDeclaration {
        reading: SlabReading::Facet96,
        ..register_slab()
    };
    for slab in [Some(&facet), None] {
        let r = a.resolve_for_context(CONCEPT, slab).unwrap();
        assert!(matches!(
            r.bind_register128(RegisterRails::One),
            Err(ActivationDrift::NotRegister128 {
                concept: CONCEPT,
                ..
            })
        ));
    }
}
