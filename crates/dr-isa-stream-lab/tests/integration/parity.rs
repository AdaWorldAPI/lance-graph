//! Both arms against the fixture oracle, over one binding.

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::cube::{fold, WhereMasks};
use dr_isa_stream_lab::fixture::{Fixture, MASKS};
use dr_isa_stream_lab::gemm::{sums, sums_whole, Mode, SelectionB};
use dr_isa_stream_lab::oracle;
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};

fn as_f32(rows: &[[i64; MASKS]]) -> Vec<[f32; MASKS]> {
    rows.iter().map(|r| r.map(|v| v as f32)).collect()
}

/// One binding, both arms, the oracle: exact for every listed size.
#[test]
fn one_binding_serves_the_cube_fold_and_the_tile_gemm_exactly() {
    for (n, props) in [(256, 16), (513, 16), (4096, 32), (65_536, 16)] {
        let fx = Fixture {
            n,
            props,
            seed: 0x15A_0000 + n as u64,
        };
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, props);
        let world = World::new();
        let fields: Vec<u8> = (0..props as u8).collect();
        let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
        let masks = WhereMasks::from_fixture(&fx, SPACE_A);
        let props_us: Vec<usize> = (0..props).collect();
        let want = oracle::sums(&fx, &props_us);

        let cube = fold(&world, &binding, &socket, &masks, &fields).unwrap();
        assert_eq!(cube, want, "cube fold, n={n}");

        let (direct, report) =
            sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap();
        assert_eq!(
            direct,
            as_f32(&want),
            "direct GEMM, n={n}, tier {}",
            report.tier
        );
        assert_eq!(report.a_staged_bytes, 0, "direct read copied A, n={n}");
        assert_eq!(report.direct_tiles, props / 16);

        let (staged, report) =
            sums(&world, &binding, &socket, &masks, &fields, Mode::Staged).unwrap();
        assert_eq!(staged, as_f32(&want), "staged GEMM, n={n}");
        assert!(report.a_staged_bytes > 0);

        let sel = SelectionB::stage(&masks);
        let (whole, report) =
            sums_whole(&world, &binding, &socket, &sel, &fields, Mode::Direct).unwrap();
        assert_eq!(whole, as_f32(&want), "whole-lane direct GEMM, n={n}");
        assert_eq!((report.a_staged_bytes, report.gemms), (0, props / 16));
        let (whole, report) =
            sums_whole(&world, &binding, &socket, &sel, &fields, Mode::Staged).unwrap();
        assert_eq!(whole, as_f32(&want), "whole-lane staged GEMM, n={n}");
        assert!(report.a_staged_bytes > 0);
    }
}

/// The fixture is not degenerate: masks differ in density and the sums are
/// not all equal, so a wrong mask or lane cannot pass by coincidence.
#[test]
fn the_fixture_discriminates() {
    let fx = Fixture {
        n: 4096,
        props: 16,
        seed: 7,
    };
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    let counts = masks.counts();
    assert!(
        counts.windows(2).all(|w| w[0] < w[1]),
        "mask densities {counts:?}"
    );
    assert!(counts[0] > 0 && (counts[MASKS - 1] as usize) < fx.n);
    let want = oracle::sums(&fx, &(0..16).collect::<Vec<_>>());
    let distinct: std::collections::BTreeSet<i64> = want.iter().flatten().copied().collect();
    assert!(
        distinct.len() > 100,
        "only {} distinct sums",
        distinct.len()
    );
}

/// The binding resolves each demanded lane once per arm, independent of n:
/// no per-identity lookup happens on the hot path.
#[test]
fn lanes_are_resolved_once_per_arm_not_per_identity() {
    for n in [256, 65_536] {
        let fx = Fixture {
            n,
            props: 16,
            seed: 3,
        };
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, 16);
        let world = World::new();
        let fields: Vec<u8> = (0..16).collect();
        let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
        let masks = WhereMasks::from_fixture(&fx, SPACE_A);
        let before = socket.borrows.get();
        sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap();
        assert_eq!(socket.borrows.get() - before, 16, "GEMM borrows, n={n}");
        let before = socket.borrows.get();
        fold(&world, &binding, &socket, &masks, &fields).unwrap();
        assert_eq!(socket.borrows.get() - before, 16, "fold borrows, n={n}");
    }
}

/// Which GEMM tier ran is reported, never assumed.
#[test]
fn the_tier_is_reported() {
    let fx = Fixture {
        n: 256,
        props: 16,
        seed: 1,
    };
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 16);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    let (_, report) = sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap();
    eprintln!("GEMM tier: {}", report.tier);
    if !report.tier.starts_with("AMX") {
        eprintln!(
            "SKIP (honestly): AMX not available; the direct path ran on {}",
            report.tier
        );
    }
}
