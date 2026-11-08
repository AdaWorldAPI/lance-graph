//! When A can be read in place, and what happens when it cannot.

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::cube::WhereMasks;
use dr_isa_stream_lab::fixture::{Fixture, MASKS};
use dr_isa_stream_lab::gemm::{sums, DirectRefusal, GemmRefusal, Mode};
use dr_isa_stream_lab::oracle;
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};

fn want(fx: &Fixture, props: &[usize]) -> Vec<[f32; MASKS]> {
    oracle::sums(fx, props)
        .iter()
        .map(|r| r.map(|v| v as f32))
        .collect()
}

/// Separately allocated lanes cannot be one tile load: direct refuses,
/// staging answers correctly and reports what it copied.
#[test]
fn scattered_lanes_refuse_direct_and_stage_correctly() {
    let fx = Fixture {
        n: 1024,
        props: 16,
        seed: 11,
    };
    let lanes = Bf16Lanes::scattered(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 16);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    assert_eq!(
        sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Direct(DirectRefusal::NotOneSlab { field: 1 })
    );
    let (got, report) = sums(&world, &binding, &socket, &masks, &fields, Mode::Staged).unwrap();
    assert_eq!(got, want(&fx, &(0..16).collect::<Vec<_>>()));
    assert_eq!(
        report.a_staged_bytes,
        16 * 1024 * 2,
        "every A byte staged once"
    );
}

/// A slab whose fields are not consecutive rows refuses direct, and is
/// correct staged.
#[test]
fn permuted_rows_refuse_direct_and_stage_correctly() {
    let fx = Fixture {
        n: 512,
        props: 16,
        seed: 12,
    };
    let rows: Vec<usize> = (0..16).rev().collect();
    let lanes = Bf16Lanes::slab(&fx, &rows).unwrap();
    let socket = Socket::new(&lanes, rows);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    assert_eq!(
        sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Direct(DirectRefusal::RowsNotConsecutive { field: 1 })
    );
    let (got, _) = sums(&world, &binding, &socket, &masks, &fields, Mode::Staged).unwrap();
    assert_eq!(got, want(&fx, &(0..16).collect::<Vec<_>>()));
}

/// The row check is load-bearing: the same permuted slab read as if its rows
/// were in field order gives a different answer from the oracle.
#[test]
fn without_the_row_check_a_permuted_slab_reads_wrong() {
    let fx = Fixture {
        n: 512,
        props: 16,
        seed: 12,
    };
    let rows: Vec<usize> = (0..16).rev().collect();
    let lanes = Bf16Lanes::slab(&fx, &rows).unwrap();
    // A socket that claims field f is row f: what an unchecked direct read
    // of "16 rows from the first lane's row" would assume.
    let lying = Socket::in_order(&lanes, 16);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&lying, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    let (got, _) = sums(&world, &binding, &lying, &masks, &fields, Mode::Direct).unwrap();
    assert_ne!(got, want(&fx, &(0..16).collect::<Vec<_>>()));
}

/// Fewer than 16 demanded lanes is a partial row tile: direct refuses (it
/// would read undemanded lanes), staging answers with zero rows.
#[test]
fn partial_row_tiles_are_staged() {
    for props in [3usize, 8] {
        let fx = Fixture {
            n: 700,
            props,
            seed: 13,
        };
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, props);
        let world = World::new();
        let fields: Vec<u8> = (0..props as u8).collect();
        let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
        let masks = WhereMasks::from_fixture(&fx, SPACE_A);
        assert_eq!(
            sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap_err(),
            GemmRefusal::Direct(DirectRefusal::PartialRowTile { lanes: props })
        );
        let (got, _) = sums(&world, &binding, &socket, &masks, &fields, Mode::Staged).unwrap();
        assert_eq!(
            got,
            want(&fx, &(0..props).collect::<Vec<_>>()),
            "props={props}"
        );
    }
}

/// Reusing the selection: 32 lanes are two row tiles over one staged B per
/// block, so B is staged once, not once per tile.
#[test]
fn b_is_staged_once_per_block_for_all_row_tiles() {
    let fx = Fixture {
        n: 2048,
        props: 32,
        seed: 14,
    };
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 32);
    let world = World::new();
    let fields32: Vec<u8> = (0..32).collect();
    let binding = world.bind(&socket, &fields32, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    let (_, r32) = sums(&world, &binding, &socket, &masks, &fields32, Mode::Direct).unwrap();
    let (_, r16) = sums(
        &world,
        &binding,
        &socket,
        &masks,
        &fields32[..16],
        Mode::Direct,
    )
    .unwrap();
    assert_eq!(r32.b_entries, r16.b_entries);
    assert_eq!(r32.gemms, 2 * r16.gemms);
    assert_eq!(r32.b_entries as u32, masks.counts().iter().sum::<u32>());
}

/// A pitch that leaves no padding for the last 32-wide block refuses direct
/// (the read would run into the next row or past the slab), and stages.
#[test]
fn an_unpadded_pitch_refuses_direct_and_stages_correctly() {
    let fx = Fixture {
        n: 513,
        props: 16,
        seed: 15,
    };
    let rows: Vec<usize> = (0..16).collect();
    let lanes = Bf16Lanes::slab_with_pitch(&fx, &rows, 513).unwrap();
    let socket = Socket::new(&lanes, rows);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let masks = WhereMasks::from_fixture(&fx, SPACE_A);
    assert_eq!(
        sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Direct(DirectRefusal::ShortPitch {
            pitch: 513,
            needed: 544
        })
    );
    let (got, _) = sums(&world, &binding, &socket, &masks, &fields, Mode::Staged).unwrap();
    assert_eq!(got, want(&fx, &(0..16).collect::<Vec<_>>()));
    // A padded pitch over the same values is read in place.
    let padded = Bf16Lanes::slab_with_pitch(&fx, &(0..16).collect::<Vec<_>>(), 544).unwrap();
    let socket = Socket::in_order(&padded, 16);
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let (got, r) = sums(&world, &binding, &socket, &masks, &fields, Mode::Direct).unwrap();
    assert_eq!(got, want(&fx, &(0..16).collect::<Vec<_>>()));
    assert_eq!(r.a_staged_bytes, 0);
}
