//! D-DT64-GEMM-SHAPE-1/2: the virtual 64 x 64 and 256 x 256 objects, four
//! arms, and every refusal the shape depends on.

use dr_isa_stream_lab::bind::{World, READER, SPACE_A, SPACE_OTHER};
use dr_isa_stream_lab::fixture::{bf16_exact, bf16_round, bf16_value, Fixture};
use dr_isa_stream_lab::gemm::DirectRefusal;
use dr_isa_stream_lab::shape::{
    cblas, direct, oracle, staged, ArmReport, BOperand, PreparedB, ReuseRefusal, Selection,
    ShapeRefusal, VirtualObject, TILE,
};
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};
use lance_graph_dt64::dr::{FrameRefusal, FrameRevisions, Scope};
use lance_graph_dt64::{CoordSpace, Extent, LaneRefusal};
use ndarray::simd::bf16_tile_gemm_16x16_packed_strided;

const B: BOperand = BOperand {
    seed: 0xB0B,
    rev: 1,
};
/// The row order a property-major (transposed) slab is really in.
const SPACE_PROPS: CoordSpace = CoordSpace {
    population: 2,
    coordinates: 1,
};

fn world(dim: usize) -> World {
    let extent = if dim <= 64 {
        Extent::DR64_CUBE
    } else {
        Extent::DR256_SQ_X64
    };
    World::with_extent(dim, "isa:f", extent)
}

fn fx(n: usize, props: usize) -> Fixture {
    Fixture {
        n,
        props,
        seed: 0x5A9E_0000 + (n * props) as u64,
    }
}

fn fields(dim: usize) -> Vec<u8> {
    (0..dim).map(|f| f as u8).collect()
}

fn want(fx: &Fixture, sel: &Selection, dim: usize) -> Vec<f32> {
    oracle(
        sel,
        &fields(dim),
        dim,
        |i, p| f64::from(fx.value(i, p)),
        |i, j| f64::from(B.value(i, j)),
    )
    .iter()
    .map(|&v| v as f32)
    .collect()
}

struct Run {
    direct: Result<(Vec<f32>, ArmReport), ShapeRefusal>,
    staged: (Vec<f32>, ArmReport),
    cblas: (Vec<f32>, ArmReport),
}

fn run(world: &World, socket: &Socket<'_>, dim: usize, sel: Selection) -> Run {
    let obj = VirtualObject::bind(world, socket, &fields(dim), dim, sel).unwrap();
    let lanes = obj.resolve(world, socket).unwrap();
    let (prep, _) = PreparedB::prepare(&obj, &B);
    let mut c = vec![0f32; obj.c_len()];
    let direct = direct(&obj, &lanes, &prep, B.rev, &mut c).map(|r| (c.clone(), r));
    let mut c = vec![0f32; obj.c_len()];
    let r = staged(&obj, &lanes, &prep, B.rev, &mut Vec::new(), &mut c).unwrap();
    let staged = (c, r);
    let mut c = vec![0f32; obj.c_len()];
    let r = cblas(&obj, &lanes, &B, &mut Vec::new(), &mut Vec::new(), &mut c).unwrap();
    Run {
        direct,
        staged,
        cblas: (c, r),
    }
}

/// All four arms agree exactly at 64 x 64 and 256 x 256, over a region of the
/// universe, and each reports what it did.
#[test]
fn four_arms_agree_exactly_at_both_sizes() {
    for dim in [64, 256] {
        let fx = fx(4096, dim);
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, dim);
        let world = world(dim);
        let sel = Selection::region(SPACE_A, 1024, dim);
        let want = want(&fx, &sel, dim);
        let r = run(&world, &socket, dim, sel);
        let (d, dr) = r.direct.unwrap();
        assert_eq!(d, want, "direct, {dim}");
        assert_eq!(r.staged.0, want, "staged, {dim}");
        assert_eq!(r.cblas.0, want, "cblas, {dim}");
        let tiles = (dim / TILE) * (dim / TILE);
        assert_eq!((dr.tiles, dr.calls, dr.a_copied), (tiles, tiles, 0));
        // 64: 16 tiles x 2 steps = 32. 256: 256 tiles x 8 steps = 2048.
        assert_eq!(dr.dp_steps, if dim == 64 { 32 } else { 2048 });
        assert_eq!(r.staged.1.a_copied, dim * dim * 2, "8 KiB / 128 KiB of A");
        assert_eq!(r.cblas.1.b_materialized, dim * dim * 2);
    }
}

/// The fixture is not degenerate: every output tile differs from every
/// other, so a misplaced tile cannot pass. Swapping two tiles fails.
#[test]
fn tiles_are_distinct_and_their_order_is_checked() {
    let dim = 64;
    let fx = fx(4096, dim);
    let sel = Selection::region(SPACE_A, 0, dim);
    let c = want(&fx, &sel, dim);
    let tile = |g: usize, t: usize| -> Vec<f32> {
        (0..TILE)
            .flat_map(|r| c[(g * TILE + r) * dim + t * TILE..][..TILE].to_vec())
            .collect()
    };
    let all: Vec<Vec<f32>> = (0..4)
        .flat_map(|g| (0..4).map(move |t| (g, t)))
        .map(|(g, t)| tile(g, t))
        .collect();
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            assert_ne!(all[i], all[j], "tiles {i} and {j} equal");
        }
    }
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, dim);
    let (mut got, _) = run(&world(dim), &socket, dim, sel).direct.unwrap();
    assert_eq!(got, c);
    for r in 0..TILE {
        let (a, b) = (r * dim, r * dim + TILE);
        for k in 0..TILE {
            got.swap(a + k, b + k);
        }
    }
    assert_ne!(got, c, "swapping tile (0,0) with (0,1) must be caught");
}

/// K = 32, 64 and 256 accumulate exactly; the step count follows K.
#[test]
fn k_steps_of_32_64_and_256_accumulate_exactly() {
    let dim = 64;
    let fx = fx(4096, dim);
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, dim);
    let world = world(dim);
    for k in [32, 64, 256] {
        let sel = Selection::region(SPACE_A, 512, k);
        let want = want(&fx, &sel, dim);
        let (got, r) = run(&world, &socket, dim, sel).direct.unwrap();
        assert_eq!(got, want, "K = {k}");
        assert_eq!(r.dp_steps, 16 * k / 32);
    }
}

/// Tails 63/64/65: at the end of the universe the padding is the slab's own
/// zeros and direct is exact; in the middle direct refuses (it would read
/// live identities) and staging is exact.
#[test]
fn tails_read_padding_at_the_end_and_refuse_in_the_middle() {
    let dim = 64;
    let world = world(dim);
    for len in [63, 64, 65] {
        let fx = fx(1024 + len, dim);
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, dim);
        let end = Selection::region(SPACE_A, 1024, len);
        let (got, _) = run(&world, &socket, dim, end.clone()).direct.unwrap();
        assert_eq!(got, want(&fx, &end, dim), "end, len {len}");

        let mid = Selection::region(SPACE_A, 500, len);
        let r = run(&world, &socket, dim, mid.clone());
        if len == 64 {
            assert_eq!(r.direct.unwrap().0, want(&fx, &mid, dim));
        } else {
            assert!(
                matches!(r.direct, Err(ShapeRefusal::OverRead { .. })),
                "mid, len {len}"
            );
        }
        assert_eq!(r.staged.0, want(&fx, &mid, dim), "staged mid, len {len}");
        assert_eq!(r.cblas.0, want(&fx, &mid, dim), "cblas mid, len {len}");
    }
}

/// The over-read refusal is load-bearing: a NaN in the identity just past a
/// 63-wide region poisons a padded in-place read even though B's padding row
/// is zero (0 * NaN = NaN). Staging the same region is clean.
#[test]
fn without_the_over_read_refusal_a_neighbouring_nan_poisons_the_tile() {
    let dim = 64;
    let fx = fx(1024, dim);
    let (start, len) = (256, 63);
    let lanes = Bf16Lanes::slab_encoded(1024, dim, |i, p| {
        if i == start + len {
            0x7FC0 // bf16 NaN
        } else {
            bf16_exact(fx.value(i, p)).unwrap()
        }
    });
    let socket = Socket::in_order(&lanes, dim);
    let world = world(dim);
    let sel = Selection::region(SPACE_A, start, len);
    let r = run(&world, &socket, dim, sel.clone());
    assert!(matches!(r.direct, Err(ShapeRefusal::OverRead { .. })));
    assert_eq!(
        r.staged.0,
        want(&fx, &sel, dim),
        "staging never reads the NaN"
    );

    // The read the refusal prevents, done by hand.
    let obj = VirtualObject::bind(&world, &socket, &fields(dim), dim, sel).unwrap();
    let (prep, _) = PreparedB::prepare(&obj, &B);
    let row0 = socket.lanes.row(0).unwrap();
    let mut tile = [0f32; TILE * TILE];
    bf16_tile_gemm_16x16_packed_strided(&row0.slab[start..], row0.pitch, prep.panel(0), &mut tile);
    assert!(
        tile.iter().any(|v| v.is_nan()),
        "the padded read must see the NaN"
    );
}

/// Sparse demand: a scattered WHERE mask. Direct refuses; staging and CBLAS
/// are exact. Logical selection, bytes copied and dense arithmetic are
/// different numbers: the zero padding is still multiplied.
#[test]
fn a_scattered_selection_is_staged_and_its_arithmetic_stays_dense() {
    let dim = 256;
    let fx = fx(4096, dim);
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, dim);
    let ids: Vec<usize> = (0..40).map(|k| (k * 97 + 13) % 4096).collect();
    let sel = Selection::mask(SPACE_A, 4096, ids.iter().copied());
    assert_eq!((sel.len(), sel.k_pad()), (40, 64));
    let want = want(&fx, &sel, dim);
    let r = run(&world(dim), &socket, dim, sel);
    assert_eq!(r.direct.unwrap_err(), ShapeRefusal::NotARegion);
    assert_eq!(r.staged.0, want);
    assert_eq!(r.cblas.0, want);
    assert_eq!(
        r.staged.1.a_copied,
        dim * 40 * 2,
        "only selected values copied"
    );
    assert_eq!(
        r.staged.1.dp_steps,
        256 * 2,
        "but K is padded to 64: 2 steps per tile"
    );
}

/// Lanes in separate allocations, or in a permuted row order, are not one
/// tile load: direct refuses with the reason, staging is exact.
#[test]
fn scattered_and_permuted_lanes_refuse_direct_and_stage_exactly() {
    let dim = 64;
    let fx = fx(2048, dim);
    let world = world(dim);
    let sel = Selection::region(SPACE_A, 64, dim);
    let want = want(&fx, &sel, dim);

    let lanes = Bf16Lanes::scattered(&fx).unwrap();
    let socket = Socket::in_order(&lanes, dim);
    let r = run(&world, &socket, dim, sel.clone());
    assert_eq!(
        r.direct.unwrap_err(),
        ShapeRefusal::Direct(DirectRefusal::NotOneSlab { field: 1 })
    );
    assert_eq!(r.staged.0, want);

    let rows: Vec<usize> = (0..dim).rev().collect();
    let lanes = Bf16Lanes::slab(&fx, &rows).unwrap();
    let socket = Socket::new(&lanes, rows);
    let r = run(&world, &socket, dim, sel);
    assert_eq!(
        r.direct.unwrap_err(),
        ShapeRefusal::Direct(DirectRefusal::RowsNotConsecutive { field: 1 })
    );
    assert_eq!(r.staged.0, want);
}

/// A pitch without block padding refuses a direct read of the tail region.
#[test]
fn a_short_pitch_refuses_direct() {
    let dim = 64;
    let fx = fx(4001, dim);
    let rows: Vec<usize> = (0..dim).collect();
    let lanes = Bf16Lanes::slab_with_pitch(&fx, &rows, 4001).unwrap();
    let socket = Socket::new(&lanes, rows);
    let sel = Selection::region(SPACE_A, 3937, 64);
    let r = run(&world(dim), &socket, dim, sel.clone());
    assert_eq!(
        r.direct.unwrap_err(),
        ShapeRefusal::Direct(DirectRefusal::ShortPitch {
            pitch: 4001,
            needed: 4032
        })
    );
    assert_eq!(r.staged.0, want(&fx, &sel, dim));
}

/// A square property-major slab has the same shape as the lanes. Declared in
/// the identity row order it reads wrong; declared in its true row order it
/// is refused at bind.
#[test]
fn a_transposed_square_is_wrong_unless_refused() {
    let dim = 64;
    let fx = fx(64, dim);
    // Row r holds identity r's properties: lane "p" is really identity p.
    let lanes = Bf16Lanes::slab_encoded(64, dim, |i, p| bf16_exact(fx.value(p, i)).unwrap());
    let world = world(dim);
    let sel = Selection::region(SPACE_A, 0, 64);
    let socket = Socket::in_order(&lanes, dim);
    let (got, _) = run(&world, &socket, dim, sel.clone()).direct.unwrap();
    assert_ne!(
        got,
        want(&fx, &sel, dim),
        "a transposed operand must read wrong"
    );

    let mut socket = Socket::in_order(&lanes, dim);
    socket.spaces = vec![SPACE_PROPS; dim];
    match VirtualObject::bind(&world, &socket, &fields(dim), dim, sel) {
        Err(ShapeRefusal::Lane(LaneRefusal::CoordinateSpace { field: 0, .. })) => {}
        other => panic!("expected a coordinate-space refusal, got {other:?}"),
    }
}

/// A selection in another row order than the lanes is refused at bind.
#[test]
fn a_cross_space_selection_is_refused() {
    let dim = 64;
    let fx = fx(1024, dim);
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, dim);
    let sel = Selection::region(SPACE_OTHER, 0, 64);
    assert!(matches!(
        VirtualObject::bind(&world(dim), &socket, &fields(dim), dim, sel),
        Err(ShapeRefusal::Lane(LaneRefusal::CoordinateSpace { .. }))
    ));
}

/// Stale frame, foreign frame, wrong reader scope, changed mapping and a
/// changed basis are all refused before any lane is borrowed.
#[test]
fn frame_scope_mapping_and_basis_refusals() {
    let dim = 64;
    let fx = fx(1024, dim);
    let world = world(dim);
    let sel = Selection::region(SPACE_A, 0, 64);

    // Stale: the source advances after the bind.
    let mut lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let obj = {
        let socket = Socket::in_order(&lanes, dim);
        VirtualObject::bind(&world, &socket, &fields(dim), dim, sel.clone()).unwrap()
    };
    lanes.frames_mut().advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let socket = Socket::in_order(&lanes, dim);
    let before = socket.borrows.get();
    assert_eq!(
        obj.resolve(&world, &socket).unwrap_err(),
        ShapeRefusal::Lane(LaneRefusal::Frame(FrameRefusal::SourceChanged))
    );
    assert_eq!(socket.borrows.get(), before, "nothing borrowed");

    // Foreign: the same layout, another owner.
    let other = Bf16Lanes::slab_in_order(&fx).unwrap();
    let other_socket = Socket::in_order(&other, dim);
    let obj = VirtualObject::bind(&world, &socket, &fields(dim), dim, sel.clone()).unwrap();
    assert_eq!(
        obj.resolve(&world, &other_socket).unwrap_err(),
        ShapeRefusal::Lane(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );

    // Scope: the lanes serve another reader.
    let mut scoped = Socket::in_order(&lanes, dim);
    scoped.scope = Scope { id: READER.id + 1 };
    assert!(matches!(
        VirtualObject::bind(&world, &scoped, &fields(dim), dim, sel.clone()),
        Err(ShapeRefusal::Lane(LaneRefusal::Scope { .. }))
    ));

    // Mapping: the source moves field 0 to another lane after the bind.
    let mut remapped = Socket::in_order(&lanes, dim);
    let obj = VirtualObject::bind(&world, &remapped, &fields(dim), dim, sel.clone()).unwrap();
    remapped.rows.swap(0, 1);
    assert_eq!(
        obj.resolve(&world, &remapped).unwrap_err(),
        ShapeRefusal::Lane(LaneRefusal::MappingChanged { field: 0 })
    );

    // Basis: checked against a store that binds the class over other fields.
    let other_world = World::with_extent(dim, "isa:g", Extent::DR64_CUBE);
    assert!(matches!(
        obj.resolve(&other_world, &socket).unwrap_err(),
        ShapeRefusal::Lane(LaneRefusal::StoreBasisMismatch { .. })
    ));
}

/// Prepared B is packed once and serves every output tile and repeated runs;
/// it is refused after a selection change, a frame change, a new B revision,
/// or for another width. Using a stale one would be wrong.
#[test]
fn prepared_b_is_reused_and_invalidated() {
    for dim in [64, 256] {
        let fx = fx(4096, dim);
        let mut lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let world = world(dim);
        let sel = Selection::region(SPACE_A, 1024, dim);
        let want1 = want(&fx, &sel, dim);
        let (prep, rep, obj) = {
            let socket = Socket::in_order(&lanes, dim);
            let obj = VirtualObject::bind(&world, &socket, &fields(dim), dim, sel.clone()).unwrap();
            let (prep, rep) = PreparedB::prepare(&obj, &B);
            let l = obj.resolve(&world, &socket).unwrap();
            for _ in 0..3 {
                let mut c = vec![0f32; obj.c_len()];
                let r = direct(&obj, &l, &prep, B.rev, &mut c).unwrap();
                assert_eq!(c, want1);
                assert_eq!(r.tiles, (dim / 16) * (dim / 16));
            }
            (prep, rep, obj)
        };
        assert_eq!(
            rep.panels,
            dim / 16,
            "one panel per 16 columns serves every row tile"
        );
        assert_eq!(rep.bytes_written, dim * dim * 2);

        assert_eq!(prep.admit(&obj, B.rev + 1), Err(ReuseRefusal::Operand));
        let socket = Socket::in_order(&lanes, dim);
        let moved = Selection::region(SPACE_A, 2048, dim);
        let obj2 = VirtualObject::bind(&world, &socket, &fields(dim), dim, moved.clone()).unwrap();
        assert_eq!(prep.admit(&obj2, B.rev), Err(ReuseRefusal::Selection));
        let narrow = VirtualObject::bind(&world, &socket, &fields(dim), 32, sel.clone()).unwrap();
        assert_eq!(prep.admit(&narrow, B.rev), Err(ReuseRefusal::Columns));

        // Load-bearing: obj2's A against obj1's B gives the wrong answer.
        let l2 = obj2.resolve(&world, &socket).unwrap();
        let lane0 = l2.lanes()[0];
        let mut tile = [0f32; 256];
        bf16_tile_gemm_16x16_packed_strided(
            &lane0.slab[2048..],
            lane0.pitch,
            prep.panel(0),
            &mut tile,
        );
        let want2 = want(&fx, &moved, dim);
        assert_ne!(tile[..16], want2[..16], "a stale B must read wrong");

        drop(socket);
        lanes.frames_mut().advance(FrameRevisions {
            source: 2,
            relation_map: 1,
            coordinates: 1,
        });
        let socket = Socket::in_order(&lanes, dim);
        let rebound = VirtualObject::bind(&world, &socket, &fields(dim), dim, sel).unwrap();
        assert_eq!(prep.admit(&rebound, B.rev), Err(ReuseRefusal::Frame));
    }
}

/// Inexact values are refused at ingest; values rounded to bf16 on purpose
/// agree across arms to f32 accumulation order and with the unrounded f64
/// answer to within bf16's input error.
#[test]
fn rounded_inputs_meet_the_documented_tolerance() {
    assert!(bf16_exact(257).is_none(), "257 is not exact in bf16");
    let dim = 64;
    let n = 1024;
    let raw = |i: usize, p: usize| ((i * 131 + p * 17) % 997) as f32 * 0.0137 - 6.3;
    let lanes = Bf16Lanes::slab_encoded(n, dim, |i, p| bf16_round(raw(i, p)));
    let socket = Socket::in_order(&lanes, dim);
    let sel = Selection::region(SPACE_A, 128, 256);
    let r = run(&world(dim), &socket, dim, sel.clone());
    let d = r.direct.unwrap().0;
    let on_rounded = oracle(
        &sel,
        &fields(dim),
        dim,
        |i, p| f64::from(bf16_value(bf16_round(raw(i, p)))),
        |i, j| f64::from(B.value(i, j)),
    );
    let on_raw = oracle(
        &sel,
        &fields(dim),
        dim,
        |i, p| f64::from(raw(i, p)),
        |i, j| f64::from(B.value(i, j)),
    );
    let mag = oracle(
        &sel,
        &fields(dim),
        dim,
        |i, p| f64::from(raw(i, p)).abs(),
        |i, j| f64::from(B.value(i, j)).abs(),
    );
    for (idx, &m) in mag.iter().enumerate() {
        // f32 accumulation over K = 256: |err| <= K * 2^-24 * sum|a*b|, with
        // a factor 2 for the rounded-vs-raw magnitude and order differences.
        let acc = 2.0 * 256.0 * f64::powi(2.0, -24) * m + 1e-6;
        for (arm, got) in [
            ("direct", &d),
            ("staged", &r.staged.0),
            ("cblas", &r.cblas.0),
        ] {
            assert!(
                (f64::from(got[idx]) - on_rounded[idx]).abs() <= acc,
                "{arm} vs rounded at {idx}"
            );
        }
        // bf16 round-to-nearest (8-bit significand): |a - bf16(a)| <= 2^-8 |a|;
        // B is exact.
        let input = f64::powi(2.0, -8) * m;
        assert!(
            (f64::from(d[idx]) - on_raw[idx]).abs() <= input + acc,
            "direct vs raw at {idx}"
        );
    }
}

/// Routing needs no global G2L: one resolve borrows exactly one lane per
/// demanded field, whatever the universe size.
#[test]
fn resolution_borrows_one_lane_per_field_independent_of_universe() {
    for n in [256, 65_536] {
        for dim in [64, 256] {
            let fx = fx(n, dim);
            let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
            let socket = Socket::in_order(&lanes, dim);
            let world = world(dim);
            let sel = Selection::region(SPACE_A, 0, 64.min(n));
            let obj = VirtualObject::bind(&world, &socket, &fields(dim), dim, sel).unwrap();
            let before = socket.borrows.get();
            obj.resolve(&world, &socket).unwrap();
            assert_eq!(
                socket.borrows.get() - before,
                dim as u64,
                "n {n}, dim {dim}"
            );
        }
    }
}

/// Lanes resolved by one object cannot feed another: another source (and
/// so another frame issue), another field count, or another lane mapping is
/// refused by every arm before any arithmetic.
#[test]
fn lanes_from_another_object_are_refused_by_every_arm() {
    let fx = fx(4096, 256);
    let world256 = world(256);
    let world64 = world(64);
    let a = Bf16Lanes::slab_in_order(&fx).unwrap();
    let b = Bf16Lanes::slab_in_order(&fx).unwrap();
    // `sb` is over the SAME source as `sa`, so "narrow" differs only in count.
    let (sa, sb) = (Socket::in_order(&a, 256), Socket::in_order(&a, 64));
    let sel = Selection::region(SPACE_A, 0, 64);
    let obj = VirtualObject::bind(&world256, &sa, &fields(256), 256, sel.clone()).unwrap();
    let (prep, _) = PreparedB::prepare(&obj, &B);
    let own = obj.resolve(&world256, &sa).unwrap();
    assert!(obj.admit_lanes(&own).is_ok());

    // Another source: identical bytes, another owner.
    let sb256 = Socket::in_order(&b, 256);
    let other = VirtualObject::bind(&world256, &sb256, &fields(256), 256, sel.clone()).unwrap();
    let foreign = other.resolve(&world256, &sb256).unwrap();
    // Too few rows: 64 lanes offered to the 256-field object.
    let narrow_obj = VirtualObject::bind(&world64, &sb, &fields(64), 256, sel.clone()).unwrap();
    let narrow = narrow_obj.resolve(&world64, &sb).unwrap();
    // Same source and frame, another mapping (rows reversed).
    let rev_rows: Vec<usize> = (0..256).rev().collect();
    let sr = Socket::new(&a, rev_rows);
    let remapped_obj = VirtualObject::bind(&world256, &sr, &fields(256), 256, sel).unwrap();
    let remapped = remapped_obj.resolve(&world256, &sr).unwrap();

    for (name, lanes) in [
        ("foreign", &foreign),
        ("narrow", &narrow),
        ("remapped", &remapped),
    ] {
        let mut c = vec![0f32; obj.c_len()];
        assert_eq!(
            direct(&obj, lanes, &prep, B.rev, &mut c).unwrap_err(),
            ShapeRefusal::ForeignLanes,
            "direct, {name}"
        );
        assert_eq!(
            staged(&obj, lanes, &prep, B.rev, &mut Vec::new(), &mut c).unwrap_err(),
            ShapeRefusal::ForeignLanes,
            "staged, {name}"
        );
        assert_eq!(
            cblas(&obj, lanes, &B, &mut Vec::new(), &mut Vec::new(), &mut c).unwrap_err(),
            ShapeRefusal::ForeignLanes,
            "cblas, {name}"
        );
        assert!(c.iter().all(|&v| v == 0.0), "nothing computed, {name}");
    }
}
