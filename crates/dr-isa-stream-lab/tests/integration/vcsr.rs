//! The virtual CSR and neighbor aggregation over it.

use dr_isa_stream_lab::bind::{World, SPACE_A, SPACE_OTHER};
use dr_isa_stream_lab::cube::{ArmRefusal, TILE};
use dr_isa_stream_lab::fixture::Fixture;
use dr_isa_stream_lab::gemm::{DirectRefusal, GemmRefusal, Mode};
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};
use dr_isa_stream_lab::spmm::{aggregate, oracle};
use dr_isa_stream_lab::vcsr::{ExplicitCsr, VirtualCsr};
use lance_graph_dt64::dr::{FrameRefusal, FrameRevisions};
use lance_graph_dt64::LaneRefusal;

fn edges(fx: &Fixture) -> Vec<(usize, usize)> {
    (0..fx.n)
        .flat_map(|i| fx.neighbors(i).map(move |j| (i, j)))
        .collect()
}

fn graph(fx: &Fixture, space: lance_graph_dt64::CoordSpace) -> (VirtualCsr, ExplicitCsr) {
    let e = edges(fx);
    (
        VirtualCsr::from_edges(space, fx.n, e.iter().copied()),
        ExplicitCsr::from_edges(fx.n, e),
    )
}

fn as_f32(rows: Vec<Vec<i64>>) -> Vec<Vec<f32>> {
    rows.into_iter()
        .map(|r| r.into_iter().map(|v| v as f32).collect())
        .collect()
}

#[test]
fn the_virtual_csr_has_the_explicit_csr_rows() {
    for n in [300, 4096] {
        let fx = Fixture {
            n,
            props: 16,
            seed: 31,
        };
        let (v, e) = graph(&fx, SPACE_A);
        for i in 0..n {
            assert_eq!(v.row(i).collect::<Vec<_>>(), e.row(i), "row {i}, n={n}");
        }
        assert_eq!(v.nnz(), e.col_idx.len());
    }
}

/// The fixture has local and long-range blocks, and is far from dense: a
/// block-sparse layout is doing real work.
#[test]
fn the_graph_is_block_sparse_with_long_range_blocks() {
    let fx = Fixture {
        n: 65_536,
        props: 16,
        seed: 32,
    };
    let (v, _) = graph(&fx, SPACE_A);
    let tiles = v.row_tiles();
    assert_eq!(tiles, 256);
    // Measured 5,186 of 65,536 tile pairs at the default rate.
    assert!(
        v.blocks() * 10 < tiles * tiles,
        "{} of {} blocks present",
        v.blocks(),
        tiles * tiles
    );
    let far = (0..tiles)
        .flat_map(|tr| v.row_tile(tr).iter().map(move |b| (tr, b.tc)))
        .filter(|&(tr, tc)| tr.abs_diff(tc) > 2 && tr.abs_diff(tc) < tiles - 2)
        .count();
    assert!(far > tiles, "only {far} long-range blocks");
}

/// The block count is set by locality: the same degree with a long-range
/// edge on every identity fills most tile pairs.
#[test]
fn long_range_edges_fill_the_block_layout() {
    let fx = Fixture {
        n: 65_536,
        props: 16,
        seed: 32,
    };
    let local = VirtualCsr::from_edges(SPACE_A, fx.n, edges(&fx)).blocks();
    let everywhere = VirtualCsr::from_edges(
        SPACE_A,
        fx.n,
        (0..fx.n).flat_map(|i| fx.neighbors_with(i, 1).map(move |j| (i, j))),
    )
    .blocks();
    // Measured 41,997 vs 5,186 blocks.
    assert!(everywhere > 5 * local, "{everywhere} vs {local} blocks");
}

#[test]
fn aggregation_is_exact_direct_and_staged() {
    for n in [256, 513, 4096, 65_536] {
        let fx = Fixture {
            n,
            props: 16,
            seed: 33 + n as u64,
        };
        let (v, e) = graph(&fx, SPACE_A);
        let want = as_f32(oracle(&fx, &e, &(0..16).collect::<Vec<_>>()));
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, 16);
        let world = World::new();
        let fields: Vec<u8> = (0..16).collect();
        let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
        let (direct, r) = aggregate(&world, &binding, &socket, &v, &fields, Mode::Direct).unwrap();
        assert_eq!(direct, want, "direct, n={n}, tier {}", r.tier);
        assert_eq!(r.a_staged_bytes, 0, "direct copied A, n={n}");
        assert_eq!(r.blocks, v.blocks());
        let (staged, r) = aggregate(&world, &binding, &socket, &v, &fields, Mode::Staged).unwrap();
        assert_eq!(staged, want, "staged, n={n}");
        assert!(r.a_staged_bytes > 0);
    }
}

#[cfg(feature = "mkl")]
#[test]
fn mkl_aggregation_reads_the_slab_in_place_and_matches() {
    use dr_isa_stream_lab::spmm::aggregate_mkl;
    for n in [513, 65_536] {
        let fx = Fixture {
            n,
            props: 16,
            seed: 34,
        };
        let (v, e) = graph(&fx, SPACE_A);
        let want = as_f32(oracle(&fx, &e, &(0..16).collect::<Vec<_>>()));
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, 16);
        let world = World::new();
        let fields: Vec<u8> = (0..16).collect();
        let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
        let (got, r) = aggregate_mkl(&world, &binding, &socket, &v, &fields).unwrap();
        assert_eq!(got, want, "MKL, n={n}");
        assert_eq!(r.gemms, v.blocks(), "one MKL call per block");
    }
}

#[test]
fn lanes_are_borrowed_once_per_aggregation() {
    for n in [256, 65_536] {
        let fx = Fixture {
            n,
            props: 16,
            seed: 35,
        };
        let (v, _) = graph(&fx, SPACE_A);
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, 16);
        let world = World::new();
        let fields: Vec<u8> = (0..16).collect();
        let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
        let before = socket.borrows.get();
        aggregate(&world, &binding, &socket, &v, &fields, Mode::Direct).unwrap();
        assert_eq!(socket.borrows.get() - before, 16, "n={n}");
    }
}

#[test]
fn a_graph_in_another_row_order_is_refused() {
    let fx = Fixture {
        n: 1024,
        props: 16,
        seed: 36,
    };
    let (v, _) = graph(&fx, SPACE_OTHER);
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 16);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    assert_eq!(
        aggregate(&world, &binding, &socket, &v, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Arm(ArmRefusal::GraphSpace {
            graph: SPACE_OTHER,
            selection: SPACE_A
        })
    );
}

#[test]
fn a_stale_frame_refuses_aggregation() {
    let fx = Fixture {
        n: 1024,
        props: 16,
        seed: 37,
    };
    let (v, _) = graph(&fx, SPACE_A);
    let mut lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world
        .bind(&Socket::in_order(&lanes, 16), &fields, SPACE_A)
        .unwrap();
    lanes.frames_mut().advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let socket = Socket::in_order(&lanes, 16);
    assert_eq!(
        aggregate(&world, &binding, &socket, &v, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Arm(ArmRefusal::Lane(LaneRefusal::Frame(
            FrameRefusal::SourceChanged
        )))
    );
}

#[test]
fn fewer_than_sixteen_lanes_refuse_direct_and_stage_correctly() {
    let fx = Fixture {
        n: 700,
        props: 5,
        seed: 38,
    };
    let (v, e) = graph(&fx, SPACE_A);
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 5);
    let world = World::new();
    let fields: Vec<u8> = (0..5).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    assert_eq!(
        aggregate(&world, &binding, &socket, &v, &fields, Mode::Direct).unwrap_err(),
        GemmRefusal::Direct(DirectRefusal::PartialRowTile { lanes: 5 })
    );
    let (got, _) = aggregate(&world, &binding, &socket, &v, &fields, Mode::Staged).unwrap();
    assert_eq!(got, as_f32(oracle(&fx, &e, &(0..5).collect::<Vec<_>>())));
}

/// The one-edge case pins orientation: row i gets lane_p[j], not lane_p[i].
#[test]
fn a_single_edge_aggregates_the_neighbor_not_the_row() {
    let fx = Fixture {
        n: TILE * 2,
        props: 16,
        seed: 39,
    };
    let (i, j) = (3usize, TILE + 77);
    let v = VirtualCsr::from_edges(SPACE_A, fx.n, [(i, j)]);
    let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
    let socket = Socket::in_order(&lanes, 16);
    let world = World::new();
    let fields: Vec<u8> = (0..16).collect();
    let binding = world.bind(&socket, &fields, SPACE_A).unwrap();
    let (y, _) = aggregate(&world, &binding, &socket, &v, &fields, Mode::Direct).unwrap();
    for (p, yp) in y.iter().enumerate() {
        assert_eq!(yp[i], fx.value(j, p) as f32, "prop {p}");
        assert_eq!(
            yp.iter().filter(|&&x| x != 0.0).count(),
            usize::from(fx.value(j, p) != 0)
        );
    }
    assert!(
        (0..16).any(|p| fx.value(i, p) != fx.value(j, p)),
        "fixture cannot tell i from j"
    );
}
