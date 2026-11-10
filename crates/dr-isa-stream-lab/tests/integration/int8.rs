//! D-DT64-GEMM-SHAPE-3: u8 x i8 -> i32 over the same virtual object.

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::fixture::Fixture;
use dr_isa_stream_lab::int8::{self, U8Lanes, U8Socket};
use dr_isa_stream_lab::shape::{
    oracle as bf16_oracle, BOperand, Selection, ShapeRefusal, VirtualObject,
};
use lance_graph_dt64::dr::{FrameRefusal, FrameRevisions};
use lance_graph_dt64::{Extent, LaneRefusal};

const B: BOperand = BOperand {
    seed: 0xB0B,
    rev: 1,
};

fn world(dim: usize) -> World {
    let extent = if dim <= 64 {
        Extent::DR64_CUBE
    } else {
        Extent::DR256_SQ_X64
    };
    World::with_extent(dim, "isa:f", extent)
}

fn fields(dim: usize) -> Vec<u8> {
    (0..dim).map(|f| f as u8).collect()
}

fn fx(n: usize, props: usize) -> Fixture {
    Fixture {
        n,
        props,
        seed: 0x5A9E_0000 + (n * props) as u64,
    }
}

fn run(
    world: &World,
    socket: &U8Socket<'_>,
    dim: usize,
    sel: Selection,
) -> (Vec<i32>, int8::Int8Report) {
    let obj = VirtualObject::bind(world, socket, &fields(dim), dim, sel).unwrap();
    let lanes = int8::resolve(&obj, world, socket).unwrap();
    let mut c = vec![0i32; obj.c_len()];
    let r = int8::staged(&obj, &lanes, &B, &mut Vec::new(), &mut Vec::new(), &mut c).unwrap();
    (c, r)
}

/// Exact against the i64 oracle at both sizes, over a region, K tails 63
/// and 65 (padded to 64 and 128), and a scattered mask.
#[test]
fn int8_is_exact_at_both_sizes_tails_and_masks() {
    for dim in [64, 256] {
        let fx = fx(4096, dim);
        let lanes = U8Lanes::slab_in_order(&fx);
        let socket = U8Socket::in_order(&lanes, dim);
        let world = world(dim);
        let sels = [
            Selection::region(SPACE_A, 1024, dim),
            Selection::region(SPACE_A, 1000, 63),
            Selection::region(SPACE_A, 1000, 65),
            Selection::mask(SPACE_A, 4096, (0..40).map(|k| (k * 97 + 13) % 4096)),
        ];
        for sel in sels {
            let want: Vec<i32> = int8::oracle(&fx, &sel, &fields(dim), dim, &B)
                .iter()
                .map(|&v| i32::try_from(v).unwrap())
                .collect();
            let (got, r) = run(&world, &socket, dim, sel.clone());
            assert_eq!(got, want, "dim {dim}, {:?}", sel.kind);
            assert_eq!(r.k_pad, sel.len().div_ceil(64) * 64);
            assert_eq!(r.a_copied, dim * sel.len(), "one byte per selected value");
        }
    }
}

/// u8 A is unsigned: 200 x (-1) summed over K is -200 K. Read as i8, 200
/// would be -56 and the sum +56 K.
#[test]
fn a_is_never_reinterpreted_as_signed() {
    let dim = 64;
    let lanes = U8Lanes::from_fn(256, dim, |_, _| 200);
    let socket = U8Socket::in_order(&lanes, dim);
    let world = world(dim);
    let sel = Selection::region(SPACE_A, 0, 64);
    let obj = VirtualObject::bind(&world, &socket, &fields(dim), dim, sel).unwrap();
    let l = int8::resolve(&obj, &world, &socket).unwrap();
    let mut a = Vec::new();
    int8::stage_a(&obj, &l, &mut a).unwrap();
    let bm = vec![-1i8; 64 * dim];
    let mut c = vec![0i32; obj.c_len()];
    int8::gemm(&obj, &a, &bm, &mut c);
    assert!(c.iter().all(|&v| v == -200 * 64), "got {}", c[0]);
}

/// The INT8 and BF16 arms see the same fixture: A_u8 = A + 8, so
/// C_int8 = C_bf16 + 8 x colsum(B) over the selection, exactly.
#[test]
fn int8_and_bf16_agree_on_the_same_fixture() {
    let dim = 64;
    let fx = fx(4096, dim);
    let sel = Selection::region(SPACE_A, 512, 64);
    let bf = bf16_oracle(
        &sel,
        &fields(dim),
        dim,
        |i, p| f64::from(fx.value(i, p)),
        |i, j| f64::from(B.value(i, j)),
    );
    let mut colsum = vec![0i64; dim];
    sel.for_each(|_, i| (0..dim).for_each(|j| colsum[j] += i64::from(B.value(i, j))));
    let lanes = U8Lanes::slab_in_order(&fx);
    let socket = U8Socket::in_order(&lanes, dim);
    let (got, _) = run(&world(dim), &socket, dim, sel);
    for (idx, &g) in got.iter().enumerate() {
        assert_eq!(i64::from(g), bf[idx] as i64 + 8 * colsum[idx % dim]);
    }
}

/// The binding refusals hold for u8 lanes too, and foreign or out-of-range
/// lanes are refused before staging.
#[test]
fn int8_refusals() {
    let dim = 64;
    let fx = fx(1024, dim);
    let world = world(dim);
    let sel = Selection::region(SPACE_A, 0, 64);

    let mut lanes = U8Lanes::slab_in_order(&fx);
    let obj = {
        let socket = U8Socket::in_order(&lanes, dim);
        VirtualObject::bind(&world, &socket, &fields(dim), dim, sel.clone()).unwrap()
    };
    lanes.frames_mut().advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let socket = U8Socket::in_order(&lanes, dim);
    assert_eq!(
        int8::resolve(&obj, &world, &socket).unwrap_err(),
        ShapeRefusal::Lane(LaneRefusal::Frame(FrameRefusal::SourceChanged))
    );

    // Mapping changed after the bind.
    let mut remapped = U8Socket::in_order(&lanes, dim);
    let obj = VirtualObject::bind(&world, &remapped, &fields(dim), dim, sel.clone()).unwrap();
    remapped.rows.swap(0, 1);
    assert_eq!(
        int8::resolve(&obj, &world, &remapped).unwrap_err(),
        ShapeRefusal::Lane(LaneRefusal::MappingChanged { field: 0 })
    );

    // Lanes resolved from another source.
    let other = U8Lanes::slab_in_order(&fx);
    let other_socket = U8Socket::in_order(&other, dim);
    let other_obj =
        VirtualObject::bind(&world, &other_socket, &fields(dim), dim, sel.clone()).unwrap();
    let foreign = int8::resolve(&other_obj, &world, &other_socket).unwrap();
    assert_eq!(
        int8::stage_a(&obj, &foreign, &mut Vec::new()).unwrap_err(),
        ShapeRefusal::ForeignLanes
    );

    // A selection past the source.
    let past = VirtualObject::bind(
        &world,
        &socket,
        &fields(dim),
        dim,
        Selection::region(SPACE_A, 1000, 64),
    )
    .unwrap();
    let l = int8::resolve(&past, &world, &socket).unwrap();
    assert_eq!(
        int8::stage_a(&past, &l, &mut Vec::new()).unwrap_err(),
        ShapeRefusal::OutOfRange {
            last: 1063,
            n: 1024
        }
    );
}

/// One borrow per field, whatever the universe.
#[test]
fn int8_resolution_borrows_one_lane_per_field() {
    for n in [256, 65_536] {
        let fx = fx(n, 256);
        let lanes = U8Lanes::slab_in_order(&fx);
        let socket = U8Socket::in_order(&lanes, 256);
        let world = world(256);
        let obj = VirtualObject::bind(
            &world,
            &socket,
            &fields(256),
            256,
            Selection::region(SPACE_A, 0, 64),
        )
        .unwrap();
        let before = socket.borrows.get();
        int8::resolve(&obj, &world, &socket).unwrap();
        assert_eq!(socket.borrows.get() - before, 256);
    }
}

/// B is borrowed in place from a table for a full-width region (0 bytes
/// written) and materialized otherwise; both are exact. The region copy of A
/// and the per-value walk stage the same bytes.
#[test]
fn b_in_place_and_region_copy_are_exact_and_counted() {
    use dr_isa_stream_lab::shape::BTable;
    let dim = 64;
    let fx = fx(4096, dim);
    let lanes = U8Lanes::slab_in_order(&fx);
    let socket = U8Socket::in_order(&lanes, dim);
    let world = world(dim);
    let table = BTable::new(&B, 4096, dim);
    let cases = [
        (Selection::region(SPACE_A, 1024, 64), true),
        (Selection::region(SPACE_A, 1024, 63), false), // K pads to 64
        (Selection::mask(SPACE_A, 4096, [5, 77, 300]), false),
    ];
    for (sel, in_place) in cases {
        let obj = VirtualObject::bind(&world, &socket, &fields(dim), dim, sel.clone()).unwrap();
        let l = int8::resolve(&obj, &world, &socket).unwrap();
        let mut c = vec![0i32; obj.c_len()];
        let r = int8::staged(&obj, &l, &table, &mut Vec::new(), &mut Vec::new(), &mut c).unwrap();
        let want: Vec<i32> = int8::oracle(&fx, &sel, &fields(dim), dim, &B)
            .iter()
            .map(|&v| i32::try_from(v).unwrap())
            .collect();
        assert_eq!(c, want, "{:?}", sel.kind);
        assert_eq!(r.b_materialized == 0, in_place, "{:?}", sel.kind);
        let (mut a1, mut a2) = (Vec::new(), Vec::new());
        int8::stage_a(&obj, &l, &mut a1).unwrap();
        int8::stage_a_per_value(&obj, &l, &mut a2).unwrap();
        assert_eq!(a1, a2);
    }
}
