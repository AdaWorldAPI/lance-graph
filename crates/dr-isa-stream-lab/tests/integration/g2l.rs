//! D-DT64-G2L-1: one witnessed global -> local translation, made once and
//! reused across repeated GEMMs, against an oracle computed in global ids
//! alone.

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::fixture::Fixture;
use dr_isa_stream_lab::g2l::{
    global_of, identity_of, translate, G2lRefusal, OrderedSource, TranslatedB,
};
use dr_isa_stream_lab::int8::{self, a_u8, b_i8};
use dr_isa_stream_lab::shape::{
    BOperand, BTable, ReuseRefusal, Selection, ShapeRefusal, VirtualObject,
};
use lance_graph_dt64::dr::{FrameRefusal, LocateRefusal};
use lance_graph_dt64::{Extent, LaneRefusal};

const B: BOperand = BOperand {
    seed: 0xB0B,
    rev: 1,
};
const N: usize = 4096;
const DIM: usize = 64;

fn world() -> World {
    World::with_extent(DIM, "isa:f", Extent::DR64_CUBE)
}

fn fields() -> Vec<u8> {
    (0..DIM).map(|f| f as u8).collect()
}

fn fx() -> Fixture {
    Fixture {
        n: N,
        props: DIM,
        seed: 0x6211,
    }
}

/// A permutation of `0..N` (`mul` odd, so a bijection mod 2^12).
fn order(mul: usize, add: usize) -> Vec<usize> {
    (0..N).map(|l| (l * mul + add) % N).collect()
}

/// The consumer's selection, in global ids, in the consumer's own order.
fn wanted() -> Vec<u64> {
    (0..200).map(|k| global_of((k * 977 + 5) % N)).collect()
}

/// `C[p][j] = sum over the selected global ids of A(g, p) * B(g, j)`,
/// computed from the fixture and B by identity: no map, no source order.
fn oracle(fx: &Fixture, globals: &[u64]) -> Vec<i32> {
    let mut c = vec![0i64; DIM * DIM];
    for &g in globals {
        let i = identity_of(g);
        for p in 0..DIM {
            let a = i64::from(a_u8(fx, i, p));
            for j in 0..DIM {
                c[p * DIM + j] += a * i64::from(b_i8(&B, i, j));
            }
        }
    }
    c.iter().map(|&v| i32::try_from(v).unwrap()).collect()
}

#[test]
fn a_translation_is_exact_and_reused_with_zero_further_lookups() {
    let fx = fx();
    let source = OrderedSource::new(&fx, DIM, order(1237, 11));
    let socket = source.socket(DIM);
    let world = world();
    let globals = wanted();
    let want = oracle(&fx, &globals);

    let tr = translate(&source, &globals).unwrap();
    // Every id is located once across the 16 tiles of the population.
    assert_eq!(tr.lookups(), (globals.len() * N / 256) as u64);
    let obj = VirtualObject::bind(&world, &socket, &fields(), DIM, tr.selection().clone()).unwrap();
    let bt = BTable::new(&B, N, DIM);
    let tb = TranslatedB::new(&bt, &source, &tr).unwrap();
    let prep = int8::PreparedI8B::prepare(&obj, &tb).unwrap();
    let (lookups, reads) = (source.lookups(), tb.reads());
    // Packing resolves each selected row once, not once per column.
    assert_eq!(reads, globals.len() as u64);

    let (mut a, mut c) = (Vec::new(), vec![0i32; obj.c_len()]);
    for run in 0..5 {
        let lanes = int8::resolve(&obj, &world, &socket).unwrap();
        int8::staged_prepared(&obj, &lanes, &prep, &tb, &mut a, &mut c).unwrap();
        assert_eq!(c, want, "run {run}");
    }
    assert_eq!(source.lookups(), lookups, "execution never looks an id up");
    assert_eq!(tb.reads(), reads, "execution never reads B through the map");
}

/// Without a translation, the consumer's order cannot be read on the source.
#[test]
fn the_untranslated_cross_space_read_stays_refused() {
    let fx = fx();
    let source = OrderedSource::new(&fx, DIM, order(1237, 11));
    let socket = source.socket(DIM);
    let ids = wanted().iter().map(|&g| identity_of(g)).collect::<Vec<_>>();
    let sel = Selection::mask(SPACE_A, N, ids);
    let err = VirtualObject::bind(&world(), &socket, &fields(), DIM, sel).unwrap_err();
    assert!(
        matches!(err, ShapeRefusal::Lane(LaneRefusal::CoordinateSpace { .. })),
        "{err:?}"
    );
}

#[test]
fn a_reorder_refuses_everything_prepared_under_the_old_order() {
    let fx = fx();
    let mut source = OrderedSource::new(&fx, DIM, order(1237, 11));
    let world = world();
    let globals = wanted();
    let bt = BTable::new(&B, N, DIM);
    let (tr, obj, prep) = {
        let socket = source.socket(DIM);
        let tr = translate(&source, &globals).unwrap();
        let obj =
            VirtualObject::bind(&world, &socket, &fields(), DIM, tr.selection().clone()).unwrap();
        let tb = TranslatedB::new(&bt, &source, &tr).unwrap();
        let prep = int8::PreparedI8B::prepare(&obj, &tb).unwrap();
        (tr, obj, prep)
    };

    source.reorder(&fx, order(3001, 7));
    let socket = source.socket(DIM);
    assert!(matches!(
        int8::resolve(&obj, &world, &socket),
        Err(ShapeRefusal::Lane(LaneRefusal::Frame(
            FrameRefusal::CoordinatesChanged
        )))
    ));
    assert!(matches!(tr.check(&source), Err(G2lRefusal::Stale { .. })));
    assert!(matches!(
        TranslatedB::new(&bt, &source, &tr),
        Err(G2lRefusal::Stale { .. })
    ));
    drop(prep);

    // A fresh translation under the new order is exact again.
    let tr2 = translate(&source, &globals).unwrap();
    let obj2 =
        VirtualObject::bind(&world, &socket, &fields(), DIM, tr2.selection().clone()).unwrap();
    let tb2 = TranslatedB::new(&bt, &source, &tr2).unwrap();
    let prep2 = int8::PreparedI8B::prepare(&obj2, &tb2).unwrap();
    let lanes = int8::resolve(&obj2, &world, &socket).unwrap();
    let mut c = vec![0i32; obj2.c_len()];
    int8::staged_prepared(&obj2, &lanes, &prep2, &tb2, &mut Vec::new(), &mut c).unwrap();
    assert_eq!(c, oracle(&fx, &globals));
}

/// The source reordered its lanes without its map: the map no longer
/// describes the source, so no translation is made from it.
#[test]
fn a_stale_mapping_is_refused() {
    let fx = fx();
    let mut source = OrderedSource::new(&fx, DIM, order(1237, 11));
    source.reorder_lanes_only(&fx, order(3001, 7));
    assert!(matches!(
        translate(&source, &wanted()),
        Err(G2lRefusal::MapNotSource { .. })
    ));
}

#[test]
fn unknown_and_repeated_ids_are_refused() {
    let fx = fx();
    let source = OrderedSource::new(&fx, DIM, order(1237, 11));
    assert!(matches!(
        translate(&source, &[global_of(3), 5]),
        Err(G2lRefusal::Locate {
            global: 5,
            refusal: LocateRefusal::Unmapped
        })
    ));
    assert!(matches!(
        translate(&source, &[global_of(3), global_of(9), global_of(3)]),
        Err(G2lRefusal::Duplicate { .. })
    ));
}

/// Two translations of the same selection are two operands: B prepared
/// through one is refused through the other.
#[test]
fn a_prepared_b_belongs_to_its_translation() {
    let fx = fx();
    let source = OrderedSource::new(&fx, DIM, order(1237, 11));
    let socket = source.socket(DIM);
    let world = world();
    let bt = BTable::new(&B, N, DIM);
    let tr1 = translate(&source, &wanted()).unwrap();
    let tr2 = translate(&source, &wanted()).unwrap();
    let obj =
        VirtualObject::bind(&world, &socket, &fields(), DIM, tr1.selection().clone()).unwrap();
    let tb1 = TranslatedB::new(&bt, &source, &tr1).unwrap();
    let tb2 = TranslatedB::new(&bt, &source, &tr2).unwrap();
    let prep = int8::PreparedI8B::prepare(&obj, &tb1).unwrap();
    let lanes = int8::resolve(&obj, &world, &socket).unwrap();
    let mut c = vec![0i32; obj.c_len()];
    assert!(matches!(
        int8::staged_prepared(&obj, &lanes, &prep, &tb2, &mut Vec::new(), &mut c),
        Err(ShapeRefusal::Reuse(ReuseRefusal::Operand))
    ));
    int8::staged_prepared(&obj, &lanes, &prep, &tb1, &mut Vec::new(), &mut c).unwrap();
    assert_eq!(c, oracle(&fx, &wanted()));
}

/// #1990's OperandShape contract survives translation. The inner B table
/// covers identities `0..100` only; the source holds all `N`. One selected
/// global maps to identity 3000, past the table. The source position it
/// lands on is inside `0..N`, so checking positions against the source's
/// length would admit it and packing would read past the table. Preparation
/// must refuse it, with the inner row and the table's extent, before any
/// packing.
#[test]
fn a_translated_identity_past_the_inner_table_is_refused_before_packing() {
    let fx = fx();
    let source = OrderedSource::new(&fx, DIM, order(1237, 11));
    let socket = source.socket(DIM);
    let short = BTable::new(&B, 100, DIM);
    let globals = vec![global_of(3), global_of(3000), global_of(42)];
    let tr = translate(&source, &globals).unwrap();
    let w = world();
    let obj = VirtualObject::bind(&w, &socket, &fields(), DIM, tr.selection().clone()).unwrap();
    let tb = TranslatedB::new(&short, &source, &tr).unwrap();
    // Anti-vacuity: the positions really are inside the source.
    assert!(tr.selection().last().unwrap() < N);
    match int8::PreparedI8B::prepare(&obj, &tb) {
        Err(ShapeRefusal::OperandShape { last, rows, .. }) => assert_eq!((last, rows), (3000, 100)),
        Err(other) => panic!("expected OperandShape, got {other:?}"),
        Ok(_) => panic!("expected OperandShape, got a prepared B"),
    }
    assert_eq!(tb.reads(), 0, "refused before packing read anything");
    // Silence twin: identities that all fit the table are admitted.
    let ok = vec![global_of(3), global_of(99), global_of(42)];
    let tr = translate(&source, &ok).unwrap();
    let obj = VirtualObject::bind(&w, &socket, &fields(), DIM, tr.selection().clone()).unwrap();
    let tb = TranslatedB::new(&short, &source, &tr).unwrap();
    int8::PreparedI8B::prepare(&obj, &tb).expect("every identity is inside the table");
}

/// A translation belongs to the source whose map table issued its frame.
/// Two sources built in the same order have equal coordinate digests, so a
/// coordinate check alone admits a translation from the other one; packing
/// would then read the second table with a frame it never issued. It must be
/// refused before any packing.
#[test]
fn a_translation_from_another_source_in_the_same_order_is_refused() {
    let fx = fx();
    let first = OrderedSource::new(&fx, DIM, order(1237, 11));
    let second = OrderedSource::new(&fx, DIM, order(1237, 11));
    let bt = BTable::new(&B, N, DIM);
    let tr = translate(&first, &wanted()).unwrap();
    // Anti-vacuity: the two sources really agree on coordinates.
    assert_eq!(first.coordinates(), second.coordinates());
    assert!(matches!(
        TranslatedB::new(&bt, &second, &tr),
        Err(G2lRefusal::Foreign)
    ));
    assert!(matches!(tr.check(&second), Err(G2lRefusal::Foreign)));
    // Silence twin: the issuing source still admits it.
    assert!(TranslatedB::new(&bt, &first, &tr).is_ok());
}
