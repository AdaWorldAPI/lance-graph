//! The lane socket refuses what it must, and a passing bind borrows the
//! source's own lanes.

use lance_graph_dt64::dr::{FrameRefusal, Scope};
use lance_graph_dt64::LaneRefusal;
use soa64_u64_lab::arms::{Query, SoaU64, K};
use soa64_u64_lab::fixture::Fixture;
use soa64_u64_lab::oracle::oracle;
use soa64_u64_lab::socket::{
    bind_demanded, reordered_b, sum_where_via_socket, U64Source, World, READER, SPACE_A,
    SPACE_OTHER,
};

const SEED: u64 = 0x5EED_0064;
const N: usize = 513;

fn setup() -> (Fixture, SoaU64) {
    let fx = Fixture::new(SEED, N);
    let soa = SoaU64::build(&fx);
    (fx, soa)
}

#[test]
fn one_coordinate_space_binds_sums_exactly_and_borrows_in_place() {
    let (fx, soa) = setup();
    let world = World::new();
    let source = U64Source::new(&soa);
    let binding = bind_demanded(&world, &source, &[0, 1], SPACE_A).unwrap();
    assert_eq!(source.borrows.get(), 0, "binding holds addresses only");

    let got = sum_where_via_socket(&world, &binding, &source).unwrap();
    assert_eq!(got, oracle(&fx, &Query::SumWhere));

    let a = binding
        .lane(&world.store, &world.view, soa.frames(), &source, 0)
        .unwrap();
    let b = binding
        .lane(&world.store, &world.view, soa.frames(), &source, 1)
        .unwrap();
    assert!(std::ptr::eq(a, soa.lane(0)));
    assert!(std::ptr::eq(b, soa.lane(1)));
}

#[test]
fn a_lane_in_another_row_order_is_refused() {
    let (_fx, soa) = setup();
    let world = World::new();
    let mut source = U64Source::new(&soa);
    source.spaces[1] = SPACE_OTHER;
    assert_eq!(
        bind_demanded(&world, &source, &[0, 1], SPACE_A).err(),
        Some(LaneRefusal::CoordinateSpace {
            field: 1,
            selection: SPACE_A,
            lane: SPACE_OTHER,
        })
    );
}

#[test]
fn without_the_refusal_a_reordered_lane_reads_wrong() {
    let (fx, soa) = setup();
    let b = reordered_b(&fx);
    assert_eq!(b.len(), N);
    assert_eq!(b[0], soa.lane(1)[N - 1], "reversed row order");

    // What a missing guard would do: read the other-order lane against A.
    let mut s = 0u128;
    for (&a, &y) in soa.lane(0).iter().zip(&b) {
        if a == K {
            s += y as u128;
        }
    }
    assert_ne!(s, oracle(&fx, &Query::SumWhere));
}

#[test]
fn a_lane_for_another_scope_is_refused() {
    let (_fx, soa) = setup();
    let world = World::new();
    let mut source = U64Source::new(&soa);
    source.scope = Scope { id: 8 };
    assert_eq!(
        bind_demanded(&world, &source, &[0, 1], SPACE_A).err(),
        Some(LaneRefusal::Scope {
            field: 0,
            reader: READER,
            lane: Scope { id: 8 },
        })
    );
}

/// Swapping the SoA under a bound source swaps the issuer with it: the same
/// fixture rebuilt is equal data but another store, and the binding made over
/// the first is refused, never read, against the second.
#[test]
fn a_source_over_swapped_data_is_refused() {
    let (fx, soa) = setup();
    let world = World::new();
    let source = U64Source::new(&soa);
    let binding = bind_demanded(&world, &source, &[0, 1], SPACE_A).unwrap();

    let twin = SoaU64::build(&fx);
    let swapped = U64Source {
        soa: &twin,
        ..U64Source::new(&soa)
    };
    assert_eq!(
        sum_where_via_socket(&world, &binding, &swapped).err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    assert_eq!(swapped.borrows.get(), 0, "refused before any borrow");
    // The original pairing still reads.
    assert_eq!(
        sum_where_via_socket(&world, &binding, &source).unwrap(),
        oracle(&fx, &Query::SumWhere)
    );
}
