//! Every arm equals the independent oracle, exactly, on every query.

use soa64_u64_lab::arms::{
    Aos, Arm, HashRow, NarrowLane, Query, SoaBitpack, SoaNarrow, SoaU64, K,
};
use soa64_u64_lab::fixture::{present, probes, value, Fixture};
use soa64_u64_lab::oracle::oracle;

const SEED: u64 = 0x5EED_0064;

fn arms(fx: &Fixture) -> Vec<Box<dyn Arm>> {
    vec![
        Box::new(Aos::build(fx)),
        Box::new(SoaU64::build(fx)),
        Box::new(SoaNarrow::build(fx)),
        Box::new(SoaBitpack::build(fx)),
        Box::new(HashRow::build(fx)),
    ]
}

#[test]
fn every_arm_equals_the_oracle_on_every_query() {
    for n in [64usize, 256, 257, 513, 4096] {
        let fx = Fixture::new(SEED, n);
        let pr = probes(SEED, n, 1024);
        let mut queries = vec![
            Query::CountWhere,
            Query::Exists,
            Query::SumWhere,
            Query::SumPresentC,
            Query::Point { probes: &pr },
        ];
        for d in [1usize, 3, 8, 32, 64] {
            queries.push(Query::DemandFold { d });
        }
        for arm in arms(&fx) {
            for q in &queries {
                assert_eq!(
                    arm.run(q),
                    oracle(&fx, q),
                    "arm {} n {} query {:?}",
                    arm.name(),
                    n,
                    q
                );
            }
        }
    }
}

#[test]
fn narrow_lane_widths_are_narrow_but_field_one_is_u64() {
    let fx = Fixture::new(SEED, 513);
    let s = SoaNarrow::build(&fx);
    assert!(matches!(s.lane(0), NarrowLane::U8(_)));
    assert!(matches!(s.lane(1), NarrowLane::U64(_)));
    assert!(matches!(s.lane(2), NarrowLane::U16(_)));
}

#[test]
fn the_fixture_is_not_vacuous() {
    let n = 513;
    let max_rows = (0..n).filter(|&i| value(SEED, i, 1) == u64::MAX).count();
    assert!(max_rows >= 5, "only {max_rows} u64::MAX rows");

    let n = 4096;
    let fx = Fixture::new(SEED, n);
    let selected = (0..n).filter(|&i| value(SEED, i, 0) == K).count();
    assert!(selected > 0);
    let absent = (0..n).filter(|&i| !present(SEED, i, 2)).count();
    assert!(absent > 0, "no absent C values");

    // Absence matters: treating absent as present changes the answer.
    let with_absent_as_present: u128 = (0..n)
        .filter(|&i| value(SEED, i, 0) == K)
        .map(|i| value(SEED, i, 2) as u128)
        .sum();
    assert_ne!(oracle(&fx, &Query::SumPresentC), with_absent_as_present);

    // The u128 accumulator matters: the exact sum exceeds u64.
    assert!(oracle(&fx, &Query::SumWhere) > u64::MAX as u128);
}
