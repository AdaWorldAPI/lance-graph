//! The arm without staging: answers equal the oracle, no lane value is
//! copied, and a binding refuses a source that has moved on.

use dr_select_k_lab::counters::Counters;
use dr_select_k_lab::fixture::Pattern;
use dr_select_k_lab::{oracle, InPlace, InPlaceRefusal, Order, Version};

use crate::common;

const KS: [usize; 4] = [4, 64, 1000, 12_000];
const THRESHOLDS: [i32; 3] = [-1, 500, 990];

// Fails if any in-place fold differs from the row-at-a-time oracle.
#[test]
fn in_place_folds_equal_the_oracle() {
    let mut cells = 0;
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        for order in common::BINDINGS {
            for k in KS {
                let c = Counters::new();
                let range = common::range_for_k(k);
                let q = common::query(range, None);
                let sel = common::select(&src, range, order, &c);
                let mut b = InPlace::bind(&src, sel, &c).unwrap();
                assert_eq!(b.k(), k);
                let want_sum = oracle::sum_a(&src, q);
                assert_eq!(b.sum_a(&src, &c), Ok(want_sum));
                assert_eq!(b.via_positions_sum_a(&src, &c), Ok(want_sum));
                for t in THRESHOLDS {
                    let want = oracle::count_b_gt(&src, q, t);
                    assert_eq!(b.count_b_gt(&src, t, &c), Ok(want));
                    assert_eq!(b.via_positions_count_b_gt(&src, t, &c), Ok(want));
                    if b.is_run() {
                        assert_eq!(b.run_count_b_gt(&src, t, &c), Ok(want));
                    }
                    // The counts discriminate: neither nothing nor everything.
                    if t == 500 {
                        assert!(want > 0 && (want as usize) < k);
                    }
                    cells += 1;
                }
                if b.is_run() {
                    assert_eq!(b.run_sum_a(&src, &c), Ok(want_sum));
                }
            }
        }
    }
    assert_eq!(cells, 2 * 3 * 4 * 3);
}

// Fails if a clustered filter selection is not seen as a run, or a scattered
// one is, or the run arm answers for positions that are not a run.
#[test]
fn a_run_is_recognised_and_only_a_run() {
    let c = Counters::new();
    let range = common::range_for_k(1000);
    let clustered = common::source(Pattern::Clustered);
    let sel = common::select(&clustered, range, Order::Source, &c);
    let mut run = InPlace::bind(&clustered, sel, &c).unwrap();
    assert!(run.is_run());

    let random = common::source(Pattern::Random);
    let sel = common::select(&random, range, Order::Source, &c);
    let mut scattered = InPlace::bind(&random, sel, &c).unwrap();
    assert!(!scattered.is_run());
    assert_eq!(
        scattered.run_count_b_gt(&random, 0, &c),
        Err(InPlaceRefusal::NotARun)
    );
    assert_eq!(
        scattered.run_sum_a(&random, &c),
        Err(InPlaceRefusal::NotARun)
    );
    assert!(run.run_sum_a(&clustered, &c).is_ok());
}

// Fails if binding copies lane values, or if a fold touches storage.
#[test]
fn no_lane_value_is_copied_and_storage_does_not_move() {
    let src = common::source(Pattern::Random);
    let c = Counters::new();
    let k = 1000;
    let sel = common::select(&src, common::range_for_k(k), Order::Source, &c);
    let after_select = c.snapshot();
    assert_eq!(after_select[4], (k * 4) as u64);
    let mut b = InPlace::bind(&src, sel, &c).unwrap();
    let after_bind = c.snapshot();
    // One bit per source row, and nothing per selected row: a staged bind
    // adds 28 bytes for each of the K rows.
    let plane_bytes = (common::N.div_ceil(64) * 8) as u64;
    assert_eq!(after_bind[4] - after_select[4], plane_bytes);
    assert!(plane_bytes < (k * 28) as u64);
    assert_eq!(after_bind[0], after_select[0]);
    assert_eq!(after_bind[1], 1);
    b.count_b_gt(&src, 500, &c).unwrap();
    b.sum_a(&src, &c).unwrap();
    b.via_positions_sum_a(&src, &c).unwrap();
    let after_folds = c.snapshot();
    assert_eq!(after_folds[0], after_bind[0]);
    assert_eq!(after_folds[4], after_bind[4]);
    assert_eq!(after_folds[2], (2 * common::N + k) as u64);
}

// Fails if a stale selection binds, or if any fold answers after the source
// has moved on.
#[test]
fn a_binding_refuses_a_source_that_moved() {
    let mut src = common::source(Pattern::Clustered);
    let c = Counters::new();
    let range = common::range_for_k(64);
    let sel = common::select(&src, range, Order::Source, &c);
    let mut b = InPlace::bind(&src, sel.clone(), &c).unwrap();
    assert!(b.count_b_gt(&src, 500, &c).is_ok());
    src.advance();
    let stale = InPlaceRefusal::Stale {
        bound: Version(1),
        source: Version(2),
    };
    assert_eq!(b.count_b_gt(&src, 500, &c), Err(stale));
    assert_eq!(b.sum_a(&src, &c), Err(stale));
    assert_eq!(b.run_count_b_gt(&src, 500, &c), Err(stale));
    assert_eq!(b.run_sum_a(&src, &c), Err(stale));
    assert_eq!(b.via_positions_count_b_gt(&src, 500, &c), Err(stale));
    assert_eq!(b.via_positions_sum_a(&src, &c), Err(stale));
    assert_eq!(InPlace::bind(&src, sel, &c).err(), Some(stale));
    let fresh = common::select(&src, range, Order::Source, &c);
    assert!(InPlace::bind(&src, fresh, &c).is_ok());
}
