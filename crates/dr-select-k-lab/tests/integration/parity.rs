//! Every admissible view equals the oracle.

use std::collections::HashMap;

use dr_select_k_lab::counters::Counters;
use dr_select_k_lab::fixture::Pattern;
use dr_select_k_lab::ops::{self, baseline, Inadmissible, PreparedFolds};
use dr_select_k_lab::oracle;
use dr_select_k_lab::{Order, Residual};

use crate::common::{self, DUP};

const KS: [usize; 6] = [0, 4, 64, 256, 260, 15_000];
const T_AS: [Option<i32>; 5] = [None, Some(1), Some(50), Some(99), Some(100)];
const T_B: i32 = 500;

fn check_case(pattern: Pattern, ordered: bool, k: usize, t_a: Option<i32>) {
    let src = common::source(pattern);
    let range = common::range_for_k(k);
    let c = Counters::new();
    let bound = common::bind(&src, range, ordered, &c);
    // Fails if the fixture does not select the K the case names.
    assert_eq!(bound.k(), k);
    let q = common::query(range, t_a);
    let want_count = oracle::count_b_gt(&src, q, T_B);
    let want_sum = oracle::sum_a(&src, q);
    let mut folds = PreparedFolds::new(&bound);
    for (like, t) in common::templates(t_a) {
        let res = ops::residual_a_ge(&bound, t, &like, &c);
        let view = res.view();
        let ctx = format!("{pattern:?} ordered={ordered} k={k} t_a={t_a:?}");
        assert_eq!(
            folds.count_b_gt(&bound, view, T_B, &c),
            want_count,
            "count {ctx}"
        );
        assert_eq!(folds.sum_a(&bound, view, &c), want_sum, "sum {ctx}");
        for limit in [0usize, 1, 10, k + 5] {
            let mut out = vec![7u128];
            let r = folds.top_by_key(&bound, view, limit, &mut out, &c);
            if ordered {
                assert_eq!(r, Ok(()), "top admitted {ctx}");
                assert_eq!(
                    out,
                    oracle::top_by_key(&src, q, limit),
                    "top limit={limit} {ctx}"
                );
            } else {
                assert_eq!(
                    r,
                    Err(Inadmissible::OrderNotWitnessed {
                        have: Order::Source
                    }),
                    "top refused {ctx}"
                );
            }
        }
    }
}

// Fails if any view differs from the oracle on any K, residual or pattern.
#[test]
fn filter_selection_views_equal_the_oracle() {
    for pattern in common::PATTERNS {
        for k in KS {
            for t_a in T_AS {
                check_case(pattern, false, k, t_a);
            }
        }
    }
}

// Fails if the ordered selection, or top-by-key over it, differs from the oracle.
#[test]
fn ordered_selection_views_equal_the_oracle() {
    for pattern in common::PATTERNS {
        for k in KS {
            for t_a in T_AS {
                check_case(pattern, true, k, t_a);
            }
        }
    }
}

// Fails if ties on the key are broken by source row instead of guid.
#[test]
fn ties_are_broken_by_guid_not_by_row() {
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        let range = common::range_for_k(256);
        let c = Counters::new();
        let bound = common::bind(&src, range, true, &c);
        let q = common::query(range, None);
        let mut folds = PreparedFolds::new(&bound);
        let res = Residual::All;
        let mut full = Vec::new();
        folds
            .top_by_key(&bound, res.view(), 256, &mut full, &c)
            .unwrap();
        assert_eq!(full, oracle::top_by_key(&src, q, 256));
        // A limit that cuts inside a group of DUP equal keys (6 = 4 + 2).
        let mut cut = Vec::new();
        folds
            .top_by_key(&bound, res.view(), 6, &mut cut, &c)
            .unwrap();
        assert_eq!(cut, oracle::top_by_key(&src, q, 6));
        assert_eq!(cut[..], full[..6]);
        // Within each group of DUP equal keys the guids ascend; and for at
        // least one group the source rows do not (so row-order tie-breaking
        // would have given another answer). Needs orchestrator check: the
        // second assertion depends on the fixture's guids for this seed.
        let (guid, _, _, _) = src.lanes();
        let row_of: HashMap<u128, usize> = guid.iter().enumerate().map(|(r, g)| (*g, r)).collect();
        let mut groups_out_of_row_order = 0;
        for group in full.chunks(DUP as usize) {
            assert!(
                group.windows(2).all(|w| w[0] < w[1]),
                "guids ascend in a group"
            );
            if !group.windows(2).all(|w| row_of[&w[0]] < row_of[&w[1]]) {
                groups_out_of_row_order += 1;
            }
        }
        assert!(
            groups_out_of_row_order > 0,
            "{pattern:?}: guid order equals row order everywhere; pick another seed"
        );
    }
}

// Fails if the residual is trivially empty or full, or the forms disagree.
#[test]
fn residual_is_a_strict_nonempty_subset_and_forms_agree() {
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        let range = common::range_for_k(256);
        let c = Counters::new();
        let bound = common::bind(&src, range, false, &c);
        let k = bound.k();
        let mut folds = PreparedFolds::new(&bound);
        let pos = ops::residual_a_ge(&bound, 50, &Residual::Positions(Vec::new()), &c);
        let mask = ops::residual_a_ge(&bound, 50, &Residual::Mask(Vec::new()), &c);
        let m = common::residual_len(&pos, k);
        assert!(m > 0 && m * 3 < k * 2, "m={m} k={k}");
        assert_eq!(common::residual_len(&mask, k), m);
        if let Residual::Positions(p) = &pos {
            assert!(p.windows(2).all(|w| w[0] < w[1]));
        } else {
            panic!("positions form expected");
        }
        let cp = folds.count_b_gt(&bound, pos.view(), T_B, &c);
        let cm = folds.count_b_gt(&bound, mask.view(), T_B, &c);
        assert_eq!(cp, cm);
        assert!(cp > 0);
        // The three views at threshold 0 (keeps every row): same non-zero count.
        let all = Residual::All;
        let pos0 = ops::residual_a_ge(&bound, 0, &Residual::Positions(Vec::new()), &c);
        let mask0 = ops::residual_a_ge(&bound, 0, &Residual::Mask(Vec::new()), &c);
        let ca = folds.count_b_gt(&bound, all.view(), T_B, &c);
        assert!(ca > 0 && ca > cp);
        assert_eq!(folds.count_b_gt(&bound, pos0.view(), T_B, &c), ca);
        assert_eq!(folds.count_b_gt(&bound, mask0.view(), T_B, &c), ca);
    }
}

// Fails if the resident baseline over N rows differs from the oracle.
#[test]
fn resident_baseline_equals_the_oracle() {
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        for k in KS {
            for t_a in T_AS {
                let range = common::range_for_k(k);
                let q = common::query(range, t_a);
                let plane = common::selected_plane(&src, &q);
                let c = Counters::new();
                assert_eq!(
                    baseline::count_b_gt(&src, &plane, T_B, &c),
                    oracle::count_b_gt(&src, q, T_B),
                    "{pattern:?} k={k} t_a={t_a:?}"
                );
                assert_eq!(
                    baseline::sum_a(&src, &plane, &c),
                    oracle::sum_a(&src, q),
                    "{pattern:?} k={k} t_a={t_a:?}"
                );
            }
        }
    }
}
