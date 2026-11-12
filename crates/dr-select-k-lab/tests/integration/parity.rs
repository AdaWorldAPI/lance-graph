//! Every admissible view equals the oracle.

use std::collections::HashMap;

use dr_select_k_lab::counters::Counters;
use dr_select_k_lab::fixture::Pattern;
use dr_select_k_lab::ops::{self, baseline, Inadmissible, PreparedFolds};
use dr_select_k_lab::oracle;
use dr_select_k_lab::{Bound, Order, Residual, Source, View};

use crate::common::{self, DUP};

const KS: [usize; 6] = [0, 4, 64, 256, 260, 15_000];
const T_AS: [Option<i32>; 5] = [None, Some(1), Some(50), Some(99), Some(100)];
const T_B: i32 = 500;

/// `order == Order::Source` is the filter binding; a key order is the index
/// binding of that order.
fn check_case(pattern: Pattern, order: Order, k: usize, t_a: Option<i32>) {
    let src = common::source(pattern);
    let range = common::range_for_k(k);
    let c = Counters::new();
    let bound = common::bind(&src, range, order, &c);
    // Fails if the fixture does not select the K the case names.
    assert_eq!(bound.k(), k);
    let q = common::query(range, t_a);
    let want_count = oracle::count_b_gt(&src, q, T_B);
    let want_sum = oracle::sum_a(&src, q);
    let limits = [0usize, 1, 10, k + 5];
    // The only admissible request is the binding's own order, so the oracle
    // is needed for that order alone.
    let want_tops: Vec<Vec<u128>> = if order == Order::Source {
        Vec::new()
    } else {
        limits
            .iter()
            .map(|&l| oracle::top_by_key(&src, q, l, order))
            .collect()
    };
    let mut folds = PreparedFolds::new(&bound);
    for (like, t) in common::templates(t_a) {
        let res = ops::residual_a_ge(&bound, t, &like, &c);
        let view = res.view();
        let ctx = format!("{pattern:?} binding={order:?} k={k} t_a={t_a:?}");
        assert_eq!(
            folds.count_b_gt(&bound, view, T_B, &c),
            want_count,
            "count {ctx}"
        );
        assert_eq!(folds.sum_a(&bound, view, &c), want_sum, "sum {ctx}");
        for (li, &limit) in limits.iter().enumerate() {
            for want in common::ORDERS {
                let mut out = vec![7u128];
                let r = folds.top_by_key(&bound, view, limit, want, &mut out, &c);
                if bound.order() == want {
                    assert_eq!(r, Ok(()), "top admitted want={want:?} {ctx}");
                    assert_eq!(out, want_tops[li], "top limit={limit} want={want:?} {ctx}");
                } else {
                    assert_eq!(
                        r,
                        Err(Inadmissible::OrderNotWitnessed {
                            want,
                            have: bound.order()
                        }),
                        "top refused want={want:?} {ctx}"
                    );
                }
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
                check_case(pattern, Order::Source, k, t_a);
            }
        }
    }
}

// Fails if the ordered selection, or top-by-key over it, differs from the
// oracle called with the same order, for either declared order.
#[test]
fn ordered_selection_views_equal_the_oracle_for_both_orders() {
    for order in common::ORDERS {
        for pattern in common::PATTERNS {
            for k in KS {
                for t_a in T_AS {
                    check_case(pattern, order, k, t_a);
                }
            }
        }
    }
}

/// The full top-`limit` of a binding over the implicit ALL residual.
fn top_all(bound: &Bound, limit: usize, order: Order, c: &Counters) -> Vec<u128> {
    let mut folds = PreparedFolds::new(bound);
    let mut out = Vec::new();
    folds
        .top_by_key(bound, View::All, limit, order, &mut out, c)
        .unwrap();
    out
}

/// guid -> source row, for the whole source.
fn row_of(src: &Source) -> HashMap<u128, usize> {
    let (guid, _, _, _) = src.lanes();
    guid.iter().enumerate().map(|(r, g)| (*g, r)).collect()
}

/// A full ordered result of a range with `DUP` rows per key: every chunk of
/// `DUP` guids is exactly one key, and the chunk keys ascend strictly.
fn assert_key_groups(src: &Source, seq: &[u128]) {
    let (_, key, _, _) = src.lanes();
    let row_of = row_of(src);
    let mut prev: Option<i32> = None;
    for group in seq.chunks(DUP as usize) {
        let k0 = key[row_of[&group[0]]];
        assert!(
            group.iter().all(|g| key[row_of[g]] == k0),
            "a chunk of DUP guids is one key group"
        );
        if let Some(p) = prev {
            assert!(p < k0, "key groups ascend");
        }
        prev = Some(k0);
    }
}

// Fails if, under `KeyThenGuid`, ties on the key are broken by anything but
// the guid. The distinctness test below proves the fixture can tell guid
// order from row order, so a row-order tie break cannot pass this one.
#[test]
fn key_then_guid_breaks_ties_by_guid() {
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        let range = common::range_for_k(256);
        let c = Counters::new();
        let bound = common::bind(&src, range, Order::KeyThenGuid, &c);
        let q = common::query(range, None);
        let full = top_all(&bound, 256, Order::KeyThenGuid, &c);
        assert_eq!(
            full,
            oracle::top_by_key(&src, q, 256, Order::KeyThenGuid),
            "{pattern:?}"
        );
        // A limit that cuts inside a group of DUP equal keys (6 = 4 + 2).
        let cut = top_all(&bound, 6, Order::KeyThenGuid, &c);
        assert_eq!(
            cut,
            oracle::top_by_key(&src, q, 6, Order::KeyThenGuid),
            "{pattern:?}"
        );
        assert_eq!(cut[..], full[..6], "{pattern:?}");
        assert_key_groups(&src, &full);
        for group in full.chunks(DUP as usize) {
            assert!(
                group.windows(2).all(|w| w[0] < w[1]),
                "{pattern:?}: guids ascend in a key group"
            );
        }
    }
}

// Fails if, under `KeyThenRow`, ties on the key are broken by anything but
// the source row position.
#[test]
fn key_then_row_breaks_ties_by_source_row() {
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        let range = common::range_for_k(256);
        let c = Counters::new();
        let bound = common::bind(&src, range, Order::KeyThenRow, &c);
        let q = common::query(range, None);
        let full = top_all(&bound, 256, Order::KeyThenRow, &c);
        assert_eq!(
            full,
            oracle::top_by_key(&src, q, 256, Order::KeyThenRow),
            "{pattern:?}"
        );
        let cut = top_all(&bound, 6, Order::KeyThenRow, &c);
        assert_eq!(
            cut,
            oracle::top_by_key(&src, q, 6, Order::KeyThenRow),
            "{pattern:?}"
        );
        assert_eq!(cut[..], full[..6], "{pattern:?}");
        assert_key_groups(&src, &full);
        let row_of = row_of(&src);
        for group in full.chunks(DUP as usize) {
            assert!(
                group.windows(2).all(|w| row_of[&w[0]] < row_of[&w[1]]),
                "{pattern:?}: source rows ascend in a key group"
            );
        }
    }
}

// Fails if the two contracts collapse into one: if, for every key group, the
// guid sequence is the same under both orders (for example because the
// fixture's guids rise with the row, or because an index ignored its order),
// or if a limit that cuts inside a differing group gives the same top-T for
// both. The search below must find such a group; it asserts one exists.
#[test]
fn the_two_orders_are_distinct_contracts() {
    let dup = DUP as usize;
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        let range = common::range_for_k(256);
        let q = common::query(range, None);
        let c = Counters::new();
        let b_row = common::bind(&src, range, Order::KeyThenRow, &c);
        let b_guid = common::bind(&src, range, Order::KeyThenGuid, &c);
        let by_row = top_all(&b_row, 256, Order::KeyThenRow, &c);
        let by_guid = top_all(&b_guid, 256, Order::KeyThenGuid, &c);
        assert_ne!(by_row, by_guid, "{pattern:?}: the full sequences differ");

        // Search the key groups for one whose guid sequence differs.
        let mut differing = None;
        for (gi, (r, g)) in by_row.chunks(dup).zip(by_guid.chunks(dup)).enumerate() {
            let mut rs = r.to_vec();
            let mut gs = g.to_vec();
            rs.sort_unstable();
            gs.sort_unstable();
            assert_eq!(rs, gs, "{pattern:?}: both orders hold the same group");
            if differing.is_none() && r != g {
                differing = Some(gi);
            }
        }
        let gi = differing.unwrap_or_else(|| panic!("{pattern:?}: no key group differs"));

        // Two permutations of one set that differ must differ in some proper
        // prefix, so a cut inside the group exists.
        let n = (1..dup)
            .find(|&n| by_row[gi * dup..gi * dup + n] != by_guid[gi * dup..gi * dup + n])
            .unwrap_or_else(|| panic!("{pattern:?}: no cut inside group {gi} separates"));
        let limit = gi * dup + n;
        let top_row = top_all(&b_row, limit, Order::KeyThenRow, &c);
        let top_guid = top_all(&b_guid, limit, Order::KeyThenGuid, &c);
        assert_ne!(
            top_row, top_guid,
            "{pattern:?}: top-{limit} cuts inside group {gi}"
        );
        assert_eq!(
            top_row,
            oracle::top_by_key(&src, q, limit, Order::KeyThenRow)
        );
        assert_eq!(
            top_guid,
            oracle::top_by_key(&src, q, limit, Order::KeyThenGuid)
        );
    }
}

// Fails if the residual view changes the result of a top-T over one fixed
// declared order. With threshold 0 the positions and the mask keep every row,
// so all three views must agree with each other and with the oracle; at
// threshold 50 positions and mask must still agree with each other.
#[test]
fn a_view_change_does_not_change_the_ordered_result() {
    for pattern in common::PATTERNS {
        for order in common::ORDERS {
            let src = common::source(pattern);
            let range = common::range_for_k(256);
            let c = Counters::new();
            let bound = common::bind(&src, range, order, &c);
            let k = bound.k();
            let mut folds = PreparedFolds::new(&bound);
            let all = Residual::All;
            let pos0 = ops::residual_a_ge(&bound, 0, &Residual::Positions(Vec::new()), &c);
            let mask0 = ops::residual_a_ge(&bound, 0, &Residual::Mask(Vec::new()), &c);
            assert_eq!(
                common::residual_len(&pos0, k),
                k,
                "positions keep every row"
            );
            assert_eq!(common::residual_len(&mask0, k), k, "mask keeps every row");
            let pos50 = ops::residual_a_ge(&bound, 50, &Residual::Positions(Vec::new()), &c);
            let mask50 = ops::residual_a_ge(&bound, 50, &Residual::Mask(Vec::new()), &c);
            let m50 = common::residual_len(&pos50, k);
            assert!(m50 > 0 && m50 < k, "threshold 50 is a strict subset");
            let q_all = common::query(range, None);
            let q50 = common::query(range, Some(50));
            for limit in [0usize, 1, 6, 256] {
                let ctx = format!("{pattern:?} {order:?} limit={limit}");
                let mut base = Vec::new();
                folds
                    .top_by_key(&bound, all.view(), limit, order, &mut base, &c)
                    .unwrap();
                assert_eq!(base.len(), limit.min(k), "{ctx}");
                assert_eq!(base, oracle::top_by_key(&src, q_all, limit, order), "{ctx}");
                for res in [&pos0, &mask0] {
                    let mut out = Vec::new();
                    folds
                        .top_by_key(&bound, res.view(), limit, order, &mut out, &c)
                        .unwrap();
                    assert_eq!(out, base, "{ctx}");
                }
                let mut p = Vec::new();
                let mut m = Vec::new();
                folds
                    .top_by_key(&bound, pos50.view(), limit, order, &mut p, &c)
                    .unwrap();
                folds
                    .top_by_key(&bound, mask50.view(), limit, order, &mut m, &c)
                    .unwrap();
                assert_eq!(p, m, "{ctx} t_a=50");
                assert_eq!(p, oracle::top_by_key(&src, q50, limit, order), "{ctx}");
            }
        }
    }
}

// Fails if the residual is trivially empty or full, or the forms disagree.
#[test]
fn residual_is_a_strict_nonempty_subset_and_forms_agree() {
    for pattern in common::PATTERNS {
        let src = common::source(pattern);
        let range = common::range_for_k(256);
        let c = Counters::new();
        let bound = common::bind(&src, range, Order::Source, &c);
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
