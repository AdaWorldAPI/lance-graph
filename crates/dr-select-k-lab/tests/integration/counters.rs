//! The counters: no second read after the bind, and what a conversion costs.

use dr_select_k_lab::counters::Counters;
use dr_select_k_lab::fixture::Pattern;
use dr_select_k_lab::ops::{self, PreparedFolds};
use dr_select_k_lab::oracle;
use dr_select_k_lab::Residual;

use crate::common;

const K: usize = 256;

// Fails if produce-and-bind costs other than one select plus one take.
#[test]
fn produce_and_bind_is_two_storage_accesses_one_bind() {
    for ordered in [false, true] {
        let src = common::source(Pattern::Random);
        let c = Counters::new();
        let bound = common::bind(&src, common::range_for_k(K), ordered, &c);
        let s = c.snapshot();
        assert_eq!(s[0], 2, "storage ordered={ordered}");
        assert_eq!(s[1], 1, "binds ordered={ordered}");
        assert_eq!(s[3], 0, "conversions");
        assert_eq!(s[4], (bound.k() * 28) as u64, "bytes copied by take");
    }
}

// Fails if any residual build or fold touches storage or binds again.
#[test]
fn no_second_read_after_the_bind() {
    let src = common::source(Pattern::Random);
    let c = Counters::new();
    let bound = common::bind(&src, common::range_for_k(K), true, &c);
    let after_bind = c.snapshot();
    let mut folds = PreparedFolds::new(&bound);
    let pos = ops::residual_a_ge(&bound, 50, &Residual::Positions(Vec::new()), &c);
    let mask = ops::residual_a_ge(&bound, 50, &Residual::Mask(Vec::new()), &c);
    let all = Residual::All;
    let before_folds = c.snapshot();
    for _ in 0..3 {
        folds.count_b_gt(&bound, mask.view(), 500, &c);
    }
    for _ in 0..2 {
        folds.sum_a(&bound, mask.view(), &c);
    }
    // 5 folds over the K-bit mask: K operand rows each.
    assert_eq!(c.snapshot()[2] - before_folds[2], 5 * K as u64);
    let before_all = c.snapshot()[2];
    folds.count_b_gt(&bound, all.view(), 500, &c);
    folds.sum_a(&bound, all.view(), &c);
    assert_eq!(c.snapshot()[2] - before_all, 2 * K as u64);
    // Positions hand the fold only the surviving rows.
    let m = common::residual_len(&pos, bound.k()) as u64;
    assert!(m > 0 && m < K as u64);
    let before_pos = c.snapshot()[2];
    folds.count_b_gt(&bound, pos.view(), 500, &c);
    folds.sum_a(&bound, pos.view(), &c);
    assert_eq!(c.snapshot()[2] - before_pos, 2 * m);
    let mut out = Vec::new();
    folds
        .top_by_key(&bound, all.view(), 10, &mut out, &c)
        .unwrap();
    let end = c.snapshot();
    assert_eq!(end[0], after_bind[0], "storage moved after bind");
    assert_eq!(end[1], after_bind[1], "bind count moved after bind");
    assert!(end[2] > after_bind[2], "operand rows must grow");
}

// Fails if ALL costs a conversion or a copy, or a stored form costs other
// than one conversion and the bytes of its result.
#[test]
fn conversions_and_bytes_per_form() {
    let src = common::source(Pattern::Clustered);
    let range = common::range_for_k(K);
    let c = Counters::new();
    let bound = common::bind(&src, range, false, &c);
    let k = bound.k();
    let t = 50;
    let m = oracle::count_b_gt(&src, common::query(range, Some(t)), -1) as usize;
    assert!(m > 0 && m < k);

    let s0 = c.snapshot();
    let all = ops::residual_a_ge(&bound, t, &Residual::All, &c);
    let s1 = c.snapshot();
    assert!(matches!(all, Residual::All));
    assert_eq!((s1[3], s1[4]), (s0[3], s0[4]), "ALL adds nothing");

    let pos = ops::residual_a_ge(&bound, t, &Residual::Positions(Vec::new()), &c);
    let s2 = c.snapshot();
    assert_eq!(s2[3] - s1[3], 1);
    assert_eq!(s2[4] - s1[4], (m * 4) as u64);

    let mask = ops::residual_a_ge(&bound, t, &Residual::Mask(Vec::new()), &c);
    let s3 = c.snapshot();
    assert_eq!(s3[3] - s2[3], 1);
    assert_eq!(s3[4] - s2[4], (k.div_ceil(64) * 8) as u64);

    assert_eq!(all.retained_bytes(), 0);
    assert_eq!(pos.retained_bytes(), m * 4);
    assert_eq!(mask.retained_bytes(), k.div_ceil(64) * 8);
}
