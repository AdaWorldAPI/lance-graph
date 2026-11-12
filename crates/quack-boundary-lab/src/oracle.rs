//! Scalar oracles. Each reads logical rows one at a time through
//! [`Source::row`] and names rows by their values, never by a row ordinal a
//! program was lowered with. None calls Quack, mask-RISC or a SIMD facade.

use crate::{Source, REGIONS, ROWS};

/// `SUM(qty) WHERE lo <= id < hi AND amount > t`.
pub fn sum_qty_id_in_amount_gt(s: &Source, lo: i32, hi: i32, t: i32) -> i64 {
    let mut sum = 0i64;
    for r in 0..ROWS {
        let row = s.row(r);
        if row.id >= lo && row.id < hi && row.amount > t {
            sum += i64::from(row.qty);
        }
    }
    sum
}

/// `COUNT(*) WHERE amount > t`.
pub fn count_amount_gt(s: &Source, t: i32) -> usize {
    (0..ROWS).filter(|&r| s.row(r).amount > t).count()
}

/// `SUM(qty) WHERE amount > t`.
pub fn sum_qty_amount_gt(s: &Source, t: i32) -> i64 {
    let mut sum = 0i64;
    for r in 0..ROWS {
        let row = s.row(r);
        if row.amount > t {
            sum += i64::from(row.qty);
        }
    }
    sum
}

/// One region's `(COUNT(*), SUM(qty), MIN(qty), MAX(qty))`.
pub type RegionFold = (i64, i64, Option<i32>, Option<i32>);

/// Per region, over `amount > t AND region <> skip`. MIN and MAX are `None`
/// for a region no row reached.
pub fn by_region_amount_gt(s: &Source, t: i32, skip: u32) -> Vec<RegionFold> {
    let mut out = vec![(0i64, 0i64, None, None); REGIONS as usize];
    for r in 0..ROWS {
        let row = s.row(r);
        if row.amount > t && row.region != skip {
            let g = &mut out[row.region as usize];
            g.0 += 1;
            g.1 += i64::from(row.qty);
            g.2 = Some(g.2.map_or(row.qty, |m: i32| m.min(row.qty)));
            g.3 = Some(g.3.map_or(row.qty, |m: i32| m.max(row.qty)));
        }
    }
    out
}

/// The rows with `amount > t`, as one bit per row ordinal of `s`.
pub fn mask_amount_gt(s: &Source, t: i32) -> Vec<u64> {
    let mut words = vec![0u64; ROWS.div_ceil(64)];
    for r in 0..ROWS {
        if s.row(r).amount > t {
            words[r / 64] |= 1u64 << (r % 64);
        }
    }
    words
}

/// `COUNT(*) WHERE discount > t`, or `WHERE NOT (discount > t)` when
/// `negated`, under SQL three-valued logic: a NULL discount makes either
/// predicate UNKNOWN, and `WHERE` keeps only TRUE.
pub fn count_discount_gt_3vl(s: &Source, t: i32, negated: bool) -> usize {
    (0..ROWS)
        .filter(|&r| match s.row(r).discount {
            None => false,
            Some(d) => (d > t) != negated,
        })
        .count()
}

/// `COUNT(*) FROM orders o WHERE EXISTS (partner p WHERE p.row = o.partner
/// AND p.tier > t)`, against the partners table as it is now.
pub fn count_orders_with_partner_tier_gt(s: &Source, tier: &[i32], t: i32) -> usize {
    (0..ROWS)
        .filter(|&r| tier.get(s.row(r).partner as usize).is_some_and(|&v| v > t))
        .count()
}
