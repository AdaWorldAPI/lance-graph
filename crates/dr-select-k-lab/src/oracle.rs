//! The independent oracle.
//!
//! Row at a time over the source lanes, with no use of `bind`, `ops`, the
//! order index or mask-RISC. Reads `Source::lanes()`.

use crate::bind::Order;
use crate::source::{KeyRange, Source};

/// The full query the lab asks: rows with `key in range AND a >= t_a`.
#[derive(Debug, Clone, Copy)]
pub struct Query {
    pub range: KeyRange,
    /// `None` means no residual (the ALL case).
    pub t_a: Option<i32>,
}

/// `count(query AND b > t)`.
pub fn count_b_gt(source: &Source, q: Query, t: i32) -> u64 {
    let (_, key, a, b) = source.lanes();
    let mut n = 0u64;
    for r in 0..key.len() {
        if in_query(&q, key[r], a[r]) && b[r] > t {
            n += 1;
        }
    }
    n
}

/// `sum(a)` over the query's rows.
pub fn sum_a(source: &Source, q: Query) -> i64 {
    let (_, key, a, _) = source.lanes();
    let mut sum = 0i64;
    for r in 0..key.len() {
        if in_query(&q, key[r], a[r]) {
            sum += i64::from(a[r]);
        }
    }
    sum
}

/// The first `limit` guids of the query's rows in the declared `order`:
/// key ascending, ties by source row (`Order::KeyThenRow`) or by guid
/// (`Order::KeyThenGuid`). Sorts its own copy; it shares nothing with the
/// source's order indexes.
///
/// # Panics
/// If `order` is `Order::Source`, which declares no key order.
pub fn top_by_key(source: &Source, q: Query, limit: usize, order: Order) -> Vec<u128> {
    let (guid, key, a, _) = source.lanes();
    let mut rows: Vec<(i32, u128, usize)> = Vec::new();
    for r in 0..key.len() {
        if in_query(&q, key[r], a[r]) {
            rows.push((key[r], guid[r], r));
        }
    }
    match order {
        Order::KeyThenRow => rows.sort_by_key(|&(k, _, r)| (k, r)),
        Order::KeyThenGuid => rows.sort_by_key(|&(k, g, _)| (k, g)),
        Order::Source => panic!("the oracle needs a declared key order"),
    }
    rows.into_iter().take(limit).map(|(_, g, _)| g).collect()
}

/// Whether one source row belongs to the query.
fn in_query(q: &Query, key: i32, a: i32) -> bool {
    q.range.has(key) && q.t_a.is_none_or(|t| a >= t)
}
