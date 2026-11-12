//! The independent oracle.
//!
//! Row at a time over the source lanes, with no use of `bind`, `ops`, the
//! order index or mask-RISC. Reads `Source::lanes()`.

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

/// The first `limit` guids of the query's rows, key ascending, ties by guid.
/// Sorts its own copy; it shares nothing with the source's order index.
pub fn top_by_key(source: &Source, q: Query, limit: usize) -> Vec<u128> {
    let (guid, key, a, _) = source.lanes();
    let mut rows: Vec<(i32, u128)> = Vec::new();
    for r in 0..key.len() {
        if in_query(&q, key[r], a[r]) {
            rows.push((key[r], guid[r]));
        }
    }
    rows.sort();
    rows.into_iter().take(limit).map(|(_, g)| g).collect()
}

/// Whether one source row belongs to the query.
fn in_query(q: &Query, key: i32, a: i32) -> bool {
    q.range.has(key) && q.t_a.is_none_or(|t| a >= t)
}
