//! The independent oracle. SKELETON: bodies are `todo!()`.
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
    let _ = (source, q, t);
    todo!()
}

/// `sum(a)` over the query's rows.
pub fn sum_a(source: &Source, q: Query) -> i64 {
    let _ = (source, q);
    todo!()
}

/// The first `limit` guids of the query's rows, key ascending, ties by guid.
/// Sorts its own copy; it shares nothing with the source's order index.
pub fn top_by_key(source: &Source, q: Query, limit: usize) -> Vec<u128> {
    let _ = (source, q, limit);
    todo!()
}
