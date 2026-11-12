//! Shared fixtures for the integration tests.

use dr_select_k_lab::counters::Counters;
use dr_select_k_lab::fixture::{self, Pattern};
use dr_select_k_lab::oracle::Query;
use dr_select_k_lab::source::{KeyRange, Source};
use dr_select_k_lab::{Bound, Order, Residual, Selection};

/// Rows in every test source (small enough for a debug build).
pub const N: usize = 20_000;
/// Rows per key value, so key ties exist.
pub const DUP: u32 = 4;
pub const SEED: u64 = 0x5EED_1234;

pub const PATTERNS: [Pattern; 2] = [Pattern::Clustered, Pattern::Random];

/// The two declared key orders an index can witness.
pub const ORDERS: [Order; 2] = [Order::KeyThenRow, Order::KeyThenGuid];

/// Every order a binding can carry: the filter's source order, then the two
/// declared key orders.
pub const BINDINGS: [Order; 3] = [Order::Source, Order::KeyThenRow, Order::KeyThenGuid];

pub fn source(pattern: Pattern) -> Source {
    fixture::source(N, pattern, DUP, SEED)
}

/// A key range selecting exactly `k` rows (`k` a multiple of `DUP`), or no
/// row at all for `k == 0` (keys top out at `N / DUP - 1`, below 6000).
pub fn range_for_k(k: usize) -> KeyRange {
    if k == 0 {
        KeyRange { lo: 6000, hi: 6000 }
    } else {
        assert!(k.is_multiple_of(DUP as usize));
        KeyRange {
            lo: 0,
            hi: (k / DUP as usize) as i32 - 1,
        }
    }
}

/// Produce one selection: the filter for `Order::Source`, the matching index
/// order otherwise.
pub fn select(src: &Source, range: KeyRange, order: Order, counters: &Counters) -> Selection {
    match order {
        Order::Source => src.select_filter(range, counters),
        Order::KeyThenRow | Order::KeyThenGuid => src.select_ordered(range, order, counters),
    }
}

/// Produce and bind one selection. `Order::Source` is the filter binding.
pub fn bind(src: &Source, range: KeyRange, order: Order, counters: &Counters) -> Bound {
    let sel = select(src, range, order, counters);
    Bound::bind(src, sel, counters).unwrap()
}

/// Residual templates (the `like` argument of `ops::residual_a_ge`) with the
/// threshold to pass. `All` means no residual, so it is paired only with
/// `t_a == None`; there `Positions`/`Mask` with threshold 0 keep every row
/// (`a` is never negative) and must agree with `All`.
pub fn templates(t_a: Option<i32>) -> Vec<(Residual, i32)> {
    match t_a {
        None => vec![
            (Residual::All, 0),
            (Residual::Positions(Vec::new()), 0),
            (Residual::Mask(Vec::new()), 0),
        ],
        Some(t) => vec![
            (Residual::Positions(Vec::new()), t),
            (Residual::Mask(Vec::new()), t),
        ],
    }
}

pub fn query(range: KeyRange, t_a: Option<i32>) -> Query {
    Query { range, t_a }
}

/// The N-bit plane of source rows in the query, built row at a time.
pub fn selected_plane(src: &Source, q: &Query) -> Vec<u64> {
    let (_, key, a, _) = src.lanes();
    let mut plane = vec![0u64; key.len().div_ceil(64)];
    for r in 0..key.len() {
        if q.range.has(key[r]) && q.t_a.is_none_or(|t| a[r] >= t) {
            plane[r / 64] |= 1u64 << (r % 64);
        }
    }
    plane
}

/// Number of local rows in a residual of any form.
pub fn residual_len(res: &Residual, k: usize) -> usize {
    match res {
        Residual::All => k,
        Residual::Positions(p) => p.len(),
        Residual::Mask(m) => m.iter().map(|w| w.count_ones() as usize).sum(),
    }
}
