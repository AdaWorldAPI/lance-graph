//! Independent scalar reference: every query computed directly from
//! [`fixture::value`] / [`fixture::present`] per `(row, field)`, never from an arm.

use crate::arms::{Query, K};
use crate::fixture::{present, value, Fixture, FIELDS};

/// Reference answer of `q` over `fx`.
pub fn oracle(fx: &Fixture, q: &Query<'_>) -> u128 {
    let (seed, n) = (fx.seed, fx.n);
    match *q {
        Query::CountWhere => (0..n).filter(|&i| value(seed, i, 0) == K).count() as u128,
        Query::Exists => (0..n).any(|i| value(seed, i, 0) == K) as u128,
        Query::SumWhere => (0..n)
            .filter(|&i| value(seed, i, 0) == K)
            .map(|i| value(seed, i, 1) as u128)
            .sum(),
        Query::SumPresentC => (0..n)
            .filter(|&i| value(seed, i, 0) == K && present(seed, i, 2))
            .map(|i| value(seed, i, 2) as u128)
            .sum(),
        Query::DemandFold { d } => {
            let mut acc = 0u64;
            for i in 0..n {
                for f in 0..d.min(FIELDS) {
                    acc = acc.wrapping_add(value(seed, i, f) ^ f as u64);
                }
            }
            acc as u128
        }
        Query::Point { probes } => {
            let mut acc = 0u64;
            for &p in probes {
                let mut x = 0u64;
                for f in 0..5 {
                    x ^= value(seed, p, f);
                }
                acc = acc.wrapping_add(x);
            }
            acc as u128
        }
    }
}
