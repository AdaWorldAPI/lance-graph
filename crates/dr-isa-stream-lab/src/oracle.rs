//! The answer straight from the fixture: no lanes, no masks, no binding.

use crate::fixture::{Fixture, MASKS};

/// `answer[p][m] = sum of value(i, p) over identities selected by mask m`,
/// for the listed properties.
pub fn sums(fx: &Fixture, props: &[usize]) -> Vec<[i64; MASKS]> {
    props
        .iter()
        .map(|&p| {
            let mut row = [0i64; MASKS];
            for i in 0..fx.n {
                let v = i64::from(fx.value(i, p));
                for (m, slot) in row.iter_mut().enumerate() {
                    if fx.selected(i, m) {
                        *slot += v;
                    }
                }
            }
            row
        })
        .collect()
}
