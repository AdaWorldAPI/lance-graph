//! The independent oracle: plain `i128` substitution over the ORIGINAL
//! premises. It shares no code with the egglog encoding, the extraction, or
//! the lowering; it never sees the extracted term.

use std::collections::BTreeMap;

use crate::Problem;

/// Solves the premises forward from `target_var = c` and returns the value of
/// `goal`. `None` when the premises do not connect `goal` to `target_var`
/// (a missing premise), or when an `i128` product would itself overflow.
pub fn solve(problem: &Problem, c: i64) -> Option<i128> {
    let mut known: BTreeMap<&str, i128> = BTreeMap::new();
    known.insert(problem.target_var, i128::from(c));
    // Each pass can resolve at least one more variable; premises.len() + 1
    // passes reach every variable that is reachable at all.
    for _ in 0..=problem.premises.len() {
        let mut progressed = false;
        for eq in &problem.premises {
            if known.contains_key(eq.lhs) {
                continue;
            }
            if let Some(&r) = known.get(eq.rhs) {
                let v = r.checked_mul(i128::from(eq.coeff))?;
                known.insert(eq.lhs, v);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    known.get(problem.goal).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Eq, e1_problem};

    /// FAILS IF: the oracle stops computing 8*c for E1, stops refusing a
    /// disconnected goal, or truncates to i64 (the overflow counterexample
    /// must come back as an i128 outside the i64 range).
    #[test]
    fn e1_is_8c_and_a_missing_premise_is_none() {
        let p = e1_problem();
        assert_eq!(solve(&p, 7), Some(56));
        assert_eq!(solve(&p, -123_456), Some(-987_648));
        let big = solve(&p, i64::MAX).expect("i128 holds it");
        assert!(big > i128::from(i64::MAX));

        let broken = Problem {
            premises: vec![
                Eq {
                    lhs: "a",
                    coeff: 2,
                    rhs: "b",
                },
                Eq {
                    lhs: "z",
                    coeff: 2,
                    rhs: "a",
                },
            ],
            goal: "z",
            target_var: "c",
        };
        assert_eq!(solve(&broken, 7), None);
    }
}
