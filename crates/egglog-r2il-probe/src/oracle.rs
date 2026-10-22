//! The independent oracle: plain `i128` arithmetic over the ORIGINAL
//! premises. It shares no code with the egglog encoding, the extraction, or
//! the lowering; it never sees the extracted term.
//!
//! Every premise is linear in the target, so the oracle works on
//! coefficients: `coeff[v]` is the integer `k` with `v = k * target`. A
//! premise is never skipped. One that defines an already-known variable must
//! AGREE with it, so `[z = 2c, z = 3c]` is refused as inconsistent instead of
//! being answered by whichever premise comes first.

use std::collections::BTreeMap;

use crate::Problem;

/// Why the premises do not define the goal as a function of the target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleRefusal {
    /// Two premises give `lhs` different coefficients (this includes a cycle
    /// such as `c = 2c`, which only holds at `c = 0`).
    Inconsistent { lhs: &'static str },
    /// A premise whose right-hand side the target never reaches. Its meaning
    /// would be a constraint on an unknown, not a definition.
    Unreachable { lhs: &'static str },
    /// The goal is not reached from the target.
    GoalUnreachable,
    /// A coefficient left `i128`.
    Overflow,
}

/// `coeff[v]` for every variable the premises define from the target,
/// checking every premise.
pub fn coefficients(problem: &Problem) -> Result<BTreeMap<&'static str, i128>, OracleRefusal> {
    let mut known: BTreeMap<&'static str, i128> = BTreeMap::new();
    known.insert(problem.target_var, 1);
    let mut used = vec![false; problem.premises.len()];
    // Each pass resolves at least one more premise; premises.len() + 1 passes
    // reach every premise that is reachable at all.
    for _ in 0..=problem.premises.len() {
        let mut progressed = false;
        for (i, eq) in problem.premises.iter().enumerate() {
            if used[i] {
                continue;
            }
            let Some(&r) = known.get(eq.rhs) else {
                continue;
            };
            let k = r
                .checked_mul(i128::from(eq.coeff))
                .ok_or(OracleRefusal::Overflow)?;
            match known.get(eq.lhs) {
                Some(&have) if have != k => {
                    return Err(OracleRefusal::Inconsistent { lhs: eq.lhs });
                }
                Some(_) => {}
                None => {
                    known.insert(eq.lhs, k);
                }
            }
            used[i] = true;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    if let Some(i) = used.iter().position(|u| !u) {
        return Err(OracleRefusal::Unreachable {
            lhs: problem.premises[i].lhs,
        });
    }
    if !known.contains_key(problem.goal) {
        return Err(OracleRefusal::GoalUnreachable);
    }
    Ok(known)
}

/// The value of `goal` at `target_var = c`. `None` when the premises are
/// refused by [`coefficients`] or the value leaves `i128`.
pub fn solve(problem: &Problem, c: i64) -> Option<i128> {
    coefficients(problem)
        .ok()?
        .get(problem.goal)?
        .checked_mul(i128::from(c))
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

    fn eq(lhs: &'static str, coeff: i64, rhs: &'static str) -> Eq {
        Eq { lhs, coeff, rhs }
    }

    fn problem(premises: Vec<Eq>) -> Problem {
        Problem {
            premises,
            goal: "z",
            target_var: "c",
        }
    }

    /// FAILS IF: contradictory definitions are answered (in either premise
    /// order) instead of refused, an agreeing duplicate stops being accepted
    /// (two-sided), or a cycle through the target is accepted.
    #[test]
    fn every_premise_is_checked_not_just_the_first_definition() {
        for premises in [
            vec![eq("z", 2, "c"), eq("z", 3, "c")],
            vec![eq("z", 3, "c"), eq("z", 2, "c")],
        ] {
            let p = problem(premises);
            assert_eq!(
                coefficients(&p),
                Err(OracleRefusal::Inconsistent { lhs: "z" })
            );
            assert_eq!(solve(&p, 7), None);
        }
        let agreeing = problem(vec![eq("a", 2, "c"), eq("z", 4, "c"), eq("z", 2, "a")]);
        assert_eq!(solve(&agreeing, 7), Some(28));
        let cycle = problem(vec![eq("c", 2, "c"), eq("z", 1, "c")]);
        assert_eq!(
            coefficients(&cycle),
            Err(OracleRefusal::Inconsistent { lhs: "c" })
        );
        let dangling = problem(vec![eq("z", 2, "c"), eq("z", 3, "q")]);
        assert_eq!(
            coefficients(&dangling),
            Err(OracleRefusal::Unreachable { lhs: "z" })
        );
    }
}
