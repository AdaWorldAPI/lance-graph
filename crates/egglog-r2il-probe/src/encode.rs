//! Problem -> egglog program text.
//!
//! One datatype, one `union` per premise, the admitted rewrites in a single
//! `arith` ruleset. Saturation is driven from Rust one `(run arith 1)` step at
//! a time (see `crate::session`), because a combined `(run arith FUEL)` report
//! cannot tell saturation from fuel exhaustion in proofs mode (the report
//! ANDs `can_stop` across the rebuild iterations; see the finding in
//! `session.rs`).
//!
//! Constants: premise coefficients are used as given. Nothing here can
//! overflow egglog: its `i64` `*` is `checked_mul` and a failing primitive
//! just stops the rule from firing.

use crate::laws::RuleId;
use crate::{Problem, laws};

/// The datatype and the (empty) ruleset.
///
/// `Def` carries `:cost 1000` (datatype variant syntax, egglog
/// `src/ast/parse.rs:1051`). With egglog's default AST-size cost every other
/// node costs 1, so the cheapest member of a defined variable's e-class is any
/// expansion of fewer than 1000 nodes, never the name. The extraction that
/// picks the answer is therefore egglog's own; the probe adds no search.
pub const HEADER: &str = "(datatype Term (Num i64) (Var String) (Def String :cost 1000) (Mul Term Term))\n(ruleset arith)\n";

/// The one-step schedule. `(run <ruleset> <n>)` parses as `Repeat(n, Run)`
/// (egglog `src/ast/parse.rs:670-698`).
pub const STEP: &str = "(run arith 1)";

/// A variable as a term: `(Var "c")` for the target, `(Def "a")` otherwise.
pub fn sym(name: &str, target: &str) -> String {
    if name == target {
        format!("(Var {name:?})")
    } else {
        format!("(Def {name:?})")
    }
}

/// The setup program: header, premises, admitted rewrites.
pub fn setup(problem: &Problem, admitted: &[RuleId]) -> String {
    let t = problem.target_var;
    let mut s = String::from(HEADER);
    for eq in &problem.premises {
        s.push_str(&format!(
            "(union {} (Mul (Num {}) {}))\n",
            sym(eq.lhs, t),
            eq.coeff,
            sym(eq.rhs, t)
        ));
    }
    for id in admitted {
        s.push_str(laws::rewrite_text(*id));
        s.push('\n');
    }
    s
}

/// `(prove-variants <goal> n)`.
pub fn prove_variants(problem: &Problem, n: usize) -> String {
    format!(
        "(prove-variants {} {n})",
        sym(problem.goal, problem.target_var)
    )
}

/// `(prove (= <goal> <rhs term text>))`.
pub fn prove_eq(problem: &Problem, rhs_term: &str) -> String {
    format!(
        "(prove (= {} {rhs_term}))",
        sym(problem.goal, problem.target_var)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Eq;
    use lance_graph_contract::set_rules::Arith;

    /// FAILS IF: the encoding stops emitting one union per premise, or emits a
    /// rewrite for a refused law (Checked admits none).
    #[test]
    fn setup_has_one_union_per_premise_and_only_admitted_rules() {
        let p = Problem {
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
            target_var: "b",
        };
        let (exact, _) = laws::admit_all(Arith::Exact);
        let (checked, _) = laws::admit_all(Arith::Checked);
        let e = setup(&p, &exact);
        let c = setup(&p, &checked);
        assert_eq!(e.matches("(union ").count(), 2);
        assert_eq!(e.matches("(rewrite ").count(), 2);
        assert_eq!(c.matches("(union ").count(), 2);
        assert_eq!(c.matches("(rewrite ").count(), 0);
        assert!(e.contains(r#"(union (Def "a") (Mul (Num 2) (Var "b")))"#));
    }
}
