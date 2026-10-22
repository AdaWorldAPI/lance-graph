//! The logical traceroute: every hop from premises to verdict, one line each.
//!
//! Hops, in order: Premises, Laws, Saturation, Extracted, Proof, R2IL, Value,
//! Oracle, Verdict. Each line renders as `N. Label: detail`.

use std::fmt;

use crate::lower::{Step, lower_steps, run_steps};
use crate::{Arith, LowerRefusal, Problem, Saturation, compile, oracle};

/// The hop labels, in order.
pub const HOPS: [&str; 9] = [
    "Premises",
    "Laws",
    "Saturation",
    "Extracted",
    "Proof",
    "R2IL",
    "Value",
    "Oracle",
    "Verdict",
];

/// How the lowered value and the oracle relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Both answered and the values are equal.
    Agree,
    /// Both answered and the values differ, or the lowering answered where
    /// the oracle has no answer.
    Disagree,
    /// Neither produced an `i64` answer, for a reason the other side shares:
    /// no connecting premise (both refuse), or an overflow while the oracle's
    /// exact value lies outside `i64`.
    BothRefuse,
    /// The oracle answers (in range) and the lowering refuses.
    OracleOnly,
}

/// One hop.
#[derive(Debug, Clone)]
pub struct Hop {
    /// Label from [`HOPS`].
    pub label: &'static str,
    /// What was observed there.
    pub detail: String,
}

/// The whole route.
#[derive(Debug, Clone)]
pub struct Trace {
    /// Hops in order.
    pub hops: Vec<Hop>,
    /// The final comparison.
    pub verdict: Verdict,
}

impl fmt::Display for Trace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, h) in self.hops.iter().enumerate() {
            writeln!(f, "{}. {}: {}", i + 1, h.label, h.detail)?;
        }
        Ok(())
    }
}

/// Compiles, lowers, runs with `target_var = c`, consults the oracle.
pub fn traceroute(p: &Problem, arith: Arith, fuel: usize, c: i64) -> Trace {
    let compiled = compile(p, arith, fuel);
    let mut hops = Vec::with_capacity(HOPS.len());
    let mut push = |label: &'static str, detail: String| hops.push(Hop { label, detail });

    push(
        "Premises",
        p.premises
            .iter()
            .map(|e| format!("{} = {}*{}", e.lhs, e.coeff, e.rhs))
            .collect::<Vec<_>>()
            .join("; ")
            + &format!(" | goal {} in {}", p.goal, p.target_var),
    );
    let admitted: Vec<String> = compiled
        .admitted
        .iter()
        .map(|r| format!("{:?}@{:?}", r.law, r.domain))
        .collect();
    let refused: Vec<String> = compiled
        .refused
        .iter()
        .map(|(l, why)| format!("{l:?}: {why:?}"))
        .collect();
    push(
        "Laws",
        format!(
            "admitted [{}] refused [{}]",
            admitted.join(", "),
            refused.join(", ")
        ),
    );
    push(
        "Saturation",
        match (&compiled.error, compiled.saturation) {
            (Some(e), _) => format!("egglog error: {e}"),
            (None, Saturation::Saturated { iterations }) => {
                format!("saturated after {iterations} step(s)")
            }
            (None, Saturation::Incomplete { fuel }) => {
                format!("INCOMPLETE: fuel {fuel} exhausted")
            }
        },
    );
    push(
        "Extracted",
        format!("{} (closed: {})", compiled.extracted, compiled.closed),
    );
    push(
        "Proof",
        match &compiled.proof {
            Some(p) => format!("present, {} bytes", p.len()),
            None => "absent".to_string(),
        },
    );

    let steps = lower_steps(&compiled);
    push(
        "R2IL",
        match &steps {
            Ok(s) => s.iter().map(Step::mnemonic).collect::<Vec<_>>().join(", "),
            Err(e) => format!("refused: {e:?}"),
        },
    );
    let value: Result<i64, LowerRefusal> = match &steps {
        Ok(s) => run_steps(s, compiled.target_var, &[(p.target_var, c)]),
        Err(e) => Err(e.clone()),
    };
    push(
        "Value",
        match &value {
            Ok(v) => format!("{v} (at {} = {c})", p.target_var),
            Err(e) => format!("refused: {e:?}"),
        },
    );
    let truth = oracle::solve(p, c);
    push(
        "Oracle",
        match truth {
            Some(v) => format!("{v} (i128)"),
            None => "none".to_string(),
        },
    );

    let verdict = match (&value, truth) {
        (Ok(v), Some(o)) => {
            if i128::from(*v) == o {
                Verdict::Agree
            } else {
                Verdict::Disagree
            }
        }
        (Ok(_), None) => Verdict::Disagree,
        (Err(_), None) => Verdict::BothRefuse,
        (Err(LowerRefusal::Overflow), Some(o)) if i64::try_from(o).is_err() => Verdict::BothRefuse,
        (Err(_), Some(_)) => Verdict::OracleOnly,
    };
    push("Verdict", format!("{verdict:?}"));

    Trace { hops, verdict }
}
