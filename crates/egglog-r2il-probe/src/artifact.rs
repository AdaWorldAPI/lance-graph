//! Cold, in-memory proof capsule for the already-shipped E1 egglog/R2IL probe.
//!
//! The capsule retains the admitted premises, the extracted term, egglog's
//! checked proof rendering, and the exact reusable R2IL step list.
//! `recheck` independently rebuilds a fresh egglog proof session against
//! those premises before any claim of *proof replay* is made.
//!
//! This is NOT durable serialization, a cryptographic signature, a stand-alone
//! proof verifier, or a production-admitted JIT. Ordinary `execute` invokes
//! only the already-existing R2IL scalar fold, with no egglog in the hot path.

use crate::lower::{lower_steps, run_steps};
use crate::{Arith, Compiled, LowerRefusal, Problem, Saturation, Step, compile, oracle, try_prove};

/// A refusal is explicit; no incomplete proof can be silently promoted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayRefusal {
    NotExact,
    Compiler(String),
    Incomplete,
    OpenTerm,
    NoProof,
    ExtractionChanged,
    ProofChanged,
    LoweringChanged,
    InvalidPremises,
    Execution(LowerRefusal),
    ResultMismatch { compiled: i64, oracle: i128 },
}

/// Captured *mathematical* compilation. The premises' string identifiers are
/// static in Stage B's existing Problem API; a persisted/owned codec is later
/// work. No source version or data-binding validity is implied here.
#[derive(Debug, Clone)]
pub struct ReplayCapsule {
    problem: Problem,
    arith: Arith,
    fuel: usize,
    extracted: String,
    proof: String,
    program: Vec<Step>,
}

fn check_compilation(c: &Compiled) -> Result<(), ReplayRefusal> {
    if let Some(e) = &c.error {
        return Err(ReplayRefusal::Compiler(e.clone()));
    }
    if !matches!(c.saturation, Saturation::Saturated { .. }) {
        return Err(ReplayRefusal::Incomplete);
    }
    if !c.closed {
        return Err(ReplayRefusal::OpenTerm);
    }
    if c.proof.is_none() {
        return Err(ReplayRefusal::NoProof);
    }
    Ok(())
}

impl ReplayCapsule {
    /// Run the cold compiler and require a checked, closed exact result.
    pub fn certify(problem: &Problem, fuel: usize) -> Result<Self, ReplayRefusal> {
        // The premises must define the goal as ONE function of the target.
        // egglog unions contradictory premises without complaint, and its
        // extraction then picks a definition by cost or order.
        oracle::coefficients(problem).map_err(|_| ReplayRefusal::InvalidPremises)?;
        let c = compile(problem, Arith::Exact, fuel);
        check_compilation(&c)?;
        let program = lower_steps(&c).map_err(ReplayRefusal::Execution)?;
        let proof = c.proof.ok_or(ReplayRefusal::NoProof)?;
        // This asks egglog to prove the specific selected equality again,
        // rather than treating nonempty rendered proof text as certification.
        try_prove(problem, Arith::Exact, fuel, &c.extracted).map_err(ReplayRefusal::Compiler)?;
        Ok(Self {
            problem: problem.clone(),
            arith: Arith::Exact,
            fuel,
            extracted: c.extracted,
            proof,
            program,
        })
    }

    /// The canonical selected mathematical term (not a historical attribution).
    pub fn extracted(&self) -> &str {
        &self.extracted
    }

    /// The exact R2IL scalar instruction sequence saved at certification.
    pub fn program(&self) -> &[Step] {
        &self.program
    }

    /// The retained proof rendering from egglog's checked extraction.
    pub fn proof(&self) -> &str {
        &self.proof
    }

    /// The original premises used to certify the equality.
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Reconstruct and check the derivation under the ORIGINAL premises.
    ///
    /// Byte equality to freshly generated proof text is a conservative
    /// integrity control for this deterministic E1 lab. It is NOT an
    /// independently verified serialized proof format. Failure refuses.
    pub fn recheck(&self) -> Result<(), ReplayRefusal> {
        if self.arith != Arith::Exact {
            return Err(ReplayRefusal::NotExact);
        }
        let fresh = compile(&self.problem, self.arith, self.fuel);
        check_compilation(&fresh)?;
        if fresh.extracted != self.extracted {
            return Err(ReplayRefusal::ExtractionChanged);
        }
        if fresh.proof.as_deref() != Some(self.proof.as_str()) {
            return Err(ReplayRefusal::ProofChanged);
        }
        let steps = lower_steps(&fresh).map_err(ReplayRefusal::Execution)?;
        if steps != self.program {
            return Err(ReplayRefusal::LoweringChanged);
        }
        try_prove(&self.problem, self.arith, self.fuel, &self.extracted)
            .map_err(ReplayRefusal::Compiler)?;
        Ok(())
    }

    /// Hot-compatible execution: no egglog and no oracle. This requires
    /// *prior* certification and must not be confused with proof checking.
    pub fn execute(&self, target_value: i64) -> Result<i64, ReplayRefusal> {
        run_steps(
            &self.program,
            self.problem.target_var,
            &[(self.problem.target_var, target_value)],
        )
        .map_err(ReplayRefusal::Execution)
    }

    /// Cold replay with re-certification and independent i128 value oracle.
    pub fn replay(&self, target_value: i64) -> Result<i64, ReplayRefusal> {
        self.recheck()?;
        let v = self.execute(target_value)?;
        let expected =
            oracle::solve(&self.problem, target_value).ok_or(ReplayRefusal::InvalidPremises)?;
        if i128::from(v) != expected {
            return Err(ReplayRefusal::ResultMismatch {
                compiled: v,
                oracle: expected,
            });
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e1_problem;

    /// FAILS IF: contradictory premises are certified (in either order).
    #[test]
    fn contradictory_premises_are_not_certified() {
        use crate::Eq;
        for premises in [
            vec![
                Eq {
                    lhs: "z",
                    coeff: 2,
                    rhs: "c",
                },
                Eq {
                    lhs: "z",
                    coeff: 3,
                    rhs: "c",
                },
            ],
            vec![
                Eq {
                    lhs: "z",
                    coeff: 3,
                    rhs: "c",
                },
                Eq {
                    lhs: "z",
                    coeff: 2,
                    rhs: "c",
                },
            ],
        ] {
            let p = Problem {
                premises,
                goal: "z",
                target_var: "c",
            };
            assert_eq!(
                ReplayCapsule::certify(&p, 8).map(|_| ()),
                Err(ReplayRefusal::InvalidPremises)
            );
        }
    }

    #[test]
    fn corrupt_proof_and_program_refuse_on_replay() {
        let good = ReplayCapsule::certify(&e1_problem(), 8).expect("E1");
        assert_eq!(good.replay(7), Ok(56));

        let mut forged = good.clone();
        forged.proof.push_str("(invented proof)");
        assert_eq!(forged.recheck(), Err(ReplayRefusal::ProofChanged));

        let mut forged = good.clone();
        forged.program.pop();
        assert_eq!(forged.recheck(), Err(ReplayRefusal::LoweringChanged));

        let mut forged = good;
        forged.extracted = "(Num 999)".into();
        assert_eq!(forged.recheck(), Err(ReplayRefusal::ExtractionChanged));
    }
}
