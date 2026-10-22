//! The only module that touches egglog, and only through its public API:
//! `EGraph::new_with_proofs`, `parse_and_run_program`, the
//! `prove` and `prove-variants` commands, `CommandOutput`, `ProvedVariants`.
//! egglog is not modified; every proof it returns has already been checked by
//! its own `check_proof` (egglog `src/proofs/proof_extraction.rs:126-139`).
//!
//! # Saturation versus fuel exhaustion
//!
//! The probe runs `(run arith 1)` itself, up to `fuel` times, and reads the
//! `changed()` flag of the FIRST iteration report of each step (the `arith`
//! ruleset's own iteration). A step that changed nothing is a fixpoint:
//! `Saturated`. If every one of the `fuel` steps changed something the probe
//! reports `Incomplete` and certifies nothing.
//!
//! Why not one `(run-schedule (repeat FUEL (run arith)))` and its `RunReport`?
//! In proofs mode egglog wraps every schedule with rebuild iterations
//! (`src/proofs/proof_encoding.rs:1498`, `instrument_schedule`), and
//! `RunReport::union` ANDs `can_stop` and ORs `updated` across all of them
//! (egglog-reports `src/lib.rs:260-264`), so the combined flags describe the
//! rebuilds as much as the rules and cannot be read as "the rules reached a
//! fixpoint".

use egglog::{CommandOutput, EGraph, ProvedVariants};

use crate::Saturation;
use crate::{Problem, encode};

/// Variants requested per class (egglog returns at most one per e-node).
pub const VARIANTS: usize = 16;

/// One proofs-mode e-graph.
pub struct Session {
    egraph: EGraph,
}

impl Session {
    /// A fresh proofs-mode e-graph.
    pub fn new() -> Self {
        Session {
            egraph: EGraph::new_with_proofs(),
        }
    }

    /// Runs program text; egglog errors become strings.
    pub fn run(&mut self, text: &str) -> Result<Vec<CommandOutput>, String> {
        self.egraph
            .parse_and_run_program(None, text)
            .map_err(|e| e.to_string())
    }

    /// Steps `arith` up to `fuel` times. `has_rules == false` (an empty
    /// ruleset, nothing admitted) is saturated at zero iterations without
    /// asking egglog to run an empty ruleset.
    pub fn saturate(&mut self, fuel: usize, has_rules: bool) -> Result<Saturation, String> {
        if !has_rules {
            return Ok(Saturation::Saturated { iterations: 0 });
        }
        for i in 1..=fuel {
            let outs = self.run(encode::STEP)?;
            if !step_changed(&outs) {
                return Ok(Saturation::Saturated { iterations: i });
            }
        }
        Ok(Saturation::Incomplete { fuel })
    }

    /// `(prove-variants <goal> n)`: `(term, proof text)` per variant, in
    /// the order egglog returns them (cheapest first).
    ///
    /// The result is recovered from `CommandOutput::UserDefined(arc)` with
    /// `arc.as_ref().as_any().downcast_ref::<ProvedVariants>()`: `as_ref()`
    /// first, because the `Arc` itself also satisfies the blanket
    /// `UserDefinedCommandOutput` impl and its `as_any` would describe the
    /// `Arc`, not the payload (egglog `src/lib.rs:135-157`).
    pub fn prove_variants(
        &mut self,
        problem: &Problem,
        n: usize,
    ) -> Result<Vec<(String, String)>, String> {
        let outs = self.run(&encode::prove_variants(problem, n))?;
        for out in &outs {
            if let CommandOutput::UserDefined(arc) = out
                && let Some(pv) = arc.as_ref().as_any().downcast_ref::<ProvedVariants>()
            {
                return Ok(pv
                    .variants
                    .iter()
                    .map(|v| (v.term.clone(), v.proof_store.proof_to_string(v.proof_id)))
                    .collect());
            }
        }
        Err("prove-variants returned no ProvedVariants".to_string())
    }

    /// `(prove (= <goal> rhs))`: the checked proof text, or egglog's error.
    pub fn prove_eq(&mut self, problem: &Problem, rhs_term: &str) -> Result<String, String> {
        let outs = self.run(&encode::prove_eq(problem, rhs_term))?;
        for out in &outs {
            if let CommandOutput::ProveExists {
                proof_store,
                proof_id,
            } = out
            {
                return Ok(proof_store.proof_to_string(*proof_id));
            }
        }
        Err("prove returned no proof".to_string())
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether the `arith` ruleset changed the e-graph in one `(run arith 1)`:
/// the first `RunSchedule` output's first iteration. Falls back to the
/// report's `updated` flag if egglog returned no iteration list.
fn step_changed(outs: &[CommandOutput]) -> bool {
    for out in outs {
        if let CommandOutput::RunSchedule(report) = out {
            return report
                .iterations
                .first()
                .map_or(report.updated, |it| it.changed());
        }
    }
    // No report at all: do not claim a fixpoint.
    true
}
