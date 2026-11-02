//! Compiler-graded diagnostics read off the codegraph's masks.
//!
//! Review semantics follow crai-rs CR-H0: the grade is `COMPILER` (the facts
//! come from the compiler's own LLBC, not from a heuristic); `POSSIBLE` is a
//! qualifier, never a grade; a run is `NOT_RUN` / `PARTIAL` / `COMPLETE`, and
//! a diagnostic that did not run is never reported as "clean" (zero findings
//! with a `NotRun` or `Refused` run means "no verdict", not "no problems").

use ndarray::simd::mask_andnot;
use ndarray::simd::mask_andnot_assign;

use crate::artifact::{P_CALLS, P_UNRESOLVED};
use crate::graph::Codegraph;

/// Which diagnostic a [`Finding`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A function nothing reaches.
    DeadFunction,
    /// A call whose target the compiler could not resolve.
    UnresolvedCall,
    /// A resolved call into a stored crate that defines no such function.
    DanglingCall,
    /// A `use` that nothing needs.
    UnusedImport,
}

/// How a finding was established. Only the compiler grade exists here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    /// Derived from compiler-emitted facts.
    Compiler,
}

/// A qualifier on a finding; never a grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qualifier {
    /// The finding holds on some resolutions, not all.
    Possible,
}

/// Lifecycle of a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingStatus {
    /// Current as of the loaded artifact.
    Open,
    /// The artifact set is an incomplete refresh; the finding may be an
    /// artifact of what was and was not re-extracted.
    Stale,
}

/// Whether the diagnostic ran, and how far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Run {
    /// Every stored crate's extractor finished.
    Complete,
    /// At least one stored crate's CRATE row is not `COMPLETE`.
    Partial,
    /// The artifact carries no facts for this diagnostic; no verdict.
    NotRun {
        /// Why nothing ran.
        reason: &'static str,
    },
    /// The diagnostic declined to answer for this input.
    Refused {
        /// Why it declined.
        reason: &'static str,
    },
}

/// Whether the analysed crate is a library or a binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrateKind {
    /// A library: its public items are entry points.
    Lib,
    /// A binary: entry reachability is not in the rows.
    Bin,
}

/// One diagnostic's result: masks, not decoded tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Which diagnostic produced this.
    pub kind: Kind,
    /// Position mask (functions), `words()` words.
    pub positions: Vec<u64>,
    /// Row mask (call sites), `rows().div_ceil(64)` words; an empty `Vec`
    /// for position-only findings.
    pub rows: Vec<u64>,
    /// How the finding was established.
    pub grade: Grade,
    /// Qualifier, if any.
    pub qualifier: Option<Qualifier>,
    /// Lifecycle status.
    pub status: FindingStatus,
    /// Whether and how far the diagnostic ran.
    pub run: Run,
}

fn set_bit(words: &mut [u64], p: usize) {
    if let Some(w) = words.get_mut(p / 64) {
        *w |= 1 << (p % 64);
    }
}

fn has_bit(words: &[u64], p: usize) -> bool {
    words.get(p / 64).is_some_and(|w| w >> (p % 64) & 1 == 1)
}

impl Codegraph {
    /// The four diagnostics, in the order DeadFunction, UnresolvedCall,
    /// DanglingCall, UnusedImport.
    ///
    /// The overall run is [`Run::Partial`] when any `CRATE` row carries a
    /// status other than `COMPLETE`, else [`Run::Complete`]; a `Partial` run
    /// still reports its findings, but the caller must not read a short list
    /// as "clean".
    pub fn diagnostics(&self, kind: CrateKind) -> [Finding; 4] {
        let words = self.words();
        let row_words = self.rows().div_ceil(64);

        // One scan over the integer lanes, into masks (not a tuple decode):
        // UNRESOLVED_CALL rows and the CALLS_RESOLVED rows that dangle. The
        // run status is the CRATE map (any non-COMPLETE crate), which is what
        // a per-CRATE-row lookup of the same map computed.
        let l = self.lanes();
        let partial = self.any_partial();
        let mut unresolved_rows = vec![0u64; row_words];
        let mut dangling_rows = vec![0u64; row_words];
        let mut dangling_callers = vec![0u64; words];
        for r in 0..self.rows() {
            match l.pred[r] {
                P_UNRESOLVED => set_bit(&mut unresolved_rows, r),
                P_CALLS => {
                    let (Some(caller), Some(callee)) =
                        (self.pos_of_id(l.subject[r]), self.pos_of_id(l.object[r]))
                    else {
                        continue;
                    };
                    let callee = callee as usize;
                    if !has_bit(self.fns(), callee) && has_bit(self.stored_owner(), callee) {
                        set_bit(&mut dangling_rows, r);
                        set_bit(&mut dangling_callers, caller as usize);
                    }
                }
                _ => {}
            }
        }
        let overall = if partial { Run::Partial } else { Run::Complete };

        // DeadFunction: defined, not public, not an impl, not synthetic, and
        // neither called nor referenced. Silent for a binary, whose entry
        // reachability the rows do not carry.
        let (dead, dead_run) = match kind {
            CrateKind::Bin => (
                vec![0u64; words],
                Run::Refused {
                    reason: "NO_ENTRY_FACTS",
                },
            ),
            CrateKind::Lib => {
                let mut m = vec![0u64; words];
                mask_andnot(self.fns(), self.public(), &mut m);
                mask_andnot_assign(&mut m, self.impl_subjects());
                mask_andnot_assign(&mut m, self.synthetic());
                mask_andnot_assign(&mut m, self.called());
                mask_andnot_assign(&mut m, self.referenced());
                (m, overall)
            }
        };

        [
            Finding {
                kind: Kind::DeadFunction,
                positions: dead,
                rows: Vec::new(),
                grade: Grade::Compiler,
                qualifier: None,
                status: FindingStatus::Open,
                run: dead_run,
            },
            // UnresolvedCall: fires for every UNRESOLVED_CALL row; silent
            // only when the artifact has none.
            Finding {
                kind: Kind::UnresolvedCall,
                positions: self.dynamic_callers().to_vec(),
                rows: unresolved_rows,
                grade: Grade::Compiler,
                qualifier: None,
                status: FindingStatus::Open,
                run: overall,
            },
            // DanglingCall: the callee belongs to a stored crate yet no
            // stored crate defines it. Callees of crates not in the store
            // (core::..., std::...) and MAY_DISPATCH rows never count.
            Finding {
                kind: Kind::DanglingCall,
                positions: dangling_callers,
                rows: dangling_rows,
                grade: Grade::Compiler,
                qualifier: None,
                status: FindingStatus::Stale,
                run: overall,
            },
            // UnusedImport: LLBC carries no `use` items, so there is no
            // verdict, and the empty masks must not be read as "clean".
            Finding {
                kind: Kind::UnusedImport,
                positions: vec![0u64; words],
                rows: vec![0u64; row_words],
                grade: Grade::Compiler,
                qualifier: None,
                status: FindingStatus::Open,
                run: Run::NotRun {
                    reason: "NO_IMPORT_FACTS",
                },
            },
        ]
    }
}
