//! The Rubicon reference flow, an optimized variant, and the mutants the
//! disable runs use.
//!
//! The flow, in words:
//!
//! 1. If deliberation is needed, deliberate.
//! 2. Wait for external input (suspend until it is ready).
//! 3. Decide, on the atom observed at that moment:
//!    * revision valid **and** MUL authorization granted (both WHERE, `{T}`)
//!      → `RequestCommit`;
//!    * revision invalid (`F`) → `Prune`;
//!    * revision valid, authorization refused (`F`) → `Plan`;
//!    * otherwise (an `U` on either side) → `Hold`.
//!
//! `RequestCommit` is an intent. The probe never writes anything; the actual
//! Lance write belongs to the supervisor / `cycle_driver`, which this crate
//! does not touch. The Kanban terminals `Commit | Plan | Prune` of
//! `kanban.rs` `next_phases(Evaluation)` correspond to `RequestCommit`,
//! `Plan` and `Prune`; `Hold` corresponds to a phase stay
//! (`GateDecision::Hold` → `advance_on_gate` returns `None`). This crate
//! does not depend on `kanban.rs` and owns no progression.

use crate::expr::{Action, BExp, Refusal};
use crate::flow::{Cond, Flow, Kind, Signature, Step};

pub const DELIBERATE: Action = Action("Deliberate");
pub const SUSPEND: Action = Action("Suspend");
pub const REQUEST_COMMIT: Action = Action("RequestCommit");
pub const HOLD: Action = Action("Hold");
pub const PLAN: Action = Action("Plan");
pub const PRUNE: Action = Action("Prune");

/// Revision validity, three-valued.
pub const REV: &str = "rev";
/// MUL authorization, three-valued.
pub const MUL: &str = "mul";
/// Whether deliberation is needed.
pub const DELIBERATE_NEEDED: &str = "deliberate";
/// Whether external input is ready at resume.
pub const INPUT_READY: &str = "input";

/// 6 primitive tests: rev (2), mul (2), deliberate (1), input (1); 64 atoms.
pub fn signature() -> Signature {
    Signature {
        observables: vec![
            (REV, Kind::K3),
            (MUL, Kind::K3),
            (DELIBERATE_NEEDED, Kind::Bool),
            (INPUT_READY, Kind::Bool),
        ],
        actions: vec![DELIBERATE, SUSPEND, REQUEST_COMMIT, HOLD, PLAN, PRUNE],
    }
}

/// The authorization the safety check requires immediately before
/// `RequestCommit`: WHERE(rev) ∧ WHERE(mul). It is fixed by the checker, not
/// taken from the flow, so a flow that authorizes with CHECK fails.
pub fn authorized() -> Result<BExp, Refusal> {
    signature().guard(&Cond::Where(REV).and(Cond::Where(MUL)))
}

fn act(p: Action) -> Step {
    Step::Act(p)
}

fn wait() -> Step {
    Step::WaitForInput {
        ready: Cond::Is(INPUT_READY),
        suspend: SUSPEND,
    }
}

/// The reference decision.
pub fn decide_reference() -> Step {
    Step::if_(
        Cond::Where(REV).and(Cond::Where(MUL)),
        act(REQUEST_COMMIT),
        Step::if_(
            Cond::Check(REV).negated(),
            act(PRUNE),
            Step::if_(
                Cond::Where(REV).and(Cond::Check(MUL).negated()),
                act(PLAN),
                act(HOLD),
            ),
        ),
    )
}

/// The reference flow: the decision tail is written out in both branches of
/// the deliberation choice.
pub fn reference() -> Flow {
    Flow {
        signature: signature(),
        body: Step::if_(
            Cond::Is(DELIBERATE_NEEDED),
            Step::Seq(vec![act(DELIBERATE), wait(), decide_reference()]),
            Step::Seq(vec![wait(), decide_reference()]),
        ),
    }
}

/// The optimized decision: tests the revision first and nests the
/// authorization inside it.
pub fn decide_optimized(mul_branch: Step) -> Step {
    Step::if_(
        Cond::Check(REV).negated(),
        act(PRUNE),
        Step::if_(Cond::Where(REV), mul_branch, act(HOLD)),
    )
}

/// The optimized decision's authorization branch.
pub fn mul_branch() -> Step {
    Step::if_(
        Cond::Where(MUL),
        act(REQUEST_COMMIT),
        Step::if_(Cond::Check(MUL), act(HOLD), act(PLAN)),
    )
}

/// The optimized flow: the shared tail is factored out of the deliberation
/// choice (G1) and the decision is reordered.
pub fn optimized() -> Flow {
    optimized_with(decide_optimized(mul_branch()))
}

fn optimized_with(decide: Step) -> Flow {
    Flow {
        signature: signature(),
        body: Step::Seq(vec![
            Step::if_(Cond::Is(DELIBERATE_NEEDED), act(DELIBERATE), Step::Skip),
            wait(),
            decide,
        ]),
    }
}

/// Disable 1: the authorization guard's polarity is reversed.
pub fn mutant_reversed_polarity() -> Flow {
    optimized_with(decide_optimized(Step::if_(
        Cond::Where(MUL).negated(),
        act(REQUEST_COMMIT),
        Step::if_(Cond::Check(MUL), act(HOLD), act(PLAN)),
    )))
}

/// Disable 2: `RequestCommit` is moved ahead of its authorization guard.
pub fn mutant_commit_before_guard() -> Flow {
    optimized_with(decide_optimized(Step::Seq(vec![
        act(REQUEST_COMMIT),
        Step::if_(
            Cond::Where(MUL),
            Step::Skip,
            Step::if_(Cond::Check(MUL), act(HOLD), act(PLAN)),
        ),
    ])))
}

/// Disable 2b: authorization is observed, then `Deliberate` runs, then
/// `RequestCommit`. The authorization was observed before an action and is
/// stale when the intent is emitted.
pub fn mutant_stale_authorization() -> Flow {
    Flow {
        signature: signature(),
        body: Step::Seq(vec![
            wait(),
            Step::if_(
                Cond::Where(REV).and(Cond::Where(MUL)),
                Step::Seq(vec![act(DELIBERATE), act(REQUEST_COMMIT)]),
                Step::if_(Cond::Check(REV).negated(), act(PRUNE), act(HOLD)),
            ),
        ]),
    }
}

/// Disable 3: unguarded nondeterminism between committing and holding.
pub fn mutant_unguarded_choice() -> Flow {
    optimized_with(decide_optimized(Step::Choice(vec![
        act(REQUEST_COMMIT),
        act(HOLD),
    ])))
}

/// Commit authorization with CHECK projections instead of WHERE: an `U`
/// revision or authorization would authorize.
pub fn reference_with_check_auth() -> Flow {
    Flow {
        signature: signature(),
        body: Step::Seq(vec![
            wait(),
            Step::if_(
                Cond::Check(REV).and(Cond::Check(MUL)),
                act(REQUEST_COMMIT),
                Step::if_(Cond::Check(REV).negated(), act(PRUNE), act(PLAN)),
            ),
        ]),
    }
}

/// An optimized form of [`reference_with_check_auth`]: equivalent to it, and
/// just as unsafe.
pub fn optimized_with_check_auth() -> Flow {
    Flow {
        signature: signature(),
        body: Step::Seq(vec![
            wait(),
            Step::if_(
                Cond::Check(REV).negated(),
                act(PRUNE),
                Step::if_(Cond::Check(MUL), act(REQUEST_COMMIT), act(PLAN)),
            ),
        ]),
    }
}

/// The flow with a `GoTo` retry instead of a structured loop.
pub fn mutant_goto() -> Flow {
    optimized_with(Step::if_(
        Cond::Check(REV).negated(),
        Step::GoTo("planning"),
        decide_optimized(mul_branch()),
    ))
}

/// The flow with a `GoBack` instead of a structured loop.
pub fn mutant_goback() -> Flow {
    optimized_with(Step::if_(
        Cond::Is(INPUT_READY).negated(),
        Step::GoBack,
        decide_optimized(mul_branch()),
    ))
}
