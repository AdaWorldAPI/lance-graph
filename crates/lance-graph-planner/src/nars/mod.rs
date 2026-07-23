//! # P1: NARS Schema + Thinking
//!
//! **NARS Schema** = truth values stored as edge properties in the adjacency store.
//! **NARS Thinking** = inference rules that OPERATE on adjacent truth values.
//!
//! Each inference type = a specific semiring for adjacent_truth_propagate().

pub mod belief;
pub mod dissolution;
pub mod elevation;
pub mod facet_fold;
pub mod inference;
pub mod insight;
pub mod reach_out;
pub mod regulate;
pub mod tactic_select;
pub mod tactics;
pub mod truth;

pub use belief::{Belief, BeliefArena, CStmt, Copula, ReviseOutcome, Stamp};
pub use dissolution::{detect_dissolution, should_elevate, staunen, wisdom, Dissolution};
pub use elevation::{elevate_field, Elevation};
pub use facet_fold::{cstmt_from_spo_facet, to_spo_facet};
pub use inference::NarsInference;
pub use insight::{arena_graph_signals, detect, flow_state, InsightMush, Snapshot};
pub use regulate::{regulate_cycle, CycleConfig, CycleOutcome};
pub use tactic_select::{tactic_for_bias, TacticChoice};
pub use tactics::{
    asc_challenge, cas_abstract, challenge_target, cr_synthesize, rcr_abduce, tr_diverge,
    AscOutcome, Candidate, Frontier, GapKind, ReasoningGap, Tactic, Throttle,
};
pub use truth::TruthValue;
