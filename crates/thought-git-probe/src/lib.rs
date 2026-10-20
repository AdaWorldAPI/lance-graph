//! D-TGB-0 — the pre-registration for thought git
//! (`.claude/plans/thought-git-basics-v1.md`, work item D-TGB-0).
//!
//! Three things live here, all written before any thought-git code exists:
//!
//! - [`fixtures`]: the thoughts every later D-TGB item is tested on, one per
//!   Rubicon outcome, a stack of unremarkable thoughts, one thought per
//!   anomaly signal, and a lane that is never written.
//! - [`oracle`]: the from-scratch reference. Given the givens (layer 0), the
//!   previous version and the thoughts that land, it computes the next
//!   version without any incremental state. Every later implementation is
//!   compared against it.
//! - `tests/`: the oracle's rules, each with a disable that turns it red, and
//!   checks that the existing contract signals the plan relies on (regression
//!   counts, the cold/hot mask diff, the kanban routes, the Gadamer verdict)
//!   fire on the anomaly fixtures and stay silent on the unremarkable ones.
//!
//! `PREREGISTRATION.md` lists the falsifier and the disable of every later
//! D-TGB item, with the fixture it runs on. What is still OPEN there (the
//! anomaly thresholds, what "satisfactory" means) is left open on purpose.

pub mod fixtures;
pub mod oracle;
