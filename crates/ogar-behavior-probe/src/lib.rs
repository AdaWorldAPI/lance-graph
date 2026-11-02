//! LAB: what the existing OGAR structures can express about method
//! behaviour, measured before any type is proposed.
//!
//! - [`revision`] (P2): a deterministic revision of an `ogar-loco::Program`,
//!   and a demonstration of what it does not cover.
//! - [`projection`] (P3): a graded, cold projection of `ActionDef` onto the
//!   DT64 `MethodBinding` fields. It never builds a binding.
//! - [`census`] (P9): which methods have which behavioural representation.
//!
//! Plan: `.claude/board/entries/2026-10-31-coresearch-mandatory-codegraph.md`
//! (probes P2, P3, P9). The README carries the results and the disable table.

pub mod census;
pub mod projection;
pub mod revision;
