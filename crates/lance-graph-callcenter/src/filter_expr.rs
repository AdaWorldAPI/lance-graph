//! DM-3 — `CommitFilter` → DataFusion `Expr` translator.
//!
//! Converts the scalar-predicate shape of `CommitFilter` into a DataFusion
//! `Expr` that can be pushed down into Lance column scans. All predicates
//! operate on Arrow-scalar columns (`u8`, `u64`, `bool`) from `CognitiveEventRow`
//! — no VSA types ever appear here.
//!
//! Column names must match the Lance schema written by `LanceMembrane::persist()`.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § DM-3

use datafusion::logical_expr::{col, lit, Expr};
use lance_graph_contract::external_membrane::CommitFilter;

/// Column name constants — keep in sync with the schema written by `LanceMembrane`.
pub mod columns {
    pub const GATE_F:      &str = "gate_f";      // u8 — free_energy at commit time
    pub const THINKING:    &str = "thinking";    // u8 — ThinkingStyle ordinal
    pub const GATE_COMMIT: &str = "gate_commit"; // bool — CollapseGate decision
    // actor_id column: not yet in schema (UNKNOWN-4); filtered in Phase B.
}

/// Translate a `CommitFilter` to a DataFusion `Expr`.
///
/// Returns `None` when all filter fields are `None` (no predicate needed —
/// caller scans the full table). Returns `Some(expr)` when at least one
/// predicate is active.
///
/// Predicates are AND-ed. Field mapping:
/// - `max_free_energy` → `gate_f <= max_free_energy`
/// - `style_ordinal`   → `thinking = style_ordinal`
/// - `is_commit`       → `gate_commit = is_commit`
/// - `actor_id`        → skipped (Phase B, no schema column yet)
pub fn commit_filter_to_expr(filter: &CommitFilter) -> Option<Expr> {
    let mut predicates: Vec<Expr> = Vec::new();

    if let Some(max_fe) = filter.max_free_energy {
        predicates.push(col(columns::GATE_F).lt_eq(lit(max_fe)));
    }

    if let Some(style) = filter.style_ordinal {
        predicates.push(col(columns::THINKING).eq(lit(style)));
    }

    if let Some(commit) = filter.is_commit {
        predicates.push(col(columns::GATE_COMMIT).eq(lit(commit)));
    }

    // actor_id: UNKNOWN-4 (right type / column name not yet decided).
    // When resolved, add: col("actor_id").eq(lit(actor_id))
    if filter.actor_id.is_some() {
        // Silently ignored for now — logged here so it's visible in grep.
        // TODO(UNKNOWN-4): wire actor_id predicate once column schema is settled.
    }

    predicates.into_iter().reduce(|acc, e| acc.and(e))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_returns_none() {
        let f = CommitFilter::default();
        assert!(commit_filter_to_expr(&f).is_none());
    }

    #[test]
    fn single_predicate_max_free_energy() {
        let f = CommitFilter { max_free_energy: Some(50), ..Default::default() };
        let expr = commit_filter_to_expr(&f).expect("should produce expr");
        let s = format!("{expr}");
        assert!(s.contains("gate_f"), "expr should reference gate_f, got: {s}");
    }

    #[test]
    fn style_predicate() {
        let f = CommitFilter { style_ordinal: Some(3), ..Default::default() };
        let expr = commit_filter_to_expr(&f).expect("should produce expr");
        let s = format!("{expr}");
        assert!(s.contains("thinking"), "expr should reference thinking, got: {s}");
    }

    #[test]
    fn commit_predicate() {
        let f = CommitFilter { is_commit: Some(true), ..Default::default() };
        let expr = commit_filter_to_expr(&f).expect("should produce expr");
        let s = format!("{expr}");
        assert!(s.contains("gate_commit"), "expr should reference gate_commit, got: {s}");
    }

    #[test]
    fn all_predicates_are_combined() {
        let f = CommitFilter {
            max_free_energy: Some(30),
            style_ordinal: Some(7),
            is_commit: Some(true),
            actor_id: None,
        };
        let expr = commit_filter_to_expr(&f).expect("should produce combined expr");
        let s = format!("{expr}");
        assert!(s.contains("gate_f"),      "combined expr missing gate_f: {s}");
        assert!(s.contains("thinking"),    "combined expr missing thinking: {s}");
        assert!(s.contains("gate_commit"), "combined expr missing gate_commit: {s}");
    }
}
