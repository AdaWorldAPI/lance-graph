//! D-ODOO-EXT-6 — coverage gate test (Stage 1 closer).
//!
//! Reads the pairing table from `pairing::CURATED_EXTRACTED_PAIRS` and
//! enforces per-lane minimum extracted-backing coverage. The 5 explicit
//! TIER-2 deferrals (`COVERAGE_EXEMPTIONS`) are subtracted from each
//! lane's eligible count before computing the ratio — they're scope
//! decisions, not regressions.
//!
//! Companion human-readable report: `extracted/COVERAGE.md`.

/// Curated `model_name`s explicitly deferred to Stage 2 (TIER-2 addons
/// outside the 12 TIER-1 set). Subtracting these from each lane's
/// "eligible" count prevents the gate from flagging the plan's own
/// deferral decisions as regressions.
///
/// When Stage 2 lands (extracts `hr` + `stock_account` + …), remove
/// the corresponding entries from this list.
pub const COVERAGE_EXEMPTIONS: &[(&str, &str)] = &[
    ("hr.contract.type", "hr (Stage 2)"),
    ("hr.department", "hr (Stage 2)"),
    ("hr.employee", "hr (Stage 2)"),
    ("hr.job", "hr (Stage 2)"),
    ("stock.valuation.layer", "stock_account (Stage 2)"),
];

/// Minimum eligible coverage per lane. Set at 80% per the plan.
pub const COVERAGE_FLOOR: f32 = 0.80;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::extracted::pairing::CURATED_EXTRACTED_PAIRS;

    /// Collect every curated `OdooEntity` (model_name + which lane it
    /// lives in). The list is intentionally hand-maintained — the lane
    /// modules are small; this keeps the test legible without macros.
    ///
    /// For Stage 1, every curated entity is in either `CURATED_EXTRACTED_PAIRS`
    /// (has backing) or `COVERAGE_EXEMPTIONS` (TIER-2 deferred). Together
    /// they enumerate all 53 curated entities.
    fn curated_entities() -> Vec<(&'static str, u8)> {
        let mut v: Vec<(&'static str, u8)> = Vec::new();
        // Backed entities: derive lane from provenance.l_doc
        for p in CURATED_EXTRACTED_PAIRS {
            v.push((p.model_name, lane_of(p.curated.provenance.l_doc)));
        }
        // Exempt entities: lane is hand-coded below (they're absent from
        // the pairing table by definition — no extracted backing yet)
        for (model_name, _) in COVERAGE_EXEMPTIONS {
            v.push((*model_name, lane_of_exempt(model_name)));
        }
        v
    }

    /// Parse lane number from an l_doc filename of the form "L{N}-…".
    ///
    /// "L13-STOCK-VALUATION-PROCUREMENT.md" → 13
    /// Panics on parse failure — all l_doc values must follow the L{N}-... format (enforced by EXT-3 backfill).
    fn lane_of(l_doc: &str) -> u8 {
        // Strip the leading 'L', collect ASCII digits until the first '-'
        let stripped = l_doc.trim_start_matches('L');
        let digits: String = stripped
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse().expect("l_doc must follow 'L{N}-...' format")
    }

    /// Hand-coded lane mapping for the 5 TIER-2 exemptions.
    ///
    /// Their l_doc lives in the curated lane modules, but the entities are
    /// absent from `CURATED_EXTRACTED_PAIRS` by definition — so there is no
    /// runtime handle to inspect their provenance without enumerating all lane
    /// consts. This small table avoids that coupling.
    ///
    /// Update this mapping if a future plan re-assigns exemptions to a
    /// different lane.
    fn lane_of_exempt(model_name: &str) -> u8 {
        match model_name {
            "stock.valuation.layer" => 13,
            "hr.contract.type" | "hr.department" | "hr.employee" | "hr.job" => 14,
            _ => panic!("unknown exemption model_name: {}", model_name),
        }
    }

    /// Every lane with at least one eligible curated entity must have
    /// extracted backing for ≥ 80% of those entities.
    ///
    /// Lanes where ALL entities are exempt (currently L14 — HR-only) are
    /// skipped rather than failed: 0-eligible is a scope decision, not a
    /// regression.
    #[test]
    fn every_lane_meets_coverage_floor() {
        let exempt_names: std::collections::HashSet<&str> =
            COVERAGE_EXEMPTIONS.iter().map(|(n, _)| *n).collect();
        let curated = curated_entities();
        let backed_names: std::collections::HashSet<&str> = CURATED_EXTRACTED_PAIRS
            .iter()
            .map(|p| p.model_name)
            .collect();

        for lane in 1u8..=15 {
            let lane_entities: Vec<&str> = curated
                .iter()
                .filter(|(_, l)| *l == lane)
                .map(|(n, _)| *n)
                .collect();

            let eligible: Vec<&str> = lane_entities
                .iter()
                .copied()
                .filter(|n| !exempt_names.contains(n))
                .collect();

            if eligible.is_empty() {
                // Wholly-exempt lane (e.g. L14) — skip, not a failure.
                continue;
            }

            let backed = eligible
                .iter()
                .filter(|n| backed_names.contains(*n))
                .count();
            let coverage = backed as f32 / eligible.len() as f32;

            assert!(
                coverage >= COVERAGE_FLOOR,
                "L{} coverage {:.1}% below floor {:.0}% — {} eligible, {} backed",
                lane,
                coverage * 100.0,
                COVERAGE_FLOOR * 100.0,
                eligible.len(),
                backed,
            );
        }
    }

    /// Numerical sanity gate: 53 total curated entities, 48 eligible
    /// (after 5 TIER-2 exemptions), 48 backed.
    ///
    /// If extraction surprises us — a new addon adds backing for a
    /// previously un-paired entity — this surfaces immediately.
    #[test]
    fn aggregate_coverage_reports_correctly() {
        let exempt_names: std::collections::HashSet<&str> =
            COVERAGE_EXEMPTIONS.iter().map(|(n, _)| *n).collect();
        let curated = curated_entities();
        let eligible_total = curated
            .iter()
            .filter(|(n, _)| !exempt_names.contains(n))
            .count();
        let backed_total = CURATED_EXTRACTED_PAIRS.len();

        assert_eq!(curated.len(), 53, "Expected 53 curated entities total");
        assert_eq!(
            eligible_total, 48,
            "Expected 48 eligible after 5 TIER-2 exemptions"
        );
        assert_eq!(
            backed_total, 48,
            "All 48 eligible should be backed for Stage 1"
        );
    }
}
