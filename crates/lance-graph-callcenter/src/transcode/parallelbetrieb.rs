//! MySQL ↔ DataFusion ↔ SPO parallelbetrieb — **the one deliberate transition bandaid**.
//!
//! Everything else under `transcode/` is reusable Foundry primitive: outer
//! ontology → inner ontology mapping that should still make sense in five
//! years. *This* module is different. It exists to run two systems in
//! parallel — a MySQL ground-truth and the new SPO substrate over a
//! DataFusion surface — long enough to reconcile every drift and prove the
//! new side is correct.
//!
//! ## Why parallelbetrieb is necessary
//!
//! No phase advances with nonzero drift, by explicit user directive. To
//! detect drift you need the two systems answering the same question and
//! a reconciler diffing the answers. That reconciler is structurally
//! transitional:
//!
//! - **F1** — both systems answer; reconciler is on; consumers still hit
//!   MySQL.
//! - **F2** — both still answer; consumers now hit the new system; MySQL
//!   becomes the witness.
//! - **F3** — both still answer; the long-tail drift is chased to zero.
//! - **F4 / F5** — Foundry features go live on top of the new system.
//!   MySQL remains as the permanent reference (per directive), but the
//!   reconciler's *primary* purpose is satisfied.
//!
//! Even at F5 the reconciler stays — MySQL is permanent — but its mode
//! shifts from "consensus required for any commit" to "background witness
//! that emits drift events when something diverges". The bandaid framing
//! is for the **parallel-evaluation overhead**, not for the witness
//! itself.
//!
//! ## What this module provides today
//!
//! - [`DriftEvent`] — the JSON-shape every reconciler emits. Matches the
//!   schema MedCareV2's C# `LanceProbe` already POSTs to
//!   `/api/__parity/csharp` (medcare-rs PR #71).
//! - [`DriftKind`] — `Match` / `ValueDrift` / `ShapeDrift` / `MissingMysql`
//!   / `MissingLance`.
//! - [`Reconciler`] trait — the contract a parallelbetrieb runner
//!   implements. Two callers fill it: the C# `LanceProbe` (cross-language
//!   diverse-redundancy witness) and the Rust-side reconciler that
//!   compares MySQL `mysql_query` results to SPO `spo_lookup` results.
//!
//! ## What's deferred
//!
//! - The Rust-side reconciler implementation. It needs an MySQL query
//!   issuer + an SPO query issuer + a canonicaliser; the canonicaliser
//!   is the same field-rule table the C# side already implements
//!   (date-only, "F4" doubles, soft-delete coercion, second-truncated
//!   timestamps). Land both rules in one place when the Rust query
//!   path is wired (Phase 3 of `.claude/plans/sql-spo-ontology-bridge-v1.md`).
//! - The drift-event sink. Today the C# side POSTs to a medcare-rs
//!   route; the Rust side will publish to the same persistent ring
//!   buffer ([`crate::audit::LanceAuditSink`]) once the wiring lands.
//!
//! ## Hard rules
//!
//! - **No Foundry primitive in this module.** If the type is reusable
//!   beyond parallelbetrieb, it goes in a sibling module. Keep this
//!   file focused on the bandaid surface so it can be deleted (or
//!   reduced) when the F5 sunset clarifies what stays.
//! - **No silent reconciliation.** If MySQL says `pf_delete=NULL` and
//!   SPO says `deleted=false`, the canonicaliser collapses both to
//!   `false` and emits `Match`; if MySQL says `pf_delete=1` and SPO
//!   says `deleted=false`, the reconciler emits `ValueDrift`. No
//!   silent agree-to-disagree.

use core::fmt;

/// Kind of drift the reconciler observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftKind {
    /// MySQL and SPO agree after canonicalisation.
    Match,
    /// Both sides have a row; one or more field values disagree.
    ValueDrift,
    /// Both sides have a row; the shape (columns / list lengths) disagrees.
    ShapeDrift,
    /// MySQL has a row, SPO doesn't.
    MissingLance,
    /// SPO has a row, MySQL doesn't.
    MissingMysql,
}

impl fmt::Display for DriftKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DriftKind::Match => "Match",
            DriftKind::ValueDrift => "ValueDrift",
            DriftKind::ShapeDrift => "ShapeDrift",
            DriftKind::MissingLance => "MissingLance",
            DriftKind::MissingMysql => "MissingMysql",
        })
    }
}

/// One field-level drift entry: identifies the field path and both
/// canonicalised representations. Mirrors the C# `DriftField`.
#[derive(Clone, Debug, PartialEq)]
pub struct DriftField {
    pub path: String,
    pub mysql: String,
    pub lance: String,
}

/// One reconciliation outcome. Identical schema to MedCareV2's C#
/// `DriftEvent.ToJson()` so the same JSON document parses on both sides.
#[derive(Clone, Debug)]
pub struct DriftEvent {
    /// Origin of the event — e.g. `medcarev2-lance-probe` (C# side) or
    /// `medcare-rs-reconciler` (Rust side).
    pub source: &'static str,
    pub route: String,
    pub method: &'static str,
    pub kind: DriftKind,
    pub fields: Vec<DriftField>,
    /// ISO 8601 UTC second-resolution timestamp when the comparison ran.
    pub captured_at: String,
}

impl DriftEvent {
    /// Construct a `Match` event — no drift, just confirms the
    /// reconciliation ran. Useful for sampling counts and "we saw nothing
    /// wrong" telemetry.
    pub fn matched(source: &'static str, route: impl Into<String>) -> Self {
        Self {
            source,
            route: route.into(),
            method: "GET",
            kind: DriftKind::Match,
            fields: Vec::new(),
            captured_at: now_iso8601_seconds(),
        }
    }

    /// Construct a `ValueDrift` event with one or more field
    /// disagreements. The fields list must be non-empty — empty `fields`
    /// + `ValueDrift` would be a contradiction.
    pub fn value_drift(
        source: &'static str,
        route: impl Into<String>,
        fields: Vec<DriftField>,
    ) -> Self {
        debug_assert!(
            !fields.is_empty(),
            "ValueDrift requires at least one DriftField; use matched() if no drift"
        );
        Self {
            source,
            route: route.into(),
            method: "GET",
            kind: DriftKind::ValueDrift,
            fields,
            captured_at: now_iso8601_seconds(),
        }
    }
}

/// Contract for any parallelbetrieb implementation.
///
/// Two implementors are anticipated:
/// 1. **C#-side** `LanceProbe` (already shipped in MedCareV2 PR #1, #2,
///    #3) — runs in the legacy desktop app's process; POSTs DriftEvents
///    to a medcare-rs route.
/// 2. **Rust-side** reconciler (deferred) — runs as part of the
///    membrane; compares MySQL query results to SPO results and
///    publishes DriftEvents directly to the audit log.
///
/// Both report drift in the same JSON shape so a single dashboard can
/// merge both feeds.
pub trait Reconciler {
    /// Reconcile one query — implementations issue the MySQL form, the
    /// SPO form, canonicalise both, diff, and return the event.
    /// Implementations must NOT throw; bubble errors as
    /// `DriftKind::ShapeDrift` with the error message in the field.
    fn reconcile(&self, route: &str) -> DriftEvent;
}

fn now_iso8601_seconds() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Cheap 32-byte ISO 8601: yyyy-mm-ddTHH:MM:SSZ. Avoids pulling chrono.
    let (y, m, d, hh, mm, ss) = unix_to_ymd_hms(secs);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Convert a Unix timestamp (seconds) to (year, month, day, hour, minute,
/// second). Self-contained so the parallelbetrieb module has no
/// chrono / time dep.
fn unix_to_ymd_hms(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let day = secs / 86_400;
    let secs_of_day = secs - day * 86_400;
    let hh = (secs_of_day / 3600) as u32;
    let mm = ((secs_of_day / 60) % 60) as u32;
    let ss = (secs_of_day % 60) as u32;

    // Civil-from-days, Howard Hinnant. Public-domain algorithm; matches
    // chrono's `NaiveDate::from_num_days_from_ce_opt` modulo offset.
    let z = day as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y_civil = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d_civil = doy - (153 * mp + 2) / 5 + 1;
    let m_civil = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y_civil + i64::from(m_civil <= 2);
    (y as u32, m_civil as u32, d_civil as u32, hh, mm, ss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_kind_displays_camelcase() {
        assert_eq!(DriftKind::Match.to_string(), "Match");
        assert_eq!(DriftKind::ValueDrift.to_string(), "ValueDrift");
        assert_eq!(DriftKind::MissingMysql.to_string(), "MissingMysql");
    }

    #[test]
    fn matched_event_has_no_fields() {
        let ev = DriftEvent::matched("test", "/api/patient/1");
        assert_eq!(ev.kind, DriftKind::Match);
        assert!(ev.fields.is_empty());
        assert_eq!(ev.method, "GET");
        assert_eq!(ev.source, "test");
    }

    #[test]
    fn value_drift_carries_field_path() {
        let ev = DriftEvent::value_drift(
            "test",
            "/api/patient/1",
            vec![DriftField {
                path: "$.name".to_string(),
                mysql: "Anna".to_string(),
                lance: "Anneliese".to_string(),
            }],
        );
        assert_eq!(ev.kind, DriftKind::ValueDrift);
        assert_eq!(ev.fields.len(), 1);
        assert_eq!(ev.fields[0].path, "$.name");
    }

    #[test]
    fn iso8601_handles_unix_epoch() {
        let (y, m, d, hh, mm, ss) = unix_to_ymd_hms(0);
        assert_eq!((y, m, d, hh, mm, ss), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn iso8601_handles_one_day_later() {
        // Unix 86400 = 1970-01-02 00:00:00 UTC (one full day after epoch).
        let (y, m, d, hh, mm, ss) = unix_to_ymd_hms(86_400);
        assert_eq!((y, m, d, hh, mm, ss), (1970, 1, 2, 0, 0, 0));
    }

    #[test]
    fn iso8601_handles_y2k_boundary() {
        // Unix 946_684_800 = 2000-01-01 00:00:00 UTC (Y2K rollover).
        let (y, m, d, hh, mm, ss) = unix_to_ymd_hms(946_684_800);
        assert_eq!((y, m, d, hh, mm, ss), (2000, 1, 1, 0, 0, 0));
    }

    #[test]
    fn iso8601_handles_leap_year_feb_29() {
        // Unix 1_582_934_400 = 2020-02-29 00:00:00 UTC (most recent
        // observed leap day before this code shipped).
        let (y, m, d, hh, mm, ss) = unix_to_ymd_hms(1_582_934_400);
        assert_eq!((y, m, d, hh, mm, ss), (2020, 2, 29, 0, 0, 0));
    }

    #[test]
    fn iso8601_round_trips_seconds_and_minutes() {
        // 1970-01-01 12:34:56 UTC = 12*3600 + 34*60 + 56 = 45_296.
        let (y, m, d, hh, mm, ss) = unix_to_ymd_hms(45_296);
        assert_eq!((y, m, d, hh, mm, ss), (1970, 1, 1, 12, 34, 56));
    }

    #[test]
    fn captured_at_format_is_well_formed() {
        let ev = DriftEvent::matched("test", "/api/patient/1");
        assert_eq!(ev.captured_at.len(), 20); // yyyy-mm-ddTHH:MM:SSZ
        assert!(ev.captured_at.ends_with('Z'));
        assert!(ev.captured_at.contains('T'));
    }
}
