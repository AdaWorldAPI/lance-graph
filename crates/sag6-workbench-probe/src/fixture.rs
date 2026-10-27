//! The smallest lawful fixture: a bounded slice of real upstream records,
//! read on 2026-10-27, each with the snapshot it came from.
//!
//! Sources, exactly as read:
//! - Linear issues EXO-111, EXO-112, EXO-140, EXO-148 and EXO-50, with their
//!   relation lists (`blocks`, `blockedBy`, `relatedTo`) and descriptions.
//!   Each record's snapshot is the issue it was read from, at that issue's
//!   `updatedAt`.
//! - lance-graph at `main` = [`LANCE_GRAPH_MAIN`]: the source lines the issues
//!   name, and the `.claude/board/ISSUES.md` rows that track them, checked at
//!   that commit.
//!
//! What is deliberately NOT here:
//! - Relations of issues that were not read (EXO-55, EXO-45, EXO-46, EXO-114,
//!   EXO-135). They appear as objects, and their own edge sets are not marked
//!   complete, so questions about them answer `Undetermined`.
//! - EXO-50's full `relatedTo` list (more than 80 issues). Only the three edges
//!   the other issues report are imported, and EXO-50's `relatedTo` is not
//!   marked complete.
//! - Any edge inferred from a name match. The one inferred edge is imported
//!   as [`Grade::Hypothesis`] and says so in its locator.

use crate::{Corpus, Direction, GlobalId, Grade, Predicate, Record, Snapshot, SourceSystem};

/// lance-graph `main` at the time of the read.
pub const LANCE_GRAPH_MAIN: &str = "aa569bc5c41ca38edf1ce3db945007543b0c82f6";
pub const REPO: &str = "AdaWorldAPI/lance-graph";
pub const FETCHED_AT: &str = "2026-10-27";

/// `updatedAt` of each Linear issue that was read.
pub const LINEAR_READ: [(&str, &str); 5] = [
    ("EXO-111", "2026-10-23T09:25:53.396Z"),
    ("EXO-112", "2026-10-23T08:48:48.110Z"),
    ("EXO-140", "2026-10-23T08:58:42.268Z"),
    ("EXO-148", "2026-10-25T15:52:08.088Z"),
    ("EXO-50", "2026-10-26T22:11:29.036Z"),
];

/// Linear's own `blockedBy` lists for the issues read, kept apart from the
/// imported records. The workbench never stores `blocked_by`; it projects it
/// from `blocks`, and the tests compare that projection against this list.
pub const LINEAR_BLOCKED_BY: [(&str, &[&str]); 5] = [
    ("EXO-111", &["EXO-112"]),
    ("EXO-112", &[]),
    ("EXO-140", &[]),
    ("EXO-148", &[]),
    ("EXO-50", &[]),
];

fn linear_snapshot(issue: &str) -> Snapshot {
    let revision = LINEAR_READ
        .iter()
        .find(|(id, _)| *id == issue)
        .map(|(_, at)| *at)
        .expect("only issues that were read have snapshots");
    Snapshot {
        system: SourceSystem::Linear,
        revision: revision.to_owned(),
        object: issue.to_owned(),
        fetched_at: FETCHED_AT.to_owned(),
    }
}

fn linear_url(issue: &str) -> String {
    format!("https://linear.app/exo-red/issue/{issue}")
}

fn blob_url(path: &str, line: u32) -> String {
    format!("https://github.com/{REPO}/blob/{LANCE_GRAPH_MAIN}/{path}#L{line}")
}

fn edge(
    subject: GlobalId,
    predicate: Predicate,
    object: GlobalId,
    from_issue: &str,
    locator: String,
    grade: Grade,
) -> Record {
    Record {
        subject,
        predicate,
        object,
        snapshot: linear_snapshot(from_issue),
        locator,
        grade,
    }
}

/// The fixture's records.
pub fn records() -> Vec<Record> {
    use Grade::*;
    use Predicate::*;
    let l = GlobalId::linear;
    let sym = |path: &str, line: u32| GlobalId::symbol(REPO, LANCE_GRAPH_MAIN, path, line);
    let row = |anchor: &str| GlobalId::debt_row(REPO, LANCE_GRAPH_MAIN, anchor);
    let pr = |n: u32| GlobalId::pull_request(REPO, n);

    vec![
        // Linear relations, read from the issue that lists them.
        edge(
            l("EXO-112"),
            Blocks,
            l("EXO-111"),
            "EXO-112",
            linear_url("EXO-112"),
            ObservedSource,
        ),
        edge(
            l("EXO-148"),
            Blocks,
            l("EXO-55"),
            "EXO-148",
            linear_url("EXO-148"),
            ObservedSource,
        ),
        edge(
            l("EXO-111"),
            RelatedTo,
            l("EXO-50"),
            "EXO-111",
            linear_url("EXO-111"),
            ObservedSource,
        ),
        edge(
            l("EXO-112"),
            RelatedTo,
            l("EXO-50"),
            "EXO-112",
            linear_url("EXO-112"),
            ObservedSource,
        ),
        edge(
            l("EXO-148"),
            RelatedTo,
            l("EXO-45"),
            "EXO-148",
            linear_url("EXO-148"),
            ObservedSource,
        ),
        edge(
            l("EXO-148"),
            RelatedTo,
            l("EXO-50"),
            "EXO-148",
            linear_url("EXO-148"),
            ObservedSource,
        ),
        edge(
            l("EXO-140"),
            RelatedTo,
            l("EXO-46"),
            "EXO-140",
            linear_url("EXO-140"),
            ObservedSource,
        ),
        edge(
            l("EXO-140"),
            RelatedTo,
            l("EXO-114"),
            "EXO-140",
            linear_url("EXO-140"),
            ObservedSource,
        ),
        edge(
            l("EXO-140"),
            RelatedTo,
            l("EXO-135"),
            "EXO-140",
            linear_url("EXO-140"),
            ObservedSource,
        ),
        // Source locations the issue texts name, pinned at LANCE_GRAPH_MAIN.
        edge(
            l("EXO-111"),
            MentionsSymbol,
            sym("crates/lance-graph-contract/src/mul.rs", 1086),
            "EXO-111",
            blob_url("crates/lance-graph-contract/src/mul.rs", 1086),
            ObservedSource,
        ),
        edge(
            l("EXO-140"),
            MentionsSymbol,
            sym("crates/lance-graph-quack/src/lib.rs", 387),
            "EXO-140",
            blob_url("crates/lance-graph-quack/src/lib.rs", 387),
            ObservedSource,
        ),
        edge(
            l("EXO-148"),
            MentionsSymbol,
            sym("crates/mork-pathmap-probe/src/rmap.rs", 70),
            "EXO-148",
            blob_url("crates/mork-pathmap-probe/src/rmap.rs", 70),
            ObservedSource,
        ),
        // Technical-debt rows the issue texts name as their GitHub row.
        edge(
            l("EXO-140"),
            TrackedByDebtRow,
            row("ISS-ADMISSION-B-CALL-SITE-CLASSIFICATION-NOT-STARTED"),
            "EXO-140",
            blob_url(".claude/board/ISSUES.md", 74),
            ObservedSource,
        ),
        edge(
            l("EXO-148"),
            TrackedByDebtRow,
            row("ISS-DURABLEREF-THOUGHTCOMMITID"),
            "EXO-148",
            blob_url(".claude/board/ISSUES.md", 176),
            ObservedSource,
        ),
        // Pull requests the issue texts cite.
        edge(
            l("EXO-140"),
            CitesPr,
            pr(1640),
            "EXO-140",
            linear_url("EXO-140"),
            ObservedSource,
        ),
        edge(
            l("EXO-140"),
            CitesPr,
            pr(1732),
            "EXO-140",
            linear_url("EXO-140"),
            ObservedSource,
        ),
        edge(
            l("EXO-148"),
            CitesPr,
            pr(1730),
            "EXO-148",
            linear_url("EXO-148"),
            ObservedSource,
        ),
        // The session id EXO-140's text records.
        edge(
            l("EXO-140"),
            AuthoredInSession,
            GlobalId::session("session_019fbZ92dFKQ5BJHrA3YcH9A"),
            "EXO-140",
            linear_url("EXO-140"),
            ObservedSource,
        ),
        // The one inference: EXO-140 says "the gather hop, behind an opt-in
        // feature" and does not name a file. graph_hop.rs:55 is the reader's
        // guess, so it is a hypothesis, not an observed edge.
        edge(
            l("EXO-140"),
            MentionsSymbol,
            sym("crates/lance-graph-mask-risc/src/graph_hop.rs", 55),
            "EXO-140",
            "inferred: EXO-140 names 'the gather hop, behind an opt-in feature' but no file"
                .to_owned(),
            Hypothesis,
        ),
    ]
}

/// The fixture corpus, with the edge sets that were read in full.
pub fn corpus() -> Corpus {
    let mut c = Corpus::default();
    c.import(records());
    for (issue, _) in LINEAR_READ {
        let id = GlobalId::linear(issue);
        c.mark_complete(id.clone(), Predicate::Blocks, Direction::Forward);
        c.mark_complete(id.clone(), Predicate::Blocks, Direction::Inverse);
        // EXO-50's relatedTo list was read but not imported in full.
        if issue != "EXO-50" {
            c.mark_complete(id, Predicate::RelatedTo, Direction::Forward);
        }
    }
    c
}
