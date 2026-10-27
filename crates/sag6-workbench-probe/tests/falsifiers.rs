//! D-SAG-6-W1 falsifiers. Gate ids are the D-SAG-6 F-gates
//! (`.claude/plans/d-sag-6-rust-codegraph-math-logic-tracert-v1.md` §5);
//! W-gates are new to the workbench.

use dr_math_binding_probe::Refusal as FrameRefusal;
use sag6_workbench_probe::fixture::{self, LANCE_GRAPH_MAIN, LINEAR_BLOCKED_BY, REPO};
use sag6_workbench_probe::{
    Answer, Corpus, Direction, GlobalId, Grade, Predicate, Reason, Record, Resolution, Witness,
    Workbench,
};

fn local(wb: &Workbench, id: &GlobalId) -> sag6_workbench_probe::LocalRef {
    match wb.resolve_identity(id) {
        Resolution::Resolved(r) => r,
        Resolution::NotInSnapshot => panic!("{id:?} is in the fixture"),
    }
}

fn ids(answer: &Answer) -> Vec<GlobalId> {
    match answer {
        Answer::Witnessed { witnesses, .. } => {
            let mut v: Vec<GlobalId> = witnesses.iter().map(|w| w.id.clone()).collect();
            v.sort();
            v
        }
        _ => Vec::new(),
    }
}

/// The independent baseline: a scan of the canonical records by `GlobalId`,
/// sharing no code with the ordinal lanes or the Quack program.
fn oracle(corpus: &Corpus, at: &GlobalId, p: Predicate, d: Direction) -> Vec<GlobalId> {
    let mut out = Vec::new();
    for r in corpus.records().filter(|r| r.predicate == p) {
        let symmetric = p.spec().symmetric;
        if (symmetric || d == Direction::Forward) && &r.subject == at {
            out.push(r.object.clone());
        }
        if (symmetric || d == Direction::Inverse) && &r.object == at {
            out.push(r.subject.clone());
        }
    }
    out.sort();
    out
}

fn every_identity(corpus: &Corpus) -> Vec<GlobalId> {
    let mut v: Vec<GlobalId> = corpus
        .records()
        .flat_map(|r| [r.subject.clone(), r.object.clone()])
        .collect();
    v.sort();
    v.dedup();
    v
}

/// F10. FAILS IF the Quack + mask-risc path and the scalar oracle disagree
/// on any identity, predicate and direction of the fixture.
#[test]
fn quack_matches_the_independent_baseline_everywhere() {
    let corpus = fixture::corpus();
    let wb = Workbench::new(corpus.clone());
    let mut compared = 0;
    let mut non_empty = 0;
    for id in every_identity(&corpus) {
        let at = local(&wb, &id);
        for p in Predicate::ALL {
            for d in [Direction::Forward, Direction::Inverse] {
                let expected = oracle(&corpus, &id, p, d);
                assert_eq!(
                    ids(&wb.query_relationships(&at, p, d)),
                    expected,
                    "{id:?} {p:?} {d:?}"
                );
                compared += 1;
                non_empty += usize::from(!expected.is_empty());
            }
        }
    }
    // Anti-vacuity: the comparison covers real edges, not only empty answers.
    assert!(non_empty >= 20, "only {non_empty} non-empty comparisons");
    assert_eq!(
        compared,
        every_identity(&corpus).len() * Predicate::ALL.len() * 2
    );
}

/// Inverse projection. FAILS IF `blocked_by` is not exactly Linear's own
/// `blockedBy` list for every issue that was read, or if any `blocked_by`
/// row was stored (the inverse is a projection of `blocks`).
#[test]
fn blocked_by_is_a_projection_that_matches_linear() {
    let wb = Workbench::new(fixture::corpus());
    for (issue, linear_says) in LINEAR_BLOCKED_BY {
        let at = local(&wb, &GlobalId::linear(issue));
        let mut expected: Vec<GlobalId> = linear_says.iter().map(|i| GlobalId::linear(i)).collect();
        expected.sort();
        let answer = wb.query_relationships(&at, Predicate::Blocks, Direction::Inverse);
        assert_eq!(ids(&answer), expected, "{issue}");
        if expected.is_empty() {
            assert_eq!(answer, Answer::NotHeld, "{issue} was read in full");
        }
    }
    let blocks_rows = wb
        .corpus()
        .records()
        .filter(|r| r.predicate == Predicate::Blocks)
        .count();
    assert_eq!(
        blocks_rows, 2,
        "EXO-112 blocks EXO-111, EXO-148 blocks EXO-55; nothing else"
    );
}

/// F06. FAILS IF an edge set that was not read answers `NotHeld`.
#[test]
fn missing_coverage_is_undetermined_not_false() {
    let wb = Workbench::new(fixture::corpus());
    // EXO-55 appears only as an object; its own relations were never read.
    let exo55 = local(&wb, &GlobalId::linear("EXO-55"));
    match wb.query_relationships(&exo55, Predicate::Blocks, Direction::Forward) {
        Answer::Undetermined { unread } => {
            assert_eq!(
                unread,
                vec![(
                    GlobalId::linear("EXO-55"),
                    Predicate::Blocks,
                    Direction::Forward
                )]
            )
        }
        other => panic!("expected Undetermined, got {other:?}"),
    }
    // EXO-50's relatedTo list was not imported in full: three edges, incomplete.
    let exo50 = local(&wb, &GlobalId::linear("EXO-50"));
    match wb.query_relationships(&exo50, Predicate::RelatedTo, Direction::Forward) {
        Answer::Witnessed {
            witnesses,
            complete,
        } => {
            assert_eq!(witnesses.len(), 3);
            assert!(!complete);
        }
        other => panic!("expected an incomplete witness set, got {other:?}"),
    }
}

/// F06, the silence twin. FAILS IF an edge set that WAS read in full and is
/// empty answers anything but `NotHeld` — a guard that always says UNKNOWN
/// is as useless as one that never does.
#[test]
fn a_fully_read_empty_edge_set_is_not_held() {
    let wb = Workbench::new(fixture::corpus());
    let exo50 = local(&wb, &GlobalId::linear("EXO-50"));
    assert_eq!(
        wb.query_relationships(&exo50, Predicate::Blocks, Direction::Inverse),
        Answer::NotHeld
    );
    let exo112 = local(&wb, &GlobalId::linear("EXO-112"));
    assert_eq!(
        wb.query_relationships(&exo112, Predicate::Blocks, Direction::Inverse),
        Answer::NotHeld
    );
}

/// F01. FAILS IF the same path at two revisions collapses into one identity,
/// or a question about one revision returns the other.
#[test]
fn the_same_symbol_at_two_revisions_is_two_identities() {
    let path = "crates/lance-graph-contract/src/mul.rs";
    let at_main = GlobalId::symbol(REPO, LANCE_GRAPH_MAIN, path, 1086);
    let at_other = GlobalId::symbol(REPO, "0000000000000000000000000000000000000000", path, 1086);
    assert_eq!(at_main.key, at_other.key, "same path and line");
    assert_ne!(at_main, at_other);

    let mut corpus = fixture::corpus();
    let original: Vec<Record> = corpus
        .records()
        .filter(|r| r.object == at_main)
        .cloned()
        .collect();
    assert_eq!(original.len(), 1);
    let mut moved = original[0].clone();
    moved.subject = GlobalId::linear("EXO-999");
    moved.object = at_other.clone();
    corpus.import([moved]);

    let wb = Workbench::new(corpus);
    let a = local(&wb, &at_main);
    let b = local(&wb, &at_other);
    assert_ne!(a.ordinal(), b.ordinal());
    assert_eq!(
        ids(&wb.query_relationships(&a, Predicate::MentionsSymbol, Direction::Inverse)),
        vec![GlobalId::linear("EXO-111")]
    );
    assert_eq!(
        ids(&wb.query_relationships(&b, Predicate::MentionsSymbol, Direction::Inverse)),
        vec![GlobalId::linear("EXO-999")]
    );
}

/// F07. FAILS IF a coordinate issued by one workbench is accepted by another
/// built from the identical corpus, where the same ordinal names the same
/// identity by coincidence.
#[test]
fn a_coordinate_from_another_frame_is_refused() {
    let a = Workbench::new(fixture::corpus());
    let b = Workbench::new(fixture::corpus());
    let id = GlobalId::linear("EXO-111");
    let from_a = local(&a, &id);
    assert_eq!(
        from_a.ordinal(),
        local(&b, &id).ordinal(),
        "same numbers on purpose"
    );
    assert_eq!(
        b.query_relationships(&from_a, Predicate::Blocks, Direction::Inverse),
        Answer::Refused(Reason::Frame(FrameRefusal::ForeignFrame))
    );
    // Silence twin: the same coordinate is accepted where it was issued.
    assert!(matches!(
        a.query_relationships(&from_a, Predicate::Blocks, Direction::Inverse),
        Answer::Witnessed { .. }
    ));
}

/// F11. FAILS IF a coordinate survives a reload: a changed source refuses
/// as `SourceChanged`, an identical reload as `FrameResealed`.
#[test]
fn a_reload_invalidates_every_earlier_coordinate() {
    let mut wb = Workbench::new(fixture::corpus());
    let id = GlobalId::linear("EXO-148");
    let before = local(&wb, &id);

    wb.reload(fixture::corpus());
    assert_eq!(
        wb.query_relationships(&before, Predicate::Blocks, Direction::Forward),
        Answer::Refused(Reason::Frame(FrameRefusal::FrameResealed))
    );

    let resealed = local(&wb, &id);
    let mut changed = fixture::corpus();
    let mut r = changed.records().next().cloned().expect("non-empty");
    r.snapshot.revision = "2026-10-28T00:00:00.000Z".to_owned();
    changed.import([r]);
    wb.reload(changed);
    assert_eq!(
        wb.query_relationships(&resealed, Predicate::Blocks, Direction::Forward),
        Answer::Refused(Reason::Frame(FrameRefusal::SourceChanged))
    );
    // A fresh coordinate under the new seal answers again.
    let fresh = local(&wb, &id);
    assert_eq!(
        ids(&wb.query_relationships(&fresh, Predicate::Blocks, Direction::Forward)),
        vec![GlobalId::linear("EXO-55")]
    );
}

/// F04 / F14. FAILS IF a prepared method binds to a predicate that has no
/// classid binding, including when the caller names a real OGAR classid.
#[test]
fn an_unbound_predicate_refuses_every_prepared_method() {
    let wb = Workbench::new(fixture::corpus());
    // 0x0102 = ogar_vocab::class_ids::PROJECT_WORK_ITEM, a real classid.
    for p in Predicate::ALL {
        assert_eq!(wb.prepare_method(p, 0x0102), Err(Reason::PredicateUnbound));
        let inspection = wb.inspect_predicate(p);
        assert_eq!(inspection.class_id, None);
        assert!(matches!(inspection.method, Answer::Unsupported(_)));
    }
    // The convergence candidate is reported, and is not a binding.
    assert_eq!(
        wb.inspect_predicate(Predicate::Blocks).spec.convergence,
        Some("ogar_vocab::class_ids::PROJECT_RELATION")
    );
}

/// Multi-hop. FAILS IF a chain of hops is answered instead of refused.
#[test]
fn multi_hop_is_refused_not_emulated() {
    let wb = Workbench::new(fixture::corpus());
    let at = local(&wb, &GlobalId::linear("EXO-111"));
    let report = wb.explain_blocker(&at);
    assert_eq!(
        report.transitive,
        Answer::Refused(Reason::MultiHopNotAdmitted)
    );
    // The one-hop parts still answer.
    assert_eq!(ids(&report.blocked_by), vec![GlobalId::linear("EXO-112")]);
    assert!(matches!(
        report.blocked_by,
        Answer::Witnessed { complete: true, .. }
    ));
    assert!(matches!(report.debt_rows, Answer::Undetermined { .. }));
}

/// F15. FAILS IF the inferred edge is dropped, hidden, or reported with any
/// grade other than `Hypothesis`.
#[test]
fn an_inferred_edge_is_reported_as_a_hypothesis() {
    let wb = Workbench::new(fixture::corpus());
    let hop = GlobalId::symbol(
        REPO,
        LANCE_GRAPH_MAIN,
        "crates/lance-graph-mask-risc/src/graph_hop.rs",
        55,
    );
    let unverified = wb.list_unverified_claims().expect("runs");
    assert_eq!(unverified.len(), 1);
    assert_eq!(unverified[0].object, hop);

    let at = local(&wb, &GlobalId::linear("EXO-140"));
    let Answer::Witnessed { witnesses, .. } =
        wb.query_relationships(&at, Predicate::MentionsSymbol, Direction::Forward)
    else {
        panic!("EXO-140 names symbols")
    };
    let graded: Vec<(&GlobalId, Grade)> = witnesses
        .iter()
        .map(|w: &Witness| (&w.id, w.grade))
        .collect();
    assert!(graded.contains(&(&hop, Grade::Hypothesis)));
    assert_eq!(
        graded
            .iter()
            .filter(|(_, g)| *g == Grade::ObservedSource)
            .count(),
        1
    );
}

/// F13. FAILS IF any query changes the canonical records.
#[test]
fn queries_do_not_change_the_canonical_records() {
    let corpus = fixture::corpus();
    let wb = Workbench::new(corpus.clone());
    for id in every_identity(&corpus) {
        let at = local(&wb, &id);
        for p in Predicate::ALL {
            let _ = wb.query_relationships(&at, p, Direction::Forward);
            let _ = wb.query_relationships(&at, p, Direction::Inverse);
        }
        let _ = wb.trace_provenance(&at);
        let _ = wb.explain_blocker(&at);
    }
    let _ = wb.list_unverified_claims();
    assert_eq!(wb.corpus(), &corpus);
}

/// W-cache. FAILS IF the derived state carries anything the canonical
/// records do not: a workbench rebuilt from the records alone answers
/// identically to the one it replaces.
#[test]
fn derived_state_is_disposable() {
    let first = Workbench::new(fixture::corpus());
    let answers = |wb: &Workbench| -> Vec<Answer> {
        every_identity(wb.corpus())
            .iter()
            .flat_map(|id| {
                let at = local(wb, id);
                Predicate::ALL.map(|p| wb.query_relationships(&at, p, Direction::Forward))
            })
            .collect()
    };
    let before = answers(&first);
    drop(first);
    let rebuilt = Workbench::new(fixture::corpus());
    assert_eq!(answers(&rebuilt), before);
}

/// W-replay. FAILS IF importing the same records twice changes the corpus
/// or any answer.
#[test]
fn a_replayed_import_is_idempotent() {
    let once = fixture::corpus();
    let mut twice = fixture::corpus();
    twice.import(fixture::records());
    assert_eq!(twice.len(), once.len());
    assert_eq!(twice, once);
    let diff = Workbench::compare_snapshots(&once, &twice);
    assert!(diff.added.is_empty() && diff.removed.is_empty());
}

/// compare_snapshots. FAILS IF a new record is not reported, or an
/// unchanged one is.
#[test]
fn compare_snapshots_reports_exactly_the_change() {
    let before = fixture::corpus();
    let mut after = fixture::corpus();
    let mut r = before.records().next().cloned().expect("non-empty");
    r.object = GlobalId::linear("EXO-777");
    after.import([r.clone()]);
    let diff = Workbench::compare_snapshots(&before, &after);
    assert_eq!(diff.added, vec![r]);
    assert!(diff.removed.is_empty());
}

/// Every witness carries its source. FAILS IF a witness has no snapshot
/// revision or no locator.
#[test]
fn every_witness_carries_its_source() {
    let wb = Workbench::new(fixture::corpus());
    let at = local(&wb, &GlobalId::linear("EXO-148"));
    let Answer::Witnessed { witnesses, .. } = wb.trace_provenance(&at) else {
        panic!("EXO-148 has edges")
    };
    assert_eq!(witnesses.len(), 6);
    for w in witnesses {
        assert!(!w.snapshot.revision.is_empty());
        assert!(w.locator.starts_with("https://"), "{}", w.locator);
    }
}
