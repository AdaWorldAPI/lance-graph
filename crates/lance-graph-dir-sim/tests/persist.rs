//! A history persists as its inputs (`image`) and restores to a store that
//! answers every read the same way (W-3, plan `dir-sim-persist-v1`).

use lance_graph_dir_sim::image::RestoreError;
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
const ALICE: u8 = 0xA1;
const BOB: u8 = 0xB0;
const EMPLOYEES: u8 = 0xE0;
const EXCHANGE: u8 = 0xEC;

const EXCHANGE_ACCESS: RuleId = RuleId {
    name: "ExchangeAccess",
    version: 1,
};
const EMPLOYEES_GET_EXCHANGE: RuleId = RuleId {
    name: "EmployeesGetExchange",
    version: 1,
};
const RENAME_MAIL: RuleId = RuleId {
    name: "RenameMail",
    version: 1,
};

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

fn observed() -> Observation {
    Observation {
        scope: SCOPE,
        nodes: vec![
            (
                g(ALICE),
                ObservedNode::user("alice@example.test", "alice@example.test"),
            ),
            (
                g(BOB),
                ObservedNode::user("bob@example.test", "bob@example.test"),
            ),
            (g(EMPLOYEES), ObservedNode::group()),
            (g(EXCHANGE), ObservedNode::group()),
        ],
        members: vec![(g(ALICE), g(EMPLOYEES)), (g(BOB), g(EMPLOYEES))],
    }
}
fn grant_alice() -> GrantGroup {
    GrantGroup {
        rule: EXCHANGE_ACCESS,
        group: g(EXCHANGE),
        to: vec![g(ALICE)],
    }
}
fn imply() -> ImplyGroup {
    ImplyGroup {
        rule: EMPLOYEES_GET_EXCHANGE,
        source: g(EMPLOYEES),
        target: g(EXCHANGE),
    }
}
fn ev(s: &str) -> Vec<EvidenceRef> {
    vec![EvidenceRef(s.into())]
}
fn chain_from(obs: Observation) -> (VersionStore, VersionId, VersionId, VersionId) {
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 1_000, obs).unwrap();
    let g1 = st.simulate(g0, &grant_alice(), &ev("REQ-1")).unwrap();
    let g2 = st.simulate(g1, &imply(), &ev("POLICY-7")).unwrap();
    (st, g0, g1, g2)
}

// g0 observed; g1, g2 simulated and g2 desired; a value interned between
// versions; g3 rejected (a verdict); g4 a second observation.
fn history() -> (VersionStore, Vec<VersionId>, ValueId) {
    let (mut st, g0, g1, g2) = chain_from(observed());
    st.promote_desired(g2).unwrap();
    let to = st.intern("Alice@Example.test");
    let g3 = st
        .simulate(
            g2,
            &SetPrimarySmtp {
                rule: RENAME_MAIL,
                user: g(BOB),
                to,
            },
            &ev("REQ-2"),
        )
        .unwrap();
    assert!(st.promote_desired(g3).is_err());
    let mut later = observed();
    later.members.pop();
    later.nodes.push((
        g(0xC0),
        ObservedNode::user("carol@example.test", "carol@example.test"),
    ));
    let g4 = st.observe("lab", 2_000, later).unwrap();
    (st, vec![g0, g1, g2, g3, g4], to)
}

fn reads(st: &VersionStore, vs: &[VersionId]) -> String {
    let mut out = String::new();
    for &v in vs {
        out += &format!(
            "{v:?} {:?} {:?} {:?} {:?}\n",
            st.version(v),
            st.diff(vs[0], v),
            st.verdict(v),
            st.view(v).unwrap().members_transitive(&g(EMPLOYEES)),
        );
    }
    out += &format!(
        "{:?} {:?} {:?}\n",
        st.tag(TAG_DESIRED),
        st.tag(TAG_OBSERVED),
        st.plan(vs[2])
    );
    out
}

#[test]
fn a_history_restores_to_the_same_reads() {
    let (st, vs, to) = history();
    let image = st.image();
    assert_eq!(image.versions.len(), 5);
    assert_eq!(image.roots.len(), 2);
    let back = VersionStore::restore(image.clone()).unwrap();
    assert_eq!(reads(&back, &vs), reads(&st, &vs));
    assert_eq!(back.value(to), Some("Alice@Example.test"));
    // The image of the restored store is the image it came from.
    assert_eq!(format!("{:?}", back.image()), format!("{image:?}"));
    // Work continues on the restored store with the same ids.
    let mut back = back;
    assert_eq!(back.intern("Alice@Example.test"), to);
}

#[test]
fn a_damaged_image_is_refused() {
    let (st, _, _) = history();
    let image = st.image();

    let mut i = image.clone();
    i.values.push(i.values[0].clone());
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::DuplicateValue(image.values.len())
    );

    let mut i = image.clone();
    i.values.clear();
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::IncompleteDictionary
    );

    let mut i = image.clone();
    let root = i.roots.remove(1).0;
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::MissingRoot(root)
    );

    let mut i = image.clone();
    i.versions.swap(1, 2);
    assert!(matches!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::VersionOrder { index: 1, .. }
    ));

    let mut i = image.clone();
    i.tags.push(("ghost".into(), VersionId(99)));
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::UnknownVersion(VersionId(99))
    );

    let mut i = image.clone();
    let extra = (VersionId(1), i.roots[0].1.clone());
    i.roots.push(extra);
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::NotARoot(VersionId(1))
    );

    // A delta that no longer applies to its parent.
    let mut i = image.clone();
    let d = i.versions[1].delta.clone();
    i.versions[2].delta.extend(d);
    assert!(matches!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::Apply(VersionId(2), _)
    ));
}

// Undo is a new version whose reads equal the earlier one's; nothing before
// it changes.
#[test]
fn a_revert_is_a_new_version_that_reads_as_the_earlier_one() {
    let (mut st, vs, _) = history();
    let (g0, g3) = (vs[0], vs[3]);
    let before = reads(&st, &vs);
    let r = st.revert(g3, g0, &ev("UNDO-1")).unwrap();
    assert_eq!(st.diff(r, g0).unwrap(), vec![]);
    assert_ne!(st.diff(g3, g0).unwrap(), vec![]);
    assert_eq!(st.version(r).unwrap().parent, Some(g3));
    assert_eq!(
        st.version(r).unwrap().origin,
        Origin::Simulated {
            rule: lance_graph_dir_sim::store::REVERT,
            evidence: vec![
                EvidenceRef("version:0".into()),
                EvidenceRef("UNDO-1".into())
            ],
        }
    );
    assert_eq!(reads(&st, &vs), before, "history is not rewritten");
    // Nothing to undo: no version.
    assert!(matches!(
        st.revert(r, g0, &[]),
        Err(SimError::EmptyProposal(_))
    ));
    // A revert persists like any version.
    let back = VersionStore::restore(st.image()).unwrap();
    assert_eq!(back.diff(r, g0).unwrap(), vec![]);
}

// ---- Persisted rules: text in, exact (name, version) out (decision 2(a)).

use lance_graph_dir_sim::image::{History, RuleRegistry, StoredImage, StoredRule, UnknownRule};

fn all_rules() -> RuleRegistry {
    RuleRegistry::new()
        .with(EXCHANGE_ACCESS)
        .with(EMPLOYEES_GET_EXCHANGE)
        .with(RENAME_MAIL)
}

// A stored image whose rules are all registered restores to a store that
// reads the same as the original.
#[test]
fn stored_rules_resolve_to_the_same_store() {
    let (st, vs, _) = history();
    let stored = StoredImage::from(st.image());
    let back = VersionStore::restore_stored(stored, &all_rules()).unwrap();
    assert_eq!(reads(&back, &vs), reads(&st, &vs));
}

// A rule registered under another version is a different behaviour: it is
// not taken in place of the stored one, and the restore names every
// version it could not resolve.
#[test]
fn another_version_of_a_rule_is_not_substituted() {
    let (st, vs, _) = history();
    let stored = StoredImage::from(st.image());
    let rules = RuleRegistry::new()
        .with(RuleId {
            name: "ExchangeAccess",
            version: 2,
        })
        .with(EMPLOYEES_GET_EXCHANGE);
    let err = VersionStore::restore_stored(stored, &rules).unwrap_err();
    assert_eq!(
        err,
        RestoreError::UnknownRules(vec![
            UnknownRule {
                version: vs[1],
                rule: StoredRule::from(EXCHANGE_ACCESS),
            },
            UnknownRule {
                version: vs[3],
                rule: StoredRule::from(RENAME_MAIL),
            },
        ])
    );
}

// The same history whose rules are unknown can still be read: every
// version's delta is stored, so its view needs no rule. The stored rule
// stays visible as it was written.
#[test]
fn an_unknown_rule_still_reads_as_history() {
    let (st, vs, _) = history();
    let stored = StoredImage::from(st.image());
    let h = History::restore(stored, &RuleRegistry::new()).unwrap();
    assert_eq!(h.unresolved().len(), 3);
    assert_eq!(h.rule(vs[1]), Some(&StoredRule::from(EXCHANGE_ACCESS)));
    assert_eq!(h.rule(vs[0]), None);
    for &v in &vs {
        assert_eq!(
            h.view(v).unwrap().members_transitive(&g(EMPLOYEES)),
            st.view(v).unwrap().members_transitive(&g(EMPLOYEES)),
        );
        assert_eq!(h.diff(vs[0], v), st.diff(vs[0], v));
        assert_eq!(h.verdict(v), st.verdict(v));
    }
    assert_eq!(h.tag(TAG_DESIRED), st.tag(TAG_DESIRED));
}

// The revert rule is the store's own and always resolves.
#[test]
fn a_revert_resolves_without_registering_it() {
    let (mut st, vs, _) = history();
    let r = st.revert(vs[2], vs[0], &ev("UNDO-1")).unwrap();
    let stored = StoredImage::from(st.image());
    let back = VersionStore::restore_stored(stored, &all_rules()).unwrap();
    assert_eq!(back.diff(vs[0], r), st.diff(vs[0], r));
}

// ---- The log: verified append-only batches (decision 3(b)).

use lance_graph_dir_sim::log::{replay, replay_prefix, LogError, LogState, Record, ReplayError};

/// Save `st` as one batch on top of `log`, returning the batch's records.
fn save(st: &VersionStore, log: &mut LogState) -> Vec<Record> {
    let batch = log
        .batch(&StoredImage::from(st.image()))
        .unwrap()
        .expect("something new");
    log.committed(&batch).unwrap();
    batch.records
}

fn restored(records: Vec<Record>) -> VersionStore {
    let (image, _) = replay(records).unwrap();
    VersionStore::restore_stored(image, &all_rules()).unwrap()
}

// Two saves, the second holding only what the first did not: the replayed
// log restores a store that reads the same as the original.
#[test]
fn a_log_of_two_batches_restores_the_same_reads() {
    let (mut st, mut vs, _) = history();
    let mut log = LogState::new();
    let mut records = save(&st, &mut log);
    let first_len = records.len();
    let r = st.revert(vs[2], vs[0], &ev("UNDO-1")).unwrap();
    st.validate(r).unwrap();
    vs.push(r);
    let second = save(&st, &mut log);
    assert!(
        second.len() < first_len,
        "the second batch is only the new part"
    );
    assert!(second.iter().all(|x| x.batch == 1));
    records.extend(second);
    let back = restored(records);
    assert_eq!(reads(&back, &vs), reads(&st, &vs));
    // Nothing new: no batch.
    assert!(log.batch(&StoredImage::from(st.image())).unwrap().is_none());
}

// The logical clock is the history's own: one batch holds five versions.
#[test]
fn one_batch_carries_many_logical_versions() {
    let (st, vs, _) = history();
    let records = save(&st, &mut LogState::new());
    assert!(records.iter().all(|r| r.batch == 0));
    let (image, state) = replay(records).unwrap();
    assert_eq!(image.versions.len(), vs.len());
    assert_eq!(state.next_batch(), 1);
}

fn two_batches() -> (Vec<Record>, Vec<Record>) {
    let (mut st, vs, _) = history();
    let mut log = LogState::new();
    let a = save(&st, &mut log);
    st.revert(vs[2], vs[0], &ev("UNDO-1")).unwrap();
    let b = save(&st, &mut log);
    (a, b)
}

// Each way a save can be torn is refused, naming the batch.
#[test]
fn a_torn_save_is_refused() {
    let (a, b) = two_batches();
    let all = || a.iter().chain(&b).cloned().collect::<Vec<_>>();

    let mut lost_tail = all();
    lost_tail.pop();
    assert!(matches!(
        replay(lost_tail),
        Err(ReplayError::RecordCount { batch: 1, .. })
    ));

    let headless: Vec<Record> = all()
        .into_iter()
        .filter(|r| !(r.batch == 1 && r.seq == 0))
        .collect();
    assert_eq!(replay(headless).unwrap_err(), ReplayError::MissingHeader(1));

    let mut flipped = all();
    let last = flipped.len() - 1;
    *flipped[last].body.last_mut().unwrap() ^= 1;
    assert_eq!(replay(flipped).unwrap_err(), ReplayError::Checksum(1));

    assert_eq!(
        replay(b.clone()).unwrap_err(),
        ReplayError::BatchGap {
            expected: 0,
            found: 1
        }
    );
}

// Reading up to the damage is its own call: it returns the good batches
// and names the bad one; `replay` returns nothing.
#[test]
fn the_prefix_up_to_the_damage_is_explicit() {
    let (a, mut b) = two_batches();
    b.pop();
    let (good, _) = replay(a.clone()).unwrap();
    let (image, state, err) = replay_prefix(a.into_iter().chain(b));
    assert!(matches!(
        err,
        Some(ReplayError::RecordCount { batch: 1, .. })
    ));
    assert_eq!(image.versions, good.versions);
    assert_eq!(state.next_batch(), 1);
}

// A batch is committed once, in order; an image that does not extend the
// log is refused.
#[test]
fn the_log_moves_forward_only() {
    let (st, _, _) = history();
    let mut log = LogState::new();
    let image = StoredImage::from(st.image());
    let first = log.batch(&image).unwrap().unwrap();
    let again = log.batch(&image).unwrap().unwrap();
    log.committed(&first).unwrap();
    assert_eq!(
        log.committed(&again),
        Err(LogError::StaleBatch {
            expected: 1,
            found: 0
        })
    );
    let mut shorter = VersionStore::new();
    shorter.observe("lab", 1, observed()).unwrap();
    assert_eq!(
        log.batch(&StoredImage::from(shorter.image())).unwrap_err(),
        LogError::NotAContinuation
    );
}
