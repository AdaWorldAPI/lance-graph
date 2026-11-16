//! A re-read of the directory is stored as the change from the previous
//! observation, not as a second whole observation (G2 first missing seam).

use lance_graph_dir_sim::image::StoredImage;
use lance_graph_dir_sim::log::{replay, LogState, RecordKind};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
const ALICE: u8 = 0xA1;
const BOB: u8 = 0xB0;
const CAROL: u8 = 0xCA;
const EMPLOYEES: u8 = 0xE0;
const EXCHANGE: u8 = 0xEC;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const OTHER: DirectoryScope = DirectoryScope(Guid128([0x6D; 16]));

const EXCHANGE_ACCESS: RuleId = RuleId {
    name: "ExchangeAccess",
    version: 1,
};

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

fn with_member(mut o: Observation, u: u8, grp: u8) -> Observation {
    o.members.push((g(u), g(grp)));
    o
}

fn roots(st: &VersionStore) -> usize {
    st.image().roots.len()
}

/// One changed membership: one Change, no new root, no Root record in the log.
#[test]
fn one_changed_membership_is_stored_as_one_change() {
    let mut st = VersionStore::new();
    let mut log = LogState::new();
    let v1 = st.observe("ad", 1_000, observed()).unwrap();
    let first = log.batch(&StoredImage::from(st.image())).unwrap().unwrap();
    log.committed(&first).unwrap();

    let v2 = st
        .reobserve("ad", 2_000, with_member(observed(), ALICE, EXCHANGE))
        .unwrap();
    let v = st.version(v2).unwrap();
    assert_eq!(v.parent, Some(v1));
    assert!(matches!(v.origin, Origin::Observed { .. }));
    assert_eq!(
        v.delta,
        vec![Change::AddMembership {
            user: g(ALICE),
            group: g(EXCHANGE)
        }]
    );
    assert_eq!(st.tag(TAG_OBSERVED), Some(v2));
    assert_eq!(roots(&st), 1, "no second root");

    let second = log.batch(&StoredImage::from(st.image())).unwrap().unwrap();
    let kinds: Vec<u8> = second.records.iter().map(|r| r.kind).collect();
    assert!(
        !kinds.contains(&(RecordKind::Root as u8)),
        "the second batch carries no whole observation: {kinds:?}"
    );
    assert!(kinds.contains(&(RecordKind::Version as u8)));
}

/// The re-observed version reads exactly as the same observation recorded as
/// a fresh root: nodes created and deleted, a property changed, memberships
/// added and removed.
#[test]
fn a_reobserved_version_reads_as_a_fresh_root() {
    let mut st = VersionStore::new();
    st.observe("ad", 1_000, observed()).unwrap();
    let mut next = observed();
    next.nodes.retain(|(id, _)| *id != g(BOB));
    next.members.retain(|(u, _)| *u != g(BOB));
    next.nodes[0].1 = ObservedNode::user("alice2@example.test", "alice@example.test");
    next.nodes.push((
        g(CAROL),
        ObservedNode::user("carol@example.test", "carol@example.test"),
    ));
    next.members.push((g(CAROL), g(EXCHANGE)));
    let v = st.reobserve("ad", 2_000, next.clone()).unwrap();
    let delta = st.version(v).unwrap().delta.clone();
    assert!(!delta.is_empty() && delta.len() < 10, "{delta:?}");

    let fresh = st.observe("ad", 2_000, next).unwrap();
    assert_eq!(st.diff(v, fresh).unwrap(), Vec::new());
    assert_eq!(st.diff(fresh, v).unwrap(), Vec::new());
}

/// A re-read that changes nothing is still recorded, with an empty delta.
#[test]
fn an_unchanged_reread_is_recorded_with_an_empty_delta() {
    let mut st = VersionStore::new();
    let v1 = st.observe("ad", 1_000, observed()).unwrap();
    let v2 = st.reobserve("ad", 2_000, observed()).unwrap();
    assert_ne!(v1, v2);
    assert!(st.version(v2).unwrap().delta.is_empty());
    assert_eq!(st.tag(TAG_OBSERVED), Some(v2));
}

/// Another directory scope, or nothing observed yet, is a new root.
#[test]
fn another_scope_or_a_first_read_is_a_root() {
    let mut st = VersionStore::new();
    let v1 = st.reobserve("ad", 1_000, observed()).unwrap();
    assert_eq!(st.version(v1).unwrap().parent, None);
    let mut other = observed();
    other.scope = OTHER;
    let v2 = st.reobserve("ad", 2_000, other).unwrap();
    assert_eq!(st.version(v2).unwrap().parent, None);
    assert_eq!(roots(&st), 2);
}

/// The log of a re-observed history replays to a store with the same reads.
#[test]
fn a_reobserved_history_persists_and_replays() {
    let mut st = VersionStore::new();
    let mut log = LogState::new();
    let v1 = st.observe("ad", 1_000, observed()).unwrap();
    let save = |st: &VersionStore, log: &mut LogState| {
        let batch = log.batch(&StoredImage::from(st.image())).unwrap().unwrap();
        log.committed(&batch).unwrap();
        batch.records
    };
    let mut records = save(&st, &mut log);
    let v2 = st
        .reobserve("ad", 2_000, with_member(observed(), BOB, EXCHANGE))
        .unwrap();
    records.extend(save(&st, &mut log));
    let (image, _) = replay(records).unwrap();
    let back = VersionStore::restore_stored(image, &image::RuleRegistry::new()).unwrap();
    assert_eq!(back.diff(v1, v2).unwrap(), st.diff(v1, v2).unwrap());
    assert_eq!(back.tag(TAG_OBSERVED), Some(v2));
}

/// Reality changes between observations are never intent. Reality drops
/// Bob from Employees, the desired version grants Alice Exchange on top of
/// that, and a later read shows Bob back: the plan holds only Alice's grant,
/// never a removal of Bob.
#[test]
fn the_plan_takes_intent_from_the_nearest_observation() {
    let mut st = VersionStore::new();
    st.observe("ad", 1_000, observed()).unwrap();
    let mut without_bob = observed();
    without_bob.members.retain(|(u, _)| *u != g(BOB));
    let o2 = st.reobserve("ad", 2_000, without_bob).unwrap();
    let grant = GrantGroup {
        rule: EXCHANGE_ACCESS,
        group: g(EXCHANGE),
        to: vec![g(ALICE)],
    };
    let target = st.simulate(o2, &grant, &[]).unwrap();
    st.promote_desired(target).unwrap();
    let o3 = st.reobserve("ad", 3_000, observed()).unwrap();
    let plan = st.plan(target).unwrap();
    assert_eq!(plan.basis, o3);
    assert_eq!(
        plan.ops,
        vec![PlannedOp::lower(Change::AddMembership {
            user: g(ALICE),
            group: g(EXCHANGE)
        })
        .unwrap()]
    );
}

/// Bytes a store's next log batch would write, over `log`.
fn batch_bytes(st: &VersionStore, log: &mut LogState) -> (usize, usize) {
    let b = log.batch(&StoredImage::from(st.image())).unwrap().unwrap();
    log.committed(&b).unwrap();
    let bytes = b.records.iter().map(|r| r.body.len()).sum();
    (bytes, b.records.len())
}

fn big_observation(users: u8) -> Observation {
    let mut o = Observation {
        scope: SCOPE,
        nodes: vec![
            (g(EMPLOYEES), ObservedNode::group()),
            (g(EXCHANGE), ObservedNode::group()),
        ],
        members: Vec::new(),
    };
    for u in 1..=users {
        let mail = format!("user{u}@example.test");
        o.nodes.push((g(u), ObservedNode::user(&mail, &mail)));
        o.members.push((g(u), g(EMPLOYEES)));
    }
    o
}

/// One changed membership in a 100-user directory: the re-observation's log
/// batch is a small fraction of what a second root costs.
#[test]
fn a_reobservation_retains_only_the_changed_coordinates() {
    let base = big_observation(100);
    let next = with_member(base.clone(), 7, EXCHANGE);

    let mut sparse = VersionStore::new();
    let mut sparse_log = LogState::new();
    sparse.observe("ad", 1_000, base.clone()).unwrap();
    batch_bytes(&sparse, &mut sparse_log);
    let v = sparse.reobserve("ad", 2_000, next.clone()).unwrap();
    let (sparse_bytes, _) = batch_bytes(&sparse, &mut sparse_log);
    assert_eq!(sparse.version(v).unwrap().delta.len(), 1);

    let mut full = VersionStore::new();
    let mut full_log = LogState::new();
    full.observe("ad", 1_000, base).unwrap();
    batch_bytes(&full, &mut full_log);
    full.observe("ad", 2_000, next).unwrap();
    let (full_bytes, _) = batch_bytes(&full, &mut full_log);

    eprintln!("second batch: reobserve {sparse_bytes} B, observe {full_bytes} B");
    assert!(
        sparse_bytes * 20 < full_bytes,
        "reobserve {sparse_bytes} B vs observe {full_bytes} B"
    );
}

/// Several re-reads in a row, each one equal to a fresh root of the same read.
#[test]
fn repeated_reobservations_each_read_as_a_fresh_root() {
    let mut st = VersionStore::new();
    st.observe("ad", 1_000, observed()).unwrap();
    let reads = [
        with_member(observed(), ALICE, EXCHANGE),
        with_member(with_member(observed(), ALICE, EXCHANGE), BOB, EXCHANGE),
        observed(),
        observed(),
    ];
    let mut prev = st.tag(TAG_OBSERVED).unwrap();
    for (i, r) in reads.iter().enumerate() {
        let v = st.reobserve("ad", 2_000 + i as i64, r.clone()).unwrap();
        assert_eq!(st.version(v).unwrap().parent, Some(prev), "read {i}");
        assert!(v > prev, "logical versions only move forward");
        // Same reads: the re-observed version against a root of the same
        // observation, compared inside one store.
        let root = st.observe("oracle", 0, r.clone()).unwrap();
        assert_eq!(st.diff(v, root).unwrap(), Vec::new(), "read {i}");
        // The oracle root took the observed tag; hand it back to the chain.
        prev = st.reobserve("ad", 2_000 + i as i64, r.clone()).unwrap();
        assert!(st.version(prev).unwrap().delta.is_empty());
    }
}

/// A simulated branch in between is neither the re-observation's parent nor
/// changed by it.
#[test]
fn a_simulated_branch_does_not_become_observed_history() {
    let mut st = VersionStore::new();
    let o1 = st.observe("ad", 1_000, observed()).unwrap();
    let grant = GrantGroup {
        rule: EXCHANGE_ACCESS,
        group: g(EXCHANGE),
        to: vec![g(ALICE)],
    };
    let s = st.simulate(o1, &grant, &[]).unwrap();
    let before = st.diff(o1, s).unwrap();
    let o2 = st
        .reobserve("ad", 2_000, with_member(observed(), BOB, EXCHANGE))
        .unwrap();
    let v = st.version(o2).unwrap();
    assert_eq!(
        v.parent,
        Some(o1),
        "parent is the observation, not the branch"
    );
    assert!(!v.delta.contains(&Change::AddMembership {
        user: g(ALICE),
        group: g(EXCHANGE)
    }));
    assert_eq!(
        st.diff(o1, s).unwrap(),
        before,
        "the branch reads as before"
    );
    assert!(matches!(
        st.version(s).unwrap().origin,
        Origin::Simulated { .. }
    ));
}

/// A refused observation creates no version, moves no tag and interns no
/// value.
#[test]
fn a_refused_observation_leaves_the_store_unchanged() {
    let mut st = VersionStore::new();
    st.observe("ad", 1_000, observed()).unwrap();
    let before = format!("{:?}", st.image());
    let mut bad = observed();
    bad.nodes.push((
        g(ALICE),
        ObservedNode::user("dup@example.test", "dup@example.test"),
    ));
    assert_eq!(
        st.reobserve("ad", 2_000, bad),
        Err(ObserveError::Build(BuildError::DuplicateNode(g(ALICE))))
    );
    assert_eq!(format!("{:?}", st.image()), before);
}

/// Source and read time are kept on the re-observation.
#[test]
fn a_reobservation_keeps_its_source_and_time() {
    let mut st = VersionStore::new();
    st.observe("ad", 1_000, observed()).unwrap();
    let v = st.reobserve("graph", 4_242, observed()).unwrap();
    assert_eq!(
        st.version(v).unwrap().origin,
        Origin::Observed {
            source: "graph".into(),
            observed_at_ms: 4_242
        }
    );
}
