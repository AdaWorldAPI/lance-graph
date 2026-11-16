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
