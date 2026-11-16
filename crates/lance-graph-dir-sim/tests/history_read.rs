//! D-DIRSIM-DELTA-READ-1: reading what changed between two observations
//! from the stored deltas (`VersionStore::history`), checked against a
//! scalar oracle that compares the two `Observation`s directly and shares no
//! code with dir-sim's diff.

use std::collections::BTreeSet;

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
const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const OTHER: DirectoryScope = DirectoryScope(Guid128([0x6D; 16]));
const GRANT: RuleId = RuleId {
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

/// A semantic change, by value: what the oracle and the store both report.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Fact {
    AddMember(Guid128, Guid128),
    RemoveMember(Guid128, Guid128),
    AddNode(Guid128),
    DeleteNode(Guid128),
    Upn(Guid128, Option<String>, Option<String>),
    Smtp(Guid128, Option<String>, Option<String>),
    Active(Guid128, Option<bool>, Option<bool>),
}

/// The oracle: the difference of two observations, by set comparison.
fn oracle(a: &Observation, b: &Observation) -> BTreeSet<Fact> {
    let mut out = BTreeSet::new();
    let (ma, mb): (BTreeSet<_>, BTreeSet<_>) = (
        a.members.iter().copied().collect(),
        b.members.iter().copied().collect(),
    );
    for &(u, gr) in mb.difference(&ma) {
        out.insert(Fact::AddMember(u, gr));
    }
    for &(u, gr) in ma.difference(&mb) {
        out.insert(Fact::RemoveMember(u, gr));
    }
    for (id, nb) in &b.nodes {
        match a.nodes.iter().find(|(x, _)| x == id) {
            None => {
                out.insert(Fact::AddNode(*id));
            }
            Some((_, na)) => {
                if na.upn != nb.upn {
                    out.insert(Fact::Upn(*id, na.upn.clone(), nb.upn.clone()));
                }
                if na.primary_smtp != nb.primary_smtp {
                    out.insert(Fact::Smtp(
                        *id,
                        na.primary_smtp.clone(),
                        nb.primary_smtp.clone(),
                    ));
                }
                if na.active != nb.active {
                    out.insert(Fact::Active(*id, na.active, nb.active));
                }
            }
        }
    }
    for (id, _) in &a.nodes {
        if !b.nodes.iter().any(|(x, _)| x == id) {
            out.insert(Fact::DeleteNode(*id));
        }
    }
    out
}

/// The store's change, hydrated to values at the presentation boundary.
fn fact(st: &VersionStore, c: &Change) -> Fact {
    let text = |v: &Option<ValueId>| v.map(|v| st.value(v).unwrap().to_string());
    match c {
        Change::AddMembership { user, group } => Fact::AddMember(*user, *group),
        Change::RemoveMembership { user, group } => Fact::RemoveMember(*user, *group),
        Change::CreateNode { node, .. } => Fact::AddNode(*node),
        Change::DeleteNode { node, .. } => Fact::DeleteNode(*node),
        Change::SetAttribute {
            node,
            attribute: Attribute::Upn,
            from,
            to,
        } => Fact::Upn(*node, text(from), text(to)),
        Change::SetAttribute {
            node,
            attribute: Attribute::PrimarySmtp,
            from,
            to,
        } => Fact::Smtp(*node, text(from), text(to)),
        Change::SetActive { node, from, to } => Fact::Active(*node, *from, *to),
        other => panic!("not in the oracle: {other:?}"),
    }
}

fn events(st: &VersionStore, from: VersionId, to: VersionId) -> Vec<(VersionId, Vec<Fact>)> {
    st.history(from, to)
        .unwrap()
        .map(|v| (v.id, v.delta.iter().map(|c| fact(st, c)).collect()))
        .collect()
}

fn net(st: &VersionStore, from: VersionId, to: VersionId) -> BTreeSet<Fact> {
    st.diff(from, to)
        .unwrap()
        .iter()
        .map(|c| fact(st, c))
        .collect()
}

/// 1. One changed membership is exactly one change, in one version.
#[test]
fn one_membership_change_is_one_change() {
    let mut st = VersionStore::new();
    let a = observed();
    let b = with_member(observed(), ALICE, EXCHANGE);
    let o1 = st.observe("ad", 1, a.clone()).unwrap();
    let o2 = st.reobserve("ad", 2, b.clone()).unwrap();
    let got = events(&st, o1, o2);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, o2);
    let want: Vec<Fact> = oracle(&a, &b).into_iter().collect();
    assert_eq!(got[0].1, want);
    assert_eq!(want.len(), 1);
}

/// 2. An unchanged re-read is an event with no changes.
#[test]
fn an_unchanged_reread_is_an_event_without_changes() {
    let mut st = VersionStore::new();
    let o1 = st.observe("ad", 1, observed()).unwrap();
    let o2 = st.reobserve("ad", 2, observed()).unwrap();
    let got = events(&st, o1, o2);
    assert_eq!(got, vec![(o2, Vec::new())]);
    let v = st.version(o2).unwrap();
    assert!(matches!(
        &v.origin,
        Origin::Observed {
            observed_at_ms: 2,
            ..
        }
    ));
}

/// 3. Four reads: each adjacent pair's events equal the oracle, and the net
/// reading across the whole interval equals the oracle of the end points.
#[test]
fn four_reads_agree_pairwise_and_across_the_interval() {
    let reads = [
        observed(),
        with_member(observed(), ALICE, EXCHANGE),
        with_member(with_member(observed(), ALICE, EXCHANGE), BOB, EXCHANGE),
        with_member(observed(), BOB, EXCHANGE),
    ];
    let mut st = VersionStore::new();
    let mut ids = vec![st.observe("ad", 0, reads[0].clone()).unwrap()];
    for (i, r) in reads.iter().enumerate().skip(1) {
        ids.push(st.reobserve("ad", i as i64, r.clone()).unwrap());
    }
    for i in 0..3 {
        let got = events(&st, ids[i], ids[i + 1]);
        assert_eq!(got.len(), 1, "pair {i}");
        let got: BTreeSet<Fact> = got[0].1.iter().cloned().collect();
        assert_eq!(got, oracle(&reads[i], &reads[i + 1]), "pair {i}");
    }
    let whole = events(&st, ids[0], ids[3]);
    assert_eq!(
        whole.iter().map(|(v, _)| *v).collect::<Vec<_>>(),
        vec![ids[3], ids[2], ids[1]],
        "newest first, one item per version"
    );
    assert_eq!(net(&st, ids[0], ids[3]), oracle(&reads[0], &reads[3]));
}

/// 4. A simulated branch is not observed history.
#[test]
fn a_simulated_branch_is_not_observed_history() {
    let mut st = VersionStore::new();
    let o1 = st.observe("ad", 1, observed()).unwrap();
    let grant = GrantGroup {
        rule: GRANT,
        group: g(EXCHANGE),
        to: vec![g(ALICE)],
    };
    let s = st.simulate(o1, &grant, &[]).unwrap();
    let o2 = st
        .reobserve("ad", 2, with_member(observed(), BOB, EXCHANGE))
        .unwrap();
    let ids: Vec<VersionId> = st.history(o1, o2).unwrap().map(|v| v.id).collect();
    assert_eq!(ids, vec![o2], "the branch is not on the observed lineage");
    assert_eq!(
        st.history(s, o2).err(),
        Some(HistoryRefusal::NotAncestor { from: s, to: o2 })
    );
    let branch: Vec<&Version> = st.history(o1, s).unwrap().collect();
    assert!(matches!(branch[0].origin, Origin::Simulated { .. }));
}

/// 5. A node deleted and created again is two events, while the net reading
/// over the same interval is empty.
#[test]
fn delete_and_recreate_are_events_not_an_unchanged_node() {
    let mut st = VersionStore::new();
    let o1 = st.observe("ad", 1, observed()).unwrap();
    let mut gone = observed();
    gone.nodes.retain(|(id, _)| *id != g(BOB));
    gone.members.retain(|(u, _)| *u != g(BOB));
    let o2 = st.reobserve("ad", 2, gone).unwrap();
    let o3 = st.reobserve("ad", 3, observed()).unwrap();
    let all: Vec<Fact> = events(&st, o1, o3)
        .into_iter()
        .flat_map(|(_, f)| f)
        .collect();
    assert!(all.contains(&Fact::DeleteNode(g(BOB))), "{all:?}");
    assert!(all.contains(&Fact::AddNode(g(BOB))), "{all:?}");
    assert!(net(&st, o1, o3).is_empty());
    assert!(oracle(&observed(), &observed()).is_empty());
    assert_eq!(events(&st, o2, o3).len(), 1);
}

/// 6. Another directory scope, an unknown id and a reversed range refuse.
#[test]
fn another_scope_an_unknown_id_and_a_reversed_range_refuse() {
    let mut st = VersionStore::new();
    let o1 = st.observe("ad", 1, observed()).unwrap();
    let o2 = st
        .reobserve("ad", 2, with_member(observed(), ALICE, EXCHANGE))
        .unwrap();
    let mut other = observed();
    other.scope = OTHER;
    let r = st.reobserve("ad", 3, other).unwrap();
    assert_eq!(
        st.history(o1, r).err(),
        Some(HistoryRefusal::NotAncestor { from: o1, to: r })
    );
    assert_eq!(
        st.history(o2, o1).err(),
        Some(HistoryRefusal::NotAncestor { from: o2, to: o1 })
    );
    let ghost = VersionId(999);
    assert_eq!(
        st.history(ghost, o2).err(),
        Some(HistoryRefusal::UnknownVersion(ghost))
    );
    assert_eq!(
        st.history(o1, ghost).err(),
        Some(HistoryRefusal::UnknownVersion(ghost))
    );
}

/// 7. A changed attribute carries its before and after values; absent,
/// explicit empty and unknown stay distinct.
#[test]
fn attribute_changes_carry_before_and_after() {
    let mut st = VersionStore::new();
    let mut a = observed();
    a.nodes[1].1.active = None;
    let o1 = st.observe("ad", 1, a.clone()).unwrap();
    let mut b = a.clone();
    b.nodes[0].1.upn = Some("alice2@example.test".into());
    b.nodes[0].1.primary_smtp = Some(String::new());
    b.nodes[1].1.active = Some(false);
    let o2 = st.reobserve("ad", 2, b.clone()).unwrap();
    let mut c = b.clone();
    c.nodes[0].1.primary_smtp = None;
    let o3 = st.reobserve("ad", 3, c.clone()).unwrap();

    let first: BTreeSet<Fact> = events(&st, o1, o2)[0].1.iter().cloned().collect();
    assert_eq!(first, oracle(&a, &b));
    assert!(first.contains(&Fact::Upn(
        g(ALICE),
        Some("alice@example.test".into()),
        Some("alice2@example.test".into())
    )));
    assert!(first.contains(&Fact::Smtp(
        g(ALICE),
        Some("alice@example.test".into()),
        Some(String::new())
    )));
    assert!(first.contains(&Fact::Active(g(BOB), None, Some(false))));
    let second: BTreeSet<Fact> = events(&st, o2, o3)[0].1.iter().cloned().collect();
    assert_eq!(
        second,
        BTreeSet::from([Fact::Smtp(g(ALICE), Some(String::new()), None)])
    );
}

/// 8. An old observation stays readable after newer ones.
#[test]
fn an_old_observation_reads_as_it_was() {
    let mut st = VersionStore::new();
    let o1 = st.observe("ad", 0, observed()).unwrap();
    for i in 1..=10u8 {
        let u = if i % 2 == 0 { ALICE } else { BOB };
        let mut r = observed();
        if i % 4 < 2 {
            r = with_member(r, u, EXCHANGE);
        }
        st.reobserve("ad", i64::from(i), r).unwrap();
    }
    let v = st.view(o1).unwrap();
    for &(u, gr) in &observed().members {
        assert!(v.is_member(&u, &gr));
    }
    assert!(!v.is_member(&g(ALICE), &g(EXCHANGE)));
    assert!(!v.is_member(&g(BOB), &g(EXCHANGE)));
}

/// 10. A refused read changes nothing.
#[test]
fn a_refused_read_changes_nothing() {
    let mut st = VersionStore::new();
    let o1 = st.observe("ad", 1, observed()).unwrap();
    let before = format!("{:?}", st.image());
    assert!(st.history(o1, VersionId(7)).is_err());
    assert_eq!(format!("{:?}", st.image()), before);
}
