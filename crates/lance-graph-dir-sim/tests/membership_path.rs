//! Why a user is in a group, and whether a role survives a change: a
//! membership chain read from one view, and a role decided on each version
//! on its own.

use lance_graph_contract::rbac_plug::RbacBinding;
use lance_graph_dir_sim::actors::{GroupRoles, RoleEvidence, RoleImpact};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::{Change, EvidenceRef, RuleId};

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

const ANNA: u8 = 0x11;
const BOB: u8 = 0x12;
/// Security groups.
const VERTRIEB: u8 = 0xA1;
const MARKETING: u8 = 0xA2;
const CRM: u8 = 0xA3;
const CMS: u8 = 0xA4;
const BRIDGE: u8 = 0xA5;
const HOP: u8 = 0xA6;
const LOOP: u8 = 0xA7;
/// Distribution group: not security-enabled.
const DL: u8 = 0xD1;
/// Security group with no members whose grants are never read.
const UNREAD: u8 = 0xE1;

fn group(security: bool) -> ObservedNode {
    let mut n = ObservedNode::group();
    n.security = Some(security);
    n
}

// anna ∈ {VERTRIEB, BRIDGE, DL}; bob ∈ {MARKETING}
// VERTRIEB ∈ CRM (short path); BRIDGE ∈ HOP ∈ CRM (long path)
// CRM ∈ LOOP ∈ CRM (a cycle); MARKETING ∈ CMS; DL ∈ CMS (distribution only);
// HOP ∈ DL (a security group inside a distribution group); UNREAD empty
fn observed() -> Observation {
    let user = |n: &str| ObservedNode::user(&format!("{n}@x.test"), &format!("{n}@x.test"));
    let mut nodes = vec![(g(ANNA), user("anna")), (g(BOB), user("bob"))];
    for n in [VERTRIEB, MARKETING, CRM, CMS, BRIDGE, HOP, LOOP, UNREAD] {
        nodes.push((g(n), group(true)));
    }
    nodes.push((g(DL), group(false)));
    let e = |m: u8, gr: u8| (g(m), g(gr));
    Observation {
        scope: SCOPE,
        nodes,
        members: vec![
            e(ANNA, VERTRIEB),
            e(ANNA, BRIDGE),
            e(ANNA, DL),
            e(BOB, MARKETING),
            e(VERTRIEB, CRM),
            e(BRIDGE, HOP),
            e(HOP, CRM),
            e(CRM, LOOP),
            e(LOOP, CRM),
            e(MARKETING, CMS),
            e(DL, CMS),
            e(HOP, DL),
        ],
    }
}

const ALL_GROUPS: [u8; 8] = [VERTRIEB, MARKETING, CRM, CMS, BRIDGE, HOP, LOOP, DL];

fn ids(ns: &[u8]) -> Vec<Guid128> {
    ns.iter().map(|&n| g(n)).collect()
}

struct Edit(Vec<Change>);
impl Rule for Edit {
    fn id(&self) -> RuleId {
        RuleId {
            name: "Edit",
            version: 1,
        }
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}

fn add(m: u8, gr: u8) -> Change {
    Change::AddMembership {
        user: g(m),
        group: g(gr),
    }
}
fn remove(m: u8, gr: u8) -> Change {
    Change::RemoveMembership {
        user: g(m),
        group: g(gr),
    }
}

const CRM_READER: &str = "crm_reader";
const CMS_EDITOR: &str = "cms_editor";

fn binding() -> RbacBinding {
    RbacBinding::new(
        "test",
        vec![],
        vec![(CRM_READER, vec![]), (CMS_EDITOR, vec![])],
        vec![],
    )
}

// CRM grants crm_reader, CMS cms_editor; every other security group is read
// with no grants except LOOP, which was never read.
fn grants() -> GroupRoles {
    let mut gr = GroupRoles::new();
    gr.read(g(CRM), &[CRM_READER])
        .read(g(CMS), &[CMS_EDITOR])
        .read(g(VERTRIEB), &[])
        .read(g(MARKETING), &[])
        .read(g(BRIDGE), &[])
        .read(g(HOP), &[]);
    gr
}

#[test]
fn a_direct_membership_is_a_one_group_chain() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let p = view.membership_path(&g(ANNA), &g(VERTRIEB)).unwrap();
    assert_eq!(p.user(), &g(ANNA));
    assert_eq!(p.groups(), ids(&[VERTRIEB]).as_slice());
    assert!(p.is_direct());
    // Not a member, or not there: no chain.
    assert!(view.membership_path(&g(ANNA), &g(MARKETING)).is_none());
    assert!(view.membership_path(&g(0x99), &g(VERTRIEB)).is_none());
    assert!(view.membership_path(&g(ANNA), &g(0x99)).is_none());
}

// Two chains reach CRM; the witness is the shorter one, and the cycle
// through LOOP terminates.
#[test]
fn the_witness_is_the_shortest_chain_and_a_cycle_ends() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let p = view.membership_path(&g(ANNA), &g(CRM)).unwrap();
    assert_eq!(p.groups(), ids(&[VERTRIEB, CRM]).as_slice());
    assert!(!p.is_direct());
    let l = view.membership_path(&g(ANNA), &g(LOOP)).unwrap();
    assert_eq!(l.groups(), ids(&[VERTRIEB, CRM, LOOP]).as_slice());
}

// The chain exists exactly when the closure says so, and every link of it
// is a live direct membership.
#[test]
fn every_chain_agrees_with_the_closure_and_each_link_holds() {
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 1, observed()).unwrap();
    let v1 = st
        .simulate(
            v0,
            &Edit(vec![remove(ANNA, VERTRIEB), add(BOB, BRIDGE)]),
            &[],
        )
        .unwrap();
    let mut chains = 0;
    for v in [v0, v1] {
        let view = st.view(v).unwrap();
        for u in [ANNA, BOB] {
            let closure = view.groups_transitive(&g(u));
            for gr in ALL_GROUPS {
                let p = view.membership_path(&g(u), &g(gr));
                assert_eq!(p.is_some(), closure.contains(&g(gr)), "{u:x} {gr:x}");
                if let Some(p) = p {
                    chains += 1;
                    assert_eq!(p.group(), &g(gr));
                    assert!(view.is_member(&g(u), &p.groups()[0]));
                    for w in p.groups().windows(2) {
                        assert!(view.is_member(&w[0], &w[1]), "{:?}", w);
                    }
                }
            }
        }
    }
    // Anti-vacuity: both versions produce chains, multi-hop ones included.
    assert!(chains >= 12, "{chains}");
}

// Through security groups only: CMS is reached through the distribution
// group, which carries no SID.
#[test]
fn a_sid_chain_skips_distribution_groups() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let sid = GroupWhere::Is(GroupProperty::SecurityEnabled);
    assert_eq!(
        view.membership_path(&g(ANNA), &g(CMS)).unwrap().groups(),
        ids(&[DL, CMS]).as_slice()
    );
    // Also not through HOP ∈ DL: the filter holds above the direct groups.
    assert!(view
        .membership_path_through(&g(ANNA), &g(CMS), &sid)
        .is_none());
    assert_eq!(
        view.membership_path_through(&g(ANNA), &g(CRM), &sid)
            .unwrap()
            .groups(),
        ids(&[VERTRIEB, CRM]).as_slice()
    );
}

// The same observation gives the same chain in two stores, and a later
// simulation leaves the earlier version's answer as it was.
#[test]
fn the_witness_is_deterministic_and_the_observed_version_is_untouched() {
    let mut a = VersionStore::new();
    let mut b = VersionStore::new();
    let va = a.observe("lab", 1, observed()).unwrap();
    let vb = b.observe("lab", 1, observed()).unwrap();
    let pa = a.view(va).unwrap().membership_path(&g(ANNA), &g(LOOP));
    assert_eq!(pa, b.view(vb).unwrap().membership_path(&g(ANNA), &g(LOOP)));
    a.simulate(va, &Edit(vec![remove(ANNA, VERTRIEB)]), &[])
        .unwrap();
    assert_eq!(a.view(va).unwrap().membership_path(&g(ANNA), &g(LOOP)), pa);
}

// Removing the witness's first edge does not take the role away: the long
// chain through BRIDGE and HOP still carries it.
#[test]
fn a_role_survives_when_another_chain_still_carries_it() {
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 1, observed()).unwrap();
    let v1 = st
        .simulate(v0, &Edit(vec![remove(ANNA, VERTRIEB)]), &[])
        .unwrap();
    let (before, after) = (st.view(v0).unwrap(), st.view(v1).unwrap());
    let impact = RoleImpact::between(&before, &after, &g(ANNA), CRM_READER, &grants(), &binding());
    match (&impact.before, &impact.after) {
        (
            RoleEvidence::Held {
                grant: gb,
                path: pb,
            },
            RoleEvidence::Held {
                grant: ga,
                path: pa,
            },
        ) => {
            assert_eq!((gb, ga), (&g(CRM), &g(CRM)));
            assert_eq!(pb.groups(), ids(&[VERTRIEB, CRM]).as_slice());
            assert_eq!(pa.groups(), ids(&[BRIDGE, HOP, CRM]).as_slice());
        }
        other => panic!("{other:?}"),
    }
    assert!(!impact.is_lost());
    assert!(!impact.is_gained());
}

// With both chains cut the role is lost, and the decision says so only
// because every remaining group was read.
#[test]
fn a_role_is_lost_only_when_no_chain_is_left() {
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 1, observed()).unwrap();
    let v1 = st
        .simulate(
            v0,
            &Edit(vec![remove(ANNA, VERTRIEB), remove(ANNA, BRIDGE)]),
            &[],
        )
        .unwrap();
    let (before, after) = (st.view(v0).unwrap(), st.view(v1).unwrap());
    let impact = RoleImpact::between(&before, &after, &g(ANNA), CRM_READER, &grants(), &binding());
    assert!(impact.before.is_held());
    assert_eq!(impact.after, RoleEvidence::NotHeld);
    assert!(impact.is_lost());
}

// Joining MARKETING adds cms_editor, but before the change anna carried
// LOOP's SID and LOOP was never read: that is undetermined, not "not held",
// so no gain is reported until LOOP is read.
#[test]
fn an_unread_group_makes_the_answer_undetermined_not_absent() {
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 1, observed()).unwrap();
    let v1 = st
        .simulate(v0, &Edit(vec![add(ANNA, MARKETING)]), &[])
        .unwrap();
    let (before, after) = (st.view(v0).unwrap(), st.view(v1).unwrap());
    let impact = RoleImpact::between(&before, &after, &g(ANNA), CMS_EDITOR, &grants(), &binding());
    assert_eq!(
        impact.before,
        RoleEvidence::Undetermined {
            unread: ids(&[LOOP])
        }
    );
    assert!(impact.after.is_held());
    assert!(!impact.is_gained());

    let mut read = grants();
    read.read(g(LOOP), &[]);
    let impact = RoleImpact::between(&before, &after, &g(ANNA), CMS_EDITOR, &read, &binding());
    assert_eq!(impact.before, RoleEvidence::NotHeld);
    match &impact.after {
        RoleEvidence::Held { grant, path } => {
            assert_eq!(grant, &g(CMS));
            assert_eq!(path.groups(), ids(&[MARKETING, CMS]).as_slice());
        }
        other => panic!("{other:?}"),
    }
    assert!(impact.is_gained());
}

// A role the binding does not declare is never conferred, whatever the
// grants say.
#[test]
fn an_undeclared_role_is_never_held() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let mut gr = grants();
    gr.read(g(CRM), &[CRM_READER, "admin"]);
    assert_eq!(
        gr.role_evidence(&view, &g(ANNA), "admin", &binding()),
        RoleEvidence::NotDeclared
    );
    assert!(gr
        .role_evidence(&view, &g(ANNA), CRM_READER, &binding())
        .is_held());
}

// Two groups grant the role; the witness is the one with the shorter chain
// (BRIDGE, direct), not the first one in ordinal order (CRM, two hops).
#[test]
fn the_witness_grant_is_the_one_with_the_shortest_chain() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let mut gr = grants();
    gr.read(g(BRIDGE), &[CRM_READER]);
    match gr.role_evidence(&view, &g(ANNA), CRM_READER, &binding()) {
        RoleEvidence::Held { grant, path } => {
            assert_eq!(grant, g(BRIDGE));
            assert!(path.is_direct());
        }
        other => panic!("{other:?}"),
    }
}

// The role was held and after the change anna carries only a group whose
// grants were never read: that is not a loss, it is unknown.
#[test]
fn an_undetermined_after_is_not_a_loss() {
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 1, observed()).unwrap();
    let v1 = st
        .simulate(
            v0,
            &Edit(vec![
                remove(ANNA, VERTRIEB),
                remove(ANNA, BRIDGE),
                add(ANNA, UNREAD),
            ]),
            &[],
        )
        .unwrap();
    let (before, after) = (st.view(v0).unwrap(), st.view(v1).unwrap());
    let impact = RoleImpact::between(&before, &after, &g(ANNA), CRM_READER, &grants(), &binding());
    assert!(impact.before.is_held());
    assert_eq!(
        impact.after,
        RoleEvidence::Undetermined {
            unread: ids(&[UNREAD])
        }
    );
    assert!(!impact.is_lost());
}
