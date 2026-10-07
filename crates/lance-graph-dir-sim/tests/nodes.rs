//! Node creation and deletion, and reconciliation against a re-observation.

use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};
use ogar_dir_sim::*;

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
const ALICE: u8 = 0xA1;
const BOB: u8 = 0xB0;
const CAROL: u8 = 0xC0;
const EMPLOYEES: u8 = 0xE0;
const EXCHANGE: u8 = 0xEC;
const NEW_USER: u8 = 0x51;
const NEW_GROUP: u8 = 0x61;
const GHOST: u8 = 0x99;

const R: RuleId = RuleId {
    name: "Propose",
    version: 1,
};

/// A rule that proposes a fixed change list (the order it is given in).
struct Propose(Vec<Change>);
impl Rule for Propose {
    fn id(&self) -> RuleId {
        R
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}

/// Every raw value these tests use, interned at ingress in this order before
/// anything is observed. The store is append-only, so `val(s)` — the index
/// here — is the id the store issues for `s` for its whole lifetime.
const VOCAB: &[&str] = &[
    "alice@example.test",
    "bob@example.test",
    "carol@example.test",
    "new@example.test",
    "other@example.test",
    "ghost@example.test",
    "heir@example.test",
    "ALICE@example.test",
    "a.smith@example.test",
    "carol.renamed@example.test",
    "old@example.test",
    "typo@example.test",
];
fn val(s: &str) -> ValueId {
    let i = VOCAB.iter().position(|v| *v == s).expect("in VOCAB");
    ValueId(i as u32)
}
fn new_store() -> VersionStore {
    let mut st = VersionStore::new();
    for (i, v) in VOCAB.iter().enumerate() {
        assert_eq!(st.intern(v), ValueId(i as u32));
    }
    st
}
const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

fn user_state(name: &str) -> NodeState {
    let v = val(&format!("{name}@example.test"));
    NodeState {
        kind: NodeKind::User,
        active: Some(true),
        upn: Some(v),
        primary_smtp: Some(v),
        dn: None,
    }
}
fn group_state() -> NodeState {
    NodeState {
        kind: NodeKind::Group,
        active: Some(true),
        upn: None,
        primary_smtp: None,
        dn: None,
    }
}
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
            // Carol is in no group.
            (
                g(CAROL),
                ObservedNode::user("carol@example.test", "carol@example.test"),
            ),
            (g(EMPLOYEES), ObservedNode::group()),
            (g(EXCHANGE), ObservedNode::group()),
        ],
        members: vec![(g(ALICE), g(EMPLOYEES)), (g(BOB), g(EMPLOYEES))],
    }
}
fn store() -> (VersionStore, VersionId) {
    let mut st = new_store();
    let g0 = st.observe("lab", 1_000, observed()).unwrap();
    (st, g0)
}
fn sim(st: &mut VersionStore, from: VersionId, cs: Vec<Change>) -> Result<VersionId, SimError> {
    st.simulate(from, &Propose(cs), &[EvidenceRef("REQ".into())])
}
fn create(n: u8, state: NodeState) -> Change {
    Change::CreateNode { node: g(n), state }
}
fn delete(n: u8, state: NodeState) -> Change {
    Change::DeleteNode { node: g(n), state }
}
fn add(u: u8, grp: u8) -> Change {
    Change::AddMembership {
        user: g(u),
        group: g(grp),
    }
}
fn remove(u: u8, grp: u8) -> Change {
    Change::RemoveMembership {
        user: g(u),
        group: g(grp),
    }
}
fn apply_err(r: Result<VersionId, SimError>) -> ApplyError {
    match r {
        Err(SimError::Apply(e)) => e,
        other => panic!("expected an apply error, got {other:?}"),
    }
}

#[test]
fn create_user_grows_the_version_by_one_delta() {
    let (mut st, g0) = store();
    let v = sim(&mut st, g0, vec![create(NEW_USER, user_state("new"))]).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(view.delta_len(), 1);
    assert!(std::ptr::eq(
        view.snapshot(),
        st.view(g0).unwrap().snapshot()
    ));
    assert_eq!((view.users_len(), view.groups_len()), (4, 2));
    let o = view.user_ordinal(&g(NEW_USER)).unwrap();
    assert_eq!(
        o,
        UserOrdinal(3),
        "created users follow the base user ordinals"
    );
    assert_eq!(view.user_guid(o), Some(g(NEW_USER)));
    assert_eq!(view.node_state(&g(NEW_USER)), Some(user_state("new")));
    assert!(view.active_users()[0] >> o.0 & 1 == 1);
    assert!(st.validate(v).unwrap().is_empty());
    assert_eq!(
        st.diff(g0, v).unwrap(),
        vec![create(NEW_USER, user_state("new"))]
    );
}

#[test]
fn create_group_and_its_membership_plan_in_a_safe_order() {
    let (mut st, g0) = store();
    // Given in the worst order: the membership before either node exists.
    let v = sim(
        &mut st,
        g0,
        vec![
            add(NEW_USER, NEW_GROUP),
            create(NEW_GROUP, group_state()),
            create(NEW_USER, user_state("new")),
        ],
    )
    .unwrap();
    assert!(st.view(v).unwrap().is_member(&g(NEW_USER), &g(NEW_GROUP)));
    st.promote_desired(v).unwrap();
    let ops: Vec<Operation> = st.plan(v).unwrap().ops.into_iter().map(|p| p.op).collect();
    assert_eq!(
        ops,
        vec![
            Operation::CreateObject {
                object: g(NEW_USER),
                state: user_state("new")
            },
            Operation::CreateObject {
                object: g(NEW_GROUP),
                state: group_state()
            },
            Operation::AddGroupMember {
                group: g(NEW_GROUP),
                member: g(NEW_USER)
            },
        ]
    );
}

#[test]
fn delete_user_and_group() {
    let (mut st, g0) = store();
    let carol = st.view(g0).unwrap().node_state(&g(CAROL)).unwrap();
    let v = sim(
        &mut st,
        g0,
        vec![
            delete(CAROL, carol.clone()),
            delete(EXCHANGE, group_state()),
        ],
    )
    .unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(view.delta_len(), 2);
    assert!(!view.exists(&g(CAROL)) && !view.exists(&g(EXCHANGE)));
    // The slot remains, cleared from every live plane.
    assert_eq!(view.users_len(), 3, "a deleted user keeps its slot");
    let carol_ord = st.view(g0).unwrap().user_ordinal(&g(CAROL)).unwrap();
    assert_eq!(view.user_guid(carol_ord), None);
    assert_eq!(view.user_ordinal(&g(CAROL)), None);
    assert!(view.active_users()[0] >> carol_ord.0 & 1 == 0);
    assert!(st.validate(v).unwrap().is_empty());
    st.promote_desired(v).unwrap();
    let plan = st.plan(v).unwrap();
    assert_eq!(
        plan.ops,
        vec![
            PlannedOp {
                op: Operation::DeleteObject { object: g(CAROL) },
                precondition: Precondition::ObjectRemovable(carol),
            },
            PlannedOp {
                op: Operation::DeleteObject {
                    object: g(EXCHANGE)
                },
                precondition: Precondition::ObjectRemovable(group_state()),
            },
        ]
    );
}

#[test]
fn create_then_delete_converges() {
    let (mut st, g0) = store();
    let v1 = sim(&mut st, g0, vec![create(NEW_USER, user_state("new"))]).unwrap();
    let v2 = sim(&mut st, v1, vec![delete(NEW_USER, user_state("new"))]).unwrap();
    assert_eq!(st.view(v2).unwrap().delta_len(), 0);
    assert!(st.diff(g0, v2).unwrap().is_empty());
    // Delete-then-recreate of an observed node unchanged is an undo too.
    let carol = st.view(g0).unwrap().node_state(&g(CAROL)).unwrap();
    let v3 = sim(&mut st, g0, vec![delete(CAROL, carol.clone())]).unwrap();
    let v4 = sim(&mut st, v3, vec![create(CAROL, carol)]).unwrap();
    assert_eq!(st.view(v4).unwrap().delta_len(), 0);
    assert!(st.diff(g0, v4).unwrap().is_empty());
}

#[test]
fn recreating_a_deleted_identity_with_new_content_is_refused() {
    let (mut st, g0) = store();
    let carol = st.view(g0).unwrap().node_state(&g(CAROL)).unwrap();
    let v = sim(&mut st, g0, vec![delete(CAROL, carol)]).unwrap();
    assert_eq!(
        apply_err(sim(&mut st, v, vec![create(CAROL, user_state("other"))])),
        ApplyError::IdentityReused(g(CAROL))
    );
}

#[test]
fn create_is_refused_for_an_existing_identity() {
    let (mut st, g0) = store();
    assert_eq!(
        apply_err(sim(&mut st, g0, vec![create(ALICE, user_state("alice"))])),
        ApplyError::NodeExists(g(ALICE))
    );
    assert_eq!(
        apply_err(sim(
            &mut st,
            g0,
            vec![
                create(NEW_USER, user_state("new")),
                create(NEW_USER, user_state("other"))
            ]
        )),
        ApplyError::NodeExists(g(NEW_USER))
    );
    let v = sim(&mut st, g0, vec![create(NEW_USER, user_state("new"))]).unwrap();
    assert_eq!(
        apply_err(sim(&mut st, v, vec![create(NEW_USER, user_state("new"))])),
        ApplyError::NodeExists(g(NEW_USER))
    );
    assert_eq!(st.lineage(v).unwrap().len(), 2, "no version was created");
}

#[test]
fn delete_is_compare_and_set() {
    let (mut st, g0) = store();
    let stale = NodeState {
        primary_smtp: Some(val("old@example.test")),
        ..st.view(g0).unwrap().node_state(&g(CAROL)).unwrap()
    };
    assert!(matches!(
        apply_err(sim(&mut st, g0, vec![delete(CAROL, stale)])),
        ApplyError::StaleNode { node, .. } if node == g(CAROL)
    ));
    assert_eq!(
        apply_err(sim(&mut st, g0, vec![delete(GHOST, user_state("ghost"))])),
        ApplyError::UnknownNode(g(GHOST))
    );
}

#[test]
fn deleting_a_node_with_memberships_cannot_strand_edges() {
    let (mut st, g0) = store();
    let alice = st.view(g0).unwrap().node_state(&g(ALICE)).unwrap();
    // User side (observed edge).
    assert_eq!(
        apply_err(sim(&mut st, g0, vec![delete(ALICE, alice.clone())])),
        ApplyError::NodeHasMemberships(g(ALICE))
    );
    // Group side (the relation has no group index: the guarded scan).
    assert_eq!(
        apply_err(sim(&mut st, g0, vec![delete(EMPLOYEES, group_state())])),
        ApplyError::NodeHasMemberships(g(EMPLOYEES))
    );
    // An edge added by simulation counts too.
    let v = sim(&mut st, g0, vec![add(CAROL, EXCHANGE)]).unwrap();
    assert_eq!(
        apply_err(sim(&mut st, v, vec![delete(EXCHANGE, group_state())])),
        ApplyError::NodeHasMemberships(g(EXCHANGE))
    );
    // Removing the edges first makes the delete legal, and the plan removes
    // before it deletes.
    let d = sim(
        &mut st,
        g0,
        vec![delete(ALICE, alice), remove(ALICE, EMPLOYEES)],
    )
    .unwrap();
    assert!(st.validate(d).unwrap().is_empty());
    st.promote_desired(d).unwrap();
    let ops: Vec<Operation> = st.plan(d).unwrap().ops.into_iter().map(|p| p.op).collect();
    assert_eq!(
        ops,
        vec![
            Operation::RemoveGroupMember {
                group: g(EMPLOYEES),
                member: g(ALICE)
            },
            Operation::DeleteObject { object: g(ALICE) },
        ]
    );
}

#[test]
fn dangling_memberships_are_detected_before_promotion() {
    let (mut st, g0) = store();
    // A new user put into a group nobody creates.
    let v = sim(
        &mut st,
        g0,
        vec![create(NEW_USER, user_state("new")), add(NEW_USER, GHOST)],
    )
    .unwrap();
    let err = st.promote_desired(v).unwrap_err();
    assert_eq!(
        err.violations,
        vec![Violation::DanglingMembership {
            user: g(NEW_USER),
            group: g(GHOST),
            missing: Endpoint::Group
        }]
    );
    assert_eq!(st.tag(TAG_DESIRED), None);
}

#[test]
fn created_and_deleted_nodes_take_part_in_uniqueness() {
    let (mut st, g0) = store();
    // A new user taking Alice's address collides…
    let mut clash = user_state("new");
    clash.primary_smtp = Some(val("ALICE@example.test"));
    let v = sim(&mut st, g0, vec![create(NEW_USER, clash)]).unwrap();
    assert_eq!(
        st.validate(v).unwrap(),
        vec![Violation::DuplicateSmtp {
            key: st.key_of(val("alice@example.test")).unwrap(),
            // Sorted by identity: 0x51.. < 0xA1..
            owners: vec![g(NEW_USER), g(ALICE)],
        }]
    );
    // …but the address of a deleted user is free.
    let carol = st.view(g0).unwrap().node_state(&g(CAROL)).unwrap();
    let mut reuse = user_state("new");
    reuse.primary_smtp = carol.primary_smtp;
    reuse.upn = carol.upn;
    let w = sim(
        &mut st,
        g0,
        vec![delete(CAROL, carol), create(NEW_USER, reuse)],
    )
    .unwrap();
    assert!(st.validate(w).unwrap().is_empty());
}

#[test]
fn subtree_sees_created_and_not_deleted_nodes() {
    let dn = |l: &[u8]| Dn128::new(l).unwrap();
    let mut obs = observed();
    obs.nodes[2].1.dn = Some(dn(&[0, 1]));
    let mut st = new_store();
    let g0 = st.observe("lab", 0, obs).unwrap();
    let carol = st.view(g0).unwrap().node_state(&g(CAROL)).unwrap();
    let mut located = user_state("new");
    located.dn = Some(dn(&[0, 2]));
    let v = sim(
        &mut st,
        g0,
        vec![delete(CAROL, carol), create(NEW_USER, located)],
    )
    .unwrap();
    let view = st.view(v).unwrap();
    let rows = subtree(&view, SCOPE, NodeKind::User, &dn(&[0]))
        .unwrap()
        .rows();
    let new = view.user_ordinal(&g(NEW_USER)).unwrap();
    assert_eq!(rows, vec![usize::from(new.0)]);
}

#[test]
fn input_order_does_not_change_the_output() {
    let changes = || {
        vec![
            create(NEW_GROUP, group_state()),
            create(NEW_USER, user_state("new")),
            add(NEW_USER, NEW_GROUP),
            add(CAROL, NEW_GROUP),
            remove(BOB, EMPLOYEES),
        ]
    };
    let run = |cs: Vec<Change>, rev_obs: bool| {
        let mut obs = observed();
        if rev_obs {
            obs.nodes.reverse();
            obs.members.reverse();
        }
        let mut st = new_store();
        let g0 = st.observe("lab", 0, obs).unwrap();
        let v = sim(&mut st, g0, cs).unwrap();
        st.promote_desired(v).unwrap();
        (st.diff(g0, v).unwrap(), st.plan(v).unwrap().ops)
    };
    let forward = run(changes(), false);
    let mut rev = changes();
    rev.reverse();
    // Reversed, the membership adds precede the creates: still applies,
    // because added memberships are held by identity.
    assert_eq!(run(rev, true), forward);
}

#[test]
fn identity_not_ordinal_reaches_change_diff_plan_and_provenance() {
    let (mut st, g0) = store();
    let v = sim(
        &mut st,
        g0,
        vec![create(NEW_USER, user_state("new")), add(NEW_USER, EXCHANGE)],
    )
    .unwrap();
    let expected_delta = vec![create(NEW_USER, user_state("new")), add(NEW_USER, EXCHANGE)];
    assert_eq!(st.version(v).unwrap().delta, expected_delta);
    let mut diff = st.diff(g0, v).unwrap();
    diff.sort();
    assert_eq!(diff, expected_delta);
    let chain = st
        .explain_membership(v, &g(NEW_USER), &g(EXCHANGE))
        .unwrap();
    assert_eq!(chain.last().unwrap().id, v);
    st.promote_desired(v).unwrap();
    assert!(st.plan(v).unwrap().ops.iter().all(|p| match &p.op {
        Operation::CreateObject { object, .. } => *object == g(NEW_USER),
        Operation::AddGroupMember { group, member } => {
            *group == g(EXCHANGE) && *member == g(NEW_USER)
        }
        _ => false,
    }));
}

// ---- reconciliation: the plan holds only the work still outstanding ----

/// Desired: create a user + membership, delete Carol, remove Bob from
/// Employees, rename Alice's address.
fn desired(st: &mut VersionStore, g0: VersionId) -> VersionId {
    let carol = st.view(g0).unwrap().node_state(&g(CAROL)).unwrap();
    let v = sim(
        st,
        g0,
        vec![
            create(NEW_USER, user_state("new")),
            add(NEW_USER, EXCHANGE),
            delete(CAROL, carol),
            remove(BOB, EMPLOYEES),
            Change::SetAttribute {
                node: g(ALICE),
                attribute: Attribute::PrimarySmtp,
                from: Some(val("alice@example.test")),
                to: Some(val("a.smith@example.test")),
            },
        ],
    )
    .unwrap();
    st.promote_desired(v).unwrap();
    v
}

#[test]
fn reconcile_a_fully_converged_observation_plans_nothing() {
    let (mut st, g0) = store();
    let d = desired(&mut st, g0);
    let mut obs = observed();
    obs.nodes.retain(|(id, _)| *id != g(CAROL));
    obs.nodes[0].1.primary_smtp = Some("a.smith@example.test".into());
    obs.nodes.push((
        g(NEW_USER),
        ObservedNode::user("new@example.test", "new@example.test"),
    ));
    obs.members = vec![(g(ALICE), g(EMPLOYEES)), (g(NEW_USER), g(EXCHANGE))];
    let o = st.observe("lab", 2_000, obs).unwrap();
    let plan = st.plan(d).unwrap();
    assert_eq!(plan.basis, o);
    assert!(plan.ops.is_empty(), "{:?}", plan.ops);
    assert!(st.diff(o, d).unwrap().is_empty());
}

#[test]
fn reconcile_a_partial_observation_plans_only_the_rest() {
    let (mut st, g0) = store();
    let d = desired(&mut st, g0);
    let full = st.plan(d).unwrap().ops;
    assert_eq!(full.len(), 5);

    // Reality now: the user exists (but with a different address), Carol is
    // already gone, Alice already has the new address, Bob is still in
    // Employees, the new user is not yet in Exchange, and someone else
    // created an unrelated group.
    let mut obs = observed();
    obs.nodes.retain(|(id, _)| *id != g(CAROL));
    obs.nodes[0].1.primary_smtp = Some("a.smith@example.test".into());
    obs.nodes.push((
        g(NEW_USER),
        ObservedNode::user("new@example.test", "typo@example.test"),
    ));
    obs.nodes.push((g(GHOST), ObservedNode::group()));
    let o = st.observe("lab", 2_000, obs).unwrap();
    let plan = st.plan(d).unwrap();
    assert_eq!(plan.basis, o);
    assert_eq!(
        plan.ops,
        vec![
            PlannedOp {
                op: Operation::RemoveGroupMember {
                    group: g(EMPLOYEES),
                    member: g(BOB)
                },
                precondition: Precondition::IsMember,
            },
            // The requested node exists: no repeated create, only the
            // attribute still differing, with the precondition read from
            // the new observation (not the stale "absent").
            PlannedOp {
                op: Operation::SetAttribute {
                    object: g(NEW_USER),
                    attribute: Attribute::PrimarySmtp,
                    value: Some(val("new@example.test")),
                },
                precondition: Precondition::AttributeEquals(Some(val("typo@example.test"))),
            },
            PlannedOp {
                op: Operation::AddGroupMember {
                    group: g(EXCHANGE),
                    member: g(NEW_USER)
                },
                precondition: Precondition::NotMember,
            },
        ]
    );
}

#[test]
fn reconcile_a_delete_reads_its_precondition_from_the_latest_observation() {
    let (mut st, g0) = store();
    let d = desired(&mut st, g0);
    let mut obs = observed();
    obs.nodes[2].1.upn = Some("carol.renamed@example.test".into());
    st.observe("lab", 2_000, obs).unwrap();
    let plan = st.plan(d).unwrap();
    let del = plan
        .ops
        .iter()
        .find(|p| p.op == Operation::DeleteObject { object: g(CAROL) })
        .unwrap();
    assert_eq!(
        del.precondition,
        Precondition::ObjectRemovable(NodeState {
            upn: Some(val("carol.renamed@example.test")),
            ..user_state("carol")
        })
    );
}

#[test]
fn a_group_has_no_enabled_flag() {
    let (mut st, g0) = store();
    assert_eq!(
        st.view(g0)
            .unwrap()
            .node_state(&g(EXCHANGE))
            .unwrap()
            .active,
        Some(true)
    );
    let inactive = NodeState {
        active: Some(false),
        ..group_state()
    };
    assert_eq!(
        apply_err(sim(&mut st, g0, vec![create(NEW_GROUP, inactive)])),
        ApplyError::InactiveGroup(g(NEW_GROUP))
    );
}

#[test]
fn a_freed_address_is_released_before_it_is_claimed() {
    // Carol leaves; a new user takes her address. Both versions are valid,
    // so execution must delete before it creates.
    let (mut st, g0) = store();
    let carol = st.view(g0).unwrap().node_state(&g(CAROL)).unwrap();
    let mut heir = user_state("heir");
    heir.primary_smtp = carol.primary_smtp;
    let v = sim(
        &mut st,
        g0,
        vec![create(NEW_USER, heir), delete(CAROL, carol)],
    )
    .unwrap();
    st.promote_desired(v).unwrap();
    let ops: Vec<Operation> = st.plan(v).unwrap().ops.into_iter().map(|p| p.op).collect();
    assert_eq!(ops[0], Operation::DeleteObject { object: g(CAROL) });
    assert!(matches!(ops[1], Operation::CreateObject { object, .. } if object == g(NEW_USER)));
}

#[test]
fn reconcile_refuses_a_create_it_cannot_converge() {
    let (mut st, g0) = store();
    let d = desired(&mut st, g0);
    // The requested user exists in reality, but disabled: no change can
    // enable it, so the create is neither done nor doable.
    let mut disabled = ObservedNode::user("new@example.test", "new@example.test");
    disabled.active = Some(false);
    let mut obs = observed();
    obs.nodes.push((g(NEW_USER), disabled));
    let o = st.observe("lab", 2_000, obs).unwrap();
    assert_eq!(
        st.plan(d),
        Err(PlanError::Unconvergeable {
            basis: o,
            target: d,
            node: g(NEW_USER)
        })
    );
}
