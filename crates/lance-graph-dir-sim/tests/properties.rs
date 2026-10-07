//! Properties of a node that exists before and after a change: the enabled
//! flag (three-valued) and the location (`Dn128`), through every seam —
//! apply, `node_state`, the active plane and validation, subtree, diff,
//! plan and reconcile. `kind` is identity and has no change.

use lance_graph_dir_sim::validate::validate;
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};
use ogar_dir_sim::*;
use std::sync::Arc;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const R: RuleId = RuleId {
    name: "Propose",
    version: 1,
};
const ALICE: u8 = 0xA1;
const BOB: u8 = 0xB0;
const UNSURE: u8 = 0x0F;
const STAFF: u8 = 0xE0;

struct Propose(Vec<Change>);
impl Rule for Propose {
    fn id(&self) -> RuleId {
        R
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
fn dn(l: &[u8]) -> Dn128 {
    Dn128::new(l).unwrap()
}
/// Alice enabled at OU [0], Bob enabled at OU [1], an unknown-status user
/// with no location, a group at OU [0].
fn observed() -> Observation {
    let mut alice = ObservedNode::user("alice@x.test", "alice@x.test");
    alice.dn = Some(dn(&[0]));
    let mut bob = ObservedNode::user("bob@x.test", "bob@x.test");
    bob.dn = Some(dn(&[1]));
    let mut unsure = ObservedNode::user("unsure@x.test", "unsure@x.test");
    unsure.active = None;
    let mut staff = ObservedNode::group();
    staff.dn = Some(dn(&[0]));
    Observation {
        scope: SCOPE,
        nodes: vec![
            (g(ALICE), alice),
            (g(BOB), bob),
            (g(UNSURE), unsure),
            (g(STAFF), staff),
        ],
        members: vec![],
    }
}
fn store() -> (VersionStore, VersionId) {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, observed()).unwrap();
    (st, v)
}
fn sim(st: &mut VersionStore, v: VersionId, cs: Vec<Change>) -> Result<VersionId, SimError> {
    st.simulate(v, &Propose(cs), &[])
}
fn active(n: u8, from: Option<bool>, to: Option<bool>) -> Change {
    Change::SetActive {
        node: g(n),
        from,
        to,
    }
}
fn moved(n: u8, from: Option<Dn128>, to: Option<Dn128>) -> Change {
    Change::SetLocation {
        node: g(n),
        from,
        to,
    }
}
fn active_set(v: &View<'_>) -> Vec<Guid128> {
    let plane = v.active_users();
    (0..v.users_len())
        .filter(|&i| plane[i / 64] >> (i % 64) & 1 == 1)
        .filter_map(|i| v.user_guid(UserOrdinal(i as u16)))
        .collect()
}

// (1) + (2): a flag override is seen by the active plane, not only by
// node_state; all three values, from every starting value. The unknown flag
// is neither active nor stored as `false`.
#[test]
fn a_flag_override_reaches_node_state_and_the_active_plane_in_all_three_states() {
    let (mut st, g0) = store();
    let states = [Some(true), Some(false), None];
    for (n, from) in [(BOB, Some(true)), (UNSURE, None)] {
        for to in states {
            if to == from {
                continue;
            }
            let v = sim(&mut st, g0, vec![active(n, from, to)]).unwrap();
            let view = st.view(v).unwrap();
            assert_eq!(
                view.node_state(&g(n)).unwrap().active,
                to,
                "{n}: {from:?}->{to:?}"
            );
            assert_eq!(
                active_set(&view).contains(&g(n)),
                to == Some(true),
                "{n}: active plane after {from:?}->{to:?}"
            );
            // Exactly one other change: Alice stays active throughout.
            assert!(active_set(&view).contains(&g(ALICE)));
        }
    }
}

// Validation reads the simulated flag: three users share alice@. With the
// third one's flag unknown, only Alice and Bob collide; disabling Bob
// clears it; enabling the unknown user collides again — with exactly the
// owners whose simulated flag is enabled.
#[test]
fn validation_sees_the_simulated_flag() {
    let mut obs = observed();
    obs.nodes[1].1.primary_smtp = Some("alice@x.test".into());
    obs.nodes[2].1.primary_smtp = Some("alice@x.test".into());
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, obs).unwrap();
    let owners = |st: &mut VersionStore, v| -> Vec<Vec<Guid128>> {
        st.validate(v)
            .unwrap()
            .into_iter()
            .filter_map(|x| match x {
                Violation::DuplicateSmtp { owners, .. } => Some(owners),
                _ => None,
            })
            .collect()
    };
    assert_eq!(owners(&mut st, g0), vec![vec![g(ALICE), g(BOB)]]);
    let off = sim(&mut st, g0, vec![active(BOB, Some(true), Some(false))]).unwrap();
    assert!(owners(&mut st, off).is_empty(), "Bob disabled");
    let on = sim(&mut st, off, vec![active(UNSURE, None, Some(true))]).unwrap();
    assert_eq!(owners(&mut st, on), vec![vec![g(UNSURE), g(ALICE)]]);
}

// (3): the compare-and-set `from` is checked against the version, not
// ignored — and a group has no flag.
#[test]
fn a_stale_from_or_a_group_flag_is_refused() {
    let (mut st, g0) = store();
    let refusal = |r: Result<VersionId, SimError>| match r {
        Err(SimError::Apply(ApplyError::Refused { refusal, .. })) => refusal,
        other => panic!("expected a refusal, got {other:?}"),
    };
    // UNSURE is unknown, not disabled: `from: Some(false)` is stale.
    assert_eq!(
        refusal(sim(
            &mut st,
            g0,
            vec![active(UNSURE, Some(false), Some(true))]
        )),
        Refusal::Stale
    );
    assert_eq!(
        refusal(sim(
            &mut st,
            g0,
            vec![moved(BOB, Some(dn(&[0])), Some(dn(&[2])))]
        )),
        Refusal::Stale
    );
    assert_eq!(
        refusal(sim(
            &mut st,
            g0,
            vec![active(STAFF, Some(true), Some(false))]
        )),
        Refusal::NoEnabledFlag
    );
}

// (4): a move is visible to node_state, to subtree queries (at the new
// location, not the old), to diff and to the plan — with the observed
// location as its precondition.
#[test]
fn a_move_reaches_subtree_diff_and_plan() {
    let (mut st, g0) = store();
    let v = sim(
        &mut st,
        g0,
        vec![moved(BOB, Some(dn(&[1])), Some(dn(&[0, 3])))],
    )
    .unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(view.node_state(&g(BOB)).unwrap().dn, Some(dn(&[0, 3])));
    let users_in = |p: &Dn128| -> Vec<Guid128> {
        subtree(&view, SCOPE, NodeKind::User, p)
            .unwrap()
            .rows()
            .into_iter()
            .map(|o| view.user_guid(UserOrdinal(o as u16)).unwrap())
            .collect()
    };
    let mut in_0 = users_in(&dn(&[0]));
    in_0.sort();
    assert_eq!(in_0, vec![g(ALICE), g(BOB)], "BOB found at the new OU");
    assert!(
        users_in(&dn(&[1])).is_empty(),
        "and no longer at the old one"
    );
    assert_eq!(
        st.diff(g0, v).unwrap(),
        vec![moved(BOB, Some(dn(&[1])), Some(dn(&[0, 3])))]
    );
    st.promote_desired(v).unwrap();
    assert_eq!(
        st.plan(v).unwrap().ops,
        vec![PlannedOp {
            op: Operation::MoveObject {
                object: g(BOB),
                to: dn(&[0, 3])
            },
            precondition: Precondition::LocationEquals(Some(dn(&[1]))),
        }]
    );
}

// Moving a node away from its location and back nets out (no override, no
// diff); moving it to "unknown" is a reportable diff but not a plan.
#[test]
fn net_effect_and_unknown_targets() {
    let (mut st, g0) = store();
    let there = sim(
        &mut st,
        g0,
        vec![moved(BOB, Some(dn(&[1])), Some(dn(&[2])))],
    )
    .unwrap();
    let back = sim(
        &mut st,
        there,
        vec![moved(BOB, Some(dn(&[2])), Some(dn(&[1])))],
    )
    .unwrap();
    assert_eq!(st.view(back).unwrap().delta_len(), 0);
    assert!(st.diff(g0, back).unwrap().is_empty());

    let unknown = sim(&mut st, g0, vec![active(BOB, Some(true), None)]).unwrap();
    assert_eq!(
        st.diff(g0, unknown).unwrap(),
        vec![active(BOB, Some(true), None)]
    );
    st.promote_desired(unknown).unwrap();
    assert!(matches!(
        st.plan(unknown),
        Err(PlanError::NotActuatable { node, .. }) if node == g(BOB)
    ));
}

// (5) + (8): a desired change survives re-observation. Reality unchanged:
// the plan still disables Bob, its precondition read from the NEW
// observation. Reality already disabled: nothing is left to do.
#[test]
fn a_desired_flag_change_survives_reconcile() {
    let (mut st, g0) = store();
    let d = sim(&mut st, g0, vec![active(BOB, Some(true), Some(false))]).unwrap();
    st.promote_desired(d).unwrap();
    let want = vec![PlannedOp {
        op: Operation::SetEnabled {
            object: g(BOB),
            enabled: false,
        },
        precondition: Precondition::EnabledEquals(Some(true)),
    }];
    assert_eq!(st.plan(d).unwrap().ops, want);

    st.observe("lab", 1, observed()).unwrap();
    assert_eq!(st.plan(d).unwrap().ops, want, "unchanged reality");

    // Someone made Bob's flag unknown in reality: the precondition follows
    // the latest observation.
    let mut obs = observed();
    obs.nodes[1].1.active = None;
    st.observe("lab", 2, obs).unwrap();
    assert_eq!(
        st.plan(d).unwrap().ops[0].precondition,
        Precondition::EnabledEquals(None)
    );

    let mut obs = observed();
    obs.nodes[1].1.active = Some(false);
    st.observe("lab", 3, obs).unwrap();
    assert!(st.plan(d).unwrap().ops.is_empty(), "converged");
}

// Reconcile across two observations reports flag and location drift as
// property changes, and an identity whose kind differs as delete + create.
#[test]
fn a_reconcile_diff_reports_drift_and_kind_as_identity() {
    let (mut st, g0) = store();
    let mut obs = observed();
    obs.nodes[0].1.active = Some(false);
    obs.nodes[0].1.dn = None;
    obs.nodes[3] = (g(BOB + 1), ObservedNode::group());
    obs.nodes
        .push((g(STAFF), ObservedNode::user("staff@x.test", "staff@x.test")));
    let o = st.observe("lab", 1, obs).unwrap();
    let d = st.diff(g0, o).unwrap();
    assert!(d.contains(&active(ALICE, Some(true), Some(false))));
    assert!(d.contains(&moved(ALICE, Some(dn(&[0])), None)));
    let staff: Vec<_> = d
        .iter()
        .filter(|c| match c {
            Change::DeleteNode { node, .. } | Change::CreateNode { node, .. } => *node == g(STAFF),
            _ => false,
        })
        .collect();
    assert!(matches!(staff[0], Change::DeleteNode { state, .. } if state.kind == NodeKind::Group));
    assert!(matches!(staff[1], Change::CreateNode { state, .. } if state.kind == NodeKind::User));
}

// (6): one property change is one overlay entry over the shared snapshot.
// (Allocation independence of population size: tests/alloc.rs.)
#[test]
fn a_property_change_shares_the_snapshot() {
    let (mut st, g0) = store();
    let v = sim(
        &mut st,
        g0,
        vec![
            active(BOB, Some(true), Some(false)),
            moved(UNSURE, None, Some(dn(&[4]))),
        ],
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        st.snapshot(g0).unwrap(),
        st.snapshot(v).unwrap()
    ));
    assert_eq!(st.view(v).unwrap().delta_len(), 2);
    assert!(validate(&st.view(v).unwrap()).is_empty());
}

// A created node's flag and location are set in its own row; a deleted
// node's overrides do not outlive it.
#[test]
fn created_and_deleted_nodes_carry_their_own_properties() {
    let (mut st, g0) = store();
    let new = NodeState {
        kind: NodeKind::User,
        active: None,
        upn: None,
        primary_smtp: None,
        dn: None,
    };
    // Two steps: within one change list, property changes sort before
    // creates (OGAR's safe order), so a list cannot set a property of a node
    // it creates — that is the create's own state.
    let created = sim(
        &mut st,
        g0,
        vec![Change::CreateNode {
            node: g(0x77),
            state: new.clone(),
        }],
    )
    .unwrap();
    let v = sim(
        &mut st,
        created,
        vec![
            active(0x77, None, Some(true)),
            moved(0x77, None, Some(dn(&[1]))),
        ],
    )
    .unwrap();
    assert_eq!(st.view(v).unwrap().delta_len(), 1, "still one created row");
    let view = st.view(v).unwrap();
    assert_eq!(view.node_state(&g(0x77)).unwrap().active, Some(true));
    assert!(active_set(&view).contains(&g(0x77)));
    let in_1: Vec<_> = subtree(&view, SCOPE, NodeKind::User, &dn(&[1]))
        .unwrap()
        .rows();
    assert_eq!(in_1.len(), 2, "BOB and the created user");

    let off = sim(&mut st, g0, vec![active(BOB, Some(true), Some(false))]).unwrap();
    let bob = st.view(off).unwrap().node_state(&g(BOB)).unwrap();
    let gone = sim(
        &mut st,
        off,
        vec![Change::DeleteNode {
            node: g(BOB),
            state: bob,
        }],
    )
    .unwrap();
    assert_eq!(st.view(gone).unwrap().delta_len(), 1, "only the delete");
}
