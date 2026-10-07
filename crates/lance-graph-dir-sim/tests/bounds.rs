//! The bounded numeric substrate: two independent 65,536-node ordinal
//! spaces, sparse `u16 × u16` membership, the `Dn128` hierarchy, and the
//! stable `ValueId` boundary between strings and execution.

use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Dn128, Dn128Error, Guid128};
use ogar_dir_sim::{
    Attribute, Change, EvidenceRef, NodeState, Operation, Precondition, RuleId, VersionId,
};

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

/// Users are `0x00…`, groups `0x01…`, so each kind's ordinal is `i`.
fn user(i: u32) -> Guid128 {
    let mut b = [0u8; 16];
    b[1..5].copy_from_slice(&i.to_be_bytes());
    Guid128(b)
}
fn group(i: u32) -> Guid128 {
    let mut b = user(i).0;
    b[0] = 1;
    Guid128(b)
}
fn bare(kind: NodeKind) -> ObservedNode {
    ObservedNode {
        kind,
        active: Some(true),
        upn: None,
        primary_smtp: None,
        proxies: Vec::new(),
        dn: None,
        recipient: None,
        mail: None,
        alias: None,
    }
}
fn population(users: u32, groups: u32) -> Observation {
    let mut obs = Observation {
        scope: SCOPE,
        ..Observation::default()
    };
    obs.nodes
        .extend((0..users).map(|i| (user(i), bare(NodeKind::User))));
    obs.nodes
        .extend((0..groups).map(|i| (group(i), bare(NodeKind::Group))));
    obs
}

struct Propose(Vec<Change>);
impl Rule for Propose {
    fn id(&self) -> RuleId {
        RuleId {
            name: "Propose",
            version: 1,
        }
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}
fn apply_err(r: Result<VersionId, SimError>) -> ApplyError {
    match r {
        Err(SimError::Apply(e)) => e,
        other => panic!("expected an apply error, got {other:?}"),
    }
}

// ---- 1. two independent 64k ordinal spaces -------------------------------

#[test]
fn both_populations_reach_65536_at_once_and_every_u16_is_an_ordinal() {
    let n = MAX_USERS as u32;
    assert_eq!(MAX_GROUPS, MAX_USERS);
    let mut obs = population(n, n);
    // The last ordinal on both sides, as one membership row: no value of a
    // u16 is reserved, so (65535, 65535) is an ordinary row.
    obs.members.push((user(n - 1), group(n - 1)));
    obs.members.push((user(0), group(0)));
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, obs).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(
        (view.users_len(), view.groups_len()),
        (MAX_USERS, MAX_GROUPS)
    );
    // 131,072 nodes in total: one shared space would not hold them.
    assert_eq!(view.user_ordinal(&user(n - 1)), Some(UserOrdinal(u16::MAX)));
    assert_eq!(
        view.group_ordinal(&group(n - 1)),
        Some(GroupOrdinal(u16::MAX))
    );
    assert_eq!(view.user_ordinal(&user(0)), Some(UserOrdinal(0)));
    assert_eq!(view.group_ordinal(&group(0)), Some(GroupOrdinal(0)));
    assert_eq!(view.user_guid(UserOrdinal(u16::MAX)), Some(user(n - 1)));
    assert!(view.is_member(&user(n - 1), &group(n - 1)));
    assert!(!view.is_member(&user(n - 1), &group(0)));
    assert_eq!(st.snapshot(v).unwrap().membership_rows(), 2);
    assert!(st.validate(v).unwrap().is_empty());
}

#[test]
fn a_population_past_65536_is_refused_per_kind() {
    let n = MAX_USERS as u32;
    let mut st = VersionStore::new();
    assert_eq!(
        st.observe("lab", 0, population(n + 1, 0)),
        Err(BuildError::TooManyUsers(MAX_USERS + 1))
    );
    assert_eq!(
        st.observe("lab", 0, population(0, n + 1)),
        Err(BuildError::TooManyGroups(MAX_GROUPS + 1))
    );
}

#[test]
fn a_full_population_refuses_a_create_and_the_other_one_does_not() {
    let n = MAX_USERS as u32;
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, population(n, 1)).unwrap();
    let state = |kind| NodeState {
        kind,
        active: Some(true),
        upn: None,
        primary_smtp: None,
        dn: None,
        recipient: None,
    };
    let more_users = Propose(vec![Change::CreateNode {
        node: user(n),
        state: state(NodeKind::User),
    }]);
    assert_eq!(
        apply_err(st.simulate(g0, &more_users, &[])),
        ApplyError::PopulationFull(NodeKind::User)
    );
    let more_groups = Propose(vec![Change::CreateNode {
        node: group(1),
        state: state(NodeKind::Group),
    }]);
    let v = st.simulate(g0, &more_groups, &[]).unwrap();
    assert_eq!(st.view(v).unwrap().groups_len(), 2);
    // A delete frees no ordinal for reuse: the slot stays, the bound holds.
    let gone = Propose(vec![Change::DeleteNode {
        node: user(0),
        state: state(NodeKind::User),
    }]);
    let d = st.simulate(g0, &gone, &[]).unwrap();
    assert_eq!(
        apply_err(st.simulate(d, &more_users, &[])),
        ApplyError::PopulationFull(NodeKind::User)
    );
}

// ---- 2. sparse membership ------------------------------------------------

#[test]
fn an_unresolved_membership_is_kept_by_identity_never_as_a_lane_value() {
    let mut obs = population(2, 2);
    obs.members.push((user(0), group(9))); // no such group
    obs.members.push((group(0), group(1))); // a group as member
    obs.members.push((user(1), group(1)));
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, obs).unwrap();
    // Only the resolved row is in the lanes.
    assert_eq!(st.snapshot(v).unwrap().membership_rows(), 1);
    let view = st.view(v).unwrap();
    assert!(view.is_member(&user(0), &group(9)), "still observed");
    assert_eq!(st.validate(v).unwrap().len(), 2);
    // And it can be removed like any other membership.
    let fix = Propose(vec![
        Change::RemoveMembership {
            user: user(0),
            group: group(9),
        },
        Change::RemoveMembership {
            user: group(0),
            group: group(1),
        },
    ]);
    let w = st.simulate(v, &fix, &[]).unwrap();
    assert!(st.validate(w).unwrap().is_empty());
    assert_eq!(st.diff(v, w).unwrap().len(), 2);
}

// ---- 3. Dn128 hierarchy --------------------------------------------------

#[test]
fn dn128_subtree_at_depth_1_4_and_16() {
    let dn = |l: &[u8]| Dn128::new(l).unwrap();
    let path16: Vec<u8> = (0..16).map(|k| (k * 17) as u8).collect();
    let mut sibling16 = path16.clone();
    sibling16[15] ^= 1;
    let located = [
        dn(&path16),          // user 0: the full depth-16 path
        dn(&sibling16),       // user 1: same parent, different leaf
        dn(&path16[..4]),     // user 2: an ancestor at depth 4
        dn(&path16[..1]),     // user 3: an ancestor at depth 1
        dn(&[path16[0] ^ 1]), // user 4: a sibling at depth 1
        // user 5: the depth-3 ancestor. Its zero tail equals the byte a
        // `[0, 17, 34, 0]` query compares at level 3.
        dn(&path16[..3]),
    ];
    let mut obs = population(located.len() as u32 + 1, 0); // last user unlocated
    for (i, d) in located.iter().enumerate() {
        obs.nodes[i].1.dn = Some(*d);
    }
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, obs).unwrap();
    let view = st.view(v).unwrap();
    let pick = |p: &Dn128| subtree(&view, SCOPE, NodeKind::User, p).unwrap().rows();
    assert_eq!(
        pick(&dn(&path16)),
        vec![0],
        "depth 16: the leaf, not its sibling"
    );
    assert_eq!(pick(&dn(&path16[..15])), vec![0, 1], "the shared parent");
    assert_eq!(
        pick(&dn(&path16[..4])),
        vec![0, 1, 2],
        "depth 4: self and below"
    );
    // Anti-vacuity: the 16-byte compare alone accepts user 5 here …
    let (pattern, care) = dn(&[0, 17, 34, 0]).subtree_mask();
    let bytes = located[5].bytes();
    assert!((0..16).all(|k| (bytes[k] ^ pattern[k]) & care[k] == 0));
    // … and the depth gate is what keeps it out.
    assert_eq!(pick(&dn(&[0, 17, 34, 0])), Vec::<usize>::new());
    assert_eq!(pick(&dn(&path16[..3])), vec![0, 1, 2, 5], "depth 3");
    assert_eq!(pick(&dn(&path16[..1])), vec![0, 1, 2, 3, 5], "depth 1");
    assert_eq!(
        pick(&Dn128::ROOT).len(),
        located.len(),
        "every located user"
    );
    // The ancestor is answered from the node's own bytes, by OGAR's rule.
    for i in pick(&dn(&path16[..4])) {
        assert!(dn(&path16[..4]).is_ancestor_of(&located[i]));
    }
}

#[test]
fn a_seventeenth_level_and_a_257th_child_are_refused() {
    assert_eq!(Dn128::new(&[0; 17]), Err(Dn128Error::TooDeep(17)));
    // 257 OUs under one parent, through the real ingress path.
    let mut ldif = String::new();
    for i in 0..257u32 {
        let mut guid = [0u8; 16];
        guid[..4].copy_from_slice(&i.to_be_bytes());
        guid[15] = 1;
        ldif.push_str(&format!(
            "dn: CN=U{i},OU=O{i},DC=example,DC=test\nobjectGUID:: {}\nobjectClass: user\n\n",
            b64(&guid)
        ));
    }
    let (mut d, mut p) = (
        ogar_dir_core::OuDictionary::new(),
        ogar_dir_core::ValuePool::new(),
    );
    let recs: Vec<_> = ogar_ad::ldif::parse(&ldif)
        .unwrap()
        .iter()
        .map(|e| {
            ogar_ad::encode(e, SCOPE.0, &mut d, &mut p, 0)
                .unwrap()
                .record
        })
        .collect();
    // 256 children still convert…
    assert!(observe::from_ad(SCOPE, &recs[..256], &p).is_ok());
    // …the 257th fails closed, naming the record, and nothing is observed.
    assert_eq!(
        observe::from_ad(SCOPE, &recs, &p).unwrap_err(),
        observe::ObserveError::Location {
            node: recs[256].node_guid(),
            error: Dn128Error::ChildCodeOverflow {
                level: 0,
                segment: 257
            }
        }
    );
}

/// A `Dn128` holds no scope, so the scope is enforced at the edges: a record
/// from another directory is refused at ingress, and two versions from
/// different directories are neither diffed nor planned against each other.
#[test]
fn a_foreign_scope_is_refused_at_ingress_diff_and_plan() {
    let other = DirectoryScope(Guid128([0x0D; 16]));
    let ldif = |n: u8| {
        format!(
            "dn: CN=U,OU=Same,DC=example,DC=test\nobjectGUID:: {}\nobjectClass: user\n\n",
            b64(&[n; 16])
        )
    };
    let (mut d, mut p) = (
        ogar_dir_core::OuDictionary::new(),
        ogar_dir_core::ValuePool::new(),
    );
    let mut rec = |n: u8, scope: DirectoryScope| {
        let e = &ogar_ad::ldif::parse(&ldif(n)).unwrap()[0];
        ogar_ad::encode(e, scope.0, &mut d, &mut p, 0)
            .unwrap()
            .record
    };
    let (local, foreign) = (rec(1, SCOPE), rec(2, other));
    // Same OU path, so the same Dn128: only the scope tells them apart.
    assert_eq!(local.ou_hhtl(), foreign.ou_hhtl());
    assert_eq!(
        observe::from_ad(SCOPE, &[local, foreign], &p).unwrap_err(),
        (observe::ObserveError::ForeignScope {
            node: foreign.node_guid(),
            scope: other,
        })
    );

    let mut st = VersionStore::new();
    let here = st
        .observe("ad", 0, observe::from_ad(SCOPE, &[local], &p).unwrap())
        .unwrap();
    let there = st
        .observe("ad", 1, observe::from_ad(other, &[foreign], &p).unwrap())
        .unwrap();
    assert_eq!(
        st.diff(here, there),
        Err(SimError::ScopeMismatch(here, there))
    );
    // Desired lineage in one scope, latest observation in another: no plan.
    let mut st = VersionStore::new();
    let mut obs = population(1, 1);
    let g0 = st.observe("lab", 0, obs.clone()).unwrap();
    let add = Propose(vec![Change::AddMembership {
        user: user(0),
        group: group(0),
    }]);
    let g1 = st.simulate(g0, &add, &[]).unwrap();
    st.promote_desired(g1).unwrap();
    assert!(st.plan(g1).is_ok());
    obs.scope = other;
    let o = st.observe("lab", 1, obs).unwrap();
    assert_eq!(
        st.plan(g1),
        Err(ogar_dir_sim::PlanError::ScopeMismatch {
            basis: o,
            target: g1
        })
    );
}

fn b64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in bytes.chunks(3) {
        let n = c
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, b)| n | u32::from(*b) << (16 - 8 * i));
        for k in 0..4 {
            if k <= c.len() {
                out.push(T[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

// ---- 4. stable ValueId boundary ------------------------------------------

#[test]
fn a_value_id_keeps_its_meaning_from_ingress_to_plan_across_observations() {
    let mut st = VersionStore::new();
    // Ingress first: the id is fixed before anything is observed.
    let alice = st.intern("Alice@Example.test");
    let renamed = st.intern("a.smith@example.test");
    let mut obs = population(1, 0);
    obs.nodes[0].1.primary_smtp = Some("Alice@Example.test".into());
    let g0 = st.observe("lab", 0, obs.clone()).unwrap();
    let view = st.view(g0).unwrap();
    assert_eq!(view.attr(&user(0), Attribute::PrimarySmtp), Some(alice));
    drop(view);
    let rule = SetPrimarySmtp {
        rule: RuleId {
            name: "Rename",
            version: 1,
        },
        user: user(0),
        to: renamed,
    };
    let g1 = st.simulate(g0, &rule, &[]).unwrap();
    st.promote_desired(g1).unwrap();
    // A second observation re-interns the same string: same id.
    st.observe("lab", 1, obs).unwrap();
    let plan = st.plan(g1).unwrap();
    assert_eq!(
        plan.ops[0].op,
        Operation::SetAttribute {
            object: user(0),
            attribute: Attribute::PrimarySmtp,
            value: Some(renamed),
        }
    );
    assert_eq!(
        plan.ops[0].precondition,
        Precondition::AttributeEquals(Some(alice))
    );
    // Strings come back only at egress.
    assert_eq!(st.value(alice), Some("Alice@Example.test"));
    assert_eq!(st.value(renamed), Some("a.smith@example.test"));
    // Comparison happens on keys: different raw values, one key.
    let shout = st.intern("ALICE@EXAMPLE.TEST");
    assert_ne!(shout, alice);
    assert_eq!(st.key_of(shout), st.key_of(alice));
    assert_eq!(
        st.key_label(st.key_of(alice).unwrap()),
        Some("alice@example.test")
    );
}

#[test]
fn a_value_id_the_store_never_issued_is_refused() {
    let mut st = VersionStore::new();
    let mut obs = population(1, 0);
    obs.nodes[0].1.upn = Some("u@example.test".into());
    let g0 = st.observe("lab", 0, obs).unwrap();
    let from = st.view(g0).unwrap().attr(&user(0), Attribute::Upn);
    let bogus = ValueId(4_000_000);
    let set = Propose(vec![Change::SetAttribute {
        node: user(0),
        attribute: Attribute::Upn,
        from,
        to: Some(bogus),
    }]);
    assert_eq!(
        apply_err(st.simulate(g0, &set, &[])),
        ApplyError::Uninterned(bogus)
    );
}

/// An observed membership whose group was missing resolves once a version
/// creates that group: it is no longer dangling, it is counted, and a
/// population rule sees it.
#[test]
fn an_unresolved_membership_resolves_when_its_endpoint_is_created() {
    let mut obs = population(2, 1);
    obs.members.push((user(0), group(7))); // group 7 not observed yet
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, obs).unwrap();
    assert_eq!(st.validate(g0).unwrap().len(), 1, "dangling while missing");
    let create = Propose(vec![Change::CreateNode {
        node: group(7),
        state: NodeState {
            kind: NodeKind::Group,
            active: Some(true),
            upn: None,
            primary_smtp: None,
            dn: None,
            recipient: None,
        },
    }]);
    let g1 = st.simulate(g0, &create, &[]).unwrap();
    assert!(
        st.validate(g1).unwrap().is_empty(),
        "resolved by the create"
    );
    let view = st.view(g1).unwrap();
    let counts = member_counts(&view, &group(7));
    assert_eq!(counts, vec![1, 0], "counted like any membership");
    // A rule over the now-resolved edge: members of 7 should be in 0.
    let imply = ImplyGroup {
        rule: RuleId {
            name: "Imply",
            version: 1,
        },
        source: group(7),
        target: group(0),
    };
    assert_eq!(
        imply.propose(&view, &[]),
        vec![Change::AddMembership {
            user: user(0),
            group: group(0)
        }]
    );
    drop(view);
    // Removing and re-adding the edge round-trips.
    let edge = |add| {
        let c = if add {
            Change::AddMembership {
                user: user(0),
                group: group(7),
            }
        } else {
            Change::RemoveMembership {
                user: user(0),
                group: group(7),
            }
        };
        Propose(vec![c])
    };
    let g2 = st.simulate(g1, &edge(false), &[]).unwrap();
    assert!(!st.view(g2).unwrap().is_member(&user(0), &group(7)));
    assert_eq!(member_counts(&st.view(g2).unwrap(), &group(7)), vec![0, 0]);
    let g3 = st.simulate(g2, &edge(true), &[]).unwrap();
    assert_eq!(member_counts(&st.view(g3).unwrap(), &group(7)), vec![1, 0]);
    assert!(st.diff(g1, g3).unwrap().is_empty());
}
