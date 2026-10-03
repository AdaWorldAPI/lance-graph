//! observe → simulate → validate → diff → plan, over the SoA substrate.

use lance_graph_dir_sim::validate::{dangling, dangling_program, validate};
use lance_graph_dir_sim::*;
use lance_graph_mask_risc::{materialize_rows, MaskOp};
use ogar_dir_core::{Guid128, OuHhtl};
use ogar_dir_sim::*;
use std::sync::Arc;

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

fn observed() -> Observation {
    Observation {
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
fn chain() -> (VersionStore, VersionId, VersionId, VersionId) {
    chain_from(observed())
}

// 1. G0 is unchanged by simulation.
#[test]
fn t01_g0_immutable() {
    let mut fresh = VersionStore::new();
    let only = fresh.observe("lab", 1_000, observed()).unwrap();
    let (st, g0, ..) = chain();
    assert_eq!(**st.snapshot(g0).unwrap(), **fresh.snapshot(only).unwrap());
    assert_eq!(st.view(g0).unwrap().delta_len(), 0);
    assert!(!st.view(g0).unwrap().is_member(&g(ALICE), &g(EXCHANGE)));
}

// 2 + 10. G1 has parent G0; provenance names the rule and the evidence.
#[test]
fn t02_t10_child_with_provenance() {
    let (st, g0, g1, _) = chain();
    let v = st.version(g1).unwrap();
    assert_eq!(v.parent, Some(g0));
    assert_eq!(
        v.origin,
        Origin::Simulated {
            rule: EXCHANGE_ACCESS,
            evidence: ev("REQ-1")
        }
    );
    assert_eq!(
        v.delta,
        vec![Change::AddMembership {
            user: g(ALICE),
            group: g(EXCHANGE)
        }]
    );
    assert!(st.view(g1).unwrap().is_member(&g(ALICE), &g(EXCHANGE)));
}

// 3. A population rule chains from G1 and adds only Bob.
#[test]
fn t03_population_rule_chains() {
    let (mut st, g0, g1, g2) = chain();
    assert_eq!(st.version(g2).unwrap().parent, Some(g1));
    assert_eq!(
        st.version(g2).unwrap().delta,
        vec![Change::AddMembership {
            user: g(BOB),
            group: g(EXCHANGE)
        }]
    );
    assert_eq!(st.lineage(g2).unwrap(), vec![g0, g1, g2]);
    assert_eq!(
        st.simulate(g2, &imply(), &[]),
        Err(SimError::EmptyProposal(EMPLOYEES_GET_EXCHANGE))
    );
}

// 4. diff(G0, G1) is one semantic change.
#[test]
fn t04_diff_one_change() {
    let (st, g0, g1, g2) = chain();
    assert_eq!(
        st.diff(g0, g1).unwrap(),
        vec![Change::AddMembership {
            user: g(ALICE),
            group: g(EXCHANGE)
        }]
    );
    assert_eq!(st.diff(g0, g2).unwrap().len(), 2);
    assert_eq!(
        st.diff(g2, g0).unwrap().len(),
        2,
        "reverse diff removes both"
    );
    assert!(st.diff(g1, g1).unwrap().is_empty());
}

// 5. Valid memberships pass.
#[test]
fn t05_valid_passes() {
    let (mut st, g0, _, g2) = chain();
    assert!(st.validate(g0).unwrap().is_empty());
    assert!(st.validate(g2).unwrap().is_empty());
    st.promote_desired(g2).unwrap();
    assert_eq!(st.tag(TAG_DESIRED), Some(g2));
}

// 6. Dangling memberships fail — in the base relation (observed) and in the
// overlay (simulated), on both endpoint sides, with identity preserved.
#[test]
fn t06_dangling_fails() {
    let ghost = g(0x66);
    let mut obs = observed();
    obs.members.push((ghost, g(EMPLOYEES))); // unknown user
    obs.members.push((g(EMPLOYEES), g(EXCHANGE))); // a group as member
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, obs).unwrap();
    let bad = st
        .simulate(
            g0,
            &GrantGroup {
                rule: EXCHANGE_ACCESS,
                group: ghost,
                to: vec![g(BOB)],
            },
            &[],
        )
        .unwrap();
    let v = st.validate(bad).unwrap();
    assert_eq!(
        v,
        vec![
            Violation::DanglingMembership {
                user: ghost,
                group: g(EMPLOYEES),
                missing: Endpoint::User
            },
            Violation::DanglingMembership {
                user: g(BOB),
                group: ghost,
                missing: Endpoint::Group
            },
            Violation::DanglingMembership {
                user: g(EMPLOYEES),
                group: g(EXCHANGE),
                missing: Endpoint::User
            },
        ]
    );
}

// 7. Duplicate UPN fails (normalized); a disabled owner does not count.
#[test]
fn t07_duplicate_upn() {
    let mut obs = observed();
    obs.nodes.push((
        g(0x77),
        ObservedNode::user("  ALICE@example.test", "other@example.test"),
    ));
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 0, obs.clone()).unwrap();
    assert_eq!(
        st.validate(v0).unwrap(),
        vec![Violation::DuplicateUpn {
            upn: "alice@example.test".into(),
            owners: vec![g(0x77), g(ALICE)]
        }]
    );
    obs.nodes.last_mut().unwrap().1.active = false;
    let v1 = st.observe("lab", 1, obs).unwrap();
    assert!(st.validate(v1).unwrap().is_empty());
}

// 8 + 9 + rejected future: Bob.primarySMTP = alice@… is constructible,
// detected through the overlay, refused, kept — and nothing else moves.
#[test]
fn t08_t09_smtp_collision_rejected() {
    let (mut st, g0, g1, g2) = chain();
    st.promote_desired(g2).unwrap();
    let before: Vec<_> = [g0, g1, g2]
        .iter()
        .map(|v| st.diff(g0, *v).unwrap())
        .collect();
    let snap_before = Arc::clone(st.snapshot(g0).unwrap());

    let rule = SetPrimarySmtp {
        rule: RENAME_MAIL,
        user: g(BOB),
        to: "Alice@Example.test".into(),
    };
    let g3 = st
        .simulate(g2, &rule, &ev("REQ-2"))
        .expect("hypothetical future is constructible");
    let rej = st.promote_desired(g3).unwrap_err();
    assert_eq!(
        rej.violations,
        vec![Violation::DuplicateSmtp {
            address: "alice@example.test".into(),
            owners: vec![g(ALICE), g(BOB)]
        }]
    );
    assert_eq!(st.tag(TAG_DESIRED), Some(g2));
    let after: Vec<_> = [g0, g1, g2]
        .iter()
        .map(|v| st.diff(g0, *v).unwrap())
        .collect();
    assert_eq!(before, after);
    assert!(
        Arc::ptr_eq(&snap_before, st.snapshot(g3).unwrap()),
        "the base was never copied"
    );
    assert_eq!(st.verdict(g3), Some(rej.violations.as_slice()));
    let bob = st.view(g3).unwrap().ordinal(&g(BOB)).unwrap();
    assert_eq!(
        st.view(g3).unwrap().attr(bob, Attribute::PrimarySmtp),
        Some("Alice@Example.test")
    );
}

// A stale compare-and-set creates no version.
#[test]
fn stale_change_creates_no_version() {
    struct Stale;
    impl Rule for Stale {
        fn id(&self) -> RuleId {
            RENAME_MAIL
        }
        fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
            vec![Change::SetAttribute {
                node: g(BOB),
                attribute: Attribute::PrimarySmtp,
                from: Some("not-it@example.test".into()),
                to: Some("x@example.test".into()),
            }]
        }
    }
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, observed()).unwrap();
    assert!(matches!(
        st.simulate(g0, &Stale, &[]),
        Err(SimError::Apply(ApplyError::Stale { .. }))
    ));
    assert!(st.version(VersionId(1)).is_none());
}

// 11. The diff becomes a plan with basis preconditions and no transport.
#[test]
fn t11_plan() {
    let (mut st, g0, _, g2) = chain();
    assert_eq!(st.plan(g2), Err(PlanError::NotDesired(g2)));
    st.promote_desired(g2).unwrap();
    let plan = st.plan(g2).unwrap();
    assert_eq!((plan.basis, plan.target), (g0, g2));
    assert_eq!(
        plan.ops,
        vec![
            PlannedOp {
                op: Operation::AddGroupMember {
                    group: g(EXCHANGE),
                    member: g(ALICE)
                },
                precondition: Precondition::NotMember
            },
            PlannedOp {
                op: Operation::AddGroupMember {
                    group: g(EXCHANGE),
                    member: g(BOB)
                },
                precondition: Precondition::NotMember
            },
        ]
    );
    let mut s2 = VersionStore::new();
    let b0 = s2.observe("lab", 0, observed()).unwrap();
    let b1 = s2
        .simulate(
            b0,
            &SetPrimarySmtp {
                rule: RENAME_MAIL,
                user: g(BOB),
                to: "robert@example.test".into(),
            },
            &[],
        )
        .unwrap();
    s2.promote_desired(b1).unwrap();
    assert_eq!(
        s2.plan(b1).unwrap().ops,
        vec![PlannedOp {
            op: Operation::SetAttribute {
                object: g(BOB),
                attribute: Attribute::PrimarySmtp,
                value: Some("robert@example.test".into())
            },
            precondition: Precondition::AttributeEquals(Some("bob@example.test".into())),
        }]
    );
}

// 12. Same input + same rules ⇒ same desired state, diff and plan.
#[test]
fn t12_deterministic() {
    let run = || {
        let (mut st, g0, _, g2) = chain();
        st.promote_desired(g2).unwrap();
        (st.diff(g0, g2).unwrap(), st.plan(g2).unwrap())
    };
    assert_eq!(run(), run());
}

// 13. Guid128 survives the dense-ordinal mapping, including GUIDs that
// differ only in their upper 64 bits.
#[test]
fn t13_guid_ordinal_round_trip() {
    let a = Guid128::parse("00000000-0000-0001-0123-456789abcdef").unwrap();
    let b = Guid128::parse("00000000-0000-0002-0123-456789abcdef").unwrap();
    let mut obs = observed();
    obs.nodes.push((b, ObservedNode::group()));
    obs.nodes.push((a, ObservedNode::group()));
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, obs).unwrap();
    let view = st.view(v).unwrap();
    for id in [a, b, g(ALICE), g(BOB), g(EMPLOYEES), g(EXCHANGE)] {
        let o = view.ordinal(&id).unwrap();
        assert_eq!(view.guid(o), Some(id));
    }
    assert_ne!(view.ordinal(&a), view.ordinal(&b));
    assert_eq!(view.ordinal(&g(0x99)), None);
}

// 14. Membership validation reads only membership lanes and kind planes: it
// finds the dangling edge in a directory with no strings at all.
#[test]
fn t14_membership_validation_reads_no_attributes() {
    let node = |kind| ObservedNode {
        kind,
        active: true,
        upn: None,
        primary_smtp: None,
        ou: None,
    };
    let obs = Observation {
        nodes: vec![(g(1), node(NodeKind::User)), (g(2), node(NodeKind::Group))],
        members: vec![(g(1), g(2)), (g(1), g(3))],
    };
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, obs).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(
        dangling(&view),
        vec![Violation::DanglingMembership {
            user: g(1),
            group: g(3),
            missing: Endpoint::Group
        }]
    );
}

// 15. Edge integrity is an anti-join lowered to MaskOp::Gather (Quack's
// semijoin), not a hand-written hash join.
#[test]
fn t15_dangling_uses_semijoin() {
    let p = dangling_program();
    let gathers = p
        .ops
        .iter()
        .filter(|op| matches!(op, MaskOp::Gather { .. }))
        .count();
    assert_eq!(gathers, 2, "one semijoin per endpoint side: {:?}", p.ops);
}

// 16. A one-edge mutation shares the snapshot and holds one delta entry.
#[test]
fn t16_structural_sharing() {
    let (st, g0, g1, g2) = chain();
    let s0 = st.snapshot(g0).unwrap();
    assert!(Arc::ptr_eq(s0, st.snapshot(g1).unwrap()));
    assert!(Arc::ptr_eq(s0, st.snapshot(g2).unwrap()));
    assert_eq!(st.view(g1).unwrap().delta_len(), 1);
    assert_eq!(st.view(g2).unwrap().delta_len(), 2);
}

// 17. Semantic output is invariant under input ordering.
#[test]
fn t17_ordering_invariance() {
    let mut shuffled = observed();
    shuffled.nodes.reverse();
    shuffled.nodes.swap(0, 2);
    shuffled.members.reverse();
    shuffled.members.push((g(ALICE), g(EMPLOYEES))); // duplicate observation
    let outputs = |obs| {
        let (mut st, g0, _, g2) = chain_from(obs);
        st.promote_desired(g2).unwrap();
        let rule = SetPrimarySmtp {
            rule: RENAME_MAIL,
            user: g(BOB),
            to: "alice@example.test".into(),
        };
        let g3 = st.simulate(g2, &rule, &[]).unwrap();
        (
            st.diff(g0, g2).unwrap(),
            st.plan(g2).unwrap().ops,
            st.validate(g3).unwrap(),
        )
    };
    assert_eq!(outputs(observed()), outputs(shuffled));
}

// Audit: why is Alice in ExchangeUsers?
#[test]
fn audit_chain() {
    let (st, g0, g1, g2) = chain();
    let why = st.explain_membership(g2, &g(ALICE), &g(EXCHANGE)).unwrap();
    assert_eq!(why.iter().map(|v| v.id).collect::<Vec<_>>(), vec![g0, g1]);
    assert!(matches!(why[0].origin, Origin::Observed { .. }));
    match &why[1].origin {
        Origin::Simulated { rule, evidence } => {
            assert_eq!(rule.to_string(), "ExchangeAccess/v1");
            assert_eq!(evidence, &ev("REQ-1"));
        }
        o => panic!("unexpected {o:?}"),
    }
    assert_eq!(
        st.explain_membership(g2, &g(BOB), &g(EXCHANGE))
            .unwrap()
            .last()
            .unwrap()
            .id,
        g2
    );
    assert_eq!(
        st.explain_membership(g2, &g(ALICE), &g(EMPLOYEES))
            .unwrap()
            .len(),
        1
    );
    assert!(st.explain_membership(g0, &g(ALICE), &g(EXCHANGE)).is_none());
}

// Convergence: re-observing the desired state diffs empty (full-merge path).
#[test]
fn converged_observation_diffs_empty() {
    let (mut st, _, _, g2) = chain();
    st.promote_desired(g2).unwrap();
    let mut actual = observed();
    actual
        .members
        .extend([(g(ALICE), g(EXCHANGE)), (g(BOB), g(EXCHANGE))]);
    let o = st.observe("lab", 2_000, actual).unwrap();
    assert!(st.diff(o, g2).unwrap().is_empty());
    let mut drifted = observed();
    drifted.members.push((g(ALICE), g(EXCHANGE)));
    let d = st.observe("lab", 3_000, drifted).unwrap();
    assert_eq!(
        st.diff(d, g2).unwrap(),
        vec![Change::AddMembership {
            user: g(BOB),
            group: g(EXCHANGE)
        }]
    );
}

// Removal and re-add net out; removals reach the plan as RemoveGroupMember.
#[test]
fn removal_round_trip() {
    struct Drop;
    impl Rule for Drop {
        fn id(&self) -> RuleId {
            RuleId {
                name: "Leaver",
                version: 1,
            }
        }
        fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
            vec![Change::RemoveMembership {
                user: g(BOB),
                group: g(EMPLOYEES),
            }]
        }
    }
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, observed()).unwrap();
    let r1 = st.simulate(g0, &Drop, &[]).unwrap();
    assert!(!st.view(r1).unwrap().is_member(&g(BOB), &g(EMPLOYEES)));
    let r2 = st
        .simulate(
            r1,
            &GrantGroup {
                rule: EXCHANGE_ACCESS,
                group: g(EMPLOYEES),
                to: vec![g(BOB)],
            },
            &[],
        )
        .unwrap();
    assert!(st.diff(g0, r2).unwrap().is_empty());
    st.promote_desired(r1).unwrap();
    assert_eq!(
        st.plan(r1).unwrap().ops,
        vec![PlannedOp {
            op: Operation::RemoveGroupMember {
                group: g(EMPLOYEES),
                member: g(BOB)
            },
            precondition: Precondition::IsMember
        }]
    );
}

// HHTL as executable geometry: subtree selection is one prefix match.
#[test]
fn ou_subtree_is_a_prefix_match() {
    let ou = |l: &[u16]| {
        let mut h = OuHhtl::ROOT;
        h.0[..l.len()].copy_from_slice(l);
        h
    };
    let mut obs = observed();
    obs.nodes[0].1.ou = Some(ou(&[1, 1, 1])); // Stuttgart/Infrastructure/Exchange
    obs.nodes[1].1.ou = Some(ou(&[2])); // Berlin
    obs.nodes.push((
        g(0x55),
        ObservedNode {
            ou: Some(ou(&[1, 2])),
            ..ObservedNode::user("c@x", "c@x")
        },
    ));
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, obs).unwrap();
    let view = st.view(v).unwrap();
    let pick = |p: &OuHhtl| -> Vec<Guid128> {
        materialize_rows(&subtree(&view, p).unwrap(), view.len())
            .into_iter()
            .map(|o| view.guid(o as u32).unwrap())
            .collect()
    };
    assert_eq!(pick(&ou(&[1])), vec![g(0x55), g(ALICE)]);
    assert_eq!(pick(&ou(&[1, 1])), vec![g(ALICE)]);
    assert_eq!(pick(&ou(&[2])), vec![g(BOB)]);
    assert_eq!(pick(&OuHhtl::ROOT).len(), 3, "root = every located node");
    assert_eq!(
        subtree(&view, &ou(&[1, 1, 1, 1, 1])),
        Err(SubtreeTooDeep(5))
    );
}

// Observation from OGAR PR #313 records.
#[test]
fn observe_from_ogar_ad() {
    use ogar_dir_core::{OuDictionary, ValuePool};
    let ldif = "dn: CN=Alice,OU=Staff,DC=example,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAQ==\nobjectClass: user\nuserPrincipalName: alice@example.test\nproxyAddresses: smtp:a@legacy.test\nproxyAddresses: SMTP:alice@example.test\nuserAccountControl: 514\n\ndn: CN=Employees,OU=Groups,DC=example,DC=test\nobjectGUID:: 1MOyoQAAAECAAAAAAAC+7w==\nobjectClass: group\n";
    let (mut d, mut p) = (OuDictionary::new(), ValuePool::new());
    let recs: Vec<_> = ogar_ad::ldif::parse(ldif)
        .unwrap()
        .iter()
        .map(|e| {
            ogar_ad::encode(e, Guid128::NIL, &mut d, &mut p, 0)
                .unwrap()
                .record
        })
        .collect();
    let obs = observe::from_ad(&recs, &p);
    assert_eq!(
        obs.nodes[0].1.primary_smtp.as_deref(),
        Some("alice@example.test")
    );
    assert!(!obs.nodes[0].1.active, "UAC 514 = disabled");
    assert!(obs.nodes[0].1.ou.is_some());
    assert_eq!(obs.nodes[1].1.kind, NodeKind::Group);
    let mut st = VersionStore::new();
    let v = st.observe("ogar-ad:ldif", 0, obs).unwrap();
    assert!(validate(&st.view(v).unwrap()).is_empty());
}

// Overrides REPLACE the observed value: renaming the second owner away from
// an observed collision must clear it (the base owner plane drops overridden
// users; otherwise the old value would still be counted).
#[test]
fn renaming_away_resolves_an_observed_collision() {
    let carol = g(0xC0);
    let mut obs = observed();
    obs.nodes.push((
        carol,
        ObservedNode::user("carol@example.test", "BOB@example.test"),
    ));
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, obs).unwrap();
    assert_eq!(
        st.validate(g0).unwrap(),
        vec![Violation::DuplicateSmtp {
            address: "bob@example.test".into(),
            owners: vec![g(BOB), carol]
        }]
    );
    let fix = SetPrimarySmtp {
        rule: RENAME_MAIL,
        user: carol,
        to: "carol@example.test".into(),
    };
    let g1 = st.simulate(g0, &fix, &[]).unwrap();
    assert!(st.validate(g1).unwrap().is_empty());
    st.promote_desired(g1).unwrap();
}
