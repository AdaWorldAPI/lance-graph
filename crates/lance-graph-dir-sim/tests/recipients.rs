//! Exchange recipients (OGAR `exchange`): ingest, the snapshot lanes, the
//! overlay, diff, plan and reconcile — and address uniqueness by recipient,
//! not by the account's enabled flag. A shared, room or equipment mailbox is
//! a disabled account and still owns its addresses.

use lance_graph_dir_sim::validate::validate;
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const R: RuleId = RuleId {
    name: "Propose",
    version: 1,
};
const ALICE: u8 = 0xA1;
const SHARED: u8 = 0x5A;
const DORMANT: u8 = 0xD0;
const LEGACY: u8 = 0x1E;
const ROUTING: &str = "team@tenant.mail.onmicrosoft.com";

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
/// A migrated remote shared mailbox (code 100), as AD stores it.
fn shared_attrs() -> ObservedRecipient {
    ObservedRecipient {
        remote_recipient_type: Some(100),
        display_type: Some(-2_147_483_642),
        type_details: Some(34_359_738_368),
        target_address: Some(ROUTING.into()),
    }
}
fn user(smtp: &str, active: Option<bool>, rcp: Option<ObservedRecipient>) -> ObservedNode {
    let mut u = ObservedNode::user(&format!("{smtp}.upn"), smtp);
    u.active = active;
    u.recipient = rcp;
    u
}
/// Alice: an enabled user, recipient not read. Shared: a disabled remote
/// shared mailbox. Dormant: a disabled user that is not mail-enabled
/// (attributes read, all absent). Legacy: a disabled user whose recipient
/// attributes were never read.
fn observed(shared_smtp: &str, dormant_smtp: &str, legacy_smtp: &str) -> Observation {
    Observation {
        scope: SCOPE,
        nodes: vec![
            (g(ALICE), user("alice@x.test", Some(true), None)),
            (
                g(SHARED),
                user(shared_smtp, Some(false), Some(shared_attrs())),
            ),
            (
                g(DORMANT),
                user(
                    dormant_smtp,
                    Some(false),
                    Some(ObservedRecipient::default()),
                ),
            ),
            (g(LEGACY), user(legacy_smtp, Some(false), None)),
        ],
        members: vec![],
    }
}
fn store(obs: Observation) -> (VersionStore, VersionId) {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 0, obs).unwrap();
    (st, v)
}
fn quiet() -> Observation {
    observed("team@x.test", "dormant@x.test", "legacy@x.test")
}
fn sim(st: &mut VersionStore, v: VersionId, cs: Vec<Change>) -> Result<VersionId, SimError> {
    st.simulate(v, &Propose(cs), &[])
}
fn shared_recipient(st: &mut VersionStore) -> Recipient {
    let routing = st.intern(ROUTING);
    RemoteMailboxOp::SetType(RemoteKind::Shared)
        .apply(
            &RemoteMailboxOp::CompleteMove { routing }
                .apply(&Recipient::OnPremisesMailbox {
                    archive: ArchiveState::None,
                })
                .unwrap(),
        )
        .unwrap()
}
fn smtp_owners(st: &mut VersionStore, v: VersionId) -> Vec<Vec<Guid128>> {
    st.validate(v)
        .unwrap()
        .into_iter()
        .filter_map(|x| match x {
            Violation::DuplicateSmtp { owners, .. } => Some(owners),
            _ => None,
        })
        .collect()
}

// The triplet decodes into node_state: read and present, read and absent
// (not mail-enabled), and never read (None) are three different states.
#[test]
fn the_snapshot_decodes_the_recipient_and_keeps_not_read_apart() {
    let (mut st, v) = store(quiet());
    let want = shared_recipient(&mut st);
    let view = st.view(v).unwrap();
    assert_eq!(view.node_state(&g(SHARED)).unwrap().recipient, Some(want));
    assert!(matches!(
        want,
        Recipient::RemoteMailbox(m) if m.kind() == RemoteKind::Shared && m.code() == 100
    ));
    assert_eq!(
        view.node_state(&g(DORMANT)).unwrap().recipient,
        Some(Recipient::NotMailEnabled)
    );
    assert_eq!(view.node_state(&g(LEGACY)).unwrap().recipient, None);
}

// Attributes no lifecycle produces stay raw (`Other`), never guessed.
#[test]
fn an_undecodable_triplet_stays_raw() {
    let mut obs = quiet();
    obs.nodes[1].1.recipient = Some(ObservedRecipient {
        remote_recipient_type: Some(99),
        ..shared_attrs()
    });
    let (st, v) = store(obs);
    let r = st
        .view(v)
        .unwrap()
        .node_state(&g(SHARED))
        .unwrap()
        .recipient;
    assert!(
        matches!(r, Some(Recipient::Other(a)) if a.remote_recipient_type == Some(99)),
        "{r:?}"
    );
}

// The disabled shared mailbox owns its address: a collision with an enabled
// user is a duplicate (owners sorted by identity). The disabled non-mail user and the disabled user
// never read do not count (V4).
#[test]
fn a_disabled_mailbox_owns_its_addresses() {
    let (mut st, v) = store(observed("alice@x.test", "dormant@x.test", "legacy@x.test"));
    assert_eq!(smtp_owners(&mut st, v), vec![vec![g(SHARED), g(ALICE)]]);
    for (dormant, legacy) in [
        ("alice@x.test", "legacy@x.test"),
        ("dormant@x.test", "alice@x.test"),
    ] {
        let (mut st, v) = store(observed("team@x.test", dormant, legacy));
        assert!(smtp_owners(&mut st, v).is_empty(), "{dormant} {legacy}");
    }
}

// The secondary proxies count by recipient too.
#[test]
fn a_disabled_mailbox_owns_its_secondary_addresses() {
    let mut obs = quiet();
    obs.nodes[1].1.proxies = vec!["smtp:Alice@X.test".into()];
    let (mut st, v) = store(obs);
    assert_eq!(smtp_owners(&mut st, v), vec![vec![g(SHARED), g(ALICE)]]);
}

// Enabling a remote mailbox on the disabled user makes it an owner (a
// collision appears); disabling it again makes it stop.
#[test]
fn a_recipient_change_moves_ownership() {
    let (mut st, g0) = store(observed("team@x.test", "alice@x.test", "legacy@x.test"));
    assert!(smtp_owners(&mut st, g0).is_empty());
    let routing = st.intern("dormant@tenant.mail.onmicrosoft.com");
    let enabled = RemoteMailboxOp::Enable {
        kind: RemoteKind::User,
        routing,
    }
    .apply(&Recipient::NotMailEnabled)
    .unwrap();
    let set = |from, to| Change::SetRecipient {
        node: g(DORMANT),
        from: Some(from),
        to: Some(to),
    };
    let v1 = sim(&mut st, g0, vec![set(Recipient::NotMailEnabled, enabled)]).unwrap();
    assert_eq!(
        st.view(v1)
            .unwrap()
            .node_state(&g(DORMANT))
            .unwrap()
            .recipient,
        Some(enabled)
    );
    assert_eq!(smtp_owners(&mut st, v1), vec![vec![g(ALICE), g(DORMANT)]]);
    let disabled = RemoteMailboxOp::Disable.apply(&enabled).unwrap();
    let v2 = sim(&mut st, v1, vec![set(enabled, disabled)]).unwrap();
    assert!(smtp_owners(&mut st, v2).is_empty(), "deprovisioned");
}

// Compare-and-set: a stale `from` is refused; a routing address the store
// never issued is refused.
#[test]
fn a_stale_or_uninterned_recipient_is_refused() {
    let (mut st, g0) = store(quiet());
    let routing = st.intern("x@tenant.mail.onmicrosoft.com");
    let enabled = RemoteMailboxOp::Enable {
        kind: RemoteKind::User,
        routing,
    }
    .apply(&Recipient::NotMailEnabled)
    .unwrap();
    let stale = Change::SetRecipient {
        node: g(SHARED),
        from: Some(Recipient::NotMailEnabled),
        to: Some(enabled),
    };
    assert!(matches!(
        sim(&mut st, g0, vec![stale]),
        Err(SimError::Apply(ApplyError::Refused {
            refusal: Refusal::Stale,
            ..
        }))
    ));
    let foreign = RemoteMailboxOp::Enable {
        kind: RemoteKind::User,
        routing: ValueId(9_999_999),
    }
    .apply(&Recipient::NotMailEnabled)
    .unwrap();
    let bad = Change::SetRecipient {
        node: g(DORMANT),
        from: Some(Recipient::NotMailEnabled),
        to: Some(foreign),
    };
    assert!(matches!(
        sim(&mut st, g0, vec![bad]),
        Err(SimError::Apply(ApplyError::Uninterned(ValueId(9_999_999))))
    ));
}

// Net effect, diff, plan and reconcile: there and back is no override and
// no diff; a desired enable is planned as one lifecycle step guarded by the
// observed recipient, and drops out once reality shows it.
#[test]
fn diff_plan_and_reconcile_carry_the_recipient() {
    let (mut st, g0) = store(quiet());
    let routing = st.intern("dormant@tenant.mail.onmicrosoft.com");
    let enabled = RemoteMailboxOp::Enable {
        kind: RemoteKind::User,
        routing,
    }
    .apply(&Recipient::NotMailEnabled)
    .unwrap();
    let change = Change::SetRecipient {
        node: g(DORMANT),
        from: Some(Recipient::NotMailEnabled),
        to: Some(enabled),
    };
    let back = Change::SetRecipient {
        node: g(DORMANT),
        from: Some(enabled),
        to: Some(Recipient::NotMailEnabled),
    };
    let there = sim(&mut st, g0, vec![change.clone()]).unwrap();
    let undone = sim(&mut st, there, vec![back]).unwrap();
    assert_eq!(st.view(undone).unwrap().delta_len(), 0);
    assert!(st.diff(g0, undone).unwrap().is_empty());
    assert_eq!(st.diff(g0, there).unwrap(), vec![change]);

    st.promote_desired(there).unwrap();
    let want = vec![PlannedOp {
        op: Operation::RemoteMailbox {
            object: g(DORMANT),
            op: RemoteMailboxOp::Enable {
                kind: RemoteKind::User,
                routing,
            },
        },
        precondition: Precondition::RecipientEquals(Recipient::NotMailEnabled),
    }];
    assert_eq!(st.plan(there).unwrap().ops, want);
    st.observe("lab", 1, quiet()).unwrap();
    assert_eq!(st.plan(there).unwrap().ops, want, "unchanged reality");

    // Reality now shows the remote mailbox: nothing left to do.
    let mut obs = quiet();
    obs.nodes[2].1.recipient = Some(ObservedRecipient {
        remote_recipient_type: Some(1),
        display_type: Some(-2_147_483_642),
        type_details: Some(2_147_483_648),
        target_address: Some("dormant@tenant.mail.onmicrosoft.com".into()),
    });
    st.observe("lab", 2, obs).unwrap();
    assert!(st.plan(there).unwrap().ops.is_empty(), "converged");
}

// A node created with a recipient carries it in its own row, and is an
// owner by it; deleting a node drops its recipient override.
#[test]
fn created_and_deleted_nodes_keep_their_own_recipient() {
    let (mut st, g0) = store(quiet());
    let shared = shared_recipient(&mut st);
    let alias = st.intern("alice@x.test");
    let state = NodeState {
        kind: NodeKind::User,
        active: Some(false),
        upn: None,
        primary_smtp: Some(alias),
        dn: None,
        recipient: Some(shared),
    };
    let v = sim(
        &mut st,
        g0,
        vec![Change::CreateNode {
            node: g(0x77),
            state: state.clone(),
        }],
    )
    .unwrap();
    assert_eq!(
        st.view(v).unwrap().node_state(&g(0x77)).unwrap().recipient,
        Some(shared)
    );
    assert_eq!(smtp_owners(&mut st, v), vec![vec![g(0x77), g(ALICE)]]);

    let moved = sim(
        &mut st,
        g0,
        vec![Change::SetRecipient {
            node: g(SHARED),
            from: Some(shared),
            to: Some(Recipient::NotMailEnabled),
        }],
    )
    .unwrap();
    let before = st.view(moved).unwrap().node_state(&g(SHARED)).unwrap();
    let gone = sim(
        &mut st,
        moved,
        vec![Change::DeleteNode {
            node: g(SHARED),
            state: before,
        }],
    )
    .unwrap();
    assert!(!st.view(gone).unwrap().exists(&g(SHARED)));
    assert!(validate(&st.view(gone).unwrap()).is_empty());
}

// Ingest: an AD entry's triplet and targetAddress reach node_state through
// ogar-ad and from_ad, with the SMTP: prefix stripped. A record encoded
// before the triplet existed (schema 1) has not read it.
#[test]
fn from_ad_reads_the_triplet_and_strips_smtp() {
    use ogar_dir_core::{DirRecord, OuDictionary, SchemaFamily, SchemaId, ValuePool};
    let ldif = "dn: CN=Team,OU=Shared,DC=example,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAQ==\nobjectClass: user\nuserAccountControl: 514\nproxyAddresses: SMTP:team@x.test\nmsExchRemoteRecipientType: 100\nmsExchRecipientDisplayType: -2147483642\nmsExchRecipientTypeDetails: 34359738368\ntargetAddress: SMTP:team@tenant.mail.onmicrosoft.com\n";
    let (mut d, mut p) = (OuDictionary::new(), ValuePool::new());
    let rec = ogar_ad::encode(
        &ogar_ad::ldif::parse(ldif).unwrap()[0],
        SCOPE.0,
        &mut d,
        &mut p,
        0,
    )
    .unwrap()
    .record;
    let old = DirRecord::new(
        SchemaId {
            family: SchemaFamily::AdDs,
            version: 1,
        },
        ogar_ad::AdKind::User as u16,
        Guid128([0x0D; 16]),
        SCOPE.0,
        0,
    );
    let obs = observe::from_ad(SCOPE, &[rec, old], &p).unwrap();
    let node = &obs
        .nodes
        .iter()
        .find(|(g, _)| *g == rec.node_guid())
        .unwrap()
        .1;
    assert_eq!(node.active, Some(false));
    assert_eq!(
        node.recipient,
        Some(ObservedRecipient {
            remote_recipient_type: Some(100),
            display_type: Some(-2_147_483_642),
            type_details: Some(34_359_738_368),
            target_address: Some("team@tenant.mail.onmicrosoft.com".into()),
        })
    );
    let old_node = &obs
        .nodes
        .iter()
        .find(|(g, _)| *g == Guid128([0x0D; 16]))
        .unwrap()
        .1;
    assert_eq!(old_node.recipient, None);

    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, obs).unwrap();
    let r = st
        .view(v)
        .unwrap()
        .node_state(&rec.node_guid())
        .unwrap()
        .recipient;
    assert!(
        matches!(r, Some(Recipient::RemoteMailbox(m)) if m.kind() == RemoteKind::Shared),
        "{r:?}"
    );
}
