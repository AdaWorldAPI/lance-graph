//! A remote mailbox receives only when its mailbox exists in Exchange
//! Online, and that is decided by OGAR's hybrid correspondence fold on
//! GUIDs (anchor, backsync, Entra id), never on an address.
//!
//! The fold's lanes are built by hand: these tests exercise how
//! [`CloudMailboxes`] reads a completed fold, not the record encoders
//! (`ogar-az`'s correspondence tests cover those).

use lance_graph_dir_sim::validate::{address_owner, address_recipient, address_recipient_in};
use lance_graph_dir_sim::*;
use ogar_dir_core::correspond::{fold, state, Index, Lanes, Output};
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::exchange::{display_type, type_details};
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const D: &str = "d@example.org";
const ROUTING: &str = "d@tenant.mail.onmicrosoft.com";

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}

/// The AD user.
const USER: u8 = 0x01;
/// Its source anchor (`mS-DS-ConsistencyGuid`).
const ANCHOR: u8 = 0x0A;
/// Its Entra object.
const ENTRA: u8 = 0x0E;
/// Another Entra object.
const OTHER_ENTRA: u8 = 0x0F;

/// The AD user holding `D`: a remote user mailbox (code 4) or an
/// on-premises mailbox.
fn user(remote: bool) -> ObservedNode {
    let mut u = ObservedNode::user("d.upn@example.org", D);
    u.alias = Some("d".into());
    u.recipient = Some(if remote {
        u.proxies = vec![format!("smtp:{ROUTING}")];
        ObservedRecipient {
            remote_recipient_type: Some(4),
            display_type: Some(display_type::REMOTE_USER_MAILBOX),
            type_details: Some(type_details::REMOTE_USER_MAILBOX),
            target_address: Some(ROUTING.into()),
        }
    } else {
        ObservedRecipient {
            remote_recipient_type: None,
            display_type: Some(display_type::MAILBOX_USER),
            type_details: Some(type_details::USER_MAILBOX),
            target_address: None,
        }
    });
    u
}

fn observe(node: ObservedNode) -> (VersionStore, VersionId) {
    let mut st = VersionStore::new();
    let v = st
        .observe(
            "lab",
            0,
            Observation {
                scope: SCOPE,
                nodes: vec![(g(USER), node)],
                members: vec![],
            },
        )
        .unwrap();
    (st, v)
}

/// One present id cell.
fn id(col: &mut ogar_dir_core::correspond::IdColumn, v: Option<Guid128>) {
    match v {
        Some(g) => {
            col.id.push(g);
            col.state.push(state::PRESENT);
        }
        None => {
            col.id.push(Guid128::NIL);
            col.state.push(state::ABSENT);
        }
    }
}

/// The hybrid world, all in one bound scope (ordinal 0) at one time:
/// the AD user with its anchor and backsync, the Entra objects as
/// `(id, immutable id)`, and the Exchange Online mailboxes by
/// `ExternalDirectoryObjectId`.
fn cloud(
    backsync: Option<Guid128>,
    entra: &[(Guid128, Guid128)],
    exo: &[Guid128],
) -> CloudMailboxes {
    let mut l = Lanes {
        max_skew_ms: 0,
        ..Lanes::default()
    };
    let row = |r: &mut ogar_dir_core::correspond::Rows, owner: Guid128| {
        r.owner.push(owner);
        r.scope.push(0);
        r.at.push(1_000);
    };
    row(&mut l.ad, g(USER));
    id(&mut l.ad_anchor, Some(g(ANCHOR)));
    id(&mut l.ad_backsync, backsync);
    for &(e, immutable) in entra {
        row(&mut l.entra, e);
        id(&mut l.entra_anchor, Some(immutable));
    }
    for (n, &external) in exo.iter().enumerate() {
        row(&mut l.exo, g(0xE0 + n as u8));
        id(&mut l.exo_external, Some(external));
    }
    let ix = Index::build(&l);
    let mut out = Output::for_index(&l, &ix);
    fold(&l, &ix, &mut out);
    CloudMailboxes::from_fold(&l, &ix, &out)
}

/// (owner, AD-only recipient, recipient with the cloud observed) of `D`.
fn answers(
    st: &VersionStore,
    v: VersionId,
    c: &CloudMailboxes,
) -> (Option<Guid128>, Option<Guid128>, Option<Guid128>) {
    let key = st.dicts().key_lookup(D).unwrap();
    let view = st.view(v).unwrap();
    (
        address_owner(&view, key).unwrap(),
        address_recipient(&view, key).unwrap(),
        address_recipient_in(&view, key, c).unwrap(),
    )
}

// The whole chain holds: anchor -> Entra, backsync -> Entra, Entra -> one
// Exchange Online mailbox. The remote mailbox receives.
#[test]
fn a_remote_mailbox_with_its_cloud_mailbox_receives() {
    let c = cloud(Some(g(ENTRA)), &[(g(ENTRA), g(ANCHOR))], &[g(ENTRA)]);
    assert!(c.contains(&g(USER)));
    let (st, v) = observe(user(true));
    assert_eq!(
        answers(&st, v, &c),
        (Some(g(USER)), Some(g(USER)), Some(g(USER)))
    );
    assert!(c.delivers_to(&st.view(v).unwrap(), &g(USER)));
}

// The AD object says remote mailbox, but Exchange Online has none for its
// Entra object (provisioning not done, or the mailbox removed): the
// address is still held on-premises, and mail to it is delivered nowhere.
#[test]
fn a_remote_mailbox_without_a_cloud_mailbox_receives_nothing() {
    let c = cloud(Some(g(ENTRA)), &[(g(ENTRA), g(ANCHOR))], &[]);
    assert!(!c.contains(&g(USER)));
    let (st, v) = observe(user(true));
    assert_eq!(answers(&st, v, &c), (Some(g(USER)), Some(g(USER)), None));
    assert!(!c.delivers_to(&st.view(v).unwrap(), &g(USER)));
}

// The forward anchor alone suffices when there is no backsync to confirm.
#[test]
fn the_forward_anchor_alone_ties_the_mailbox() {
    let c = cloud(None, &[(g(ENTRA), g(ANCHOR))], &[g(ENTRA)]);
    assert!(c.contains(&g(USER)));
}

// Two Exchange Online mailboxes for one Entra object are no mailbox: the
// fold never picks one.
#[test]
fn two_cloud_mailboxes_are_not_one() {
    let c = cloud(
        Some(g(ENTRA)),
        &[(g(ENTRA), g(ANCHOR))],
        &[g(ENTRA), g(ENTRA)],
    );
    assert!(!c.contains(&g(USER)));
    let (st, v) = observe(user(true));
    assert_eq!(answers(&st, v, &c).2, None);
}

// The witnesses contradict: the anchor reaches one Entra object, backsync
// names another. The mailbox the anchor's object has does not count.
#[test]
fn contradicting_witnesses_are_not_a_mailbox() {
    let c = cloud(
        Some(g(OTHER_ENTRA)),
        &[(g(ENTRA), g(ANCHOR)), (g(OTHER_ENTRA), g(0x77))],
        &[g(ENTRA)],
    );
    assert!(!c.contains(&g(USER)));
}

// A mailbox for some other Entra object is not this user's mailbox.
#[test]
fn another_objects_mailbox_is_not_this_one() {
    let c = cloud(Some(g(ENTRA)), &[(g(ENTRA), g(ANCHOR))], &[g(OTHER_ENTRA)]);
    assert!(!c.contains(&g(USER)));
}

// An on-premises mailbox is delivered on-premises: the cloud observation
// does not apply to it.
#[test]
fn an_on_premises_mailbox_needs_no_cloud_mailbox() {
    let c = cloud(Some(g(ENTRA)), &[(g(ENTRA), g(ANCHOR))], &[]);
    let (st, v) = observe(user(false));
    let s = st.view(v).unwrap().node_state(&g(USER)).unwrap();
    assert!(
        matches!(s.recipient, Some(Recipient::OnPremisesMailbox { .. })),
        "fixture decodes as an on-premises mailbox: {:?}",
        s.recipient
    );
    assert_eq!(
        answers(&st, v, &c),
        (Some(g(USER)), Some(g(USER)), Some(g(USER)))
    );
    assert!(c.delivers_to(&st.view(v).unwrap(), &g(USER)));
}

// A cloud mailbox never makes a non-recipient receive: an account that is
// not mail-enabled on-premises is not delivered to, whatever Exchange
// Online still holds for its Entra object. Nor does it hold the address:
// its recipient type, not its enabled flag, decides whether its mail
// addresses are provisioned, and a non-recipient's are not.
#[test]
fn the_cloud_never_revives_a_non_recipient() {
    let c = cloud(Some(g(ENTRA)), &[(g(ENTRA), g(ANCHOR))], &[g(ENTRA)]);
    assert!(c.contains(&g(USER)));
    let mut u = user(true);
    u.recipient = Some(ObservedRecipient::default());
    let (st, v) = observe(u);
    assert!(!c.delivers_to(&st.view(v).unwrap(), &g(USER)));
    assert_eq!(answers(&st, v, &c), (None, None, None));
}

// ── ExchangeGuid: the mailbox's immutable identity, on both sides ──────────

/// The Exchange Online mailbox's `ExchangeGuid`.
const CLOUD_BOX: Guid128 = Guid128([0xC1; 16]);
/// A different `ExchangeGuid`.
const OTHER_BOX: Guid128 = Guid128([0xC2; 16]);

/// The remote mailbox with its on-premises `msExchMailboxGuid`.
fn remote_with(exchange_guid: Option<Guid128>) -> (VersionStore, VersionId) {
    let mut u = user(true);
    u.exchange_guid = exchange_guid;
    observe(u)
}

/// The full chain, with the cloud mailbox's `ExchangeGuid` read by
/// `ExternalDirectoryObjectId`.
fn chained(cloud_guids: &[(Guid128, Guid128)]) -> CloudMailboxes {
    cloud(Some(g(ENTRA)), &[(g(ENTRA), g(ANCHOR))], &[g(ENTRA)]).with_exchange_guids(cloud_guids)
}

// `Enable-RemoteMailbox` leaves `msExchMailboxGuid` empty until backsync
// fills it: mail is delivered, but the mailbox cannot be migrated.
#[test]
fn an_empty_on_premises_exchange_guid_awaits_backsync() {
    let c = chained(&[(g(ENTRA), CLOUD_BOX)]);
    let (st, v) = remote_with(None);
    let view = st.view(v).unwrap();
    let state = c.mailbox_guid(&view, &g(USER));
    assert_eq!(state, MailboxGuid::AwaitingBacksync { cloud: CLOUD_BOX });
    assert!(!state.migratable());
    assert!(c.delivers_to(&view, &g(USER)));
}

// Both sides carry the same ExchangeGuid: the only migratable state.
#[test]
fn a_matching_exchange_guid_is_migratable() {
    let c = chained(&[(g(ENTRA), CLOUD_BOX)]);
    let (st, v) = remote_with(Some(CLOUD_BOX));
    let view = st.view(v).unwrap();
    let state = c.mailbox_guid(&view, &g(USER));
    assert_eq!(state, MailboxGuid::Matched(CLOUD_BOX));
    assert!(state.migratable());
    assert!(c.delivers_to(&view, &g(USER)));
}

// A different on-premises value blocks provisioning the cloud mailbox:
// nothing is delivered and nothing can be migrated, although the chain to
// the cloud mailbox holds.
#[test]
fn a_mismatched_exchange_guid_blocks_the_cloud_mailbox() {
    let c = chained(&[(g(ENTRA), CLOUD_BOX)]);
    assert!(c.contains(&g(USER)));
    let (st, v) = remote_with(Some(OTHER_BOX));
    let view = st.view(v).unwrap();
    let state = c.mailbox_guid(&view, &g(USER));
    assert_eq!(
        state,
        MailboxGuid::Mismatch {
            on_premises: OTHER_BOX,
            cloud: CLOUD_BOX
        }
    );
    assert!(!state.migratable());
    assert!(!c.delivers_to(&view, &g(USER)));
    assert_eq!(answers(&st, v, &c), (Some(g(USER)), Some(g(USER)), None));
}

// Not read, or read ambiguously (one ExternalDirectoryObjectId, two
// values), is not a verdict: delivery stays as the fold decides, and no
// migration is claimed.
#[test]
fn an_unread_or_ambiguous_cloud_exchange_guid_decides_nothing() {
    let (st, v) = remote_with(Some(OTHER_BOX));
    let view = st.view(v).unwrap();
    for c in [
        chained(&[]),
        chained(&[(g(ENTRA), CLOUD_BOX), (g(ENTRA), OTHER_BOX)]),
        chained(&[(g(OTHER_ENTRA), CLOUD_BOX)]),
    ] {
        assert_eq!(c.mailbox_guid(&view, &g(USER)), MailboxGuid::CloudUnread);
        assert!(c.delivers_to(&view, &g(USER)));
    }
}

// The comparison applies to remote mailboxes only, and needs the cloud
// mailbox to exist.
#[test]
fn the_exchange_guid_comparison_needs_a_remote_mailbox_in_the_cloud() {
    let c = chained(&[(g(ENTRA), CLOUD_BOX)]);
    let (st, v) = observe(user(false));
    assert_eq!(
        c.mailbox_guid(&st.view(v).unwrap(), &g(USER)),
        MailboxGuid::NotRemote
    );
    let none = cloud(Some(g(ENTRA)), &[(g(ENTRA), g(ANCHOR))], &[])
        .with_exchange_guids(&[(g(ENTRA), CLOUD_BOX)]);
    let (st, v) = remote_with(Some(CLOUD_BOX));
    assert_eq!(
        none.mailbox_guid(&st.view(v).unwrap(), &g(USER)),
        MailboxGuid::NoCloudMailbox
    );
}
