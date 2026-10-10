//! `mail` is a label, not a claim. A label is the trigger for one lookup:
//! its address is resolved to the object that holds it, and that object's
//! Exchange identity (recipient types, `ExchangeGuid`, primary SMTP) is read
//! by GUID. The label never changes who owns or receives at the address.

use lance_graph_dir_sim::validate::{address_owner, mail_label};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::exchange::{display_type, type_details};
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const A: u8 = 0xA1;
const ADMIN: u8 = 0xB2;
const STALE: u8 = 0xC3;
const D1: u8 = 0xD1;
const D2: u8 = 0xD2;
const F: u8 = 0xE6;
const PLAIN: u8 = 0xF7;
/// A's mailbox: deliberately not A's objectGUID.
const A_MAILBOX: Guid128 = Guid128([0x4B; 16]);

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}

/// An on-premises mailbox at `smtp`, with its `ExchangeGuid`.
fn mailbox(upn: &str, smtp: &str, exchange_guid: Option<Guid128>) -> ObservedNode {
    let mut u = ObservedNode::user(upn, smtp);
    u.mail = Some(smtp.into());
    u.exchange_guid = exchange_guid;
    u.recipient = Some(ObservedRecipient {
        remote_recipient_type: None,
        display_type: Some(display_type::MAILBOX_USER),
        type_details: Some(type_details::USER_MAILBOX),
        target_address: None,
    });
    u
}

/// An account that is not mail-enabled, carrying `mail` as a label only.
fn labelled(upn: &str, label: &str) -> ObservedNode {
    let mut u = ObservedNode::user(upn, upn);
    u.primary_smtp = None;
    u.mail = Some(label.into());
    u.recipient = Some(ObservedRecipient::default());
    u
}

fn world() -> (VersionStore, VersionId) {
    let mut st = VersionStore::new();
    let v = st
        .observe(
            "lab",
            0,
            Observation {
                scope: SCOPE,
                nodes: vec![
                    (
                        g(A),
                        mailbox("a.upn@example.org", "a@example.org", Some(A_MAILBOX)),
                    ),
                    // The admin account's `mail` is A's mailbox (a
                    // password-reset target).
                    (g(ADMIN), labelled("admin@example.org", "a@example.org")),
                    (g(STALE), labelled("stale@example.org", "gone@example.org")),
                    (g(D1), mailbox("d1@example.org", "dup@example.org", None)),
                    (g(D2), mailbox("d2@example.org", "dup@example.org", None)),
                    (g(F), labelled("f@example.org", "dup@example.org")),
                    (
                        g(PLAIN),
                        ObservedNode::user("plain@example.org", "plain@example.org"),
                    ),
                ],
                members: vec![],
            },
        )
        .unwrap();
    (st, v)
}

fn key(st: &VersionStore, s: &str) -> KeyId {
    st.dicts().key_lookup(s).unwrap()
}

// The label is A's own primary address: A's identity, including the
// mailbox's ExchangeGuid, which is not A's objectGUID.
#[test]
fn an_own_label_hydrates_the_objects_exchange_identity() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    let Some(MailLabel::Own(id)) = mail_label(&view, &g(A)) else {
        panic!("A's label is its own address");
    };
    assert_eq!(id.node, g(A));
    assert_eq!(id.exchange_guid, Some(A_MAILBOX));
    assert_ne!(id.exchange_guid, Some(id.node));
    assert!(matches!(
        id.recipient,
        Some(Recipient::OnPremisesMailbox { .. })
    ));
    assert_eq!(
        id.primary_smtp.and_then(|p| st.dicts().value(p)),
        Some("a@example.org")
    );
}

// The admin's label points at A's mailbox: the identity hydrated is A's,
// found through the address once and read by A's GUID. A still owns the
// address; the label claims nothing.
#[test]
fn a_label_elsewhere_hydrates_the_holder_not_the_labelled_object() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    let k = key(&st, "a@example.org");
    assert_eq!(
        mail_label(&view, &g(ADMIN)),
        Some(MailLabel::Elsewhere {
            key: k,
            holder: view.exchange_identity(&g(A)).unwrap(),
        })
    );
    assert_eq!(address_owner(&view, k).unwrap(), Some(g(A)));
}

// Nobody holds the address the label names: a stale label, nothing hydrated.
#[test]
fn a_label_nobody_holds_is_unheld() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    assert_eq!(
        mail_label(&view, &g(STALE)),
        Some(MailLabel::Unheld {
            key: key(&st, "gone@example.org")
        })
    );
}

// Two objects hold the address: none is chosen, both are listed, and the
// labelling object is not among them.
#[test]
fn a_contested_label_hydrates_no_one() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    assert_eq!(
        mail_label(&view, &g(F)),
        Some(MailLabel::Contested {
            key: key(&st, "dup@example.org"),
            holders: vec![g(D1), g(D2)],
        })
    );
}

// No `mail` value, no label, no lookup; and an object that does not exist
// has none either.
#[test]
fn no_label_no_lookup() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    assert_eq!(mail_label(&view, &g(PLAIN)), None);
    assert_eq!(mail_label(&view, &g(0x99)), None);
    assert_eq!(view.exchange_guid(&g(PLAIN)), None);
}

// Ingest: `msExchMailboxGuid` reaches the node as an id from a schema-6
// record; a record encoded before schema 6 has not read it.
#[test]
fn from_ad_reads_the_exchange_guid_from_schema_6_only() {
    use ogar_dir_core::{DirRecord, OuDictionary, SchemaFamily, SchemaId, ValuePool};
    const BOX: &str = "Ab5xLqKmTkiZ8gH0cDeFqw==";
    let ldif = format!(
        "dn: CN=A,OU=Staff,DC=example,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAQ==\nobjectClass: user\nuserAccountControl: 512\nmail: a@example.org\nmsExchMailboxGuid:: {BOX}\n"
    );
    let (mut d, mut p) = (OuDictionary::new(), ValuePool::new());
    let rec = ogar_ad::encode(
        &ogar_ad::ldif::parse(&ldif).unwrap()[0],
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
            version: 5,
        },
        ogar_ad::AdKind::User as u16,
        Guid128([0x0D; 16]),
        SCOPE.0,
        0,
    );
    let obs = observe::from_ad(SCOPE, &[rec, old], &p).unwrap();
    let node = |id: Guid128| &obs.nodes.iter().find(|(x, _)| *x == id).unwrap().1;
    let raw = ogar_dir_core::base64::decode(BOX).unwrap();
    assert_eq!(
        node(rec.node_guid()).exchange_guid,
        Some(Guid128::from_ms_bytes(&raw).unwrap())
    );
    assert_eq!(node(rec.node_guid()).mail.as_deref(), Some("a@example.org"));
    assert_eq!(node(Guid128([0x0D; 16])).exchange_guid, None);
}
