//! The recipient is identified by its `PrimarySmtpAddress`; after
//! provisioning the mailbox is identified by its `ExchangeGuid`, and
//! `ExternalDirectoryObjectId` links it to its user, the Entra object
//! (formerly the MsolUser). Exchange identity is read by the object's GUID.
//! `mail` is a property on the user's business card, like the telephone
//! number: kept as written, it follows the user rather than the mailbox, and
//! is neither identity nor a receiving address nor provisioned.

use lance_graph_dir_sim::validate::address_owner;
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::exchange::{display_type, type_details};
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const A: u8 = 0xA1;
const ADMIN: u8 = 0xB2;
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

/// An account that is not mail-enabled, carrying a `mail` label only.
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

// A's identity is read by A's GUID: the mailbox's ExchangeGuid, which is
// not A's objectGUID, with A's recipient types and primary address.
#[test]
fn identity_is_read_by_guid() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    let id = view.exchange_identity(&g(A)).unwrap();
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

// The admin's `mail` names A's mailbox (a password-reset target). It hydrates
// nothing: the admin's identity is its own, without A's ExchangeGuid,
// recipient type or address, and A still holds the address.
#[test]
fn mail_hydrates_no_identity() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    let admin = view.exchange_identity(&g(ADMIN)).unwrap();
    assert_eq!(admin.node, g(ADMIN));
    assert_eq!(admin.exchange_guid, None);
    assert_eq!(admin.primary_smtp, None);
    assert_eq!(admin.recipient, Some(Recipient::NotMailEnabled));
    assert_ne!(Some(admin), view.exchange_identity(&g(A)));
    let k = st.dicts().key_lookup("a@example.org").unwrap();
    assert_eq!(address_owner(&view, k).unwrap(), Some(g(A)));
}

// No mailbox, no ExchangeGuid; an object that does not exist has no identity.
#[test]
fn no_mailbox_no_exchange_guid() {
    let (st, v) = world();
    let view = st.view(v).unwrap();
    assert_eq!(view.exchange_guid(&g(PLAIN)), None);
    assert_eq!(view.exchange_identity(&g(0x99)), None);
}

// The business-card property is kept as written, on the user whether or not
// it is mail-enabled, and when it names another user's address. Reading it
// changes nothing: the user carrying it is not the address's owner, and the
// address's owner and recipient are unchanged.
#[test]
fn mail_is_a_business_card_property_as_written() {
    let mut card = ObservedNode::user("p.upn@example.org", "p@example.org");
    card.mail = Some("Pat.Example@Example.ORG".into());
    let (st, v) = {
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
                        (g(ADMIN), labelled("admin@example.org", "a@example.org")),
                        (g(PLAIN), card),
                    ],
                    members: vec![],
                },
            )
            .unwrap();
        (st, v)
    };
    let view = st.view(v).unwrap();
    let text = |g: Guid128| view.mail(&g).and_then(|m| st.dicts().value(m));
    assert_eq!(text(g(PLAIN)), Some("Pat.Example@Example.ORG"));
    assert_eq!(text(g(ADMIN)), Some("a@example.org"));
    assert_eq!(text(g(A)), Some("a@example.org"));
    assert_eq!(view.mail(&g(0x99)), None);
    // Nobody holds the mail text as an address; the admin's mail does
    // not make it A's co-holder.
    let k = st.dicts().key_lookup("Pat.Example@Example.ORG").unwrap();
    assert_eq!(address_owner(&view, k).unwrap(), None);
    let a = st.dicts().key_lookup("a@example.org").unwrap();
    assert_eq!(address_owner(&view, a).unwrap(), Some(g(A)));
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
