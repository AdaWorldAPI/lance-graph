//! One address space (OGAR `Violation::AddressConflict`) and the routing
//! address rules (`RoutingMismatch`, `RoutingNotInProxies`).
//!
//! AD: `proxyAddresses` = `SMTP:{primary}`, `smtp:{secondary}`,
//! `smtp:{alias}@{tenant}.mail.onmicrosoft.com`; `targetAddress` =
//! `SMTP:{alias}@{tenant}.mail.onmicrosoft.com`; the alias is `mailNickname`.

use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const ALICE: u8 = 0xA1;
const BOB: u8 = 0xB0;
const ADMIN: u8 = 0xAD;
const TENANT: &str = "contoso.mail.onmicrosoft.com";

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
/// An enabled user `{name}@x.test` with UPN `{name}.upn@x.test`, recipient
/// not read.
fn user(name: &str) -> ObservedNode {
    ObservedNode::user(&format!("{name}.upn@x.test"), &format!("{name}@x.test"))
}
/// A migrated remote user mailbox: alias `alias`, routing address
/// `routing`, with the routing proxy present when `proxied`.
fn mailbox(name: &str, alias: &str, routing: &str, proxied: bool) -> ObservedNode {
    let mut u = user(name);
    u.alias = Some(alias.into());
    if proxied {
        u.proxies.push(format!("smtp:{routing}"));
    }
    u.recipient = Some(ObservedRecipient {
        remote_recipient_type: Some(4),
        display_type: Some(-2_147_483_642),
        type_details: Some(2_147_483_648),
        target_address: Some(routing.into()),
    });
    u
}
fn validate_nodes(nodes: Vec<(Guid128, ObservedNode)>) -> (VersionStore, Vec<Violation>) {
    let mut st = VersionStore::new();
    let v = st
        .observe(
            "lab",
            0,
            Observation {
                scope: SCOPE,
                nodes,
                members: vec![],
            },
        )
        .unwrap();
    let out = st.validate(v).unwrap();
    (st, out)
}
fn key(st: &mut VersionStore, s: &str) -> KeyId {
    let v = st.intern(s);
    st.key_of(v).unwrap()
}
fn conflicts(out: &[Violation]) -> Vec<&Violation> {
    out.iter()
        .filter(|v| matches!(v, Violation::AddressConflict { .. }))
        .collect()
}

// A well-formed remote mailbox is clean: its routing address is its own
// alias at the tenant, and it owns it as a proxy.
#[test]
fn a_well_formed_mailbox_is_clean() {
    let (_, out) = validate_nodes(vec![(
        g(ALICE),
        mailbox("alice", "alice", &format!("alice@{TENANT}"), true),
    )]);
    assert!(out.is_empty(), "{out:?}");
}

// Rule 1: the routing address is {alias}@{tenant}.mail.onmicrosoft.com for
// the node's own mailNickname — another alias, or not the template at all,
// is a mismatch. Alias comparison is case-insensitive.
#[test]
fn the_routing_address_must_be_the_nodes_own_alias() {
    for (alias, routing, bad) in [
        ("Alice", format!("alice@{TENANT}"), false),
        // Addresses compare case-insensitively, the routing domain included.
        (
            "alice",
            "Alice@Contoso.MAIL.onmicrosoft.com".to_string(),
            false,
        ),
        ("alice", format!("bob@{TENANT}"), true),
        ("alice", "alice@contoso.onmicrosoft.com".to_string(), true),
        ("alice", "alice@x.test".to_string(), true),
    ] {
        let (_, out) = validate_nodes(vec![(g(ALICE), mailbox("alice", alias, &routing, true))]);
        assert_eq!(
            out.contains(&Violation::RoutingMismatch { node: g(ALICE) }),
            bad,
            "{alias} {routing}: {out:?}"
        );
    }
}

// Rule 3: the mailbox must own its routing address as an SMTP proxy.
#[test]
fn the_routing_address_must_be_a_proxy() {
    let (_, out) = validate_nodes(vec![(
        g(ALICE),
        mailbox("alice", "alice", &format!("alice@{TENANT}"), false),
    )]);
    assert_eq!(out, vec![Violation::RoutingNotInProxies { node: g(ALICE) }]);
}

// Rule 4: another user's SMTP that is someone's UPN is a conflict; so is an
// address that is someone's routing address.
#[test]
fn upn_smtp_and_routing_share_one_space() {
    let mut bob = user("bob");
    bob.proxies = vec!["smtp:alice.upn@x.test".into()];
    let (mut st, out) = validate_nodes(vec![(g(ALICE), user("alice")), (g(BOB), bob)]);
    let k = key(&mut st, "alice.upn@x.test");
    assert_eq!(
        conflicts(&out),
        vec![&Violation::AddressConflict {
            key: k,
            holders: vec![
                (g(ALICE), AddressRole::Upn),
                (g(BOB), AddressRole::SecondarySmtp)
            ],
        }]
    );

    // Bob holds Alice's routing address as a proxy; Alice routes to it but
    // does not hold it as a proxy herself (else it is DuplicateSmtp).
    let routing = format!("alice@{TENANT}");
    let mut bob = user("bob");
    bob.proxies = vec![format!("smtp:{routing}")];
    let (mut st, out) = validate_nodes(vec![
        (g(ALICE), mailbox("alice", "alice", &routing, false)),
        (g(BOB), bob),
    ]);
    assert!(out.contains(&Violation::RoutingNotInProxies { node: g(ALICE) }));
    let k = key(&mut st, &routing);
    let c = conflicts(&out);
    assert_eq!(c.len(), 1, "{out:?}");
    assert!(matches!(c[0], Violation::AddressConflict { key, holders }
        if *key == k && holders.contains(&(g(ALICE), AddressRole::Routing))
            && holders.contains(&(g(BOB), AddressRole::SecondarySmtp))));
}

// Rule 5: a `mail` held by another object conflicts — the admin account
// whose mail points at Alice's mailbox as a password-reset target. It
// counts even though the admin account is disabled and no recipient.
#[test]
fn a_mail_on_another_object_conflicts_even_when_disabled() {
    let mut admin = ObservedNode::user("adm-alice@x.test", "adm-alice@x.test");
    admin.primary_smtp = None;
    admin.active = Some(false);
    admin.mail = Some("alice@x.test".into());
    let (mut st, out) = validate_nodes(vec![(g(ALICE), user("alice")), (g(ADMIN), admin)]);
    let k = key(&mut st, "alice@x.test");
    assert_eq!(
        conflicts(&out),
        vec![&Violation::AddressConflict {
            key: k,
            holders: vec![
                (g(ALICE), AddressRole::PrimarySmtp),
                (g(ADMIN), AddressRole::Mail)
            ],
        }]
    );
}

// One object holding an address under several attributes is not a
// conflict (UPN = mail = primary is the common case); a pure SMTP or pure
// UPN collision stays DuplicateSmtp / DuplicateUpn only.
#[test]
fn one_holder_or_one_attribute_is_not_an_address_conflict() {
    let mut alice = ObservedNode::user("alice@x.test", "alice@x.test");
    alice.mail = Some("Alice@x.test".into());
    let (_, out) = validate_nodes(vec![(g(ALICE), alice)]);
    assert!(out.is_empty(), "{out:?}");

    let mut bob = user("bob");
    bob.proxies = vec!["smtp:alice@x.test".into()];
    let (_, out) = validate_nodes(vec![(g(ALICE), user("alice")), (g(BOB), bob)]);
    assert!(conflicts(&out).is_empty(), "{out:?}");
    assert!(out
        .iter()
        .any(|v| matches!(v, Violation::DuplicateSmtp { .. })));
}

// The owner gate of the other roles is unchanged: a disabled non-recipient
// user's UPN does not count; its mail does.
#[test]
fn only_mail_ignores_the_owner_gate() {
    let mut ghost = user("ghost");
    ghost.active = Some(false);
    ghost.upn = Some("alice@x.test".into());
    let (_, out) = validate_nodes(vec![(g(ALICE), user("alice")), (g(BOB), ghost)]);
    assert!(conflicts(&out).is_empty(), "{out:?}");
}

// Ingest: from_ad reads mail and mailNickname.
#[test]
fn from_ad_reads_mail_and_alias() {
    use ogar_dir_core::{OuDictionary, ValuePool};
    let ldif = "dn: CN=A,OU=Staff,DC=example,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAQ==\nobjectClass: user\nmail: alice@x.test\nmailNickname: alice\n";
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
    let obs = observe::from_ad(SCOPE, &[rec], &p).unwrap();
    assert_eq!(obs.nodes[0].1.mail.as_deref(), Some("alice@x.test"));
    assert_eq!(obs.nodes[0].1.alias.as_deref(), Some("alice"));
}
