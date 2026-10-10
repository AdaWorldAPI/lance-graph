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
const CAROL: u8 = 0xCA;
const DAVE: u8 = 0xDA;
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

// Rule 5: `mail` is a label, not a claim. Exchange enforces address
// uniqueness on proxy addresses (and the UPN namespace), never on `mail`:
// the admin account whose `mail` points at Alice's mailbox as a
// password-reset target holds nothing, enabled or not.
#[test]
fn a_mail_on_another_object_is_a_label_not_a_claim() {
    for active in [Some(false), Some(true)] {
        let mut admin = ObservedNode::user("adm-alice@x.test", "adm-alice@x.test");
        admin.primary_smtp = None;
        admin.active = active;
        admin.mail = Some("alice@x.test".into());
        let mut st = VersionStore::new();
        let v = st
            .observe(
                "lab",
                0,
                Observation {
                    scope: SCOPE,
                    nodes: vec![(g(ALICE), user("alice")), (g(ADMIN), admin)],
                    members: vec![],
                },
            )
            .unwrap();
        let out = st.validate(v).unwrap();
        assert!(conflicts(&out).is_empty(), "{active:?}: {out:?}");
        let k = key(&mut st, "alice@x.test");
        assert_eq!(
            lance_graph_dir_sim::validate::address_owner(&st.view(v).unwrap(), k),
            Ok(Some(g(ALICE))),
            "{active:?}: Alice alone holds her address"
        );
    }
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

// The owner gate: a disabled non-recipient user's UPN does not count.
#[test]
fn a_disabled_non_recipients_upn_is_not_a_claim() {
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

/// Enable a remote user mailbox routing to `routing` on Carol, an enabled
/// user with alias `carol` that is observed as not mail-enabled and owns no
/// routing proxy; `others` are observed beside her. Returns the version's
/// violations.
fn enable_carol(routing: &str, others: Vec<(Guid128, ObservedNode)>) -> Vec<Violation> {
    let mut carol = user("carol");
    carol.alias = Some("carol".into());
    carol.recipient = Some(ObservedRecipient {
        remote_recipient_type: None,
        display_type: None,
        type_details: None,
        target_address: None,
    });
    let mut nodes = vec![(g(CAROL), carol)];
    nodes.extend(others);
    let mut st = VersionStore::new();
    let v0 = st
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
    let out = st.validate(v0).unwrap();
    assert!(out.is_empty(), "observed state is clean: {out:?}");
    let routing = st.intern(routing);
    let enabled = RemoteMailboxOp::Enable {
        kind: RemoteKind::User,
        routing,
    }
    .apply(&Recipient::NotMailEnabled)
    .unwrap();
    let v1 = st
        .simulate(
            v0,
            &Propose(vec![Change::SetRecipient {
                node: g(CAROL),
                from: Some(Recipient::NotMailEnabled),
                to: Some(enabled),
            }]),
            &[],
        )
        .unwrap();
    st.validate(v1).unwrap()
}

// Codex P1 on #1392: Enable-RemoteMailbox stamps the routing address as a
// proxy itself, so a version that enables a mailbox is valid before AD holds
// that proxy. Only an observed routing address must already be a proxy
// (`the_routing_address_must_be_a_proxy`).
#[test]
fn enabling_a_mailbox_stamps_its_routing_proxy() {
    assert_eq!(enable_carol(&format!("carol@{TENANT}"), vec![]), vec![]);
}

// The introduced routing address is still checked: it must be the template
// for the node's own alias ...
#[test]
fn an_introduced_routing_address_must_match_the_alias() {
    assert_eq!(
        enable_carol(&format!("dave@{TENANT}"), vec![]),
        vec![Violation::RoutingMismatch { node: g(CAROL) }]
    );
}

// ... and it is in the address space: an introduced routing address that is
// another user's SMTP address conflicts, as an observed one would.
#[test]
fn an_introduced_routing_address_conflicts_like_an_observed_one() {
    let routing = format!("carol@{TENANT}");
    let mut dave = user("dave");
    dave.proxies.push(format!("smtp:{routing}"));
    let out = enable_carol(&routing, vec![(g(DAVE), dave)]);
    assert!(
        out.iter().any(|v| matches!(v,
            Violation::AddressConflict { holders, .. }
                if holders.contains(&(g(CAROL), AddressRole::Routing))
                    && holders.contains(&(g(DAVE), AddressRole::SecondarySmtp)))),
        "{out:?}"
    );
    assert!(
        !out.contains(&Violation::RoutingNotInProxies { node: g(CAROL) }),
        "{out:?}"
    );
}

/// An enabled account that is no longer mail-enabled, still carrying its
/// old primary SMTP and a secondary proxy.
fn deprovisioned(name: &str, smtp: &str, proxy: &str) -> ObservedNode {
    let mut u = ObservedNode::user(&format!("{name}.upn@x.test"), smtp);
    u.proxies.push(format!("smtp:{proxy}"));
    u.recipient = Some(ObservedRecipient::default());
    u
}

// Provisioning follows the recipient type: a non-mail-enabled account's
// leftover SMTP and proxies reserve nothing, so a mailbox now holding the
// same addresses is not in conflict; its UPN is still its own and still
// collides. The paired half — the same leftovers on a mail-enabled account
// — must conflict, or the first assertion proves nothing.
#[test]
fn leftover_mail_addresses_of_a_non_recipient_claim_nothing_but_its_upn_does() {
    let mut taken = user("carol");
    taken.proxies.push("smtp:old@x.test".into());
    let (_, out) = validate_nodes(vec![
        (g(CAROL), taken),
        (g(DAVE), deprovisioned("dave", "carol@x.test", "old@x.test")),
    ]);
    assert!(out.is_empty(), "{out:?}");

    // Same UPN: the UPN plane is not gated on the recipient type.
    let mut upn_clash = deprovisioned("dave", "dave@x.test", "dave2@x.test");
    upn_clash.upn = Some("carol.upn@x.test".into());
    let (_, out) = validate_nodes(vec![(g(CAROL), user("carol")), (g(DAVE), upn_clash)]);
    assert!(
        out.iter()
            .any(|v| matches!(v, Violation::DuplicateUpn { .. })),
        "the UPN is still claimed: {out:?}"
    );

    // Paired: the same leftovers on a mail-enabled account conflict.
    let mut live = mailbox("dave", "dave", &format!("dave@{TENANT}"), true);
    live.primary_smtp = Some("carol@x.test".into());
    live.proxies.push("smtp:old@x.test".into());
    let mut taken = user("carol");
    taken.proxies.push("smtp:old@x.test".into());
    let (_, out) = validate_nodes(vec![(g(CAROL), taken), (g(DAVE), live)]);
    let clashes: Vec<_> = out
        .iter()
        .filter(|v| matches!(v, Violation::DuplicateSmtp { .. }))
        .collect();
    assert_eq!(clashes.len(), 2, "primary and proxy both clash: {out:?}");
}
