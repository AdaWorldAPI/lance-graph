//! `proxyAddresses` as a relation `(owner, KeyId, meta)`, and SMTP
//! uniqueness over it: distinct owners per normalized SMTP key, counted with
//! the existing `GroupReduce Count`.

use lance_graph_dir_sim::proxy::{parse_proxy, ProxyKind};
use lance_graph_dir_sim::validate::{smtp_duplicates, validate};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::{Attribute, KeyId, RuleId, VersionId, Violation};

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const RENAME: RuleId = RuleId {
    name: "RenameMail",
    version: 1,
};

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
fn user(primary: &str, proxies: &[&str]) -> ObservedNode {
    ObservedNode {
        proxies: proxies.iter().map(|s| s.to_string()).collect(),
        ..ObservedNode::user(&format!("{primary}.upn"), primary)
    }
}
fn store(nodes: Vec<(Guid128, ObservedNode)>) -> (VersionStore, VersionId) {
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
    (st, v)
}
fn key(st: &mut VersionStore, s: &str) -> KeyId {
    let v = st.intern(s);
    st.key_of(v).unwrap()
}
fn dup(key: KeyId, owners: &[u8]) -> Violation {
    Violation::DuplicateSmtp {
        key,
        owners: owners.iter().map(|&o| g(o)).collect(),
    }
}

#[test]
fn the_prefix_names_the_kind_and_the_case_of_smtp_marks_the_primary() {
    assert_eq!(
        parse_proxy("SMTP:A@x.de"),
        (ProxyKind::Smtp, true, "A@x.de")
    );
    assert_eq!(
        parse_proxy("smtp:A@x.de"),
        (ProxyKind::Smtp, false, "A@x.de")
    );
    assert_eq!(
        parse_proxy("X500:/o=Org/cn=a"),
        (ProxyKind::X500, false, "/o=Org/cn=a")
    );
    assert_eq!(
        parse_proxy("x500:/o=Org/cn=a"),
        (ProxyKind::X500, false, "/o=Org/cn=a")
    );
    assert_eq!(parse_proxy("SIP:a@x.de"), (ProxyKind::Sip, false, "a@x.de"));
    assert_eq!(parse_proxy("sip:a@x.de"), (ProxyKind::Sip, false, "a@x.de"));
    // Unknown namespaces keep their whole text: the namespace is not
    // compared, so nothing is lost by not splitting it.
    assert_eq!(
        parse_proxy("EUM:123;phone"),
        (ProxyKind::Other, false, "EUM:123;phone")
    );
    assert_eq!(
        parse_proxy("no-colon"),
        (ProxyKind::Other, false, "no-colon")
    );
    // A mixed-case `Smtp:` is not the upper-case marker.
    assert!(!parse_proxy("Smtp:a@x.de").1);
}

#[test]
fn from_ad_keeps_the_primary_apart_and_every_other_proxy_raw() {
    use ogar_dir_core::{OuDictionary, ValuePool};
    let ldif = "dn: CN=Alice,OU=Staff,DC=example,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAQ==\nobjectClass: user\nuserPrincipalName: alice@example.test\nproxyAddresses: smtp:a@legacy.test\nproxyAddresses: SMTP:alice@example.test\nproxyAddresses: X500:/o=Org/cn=alice\nuserAccountControl: 512\n";
    let (mut d, mut p) = (OuDictionary::new(), ValuePool::new());
    let recs: Vec<_> = ogar_ad::ldif::parse(ldif)
        .unwrap()
        .iter()
        .map(|e| {
            ogar_ad::encode(e, SCOPE.0, &mut d, &mut p, 0)
                .unwrap()
                .record
        })
        .collect();
    let obs = observe::from_ad(SCOPE, &recs, &p).unwrap();
    let n = &obs.nodes[0].1;
    assert_eq!(n.primary_smtp.as_deref(), Some("alice@example.test"));
    let mut proxies = n.proxies.clone();
    proxies.sort();
    assert_eq!(proxies, vec!["X500:/o=Org/cn=alice", "smtp:a@legacy.test"]);
}

// The relation: one row per proxy, the primary included, ordered by owner.
// The exact spelling survives as a ValueId; comparison is by KeyId.
#[test]
fn the_relation_holds_every_proxy_with_its_exact_spelling() {
    let (mut st, v) = store(vec![(
        g(1),
        user(
            "Alice@X.de",
            &["smtp:ALIAS@x.de", "X500:/o=Org/cn=a", "sip:alice@x.de"],
        ),
    )]);
    let alias = key(&mut st, "alias@x.de");
    let snap = st.snapshot(v).unwrap().clone();
    let rel = snap.proxies();
    assert_eq!(rel.len(), 4);
    let rows: Vec<ProxyRow> = (0..rel.len()).map(|i| rel.row(i)).collect();
    assert!(rows.iter().all(|r| r.owner == UserOrdinal(0)));
    let kinds: Vec<_> = rows.iter().map(|r| (r.kind, r.primary)).collect();
    assert!(kinds.contains(&(ProxyKind::Smtp, true)));
    assert!(kinds.contains(&(ProxyKind::Smtp, false)));
    assert!(kinds.contains(&(ProxyKind::X500, false)));
    assert!(kinds.contains(&(ProxyKind::Sip, false)));
    let secondary = rows
        .iter()
        .find(|r| r.kind == ProxyKind::Smtp && !r.primary)
        .unwrap();
    assert_eq!(secondary.key, alias);
    assert_eq!(st.dicts().value(secondary.value), Some("ALIAS@x.de"));
}

// A secondary address of one user collides with the primary of another:
// the old primary-only check could not see it.
#[test]
fn a_secondary_colliding_with_another_users_primary_is_a_duplicate() {
    let (mut st, v) = store(vec![
        (g(1), user("alice@x.de", &[])),
        (g(2), user("bob@x.de", &["smtp:ALICE@X.DE"])),
    ]);
    let k = key(&mut st, "alice@x.de");
    assert_eq!(st.validate(v).unwrap(), vec![dup(k, &[1, 2])]);
}

// Two users sharing only secondaries are a duplicate as well.
#[test]
fn two_secondaries_on_different_owners_collide() {
    let (mut st, v) = store(vec![
        (g(1), user("a@x.de", &["smtp:team@x.de"])),
        (g(2), user("b@x.de", &["smtp:Team@X.de"])),
    ]);
    let k = key(&mut st, "team@x.de");
    assert_eq!(st.validate(v).unwrap(), vec![dup(k, &[1, 2])]);
}

// Distinct owners, not raw rows: one user holding the same address twice
// (`SMTP:` and `smtp:`, any case) is one owner, not a collision.
#[test]
fn one_owner_holding_an_address_twice_is_not_a_duplicate() {
    let (st, v) = store(vec![
        (
            g(1),
            user("Alice@x.de", &["smtp:alice@X.de", "smtp:ALICE@x.de"]),
        ),
        (g(2), user("bob@x.de", &[])),
    ]);
    assert!(smtp_duplicates(&st.view(v).unwrap()).is_empty());
    // ...but it still counts once against another owner.
    let (mut st, v) = store(vec![
        (g(1), user("Alice@x.de", &["smtp:alice@X.de"])),
        (g(2), user("bob@x.de", &["smtp:alice@x.de"])),
    ]);
    let k = key(&mut st, "alice@x.de");
    assert_eq!(st.validate(v).unwrap(), vec![dup(k, &[1, 2])]);
}

// Only SMTP takes part: equal X500 / SIP / other addresses are not an SMTP
// collision, and an SMTP address does not collide with a SIP one.
#[test]
fn other_namespaces_do_not_take_part() {
    let (st, v) = store(vec![
        (
            g(1),
            user("a@x.de", &["X500:/o=Org/cn=same", "sip:same@x.de", "EUM:1"]),
        ),
        (
            g(2),
            user("b@x.de", &["X500:/o=Org/cn=same", "SIP:same@x.de", "EUM:1"]),
        ),
        (g(3), user("same@x.de", &[])),
    ]);
    assert!(validate(&st.view(v).unwrap()).is_empty());
}

// V4: only active owners count. A disabled or unknown-status user holding an
// address is not a collision.
#[test]
fn only_active_owners_count() {
    for active in [Some(false), None] {
        let mut off = user("b@x.de", &["smtp:alice@x.de"]);
        off.active = active;
        let (st, v) = store(vec![(g(1), user("alice@x.de", &[])), (g(2), off)]);
        assert!(validate(&st.view(v).unwrap()).is_empty(), "{active:?}");
    }
}

// A SetPrimarySmtp override REPLACES the observed primary row; the owner's
// secondaries stay.
#[test]
fn an_override_replaces_the_primary_and_keeps_the_secondaries() {
    let (mut st, g0) = store(vec![
        (g(1), user("alice@x.de", &[])),
        (
            g(2),
            ObservedNode {
                upn: Some("second.upn".into()),
                ..user("alice@x.de", &["smtp:shared@x.de"])
            },
        ),
        (g(3), user("c@x.de", &["smtp:shared@x.de"])),
    ]);
    let alice = key(&mut st, "alice@x.de");
    let shared = key(&mut st, "shared@x.de");
    assert_eq!(
        st.validate(g0).unwrap(),
        vec![dup(alice, &[1, 2]), dup(shared, &[2, 3])]
    );
    let to = st.intern("bob@x.de");
    let g1 = st
        .simulate(
            g0,
            &SetPrimarySmtp {
                rule: RENAME,
                user: g(2),
                to,
            },
            &[],
        )
        .unwrap();
    // The primary collision is gone; the secondary one is not.
    assert_eq!(st.validate(g1).unwrap(), vec![dup(shared, &[2, 3])]);
}

// An override onto one of the owner's own secondaries is still one owner;
// onto another user's secondary it is a collision.
#[test]
fn an_override_counts_its_owner_once() {
    let (mut st, g0) = store(vec![
        (g(1), user("a@x.de", &["smtp:alias@x.de"])),
        (g(2), user("b@x.de", &["smtp:taken@x.de"])),
    ]);
    let own = st.intern("ALIAS@x.de");
    let g1 = st
        .simulate(
            g0,
            &SetPrimarySmtp {
                rule: RENAME,
                user: g(1),
                to: own,
            },
            &[],
        )
        .unwrap();
    assert!(st.validate(g1).unwrap().is_empty());
    let taken = st.intern("taken@x.de");
    let g2 = st
        .simulate(
            g0,
            &SetPrimarySmtp {
                rule: RENAME,
                user: g(1),
                to: taken,
            },
            &[],
        )
        .unwrap();
    let k = key(&mut st, "taken@x.de");
    assert_eq!(st.validate(g2).unwrap(), vec![dup(k, &[1, 2])]);
    assert_eq!(
        st.view(g2).unwrap().attr(&g(1), Attribute::PrimarySmtp),
        Some(taken)
    );
}

// The primary and a secondary spelling of one address on one owner: after an
// override drops the primary row, the secondary still holds the address, so
// a collision with another owner survives. (One row per owner and address
// must be the secondary, or the override would take the address with it.)
#[test]
fn an_override_does_not_take_a_secondary_of_the_same_address_with_it() {
    let (mut st, g0) = store(vec![
        (g(1), user("alice@x.de", &["smtp:ALICE@x.de"])),
        (g(2), user("b@x.de", &["smtp:alice@x.de"])),
    ]);
    let k = key(&mut st, "alice@x.de");
    assert_eq!(st.validate(g0).unwrap(), vec![dup(k, &[1, 2])]);
    let to = st.intern("new@x.de");
    let g1 = st
        .simulate(
            g0,
            &SetPrimarySmtp {
                rule: RENAME,
                user: g(1),
                to,
            },
            &[],
        )
        .unwrap();
    assert_eq!(st.validate(g1).unwrap(), vec![dup(k, &[1, 2])]);
}
