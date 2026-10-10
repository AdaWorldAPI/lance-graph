//! A mail recipient and a directory recipient converge on one address
//! identity, and the directory decides whether it names exactly one owner.
//!
//! The mail side is the address text a recipient arrives with: what Stalwart
//! hands over as `SessionAddress::address` and what Spear stores in its
//! `to_addrs` column. It crosses into ids exactly once, through
//! [`Dicts::key_lookup`] (counted, never minting). From there
//! [`validate::address_owner`] works on `KeyId`s only, over the same
//! `(key, holder, role)` rows as [`validate::address_rules`]: UPN,
//! primary and secondary SMTP, and the routing address. `mail` is a label
//! and holds nothing.
//!
//! Several holders are never resolved to one: no first match, no insertion
//! order, no preferred attribute.

use lance_graph_dir_sim::validate::{address_owner, validate};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const A: u8 = 0xA1;
const B: u8 = 0xB0;
const MAIL: &str = "a@example.org";
const ROUTING: &str = "a@tenant.mail.onmicrosoft.com";

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}

/// User A: `mail` = primary SMTP = `a@example.org`, a remote user mailbox
/// routing to `a@tenant.mail.onmicrosoft.com`, which it owns as a proxy.
fn user_a() -> ObservedNode {
    let mut u = ObservedNode::user("a.upn@example.org", MAIL);
    u.mail = Some(MAIL.into());
    u.alias = Some("a".into());
    u.proxies = vec![format!("smtp:{ROUTING}")];
    u.recipient = Some(ObservedRecipient {
        remote_recipient_type: Some(4),
        display_type: Some(-2_147_483_642),
        type_details: Some(2_147_483_648),
        target_address: Some(ROUTING.into()),
    });
    u
}

/// User B: an enabled user with its own addresses.
fn user_b() -> ObservedNode {
    ObservedNode::user("b.upn@example.org", "b@example.org")
}

fn observe(nodes: Vec<(Guid128, ObservedNode)>) -> (VersionStore, VersionId) {
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

/// Ingress for the mail side: the recipient text becomes a key once.
fn recipient_key(st: &VersionStore, address: &str) -> Option<KeyId> {
    st.dicts().key_lookup(address)
}

fn owner(st: &VersionStore, v: VersionId, address: &str) -> Result<Option<Guid128>, Violation> {
    let key = recipient_key(st, address).expect("the directory knows this address");
    address_owner(&st.view(v).unwrap(), key)
}

// 1. The message recipient and the directory's SMTP, `mail` and proxy
// spellings are one identity.
#[test]
fn the_message_address_and_the_directory_address_are_one_key() {
    let (mut st, _) = observe(vec![(g(A), user_a())]);
    let from_mail = recipient_key(&st, MAIL).unwrap();
    // The directory interned A's primary SMTP and `mail` (both `MAIL`) at
    // observation; re-interning finds the same value and so the same key.
    let primary = st.intern(MAIL);
    assert_eq!(st.key_of(primary), Some(from_mail));
    let routing = st.intern(ROUTING);
    assert_ne!(
        st.key_of(routing),
        Some(from_mail),
        "routing is its own address"
    );
}

// 2. Case and surrounding whitespace do not change the identity.
#[test]
fn casing_does_not_change_the_identity() {
    let (st, _) = observe(vec![(g(A), user_a())]);
    let k = recipient_key(&st, MAIL).unwrap();
    for spelling in ["A@Example.ORG", "  a@EXAMPLE.org ", "A@EXAMPLE.ORG"] {
        assert_eq!(recipient_key(&st, spelling), Some(k), "{spelling}");
    }
}

// 3. One owner: the message recipient resolves to user A, and so does A's
// routing address (one holder under two roles is still one owner).
#[test]
fn a_uniquely_owned_address_resolves_to_its_owner() {
    let (st, v) = observe(vec![(g(A), user_a()), (g(B), user_b())]);
    assert_eq!(owner(&st, v, MAIL), Ok(Some(g(A))));
    assert_eq!(owner(&st, v, "A@EXAMPLE.ORG"), Ok(Some(g(A))));
    assert_eq!(owner(&st, v, ROUTING), Ok(Some(g(A))));
    assert_eq!(owner(&st, v, "b@example.org"), Ok(Some(g(B))));
    assert!(validate(&st.view(v).unwrap()).is_empty());
}

// An address no directory object holds has no owner, and resolving it
// mints nothing: a key the directory never saw is not a key.
#[test]
fn an_address_the_directory_never_saw_has_no_key() {
    let (st, _) = observe(vec![(g(A), user_a())]);
    let keys = st.dicts().key_count();
    assert_eq!(recipient_key(&st, "nobody@example.org"), None);
    assert_eq!(st.dicts().key_count(), keys, "lookup never mints");
}

// A key the directory knows but no live directory object holds resolves to
// no owner (here: B's UPN key after B is gone is covered below; this one is
// a value only the dictionary saw).
#[test]
fn a_known_key_without_a_holder_has_no_owner() {
    let (mut st, v) = observe(vec![(g(A), user_a())]);
    let stray = st.intern("stray@example.org");
    let key = st.key_of(stray).unwrap();
    assert_eq!(address_owner(&st.view(v).unwrap(), key), Ok(None));
}

// 4. + 5. Another object claims the same address through a different
// surface (B's UPN). The recipient no longer names one owner: the
// resolution reports both holders with their roles, and validation reports
// the same conflict.
#[test]
fn a_cross_attribute_claim_makes_the_recipient_ambiguous() {
    let mut b = user_b();
    b.upn = Some(MAIL.into());
    let (st, v) = observe(vec![(g(A), user_a()), (g(B), b)]);
    let key = recipient_key(&st, MAIL).unwrap();
    let Err(Violation::AddressConflict { key: k, holders }) = owner(&st, v, MAIL) else {
        panic!("two holders must not resolve to one owner");
    };
    assert_eq!(k, key);
    // Exactly the two claims: `mail` holds nothing, so no third holder.
    assert_eq!(
        holders,
        vec![(g(A), AddressRole::PrimarySmtp), (g(B), AddressRole::Upn)]
    );
    let found = validate(&st.view(v).unwrap());
    assert!(
        found.iter().any(|x| matches!(x,
            Violation::AddressConflict { key: k, .. } if *k == key)),
        "validation reports the same conflict: {found:?}"
    );
}

// 5. Ambiguity is not decided by observation order or by the attribute: B
// observed first gives the same answer, and a same-attribute collision (two
// primary SMTPs, `DuplicateSmtp` to validation) is just as unresolved.
#[test]
fn ambiguity_never_picks_an_owner() {
    let mut b = user_b();
    b.upn = Some(MAIL.into());
    let (st1, v1) = observe(vec![(g(A), user_a()), (g(B), b.clone())]);
    let (st2, v2) = observe(vec![(g(B), b), (g(A), user_a())]);
    let e1 = owner(&st1, v1, MAIL).unwrap_err();
    let e2 = owner(&st2, v2, MAIL).unwrap_err();
    assert_eq!(e1, e2, "the answer does not depend on observation order");

    let twin = ObservedNode::user("twin.upn@example.org", MAIL);
    let (st, v) = observe(vec![(g(A), user_a()), (g(B), twin)]);
    assert!(
        matches!(owner(&st, v, MAIL), Err(Violation::AddressConflict { .. })),
        "two primary SMTPs are two holders too"
    );
}

// 6. After the recipient became a key, resolution does no text work at all.
#[test]
fn resolution_is_string_free_after_ingress() {
    let mut b = user_b();
    b.upn = Some(MAIL.into());
    for nodes in [vec![(g(A), user_a())], vec![(g(A), user_a()), (g(B), b)]] {
        let (st, v) = observe(nodes);
        let key = recipient_key(&st, "A@Example.org").unwrap();
        let view = st.view(v).unwrap();
        let before = st.dicts().counters.snapshot();
        let _ = address_owner(&view, key);
        assert_eq!(
            st.dicts().counters.snapshot(),
            before,
            "no intern, lookup or resolution during resolution"
        );
    }
}

// `mail` is a label: B carrying A's address as `mail` claims nothing, and
// the address still names A alone. In Exchange, uniqueness is enforced on
// proxy addresses (and the UPN namespace), never on `mail`.
#[test]
fn a_mail_label_on_another_object_is_not_a_claim() {
    let mut b = user_b();
    b.mail = Some(MAIL.into());
    let (st, v) = observe(vec![(g(A), user_a()), (g(B), b)]);
    assert_eq!(owner(&st, v, MAIL), Ok(Some(g(A))));
    assert!(validate(&st.view(v).unwrap()).is_empty());
}

// 7. A deleted holder stops holding: deleting B in a simulated version
// makes A the only owner again, through the same view semantics every
// other rule reads.
#[test]
fn a_deleted_holder_no_longer_holds_the_address() {
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
    let mut b = user_b();
    b.upn = Some(MAIL.into());
    let (mut st, v0) = observe(vec![(g(A), user_a()), (g(B), b)]);
    assert!(owner(&st, v0, MAIL).is_err());
    let state = st.view(v0).unwrap().node_state(&g(B)).unwrap();
    let v1 = st
        .simulate(
            v0,
            &Propose(vec![Change::DeleteNode { node: g(B), state }]),
            &[],
        )
        .unwrap();
    assert_eq!(owner(&st, v1, MAIL), Ok(Some(g(A))));
    assert!(
        owner(&st, v0, MAIL).is_err(),
        "the base version is unchanged"
    );
}

/// A mail-enabled group with its own primary SMTP address.
fn group(smtp: &str) -> ObservedNode {
    let mut g = ObservedNode::group();
    g.primary_smtp = Some(smtp.into());
    g
}

const G: u8 = 0x61;

// 9. A group's primary SMTP address is in the one address space: it names the
// group, so a recipient can be a group (a distribution list), not only a user.
#[test]
fn a_group_address_names_the_group() {
    let (st, v) = observe(vec![(g(A), user_a()), (g(G), group("sales@example.org"))]);
    assert_eq!(owner(&st, v, "sales@example.org"), Ok(Some(g(G))));
    // The user's own address is untouched.
    assert_eq!(owner(&st, v, MAIL), Ok(Some(g(A))));
    assert!(validate(&st.view(v).unwrap()).is_empty());
}

// 10. A group and a user cannot share an address: Exchange refuses it, so the
// recipient is ambiguous and the version is invalid. Without groups in the
// address space this collision was silent.
#[test]
fn a_group_and_a_user_sharing_an_address_conflict() {
    let (st, v) = observe(vec![(g(B), user_b()), (g(G), group("b@example.org"))]);
    let Err(Violation::AddressConflict { holders, .. }) = owner(&st, v, "b@example.org") else {
        panic!("a shared address must be a conflict");
    };
    assert!(holders.contains(&(g(B), AddressRole::PrimarySmtp)));
    assert!(holders.contains(&(g(G), AddressRole::PrimarySmtp)));
    let key = recipient_key(&st, "b@example.org").unwrap();
    assert!(
        validate(&st.view(v).unwrap())
            .iter()
            .any(|x| matches!(x, Violation::AddressConflict { key: k, .. } if *k == key)),
        "validate must report the user/group collision"
    );
}

// 11. Two groups cannot share an address either.
#[test]
fn two_groups_sharing_an_address_conflict() {
    let (st, v) = observe(vec![
        (g(G), group("team@example.org")),
        (g(0x62), group("team@example.org")),
    ]);
    assert!(matches!(
        owner(&st, v, "team@example.org"),
        Err(Violation::AddressConflict { .. })
    ));
    assert!(!validate(&st.view(v).unwrap()).is_empty());
}
