//! Nested groups expand through two closures (OGAR plan V19): delivery
//! walks mail-enabled groups, security walks only groups known to be
//! security-enabled.

use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

const U1: u8 = 0x11;
const U2: u8 = 0x12;
const U3: u8 = 0x13;
const U4: u8 = 0x14;
const U5: u8 = 0x15;
/// Mail-enabled security group.
const S1: u8 = 0xA1;
/// Security group, not mail-enabled.
const S2: u8 = 0xA2;
/// Mail-enabled security group, reached only through D1.
const S3: u8 = 0xA3;
/// Distribution group: mail-enabled, not security-enabled.
const D1: u8 = 0xD1;
/// Mail-enabled group whose security flag was not read.
const UNREAD: u8 = 0xE1;

fn group(smtp: Option<&str>, security: Option<bool>) -> ObservedNode {
    let mut n = ObservedNode::group();
    n.primary_smtp = smtp.map(str::to_string);
    n.security = security;
    n
}

// S1 ⊃ {u1, S2, D1, UNREAD}; S2 ⊃ {u2, S1} (a cycle); D1 ⊃ {u3, S3};
// S3 ⊃ {u4}; UNREAD ⊃ {u5}.
fn observed() -> Observation {
    let user = |n: &str| ObservedNode::user(&format!("{n}@x.test"), &format!("{n}@x.test"));
    Observation {
        scope: SCOPE,
        nodes: vec![
            (g(U1), user("u1")),
            (g(U2), user("u2")),
            (g(U3), user("u3")),
            (g(U4), user("u4")),
            (g(U5), user("u5")),
            (g(S1), group(Some("s1@x.test"), Some(true))),
            (g(S2), group(None, Some(true))),
            (g(S3), group(Some("s3@x.test"), Some(true))),
            (g(D1), group(Some("d1@x.test"), Some(false))),
            (g(UNREAD), group(Some("unread@x.test"), None)),
        ],
        members: vec![
            (g(U1), g(S1)),
            (g(S2), g(S1)),
            (g(D1), g(S1)),
            (g(UNREAD), g(S1)),
            (g(U2), g(S2)),
            (g(S1), g(S2)),
            (g(U3), g(D1)),
            (g(S3), g(D1)),
            (g(U4), g(S3)),
            (g(U5), g(UNREAD)),
        ],
    }
}

fn ids(ns: &[u8]) -> Vec<Guid128> {
    ns.iter().map(|&n| g(n)).collect()
}

#[test]
fn the_security_flag_is_kept_as_read() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(view.is_security_enabled(&g(S1)), Some(true));
    assert_eq!(view.is_security_enabled(&g(D1)), Some(false));
    assert_eq!(view.is_security_enabled(&g(UNREAD)), None);
    // Not a group, or not there.
    assert_eq!(view.is_security_enabled(&g(U1)), None);
    assert_eq!(view.is_security_enabled(&g(0x99)), None);
    // Mail-enabled is the other, independent property.
    assert!(view.is_mail_recipient(&g(S1)));
    assert!(!view.is_mail_recipient(&g(S2)));
    assert!(view.is_mail_recipient(&g(D1)));
}

// Security walks S1 → S2 (and back, once) but never through the
// distribution group D1, nor through the group whose flag was not read.
#[test]
fn the_security_closure_walks_only_security_groups() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(
        view.members_transitive(&g(S1), Closure::Security),
        ids(&[U1, U2])
    );
    // The cycle S1 ↔ S2 is the same set from either side.
    assert_eq!(
        view.members_transitive(&g(S2), Closure::Security),
        ids(&[U1, U2])
    );
    // A security group reached only through a distribution group keeps its
    // own members.
    assert_eq!(
        view.members_transitive(&g(S3), Closure::Security),
        ids(&[U4])
    );
    // A distribution group, or an unread flag, holds no permission at all.
    assert!(view
        .members_transitive(&g(D1), Closure::Security)
        .is_empty());
    assert!(view
        .members_transitive(&g(UNREAD), Closure::Security)
        .is_empty());
}

// Delivery walks every mail-enabled group, security-enabled or not, and
// skips the one without an address.
#[test]
fn the_delivery_closure_walks_mail_enabled_groups() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(
        view.members_transitive(&g(S1), Closure::Delivery),
        ids(&[U1, U3, U4, U5])
    );
    assert_eq!(
        view.members_transitive(&g(D1), Closure::Delivery),
        ids(&[U3, U4])
    );
    assert!(view
        .members_transitive(&g(S2), Closure::Delivery)
        .is_empty());
    // A user, or nothing, is not a group.
    assert!(view
        .members_transitive(&g(U1), Closure::Delivery)
        .is_empty());
    assert!(view
        .members_transitive(&g(0x99), Closure::Delivery)
        .is_empty());
}

// Membership walks every group: the address-less S2 and the unread group
// that Delivery or Security stop at are both followed.
#[test]
fn the_membership_closure_walks_every_group() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let all = ids(&[U1, U2, U3, U4, U5]);
    assert_eq!(view.members_transitive(&g(S1), Closure::Membership), all);
    // S2 has no address, so Delivery from it is empty; Membership is not.
    assert!(view
        .members_transitive(&g(S2), Closure::Delivery)
        .is_empty());
    assert_eq!(view.members_transitive(&g(S2), Closure::Membership), all);
    // A user, or nothing, is still not a group.
    assert!(view
        .members_transitive(&g(U1), Closure::Membership)
        .is_empty());
}

struct Edit(Vec<Change>);
impl Rule for Edit {
    fn id(&self) -> RuleId {
        RuleId {
            name: "Edit",
            version: 1,
        }
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}

// A version's added and removed memberships are seen, nested pairs included.
#[test]
fn a_version_s_memberships_are_followed() {
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 1, observed()).unwrap();
    let v1 = st
        .simulate(
            v0,
            &Edit(vec![
                // u5 joins S2 directly; S2 leaves S1; S3 joins S1; u1 leaves
                // S1 (an observed, resolved row).
                Change::RemoveMembership {
                    user: g(U1),
                    group: g(S1),
                },
                Change::AddMembership {
                    user: g(U5),
                    group: g(S2),
                },
                Change::RemoveMembership {
                    user: g(S2),
                    group: g(S1),
                },
                Change::AddMembership {
                    user: g(S3),
                    group: g(S1),
                },
            ]),
            &[EvidenceRef("t".into())],
        )
        .unwrap();
    let view = st.view(v1).unwrap();
    // S1 no longer contains u1 or S2 (so u2 and u5 are gone); S3 now
    // reaches u4.
    assert_eq!(
        view.members_transitive(&g(S1), Closure::Security),
        ids(&[U4])
    );
    // S2 still contains S1, and u5 directly.
    assert_eq!(
        view.members_transitive(&g(S2), Closure::Security),
        ids(&[U2, U4, U5])
    );
    // The observed version is unchanged.
    let view0 = st.view(v0).unwrap();
    assert_eq!(
        view0.members_transitive(&g(S1), Closure::Security),
        ids(&[U1, U2])
    );
}

// AD's groupType reaches the observation: its high bit is the flag, and a
// record whose schema predates groupType stays unread.
#[test]
fn ad_group_type_sets_the_flag() {
    use ogar_ad::{encode, ldif};
    use ogar_dir_core::{OuDictionary, ValuePool};
    let ldif_text = "dn: CN=Sec,OU=G,DC=x,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAQ==\nobjectClass: group\ngroupType: -2147483646\n\ndn: CN=Dist,OU=G,DC=x,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAg==\nobjectClass: group\ngroupType: 8\n\ndn: CN=None,OU=G,DC=x,DC=test\nobjectGUID:: 4AQlP4lP0xGaDAMF6CwzAw==\nobjectClass: group\n";
    let (mut dict, mut pool) = (OuDictionary::new(), ValuePool::new());
    let recs: Vec<_> = ldif::parse(ldif_text)
        .unwrap()
        .iter()
        .map(|e| encode(e, SCOPE.0, &mut dict, &mut pool, 1).unwrap().record)
        .collect();
    let obs = observe::from_ad(SCOPE, &recs, &pool).unwrap();
    let flags: Vec<Option<bool>> = recs
        .iter()
        .map(|r| {
            obs.nodes
                .iter()
                .find(|(id, _)| *id == r.node_guid())
                .unwrap()
                .1
                .security
        })
        .collect();
    assert_eq!(flags, vec![Some(true), Some(false), None]);
}
