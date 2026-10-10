//! Nesting is one global pattern (OGAR plan V19): every nested group is
//! followed, whatever its kind, in both directions (the users a group
//! reaches, the groups a user is in). Mail-enabled and security-enabled are
//! independent properties, filtered with one Quack program.

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

// Nesting follows every group, whatever its kind: from S1 through the
// address-less S2, the distribution group D1 and the unread group alike.
#[test]
fn nesting_follows_every_group() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let all = ids(&[U1, U2, U3, U4, U5]);
    assert_eq!(view.members_transitive(&g(S1)), all);
    // S2 contains S1, which closes the cycle: the same set from either side.
    assert_eq!(view.members_transitive(&g(S2)), all);
    assert_eq!(view.members_transitive(&g(D1)), ids(&[U3, U4]));
    assert_eq!(view.members_transitive(&g(S3)), ids(&[U4]));
    assert_eq!(view.members_transitive(&g(UNREAD)), ids(&[U5]));
    // A user, or nothing, is not a group.
    assert!(view.members_transitive(&g(U1)).is_empty());
    assert!(view.members_transitive(&g(0x99)).is_empty());
}

// The groups a user is in, upward through every kind of group: u4 is in S3,
// S3 in D1, D1 in S1, S1 in S2.
#[test]
fn a_user_s_groups_follow_nesting_upward() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(view.groups_transitive(&g(U4)), ids(&[S1, S2, S3, D1]));
    assert_eq!(view.groups_transitive(&g(U2)), ids(&[S1, S2]));
    assert_eq!(view.groups_transitive(&g(U5)), ids(&[S1, S2, UNREAD]));
    // A group, or nothing, is not a user.
    assert!(view.groups_transitive(&g(S1)).is_empty());
    assert!(view.groups_transitive(&g(0x99)).is_empty());
}

// The two directions agree on every user and group.
#[test]
fn members_and_groups_are_inverse() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let users = ids(&[U1, U2, U3, U4, U5]);
    let groups = ids(&[S1, S2, S3, D1, UNREAD]);
    let mut held = 0;
    for gr in &groups {
        let members = view.members_transitive(gr);
        for u in &users {
            let up = view.groups_transitive(u);
            assert_eq!(members.contains(u), up.contains(gr), "{u:?} {gr:?}");
            held += usize::from(members.contains(u));
        }
    }
    // Not vacuous: some pairs hold and some do not.
    assert!(held > 0 && held < users.len() * groups.len(), "{held}");
}

fn selected(view: &View<'_>, w: &GroupWhere) -> Vec<Guid128> {
    groups_where(view, w)
        .rows()
        .into_iter()
        .filter_map(|i| view.group_guid(GroupOrdinal(i as u16)))
        .collect()
}

fn is(p: GroupProperty) -> GroupWhere {
    GroupWhere::Is(p)
}

fn not(w: GroupWhere) -> GroupWhere {
    GroupWhere::Not(Box::new(w))
}

// Mail-enabled and security-enabled are independent properties, selected
// by one Quack filter: neither, either or both.
#[test]
fn group_properties_are_a_quack_filter() {
    use GroupProperty::{MailEnabled, SecurityEnabled};
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!(
        selected(&view, &is(MailEnabled)),
        ids(&[S1, S3, D1, UNREAD])
    );
    assert_eq!(selected(&view, &is(SecurityEnabled)), ids(&[S1, S2, S3]));
    // Both is both.
    assert_eq!(selected(&view, &GroupWhere::both()), ids(&[S1, S3]));
    // A security group without an address.
    assert_eq!(
        selected(
            &view,
            &GroupWhere::And(vec![is(SecurityEnabled), not(is(MailEnabled))])
        ),
        ids(&[S2])
    );
    // A distribution group: mail-enabled, not known to be security-enabled.
    assert_eq!(
        selected(
            &view,
            &GroupWhere::And(vec![is(MailEnabled), not(is(SecurityEnabled))])
        ),
        ids(&[D1, UNREAD])
    );
    // The filter agrees with the per-group reads on every group.
    for gr in ids(&[S1, S2, S3, D1, UNREAD]) {
        assert_eq!(
            selected(&view, &is(MailEnabled)).contains(&gr),
            view.is_mail_recipient(&gr)
        );
        assert_eq!(
            selected(&view, &is(SecurityEnabled)).contains(&gr),
            view.is_security_enabled(&gr) == Some(true)
        );
    }
}

// Only a security group has a SID, so permission inheritance runs through
// security groups only. Membership nesting still reaches every group.
#[test]
fn only_a_security_group_passes_on_its_sid() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    // u3 is in the distribution group D1, which is nested in S1: a member of
    // S1 by nesting, but D1 has no SID, so u3 holds none of S1's.
    assert_eq!(view.groups_transitive(&g(U3)), ids(&[S1, S2, D1]));
    assert!(view.security_identifiers(&g(U3)).is_empty());
    // u4 is in S3, which is nested in D1: S3's SID, but not S1's through D1.
    assert_eq!(view.security_identifiers(&g(U4)), ids(&[S3]));
    // u1 and u2 reach S1 and S2 through security groups only.
    assert_eq!(view.security_identifiers(&g(U1)), ids(&[S1, S2]));
    assert_eq!(view.security_identifiers(&g(U2)), ids(&[S1, S2]));
    // An unread flag is not a SID.
    assert!(view.security_identifiers(&g(U5)).is_empty());
    // Who inherits a permission granted to S1: down through S2 only, not
    // through D1 or the unread group.
    let sec = GroupWhere::Is(GroupProperty::SecurityEnabled);
    assert_eq!(
        view.members_transitive_through(&g(S1), &sec),
        ids(&[U1, U2])
    );
    // A permission granted to a group without a SID reaches nobody.
    assert!(view.members_transitive_through(&g(D1), &sec).is_empty());
    // The two directions agree on every pair.
    let mut held = 0;
    for gr in ids(&[S1, S2, S3, D1, UNREAD]) {
        let down = view.members_transitive_through(&gr, &sec);
        for u in ids(&[U1, U2, U3, U4, U5]) {
            let up = view.groups_transitive_through(&u, &sec);
            assert_eq!(down.contains(&u), up.contains(&gr), "{u:?} {gr:?}");
            held += usize::from(down.contains(&u));
        }
    }
    assert!(held > 0 && held < 25, "{held}");
}

// The empty junctions are the identities, not a refused query: an empty And
// holds for every group, an empty Or for none, also when nested.
#[test]
fn empty_property_junctions_are_identities() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let all = ids(&[S1, S2, S3, D1, UNREAD]);
    assert_eq!(selected(&view, &GroupWhere::And(vec![])), all);
    assert!(selected(&view, &GroupWhere::Or(vec![])).is_empty());
    assert_eq!(selected(&view, &not(GroupWhere::Or(vec![]))), all);
    assert!(selected(
        &view,
        &GroupWhere::And(vec![GroupWhere::Or(vec![]), is(GroupProperty::MailEnabled)])
    )
    .is_empty());
}

// Nesting and properties compose: the groups u4 is in, filtered. Its SIDs
// come from the security groups, its lists from the mail-enabled ones.
#[test]
fn nesting_and_properties_compose() {
    use GroupProperty::{MailEnabled, SecurityEnabled};
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let up = view.groups_transitive(&g(U4));
    let filter = |w: &GroupWhere| -> Vec<Guid128> {
        let keep = selected(&view, w);
        up.iter().copied().filter(|gr| keep.contains(gr)).collect()
    };
    assert_eq!(filter(&is(SecurityEnabled)), ids(&[S1, S2, S3]));
    assert_eq!(filter(&is(MailEnabled)), ids(&[S1, S3, D1]));
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

// A version's added and removed memberships are seen, nested pairs included,
// and a version's address change is seen by the property filter.
#[test]
fn a_version_s_memberships_are_followed() {
    let mut st = VersionStore::new();
    let v0 = st.observe("lab", 1, observed()).unwrap();
    let s1_smtp = st.view(v0).unwrap().attr(&g(S1), Attribute::PrimarySmtp);
    assert!(s1_smtp.is_some());
    let v1 = st
        .simulate(
            v0,
            &Edit(vec![
                // u1 leaves S1 (an observed, resolved row); u5 joins S2
                // directly; S2 leaves S1; S3 joins S1; S1 loses its address.
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
                Change::SetAttribute {
                    node: g(S1),
                    attribute: Attribute::PrimarySmtp,
                    from: s1_smtp,
                    to: None,
                },
            ]),
            &[EvidenceRef("t".into())],
        )
        .unwrap();
    let view = st.view(v1).unwrap();
    // S1: D1 (u3, S3 → u4), UNREAD (u5), S3 (u4).
    assert_eq!(view.members_transitive(&g(S1)), ids(&[U3, U4, U5]));
    // S2: u2 and u5 directly, S1's set through S1.
    assert_eq!(view.members_transitive(&g(S2)), ids(&[U2, U3, U4, U5]));
    assert_eq!(view.groups_transitive(&g(U1)), Vec::<Guid128>::new());
    // S1 is no longer mail-enabled in this version; still security-enabled.
    assert_eq!(
        selected(&view, &GroupWhere::Is(GroupProperty::MailEnabled)),
        ids(&[S3, D1, UNREAD])
    );
    assert_eq!(selected(&view, &GroupWhere::both()), ids(&[S3]));
    // The observed version is unchanged.
    let view0 = st.view(v0).unwrap();
    assert_eq!(view0.members_transitive(&g(S1)), ids(&[U1, U2, U3, U4, U5]));
    assert_eq!(selected(&view0, &GroupWhere::both()), ids(&[S1, S3]));
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
