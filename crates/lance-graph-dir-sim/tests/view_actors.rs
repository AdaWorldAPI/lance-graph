//! The roles a principal holds through a view (`actors`, feature `rbac`):
//! granted to security groups by GUID, inherited through SID chains only,
//! filtered by the consumer's binding, nothing from an unread group.

use lance_graph_contract::rbac_plug::{ActorSource, RbacBinding};
use lance_graph_dir_sim::actors::{GroupRoles, ViewActors};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};

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

fn binding(roles: &[&'static str]) -> RbacBinding {
    RbacBinding::new(
        "test",
        vec![],
        roles.iter().map(|r| (*r, vec![])).collect(),
        vec![],
    )
}

// S1 grants `reader`, S2 `editor` and a role the plug does not declare, D1
// (a distribution group) `dl`; S3's grants were never read.
fn grants() -> GroupRoles {
    let mut gr = GroupRoles::new();
    gr.read(g(S1), &["reader"])
        .read(g(S2), &["editor", "undeclared"])
        .read(g(D1), &["dl"])
        .read(g(UNREAD), &["reader"]);
    gr
}

const DECLARED: &[&str] = &["reader", "editor", "dl"];

#[test]
fn roles_come_from_the_security_groups_a_principal_carries() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let b = binding(DECLARED);
    // u1 is in S1 directly and in S2 through S1: both grants, the undeclared
    // role dropped.
    let a = ViewActors::for_principal(&view, &g(U1), &grants(), &b);
    assert_eq!(a.roles(), &["editor", "reader"]);
    assert_eq!(a.roles_of(a.subject()), &["editor", "reader"]);
    assert!(a.unread_groups().is_empty());
    // u3 is in D1, nested in S1: D1 has no SID, so neither D1's grant nor
    // S1's reaches u3.
    let a = ViewActors::for_principal(&view, &g(U3), &grants(), &b);
    assert!(a.roles().is_empty());
    // u5 is in the group whose security flag was not read: no SID, no role.
    let a = ViewActors::for_principal(&view, &g(U5), &grants(), &b);
    assert!(a.roles().is_empty());
}

#[test]
fn an_unread_group_grants_nothing_and_is_named() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    // u4 carries only S3's SID, whose grants were never read.
    let a = ViewActors::for_principal(&view, &g(U4), &grants(), &binding(DECLARED));
    assert!(a.roles().is_empty());
    assert_eq!(a.unread_groups(), &[g(S3)]);
    // Read with no grants is not the same as unread.
    let mut gr = grants();
    gr.read(g(S3), &[]);
    let a = ViewActors::for_principal(&view, &g(U4), &gr, &binding(DECLARED));
    assert!(a.roles().is_empty());
    assert!(a.unread_groups().is_empty());
}

#[test]
fn the_binding_decides_which_roles_survive() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    // Declaring the role lets the same grant through: the filter binds.
    let a = ViewActors::for_principal(
        &view,
        &g(U2),
        &grants(),
        &binding(&["editor", "undeclared"]),
    );
    assert_eq!(a.roles(), &["editor", "undeclared"]);
    let a = ViewActors::for_principal(&view, &g(U2), &grants(), &binding(&["reader"]));
    assert_eq!(a.roles(), &["reader"]);
}

#[test]
fn only_the_principal_holds_roles() {
    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let b = binding(DECLARED);
    let a = ViewActors::for_principal(&view, &g(U1), &grants(), &b);
    assert_eq!(a.subject(), g(U1).to_string());
    assert!(a.roles_of(&g(U2).to_string()).is_empty());
    assert!(a.roles_of("").is_empty());
    // Not a user in the view: nothing.
    let a = ViewActors::for_principal(&view, &g(0x99), &grants(), &b);
    assert!(a.roles().is_empty());
    let a = ViewActors::for_principal(&view, &g(S1), &grants(), &b);
    assert!(a.roles().is_empty());
}

// Grants come from the authority's versioned policy, not from the directory.
// A group the policy never named stays unread; a group whose grants were
// all revoked is read with no roles; an earlier policy version gives the
// roles that stood then.
#[test]
fn roles_come_from_one_version_of_the_group_role_policy() {
    use ogar_rbac::group_policy::{GrantProvenance, GroupRolePolicy, PolicyChange};
    let by = |e: &str| GrantProvenance {
        by: "admin".into(),
        evidence: e.into(),
        at_ms: 1,
    };
    let mut policy = GroupRolePolicy::new();
    policy
        .apply(
            PolicyChange::Grant {
                group: g(S1),
                role: "reader",
            },
            by("t1"),
        )
        .unwrap();
    let granted = policy
        .apply(
            PolicyChange::Grant {
                group: g(S3),
                role: "editor",
            },
            by("t2"),
        )
        .unwrap();
    policy
        .apply(
            PolicyChange::Revoke {
                group: g(S3),
                role: "editor",
            },
            by("t3"),
        )
        .unwrap();

    let mut st = VersionStore::new();
    let v = st.observe("lab", 1, observed()).unwrap();
    let view = st.view(v).unwrap();
    let b = binding(DECLARED);

    let now = GroupRoles::from_policy(&policy.as_of(policy.head()).unwrap());
    // u1: S1 grants `reader`; S2 (reached through S1) was never named by the
    // policy, so it is unread, not empty.
    let a = ViewActors::for_principal(&view, &g(U1), &now, &b);
    assert_eq!(a.roles(), &["reader"]);
    assert_eq!(a.unread_groups(), &[g(S2)]);
    // u4: S3's only grant was revoked. Read with no roles, so not unread.
    let a = ViewActors::for_principal(&view, &g(U4), &now, &b);
    assert!(a.roles().is_empty());
    assert!(a.unread_groups().is_empty());

    // At the version where S3 still granted `editor`, u4 holds it.
    let then = GroupRoles::from_policy(&policy.as_of(granted).unwrap());
    let a = ViewActors::for_principal(&view, &g(U4), &then, &b);
    assert_eq!(a.roles(), &["editor"]);
}
