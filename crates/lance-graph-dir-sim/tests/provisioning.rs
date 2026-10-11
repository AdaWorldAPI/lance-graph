//! The provisioning plane: an address claimed by more than one directory
//! object, whatever state the objects are in.
//!
//! Entra ID keeps `userPrincipalName` and `proxyAddresses` unique across
//! every object of the tenant; a duplicate value from a sync is quarantined
//! on the object that brought it, enabled or not, mail-enabled or not. That
//! is a separate question from who holds an address in Exchange (only
//! address owners: [`validate::address_owner`]) and from who receives mail
//! at it ([`validate::address_recipient`]), and it must not change either.

use lance_graph_dir_sim::validate::{
    address_owner, address_recipient, provisioning_conflicts, ProvisioningConflict,
};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
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

fn group(smtp: &str) -> ObservedNode {
    let mut grp = ObservedNode::group();
    grp.primary_smtp = Some(smtp.into());
    grp
}

/// An enabled account whose recipient type was read and is empty: not
/// mail-enabled. Its SMTP and proxies are leftovers in Exchange terms.
fn not_mail_enabled(name: &str, smtp: &str) -> ObservedNode {
    let mut u = ObservedNode::user(&format!("{name}.upn@x.test"), smtp);
    u.recipient = Some(ObservedRecipient::default());
    u
}

// A non-mail-enabled account still carrying a group's address: Exchange
// holds and delivers the address to the group, and validation raises no
// address conflict, but provisioning reports the key as claimed twice.
#[test]
fn a_leftover_address_is_a_provisioning_conflict_not_a_delivery_change() {
    let (mut st, v) = store(vec![
        (g(0xB0), group("team@x.test")),
        (g(0xDA), not_mail_enabled("dave", "team@x.test")),
    ]);
    let key = {
        let id = st.intern("team@x.test");
        st.key_of(id).unwrap()
    };
    let out = st.validate(v).unwrap();
    assert!(
        !out.iter()
            .any(|x| matches!(x, Violation::AddressConflict { .. })),
        "{out:?}"
    );
    let view = st.view(v).unwrap();
    assert_eq!(address_owner(&view, key), Ok(Some(g(0xB0))));
    assert_eq!(address_recipient(&view, key), Ok(Some(g(0xB0))));
    assert_eq!(
        provisioning_conflicts(&view),
        vec![ProvisioningConflict {
            key,
            claims: vec![
                (g(0xB0), AddressRole::PrimarySmtp),
                (g(0xDA), AddressRole::PrimarySmtp),
            ],
        }]
    );
}

// A disabled account's UPN is still claimed in the tenant.
#[test]
fn a_disabled_accounts_upn_is_still_claimed() {
    let mut disabled = ObservedNode::user("shared@x.test", "old@x.test");
    disabled.active = Some(false);
    disabled.recipient = Some(ObservedRecipient::default());
    let (mut st, v) = store(vec![
        (g(0xD1), disabled),
        (
            g(0xA1),
            ObservedNode::user("alice.upn@x.test", "shared@x.test"),
        ),
    ]);
    let key = {
        let id = st.intern("shared@x.test");
        st.key_of(id).unwrap()
    };
    let view = st.view(v).unwrap();
    assert_eq!(address_owner(&view, key), Ok(Some(g(0xA1))));
    assert_eq!(
        provisioning_conflicts(&view),
        vec![ProvisioningConflict {
            key,
            claims: vec![
                (g(0xA1), AddressRole::PrimarySmtp),
                (g(0xD1), AddressRole::Upn),
            ],
        }]
    );
}

// The silence half: distinct addresses, and one object claiming one key
// under two roles (its UPN is its primary SMTP), conflict with nothing.
#[test]
fn distinct_claims_and_one_object_under_two_roles_are_not_conflicts() {
    let (st, v) = store(vec![
        (g(0xA1), ObservedNode::user("alice@x.test", "alice@x.test")),
        (g(0xB0), group("team@x.test")),
        (g(0xDA), not_mail_enabled("dave", "dave@x.test")),
    ]);
    let view = st.view(v).unwrap();
    assert_eq!(provisioning_conflicts(&view), vec![]);
}
