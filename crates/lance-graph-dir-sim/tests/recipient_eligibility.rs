//! Who holds an address and who receives mail at it are two questions.
//!
//! [`address_owner`] answers the first: the claim that keeps anyone else
//! from taking an address, held through UPN, SMTP proxies and the routing
//! address. `mail` is a label and holds nothing, as in Exchange, where
//! uniqueness is enforced on proxy addresses. [`address_recipient`]
//! answers the second, from OGAR's recipient lifecycle
//! (`Recipient::is_recipient`) through [`View::is_mail_recipient`].
//!
//! The first four cases are D-IAM-IDENTITY-0's falsifiers, now regressions:
//! every one validated clean and still resolved the departed user as the
//! one who receives.

use lance_graph_dir_sim::validate::{address_owner, address_recipient, validate};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::exchange::type_details;
use ogar_dir_sim::*;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const D: &str = "d@example.org";
const ROUTING: &str = "d@tenant.mail.onmicrosoft.com";

struct Plan(Vec<Change>);
impl Rule for Plan {
    fn id(&self) -> RuleId {
        RuleId {
            name: "Offboard",
            version: 1,
        }
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}

/// A remote mailbox's recipient attributes (`code` = msExchRemoteRecipientType).
fn remote(code: u32, details: u64, display: i32) -> ObservedRecipient {
    ObservedRecipient {
        remote_recipient_type: Some(code),
        display_type: Some(display),
        type_details: Some(details),
        target_address: Some(ROUTING.into()),
    }
}

fn remote_user_mailbox() -> ObservedRecipient {
    remote(4, type_details::REMOTE_USER_MAILBOX, -2_147_483_642)
}

/// A user holding `D` as `mail` and primary SMTP, plus the routing proxy.
fn user(active: Option<bool>, recipient: Option<ObservedRecipient>) -> ObservedNode {
    let mut u = ObservedNode::user("d.upn@example.org", D);
    u.active = active;
    u.mail = Some(D.into());
    u.alias = Some("d".into());
    u.proxies = vec![format!("smtp:{ROUTING}")];
    u.recipient = recipient;
    u
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

fn both(st: &VersionStore, v: VersionId, addr: &str) -> (Option<Guid128>, Option<Guid128>) {
    let key = st.dicts().key_lookup(addr).expect("known address");
    let view = st.view(v).unwrap();
    (
        address_owner(&view, key).unwrap(),
        address_recipient(&view, key).unwrap(),
    )
}

// 1. The departed user: disabled, not mail-enabled, no proxy addresses, a
// stale `mail` label.
#[test]
fn a_departed_user_holds_no_address_and_receives_nothing() {
    let mut d = user(Some(false), Some(ObservedRecipient::default()));
    d.primary_smtp = None;
    d.proxies.clear();
    let (st, v) = observe(vec![(g(1), d)]);
    assert!(validate(&st.view(v).unwrap()).is_empty());
    // The stale `mail` is a label: it reserves nothing ...
    let (owner, rcpt) = both(&st, v, D);
    assert_eq!(owner, None);
    // ... and mail to it is delivered nowhere.
    assert_eq!(rcpt, None);
    assert!(!st.view(v).unwrap().is_mail_recipient(&g(1)));
}

// 2. The offboarding a simulator can plan: disable the account and run
// Disable-RemoteMailbox. No change here clears `mail`, and none needs to:
// it is a label, and the deprovisioned account claims nothing through it.
#[test]
fn offboarding_by_disable_remote_mailbox_stops_delivery() {
    let (mut st, v0) = observe(vec![(g(1), user(Some(true), Some(remote_user_mailbox())))]);
    assert_eq!(both(&st, v0, D), (Some(g(1)), Some(g(1))));
    let before = st
        .view(v0)
        .unwrap()
        .node_state(&g(1))
        .unwrap()
        .recipient
        .unwrap();
    let after = RemoteMailboxOp::Disable.apply(&before).unwrap();
    assert_eq!(
        RemoteMailboxOp::between(&before, &after),
        Some(RemoteMailboxOp::Disable),
        "an actuatable step"
    );
    let plan = Plan(vec![
        Change::SetActive {
            node: g(1),
            from: Some(true),
            to: Some(false),
        },
        Change::SetRecipient {
            node: g(1),
            from: Some(before),
            to: Some(after),
        },
    ]);
    let v1 = st.simulate(v0, &plan, &[]).unwrap();
    assert!(validate(&st.view(v1).unwrap()).is_empty());
    let (owner, rcpt) = both(&st, v1, D);
    assert_eq!(
        owner, None,
        "a deprovisioned object holds none of its addresses"
    );
    assert_eq!(rcpt, None, "and nothing is delivered");
}

// 3. A disabled shared mailbox is a recipient: disabled accounts are how
// shared, room and equipment mailboxes are kept.
#[test]
fn a_disabled_shared_mailbox_still_receives() {
    let shared = remote(97, type_details::REMOTE_SHARED_MAILBOX, -2_147_483_642);
    let (st, v) = observe(vec![(g(1), user(Some(false), Some(shared)))]);
    let s = st.view(v).unwrap().node_state(&g(1)).unwrap();
    assert!(
        matches!(s.recipient, Some(Recipient::RemoteMailbox(m)) if m.kind() == RemoteKind::Shared),
        "fixture decodes as a shared mailbox: {:?}",
        s.recipient
    );
    assert_eq!(both(&st, v, D), (Some(g(1)), Some(g(1))));
}

// 4. An enabled account that is not mail-enabled is not a mailbox, whatever
// SMTP values it still carries.
#[test]
fn an_enabled_non_mail_enabled_user_is_not_a_recipient() {
    let (st, v) = observe(vec![(
        g(1),
        user(Some(true), Some(ObservedRecipient::default())),
    )]);
    let (owner, rcpt) = both(&st, v, D);
    assert_eq!(owner, Some(g(1)));
    assert_eq!(rcpt, None);
}

// 5. Unknown lifecycle: neither the flag nor the recipient attributes were
// read. Not a recipient.
#[test]
fn an_unknown_lifecycle_is_not_a_recipient() {
    let (st, v) = observe(vec![(g(1), user(None, None))]);
    assert_eq!(both(&st, v, D).1, None);
    assert!(!st.view(v).unwrap().is_mail_recipient(&g(1)));
}

// 6. Without Exchange data at all, an enabled account receives — the
// assumption every snapshot without recipient attributes was built on.
#[test]
fn an_enabled_user_without_exchange_data_receives() {
    let (st, v) = observe(vec![(g(1), user(Some(true), None))]);
    assert_eq!(both(&st, v, D), (Some(g(1)), Some(g(1))));
}

// 7. A UPN-only address is a claim, and delivery still follows the holder's
// recipient state, never the UPN alone.
#[test]
fn a_upn_is_an_address_only_of_a_recipient() {
    let mut u = user(Some(true), Some(ObservedRecipient::default()));
    u.primary_smtp = None;
    u.mail = None;
    u.proxies.clear();
    let (st, v) = observe(vec![(g(1), u)]);
    let (owner, rcpt) = both(&st, v, "d.upn@example.org");
    assert_eq!(owner, Some(g(1)));
    assert_eq!(rcpt, None, "a non-mail-enabled UPN delivers nowhere");
}

// 8. Two live recipients claiming one proxy: no recipient is chosen.
#[test]
fn conflicting_proxies_name_no_recipient() {
    let mut other = ObservedNode::user("o.upn@example.org", "o@example.org");
    other.proxies = vec![format!("smtp:{D}")];
    let (st, v) = observe(vec![
        (g(1), user(Some(true), Some(remote_user_mailbox()))),
        (g(2), other),
    ]);
    let key = st.dicts().key_lookup(D).unwrap();
    let view = st.view(v).unwrap();
    assert!(address_owner(&view, key).is_err());
    assert!(address_recipient(&view, key).is_err());
}

// 9. A group with an address receives; one without does not.
#[test]
fn groups_receive_by_their_address() {
    let mut list = ObservedNode::group();
    list.primary_smtp = Some("team@example.org".into());
    let (st, v) = observe(vec![(g(1), list), (g(2), ObservedNode::group())]);
    let view = st.view(v).unwrap();
    assert!(view.is_mail_recipient(&g(1)));
    assert!(!view.is_mail_recipient(&g(2)));
    assert_eq!(both(&st, v, "team@example.org").1, Some(g(1)));
}

// 10. The usual departure keeps the mailbox: login disabled, group
// memberships removed and the mailbox converted to shared (Set-RemoteMailbox
// -Type Shared) so it needs no license. The account stops; the mailbox
// keeps receiving at its addresses.
#[test]
fn departure_by_conversion_to_shared_keeps_the_mailbox() {
    let mut st = VersionStore::new();
    let mut team = ObservedNode::group();
    team.primary_smtp = Some("team@example.org".into());
    let v0 = st
        .observe(
            "lab",
            0,
            Observation {
                scope: SCOPE,
                nodes: vec![
                    (g(1), user(Some(true), Some(remote_user_mailbox()))),
                    (g(2), team),
                ],
                members: vec![(g(1), g(2))],
            },
        )
        .unwrap();
    let before = st
        .view(v0)
        .unwrap()
        .node_state(&g(1))
        .unwrap()
        .recipient
        .unwrap();
    let after = RemoteMailboxOp::SetType(RemoteKind::Shared)
        .apply(&before)
        .unwrap();
    assert_eq!(
        RemoteMailboxOp::between(&before, &after),
        Some(RemoteMailboxOp::SetType(RemoteKind::Shared)),
        "an actuatable step"
    );
    assert_eq!(after.attributes().remote_recipient_type, Some(100));
    let plan = Plan(vec![
        Change::RemoveMembership {
            user: g(1),
            group: g(2),
        },
        Change::SetActive {
            node: g(1),
            from: Some(true),
            to: Some(false),
        },
        Change::SetRecipient {
            node: g(1),
            from: Some(before),
            to: Some(after),
        },
    ]);
    let v1 = st.simulate(v0, &plan, &[]).unwrap();
    let view = st.view(v1).unwrap();
    assert!(validate(&view).is_empty());
    let state = view.node_state(&g(1)).unwrap();
    assert_eq!(state.active, Some(false), "login disabled");
    assert!(!view.is_member(&g(1), &g(2)), "memberships removed");
    assert!(
        matches!(state.recipient, Some(Recipient::RemoteMailbox(m)) if m.kind() == RemoteKind::Shared),
        "the mailbox is now shared: {:?}",
        state.recipient
    );
    assert!(view.is_mail_recipient(&g(1)), "the shared mailbox receives");
    assert_eq!(both(&st, v1, D), (Some(g(1)), Some(g(1))));
}
