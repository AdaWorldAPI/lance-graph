//! Entra users read through Graph, shown as an Active Directory emulation:
//! a synchronized user is mirrored at its on-premises OU, a cloud-only user
//! is placed under the synthetic container.

use lance_graph_dir_sim::observe::{from_graph, ObserveError, SYNTHETIC_OU};
use ogar_dir_core::{DirectoryScope, Dn128, Guid128, OuDictionary, ValuePool};

const TENANT: &str = "c0ffee00-1234-4abc-8def-000000000042";
const HYBRID: &str = "7a3c9e11-2b44-4c6d-9e8f-0123456789ab";
const CLOUD: &str = "7a3c9e11-2b44-4c6d-9e8f-0123456789ac";

fn g(s: &str) -> Guid128 {
    Guid128::parse(s).unwrap()
}

fn user(id: &str, extra: &str) -> String {
    format!(r#"{{"id":"{id}","userPrincipalName":"{id}@example.de"{extra}}}"#)
}

fn ingest(
    users: &[String],
    dict: &mut OuDictionary,
    pool: &mut ValuePool,
) -> Vec<ogar_dir_core::DirRecord> {
    let body = format!(r#"{{"value":[{}]}}"#, users.join(","));
    ogar_az::ingest_page(&body, g(TENANT), dict, pool, 0)
        .unwrap()
        .records
}

const HYBRID_DN: &str =
    r#","onPremisesDistinguishedName":"CN=Erika,OU=Exchange,OU=Infrastructure,DC=example,DC=de""#;

#[test]
fn a_synchronized_user_is_mirrored_and_a_cloud_only_user_is_synthetic() {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let recs = ingest(
        &[
            user(
                HYBRID,
                &format!(
                    r#"{HYBRID_DN},"accountEnabled":true,"mail":"erika@example.de","mailNickname":"emueller","proxyAddresses":["smtp:e@example.mail.onmicrosoft.com","SMTP:erika@example.de"]"#
                ),
            ),
            user(CLOUD, r#","accountEnabled":false"#),
        ],
        &mut dict,
        &mut pool,
    );
    let out = from_graph(DirectoryScope(g(TENANT)), &recs, &pool, &mut dict).unwrap();
    assert_eq!(out.synthetic, vec![g(CLOUD)]);
    let node = |id| {
        &out.observation
            .nodes
            .iter()
            .find(|(n, _)| *n == g(id))
            .unwrap()
            .1
    };

    // Mirrored: the same location the AD object of this user has.
    let ad = dict.intern(&["Infrastructure", "Exchange"]).unwrap();
    let h = node(HYBRID);
    assert_eq!(h.dn, Some(Dn128::from_ou_hhtl(&ad).unwrap()));
    assert_eq!(h.active, Some(true));
    assert_eq!(h.primary_smtp.as_deref(), Some("erika@example.de"));
    assert_eq!(
        h.proxies,
        vec!["smtp:e@example.mail.onmicrosoft.com".to_string()]
    );
    assert_eq!(h.mail.as_deref(), Some("erika@example.de"));
    assert_eq!(h.alias.as_deref(), Some("emueller"));
    assert_eq!(h.recipient, None);
    assert_eq!(h.exchange_guid, None);

    // Synthetic: under the marked container, and the container explains as such.
    let syn = dict.intern(&[SYNTHETIC_OU]).unwrap();
    let c = node(CLOUD);
    assert_eq!(c.dn, Some(Dn128::from_ou_hhtl(&syn).unwrap()));
    assert_ne!(c.dn, h.dn);
    assert_eq!(dict.explain(&syn), Some(vec![SYNTHETIC_OU.to_string()]));
    assert_eq!(c.active, Some(false));
}

// Absent accountEnabled is unknown, never enabled.
#[test]
fn an_unread_account_flag_stays_unknown() {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let recs = ingest(&[user(CLOUD, "")], &mut dict, &mut pool);
    let out = from_graph(DirectoryScope(g(TENANT)), &recs, &pool, &mut dict).unwrap();
    assert_eq!(out.observation.nodes[0].1.active, None);
}

// A real OU named like the synthetic container would make the two
// placements indistinguishable: refused.
#[test]
fn a_mirrored_user_inside_the_synthetic_container_is_refused() {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let dn = format!(
        r#","onPremisesDistinguishedName":"CN=x,OU=Sub,OU={SYNTHETIC_OU},DC=example,DC=de""#
    );
    let recs = ingest(&[user(HYBRID, &dn)], &mut dict, &mut pool);
    assert_eq!(
        from_graph(DirectoryScope(g(TENANT)), &recs, &pool, &mut dict).unwrap_err(),
        ObserveError::SyntheticClash { node: g(HYBRID) }
    );
}

// A synchronized user whose DN could not be encoded is never demoted to
// the synthetic container.
#[test]
fn an_unencoded_on_premises_dn_is_refused_not_made_synthetic() {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let recs = ingest(
        &[user(HYBRID, r#","onPremisesDistinguishedName":"not a dn""#)],
        &mut dict,
        &mut pool,
    );
    assert_eq!(
        from_graph(DirectoryScope(g(TENANT)), &recs, &pool, &mut dict).unwrap_err(),
        ObserveError::UnplacedMirror { node: g(HYBRID) }
    );
}

#[test]
fn a_record_of_another_tenant_is_refused() {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let recs = ingest(&[user(CLOUD, "")], &mut dict, &mut pool);
    let other = g("c0ffee00-1234-4abc-8def-000000000043");
    assert!(matches!(
        from_graph(DirectoryScope(other), &recs, &pool, &mut dict),
        Err(ObserveError::ForeignScope { .. })
    ));
}

// Mailbox records carry other slots; they are not users.
#[test]
fn mailbox_records_are_not_observed_as_users() {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let mbx = ogar_az::mailbox::encode_mailbox(
        &ogar_az::mailbox::MailboxBodies {
            user: g(CLOUD),
            mailbox_settings: Some(r#"{"userPurpose":"user"}"#),
            exchange_settings: None,
        },
        g(TENANT),
        &mut pool,
        0,
    )
    .unwrap();
    let out = from_graph(DirectoryScope(g(TENANT)), &[mbx], &pool, &mut dict).unwrap();
    assert!(out.observation.nodes.is_empty());
    assert!(out.synthetic.is_empty());
}
