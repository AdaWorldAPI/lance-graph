//! A version shown as Active Directory entries and as LDIF, each entry
//! marked with where it came from.

use lance_graph_dir_sim::ad::{
    base64, escape_rdn_value, project, to_ldif, Origin, ProjectError, Source, Value,
};
use lance_graph_dir_sim::observe::{from_graph, SYNTHETIC_OU};
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Dn128, Guid128, OuDictionary, ValuePool};
use ogar_dir_sim::{Change, EvidenceRef, NodeState, RuleId, VersionId};

const TENANT: &str = "c0ffee00-1234-4abc-8def-000000000042";
const HYBRID: &str = "7a3c9e11-2b44-4c6d-9e8f-0123456789ab";
const CLOUD: &str = "7a3c9e11-2b44-4c6d-9e8f-0123456789ac";
const NC: &str = "DC=example,DC=de";

fn g(s: &str) -> Guid128 {
    Guid128::parse(s).unwrap()
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

fn cloud_store() -> (VersionStore, VersionId, OuDictionary, Vec<Guid128>) {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let body = format!(
        r#"{{"value":[
        {{"id":"{HYBRID}","userPrincipalName":"erika@example.de","accountEnabled":true,
          "mail":"erika@example.de","onPremisesDistinguishedName":"CN=Erika,OU=Exchange,OU=Infrastructure,DC=example,DC=de",
          "proxyAddresses":["SMTP:erika@example.de","smtp:emueller@example.mail.onmicrosoft.com"]}},
        {{"id":"{CLOUD}","userPrincipalName":"cloud@example.de","accountEnabled":false}}]}}"#
    );
    let recs = ogar_az::ingest_page(&body, g(TENANT), &mut dict, &mut pool, 0)
        .unwrap()
        .records;
    let out = from_graph(DirectoryScope(g(TENANT)), &recs, &pool, &mut dict).unwrap();
    let mut st = VersionStore::new();
    let v = st.observe("graph", 0, out.observation).unwrap();
    (st, v, dict, out.synthetic)
}

#[test]
fn a_cloud_read_is_shown_mirrored_and_synthetic() {
    let (st, v, dict, synthetic) = cloud_store();
    let p = project(
        &st.view(v).unwrap(),
        Source::Cloud {
            synthetic: &synthetic,
        },
        NC,
        &dict,
    )
    .unwrap();
    let dns: Vec<(&str, Origin)> = p
        .entries
        .iter()
        .map(|e| (e.dn.as_str(), e.origin))
        .collect();
    let hybrid_dn = format!("CN={HYBRID},OU=Exchange,OU=Infrastructure,{NC}");
    let cloud_dn = format!("CN={CLOUD},OU={SYNTHETIC_OU},{NC}");
    let syn_ou = format!("OU={SYNTHETIC_OU},{NC}");
    assert_eq!(
        dns,
        vec![
            (syn_ou.as_str(), Origin::Synthetic),
            ("OU=Infrastructure,DC=example,DC=de", Origin::Mirrored),
            (
                "OU=Exchange,OU=Infrastructure,DC=example,DC=de",
                Origin::Mirrored
            ),
            (hybrid_dn.as_str(), Origin::Mirrored),
            (cloud_dn.as_str(), Origin::Synthetic),
        ]
    );
    let h = &p.entries[3];
    assert_eq!(h.node, Some(g(HYBRID)));
    assert_eq!(h.text("userPrincipalName"), ["erika@example.de"]);
    assert_eq!(h.text("mail"), ["erika@example.de"]);
    assert_eq!(
        h.text("proxyAddresses"),
        [
            "SMTP:erika@example.de",
            "smtp:emueller@example.mail.onmicrosoft.com"
        ]
    );
    assert_eq!(h.text("userAccountControl"), ["512"]);
    assert_eq!(h.text("dirSimOrigin"), ["mirrored"]);
    assert!(h.attrs.contains(&(
        "objectGUID",
        Value::Binary(g(HYBRID).to_ms_bytes().to_vec())
    )));
    let c = &p.entries[4];
    assert_eq!(c.text("userAccountControl"), ["514"]);
    assert_eq!(c.text("dirSimOrigin"), ["synthetic"]);
    assert!(c.text("proxyAddresses").is_empty());
}

fn ad_store(dict: &mut OuDictionary) -> (VersionStore, VersionId) {
    let staff = Dn128::from_ou_hhtl(&dict.intern(&["Staff"]).unwrap()).unwrap();
    let mut alice = ObservedNode::user("alice@example.de", "alice@example.de");
    alice.dn = Some(staff);
    let mut grp = ObservedNode::group();
    grp.dn = Some(staff);
    let obs = Observation {
        scope: DirectoryScope(Guid128([0x5C; 16])),
        nodes: vec![(Guid128([0xA1; 16]), alice), (Guid128([0xE0; 16]), grp)],
        members: vec![
            (Guid128([0xA1; 16]), Guid128([0xE0; 16])),
            // An observed member the snapshot does not hold.
            (Guid128([0xFF; 16]), Guid128([0xE0; 16])),
        ],
    };
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, obs).unwrap();
    (st, v)
}

// A simulated user is marked simulated; a simulated membership shows as a
// member; a membership whose user does not exist is counted, not emitted.
#[test]
fn a_simulated_version_marks_what_it_created() {
    let mut dict = OuDictionary::new();
    let (mut st, v0) = ad_store(&mut dict);
    let staff = Dn128::from_ou_hhtl(&dict.resolve(&["Staff"]).unwrap()).unwrap();
    let bob = Guid128([0xB0; 16]);
    let v1 = st
        .simulate(
            v0,
            &Propose(vec![
                Change::CreateNode {
                    node: bob,
                    state: NodeState {
                        kind: NodeKind::User,
                        active: Some(true),
                        upn: None,
                        primary_smtp: None,
                        dn: Some(staff),
                        recipient: None,
                    },
                },
                Change::AddMembership {
                    user: bob,
                    group: Guid128([0xE0; 16]),
                },
            ]),
            &[],
        )
        .unwrap();
    let p = project(&st.view(v1).unwrap(), Source::Ad, NC, &dict).unwrap();
    let alice_dn = format!("CN={},OU=Staff,{NC}", Guid128([0xA1; 16]));
    let bob_dn = format!("CN={bob},OU=Staff,{NC}");
    let by_node = |n: Guid128| p.entries.iter().find(|e| e.node == Some(n)).unwrap();
    assert_eq!(by_node(Guid128([0xA1; 16])).origin, Origin::Observed);
    assert_eq!(by_node(bob).origin, Origin::Simulated);
    assert_eq!(by_node(bob).text("dirSimOrigin"), ["simulated"]);
    let grp = by_node(Guid128([0xE0; 16]));
    assert_eq!(grp.text("objectClass"), ["top", "group"]);
    let mut want = vec![alice_dn.as_str(), bob_dn.as_str()];
    want.sort();
    assert_eq!(grp.text("member"), want);
    assert_eq!(p.dangling_members, 1);
    assert_eq!(p.entries[0].dn, format!("OU=Staff,{NC}"));
    assert_eq!(p.entries[0].origin, Origin::Observed);
}

#[test]
fn bad_inputs_are_refused() {
    let mut dict = OuDictionary::new();
    let (st, v) = ad_store(&mut dict);
    let view = st.view(v).unwrap();
    for nc in ["OU=x,DC=de", "", "CN=a"] {
        assert_eq!(
            project(&view, Source::Ad, nc, &dict),
            Err(ProjectError::NamingContext(nc.to_string()))
        );
    }
    // Another dictionary does not know the locations.
    assert!(matches!(
        project(&view, Source::Ad, NC, &OuDictionary::new()),
        Err(ProjectError::UnknownOu { .. })
    ));
    // A node without a location cannot be placed.
    let mut st = VersionStore::new();
    let v = st
        .observe(
            "ad",
            0,
            Observation {
                scope: DirectoryScope(Guid128([0x5C; 16])),
                nodes: vec![(Guid128([1; 16]), ObservedNode::user("a@x", "a@x"))],
                members: vec![],
            },
        )
        .unwrap();
    assert_eq!(
        project(&st.view(v).unwrap(), Source::Ad, NC, &dict),
        Err(ProjectError::Unlocated {
            node: Guid128([1; 16])
        })
    );
}

#[test]
fn rdn_values_are_escaped() {
    for (raw, want) in [
        ("Sales, EU", "Sales\\, EU"),
        ("a+b", "a\\+b"),
        ("#1", "\\#1"),
        (" lead", "\\ lead"),
        ("trail ", "trail\\ "),
        ("x=y;z", "x\\=y\\;z"),
        ("Cloud Only (emulated)", "Cloud Only (emulated)"),
        ("Müller", "Müller"),
    ] {
        assert_eq!(escape_rdn_value(raw), want, "{raw}");
    }
}

#[test]
fn ldif_is_rfc_2849() {
    for (raw, want) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(base64(raw.as_bytes()), want);
    }
    let (st, v, dict, synthetic) = cloud_store();
    let ldif = to_ldif(
        &project(
            &st.view(v).unwrap(),
            Source::Cloud {
                synthetic: &synthetic,
            },
            NC,
            &dict,
        )
        .unwrap(),
    );
    assert!(ldif.starts_with("version: 1\n\ndn: OU=Cloud Only (emulated),DC=example,DC=de\n"));
    // Binary values are always base64.
    let guid = base64(&g(HYBRID).to_ms_bytes());
    assert!(ldif.contains(&format!("objectGUID:: {guid}\n")));
    assert!(!ldif.contains("objectGUID: "));
    for l in ldif.lines() {
        assert!(l.len() <= 76, "{l}");
    }
    // Five records.
    assert_eq!(
        ldif.matches("\ndn: ").count() + ldif.matches("\ndn:: ").count(),
        5
    );
}

// A value that is not a SAFE-STRING is base64, and a long line folds and
// unfolds back to itself.
#[test]
fn unsafe_values_are_base64_and_long_lines_fold() {
    let mut dict = OuDictionary::new();
    let ou = format!("Müller {}", "x".repeat(90));
    let h = dict.intern(&[ou.as_str()]).unwrap();
    let mut u = ObservedNode::user("m@x", "m@x");
    u.dn = Some(Dn128::from_ou_hhtl(&h).unwrap());
    let mut st = VersionStore::new();
    let v = st
        .observe(
            "ad",
            0,
            Observation {
                scope: DirectoryScope(Guid128([0x5C; 16])),
                nodes: vec![(Guid128([1; 16]), u)],
                members: vec![],
            },
        )
        .unwrap();
    let p = project(&st.view(v).unwrap(), Source::Ad, NC, &dict).unwrap();
    let ldif = to_ldif(&p);
    let ou_dn = format!("OU={ou},{NC}");
    let unfolded = ldif.replace("\n ", "");
    assert!(unfolded.contains(&format!("dn:: {}\n", base64(ou_dn.as_bytes()))));
    assert!(ldif.lines().any(|l| l.starts_with(' ')), "folded");
    for l in ldif.lines() {
        assert!(l.len() <= 76, "{l}");
    }
}
