//! A history persists as its inputs (`image`) and restores to a store that
//! answers every read the same way (W-3, plan `dir-sim-persist-v1`).

use lance_graph_dir_sim::image::RestoreError;
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::*;

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}
const ALICE: u8 = 0xA1;
const BOB: u8 = 0xB0;
const EMPLOYEES: u8 = 0xE0;
const EXCHANGE: u8 = 0xEC;

const EXCHANGE_ACCESS: RuleId = RuleId {
    name: "ExchangeAccess",
    version: 1,
};
const EMPLOYEES_GET_EXCHANGE: RuleId = RuleId {
    name: "EmployeesGetExchange",
    version: 1,
};
const RENAME_MAIL: RuleId = RuleId {
    name: "RenameMail",
    version: 1,
};

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

fn observed() -> Observation {
    Observation {
        scope: SCOPE,
        nodes: vec![
            (
                g(ALICE),
                ObservedNode::user("alice@example.test", "alice@example.test"),
            ),
            (
                g(BOB),
                ObservedNode::user("bob@example.test", "bob@example.test"),
            ),
            (g(EMPLOYEES), ObservedNode::group()),
            (g(EXCHANGE), ObservedNode::group()),
        ],
        members: vec![(g(ALICE), g(EMPLOYEES)), (g(BOB), g(EMPLOYEES))],
    }
}
fn grant_alice() -> GrantGroup {
    GrantGroup {
        rule: EXCHANGE_ACCESS,
        group: g(EXCHANGE),
        to: vec![g(ALICE)],
    }
}
fn imply() -> ImplyGroup {
    ImplyGroup {
        rule: EMPLOYEES_GET_EXCHANGE,
        source: g(EMPLOYEES),
        target: g(EXCHANGE),
    }
}
fn ev(s: &str) -> Vec<EvidenceRef> {
    vec![EvidenceRef(s.into())]
}
fn chain_from(obs: Observation) -> (VersionStore, VersionId, VersionId, VersionId) {
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 1_000, obs).unwrap();
    let g1 = st.simulate(g0, &grant_alice(), &ev("REQ-1")).unwrap();
    let g2 = st.simulate(g1, &imply(), &ev("POLICY-7")).unwrap();
    (st, g0, g1, g2)
}

// g0 observed; g1, g2 simulated and g2 desired; a value interned between
// versions; g3 rejected (a verdict); g4 a second observation.
fn history() -> (VersionStore, Vec<VersionId>, ValueId) {
    let (mut st, g0, g1, g2) = chain_from(observed());
    st.promote_desired(g2).unwrap();
    let to = st.intern("Alice@Example.test");
    let g3 = st
        .simulate(
            g2,
            &SetPrimarySmtp {
                rule: RENAME_MAIL,
                user: g(BOB),
                to,
            },
            &ev("REQ-2"),
        )
        .unwrap();
    assert!(st.promote_desired(g3).is_err());
    let mut later = observed();
    later.members.pop();
    later.nodes.push((
        g(0xC0),
        ObservedNode::user("carol@example.test", "carol@example.test"),
    ));
    let g4 = st.observe("lab", 2_000, later).unwrap();
    (st, vec![g0, g1, g2, g3, g4], to)
}

fn reads(st: &VersionStore, vs: &[VersionId]) -> String {
    let mut out = String::new();
    for &v in vs {
        out += &format!(
            "{v:?} {:?} {:?} {:?} {:?}\n",
            st.version(v),
            st.diff(vs[0], v),
            st.verdict(v),
            st.view(v).unwrap().members_transitive(&g(EMPLOYEES)),
        );
    }
    out += &format!(
        "{:?} {:?} {:?}\n",
        st.tag(TAG_DESIRED),
        st.tag(TAG_OBSERVED),
        st.plan(vs[2])
    );
    out
}

#[test]
fn a_history_restores_to_the_same_reads() {
    let (st, vs, to) = history();
    let image = st.image();
    assert_eq!(image.versions.len(), 5);
    assert_eq!(image.roots.len(), 2);
    let back = VersionStore::restore(image.clone()).unwrap();
    assert_eq!(reads(&back, &vs), reads(&st, &vs));
    assert_eq!(back.value(to), Some("Alice@Example.test"));
    // The image of the restored store is the image it came from.
    assert_eq!(format!("{:?}", back.image()), format!("{image:?}"));
    // Work continues on the restored store with the same ids.
    let mut back = back;
    assert_eq!(back.intern("Alice@Example.test"), to);
}

#[test]
fn a_damaged_image_is_refused() {
    let (st, _, _) = history();
    let image = st.image();

    let mut i = image.clone();
    i.values.push(i.values[0].clone());
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::DuplicateValue(image.values.len())
    );

    let mut i = image.clone();
    i.values.clear();
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::IncompleteDictionary
    );

    let mut i = image.clone();
    let root = i.roots.remove(1).0;
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::MissingRoot(root)
    );

    let mut i = image.clone();
    i.versions.swap(1, 2);
    assert!(matches!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::VersionOrder { index: 1, .. }
    ));

    let mut i = image.clone();
    i.tags.push(("ghost".into(), VersionId(99)));
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::UnknownVersion(VersionId(99))
    );

    let mut i = image.clone();
    let extra = (VersionId(1), i.roots[0].1.clone());
    i.roots.push(extra);
    assert_eq!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::NotARoot(VersionId(1))
    );

    // A delta that no longer applies to its parent.
    let mut i = image.clone();
    let d = i.versions[1].delta.clone();
    i.versions[2].delta.extend(d);
    assert!(matches!(
        VersionStore::restore(i).unwrap_err(),
        RestoreError::Apply(VersionId(2), _)
    ));
}
