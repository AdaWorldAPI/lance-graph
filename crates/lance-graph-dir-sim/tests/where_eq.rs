//! `WHERE smtp = 'Alice@X.de'` as a contract, not a frontend: the literal is
//! resolved once at the boundary, the executable query holds only numbers,
//! and execution performs no text operation.

use lance_graph_dir_sim::*;
use lance_graph_mask_risc::{MaskOp, Pred};
use lance_graph_quack::bind::{table, BindError};
use lance_graph_quack::lower;
use ogar_dir_core::Guid128;
use ogar_dir_sim::{Attribute, Change, EvidenceRef, NodeState, RuleId};

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
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

#[test]
fn a_text_literal_is_resolved_once_and_execution_is_numeric() {
    let mut st = VersionStore::new();
    let obs = Observation {
        nodes: vec![
            (g(1), ObservedNode::user("a@x.de", "alice@x.de")),
            (g(2), ObservedNode::user("b@x.de", "bob@x.de")),
            (g(3), ObservedNode::user("c@x.de", "ALICE@X.DE")), // same key
            (g(4), ObservedNode::user("d@x.de", "carol@x.de")),
        ],
        ..Observation::default()
    };
    let g0 = st.observe("lab", 0, obs).unwrap();
    // A version that moves one owner off the key, moves another onto it,
    // and creates a third: the query must see base, override and created.
    let carol = st.lookup("carol@x.de");
    let alice2 = st.intern("Alice@X.de");
    let newcomer = st.intern("n@x.de");
    let g1 = st
        .simulate(
            g0,
            &Propose(vec![
                Change::SetAttribute {
                    node: g(3),
                    attribute: Attribute::PrimarySmtp,
                    from: st.view(g0).unwrap().attr(&g(3), Attribute::PrimarySmtp),
                    to: Some(newcomer),
                },
                Change::SetAttribute {
                    node: g(4),
                    attribute: Attribute::PrimarySmtp,
                    from: carol,
                    to: Some(alice2),
                },
                Change::CreateNode {
                    node: g(5),
                    state: NodeState {
                        kind: NodeKind::User,
                        active: Some(true),
                        upn: None,
                        primary_smtp: Some(alice2),
                        dn: None,
                    },
                },
            ]),
            &[],
        )
        .unwrap();
    let view = st.view(g1).unwrap();
    let counters = &view.dicts().counters;

    // 1. Boundary: one text operation turns the literal into a number.
    let before = counters.snapshot();
    let key = view.dicts().key_lookup("Alice@X.de").expect("key observed");
    assert_eq!(counters.snapshot()[1], before[1] + 1, "exactly one lookup");

    // 2. The executable query holds only that number.
    let p = key_eq_program(key);
    let eqs: Vec<u32> = p
        .ops
        .iter()
        .filter_map(|op| match op {
            MaskOp::Pred {
                pred: Pred::EqU32 { v, .. },
                ..
            } => Some(*v),
            _ => None,
        })
        .collect();
    assert_eq!(eqs, vec![key.0]);

    // 3. Execution: no intern, lookup or resolution.
    let at = counters.snapshot();
    let rows = users_with_key(&view, Attribute::PrimarySmtp, key).rows();
    assert_eq!(counters.snapshot(), at, "execution did no text work");
    let owners: Vec<Guid128> = rows
        .into_iter()
        .map(|o| view.user_guid(UserOrdinal(o as u16)).unwrap())
        .collect();
    // g(1) observed; g(3) moved off; g(4) moved on; g(5) created.
    assert_eq!(owners, vec![g(1), g(4), g(5)]);

    // An unknown literal is refused at the boundary, before any program.
    assert_eq!(view.dicts().key_lookup("nobody@x.de"), None);

    // 4. The text form: the generic Quack frontend over the store's own
    // binder lowers to exactly this program, and keeps exactly these rows
    // (base, override and created), with one lookup at bind.
    let ub = UserBinder::new(view.dicts());
    let bound = table("users")
        .where_eq("smtp", "ALICE@x.de")
        .bind(&ub)
        .unwrap();
    assert_eq!(lower(&bound).unwrap(), key_eq_program(key));
    assert_eq!(ub.issued(), Some((Attribute::PrimarySmtp, key)));
    let at = counters.snapshot();
    let by_text = where_eq(&view, "smtp", "ALICE@x.de").unwrap().rows();
    assert_eq!(counters.snapshot()[1], at[1] + 1, "one lookup, at bind");
    assert_eq!(
        by_text,
        users_with_key(&view, Attribute::PrimarySmtp, key).rows()
    );
    // The other key field binds to its own attribute.
    let upn = where_eq(&view, "upn", "B@X.DE").unwrap().rows();
    let upn_owners: Vec<Guid128> = upn
        .into_iter()
        .map(|o| view.user_guid(UserOrdinal(o as u16)).unwrap())
        .collect();
    assert_eq!(upn_owners, vec![g(2)]);
    // Refused at the boundary, in the developer's vocabulary.
    assert!(matches!(
        where_eq(&view, "mail", "alice@x.de"),
        Err(WhereEqError::Bind(BindError::UnknownField { .. }))
    ));
    assert!(matches!(
        where_eq(&view, "smtp", "nobody@x.de"),
        Err(WhereEqError::Bind(BindError::UnknownValue { .. }))
    ));
}
