//! The Quack frontend over IAM's label/value store. Two fields bind the same
//! attribute under different identity contracts — `smtp` by comparison key
//! (`KeyId`, case-insensitive), `smtp_exact` by exact value (`ValueId`) —
//! and the generic frontend must keep them apart even though both end as
//! `EqU32`. The key form must bind to exactly the program dir-sim runs.

use lance_graph_dir_sim::*;
use lance_graph_quack::bind::{table, Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{lower, Col, Mask, Query};
use ogar_dir_core::Guid128;

/// Lane 0 = SMTP key ids, lane 1 = SMTP value ids; plane 0 = candidates.
struct IamBinder<'a>(&'a Dicts);

impl Binder for IamBinder<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == "users").then_some(TableId(0))
    }
    fn live(&self, _: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, _: TableId, name: &str) -> Option<BoundField> {
        let col = match name {
            "smtp" => Col(0),
            "smtp_exact" => Col(1),
            _ => return None,
        };
        Some(BoundField {
            col,
            kind: FieldKind::Code,
            // IAM's dictionary ids are never NULL here.
            validity: None,
        })
    }
    fn code(&self, _: TableId, col: Col, literal: &str) -> Option<u32> {
        match col {
            Col(0) => self.0.key_lookup(literal).map(|k| k.0),
            _ => self.0.lookup(literal).map(|v| v.0),
        }
    }
}

fn eq(field: &str, value: &str, b: &IamBinder<'_>) -> Query {
    table("users").where_eq(field, value).bind(b).unwrap()
}

#[test]
fn exact_value_and_comparison_key_bind_under_their_own_contracts() {
    let mut st = VersionStore::new();
    let obs = Observation {
        nodes: vec![
            (Guid128([1; 16]), ObservedNode::user("a@x.de", "alice@x.de")),
            (Guid128([2; 16]), ObservedNode::user("b@x.de", "Alice@X.de")),
        ],
        ..Observation::default()
    };
    st.observe("lab", 0, obs).unwrap();
    let b = IamBinder(st.dicts());
    let lookups = || b.0.counters.snapshot()[1];

    let before = lookups();
    let by_key = eq("smtp", "alice@x.de", &b);
    assert_eq!(lookups(), before + 1, "one lookup, at bind");
    // Comparison form: both spellings are one key.
    assert_eq!(eq("smtp", "ALICE@x.DE", &b), by_key);
    // Exact form: the two spellings are two values.
    assert_ne!(
        eq("smtp_exact", "alice@x.de", &b),
        eq("smtp_exact", "Alice@X.de", &b)
    );
    // The key-bound query is exactly the program dir-sim executes.
    let key = b.0.key_lookup("alice@x.de").unwrap();
    assert_eq!(lower(&by_key).unwrap(), key_eq_program(key));
}
