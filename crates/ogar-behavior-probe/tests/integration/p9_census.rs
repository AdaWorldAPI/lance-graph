//! P9: coverage census over every ActionDef producer that runs in this lab.

use ogar_behavior_probe::census::{classify, tally, Coverage};
use ogar_vocab::ActionDef;

use crate::common::compiled;

fn sources() -> Vec<(&'static str, Vec<ActionDef>)> {
    let odoo: Vec<ActionDef> = compiled().into_iter().flat_map(|c| c.actions).collect();
    vec![
        (
            "ogar-vocab::document_actions",
            ogar_vocab::document_actions::document_actions(),
        ),
        (
            "ogar-vocab::geo_actions",
            ogar_vocab::geo_actions::geo_actions(),
        ),
        (
            "ogar-vocab::healthcare_actions",
            ogar_vocab::healthcare_actions::healthcare_actions(),
        ),
        ("ogar-from-ruff (Odoo account_payment_term.py)", odoo),
    ]
}

#[test]
fn no_actiondef_is_counted_as_having_a_program() {
    for (name, actions) in sources() {
        for a in &actions {
            assert_ne!(
                classify(a),
                Coverage::ProgramAndAction,
                "{name}: {}",
                a.identity
            );
        }
    }
}

#[test]
fn census_tally() {
    let mut total = 0;
    for (name, actions) in sources() {
        let t = tally(&actions);
        total += t.methods;
        println!(
            "P9 {name}: methods={} cat1={} cat2={} kausal={} body_source={} reads={} writes={} on_enter={}",
            t.methods,
            t.by_category[0],
            t.by_category[1],
            t.with_kausal,
            t.with_body_source,
            t.with_reads,
            t.with_writes,
            t.with_on_enter
        );
        let expected = match name {
            "ogar-vocab::document_actions" => 3,
            "ogar-vocab::geo_actions" => 6,
            "ogar-vocab::healthcare_actions" => 20,
            _ => 21,
        };
        assert_eq!(
            t.methods, expected,
            "{name}: producer output moved; re-measure"
        );
        assert_eq!(t.by_category[0], 0);
        assert_eq!(t.by_category[1], t.methods);
    }
    println!("P9 total ActionDefs: {total}");
    assert_eq!(total, 50);
}
