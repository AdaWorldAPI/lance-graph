//! P3: grading the cold projection of real ActionDefs onto MethodBinding.

use std::collections::BTreeMap;

use ogar_behavior_probe::projection::{project, Field, Grade, FIELDS};
use ogar_vocab::{ActionDef, KausalSpec};

use crate::common::{basis, class, compiled};

fn action<'a>(actions: &'a [ActionDef], name: &str) -> &'a ActionDef {
    actions
        .iter()
        .find(|a| a.predicate == name)
        .unwrap_or_else(|| panic!("{name} harvested"))
}

#[test]
fn a_single_field_dependency_projects_onto_on() {
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let b = basis(term);
    let p = project(action(&term.actions, "_compute_currency_id"), &b);
    assert_eq!(p.grade(Field::On), Grade::Direct);
    let pos = b
        .position("company_id")
        .expect("company_id is a field of the class");
    assert!(p.on.as_ref().expect("an ON mask").has(pos));
}

#[test]
fn a_dotted_dependency_is_not_representable_in_on() {
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let p = project(
        action(&term.actions, "_compute_example_preview"),
        &basis(term),
    );
    assert_eq!(p.grade(Field::On), Grade::NotRepresentable);
    assert!(p.on.is_none());
}

#[test]
fn a_constrains_trigger_projects_onto_on() {
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let p = project(action(&term.actions, "_check_lines"), &basis(term));
    assert_eq!(p.grade(Field::On), Grade::Direct);
    assert_eq!(p.on.as_ref().unwrap().count(), 2);
}

#[test]
fn a_method_without_a_kausal_fact_has_an_unknown_on() {
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let p = project(action(&term.actions, "copy_data"), &basis(term));
    assert_eq!(p.grade(Field::On), Grade::Unknown);
}

#[test]
fn a_state_guard_is_not_representable_as_a_dt64_guard() {
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let mut a = action(&term.actions, "copy_data").clone();
    a.kausal = Some(KausalSpec::StateGuard {
        guard_field: "active".into(),
        guard_values: vec!["True".into()],
    });
    let p = project(&a, &basis(term));
    assert_eq!(p.grade(Field::When), Grade::NotRepresentable);
}

#[test]
fn inferred_reads_are_never_graded_complete() {
    let c = compiled();
    for cc in &c {
        let b = basis(cc);
        for a in &cc.actions {
            assert_eq!(
                project(a, &b).grade(Field::Reads),
                Grade::Unknown,
                "{}",
                a.identity
            );
        }
    }
}

#[test]
fn two_writes_without_order_are_not_representable_as_an_ordered_do() {
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let mut a = action(&term.actions, "copy_data").clone();
    a.writes = vec!["active".into(), "note".into()];
    assert_eq!(
        project(&a, &basis(term)).grade(Field::EffectOrder),
        Grade::NotRepresentable
    );
    a.writes.truncate(1);
    assert_eq!(
        project(&a, &basis(term)).grade(Field::EffectOrder),
        Grade::SoundlyDerivable
    );
}

#[test]
fn no_harvested_method_yields_a_sound_binding() {
    // The headline P3 result, over every method of the real fixture: each
    // projection has blockers, so a MethodBinding cannot be derived without
    // inventing something. Also prints the per-field grade distribution.
    let c = compiled();
    let mut methods = 0;
    let mut dist: BTreeMap<(Field, Grade), usize> = BTreeMap::new();
    for cc in &c {
        let b = basis(cc);
        for a in &cc.actions {
            methods += 1;
            let p = project(a, &b);
            assert_eq!(p.grades.len(), FIELDS.len());
            assert!(
                !p.blockers().is_empty(),
                "{} projected without blockers",
                a.identity
            );
            assert_eq!(p.grade(Field::EffectAlgebra), Grade::NotRepresentable);
            for gr in &p.grades {
                *dist.entry((gr.field, gr.grade)).or_default() += 1;
            }
        }
    }
    println!("P3 methods graded: {methods}");
    for ((f, gr), n) in &dist {
        println!("P3 {f:?} {gr:?} {n}");
    }
    assert_eq!(methods, 21);
    assert_eq!(dist[&(Field::On, Grade::Direct)], 10);
    assert_eq!(dist[&(Field::EffectTargets, Grade::Direct)], 8);
}

#[test]
fn a_single_segment_name_outside_the_basis_is_not_representable() {
    // Reaches the "not a field of this class" branch; the dotted fixture is
    // refused earlier, by the dotted-path check.
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let mut a = action(&term.actions, "_compute_currency_id").clone();
    a.kausal = Some(KausalSpec::Depends {
        paths: vec!["company_id".into(), "no_such_field".into()],
    });
    let p = project(&a, &basis(term));
    assert_eq!(p.grade(Field::On), Grade::NotRepresentable);
    assert!(p.on.is_none());
}
