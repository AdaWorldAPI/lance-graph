//! The native fixpoint against the Clingo goldens, and the falsifiers.

use std::fs;
use std::path::PathBuf;

use lance_graph_contract::class_view::WideFieldMask;
use mistaber_kripke_probe::{Golden, Kb, Projection, Refusal, parse_golden};

fn goldens() -> Vec<Golden> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens");
    let mut out: Vec<Golden> = fs::read_dir(&dir)
        .expect("goldens/")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
        .map(|p| parse_golden(&fs::read_to_string(p).expect("read")).expect("parse"))
        .collect();
    out.sort_by(|a, b| a.fixture_id.cmp(&b.fixture_id));
    out
}

fn golden(id: &str) -> Golden {
    goldens()
        .into_iter()
        .find(|g| g.fixture_id == id)
        .expect(id)
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

#[test]
fn every_golden_is_reproduced_exactly() {
    let all = goldens();
    // Anti-vacuity: the ten exported fixtures, not an empty directory.
    assert_eq!(all.len(), 10);
    for g in &all {
        let kb = Kb::load(&g.facts).expect(&g.fixture_id);
        let got = kb.solve().materialize_atoms(Projection::Set);
        assert_eq!(got, sorted(g.atoms.clone()), "{}", g.fixture_id);
    }
}

#[test]
fn the_goldens_are_not_trivial() {
    // Every derived predicate appears somewhere, and inheritance happens.
    let atoms: Vec<String> = goldens().into_iter().flat_map(|g| g.atoms).collect();
    for pred in [
        "holds(",
        "holds_via(inherited(",
        "holds_via_rule(inherited(",
        "holding_strength(weak",
        "inheritance_source(",
        "overridden(",
    ] {
        assert!(atoms.iter().any(|a| a.starts_with(pred)), "{pred}");
    }
}

#[test]
fn orientation_is_load_bearing() {
    let a = golden("F1-modes");
    let b = golden("F1-reversed");
    let (ka, kb) = (Kb::load(&a.facts).unwrap(), Kb::load(&b.facts).unwrap());
    let sa = ka.solve().materialize_atoms(Projection::Set);
    let sb = kb.solve().materialize_atoms(Projection::Set);
    assert!(sa.contains(&"holds(p,a)".to_owned()));
    assert!(!sb.contains(&"holds(p,a)".to_owned()));
}

#[test]
fn a_cycle_supports_nothing_and_terminates() {
    let g = golden("F2-cycle");
    let kb = Kb::load(&g.facts).unwrap();
    let h = kb.solve();
    let got = h.materialize_atoms(Projection::Set);
    let held: Vec<_> = got.iter().filter(|a| a.starts_with("holds(")).collect();
    assert_eq!(held, ["holds(t,z)"]);
    assert!(h.steps <= 4, "{} steps", h.steps);
}

#[test]
fn set_projection_is_required_on_the_diamond() {
    // p holds in d by inherited(a) and inherited(b): two lens rows, one fact.
    let g = golden("F1-modes");
    let kb = Kb::load(&g.facts).unwrap();
    let h = kb.solve();
    assert_eq!(h.bag_multiplicity("p", "d"), 2);
    assert_eq!(h.bag_multiplicity("p", "a"), 1);
    let sel = WideFieldMask::from_positions(&[0, 1]);
    let set = h.materialize_selection(&sel, Projection::Set).unwrap();
    let bag = h.materialize_selection(&sel, Projection::Bag).unwrap();
    let count = |v: &[String]| v.iter().filter(|a| *a == "holds(p,d)").count();
    assert_eq!(count(&set), 1);
    assert_eq!(count(&bag), 2);
    // The bag reading does not reproduce the golden.
    assert_ne!(h.materialize_atoms(Projection::Bag), sorted(g.atoms));
}

#[test]
fn bag_and_set_agree_where_no_holding_has_two_modes() {
    // Silence twin: on a chain without a diamond the two projections coincide.
    let g = golden("C-mammal-C4");
    let kb = Kb::load(&g.facts).unwrap();
    let h = kb.solve();
    assert_eq!(
        h.materialize_atoms(Projection::Bag),
        h.materialize_atoms(Projection::Set)
    );
}

#[test]
fn a_fact_outside_the_kripke_inputs_is_refused() {
    let facts = vec![vec![
        "is_a".to_owned(),
        "monotreme".to_owned(),
        "mammal".to_owned(),
    ]];
    assert!(matches!(Kb::load(&facts), Err(Refusal::UnknownFact(k)) if k == "is_a"));
    let facts = vec![vec!["asserts".to_owned(), "w".to_owned()]];
    assert!(matches!(Kb::load(&facts), Err(Refusal::Arity(_))));
}

#[test]
fn a_65th_world_is_refused_not_wrapped() {
    let facts: Vec<Vec<String>> = (0..65)
        .map(|i| vec!["world".to_owned(), format!("w{i}")])
        .collect();
    assert_eq!(Kb::load(&facts[..64]).err(), None);
    assert!(matches!(Kb::load(&facts), Err(Refusal::TooMany("world"))));
}

#[test]
fn a_selection_that_is_no_predicate_is_refused() {
    let g = golden("F1-modes");
    let kb = Kb::load(&g.facts).unwrap();
    let h = kb.solve();
    for f in [&[0u8][..], &[0, 2], &[0, 1, 3], &[0, 1, 2, 4]] {
        let sel = WideFieldMask::from_positions(f);
        assert_eq!(
            h.materialize_selection(&sel, Projection::Set),
            Err(Refusal::Selection),
            "{f:?}"
        );
    }
}
