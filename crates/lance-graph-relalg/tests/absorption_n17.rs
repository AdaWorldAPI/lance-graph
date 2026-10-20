//! N17 / ISS-RELALG-ABSORPTION-AFTER-DE-MORGAN, through relalg's public
//! `normalize`, which delegates to `lance_graph_contract::set_algebra`.

use lance_graph_relalg::{normalize, Rel};

fn not(r: Rel) -> Rel {
    Rel::Not(Box::new(r))
}

/// FAILS IF: `¬(x ∧ (x ∨ y))` with `x = P1 ∧ P4` keeps the redundant
/// `(¬P1 ∨ ¬P4) ∧ ¬P5 ∧ ¬P7` arm De Morgan emits; it is `¬P1 ∨ ¬P4`.
#[test]
fn absorption_after_de_morgan_is_as_narrow_as_written() {
    let x = Rel::And(vec![Rel::Plane(1), Rel::Plane(4)]);
    let y = Rel::And(vec![Rel::Plane(5), Rel::Plane(7)]);
    let rel = not(Rel::And(vec![x.clone(), Rel::Or(vec![x, y])]));
    assert_eq!(
        normalize(&rel),
        Rel::Or(vec![not(Rel::Plane(1)), not(Rel::Plane(4))])
    );
}

/// FAILS IF: nesting from the parser reaches lowering: `Or(a, Or(b, c))` is
/// one flat `Or` after `normalize`, which `run` applies before the slot check.
#[test]
fn nested_disjunction_is_flat_before_admission() {
    let mut nested = Rel::eq_u32(0, 0);
    for v in 1..8 {
        nested = Rel::Or(vec![nested, Rel::eq_u32(1, v)]);
    }
    let Rel::Or(cs) = normalize(&nested) else {
        panic!("not an Or")
    };
    assert_eq!(cs.len(), 8);
    assert!(cs.iter().all(|c| matches!(c, Rel::Atom(_))));
}
