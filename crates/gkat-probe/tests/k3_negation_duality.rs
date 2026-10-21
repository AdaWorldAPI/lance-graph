//! The SQL-negation / projection boundary of the K3 encoding (#1679).
//!
//! `Cond::Not` is Boolean negation of a guard that is already projected. SQL
//! `NOT` acts on the K3 value before projection. With `Where(p) = valid ∧ leaf`
//! and `Check(p) = ¬valid ∨ leaf` the two meet only through the duality
//!
//! * `Where(NOT p) = NOT Check(p)`
//! * `Check(NOT p) = NOT Where(p)`
//!
//! so `Cond::Not(Where(p))` is SQL `CHECK(NOT p)`, never SQL `WHERE NOT p`;
//! they differ exactly where `p` is UNKNOWN. Checked on all four atoms of one
//! observable: TRUE `(1,1)`, FALSE `(0,1)`, and NULL under both payloads.

use gkat_probe::Atom;
use gkat_probe::flow::{Cond, Kind, Signature};

const P: &str = "p";

fn sig() -> Signature {
    Signature {
        observables: vec![(P, Kind::K3)],
        actions: vec![],
    }
}

/// (leaf, valid) bits: test 0 is the leaf, test 1 its validity.
const T: Atom = Atom(0b11);
const F: Atom = Atom(0b10);
const NULL_0: Atom = Atom(0b00);
const NULL_1: Atom = Atom(0b01);
const ATOMS: [Atom; 4] = [T, F, NULL_0, NULL_1];

/// SQL semantics of the K3 value of `p` on an atom: `None` is UNKNOWN.
fn k3(a: Atom) -> Option<bool> {
    a.holds(1).then(|| a.holds(0))
}

fn eval(c: Cond) -> [bool; 4] {
    let g = sig().guard(&c).expect("guard");
    ATOMS.map(|a| g.eval(a))
}

/// FAILS IF: the translator's negation is taken for SQL negation, or a
/// projection reads a NULL payload. Each side is compared with the SQL
/// meaning computed from `k3`, not with the other side.
#[test]
fn negating_a_projection_is_the_dual_projection_of_the_sql_negation() {
    // SQL meanings, per atom.
    let where_not_p = ATOMS.map(|a| k3(a).map(|b| !b) == Some(true));
    let check_not_p = ATOMS.map(|a| k3(a).map(|b| !b) != Some(false));
    assert_eq!(where_not_p, [false, true, false, false]);
    assert_eq!(check_not_p, [false, true, true, true]);

    // The only spellings of the SQL forms in the flow language.
    assert_eq!(
        eval(Cond::Check(P).negated()),
        where_not_p,
        "Where(NOT p) = NOT Check(p)"
    );
    assert_eq!(
        eval(Cond::Where(P).negated()),
        check_not_p,
        "Check(NOT p) = NOT Where(p)"
    );

    // Boolean NOT of WHERE is not SQL WHERE NOT: they differ on both NULL atoms
    // and nowhere else.
    let not_where = eval(Cond::Where(P).negated());
    let differs: Vec<Atom> = ATOMS
        .iter()
        .zip(not_where.iter().zip(where_not_p.iter()))
        .filter(|(_, (a, b))| a != b)
        .map(|(atom, _)| *atom)
        .collect();
    assert_eq!(differs, vec![NULL_0, NULL_1]);

    // Payload blindness under NULL, for every form above.
    for c in [
        Cond::Where(P),
        Cond::Check(P),
        Cond::Where(P).negated(),
        Cond::Check(P).negated(),
    ] {
        let v = eval(c.clone());
        assert_eq!(v[2], v[3], "{c:?} reads the payload of a NULL");
    }
}
