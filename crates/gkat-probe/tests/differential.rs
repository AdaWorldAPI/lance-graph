//! Differential check of the automaton against the independent oracle over
//! random expressions and single-point mutations of them.
//!
//! The direction of each assertion follows what a bounded oracle can show:
//! automaton-equivalent ⇒ oracle-equal at every bound; oracle-different ⇒
//! automaton-different. The count of automaton-different pairs the oracle
//! also separates is asserted non-trivial so the check cannot pass vacuously.

use gkat_probe::oracle::same_up_to;
use gkat_probe::{Action, Atoms, BExp, Exp, equivalent};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const ACTIONS: [Action; 2] = [Action("p"), Action("q")];

fn bexp(r: &mut Rng, depth: u32) -> BExp {
    match if depth == 0 { r.below(3) } else { r.below(6) } {
        0 => BExp::Test(0),
        1 => BExp::Test(1),
        2 => [BExp::True, BExp::False][r.below(2) as usize].clone(),
        3 => bexp(r, depth - 1).negated(),
        4 => bexp(r, depth - 1).and(bexp(r, depth - 1)),
        _ => bexp(r, depth - 1).or(bexp(r, depth - 1)),
    }
}

fn exp(r: &mut Rng, depth: u32) -> Exp {
    match if depth == 0 { r.below(4) } else { r.below(7) } {
        0 => Exp::Skip,
        1 => Exp::Fail,
        2 => Exp::Assert(bexp(r, 1)),
        3 => Exp::Do(ACTIONS[r.below(2) as usize]),
        4 => exp(r, depth - 1).seq(exp(r, depth - 1)),
        5 => Exp::if_(bexp(r, 1), exp(r, depth - 1), exp(r, depth - 1)),
        _ => Exp::while_(bexp(r, 1), exp(r, depth - 1)),
    }
}

/// One random local change: flip an action label, negate a guard, or swap
/// the branches of a conditional.
fn mutate(e: &Exp, r: &mut Rng) -> Exp {
    match e {
        Exp::Do(p) if r.below(2) == 0 => Exp::Do(if *p == ACTIONS[0] {
            ACTIONS[1]
        } else {
            ACTIONS[0]
        }),
        Exp::If(b, x, y) => match r.below(3) {
            0 => Exp::If(b.clone().negated(), x.clone(), y.clone()),
            1 => Exp::If(b.clone(), Box::new(mutate(x, r)), y.clone()),
            _ => Exp::If(b.clone(), x.clone(), Box::new(mutate(y, r))),
        },
        Exp::Seq(x, y) => {
            if r.below(2) == 0 {
                Exp::Seq(Box::new(mutate(x, r)), y.clone())
            } else {
                Exp::Seq(x.clone(), Box::new(mutate(y, r)))
            }
        }
        Exp::While(b, x) => {
            if r.below(2) == 0 {
                Exp::While(b.clone().negated(), x.clone())
            } else {
                Exp::While(b.clone(), Box::new(mutate(x, r)))
            }
        }
        other => other.clone(),
    }
}

#[test]
fn automaton_and_oracle_agree_on_random_pairs() {
    let atoms = Atoms::new(2, &BExp::True).unwrap();
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut equal, mut separated, mut different) = (0, 0, 0);
    for _ in 0..600 {
        let e = exp(&mut r, 3);
        let f = if r.below(2) == 0 {
            mutate(&e, &mut r)
        } else {
            exp(&mut r, 3)
        };
        let v = equivalent(&e, &f, &atoms).unwrap().holds();
        let oracle = same_up_to(&e, &f, &atoms, 3);
        if v {
            equal += 1;
            assert!(
                oracle,
                "automaton equivalent, oracle differs:\n{e:?}\n{f:?}"
            );
            for n in 0..=4 {
                assert!(same_up_to(&e, &f, &atoms, n), "bound {n}:\n{e:?}\n{f:?}");
            }
        } else {
            different += 1;
            if !oracle {
                separated += 1;
            }
        }
        // Reflexivity: every expression is equivalent to itself.
        assert!(equivalent(&e, &e, &atoms).unwrap().holds());
    }
    eprintln!("equivalent {equal}, different {different}, oracle-separated {separated}");
    // Anti-vacuity: both verdicts occur, and the oracle confirms most
    // differences within its bound.
    assert!(equal > 50, "{equal}");
    assert!(different > 50, "{different}");
    assert!(separated * 10 >= different * 8, "{separated}/{different}");
}

/// Oracle-different ⇒ automaton-different, checked from the oracle's side.
#[test]
fn every_oracle_difference_is_an_automaton_difference() {
    let atoms = Atoms::new(2, &BExp::True).unwrap();
    let mut r = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..600 {
        let e = exp(&mut r, 3);
        let f = mutate(&e, &mut r);
        if !same_up_to(&e, &f, &atoms, 3) {
            assert!(
                !equivalent(&e, &f, &atoms).unwrap().holds(),
                "\n{e:?}\n{f:?}"
            );
        }
    }
}

/// A terminating stretch of more silent steps than any fixed budget is not a
/// divergence (Codex review, #1676): a balanced sequence of 2^14 `skip`s is
/// equivalent to `skip`, and the oracle must agree at bound 0.
#[test]
fn long_silent_stretches_are_not_divergence() {
    fn skips(depth: u32) -> Exp {
        if depth == 0 {
            Exp::Skip
        } else {
            skips(depth - 1).seq(skips(depth - 1))
        }
    }
    let atoms = Atoms::new(1, &BExp::True).unwrap();
    let e = skips(14);
    assert!(equivalent(&e, &Exp::Skip, &atoms).unwrap().holds());
    let x = gkat_probe::oracle::explore(&e, &atoms, 0);
    assert_eq!(x.diverged, 0);
    assert!(same_up_to(&e, &Exp::Skip, &atoms, 0));
    // A genuinely silent loop still diverges.
    let spin = gkat_probe::oracle::explore(&Exp::while_(BExp::True, Exp::Skip), &atoms, 0);
    assert_eq!(spin.diverged, atoms.len());
}
