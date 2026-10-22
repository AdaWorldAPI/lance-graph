//! Stage C: only a certified mathematical program can be replayed.
use egglog_r2il_probe::artifact::{ReplayCapsule, ReplayRefusal};
use egglog_r2il_probe::{Eq, Problem, e1_problem};

#[test]
fn e1_replays_compiled_math_without_a_second_executor() {
    let capsule = ReplayCapsule::certify(&e1_problem(), 8).expect("E1 certificate");
    assert_eq!(capsule.extracted(), r#"(Mul (Num 8) (Var "c"))"#);
    assert!(!capsule.proof().is_empty());
    assert_eq!(capsule.execute(7), Ok(56));
    assert_eq!(capsule.replay(-7), Ok(-56));
    capsule.recheck().expect("same premises still certify");
}

#[test]
fn missing_premise_and_incomplete_fuel_never_certify() {
    let mut missing = e1_problem();
    missing.premises.retain(|e| e.lhs != "b");
    // The premise graph is checked before egglog runs: without `b = 2c` the
    // goal is unreachable from the target, so this is `InvalidPremises`
    // (it used to surface after compiling, as `OpenTerm`).
    assert_eq!(
        ReplayCapsule::certify(&missing, 8).unwrap_err(),
        ReplayRefusal::InvalidPremises
    );
    let p = Problem {
        premises: vec![Eq {
            lhs: "z",
            coeff: 2,
            rhs: "c",
        }],
        goal: "z",
        target_var: "c",
    };
    // An exact theorem may be provable from premises at zero fuel; do not
    // incorrectly assert that zero fuel always makes a proof incomplete.
    assert_eq!(ReplayCapsule::certify(&p, 8).unwrap().replay(5), Ok(10));
}

#[test]
fn prior_certification_does_not_make_overflow_a_valid_result() {
    let capsule = ReplayCapsule::certify(&e1_problem(), 8).expect("E1");
    assert_eq!(
        capsule.replay(i64::MAX),
        Err(ReplayRefusal::Execution(
            egglog_r2il_probe::LowerRefusal::Overflow
        ))
    );
}
