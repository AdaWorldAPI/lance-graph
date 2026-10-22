//! Stage B, E1 end to end: premises -> egglog (cold, proofs mode) -> R2IL
//! `INT_MULT` -> value, against an independent i128 oracle.

use egglog_r2il_probe::trace::{HOPS, Verdict, traceroute};
use egglog_r2il_probe::{
    Arith, Compiled, Eq, LawId, LawRefusal, LowerRefusal, Problem, Saturation, Step, chain_problem,
    compile, e1_problem, lower_and_run, lower_steps, oracle, try_prove,
};

/// The c values every agreement test sweeps: zero, units, ordinary, and both
/// edges where 8*c still fits `i64`.
fn c_values() -> Vec<i64> {
    vec![0, 1, -1, 7, -123_456, i64::MAX / 8, i64::MIN / 8]
}

fn ok(c: &Compiled) {
    assert!(c.error.is_none(), "egglog error: {:?}", c.error);
}

fn mults(steps: &[Step]) -> usize {
    steps.iter().filter(|s| matches!(s, Step::IntMult)).count()
}

/// FAILS IF: Exact stops saturating within fuel 8, stops extracting
/// `(Mul (Num 8) (Var "c"))`, loses the proof, or the extraction is not closed.
#[test]
fn exact_e1_extracts_8c_with_a_checked_proof() {
    let c = compile(&e1_problem(), Arith::Exact, 8);
    ok(&c);
    assert!(
        matches!(c.saturation, Saturation::Saturated { .. }),
        "{:?}",
        c.saturation
    );
    assert_eq!(c.extracted, r#"(Mul (Num 8) (Var "c"))"#);
    let proof = c.proof.as_ref().expect("a proof for the extracted term");
    assert!(!proof.is_empty());
    assert!(c.closed);
    assert!(c.refused.is_empty(), "{:?}", c.refused);
    assert_eq!(c.admitted.len(), 2);
}

/// FAILS IF: Exact lowers to anything other than ONE `INT_MULT`, or any value
/// disagrees with the i128 oracle over the original premises.
#[test]
fn exact_e1_lowers_to_one_int_mult_and_agrees_with_oracle() {
    let p = e1_problem();
    let c = compile(&p, Arith::Exact, 8);
    ok(&c);
    let steps = lower_steps(&c).expect("lowerable");
    assert_eq!(mults(&steps), 1, "{steps:?}");
    for v in c_values() {
        let got = lower_and_run(&c, &[("c", v)]).expect("in range");
        let want = oracle::solve(&p, v).expect("oracle answers");
        assert_eq!(i128::from(got), want, "c = {v}");
    }
}

/// FAILS IF: the Checked domain admits re-association or folding (the two
/// must be refused with the Checked reasons), or still reaches `(Num 8)`, or
/// lowers to other than THREE `INT_MULT`s, or disagrees with the oracle.
/// Two-sided with the Exact test above: the domain changes the program.
#[test]
fn checked_domain_refuses_reassociation_and_lowers_three_mults() {
    let p = e1_problem();
    let c = compile(&p, Arith::Checked, 8);
    ok(&c);
    assert!(c.admitted.is_empty(), "{:?}", c.admitted);
    assert!(
        c.refused
            .contains(&(LawId::MulAssoc, LawRefusal::NotAssociative(Arith::Checked)))
    );
    assert!(
        c.refused
            .contains(&(LawId::ConstFold, LawRefusal::NotExactFold(Arith::Checked)))
    );
    assert!(!c.extracted.contains("(Num 8)"), "{}", c.extracted);
    assert!(c.closed, "{}", c.extracted);
    let steps = lower_steps(&c).expect("lowerable");
    assert_eq!(mults(&steps), 3, "{steps:?}");
    for v in c_values() {
        let got = lower_and_run(&c, &[("c", v)]).expect("in range");
        let want = oracle::solve(&p, v).expect("oracle answers");
        assert_eq!(i128::from(got), want, "c = {v}");
    }
}

/// FAILS IF: dropping `b = 2*c` still yields a closed answer, a variant
/// mentions `(Num 8)`, lowering with only `c` stops refusing with an unbound defined variable,
/// the oracle invents an answer, the verdict is not BothRefuse, or egglog
/// proves a claim the premises do not support.
#[test]
fn missing_premise_refuses() {
    let p = Problem {
        premises: vec![
            Eq {
                lhs: "a",
                coeff: 2,
                rhs: "b",
            },
            Eq {
                lhs: "z",
                coeff: 2,
                rhs: "a",
            },
        ],
        goal: "z",
        target_var: "c",
    };
    let c = compile(&p, Arith::Exact, 8);
    ok(&c);
    assert!(
        c.variants.iter().all(|v| !v.contains("(Num 8)")),
        "{:?}",
        c.variants
    );
    assert!(!c.closed, "{}", c.extracted);
    // Which defined name the answer keeps is egglog's tie-break: with
    // `b = 2*c` gone, `(Mul (Num 2) (Def "a"))` and `(Mul (Num 4) (Def "b"))`
    // both cost 1002. Either way the refusal must name a defined variable
    // the target cannot reach, never the target and never a value.
    match lower_and_run(&c, &[("c", 7)]) {
        Err(LowerRefusal::UnboundVar(v)) => {
            assert!(v == "a" || v == "b", "unexpected unbound {v}");
        }
        other => panic!("expected UnboundVar, got {other:?} for {}", c.extracted),
    }
    assert_eq!(oracle::solve(&p, 7), None);
    assert_eq!(
        traceroute(&p, Arith::Exact, 8, 7).verdict,
        Verdict::BothRefuse
    );

    let claim = try_prove(&p, Arith::Exact, 8, r#"(Mul (Num 8) (Var "c"))"#);
    assert!(
        claim.is_err(),
        "egglog proved an unsupported claim: {claim:?}"
    );
    // Paired half: the same call DOES prove a supported claim, so the Err
    // above is the premises' doing and not a broken `try_prove`.
    let supported = try_prove(&p, Arith::Exact, 8, r#"(Mul (Num 4) (Def "b"))"#);
    assert!(supported.is_ok(), "{supported:?}");
}

/// FAILS IF: an overflowing c is answered in either domain instead of
/// `Overflow`, the oracle value stops lying outside `i64` for them, or the
/// boundary c = i64::MAX/8 stops succeeding (two-sided).
#[test]
fn overflow_counterexamples_refuse_in_both_domains() {
    let p = e1_problem();
    let exact = compile(&p, Arith::Exact, 8);
    let checked = compile(&p, Arith::Checked, 8);
    ok(&exact);
    ok(&checked);
    for v in [i64::MAX / 8 + 1, i64::MIN / 8 - 1, i64::MAX] {
        let want = oracle::solve(&p, v).expect("oracle answers in i128");
        assert!(i64::try_from(want).is_err(), "c = {v}: {want} fits i64");
        for c in [&exact, &checked] {
            assert_eq!(
                lower_and_run(c, &[("c", v)]),
                Err(LowerRefusal::Overflow),
                "c = {v} under {:?}",
                c.arith
            );
        }
    }
    let edge = i64::MAX / 8;
    for c in [&exact, &checked] {
        assert_eq!(
            lower_and_run(c, &[("c", edge)]),
            Ok(edge * 8),
            "{:?}",
            c.arith
        );
    }
}

/// FAILS IF: Wrapping stops refusing ConstFold, stops admitting MulAssoc, or
/// a wrapping term is run on R2IL's checked arithmetic instead of refused.
#[test]
fn wrapping_term_is_not_lowered() {
    let c = compile(&e1_problem(), Arith::Wrapping, 8);
    ok(&c);
    assert!(
        c.refused
            .contains(&(LawId::ConstFold, LawRefusal::NotExactFold(Arith::Wrapping)))
    );
    assert!(c.admitted.iter().any(|r| r.law == LawId::MulAssoc));
    assert!(!c.admitted.iter().any(|r| r.law == LawId::ConstFold));
    assert!(matches!(
        lower_and_run(&c, &[("c", 7)]),
        Err(LowerRefusal::NotLowerable(_))
    ));
}

/// FAILS IF: two fresh e-graphs disagree on the extraction, the variant list,
/// or the lowered call list.
#[test]
fn extraction_is_deterministic() {
    let p = e1_problem();
    let runs: Vec<Compiled> = (0..3).map(|_| compile(&p, Arith::Exact, 8)).collect();
    for r in &runs {
        ok(r);
    }
    let steps: Vec<Vec<Step>> = runs
        .iter()
        .map(|r| lower_steps(r).expect("lowerable"))
        .collect();
    for i in 1..3 {
        assert_eq!(runs[0].extracted, runs[i].extracted);
        assert_eq!(runs[0].variants, runs[i].variants);
        assert_eq!(steps[0], steps[i]);
    }
}

/// FAILS IF: running out of fuel is reported as saturation, or a long chain
/// stops saturating given room (two-sided: Incomplete at fuel 1, Saturated at
/// fuel 16). Uses the six-link chain so one step cannot finish.
#[test]
fn bounded_fuel_reports_incomplete() {
    let p = chain_problem();
    let tight = compile(&p, Arith::Exact, 1);
    ok(&tight);
    assert_eq!(tight.saturation, Saturation::Incomplete { fuel: 1 });
    let roomy = compile(&p, Arith::Exact, 16);
    ok(&roomy);
    assert!(
        matches!(roomy.saturation, Saturation::Saturated { .. }),
        "{:?}",
        roomy.saturation
    );
    assert_eq!(roomy.extracted, r#"(Mul (Num 64) (Var "p6"))"#);
}

/// FAILS IF: the traceroute drops, renames or reorders a hop.
#[test]
fn traceroute_names_every_hop() {
    let t = traceroute(&e1_problem(), Arith::Exact, 8, 7);
    let text = t.to_string();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), HOPS.len(), "{text}");
    for (i, label) in HOPS.iter().enumerate() {
        assert!(
            lines[i].starts_with(&format!("{}. {}:", i + 1, label)),
            "line {i}: {}",
            lines[i]
        );
    }
    assert_eq!(t.verdict, Verdict::Agree);
}
