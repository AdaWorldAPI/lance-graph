//! Stage D: certified cache hit, explicit witness fallbacks, no hot egglog.
use egglog_r2il_probe::artifact::ReplayCapsule;
use egglog_r2il_probe::catalogue::{
    BindingWitness, Catalogue, CostEvidence, FallbackReason, PromotionRefusal, Route, TargetAbi,
};
use egglog_r2il_probe::{LowerRefusal, e1_problem};

fn bind() -> BindingWitness {
    BindingWitness {
        source_generation: 1,
        scope: 42,
        coordinate_generation: 9,
    }
}

fn abi() -> TargetAbi {
    TargetAbi {
        revision: 1,
        target: "r2il-checked-i64".into(),
    }
}

fn measured() -> CostEvidence {
    // Synthetic evidence for testing the policy, NOT measured performance.
    CostEvidence {
        baseline_ns: 300,
        candidate_ns: 100,
        compile_and_certify_ns: 1_000,
        expected_reuse: 10,
    }
}

fn trusted(x: i64) -> Result<i64, LowerRefusal> {
    x.checked_mul(2)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(2))
        .ok_or(LowerRefusal::Overflow)
}

#[test]
fn certified_hit_uses_saved_r2il_and_zero_baseline_calls() {
    let mut cache = Catalogue::new();
    let p = e1_problem();
    let compiled = ReplayCapsule::certify(&p, 8).expect("cold E1");
    cache.promote(compiled, bind(), abi(), measured()).unwrap();
    assert_eq!(cache.len(), 1);
    let hit = cache.execute_or(&p, bind(), &abi(), 7, |_| panic!("baseline on hit"));
    assert_eq!(hit.route, Route::Certified);
    assert_eq!(hit.result, Ok(56));
}

#[test]
fn no_certificate_is_installed_without_economic_gate() {
    let mut cache = Catalogue::new();
    let mut losing = measured();
    losing.candidate_ns = 500;
    let c = ReplayCapsule::certify(&e1_problem(), 8).expect("cold E1");
    assert_eq!(
        cache.promote(c, bind(), abi(), losing),
        Err(PromotionRefusal::NotAmortized)
    );
    assert!(cache.is_empty());
}

#[test]
fn changed_coordinates_target_and_semantics_call_baseline() {
    let mut cache = Catalogue::new();
    let p = e1_problem();
    cache
        .promote(
            ReplayCapsule::certify(&p, 8).unwrap(),
            bind(),
            abi(),
            measured(),
        )
        .unwrap();

    let mut stale = bind();
    stale.source_generation += 1;
    let out = cache.execute_or(&p, stale, &abi(), 7, trusted);
    assert_eq!(out.route, Route::Baseline(FallbackReason::BindingChanged));
    assert_eq!(out.result, Ok(56));

    let other = TargetAbi {
        revision: 2,
        target: "r2il-checked-i64".into(),
    };
    let out = cache.execute_or(&p, bind(), &other, 7, trusted);
    assert_eq!(out.route, Route::Baseline(FallbackReason::TargetChanged));

    let mut altered = p.clone();
    altered.premises[0].coeff = 3;
    let out = cache.execute_or(&altered, bind(), &abi(), 7, |_| Ok(-123));
    assert_eq!(out.route, Route::Baseline(FallbackReason::CacheMiss));
    assert_eq!(out.result, Ok(-123));
}

#[test]
fn overflow_must_fallback_without_manufacturing_an_answer() {
    let mut cache = Catalogue::new();
    let p = e1_problem();
    cache
        .promote(
            ReplayCapsule::certify(&p, 8).unwrap(),
            bind(),
            abi(),
            measured(),
        )
        .unwrap();
    let out = cache.execute_or(&p, bind(), &abi(), i64::MAX, trusted);
    assert_eq!(out.route, Route::Baseline(FallbackReason::CompiledRefused));
    assert_eq!(out.result, Err(LowerRefusal::Overflow));
}
