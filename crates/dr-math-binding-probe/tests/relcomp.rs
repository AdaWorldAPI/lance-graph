//! D-DRMC-5: associativity of Boolean relation composition.
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, Truth, bind_dr64,
    recipe::{Backend, CompiledReference, LawReference, LawShape, Primitive},
    relcomp::{
        CertifyRefusal, Chain, ComposeRefusal, LawRefusal, ProofRefusal, RelExpr, Rows, Step, Work,
        admit_composition_assoc, all_orders, certify_reassociation, compose, lower,
        lowering_digest, prove_equation, prove_reassociation, raise, validate_lowering,
    },
};

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

/// Five distinct domains, so every adjacent pair in a chain is typed.
fn axis(i: u16) -> Axis {
    let kind = [
        AxisKind::Identity,
        AxisKind::Property,
        AxisKind::Community,
        AxisKind::Capability,
        AxisKind::Identity,
        AxisKind::Property,
    ][i as usize];
    Axis {
        class: 10 + i,
        kind,
    }
}
fn method(i: u16) -> MethodSignature {
    MethodSignature::presence(10 + i, i, axis(i), axis(i + 1))
}
fn store() -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    })
}
fn bind<'a>(st: &FrameStore, m: MethodSignature, rows: &'a Rows, focus: Focus) -> BoundDr64<'a> {
    let at = Guards {
        frame: st.seal(),
        focus,
        scope: SCOPE,
    };
    bind_dr64(
        st,
        m,
        m,
        &at,
        (focus, SCOPE),
        Operand {
            rows,
            validity: None,
        },
    )
    .unwrap()
}
/// Deterministic xorshift fill with a density knob (bits kept per 64).
fn fill(seed: u64, keep_of_64: u32) -> Rows {
    let mut x = seed | 1;
    let mut rows = [0u64; 64];
    for r in rows.iter_mut() {
        let mut word = 0u64;
        for b in 0..64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if (x % 64) < u64::from(keep_of_64) {
                word |= 1 << b;
            }
        }
        *r = word;
    }
    rows
}
/// Independent oracle: (a∘b)[i][k] iff some j has a[i][j] and b[j][k],
/// enumerated pair by pair, sharing no code with `compose`.
fn oracle(a: &Rows, b: &Rows) -> Rows {
    let mut out = [0u64; 64];
    for i in 0..64 {
        for k in 0..64 {
            let hit = (0..64).any(|j| (a[i] >> j) & 1 == 1 && (b[j] >> k) & 1 == 1);
            if hit {
                out[i] |= 1 << k;
            }
        }
    }
    out
}
fn oracle_eval(ops: &[Rows], order: &RelExpr) -> Rows {
    match order {
        RelExpr::Leaf(i) => ops[*i],
        RelExpr::Comp(a, b) => oracle(&oracle_eval(ops, a), &oracle_eval(ops, b)),
    }
}
fn left_deep(n: usize) -> RelExpr {
    (1..n).fold(RelExpr::Leaf(0), |acc, i| {
        RelExpr::comp(acc, RelExpr::Leaf(i))
    })
}
fn right_deep(n: usize) -> RelExpr {
    (0..n - 1).rev().fold(RelExpr::Leaf(n - 1), |acc, i| {
        RelExpr::comp(RelExpr::Leaf(i), acc)
    })
}
fn shape() -> LawShape {
    LawShape::for_method(method(0), Primitive::BooleanCompose)
}
fn law() -> LawReference {
    LawReference {
        id: 7,
        revision: 1,
        shape: shape(),
    }
}
/// The compiled reference for evaluating `to`: its digest is the digest of
/// `to`'s lowering, the only digest a certificate accepts.
fn compiled(to: &RelExpr) -> CompiledReference {
    CompiledReference {
        law: law(),
        compiled_digest: lowering_digest(&lower(to)),
        compiler_revision: 1,
        target_abi: 1,
        backend: Backend::ResidentDr64Simd,
    }
}

#[test]
fn the_kernel_matches_pair_enumeration() {
    for (seed, density) in [(1, 3), (2, 8), (3, 32), (4, 63), (5, 0)] {
        let (a, b) = (fill(seed, density), fill(seed * 31 + 7, density));
        let mut w = Work::default();
        assert_eq!(compose(&a, &b, &mut w), oracle(&a, &b), "seed {seed}");
        assert_eq!(
            w.row_ors,
            a.iter().map(|r| u64::from(r.count_ones())).sum::<u64>()
        );
    }
    // Anti-vacuity: the fixtures are neither empty nor full after composing.
    let c = oracle(&fill(1, 3), &fill(38, 3));
    assert!(c.iter().any(|r| *r != 0) && c.iter().any(|r| *r != u64::MAX));
}

/// Every association order of a 4- and a 5-chain computes the oracle's
/// relation, and the fixtures make the orders differ in work.
#[test]
fn every_order_computes_the_same_relation() {
    let st = store();
    for n in [4usize, 5] {
        let ops: Vec<Rows> = (0..n as u64)
            .map(|i| fill(i + 11, 2 + 6 * i as u32))
            .collect();
        let bound = ops
            .iter()
            .enumerate()
            .map(|(i, r)| bind(&st, method(i as u16), r, FOCUS))
            .collect();
        let chain = Chain::new(&st, bound).unwrap();
        let orders = all_orders(n);
        assert_eq!(orders.len(), if n == 4 { 5 } else { 14 });
        let expected = oracle_eval(&ops, &left_deep(n));
        assert!(expected.iter().any(|r| *r != 0) && expected.iter().any(|r| *r != u64::MAX));
        let mut works = Vec::new();
        for o in &orders {
            let (rows, work) = chain.eval(o).unwrap();
            assert_eq!(rows, expected, "order {o:?}");
            works.push(work.row_ors);
        }
        let (lo, hi) = (works.iter().min().unwrap(), works.iter().max().unwrap());
        assert!(
            lo < hi,
            "orders must differ in cost on these fixtures: {works:?}"
        );
    }
}

#[test]
fn a_chain_refuses_mistyped_or_misguarded_operands() {
    let mut st = store();
    let r = fill(1, 8);
    let ok = |i| bind(&st, method(i), &r, FOCUS);
    // Domains: operand 1's row must be operand 0's column.
    let skip = bind(&st, method(2), &r, FOCUS);
    assert_eq!(
        Chain::new(&st, vec![ok(0), skip]).unwrap_err(),
        ComposeRefusal::DomainMismatch { at: 0 }
    );
    assert_eq!(
        Chain::new(&st, vec![]).unwrap_err(),
        ComposeRefusal::EmptyChain
    );
    let other_focus = bind(&st, method(1), &r, Focus { revision: 9 });
    assert_eq!(
        Chain::new(&st, vec![ok(0), other_focus]).unwrap_err(),
        ComposeRefusal::DifferentFocus { at: 1 }
    );
    let other = store();
    let foreign = bind(&other, method(1), &r, FOCUS);
    assert_eq!(
        Chain::new(&st, vec![ok(0), foreign]).unwrap_err(),
        ComposeRefusal::DifferentFrame { at: 1 }
    );
    let (a, b) = (ok(0), ok(1));
    st.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    assert_eq!(
        Chain::new(&st, vec![a, b]).unwrap_err(),
        ComposeRefusal::StaleFrame
    );
}

#[test]
fn a_three_valued_operand_does_not_compose() {
    let st = store();
    let (r, v) = (fill(1, 8), [u64::MAX; 64]);
    let mut m = method(1);
    m.truth = Truth::ThreeValued;
    let at = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    let k3 = bind_dr64(
        &st,
        m,
        m,
        &at,
        (FOCUS, SCOPE),
        Operand {
            rows: &r,
            validity: Some(&v),
        },
    )
    .unwrap();
    assert_eq!(
        Chain::new(&st, vec![bind(&st, method(0), &r, FOCUS), k3]).unwrap_err(),
        ComposeRefusal::NotBooleanDistinct { at: 1 }
    );
}

#[test]
fn an_order_over_other_positions_is_refused() {
    let st = store();
    let r = fill(1, 8);
    let chain = Chain::new(
        &st,
        vec![
            bind(&st, method(0), &r, FOCUS),
            bind(&st, method(1), &r, FOCUS),
        ],
    )
    .unwrap();
    let swapped = RelExpr::comp(RelExpr::Leaf(1), RelExpr::Leaf(0));
    assert_eq!(
        chain.eval(&swapped).unwrap_err(),
        ComposeRefusal::NotThisChain
    );
}

#[test]
fn associativity_is_admitted_only_for_boolean_composition() {
    assert!(admit_composition_assoc(shape()).is_ok());
    let mut other = shape();
    other.primitive = Primitive::Selection;
    assert_eq!(
        admit_composition_assoc(other).unwrap_err(),
        LawRefusal::NotComposition
    );
    for (truth, population) in [
        (
            Truth::ThreeValued,
            dr_math_binding_probe::Population::Distinct,
        ),
        (Truth::TwoValued, dr_math_binding_probe::Population::Bag),
    ] {
        let mut s = shape();
        s.truth = truth;
        s.population = population;
        assert_eq!(
            admit_composition_assoc(s).unwrap_err(),
            LawRefusal::NotBooleanSemiring
        );
    }
}

#[test]
fn egglog_proves_every_reassociation_of_a_five_chain() {
    let law = admit_composition_assoc(shape()).unwrap();
    for to in all_orders(5) {
        let proof = prove_reassociation(&law, &left_deep(5), &to).unwrap();
        assert!(!proof.is_empty());
    }
}

/// Can-it-refuse: commutation is not implied by associativity, so egglog
/// finds no proof even with both terms in the e-graph.
#[test]
fn egglog_refuses_an_equation_the_admitted_rule_does_not_imply() {
    let law = admit_composition_assoc(shape()).unwrap();
    let ab = RelExpr::comp(RelExpr::Leaf(0), RelExpr::Leaf(1));
    let ba = RelExpr::comp(RelExpr::Leaf(1), RelExpr::Leaf(0));
    assert!(matches!(
        prove_equation(&law, &ab.egglog_term(), &ba.egglog_term(), 4),
        Err(ProofRefusal::NotProved(_))
    ));
    assert_eq!(
        prove_reassociation(&law, &ab, &ba).unwrap_err(),
        ProofRefusal::DifferentLeaves
    );
}

#[test]
fn lowering_round_trips_and_keeps_leaf_order() {
    for o in all_orders(5) {
        let steps = lower(&o);
        assert_eq!(steps.iter().filter(|s| **s == Step::Compose).count(), 4);
        assert_eq!(raise(&steps).as_ref(), Some(&o));
        assert!(validate_lowering(&o, &steps).is_ok());
    }
    let wrong = lower(&left_deep(4));
    assert!(validate_lowering(&right_deep(4), &wrong).is_err());
    assert_eq!(raise(&[Step::Compose]), None);
}

#[test]
fn a_reassociation_recipe_gets_a_certificate() {
    let law_ok = admit_composition_assoc(shape()).unwrap();
    let to = right_deep(4);
    let cert = certify_reassociation(&law_ok, law(), &left_deep(4), &to, compiled(&to)).unwrap();
    assert_eq!(cert.law(), &law());
    assert_eq!(
        cert.compiled().compiled_digest,
        lowering_digest(&lower(&to))
    );
    assert!(cert.proof().contains("Comp"));
}

/// The digest binds the certificate to the checked lowering: a nonzero digest
/// of another order, or an arbitrary one, is refused, though the proof and the
/// lowering check both pass.
#[test]
fn a_certificate_is_bound_to_the_checked_lowering() {
    let admitted = admit_composition_assoc(shape()).unwrap();
    let (from, to) = (left_deep(4), right_deep(4));
    // Anti-vacuity: the two orders lower to different digests.
    assert_ne!(lowering_digest(&lower(&from)), lowering_digest(&lower(&to)));
    for digest in [lowering_digest(&lower(&from)), [9; 32]] {
        let mut c = compiled(&to);
        c.compiled_digest = digest;
        assert_eq!(
            certify_reassociation(&admitted, law(), &from, &to, c).unwrap_err(),
            CertifyRefusal::CompiledMismatch
        );
    }
    assert!(certify_reassociation(&admitted, law(), &from, &to, compiled(&to)).is_ok());
}

#[test]
fn certification_refuses_without_each_ingredient() {
    let admitted = admit_composition_assoc(shape()).unwrap();
    // Law reference for another shape than the admitted one.
    let mut foreign = law();
    foreign.shape.row_kind = AxisKind::Capability;
    let mut c = compiled(&right_deep(3));
    c.law = foreign;
    assert!(certify_reassociation(&admitted, foreign, &left_deep(3), &right_deep(3), c).is_err());
    // Not a reassociation: different leaves, no proof.
    let swapped = RelExpr::comp(RelExpr::Leaf(1), RelExpr::Leaf(0));
    let two = RelExpr::comp(RelExpr::Leaf(0), RelExpr::Leaf(1));
    assert!(certify_reassociation(&admitted, law(), &two, &swapped, compiled(&swapped)).is_err());
    // No code identity.
    let mut no_code = compiled(&right_deep(3));
    no_code.compiled_digest = [0; 32];
    assert!(
        certify_reassociation(&admitted, law(), &left_deep(3), &right_deep(3), no_code).is_err()
    );
}
