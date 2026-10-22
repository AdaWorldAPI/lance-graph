//! Proof and target identity come from Stage B. The statement used below is
//! Stage B's E1 substitution problem: it is the only kind of statement a
//! checker exists for. Pairing it with a Boolean-DR64 law shape exercises the
//! admission plumbing; it does not claim that E1 proves anything about DR64.
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, bind_dr64,
    recipe::{
        Backend, Capability, CompiledReference, LawReference, LawShape, Primitive, ProofReference,
        RecipeRefusal, offer_recipe,
    },
};
use egglog_r2il_probe::artifact::ReplayCapsule;
use egglog_r2il_probe::catalogue::{SemanticKey, TargetAbi};
use egglog_r2il_probe::{chain_problem, e1_problem};

const FOCUS: Focus = Focus { revision: 4 };
const SCOPE: Scope = Scope { id: 5 };

fn store() -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 2,
        coordinates: 3,
    })
}
/// Non-empty operand, so a test cannot pass only because the register is zero.
fn rows() -> [u64; 64] {
    let mut rows = [0u64; 64];
    rows[3] = (1 << 5) | (1 << 40);
    rows[60] = 1 << 1;
    rows
}
fn method(class: u16, ordinal: u16, property: bool) -> MethodSignature {
    let kind = if property {
        AxisKind::Property
    } else {
        AxisKind::Identity
    };
    MethodSignature::presence(
        class,
        ordinal,
        Axis {
            class,
            kind: AxisKind::Identity,
        },
        Axis { class, kind },
    )
}
fn bind<'a>(st: &FrameStore, m: MethodSignature, rows: &'a [u64; 64]) -> BoundDr64<'a> {
    let at = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    bind_dr64(
        st,
        m,
        m,
        &at,
        (FOCUS, SCOPE),
        Operand {
            rows,
            validity: None,
        },
    )
    .unwrap()
}
fn law(m: MethodSignature) -> LawReference {
    LawReference {
        id: 41,
        revision: 7,
        shape: LawShape::for_method(m, Primitive::CountViolations),
        statement: SemanticKey::from(&e1_problem()),
    }
}
fn capsule() -> ReplayCapsule {
    ReplayCapsule::certify(&e1_problem(), 8).expect("cold E1 certificate")
}
fn proof(m: MethodSignature) -> ProofReference {
    ProofReference::rechecked(law(m), capsule()).expect("E1 re-checks")
}
fn abi() -> TargetAbi {
    TargetAbi {
        revision: 3,
        target: "r2il-scalar".to_owned(),
    }
}
fn compiled(p: &ProofReference) -> CompiledReference {
    CompiledReference {
        law: p.law().clone(),
        compiled_digest: [18; 32],
        compiler_revision: 2,
        target: abi(),
        backend: Backend::R2ilMaskRisc,
    }
}

#[test]
fn one_law_shape_reuses_two_classes_without_reusing_their_source_binding() {
    let st = store();
    let r = rows();
    let (m1, m2) = (method(1, 2, true), method(22, 8, true));
    let p = proof(m1);
    let c = compiled(&p);
    let a = offer_recipe(
        bind(&st, m1, &r),
        Primitive::CountViolations,
        p.clone(),
        c.clone(),
        &abi(),
        Capability::PureMathematics,
    )
    .unwrap();
    let b = offer_recipe(
        bind(&st, m2, &r),
        Primitive::CountViolations,
        p,
        c,
        &abi(),
        Capability::PureMathematics,
    )
    .unwrap();
    assert_eq!(a.proof.law(), b.proof.law());
    assert_ne!(a.operand.method.owner_class, b.operand.method.owner_class);
    assert!(!a.proof.proof().is_empty());
}

#[test]
fn same_bytes_but_different_axes_cannot_reuse_law() {
    let st = store();
    let r = rows();
    let p = proof(method(1, 2, true));
    let c = compiled(&p);
    assert_eq!(
        offer_recipe(
            bind(&st, method(1, 2, false), &r),
            Primitive::CountViolations,
            p,
            c,
            &abi(),
            Capability::PureMathematics
        )
        .unwrap_err(),
        RecipeRefusal::UnsupportedMathematicalShape
    );
}

/// The forged-digest hole is closed: there is no status flag and no digest
/// field to set, and a real, re-checking proof of a DIFFERENT statement is
/// refused. (Building a `ProofReference` by struct literal does not compile;
/// see the `compile_fail` doctest on the type.)
#[test]
fn a_proof_of_another_statement_is_refused() {
    let mut l = law(method(1, 2, true));
    l.statement = SemanticKey::from(&chain_problem());
    assert_ne!(l.statement, SemanticKey::from(&e1_problem()));
    assert_eq!(
        ProofReference::rechecked(l, capsule()).unwrap_err(),
        RecipeRefusal::ProofOfOtherStatement
    );
}

#[test]
fn changed_law_revision_or_target_refuses_cached_recipe() {
    let st = store();
    let r = rows();
    let m = method(1, 2, true);
    let p = proof(m);
    let mut c = compiled(&p);
    c.law.revision += 1;
    assert_eq!(
        offer_recipe(
            bind(&st, m, &r),
            Primitive::CountViolations,
            p.clone(),
            c,
            &abi(),
            Capability::PureMathematics
        )
        .unwrap_err(),
        RecipeRefusal::LawChanged
    );
    for runtime in [
        TargetAbi {
            revision: 4,
            ..abi()
        },
        TargetAbi {
            target: "resident-dr64-simd".to_owned(),
            ..abi()
        },
    ] {
        assert_eq!(
            offer_recipe(
                bind(&st, m, &r),
                Primitive::CountViolations,
                p.clone(),
                compiled(&p),
                &runtime,
                Capability::PureMathematics
            )
            .unwrap_err(),
            RecipeRefusal::WrongTarget
        );
    }
}

#[test]
fn missing_code_identity_refuses() {
    let st = store();
    let r = rows();
    let m = method(1, 2, true);
    let p = proof(m);
    for broken in [
        CompiledReference {
            compiled_digest: [0; 32],
            ..compiled(&p)
        },
        CompiledReference {
            compiler_revision: 0,
            ..compiled(&p)
        },
    ] {
        assert_eq!(
            offer_recipe(
                bind(&st, m, &r),
                Primitive::CountViolations,
                p.clone(),
                broken,
                &abi(),
                Capability::PureMathematics
            )
            .unwrap_err(),
            RecipeRefusal::MissingCodeIdentity
        );
    }
}

#[test]
fn proof_of_math_never_authorizes_a_cognitive_commit() {
    let st = store();
    let r = rows();
    let m = method(1, 2, true);
    let p = proof(m);
    for kind in [Capability::Proposal, Capability::EffectfulAction] {
        assert_eq!(
            offer_recipe(
                bind(&st, m, &r),
                Primitive::CountViolations,
                p.clone(),
                compiled(&p),
                &abi(),
                kind
            )
            .unwrap_err(),
            RecipeRefusal::EffectRequiresSeparateAuthorization
        );
    }
}
