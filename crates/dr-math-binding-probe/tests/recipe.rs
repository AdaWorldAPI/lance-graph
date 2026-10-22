//! No certified-compiler boundary exists for a Boolean-DR64 law, so every
//! admission that passes the metadata checks ends in `NoCertifiedCompiler`.
//! The tests pin that each earlier check refuses on its own, and that nothing
//! (no flag, no digest, no finite-population report) produces a certificate.
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, bind_dr64,
    recipe::{
        Backend, Capability, CompiledReference, LawReference, LawShape, Primitive, RecipeRefusal,
        admit_recipe,
    },
};

const FOCUS: Focus = Focus { revision: 4 };
const SCOPE: Scope = Scope { id: 5 };
const ABI: u32 = 3;

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
    }
}
fn compiled(l: LawReference) -> CompiledReference {
    CompiledReference {
        law: l,
        compiled_digest: [18; 32],
        compiler_revision: 2,
        target_abi: ABI,
        backend: Backend::R2ilMaskRisc,
    }
}
fn admit(
    st: &FrameStore,
    m: MethodSignature,
    l: LawReference,
    c: CompiledReference,
    runtime_abi: u32,
    capability: Capability,
) -> RecipeRefusal {
    let r = rows();
    admit_recipe(
        bind(st, m, &r),
        Primitive::CountViolations,
        l,
        c,
        runtime_abi,
        capability,
    )
    .unwrap_err()
}

/// One law key serves two classes (the key holds no ClassId); both pass every
/// metadata check and stop at certification, which does not exist for DR64.
#[test]
fn one_law_key_spans_two_classes_and_neither_is_certified() {
    let st = store();
    let (m1, m2) = (method(1, 2, true), method(22, 8, true));
    assert_eq!(law(m1), law(m2));
    assert_ne!(m1.owner_class, m2.owner_class);
    for m in [m1, m2] {
        assert_eq!(
            admit(
                &st,
                m,
                law(m1),
                compiled(law(m1)),
                ABI,
                Capability::PureMathematics
            ),
            RecipeRefusal::NoCertifiedCompiler
        );
    }
}

#[test]
fn same_bytes_but_different_axes_cannot_reuse_law() {
    let st = store();
    let l = law(method(1, 2, true));
    assert_eq!(
        admit(
            &st,
            method(1, 2, false),
            l,
            compiled(l),
            ABI,
            Capability::PureMathematics
        ),
        RecipeRefusal::UnsupportedMathematicalShape
    );
}

#[test]
fn changed_law_revision_or_target_refuses_before_certification() {
    let st = store();
    let m = method(1, 2, true);
    let mut c = compiled(law(m));
    c.law.revision += 1;
    assert_eq!(
        admit(&st, m, law(m), c, ABI, Capability::PureMathematics),
        RecipeRefusal::LawChanged
    );
    assert_eq!(
        admit(
            &st,
            m,
            law(m),
            compiled(law(m)),
            ABI + 1,
            Capability::PureMathematics
        ),
        RecipeRefusal::WrongTarget
    );
}

#[test]
fn missing_code_identity_refuses() {
    let st = store();
    let m = method(1, 2, true);
    for broken in [
        CompiledReference {
            compiled_digest: [0; 32],
            ..compiled(law(m))
        },
        CompiledReference {
            compiler_revision: 0,
            ..compiled(law(m))
        },
    ] {
        assert_eq!(
            admit(&st, m, law(m), broken, ABI, Capability::PureMathematics),
            RecipeRefusal::MissingCodeIdentity
        );
    }
}

#[test]
fn proof_of_math_never_authorizes_a_cognitive_commit() {
    let st = store();
    let m = method(1, 2, true);
    for kind in [Capability::Proposal, Capability::EffectfulAction] {
        assert_eq!(
            admit(&st, m, law(m), compiled(law(m)), ABI, kind),
            RecipeRefusal::EffectRequiresSeparateAuthorization
        );
    }
}
