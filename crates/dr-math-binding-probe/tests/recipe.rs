use dr_math_binding_probe::{
    bind_dr64, Axis, AxisKind, BindingWitness, MethodSignature,
    recipe::{
        offer_recipe, Backend, Capability, CompiledReference, LawReference,
        LawShape, Primitive, ProofDisposition, ProofReference, RecipeRefusal,
    },
};
fn witness() -> BindingWitness {
    BindingWitness { source_revision:1, relation_map_revision:2,coordinate_revision:3,
        focus_revision:4,authorization_scope:5 }
}
fn method(class:u16,ordinal:u16,property:bool) -> MethodSignature {
    let kind = if property { AxisKind::Property } else { AxisKind::Identity };
    MethodSignature::presence(class,ordinal,
        Axis { class, kind:AxisKind::Identity },
        Axis { class, kind })
}
fn proof(m:MethodSignature) -> ProofReference {
    ProofReference {
        law: LawReference { id:41, revision:7,
            shape:LawShape::for_method(m,Primitive::CountViolations) },
        checker_revision:1, proof_digest:[17;32],
        status:ProofDisposition::CheckedByExternalAuthority,
    }
}
fn compiled(p:ProofReference) -> CompiledReference {
    CompiledReference { law:p.law, compiled_digest:[18;32],
        compiler_revision:2,target_abi:3,backend:Backend::R2ilMaskRisc }
}
#[test]
fn one_law_shape_reuses_two_classes_without_reusing_their_source_binding() {
    let rows=[0u64;64];
    let m1=method(1,2,true);
    let m2=method(22,8,true);
    let a=bind_dr64(m1,m1,witness(),witness(),&rows).unwrap();
    let b=bind_dr64(m2,m2,witness(),witness(),&rows).unwrap();
    let p=proof(m1);
    let c=compiled(p);
    let aa=offer_recipe(a,Primitive::CountViolations,p,c,3,Capability::PureMathematics).unwrap();
    let bb=offer_recipe(b,Primitive::CountViolations,p,c,3,Capability::PureMathematics).unwrap();
    assert_eq!(aa.proof.law,bb.proof.law);
    assert_ne!(aa.operand.method.owner_class,bb.operand.method.owner_class);
}
#[test]
fn same_bytes_but_different_axes_cannot_reuse_law() {
    let rows=[0u64;64];
    let id2=method(1,2,false);
    let bound=bind_dr64(id2,id2,witness(),witness(),&rows).unwrap();
    let p=proof(method(1,2,true));
    assert_eq!(offer_recipe(bound,Primitive::CountViolations,p,compiled(p),3,Capability::PureMathematics)
        .unwrap_err(),RecipeRefusal::UnsupportedMathematicalShape);
}
#[test]
fn unverified_conjecture_is_not_admitted_by_a_bare_flag_free_path() {
    let rows=[0u64;64];let m=method(1,2,true);
    let bound=bind_dr64(m,m,witness(),witness(),&rows).unwrap();
    let mut p=proof(m);p.status=ProofDisposition::Candidate;
    assert_eq!(offer_recipe(bound,Primitive::CountViolations,p,compiled(p),3,Capability::PureMathematics)
        .unwrap_err(),RecipeRefusal::UncheckedLaw);
}
#[test]
fn changed_law_revision_or_target_refuses_cached_recipe() {
    let rows=[0u64;64];let m=method(1,2,true);
    let bound=bind_dr64(m,m,witness(),witness(),&rows).unwrap();
    let p=proof(m);let mut c=compiled(p);c.law.revision+=1;
    assert_eq!(offer_recipe(bound,Primitive::CountViolations,p,c,3,Capability::PureMathematics)
        .unwrap_err(),RecipeRefusal::LawChanged);
    let c=compiled(p);
    assert_eq!(offer_recipe(bound,Primitive::CountViolations,p,c,4,Capability::PureMathematics)
        .unwrap_err(),RecipeRefusal::WrongTarget);
}
#[test]
fn missing_proof_and_code_identities_refuse() {
    let rows=[0u64;64];let m=method(1,2,true);
    let bound=bind_dr64(m,m,witness(),witness(),&rows).unwrap();
    let mut p=proof(m);p.proof_digest=[0;32];
    assert_eq!(offer_recipe(bound,Primitive::CountViolations,p,compiled(p),3,Capability::PureMathematics)
        .unwrap_err(),RecipeRefusal::MissingProofIdentity);
    let p=proof(m);let mut c=compiled(p);c.compiled_digest=[0;32];
    assert_eq!(offer_recipe(bound,Primitive::CountViolations,p,c,3,Capability::PureMathematics)
        .unwrap_err(),RecipeRefusal::MissingCodeIdentity);
}
#[test]
fn proof_of_math_never_authorizes_a_cognitive_commit() {
    let rows=[0u64;64];let m=method(1,2,true);
    let bound=bind_dr64(m,m,witness(),witness(),&rows).unwrap();
    let p=proof(m);
    for kind in [Capability::Proposal,Capability::EffectfulAction] {
        assert_eq!(offer_recipe(bound,Primitive::CountViolations,p,compiled(p),3,kind)
            .unwrap_err(),RecipeRefusal::EffectRequiresSeparateAuthorization);
    }
}
