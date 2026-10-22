//! D-DRMC-4: cross-layer *contract* fixture, no proof engine or write path.
//! The checked proof is a synthetic external receipt, NOT a real verified law.
use dr_math_binding_probe::{
    bind_dr64, Axis, AxisKind, BindingWitness, MethodSignature,
    population::observe_focus,
    recipe::{
        offer_recipe, Backend, Capability, CompiledReference, LawReference,
        LawShape, Primitive, ProofDisposition, ProofReference,
    },
};
use lance_graph_contract::class_view::{
    ClassId, ClassView, ComputeEdge, compute_dag_topo_order,
};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_act_r::{
    arbitrate, witness_matches, Candidate, Refusal as ArbitrationRefusal, Snapshot,
};

static COMPUTED_ROLES: [ComputeEdge; 1] = [ComputeEdge {
    target: 1, inputs: &[0],
}];
struct FixtureClass {
    fields: [FieldRef; 2],
}
impl FixtureClass {
    fn new() -> Self {
        Self {
            fields: [
                FieldRef::new("urn:fixture:memberOf", "memberOf"),
                FieldRef::new("urn:fixture:effectiveRoles", "effectiveRoles"),
            ],
        }
    }
}
impl ClassView for FixtureClass {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == 7 { &self.fields } else { &[] }
    }
    fn template(&self, _: ClassId) -> DisplayTemplate { DisplayTemplate::Detail }
    fn dolce_category_id(&self, _: ClassId) -> u8 { 0 }
    fn compute_dag(&self, class: ClassId) -> &[ComputeEdge] {
        if class == 7 { &COMPUTED_ROLES } else { &[] }
    }
}
fn signature(method: u16) -> MethodSignature {
    MethodSignature::presence(7, method,
        Axis { class:7, kind:AxisKind::Identity },
        Axis { class:7, kind:AxisKind::Property })
}
fn witness() -> BindingWitness {
    BindingWitness { source_revision:9, relation_map_revision:10,
        coordinate_revision:11, focus_revision:12, authorization_scope:13 }
}
fn checked_fixture_law() -> ProofReference {
    ProofReference {
        law:LawReference { id:42, revision:1, shape:LawShape::for_method(
            signature(1),Primitive::CountViolations) },
        checker_revision:1, proof_digest:[1;32],
        status:ProofDisposition::CheckedByExternalAuthority,
    }
}
#[test]
fn class_dependency_to_mathematical_offer_to_act_r_proposal_has_no_effect_path() {
    let view=FixtureClass::new();
    assert_eq!(compute_dag_topo_order(view.compute_dag(7)),Some(vec![1]));
    let mut selected=[0u64;64];
    selected[2]=1u64<<17;
    let m=signature(1);
    let g=bind_dr64(m,m,witness(),witness(),&selected).unwrap();
    let r=bind_dr64(m,m,witness(),witness(),&selected).unwrap();
    let report=observe_focus(&g,&r,None,witness()).unwrap();
    let checked=report.universal_on_focus().unwrap();
    assert_eq!((checked.eligible,checked.false_pairs,checked.unknown_pairs),(1,0,0));

    let proof=checked_fixture_law();
    let compiled=CompiledReference {
        law:proof.law,compiled_digest:[2;32],compiler_revision:1,
        target_abi:3,backend:Backend::R2ilMaskRisc,
    };
    let offer=offer_recipe(r,Primitive::CountViolations,proof,compiled,3,
        Capability::PureMathematics).unwrap();
    assert_eq!(offer.operand.method,signature(1));

    // ACT-R consumes external eligibility only. No action executor or write.
    let observed=Snapshot { thought_address:77,witness_version:9 };
    let candidate=Candidate {
        rule_id:19,eligible:checked.false_pairs==0,
        utility:0.7,activation:0.4,evaluated_at:observed,
    };
    let decision=arbitrate(observed,&[candidate]).unwrap();
    assert_eq!(decision.rule_id,19);
    assert!(witness_matches(&decision,observed));

    let moved=Snapshot { thought_address:77,witness_version:10 };
    assert!(matches!(
        arbitrate(moved,&[candidate]),
        Err(ArbitrationRefusal::StaleCandidate {rule_id:19,..})
    ));
}
#[test]
fn class_declares_a_dependency_not_permission_to_write() {
    let view=FixtureClass::new();
    assert_eq!(view.compute_dag(7)[0].inputs,&[0]);
    let mut selected=[0u64;64];
    selected[2]=1u64<<17;
    let m=signature(1);
    let b=bind_dr64(m,m,witness(),witness(),&selected).unwrap();
    let p=checked_fixture_law();
    let c=CompiledReference { law:p.law,compiled_digest:[2;32],
        compiler_revision:1,target_abi:3,backend:Backend::ResidentDr64Simd };
    assert!(offer_recipe(b,Primitive::CountViolations,p,c,3,
        Capability::EffectfulAction).is_err());
}
