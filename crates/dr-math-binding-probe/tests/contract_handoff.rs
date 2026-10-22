//! D-DRMC-4: cross-layer *contract* fixture, no write path.
//! The proof is Stage B's re-checked E1 capsule; its statement is an integer
//! substitution, so pairing it with this DR64 law exercises the plumbing only.
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, bind_dr64,
    population::observe_focus,
    recipe::{
        Backend, Capability, CompiledReference, LawReference, LawShape, Primitive, ProofReference,
        RecipeRefusal, offer_recipe,
    },
};
use egglog_r2il_probe::artifact::ReplayCapsule;
use egglog_r2il_probe::catalogue::{SemanticKey, TargetAbi};
use egglog_r2il_probe::e1_problem;
use lance_graph_act_r::{
    Candidate, Refusal as ArbitrationRefusal, Snapshot, arbitrate, witness_matches,
};
use lance_graph_contract::class_view::{ClassId, ClassView, ComputeEdge, compute_dag_topo_order};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};

static COMPUTED_ROLES: [ComputeEdge; 1] = [ComputeEdge {
    target: 1,
    inputs: &[0],
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
    fn template(&self, _: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _: ClassId) -> u8 {
        0
    }
    fn compute_dag(&self, class: ClassId) -> &[ComputeEdge] {
        if class == 7 { &COMPUTED_ROLES } else { &[] }
    }
}
const FOCUS: Focus = Focus { revision: 12 };
const SCOPE: Scope = Scope { id: 13 };

fn signature(method: u16) -> MethodSignature {
    MethodSignature::presence(
        7,
        method,
        Axis {
            class: 7,
            kind: AxisKind::Identity,
        },
        Axis {
            class: 7,
            kind: AxisKind::Property,
        },
    )
}
fn store() -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 9,
        relation_map: 10,
        coordinates: 11,
    })
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
fn abi() -> TargetAbi {
    TargetAbi {
        revision: 3,
        target: "r2il-scalar".to_owned(),
    }
}
fn checked_law() -> ProofReference {
    let law = LawReference {
        id: 42,
        revision: 1,
        shape: LawShape::for_method(signature(1), Primitive::CountViolations),
        statement: SemanticKey::from(&e1_problem()),
    };
    let capsule = ReplayCapsule::certify(&e1_problem(), 8).expect("cold E1 certificate");
    ProofReference::rechecked(law, capsule).expect("E1 re-checks")
}
fn compiled(p: &ProofReference, backend: Backend) -> CompiledReference {
    CompiledReference {
        law: p.law().clone(),
        compiled_digest: [2; 32],
        compiler_revision: 1,
        target: abi(),
        backend,
    }
}

#[test]
fn class_dependency_to_mathematical_offer_to_act_r_proposal_has_no_effect_path() {
    let view = FixtureClass::new();
    assert_eq!(compute_dag_topo_order(view.compute_dag(7)), Some(vec![1]));
    let st = store();
    // Gamma and R are different bits: R holds the selected pair and one more.
    let mut selected = [0u64; 64];
    selected[2] = 1u64 << 17;
    let mut relation = [0u64; 64];
    relation[2] = (1u64 << 17) | (1u64 << 18);
    assert_ne!(selected, relation);
    let g = bind(&st, signature(2), &selected);
    let r = bind(&st, signature(1), &relation);
    let report = observe_focus(&st, &g, &r).unwrap();
    let checked = report.universal_on_focus().unwrap();
    assert_eq!(
        (
            checked.eligible,
            checked.true_pairs,
            checked.false_pairs,
            checked.unknown_pairs
        ),
        (1, 1, 0, 0)
    );

    let proof = checked_law();
    let c = compiled(&proof, Backend::R2ilMaskRisc);
    let offer = offer_recipe(
        r.clone(),
        Primitive::CountViolations,
        proof,
        c,
        &abi(),
        Capability::PureMathematics,
    )
    .unwrap();
    assert_eq!(offer.operand.method, signature(1));

    // ACT-R consumes external eligibility only. No action executor or write.
    // The snapshot version comes from the bound frame, not a second literal.
    let observed = Snapshot {
        thought_address: 77,
        witness_version: r.frame.revisions().source,
    };
    let candidate = Candidate {
        rule_id: 19,
        eligible: checked.false_pairs == 0,
        utility: 0.7,
        activation: 0.4,
        evaluated_at: observed,
    };
    let decision = arbitrate(observed, &[candidate]).unwrap();
    assert_eq!(decision.rule_id, 19);
    assert!(witness_matches(&decision, observed));

    let moved = Snapshot {
        thought_address: 77,
        witness_version: observed.witness_version + 1,
    };
    assert!(matches!(
        arbitrate(moved, &[candidate]),
        Err(ArbitrationRefusal::StaleCandidate { rule_id: 19, .. })
    ));
}

#[test]
fn class_declares_a_dependency_not_permission_to_write() {
    let view = FixtureClass::new();
    assert_eq!(view.compute_dag(7)[0].inputs, &[0]);
    let st = store();
    let mut selected = [0u64; 64];
    selected[2] = 1u64 << 17;
    let b = bind(&st, signature(1), &selected);
    let p = checked_law();
    let c = compiled(&p, Backend::ResidentDr64Simd);
    assert_eq!(
        offer_recipe(
            b,
            Primitive::CountViolations,
            p,
            c,
            &abi(),
            Capability::EffectfulAction
        )
        .unwrap_err(),
        RecipeRefusal::EffectRequiresSeparateAuthorization
    );
}
