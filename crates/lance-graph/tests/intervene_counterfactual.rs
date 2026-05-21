// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for Pearl rung-2/3 intervention and counterfactual verbs.
//!
//! Covers:
//! 1. `TripletGraph::intervene_on` returns `CounterfactualSpoG` tagged `Intervention`
//! 2. `intervene_on` does NOT mutate the original graph
//! 3. `NarsInferenceType::Intervention` confidence_modifier returns 0.85
//! 4. `NarsInferenceType::Counterfactual` confidence_modifier returns 0.70
//! 5. `CausalEdge64` roundtrip with `InferenceType::Intervention`
//! 6. `CausalEdge64` roundtrip with `InferenceType::Counterfactual`
//! 7. Pearl rung distinction: Intervention ≠ Counterfactual in the type system
//! 8. Three-step counterfactual chain (abduce → intervene → predict) — structural smoke test

use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use lance_graph::graph::arigraph::triplet_graph::{ContextTag, Triplet, TripletGraph};
use lance_graph_planner::thinking::nars_dispatch::NarsInferenceType;

// ---------------------------------------------------------------------------
// Test 1: intervene_on produces CounterfactualSpoG tagged Intervention
// ---------------------------------------------------------------------------

#[test]
fn intervene_on_produces_counterfactual_spog() {
    let mut g = TripletGraph::new();
    g.add_triplets(&[Triplet::new("patient_a", "healthy", "joint_status", 10)]);

    let cfact = g.intervene_on("patient_a", "joint_status", "improved");

    // Context must be Intervention (Pearl rung 2)
    assert_eq!(
        cfact.context,
        ContextTag::Intervention,
        "intervene_on must tag CounterfactualSpoG as Intervention"
    );

    // The substituted object must equal the supplied new_object
    assert_eq!(
        cfact.triplet.object, "improved",
        "counterfactual triplet new_object must match the do() argument"
    );

    // Subject and predicate must be preserved
    assert_eq!(cfact.triplet.subject, "patient_a");
    assert_eq!(cfact.triplet.relation, "joint_status");
}

// ---------------------------------------------------------------------------
// Test 2: intervene_on does NOT mutate the original graph
// ---------------------------------------------------------------------------

#[test]
fn intervene_does_not_mutate_original_graph() {
    let mut g = TripletGraph::new();
    g.add_triplets(&[
        Triplet::new("alice", "bob", "knows", 1),
        Triplet::new("alice", "ceo", "role", 2),
    ]);

    let triplet_count_before = g.len();

    // Perform an intervention
    let _cfact = g.intervene_on("alice", "role", "engineer");

    // Graph must not have gained any triplets
    assert_eq!(
        g.len(),
        triplet_count_before,
        "intervene_on must not add triplets to the original graph"
    );

    // The existing role triplet must still carry the original object
    let role_triplet = g
        .triplets
        .iter()
        .find(|t| t.subject == "alice" && t.relation == "role")
        .expect("original triplet must still be present");

    assert_eq!(
        role_triplet.object, "ceo",
        "intervene_on must not modify existing triplet objects"
    );
}

// ---------------------------------------------------------------------------
// Test 3: NarsInferenceType::Intervention confidence_modifier = 0.85
// ---------------------------------------------------------------------------

#[test]
fn nars_inference_type_intervention_routes() {
    let modifier = NarsInferenceType::Intervention.confidence_modifier();
    assert!(
        (modifier - 0.85).abs() < f64::EPSILON,
        "Intervention confidence_modifier must be 0.85, got {modifier}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: NarsInferenceType::Counterfactual confidence_modifier = 0.70
// ---------------------------------------------------------------------------

#[test]
fn nars_inference_type_counterfactual_routes() {
    let modifier = NarsInferenceType::Counterfactual.confidence_modifier();
    assert!(
        (modifier - 0.70).abs() < f64::EPSILON,
        "Counterfactual confidence_modifier must be 0.70, got {modifier}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: CausalEdge64 roundtrip with InferenceType::Intervention
// ---------------------------------------------------------------------------

#[test]
fn causal_edge_intervention_roundtrip() {
    let edge = CausalEdge64::pack(
        10,
        20,
        30,
        200,
        200,
        CausalMask::PO, // Level 2: Intervention — P and O active, S projected out
        0,
        InferenceType::Intervention,
        PlasticityState::ALL_FROZEN,
        42,
    );

    // v2 migration: read via signed mantissa per causal-edge 0.2.0 deprecation.
    // `pack(InferenceType::X, ...)` stores `X.to_mantissa()`; reading back via
    // `from_mantissa(inference_mantissa())` round-trips the enum identity
    // (Intervention ↔ +6, Counterfactual ↔ −6). See
    // pr-ce64-mb-2-causaledge64-v2.md §"Signed Mantissa Rationale".
    let decoded = causal_edge::edge::InferenceType::from_mantissa(edge.inference_mantissa());
    assert_eq!(
        decoded,
        InferenceType::Intervention,
        "InferenceType::Intervention must survive pack/unpack roundtrip"
    );

    // Causal mask must also survive
    assert_eq!(edge.causal_mask(), CausalMask::PO);
    assert!(edge.is_interventional());
    assert!(!edge.is_counterfactual());
}

// ---------------------------------------------------------------------------
// Test 6: CausalEdge64 roundtrip with InferenceType::Counterfactual
// ---------------------------------------------------------------------------

#[test]
fn causal_edge_counterfactual_roundtrip() {
    let edge = CausalEdge64::pack(
        10,
        20,
        30,
        200,
        200,
        CausalMask::SPO, // Level 3: Counterfactual — all planes active
        0,
        InferenceType::Counterfactual,
        PlasticityState::ALL_FROZEN,
        99,
    );

    // v2 migration: read via signed mantissa per causal-edge 0.2.0 deprecation.
    // `pack(InferenceType::X, ...)` stores `X.to_mantissa()`; reading back via
    // `from_mantissa(inference_mantissa())` round-trips the enum identity
    // (Intervention ↔ +6, Counterfactual ↔ −6). See
    // pr-ce64-mb-2-causaledge64-v2.md §"Signed Mantissa Rationale".
    let decoded = causal_edge::edge::InferenceType::from_mantissa(edge.inference_mantissa());
    assert_eq!(
        decoded,
        InferenceType::Counterfactual,
        "InferenceType::Counterfactual must survive pack/unpack roundtrip"
    );

    // Causal mask must also survive
    assert_eq!(edge.causal_mask(), CausalMask::SPO);
    assert!(edge.is_counterfactual());
    assert!(!edge.is_interventional());
}

// ---------------------------------------------------------------------------
// Test 7: Pearl rung distinction — Intervention ≠ Counterfactual
// ---------------------------------------------------------------------------

#[test]
fn pearl_rung_distinction() {
    // At the InferenceType layer (causal-edge)
    let i = InferenceType::Intervention;
    let c = InferenceType::Counterfactual;
    assert_ne!(
        i, c,
        "Intervention and Counterfactual must be distinct variants"
    );
    assert_ne!(i as u8, c as u8, "Their discriminants must differ");

    // Intervention is rung 2 (discriminant 5), Counterfactual is rung 3 (discriminant 6)
    assert_eq!(i as u8, 5, "Intervention discriminant must be 5");
    assert_eq!(c as u8, 6, "Counterfactual discriminant must be 6");

    // At the NarsInferenceType layer (lance-graph-planner)
    let ni = NarsInferenceType::Intervention;
    let nc = NarsInferenceType::Counterfactual;
    assert_ne!(ni, nc, "NarsInferenceType variants must be distinct");

    // Confidence modifiers must differ (rung 3 < rung 2)
    assert!(
        nc.confidence_modifier() < ni.confidence_modifier(),
        "Counterfactual modifier ({}) must be less than Intervention ({})",
        nc.confidence_modifier(),
        ni.confidence_modifier()
    );

    // At the ContextTag layer (arigraph) — Intervention tag has raw_g = 0xFF
    assert_eq!(
        ContextTag::Intervention.raw_g(),
        ContextTag::INTERVENTION_RAW_G
    );
    assert_ne!(
        ContextTag::Intervention.raw_g(),
        ContextTag::Observation.raw_g(),
        "Intervention and Observation G-slot bytes must differ"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Three-step counterfactual chain (structural smoke test)
//
// Abduce (observe) → Intervene (do-calculus) → Predict (deduction)
//
// IGNORED for PR-LL-1 — depends on `TripletGraph::infer_deductions` shape
// that doesn't yet match the abduction substrate. The full 3-step chain is
// the target of PR-LL-4 (GRPO trainer + per-step semiring composition);
// PR-LL-1 only lands the *verbs* (Intervention/Counterfactual variants +
// intervene_on). TODO PR-LL-4: unignore + wire abduction path.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "PR-LL-4: full 3-step chain needs abduction substrate wiring"]
fn three_step_counterfactual_chain() {
    // Step 0: build a world with two causal facts
    let mut g = TripletGraph::new();
    g.add_triplets(&[
        Triplet::new("patient_a", "high_risk", "blood_pressure", 1),
        Triplet::new("high_risk", "likely_event", "cardiac_event", 2),
    ]);

    // Step 1 — Abduce: observe background context via 2-hop deduction
    let abduced = g.infer_deductions();
    assert!(
        !abduced.is_empty(),
        "Abduction step must yield at least one inferred triplet"
    );
    // The deduced path: patient_a → high_risk → cardiac_event
    let deduced_chain = abduced
        .iter()
        .find(|t| t.subject == "patient_a" && t.object == "cardiac_event");
    assert!(
        deduced_chain.is_some(),
        "Abduction must derive patient_a → cardiac_event chain"
    );

    // Step 2 — Intervene: do(patient_a, blood_pressure, controlled)
    let cfact = g.intervene_on("patient_a", "blood_pressure", "controlled");
    assert_eq!(cfact.context, ContextTag::Intervention);
    assert_eq!(cfact.triplet.object, "controlled");

    // Step 3 — Predict: in the counterfactual world, build a shadow graph
    // and check that deduction now yields a different outcome
    let mut shadow = g.clone();
    shadow.add_triplets(std::slice::from_ref(&cfact.triplet));

    // The intervention adds a new blood_pressure fact; the original high_risk
    // chain is still present but the counterfactual fact is now in the graph.
    assert!(
        shadow.len() > g.len(),
        "Shadow graph must contain the counterfactual triplet"
    );

    // Verify the counterfactual triple is findable in the shadow graph
    let cf_triple = shadow.triplets.iter().find(|t| {
        t.subject == "patient_a" && t.relation == "blood_pressure" && t.object == "controlled"
    });
    assert!(
        cf_triple.is_some(),
        "Shadow graph must contain the do(blood_pressure = controlled) triple"
    );
}
