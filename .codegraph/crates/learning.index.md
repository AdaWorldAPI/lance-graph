# learning

crate: learning
path: crates/learning
workspace: excluded
digest: df68f779a8369953
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph-contract normal — crates/learning/Cargo.toml:17 CONFIRMED
- dep ndarray normal — crates/learning/Cargo.toml:16 CONFIRMED

## Depended on by


## Public items

- struct IceCakedLayer [pub] — crates/learning/src/blackboard.rs:8 CONFIRMED
- struct Decision [pub] — crates/learning/src/blackboard.rs:36 CONFIRMED
- struct TaskState [pub] — crates/learning/src/blackboard.rs:47 CONFIRMED
- struct ConsciousnessState [pub] — crates/learning/src/blackboard.rs:55 CONFIRMED
- struct Blackboard [pub] — crates/learning/src/blackboard.rs:63 CONFIRMED
- fn Blackboard::new [pub] — crates/learning/src/blackboard.rs:78 CONFIRMED
- fn Blackboard::update_from_session [pub] — crates/learning/src/blackboard.rs:104 CONFIRMED
- fn Blackboard::record_decision [pub] — crates/learning/src/blackboard.rs:112 CONFIRMED
- fn Blackboard::add_ice_cake [pub] — crates/learning/src/blackboard.rs:125 CONFIRMED
- fn Blackboard::record_file_modified [pub] — crates/learning/src/blackboard.rs:131 CONFIRMED
- fn Blackboard::add_next_step [pub] — crates/learning/src/blackboard.rs:137 CONFIRMED
- fn Blackboard::to_yaml [pub] — crates/learning/src/blackboard.rs:141 CONFIRMED
- fn Blackboard::to_json [pub] — crates/learning/src/blackboard.rs:145 CONFIRMED
- fn Blackboard::handover_summary [pub] — crates/learning/src/blackboard.rs:149 CONFIRMED
- enum OpResult [pub] — crates/learning/src/cam_ops.rs:35 CONFIRMED
- fn OpResult::to_fingerprint [pub] — crates/learning/src/cam_ops.rs:54 CONFIRMED
- struct OpSignature [pub] — crates/learning/src/cam_ops.rs:76 CONFIRMED
- enum OpType [pub] — crates/learning/src/cam_ops.rs:84 CONFIRMED
- struct OpMeta [pub] — crates/learning/src/cam_ops.rs:95 CONFIRMED
- enum OpCategory [pub] — crates/learning/src/cam_ops.rs:107 CONFIRMED
- fn OpCategory::from_id [pub] — crates/learning/src/cam_ops.rs:127 CONFIRMED
- enum LanceOp [pub] — crates/learning/src/cam_ops.rs:156 CONFIRMED
- enum SqlOp [pub] — crates/learning/src/cam_ops.rs:244 CONFIRMED
- enum CypherOp [pub] — crates/learning/src/cam_ops.rs:325 CONFIRMED
- enum HammingOp [pub] — crates/learning/src/cam_ops.rs:408 CONFIRMED
- enum NarsOp [pub] — crates/learning/src/cam_ops.rs:479 CONFIRMED
- enum FilesystemOp [pub] — crates/learning/src/cam_ops.rs:578 CONFIRMED
- enum CrystalOp [pub] — crates/learning/src/cam_ops.rs:676 CONFIRMED
- enum NsmOp [pub] — crates/learning/src/cam_ops.rs:768 CONFIRMED
- enum ActrOp [pub] — crates/learning/src/cam_ops.rs:872 CONFIRMED
- enum RlOp [pub] — crates/learning/src/cam_ops.rs:951 CONFIRMED
- enum CausalOp [pub] — crates/learning/src/cam_ops.rs:1036 CONFIRMED
- enum QualiaOp [pub] — crates/learning/src/cam_ops.rs:1120 CONFIRMED
- enum RungOp [pub] — crates/learning/src/cam_ops.rs:1220 CONFIRMED
- enum MetaOp [pub] — crates/learning/src/cam_ops.rs:1318 CONFIRMED
- enum VerbOp [pub] — crates/learning/src/cam_ops.rs:1407 CONFIRMED
- enum MemoryOp [pub] — crates/learning/src/cam_ops.rs:1465 CONFIRMED
- enum UserOp [pub] — crates/learning/src/cam_ops.rs:1500 CONFIRMED
- enum LearnOp [pub] — crates/learning/src/cam_ops.rs:1565 CONFIRMED
- type OpFn [pub] — crates/learning/src/cam_ops.rs:1702 CONFIRMED
- struct OpContext [pub] — crates/learning/src/cam_ops.rs:1716 CONFIRMED
- enum OpParam [pub] — crates/learning/src/cam_ops.rs:1729 CONFIRMED
- trait LanceDbOps [pub] — crates/learning/src/cam_ops.rs:1738 CONFIRMED
- struct CognitiveCodebook [pub] — crates/learning/src/cam_ops.rs:1762 CONFIRMED
- struct CrystalLM [pub] — crates/learning/src/cam_ops.rs:1763 CONFIRMED
- struct OpDictionary [pub] — crates/learning/src/cam_ops.rs:1766 CONFIRMED
- fn OpDictionary::new [pub] — crates/learning/src/cam_ops.rs:1781 CONFIRMED
- fn OpDictionary::execute [pub] — crates/learning/src/cam_ops.rs:2501 CONFIRMED
- fn OpDictionary::execute_by_name [pub] — crates/learning/src/cam_ops.rs:2510 CONFIRMED
- fn OpDictionary::execute_semantic [pub] — crates/learning/src/cam_ops.rs:2519 CONFIRMED
- fn OpDictionary::get_meta [pub] — crates/learning/src/cam_ops.rs:2558 CONFIRMED
- fn OpDictionary::list_category [pub] — crates/learning/src/cam_ops.rs:2563 CONFIRMED
- fn OpDictionary::count [pub] — crates/learning/src/cam_ops.rs:2571 CONFIRMED
- fn bundle_fingerprints [pub] — crates/learning/src/cam_ops.rs:2587 CONFIRMED
- fn fold_to_48 [pub] — crates/learning/src/cam_ops.rs:2609 CONFIRMED
- struct GrammarCausalEdge [pub] — crates/learning/src/causal_bridge.rs:58 CONFIRMED
- struct CausalBridge [pub] — crates/learning/src/causal_bridge.rs:79 CONFIRMED
- fn CausalBridge::new [pub] — crates/learning/src/causal_bridge.rs:87 CONFIRMED
- fn CausalBridge::with_engine [pub] — crates/learning/src/causal_bridge.rs:95 CONFIRMED
- fn CausalBridge::engine [pub] — crates/learning/src/causal_bridge.rs:103 CONFIRMED
- fn CausalBridge::engine_mut [pub] — crates/learning/src/causal_bridge.rs:108 CONFIRMED
- fn CausalBridge::into_engine [pub] — crates/learning/src/causal_bridge.rs:113 CONFIRMED
- fn CausalBridge::edge_count [pub] — crates/learning/src/causal_bridge.rs:118 CONFIRMED
- fn CausalBridge::extract_edge [pub] — crates/learning/src/causal_bridge.rs:131 CONFIRMED
- fn CausalBridge::extract_from_triangle [pub] — crates/learning/src/causal_bridge.rs:172 CONFIRMED
- fn CausalBridge::ingest_edge [pub] — crates/learning/src/causal_bridge.rs:204 CONFIRMED
- fn CausalBridge::ingest_text [pub] — crates/learning/src/causal_bridge.rs:242 CONFIRMED
- fn CausalBridge::ingest_flow [pub] — crates/learning/src/causal_bridge.rs:250 CONFIRMED
- fn CausalBridge::query_do [pub] — crates/learning/src/causal_bridge.rs:263 CONFIRMED
- fn CausalBridge::query_cause [pub] — crates/learning/src/causal_bridge.rs:270 CONFIRMED
- fn CausalBridge::query_counterfactual [pub] — crates/learning/src/causal_bridge.rs:281 CONFIRMED
- fn CausalBridge::query_correlates [pub] — crates/learning/src/causal_bridge.rs:292 CONFIRMED
- enum CausalOp [pub] — crates/learning/src/causal_ops.rs:38 CONFIRMED
- struct CausalEngine [pub] — crates/learning/src/causal_ops.rs:246 CONFIRMED
- struct GraphEdge [pub] — crates/learning/src/causal_ops.rs:257 CONFIRMED
- enum CausalEdgeType [pub] — crates/learning/src/causal_ops.rs:266 CONFIRMED
- fn CausalEngine::new [pub] — crates/learning/src/causal_ops.rs:279 CONFIRMED
- fn CausalEngine::store_correlation [pub] — crates/learning/src/causal_ops.rs:292 CONFIRMED
- fn CausalEngine::query_correlates [pub] — crates/learning/src/causal_ops.rs:297 CONFIRMED
- fn CausalEngine::store_intervention [pub] — crates/learning/src/causal_ops.rs:306 CONFIRMED
- fn CausalEngine::query_do [pub] — crates/learning/src/causal_ops.rs:326 CONFIRMED
- fn CausalEngine::query_cause [pub] — crates/learning/src/causal_ops.rs:331 CONFIRMED
- fn CausalEngine::detect_confounders [pub] — crates/learning/src/causal_ops.rs:336 CONFIRMED
- fn CausalEngine::average_treatment_effect [pub] — crates/learning/src/causal_ops.rs:345 CONFIRMED
- fn CausalEngine::store_counterfactual [pub] — crates/learning/src/causal_ops.rs:371 CONFIRMED
- fn CausalEngine::query_counterfactual [pub] — crates/learning/src/causal_ops.rs:383 CONFIRMED
- fn CausalEngine::compute_regret [pub] — crates/learning/src/causal_ops.rs:392 CONFIRMED
- fn CausalEngine::prob_necessity [pub] — crates/learning/src/causal_ops.rs:404 CONFIRMED
- fn CausalEngine::prob_sufficiency [pub] — crates/learning/src/causal_ops.rs:435 CONFIRMED
- fn CausalEngine::add_edge [pub] — crates/learning/src/causal_ops.rs:465 CONFIRMED
- fn CausalEngine::parents [pub] — crates/learning/src/causal_ops.rs:481 CONFIRMED
- fn CausalEngine::children [pub] — crates/learning/src/causal_ops.rs:490 CONFIRMED
- fn CausalEngine::ancestors [pub] — crates/learning/src/causal_ops.rs:499 CONFIRMED
- fn CausalEngine::descendants [pub] — crates/learning/src/causal_ops.rs:519 CONFIRMED
- fn CausalEngine::d_separated [pub] — crates/learning/src/causal_ops.rs:539 CONFIRMED
- fn CausalEngine::total_effect [pub] — crates/learning/src/causal_ops.rs:572 CONFIRMED
- fn CausalEngine::natural_direct_effect [pub] — crates/learning/src/causal_ops.rs:584 CONFIRMED
- struct TruthValue [pub] — crates/learning/src/cognitive_frameworks.rs:19 CONFIRMED
- fn TruthValue::new [pub] — crates/learning/src/cognitive_frameworks.rs:27 CONFIRMED
- fn TruthValue::expectation [pub] — crates/learning/src/cognitive_frameworks.rs:35 CONFIRMED
- fn TruthValue::positive_evidence [pub] — crates/learning/src/cognitive_frameworks.rs:40 CONFIRMED
- fn TruthValue::total_evidence [pub] — crates/learning/src/cognitive_frameworks.rs:48 CONFIRMED
- fn TruthValue::from_evidence [pub] — crates/learning/src/cognitive_frameworks.rs:56 CONFIRMED
- enum NarsCopula [pub] — crates/learning/src/cognitive_frameworks.rs:65 CONFIRMED
- fn NarsCopula::fingerprint [pub] — crates/learning/src/cognitive_frameworks.rs:79 CONFIRMED
- struct NarsInference [pub] — crates/learning/src/cognitive_frameworks.rs:85 CONFIRMED
- fn NarsInference::deduction [pub] — crates/learning/src/cognitive_frameworks.rs:91 CONFIRMED
- fn NarsInference::induction [pub] — crates/learning/src/cognitive_frameworks.rs:98 CONFIRMED
- fn NarsInference::abduction [pub] — crates/learning/src/cognitive_frameworks.rs:105 CONFIRMED
- fn NarsInference::revision [pub] — crates/learning/src/cognitive_frameworks.rs:112 CONFIRMED
- fn NarsInference::negation [pub] — crates/learning/src/cognitive_frameworks.rs:122 CONFIRMED
- fn NarsInference::intersection [pub] — crates/learning/src/cognitive_frameworks.rs:127 CONFIRMED
- fn NarsInference::union [pub] — crates/learning/src/cognitive_frameworks.rs:132 CONFIRMED
- fn NarsInference::choice [pub] — crates/learning/src/cognitive_frameworks.rs:138 CONFIRMED
- fn NarsInference::decision [pub] — crates/learning/src/cognitive_frameworks.rs:147 CONFIRMED
- struct NarsStatement [pub] — crates/learning/src/cognitive_frameworks.rs:154 CONFIRMED
- fn NarsStatement::new [pub] — crates/learning/src/cognitive_frameworks.rs:163 CONFIRMED
- enum ActrBuffer [pub] — crates/learning/src/cognitive_frameworks.rs:189 CONFIRMED
- fn ActrBuffer::fingerprint [pub] — crates/learning/src/cognitive_frameworks.rs:201 CONFIRMED
- struct ActrChunk [pub] — crates/learning/src/cognitive_frameworks.rs:208 CONFIRMED
- fn ActrChunk::new [pub] — crates/learning/src/cognitive_frameworks.rs:229 CONFIRMED
- fn ActrChunk::with_slot [pub] — crates/learning/src/cognitive_frameworks.rs:240 CONFIRMED
- fn ActrChunk::base_level_activation [pub] — crates/learning/src/cognitive_frameworks.rs:255 CONFIRMED
- fn ActrChunk::record_access [pub] — crates/learning/src/cognitive_frameworks.rs:267 CONFIRMED
- fn ActrChunk::get_slot [pub] — crates/learning/src/cognitive_frameworks.rs:272 CONFIRMED
- fn ActrChunk::partial_match [pub] — crates/learning/src/cognitive_frameworks.rs:280 CONFIRMED
- struct ActrProduction [pub] — crates/learning/src/cognitive_frameworks.rs:307 CONFIRMED
- fn ActrProduction::new [pub] — crates/learning/src/cognitive_frameworks.rs:325 CONFIRMED
- fn ActrProduction::with_utility [pub] — crates/learning/src/cognitive_frameworks.rs:339 CONFIRMED
- fn ActrProduction::matches [pub] — crates/learning/src/cognitive_frameworks.rs:345 CONFIRMED
- struct StateAction [pub] — crates/learning/src/cognitive_frameworks.rs:356 CONFIRMED
- fn StateAction::new [pub] — crates/learning/src/cognitive_frameworks.rs:363 CONFIRMED
- struct QValue [pub] — crates/learning/src/cognitive_frameworks.rs:375 CONFIRMED
- struct RlAgent [pub] — crates/learning/src/cognitive_frameworks.rs:382 CONFIRMED
- fn RlAgent::new [pub] — crates/learning/src/cognitive_frameworks.rs:397 CONFIRMED
- fn RlAgent::get_q [pub] — crates/learning/src/cognitive_frameworks.rs:407 CONFIRMED
- fn RlAgent::update_q [pub] — crates/learning/src/cognitive_frameworks.rs:414 CONFIRMED
- fn RlAgent::select_action [pub] — crates/learning/src/cognitive_frameworks.rs:452 CONFIRMED
- fn RlAgent::td_error [pub] — crates/learning/src/cognitive_frameworks.rs:480 CONFIRMED
- enum CausalRelation [pub] — crates/learning/src/cognitive_frameworks.rs:500 CONFIRMED
- struct CausalNode [pub] — crates/learning/src/cognitive_frameworks.rs:511 CONFIRMED
- struct CausalEdge [pub] — crates/learning/src/cognitive_frameworks.rs:519 CONFIRMED
- struct DoOperator [pub] — crates/learning/src/cognitive_frameworks.rs:528 CONFIRMED
- fn DoOperator::new [pub] — crates/learning/src/cognitive_frameworks.rs:536 CONFIRMED
- fn DoOperator::fingerprint [pub] — crates/learning/src/cognitive_frameworks.rs:541 CONFIRMED
- struct Counterfactual [pub] — crates/learning/src/cognitive_frameworks.rs:549 CONFIRMED
- fn Counterfactual::fingerprint [pub] — crates/learning/src/cognitive_frameworks.rs:559 CONFIRMED
- enum QualiaChannel [pub] — crates/learning/src/cognitive_frameworks.rs:578 CONFIRMED
- fn QualiaChannel::fingerprint [pub] — crates/learning/src/cognitive_frameworks.rs:590 CONFIRMED
- fn QualiaChannel::all [pub] — crates/learning/src/cognitive_frameworks.rs:594 CONFIRMED
- struct QualiaState [pub] — crates/learning/src/cognitive_frameworks.rs:610 CONFIRMED
- fn QualiaState::neutral [pub] — crates/learning/src/cognitive_frameworks.rs:619 CONFIRMED
- fn QualiaState::from_values [pub] — crates/learning/src/cognitive_frameworks.rs:623 CONFIRMED
- fn QualiaState::get [pub] — crates/learning/src/cognitive_frameworks.rs:648 CONFIRMED
- fn QualiaState::set [pub] — crates/learning/src/cognitive_frameworks.rs:652 CONFIRMED
- fn QualiaState::blend [pub] — crates/learning/src/cognitive_frameworks.rs:658 CONFIRMED
- fn QualiaState::distance [pub] — crates/learning/src/cognitive_frameworks.rs:665 CONFIRMED
- fn QualiaState::classify_emotion [pub] — crates/learning/src/cognitive_frameworks.rs:676 CONFIRMED
- enum Rung [pub] — crates/learning/src/cognitive_frameworks.rs:702 CONFIRMED
- fn Rung::fingerprint [pub] — crates/learning/src/cognitive_frameworks.rs:716 CONFIRMED
- fn Rung::from_u8 [pub] — crates/learning/src/cognitive_frameworks.rs:720 CONFIRMED
- fn Rung::name [pub] — crates/learning/src/cognitive_frameworks.rs:735 CONFIRMED
- fn Rung::can_ascend [pub] — crates/learning/src/cognitive_frameworks.rs:751 CONFIRMED
- fn Rung::can_descend [pub] — crates/learning/src/cognitive_frameworks.rs:756 CONFIRMED
- fn Rung::ascend [pub] — crates/learning/src/cognitive_frameworks.rs:761 CONFIRMED
- fn Rung::descend [pub] — crates/learning/src/cognitive_frameworks.rs:766 CONFIRMED
- struct RungClassifier [pub] — crates/learning/src/cognitive_frameworks.rs:772 CONFIRMED
- fn RungClassifier::new [pub] — crates/learning/src/cognitive_frameworks.rs:778 CONFIRMED
- fn RungClassifier::classify [pub] — crates/learning/src/cognitive_frameworks.rs:785 CONFIRMED
- enum Operator [pub] — crates/learning/src/cognitive_styles.rs:48 CONFIRMED
- const Operator::ALL [pub] — crates/learning/src/cognitive_styles.rs:70 CONFIRMED
- fn Operator::from_index [pub] — crates/learning/src/cognitive_styles.rs:82 CONFIRMED
- enum Atom [pub] — crates/learning/src/cognitive_styles.rs:90 CONFIRMED
- const Atom::ALL [pub] — crates/learning/src/cognitive_styles.rs:112 CONFIRMED
- fn Atom::from_index [pub] — crates/learning/src/cognitive_styles.rs:124 CONFIRMED
- type StyleFingerprint [pub] — crates/learning/src/cognitive_styles.rs:134 CONFIRMED
- struct CognitiveStyle [pub] — crates/learning/src/cognitive_styles.rs:138 CONFIRMED
- enum StyleOrigin [pub] — crates/learning/src/cognitive_styles.rs:163 CONFIRMED
- fn CognitiveStyle::to_fingerprint [pub] — crates/learning/src/cognitive_styles.rs:174 CONFIRMED
- fn CognitiveStyle::from_fingerprint [pub] — crates/learning/src/cognitive_styles.rs:195 CONFIRMED
- fn CognitiveStyle::distance [pub] — crates/learning/src/cognitive_styles.rs:224 CONFIRMED
- fn CognitiveStyle::mutate [pub] — crates/learning/src/cognitive_styles.rs:235 CONFIRMED
- fn CognitiveStyle::crossover [pub] — crates/learning/src/cognitive_styles.rs:266 CONFIRMED
- fn create_base_styles [pub] — crates/learning/src/cognitive_styles.rs:303 CONFIRMED
- struct TaskContext [pub] — crates/learning/src/cognitive_styles.rs:581 CONFIRMED
- fn TaskContext::key [pub] — crates/learning/src/cognitive_styles.rs:596 CONFIRMED
- fn TaskContext::discretize [pub] — crates/learning/src/cognitive_styles.rs:605 CONFIRMED
- struct TaskOutcome [pub] — crates/learning/src/cognitive_styles.rs:623 CONFIRMED
- fn TaskOutcome::reward [pub] — crates/learning/src/cognitive_styles.rs:636 CONFIRMED
- struct RLConfig [pub] — crates/learning/src/cognitive_styles.rs:645 CONFIRMED
- struct StyleSelector [pub] — crates/learning/src/cognitive_styles.rs:691 CONFIRMED
- fn StyleSelector::new [pub] — crates/learning/src/cognitive_styles.rs:714 CONFIRMED
- fn StyleSelector::select [pub] — crates/learning/src/cognitive_styles.rs:739 CONFIRMED
- fn StyleSelector::update [pub] — crates/learning/src/cognitive_styles.rs:825 CONFIRMED
- fn StyleSelector::all_styles [pub] — crates/learning/src/cognitive_styles.rs:902 CONFIRMED
- fn StyleSelector::get_style [pub] — crates/learning/src/cognitive_styles.rs:912 CONFIRMED
- fn StyleSelector::triangle_weights [pub] — crates/learning/src/cognitive_styles.rs:926 CONFIRMED
- fn StyleSelector::shift_weights [pub] — crates/learning/src/cognitive_styles.rs:934 CONFIRMED
- fn StyleSelector::export_q_table [pub] — crates/learning/src/cognitive_styles.rs:942 CONFIRMED
- fn StyleSelector::stats [pub] — crates/learning/src/cognitive_styles.rs:951 CONFIRMED
- truncated (444 total), see symbols.tsv

## Imports from in-scope crates

- none

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n39["lance-graph-contract"]
  n67["learning"]
  n67 --> n39
```
