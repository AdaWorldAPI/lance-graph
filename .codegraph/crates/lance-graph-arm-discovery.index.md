# lance-graph-arm-discovery

crate: lance-graph-arm-discovery
path: crates/lance-graph-arm-discovery
workspace: excluded
digest: 7ae7fe00d549f9e4
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph-contract normal optional — crates/lance-graph-arm-discovery/Cargo.toml:52 CONFIRMED
- dep ndarray normal optional — crates/lance-graph-arm-discovery/Cargo.toml:45 CONFIRMED

## Depended on by

- rdep deepnsm-v2 normal — crates/deepnsm-v2/Cargo.toml:35 CONFIRMED
- rdep lance-graph-osint dev — crates/lance-graph-osint/Cargo.toml:46 CONFIRMED
- rdep lance-graph-quack dev — crates/lance-graph-quack/Cargo.toml:23 CONFIRMED

## Public items

- trait CodebookDistance [pub] — crates/lance-graph-arm-discovery/src/aerial/codebook.rs:35 CONFIRMED
- struct MatrixDistance [pub] — crates/lance-graph-arm-discovery/src/aerial/codebook.rs:44 CONFIRMED
- fn MatrixDistance::new [pub] — crates/lance-graph-arm-discovery/src/aerial/codebook.rs:56 CONFIRMED
- struct TopKDistance [pub] — crates/lance-graph-arm-discovery/src/aerial/codebook.rs:104 CONFIRMED
- fn TopKDistance::new [pub] — crates/lance-graph-arm-discovery/src/aerial/codebook.rs:116 CONFIRMED
- fn antecedent_distance [pub] — crates/lance-graph-arm-discovery/src/aerial/codebook.rs:155 CONFIRMED
- struct ExtractParams [pub] — crates/lance-graph-arm-discovery/src/aerial/extract.rs:31 CONFIRMED
- fn extract_rules [pub] — crates/lance-graph-arm-discovery/src/aerial/extract.rs:59 CONFIRMED
- mod codebook [pub] — crates/lance-graph-arm-discovery/src/aerial/mod.rs:30 CONFIRMED
- mod extract [pub] — crates/lance-graph-arm-discovery/src/aerial/mod.rs:31 CONFIRMED
- mod ontology [pub] — crates/lance-graph-arm-discovery/src/aerial/mod.rs:32 CONFIRMED
- type AerialParams [pub] — crates/lance-graph-arm-discovery/src/aerial/mod.rs:43 CONFIRMED
- struct AerialProposer [pub] — crates/lance-graph-arm-discovery/src/aerial/mod.rs:47 CONFIRMED
- fn AerialProposer<D>::new [pub] — crates/lance-graph-arm-discovery/src/aerial/mod.rs:58 CONFIRMED
- fn AerialProposer<D>::mine [pub] — crates/lance-graph-arm-discovery/src/aerial/mod.rs:69 CONFIRMED
- enum DolceCategory [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:22 CONFIRMED
- const DolceCategory::ALL [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:35 CONFIRMED
- fn DolceCategory::basin [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:44 CONFIRMED
- fn DolceCategory::iri [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:50 CONFIRMED
- fn DolceCategory::from_basin [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:61 CONFIRMED
- fn DolceCategory::from_index [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:69 CONFIRMED
- struct OntologyProjector [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:87 CONFIRMED
- fn OntologyProjector::subclass_of [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:102 CONFIRMED
- fn OntologyProjector::instance_of [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:113 CONFIRMED
- fn OntologyProjector::dolce_subclass [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:125 CONFIRMED
- fn OntologyProjector::dolce_id [pub] — crates/lance-graph-arm-discovery/src/aerial/ontology.rs:148 CONFIRMED
- struct RowMasks [pub] — crates/lance-graph-arm-discovery/src/bitset.rs:20 CONFIRMED
- fn RowMasks::build [pub] — crates/lance-graph-arm-discovery/src/bitset.rs:34 CONFIRMED
- fn RowMasks::len [pub] — crates/lance-graph-arm-discovery/src/bitset.rs:59 CONFIRMED
- fn RowMasks::is_empty [pub] — crates/lance-graph-arm-discovery/src/bitset.rs:65 CONFIRMED
- fn RowMasks::support_count [pub] — crates/lance-graph-arm-discovery/src/bitset.rs:78 CONFIRMED
- fn RowMasks::and_count [pub] — crates/lance-graph-arm-discovery/src/bitset.rs:86 CONFIRMED
- struct FeatureSpec [pub] — crates/lance-graph-arm-discovery/src/encode.rs:19 CONFIRMED
- fn FeatureSpec::new [pub] — crates/lance-graph-arm-discovery/src/encode.rs:32 CONFIRMED
- fn FeatureSpec::dim [pub] — crates/lance-graph-arm-discovery/src/encode.rs:52 CONFIRMED
- fn FeatureSpec::num_features [pub] — crates/lance-graph-arm-discovery/src/encode.rs:58 CONFIRMED
- fn FeatureSpec::cardinality [pub] — crates/lance-graph-arm-discovery/src/encode.rs:64 CONFIRMED
- fn FeatureSpec::block [pub] — crates/lance-graph-arm-discovery/src/encode.rs:70 CONFIRMED
- fn FeatureSpec::slot [pub] — crates/lance-graph-arm-discovery/src/encode.rs:76 CONFIRMED
- fn FeatureSpec::checked_slot [pub] — crates/lance-graph-arm-discovery/src/encode.rs:84 CONFIRMED
- struct Dataset [pub] — crates/lance-graph-arm-discovery/src/encode.rs:104 CONFIRMED
- fn Dataset::new [pub] — crates/lance-graph-arm-discovery/src/encode.rs:117 CONFIRMED
- fn Dataset::len [pub] — crates/lance-graph-arm-discovery/src/encode.rs:134 CONFIRMED
- fn Dataset::is_empty [pub] — crates/lance-graph-arm-discovery/src/encode.rs:140 CONFIRMED
- fn Dataset::count_matching [pub] — crates/lance-graph-arm-discovery/src/encode.rs:147 CONFIRMED
- mod aerial [pub] — crates/lance-graph-arm-discovery/src/lib.rs:62 CONFIRMED
- mod bitset [pub] — crates/lance-graph-arm-discovery/src/lib.rs:63 CONFIRMED
- mod encode [pub] — crates/lance-graph-arm-discovery/src/lib.rs:64 CONFIRMED
- mod ndjson [pub] — crates/lance-graph-arm-discovery/src/lib.rs:65 CONFIRMED
- mod rule [pub] — crates/lance-graph-arm-discovery/src/lib.rs:66 CONFIRMED
- mod simd [pub] — crates/lance-graph-arm-discovery/src/lib.rs:67 CONFIRMED
- mod translator [pub] — crates/lance-graph-arm-discovery/src/lib.rs:68 CONFIRMED
- fn to_ndjson [pub] — crates/lance-graph-arm-discovery/src/ndjson.rs:29 CONFIRMED
- struct Item [pub] — crates/lance-graph-arm-discovery/src/rule.rs:28 CONFIRMED
- fn Item::new [pub] — crates/lance-graph-arm-discovery/src/rule.rs:38 CONFIRMED
- const PPM [pub] — crates/lance-graph-arm-discovery/src/rule.rs:45 CONFIRMED
- struct CandidateRule [pub] — crates/lance-graph-arm-discovery/src/rule.rs:58 CONFIRMED
- fn CandidateRule::evidence [pub] — crates/lance-graph-arm-discovery/src/rule.rs:77 CONFIRMED
- fn CandidateRule::support_ppm [pub] — crates/lance-graph-arm-discovery/src/rule.rs:83 CONFIRMED
- fn CandidateRule::confidence_ppm [pub] — crates/lance-graph-arm-discovery/src/rule.rs:93 CONFIRMED
- fn CandidateRule::passes [pub] — crates/lance-graph-arm-discovery/src/rule.rs:106 CONFIRMED
- fn CandidateRule::support_f32 [pub] — crates/lance-graph-arm-discovery/src/rule.rs:113 CONFIRMED
- fn CandidateRule::confidence_f32 [pub] — crates/lance-graph-arm-discovery/src/rule.rs:119 CONFIRMED
- trait Proposer [pub] — crates/lance-graph-arm-discovery/src/rule.rs:126 CONFIRMED
- fn popcount [pub] cfg=not(feature="ndarray-simd") — crates/lance-graph-arm-discovery/src/simd.rs:21 CONFIRMED
- fn and_popcount [pub] cfg=not(feature="ndarray-simd") — crates/lance-graph-arm-discovery/src/simd.rs:28 CONFIRMED
- fn popcount [pub] cfg=feature="ndarray-simd" — crates/lance-graph-arm-discovery/src/simd.rs:44 CONFIRMED
- fn and_popcount [pub] cfg=feature="ndarray-simd" — crates/lance-graph-arm-discovery/src/simd.rs:64 CONFIRMED
- const NARS_PERSONALITY_K [pub] — crates/lance-graph-arm-discovery/src/translator.rs:23 CONFIRMED
- struct TruthU8 [pub] — crates/lance-graph-arm-discovery/src/translator.rs:61 CONFIRMED
- fn arm_to_truth_u8 [pub] — crates/lance-graph-arm-discovery/src/translator.rs:71 CONFIRMED
- struct NarsTruth [pub] — crates/lance-graph-arm-discovery/src/translator.rs:94 CONFIRMED
- fn NarsTruth::expectation [pub] — crates/lance-graph-arm-discovery/src/translator.rs:105 CONFIRMED
- fn arm_to_nars [pub] — crates/lance-graph-arm-discovery/src/translator.rs:113 CONFIRMED
- trait FeedProjector [pub] — crates/lance-graph-arm-discovery/src/translator.rs:122 CONFIRMED
- struct CandidateTriple [pub] — crates/lance-graph-arm-discovery/src/translator.rs:138 CONFIRMED
- fn CandidateTriple::from_rule [pub] — crates/lance-graph-arm-discovery/src/translator.rs:154 CONFIRMED
- struct DebugProjector [pub] — crates/lance-graph-arm-discovery/src/translator.rs:168 CONFIRMED

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
  n11["deepnsm-v2"]
  n31["lance-graph-arm-discovery"]
  n39["lance-graph-contract"]
  n11 --> n31
  n31 --> n39
```
