// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge between DeepNSM evaluation data and the SPO triple store.
//!
//! Natural Semantic Metalanguage (NSM) explications describe word meanings
//! using a small set of semantic primes. This module maps NSM evaluation
//! results into SPO triples with NARS truth values, enabling graph-based
//! reasoning about the quality and correctness of NSM explications.
//!
//! # Overview
//!
//! - [`NsmSpoMapping`] converts a single NSM evaluation into SPO records
//! - [`NsmEvalStore`] aggregates evaluations across words and models
//! - [`ModelComparison`] summarizes head-to-head model comparison
//! - Arrow RecordBatch export for DataFusion integration

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{BooleanArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use crate::graph::fingerprint::{dn_hash, label_fp};

use super::builder::{SpoBuilder, SpoRecord};
use super::truth::TruthValue;

/// An NSM explication mapped to SPO triples for graph-based reasoning.
///
/// Each mapping captures the evaluation of one model's explication for a
/// single word, including the primes/molecules ratios and total score.
#[derive(Debug, Clone)]
pub struct NsmSpoMapping {
    /// The target word being explicated.
    pub word: String,
    /// The full explication text produced by a model.
    pub explication_text: String,
    /// Ratio of semantic primes used in the explication (0.0 to 1.0).
    pub primes_ratio: f32,
    /// Ratio of semantic molecules used in the explication (0.0 to 1.0).
    pub molecules_ratio: f32,
    /// Total evaluation score (may exceed 1.0; higher is better).
    pub total_score: f32,
    /// Whether the explication contains the original word (circularity).
    pub uses_original_word: bool,
}

impl NsmSpoMapping {
    /// Convert NSM evaluation result to SPO records with NARS truth values.
    ///
    /// Creates three triples:
    /// - `(word, "has_explication", explication_hash)` with truth from primes_ratio
    /// - `(word, "legality", score_label)` with truth from legality metrics
    /// - `(word, "circularity", bool_label)` with certain or unknown truth
    pub fn to_spo_records(&self) -> Vec<(SpoRecord, TruthValue)> {
        let subj = label_fp(&self.word);

        // Triple 1: word has_explication explication_hash
        let pred_expl = label_fp("has_explication");
        // Hash the explication text to get a stable fingerprint for the object
        let expl_hash = format!("expl:{:016x}", dn_hash(&self.explication_text));
        let obj_expl = label_fp(&expl_hash);
        let truth_expl = TruthValue::new(self.primes_ratio, self.primes_ratio.min(0.9));
        let rec_expl = SpoBuilder::build_edge(&subj, &pred_expl, &obj_expl, truth_expl);

        // Triple 2: word legality score_label
        let pred_legal = label_fp("legality");
        let score_label = format!("score:{:.4}", self.total_score);
        let obj_legal = label_fp(&score_label);
        let truth_legal = self.legality_truth();
        let rec_legal = SpoBuilder::build_edge(&subj, &pred_legal, &obj_legal, truth_legal);

        // Triple 3: word circularity bool
        let pred_circ = label_fp("circularity");
        let circ_label = if self.uses_original_word {
            "circular:true"
        } else {
            "circular:false"
        };
        let obj_circ = label_fp(circ_label);
        let truth_circ = if self.uses_original_word {
            // Circularity detected with certainty
            TruthValue::certain()
        } else {
            // No circularity detected, but we can't be fully certain
            TruthValue::new(0.0, 0.8)
        };
        let rec_circ = SpoBuilder::build_edge(&subj, &pred_circ, &obj_circ, truth_circ);

        vec![
            (rec_expl, truth_expl),
            (rec_legal, truth_legal),
            (rec_circ, truth_circ),
        ]
    }

    /// Compute a NARS truth value from legality metrics.
    ///
    /// Frequency is set to `primes_ratio` (proportion of semantic primes used),
    /// and confidence is `1.0 - molecules_ratio` (lower molecule usage means
    /// more confidence that the explication uses proper primitives).
    pub fn legality_truth(&self) -> TruthValue {
        let frequency = self.primes_ratio.clamp(0.0, 1.0);
        let confidence = (1.0 - self.molecules_ratio).clamp(0.0, 1.0);
        TruthValue::new(frequency, confidence)
    }

    /// Compute a NARS truth value from the total_score.
    ///
    /// Uses a sigmoid-like mapping to normalize the score to [0,1]:
    /// `f(x) = x / (x + 1)` for non-negative x, giving a smooth
    /// saturation curve. Confidence is derived from how far the score
    /// is from the ambiguous midpoint.
    pub fn score_truth(&self) -> TruthValue {
        let s = self.total_score.max(0.0);
        // Sigmoid normalization: maps [0, inf) -> [0, 1)
        let frequency = s / (s + 1.0);
        // Confidence increases with distance from 0.5 (the uncertain midpoint)
        let confidence = (2.0 * (frequency - 0.5).abs()).clamp(0.0, 1.0);
        TruthValue::new(frequency, confidence)
    }
}

/// A single grader's score for an NSM explication.
#[derive(Debug, Clone)]
pub struct NsmGraderScore {
    /// Name of the grader model (e.g., "gpt-4o", "claude-sonnet").
    pub grader_model: String,
    /// Adjusted score from this grader.
    pub adj_score: f32,
    /// Average delta log-probability from this grader.
    pub avg_delta_log: f32,
    /// Total number of matches found by the grader.
    pub total_match: usize,
}

/// A complete model evaluation for one word, including grader scores.
#[derive(Debug, Clone)]
pub struct NsmModelEval {
    /// Name of the model that produced the explication.
    pub model_name: String,
    /// The NSM-to-SPO mapping for this evaluation.
    pub mapping: NsmSpoMapping,
    /// Scores from each grader model.
    pub grader_scores: Vec<NsmGraderScore>,
}

/// Result of comparing two models across all words.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// First model name.
    pub model_a: String,
    /// Second model name.
    pub model_b: String,
    /// Average total_score for model A across shared words.
    pub a_avg_score: f32,
    /// Average total_score for model B across shared words.
    pub b_avg_score: f32,
    /// Number of words where model A scored higher.
    pub a_wins: usize,
    /// Number of words where model B scored higher.
    pub b_wins: usize,
    /// Number of words where both models scored equally.
    pub ties: usize,
    /// NARS truth of the proposition "model_a is better than model_b".
    pub truth: TruthValue,
}

/// Stores NSM evaluation results as SPO triples for graph reasoning.
///
/// Aggregates evaluations by word, enabling queries like "which model
/// produced the best explication for word X?" and cross-model comparisons.
pub struct NsmEvalStore {
    /// Maps word -> list of model evaluations.
    evaluations: HashMap<String, Vec<NsmModelEval>>,
}

impl NsmEvalStore {
    /// Create a new empty evaluation store.
    pub fn new() -> Self {
        Self {
            evaluations: HashMap::new(),
        }
    }

    /// Add an evaluation result for a word.
    ///
    /// Multiple evaluations (from different models) can be added for the
    /// same word. The word key is taken from the evaluation's mapping.
    pub fn add_eval(&mut self, word: &str, eval: NsmModelEval) {
        self.evaluations
            .entry(word.to_string())
            .or_default()
            .push(eval);
    }

    /// Query: which model produced the best explication for a word?
    ///
    /// "Best" is determined by the highest `total_score` in the mapping.
    /// Returns `None` if no evaluations exist for the word.
    pub fn best_model_for_word(&self, word: &str) -> Option<&NsmModelEval> {
        self.evaluations.get(word).and_then(|evals| {
            evals.iter().max_by(|a, b| {
                a.mapping
                    .total_score
                    .partial_cmp(&b.mapping.total_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
    }

    /// Query: compare two models across all words where both have evaluations.
    ///
    /// Returns a [`ModelComparison`] with win/loss/tie counts and a NARS
    /// truth value expressing the confidence that model_a is better.
    pub fn compare_models(&self, model_a: &str, model_b: &str) -> ModelComparison {
        let mut a_wins = 0usize;
        let mut b_wins = 0usize;
        let mut ties = 0usize;
        let mut a_total = 0.0f32;
        let mut b_total = 0.0f32;
        let mut count = 0usize;

        for evals in self.evaluations.values() {
            let eval_a = evals.iter().find(|e| e.model_name == model_a);
            let eval_b = evals.iter().find(|e| e.model_name == model_b);

            if let (Some(ea), Some(eb)) = (eval_a, eval_b) {
                count += 1;
                a_total += ea.mapping.total_score;
                b_total += eb.mapping.total_score;

                let diff = (ea.mapping.total_score - eb.mapping.total_score).abs();
                if diff < 1e-6 {
                    ties += 1;
                } else if ea.mapping.total_score > eb.mapping.total_score {
                    a_wins += 1;
                } else {
                    b_wins += 1;
                }
            }
        }

        let a_avg = if count > 0 {
            a_total / count as f32
        } else {
            0.0
        };
        let b_avg = if count > 0 {
            b_total / count as f32
        } else {
            0.0
        };

        // NARS truth: frequency = proportion of wins for A, confidence from sample size
        let total_decided = a_wins + b_wins;
        let frequency = if total_decided > 0 {
            a_wins as f32 / total_decided as f32
        } else {
            0.5 // no evidence either way
        };
        // Confidence grows with evidence: c = n / (n + k), k=5 as horizon
        let confidence = count as f32 / (count as f32 + 5.0);

        ModelComparison {
            model_a: model_a.to_string(),
            model_b: model_b.to_string(),
            a_avg_score: a_avg,
            b_avg_score: b_avg,
            a_wins,
            b_wins,
            ties,
            truth: TruthValue::new(frequency, confidence),
        }
    }

    /// Export all evaluations as SPO records for the triple store.
    ///
    /// Returns `(key, record, truth)` tuples where the key is derived
    /// from a deterministic hash of word + model + triple index.
    pub fn to_spo_records(&self) -> Vec<(u64, SpoRecord, TruthValue)> {
        let mut results = Vec::new();

        for (word, evals) in &self.evaluations {
            for eval in evals {
                let triples = eval.mapping.to_spo_records();
                for (i, (record, truth)) in triples.into_iter().enumerate() {
                    let key_str = format!("nsm:{}:{}:{}", word, eval.model_name, i);
                    let key = dn_hash(&key_str);
                    results.push((key, record, truth));
                }

                // Additional triples for grader scores
                let subj = label_fp(word);
                for gs in &eval.grader_scores {
                    let pred = label_fp("is_graded_by");
                    let obj = label_fp(&gs.grader_model);
                    // Truth from grader's adjusted score (sigmoid-normalized)
                    let s = gs.adj_score.max(0.0);
                    let freq = s / (s + 1.0);
                    let conf = (2.0 * (freq - 0.5).abs()).clamp(0.0, 1.0);
                    let truth = TruthValue::new(freq, conf);
                    let record = SpoBuilder::build_edge(&subj, &pred, &obj, truth);
                    let key_str = format!(
                        "nsm:{}:{}:grader:{}",
                        word, eval.model_name, gs.grader_model
                    );
                    let key = dn_hash(&key_str);
                    results.push((key, record, truth));
                }

                // Triple for model attribution
                let subj = label_fp(word);
                let pred = label_fp("is_scored_by");
                let obj = label_fp(&eval.model_name);
                let truth = eval.mapping.score_truth();
                let record = SpoBuilder::build_edge(&subj, &pred, &obj, truth);
                let key_str = format!("nsm:{}:{}:scored_by", word, eval.model_name);
                let key = dn_hash(&key_str);
                results.push((key, record, truth));
            }
        }

        results
    }

    /// Build an Arrow RecordBatch from all evaluation results.
    ///
    /// The schema includes: word, model, explication, primes_ratio,
    /// molecules_ratio, total_score, uses_original_word, adj_score,
    /// avg_delta_log. One row per (word, model, grader) combination.
    /// If a model has no grader scores, one row is emitted with
    /// NaN for adj_score and avg_delta_log.
    pub fn to_record_batch(&self) -> RecordBatch {
        let mut words = Vec::new();
        let mut models = Vec::new();
        let mut explications = Vec::new();
        let mut primes_ratios = Vec::new();
        let mut molecules_ratios = Vec::new();
        let mut total_scores = Vec::new();
        let mut uses_original = Vec::new();
        let mut adj_scores = Vec::new();
        let mut avg_delta_logs = Vec::new();

        for evals in self.evaluations.values() {
            for eval in evals {
                if eval.grader_scores.is_empty() {
                    // Emit one row with NaN grader columns
                    words.push(eval.mapping.word.clone());
                    models.push(eval.model_name.clone());
                    explications.push(eval.mapping.explication_text.clone());
                    primes_ratios.push(eval.mapping.primes_ratio);
                    molecules_ratios.push(eval.mapping.molecules_ratio);
                    total_scores.push(eval.mapping.total_score);
                    uses_original.push(eval.mapping.uses_original_word);
                    adj_scores.push(f32::NAN);
                    avg_delta_logs.push(f32::NAN);
                } else {
                    for gs in &eval.grader_scores {
                        words.push(eval.mapping.word.clone());
                        models.push(eval.model_name.clone());
                        explications.push(eval.mapping.explication_text.clone());
                        primes_ratios.push(eval.mapping.primes_ratio);
                        molecules_ratios.push(eval.mapping.molecules_ratio);
                        total_scores.push(eval.mapping.total_score);
                        uses_original.push(eval.mapping.uses_original_word);
                        adj_scores.push(gs.adj_score);
                        avg_delta_logs.push(gs.avg_delta_log);
                    }
                }
            }
        }

        let schema = Arc::new(nsm_eval_schema());

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(words)),
                Arc::new(StringArray::from(models)),
                Arc::new(StringArray::from(explications)),
                Arc::new(Float32Array::from(primes_ratios)),
                Arc::new(Float32Array::from(molecules_ratios)),
                Arc::new(Float32Array::from(total_scores)),
                Arc::new(BooleanArray::from(uses_original)),
                Arc::new(Float32Array::from(adj_scores)),
                Arc::new(Float32Array::from(avg_delta_logs)),
            ],
        )
        .expect("schema and arrays must align")
    }
}

impl Default for NsmEvalStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Arrow schema for NSM evaluation results.
///
/// Fields: word, model, explication, primes_ratio, molecules_ratio,
/// total_score, uses_original_word, adj_score, avg_delta_log.
pub fn nsm_eval_schema() -> Schema {
    Schema::new(vec![
        Field::new("word", DataType::Utf8, false),
        Field::new("model", DataType::Utf8, false),
        Field::new("explication", DataType::Utf8, false),
        Field::new("primes_ratio", DataType::Float32, false),
        Field::new("molecules_ratio", DataType::Float32, false),
        Field::new("total_score", DataType::Float32, false),
        Field::new("uses_original_word", DataType::Boolean, false),
        Field::new("adj_score", DataType::Float32, false),
        Field::new("avg_delta_log", DataType::Float32, false),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mapping(word: &str, score: f32) -> NsmSpoMapping {
        NsmSpoMapping {
            word: word.to_string(),
            explication_text: format!("someone thinks something good about {}", word),
            primes_ratio: 0.75,
            molecules_ratio: 0.2,
            total_score: score,
            uses_original_word: false,
        }
    }

    fn sample_eval(word: &str, model: &str, score: f32) -> NsmModelEval {
        NsmModelEval {
            model_name: model.to_string(),
            mapping: sample_mapping(word, score),
            grader_scores: vec![
                NsmGraderScore {
                    grader_model: "gpt-4o".to_string(),
                    adj_score: score * 0.9,
                    avg_delta_log: -0.3,
                    total_match: 5,
                },
                NsmGraderScore {
                    grader_model: "claude-sonnet".to_string(),
                    adj_score: score * 0.85,
                    avg_delta_log: -0.25,
                    total_match: 4,
                },
            ],
        }
    }

    #[test]
    fn test_nsm_to_spo_records() {
        let mapping = sample_mapping("happy", 3.5);
        let records = mapping.to_spo_records();

        // Should produce 3 triples
        assert_eq!(records.len(), 3);

        // First triple: has_explication
        let (rec, tv) = &records[0];
        assert_eq!(rec.subject, label_fp("happy"));
        assert_eq!(rec.predicate, label_fp("has_explication"));
        assert!(tv.frequency > 0.0);

        // Second triple: legality
        let (rec, _tv) = &records[1];
        assert_eq!(rec.predicate, label_fp("legality"));

        // Third triple: circularity (non-circular)
        let (rec, tv) = &records[2];
        assert_eq!(rec.predicate, label_fp("circularity"));
        assert_eq!(rec.object, label_fp("circular:false"));
        // Non-circular: frequency should be 0.0
        assert!((tv.frequency - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_nsm_to_spo_records_circular() {
        let mapping = NsmSpoMapping {
            word: "sad".to_string(),
            explication_text: "this person feels sad inside".to_string(),
            primes_ratio: 0.5,
            molecules_ratio: 0.3,
            total_score: 1.0,
            uses_original_word: true,
        };
        let records = mapping.to_spo_records();
        // Circularity triple should have certain truth
        let (rec, tv) = &records[2];
        assert_eq!(rec.object, label_fp("circular:true"));
        assert_eq!(tv.frequency, 1.0);
        assert_eq!(tv.confidence, 1.0);
    }

    #[test]
    fn test_legality_truth() {
        let mapping = NsmSpoMapping {
            word: "angry".to_string(),
            explication_text: "someone feels something bad".to_string(),
            primes_ratio: 0.8,
            molecules_ratio: 0.1,
            total_score: 4.0,
            uses_original_word: false,
        };
        let tv = mapping.legality_truth();
        assert!((tv.frequency - 0.8).abs() < 1e-6);
        assert!((tv.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_legality_truth_clamped() {
        let mapping = NsmSpoMapping {
            word: "test".to_string(),
            explication_text: "test".to_string(),
            primes_ratio: 1.5,     // exceeds 1.0
            molecules_ratio: -0.2, // below 0.0
            total_score: 0.0,
            uses_original_word: false,
        };
        let tv = mapping.legality_truth();
        assert!(tv.frequency <= 1.0);
        assert!(tv.confidence <= 1.0);
        assert!(tv.frequency >= 0.0);
        assert!(tv.confidence >= 0.0);
    }

    #[test]
    fn test_score_truth() {
        // Score of 0 -> frequency near 0, low confidence
        let m0 = sample_mapping("w", 0.0);
        let tv0 = m0.score_truth();
        assert!((tv0.frequency - 0.0).abs() < 1e-6);

        // Score of 1 -> frequency = 0.5, confidence = 0
        let m1 = sample_mapping("w", 1.0);
        let tv1 = m1.score_truth();
        assert!((tv1.frequency - 0.5).abs() < 1e-6);
        assert!((tv1.confidence - 0.0).abs() < 1e-6);

        // High score -> frequency approaches 1, high confidence
        let m_high = sample_mapping("w", 100.0);
        let tv_high = m_high.score_truth();
        assert!(tv_high.frequency > 0.9);
        assert!(tv_high.confidence > 0.8);
    }

    #[test]
    fn test_score_truth_negative() {
        // Negative scores are clamped to 0
        let m = sample_mapping("w", -5.0);
        let tv = m.score_truth();
        assert!((tv.frequency - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_eval_store_best_model() {
        let mut store = NsmEvalStore::new();
        store.add_eval("happy", sample_eval("happy", "gpt-4o", 3.5));
        store.add_eval("happy", sample_eval("happy", "claude-sonnet", 4.2));
        store.add_eval("happy", sample_eval("happy", "llama-70b", 2.8));

        let best = store.best_model_for_word("happy").unwrap();
        assert_eq!(best.model_name, "claude-sonnet");
        assert!((best.mapping.total_score - 4.2).abs() < 1e-6);
    }

    #[test]
    fn test_eval_store_best_model_missing_word() {
        let store = NsmEvalStore::new();
        assert!(store.best_model_for_word("nonexistent").is_none());
    }

    #[test]
    fn test_compare_models() {
        let mut store = NsmEvalStore::new();

        // Model A wins on "happy" and "sad", model B wins on "angry"
        store.add_eval("happy", sample_eval("happy", "model-a", 4.0));
        store.add_eval("happy", sample_eval("happy", "model-b", 3.0));

        store.add_eval("sad", sample_eval("sad", "model-a", 3.5));
        store.add_eval("sad", sample_eval("sad", "model-b", 2.5));

        store.add_eval("angry", sample_eval("angry", "model-a", 2.0));
        store.add_eval("angry", sample_eval("angry", "model-b", 3.0));

        let cmp = store.compare_models("model-a", "model-b");
        assert_eq!(cmp.model_a, "model-a");
        assert_eq!(cmp.model_b, "model-b");
        assert_eq!(cmp.a_wins, 2);
        assert_eq!(cmp.b_wins, 1);
        assert_eq!(cmp.ties, 0);
        assert!(cmp.a_avg_score > cmp.b_avg_score);

        // Truth should favor model A
        assert!(cmp.truth.frequency > 0.5);
        // Confidence should be moderate (3 data points, k=5 horizon)
        assert!(cmp.truth.confidence > 0.0 && cmp.truth.confidence < 1.0);
        let expected_conf = 3.0 / (3.0 + 5.0);
        assert!((cmp.truth.confidence - expected_conf).abs() < 1e-6);
    }

    #[test]
    fn test_compare_models_no_overlap() {
        let mut store = NsmEvalStore::new();
        store.add_eval("happy", sample_eval("happy", "model-a", 4.0));
        store.add_eval("sad", sample_eval("sad", "model-b", 3.0));

        let cmp = store.compare_models("model-a", "model-b");
        assert_eq!(cmp.a_wins, 0);
        assert_eq!(cmp.b_wins, 0);
        assert_eq!(cmp.ties, 0);
        // No overlap -> frequency 0.5 (unknown), confidence 0.0
        assert!((cmp.truth.frequency - 0.5).abs() < 1e-6);
        assert!((cmp.truth.confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_record_batch() {
        let mut store = NsmEvalStore::new();
        store.add_eval("happy", sample_eval("happy", "gpt-4o", 3.5));

        let batch = store.to_record_batch();

        // sample_eval has 2 grader scores, so 2 rows
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 9);

        // Verify schema field names
        let schema = batch.schema();
        assert_eq!(schema.field(0).name(), "word");
        assert_eq!(schema.field(1).name(), "model");
        assert_eq!(schema.field(2).name(), "explication");
        assert_eq!(schema.field(3).name(), "primes_ratio");
        assert_eq!(schema.field(4).name(), "molecules_ratio");
        assert_eq!(schema.field(5).name(), "total_score");
        assert_eq!(schema.field(6).name(), "uses_original_word");
        assert_eq!(schema.field(7).name(), "adj_score");
        assert_eq!(schema.field(8).name(), "avg_delta_log");

        // Verify data types
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(3).data_type(), &DataType::Float32);
        assert_eq!(schema.field(6).data_type(), &DataType::Boolean);

        // Verify values
        let words = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(words.value(0), "happy");
        assert_eq!(words.value(1), "happy");

        let scores = batch
            .column(5)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!((scores.value(0) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_to_record_batch_no_graders() {
        let mut store = NsmEvalStore::new();
        let eval = NsmModelEval {
            model_name: "test-model".to_string(),
            mapping: sample_mapping("word", 2.0),
            grader_scores: vec![], // no graders
        };
        store.add_eval("word", eval);

        let batch = store.to_record_batch();
        // One row with NaN grader columns
        assert_eq!(batch.num_rows(), 1);

        let adj = batch
            .column(7)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!(adj.value(0).is_nan());
    }

    #[test]
    fn test_roundtrip_spo() {
        let mut store = NsmEvalStore::new();
        store.add_eval("happy", sample_eval("happy", "gpt-4o", 3.5));
        store.add_eval("happy", sample_eval("happy", "claude-sonnet", 4.0));

        let spo_records = store.to_spo_records();

        // Each eval produces: 3 mapping triples + 2 grader triples + 1 scored_by = 6
        // Two evals = 12 total
        assert_eq!(spo_records.len(), 12);

        // All keys should be unique
        let keys: Vec<u64> = spo_records.iter().map(|(k, _, _)| *k).collect();
        let mut deduped = keys.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(keys.len(), deduped.len(), "All SPO keys must be unique");

        // All truth values should be valid (frequency and confidence in [0,1])
        for (_, _, tv) in &spo_records {
            assert!(tv.frequency >= 0.0 && tv.frequency <= 1.0);
            assert!(tv.confidence >= 0.0 && tv.confidence <= 1.0);
        }

        // Verify we can insert into an SpoStore
        let mut spo_store = super::super::store::SpoStore::new();
        for (key, record, _) in &spo_records {
            spo_store.insert(*key, record);
        }
        assert_eq!(spo_store.len(), 12);
    }
}
