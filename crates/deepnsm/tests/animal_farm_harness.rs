//! Animal Farm forward-validation harness — D10 from the original plan.
//!
//! Scaffold only. The actual book text + ground-truth labels live in a
//! follow-up data PR. This file fixes the harness API + asserts the metric
//! shape we'll measure.
//!
//! META-AGENT: integration-test scaffold; no lib.rs wiring required —
//! `cargo test -p deepnsm --test animal_farm_harness` runs it.

#![cfg(test)]

#[derive(Debug, Clone)]
pub struct EpiphanyPrediction {
    pub at_chapter: u32,
    pub direction_phase: f32,
    pub initial_truth_freq: f32,
    pub initial_truth_conf: f32,
}

#[derive(Debug, Clone)]
pub struct GroundTruthBeat {
    pub at_chapter: u32,
    pub confirmed_direction: f32,
    pub matched_predictions: Vec<u32>,
}

#[derive(Debug)]
pub struct HarnessMetrics {
    pub epiphany_precision: f32,
    pub epiphany_recall: f32,
    pub arc_shift_f1: f32,
    pub direction_accuracy: f32,
}

pub fn evaluate(
    predictions: &[EpiphanyPrediction],
    ground_truth: &[GroundTruthBeat],
) -> HarnessMetrics {
    if predictions.is_empty() {
        return HarnessMetrics {
            epiphany_precision: 0.0,
            epiphany_recall: 0.0,
            arc_shift_f1: 0.0,
            direction_accuracy: 0.0,
        };
    }
    let confirmed: u32 = predictions
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            ground_truth
                .iter()
                .any(|b| b.matched_predictions.contains(&(*i as u32)))
        })
        .count() as u32;
    let precision = confirmed as f32 / predictions.len() as f32;
    let recall = confirmed as f32 / ground_truth.len().max(1) as f32;
    HarnessMetrics {
        epiphany_precision: precision,
        epiphany_recall: recall,
        arc_shift_f1: 2.0 * precision * recall / (precision + recall + 1e-9),
        direction_accuracy: 0.0, // populated when phase-direction comparison lands
    }
}

#[test]
fn harness_metrics_zero_for_empty_predictions() {
    let m = evaluate(&[], &[]);
    assert_eq!(m.epiphany_precision, 0.0);
    assert_eq!(m.epiphany_recall, 0.0);
}

#[test]
fn harness_metrics_perfect_for_all_confirmed() {
    let preds = vec![EpiphanyPrediction {
        at_chapter: 3,
        direction_phase: 0.0,
        initial_truth_freq: 0.6,
        initial_truth_conf: 0.5,
    }];
    let gt = vec![GroundTruthBeat {
        at_chapter: 5,
        confirmed_direction: 0.0,
        matched_predictions: vec![0],
    }];
    let m = evaluate(&preds, &gt);
    assert_eq!(m.epiphany_precision, 1.0);
    assert_eq!(m.epiphany_recall, 1.0);
}

#[test]
fn harness_metrics_zero_when_no_predictions_match() {
    let preds = vec![EpiphanyPrediction {
        at_chapter: 1,
        direction_phase: 0.0,
        initial_truth_freq: 0.5,
        initial_truth_conf: 0.5,
    }];
    let gt = vec![GroundTruthBeat {
        at_chapter: 2,
        confirmed_direction: 0.0,
        matched_predictions: vec![], // no matches
    }];
    let m = evaluate(&preds, &gt);
    assert_eq!(m.epiphany_precision, 0.0);
    assert_eq!(m.epiphany_recall, 0.0);
}

#[test]
fn harness_f1_is_harmonic_mean() {
    // 2 predictions, 1 confirmed -> precision 0.5
    // 1 ground-truth beat -> recall 1.0
    // F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 0.6666...
    let preds = vec![
        EpiphanyPrediction {
            at_chapter: 1,
            direction_phase: 0.0,
            initial_truth_freq: 0.5,
            initial_truth_conf: 0.5,
        },
        EpiphanyPrediction {
            at_chapter: 2,
            direction_phase: 0.0,
            initial_truth_freq: 0.5,
            initial_truth_conf: 0.5,
        },
    ];
    let gt = vec![GroundTruthBeat {
        at_chapter: 4,
        confirmed_direction: 0.0,
        matched_predictions: vec![0],
    }];
    let m = evaluate(&preds, &gt);
    assert!((m.arc_shift_f1 - 2.0 / 3.0).abs() < 1e-3);
}
