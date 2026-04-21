//! **LAB-ONLY.** D2.1 — token-agreement harness scaffold.
//!
//! The I11 cert gate per `lab-vs-canonical-surface.md` + the PR #219 → #220
//! lesson: a codec passes when decoded weights produce the **same top-k
//! tokens** as the Passthrough baseline on a real prompt set.
//! Reconstruction ICC is necessary but not sufficient.
//!
//! **Phase 2 scope split:**
//!
//! - **D2.1 (this module, scaffold):** `ReferenceModel` loader stub +
//!   `TopKAgreement` comparator + `TokenAgreementHarness::measure()` that
//!   returns `WireTokenAgreementResult { stub: true, backend: "stub", … }`.
//!   Tests the plumbing — top-k aggregation, divergence-position tracking,
//!   latency fields — without depending on safetensors I/O or real decode.
//! - **D2.2 (queued):** real decode-and-compare loop — load safetensors,
//!   run N token decodes per prompt, compare top-1 / top-5 per position.
//! - **D2.3 (queued):** `/v1/shader/token-agreement` handler wiring.
//!
//! **Pass gates** (when D2.2 lands):
//!
//! - `top1_rate ≥ 0.99`
//! - `top5_rate ≥ 0.999`
//!
//! Those thresholds are what certifies a codec for the canonical
//! `OrchestrationBridge` graduation (Phase 5 D5).

use crate::wire::{WireBaseline, WireCodecParams, WireTokenAgreementResult};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Stub reference-model descriptor. D2.2 replaces this with a real
/// safetensors loader that exposes the tensors + tokenizer + runtime
/// decoder. For now we just carry enough metadata to key the harness.
#[derive(Debug, Clone)]
pub struct ReferenceModel {
    pub path: PathBuf,
    pub path_hash: u64,
    /// When non-zero, synthetic token stream length for deterministic
    /// top-k comparison tests. D2.2 replaces with actual decode output.
    pub stub_token_count: u32,
}

impl ReferenceModel {
    /// Load a reference model from a safetensors root directory.
    ///
    /// **D2.1 stub behaviour:** records the path + computes a path hash;
    /// does NOT actually read safetensors or instantiate a decoder.
    /// D2.2 replaces this with a real loader driven by the existing
    /// `auto_detect::detect()` that returns a `ModelFingerprint`.
    pub fn load(path: &Path) -> Result<Self, TokenAgreementError> {
        // Minimal validation: the path exists or we emit a typed error.
        if !path.exists() {
            return Err(TokenAgreementError::ModelPathMissing {
                path: path.display().to_string(),
            });
        }
        let mut h = DefaultHasher::new();
        path.display().to_string().hash(&mut h);
        Ok(Self {
            path: path.to_path_buf(),
            path_hash: h.finish(),
            stub_token_count: 0,
        })
    }

    /// Construct a stub reference model without touching the filesystem.
    /// Used by tests that don't care about path validity.
    pub fn stub(tag: u64, stub_token_count: u32) -> Self {
        Self {
            path: PathBuf::from(format!("stub://{tag:#x}")),
            path_hash: tag,
            stub_token_count,
        }
    }
}

/// Error from the harness. Typed so callers can branch on cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenAgreementError {
    ModelPathMissing { path: String },
    EmptyPromptSet,
    TokenCountMismatch { reference: usize, candidate: usize },
    NotImplementedYet { what: &'static str },
}

impl std::fmt::Display for TokenAgreementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelPathMissing { path } => write!(f, "model path missing: {path}"),
            Self::EmptyPromptSet => write!(f, "prompt set empty"),
            Self::TokenCountMismatch { reference, candidate } => {
                write!(f, "token count mismatch: ref={reference} cand={candidate}")
            }
            Self::NotImplementedYet { what } => {
                write!(f, "not implemented yet: {what} (D2.2 scope)")
            }
        }
    }
}

impl std::error::Error for TokenAgreementError {}

/// Result of a top-k comparison for one prompt's token stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopKAgreement {
    pub top1_matches: u32,
    pub top5_matches: u32,
    pub total_positions: u32,
    /// Indices (within the prompt's token stream) where candidate's top-1
    /// token differs from reference's top-1 token. Useful for failure-mode
    /// analysis ("late-sequence drift" vs "random errors everywhere").
    pub divergence_positions: Vec<u32>,
}

impl TopKAgreement {
    /// Compare two token streams position-by-position.
    ///
    /// Reference and candidate each carry per-position top-k candidate
    /// lists. For each position, top-1 match = ref\[0\] == cand\[0\];
    /// top-5 match = ref\[0\] in cand\[..5\].
    pub fn compare(
        reference_topk: &[Vec<u32>],
        candidate_topk: &[Vec<u32>],
    ) -> Result<Self, TokenAgreementError> {
        if reference_topk.len() != candidate_topk.len() {
            return Err(TokenAgreementError::TokenCountMismatch {
                reference: reference_topk.len(),
                candidate: candidate_topk.len(),
            });
        }
        let total = reference_topk.len() as u32;
        let mut top1 = 0u32;
        let mut top5 = 0u32;
        let mut divergence = Vec::new();
        for (pos, (r, c)) in reference_topk.iter().zip(candidate_topk.iter()).enumerate() {
            let ref_top1 = r.first().copied();
            let cand_top1 = c.first().copied();
            if ref_top1.is_some() && ref_top1 == cand_top1 {
                top1 += 1;
            } else if ref_top1.is_some() {
                divergence.push(pos as u32);
            }
            if let Some(rt) = ref_top1 {
                if c.iter().take(5).any(|&t| t == rt) {
                    top5 += 1;
                }
            }
        }
        Ok(Self {
            top1_matches: top1,
            top5_matches: top5,
            total_positions: total,
            divergence_positions: divergence,
        })
    }

    /// top1 match rate ∈ [0, 1]. Passes the cert gate at ≥ 0.99 per D0.2.
    pub fn top1_rate(&self) -> f32 {
        if self.total_positions == 0 { 0.0 } else { self.top1_matches as f32 / self.total_positions as f32 }
    }

    /// top5 match rate ∈ [0, 1]. Passes the cert gate at ≥ 0.999 per D0.2.
    pub fn top5_rate(&self) -> f32 {
        if self.total_positions == 0 { 0.0 } else { self.top5_matches as f32 / self.total_positions as f32 }
    }

    /// Meets D0.2 acceptance thresholds (top1 ≥ 0.99 AND top5 ≥ 0.999).
    pub fn meets_cert_gate(&self) -> bool {
        self.top1_rate() >= 0.99 && self.top5_rate() >= 0.999
    }

    /// Aggregate across multiple per-prompt comparisons.
    /// All summed counters; divergence_positions concatenated with a prompt
    /// offset so callers can still localise failures.
    pub fn aggregate(per_prompt: &[TopKAgreement]) -> TopKAgreement {
        let mut agg = TopKAgreement {
            top1_matches: 0,
            top5_matches: 0,
            total_positions: 0,
            divergence_positions: Vec::new(),
        };
        let mut offset = 0u32;
        for p in per_prompt {
            agg.top1_matches += p.top1_matches;
            agg.top5_matches += p.top5_matches;
            for &d in &p.divergence_positions {
                agg.divergence_positions.push(offset + d);
            }
            offset += p.total_positions;
            agg.total_positions = offset;
        }
        agg
    }
}

/// Harness carrying the reference model + comparison context.
#[derive(Debug)]
pub struct TokenAgreementHarness {
    pub reference: ReferenceModel,
    pub baseline: WireBaseline,
    pub candidate: WireCodecParams,
    pub n_tokens: u32,
}

impl TokenAgreementHarness {
    pub fn new(
        reference: ReferenceModel,
        baseline: WireBaseline,
        candidate: WireCodecParams,
        n_tokens: u32,
    ) -> Self {
        Self { reference, baseline, candidate, n_tokens }
    }

    /// D2.1 STUB: returns a deterministic zero-rate result with `stub: true`
    /// + `backend: "stub"`. D2.2 replaces with real decode-and-compare.
    ///
    /// The stub result has zero latencies + empty divergence_positions so
    /// that a client inspecting it cannot confuse it for a real measurement
    /// — the `stub` flag makes the Phase 0 honesty machine-checkable at the
    /// type level (see EPIPHANIES 2026-04-20 "D0.2 stub flag is anti-#219
    /// defense at the type level").
    pub fn measure_stub(&self) -> Result<WireTokenAgreementResult, TokenAgreementError> {
        if self.n_tokens == 0 {
            return Err(TokenAgreementError::EmptyPromptSet);
        }
        Ok(WireTokenAgreementResult {
            top1_rate: 0.0,
            top5_rate: 0.0,
            divergence_positions: Vec::new(),
            per_layer_mse: Vec::new(),
            candidate_latency_us: 0,
            reference_latency_us: 0,
            stub: true,
            backend: "stub".to_string(),
        })
    }

    /// Full measure — D2.2 wires the real decode loop here.
    /// For now returns `NotImplementedYet` with a clear pointer.
    pub fn measure_full(&self) -> Result<WireTokenAgreementResult, TokenAgreementError> {
        Err(TokenAgreementError::NotImplementedYet {
            what: "real safetensors decode + top-k comparison (D2.2)",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{WireDistance, WireLaneWidth, WireResidualSpec, WireRotation};

    fn stub_candidate() -> WireCodecParams {
        WireCodecParams {
            subspaces: 6,
            centroids: 256,
            residual: WireResidualSpec { depth: 0, centroids: 256 },
            lane_width: WireLaneWidth::F32x16,
            pre_rotation: WireRotation::Identity,
            distance: WireDistance::AdcU8,
            calibration_rows: 2048,
            measurement_rows: 512,
            seed: 42,
        }
    }

    #[test]
    fn reference_model_stub_builds_without_filesystem() {
        let rm = ReferenceModel::stub(0xDEADBEEF, 128);
        assert_eq!(rm.path_hash, 0xDEADBEEF);
        assert_eq!(rm.stub_token_count, 128);
        assert!(rm.path.to_string_lossy().starts_with("stub://"));
    }

    #[test]
    fn reference_model_load_missing_path_yields_typed_error() {
        let err = ReferenceModel::load(Path::new("/nonexistent/xyz/model.safetensors")).unwrap_err();
        assert!(matches!(err, TokenAgreementError::ModelPathMissing { .. }));
    }

    #[test]
    fn topk_compare_identical_streams_is_perfect() {
        let r = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let c = r.clone();
        let a = TopKAgreement::compare(&r, &c).unwrap();
        assert_eq!(a.top1_matches, 3);
        assert_eq!(a.top5_matches, 3);
        assert_eq!(a.total_positions, 3);
        assert!(a.divergence_positions.is_empty());
        assert_eq!(a.top1_rate(), 1.0);
        assert_eq!(a.top5_rate(), 1.0);
        assert!(a.meets_cert_gate());
    }

    #[test]
    fn topk_compare_all_different_fails_cert_gate() {
        let r = vec![vec![1], vec![2], vec![3]];
        let c = vec![vec![10], vec![20], vec![30]];
        let a = TopKAgreement::compare(&r, &c).unwrap();
        assert_eq!(a.top1_matches, 0);
        assert_eq!(a.top5_matches, 0);
        assert_eq!(a.divergence_positions, vec![0, 1, 2]);
        assert!(!a.meets_cert_gate());
    }

    #[test]
    fn topk_top5_matches_when_top1_misses_but_in_top5() {
        // Reference top-1 = 7; candidate has 7 at position 3 in top-5.
        let r = vec![vec![7, 1, 2, 3, 4]];
        let c = vec![vec![1, 2, 7, 3, 4]];
        let a = TopKAgreement::compare(&r, &c).unwrap();
        assert_eq!(a.top1_matches, 0);
        assert_eq!(a.top5_matches, 1);
        assert_eq!(a.divergence_positions, vec![0]);
    }

    #[test]
    fn topk_mismatched_stream_lengths_yield_typed_error() {
        let r = vec![vec![1], vec![2]];
        let c = vec![vec![1]];
        let err = TopKAgreement::compare(&r, &c).unwrap_err();
        assert!(matches!(err, TokenAgreementError::TokenCountMismatch { reference: 2, candidate: 1 }));
    }

    #[test]
    fn topk_aggregate_sums_counters_and_offsets_divergence() {
        let p1 = TopKAgreement {
            top1_matches: 8,
            top5_matches: 10,
            total_positions: 10,
            divergence_positions: vec![2, 7],
        };
        let p2 = TopKAgreement {
            top1_matches: 5,
            top5_matches: 5,
            total_positions: 5,
            divergence_positions: vec![4],
        };
        let agg = TopKAgreement::aggregate(&[p1, p2]);
        assert_eq!(agg.top1_matches, 13);
        assert_eq!(agg.top5_matches, 15);
        assert_eq!(agg.total_positions, 15);
        // First prompt's divergences pass through at offset 0; second's at +10.
        assert_eq!(agg.divergence_positions, vec![2, 7, 14]);
    }

    #[test]
    fn cert_gate_passes_at_exact_thresholds() {
        // 990/1000 = 0.99 exactly (top1 threshold)
        // 999/1000 = 0.999 exactly (top5 threshold)
        let a = TopKAgreement {
            top1_matches: 990,
            top5_matches: 999,
            total_positions: 1000,
            divergence_positions: vec![],
        };
        assert!((a.top1_rate() - 0.99).abs() < 1e-6);
        assert!((a.top5_rate() - 0.999).abs() < 1e-6);
        assert!(a.meets_cert_gate(), "exactly at thresholds should pass");
    }

    #[test]
    fn cert_gate_fails_when_top1_below_threshold_even_if_top5_passes() {
        // 989/1000 = 0.989 (just under top1 0.99)
        // 999/1000 = 0.999 (exactly at top5)
        let a = TopKAgreement {
            top1_matches: 989,
            top5_matches: 999,
            total_positions: 1000,
            divergence_positions: vec![],
        };
        assert!(!a.meets_cert_gate(), "top1 just below threshold should fail even if top5 passes");
    }

    #[test]
    fn cert_gate_fails_when_top5_below_threshold_even_if_top1_passes() {
        // 990/1000 = 0.99 (exactly at top1)
        // 998/1000 = 0.998 (just under top5 0.999)
        let a = TopKAgreement {
            top1_matches: 990,
            top5_matches: 998,
            total_positions: 1000,
            divergence_positions: vec![],
        };
        assert!(!a.meets_cert_gate(), "top5 just below threshold should fail even if top1 passes");
    }

    #[test]
    fn harness_measure_stub_returns_machine_checkable_stub_flag() {
        let ref_model = ReferenceModel::stub(1, 16);
        let harness = TokenAgreementHarness::new(
            ref_model,
            WireBaseline::Passthrough,
            stub_candidate(),
            128,
        );
        let result = harness.measure_stub().unwrap();
        assert!(result.stub, "stub flag MUST be true so clients cannot confuse stub for real measurement");
        assert_eq!(result.backend, "stub");
        assert_eq!(result.top1_rate, 0.0);
        assert_eq!(result.top5_rate, 0.0);
        assert_eq!(result.candidate_latency_us, 0);
    }

    #[test]
    fn harness_measure_full_returns_not_implemented_pointing_at_d22() {
        let ref_model = ReferenceModel::stub(1, 16);
        let harness = TokenAgreementHarness::new(
            ref_model,
            WireBaseline::Passthrough,
            stub_candidate(),
            128,
        );
        let err = harness.measure_full().unwrap_err();
        assert!(matches!(err, TokenAgreementError::NotImplementedYet { .. }));
    }

    #[test]
    fn harness_measure_stub_rejects_zero_n_tokens() {
        let ref_model = ReferenceModel::stub(1, 16);
        let harness = TokenAgreementHarness::new(
            ref_model,
            WireBaseline::Passthrough,
            stub_candidate(),
            0,
        );
        let err = harness.measure_stub().unwrap_err();
        assert!(matches!(err, TokenAgreementError::EmptyPromptSet));
    }
}
