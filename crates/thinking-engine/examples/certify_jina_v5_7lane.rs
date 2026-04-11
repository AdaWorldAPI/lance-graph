//! Certify: Jina v5 7-lane encoder output vs F32 reference.
//!
//! Reads the 7-lane encoder output at
//! `crates/thinking-engine/data/jina-v5-7lane/`, loads the
//! `cosine_matrix_256x256.f32` as the atomic-clock reference, and measures
//! how much information each lane preserves via Pearson r, Spearman ρ, and
//! Cronbach α, all reported to 4 decimal places.
//!
//! **Prerequisites**: the 7-lane encoder must have been run first to
//! produce the 8 output files. Run:
//!
//! ```sh
//! cargo run --release --features calibration --example seven_lane_encoder \
//!     --manifest-path crates/thinking-engine/Cargo.toml -- jina-v5
//! ```
//!
//! After that, run this certification:
//!
//! ```sh
//! cargo run --release --features calibration --example certify_jina_v5_7lane \
//!     --manifest-path crates/thinking-engine/Cargo.toml
//! ```
//!
//! **Certification policy** per `lance-graph/CLAUDE.md § Certification Process`
//! and the `certification-officer` agent card:
//!
//! | Lane | Primary metric | Target     | Role |
//! |------|----------------|------------|------|
//! | 1    | Spearman ρ     | ≥ 0.9990   | u8 CDF (rank preservation) |
//! | 2    | Pearson r      | ≥ 0.9980   | i8 direct `round(cos × 127)` |
//! | 3    | Spearman ρ     | ≥ 0.9990   | u8 γ+φ (CDF after gamma+phi) |
//! | 4    | Pearson r      | ≥ 0.9980   | i8 γ+φ signed |
//! | 5    | —              | informational | SiLU delta L2 norm (zero for token_embd) |
//! | 6    | all three      | **≥ 0.9999** | **BF16 RNE — atomic clock** |
//! | 7    | —              | informational | spiral drift mean/max |
//!
//! Lane 6 is the atomic-clock lab-BF16 lane. It must hit 0.9999 or better
//! on ALL THREE metrics simultaneously. The other lanes have a single
//! primary metric; Lanes 5 and 7 are informational and do not contribute
//! to the pass/fail verdict.
//!
//! NaN scan runs at every stage. Any NaN halts with exit code 2.
//!
//! Output: JSON report at `.claude/knowledge/certification/jina-v5-small_7lane.json`.

#[cfg(not(feature = "calibration"))]
fn main() { eprintln!("Requires --features calibration"); }

#[cfg(feature = "calibration")]
fn main() {
    use bgz_tensor::quality;

    const DATA_DIR: &str = "crates/thinking-engine/data/jina-v5-7lane";
    const N_CENT: usize = 256;
    const N_PAIRS_UPPER: usize = N_CENT * (N_CENT - 1) / 2;

    // Target thresholds (all at 4 decimal places).
    const TARGET_LAB_BF16: f64 = 0.9999;
    const TARGET_COMPRESSED: f64 = 0.9980;
    const TARGET_RANK: f64 = 0.9990;

    println!("═══════════════════════════════════════════════════════════");
    println!("  CERTIFY: Jina v5 7-lane encoder output");
    println!("  Source: {}", DATA_DIR);
    println!("═══════════════════════════════════════════════════════════\n");

    // ─── Step 1: Load the F32 reference matrix ───
    //
    // cosine_matrix_256x256.f32 is produced by the encoder from the live
    // safetensors source at run time. It IS the atomic clock for this
    // certification run — we do not need to re-derive it from the
    // safetensors because the encoder's Step 5 (centroid averaging +
    // pairwise cosine) is deterministic given the same source + the
    // hardcoded centroid-0 start. The reference is the f32 matrix on
    // disk; every lane is measured against it.
    println!("[1] Loading F32 reference matrix");
    let ref_path = format!("{}/cosine_matrix_{}x{}.f32", DATA_DIR, N_CENT, N_CENT);
    let ref_bytes = std::fs::read(&ref_path).unwrap_or_else(|e| {
        eprintln!("  FAILED to read {}: {}", ref_path, e);
        eprintln!("  Prerequisite: run seven_lane_encoder first to produce the 7-lane tables.");
        eprintln!("  See the module docstring for the exact command.");
        std::process::exit(1);
    });
    let ref_f32: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if ref_f32.len() != N_CENT * N_CENT {
        eprintln!(
            "  ref length mismatch: got {}, expected {}",
            ref_f32.len(),
            N_CENT * N_CENT
        );
        std::process::exit(1);
    }

    // Extract upper-triangular (off-diagonal) into a flat Vec<f64> for
    // pairwise comparison. The diagonal is always 1.0 (self-similarity)
    // and provides no information; including it would bias every metric
    // toward 1.0.
    let ref_upper: Vec<f64> = upper_triangular_f32(&ref_f32, N_CENT);
    assert_eq!(ref_upper.len(), N_PAIRS_UPPER);

    // NaN scan stage 1: reference matrix.
    let nan_in_ref = ref_upper.iter().filter(|v| v.is_nan()).count();
    if nan_in_ref > 0 {
        eprintln!("  NaN in reference matrix: {} / {} values. Halting.",
            nan_in_ref, ref_upper.len());
        std::process::exit(2);
    }
    let ref_min = ref_upper.iter().copied().fold(f64::INFINITY, f64::min);
    let ref_max = ref_upper.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ref_mean = ref_upper.iter().sum::<f64>() / ref_upper.len() as f64;
    println!(
        "  {} pairs, cos ∈ [{:.6}, {:.6}], mean {:.6}",
        ref_upper.len(), ref_min, ref_max, ref_mean
    );

    // Accumulators for the JSON report.
    let mut lanes: Vec<LaneReport> = Vec::new();

    // ─── Step 2: Lane 1 — u8 CDF (percentile rank) ───
    println!("\n[2] Lane 1: u8 CDF (percentile rank)");
    let lane1 = read_lane_u8(&format!(
        "{}/distance_table_{}x{}.u8", DATA_DIR, N_CENT, N_CENT
    ));
    let lane1_upper = upper_triangular_u8(&lane1, N_CENT);
    assert_eq!(lane1_upper.len(), N_PAIRS_UPPER);
    lanes.push(measure_lane(
        "lane_1_u8_cdf", "spearman", TARGET_RANK,
        &ref_upper, &lane1_upper,
    ));

    // ─── Step 3: Lane 2 — i8 direct ───
    println!("\n[3] Lane 2: i8 direct (round(cos × 127))");
    let lane2 = read_lane_i8(&format!(
        "{}/distance_table_{}x{}.i8", DATA_DIR, N_CENT, N_CENT
    ));
    let lane2_upper = upper_triangular_i8(&lane2, N_CENT);
    lanes.push(measure_lane(
        "lane_2_i8_direct", "pearson", TARGET_COMPRESSED,
        &ref_upper, &lane2_upper,
    ));

    // ─── Step 4: Lane 3 — u8 γ+φ (CDF after gamma+phi) ───
    println!("\n[4] Lane 3: u8 γ+φ (CDF after gamma+phi)");
    let lane3 = read_lane_u8(&format!(
        "{}/distance_table_{}x{}.gamma_phi.u8", DATA_DIR, N_CENT, N_CENT
    ));
    let lane3_upper = upper_triangular_u8(&lane3, N_CENT);
    lanes.push(measure_lane(
        "lane_3_u8_gamma_phi", "spearman", TARGET_RANK,
        &ref_upper, &lane3_upper,
    ));

    // ─── Step 5: Lane 4 — i8 γ+φ signed ───
    //
    // Primary metric is Spearman (rank preservation), NOT Pearson.
    // Lane 4 applies a nonlinear γ+φ transform (log-gamma + φ-log) and
    // then quantizes to i8. Pearson through a nonlinear monotonic
    // transform is degraded by design even when the transform is
    // perfect; Spearman captures rank preservation correctly across
    // any monotone map. Same rationale as Lanes 1 and 3.
    println!("\n[5] Lane 4: i8 γ+φ signed");
    let lane4 = read_lane_i8(&format!(
        "{}/distance_table_{}x{}.gamma_phi.i8", DATA_DIR, N_CENT, N_CENT
    ));
    let lane4_upper = upper_triangular_i8(&lane4, N_CENT);
    lanes.push(measure_lane(
        "lane_4_i8_gamma_phi_signed", "spearman", TARGET_RANK,
        &ref_upper, &lane4_upper,
    ));

    // ─── Step 6: Lane 5 — SiLU delta (informational) ───
    println!("\n[6] Lane 5: SiLU delta (informational, expected zero for token_embd)");
    let lane5_path = format!("{}/silu_deltas_{}x{}.f32", DATA_DIR, N_CENT, N_CENT);
    let lane5_bytes = std::fs::read(&lane5_path).unwrap_or_default();
    let lane5_f32: Vec<f32> = lane5_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let lane5_l2_norm: f64 = lane5_f32
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    let lane5_max_abs: f64 = lane5_f32
        .iter()
        .map(|&v| (v as f64).abs())
        .fold(0.0f64, f64::max);
    let lane5_nan = lane5_f32.iter().filter(|v| v.is_nan()).count();
    if lane5_nan > 0 {
        eprintln!("  NaN in lane 5: {} values. Halting.", lane5_nan);
        std::process::exit(2);
    }
    println!(
        "  L2 norm = {:.6}, max |delta| = {:.6} ({} elements)",
        lane5_l2_norm, lane5_max_abs, lane5_f32.len()
    );

    // ─── Step 7: Lane 6 — BF16 RNE (atomic clock) ───
    //
    // This is the certification-critical lane. After the Lane 6 swap in
    // commit `e25e97d`, the encoder produces these BF16 bytes via
    // `f32_to_bf16_batch_rne` (commit `c489d31`), which matches hardware
    // `_mm512_cvtneps_pbh` byte-for-byte on 1M inputs. Decoding BF16 back
    // to F32 is the trivial lossless shift via `bf16_to_f32_batch`.
    //
    // The metric compares the F32 reference (the encoder's
    // `cosine_matrix_256x256.f32`) against the BF16-roundtripped values.
    // The only difference is the ~1 ULP RNE truncation per element. All
    // three metrics (Pearson, Spearman, Cronbach α) must round to
    // ≥ 0.9999 at 4 decimal places.
    println!("\n[7] Lane 6: BF16 RNE (atomic clock lab-BF16 lane)");
    let lane6_bytes = std::fs::read(format!(
        "{}/distance_table_{}x{}.bf16", DATA_DIR, N_CENT, N_CENT
    )).unwrap_or_else(|e| {
        eprintln!("  FAILED to read lane 6: {}", e);
        std::process::exit(1);
    });
    let lane6_bf16: Vec<u16> = lane6_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut lane6_f32 = vec![0.0f32; lane6_bf16.len()];
    ndarray::simd::bf16_to_f32_batch(&lane6_bf16, &mut lane6_f32);
    let lane6_upper = upper_triangular_f32(&lane6_f32, N_CENT);

    // NaN scan stage 2: BF16-decoded values.
    let nan_in_lane6 = lane6_upper.iter().filter(|v| v.is_nan()).count();
    if nan_in_lane6 > 0 {
        eprintln!("  NaN in lane 6 after BF16 decode: {} values. Halting.", nan_in_lane6);
        std::process::exit(2);
    }

    // Measure all three metrics simultaneously — Lane 6 is the atomic clock.
    // Cronbach α uses z-score-normalized inputs for consistency with the
    // other lanes. For Lane 6 specifically, both f32 columns are on the
    // same scale so raw and z-scored results are nearly identical (to 4
    // decimal places), but the normalization keeps the methodology uniform
    // across lanes and makes the metric definition scale-invariant.
    let pearson = quality::pearson(&ref_upper, &lane6_upper);
    let spearman = quality::spearman(&ref_upper, &lane6_upper);
    let ref_z = z_score_normalize(&ref_upper);
    let lane6_z = z_score_normalize(&lane6_upper);
    let cronbach_a = thinking_engine::cronbach::cronbach_alpha(
        &[ref_z.as_slice(), lane6_z.as_slice()]
    );
    let lane6_verdict = if pearson >= TARGET_LAB_BF16
        && spearman >= TARGET_LAB_BF16
        && (cronbach_a as f64) >= TARGET_LAB_BF16
    {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "  Pearson r       = {:.4}  target {:.4}  [{}]",
        pearson, TARGET_LAB_BF16,
        if pearson >= TARGET_LAB_BF16 { "pass" } else { "FAIL" }
    );
    println!(
        "  Spearman ρ      = {:.4}  target {:.4}  [{}]",
        spearman, TARGET_LAB_BF16,
        if spearman >= TARGET_LAB_BF16 { "pass" } else { "FAIL" }
    );
    println!(
        "  Cronbach α      = {:.4}  target {:.4}  [{}]",
        cronbach_a, TARGET_LAB_BF16,
        if (cronbach_a as f64) >= TARGET_LAB_BF16 { "pass" } else { "FAIL" }
    );
    println!("  overall lane 6 : {}", lane6_verdict);
    lanes.push(LaneReport {
        name: "lane_6_bf16_rne".to_string(),
        primary_metric: "all_three".to_string(),
        target: TARGET_LAB_BF16,
        pearson,
        spearman,
        cronbach_alpha: cronbach_a as f64,
        verdict: lane6_verdict.to_string(),
    });

    // ─── Step 8: Lane 7 — spiral drift (informational) ───
    println!("\n[8] Lane 7: spiral drift (informational, golden-step stride=11)");
    let lane7_path = format!("{}/spiral_drift_{}x{}.u8", DATA_DIR, N_CENT, N_CENT);
    let lane7 = std::fs::read(&lane7_path).unwrap_or_default();
    let drift_upper = upper_triangular_u8(&lane7, N_CENT);
    let drift_sum: f64 = drift_upper.iter().sum();
    let drift_mean = drift_sum / drift_upper.len().max(1) as f64;
    let drift_max = drift_upper
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    println!(
        "  mean = {:.2}, max = {:.2}  (u8 encoding of drift × 2550)",
        drift_mean, drift_max
    );

    // ─── Step 9: Overall verdict ───
    let required_lanes = ["lane_1_u8_cdf", "lane_2_i8_direct",
                          "lane_3_u8_gamma_phi", "lane_4_i8_gamma_phi_signed",
                          "lane_6_bf16_rne"];
    let overall = lanes
        .iter()
        .filter(|l| required_lanes.contains(&l.name.as_str()))
        .all(|l| l.verdict == "PASS");

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  OVERALL VERDICT: {}", if overall { "PASS" } else { "FAIL" });
    println!("═══════════════════════════════════════════════════════════");
    for l in &lanes {
        println!(
            "  {:30} Pearson {:.4}  Spearman {:.4}  Cronbach {:.4}  target {:.4}  [{}]",
            l.name, l.pearson, l.spearman, l.cronbach_alpha, l.target, l.verdict
        );
    }
    println!();

    // ─── Step 10: Write JSON report ───
    let report_dir = ".claude/knowledge/certification";
    std::fs::create_dir_all(report_dir).ok();
    let json_path = format!("{}/jina-v5-small_7lane.json", report_dir);

    let lanes_json: Vec<serde_json::Value> = lanes
        .iter()
        .map(|l| {
            serde_json::json!({
                "name": l.name,
                "primary_metric": l.primary_metric,
                "target": l.target,
                "pearson": round4(l.pearson),
                "spearman": round4(l.spearman),
                "cronbach_alpha": round4(l.cronbach_alpha),
                "verdict": l.verdict,
            })
        })
        .collect();

    let report = serde_json::json!({
        "source": {
            "model": "jina-v5-small",
            "data_dir": DATA_DIR,
            "reference_file": format!("cosine_matrix_{}x{}.f32", N_CENT, N_CENT),
            "n_centroids": N_CENT,
            "n_pairs_upper": N_PAIRS_UPPER,
            "reference_cos_min": round4(ref_min),
            "reference_cos_max": round4(ref_max),
            "reference_cos_mean": round4(ref_mean),
        },
        "derivation": {
            "class": "seven_lane_encoder",
            "lane_6_method": "ndarray::simd::f32_to_bf16_batch_rne",
            "lane_6_commit": "c489d31 (ndarray) + e25e97d (lance-graph)",
        },
        "nan_scan": { "passed": true, "events": [] },
        "lanes": lanes_json,
        "lane_5_silu_delta": {
            "l2_norm": round6(lane5_l2_norm),
            "max_abs": round6(lane5_max_abs),
            "note": "zero expected for token_embd (no gate)",
        },
        "lane_7_spiral_drift": {
            "mean": round4(drift_mean),
            "max": round4(drift_max),
            "stride": 11,
            "note": "u8 encoding of drift × 2550",
        },
        "verdict": if overall { "PASS" } else { "FAIL" },
        "provenance": {
            "branch": "claude/risc-thought-engine-TCZw7",
            "agent": ".claude/agents/certification-officer.md",
            "doctrine": ".claude/knowledge/certification-harness.md",
        },
    });

    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    ).unwrap_or_else(|e| {
        eprintln!("FAILED to write report: {}", e);
    });

    println!("Report written to: {}\n", json_path);

    if !overall {
        std::process::exit(1);
    }
}

// ─── Helpers ───

#[cfg(feature = "calibration")]
#[derive(Debug)]
struct LaneReport {
    name: String,
    primary_metric: String,
    target: f64,
    pearson: f64,
    spearman: f64,
    cronbach_alpha: f64,
    verdict: String,
}

#[cfg(feature = "calibration")]
fn read_lane_u8(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("  FAILED to read {}: {}", path, e);
        std::process::exit(1);
    })
}

#[cfg(feature = "calibration")]
fn read_lane_i8(path: &str) -> Vec<i8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("  FAILED to read {}: {}", path, e);
        std::process::exit(1);
    });
    bytes.into_iter().map(|b| b as i8).collect()
}

#[cfg(feature = "calibration")]
fn upper_triangular_f32(mat: &[f32], n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(mat[i * n + j] as f64);
        }
    }
    out
}

#[cfg(feature = "calibration")]
fn upper_triangular_u8(mat: &[u8], n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(mat[i * n + j] as f64);
        }
    }
    out
}

#[cfg(feature = "calibration")]
fn upper_triangular_i8(mat: &[i8], n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(mat[i * n + j] as f64);
        }
    }
    out
}

#[cfg(feature = "calibration")]
fn measure_lane(
    name: &str,
    primary: &str,
    target: f64,
    reference: &[f64],
    lane: &[f64],
) -> LaneReport {
    use bgz_tensor::quality;

    // NaN scan stage: lane values.
    let nan_count = lane.iter().filter(|v| v.is_nan()).count();
    if nan_count > 0 {
        eprintln!("  NaN in {}: {} values. Halting.", name, nan_count);
        std::process::exit(2);
    }

    let pearson = quality::pearson(reference, lane);
    let spearman = quality::spearman(reference, lane);

    // Cronbach α on z-score-normalized inputs so the metric is scale-
    // invariant across lenses with very different ranges (e.g., f32
    // cosines in [-0.19, 0.68] vs u8 levels in [0, 255]). Without this
    // normalization, Cronbach's variance-based formula collapses to
    // ~0 whenever the two lenses span incompatible scales, even when
    // they are perfectly rank-correlated. Standard psychometric
    // practice when items are on different measurement scales.
    let ref_z: Vec<f32> = z_score_normalize(reference);
    let lane_z: Vec<f32> = z_score_normalize(lane);
    let cronbach_a = thinking_engine::cronbach::cronbach_alpha(
        &[ref_z.as_slice(), lane_z.as_slice()]
    ) as f64;

    let metric_value = match primary {
        "pearson" => pearson,
        "spearman" => spearman,
        _ => pearson.min(spearman).min(cronbach_a),
    };
    let verdict = if metric_value >= target { "PASS" } else { "FAIL" };

    println!(
        "  Pearson {:.4}  Spearman {:.4}  Cronbach {:.4}  [primary {}: {:.4} target {:.4}] [{}]",
        pearson, spearman, cronbach_a, primary, metric_value, target, verdict
    );

    LaneReport {
        name: name.to_string(),
        primary_metric: primary.to_string(),
        target,
        pearson,
        spearman,
        cronbach_alpha: cronbach_a,
        verdict: verdict.to_string(),
    }
}

/// Z-score normalize: subtract mean, divide by std. Returns f32 for
/// feeding into `cronbach_alpha` which takes `&[&[f32]]`.
#[cfg(feature = "calibration")]
fn z_score_normalize(values: &[f64]) -> Vec<f32> {
    let n = values.len() as f64;
    if n < 2.0 {
        return values.iter().map(|&v| v as f32).collect();
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-12);
    values.iter().map(|&v| ((v - mean) / std) as f32).collect()
}

#[cfg(feature = "calibration")]
fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

#[cfg(feature = "calibration")]
fn round6(v: f64) -> f64 {
    (v * 1000000.0).round() / 1000000.0
}
