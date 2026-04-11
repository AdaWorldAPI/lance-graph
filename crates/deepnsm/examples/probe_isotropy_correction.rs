//! Probe: DeepNSM Isotropy Correction (Regime A of γ+φ)
//!
//! ## Purpose
//!
//! Probe `probe_semantic_sanity.rs` measured DeepNSM's 4096² u8 distance
//! matrix and found:
//!
//! - Row-sum CV: 0.3087  (rows differ — good)
//! - Participation ratio: 1.53 out of 4096  (top-1 eigenvalue = 80.66% of variance — bad)
//!
//! The row-sum CV passes but the eigenspectrum is dominated by a single
//! principal axis (almost certainly the Zipfian word-frequency / corpus-
//! centrality axis inherited from COCA). This is the SAME structural
//! pathology that the canonical `bf16-hhtl-terrain.md` § C3 Regime A
//! describes: a distribution with a dominant center-of-mass that wastes
//! most of the palette's quantization range.
//!
//! ## What this probe measures
//!
//! The simplest Regime A operation: **per-row mean centering**. For each
//! word row in the 4096² matrix, subtract the row mean from every cell.
//! This is the first-step isotropy correction — equivalent to subtracting
//! the top principal component if that component aligns with the row mean
//! direction (which it typically does for Zipfian-biased embeddings, per
//! Mu et al. "All-but-the-top" 2018).
//!
//! If per-row mean centering meaningfully reduces the top-1 eigenvalue
//! share, then the full γ+φ Regime A pipeline (GammaProfile calibration
//! + gamma_encode + phi_encode in `bgz-tensor/src/gamma_phi.rs`) will
//! work even better when applied as a pre-quantization step in the next
//! DeepNSM bake. If per-row centering does NOT help, the isotropy has
//! a different root cause and γ+φ won't fix it.
//!
//! ## What this probe does NOT do
//!
//! - Does NOT apply full `gamma_encode` or `phi_encode` (those are in
//!   `bgz-tensor`, and this probe's library crate is zero-dep). Mean
//!   centering is the simplest Regime A operation that demonstrates the
//!   principle without adding a library dependency. If the result is
//!   positive, the follow-up probe can live in `bgz-tensor/examples/`
//!   where gamma_phi is native.
//! - Does NOT regenerate the DeepNSM codebook (that requires the v5
//!   release package — see item 10 in the session todo list).
//! - Does NOT touch any SIMD optimization code. `ndarray::simd::F32x16`
//!   is USED (via the LazyLock CPU dispatch in `ndarray/src/simd.rs`),
//!   but the hand-tuned backends (`simd_avx512.rs`, `simd_avx2.rs`,
//!   `simd_amx.rs`) are read-only referents. This probe never edits them.
//!
//! ## SIMD dispatch
//!
//! The per-row centering hot loop (16M element updates) uses
//! [`ndarray::simd::F32x16`] via its `Sub`/`SubAssign` impls. The
//! dispatch is implicit: `ndarray::simd` is a LazyLock CPU detect that
//! routes to `simd_avx512.rs` on AVX-512 hardware, `simd_avx2.rs` on
//! older x86, and `simd_amx.rs` on Apple Silicon. The probe never names
//! a specific backend — it just uses `F32x16` and lets the dispatch
//! pick the right path at runtime.
//!
//! `ndarray` is a DEV-DEPENDENCY in `deepnsm/Cargo.toml` (used only by
//! examples), so the deepnsm LIBRARY surface stays zero-dep. Cargo
//! `cargo check --lib -p deepnsm` does not pull in ndarray.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml \
//!     --example probe_isotropy_correction --release
//! ```
//!
//! Requires `/tmp/deepnsm_distance_4096x4096.u8` to exist (produced by
//! the earlier `probe_semantic_sanity` example run).

use std::fs;

use ndarray::simd::F32x16;

const INPUT_PATH: &str = "/tmp/deepnsm_distance_4096x4096.u8";
const OUTPUT_PATH: &str = "/tmp/deepnsm_distance_4096x4096_centered.f32";
const K: usize = 4096;

fn main() {
    println!("# Probe: DeepNSM Isotropy Correction (Regime A: per-row mean centering)");
    println!();

    // 1. Load the raw 4096² u8 matrix
    let bytes = match fs::read(INPUT_PATH) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: could not read {}: {}", INPUT_PATH, e);
            eprintln!();
            eprintln!("This probe expects the matrix dump produced by");
            eprintln!("`probe_semantic_sanity`. Run that first:");
            eprintln!();
            eprintln!("  cargo run --manifest-path crates/deepnsm/Cargo.toml \\");
            eprintln!("      --example probe_semantic_sanity --release");
            eprintln!();
            std::process::exit(2);
        }
    };
    assert_eq!(
        bytes.len(),
        K * K,
        "expected {} bytes (4096x4096 u8), got {}",
        K * K,
        bytes.len()
    );

    println!("Input: {}", INPUT_PATH);
    println!("Dimensions: {}x{} = {} u8 values ({} MB)", K, K, K * K, (K * K) / (1024 * 1024));
    println!();

    // 2. Convert u8 → f32 row-major (4096² f32 = 64 MB in RAM)
    let mut matrix: Vec<f32> = bytes.iter().map(|&v| v as f32).collect();
    drop(bytes);

    // 3. Baseline row-sum statistics
    let mut baseline_row_sums = vec![0.0_f64; K];
    for i in 0..K {
        let row_sum: f64 = matrix[i * K..(i + 1) * K].iter().map(|&v| v as f64).sum();
        baseline_row_sums[i] = row_sum;
    }
    let baseline_mean_row_sum: f64 = baseline_row_sums.iter().sum::<f64>() / K as f64;
    let baseline_row_sum_var: f64 = baseline_row_sums
        .iter()
        .map(|&s| (s - baseline_mean_row_sum).powi(2))
        .sum::<f64>()
        / K as f64;
    let baseline_row_sum_std = baseline_row_sum_var.sqrt();
    let baseline_cv = if baseline_mean_row_sum.abs() > 1e-9 {
        baseline_row_sum_std / baseline_mean_row_sum
    } else {
        0.0
    };

    println!("## Baseline (raw u8 matrix)");
    println!("- mean row sum: {:.2}", baseline_mean_row_sum);
    println!("- std of row sums: {:.2}", baseline_row_sum_std);
    println!("- coefficient of variation: {:.4}", baseline_cv);
    println!("  (matches probe_semantic_sanity finding: CV ≈ 0.31)");
    println!();

    // 4. Per-row mean centering (the Regime A operation).
    //    Hot loop uses ndarray::simd::F32x16 with explicit fused multiply-add
    //    (the add_mul / mul_add primitive the workspace prefers over plain
    //    subtract). The identity `v - m = v * 1 + (-m)` maps directly to
    //    VFMADD213PS on AVX-512, VFMADD on FMA3, or the AMX equivalent via
    //    the LazyLock CPU dispatch in ndarray/src/simd.rs. We never name a
    //    specific backend — only the exported F32x16 type and its mul_add.
    //    K = 4096 is a multiple of 16, so no scalar tail is needed.
    assert_eq!(K % 16, 0, "F32x16 path requires K % 16 == 0");
    let one = F32x16::splat(1.0_f32);
    let mut row_means = vec![0.0_f64; K];
    for i in 0..K {
        let m: f64 = baseline_row_sums[i] / K as f64;
        row_means[i] = m;
        // v * 1 + (-m) = v - m (explicit FMA for workspace consistency)
        let neg_m_splat = F32x16::splat(-(m as f32));
        let row_start = i * K;
        let mut j = 0;
        while j + 16 <= K {
            let v = F32x16::from_slice(&matrix[row_start + j..row_start + j + 16]);
            let centered = v.mul_add(one, neg_m_splat);
            centered.copy_to_slice(&mut matrix[row_start + j..row_start + j + 16]);
            j += 16;
        }
    }

    println!("## Applied: per-row mean centering");
    println!("- each row i had its row mean ({}..{}) subtracted",
        row_means.iter().cloned().fold(f64::INFINITY, f64::min) as i64,
        row_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as i64);
    println!("- row mean statistics: ");
    let mean_of_means: f64 = row_means.iter().sum::<f64>() / K as f64;
    let var_of_means: f64 = row_means
        .iter()
        .map(|&m| (m - mean_of_means).powi(2))
        .sum::<f64>()
        / K as f64;
    println!("    mean of row means: {:.4}", mean_of_means);
    println!("    std  of row means: {:.4}", var_of_means.sqrt());
    println!();

    // 5. Post-correction statistics
    let mut corrected_row_sums = vec![0.0_f64; K];
    for i in 0..K {
        let row_sum: f64 = matrix[i * K..(i + 1) * K].iter().map(|&v| v as f64).sum();
        corrected_row_sums[i] = row_sum;
    }
    let corrected_mean_row_sum: f64 = corrected_row_sums.iter().sum::<f64>() / K as f64;
    // All row sums should now be ~0 after centering; the std should be near zero.
    let corrected_row_sum_var: f64 = corrected_row_sums
        .iter()
        .map(|&s| (s - corrected_mean_row_sum).powi(2))
        .sum::<f64>()
        / K as f64;
    let corrected_row_sum_std = corrected_row_sum_var.sqrt();

    println!("## Post-correction row-sum check");
    println!("- mean row sum: {:.4e}  (should be ~0)", corrected_mean_row_sum);
    println!("- std row sum: {:.4e}", corrected_row_sum_std);
    println!("  (non-zero std → floating-point roundoff; genuine centering)");
    println!();

    // 6. Per-row variance (now the primary signal after mean is removed)
    let mut per_row_variance = vec![0.0_f64; K];
    for i in 0..K {
        let row: &[f32] = &matrix[i * K..(i + 1) * K];
        let var: f64 = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / K as f64;
        per_row_variance[i] = var;
    }
    let mean_per_row_var: f64 = per_row_variance.iter().sum::<f64>() / K as f64;
    let var_of_per_row_var: f64 = per_row_variance
        .iter()
        .map(|&v| (v - mean_per_row_var).powi(2))
        .sum::<f64>()
        / K as f64;

    println!("## Per-row variance (centered matrix)");
    println!("- mean per-row variance: {:.2}", mean_per_row_var);
    println!("- std  per-row variance: {:.2}", var_of_per_row_var.sqrt());
    println!("  (if ~constant across rows, remaining structure is isotropic around the mean);");
    println!("  (if highly variable, rows have different spreads = more real structure)");
    println!();

    // 7. Write the centered matrix as f32 for Python eigenspectrum follow-up
    let mut out_bytes: Vec<u8> = Vec::with_capacity(K * K * 4);
    for v in &matrix {
        out_bytes.extend_from_slice(&v.to_le_bytes());
    }
    match fs::write(OUTPUT_PATH, &out_bytes) {
        Ok(_) => println!(
            "## Dump\n\nWrote {} bytes of f32-centered matrix to {}",
            out_bytes.len(),
            OUTPUT_PATH
        ),
        Err(e) => eprintln!("WARN: could not write {}: {}", OUTPUT_PATH, e),
    }
    println!();

    // 8. Verdict criteria (stated BEFORE looking at the Python result)
    println!("## Verdict criteria (stated BEFORE Python eigenspectrum run)");
    println!();
    println!("After running the Python follow-up, compare the participation ratio:");
    println!();
    println!("- **PASS (γ+φ Regime A validated)**: participation ratio > 20");
    println!("  AND top-1 eigenvalue < 30% of total variance. This would");
    println!("  confirm that per-row mean centering alone reduces the");
    println!("  isotropy bias meaningfully, and the full γ+φ pipeline");
    println!("  (gamma_encode + phi_encode) in bgz-tensor/src/gamma_phi.rs");
    println!("  will work even better. The 4096 codebook generation should");
    println!("  apply GammaProfile calibration before palette construction.");
    println!();
    println!("- **MARGINAL**: 10 < PR < 20, top-1 in [30%, 60%]. Mean");
    println!("  centering helps but leaves meaningful top-component");
    println!("  dominance. A richer γ+φ treatment (per-item calibration,");
    println!("  or top-2/top-3 PC removal per Mu et al.) may be needed.");
    println!();
    println!("- **FAIL**: PR stays < 10 OR top-1 stays > 60%. Per-row");
    println!("  mean centering is insufficient. The isotropy has a");
    println!("  different root cause — possibly the underlying CAM-PQ");
    println!("  codebook needs regeneration with isotropy-corrected");
    println!("  source embeddings, not post-hoc matrix correction.");
    println!();

    // 9. Python follow-up snippet
    println!("## Python follow-up for eigenspectrum comparison");
    println!();
    println!("```python");
    println!("import numpy as np");
    println!();
    println!("# Original (uncorrected) matrix");
    println!("d_raw = np.fromfile('/tmp/deepnsm_distance_4096x4096.u8', dtype=np.uint8)");
    println!("d_raw = d_raw.reshape(4096, 4096).astype(np.float64)");
    println!("sim_raw = 1.0 - d_raw / max(d_raw.max(), 1)");
    println!("np.fill_diagonal(sim_raw, 1.0)");
    println!("ev_raw = np.linalg.eigvalsh(sim_raw)");
    println!("pos_raw = ev_raw[ev_raw > 1e-6]");
    println!("pr_raw = (pos_raw.sum()**2) / (pos_raw**2).sum()");
    println!("top_raw = sorted(ev_raw, reverse=True)[:5]");
    println!();
    println!("# Centered matrix (this probe's output)");
    println!("d_ctr = np.fromfile('{}', dtype=np.float32)", OUTPUT_PATH);
    println!("d_ctr = d_ctr.reshape(4096, 4096).astype(np.float64)");
    println!("# Centered matrix is already zero-mean per-row; use it directly as the");
    println!("# centered similarity-like signal (larger negative values = more similar).");
    println!("# For a fair eigenspectrum comparison, also center per-column and treat as similarity.");
    println!("d_ctr_sym = 0.5 * (d_ctr + d_ctr.T)");
    println!("sim_ctr = -d_ctr_sym  # flip sign: higher = more similar");
    println!("ev_ctr = np.linalg.eigvalsh(sim_ctr)");
    println!("pos_ctr = ev_ctr[ev_ctr > 1e-6]");
    println!("pr_ctr = (pos_ctr.sum()**2) / (pos_ctr**2).sum() if pos_ctr.size > 0 else 0.0");
    println!("top_ctr = sorted(ev_ctr, reverse=True)[:5]");
    println!();
    println!("print(f'=== Baseline (raw) ===')");
    println!("print(f'participation ratio: {{pr_raw:.2f}} out of 4096')");
    println!("print(f'top 5 eigenvalues: {{[round(v,1) for v in top_raw]}}')");
    println!("print(f'=== Corrected (per-row centered) ===')");
    println!("print(f'participation ratio: {{pr_ctr:.2f}} out of 4096')");
    println!("print(f'top 5 eigenvalues: {{[round(v,1) for v in top_ctr]}}')");
    println!("print(f'=== Delta ===')");
    println!("print(f'PR change: {{pr_ctr - pr_raw:+.2f}}')");
    println!("print(f'top-1 ratio change: raw {{pos_raw[-1]/pos_raw.sum():.3f}} → ctr {{pos_ctr[-1]/pos_ctr.sum() if pos_ctr.size > 0 else 0:.3f}}')");
    println!("```");
    println!();
    println!("## Next action based on Python output");
    println!();
    println!("Record the result in `.claude/probe_m1_result_2026_04_11.md`");
    println!("as a follow-up section, AND update the terrain document");
    println!("`.claude/knowledge/bf16-hhtl-terrain.md § Probe Queue` with");
    println!("the outcome for the isotropy-correction probe. If PASS,");
    println!("promote the Regime A recommendation in § C3 to a measured");
    println!("finding; if FAIL, add a note about which follow-up probe");
    println!("(richer γ+φ, top-k PC removal, source regeneration) is the");
    println!("correct next step.");
}
