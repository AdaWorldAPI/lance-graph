//! Probe: DeepNSM Semantic Layer Sanity
//!
//! Purpose: verify the CANONICAL semantic layer — the 4096-word COCA/CAM-PQ
//! distance matrix computed by `DeepNsmEngine::load` — has real cluster
//! structure, as opposed to the EXPERIMENTAL Jina v5 256-centroid codebook
//! which was measured externally to have:
//!   - off-diag cosine mean 0.64
//!   - effective rank (participation ratio) 1.82 out of 256
//!   - 43.76% of pairs with cos > 0.9
//!   → degenerate null-context artifact, not a real semantic manifold
//!
//! The DeepNSM matrix is a completely different source: 96-dimensional
//! distributional vectors from COCA subgenre frequencies (1-billion-word
//! corpus, 96 subgenres), PQ-compressed via 6-subspace × 256-centroid
//! codebook, then pairwise L2 distances quantized to u8. Structurally
//! independent from transformer forward-pass isotropy bias.
//!
//! This probe answers one question: does the DeepNSM 4096² matrix have
//! real structure, or is it also degenerate?
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml \
//!     --example probe_semantic_sanity --release
//! ```
//!
//! ## What this probe does NOT do (no duplicate code)
//!
//! - Does NOT reimplement the CAM-PQ distance (calls deepnsm::Codebook)
//! - Does NOT reimplement the codebook loader (calls DeepNsmEngine::load)
//! - Does NOT reimplement the distance matrix (reads engine.distance_matrix)
//! - Does NOT reimplement vocabulary loading (calls Vocabulary::load)
//! - Does NOT add external crates (pure std + deepnsm's own public API)
//!
//! ## What this probe DOES do
//!
//! 1. Loads `DeepNsmEngine` from the on-disk word_frequency/ data
//! 2. Iterates the 4096² distance matrix via the public `get(a, b)` API
//! 3. Computes scalar statistics (mean, std, percentiles, row-sum CV)
//! 4. Dumps the raw 16 MB matrix to `/tmp/deepnsm_distance_4096x4096.u8`
//!    for Python post-processing (eigenspectrum, participation ratio)
//! 5. Prints a side-by-side comparison to the Jina v5 numbers
//!
//! Participation ratio and eigendecomposition require LAPACK which deepnsm
//! cannot pull in (zero-dep invariant). The raw matrix dump lets Python
//! compute these via numpy in a separate step.

use std::fs;
use std::path::PathBuf;

use deepnsm::DeepNsmEngine;
use deepnsm::spo::WordDistanceMatrix;

fn main() {
    println!("# Probe: DeepNSM Semantic Layer Sanity");
    println!();

    // 1. Load engine from on-disk data. CARGO_MANIFEST_DIR resolves at compile
    //    time to the deepnsm crate root regardless of where `cargo run` is invoked.
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("word_frequency");
    println!("Data dir: {}", data_dir.display());

    let engine = match DeepNsmEngine::load(&data_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ERROR: DeepNsmEngine::load failed: {}", e);
            std::process::exit(1);
        }
    };
    println!();

    let dm = &engine.distance_matrix;
    let k = WordDistanceMatrix::K;

    println!("## Matrix dimensions");
    println!("- K (vocabulary size) = {}", k);
    println!("- total cells = {} ({} MB)", k * k, (k * k) / (1024 * 1024));
    println!();

    // 2. Diagonal check (self-distance should be 0)
    let diag_all_zero = (0..k).all(|i| dm.get(i as u16, i as u16) == 0);
    println!("## Diagonal");
    println!("- all self-distances zero: {}", diag_all_zero);
    if !diag_all_zero {
        let nonzero_diagonals: Vec<(usize, u8)> = (0..k)
            .filter_map(|i| {
                let d = dm.get(i as u16, i as u16);
                if d != 0 { Some((i, d)) } else { None }
            })
            .take(5)
            .collect();
        println!("  (first 5 non-zero diagonals: {:?})", nonzero_diagonals);
    }
    println!();

    // 3. Upper-triangle statistics + row sums
    let n_pairs = k * (k - 1) / 2;
    let mut off: Vec<u32> = Vec::with_capacity(n_pairs);
    let mut row_sum: Vec<u64> = vec![0u64; k];
    for a in 0..k {
        for b in (a + 1)..k {
            let d = dm.get(a as u16, b as u16) as u32;
            off.push(d);
            row_sum[a] += d as u64;
            row_sum[b] += d as u64;
        }
    }

    // Also fill symmetric row sums (each cell counted twice in the full row)
    // The above already accounts for both directions via the pair loop.

    // Convert to f64 for stats
    let n = off.len() as f64;
    let mean: f64 = off.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = off.iter().map(|&v| {
        let diff = v as f64 - mean;
        diff * diff
    }).sum::<f64>() / n;
    let std_dev = var.sqrt();

    // Percentiles via sort
    let mut sorted = off.clone();
    sorted.sort_unstable();
    let pct = |p: f64| -> f64 {
        let idx = ((p * n).floor() as usize).min(sorted.len() - 1);
        sorted[idx] as f64
    };

    println!("## Off-diagonal distance distribution");
    println!("- pairs: {} (upper triangle of {}×{})", off.len(), k, k);
    println!("- mean:  {:.2}", mean);
    println!("- std:   {:.2}", std_dev);
    println!("- min:   {}", sorted[0]);
    println!("- max:   {}", sorted[sorted.len() - 1]);
    println!("- median: {:.1}", pct(0.5));
    println!("- 5%:    {:.1}", pct(0.05));
    println!("- 25%:   {:.1}", pct(0.25));
    println!("- 75%:   {:.1}", pct(0.75));
    println!("- 95%:   {:.1}", pct(0.95));
    println!();

    // 4. Histogram and tail concentration (u8 range, so 256 bins is exact)
    let mut histogram = [0u64; 256];
    for &d in &off {
        histogram[d as usize] += 1;
    }
    let bottom10: u64 = histogram.iter().take(10).sum();
    let top10: u64 = histogram.iter().rev().take(10).sum();
    println!("## Tail concentration");
    println!(
        "- pairs in bottom 10 u8 bins [0..10]: {} ({:.3}%)",
        bottom10,
        bottom10 as f64 / n * 100.0
    );
    println!(
        "- pairs in top 10 u8 bins [246..256]: {} ({:.3}%)",
        top10,
        top10 as f64 / n * 100.0
    );
    println!();

    // 5. Row-sum constancy (proxy for matrix isotropy)
    //    If all row sums are identical, the matrix is nearly isotropic and
    //    has no per-row distinguishing structure → degenerate.
    let row_sum_f64: Vec<f64> = row_sum.iter().map(|&s| s as f64).collect();
    let mean_rs = row_sum_f64.iter().sum::<f64>() / k as f64;
    let var_rs = row_sum_f64.iter().map(|&s| {
        let diff = s - mean_rs;
        diff * diff
    }).sum::<f64>() / k as f64;
    let std_rs = var_rs.sqrt();
    let cv = if mean_rs.abs() > 1e-9 { std_rs / mean_rs } else { 0.0 };
    println!("## Row-sum constancy (matrix isotropy proxy)");
    println!("- mean row sum: {:.2}", mean_rs);
    println!("- std  row sum: {:.2}", std_rs);
    println!("- coefficient of variation: {:.4}", cv);
    println!("  interpretation:");
    println!("    CV < 0.01 → near-isotropic, all rows look the same, degenerate");
    println!("    0.01 < CV < 0.10 → mild structure, probably usable");
    println!("    CV > 0.10 → strong per-row structure, real semantic surface");
    println!();

    // 6. Nearest-neighbor distance (excluding self)
    let mut nn_dist: Vec<u32> = Vec::with_capacity(k);
    for i in 0..k {
        let mut best = u32::MAX;
        for j in 0..k {
            if i == j { continue; }
            let d = dm.get(i as u16, j as u16) as u32;
            if d < best { best = d; }
        }
        nn_dist.push(best);
    }
    let nn_mean: f64 = nn_dist.iter().map(|&v| v as f64).sum::<f64>() / k as f64;
    let nn_var: f64 = nn_dist.iter().map(|&v| {
        let diff = v as f64 - nn_mean;
        diff * diff
    }).sum::<f64>() / k as f64;
    let nn_std = nn_var.sqrt();
    println!("## Nearest-neighbor distance (excluding self)");
    println!("- mean: {:.2}", nn_mean);
    println!("- std:  {:.2}", nn_std);
    println!("- min:  {}", nn_dist.iter().min().unwrap());
    println!("- max:  {}", nn_dist.iter().max().unwrap());
    println!("  (low mean/std → tight clustering; high mean → spread codebook)");
    println!();

    // 7. Dump raw matrix to /tmp for Python eigendecomposition
    let dump_path = "/tmp/deepnsm_distance_4096x4096.u8";
    let mut flat: Vec<u8> = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            flat.push(dm.get(i as u16, j as u16));
        }
    }
    match fs::write(dump_path, &flat) {
        Ok(_) => println!(
            "## Raw matrix dump\n\nWrote {} bytes to {}",
            flat.len(),
            dump_path
        ),
        Err(e) => eprintln!("WARN: failed to write dump file {}: {}", dump_path, e),
    }
    println!();

    // 8. Side-by-side comparison with Jina v5 (externally measured)
    println!("## Comparison to Jina v5 256 codebook (degenerate reference)");
    println!();
    println!("| metric | Jina v5 256 | DeepNSM 4096 |");
    println!("|---|---|---|");
    println!("| matrix size | 256×256 | {}×{} |", k, k);
    println!("| off-diag mean | 0.640 (cos) | {:.2} (u8 dist) |", mean);
    println!("| effective rank | 1.82 | see Python follow-up |");
    println!("| frac > 0.9 (cos) / high u8 | 43.76% | {:.2}% (top 10 bins) |",
             top10 as f64 / n * 100.0);
    println!("| nearest-neighbor similarity | 0.9407 (cos) | see std above |");
    println!();

    // 9. Python follow-up snippet
    println!("## Python follow-up for eigenspectrum / participation ratio");
    println!();
    println!("```python");
    println!("import numpy as np");
    println!("d = np.fromfile('{}', dtype=np.uint8).reshape(4096, 4096).astype(np.float64)", dump_path);
    println!("# Convert distance to similarity: normalize [0,255] → [0,1], invert");
    println!("max_d = d.max()");
    println!("sim = 1.0 - d / max(max_d, 1)");
    println!("np.fill_diagonal(sim, 1.0)  # self-similarity = 1");
    println!("# Eigenspectrum of the symmetric similarity matrix");
    println!("eigvals = np.linalg.eigvalsh(sim)");
    println!("pos = eigvals[eigvals > 1e-6]");
    println!("pr = (pos.sum()**2) / (pos**2).sum()");
    println!("print(f'participation ratio: {{pr:.2f}} out of 4096')");
    println!("top = sorted(eigvals, reverse=True)[:10]");
    println!("print(f'top 10 eigenvalues: {{[f\"{{v:.1f}}\" for v in top]}}')");
    println!("cumvar = np.cumsum(sorted(pos, reverse=True)) / pos.sum()");
    println!("for k in [1, 2, 4, 8, 16, 32, 64, 128]:");
    println!("    if k <= len(cumvar): print(f'  top-{{k:3d}}: {{cumvar[k-1]*100:.1f}}%')");
    println!("```");
    println!();

    // 10. Verdict
    println!("## What a pass/fail looks like");
    println!();
    println!("- **PASS** (semantic layer is real): CV > 0.10 AND");
    println!("  (Python follow-up) participation ratio > 100.");
    println!("  → DeepNSM is the canonical semantic surface, use it for all probes.");
    println!();
    println!("- **MARGINAL** (usable but needs care): 0.03 < CV < 0.10 AND");
    println!("  50 < participation ratio < 100.");
    println!("  → DeepNSM is structured but narrower than ideal.");
    println!();
    println!("- **FAIL** (degenerate like Jina v5): CV < 0.02 AND");
    println!("  participation ratio < 10.");
    println!("  → the CAM-PQ compression is lossy enough that semantic structure");
    println!("    collapses; need to investigate upstream before any codec work.");
}
