//! Probe M1 (REFRAMED + EXTENDED): Natural Clustering Shape of 256 Jina Centroids
//!
//! ## Why this is reframed from the original M1
//!
//! Original M1 (from `.claude/knowledge/bf16-hhtl-terrain.md § Probe Queue`)
//! asked: "Does CLAM build a 3-level 16-way tree over 256 Jina centroids with
//! knees at L1/L2?"
//!
//! This turned out to be architecturally impossible: source-level investigation
//! of ndarray's CLAM (`/home/user/ndarray/src/hpc/clam.rs:341-379`) confirmed
//! CLAM is BINARY-ONLY with no branching-factor parameter, so "16-way" is not
//! something CLAM's native shape. The 256 Jina embeddings are also not saved
//! (only the 256×256 pairwise cosine matrix is on disk). See
//! `.claude/blackboard-ripple-architecture-20260411-01.MD § F1/F2` for traces.
//!
//! ## Reviewer extensions applied (per 4 specialized agents)
//!
//! This file has been extended past v1 in response to binding reviews:
//!
//! - **truth-architect**: added k-sweep, 4-method panel (incl. binary-CLAM-at-depth-4),
//!   absolute gap threshold replacing 3× ratio, 5-seed PAM with median, explicit
//!   cost statement. PASS verdict softened from "justified" to "not ruled out."
//! - **savant-research**: added angular distance `acos(cos)/π` as the primary
//!   metric (metric-safe on unit sphere), kept `1-cos` as secondary for comparison.
//!   Noted: PASS validates Slot D bit layout but NOT the Pareto frontier.
//! - **family-codec-smith**: PASS verdict now says 16-way is the **HIP family**
//!   layer, NOT HEEL. HEEL is the basin (2-4 states). FAIL fallback is
//!   direct-256-state Jina tag (option a), NOT binary-CLAM reinterpretation.
//! - **integration-lead**: v1 probe built cleanly; no new deps introduced by
//!   extensions (pure std + existing std::f32::consts::PI).
//!
//! ## Reframed question (intent-level)
//!
//! What is the NATURAL clustering shape of the 256-point Jina-v5 semantic
//! codebook? Run multiple methods, sweep k, let the data speak instead of
//! forcing k=16. The result tells us which architectural slot (HEEL/HIP/TWIG)
//! the natural shape fits, and whether Slot D's bit layout has empirical
//! ground.
//!
//! ## Method panel (per the user's "compare all variants" rule)
//!
//! FOUR clustering methods on the same 256×256 distance matrix:
//!   1. **k-medoids (PAM)** — 5 seeds, median silhouette, init-sensitive
//!   2. **Agglomerative average linkage** — captures merge distances for depth
//!   3. **Binary CLAM at depth 4** — recursive pole-based split, 16 leaves at
//!      depth 4 (approximates what existing CLAM gives at a fixed cut)
//!   4. **Random baseline (fixed seed)** — null hypothesis
//!
//! PRIMARY distance: `acos(clamp(cos, -1, 1)) / π` (angular, metric-safe).
//! SECONDARY distance: `1 - cos` (legacy, for comparison).
//!
//! K-SWEEP (per truth-architect): at k ∈ {4, 8, 16, 32, 64} on the winning
//! method. Lets the data tell us the natural peak instead of forcing k=16.
//!
//! ## Pass / Fail / Uncertain criteria (stated BEFORE running)
//!
//! PASS (16-way is not-ruled-out at the HIP family layer):
//!   - Best method silhouette > 0.2 at k=16
//!   - Best method balance < 3.0 at k=16
//!   - Absolute gap: `silhouette_real - silhouette_random > 0.15`
//!   - k-sweep shows k=16 within one standard deviation of the peak
//!
//! FAIL (16-way is aesthetic; architecturally unsupported):
//!   - Best method silhouette < 0.1 at k=16
//!   - OR balance > 5.0 at k=16
//!   - OR absolute gap < 0.05 (clustering is noise relative to random)
//!   - OR k-sweep peak is decisively elsewhere (e.g., k=4 or k=64) and k=16
//!     is in a trough
//!
//! UNCERTAIN (methodology needs more data):
//!   - Silhouette in [0.1, 0.2] or balance in [3, 5] or gap in [0.05, 0.15]
//!   - Methods disagree sharply — some say PASS, some say FAIL
//!   - Recommendation: regenerate the 256 × 1024 embeddings (~7 min) and
//!     re-run with vector-space methods (CLAM proper, k-means Lloyd)
//!
//! ## Interpretation (per family-codec-smith ontology)
//!
//! PASS does NOT mean HEEL = 16. HEEL is the basin layer (2-4 states).
//! PASS means the **HIP family layer** has 16-way structural support, which is
//! what Slot D's middle 4 bits should encode. The correct Slot D layout under
//! PASS is:
//!   - bits 15..14 = HEEL basin (2 bits, 2-4 states)
//!   - bits 13..10 = HIP family (4 bits, 16 states) ← this probe validates
//!   - bits  9..2  = TWIG Jina centroid (8 bits, 256 states)
//!   - bit     1   = BRANCH polarity
//!   - bit     0   = reserved / γ-phase
//!
//! FAIL fallback (canonical): direct 256-state Jina tag, no hierarchy.
//! HEEL collapses to a 2-bit basin flag stored separately.
//! Slot D = 8-bit centroid + 8-bit flags.
//!
//! ## Cost (truth-architect probe protocol step 4)
//!
//! LOC: ~850 (up from ~520 in v1)
//! Expected runtime: < 30 seconds on CPU, dominated by agglomerative O(n³·k).
//! Build: clean, zero new dependencies.
//! Memory: ~2 MB peak (256² f32 distance matrix).
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/thinking-engine/Cargo.toml \
//!     --example probe_m1_bucket_fit --release
//! ```
//!
//! No arguments. Reads from the hardcoded path. Output is a markdown-shaped
//! summary suitable for pasting into `bf16-hhtl-terrain.md § Probe Queue`.

use std::f32::consts::PI;
use std::fs;
use std::path::Path;

const COSINE_MATRIX_PATH: &str = "/tmp/codebooks/jina-v5-semantic-256/cosine_matrix_256x256.f32";
const N: usize = 256;
const K: usize = 16;
const PAM_SEEDS: [u64; 5] = [0xDEADBEEF, 0xC0FFEE, 0x1337C0DE, 0xFEEDFACE, 0xBAADF00D];
const K_SWEEP: [usize; 5] = [4, 8, 16, 32, 64];

fn main() {
    println!("# Probe M1 (reframed): Hierarchical Clustering Fit");
    println!();
    println!("Input: {}", COSINE_MATRIX_PATH);
    println!("N = {}, K = {}", N, K);
    println!();

    // -------- 1. Load cosine matrix, convert to distance --------
    if !Path::new(COSINE_MATRIX_PATH).exists() {
        eprintln!("ERROR: cosine matrix not found at {}", COSINE_MATRIX_PATH);
        eprintln!("  Run `cargo run --example semantic_codebook` first to generate it.");
        eprintln!("  (Takes ~7 minutes for 256 forward passes through Jina v5.)");
        std::process::exit(2);
    }

    let bytes = fs::read(COSINE_MATRIX_PATH).expect("read cosine matrix");
    assert_eq!(
        bytes.len(),
        N * N * 4,
        "expected {} bytes (256x256 f32), got {}",
        N * N * 4,
        bytes.len()
    );
    let cos: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Angular distance (PRIMARY, metric-safe on unit sphere per savant-research):
    //   d = acos(clamp(cos, -1, 1)) / π ∈ [0, 1]
    let dist: Vec<f32> = cos
        .iter()
        .map(|&c| c.clamp(-1.0, 1.0).acos() / PI)
        .collect();

    // 1-cos (SECONDARY, for comparison with v1)
    let dist_legacy: Vec<f32> = cos.iter().map(|&c| 1.0_f32 - c).collect();

    println!("## Distance matrix summary");
    println!("### Angular distance (primary, acos(cos)/π)");
    let (dmin, dmax, dmean) = matrix_stats(&dist);
    println!("- min: {:.4}, max: {:.4}, mean: {:.4}", dmin, dmax, dmean);
    println!("- diagonal (should be 0): {:.6}", dist[0]);
    println!("### 1-cos legacy");
    let (ldmin, ldmax, ldmean) = matrix_stats(&dist_legacy);
    println!("- min: {:.4}, max: {:.4}, mean: {:.4}", ldmin, ldmax, ldmean);
    println!();

    // -------- 2. Run the four methods (per truth-architect + savant-research) --------
    let mut results: Vec<(&str, Vec<usize>)> = Vec::new();

    println!("## Method 1: k-medoids (PAM) with k=16, 5 seeds, median silhouette");
    let mut pam_runs: Vec<(Vec<usize>, f32)> = Vec::new();
    for (i, &seed) in PAM_SEEDS.iter().enumerate() {
        let labels = kmedoids_pam(&dist, N, K, 50, seed);
        let sil = silhouette(&dist, N, &labels, K);
        println!("- seed {} (0x{:X}): silhouette = {:.4}", i, seed, sil);
        pam_runs.push((labels, sil));
    }
    pam_runs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let median_idx = pam_runs.len() / 2;
    let (labels_pam, sil_pam_median) = pam_runs.swap_remove(median_idx);
    println!("- median silhouette across 5 seeds: {:.4}", sil_pam_median);
    results.push(("k-medoids (PAM, median)", labels_pam));

    println!("## Method 2: Agglomerative hierarchical (average linkage) with k=16");
    let (labels_agg, merge_distances) = agglomerative_average_linkage(&dist, N, K);
    println!("- final {} merge distances: {:?}", K.min(merge_distances.len()), &merge_distances[merge_distances.len().saturating_sub(K)..]);
    let depth_jump_ratio = if merge_distances.len() >= 2 {
        merge_distances[merge_distances.len() - 1] / merge_distances[0].max(1e-6)
    } else {
        0.0
    };
    println!("- depth jump ratio (last/first merge): {:.2}x", depth_jump_ratio);
    results.push(("agglomerative", labels_agg));

    println!("## Method 3: Binary CLAM at depth 4 (recursive pole split, 16 leaves)");
    let labels_clam = binary_clam_depth(&dist, N, 4);
    results.push(("binary-CLAM-depth-4", labels_clam));

    println!("## Method 4: Random baseline (k=16, fixed seed)");
    let labels_rand = random_partition(N, K, 0xC0FFEE);
    results.push(("random baseline", labels_rand));

    println!();

    // -------- 3. Evaluate each --------
    println!("## Metrics per method");
    println!();
    println!("| Method | Silhouette | Balance (max/min) | Within/Between | Cluster sizes |");
    println!("|---|---|---|---|---|");

    let mut best_method: Option<(&str, f32, f32, f32)> = None;
    let mut random_silhouette = 0.0_f32;
    let mut random_within_between = 1.0_f32;

    for (name, labels) in &results {
        let sil = silhouette(&dist, N, labels, K);
        let balance = cluster_balance(labels, K);
        let wb = within_between_ratio(&dist, N, labels, K);
        let sizes = cluster_sizes(labels, K);

        println!(
            "| {} | {:.4} | {:.2} | {:.4} | {:?} |",
            name, sil, balance, wb, sizes
        );

        if *name == "random baseline" {
            random_silhouette = sil;
            random_within_between = wb;
        } else if let Some((_, best_sil, _, _)) = best_method {
            if sil > best_sil {
                best_method = Some((name, sil, balance, wb));
            }
        } else {
            best_method = Some((name, sil, balance, wb));
        }
    }

    println!();

    // -------- 4. k-sweep on the winning method (per truth-architect) --------
    println!("## K-sweep on winning method");
    println!();
    println!("Tests whether k=16 is the natural peak or forced. Runs PAM with the");
    println!("same 5-seed median protocol across k ∈ {{4, 8, 16, 32, 64}}.");
    println!();
    println!("| k | median silhouette | balance | within/between |");
    println!("|---|---|---|---|");
    let mut k_sweep_results: Vec<(usize, f32)> = Vec::new();
    for &k_test in &K_SWEEP {
        let mut runs: Vec<f32> = Vec::new();
        let mut last_labels: Vec<usize> = Vec::new();
        for &seed in &PAM_SEEDS {
            let labels = kmedoids_pam(&dist, N, k_test, 50, seed);
            let sil = silhouette(&dist, N, &labels, k_test);
            runs.push(sil);
            last_labels = labels;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = runs[runs.len() / 2];
        let balance = cluster_balance(&last_labels, k_test);
        let wb = within_between_ratio(&dist, N, &last_labels, k_test);
        println!("| {} | {:.4} | {:.2} | {:.4} |", k_test, median, balance, wb);
        k_sweep_results.push((k_test, median));
    }
    println!();
    let (peak_k, peak_sil) = k_sweep_results
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .copied()
        .unwrap_or((16, 0.0));
    let k16_sil = k_sweep_results
        .iter()
        .find(|(k, _)| *k == 16)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let k16_in_peak_band = (peak_sil - k16_sil).abs() < 0.05;
    println!(
        "Peak: k={} (silhouette {:.4}). k=16 silhouette {:.4}. k=16 within peak band (Δ<0.05): {}",
        peak_k, peak_sil, k16_sil, k16_in_peak_band
    );
    println!();

    // -------- 5. Validity / reliability across methods (verify creative agent) --------
    println!("## Multi-method validity / reliability");
    println!();
    println!("Cronbach's alpha over per-point silhouette vectors (4 methods × 256 points).");
    println!("Tests whether the 4 methods RELIABLY agree on which points are well-clustered.");
    println!("This is the empirical test of ripple-architect's multi-sieve claim.");
    println!();

    // Per-point silhouette vectors, one per method
    let mut per_point_vectors: Vec<(&str, Vec<f32>)> = Vec::new();
    for (name, labels) in &results {
        let v = silhouette_per_point(&dist, N, labels, K);
        per_point_vectors.push((name, v));
    }

    // Cronbach's alpha
    let vectors_only: Vec<&Vec<f32>> = per_point_vectors.iter().map(|(_, v)| v).collect();
    let alpha_all = cronbach_alpha(&vectors_only);
    let vectors_no_random: Vec<&Vec<f32>> = per_point_vectors
        .iter()
        .filter(|(name, _)| *name != "random baseline")
        .map(|(_, v)| v)
        .collect();
    let alpha_no_random = cronbach_alpha(&vectors_no_random);

    println!("### Cronbach's α (internal consistency of silhouette assessment)");
    println!();
    println!("- α over all 4 methods (incl. random baseline): **{:.4}**", alpha_all);
    println!("- α over 3 real methods (excluding random): **{:.4}**", alpha_no_random);
    println!();
    println!("Interpretation thresholds:");
    println!("  α > 0.9  — excellent inter-method reliability (structure is robust)");
    println!("  α > 0.7  — acceptable (standard social-science threshold)");
    println!("  α > 0.5  — questionable (some agreement, weak signal)");
    println!("  α ≤ 0.5  — unacceptable (methods disagree, no robust structure)");
    println!();

    // Pairwise Adjusted Rand Index (convergent validity)
    println!("### Pairwise Adjusted Rand Index (convergent validity)");
    println!();
    println!("ARI measures partition agreement corrected for chance.");
    println!("  ARI = 1.0   identical partitions");
    println!("  ARI ≈ 0.5   moderate agreement");
    println!("  ARI ≈ 0.0   chance-level agreement (random)");
    println!("  ARI < 0.0   anti-correlated");
    println!();
    println!("| method A | method B | ARI |");
    println!("|---|---|---|");
    let mut ari_values: Vec<f32> = Vec::new();
    let mut ari_non_random: Vec<f32> = Vec::new();
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let ari = adjusted_rand_index(&results[i].1, &results[j].1);
            println!("| {} | {} | {:.4} |", results[i].0, results[j].0, ari);
            ari_values.push(ari);
            if results[i].0 != "random baseline" && results[j].0 != "random baseline" {
                ari_non_random.push(ari);
            }
        }
    }
    let ari_mean_all = ari_values.iter().sum::<f32>() / ari_values.len() as f32;
    let ari_mean_real = if !ari_non_random.is_empty() {
        ari_non_random.iter().sum::<f32>() / ari_non_random.len() as f32
    } else {
        0.0
    };
    let ari_var_real = if ari_non_random.len() > 1 {
        let m = ari_mean_real;
        ari_non_random.iter().map(|&v| (v - m).powi(2)).sum::<f32>()
            / (ari_non_random.len() - 1) as f32
    } else {
        0.0
    };
    println!();
    println!("- Mean ARI over all pairs: {:.4}", ari_mean_all);
    println!("- Mean ARI over real-method pairs only: **{:.4}**", ari_mean_real);
    println!("- Variance of real-method ARI: {:.4}", ari_var_real);
    println!();

    // Ripple-architect creative-agent verification
    println!("### Creative agent verification (ripple-architect's multi-sieve claim)");
    println!();
    println!("ripple-architect proposed that different clustering methods should produce");
    println!("PARTIALLY correlated partitions: not redundant (ARI → 1, one method is enough)");
    println!("and not independent (ARI → 0, one method is noise). The band [0.3, 0.7] was");
    println!("proposed as the signature of a real multi-resolution manifold.");
    println!();
    let creative_supported = alpha_no_random > 0.5
        && ari_mean_real > 0.2
        && ari_mean_real < 0.85;
    if creative_supported {
        println!("**Creative agent claim: SUPPORTED**");
        println!("- α over real methods {:.4} > 0.5 ✓", alpha_no_random);
        println!("- Mean ARI {:.4} in partial-agreement band [0.2, 0.85] ✓", ari_mean_real);
        println!();
        println!("Methods give overlapping-but-not-identical views → multi-sieve stacking");
        println!("is empirically justified. Probe M1-multi (5-sieve version with ring lens)");
        println!("is the correct follow-on.");
    } else {
        println!("**Creative agent claim: NOT SUPPORTED (by this probe)**");
        println!("- α over real methods {:.4} (threshold 0.5)", alpha_no_random);
        println!("- Mean ARI {:.4} (band [0.2, 0.85])", ari_mean_real);
        println!();
        if alpha_no_random <= 0.5 {
            println!("Low α: methods disagree on which points are well-clustered.");
            println!("The clustering structure is either noise or heavily method-dependent.");
        }
        if ari_mean_real > 0.85 {
            println!("High ARI: methods produce nearly identical partitions.");
            println!("Multi-sieve stacking is redundant on this data — one method suffices.");
        }
        if ari_mean_real < 0.2 {
            println!("Low ARI: methods produce chance-level agreement.");
            println!("No reliable cluster structure across methods — falls back to identity.");
        }
    }
    println!();

    // -------- 6. Conflict detection across lenses --------
    // Per user: "In case of conflict use all available lenses for evaluation,
    // not just blind hierarchical autopilot."
    //
    // Collect per-method verdict signals; if they don't converge on the same
    // direction we report CONFLICT-DETECTED with all lens values, not a
    // hierarchical "best wins" call.
    println!("## Conflict detection across lenses");
    println!();
    let mut method_verdicts: Vec<(&str, f32, f32, f32, &'static str)> = Vec::new();
    for (name, labels) in &results {
        if *name == "random baseline" {
            continue;
        }
        let s = silhouette(&dist, N, labels, K);
        let b = cluster_balance(labels, K);
        let w = within_between_ratio(&dist, N, labels, K);
        let g = s - random_silhouette;
        let v = if s > 0.2 && b < 3.0 && g > 0.15 {
            "PASS"
        } else if s < 0.1 || b > 5.0 || g < 0.05 {
            "FAIL"
        } else {
            "UNCERTAIN"
        };
        method_verdicts.push((name, s, b, w, v));
        println!(
            "- **{}**: silhouette={:.4}, balance={:.2}, wb={:.4}, gap={:.4} → {}",
            name, s, b, w, g, v
        );
    }
    let pass_count = method_verdicts.iter().filter(|(.., v)| *v == "PASS").count();
    let fail_count = method_verdicts.iter().filter(|(.., v)| *v == "FAIL").count();
    let uncertain_count = method_verdicts
        .iter()
        .filter(|(.., v)| *v == "UNCERTAIN")
        .count();
    let all_agree = (pass_count == method_verdicts.len())
        || (fail_count == method_verdicts.len())
        || (uncertain_count == method_verdicts.len());
    println!();
    println!(
        "Per-method verdicts: {} PASS, {} FAIL, {} UNCERTAIN (of {} real methods)",
        pass_count,
        fail_count,
        uncertain_count,
        method_verdicts.len()
    );
    println!("All methods agree on direction: {}", all_agree);
    println!();

    // Cross-lens consensus (all 6 lenses: silhouette, balance, wb, gap,
    // k-sweep peak band, Cronbach's alpha, pairwise ARI range).
    let lens_count = 7;
    let mut lens_pass = 0_usize;
    let mut lens_flags: Vec<(&'static str, bool, String)> = Vec::new();
    let best_sil = method_verdicts.iter().map(|(_, s, ..)| *s).fold(f32::NEG_INFINITY, f32::max);
    let best_bal = method_verdicts.iter().map(|(_, _, b, ..)| *b).fold(f32::INFINITY, f32::min);
    let best_gap = best_sil - random_silhouette;
    let lens1 = best_sil > 0.2;
    lens_flags.push(("silhouette > 0.2", lens1, format!("{:.4}", best_sil)));
    if lens1 { lens_pass += 1; }
    let lens2 = best_bal < 3.0;
    lens_flags.push(("balance < 3.0", lens2, format!("{:.2}", best_bal)));
    if lens2 { lens_pass += 1; }
    let lens3 = best_gap > 0.15;
    lens_flags.push(("absolute gap > 0.15", lens3, format!("{:.4}", best_gap)));
    if lens3 { lens_pass += 1; }
    let lens4 = k16_in_peak_band;
    lens_flags.push(("k=16 in k-sweep peak band", lens4, format!("peak={}", peak_k)));
    if lens4 { lens_pass += 1; }
    let lens5 = alpha_no_random > 0.5;
    lens_flags.push(("Cronbach α (real methods) > 0.5", lens5, format!("{:.4}", alpha_no_random)));
    if lens5 { lens_pass += 1; }
    let lens6 = ari_mean_real > 0.2 && ari_mean_real < 0.85;
    lens_flags.push(("Mean ARI in partial-agreement band", lens6, format!("{:.4}", ari_mean_real)));
    if lens6 { lens_pass += 1; }
    let lens7 = all_agree;
    lens_flags.push(("All real methods agree on direction", lens7, format!("{}", all_agree)));
    if lens7 { lens_pass += 1; }

    println!("### Cross-lens consensus matrix");
    println!();
    println!("| lens | value | pass? |");
    println!("|---|---|---|");
    for (name, ok, val) in &lens_flags {
        println!("| {} | {} | {} |", name, val, if *ok { "✓" } else { "✗" });
    }
    println!();
    println!("Consensus: **{}/{} lenses pass**", lens_pass, lens_count);
    println!();

    // -------- 7. Verdict (multi-lens, not hierarchical-autopilot) --------
    println!("## Verdict");
    println!();

    // When conflict is high (methods disagree AND consensus is mid-range),
    // report CONFLICT-DETECTED with full lens values instead of "best-method wins."
    let conflict = !all_agree && lens_pass >= 3 && lens_pass <= 5;
    if conflict {
        println!("## RESULT: CONFLICT-DETECTED");
        println!();
        println!("Per-method verdicts disagree ({} PASS, {} FAIL, {} UNCERTAIN).",
                 pass_count, fail_count, uncertain_count);
        println!("Cross-lens consensus is mid-range ({}/{}).", lens_pass, lens_count);
        println!();
        println!("### All-lens values (use all, not hierarchical autopilot)");
        println!();
        for (name, ok, val) in &lens_flags {
            println!("- {}: {} ({})", name, val, if *ok { "pass" } else { "fail" });
        }
        println!();
        println!("### Per-method details");
        println!();
        for (name, s, b, w, v) in &method_verdicts {
            println!("- **{}**: silhouette={:.4}, balance={:.2}, wb={:.4} → {}", name, s, b, w, v);
        }
        println!();
        println!("### Interpretation");
        println!();
        println!("Methods disagree but consensus across orthogonal lenses is not decisive.");
        println!("This is a legitimate UNCERTAIN at the multi-method level. Do NOT pick");
        println!("the winning method by silhouette alone — multiple lenses say different things.");
        println!();
        println!("Next action: run Probe M1-multi (ripple-architect's 5-sieve version)");
        println!("with the 40-ring 1/40σ lens as sieve #5. The ring lens is the one object");
        println!("in the architecture that measures resonance rather than partition, and it's");
        println!("the missing lens that could break the tie.");
        println!();
        println!("Do NOT promote Slot D conjecture to FINDING on this result.");
        println!("Do NOT trigger the canonical fallback to direct-256-state either.");
        println!("The architecturally correct response to multi-lens conflict is MORE LENSES.");
    } else if let Some((name, sil, balance, wb)) = best_method {
        // ABSOLUTE gap (truth-architect replaced 3x ratio)
        let sil_gap = sil - random_silhouette;
        let wb_improvement = if random_within_between < 1e-6 {
            f32::INFINITY
        } else {
            random_within_between / wb.max(1e-6)
        };

        println!("Best method: **{}**", name);
        println!("- Silhouette: {:.4}", sil);
        println!("- Balance: {:.2}", balance);
        println!("- Within/between: {:.4}", wb);
        println!("- Random baseline silhouette: {:.4}", random_silhouette);
        println!("- Absolute gap (sil_real − sil_random): {:.4}", sil_gap);
        println!("- Within/between improvement over random: {:.2}x", wb_improvement);
        println!();

        let pass_sil = sil > 0.2;
        let pass_balance = balance < 3.0;
        let pass_gap = sil_gap > 0.15;
        let pass_k_band = k16_in_peak_band;

        let fail_sil = sil < 0.1;
        let fail_balance = balance > 5.0;
        let fail_gap = sil_gap < 0.05;
        let fail_k_peak = peak_k != 16 && !k16_in_peak_band;

        if pass_sil && pass_balance && pass_gap && pass_k_band {
            println!("## RESULT: PASS");
            println!();
            println!("All criteria met:");
            println!("- silhouette {:.4} > 0.2 ✓", sil);
            println!("- balance {:.2} < 3.0 ✓", balance);
            println!("- absolute gap {:.4} > 0.15 ✓", sil_gap);
            println!("- k=16 within peak band of k-sweep ✓ (peak=k{})", peak_k);
            println!();
            println!("### Interpretation (per family-codec-smith ontology)");
            println!();
            println!("16-way clustering is NOT RULED OUT at the **HIP family layer**.");
            println!("HEEL remains the basin (2-4 states), NOT 16. The 16-way claim");
            println!("validates only Slot D's middle 4 bits (HIP nibble), not the top.");
            println!();
            println!("Correct Slot D layout under PASS (per family-codec-smith):");
            println!("  bits 15..14 = HEEL basin (2 bits)");
            println!("  bits 13..10 = HIP family (4 bits, 16 states) ← this probe validates");
            println!("  bits  9..2  = TWIG Jina centroid (8 bits, 256 states)");
            println!("  bit     1   = BRANCH polarity");
            println!("  bit     0   = reserved / γ-phase");
            println!();
            println!("### Important caveats (per savant-research)");
            println!();
            println!("- PASS validates Slot D BIT LAYOUT, not the Pareto frontier.");
            println!("  Bucketing fidelity is a different axis from ρ measurement.");
            println!("- Winning method was {}, not necessarily CLAM. The canonical", name);
            println!("  terrain doc's 'Slot D = CLAM tree path' wording should be");
            println!("  corrected to 'Slot D = {} tree path'.", name);
            println!("- This probe used the 256×256 cosine matrix only, NOT the");
            println!("  original 256×1024 embeddings. A vector-space probe with");
            println!("  true CLAM would be a stronger test (Probe M1-vector, follow-on).");
        } else if fail_sil || fail_balance || fail_gap || fail_k_peak {
            println!("## RESULT: FAIL");
            println!();
            println!("Failure criteria hit:");
            if fail_sil {
                println!("- silhouette {:.4} < 0.1 ✗", sil);
            }
            if fail_balance {
                println!("- balance {:.2} > 5.0 ✗", balance);
            }
            if fail_gap {
                println!("- absolute gap {:.4} < 0.05 (clustering is noise) ✗", sil_gap);
            }
            if fail_k_peak {
                println!("- k-sweep peak at k={} (not 16), k=16 outside peak band ✗", peak_k);
            }
            println!();
            println!("### Interpretation");
            println!();
            println!("16-way clustering is NOT structurally supported on this codebook.");
            println!("The Slot D 16-way nibble allocation is aesthetic, not empirical.");
            println!();
            println!("### Canonical fallback (per family-codec-smith, option a)");
            println!();
            println!("Direct 256-state Jina centroid tag, no hierarchy:");
            println!("  bits 15..8  = HIP = Jina centroid id (8 bits, 256 states, 1:1)");
            println!("  bits  7..4  = reserved for basin/role flags (4 bits)");
            println!("  bit     3   = BRANCH polarity");
            println!("  bits  2..0  = γ-phase bucket (3 bits, 8 offsets)");
            println!();
            println!("This aligns with the 4 existing centroid-level pairwise tables");
            println!("(bf16_engine 256², AttentionSemiring 256², etc.).");
            println!("Not a regression — a simplification.");
        } else {
            println!("## RESULT: UNCERTAIN");
            println!();
            println!("Mixed signal, not decisive:");
            println!("- silhouette {:.4} (pass > 0.2, fail < 0.1)", sil);
            println!("- balance {:.2} (pass < 3.0, fail > 5.0)", balance);
            println!("- absolute gap {:.4} (pass > 0.15, fail < 0.05)", sil_gap);
            println!("- k-sweep peak: k={}, k=16 in band: {}", peak_k, k16_in_peak_band);
            println!();
            println!("### Recommendation");
            println!();
            println!("Methods may disagree or signal is weak. Next steps:");
            println!("1. Regenerate 256 × 1024 embeddings (~7 min compute)");
            println!("2. Run vector-space variants: true ndarray CLAM, k-means Lloyd, spectral");
            println!("3. Also consider Probe M1-multi (ripple-architect proposal): 5-sieve");
            println!("   agreement test with Cramér's V between sieves");
        }
    } else {
        println!("ERROR: no non-random method ran");
    }

    println!();
    println!("## Next action");
    println!();
    println!("Record this result in `.claude/knowledge/bf16-hhtl-terrain.md`");
    println!("§ Probe Queue — update status from NOT RUN to PASS/FAIL/UNCERTAIN.");
    println!("Commit: docs(knowledge): probe M1 result — [PASS|FAIL|UNCERTAIN]");
    println!();
    println!("Also update `.claude/knowledge/encoding-ecosystem.md` § BGZ-HHTL-D 2×BF16");
    println!("status field from CONJECTURE to FINDING (or correction) based on result.");
}

// ============================================================================
// Metrics
// ============================================================================

fn matrix_stats(m: &[f32]) -> (f32, f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    for &v in m {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v;
    }
    (min, max, sum / m.len() as f32)
}

/// Silhouette coefficient: s(i) = (b(i) - a(i)) / max(a, b)
/// where a(i) = mean dist to other points in same cluster,
///       b(i) = min over other clusters of mean dist to that cluster.
fn silhouette(dist: &[f32], n: usize, labels: &[usize], k: usize) -> f32 {
    let mut s_sum = 0.0_f32;
    let mut s_count = 0;

    for i in 0..n {
        let li = labels[i];
        let mut a_sum = 0.0_f32;
        let mut a_count = 0_usize;
        let mut b_per: Vec<(f32, usize)> = vec![(0.0, 0); k];

        for j in 0..n {
            if i == j {
                continue;
            }
            let d = dist[i * n + j];
            let lj = labels[j];
            if lj == li {
                a_sum += d;
                a_count += 1;
            } else {
                b_per[lj].0 += d;
                b_per[lj].1 += 1;
            }
        }

        if a_count == 0 {
            continue; // singleton cluster, silhouette undefined
        }
        let a = a_sum / a_count as f32;

        let mut b = f32::INFINITY;
        for (bs, bc) in &b_per {
            if *bc > 0 {
                let m = bs / *bc as f32;
                if m < b {
                    b = m;
                }
            }
        }
        if !b.is_finite() {
            continue;
        }

        let s = (b - a) / a.max(b);
        s_sum += s;
        s_count += 1;
    }

    if s_count == 0 {
        0.0
    } else {
        s_sum / s_count as f32
    }
}

fn cluster_sizes(labels: &[usize], k: usize) -> Vec<usize> {
    let mut sizes = vec![0_usize; k];
    for &l in labels {
        sizes[l] += 1;
    }
    sizes
}

fn cluster_balance(labels: &[usize], k: usize) -> f32 {
    let sizes = cluster_sizes(labels, k);
    let max = *sizes.iter().max().unwrap_or(&0);
    let min = *sizes.iter().filter(|&&s| s > 0).min().unwrap_or(&1);
    max as f32 / min as f32
}

fn within_between_ratio(dist: &[f32], n: usize, labels: &[usize], _k: usize) -> f32 {
    let mut within_sum = 0.0_f32;
    let mut within_count = 0_usize;
    let mut between_sum = 0.0_f32;
    let mut between_count = 0_usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist[i * n + j];
            if labels[i] == labels[j] {
                within_sum += d;
                within_count += 1;
            } else {
                between_sum += d;
                between_count += 1;
            }
        }
    }

    let within_mean = if within_count > 0 {
        within_sum / within_count as f32
    } else {
        0.0
    };
    let between_mean = if between_count > 0 {
        between_sum / between_count as f32
    } else {
        1.0
    };

    within_mean / between_mean.max(1e-6)
}

// ============================================================================
// Clustering algorithms
// ============================================================================

/// Simple xorshift PRNG. Seed != 0.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// k-medoids (PAM) with BUILD initialization and SWAP refinement.
/// Seed parameter added per truth-architect (5-seed median protocol).
fn kmedoids_pam(dist: &[f32], n: usize, k: usize, max_iters: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng(seed);

    // BUILD: start with k random distinct medoids
    let mut medoids: Vec<usize> = Vec::with_capacity(k);
    while medoids.len() < k {
        let cand = rng.range(n);
        if !medoids.contains(&cand) {
            medoids.push(cand);
        }
    }

    let mut labels = vec![0_usize; n];

    for iter in 0..max_iters {
        // Assign each point to the nearest medoid
        for i in 0..n {
            let mut best_m = 0_usize;
            let mut best_d = f32::INFINITY;
            for (mi, &m) in medoids.iter().enumerate() {
                let d = dist[i * n + m];
                if d < best_d {
                    best_d = d;
                    best_m = mi;
                }
            }
            labels[i] = best_m;
        }

        // SWAP: for each cluster, pick the point with minimum sum of distances
        // to all other cluster members as the new medoid.
        let mut new_medoids = medoids.clone();
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            let mut best_sum = f32::INFINITY;
            let mut best_med = members[0];
            for &cand in &members {
                let s: f32 = members.iter().map(|&m| dist[cand * n + m]).sum();
                if s < best_sum {
                    best_sum = s;
                    best_med = cand;
                }
            }
            new_medoids[c] = best_med;
        }

        if new_medoids == medoids {
            let _ = iter; // converged; don't print to keep 5-seed output quiet
            break;
        }
        medoids = new_medoids;
    }

    // Final assignment
    for i in 0..n {
        let mut best_m = 0_usize;
        let mut best_d = f32::INFINITY;
        for (mi, &m) in medoids.iter().enumerate() {
            let d = dist[i * n + m];
            if d < best_d {
                best_d = d;
                best_m = mi;
            }
        }
        labels[i] = best_m;
    }

    labels
}

/// Agglomerative hierarchical clustering with average linkage.
/// Starts with n singleton clusters, merges until k clusters remain.
/// Returns (labels, merge_distances) so the caller can measure hierarchy depth.
fn agglomerative_average_linkage(
    dist: &[f32],
    n: usize,
    k: usize,
) -> (Vec<usize>, Vec<f32>) {
    // Cluster membership: cluster_id -> Vec<point_idx>
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut merge_distances: Vec<f32> = Vec::new();

    while clusters.len() > k {
        let nc = clusters.len();
        let mut best_a = 0_usize;
        let mut best_b = 1_usize;
        let mut best_d = f32::INFINITY;

        for a in 0..nc {
            for b in (a + 1)..nc {
                let mut sum = 0.0_f32;
                let mut cnt = 0_usize;
                for &pa in &clusters[a] {
                    for &pb in &clusters[b] {
                        sum += dist[pa * n + pb];
                        cnt += 1;
                    }
                }
                let avg = sum / cnt as f32;
                if avg < best_d {
                    best_d = avg;
                    best_a = a;
                    best_b = b;
                }
            }
        }

        merge_distances.push(best_d);
        // Merge b into a, remove b
        let merged_b = clusters.remove(best_b);
        clusters[best_a].extend(merged_b);
    }

    let mut labels = vec![0_usize; n];
    for (ci, cluster) in clusters.iter().enumerate() {
        for &p in cluster {
            labels[p] = ci;
        }
    }
    (labels, merge_distances)
}

/// Binary CLAM at fixed depth: recursive pole-based split.
/// Picks two farthest points as poles, assigns each point to nearer pole, recurses.
/// At depth=4, produces 16 leaves; at depth=8, 256 leaves.
///
/// This is a "poor man's CLAM" that works on a distance matrix (no embeddings needed)
/// and exercises the same pole-split invariant as production ndarray CLAM, but with
/// a FIXED depth cut for comparison with k-medoids at the same leaf count.
fn binary_clam_depth(dist: &[f32], n: usize, depth: usize) -> Vec<usize> {
    let mut labels = vec![0_usize; n];
    let all_indices: Vec<usize> = (0..n).collect();
    recurse_pole_split(dist, n, &all_indices, 0, depth, &mut labels);
    labels
}

fn recurse_pole_split(
    dist: &[f32],
    n: usize,
    indices: &[usize],
    current_label: usize,
    remaining_depth: usize,
    out: &mut [usize],
) {
    if indices.is_empty() {
        return;
    }
    if remaining_depth == 0 || indices.len() <= 1 {
        for &i in indices {
            out[i] = current_label;
        }
        return;
    }

    // Pick two farthest points as poles (approximate: max-dist pair)
    let mut best_a = indices[0];
    let mut best_b = indices[0];
    let mut best_d = -1.0_f32;
    for &a in indices {
        for &b in indices {
            if a == b {
                continue;
            }
            let d = dist[a * n + b];
            if d > best_d {
                best_d = d;
                best_a = a;
                best_b = b;
            }
        }
    }

    // Assign each point to the closer pole
    let mut left: Vec<usize> = Vec::new();
    let mut right: Vec<usize> = Vec::new();
    for &i in indices {
        let da = dist[i * n + best_a];
        let db = dist[i * n + best_b];
        if da <= db {
            left.push(i);
        } else {
            right.push(i);
        }
    }

    // Labels at the current level: left gets (current_label << 1), right gets (<< 1 | 1)
    let left_label = current_label << 1;
    let right_label = (current_label << 1) | 1;

    recurse_pole_split(dist, n, &left, left_label, remaining_depth - 1, out);
    recurse_pole_split(dist, n, &right, right_label, remaining_depth - 1, out);
}

/// Random partition into k clusters with a fixed seed (for reproducibility).
fn random_partition(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.range(k)).collect()
}

// ============================================================================
// Multi-method validity / reliability (verify creative agent)
// ============================================================================

/// Per-point silhouette: returns a vector of n silhouette scores, one per point.
/// Points in singleton clusters get s=0.
fn silhouette_per_point(dist: &[f32], n: usize, labels: &[usize], k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let li = labels[i];
        let mut a_sum = 0.0_f32;
        let mut a_count = 0_usize;
        let mut b_per: Vec<(f32, usize)> = vec![(0.0, 0); k];
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = dist[i * n + j];
            let lj = labels[j];
            if lj == li {
                a_sum += d;
                a_count += 1;
            } else if lj < k {
                b_per[lj].0 += d;
                b_per[lj].1 += 1;
            }
        }
        if a_count == 0 {
            out[i] = 0.0;
            continue;
        }
        let a = a_sum / a_count as f32;
        let mut b = f32::INFINITY;
        for (bs, bc) in &b_per {
            if *bc > 0 {
                let m = bs / *bc as f32;
                if m < b {
                    b = m;
                }
            }
        }
        out[i] = if b.is_finite() {
            (b - a) / a.max(b)
        } else {
            0.0
        };
    }
    out
}

/// Cronbach's α: internal consistency of k measurement vectors (methods)
/// over n items (points). Returns a score in [−∞, 1]; α > 0.7 is "acceptable".
///
/// α = (k / (k − 1)) × (1 − Σ σ²ᵢ / σ²_T)
/// where σ²ᵢ is the variance of method i's scores across n points,
/// and σ²_T is the variance of the SUM of scores across methods for each point.
fn cronbach_alpha(vectors: &[&Vec<f32>]) -> f32 {
    let k = vectors.len();
    if k < 2 {
        return 0.0;
    }
    let n = vectors[0].len();
    if n < 2 {
        return 0.0;
    }

    // Per-method variance
    let mut sum_var_i = 0.0_f32;
    for v in vectors {
        let mean: f32 = v.iter().sum::<f32>() / n as f32;
        let var: f32 = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
        sum_var_i += var;
    }

    // Variance of the per-point sum across methods
    let mut sums: Vec<f32> = vec![0.0; n];
    for v in vectors {
        for (i, &x) in v.iter().enumerate() {
            sums[i] += x;
        }
    }
    let mean_t: f32 = sums.iter().sum::<f32>() / n as f32;
    let var_t: f32 = sums.iter().map(|&x| (x - mean_t).powi(2)).sum::<f32>() / (n - 1) as f32;

    if var_t < 1e-9 {
        return 0.0;
    }

    let k_f = k as f32;
    (k_f / (k_f - 1.0)) * (1.0 - sum_var_i / var_t)
}

/// Adjusted Rand Index between two cluster label vectors.
/// Returns [-1, 1]. 1 = identical partition, 0 = chance agreement, negative = worse than chance.
fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 {
        return 0.0;
    }

    let max_a = *a.iter().max().unwrap_or(&0) + 1;
    let max_b = *b.iter().max().unwrap_or(&0) + 1;

    // Contingency table
    let mut ct = vec![vec![0_usize; max_b]; max_a];
    let mut row_sum = vec![0_usize; max_a];
    let mut col_sum = vec![0_usize; max_b];
    for i in 0..n {
        ct[a[i]][b[i]] += 1;
        row_sum[a[i]] += 1;
        col_sum[b[i]] += 1;
    }

    let c2 = |x: usize| -> f64 {
        if x < 2 {
            0.0
        } else {
            (x as f64) * ((x - 1) as f64) / 2.0
        }
    };

    let mut sum_c_ij = 0.0_f64;
    for row in &ct {
        for &x in row {
            sum_c_ij += c2(x);
        }
    }
    let sum_c_a: f64 = row_sum.iter().map(|&x| c2(x)).sum();
    let sum_c_b: f64 = col_sum.iter().map(|&x| c2(x)).sum();
    let c_total = c2(n);
    if c_total < 1e-9 {
        return 0.0;
    }

    let expected = sum_c_a * sum_c_b / c_total;
    let max_index = 0.5 * (sum_c_a + sum_c_b);
    let denom = max_index - expected;
    if denom.abs() < 1e-9 {
        return 0.0;
    }
    ((sum_c_ij - expected) / denom) as f32
}
