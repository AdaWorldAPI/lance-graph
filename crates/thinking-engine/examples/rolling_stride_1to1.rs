//! Rolling stride experiment on 1:1 F16 distance tables.
//! Measures Spearman rank correlation at each stride to find the topology
//! preservation knee: at what stride does subsampling destroy ranking?
//!
//! Tables: /tmp/codebooks/bge-m3-roles-f16/<role>/distance_table_NxN.u8
//! These are exact cosine topology (Pearson = 1.000 by construction).
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example rolling_stride_1to1

use rayon::prelude::*;

const BASE_DIR: &str = "/tmp/codebooks/bge-m3-roles-f16";
const STRIDES: &[usize] = &[1, 2, 4, 8, 16, 32];
const N_PAIRS: usize = 10_000;

/// Role name, expected table dimension.
const ROLES: &[(&str, usize)] = &[
    ("attn_q",      1024),
    ("attn_k",      1024),
    ("attn_v",      1024),
    ("attn_output", 1024),
    ("ffn_up",      1024),
    ("ffn_down",    4096),
];

fn main() {
    println!("=== Rolling Stride on 1:1 F16 Distance Tables ===");
    println!("    Spearman knee detection — BGE-M3 roles");
    println!("    {} random pairs per stride\n", N_PAIRS);

    // ── Per-role results ────────────────────────────────────────────
    let mut all_results: Vec<(&str, Vec<StrideResult>)> = Vec::new();

    for &(role, expected_n) in ROLES {
        let table_path = format!(
            "{}/{}/distance_table_{}x{}.u8",
            BASE_DIR, role, expected_n, expected_n
        );

        let table = match std::fs::read(&table_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP {} — cannot read {}: {}", role, table_path, e);
                continue;
            }
        };

        let n = (table.len() as f64).sqrt() as usize;
        if n * n != table.len() {
            eprintln!("SKIP {} — table size {} is not a perfect square", role, table.len());
            continue;
        }

        println!("════════════════════════════════════════════════════════════");
        println!("Role: {}  ({}×{} = {:.1} MB)",
            role, n, n, table.len() as f64 / 1_000_000.0);
        println!("════════════════════════════════════════════════════════════");
        println!();
        println!("  Stride │ New Size │ Table Bytes │ Spearman ρ │    Time");
        println!("  ───────┼──────────┼─────────────┼────────────┼─────────");

        let results: Vec<StrideResult> = STRIDES.iter()
            .filter_map(|&stride| {
                let sub_n = n / stride;
                if sub_n < 4 { return None; }
                Some(run_stride(&table, n, stride))
            })
            .collect();

        for r in &results {
            let bytes_display = if r.table_bytes > 1_000_000 {
                format!("{:.1} MB", r.table_bytes as f64 / 1_000_000.0)
            } else if r.table_bytes > 1024 {
                format!("{:.1} KB", r.table_bytes as f64 / 1024.0)
            } else {
                format!("{} B", r.table_bytes)
            };
            println!("  {:>6} │ {:>8} │ {:>11} │ {:>10.6} │ {:>6.1}ms",
                r.stride, format!("{}×{}", r.sub_n, r.sub_n),
                bytes_display, r.spearman, r.time_ms);
        }
        println!();

        // Find knee: first stride where Spearman drops below 0.99
        if let Some(knee) = results.iter().find(|r| r.spearman < 0.99) {
            println!("  → Knee at stride {} (ρ = {:.4})", knee.stride, knee.spearman);
        } else {
            println!("  → No knee found — ρ ≥ 0.99 at all strides");
        }

        // Compression ratio at stride 32 (or max)
        if let Some(last) = results.last() {
            let ratio = (n * n) as f64 / last.table_bytes as f64;
            println!("  → Max compression: {:.0}× at stride {} (ρ = {:.4})",
                ratio, last.stride, last.spearman);
        }
        println!();

        all_results.push((role, results));
    }

    // ── Summary comparison across roles ──────────────────────────────
    println!("════════════════════════════════════════════════════════════");
    println!("Cross-Role Comparison: Spearman ρ at each stride");
    println!("════════════════════════════════════════════════════════════");
    println!();

    // Header
    print!("  {:>12}", "Stride →");
    for &s in STRIDES { print!(" {:>8}", s); }
    println!();
    print!("  {:>12}", "");
    for _ in STRIDES { print!(" {:>8}", "────────"); }
    println!();

    // Rows
    for (role, results) in &all_results {
        print!("  {:>12}", role);
        for &s in STRIDES {
            if let Some(r) = results.iter().find(|r| r.stride == s) {
                print!(" {:>8.4}", r.spearman);
            } else {
                print!(" {:>8}", "—");
            }
        }
        println!();
    }
    println!();

    // Most/least compressible
    let mut at_stride_8: Vec<(&str, f64)> = all_results.iter()
        .filter_map(|(role, results)| {
            results.iter().find(|r| r.stride == 8).map(|r| (*role, r.spearman))
        })
        .collect();
    at_stride_8.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if !at_stride_8.is_empty() {
        println!("  At stride 8:");
        println!("    Most  compressible: {} (ρ = {:.4})", at_stride_8[0].0, at_stride_8[0].1);
        if at_stride_8.len() > 1 {
            let last = at_stride_8.last().unwrap();
            println!("    Least compressible: {} (ρ = {:.4})", last.0, last.1);
        }
    }
}

struct StrideResult {
    stride: usize,
    sub_n: usize,
    table_bytes: usize,
    spearman: f64,
    time_ms: f64,
}

fn run_stride(table: &[u8], n: usize, stride: usize) -> StrideResult {
    let start = std::time::Instant::now();
    let sub_n = n / stride;
    let table_bytes = sub_n * sub_n;

    // Build subsampled table: take every stride-th row and column
    let sub_table: Vec<u8> = (0..sub_n)
        .flat_map(|si| {
            let orig_row = si * stride;
            (0..sub_n).map(move |sj| {
                let orig_col = sj * stride;
                table[orig_row * n + orig_col]
            })
        })
        .collect();

    // Generate deterministic random pairs (simple LCG for reproducibility)
    let max_idx = n;
    let pairs = deterministic_pairs(max_idx, N_PAIRS);

    // Collect (full_dist, sub_dist) for each pair — parallel
    let dists: Vec<(f64, f64)> = pairs.par_iter()
        .map(|&(i, j)| {
            let full_dist = table[i * n + j] as f64;

            // Map to nearest subsampled index
            let sub_i = (i / stride).min(sub_n - 1);
            let sub_j = (j / stride).min(sub_n - 1);
            let sub_dist = sub_table[sub_i * sub_n + sub_j] as f64;

            (full_dist, sub_dist)
        })
        .collect();

    let full_vals: Vec<f64> = dists.iter().map(|d| d.0).collect();
    let sub_vals: Vec<f64> = dists.iter().map(|d| d.1).collect();

    let spearman = spearman_corr(&full_vals, &sub_vals);
    let elapsed = start.elapsed();

    StrideResult {
        stride,
        sub_n,
        table_bytes,
        spearman,
        time_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

/// Deterministic pair generation (LCG-based) for reproducibility.
fn deterministic_pairs(n: usize, count: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(count);
    // LCG: x_{n+1} = (a * x_n + c) mod m
    let mut state: u64 = 0xDEAD_BEEF_CAFE_1337;
    let a: u64 = 6364136223846793005;
    let c: u64 = 1442695040888963407;

    while pairs.len() < count {
        state = state.wrapping_mul(a).wrapping_add(c);
        let i = (state >> 33) as usize % n;
        state = state.wrapping_mul(a).wrapping_add(c);
        let j = (state >> 33) as usize % n;
        if i != j {
            pairs.push((i, j));
        }
    }
    pairs
}

fn spearman_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let rx = ranks(&x[..n]);
    let ry = ranks(&y[..n]);
    pearson_corr(&rx, &ry)
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let d = (vx * vy).sqrt();
    if d < 1e-12 { 0.0 } else { cov / d }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (idx[j].1 - idx[i].1).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            r[idx[k].0] = avg;
        }
        i = j;
    }
    r
}
