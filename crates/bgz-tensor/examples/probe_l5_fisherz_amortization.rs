//! PROBE — L5 γ-fold holographic containers + certified L2 FisherZTable:
//! measure + validate amortization (build-once cost vs per-read cost) on
//! REAL model bytes (Rule 23 — no synthetic vectors).
//!
//! Two lanes, both live in this crate:
//!
//!   L5  — `euler_fold::{euler_gamma_fold, euler_gamma_unfold}`. Folds N
//!         similar 17-dim vectors into ONE holographic `StackedN` container
//!         (centroid + γ-rotated residual sum). Documented expectation:
//!         recovery Pearson ~0.96 at SNR ≈ 9.5 (SPD=32, d=17, N=6).
//!
//!   L2  — `fisher_z::FisherZTable`, the certified palette256 cosine-
//!         replacement (see `examples/nnue_palette_cosine.rs` for the API
//!         usage precedent this probe follows). k×k i8 pairwise-cosine
//!         table via Fisher-z (arctanh) encoding + per-family 3σ gamma.
//!
//! Real bytes only: rows come from a bgz7 shard of real model weights
//! (bge-m3-f16). Deterministic sampling: SplitMix64, seed 0x9E3779B97F4A7C15.
//! No floats are stored anywhere except inside the sanctioned bf16 StackedN
//! container and the FamilyGamma 8-byte record — everything else here is
//! printed only, never persisted.
//!
//! ```text
//! cargo run --manifest-path crates/bgz-tensor/Cargo.toml --release \
//!   --example probe_l5_fisherz_amortization
//! ```

use bgz_tensor::euler_fold::{euler_gamma_fold, euler_gamma_unfold};
use bgz_tensor::fisher_z::FisherZTable;
use bgz_tensor::stacked_n::StackedN;
use ndarray::hpc::gguf_indexer::CompressedTensor;
use std::io::Read;
use std::time::Instant;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const DIM: usize = 17;
const SPD: usize = 32;
const N_FOLD_MEMBERS: usize = 6;
const K_REPS: usize = 256;
const N_COSINE_PAIRS: usize = 2000;
const LOOKUP_ITERS: usize = 1_000_000;
const FLOAT_COSINE_ITERS: usize = 10_000;

const DEFAULT_SHARD: &str =
    "/tmp/claude-0/-home-user/bcd29cfc-5bae-5b23-b86b-0de9582a87da/scratchpad/bge-m3-f16.bgz7";

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt()).max(1e-300)
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite").then(a.cmp(&b)));
    let mut r = vec![0f64; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && (v[idx[j + 1]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

/// Lenient bgz7 read: the shard declares N tensors but may hold fewer then
/// exact EOF (a known published-asset truncation). `read_bgz7_file` hard-
/// fails on this; parse the complete prefix instead of erroring out.
fn load_rows(shard: &str) -> Vec<[f32; DIM]> {
    let mut reader = std::io::BufReader::new(std::fs::File::open(shard).expect("open shard"));
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).expect("magic");
    assert_eq!(&magic, b"BGZ7", "bad magic");
    let mut u32_buf = [0u8; 4];
    reader.read_exact(&mut u32_buf).expect("n_tensors");
    let declared = u32::from_le_bytes(u32_buf) as usize;
    let mut tensors: Vec<CompressedTensor> = Vec::with_capacity(declared);
    for _ in 0..declared {
        match CompressedTensor::read_from(&mut reader) {
            Ok(t) => tensors.push(t),
            Err(_) => break,
        }
    }
    println!("declared tensors: {declared}  parsed: {}", tensors.len());

    let mut rows: Vec<[f32; DIM]> = Vec::new();
    for t in &tensors {
        for r in &t.rows {
            let mut v = [0f32; DIM];
            let mut nz = false;
            for (i, d) in r.dims.iter().enumerate() {
                v[i] = f32::from(*d) / 256.0;
                nz |= *d != 0;
            }
            if nz {
                rows.push(v);
            }
        }
    }
    rows
}

fn main() {
    let shard = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_SHARD.to_string());
    let rows = load_rows(&shard);
    println!("shard: {shard}\nusable rows: {}\n", rows.len());
    assert!(
        rows.len() >= K_REPS + N_FOLD_MEMBERS,
        "not enough real rows sampled ({}) for K_REPS={K_REPS} + N_FOLD_MEMBERS={N_FOLD_MEMBERS}",
        rows.len()
    );

    let mut rng = SplitMix64(SEED);
    let mut taken = vec![false; rows.len()];
    let draw = |rng: &mut SplitMix64, taken: &mut Vec<bool>| loop {
        let i = rng.below(taken.len());
        if !taken[i] {
            taken[i] = true;
            return i;
        }
    };

    // ═══════════════════════════════════════════════════════════════════
    // PART A — L5 γ-fold holographic container amortization
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ PART A — L5 γ-fold (euler_gamma_fold / euler_gamma_unfold) ═══\n");

    let fold_idx: Vec<usize> = (0..N_FOLD_MEMBERS).map(|_| draw(&mut rng, &mut taken)).collect();
    let members: Vec<Vec<f32>> = fold_idx.iter().map(|&i| rows[i].to_vec()).collect();

    let t0 = Instant::now();
    let family = euler_gamma_fold(&members, SPD);
    let fold_build_ns = t0.elapsed().as_nanos() as f64;

    // Correct-index recovery: unfold member j, compare to its own StackedN
    // hydration (the same reference gate_test() in euler_fold.rs uses).
    let mut correct_pearsons = Vec::with_capacity(N_FOLD_MEMBERS);
    let mut unfold_ns_samples = Vec::with_capacity(N_FOLD_MEMBERS);
    for (j, member) in members.iter().enumerate() {
        let t1 = Instant::now();
        let recovered = euler_gamma_unfold(&family, j);
        unfold_ns_samples.push(t1.elapsed().as_nanos() as f64);

        let orig_f32 = StackedN::from_f32(member, SPD).hydrate_f32();
        let r = pearson(
            &orig_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            &recovered.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        );
        correct_pearsons.push(r);
    }
    let mean_correct = correct_pearsons.iter().sum::<f64>() / N_FOLD_MEMBERS as f64;
    let min_correct = correct_pearsons.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_unfold_ns = unfold_ns_samples.iter().sum::<f64>() / N_FOLD_MEMBERS as f64;

    // Falsifier: unfold with a WRONG member_index — ask for (i+1)%N and
    // compare recovery quality against member i's ORIGINAL. If the
    // container isn't actually addressing members, wrong-index recovery
    // would be statistically indistinguishable from correct-index recovery.
    let mut wrong_pearsons = Vec::with_capacity(N_FOLD_MEMBERS);
    for (i, member) in members.iter().enumerate() {
        let wrong_j = (i + 1) % N_FOLD_MEMBERS;
        let recovered_wrong = euler_gamma_unfold(&family, wrong_j);
        let orig_f32 = StackedN::from_f32(member, SPD).hydrate_f32();
        let r = pearson(
            &orig_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            &recovered_wrong.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        );
        wrong_pearsons.push(r);
    }
    let mean_wrong = wrong_pearsons.iter().sum::<f64>() / N_FOLD_MEMBERS as f64;

    let margin = mean_correct - mean_wrong;
    let falsifier_fires = margin > 0.05;

    let raw_member_bytes = N_FOLD_MEMBERS * DIM * 4;
    let folded_bytes = family.byte_size();
    let ratio_vs_raw = raw_member_bytes as f64 / folded_bytes as f64;
    let ratio_family_reported = family.compression_ratio();

    println!("N_FOLD_MEMBERS = {N_FOLD_MEMBERS}, SPD = {SPD}, dim = {DIM}");
    println!("fold indices (into `rows`): {fold_idx:?}\n");
    println!("build_once (euler_gamma_fold): {fold_build_ns:.0} ns");
    println!(
        "correct-index recovery: mean ρ = {mean_correct:.4}, min ρ = {min_correct:.4}  (per-member: {:?})",
        correct_pearsons.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
    );
    println!(
        "wrong-index  recovery: mean ρ = {mean_wrong:.4}  (per-member: {:?})",
        wrong_pearsons.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
    );
    println!(
        "falsifier margin (correct − wrong) = {margin:.4} → {}",
        if falsifier_fires { "FIRES (container addresses members)" } else { "DOES-NOT-FIRE (no addressing signal)" }
    );
    println!(
        "bytes: raw {raw_member_bytes} B vs folded {folded_bytes} B → ratio_vs_raw {ratio_vs_raw:.2}×, FoldedFamily::compression_ratio() = {ratio_family_reported:.2}×"
    );
    println!("per-read unfold cost: mean {mean_unfold_ns:.0} ns/member\n");

    // ═══════════════════════════════════════════════════════════════════
    // PART B — certified L2 FisherZTable amortization
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ PART B — certified L2 FisherZTable (fisher_z::FisherZTable) ═══\n");

    let rep_idx: Vec<usize> = (0..K_REPS).map(|_| draw(&mut rng, &mut taken)).collect();
    let reps: Vec<Vec<f32>> = rep_idx.iter().map(|&i| rows[i].to_vec()).collect();

    let t2 = Instant::now();
    let table = FisherZTable::build(&reps, K_REPS);
    let table_build_ns = t2.elapsed().as_nanos() as f64;
    let table_bytes = table.byte_size();

    println!("K_REPS = {K_REPS}");
    println!("build_once (FisherZTable::build): {table_build_ns:.0} ns");
    println!(
        "byte_size() = {table_bytes} B ({:.1} KB) [expect k×k + 8 = {} B]\n",
        table_bytes as f64 / 1024.0,
        K_REPS * K_REPS + 8
    );

    // Ground truth: true pairwise cosine for a deterministic sample of
    // N_COSINE_PAIRS (a, b) pairs drawn from the k representatives.
    let mut pair_a: Vec<u8> = Vec::with_capacity(N_COSINE_PAIRS);
    let mut pair_b: Vec<u8> = Vec::with_capacity(N_COSINE_PAIRS);
    let mut true_cos: Vec<f64> = Vec::with_capacity(N_COSINE_PAIRS);
    for _ in 0..N_COSINE_PAIRS {
        let a = rng.below(K_REPS);
        let mut b = rng.below(K_REPS);
        while b == a {
            b = rng.below(K_REPS);
        }
        pair_a.push(a as u8);
        pair_b.push(b as u8);
        true_cos.push(cosine_f32(&reps[a], &reps[b]) as f64);
    }

    let decoded_cos: Vec<f64> = pair_a
        .iter()
        .zip(&pair_b)
        .map(|(&a, &b)| table.lookup_f32(a, b) as f64)
        .collect();

    let r_pearson = pearson(&true_cos, &decoded_cos);
    let r_spearman = spearman(&true_cos, &decoded_cos);
    let cert_pass = r_spearman >= 0.9990;

    println!(
        "certification over {N_COSINE_PAIRS} deterministic pairs: Pearson r = {r_pearson:.4}, Spearman ρ = {r_spearman:.4}"
    );
    println!(
        "gate (Spearman ≥ 0.9990, u8-lane): {}\n",
        if cert_pass { "PASS" } else { "FAIL (real measured number, not fudged)" }
    );

    // Per-read cost: tight loop of LOOKUP_ITERS lookup_i8 calls.
    let mut sink_i8: i64 = 0;
    let t3 = Instant::now();
    for k in 0..LOOKUP_ITERS {
        let a = (k % K_REPS) as u8;
        let b = ((k / K_REPS + 1) % K_REPS) as u8;
        sink_i8 += i64::from(table.lookup_i8(a, b));
    }
    let lookup_total_ns = t3.elapsed().as_nanos() as f64;
    let lookup_ns_per_read = lookup_total_ns / LOOKUP_ITERS as f64;
    std::hint::black_box(sink_i8);

    // Alternative cost: tight loop of FLOAT_COSINE_ITERS true float cosine
    // computations over the same representative rows.
    let mut sink_f: f64 = 0.0;
    let t4 = Instant::now();
    for k in 0..FLOAT_COSINE_ITERS {
        let a = k % K_REPS;
        let b = (k / K_REPS + 1) % K_REPS;
        sink_f += f64::from(cosine_f32(&reps[a], &reps[b]));
    }
    let cosine_total_ns = t4.elapsed().as_nanos() as f64;
    let cosine_ns_per_op = cosine_total_ns / FLOAT_COSINE_ITERS as f64;
    std::hint::black_box(sink_f);

    let break_even_reads = if cosine_ns_per_op > lookup_ns_per_read {
        table_build_ns / (cosine_ns_per_op - lookup_ns_per_read)
    } else {
        f64::INFINITY
    };

    println!("per-read cost: lookup_i8 = {lookup_ns_per_read:.2} ns/lookup ({LOOKUP_ITERS} iters)");
    println!(
        "float alternative cost: true cosine = {cosine_ns_per_op:.2} ns/cosine ({FLOAT_COSINE_ITERS} iters)"
    );
    println!(
        "break_even_reads = build_ns / (cosine_ns − lookup_ns) = {table_build_ns:.0} / ({cosine_ns_per_op:.2} − {lookup_ns_per_read:.2}) = {break_even_reads:.1}"
    );
    println!(
        "compare to the certification pass alone ({N_COSINE_PAIRS} reads): {}",
        if break_even_reads.is_finite() && break_even_reads <= N_COSINE_PAIRS as f64 {
            "amortized WITHIN the certification run itself"
        } else if break_even_reads.is_finite() {
            "amortizes only AFTER the certification run (more reads needed)"
        } else {
            "never amortizes at these measured costs (lookup not cheaper than float cosine here)"
        }
    );

    // ═══════════════════════════════════════════════════════════════════
    // SUMMARY TABLE
    // ═══════════════════════════════════════════════════════════════════
    println!("\n═══ SUMMARY ═══\n");
    println!(
        "{:<8} | {:>12} | {:>13} | {:>12} | {:>13} | {:>17} | {:>9} | falsifier",
        "lane", "build_once", "product_bytes", "per_read_ns", "float_alt_ns", "break_even_reads", "fidelity"
    );
    println!("{}", "-".repeat(112));
    println!(
        "{:<8} | {:>9.0} ns | {:>10} B  | {:>9.0} ns | {:>13} | {:>17} | ρ={:>6.4} | {}",
        "L5",
        fold_build_ns,
        folded_bytes,
        mean_unfold_ns,
        "N/A",
        "N/A",
        mean_correct,
        if falsifier_fires { "FIRES" } else { "DOES-NOT-FIRE" }
    );
    println!(
        "{:<8} | {:>9.0} ns | {:>10} B  | {:>9.2} ns | {:>10.2} ns | {:>17.1} | ρ={:>6.4} | {}",
        "L2",
        table_build_ns,
        table_bytes,
        lookup_ns_per_read,
        cosine_ns_per_op,
        break_even_reads,
        r_spearman,
        if cert_pass { "CERT-PASS" } else { "CERT-FAIL" }
    );
}
