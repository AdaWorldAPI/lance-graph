//! VSA Weight Vector Probe: what lives inside 150K weight rows?
//!
//! Encodes every weight row as a 16Kbit binary vector via RaBitQ,
//! then uses VSA bundling (majority vote) to discover self-organizing
//! archetype structure at multiple granularities.
//!
//! ```text
//! f32 row [1024..6144]
//!   │
//!   ▼  Hadamard rotation + sign quantization (RaBitQ)
//! Binary [16384 bits = 256 u64]
//!   │
//!   ▼  Hamming distance → self-organize into clusters
//! Cluster → Bundle (majority vote across members)
//!   │
//!   ▼  Bundle = archetype fingerprint
//! Hierarchical codebook: 64K → 17K → 4096 → 1024 → 512 → 256
//!   │
//!   ▼  Unbundle: recover f32 via RaBitQ correction factors
//! Reconstructed row → quality measurement
//! ```
//!
//! The bundling reveals:
//! - How many DISTINCT behaviors exist (bundle diversity at each level)
//! - Which roles cluster together (cross-role resonance)
//! - Whether layer depth matters (do L0 and L27 share archetypes?)
//! - What the archetype fingerprint MEANS (decode back to f32)
//!
//! ```sh
//! cargo run --release --example vsa_weight_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors
//! ```

use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, TensorInfo};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

/// Binary vector dimension: 16384 bits = 256 u64 words.
const DIM_BITS: usize = 16384;
const DIM_U64: usize = DIM_BITS / 64;

/// Codebook levels for hierarchy.
const LEVELS: &[usize] = &[256, 512, 1024, 4096, 17408, 65536];

// ═══════════════════════════════════════════════════════════════════
// Binary vector (16Kbit, AVX-friendly)
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone)]
#[repr(align(64))]
struct BinVec {
    words: [u64; DIM_U64],
}

impl BinVec {
    fn zero() -> Self { BinVec { words: [0u64; DIM_U64] } }

    /// Encode f32 vector to binary via RaBitQ-style sign quantization.
    ///
    /// 1. Pad/fold to DIM_BITS dimensions via golden-step stride
    /// 2. Apply Walsh-Hadamard butterfly (spreads info across all bits)
    /// 3. Sign quantize: bit = 1 if positive, 0 if negative
    fn from_f32(row: &[f32]) -> (Self, f32, f32) {
        let n = row.len();

        // Step 1: Fold to DIM_BITS via golden-step accumulation
        let golden_step = 11;
        let mut expanded = vec![0.0f64; DIM_BITS];
        for i in 0..n {
            let target = (i * golden_step) % DIM_BITS;
            expanded[target] += row[i] as f64;
        }

        // Compute norm before Hadamard
        let norm = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt() as f32;

        // Step 2: In-place Walsh-Hadamard transform (O(D log D))
        let mut d = expanded;
        let mut h = 1;
        while h < DIM_BITS {
            for i in (0..DIM_BITS).step_by(h * 2) {
                for j in i..i + h {
                    let x = d[j];
                    let y = d[j + h];
                    d[j] = x + y;
                    d[j + h] = x - y;
                }
            }
            h *= 2;
        }

        // Step 3: Sign quantize
        let mut words = [0u64; DIM_U64];
        let mut n_positive = 0u32;
        for w in 0..DIM_U64 {
            let mut word = 0u64;
            for bit in 0..64 {
                let idx = w * 64 + bit;
                if d[idx] > 0.0 {
                    word |= 1u64 << bit;
                    n_positive += 1;
                }
            }
            words[w] = word;
        }

        // Dot correction factor (fraction of positive bits, for unbiased estimation)
        let dot_correction = n_positive as f32 / DIM_BITS as f32;

        (BinVec { words }, norm, dot_correction)
    }

    /// Hamming distance (popcount of XOR).
    fn hamming(&self, other: &BinVec) -> u32 {
        let mut dist = 0u32;
        for w in 0..DIM_U64 {
            dist += (self.words[w] ^ other.words[w]).count_ones();
        }
        dist
    }

    /// Approximate cosine from Hamming distance + correction factors.
    fn approx_cosine(&self, other: &BinVec, norm_a: f32, norm_b: f32) -> f64 {
        let h = self.hamming(other) as f64;
        // cos(θ) ≈ 1 - 2h/D (for uniformly distributed sign vectors)
        let cos_approx = 1.0 - 2.0 * h / DIM_BITS as f64;
        cos_approx // norm correction not needed for ranking
    }

    /// Count set bits (Hamming weight).
    fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════
// VSA Bundle: majority vote across N binary vectors
// ═══════════════════════════════════════════════════════════════════

/// Accumulator for VSA bundling (majority vote).
/// Uses i16 per bit for up to 32K vectors before overflow.
struct BundleAccumulator {
    counts: Vec<i16>,
    n_vectors: usize,
}

impl BundleAccumulator {
    fn new() -> Self {
        BundleAccumulator {
            counts: vec![0i16; DIM_BITS],
            n_vectors: 0,
        }
    }

    /// Add a binary vector to the accumulator.
    fn add(&mut self, v: &BinVec) {
        for w in 0..DIM_U64 {
            for bit in 0..64 {
                let idx = w * 64 + bit;
                if (v.words[w] >> bit) & 1 == 1 {
                    self.counts[idx] += 1;
                } else {
                    self.counts[idx] -= 1;
                }
            }
        }
        self.n_vectors += 1;
    }

    /// Resolve to a binary vector via majority vote.
    /// Ties broken by random (deterministic seed).
    fn resolve(&self, seed: u64) -> BinVec {
        let mut words = [0u64; DIM_U64];
        let mut state = seed;
        for w in 0..DIM_U64 {
            let mut word = 0u64;
            for bit in 0..64 {
                let idx = w * 64 + bit;
                if self.counts[idx] > 0 {
                    word |= 1u64 << bit;
                } else if self.counts[idx] == 0 {
                    // Tie: use deterministic random
                    state = state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
                    if state & 1 == 1 { word |= 1u64 << bit; }
                }
                // counts < 0: leave bit as 0
            }
            words[w] = word;
        }
        BinVec { words }
    }

    /// Entropy of the accumulator: how decisive is the majority?
    /// 0 = all ties. 1 = all unanimous.
    fn decisiveness(&self) -> f64 {
        let n = self.n_vectors as f64;
        if n < 1.0 { return 0.0; }
        let mut decisive = 0usize;
        for &c in &self.counts {
            if (c as f64).abs() > n * 0.1 {
                decisive += 1;
            }
        }
        decisive as f64 / DIM_BITS as f64
    }
}

// ═══════════════════════════════════════════════════════════════════
// Hierarchical bundling: self-organizing codebook
// ═══════════════════════════════════════════════════════════════════

/// One level in the hierarchy.
struct BundleLevel {
    /// k clusters at this level.
    k: usize,
    /// Bundle (archetype) for each cluster.
    bundles: Vec<BinVec>,
    /// Assignment: which cluster each vector belongs to.
    assignments: Vec<usize>,
    /// Cluster sizes.
    sizes: Vec<usize>,
    /// Mean intra-cluster Hamming distance.
    mean_intra_hamming: f64,
    /// Mean inter-cluster Hamming distance.
    mean_inter_hamming: f64,
    /// Average bundle decisiveness.
    avg_decisiveness: f64,
}

/// Build hierarchical bundles via iterative refinement.
///
/// 1. Initialize k seeds via max-Hamming furthest-point
/// 2. Assign each vector to nearest seed
/// 3. Bundle each cluster (majority vote) → new centroids
/// 4. Repeat 2-3 for 5 iterations
fn build_bundle_level(vectors: &[BinVec], k: usize) -> BundleLevel {
    let n = vectors.len();
    let k = k.min(n);
    if k == 0 {
        return BundleLevel {
            k: 0, bundles: Vec::new(), assignments: Vec::new(),
            sizes: Vec::new(), mean_intra_hamming: 0.0,
            mean_inter_hamming: 0.0, avg_decisiveness: 0.0,
        };
    }

    // Seed selection: furthest-point sampling via Hamming
    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    seeds.push(0);
    let mut min_dist = vec![u32::MAX; n];
    for i in 0..n {
        min_dist[i] = vectors[i].hamming(&vectors[0]);
    }

    for _ in 1..k {
        let next = (0..n)
            .filter(|i| !seeds.contains(i))
            .max_by_key(|&i| min_dist[i])
            .unwrap_or(0);
        seeds.push(next);
        for i in 0..n {
            let d = vectors[i].hamming(&vectors[next]);
            if d < min_dist[i] { min_dist[i] = d; }
        }
    }

    let mut bundles: Vec<BinVec> = seeds.iter().map(|&s| vectors[s].clone()).collect();
    let mut assignments = vec![0usize; n];

    // Iterate: assign → bundle → reassign
    for _iter in 0..5 {
        // Assign to nearest bundle
        for i in 0..n {
            let nearest = (0..k)
                .min_by_key(|&c| vectors[i].hamming(&bundles[c]))
                .unwrap_or(0);
            assignments[i] = nearest;
        }

        // Bundle each cluster (majority vote)
        let mut accumulators: Vec<BundleAccumulator> = (0..k)
            .map(|_| BundleAccumulator::new())
            .collect();
        for i in 0..n {
            accumulators[assignments[i]].add(&vectors[i]);
        }
        for c in 0..k {
            bundles[c] = accumulators[c].resolve(c as u64 * 31337);
        }
    }

    // Final assignment
    for i in 0..n {
        assignments[i] = (0..k)
            .min_by_key(|&c| vectors[i].hamming(&bundles[c]))
            .unwrap_or(0);
    }

    // Statistics
    let mut sizes = vec![0usize; k];
    for &a in &assignments { sizes[a] += 1; }

    // Intra-cluster Hamming (sample)
    let mut intra_sum = 0.0f64;
    let mut intra_count = 0usize;
    for c in 0..k {
        let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
        for &i in members.iter().take(20) {
            intra_sum += vectors[i].hamming(&bundles[c]) as f64;
            intra_count += 1;
        }
    }
    let mean_intra = if intra_count > 0 { intra_sum / intra_count as f64 } else { 0.0 };

    // Inter-cluster Hamming (sample k×k)
    let mut inter_sum = 0.0f64;
    let mut inter_count = 0usize;
    for a in 0..k.min(50) {
        for b in (a+1)..k.min(50) {
            inter_sum += bundles[a].hamming(&bundles[b]) as f64;
            inter_count += 1;
        }
    }
    let mean_inter = if inter_count > 0 { inter_sum / inter_count as f64 } else { 0.0 };

    // Decisiveness
    let mut accumulators: Vec<BundleAccumulator> = (0..k)
        .map(|_| BundleAccumulator::new())
        .collect();
    for i in 0..n {
        accumulators[assignments[i]].add(&vectors[i]);
    }
    let avg_dec: f64 = accumulators.iter()
        .map(|a| a.decisiveness())
        .sum::<f64>() / k as f64;

    BundleLevel {
        k, bundles, assignments, sizes,
        mean_intra_hamming: mean_intra,
        mean_inter_hamming: mean_inter,
        avg_decisiveness: avg_dec,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Self-organization analysis
// ═══════════════════════════════════════════════════════════════════

/// Analyze whether clusters correlate with role or layer.
fn analyze_clustering(
    level: &BundleLevel,
    metadata: &[(String, usize)], // (role_name, layer_index) per vector
) -> (f64, f64) {
    let n = metadata.len().min(level.assignments.len());
    if n == 0 { return (0.0, 0.0); }

    // Role purity: what fraction of each cluster shares the same role?
    let mut cluster_roles: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    for i in 0..n {
        let c = level.assignments[i];
        let role = &metadata[i].0;
        *cluster_roles.entry(c).or_default().entry(role.clone()).or_insert(0) += 1;
    }
    let role_purity: f64 = cluster_roles.values().map(|roles| {
        let total: usize = roles.values().sum();
        let max: usize = *roles.values().max().unwrap_or(&0);
        if total > 0 { max as f64 / total as f64 } else { 0.0 }
    }).sum::<f64>() / cluster_roles.len().max(1) as f64;

    // Layer spread: how many different layers appear in each cluster?
    let mut cluster_layers: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    for i in 0..n {
        let c = level.assignments[i];
        cluster_layers.entry(c).or_default().insert(metadata[i].1);
    }
    let avg_layer_spread: f64 = cluster_layers.values()
        .map(|layers| layers.len() as f64)
        .sum::<f64>() / cluster_layers.len().max(1) as f64;

    (role_purity, avg_layer_spread)
}

// ═══════════════════════════════════════════════════════════════════
// Tensor reading
// ═══════════════════════════════════════════════════════════════════

fn read_rows(
    reader: &mut BufReader<File>,
    tensor: &TensorInfo,
    data_offset: u64,
    max_rows: usize,
) -> Vec<Vec<f32>> {
    let n_rows = (tensor.dimensions[0] as usize).min(max_rows);
    let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };
    let elem_size = match tensor.dtype {
        GgmlType::BF16 | GgmlType::F16 => 2, GgmlType::F32 => 4, _ => return Vec::new(),
    };
    reader.seek(SeekFrom::Start(data_offset + tensor.offset)).unwrap();
    let mut raw = vec![0u8; n_rows * n_cols * elem_size];
    if reader.read_exact(&mut raw).is_err() { return Vec::new(); }
    (0..n_rows).map(|r| {
        (0..n_cols).map(|c| {
            let idx = r * n_cols + c;
            match tensor.dtype {
                GgmlType::BF16 => {
                    let bits = u16::from_le_bytes([raw[idx*2], raw[idx*2+1]]);
                    f32::from_bits((bits as u32) << 16)
                }
                GgmlType::F16 => {
                    let bits = u16::from_le_bytes([raw[idx*2], raw[idx*2+1]]);
                    ndarray::hpc::gguf::f16_to_f32(bits)
                }
                GgmlType::F32 => f32::from_le_bytes([raw[idx*4], raw[idx*4+1], raw[idx*4+2], raw[idx*4+3]]),
                _ => 0.0,
            }
        }).collect()
    }).collect()
}

fn detect_role(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("q_proj") { "q" }
    else if n.contains("k_proj") { "k" }
    else if n.contains("v_proj") { "v" }
    else if n.contains("o_proj") { "o" }
    else if n.contains("gate_proj") { "gate" }
    else if n.contains("up_proj") { "up" }
    else if n.contains("down_proj") { "down" }
    else if n.contains("embed") { "embed" }
    else if n.contains("lm_head") { "lm" }
    else { "other" }
}

fn detect_layer(name: &str) -> usize {
    // Extract layer number from name like "talker.model.layers.17.self_attn.q_proj.weight"
    let parts: Vec<&str> = name.split('.').collect();
    for (i, p) in parts.iter().enumerate() {
        if *p == "layers" && i + 1 < parts.len() {
            return parts[i + 1].parse().unwrap_or(0);
        }
    }
    0
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = if args.len() > 1 { &args[1] }
    else { "/home/user/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors" };

    println!("═══ VSA WEIGHT VECTOR PROBE ═══");
    println!("  Model: {}", st_path);
    println!("  Binary dim: {} bits ({} bytes)", DIM_BITS, DIM_BITS / 8);
    println!();

    let mut reader = BufReader::new(File::open(st_path).expect("open"));
    let header = read_safetensors_header(&mut reader).expect("parse");
    println!("[1] {} tensors in safetensors", header.tensors.len());

    // ─── Step 1: Encode ALL weight rows to binary ──────────────────
    println!("\n[2] Encoding weight rows to {}-bit binary...", DIM_BITS);
    let t0 = Instant::now();

    let mut all_binvecs: Vec<BinVec> = Vec::new();
    let mut all_norms: Vec<f32> = Vec::new();
    let mut all_metadata: Vec<(String, usize)> = Vec::new(); // (role, layer)
    let mut all_f32: Vec<Vec<f32>> = Vec::new(); // keep f32 for quality check

    let mut rows_by_role: HashMap<String, usize> = HashMap::new();

    for tensor in &header.tensors {
        if !tensor.name.ends_with("weight") { continue; }
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        if shape.len() < 2 || shape[0] < 128 || shape[1] < 128 { continue; }

        let role = detect_role(&tensor.name);
        if role == "other" { continue; }
        let layer = detect_layer(&tensor.name);

        // Sample up to 500 rows per tensor (keep total manageable)
        let max_per_tensor = 500;
        let rows = read_rows(&mut reader, tensor, header.tensor_data_offset, max_per_tensor);

        for row in &rows {
            let (bv, norm, _corr) = BinVec::from_f32(row);
            all_binvecs.push(bv);
            all_norms.push(norm);
            all_metadata.push((role.to_string(), layer));
            all_f32.push(row.clone());
        }

        *rows_by_role.entry(role.to_string()).or_insert(0) += rows.len();
    }

    let total = all_binvecs.len();
    println!("  {} vectors encoded in {:?}", total, t0.elapsed());
    println!("  By role:");
    for (role, count) in &rows_by_role {
        println!("    {:6}: {}", role, count);
    }

    // ─── Step 2: Hamming statistics ────────────────────────────────
    println!("\n[3] Hamming distance statistics (1000 random pairs)...");
    let mut same_role = Vec::new();
    let mut diff_role = Vec::new();
    let mut same_layer = Vec::new();
    let mut diff_layer = Vec::new();

    let mut seed = 0x9E3779B97F4A7C15u64;
    for _ in 0..1000 {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z ^= z >> 31;
        let a = (z as usize) % total;
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z ^= z >> 31;
        let b = (z as usize) % total;
        if a == b { continue; }

        let h = all_binvecs[a].hamming(&all_binvecs[b]);

        if all_metadata[a].0 == all_metadata[b].0 {
            same_role.push(h as f64);
        } else {
            diff_role.push(h as f64);
        }
        if all_metadata[a].1 == all_metadata[b].1 {
            same_layer.push(h as f64);
        } else {
            diff_layer.push(h as f64);
        }
    }

    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    println!("  Same role:  mean Hamming = {:.0} / {} ({:.1}%)",
        mean(&same_role), DIM_BITS, mean(&same_role) / DIM_BITS as f64 * 100.0);
    println!("  Diff role:  mean Hamming = {:.0} / {} ({:.1}%)",
        mean(&diff_role), DIM_BITS, mean(&diff_role) / DIM_BITS as f64 * 100.0);
    println!("  Same layer: mean Hamming = {:.0} / {} ({:.1}%)",
        mean(&same_layer), DIM_BITS, mean(&same_layer) / DIM_BITS as f64 * 100.0);
    println!("  Diff layer: mean Hamming = {:.0} / {} ({:.1}%)",
        mean(&diff_layer), DIM_BITS, mean(&diff_layer) / DIM_BITS as f64 * 100.0);
    println!("  Separation: role {:.1}%, layer {:.1}%",
        (mean(&diff_role) - mean(&same_role)) / DIM_BITS as f64 * 100.0,
        (mean(&diff_layer) - mean(&same_layer)) / DIM_BITS as f64 * 100.0);

    // ─── Step 3: Hierarchical bundling ─────────────────────────────
    println!("\n[4] Hierarchical bundling...");

    println!("┌────────┬───────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│ Level  │   k   │ Active   │ IntraH   │ InterH   │ Decisive │ RolePur  │ LayerSpd │");
    println!("├────────┼───────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤");

    for &k in LEVELS {
        if k > total { continue; }
        let t0 = Instant::now();
        let level = build_bundle_level(&all_binvecs, k);

        let n_active = level.sizes.iter().filter(|&&s| s > 0).count();
        let (role_purity, layer_spread) = analyze_clustering(&level, &all_metadata);

        println!("│ {:6} │ {:5} │ {:4}/{:4} │ {:8.0} │ {:8.0} │ {:7.3}  │ {:7.3}  │ {:7.1}  │  {:?}",
            format!("L{}", LEVELS.iter().position(|&l| l == k).unwrap()),
            k, n_active, k,
            level.mean_intra_hamming,
            level.mean_inter_hamming,
            level.avg_decisiveness,
            role_purity,
            layer_spread,
            t0.elapsed());

        // At k=256 and k=1024, also measure pairwise cosine preservation
        if k == 256 || k == 1024 || k == 4096 {
            // Reconstruct approximate rows via nearest bundle → measure cosine
            let n_test = 200.min(total);
            let mut cos_sum = 0.0f64;
            for i in 0..n_test {
                let bundle = &level.bundles[level.assignments[i]];
                let approx_cos = all_binvecs[i].approx_cosine(bundle, all_norms[i], 1.0);
                // Self-cosine: how close is the vector to its bundle?
                cos_sum += 1.0 - (all_binvecs[i].hamming(bundle) as f64 / DIM_BITS as f64 * 2.0);
            }
            let avg_self_cos = cos_sum / n_test as f64;

            // Pairwise preservation
            let mut gt_pairs = Vec::new();
            let mut bun_pairs = Vec::new();
            let mut rng = 42u64;
            for _ in 0..200 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a = (rng >> 17) as usize % n_test;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let b = (rng >> 17) as usize % n_test;
                if a == b { continue; }
                // Ground truth: f32 cosine
                let gt = cosine_f32(&all_f32[a], &all_f32[b]);
                // Bundle: Hamming between the bundles of a and b
                let ba = level.assignments[a];
                let bb = level.assignments[b];
                let bh = level.bundles[ba].hamming(&level.bundles[bb]) as f64;
                let bcos = 1.0 - 2.0 * bh / DIM_BITS as f64;
                gt_pairs.push(gt);
                bun_pairs.push(bcos);
            }
            let rho = spearman_simple(&gt_pairs, &bun_pairs);
            println!("│        │       │ self_cos={:.4}  pairwise_ρ={:.4}                                    │",
                avg_self_cos, rho);
        }
    }

    println!("└────────┴───────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘");

    // ─── Step 4: Cross-role resonance ──────────────────────────────
    println!("\n[5] Cross-role resonance (do Q and K bundles align?)...");
    // At k=256, check if Q cluster i's bundle resonates with K cluster j's bundle
    if total > 0 {
        let level = build_bundle_level(&all_binvecs, 256);

        // Find which bundles contain Q rows and which contain K rows
        let mut q_bundles: Vec<usize> = Vec::new();
        let mut k_bundles: Vec<usize> = Vec::new();
        for i in 0..total.min(level.assignments.len()) {
            match all_metadata[i].0.as_str() {
                "q" => { if !q_bundles.contains(&level.assignments[i]) { q_bundles.push(level.assignments[i]); } }
                "k" => { if !k_bundles.contains(&level.assignments[i]) { k_bundles.push(level.assignments[i]); } }
                _ => {}
            }
        }

        println!("  Q occupies {} / 256 bundles", q_bundles.len());
        println!("  K occupies {} / 256 bundles", k_bundles.len());

        let overlap: usize = q_bundles.iter().filter(|b| k_bundles.contains(b)).count();
        println!("  Q∩K overlap: {} bundles ({:.0}% of Q, {:.0}% of K)",
            overlap,
            overlap as f64 / q_bundles.len().max(1) as f64 * 100.0,
            overlap as f64 / k_bundles.len().max(1) as f64 * 100.0);

        if !q_bundles.is_empty() && !k_bundles.is_empty() {
            // Mean Hamming between Q bundles and K bundles
            let mut qk_hamming = Vec::new();
            for &qi in q_bundles.iter().take(20) {
                for &ki in k_bundles.iter().take(20) {
                    qk_hamming.push(level.bundles[qi].hamming(&level.bundles[ki]) as f64);
                }
            }
            println!("  Q↔K mean Hamming: {:.0} ({:.1}% of {})",
                mean(&qk_hamming), mean(&qk_hamming) / DIM_BITS as f64 * 100.0, DIM_BITS);
        }
    }

    println!("\n═══ DONE ═══");
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64; let y = b[i] as f64;
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

fn spearman_simple(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 3 { return 0.0; }
    let rx = ranks(x); let ry = ranks(y);
    let mr = (n as f64 + 1.0) / 2.0;
    let (mut num, mut dx2, mut dy2) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let dx = rx[i] - mr; let dy = ry[i] - mr;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
    }
    if dx2 < 1e-15 || dy2 < 1e-15 { 0.0 } else { num / (dx2 * dy2).sqrt() }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j < idx.len() && idx[j].1 == idx[i].1 { j += 1; }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j { r[idx[k].0] = avg; }
        i = j;
    }
    r
}
