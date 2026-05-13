//! Compare randomized-signature cosine vs XOR-Hamming on perturbed paths.
//!
//! This example reproduces the kind of correlation measurement that justifies
//! sigker as a third codec lane. We generate paths, perturb them by varying
//! amounts, and measure how each encoding tracks the perturbation.
//!
//! Run:
//!   cargo run --manifest-path crates/sigker/Cargo.toml \
//!             --example sig_vs_hamming --release

use sigker::randomized::RandomizedSignatureBuilder;
use std::time::Instant;

const PATH_DIM: usize = 4;
const STATE_DIM: usize = 256; // sigker carrier
const HAMMING_BITS: usize = 1024; // representative bitpacked carrier
const PATH_LEN: usize = 32;
const N_PERT_LEVELS: usize = 12;
const N_PAIRS_PER_LEVEL: usize = 50;
const SEED_BUILDER: u64 = 0xCAFEF00D;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_uniform(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn rand_normal(state: &mut u64) -> f64 {
    let u1 = rand_uniform(state).max(1e-300);
    let u2 = rand_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn random_path(state: &mut u64) -> Vec<Vec<f64>> {
    let mut path = Vec::with_capacity(PATH_LEN);
    let mut pt = vec![0.0; PATH_DIM];
    path.push(pt.clone());
    for _ in 1..PATH_LEN {
        for x in pt.iter_mut() {
            *x += rand_normal(state) * 0.5;
        }
        path.push(pt.clone());
    }
    path
}

fn perturb(path: &[Vec<f64>], state: &mut u64, magnitude: f64) -> Vec<Vec<f64>> {
    path.iter()
        .map(|p| {
            p.iter()
                .map(|&v| v + rand_normal(state) * magnitude)
                .collect()
        })
        .collect()
}

/// Stand-in for a bgz17-style bitpacked encoding: hash each path step into
/// a fixed-bit fingerprint via simple feature hashing (FNV-1a). Hamming
/// similarity = 1 − HammingDist / HAMMING_BITS.
fn bitpack_encode(path: &[Vec<f64>]) -> Vec<u8> {
    let n_bytes = HAMMING_BITS / 8;
    let mut bits = vec![0u8; n_bytes];
    for (t, point) in path.iter().enumerate() {
        for (i, &v) in point.iter().enumerate() {
            // Quantize the float into a 6-bit bucket then hash position+value.
            let bucket = ((v * 8.0).round() as i64) as u64;
            let mut h = 0xCBF2_9CE4_8422_2325u64;
            for &byte in &t.to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x100_0000_01B3);
            }
            for &byte in &i.to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x100_0000_01B3);
            }
            for &byte in &bucket.to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x100_0000_01B3);
            }
            // Set 4 bits per (t, i, bucket).
            for _ in 0..4 {
                let bit = (h as usize) % HAMMING_BITS;
                bits[bit / 8] |= 1 << (bit % 8);
                h = h.wrapping_mul(0x100_0000_01B3);
            }
        }
    }
    bits
}

fn hamming_similarity(a: &[u8], b: &[u8]) -> f64 {
    let dist: u32 = a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum();
    1.0 - dist as f64 / HAMMING_BITS as f64
}

fn main() {
    let builder = RandomizedSignatureBuilder::new(PATH_DIM, STATE_DIM, SEED_BUILDER);
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;

    println!("sigker vs bitpacked-Hamming — perturbation tracking");
    println!(
        "PATH_DIM={PATH_DIM}, STATE_DIM={STATE_DIM} (sigker), HAMMING_BITS={HAMMING_BITS}, \
         PATH_LEN={PATH_LEN}"
    );
    println!(
        "{:>8} | {:>16} | {:>16} | {:>10}",
        "pert", "sigker_cos_mean", "hamming_sim_mean", "n_pairs"
    );
    println!("{}", "-".repeat(60));

    let t0 = Instant::now();
    for level in 0..N_PERT_LEVELS {
        let pert = 0.05 * (level + 1) as f64;
        let mut sig_acc = 0.0f64;
        let mut ham_acc = 0.0f64;
        for _ in 0..N_PAIRS_PER_LEVEL {
            let p1 = random_path(&mut rng_state);
            let p2 = perturb(&p1, &mut rng_state, pert);
            // sigker cosine
            let s1 = builder.encode(&p1);
            let s2 = builder.encode(&p2);
            sig_acc += s1.cosine(&s2);
            // bitpacked Hamming similarity
            let b1 = bitpack_encode(&p1);
            let b2 = bitpack_encode(&p2);
            ham_acc += hamming_similarity(&b1, &b2);
        }
        let sig_mean = sig_acc / N_PAIRS_PER_LEVEL as f64;
        let ham_mean = ham_acc / N_PAIRS_PER_LEVEL as f64;
        println!(
            "{:>8.3} | {:>16.4} | {:>16.4} | {:>10}",
            pert, sig_mean, ham_mean, N_PAIRS_PER_LEVEL
        );
    }
    println!();
    println!("Elapsed: {} ms", t0.elapsed().as_millis());
    println!();
    println!("Reading: sigker_cos_mean should decay smoothly with perturbation.");
    println!("hamming_sim_mean tends to plateau or jump because hash buckets either");
    println!("collide (similarity stays high) or do not (similarity drops sharply).");
    println!("Sigker's smoothness is the Index-regime guarantee; Hamming's plateau is");
    println!("the cost of CAM-PQ-style codebook quantization on path-structured data.");
}
