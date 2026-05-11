//! Depth-scaling bench: how the three sigker representations scale.
//!
//!   - Truncated kernel:    O(d^(2N)) per pair, materializes the signature
//!   - Goursat-PDE kernel:  O(T₁·T₂) per pair, NEVER materializes the signature
//!   - Log-signature:       O(d^(2N)) compute (still has Magnus expansion),
//!                          but storage is dim L_N(d) — 7-13× smaller
//!
//! Run:
//!   cargo run --manifest-path crates/sigker/Cargo.toml \
//!             --example depth_scaling --release

use sigker::kernel::{signature_kernel, signature_kernel_pde};
use sigker::log_signature::{log_signature_truncated, witt_dimension};
use std::time::Instant;

const PATH_DIM: usize = 4;
const PATH_LEN: usize = 32;
const N_PAIRS_PER_DEPTH: usize = 5;

fn make_path(seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    let mut path = Vec::with_capacity(PATH_LEN);
    let mut pt = vec![0.0; PATH_DIM];
    path.push(pt.clone());
    for _ in 1..PATH_LEN {
        for x in pt.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
            *x += r * 0.3;
        }
        path.push(pt.clone());
    }
    path
}

fn main() {
    println!("Depth scaling bench — d={PATH_DIM}, T={PATH_LEN}");
    println!();
    println!(
        "{:>5} | {:>10} | {:>10} | {:>14} | {:>14} | {:>14}",
        "depth", "full_dim", "log_dim", "trunc_kernel", "pde_kernel", "log_sig_compute"
    );
    println!("{}", "-".repeat(80));

    let paths: Vec<Vec<Vec<f64>>> = (0..N_PAIRS_PER_DEPTH * 2)
        .map(|i| make_path(0xC0FFEE + i as u64))
        .collect();

    for depth in [2usize, 3, 4, 5, 6, 7, 8].iter().copied() {
        let full_dim = if PATH_DIM == 1 {
            depth + 1
        } else {
            (PATH_DIM.pow((depth + 1) as u32) - 1) / (PATH_DIM - 1)
        };
        let log_dim = witt_dimension(PATH_DIM, depth);

        let t0 = Instant::now();
        for i in 0..N_PAIRS_PER_DEPTH {
            let _ = signature_kernel(&paths[2 * i], &paths[2 * i + 1], depth);
        }
        let trunc_us = t0.elapsed().as_micros() as f64 / N_PAIRS_PER_DEPTH as f64;

        let t0 = Instant::now();
        for i in 0..N_PAIRS_PER_DEPTH {
            let _ = signature_kernel_pde(&paths[2 * i], &paths[2 * i + 1]);
        }
        let pde_us = t0.elapsed().as_micros() as f64 / N_PAIRS_PER_DEPTH as f64;

        let t0 = Instant::now();
        for i in 0..N_PAIRS_PER_DEPTH {
            let _ = log_signature_truncated(&paths[2 * i], depth);
        }
        let log_us = t0.elapsed().as_micros() as f64 / N_PAIRS_PER_DEPTH as f64;

        println!(
            "{:>5} | {:>10} | {:>10} | {:>11.1} µs | {:>11.1} µs | {:>11.1} µs",
            depth, full_dim, log_dim, trunc_us, pde_us, log_us
        );
    }

    println!();
    println!("Reading:");
    println!("  - trunc_kernel grows ~d^(2N) — the wall.");
    println!("  - pde_kernel stays flat in depth (depth-∞ in O(T·T) flops).");
    println!("  - log_sig_compute pays the same Magnus cost but stores 7-13× less.");
    println!();
    println!("Production guidance:");
    println!("  - Need a kernel matrix? → signature_kernel_pde");
    println!("  - Need to STORE many signatures? → log_signature_truncated");
    println!("  - Need a fixed-width fingerprint? → RandomizedSignatureBuilder");
    println!("  - Need depth-2 features for an interpretable pipeline? → signature_truncated");
}
