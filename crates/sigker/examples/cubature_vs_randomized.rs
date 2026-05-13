//! Compare runtime of three sigker computation modes at production
//! carrier widths: trivial cubature hydration, randomized signature
//! encoding, and Goursat-PDE kernel evaluation.
//!
//! Run:
//!   cargo run --manifest-path crates/sigker/Cargo.toml \
//!             --example cubature_vs_randomized --release

use sigker::cubature::{trivial_constant_cubature, hydrate_signature};
use sigker::kernel::signature_kernel_pde;
use sigker::randomized::RandomizedSignatureBuilder;

use std::time::Instant;

const PATH_DIM: usize = 4;     // OSINT-typical edge feature dim
const PATH_LEN: usize = 64;    // OSINT-typical sub-path length
const N_PATHS: usize = 256;    // Bench batch size
const SIG_RAND_DIM: usize = 256;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_u(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn rand_n(state: &mut u64) -> f64 {
    let u1 = rand_u(state).max(1e-300);
    let u2 = rand_u(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn random_path(state: &mut u64) -> Vec<Vec<f64>> {
    let mut path = Vec::with_capacity(PATH_LEN);
    let mut pt = vec![0.0; PATH_DIM];
    path.push(pt.clone());
    for _ in 1..PATH_LEN {
        for x in pt.iter_mut() {
            *x += rand_n(state) * 0.3;
        }
        path.push(pt.clone());
    }
    path
}

fn main() {
    println!("sigker compute-mode bench");
    println!(
        "PATH_DIM={PATH_DIM}, PATH_LEN={PATH_LEN}, N_PATHS={N_PATHS}, \
         randomized state dim = {SIG_RAND_DIM}"
    );
    println!();

    // Generate the path bank.
    let mut state: u64 = 0xADAF_00D0_C0DE_B175;
    let paths: Vec<Vec<Vec<f64>>> = (0..N_PATHS).map(|_| random_path(&mut state)).collect();

    // ────────────────────────────────────────────────────────────────────────
    // 1. Trivial cubature hydration (degree 0 — framework correctness anchor).
    //    This is the baseline for "hydrate via basis lookup". The trivial
    //    cubature returns the constant 1 regardless of query, so timing
    //    reflects only the basis dispatch overhead — the lower bound on
    //    cubature-style hydration cost.
    // ────────────────────────────────────────────────────────────────────────
    let basis = trivial_constant_cubature(PATH_DIM);
    let t0 = Instant::now();
    let mut hydration_sink = 0.0f64;
    for path in &paths {
        let sig = hydrate_signature(path, &basis);
        hydration_sink += sig.levels[0][0];
    }
    let t_hydrate = t0.elapsed();
    println!(
        "  trivial cubature hydration  : {:>8.2} µs / path  ({:>10.2} ms total, sink={hydration_sink})",
        t_hydrate.as_secs_f64() * 1e6 / N_PATHS as f64,
        t_hydrate.as_secs_f64() * 1e3,
    );

    // ────────────────────────────────────────────────────────────────────────
    // 2. Randomized signature encoding (Cuchiero et al. 2021).
    //    Universal approximator at fixed carrier width; dominant runtime mode
    //    for sigker today.
    // ────────────────────────────────────────────────────────────────────────
    let builder = RandomizedSignatureBuilder::new(PATH_DIM, SIG_RAND_DIM, 0xCAFE);
    let t0 = Instant::now();
    let mut rand_sink = 0.0f64;
    for path in &paths {
        let s = builder.encode(path);
        rand_sink += s.state[0];
    }
    let t_rand = t0.elapsed();
    println!(
        "  randomized signature        : {:>8.2} µs / path  ({:>10.2} ms total, sink={rand_sink:.4})",
        t_rand.as_secs_f64() * 1e6 / N_PATHS as f64,
        t_rand.as_secs_f64() * 1e3,
    );

    // ────────────────────────────────────────────────────────────────────────
    // 3. Goursat-PDE kernel — pairwise (the kernel-matrix consumer pattern).
    //    Cost is per-pair, so we measure for N_PATHS pairs (path_i vs path_0)
    //    rather than the full O(N²) matrix.
    // ────────────────────────────────────────────────────────────────────────
    let pivot = &paths[0];
    let t0 = Instant::now();
    let mut pde_sink = 0.0f64;
    for path in &paths {
        pde_sink += signature_kernel_pde(pivot, path);
    }
    let t_pde = t0.elapsed();
    println!(
        "  Goursat-PDE kernel (pair)   : {:>8.2} µs / pair  ({:>10.2} ms total, sink={pde_sink:.2})",
        t_pde.as_secs_f64() * 1e6 / N_PATHS as f64,
        t_pde.as_secs_f64() * 1e3,
    );

    println!();
    println!("Reading the numbers:");
    println!("  Hydration via the trivial degree-0 cubature is the framework lower bound");
    println!("  — replacing it with a real Lyons-Victoir basis adds the basis-projection");
    println!("  flops (M · N²·d for an M-path basis at degree N).");
    println!();
    println!("  Randomized signature is the production speed today: O(L · k²) per path,");
    println!("  giving fixed-width fingerprints with universality guarantees.");
    println!();
    println!("  Goursat PDE is per-pair: O(L₁·L₂·d) flops with no signature");
    println!("  materialization, sidestepping the d^(2N) wall entirely. Use for");
    println!("  kernel-matrix consumers (SVMs, GPs, kernel ridge regression).");
}
