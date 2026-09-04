//! σ-probe for `blasgraph::typed_graph::masked_traverse` — MEASUREMENT ONLY.
//!
//! `masked_traverse` today computes the FULL `A·A` product, then filters every
//! nonzero against a `Vec<bool>` label mask and rebuilds a COO. The open
//! question is whether pushing the mask *inside* the product could remove real
//! work, or whether it would only remove the filter+rebuild overhead.
//!
//! That is decided by selectivity, which is measurable BEFORE any inward-mask
//! implementation exists. This probe changes no library code: it calls `mxm`
//! directly for the unmasked baseline and `masked_traverse` for the masked
//! result, and reports per-call rows.
//!
//! Two selectivities, deliberately separated:
//!
//!   σ_result  = nnz(masked(A²)) / nnz(A²)     — result-space compression
//!   σ_columns = |M| / N                        — mask density (column space)
//!
//! Both tiny  ⟹ the case for mask pushdown is clean.
//! σ_columns ≪ σ_result ⟹ topology CONCENTRATES into the selected columns, and
//!   a naive 1/σ_columns speedup prediction would exaggerate the win.
//! σ_result ≈ 1 ⟹ pushdown cannot remove result work; stop.
//!
//! Distributions: dense random, uniform sparse, and block-clustered — the last
//! matters most, since uniform random sparsity alone is not a sufficient
//! benchmark for graph workloads. The clustered case is run with BOTH a
//! block-aligned mask and a random mask of identical density, which is what
//! isolates concentration from selectivity.
//!
//! cargo run -p lance-graph --release --example sigma_probe_masked_traverse

use lance_graph::graph::blasgraph::typed_graph::TypedGraph;
use lance_graph::graph::blasgraph::{BitVec, CooStorage, GrBDesc, GrBMatrix, HdrSemiring};

/// SplitMix64 — deterministic, so every run measures the same graphs.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`, top 53 bits so every draw is exact in `f64`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Edge list -> adjacency matrix. Each edge carries a deterministic BitVec
/// payload so the semiring has something non-trivial to compose.
fn matrix_from_edges(n: usize, edges: &[(usize, usize)]) -> GrBMatrix {
    let mut coo = CooStorage::new(n, n);
    for (idx, &(r, c)) in edges.iter().enumerate() {
        coo.push(r, c, BitVec::random(0x51ED_0000 + idx as u64));
    }
    GrBMatrix::from_coo(&coo)
}

/// Erdos-Renyi: every ordered pair independently present with probability `d`.
fn dist_random(n: usize, d: f64, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = Rng(seed);
    let mut e = Vec::new();
    for r in 0..n {
        for c in 0..n {
            if rng.unit() < d {
                e.push((r, c));
            }
        }
    }
    e
}

/// Block-clustered: `blocks` communities, dense inside, sparse across. This is
/// the graph-like case — the one where tile/mask structure is expected to pay,
/// and the one uniform random sparsity cannot stand in for.
fn dist_clustered(n: usize, blocks: usize, d_in: f64, d_out: f64, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = Rng(seed);
    let bsz = n / blocks;
    let mut e = Vec::new();
    for r in 0..n {
        for c in 0..n {
            let same = bsz > 0 && (r / bsz) == (c / bsz);
            if rng.unit() < if same { d_in } else { d_out } {
                e.push((r, c));
            }
        }
    }
    e
}

/// `count` node ids drawn uniformly without replacement.
fn mask_random(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng(seed);
    let mut ids: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    ids.truncate(count);
    ids.sort_unstable();
    ids
}

/// `count` CONTIGUOUS node ids — i.e. whole communities under `dist_clustered`.
/// Same density as `mask_random`, different alignment: the difference between
/// the two rows is precisely the concentration effect.
fn mask_blocks(count: usize) -> Vec<usize> {
    (0..count).collect()
}

/// One measured row. `nnz_full` comes from `mxm` directly; `nnz_masked` from
/// `masked_traverse`, i.e. the shipped path, unmodified.
#[allow(clippy::too_many_arguments)]
fn probe(label: &str, n: usize, edges: &[(usize, usize)], mask_name: &str, mask_ids: &[usize]) {
    let mut g = TypedGraph::new(n);
    g.add_relation("R", matrix_from_edges(n, edges));
    g.add_label("M", mask_ids);

    let sr = HdrSemiring::XorBundle;
    let a = g.relation("R").expect("relation R");
    let full = a.mxm(a, &sr, &GrBDesc::default());
    let nnz_full = full.nnz();

    let masked = g.masked_traverse("R", "M", &sr).expect("masked_traverse");
    let nnz_masked = masked.nnz();

    let sigma_result = if nnz_full == 0 {
        f64::NAN
    } else {
        nnz_masked as f64 / nnz_full as f64
    };
    let sigma_columns = mask_ids.len() as f64 / n as f64;
    // >1 means the mask's columns hold MORE than their share of the product:
    // topology concentrates into the selected region.
    let concentration = sigma_result / sigma_columns;

    println!(
        "{label:<26} {mask_name:<14} n={n:<5} edges={:<7} nnz_full={nnz_full:<7} \
         nnz_masked={nnz_masked:<7} sigma_result={sigma_result:>7.4} \
         sigma_columns={sigma_columns:>7.4} concentration={concentration:>6.2}x",
        edges.len()
    );
}

fn main() {
    // Kept small deliberately: `mxm` is O(N^3) over 2 KB `BitVec` values, so a
    // debug build at N=256 does not finish in useful time. Override with
    // `SIGMA_PROBE_N` for a --release run.
    let n: usize = std::env::var("SIGMA_PROBE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    println!("sigma-probe :: masked_traverse (measurement only, no library change)\n");
    println!("sigma_result  = nnz(masked(A^2)) / nnz(A^2)");
    println!("sigma_columns = |M| / N");
    println!("concentration = sigma_result / sigma_columns  (>1 = mass concentrates in M)\n");

    let densities = [(0.02_f64, "2%"), (0.10, "10%"), (0.40, "40%")];

    // A. dense random
    let a = dist_random(n, 0.15, 0xA11CE);
    for (frac, tag) in densities {
        let ids = mask_random(n, (n as f64 * frac) as usize, 0xD1);
        probe("A/dense-random d=0.15", n, &a, tag, &ids);
    }
    println!();

    // B. uniform sparse
    let b = dist_random(n, 0.02, 0xB0B);
    for (frac, tag) in densities {
        let ids = mask_random(n, (n as f64 * frac) as usize, 0xD2);
        probe("B/uniform-sparse d=0.02", n, &b, tag, &ids);
    }
    println!();

    // C. clustered — 16 communities of 16. Random mask first...
    let c = dist_clustered(n, 8, 0.50, 0.002, 0xC0FFEE);
    for (frac, tag) in densities {
        let ids = mask_random(n, (n as f64 * frac) as usize, 0xD3);
        probe("C/clustered rand-mask", n, &c, tag, &ids);
    }
    println!();
    // ...then a BLOCK-ALIGNED mask of identical density. The delta between
    // these two groups is the concentration effect, isolated.
    for (frac, tag) in densities {
        let ids = mask_blocks((n as f64 * frac) as usize);
        probe("C/clustered block-mask", n, &c, tag, &ids);
    }
}
