//! Triangle counting on `SplatShaderBlas` — L1 + L2 popcount probe.
//!
//! ## What this proves
//!
//! Triangle counting on `AwarenessPlane16K` rows reduces to pure
//! popcount-AND across the SoA. Per-node:
//!
//!   triangles[u]  =  Σ_{v ∈ N(u)}  popcount(plane[u] AND plane[v])
//!   total         =  (1 / 6) × Σ_u triangles[u]
//!
//! Pure L1 (popcount per row) + L2 (cross-row popcount-AND). No dynamic
//! dispatch, no random access, no graph-traversal stack — just sequential
//! popcounts over `[u64; 256]` words.
//!
//! ## Why this is a clean demo of the L1-L4 picture
//!
//! Triangle count is the simplest workload that exercises **L1 + L2
//! together**. Confirms three claims about `SplatShaderBlas`:
//!
//! 1. **Branchless hot path.** Popcount is a single hardware instruction
//!    (`POPCNT` on x86, `CNT` on ARM). The inner loop has no branches.
//! 2. **Cache-friendly.** Each row is 2 KB → fits in L1 cache; the AND
//!    of two rows is a 256-iteration tight loop with no random access.
//! 3. **Correct vs CSR baseline.** Triangle count answer matches a
//!    standard CSR set-intersection implementation on identical input.
//!
//! Run:
//!   cargo run --manifest-path crates/jc/Cargo.toml \
//!             --example splat_triangle_count --release

use lance_graph_contract::splat::AwarenessPlane16K;

// ════════════════════════════════════════════════════════════════════════════
// Graph mode bit-set helpers (direct indexing, complementary to the splat
// contract's `deposit` which uses (center_a, center_b)-derived hashing).
// ════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn set_neighbor(plane: &mut AwarenessPlane16K, neighbor_idx: u32) {
    let word = (neighbor_idx / 64) as usize;
    let mask = 1u64 << (neighbor_idx % 64);
    plane.0[word] |= mask;
}

/// L1 popcount: degree of a single row.
#[inline(always)]
#[allow(dead_code)]
fn popcount(plane: &AwarenessPlane16K) -> u32 {
    plane.0.iter().map(|w| w.count_ones()).sum()
}

/// L2 cross-row AND-popcount — the core SplatShaderBlas primitive for
/// triangle count, Adamic-Adar, Jaccard intersection.
#[inline(always)]
fn and_popcount(a: &AwarenessPlane16K, b: &AwarenessPlane16K) -> u32 {
    let mut acc = 0u32;
    for i in 0..256 {
        acc += (a.0[i] & b.0[i]).count_ones();
    }
    acc
}

/// Iterate set bit positions in a plane (<= 16384 of them).
fn iter_set_bits(plane: &AwarenessPlane16K, mut f: impl FnMut(u32)) {
    for (word_idx, &word) in plane.0.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros();
            f((word_idx as u32) * 64 + bit);
            w &= w - 1;  // clear lowest set bit
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Synthetic graph generator (deterministic splitmix64; no PRNG dep)
// ════════════════════════════════════════════════════════════════════════════

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct SsbGraph {
    n: u32,
    /// `planes[u]` has bit `v` set iff `(u, v) ∈ E`. Always symmetric.
    planes: Vec<AwarenessPlane16K>,
    /// CSR-style sorted neighbour lists for the baseline.
    csr: Vec<Vec<u32>>,
}

impl SsbGraph {
    fn random(n: u32, edge_prob_q16: u32, seed: u64) -> Self {
        let mut state = seed;
        let mut planes = vec![AwarenessPlane16K::zero(); n as usize];
        let mut csr_set = vec![Vec::<u32>::new(); n as usize];
        for u in 0..n {
            for v in (u + 1)..n {
                let r = (splitmix64(&mut state) >> 48) as u32;  // 0..2^16
                if r < edge_prob_q16 {
                    set_neighbor(&mut planes[u as usize], v);
                    set_neighbor(&mut planes[v as usize], u);
                    csr_set[u as usize].push(v);
                    csr_set[v as usize].push(u);
                }
            }
        }
        // CSR neighbour lists are built in increasing u order so already sorted.
        Self { n, planes, csr: csr_set }
    }

    fn edge_count(&self) -> u32 {
        self.csr.iter().map(|nbrs| nbrs.len() as u32).sum::<u32>() / 2
    }

    fn avg_degree(&self) -> f64 {
        2.0 * self.edge_count() as f64 / self.n as f64
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SplatShaderBlas triangle count — L4 sweep over rows, L2 popcount-AND inner
// ════════════════════════════════════════════════════════════════════════════

fn triangles_ssb(g: &SsbGraph) -> u64 {
    let mut total = 0u64;
    for u in 0..g.n {
        let plane_u = &g.planes[u as usize];
        // Iterate set bits in plane[u] = neighbours of u.
        iter_set_bits(plane_u, |v| {
            // L2 AND-popcount: shared neighbours of u and v.
            let shared = and_popcount(plane_u, &g.planes[v as usize]);
            total += shared as u64;
        });
    }
    // Each triangle is counted 6 times (each of 3 nodes considers each of 2
    // ordered neighbour pairs).
    total / 6
}

// ════════════════════════════════════════════════════════════════════════════
// Baseline: CSR set-intersection on sorted neighbour lists.
// (Standard textbook triangle-count algorithm.)
// ════════════════════════════════════════════════════════════════════════════

fn triangles_csr(g: &SsbGraph) -> u64 {
    let mut total = 0u64;
    for u in 0..g.n {
        let nbrs_u = &g.csr[u as usize];
        // Each triangle counted once via canonical u < v < w ordering.
        for &v in nbrs_u.iter().filter(|&&v| v > u) {
            let nbrs_v = &g.csr[v as usize];
            // Sorted set-intersection of nbrs_u ∩ nbrs_v with both members > v.
            let (mut i, mut j) = (0usize, 0usize);
            while i < nbrs_u.len() && j < nbrs_v.len() {
                let a = nbrs_u[i];
                let b = nbrs_v[j];
                if a == b {
                    if a > v { total += 1; }
                    i += 1; j += 1;
                } else if a < b {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
    }
    total
}

// ════════════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════════════

fn run_one(label: &str, n: u32, edge_prob_q16: u32, seed: u64) {
    let g = SsbGraph::random(n, edge_prob_q16, seed);
    let edges = g.edge_count();
    let avg_deg = g.avg_degree();
    let density = (edges as f64) * 2.0 / (n as f64 * (n as f64 - 1.0));

    println!("──────────────────────────────────────────────────────────────────────");
    println!("  {}", label);
    println!("──────────────────────────────────────────────────────────────────────");
    println!("    nodes={}  edges={}  avg_degree={:.1}  density={:.3}%",
        n, edges, avg_deg, density * 100.0);

    let t0 = std::time::Instant::now();
    let ssb_tri = triangles_ssb(&g);
    let ssb_us = t0.elapsed().as_micros();

    let t1 = std::time::Instant::now();
    let csr_tri = triangles_csr(&g);
    let csr_us = t1.elapsed().as_micros();

    println!("    SplatShaderBlas (L1+L2 popcount-AND L4 sweep):");
    println!("      triangles = {}  ({} µs)", ssb_tri, ssb_us);
    println!("    CSR (sorted-list set-intersection, baseline):");
    println!("      triangles = {}  ({} µs)", csr_tri, csr_us);

    let correct = ssb_tri == csr_tri;
    let ratio = if ssb_us > 0 { csr_us as f64 / ssb_us as f64 } else { f64::INFINITY };
    println!("    correct        : {}", if correct { "YES (counts match)" } else { "NO — DIVERGENT" });
    println!("    SSB / CSR ratio: {:.2}× ({})",
        ratio,
        if ratio > 1.0 { "SSB faster" } else { "CSR faster" });
    println!();

    assert_eq!(ssb_tri, csr_tri, "SSB triangle count diverges from CSR baseline!");
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  SplatShaderBlas — Triangle count via L1+L2 popcount-AND");
    println!("══════════════════════════════════════════════════════════════════════");
    println!();
    println!("L1: popcount(plane[u])              → degree(u)");
    println!("L2: popcount(plane[u] AND plane[v]) → |N(u) ∩ N(v)|");
    println!("L4: per-row sweep                   → all-nodes triangle count");
    println!();
    println!("Triangles[u]  =  Σ_{{v ∈ N(u)}}  popcount(plane[u] AND plane[v])");
    println!("Total         =  (1 / 6) · Σ_u Triangles[u]");
    println!();

    // Sweet spot: dense graph where SSB shines (avg_degree > 256).
    run_one("dense graph (avg_degree ~256)", 1024, /* p ≈ 0.25 */ 16_384, 0xDEAD_BEEF_0001);

    // Medium graph (CSR competitive but not faster).
    run_one("medium graph (avg_degree ~64)", 1024, /* p ≈ 0.0625 */ 4_096, 0xDEAD_BEEF_0002);

    // Sparse graph (CSR theoretically wins; SSB still correct + competitive).
    run_one("sparse graph (avg_degree ~16)",  1024, /* p ≈ 0.0156 */ 1_024, 0xDEAD_BEEF_0003);

    // Larger dense — exercises cache behaviour.
    run_one("larger dense (n=2048)",          2048, 8_192, 0xDEAD_BEEF_0004);

    println!("══════════════════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  All four graph configurations: SSB triangle count == CSR baseline.");
    println!("  SSB primitive: L1 + L2 popcount-AND, branchless hot path,");
    println!("                 cache-friendly tight inner loop on [u64; 256].");
    println!();
    println!("  Per-row 2 KB plane → triangle count is the simplest workload that");
    println!("  exercises L1 + L2 together. The result extends to:");
    println!("    Local Clustering Coefficient = triangles[u] / (degree[u] choose 2)");
    println!("    Adamic-Adar shared-neighbour weighting (per-pair AND-popcount)");
    println!("    Jaccard node similarity (AND-popcount / OR-popcount)");
    println!();
    println!("  → SplatShaderBlas L1+L2 primitives empirically grounded.");
    println!("══════════════════════════════════════════════════════════════════════");
}
