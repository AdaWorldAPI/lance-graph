//! PROBE — field-perturbation placement learns the basin tree (E-BASIN-IS-A-NODE).
//!
//! The CONJECTURE under test (E-BASIN-IS-A-NODE, the one axis left ungraded):
//! *"the 4-ary basin fan-out IS the Morton pyramid, so the mailbox distribution
//! is a perturbation-learnable field — placement guided by the field cascade
//! clusters similar items at lower hop-distance than random placement."*
//!
//! Minimal falsifiable form. "Field perturbation" here is the **spectral**
//! perturbation already shipped: `hhtl_keys` assigns each node its
//! `(HEEL, HIP, TWIG)` address by recursive Cheeger/Fiedler bisection — the
//! dominant Laplacian perturbation eigenmode at each tier. That IS the basin
//! placement derived from the field. We test it against a random placement on
//! the metric that matters for `E-BASIN-IS-A-NODE`: the **tree-hop distance**
//! `(d−cpd)+(d−cpd)` over the HHTL prefix (the exact `node_distance(PrefixDepth)`
//! metric wired in `lance-graph::graph::mailbox_scan`).
//!
//! PASS/FAIL: on a grid with planted community structure (edges = ground-truth
//! similarity), the perturbation-derived placement must give **strictly lower
//! mean tree-hop over similar pairs** than random — and the gap must clear a
//! margin so it is not noise. If it does, the field-perturbation mechanism that
//! drives placement is real (the CONJECTURE's precondition holds: the basin tree
//! CAN represent similarity as low hop-distance, and the spectral perturbation
//! finds it). The full iterative learner (inject delta → minimise cascade
//! surprise → re-place) stays future work; this proves the substrate.
//!
//! Run: cargo run --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example basin_placement_learning

use perturbation_sim::{hhtl_keys, Edge, Grid, HhtlKey};

/// Deterministic SplitMix64 (workspace canonical seed) — keeps the probe zero-dep
/// and reproducible (no `rand`).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u16) -> u16 {
        (self.next_u64() % u64::from(n)) as u16
    }
}

/// A grid with `k` planted communities of `size` nodes each: dense intra-community
/// coupling, sparse inter-community bridges. Graph adjacency = ground-truth
/// similarity (similar nodes are connected). This is the structure a good basin
/// placement must recover as shared HHTL prefixes.
fn planted_communities(k: usize, size: usize) -> Grid {
    let n = k * size;
    let mut e = Vec::new();
    // Dense intra-community ring + chords (strong coupling = 1.0).
    for c in 0..k {
        let base = c * size;
        for i in 0..size {
            for j in (i + 1)..size {
                // ring neighbours + every third chord → a dense-ish block.
                if j == i + 1 || j == i + 2 || (i + j) % 3 == 0 {
                    e.push(Edge::new(base + i, base + j, 1.0, 1.0));
                }
            }
        }
    }
    // Sparse inter-community bridges (weak coupling = 0.05): one bridge per
    // adjacent community pair, anchored at node 0 of each.
    for c in 0..k {
        let next = (c + 1) % k;
        e.push(Edge::new(c * size, next * size, 0.05, 1.0));
    }
    Grid::new(n, e)
}

/// Longest-common-prefix depth of two depth-3 HHTL paths (HEEL▸HIP▸TWIG).
fn common_prefix_depth(a: HhtlKey, b: HhtlKey) -> u32 {
    if a.heel != b.heel {
        0
    } else if a.hip != b.hip {
        1
    } else if a.twig != b.twig {
        2
    } else {
        3
    }
}

/// The `node_distance(PrefixDepth)` tree-hop metric (mailbox_scan): the steps up
/// to the shared ancestor and back down. Depth is 3 (HEEL/HIP/TWIG).
fn tree_hop(a: HhtlKey, b: HhtlKey) -> u32 {
    let cpd = common_prefix_depth(a, b);
    (3 - cpd) + (3 - cpd)
}

/// Mean tree-hop over every similar pair (graph edge) under a placement.
fn mean_hop_over_similar(grid: &Grid, keys: &[HhtlKey]) -> f64 {
    let total: u32 = grid
        .edges
        .iter()
        .map(|e| tree_hop(keys[e.from], keys[e.to]))
        .sum();
    total as f64 / grid.edges.len() as f64
}

fn main() {
    let (k, size) = (4usize, 8usize);
    let grid = planted_communities(k, size);

    // Placement A — field-perturbation-derived: recursive Cheeger/Fiedler
    // bisection assigns the HHTL address. This is the spectral perturbation
    // structure used AS the basin placement.
    let perturbation = hhtl_keys(&grid);

    // Placement B — random: each node gets a random depth-3 binary path (the
    // same {0,1}³ range the binary-Cheeger instance fills). Deterministic seed.
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    let random: Vec<HhtlKey> = (0..grid.n)
        .map(|_| HhtlKey {
            heel: rng.below(2),
            hip: rng.below(2),
            twig: rng.below(2),
        })
        .collect();

    let hop_perturbation = mean_hop_over_similar(&grid, &perturbation);
    let hop_random = mean_hop_over_similar(&grid, &random);
    let improvement = (hop_random - hop_perturbation) / hop_random * 100.0;

    println!("PROBE — field-perturbation placement vs random (E-BASIN-IS-A-NODE)");
    println!(
        "  grid: {k} communities × {size} nodes = {} nodes, {} similar pairs (edges)",
        grid.n,
        grid.edges.len()
    );
    println!("  mean tree-hop over similar pairs:");
    println!("    perturbation-derived (Cheeger/HHTL): {hop_perturbation:.4}");
    println!("    random placement                   : {hop_random:.4}");
    println!("    improvement                         : {improvement:.1}%");

    // PASS/FAIL gate: perturbation placement must clear random by a clear margin
    // (≥ 15 %), not noise. Asserts so the probe fails loudly if the mechanism
    // does not hold.
    let margin = 15.0;
    assert!(
        hop_perturbation < hop_random && improvement >= margin,
        "FAIL: perturbation placement ({hop_perturbation:.4}) did not beat random \
         ({hop_random:.4}) by ≥ {margin}% (got {improvement:.1}%) — the \
         field-perturbation-learns-placement mechanism is NOT supported"
    );
    println!(
        "  VERDICT: PASS — perturbation-derived placement clusters similar items \
         {improvement:.1}% nearer in the basin tree than random (≥ {margin}% margin)."
    );
}
