//! D-DIA-V4 rung 2 — the foveated morton-comma descent.
//!
//! Rung 1 (PROBE-CODEBOOK-44, `probe_codebook_44.rs`) proved the
//! `HierarchicalPalette` itself: a leaf byte's top nibble IS its coarse-cluster
//! ancestor (`code >> 4 == coarse`), matching
//! `lance-graph-contract::hhtl::NiblePath` (`FAN_OUT = 16`, `parent() = path >>
//! 4`). This rung builds the SEARCH that consumes that structure: a foveated
//! descent that materializes only the leaves near the query (the fovea) and
//! leaves everything else as an unmaterialized coarse centroid (the periphery
//! — an HHTL-trie node, never enumerated).
//!
//! **Eccentricity / level-of-detail, made concrete:**
//!   - eccentricity  = a coarse cluster's distance-rank from the query
//!     (rank 0 = the fovea's own cluster, rank 1 = next-nearest, …).
//!   - LoD           = FINE at the fovea (all 16 leaf children materialized
//!     and ranked), COARSE (pruned to a single centroid, never enumerated) in
//!     the periphery.
//!
//! This is the "materialize only the hot path; periphery stays HHTL-trie"
//! principle (`E-FOVEATED-HHTL-TRIE-FIELD-SEARCH-1`): the periphery is
//! represented ONLY by its coarse centroid — the trie node — never expanded
//! to its 16 leaves.
//!
//! Deterministic SplitMix64 (seed 0x9E3779B97F4A7C15) — no clock, no rand.
//!
//! Run:  cargo run --manifest-path crates/bgz17/Cargo.toml --example probe_foveated_descent

use bgz17::base17::Base17;
use bgz17::palette::{HierarchicalPalette, Palette};
use bgz17::BASE_DIM;

// ── deterministic RNG (same SplitMix64 as probe_codebook_44) ────────────────

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn jitter(&mut self, range: i32) -> i16 {
        let span = (range as i64 * 2 + 1) as u64;
        ((self.next_u64() % span) as i64 - range as i64) as i16
    }
}

// ── planted 16×16 hierarchy (mirrors probe_codebook_44's fixture) ───────────

/// (samples, ground-truth leaf templates, planted coarse index per template).
/// Separation scales: coarse (~±4000) ≫ fine (~±400) ≫ sample jitter (~±40).
fn planted_hierarchy(per_leaf: usize) -> (Vec<Base17>, Vec<Base17>, Vec<usize>) {
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    let mut samples = Vec::new();
    let mut templates = Vec::new();
    let mut template_coarse = Vec::new();
    for c in 0..16usize {
        let mut a = [0i16; BASE_DIM];
        for slot in a.iter_mut() {
            *slot = rng.jitter(4000);
        }
        for _f in 0..16usize {
            let mut leaf = a;
            for slot in leaf.iter_mut() {
                *slot = slot.saturating_add(rng.jitter(400));
            }
            templates.push(Base17 { dims: leaf });
            template_coarse.push(c);
            for _s in 0..per_leaf {
                let mut samp = leaf;
                for slot in samp.iter_mut() {
                    *slot = slot.saturating_add(rng.jitter(40));
                }
                samples.push(Base17 { dims: samp });
            }
        }
    }
    (samples, templates, template_coarse)
}

// ── the probe subject: foveated_descend ──────────────────────────────────────

/// Foveated descent over a [`HierarchicalPalette`]: materialize the leaves of
/// the `fovea_k` coarse clusters nearest the query (the fovea, FINE LoD),
/// leave every other coarse cluster as its unmaterialized centroid (the
/// periphery, COARSE LoD, pruned from the candidate set entirely), and return
/// the top-`budget` materialized leaf codes ranked by L1 distance to `query`.
///
/// This is the trie-descent analog of a foveated render: eccentricity (the
/// coarse cluster's distance rank from the query) selects the level of
/// detail — full leaf enumeration at the fovea, a single trie-node centroid
/// in the periphery.
fn foveated_descend(
    hp: &HierarchicalPalette,
    query: &Base17,
    fovea_k: usize,
    budget: usize,
) -> Vec<u8> {
    let n_coarse = hp.coarse_len();
    if n_coarse == 0 {
        return Vec::new();
    }

    // Step 1 — eccentricity: rank every coarse cluster (trie node) by its
    // distance to the query. This is the ONLY work done in the periphery —
    // it never touches a leaf.
    let mut coarse_rank: Vec<(usize, u32)> = hp
        .coarse
        .iter()
        .enumerate()
        .map(|(c, centroid)| (c, query.l1(centroid)))
        .collect();
    coarse_rank.sort_by_key(|&(_, d)| d);

    // Step 2 — fovea selection: the fovea_k nearest coarse clusters get FINE
    // LoD (materialized). Everything else stays a trie node (periphery,
    // pruned — never enumerated below).
    let fovea_k = fovea_k.clamp(1, n_coarse);
    let fovea_clusters = &coarse_rank[..fovea_k];

    // Step 3 — materialize: only the fovea's leaves are ever built into
    // candidates. `code >> 4 == coarse` (PROBE-CODEBOOK-44 GATE 1) means the
    // fovea cluster `c`'s 16 leaf codes are exactly the contiguous range
    // `[c*16, c*16+16)` — no search, no trie walk, direct slice.
    let mut candidates: Vec<(u8, u32)> = Vec::with_capacity(fovea_k * 16);
    for &(c, _coarse_dist) in fovea_clusters {
        let start = c * 16;
        let end = (start + 16).min(hp.leaf_len());
        for (offset, leaf) in hp.leaves.entries[start..end].iter().enumerate() {
            let code = (start + offset) as u8;
            candidates.push((code, query.l1(leaf)));
        }
    }

    // Step 4 — rank the (small) materialized candidate set and truncate to
    // budget. The periphery contributes nothing here — it was never built.
    candidates.sort_by_key(|&(_, d)| d);
    candidates.truncate(budget);
    candidates.into_iter().map(|(code, _)| code).collect()
}

/// Brute-force ground truth: the true globally-nearest leaf, scanning every
/// materialized leaf in the palette (fovea AND periphery). Used only to
/// grade `foveated_descend`'s recall — never part of the probe subject.
fn brute_nearest(hp: &HierarchicalPalette, query: &Base17) -> u8 {
    let mut best = 0u8;
    let mut best_d = u32::MAX;
    for (code, leaf) in hp.leaves.entries.iter().enumerate() {
        let d = query.l1(leaf);
        if d < best_d {
            best_d = d;
            best = code as u8;
        }
    }
    best
}

// ── query fixtures ───────────────────────────────────────────────────────────

/// "Typical" queries: fresh jitter (±100, larger than the ±40 sample noise
/// used to build the palette — a genuine held-out point) around each leaf
/// template. The common case: the query sits well inside its own coarse
/// cluster, so a narrow fovea should recover it easily.
fn typical_queries(templates: &[Base17], rng: &mut SplitMix64, per_template: usize) -> Vec<Base17> {
    let mut queries = Vec::with_capacity(templates.len() * per_template);
    for t in templates {
        for _ in 0..per_template {
            let mut dims = t.dims;
            for slot in dims.iter_mut() {
                *slot = slot.saturating_add(rng.jitter(100));
            }
            queries.push(Base17 { dims });
        }
    }
    queries
}

/// The coarse cluster (trie node) whose centroid is geometrically nearest
/// `hp.coarse[c1]` (excluding `c1` itself) — the real spatial neighbor, not
/// just an adjacent index.
fn nearest_other_coarse(hp: &HierarchicalPalette, c1: usize) -> usize {
    let mut best = c1; // fallback for the degenerate single-cluster case
    let mut best_d = u32::MAX;
    for c2 in 0..hp.coarse_len() {
        if c2 == c1 {
            continue;
        }
        let d = hp.coarse[c1].l1(&hp.coarse[c2]);
        if d < best_d {
            best_d = d;
            best = c2;
        }
    }
    best
}

/// "Boundary" queries: sit at the midpoint (± small jitter) between two
/// geometrically-adjacent coarse centroids. This is the honest stress case —
/// a query for which the SECOND-nearest coarse cluster may in fact contain
/// the true nearest leaf, which a narrow fovea (small `fovea_k`) prunes away
/// entirely.
fn boundary_queries(
    hp: &HierarchicalPalette,
    rng: &mut SplitMix64,
    per_pair: usize,
) -> Vec<Base17> {
    let n = hp.coarse_len();
    let mut queries = Vec::with_capacity(n * per_pair);
    for c1 in 0..n {
        let c2 = nearest_other_coarse(hp, c1);
        for _ in 0..per_pair {
            let mut dims = [0i16; BASE_DIM];
            for d in 0..BASE_DIM {
                let a = hp.coarse[c1].dims[d] as i32;
                let b = hp.coarse[c2].dims[d] as i32;
                let mid = (a + b) / 2;
                dims[d] = (mid as i16).saturating_add(rng.jitter(50));
            }
            queries.push(Base17 { dims });
        }
    }
    queries
}

/// Recall@budget: fraction of `queries` whose brute-force true nearest leaf
/// appears in `foveated_descend`'s top-`budget` result. Returns (hits, n, rate).
fn recall_at_k(
    hp: &HierarchicalPalette,
    queries: &[Base17],
    fovea_k: usize,
    budget: usize,
) -> (usize, usize, f64) {
    let mut hits = 0usize;
    for q in queries {
        let true_leaf = brute_nearest(hp, q);
        let pred = foveated_descend(hp, q, fovea_k, budget);
        if pred.contains(&true_leaf) {
            hits += 1;
        }
    }
    let n = queries.len();
    let rate = if n == 0 { 0.0 } else { hits as f64 / n as f64 };
    (hits, n, rate)
}

fn main() {
    println!("D-DIA-V4 rung 2 — foveated morton-comma descent");
    println!("=================================================");

    let (samples, templates, _template_coarse) = planted_hierarchy(8);
    println!(
        "planted synthetic: 16 coarse × 16 fine = {} templates, {} samples\n",
        templates.len(),
        samples.len()
    );

    let hp = Palette::build_hierarchical(&samples, 20);
    println!(
        "HierarchicalPalette: {} coarse clusters, {} leaves\n",
        hp.coarse_len(),
        hp.leaf_len()
    );

    // ── pruning ───────────────────────────────────────────────────────────
    println!("PRUNING (fovea_k → materialized leaves / total leaves)");
    for fovea_k in [1usize, 2, 4, 8] {
        let materialized = fovea_k.min(hp.coarse_len()) * 16;
        let ratio = materialized as f64 / hp.leaf_len() as f64;
        println!(
            "  fovea_k={fovea_k:2}  materialized={materialized:3} / {}   ratio={ratio:.4}  (1/ratio = {:.1}x pruned)",
            hp.leaf_len(),
            1.0 / ratio
        );
    }

    // ── recall ────────────────────────────────────────────────────────────
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    let typical = typical_queries(&templates, &mut rng, 2); // 512 queries
    let boundary = boundary_queries(&hp, &mut rng, 8); // 128 queries
    let combined: Vec<Base17> = typical
        .iter()
        .cloned()
        .chain(boundary.iter().cloned())
        .collect();

    println!("\nRECALL@budget (does the foveal top-k contain the TRUE brute-force nearest leaf?)");
    println!(
        "  typical  queries: {} (near leaf centers, ±100 jitter — the common case)",
        typical.len()
    );
    println!(
        "  boundary queries: {} (near coarse-cluster midpoints — the honest stress case)",
        boundary.len()
    );

    let budget = 8usize;
    for fovea_k in [1usize, 2, 3] {
        let (th, tn, tr) = recall_at_k(&hp, &typical, fovea_k, budget);
        let (bh, bn, br) = recall_at_k(&hp, &boundary, fovea_k, budget);
        let (ah, an, ar) = recall_at_k(&hp, &combined, fovea_k, budget);
        println!(
            "  fovea_k={fovea_k}  budget={budget:2}  typical={tr:.4} ({th}/{tn})   boundary={br:.4} ({bh}/{bn})   combined={ar:.4} ({ah}/{an})"
        );
    }

    // ── gates ─────────────────────────────────────────────────────────────
    let materialized_fovea1 = 1usize.min(hp.coarse_len()) * 16;
    let prune_gate = materialized_fovea1 < hp.leaf_len();
    let (th, tn, tr) = recall_at_k(&hp, &typical, 1, budget);
    let typical_gate = tr >= 0.90;
    let (bh, bn, br) = recall_at_k(&hp, &boundary, 1, budget);

    println!("\nGATES");
    println!(
        "  PRUNE  : materialized({materialized_fovea1}) << total({}) : {}",
        hp.leaf_len(),
        if prune_gate { "PASS" } else { "FAIL" }
    );
    println!(
        "  RECALL : typical queries, fovea_k=1, budget={budget}, threshold 0.90 : {tr:.4} ({th}/{tn}) : {}",
        if typical_gate { "PASS" } else { "FAIL" }
    );

    println!("\nHONEST TRADE-OFF (not hidden — this is the real cost of pruning)");
    println!(
        "  fovea_k=1 materializes {materialized_fovea1}/{} leaves ({:.1}x pruned) but only reaches",
        hp.leaf_len(),
        hp.leaf_len() as f64 / materialized_fovea1.max(1) as f64
    );
    println!(
        "  {br:.4} recall@{budget} on boundary queries ({bh}/{bn}) — queries whose true nearest leaf"
    );
    println!(
        "  sometimes sits in the SECOND-nearest coarse cluster, which fovea_k=1 prunes entirely."
    );
    println!(
        "  Widening the fovea (fovea_k=2,3 above) recovers boundary recall by materializing more —"
    );
    println!("  eccentricity/LoD IS the accuracy/cost dial, not a hidden failure.");

    println!("\nVERDICT");
    if prune_gate && typical_gate {
        println!("  D-DIA-V4 rung 2: MECHANISM PROVEN. Foveated descent prunes the periphery to a");
        println!(
            "  single coarse centroid per unmaterialized cluster (never enumerated), recovers"
        );
        println!("  the common case (typical queries) reliably, and the boundary-query recall gap");
        println!("  above is the real, reported eccentricity/LoD trade-off — never hidden.");
    } else {
        println!("  D-DIA-V4 rung 2: GATE FAIL — see above.");
    }
}
