//! PROBE — CHAODA epicenter (`ndarray::hpc::clam`) on the real ES grid.
//! **Gated behind `ndarray-simd`** (uses ndarray's production `ClamTree` +
//! `anomaly_scores`, the real CLAM/CHAODA engine — not perturbation-sim's lite).
//!
//! The WHERE axis of the three-axis surge decomposition
//! (`E-FAMILY-BASIN-WEYL-HOP-LOCAL-AT-CRISP-TIER`): a surge's fail-first
//! compartment is the brittle seam — geometrically anomalous on the node
//! manifold. CHAODA's LFD-based anomaly should flag it **statically, with no
//! cascade simulation**. Hypothesis: the top-anomaly nodes concentrate on the
//! HHTL seam — the inter-HEEL-basin cut, the crisp 2-split bottleneck (the
//! "without family nodes" table's HIP 2-line seam, lines 46/150).
//!
//! Encoding: each node → a binary fingerprint = the **sign of its top-K Fiedler
//! eigenvector coordinates**, packed to bytes (the spectral embedding, the same
//! Laplacian spectrum the HHTL tiers come from). `ClamTree::build` (Hamming) +
//! `anomaly_scores` → per-node LFD anomaly. We then compare the seam-cut
//! endpoints' anomaly against the global mean and their share of the top anomalies.
//!
//! Honest gate: the epicenter↔anomaly claim is SUPPORTED only if the seam nodes'
//! mean anomaly ≥ 1.30× the global mean AND seam nodes are over-represented in
//! the top-quartile anomalies (lift ≥ 1.30). Reported as-is otherwise.
//!
//! Run: cargo run --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --features ndarray-simd --example chaoda_surge_epicenter \
//!        -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

#[cfg(not(feature = "ndarray-simd"))]
fn main() {
    eprintln!(
        "chaoda_surge_epicenter requires `--features ndarray-simd` \
         (it uses ndarray::hpc::clam — the real CLAM/CHAODA engine)."
    );
}

#[cfg(feature = "ndarray-simd")]
fn main() {
    use ndarray::hpc::clam::ClamTree;
    use perturbation_sim::{from_pypsa_csv, hhtl_keys, symmetric_eigen, Grid};
    use std::collections::HashSet;
    use std::fs;

    const K: usize = 64; // spectral-embedding bits per node (Fiedler coords 1..=K)

    let args: Vec<String> = std::env::args().collect();
    let (bpath, lpath, country) = (
        args.get(1)
            .map(String::as_str)
            .unwrap_or("/tmp/pypsa/buses.csv"),
        args.get(2)
            .map(String::as_str)
            .unwrap_or("/tmp/pypsa/lines.csv"),
        args.get(3).map(String::as_str).unwrap_or("ES"),
    );
    let buses = fs::read_to_string(bpath).expect("read buses.csv");
    let lines = fs::read_to_string(lpath).expect("read lines.csv");
    let import = from_pypsa_csv(&buses, &lines, Some(country))
        .expect("parse pypsa")
        .largest_component();
    let grid: &Grid = &import.grid;
    let (n, m) = (grid.n, grid.edges.len());

    // Spectral embedding: top-K Fiedler eigenvectors (skip j=0, the constant).
    let eig = symmetric_eigen(&grid.laplacian_of(&vec![true; m]), n);
    let k = K.min(n.saturating_sub(1));
    let vecs: Vec<Vec<f64>> = (1..=k).map(|j| eig.eigenvector(j)).collect();

    // Node fingerprint = sign bits of its K Fiedler coords, packed to bytes.
    let vec_len = k.div_ceil(8);
    let mut data = vec![0u8; n * vec_len];
    for (node, fp) in data.chunks_mut(vec_len).enumerate() {
        for (j, v) in vecs.iter().enumerate() {
            if v[node] >= 0.0 {
                fp[j / 8] |= 1 << (j % 8);
            }
        }
    }

    // The real ndarray CLAM/CHAODA engine.
    let tree = ClamTree::build(&data, vec_len, 4);
    let scores = tree.anomaly_scores(&data, vec_len);

    // HHTL seam = the inter-HEEL-basin cut (the crisp 2-split bottleneck).
    let keys = hhtl_keys(grid);
    let mut seam_nodes: HashSet<usize> = HashSet::new();
    for e in &grid.edges {
        if keys[e.from].heel != keys[e.to].heel {
            seam_nodes.insert(e.from);
            seam_nodes.insert(e.to);
        }
    }

    let mean = |idxs: &[usize]| -> f64 {
        if idxs.is_empty() {
            0.0
        } else {
            idxs.iter().map(|&i| scores[i].score).sum::<f64>() / idxs.len() as f64
        }
    };
    let all: Vec<usize> = (0..n).collect();
    let seam: Vec<usize> = seam_nodes.iter().copied().collect();
    let (anom_seam, anom_all) = (mean(&seam), mean(&all));
    let anom_ratio = if anom_all > 0.0 { anom_seam / anom_all } else { 0.0 };

    // Top-quartile anomalies: are seam nodes over-represented?
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| {
        scores[b]
            .score
            .partial_cmp(&scores[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_n = (n / 4).max(1);
    let top: HashSet<usize> = ranked[..top_n].iter().copied().collect();
    let seam_in_top = seam.iter().filter(|i| top.contains(i)).count();
    let expected = top_n as f64 * seam.len() as f64 / n as f64; // null expectation
    let lift = if expected > 0.0 {
        seam_in_top as f64 / expected
    } else {
        0.0
    };

    println!("PROBE — CHAODA surge epicenter (ndarray::hpc::clam, real {country} grid)");
    println!("  grid: {n} buses, {m} lines; {k}-bit Fiedler fingerprints ({vec_len} B each)");
    println!(
        "  seam (inter-HEEL-basin cut): {} nodes; CLAM tree built, anomaly_scores computed",
        seam.len()
    );
    println!("\n  anomaly (LFD-derived, higher = more anomalous):");
    println!("    mean anomaly, SEAM nodes : {anom_seam:.3}");
    println!("    mean anomaly, ALL nodes  : {anom_all:.3}");
    println!("    ratio (seam / all)       : {anom_ratio:.2}");
    println!(
        "    seam share of top-{top_n} anomalies: {seam_in_top}/{} (expected {expected:.1}, lift {lift:.2})",
        seam.len()
    );

    let supported = !seam.is_empty() && anom_ratio >= 1.30 && lift >= 1.30;
    println!("\n  VERDICT:");
    if supported {
        println!(
            "    [SUPPORTED] the seam (the brittle fail-first cut) IS a CHAODA anomaly — its mean \
             LFD anomaly is {anom_ratio:.2}× the global mean and it is {lift:.2}× over-represented \
             in the top-quartile anomalies. The surge EPICENTER is detectable statically from the \
             manifold geometry, no cascade simulation — the WHERE axis closes on real data."
        );
    } else {
        println!(
            "    [NOT SUPPORTED] seam anomaly ratio {anom_ratio:.2} / top-quartile lift {lift:.2} \
             do not clear 1.30 — on this grid the LFD anomaly does not single out the seam. Report \
             honestly: CHAODA flags geometric outliers, which need not coincide with the \
             spectral-cut bottleneck here. Do not promote."
        );
    }
    println!(
        "    NOTE: CHAODA anomaly = leaf-cluster LFD normalized over the tree; the seam is the \
         inter-HEEL-basin cut (the crisp 2-split). This is the WHERE axis (epicenter), distinct \
         from the Weyl HOW-MUCH and the family-basin HOW-it-spreads axes."
    );
}
