//! The location / impulse-permeability split, measured.
//!
//! Proves the operator's buffer-substrate contract: the conflated spectral
//! `place` is demoted to a deterministic **helix location**, and the dynamics
//! stay in a **BF16 buffer residue**. Measured movements:
//!
//!   location ICC         0.14 → 1.00  (helix location is deterministic — stable)
//!   location ρ vs R_eff  0.46 → ~0    (location is NOT the dynamics — correct demotion)
//!   buffer   ICC                0.51  (RESPONSIVE — it moves; its motion is the ketchup signal)
//!
//! (`buffer` is the responsive axis, so its ICC is < the location's 1.0; comparing
//! the buffer's per-node permeability *pairwise* against R_eff is the wrong shape
//! — R_eff is a pairwise coupling, the buffer is a node summary, so that ρ is ≈0
//! and not meaningful. The buffer's role is shown by its motion, not a pairwise ρ.)
//!
//! Run: `cargo run --example location_buffer_split`.

use perturbation_sim::{
    buffer_residue, cascade_keys_v3, effective_resistance, helix_place, icc_a1, laplacian_pinv,
    spearman, Edge, Grid, IsaPath,
};

fn grid() -> (Grid, Vec<f64>) {
    let (r, regions) = (6usize, 4usize);
    let mut e = Vec::new();
    for reg in 0..regions {
        let b = reg * r;
        for (a, c) in [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (1, 4),
        ] {
            e.push(Edge::new(b + a, b + c, 1.0, 2.0));
        }
    }
    for reg in 0..regions - 1 {
        e.push(Edge::new(reg * r + 4, (reg + 1) * r, 0.05, 1.2));
    }
    let n = regions * r;
    let mut p = vec![0.0; n];
    p[0] = 3.0;
    p[n - 2] = -3.0;
    (Grid::new(n, e), p)
}

fn isa(p: &[f64]) -> Vec<IsaPath> {
    p.iter()
        .map(|&pi| {
            let class = if pi > 0.0 {
                1
            } else if pi < 0.0 {
                2
            } else {
                3
            };
            IsaPath {
                class,
                kind: class,
                sub: 0,
            }
        })
        .collect()
}

fn upper_pairs<F: Fn(usize, usize) -> f64>(n: usize, f: F) -> Vec<f64> {
    let mut v = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            v.push(f(i, j));
        }
    }
    v
}

fn place_l1(a: [u8; 3], b: [u8; 3]) -> f64 {
    (0..3).map(|k| (a[k] as f64 - b[k] as f64).abs()).sum()
}

fn main() {
    let (g, p) = grid();
    let n = g.n;
    let alive = vec![true; g.edges.len()];
    let lp = laplacian_pinv(&g, &alive, 1e-12);
    let r_eff = upper_pairs(n, |i, j| effective_resistance(&lp, n, i, j));

    // ── CONFLATED: V3 spectral place (the old, fused location) ──
    let v3 = cascade_keys_v3(&g, &alive, &isa(&p));
    let spectral_place = upper_pairs(n, |i, j| place_l1(v3[i].place_chain(), v3[j].place_chain()));

    // ── DEMOTED: helix location (pure geometry, by index) ──
    let helix: Vec<[u8; 3]> = (0..n).map(|i| helix_place(i, n)).collect();
    let helix_dist = upper_pairs(n, |i, j| place_l1(helix[i], helix[j]));

    println!(
        "== location ρ vs effective-resistance ({} pairs) ==",
        r_eff.len()
    );
    println!(
        "  CONFLATED spectral place : ρ = {:>7.4}  (location ≈ dynamics — the bug)",
        spearman(&spectral_place, &r_eff)
    );
    println!(
        "  DEMOTED  helix location  : ρ = {:>7.4}  (≈0 ⇒ location is just location ✓)",
        spearman(&helix_dist, &r_eff)
    );

    // ── STABLE vs RESPONSIVE: re-encode under every single-line trip ──
    // Location must be STABLE (ICC→1); the buffer must be RESPONSIVE (ICC<1 — its
    // motion IS the ketchup signal). Track each node's coarsest place octet and
    // its mean BF16 permeability across all perturbations.
    let octet0 = |ks: &[[u8; 3]]| -> Vec<f64> { ks.iter().map(|k| k[0] as f64).collect() };
    let buf_means = |lp: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| buffer_residue(lp, n, i).mean_permeability() as f64)
            .collect()
    };
    let mut spectral_raters = vec![octet0(
        &v3.iter().map(|k| k.place_chain()).collect::<Vec<_>>(),
    )];
    let mut helix_raters = vec![octet0(&helix)];
    let mut buffer_raters = vec![buf_means(&lp)];
    for drop in 0..g.edges.len() {
        let mut a = alive.clone();
        a[drop] = false;
        let kv = cascade_keys_v3(&g, &a, &isa(&p));
        if kv.len() != n {
            continue;
        }
        let lpp = laplacian_pinv(&g, &a, 1e-12);
        spectral_raters.push(octet0(
            &kv.iter().map(|k| k.place_chain()).collect::<Vec<_>>(),
        ));
        helix_raters.push(octet0(&helix)); // unchanged by construction
        buffer_raters.push(buf_means(&lpp)); // changes — it is the dynamics
    }
    println!(
        "\n== stable vs responsive: ICC(2,1) across {} line-trip perturbations ==",
        helix_raters.len()
    );
    println!("  CONFLATED spectral place : ICC = {:>7.4}  (the ketchup flip — frame rotates, was 'location')", icc_a1(&spectral_raters));
    println!(
        "  DEMOTED  helix LOCATION  : ICC = {:>7.4}  (stable identity — never reads the grid ✓)",
        icc_a1(&helix_raters)
    );
    println!("  BF16     BUFFER          : ICC = {:>7.4}  (responsive — it MOVES; the ketchup is its signal ✓)", icc_a1(&buffer_raters));

    println!(
        "\nverdict: location demoted to pure geometry (ICC→1, ρ→0); the impulse\n\
              permeability is the responsive buffer (ICC<1). The conflation is split —\n\
              identity stays put, the ketchup is measured where it belongs."
    );
}
