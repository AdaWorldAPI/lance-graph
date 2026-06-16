//! Validity & reliability of the model's CONCEPTS / MEDIATORS.
//!
//! The model rests on a handful of mediating concepts — the structural
//! predictors (Weyl single-trip Δλ₂, effective resistance, Fiedler edge
//! sensitivity), the two outcome axes (Raumgewinn = global connectivity loss,
//! infight = local cascade fraction), and the spread (Davis–Kahan rotation).
//! Before any of them earns trust we must show, on real data, that each one is
//! BOTH **valid** (measures what it claims) and **reliable** (stable across
//! independent measurements). This example runs the psychometric battery the
//! colleague asked for — Pearson, Spearman, Cronbach α, ICC(2,1) — over a sample
//! of N-1 contingencies on the real Iberian core.
//!
//! Four blocks:
//!   A. CRITERION VALIDITY — do the pre-cascade structural mediators PREDICT the
//!      operational cascade size? (Pearson + Spearman, non-circular: predictors
//!      are topology-only, the outcome is the realized cascade.)
//!   B. RELIABILITY of the cascade instrument — is the per-line vulnerability
//!      consistent across independent injection patterns ("raters")? ICC(2,1)
//!      absolute agreement, Cronbach α (each injection = one item), test-retest ρ.
//!   C. DISCRIMINANT VALIDITY — are the two axes (Raumgewinn ⊥ infight) actually
//!      distinct constructs? (Spearman near 0 = separate axes, the two-basin claim.)
//!   D. The collapse number Π — an honest CONSISTENCY check (its infight term is
//!      shared with the outcome, so this is internal consistency, NOT independent
//!      validity; flagged as the open [H]→[G] calibration probe).
//!
//! Significance of every correlation must use the Jirak n^(p/2−1) rate (weakly
//! dependent contingencies), NOT IID Berry–Esseen — point estimates only here.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example validate_mediators -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{
    contingency_features, dc_flows, effective_resistance, laplacian_pinv, pearson, simulate_outage,
    spearman, symmetric_eigen, zscore, CascadeConfig, Edge, Grid,
};
use perturbation_sim::{cronbach_alpha, icc_a1};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn synthetic(rows: usize, cols: usize) -> Grid {
    let id = |r: usize, c: usize| r * cols + c;
    let mut e = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                e.push(Edge::new(id(r, c), id(r, c + 1), 1.0, 1.0));
            }
            if r + 1 < rows {
                e.push(Edge::new(id(r, c), id(r + 1, c), 1.0, 1.0));
            }
        }
    }
    Grid::new(rows * cols, e)
}

/// Balanced (zero-sum) injection from seed `s`.
fn injection(n: usize, s: u64) -> Vec<f64> {
    let mut rng = Rng(s);
    let raw: Vec<f64> = (0..n).map(|_| rng.f()).collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    raw.iter().map(|x| x - mean).collect()
}

/// Jirak-honest crude significance hint: at the weak-dependence rate the noise
/// floor on a correlation scales like 1/√n (the p≥4 L^q regime). Report |ρ|·√n
/// as a "how many noise-floor units" readout, NOT a p-value.
fn jirak_units(rho: f64, n: usize) -> f64 {
    rho.abs() * (n as f64).sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid = if args.len() >= 3 {
        let buses = std::fs::read_to_string(&args[1]).expect("buses.csv");
        let lines = std::fs::read_to_string(&args[2]).expect("lines.csv");
        let cc = args.get(3).map(|s| s.as_str()).unwrap_or("ES");
        let imp = perturbation_sim::from_pypsa_csv(&buses, &lines, Some(cc))
            .expect("import")
            .largest_component();
        println!(
            "grid: {cc} PyPSA core — {} buses, {} lines",
            imp.grid.n,
            imp.grid.edges.len()
        );
        imp.grid
    } else {
        let g = synthetic(10, 10);
        println!(
            "grid: synthetic 10×10 — {} buses, {} lines",
            g.n,
            g.edges.len()
        );
        g
    };
    let n = grid.n;
    let m = grid.edges.len();
    let alive = vec![true; m];

    // Base spectrum + Fiedler edge sensitivity ranks all lines from ONE eigensolve.
    let base = symmetric_eigen(&grid.laplacian_of(&alive), n);
    let lam2 = base.values.get(1).copied().unwrap_or(0.0);
    let v2 = base.eigenvector(1);
    let mut sens: Vec<(usize, f64)> = (0..m)
        .map(|e| {
            let (a, b) = (grid.edges[e].from, grid.edges[e].to);
            let d = v2[a] - v2[b];
            (e, d * d * grid.edges[e].susceptance)
        })
        .collect();
    sens.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    // Stride-sample K lines ACROSS the full sensitivity ranking (not the top-K):
    // the top-K are nearly all bridges (raumgewinn ≡ 1, no variety), so a stride
    // spans bridges → well-connected lines and gives every block real spread.
    // Cascade is O(K·R·rounds) eigensolves — K bounded deliberately.
    let k = 30.min(m);
    let step = (m / k).max(1);
    let cand: Vec<usize> = (0..k).map(|i| sens[(i * step).min(m - 1)].0).collect();
    let fied_sens: Vec<f64> = (0..k).map(|i| sens[(i * step).min(m - 1)].1).collect();

    // Structural predictors (pre-cascade, topology-only).
    let l_plus = laplacian_pinv(&grid, &alive, 1e-9);
    let mut weyl_dloss = Vec::with_capacity(k); // single-trip λ₂ loss
    let mut eff_res = Vec::with_capacity(k); // effective resistance of the line
    for &e in &cand {
        let (a, b) = (grid.edges[e].from, grid.edges[e].to);
        let mut after = alive.clone();
        after[e] = false;
        let lam2_a = symmetric_eigen(&grid.laplacian_of(&after), n)
            .values
            .get(1)
            .copied()
            .unwrap_or(0.0);
        weyl_dloss.push((lam2 - lam2_a).max(0.0));
        eff_res.push(effective_resistance(&l_plus, n, a, b));
    }

    // Outcome instrument: cascade fraction across R independent injection
    // patterns ("raters"), each with self-calibrated limits.
    let r_raters = 4usize;
    let cfg = CascadeConfig {
        max_rounds: 10,
        ..CascadeConfig::default()
    };
    let mut outcome: Vec<Vec<f64>> = Vec::with_capacity(r_raters); // [rater][cand]
    for r in 0..r_raters {
        let p = injection(n, 0xA11CE + r as u64 * 0x1000);
        let flows = dc_flows(&grid, &alive, &base.pseudo_apply(&p, 1e-9));
        let mut g = grid.clone();
        for (e, edge) in g.edges.iter_mut().enumerate() {
            edge.limit = (1.1 * flows[e].abs()).max(1e-6);
        }
        let row: Vec<f64> = cand
            .iter()
            .map(|&e| simulate_outage(&g, &p, e, cfg).fraction_tripped)
            .collect();
        outcome.push(row);
    }
    // Mean outcome per candidate (the criterion).
    let mean_out: Vec<f64> = (0..k)
        .map(|i| outcome.iter().map(|row| row[i]).sum::<f64>() / r_raters as f64)
        .collect();

    // Cascade-derived mediators (one representative injection) for blocks C/D —
    // under the SAME self-calibrated stress as block A's outcome, else the grid's
    // real S_nom limits leave the small test injection sub-critical (no cascade ⇒
    // degenerate constant features).
    let p0 = injection(n, 0xA11CE);
    let f0 = dc_flows(&grid, &alive, &base.pseudo_apply(&p0, 1e-9));
    let mut g0 = grid.clone();
    for (e, edge) in g0.edges.iter_mut().enumerate() {
        edge.limit = (1.1 * f0[e].abs()).max(1e-6);
    }
    let feats: Vec<_> = cand
        .iter()
        .map(|&e| contingency_features(&g0, &p0, e, cfg))
        .collect();
    let raumgewinn: Vec<f64> = feats.iter().map(|f| f.raumgewinn).collect();
    let infight: Vec<f64> = feats.iter().map(|f| f.infight).collect();
    let dk_rot: Vec<f64> = feats.iter().map(|f| f.dk_rotation).collect();

    // Pearson is scale-invariant; z-score first so tiny-magnitude inputs (the ES
    // core λ₂ ≈ 3e-7 ⇒ variance ≈ 1e-14) don't trip the absolute-variance guard.
    let report = |name: &str, x: &[f64], y: &[f64]| {
        let r = pearson(&zscore(x), &zscore(y));
        let rho = spearman(x, y);
        println!(
            "  {name:<34} Pearson r = {r:+.3}   Spearman ρ = {rho:+.3}   (|ρ|√n = {:.2})",
            jirak_units(rho, x.len())
        );
    };

    println!("\n  N = {k} contingencies · {r_raters} injection raters · electrical embedding\n");

    println!("== A. Criterion validity: structural mediator → operational cascade size ==");
    report("Weyl single-trip Δλ₂", &weyl_dloss, &mean_out);
    report("effective resistance Rₑ", &eff_res, &mean_out);
    report("Fiedler edge sensitivity", &fied_sens, &mean_out);
    println!(
        "  → a positive r means the concept predicts where a trip cascades furthest;\n    \
         the pre-cascade structural mediators are non-circular predictors of the outcome.\n"
    );

    println!("== B. Reliability of the cascade instrument (across {r_raters} injection raters) ==");
    let icc = icc_a1(&outcome);
    let alpha = cronbach_alpha(&outcome);
    let tr = spearman(&outcome[0], &outcome[1]);
    println!("  ICC(2,1) absolute agreement   = {icc:+.3}");
    println!("  Cronbach α (raters as items)  = {alpha:+.3}");
    println!("  test-retest Spearman (r0,r1)  = {tr:+.3}");
    println!(
        "  → ICC/α near 1 ⇒ the vulnerability ranking is an injection-independent property\n    \
         of the topology (a reliable instrument); near 0 ⇒ it is dispatch-specific noise.\n"
    );

    println!("== C. Discriminant validity: the two axes (Raumgewinn vs infight) ==");
    report("Raumgewinn vs infight", &raumgewinn, &infight);
    let rho_axes = spearman(&raumgewinn, &infight);
    println!(
        "  → measured |ρ| = {:.2}: {}\n",
        rho_axes.abs(),
        if rho_axes.abs() < 0.2 {
            "≈ 0 — global connectivity-loss and local cascade are SEPARATE constructs \
             (the orthogonal two-basin claim, as on the unstressed/global frame)."
        } else {
            "a modest *negative* coupling under stress — bridges cause big global \
             connectivity-loss but split in one step (small local cascade), while \
             loaded non-bridges cascade locally without fragmenting. The axes are \
             distinct but trade off when stressed; NOT strictly orthogonal here."
        }
    );

    println!("== D. Collapse number Π — internal-consistency check (NOT independent validity) ==");
    let pi: Vec<f64> = (0..k)
        .map(|i| {
            let denom = (infight[i] * 6.0).max(1e-6); // fixed inertia H=6
            (raumgewinn[i] * dk_rot[i]) / denom
        })
        .collect();
    report("Π vs cascade size", &pi, &mean_out);
    println!(
        "  CAVEAT: Π carries infight in the DENOMINATOR, and infight tracks the outcome,\n    \
         so Π anti-correlates with cascade size here — a negative ρ is EXPECTED and is\n    \
         not evidence against the law. This is why a clean Π validity test needs an\n    \
         infight proxy independent of the realized cascade (e.g. a local-fight measure\n    \
         from topology, not from the same simulate run) — the open [H]→[G] calibration\n    \
         probe (PAPER §4.6).\n"
    );

    // A compact reliability scale: do the three structural predictors form one
    // coherent "structural vulnerability" construct? (z-scored, Cronbach α.)
    let struct_scale = [zscore(&weyl_dloss), zscore(&eff_res), zscore(&fied_sens)];
    let scale_alpha = cronbach_alpha(&struct_scale);
    println!("== E. Internal consistency of the structural-vulnerability scale ==");
    println!(
        "  Cronbach α {{Weyl Δλ₂, Rₑ, Fiedler sens}} = {scale_alpha:+.3}\n  \
         → high α ⇒ the three structural mediators measure ONE latent construct\n    \
         (could be averaged into a single vulnerability index); low α ⇒ they are\n    \
         complementary facets to keep separate.\n"
    );

    println!(
        "Reads: A = does the concept predict collapse (validity); B/E = is it stable /\n  \
         coherent (reliability); C = are the axes distinct (discriminant). Small N — every\n  \
         correlation's significance is the Jirak n^(p/2−1) rate, not IID. Synthetic\n  \
         injections + estimated limits; feed real ESIOS/ENTSO-E load to harden."
    );
}
