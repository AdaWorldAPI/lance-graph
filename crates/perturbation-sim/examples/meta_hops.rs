//! Meta-hop cascade: the 4 HHTL tiers as 4 hops, with inertia (the clock) and
//! phase (the bipolar sign) propagating *between levels*.
//!
//! Simplification of the N-line cascade: tier `i` MODIFIES tier `i+1` (L2 is the
//! second hop after L1). Two things cross each tier→tier boundary. **Magnitude**:
//! pass-through gain `gᵢ = infightᵢ·(1−raumgewinnᵢ)` (`meta_cascade`); a tier with
//! strong local fight + weak field passes it on. **Phase** (±1): the bipolar
//! Walsh sign; the realized field is the *bundle* (running sum) of signed
//! contributions, so aligned phases reinforce (deep) and alternating phases
//! cancel (self-arrest) — `meta_cascade_phase`. **Inertia** sets each hop's clock
//! `dtᵢ` (swing-equation RoCoF), so the cumulative time at the penetration depth
//! is the event wall-clock — the 27 s.
//!
//! Per-tier residents are read off the real grid (HH = Raumgewinn = basin λ₂;
//! TL = infight = basin cascade fraction), the phase from the *sign* of the
//! tier-to-tier Raumgewinn change (a structural phase: does connectivity rise or
//! fall as we descend?), inertia from a coarse→fine ramp (coarse tiers are
//! synchronous-machine heavy = high H; leaf tiers renewable-rich = low H).
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example meta_hops -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{
    cheeger_sweep, dc_flows, mechanism_from_timescale, meta_cascade, meta_cascade_phase,
    simulate_outage, symmetric_eigen, CascadeConfig, Edge, Grid,
};
use std::collections::HashMap;

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

fn induced(grid: &Grid, members: &[usize]) -> Grid {
    let mut remap = HashMap::new();
    for (i, &m) in members.iter().enumerate() {
        remap.insert(m, i);
    }
    let edges = grid
        .edges
        .iter()
        .filter_map(|e| match (remap.get(&e.from), remap.get(&e.to)) {
            (Some(&a), Some(&b)) => Some(Edge::new(a, b, e.susceptance, e.limit)),
            _ => None,
        })
        .collect();
    Grid::new(members.len(), edges)
}

fn bisect(grid: &Grid, members: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
    if members.len() < 4 {
        return None;
    }
    let sub = induced(grid, members);
    let c = cheeger_sweep(&sub, &vec![true; sub.edges.len()]);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, &m) in members.iter().enumerate() {
        if c.partition[i] {
            a.push(m);
        } else {
            b.push(m);
        }
    }
    if a.is_empty() || b.is_empty() {
        None
    } else {
        Some((a, b))
    }
}

/// Median of per-basin values at one tier.
fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
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
        let g = synthetic(8, 8);
        println!("grid: synthetic 8×8 — {} buses", g.n);
        g
    };

    // Build the 4-tier HHTL tree by recursive spectral bisection.
    let mut levels: Vec<Vec<Vec<usize>>> = vec![vec![(0..grid.n).collect()]];
    for _ in 1..4 {
        let mut next = Vec::new();
        for basin in levels.last().unwrap() {
            match bisect(&grid, basin) {
                Some((a, b)) => {
                    next.push(a);
                    next.push(b);
                }
                None => next.push(basin.clone()),
            }
        }
        levels.push(next);
    }

    // Per-tier residents: HH = Raumgewinn (median basin λ₂), TL = infight (median
    // basin cascade fraction under a self-calibrated, most-loaded-line trip).
    let cfg = CascadeConfig {
        max_rounds: 12,
        ..CascadeConfig::default()
    };
    let mut raum = Vec::new();
    let mut fight = Vec::new();
    for level in &levels {
        let (mut lam2s, mut infs) = (Vec::new(), Vec::new());
        for basin in level {
            if basin.len() < 4 {
                continue;
            }
            let mut sub = induced(&grid, basin);
            if sub.edges.len() < 3 {
                continue;
            }
            let alive = vec![true; sub.edges.len()];
            let eig = symmetric_eigen(&sub.laplacian_of(&alive), sub.n);
            lam2s.push(eig.values.get(1).copied().unwrap_or(0.0));

            let mut rng = Rng(0x1234 + basin.len() as u64);
            let raw: Vec<f64> = (0..sub.n).map(|_| rng.f()).collect();
            let mean = raw.iter().sum::<f64>() / sub.n as f64;
            let p: Vec<f64> = raw.iter().map(|x| x - mean).collect();
            let base = dc_flows(&sub, &alive, &eig.pseudo_apply(&p, 1e-9));
            for (e, edge) in sub.edges.iter_mut().enumerate() {
                edge.limit = (1.1 * base[e].abs()).max(1e-6);
            }
            let seed = base
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.abs().partial_cmp(&y.1.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            infs.push(simulate_outage(&sub, &p, seed, cfg).fraction_tripped);
        }
        raum.push(median(lam2s));
        fight.push(median(infs));
    }

    // Phase = sign of the tier-to-tier Raumgewinn change (structural phase: does
    // the field tighten or loosen as we descend a level?). Seed phase +1.
    let mut phase = vec![1i8; raum.len()];
    for i in 1..raum.len() {
        phase[i] = if raum[i] >= raum[i - 1] { 1 } else { -1 };
    }
    // Inertia ramp coarse→fine: HEEL synchronous-heavy (H=6) → LEAF renewable
    // (H=2). The leaves are where low inertia speeds the clock.
    let inertia = [6.0, 4.5, 3.0, 2.0];

    let tiers = ["HEEL", "HIP ", "TWIG", "LEAF"];
    println!("\n== Per-tier residents (read off the real grid) ==");
    println!(
        "  {:<5} {:>12} {:>12} {:>7} {:>9}",
        "tier", "Raumgewinn λ₂", "infight", "phase", "inertia H"
    );
    for i in 0..raum.len() {
        println!(
            "  {:<5} {:>12.3e} {:>12.3} {:>7} {:>9.1}",
            tiers.get(i).unwrap_or(&"?"),
            raum[i],
            fight[i],
            if phase[i] < 0 { "−" } else { "+" },
            inertia.get(i).copied().unwrap_or(1.0),
        );
    }

    // 1. Plain meta-hop cascade (magnitude only).
    let (amps, hops) = meta_cascade(&raum, &fight, 0.25);
    println!("\n== 1. Meta-hop cascade (magnitude only) ==");
    print!("  amplitude per tier: ");
    for a in &amps {
        print!("{a:.3} ");
    }
    println!("\n  meta_hops (amp ≥ 0.25): {hops} of {} tiers", raum.len());

    // 2. Inertia × phase between-level cascade.
    let (trace, depth) = meta_cascade_phase(&raum, &fight, &phase, &inertia, 0.1, 0.2, 0.2, 0.25);
    println!("\n== 2. Inertia × phase between-level cascade ==");
    println!(
        "  {:<5} {:>11} {:>11} {:>8} {:>9}",
        "tier", "signed_amp", "field", "dt (s)", "t (s)"
    );
    for hp in &trace {
        println!(
            "  {:<5} {:>+11.3} {:>+11.3} {:>8.2} {:>9.2}",
            tiers.get(hp.tier).unwrap_or(&"?"),
            hp.signed_amp,
            hp.field,
            hp.dt,
            hp.t
        );
    }
    let total = trace.last().map(|h| h.t).unwrap_or(0.0);
    let field_peak = trace.iter().map(|h| h.field.abs()).fold(0.0, f64::max);
    println!(
        "\n  front penetration (arriving |signed_amp| ≥ 0.25): {depth} tiers\n  \
         interference field peak |Σ|: {field_peak:.3}  (phase-governed: grows if aligned)\n  \
         cumulative wall-clock: {total:.1} s  →  {}",
        mechanism_from_timescale(total)
    );

    println!(
        "\nReads: tier i modifies tier i+1 (one meta-hop). Magnitude passes when\n  \
         infight is high and the field (Raumgewinn) is weak; phase decides whether\n  \
         the between-level contributions bundle constructively (deep cascade) or\n  \
         cancel (self-arrest). Inertia is the clock: the coarse→fine H-ramp puts the\n  \
         fast seconds in the leaf tiers — the 27 s electromechanical tell. CONJECTURE\n  \
         [H]; calibrate phase + inertia against an observed multi-tier cascade before [G]."
    );
}
