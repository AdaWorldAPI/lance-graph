//! D-MOORE-PERT-AB-0: duplicate the single-line perturbation arm and compare
//! the canonical Moore-neighbour execution against the full spectral solution.
//!
//! A: independent full Laplacian eigensolve, L^+ p (existing reference arm).
//! B: independent, deterministic 8-lane Moore stencil Jacobi relaxation.
//! Both consume the same grid, injections, single-line outage and node IDs.
//! A is not consulted to initialize or update B. No CE64/OGAR field is mutated.
//!
//! This is a CONTROLLED 2-D LATTICE probe, not a generic graph approximation.
//! On this fixture every edge is a single Moore step, so B solves exactly A's
//! linear system: the A-B gap is Jacobi iteration budget, not a modelling
//! error, and full relaxation converges to A. What the sweep table measures is
//! how fast each observable converges. Measured: signed flow shift converges
//! well ahead of signed phase (relL2 0.16 vs 0.65 at 16 sweeps), because flows
//! are neighbour differences and cancel the smooth error Jacobi removes last.
//! A Moore-model fidelity question needs a graph with non-Moore edges.
//! A general electrical graph cannot always fit in eight Moore directions.
//! Here phase means signed electrical angle perturbation relative to bus 0,
//! NOT NARS frequency/confidence, and NOT circular phase modulo 2*pi.
//! Magnitude = absolute angle perturbation; signed edge flow is another output.
//!
//! Run:
//! cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!   --features moore-probe --example MoorePertubationsfeld
//! cargo test --manifest-path crates/perturbation-sim/Cargo.toml \
//!   --features moore-probe --example MoorePertubationsfeld
//!
//! Pearson, Spearman and ICC(2,1) are point estimates from perturbation-sim's
//! pre-existing statistics battery. Bootstrap ranges resample OUTAGES, not
//! individual nodes, and are descriptive only: outages share a lattice.
//! Neither a causal validation nor an IID 95% statistical confidence interval.

use lance_graph_contract::moore_tenant::MooreSlot;
use perturbation_sim::{dc_flows, icc_a1, pearson, spearman, symmetric_eigen, Edge, Grid};

const SIDE: usize = 4;
const N: usize = SIDE * SIDE;
const EIGEN_TOL: f64 = 1e-10;
const SWEEPS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 256];

#[derive(Debug, Clone, Copy)]
struct Coupling {
    neighbor: usize,
    weight: f64,
}

// The index into each row is *always* MooreSlot.index(), never "nearest kth".
type Stencil = Vec<[Option<Coupling>; 8]>;

fn fixture() -> (Grid, Vec<f64>) {
    let id = |r: usize, c: usize| r * SIDE + c;
    let mut edges = Vec::new();
    for r in 0..SIDE {
        for c in 0..SIDE {
            if c + 1 < SIDE {
                edges.push(Edge::new(id(r, c), id(r, c + 1), 1.0 + 0.1 * r as f64, 1e9));
            }
            if r + 1 < SIDE {
                edges.push(Edge::new(
                    id(r, c),
                    id(r + 1, c),
                    0.8 + 0.09 * c as f64,
                    1e9,
                ));
            }
            if r + 1 < SIDE && c + 1 < SIDE {
                edges.push(Edge::new(id(r, c), id(r + 1, c + 1), 0.31, 1e9));
            }
            if r + 1 < SIDE && c > 0 {
                edges.push(Edge::new(id(r, c), id(r + 1, c - 1), 0.23, 1e9));
            }
        }
    }
    let mut injection: Vec<f64> = (0..N)
        .map(|i| ((i * 7 + 3) % 13) as f64 / 4.0 - 1.5)
        .collect();
    let mean = injection.iter().sum::<f64>() / N as f64;
    for x in &mut injection {
        *x -= mean;
    }
    (Grid::new(N, edges), injection)
}

fn slot_for(from: usize, to: usize) -> MooreSlot {
    let dx = to as i32 % SIDE as i32 - from as i32 % SIDE as i32;
    let dy = to as i32 / SIDE as i32 - from as i32 / SIDE as i32;
    MooreSlot::ALL
        .into_iter()
        .find(|slot| {
            let (sx, sy) = slot.offset();
            dx == i32::from(sx) && dy == i32::from(sy)
        })
        .expect("fixture edge must be one Moore step")
}

fn stencil(grid: &Grid, alive: &[bool]) -> Stencil {
    assert_eq!(grid.n, N);
    assert_eq!(grid.edges.len(), alive.len());
    let mut out = vec![[None; 8]; N];
    for (i, edge) in grid.edges.iter().enumerate() {
        if !alive[i] {
            continue;
        }
        for (from, to) in [(edge.from, edge.to), (edge.to, edge.from)] {
            let slot = slot_for(from, to).index();
            assert!(
                out[from][slot]
                    .replace(Coupling {
                        neighbor: to,
                        weight: edge.susceptance
                    })
                    .is_none(),
                "one physical neighbor per Moore direction"
            );
        }
    }
    out
}

// Fix the Laplacian gauge by pinning the reference node (0) to zero.
// Solves the same balanced DC system as L^+ p, by iterative local relaxation.
// Uses ONLY the fixed Moore slots and input injections, never A's angles.
fn moore_solve(lanes: &Stencil, injection: &[f64], sweeps: usize) -> Vec<f64> {
    assert_eq!(lanes.len(), N);
    assert_eq!(injection.len(), N);
    let mut theta = vec![0.0f64; N];
    let mut next = vec![0.0f64; N];
    for _ in 0..sweeps {
        next[0] = 0.0;
        for i in 1..N {
            let mut diag = 0.0;
            let mut weighted = 0.0;
            for coupling in lanes[i].iter().flatten() {
                diag += coupling.weight;
                weighted += coupling.weight * theta[coupling.neighbor];
            }
            assert!(diag > 0.0, "all non-reference buses must stay connected");
            next[i] = (injection[i] + weighted) / diag;
        }
        std::mem::swap(&mut theta, &mut next);
    }
    theta
}

fn spectral_solve(grid: &Grid, alive: &[bool], injection: &[f64]) -> Vec<f64> {
    let eig = symmetric_eigen(&grid.laplacian_of(alive), grid.n);
    assert_eq!(eig.nullity(EIGEN_TOL), 1, "outage must not island fixture");
    let mut theta = eig.pseudo_apply(injection, EIGEN_TOL);
    let gauge = theta[0];
    for x in &mut theta {
        *x -= gauge;
    }
    theta
}

#[derive(Debug, Clone)]
struct Field {
    signed_phase: Vec<f64>,
    magnitude: Vec<f64>,
    signed_flow_shift: Vec<f64>,
}

// Compare on matching identities (bus index and line index) and a shared gauge.
// Preserve the signs BEFORE taking absolute magnitudes.
fn field(
    grid: &Grid,
    all_alive: &[bool],
    post_alive: &[bool],
    before: &[f64],
    after: &[f64],
) -> Field {
    let signed_phase: Vec<f64> = before.iter().zip(after).map(|(a, b)| b - a).collect();
    let magnitude = signed_phase.iter().map(|x| x.abs()).collect();
    let f_before = dc_flows(grid, all_alive, before);
    let f_after = dc_flows(grid, post_alive, after);
    let signed_flow_shift = f_after
        .iter()
        .zip(&f_before)
        .map(|(new, old)| new - old)
        .collect();
    Field {
        signed_phase,
        magnitude,
        signed_flow_shift,
    }
}

#[derive(Debug)]
struct Case {
    alive: Vec<bool>,
    reference: Field,
    moore: Stencil,
}

fn cases(grid: &Grid, injection: &[f64], all_alive: &[bool]) -> Vec<Case> {
    let before = spectral_solve(grid, all_alive, injection);
    let mut cases = Vec::with_capacity(grid.edges.len());
    for line in 0..grid.edges.len() {
        let mut alive = all_alive.to_vec();
        alive[line] = false;
        let after = spectral_solve(grid, &alive, injection);
        cases.push(Case {
            reference: field(grid, all_alive, &alive, &before, &after),
            moore: stencil(grid, &alive),
            alive,
        });
    }
    cases
}

fn rel_l2(reference: &[f64], estimate: &[f64]) -> f64 {
    let error2: f64 = reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let norm2: f64 = reference.iter().map(|v| v * v).sum();
    (error2 / norm2.max(1e-24)).sqrt()
}

fn mae(reference: &[f64], estimate: &[f64]) -> f64 {
    reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / reference.len() as f64
}

// Confidence-like ranges are deliberately called bootstrap *ranges*: line-trip
// cases on one lattice are dependent and are not an IID population sample.
fn outage_resampling_range(case_mae: &[f64]) -> (f64, f64) {
    let n = case_mae.len();
    assert!(n > 0);
    let mut seed = 0xB0A7_2026_A11C_E123u64;
    let mut means = Vec::with_capacity(400);
    for _ in 0..400 {
        let mut sum = 0.0;
        for _ in 0..n {
            // splitmix64, fixed seed for exact report replay.
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = seed;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            sum += case_mae[(x as usize) % n];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(f64::total_cmp);
    // 400 sorted draws: the 2.5th and 97.5th percentiles are draws 10 and 390.
    (means[10], means[390])
}

fn print_scores(label: &str, truth: &[f64], estimate: &[f64]) {
    assert_eq!(truth.len(), estimate.len());
    let r = pearson(truth, estimate);
    let rho = spearman(truth, estimate);
    let icc = icc_a1(&[truth.to_vec(), estimate.to_vec()]);
    println!(
        "  {label:<13} n={:<5} Pearson r={r:+.5}  Spearman rho={rho:+.5}  ICC(2,1)={icc:+.5}  relL2={:.5}  MAE={:.6}",
        truth.len(),
        rel_l2(truth, estimate),
        mae(truth, estimate)
    );
}

fn evaluate(
    grid: &Grid,
    injection: &[f64],
    sweep_count: usize,
    cases: &[Case],
    pre: &Stencil,
    all_alive: &[bool],
) {
    let before = moore_solve(pre, injection, sweep_count);
    let mut phase_a = Vec::new();
    let mut phase_b = Vec::new();
    let mut magnitude_a = Vec::new();
    let mut magnitude_b = Vec::new();
    let mut flow_a = Vec::new();
    let mut flow_b = Vec::new();
    let mut errors = Vec::with_capacity(cases.len());

    for case in cases {
        let after = moore_solve(&case.moore, injection, sweep_count);
        let estimate = field(grid, all_alive, &case.alive, &before, &after);
        errors.push(mae(&case.reference.signed_phase, &estimate.signed_phase));
        phase_a.extend_from_slice(&case.reference.signed_phase);
        phase_b.extend_from_slice(&estimate.signed_phase);
        magnitude_a.extend_from_slice(&case.reference.magnitude);
        magnitude_b.extend_from_slice(&estimate.magnitude);
        flow_a.extend_from_slice(&case.reference.signed_flow_shift);
        flow_b.extend_from_slice(&estimate.signed_flow_shift);
    }
    let (lo, hi) = outage_resampling_range(&errors);
    println!("\nMoore B sweeps={sweep_count:>3}");
    print_scores("signed phase", &phase_a, &phase_b);
    print_scores("magnitude", &magnitude_a, &magnitude_b);
    print_scores("signed flow", &flow_a, &flow_b);
    println!("  per-outage phase MAE resampling range [2.5%,97.5%]: [{lo:.6}, {hi:.6}] (descriptive, dependent cases)");
}

fn main() {
    let (grid, injection) = fixture();
    let all_alive = vec![true; grid.edges.len()];
    let pre = stencil(&grid, &all_alive);
    let all_cases = cases(&grid, &injection, &all_alive);
    println!(
        "D-MOORE-PERT-AB-0 | {} buses, {} physical edges, {} single-line outages",
        grid.n,
        grid.edges.len(),
        all_cases.len()
    );
    println!(
        "A = full Laplacian spectral re-solve, B = canonical 8-slot Moore relaxation (bus 0 gauge)\nNo timings: this probe measures agreement per sweep budget, not cost; no nonlinear cascade"
    );
    for sweeps in SWEEPS {
        evaluate(&grid, &injection, sweeps, &all_cases, &pre, &all_alive);
    }
    println!("\nInterpretation: magnitude correlations do NOT certify signed-phase or signed-flow validity.");
    println!(
        "All cases are synthetic and share a grid; significance / independent CI not claimed."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moore_slots_are_directional_not_sorted_by_permeability() {
        let (grid, _) = fixture();
        let lanes = stencil(&grid, &vec![true; grid.edges.len()]);
        for (i, row) in lanes.iter().enumerate() {
            for slot in MooreSlot::ALL {
                if let Some(link) = row[slot.index()] {
                    assert_eq!(slot_for(i, link.neighbor), slot);
                    let dx = slot.offset();
                    let reverse = MooreSlot::ALL
                        .into_iter()
                        .find(|s| s.offset() == (-dx.0, -dx.1))
                        .expect("reverse Moore slot");
                    assert_eq!(
                        lanes[link.neighbor][reverse.index()]
                            .expect("reverse")
                            .neighbor,
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn moore_solution_converges_toward_independent_reference() {
        let (grid, injection) = fixture();
        let all_alive = vec![true; grid.edges.len()];
        let before_a = spectral_solve(&grid, &all_alive, &injection);
        let before_b = moore_solve(&stencil(&grid, &all_alive), &injection, 4096);
        assert!(rel_l2(&before_a, &before_b) < 1e-7);

        // A line trip must alter the phase field but never the identity indexing.
        let mut after_alive = all_alive.clone();
        after_alive[grid.edges.len() / 2] = false;
        let after_a = spectral_solve(&grid, &after_alive, &injection);
        let after_b = moore_solve(&stencil(&grid, &after_alive), &injection, 4096);
        assert!(rel_l2(&after_a, &after_b) < 1e-7);
        let f_a = field(&grid, &all_alive, &after_alive, &before_a, &after_a);
        let f_b = field(&grid, &all_alive, &after_alive, &before_b, &after_b);
        assert!(f_a.magnitude.iter().any(|v| *v > 1e-7));
        assert!(rel_l2(&f_a.signed_phase, &f_b.signed_phase) < 1e-5);
        assert!(rel_l2(&f_a.signed_flow_shift, &f_b.signed_flow_shift) < 1e-5);
    }

    // Anti-vacuity for the convergence test above: a short sweep budget must be
    // measurably off, or the 1e-7 bound would hold for any solver output.
    #[test]
    fn a_short_sweep_budget_is_not_converged() {
        let (grid, injection) = fixture();
        let all_alive = vec![true; grid.edges.len()];
        let a = spectral_solve(&grid, &all_alive, &injection);
        let b = moore_solve(&stencil(&grid, &all_alive), &injection, 4);
        assert!(
            rel_l2(&a, &b) > 0.1,
            "4 Jacobi sweeps cannot reach the reference"
        );
    }

    // On this fixture every edge is one Moore step, so B solves A's linear
    // system and the A-B gap is iteration budget only. Flow shifts are
    // neighbour differences and cancel the smooth error Jacobi leaves longest,
    // so they converge faster than phase: pinned at 16 sweeps.
    #[test]
    fn flow_shift_converges_faster_than_phase() {
        let (grid, injection) = fixture();
        let all_alive = vec![true; grid.edges.len()];
        let pre = stencil(&grid, &all_alive);
        let all_cases = cases(&grid, &injection, &all_alive);
        let before = moore_solve(&pre, &injection, 16);
        let (mut pa, mut pb, mut fa, mut fb) = (vec![], vec![], vec![], vec![]);
        for case in &all_cases {
            let after = moore_solve(&case.moore, &injection, 16);
            let est = field(&grid, &all_alive, &case.alive, &before, &after);
            pa.extend_from_slice(&case.reference.signed_phase);
            pb.extend_from_slice(&est.signed_phase);
            fa.extend_from_slice(&case.reference.signed_flow_shift);
            fb.extend_from_slice(&est.signed_flow_shift);
        }
        let (phase, flow) = (rel_l2(&pa, &pb), rel_l2(&fa, &fb));
        assert!(
            flow * 2.0 < phase,
            "flow relL2 {flow} vs phase relL2 {phase}"
        );
    }
}
