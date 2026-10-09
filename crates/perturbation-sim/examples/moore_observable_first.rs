//! **D-MOORE-OBSERVABLE-FIRST-0: can the Moore relaxation make a provably
//! correct decision before its field converges?**
//!
//! ```text
//! cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!   --features moore-probe --example moore_observable_first
//! cargo test --manifest-path crates/perturbation-sim/Cargo.toml \
//!   --features moore-probe --example moore_observable_first
//! ```
//!
//! Starts from #1427 (`MoorePertubationsfeld`), which showed correlation per
//! sweep budget. Correlation says nothing about whether a DECISION read from an
//! unconverged iterate is right. This probe separates three things:
//!
//! 1. **field accuracy**: relative L2 error of the post-outage angle field;
//! 2. **observable accuracy**: error of the post-outage line flows;
//! 3. **decision correctness**: N-1 overload screening (`|f_l| > R_l`), the
//!    sign of each flow shift, and which line moves most.
//!
//! # The certificate
//!
//! Ground bus 0 (`θ_0 = 0`). The reduced Laplacian `A` of a connected grid is a
//! nonsingular M-matrix, so `A⁻¹ ≥ 0` entrywise. If a vector `w ≥ 0` satisfies
//! `A w ≥ 1` (checked by one stencil pass), then `A⁻¹ 1 ≤ w`, and for any
//! iterate `x` with residual `r = p − A x`:
//!
//! ```text
//! |θ_i − x_i| = |(A⁻¹ r)_i| ≤ (A⁻¹ |r|)_i ≤ ‖r‖∞ · w_i
//! |f_l − f̂_l| ≤ b_l (|e_i| + |e_j|) ≤ b_l ‖r‖∞ (w_i + w_j)
//! ```
//!
//! No eigenvalue is needed. `w` is computed once for the intact grid; a line
//! outage changes only two rows of `A w`, so the outage certificate is a
//! two-row recheck. When the recheck fails (the outage removes too much of the
//! margin), the probe falls back to a per-outage solve and counts it.
//!
//! A decision is CERTIFIED only when its interval clears the threshold;
//! otherwise it ABSTAINS. The same predicate read off the point estimate,
//! without the interval, is the NAIVE decision.
//!
//! **Rounding.** Residuals and the `A w ≥ 1` check carry a first-order
//! floating-point margin (`(deg + 3)·ε·Σ|terms|`). This is not interval
//! arithmetic; it is a sound bound up to that first-order model.
//!
//! # Oracles (independent of the Moore arm and of each other)
//!
//! - dense Cholesky of the grounded post-outage Laplacian, per outage;
//! - closed-form LODF from one intact-grid inverse;
//! - the crate's spectral pseudoinverse (`symmetric_eigen`), on a sample.
//!
//! They must agree before any score is printed.
//!
//! # Solvers compared
//!
//! - `jac-cold`: Jacobi from zero (the #1427 setup);
//! - `jac-warm`: Jacobi from the exact intact-grid field;
//! - `pcg-warm`: diagonally preconditioned conjugate gradient from the same
//!   start (not Moore-local: it needs global dot products);
//! - `lodf` and `chol`: exact baselines, timed for the same screening task.
//!
//! Ratings `R_l = max(|f0_l|, 0.2·max|f0|) / 0.8` (80 % intact loading) are a
//! policy pin, not data. The near-tie fixture sets some ratings to the exact
//! post-outage flow on purpose: no sound certificate may decide those.
//!
//! # Measured (board entry `2026-10-09-d-moore-observable-first-0.md`)
//!
//! - No certified decision was wrong and no bound was violated, on any
//!   fixture, arm, budget or decision kind.
//! - The `L1` bound is the one that makes early decisions possible: 1.4–25×
//!   the actual flow error (about 130× on the oscillating bipartite grid),
//!   against 10¹–10⁵× for the field bound, and it needs no certificate
//!   vector. Warm-started, it certifies 78 % of the overload
//!   decisions on `hetero-16x16` after zero sweeps.
//! - What it certifies early is ABSENCE of overload. True overloads need a
//!   nearly converged iterate (`hetero-16x16`: 0 of 225 at sweep 0, 179 after
//!   1024 Jacobi sweeps, all 225 after 256 PCG iterations).
//! - Read off the iterate without a bound, the same decisions miss real
//!   overloads: every one at sweep 0, 11 of 225 still after 1024 sweeps.
//! - For this task the exact LODF baseline is fastest (17 ms for all 930
//!   outages), then certified PCG (~160 ms), then dense Cholesky (~2.1 s) and
//!   certified Jacobi (~2.8 s). Cold start never certifies within 20 000
//!   sweeps on three of five grids.

use lance_graph_contract::moore_tenant::MooreSlot;
use perturbation_sim::{dc_flows, symmetric_eigen, Edge, Grid};
use std::time::Instant;

const EPS: f64 = f64::EPSILON;
const BUDGETS: [usize; 11] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024];
const CERT_CAP: usize = 20_000;
const LOADING: f64 = 0.8;

// ── fixtures ──────────────────────────────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone)]
struct Fixture {
    name: &'static str,
    width: usize,
    grid: Grid,
    p: Vec<f64>,
}

/// A `width × height` lattice. `weight(r, c, r2, c2)` gives the susceptance of
/// the edge between the two cells, or `None` for no edge.
fn lattice(
    name: &'static str,
    width: usize,
    height: usize,
    diagonals: bool,
    seed: u64,
    mut weight: impl FnMut(usize, usize, usize, usize) -> Option<f64>,
) -> Fixture {
    let id = |r: usize, c: usize| r * width + c;
    let mut edges = Vec::new();
    for r in 0..height {
        for c in 0..width {
            let mut push = |r2: usize, c2: usize| {
                if let Some(b) = weight(r, c, r2, c2) {
                    edges.push(Edge::new(id(r, c), id(r2, c2), b, 1e9));
                }
            };
            if c + 1 < width {
                push(r, c + 1);
            }
            if r + 1 < height {
                push(r + 1, c);
            }
            if diagonals && r + 1 < height && c + 1 < width {
                push(r + 1, c + 1);
            }
            if diagonals && r + 1 < height && c > 0 {
                push(r + 1, c - 1);
            }
        }
    }
    let n = width * height;
    let mut rng = Lcg(seed);
    let mut p: Vec<f64> = (0..n).map(|_| 2.0 * rng.f64() - 1.0).collect();
    let mean = p.iter().sum::<f64>() / n as f64;
    for x in &mut p {
        *x -= mean;
    }
    Fixture {
        name,
        width,
        grid: Grid::new(n, edges),
        p,
    }
}

/// The exact #1427 fixture: 4×4, Moore edges, the same weights and injections.
fn fixture_1427() -> Fixture {
    let mut fx = lattice("1427-4x4", 4, 4, true, 0, |r, c, r2, c2| {
        Some(match (r2 - r, c2 as i64 - c as i64) {
            (0, 1) => 1.0 + 0.1 * r as f64,
            (1, 0) => 0.8 + 0.09 * c as f64,
            (1, 1) => 0.31,
            _ => 0.23,
        })
    });
    let n = 16;
    let mut p: Vec<f64> = (0..n)
        .map(|i| ((i * 7 + 3) % 13) as f64 / 4.0 - 1.5)
        .collect();
    let mean = p.iter().sum::<f64>() / n as f64;
    for x in &mut p {
        *x -= mean;
    }
    fx.p = p;
    fx
}

fn hetero(name: &'static str, side: usize, seed: u64) -> Fixture {
    let mut rng = Lcg(seed ^ 0x5EED);
    lattice(name, side, side, true, seed, move |_, _, _, _| {
        Some(10f64.powf(4.0 * rng.f64() - 2.0))
    })
}

/// Two blocks joined by exactly two weak horizontal ties at the top and
/// bottom rows. Tripping one tie leaves a single weak path: near-islanding.
fn weak_tie(name: &'static str, width: usize, height: usize, seed: u64) -> Fixture {
    let cut = width / 2;
    lattice(name, width, height, true, seed, move |r, c, r2, c2| {
        let crosses = (c < cut) != (c2 < cut);
        if !crosses {
            Some(1.0)
        } else if r == r2 && (r == 0 || r == height - 1) {
            Some(0.05)
        } else {
            None
        }
    })
}

fn uniform(name: &'static str, width: usize, height: usize, diagonals: bool, seed: u64) -> Fixture {
    lattice(name, width, height, diagonals, seed, |_, _, _, _| Some(1.0))
}

// ── dense oracle: grounded Cholesky ───────────────────────────────────────

/// Grounded Laplacian (bus 0 removed), dense `m × m`, `m = n − 1`.
fn grounded(grid: &Grid, alive: &[bool]) -> Vec<f64> {
    let m = grid.n - 1;
    let mut a = vec![0.0; m * m];
    for (e, edge) in grid.edges.iter().enumerate() {
        if !alive[e] {
            continue;
        }
        let (u, v, b) = (edge.from, edge.to, edge.susceptance);
        for (x, y) in [(u, v), (v, u)] {
            if x > 0 {
                a[(x - 1) * m + (x - 1)] += b;
                if y > 0 {
                    a[(x - 1) * m + (y - 1)] -= b;
                }
            }
        }
    }
    a
}

/// In-place lower Cholesky. `false` if not positive definite (islanded).
fn cholesky(a: &mut [f64], m: usize) -> bool {
    for j in 0..m {
        let mut d = a[j * m + j];
        for k in 0..j {
            d -= a[j * m + k] * a[j * m + k];
        }
        if d <= 1e-14 * a[j * m + j].abs().max(1e-300) {
            return false;
        }
        let d = d.sqrt();
        a[j * m + j] = d;
        for i in j + 1..m {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = s / d;
        }
    }
    true
}

fn chol_solve(l: &[f64], m: usize, b: &mut [f64]) {
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * m + k] * b[k];
        }
        b[i] = s / l[i * m + i];
    }
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s -= l[k * m + i] * b[k];
        }
        b[i] = s / l[i * m + i];
    }
}

/// Solve the grounded system for a full-length right-hand side (entry 0 is
/// ignored); returns the full-length solution with `x[0] = 0`.
fn chol_full(grid: &Grid, alive: &[bool], rhs: &[f64]) -> Option<Vec<f64>> {
    let m = grid.n - 1;
    let mut a = grounded(grid, alive);
    if !cholesky(&mut a, m) {
        return None;
    }
    let mut b = rhs[1..].to_vec();
    chol_solve(&a, m, &mut b);
    let mut x = vec![0.0];
    x.extend(b);
    Some(x)
}

// ── Moore stencil and the iterative arms ──────────────────────────────────

type Stencil = Vec<[Option<(usize, f64)>; 8]>;

fn slot_for(width: usize, from: usize, to: usize) -> MooreSlot {
    let dx = (to % width) as i64 - (from % width) as i64;
    let dy = (to / width) as i64 - (from / width) as i64;
    MooreSlot::ALL
        .into_iter()
        .find(|s| {
            let (sx, sy) = s.offset();
            dx == i64::from(sx) && dy == i64::from(sy)
        })
        .expect("fixture edge must be one Moore step")
}

fn stencil(fx: &Fixture, alive: &[bool]) -> Stencil {
    let mut out = vec![[None; 8]; fx.grid.n];
    for (e, edge) in fx.grid.edges.iter().enumerate() {
        if !alive[e] {
            continue;
        }
        for (from, to) in [(edge.from, edge.to), (edge.to, edge.from)] {
            let slot = slot_for(fx.width, from, to).index();
            assert!(
                out[from][slot].replace((to, edge.susceptance)).is_none(),
                "one physical neighbour per Moore direction"
            );
        }
    }
    out
}

/// `(A x)_i` and the sum of absolute terms (for the rounding margin), `i ≥ 1`.
fn row_apply(st: &Stencil, x: &[f64], i: usize) -> (f64, f64, usize) {
    let (mut v, mut mag, mut deg) = (0.0, 0.0, 0usize);
    for &(j, b) in st[i].iter().flatten() {
        v += b * (x[i] - x[j]);
        mag += b * (x[i].abs() + x[j].abs());
        deg += 1;
    }
    (v, mag, deg)
}

/// Sound upper bounds on `‖p − A x‖∞` and `‖p − A x‖₁`, each with a
/// first-order rounding margin per row.
#[derive(Clone, Copy, Debug, Default)]
struct Resid {
    inf: f64,
    l1: f64,
}

impl Resid {
    fn add_row(&mut self, r: f64, margin: f64) {
        self.inf = self.inf.max(r.abs() + margin);
        self.l1 += r.abs() + margin;
    }
}

fn residual_bound(st: &Stencil, p: &[f64], x: &[f64]) -> Resid {
    let mut out = Resid::default();
    for (i, &pi) in p.iter().enumerate().skip(1) {
        let (ax, mag, deg) = row_apply(st, x, i);
        out.add_row(pi - ax, (deg as f64 + 3.0) * EPS * (mag + pi.abs()));
    }
    out
}

/// One Jacobi sweep `x → out` on the grounded system (`out[0] = 0`).
/// Returns the sound residual bound of the INPUT `x`, which the sweep computes
/// for free: `r_i = p_i + Σ b x_j − d x_i`.
fn jacobi_sweep(st: &Stencil, p: &[f64], x: &[f64], out: &mut [f64]) -> Resid {
    out[0] = 0.0;
    let mut rb = Resid::default();
    for i in 1..x.len() {
        let (mut s, mut d, mut mag, mut deg) = (0.0, 0.0, 0.0, 0usize);
        for &(j, b) in st[i].iter().flatten() {
            s += b * x[j];
            d += b;
            mag += b * x[j].abs();
            deg += 1;
        }
        let r = p[i] + s - d * x[i];
        let margin = (deg as f64 + 3.0) * EPS * (p[i].abs() + mag + d * x[i].abs());
        rb.add_row(r, margin);
        out[i] = (p[i] + s) / d;
    }
    rb
}

fn apply(st: &Stencil, x: &[f64], out: &mut [f64]) {
    out[0] = 0.0;
    for (i, o) in out.iter_mut().enumerate().skip(1) {
        *o = row_apply(st, x, i).0;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a[1..].iter().zip(&b[1..]).map(|(x, y)| x * y).sum()
}

/// Diagonally preconditioned CG, state carried between `step` calls.
struct Pcg {
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    d: Vec<f64>,
    q: Vec<f64>,
    diag: Vec<f64>,
    rz: f64,
}

impl Pcg {
    fn new(st: &Stencil, p: &[f64], x0: &[f64]) -> Self {
        let n = x0.len();
        let diag: Vec<f64> = (0..n)
            .map(|i| st[i].iter().flatten().map(|&(_, b)| b).sum::<f64>())
            .collect();
        let mut ax = vec![0.0; n];
        apply(st, x0, &mut ax);
        let mut r: Vec<f64> = (0..n).map(|i| p[i] - ax[i]).collect();
        r[0] = 0.0;
        let mut z = vec![0.0; n];
        for i in 1..n {
            z[i] = r[i] / diag[i];
        }
        let rz = dot(&r, &z);
        Pcg {
            x: x0.to_vec(),
            d: z.clone(),
            r,
            z,
            q: vec![0.0; n],
            diag,
            rz,
        }
    }

    fn step(&mut self, st: &Stencil) {
        if self.rz == 0.0 {
            return;
        }
        apply(st, &self.d, &mut self.q);
        let dq = dot(&self.d, &self.q);
        if dq <= 0.0 {
            return;
        }
        let alpha = self.rz / dq;
        for i in 1..self.x.len() {
            self.x[i] += alpha * self.d[i];
            self.r[i] -= alpha * self.q[i];
            self.z[i] = self.r[i] / self.diag[i];
        }
        let rz_new = dot(&self.r, &self.z);
        let beta = rz_new / self.rz;
        self.rz = rz_new;
        for i in 1..self.x.len() {
            self.d[i] = self.z[i] + beta * self.d[i];
        }
    }
}

// ── the certificate ───────────────────────────────────────────────────────

/// `min_i (A w)_i` minus its rounding margin, over the rows given.
fn cert_margin(st: &Stencil, w: &[f64], rows: impl Iterator<Item = usize>) -> f64 {
    rows.map(|i| {
        let (v, mag, deg) = row_apply(st, w, i);
        v - (deg as f64 + 3.0) * EPS * mag
    })
    .fold(f64::INFINITY, f64::min)
}

/// `w` with `A w ≥ 1` verified on every row of the intact grid.
fn base_certificate(fx: &Fixture, st: &Stencil) -> Vec<f64> {
    let all = vec![true; fx.grid.edges.len()];
    let ones = vec![1.0; fx.grid.n];
    let w = chol_full(&fx.grid, &all, &ones).expect("intact grid connected");
    let alpha = cert_margin(st, &w, 1..fx.grid.n);
    assert!(alpha > 0.0, "intact certificate must verify");
    w.iter().map(|x| x / alpha).collect()
}

enum OutageCert {
    /// The intact `w`, rescaled after a two-row recheck. Cost: two rows.
    Reused(Vec<f64>),
    /// The recheck failed. PCG on `A' w = 1`, warm-started from the intact
    /// `w`, ran until the stencil check passed; `iters` is its cost. How `w`
    /// was found does not matter for soundness, only that `A' w ≥ 1` verifies.
    Fallback { w: Vec<f64>, iters: usize },
}

impl OutageCert {
    fn w(&self) -> &[f64] {
        match self {
            OutageCert::Reused(w) | OutageCert::Fallback { w, .. } => w,
        }
    }
}

/// Only rows `a` and `b` of `A w` change when line `(a, b)` trips; every other
/// row keeps the `≥ 1` the intact check proved.
fn outage_certificate(fx: &Fixture, st: &Stencil, w_base: &[f64], line: usize) -> OutageCert {
    let e = &fx.grid.edges[line];
    let rows = [e.from, e.to].into_iter().filter(|&i| i > 0);
    let alpha = cert_margin(st, w_base, rows).min(1.0);
    if alpha > 1e-9 {
        return OutageCert::Reused(w_base.iter().map(|x| x / alpha).collect());
    }
    let ones = vec![1.0; fx.grid.n];
    let mut cg = Pcg::new(st, &ones, w_base);
    for iters in 1..=CERT_CAP {
        cg.step(st);
        let alpha = cert_margin(st, &cg.x, 1..fx.grid.n);
        if alpha > 0.5 {
            return OutageCert::Fallback {
                w: cg.x.iter().map(|x| x / alpha).collect(),
                iters,
            };
        }
    }
    panic!("no certificate within {CERT_CAP} PCG iterations");
}

/// Which bound a decision is allowed to use.
///
/// - `Field`: `|f_l − f̂_l| ≤ b_l ‖r‖∞ (w_i + w_j)`, from the field bound.
/// - `L1`: `|f_l − f̂_l| ≤ ‖r‖₁`, needing no certificate vector at all.
///   Proof: the flow error is `b_l Σ_k (G_ik − G_jk) r_k` with `G = A⁻¹`. By
///   symmetry `G_ik − G_jk` is the potential at `k` of a unit dipole `i → j`
///   with bus 0 held at 0. That potential is harmonic off `i, j`, so it lies
///   between its values at `j` and `i`; bus 0 is one of its values, so
///   `|G_ik − G_jk| ≤ φ_i − φ_j = R_eff(i, j) ≤ 1 / b_l`, line `l` being one
///   path between `i` and `j`. Hence `|err_l| ≤ ‖r‖₁` for every live line.
/// - `Both`: the smaller of the two, still sound.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Cert {
    Field,
    L1,
    Both,
}

impl Cert {
    fn name(self) -> &'static str {
        match self {
            Cert::Field => "field",
            Cert::L1 => "l1",
            Cert::Both => "both",
        }
    }
}

/// Per-line post-outage flow error bounds for an iterate with residual `r`.
fn flow_bounds(fx: &Fixture, alive: &[bool], w: &[f64], r: Resid, cert: Cert) -> Vec<f64> {
    fx.grid
        .edges
        .iter()
        .enumerate()
        .map(|(e, edge)| {
            if !alive[e] {
                return 0.0;
            }
            let field = edge.susceptance * r.inf * (w[edge.from] + w[edge.to]);
            match cert {
                Cert::Field => field,
                Cert::L1 => r.l1,
                Cert::Both => field.min(r.l1),
            }
        })
        .collect()
}

// ── decisions ─────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy, Debug, PartialEq)]
struct Score {
    right: u64,
    wrong: u64,
    abstain: u64,
}

#[derive(Default, Clone, Copy, Debug)]
struct Decisions {
    ovl_cert: Score,
    ovl_naive_wrong: u64,
    sign_cert: Score,
    sign_naive_wrong: u64,
    /// Lines whose true shift is zero: no sign may ever be certified.
    sign_zero: u64,
    top_cert: Score,
    top_naive_wrong: u64,
    total_ovl: u64,
    /// True overloads, and how many of them are certified as overloads.
    ovl_pos: u64,
    ovl_pos_cert: u64,
}

impl Decisions {
    fn add(&mut self, o: &Decisions) {
        let add = |a: &mut Score, b: &Score| {
            a.right += b.right;
            a.wrong += b.wrong;
            a.abstain += b.abstain;
        };
        add(&mut self.ovl_cert, &o.ovl_cert);
        add(&mut self.sign_cert, &o.sign_cert);
        add(&mut self.top_cert, &o.top_cert);
        self.ovl_naive_wrong += o.ovl_naive_wrong;
        self.sign_naive_wrong += o.sign_naive_wrong;
        self.sign_zero += o.sign_zero;
        self.top_naive_wrong += o.top_naive_wrong;
        self.total_ovl += o.total_ovl;
        self.ovl_pos += o.ovl_pos;
        self.ovl_pos_cert += o.ovl_pos_cert;
    }
}

struct Truth<'a> {
    f0: &'a [f64],
    post: &'a [f64],
    ratings: &'a [f64],
}

/// Score one outage. `est` / `eps` are the estimated post-outage flows and
/// their bounds; the oracle is `truth.post`.
fn decide(alive: &[bool], est: &[f64], eps: &[f64], truth: &Truth<'_>) -> Decisions {
    let mut d = Decisions::default();
    let scale = truth
        .post
        .iter()
        .zip(truth.f0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        .max(1e-300);
    let live: Vec<usize> = (0..est.len()).filter(|&l| alive[l]).collect();
    for &l in &live {
        // Overload.
        d.total_ovl += 1;
        let truly = truth.post[l].abs() > truth.ratings[l];
        let lo = est[l].abs() - eps[l];
        let hi = est[l].abs() + eps[l];
        if (est[l].abs() > truth.ratings[l]) != truly {
            d.ovl_naive_wrong += 1;
        }
        if truly {
            d.ovl_pos += 1;
        }
        if lo > truth.ratings[l] {
            if truly {
                d.ovl_cert.right += 1;
                d.ovl_pos_cert += 1;
            } else {
                d.ovl_cert.wrong += 1
            }
        } else if hi < truth.ratings[l] {
            if truly {
                d.ovl_cert.wrong += 1
            } else {
                d.ovl_cert.right += 1
            }
        } else {
            d.ovl_cert.abstain += 1;
        }
        // Sign of the shift.
        let t = truth.post[l] - truth.f0[l];
        let s = est[l] - truth.f0[l];
        let zero = t.abs() <= 1e-12 * scale;
        if zero {
            d.sign_zero += 1;
        } else if (s > 0.0) != (t > 0.0) {
            d.sign_naive_wrong += 1;
        }
        if s.abs() > eps[l] {
            if !zero && (s > 0.0) == (t > 0.0) {
                d.sign_cert.right += 1
            } else {
                d.sign_cert.wrong += 1
            }
        } else {
            d.sign_cert.abstain += 1;
        }
    }
    // Which line moves most.
    let shift = |l: usize, v: &[f64]| (v[l] - truth.f0[l]).abs();
    let true_max = live
        .iter()
        .map(|&l| shift(l, truth.post))
        .fold(0.0, f64::max);
    let is_top = |l: usize| shift(l, truth.post) >= true_max * (1.0 - 1e-12);
    let naive = *live
        .iter()
        .max_by(|&&a, &&b| shift(a, est).total_cmp(&shift(b, est)))
        .expect("a live line");
    if !is_top(naive) {
        d.top_naive_wrong += 1;
    }
    let lo_top = shift(naive, est) - eps[naive];
    let beaten = live
        .iter()
        .any(|&l| l != naive && shift(l, est) + eps[l] >= lo_top);
    if beaten {
        d.top_cert.abstain += 1;
    } else if is_top(naive) {
        d.top_cert.right += 1;
    } else {
        d.top_cert.wrong += 1;
    }
    d
}

// ── one fixture, prepared ─────────────────────────────────────────────────

struct Outage {
    line: usize,
    alive: Vec<bool>,
    st: Stencil,
    theta: Vec<f64>,
    post: Vec<f64>,
    cert: OutageCert,
}

struct Prepared {
    fx: Fixture,
    w_base: Vec<f64>,
    theta0: Vec<f64>,
    f0: Vec<f64>,
    ratings: Vec<f64>,
    outages: Vec<Outage>,
    islanding: usize,
    oracle_lodf_gap: f64,
    oracle_spectral_gap: f64,
    spectral_checked: usize,
}

fn ratings_for(f0: &[f64]) -> Vec<f64> {
    let max = f0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    f0.iter()
        .map(|f| f.abs().max(0.2 * max) / LOADING)
        .collect()
}

/// LODF post-outage flows for every outage, from one intact-grid inverse.
fn lodf_post(fx: &Fixture, f0: &[f64]) -> Vec<Option<Vec<f64>>> {
    let n = fx.grid.n;
    let all = vec![true; fx.grid.edges.len()];
    let m = n - 1;
    let mut l = grounded(&fx.grid, &all);
    assert!(cholesky(&mut l, m));
    // Full-size inverse, zero row/column for the reference bus.
    let mut g = vec![0.0; n * n];
    for j in 1..n {
        let mut col = vec![0.0; m];
        col[j - 1] = 1.0;
        chol_solve(&l, m, &mut col);
        for i in 1..n {
            g[i * n + j] = col[i - 1];
        }
    }
    let ptdf = |e: &Edge, c: usize, d: usize| {
        e.susceptance * (g[e.from * n + c] - g[e.from * n + d] - g[e.to * n + c] + g[e.to * n + d])
    };
    (0..fx.grid.edges.len())
        .map(|k| {
            let ek = &fx.grid.edges[k];
            let denom = 1.0 - ptdf(ek, ek.from, ek.to);
            if denom.abs() < 1e-9 {
                return None;
            }
            Some(
                fx.grid
                    .edges
                    .iter()
                    .enumerate()
                    .map(|(e, edge)| {
                        if e == k {
                            0.0
                        } else {
                            f0[e] + ptdf(edge, ek.from, ek.to) / denom * f0[k]
                        }
                    })
                    .collect(),
            )
        })
        .collect()
}

fn prepare(fx: Fixture, spectral_sample: usize) -> Prepared {
    let all = vec![true; fx.grid.edges.len()];
    let theta0 = chol_full(&fx.grid, &all, &fx.p).expect("intact grid connected");
    let f0 = dc_flows(&fx.grid, &all, &theta0);
    let ratings = ratings_for(&f0);
    let base_st = stencil(&fx, &all);
    let w_base = base_certificate(&fx, &base_st);
    let lodf = lodf_post(&fx, &f0);
    let scale = f0.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    let mut outages = Vec::new();
    let (mut islanding, mut lodf_gap, mut spec_gap, mut spec_n) = (0, 0.0f64, 0.0f64, 0);
    for line in 0..fx.grid.edges.len() {
        let mut alive = all.clone();
        alive[line] = false;
        let Some(theta) = chol_full(&fx.grid, &alive, &fx.p) else {
            islanding += 1;
            continue;
        };
        let post = dc_flows(&fx.grid, &alive, &theta);
        let lp = lodf[line]
            .as_ref()
            .expect("Cholesky and LODF agree on islanding");
        lodf_gap = lodf_gap.max(
            post.iter()
                .zip(lp)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
                / scale,
        );
        if spec_n < spectral_sample
            && line % (fx.grid.edges.len() / spectral_sample.max(1)).max(1) == 0
        {
            let eig = symmetric_eigen(&fx.grid.laplacian_of(&alive), fx.grid.n);
            let mut t = eig.pseudo_apply(&fx.p, 1e-10);
            let g0 = t[0];
            for x in &mut t {
                *x -= g0;
            }
            let sp = dc_flows(&fx.grid, &alive, &t);
            spec_gap = spec_gap.max(
                post.iter()
                    .zip(&sp)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
                    / scale,
            );
            spec_n += 1;
        }
        let st = stencil(&fx, &alive);
        let cert = outage_certificate(&fx, &st, &w_base, line);
        outages.push(Outage {
            line,
            alive,
            st,
            theta,
            post,
            cert,
        });
    }
    Prepared {
        fx,
        w_base,
        theta0,
        f0,
        ratings,
        outages,
        islanding,
        oracle_lodf_gap: lodf_gap,
        oracle_spectral_gap: spec_gap,
        spectral_checked: spec_n,
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    JacCold,
    JacWarm,
    PcgWarm,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::JacCold => "jac-cold",
            Arm::JacWarm => "jac-warm",
            Arm::PcgWarm => "pcg-warm",
        }
    }
}

#[derive(Default, Clone, Copy)]
struct BudgetRow {
    field_rel: f64,
    field_delta_rel: f64,
    flow_rel: f64,
    flow_delta_rel: f64,
    bound_over_actual: f64,
    l1_over_actual: f64,
    soundness_violations: u64,
    dec: Decisions,
    dec_l1: Decisions,
}

/// Iterate one outage and report its state at each budget in `BUDGETS`.
fn run_outage(pr: &Prepared, o: &Outage, arm: Arm, ratings: &[f64]) -> Vec<BudgetRow> {
    let fx = &pr.fx;
    let n = fx.grid.n;
    let x0 = match arm {
        Arm::JacCold => vec![0.0; n],
        _ => pr.theta0.clone(),
    };
    let truth = Truth {
        f0: &pr.f0,
        post: &o.post,
        ratings,
    };
    let w = o.cert.w();
    let rel = |a: &[f64], b: &[f64]| {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let den: f64 = b.iter().map(|y| y * y).sum();
        (num / den.max(1e-300)).sqrt()
    };
    let dtheta: Vec<f64> = o.theta.iter().zip(&pr.theta0).map(|(a, b)| a - b).collect();
    let dflow: Vec<f64> = o.post.iter().zip(&pr.f0).map(|(a, b)| a - b).collect();
    let mut rows = Vec::with_capacity(BUDGETS.len());
    let snapshot = |x: &[f64], rb: Resid| {
        let est = dc_flows(&fx.grid, &o.alive, x);
        let eps = flow_bounds(fx, &o.alive, w, rb, Cert::Field);
        let eps_l1 = flow_bounds(fx, &o.alive, w, rb, Cert::L1);
        // Soundness, both bounds: every bus error inside the field bound and
        // every line-flow error inside the L1 bound.
        let viol = (1..n)
            .filter(|&i| (x[i] - o.theta[i]).abs() > rb.inf * w[i] * (1.0 + 1e-9) + 1e-300)
            .count() as u64
            + (0..est.len())
                .filter(|&l| {
                    o.alive[l] && (est[l] - o.post[l]).abs() > eps_l1[l] * (1.0 + 1e-9) + 1e-300
                })
                .count() as u64;
        let actual: f64 = est
            .iter()
            .zip(&o.post)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let bound = eps.iter().fold(0.0, |m: f64, x| m.max(*x));
        let dx: Vec<f64> = x.iter().zip(&pr.theta0).map(|(a, b)| a - b).collect();
        let df: Vec<f64> = est.iter().zip(&pr.f0).map(|(a, b)| a - b).collect();
        BudgetRow {
            field_rel: rel(x, &o.theta),
            field_delta_rel: rel(&dx, &dtheta),
            flow_rel: rel(&est, &o.post),
            flow_delta_rel: rel(&df, &dflow),
            bound_over_actual: if actual > 0.0 {
                bound / actual
            } else {
                f64::INFINITY
            },
            l1_over_actual: if actual > 0.0 {
                rb.l1 / actual
            } else {
                f64::INFINITY
            },
            soundness_violations: viol,
            dec: decide(&o.alive, &est, &eps, &truth),
            dec_l1: decide(&o.alive, &est, &eps_l1, &truth),
        }
    };
    match arm {
        Arm::JacCold | Arm::JacWarm => {
            let (mut x, mut next) = (x0, vec![0.0; n]);
            let mut k = 0;
            for &budget in &BUDGETS {
                while k < budget {
                    jacobi_sweep(&o.st, &fx.p, &x, &mut next);
                    std::mem::swap(&mut x, &mut next);
                    k += 1;
                }
                // The sweep's free bound belongs to the iterate before it; at a
                // snapshot the bound is recomputed for `x` itself.
                let rb = residual_bound(&o.st, &fx.p, &x);
                rows.push(snapshot(&x, rb));
            }
        }
        Arm::PcgWarm => {
            let mut cg = Pcg::new(&o.st, &fx.p, &x0);
            let mut k = 0;
            for &budget in &BUDGETS {
                while k < budget {
                    cg.step(&o.st);
                    k += 1;
                }
                let rb = residual_bound(&o.st, &fx.p, &cg.x);
                rows.push(snapshot(&cg.x, rb));
            }
        }
    }
    rows
}

/// Iterations until every overload decision of the outage is certified, or
/// `None` at the cap. Uses the free residual of the Jacobi sweep.
fn iterations_to_certify(
    pr: &Prepared,
    o: &Outage,
    arm: Arm,
    ratings: &[f64],
    cert: Cert,
) -> Option<usize> {
    let fx = &pr.fx;
    let n = fx.grid.n;
    let w = o.cert.w();
    let decided = |x: &[f64], rb: Resid| {
        let est = dc_flows(&fx.grid, &o.alive, x);
        let eps = flow_bounds(fx, &o.alive, w, rb, cert);
        (0..est.len()).all(|l| {
            !o.alive[l] || est[l].abs() - eps[l] > ratings[l] || est[l].abs() + eps[l] < ratings[l]
        })
    };
    match arm {
        Arm::JacCold | Arm::JacWarm => {
            let mut x = if arm == Arm::JacCold {
                vec![0.0; n]
            } else {
                pr.theta0.clone()
            };
            let mut next = vec![0.0; n];
            for k in 0..CERT_CAP {
                // The sweep returns the residual bound of `x` (the input).
                let rb = jacobi_sweep(&o.st, &fx.p, &x, &mut next);
                if decided(&x, rb) {
                    return Some(k);
                }
                std::mem::swap(&mut x, &mut next);
            }
            None
        }
        Arm::PcgWarm => {
            let mut cg = Pcg::new(&o.st, &fx.p, &pr.theta0);
            for k in 0..CERT_CAP {
                let rb = residual_bound(&o.st, &fx.p, &cg.x);
                if decided(&cg.x, rb) {
                    return Some(k);
                }
                cg.step(&o.st);
            }
            None
        }
    }
}

fn percentile(v: &mut [usize], q: f64) -> usize {
    v.sort_unstable();
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn median_ms(mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..5)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[2]
}

/// Ratings for the near-tie adversary: the outage that most loads each of the
/// first `k` lines sets that line's rating to the exact post-outage flow. A
/// strict `>` then has no margin: no finite bound can certify either side.
fn near_tie_ratings(pr: &Prepared, k: usize) -> (Vec<f64>, usize) {
    let mut r = pr.ratings.clone();
    let mut tied = 0;
    for (l, rating) in r.iter_mut().enumerate().take(k) {
        if let Some(o) = pr
            .outages
            .iter()
            .filter(|o| o.alive[l])
            .max_by(|a, b| a.post[l].abs().total_cmp(&b.post[l].abs()))
        {
            *rating = o.post[l].abs();
            tied += 1;
        }
    }
    (r, tied)
}

fn report(pr: &Prepared, label: &str, ratings: &[f64], arms: &[Arm]) {
    let fx = &pr.fx;
    println!(
        "\n=== {label}: {} buses, {} lines, {} outages ({} islanding skipped) ===",
        fx.grid.n,
        fx.grid.edges.len(),
        pr.outages.len(),
        pr.islanding
    );
    let mut fb: Vec<usize> = pr
        .outages
        .iter()
        .filter_map(|o| match o.cert {
            OutageCert::Fallback { iters, .. } => Some(iters),
            OutageCert::Reused(_) => None,
        })
        .collect();
    let fallbacks = fb.len();
    let fb_note = if fb.is_empty() {
        String::new()
    } else {
        format!(
            " (PCG iterations median {} max {})",
            percentile(&mut fb, 0.5),
            percentile(&mut fb, 1.0)
        )
    };
    println!(
        "oracle gaps (rel. to max|f0|): Cholesky vs LODF {:.1e}, vs spectral {:.1e} ({} sampled); certificate fallbacks {}/{}{}",
        pr.oracle_lodf_gap,
        pr.oracle_spectral_gap,
        pr.spectral_checked,
        fallbacks,
        pr.outages.len(),
        fb_note
    );
    let overloads: usize = pr
        .outages
        .iter()
        .map(|o| {
            (0..o.post.len())
                .filter(|&l| o.alive[l] && o.post[l].abs() > ratings[l])
                .count()
        })
        .sum();
    println!("true N-1 overloads under these ratings: {overloads}");
    println!(
        "  columns: field/dfield/flow/dflow = max rel. L2 error of the angle field, its outage delta, the flows, their delta;"
    );
    println!(
        "  bound/actual = median ratio of the worst per-line bound to the worst actual flow error; OVL/SGN = % decisions"
    );
    println!("  certified right (abstentions); naiveX = wrong decisions read off the iterate with no bound;");
    println!("  OVL+ f/l = true overloads certified as overloads by the field / l1 bound");
    for &arm in arms {
        println!(
            "  {:<8} {:>5} | {:>8} {:>8} {:>8} {:>8} | {:>9} {:>9} | {:>14} {:>14} {:>6} {:>9} | {:>6} {:>6} {:>6} | {:>4} {:>4} {:>4} |",
            arm.name(),
            "it",
            "field",
            "dfield",
            "flow",
            "dflow",
            "fld/act",
            "l1/act",
            "OVL field",
            "OVL l1",
            "naiveX",
            "OVL+ f/l",
            "SGN f",
            "SGN l1",
            "nX",
            "TOPf",
            "TOPl",
            "nX"
        );
        let mut acc = vec![BudgetRow::default(); BUDGETS.len()];
        let mut ratio: Vec<Vec<f64>> = vec![Vec::new(); BUDGETS.len()];
        let mut ratio_l1: Vec<Vec<f64>> = vec![Vec::new(); BUDGETS.len()];
        for o in &pr.outages {
            for (b, row) in run_outage(pr, o, arm, ratings).into_iter().enumerate() {
                let a = &mut acc[b];
                a.field_rel = a.field_rel.max(row.field_rel);
                a.field_delta_rel = a.field_delta_rel.max(row.field_delta_rel);
                a.flow_rel = a.flow_rel.max(row.flow_rel);
                a.flow_delta_rel = a.flow_delta_rel.max(row.flow_delta_rel);
                a.soundness_violations += row.soundness_violations;
                a.dec.add(&row.dec);
                a.dec_l1.add(&row.dec_l1);
                if row.bound_over_actual.is_finite() {
                    ratio[b].push(row.bound_over_actual);
                    ratio_l1[b].push(row.l1_over_actual);
                }
            }
        }
        let med = |r: &mut Vec<f64>| {
            r.sort_by(f64::total_cmp);
            if r.is_empty() {
                f64::NAN
            } else {
                r[r.len() / 2]
            }
        };
        for (b, a) in acc.iter().enumerate() {
            let pct = |s: Score, tot: u64| 100.0 * s.right as f64 / tot.max(1) as f64;
            let (d, l) = (&a.dec, &a.dec_l1);
            let wrong = d.ovl_cert.wrong
                + d.sign_cert.wrong
                + d.top_cert.wrong
                + l.ovl_cert.wrong
                + l.sign_cert.wrong
                + l.top_cert.wrong;
            println!(
                "  {:<8} {:>5} | {:>8.1e} {:>8.1e} {:>8.1e} {:>8.1e} | {:>9.1} {:>9.1} | {:>6.1}% ({:>6}) {:>6.1}% ({:>6}) {:>6} {:>4}/{:<4} | {:>5.1}% {:>5.1}% {:>6} | {:>4} {:>4} {:>4} | {}{}",
                "",
                BUDGETS[b],
                a.field_rel,
                a.field_delta_rel,
                a.flow_rel,
                a.flow_delta_rel,
                med(&mut ratio[b]),
                med(&mut ratio_l1[b]),
                pct(d.ovl_cert, d.total_ovl),
                d.ovl_cert.abstain,
                pct(l.ovl_cert, l.total_ovl),
                l.ovl_cert.abstain,
                d.ovl_naive_wrong,
                d.ovl_pos_cert,
                l.ovl_pos_cert,
                pct(d.sign_cert, d.total_ovl - d.sign_zero),
                pct(l.sign_cert, l.total_ovl - l.sign_zero),
                d.sign_naive_wrong,
                d.top_cert.right,
                l.top_cert.right,
                d.top_naive_wrong,
                if wrong == 0 { "0 wrong" } else { "CERTIFIED WRONG" },
                if a.soundness_violations == 0 { "" } else { " BOUND VIOLATED" }
            );
        }
    }
}

fn certify_table(pr: &Prepared, ratings: &[f64], arms: &[Arm]) {
    println!("  iterations until every overload decision of an outage is certified:");
    for &arm in arms {
        for cert in [Cert::Field, Cert::L1, Cert::Both] {
            let mut its = Vec::new();
            let mut capped = 0;
            for o in &pr.outages {
                match iterations_to_certify(pr, o, arm, ratings, cert) {
                    Some(k) => its.push(k),
                    None => capped += 1,
                }
            }
            if its.is_empty() {
                println!(
                    "    {:<8} {:<5} none certified ({capped} at cap {CERT_CAP})",
                    arm.name(),
                    cert.name()
                );
                continue;
            }
            let (p50, p95, max) = (
                percentile(&mut its, 0.5),
                percentile(&mut its, 0.95),
                percentile(&mut its, 1.0),
            );
            println!(
            "    {:<8} {:<5} median {p50:>6}  p95 {p95:>6}  max {max:>6}  uncertified at cap: {capped}",
            arm.name(),
            cert.name()
        );
        }
    }
}

fn timing(pr: &Prepared) {
    let fx = &pr.fx;
    let ratings = &pr.ratings;
    let lodf_ms = median_ms(|| {
        std::hint::black_box(lodf_post(fx, &pr.f0));
    });
    let chol_ms = median_ms(|| {
        for o in &pr.outages {
            std::hint::black_box(chol_full(&fx.grid, &o.alive, &fx.p));
        }
    });
    // The per-outage certificate is part of the price of a certified decision.
    let cert_ms = median_ms(|| {
        for o in &pr.outages {
            std::hint::black_box(outage_certificate(fx, &o.st, &pr.w_base, o.line));
        }
    });
    let run = |arm: Arm, cert: Cert| {
        median_ms(|| {
            for o in &pr.outages {
                std::hint::black_box(iterations_to_certify(pr, o, arm, ratings, cert));
            }
        })
    };
    let (jac_b, pcg_b) = (run(Arm::JacWarm, Cert::Both), run(Arm::PcgWarm, Cert::Both));
    let (jac_l, pcg_l) = (run(Arm::JacWarm, Cert::L1), run(Arm::PcgWarm, Cert::L1));
    println!(
        "  screen all outages (median of 5, ms): lodf {lodf_ms:.2} (incl. intact inverse) | chol {chol_ms:.2}"
    );
    println!(
        "    both-bound: certificates {cert_ms:.2} + jac-warm {jac_b:.2} / pcg-warm {pcg_b:.2} | l1-only (no certificate vector): jac-warm {jac_l:.2} / pcg-warm {pcg_l:.2}"
    );
}

fn main() {
    let fixtures = vec![
        (fixture_1427(), 16usize),
        (hetero("hetero-16x16", 16, 0xC0FFEE), 3),
        (uniform("strip-64x2", 64, 2, true, 0xBEEF), 3),
        (weak_tie("weak-tie-16x8", 16, 8, 0xFACE), 3),
        (uniform("bipartite-12x12", 12, 12, false, 0xD1CE), 3),
    ];
    let all_arms = [Arm::JacCold, Arm::JacWarm, Arm::PcgWarm];
    for (fx, sample) in fixtures {
        let pr = prepare(fx, sample);
        let label = pr.fx.name;
        report(&pr, label, &pr.ratings, &all_arms);
        certify_table(&pr, &pr.ratings, &all_arms);
        timing(&pr);
        if label == "hetero-16x16" {
            let (r, tied) = near_tie_ratings(&pr, 40);
            report(&pr, "hetero-16x16 NEAR-TIE ratings", &r, &[Arm::JacWarm]);
            println!("  ({tied} ratings set to an exact post-outage flow)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Vec<Prepared> {
        vec![
            prepare(fixture_1427(), 16),
            prepare(hetero("hetero-6x6", 6, 7), 4),
            prepare(weak_tie("weak-tie-8x4", 8, 4, 9), 4),
            prepare(uniform("bipartite-5x5", 5, 5, false, 11), 4),
        ]
    }

    #[test]
    fn oracles_agree() {
        for pr in small() {
            assert!(
                pr.oracle_lodf_gap < 1e-9,
                "{}: LODF gap {}",
                pr.fx.name,
                pr.oracle_lodf_gap
            );
            assert!(
                pr.oracle_spectral_gap < 1e-7,
                "{}: spectral gap {}",
                pr.fx.name,
                pr.oracle_spectral_gap
            );
            assert!(pr.spectral_checked > 0);
        }
    }

    /// The certificate is sound: no bus error ever exceeds its bound and no
    /// certified decision is wrong, for every arm and budget.
    #[test]
    fn certified_decisions_are_never_wrong() {
        for pr in small() {
            for arm in [Arm::JacCold, Arm::JacWarm, Arm::PcgWarm] {
                for o in &pr.outages {
                    for (b, row) in run_outage(&pr, o, arm, &pr.ratings).iter().enumerate() {
                        let d = &row.dec;
                        assert_eq!(
                            row.soundness_violations, 0,
                            "{} {:?} line {} budget {}",
                            pr.fx.name, arm, o.line, BUDGETS[b]
                        );
                        assert_eq!(
                            d.ovl_cert.wrong + d.sign_cert.wrong + d.top_cert.wrong,
                            0,
                            "{} {:?} line {} budget {}",
                            pr.fx.name,
                            arm,
                            o.line,
                            BUDGETS[b]
                        );
                    }
                }
            }
        }
    }

    /// Anti-vacuity: a certificate that never certifies is trivially sound.
    /// Given enough iterations it must decide almost everything.
    #[test]
    fn certificate_can_decide() {
        for pr in small() {
            let mut d = Decisions::default();
            for o in &pr.outages {
                d.add(
                    &run_outage(&pr, o, Arm::PcgWarm, &pr.ratings)
                        .last()
                        .unwrap()
                        .dec,
                );
            }
            assert!(
                d.ovl_cert.right * 100 >= d.total_ovl * 99,
                "{}: {:?}",
                pr.fx.name,
                d.ovl_cert
            );
        }
    }

    /// Silence twin: a rating equal to the exact flow can never be certified
    /// either way, however converged the iterate is.
    #[test]
    fn exact_ties_are_never_certified() {
        let pr = prepare(hetero("hetero-6x6", 6, 7), 4);
        let (r, tied) = near_tie_ratings(&pr, 10);
        assert!(tied > 0);
        for (l, rating) in r.iter().enumerate().take(10) {
            let o = pr
                .outages
                .iter()
                .find(|o| o.alive[l] && o.post[l].abs() == *rating)
                .expect("the tying outage");
            let rows = run_outage(&pr, o, Arm::PcgWarm, &r);
            let last = rows.last().unwrap();
            assert_eq!(last.dec.ovl_cert.wrong, 0);
            let est = dc_flows(&pr.fx.grid, &o.alive, &o.theta);
            let _ = est;
            assert!(last.dec.ovl_cert.abstain >= 1, "line {l} tie was decided");
        }
    }

    /// The danger is real, not hypothetical: on near-tie ratings, a decision
    /// read off an unconverged iterate is wrong somewhere.
    #[test]
    fn naive_early_decisions_are_wrong_somewhere() {
        let pr = prepare(hetero("hetero-6x6", 6, 7), 4);
        let (r, _) = near_tie_ratings(&pr, 20);
        let mut wrong = 0;
        for o in &pr.outages {
            wrong += run_outage(&pr, o, Arm::JacWarm, &r)[3].dec.ovl_naive_wrong;
        }
        assert!(
            wrong > 0,
            "8-sweep naive decisions were all right on near ties"
        );
    }

    /// After an outage the warm-start residual is nonzero only at the two
    /// endpoints of the tripped line.
    #[test]
    fn warm_start_residual_is_local_to_the_outage() {
        let pr = prepare(hetero("hetero-6x6", 6, 7), 4);
        for o in &pr.outages {
            let e = &pr.fx.grid.edges[o.line];
            for i in 1..pr.fx.grid.n {
                let (ax, mag, _) = row_apply(&o.st, &pr.theta0, i);
                let r = (pr.fx.p[i] - ax).abs();
                if i == e.from || i == e.to {
                    assert!(r > 1e-9 * mag.max(1e-300) || pr.f0[o.line].abs() < 1e-12);
                } else {
                    assert!(
                        r <= 1e-9 * (mag + pr.fx.p[i].abs()).max(1e-12),
                        "row {i} line {}",
                        o.line
                    );
                }
            }
        }
    }

    /// The two-row shortcut is sound: every outage certificate, reused or
    /// fallback, satisfies `A' w ≥ 1` on EVERY row of the post-outage grid,
    /// not only the two rows the shortcut rechecks.
    #[test]
    fn every_outage_certificate_verifies_on_all_rows() {
        for pr in small() {
            let (mut reused, mut fallback) = (0, 0);
            for o in &pr.outages {
                match o.cert {
                    OutageCert::Reused(_) => reused += 1,
                    OutageCert::Fallback { .. } => fallback += 1,
                }
                let m = cert_margin(&o.st, o.cert.w(), 1..pr.fx.grid.n);
                assert!(
                    m >= 1.0 - 1e-9,
                    "{} line {}: margin {m}",
                    pr.fx.name,
                    o.line
                );
            }
            // Anti-vacuity: both paths are exercised across the fixtures.
            println!("{}: reused {reused}, fallback {fallback}", pr.fx.name);
        }
        let all = small();
        let count = |f: fn(&OutageCert) -> bool| -> usize {
            all.iter()
                .map(|pr| pr.outages.iter().filter(|o| f(&o.cert)).count())
                .sum()
        };
        assert!(count(|c| matches!(c, OutageCert::Reused(_))) > 0);
        assert!(count(|c| matches!(c, OutageCert::Fallback { .. })) > 0);
    }

    /// The L1 flow bound needs no certificate vector and is never violated,
    /// even on the near-islanding weak tie.
    #[test]
    fn l1_flow_bound_holds_without_a_certificate() {
        let pr = prepare(weak_tie("weak-tie-8x4", 8, 4, 9), 4);
        for o in &pr.outages {
            for arm in [Arm::JacCold, Arm::JacWarm] {
                for row in run_outage(&pr, o, arm, &pr.ratings) {
                    assert_eq!(row.soundness_violations, 0);
                    assert_eq!(row.dec_l1.ovl_cert.wrong + row.dec_l1.sign_cert.wrong, 0);
                }
            }
        }
    }
}
