//! D-ART-1 — algebraic recipe table × percentile profiles × Wankel phasors.
//!
//! Question: given a concept (a shape family with its parameters), a chain of
//! transformations and a requested terminal, how much of the geometric and
//! trigonometric work can a small, guarded recipe table eliminate before
//! anything is evaluated — and does it know when it cannot?
//!
//! Nothing here is a production primitive. The probe holds:
//!
//! - a canonicaliser that folds a transformation chain into one symbolic
//!   affine (rotations stay exact `u32` turns; translations stay lazy);
//! - a static recipe table `[shape][linear class][terminal] → recipe`, with
//!   every guard checked at evaluation, never trusted from the key;
//! - an independent oracle that materialises the vertices and applies the raw
//!   chain point by point;
//! - affine transformation of `CrossPowerSums` summaries (exact `i128`);
//! - four implementations of a projected order statistic of an `m`-fold
//!   symmetric point family, the Wankel apex triangle being `m = 3`.
//!
//! Run with:
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-mask-risc --example algebraic_recipe_probe
//! ```

use ndarray::hpc::rolling_floor::rank_per_10000;
use ndarray::simd::{masked_group_cross_power_sums_i32, CrossPowerSums};
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

const TURN: f64 = 4_294_967_296.0;

// ───────────────────────── counters and rng ─────────────────────────

/// Transcendental calls made by the planner and the oracle (not by the
/// benchmark arms, which use uncounted calls so the counter costs nothing).
static TRIG: AtomicU64 = AtomicU64::new(0);

fn sc(rad: f64) -> (f64, f64) {
    TRIG.fetch_add(1, Relaxed);
    rad.sin_cos()
}

fn trig() -> u64 {
    TRIG.load(Relaxed)
}

fn rad(p: u32) -> f64 {
    p as f64 / TURN * TAU
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn median_ns(reps: usize, per: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_nanos() as f64 / per as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[reps / 2]
}

// ───────────────────────── concepts ─────────────────────────

/// Where an `m`-fold family is centred.
#[derive(Clone, Copy, Debug)]
enum Center {
    /// The Wankel apex triangle: centre `e · e^{i3θ}`, coupled to the phase.
    Wankel { e: f64 },
    /// A fixed centre, independent of the phase.
    Fixed(f64, f64),
}

/// `m` points `c + r · e^{i(θ + 2πk/m)}`, `k = 0..m`. Holds no points.
#[derive(Clone, Copy, Debug)]
struct Regular {
    m: u32,
    r: f64,
    phase: u32,
    center: Center,
}

/// Transformations, applied left to right.
#[derive(Clone, Copy, Debug)]
enum Xf {
    Rotate(u32),
    Scale(f64),
    Translate(f64, f64),
    /// A general linear map `[a, b, c, d]`: `(x, y) ↦ (ax + by, cx + dy)`.
    Linear([f64; 4]),
}

#[derive(Clone, Copy, Debug)]
enum Term {
    Count,
    Sum,
    Centroid,
    /// `Σ |p − centroid|²`.
    SumSqCentered,
    /// `Σ |p|²`.
    SumSqOrigin,
    /// `Σ (x − x̄)(y − ȳ)`.
    CrossCentered,
    /// Order statistic of the x projections, `rank_per_10000` convention.
    QuantileX(u32),
    Vertex(u32),
    /// Does any point lie within `d` of `(px, py)`?
    Collide(f64, f64, f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
    Num(f64),
    Pt(f64, f64),
    Bool(bool),
}

// ───────────────────────── recipe table ─────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum ShapeK {
    Regular = 0,
    Population = 1,
}
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum XfK {
    Identity = 0,
    Rotation = 1,
    Similarity = 2,
    Affine = 3,
}
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum TermK {
    Count = 0,
    Sum,
    Centroid,
    SumSqCentered,
    SumSqOrigin,
    CrossCentered,
    Quantile,
    Vertex,
    Collide,
}
const NT: usize = 9;

/// What remains after the rewrite.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Rid {
    /// A constant of the concept.
    Const,
    /// One centre phasor, no vertex.
    CenterPhasor,
    /// An invariant of the isotropic centred second moment: no phase.
    IsoInvariant,
    /// Centred invariant plus `m|c'|²`: phase-free unless translated.
    OriginNorm,
    /// One cosine through the symmetry-reduced sector permutation.
    SectorQuantile,
    /// One vertex phasor.
    VertexPhasor,
    /// The summary transforms; the population is never re-read.
    SummaryAffine,
    /// No sufficient summary: materialise and evaluate.
    Materialize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Exactness {
    /// An identity of real arithmetic; `f64` rounding order changes.
    RealIdentity,
    /// Bitwise exact in integers within the stated widths.
    IntExact,
    /// The oracle itself.
    Reference,
}

fn exactness(r: Rid) -> Exactness {
    match r {
        Rid::SummaryAffine => Exactness::IntExact,
        Rid::Materialize => Exactness::Reference,
        _ => Exactness::RealIdentity,
    }
}

fn term_k(t: &Term) -> TermK {
    match t {
        Term::Count => TermK::Count,
        Term::Sum => TermK::Sum,
        Term::Centroid => TermK::Centroid,
        Term::SumSqCentered => TermK::SumSqCentered,
        Term::SumSqOrigin => TermK::SumSqOrigin,
        Term::CrossCentered => TermK::CrossCentered,
        Term::QuantileX(_) => TermK::Quantile,
        Term::Vertex(_) => TermK::Vertex,
        Term::Collide(..) => TermK::Collide,
    }
}

/// The table for a regular family does not depend on the linear class: the
/// class changes the arithmetic inside a recipe, never whether it applies.
/// Guards that do depend on it run at evaluation.
const REGULAR_ROW: [Rid; NT] = [
    Rid::Const,
    Rid::CenterPhasor,
    Rid::CenterPhasor,
    Rid::IsoInvariant,
    Rid::OriginNorm,
    Rid::IsoInvariant,
    Rid::SectorQuantile,
    Rid::VertexPhasor,
    Rid::Materialize,
];
/// A population carries only `n, Σx, Σy, Σx², Σy², Σxy`: order statistics,
/// vertices and collisions are not functions of those.
const POPULATION_ROW: [Rid; NT] = [
    Rid::SummaryAffine,
    Rid::SummaryAffine,
    Rid::SummaryAffine,
    Rid::SummaryAffine,
    Rid::SummaryAffine,
    Rid::SummaryAffine,
    Rid::Materialize,
    Rid::Materialize,
    Rid::Materialize,
];
static TABLE: [[[Rid; NT]; 4]; 2] = [[REGULAR_ROW; 4], [POPULATION_ROW; 4]];

fn lookup_array(s: ShapeK, x: XfK, t: TermK) -> Rid {
    TABLE[s as usize][x as usize][t as usize]
}

fn lookup_match(s: ShapeK, _x: XfK, t: TermK) -> Rid {
    match (s, t) {
        (ShapeK::Regular, TermK::Count) => Rid::Const,
        (ShapeK::Regular, TermK::Sum | TermK::Centroid) => Rid::CenterPhasor,
        (ShapeK::Regular, TermK::SumSqCentered | TermK::CrossCentered) => Rid::IsoInvariant,
        (ShapeK::Regular, TermK::SumSqOrigin) => Rid::OriginNorm,
        (ShapeK::Regular, TermK::Quantile) => Rid::SectorQuantile,
        (ShapeK::Regular, TermK::Vertex) => Rid::VertexPhasor,
        (ShapeK::Population, TermK::Quantile | TermK::Vertex | TermK::Collide) => Rid::Materialize,
        (ShapeK::Population, _) => Rid::SummaryAffine,
        (ShapeK::Regular, TermK::Collide) => Rid::Materialize,
    }
}

const ALL_T: [TermK; NT] = [
    TermK::Count,
    TermK::Sum,
    TermK::Centroid,
    TermK::SumSqCentered,
    TermK::SumSqOrigin,
    TermK::CrossCentered,
    TermK::Quantile,
    TermK::Vertex,
    TermK::Collide,
];
const ALL_X: [XfK; 4] = [XfK::Identity, XfK::Rotation, XfK::Similarity, XfK::Affine];
const ALL_S: [ShapeK; 2] = [ShapeK::Regular, ShapeK::Population];

/// A hasher that sends every key to one bucket: the map must still answer
/// correctly, by comparing keys.
#[derive(Default, Clone)]
struct Collide;
impl Hasher for Collide {
    fn finish(&self) -> u64 {
        0
    }
    fn write(&mut self, _: &[u8]) {}
}
impl BuildHasher for Collide {
    type Hasher = Collide;
    fn build_hasher(&self) -> Collide {
        Collide
    }
}

// ───────────────────────── canonical form ─────────────────────────

/// `p ↦ L p + t`. While the linear part is a similarity it is held as exact
/// `(rot, scale)` and translations stay lazy, each tagged with the rotation
/// and scale in force when it was applied; nothing trigonometric is computed
/// until a terminal asks for it.
#[derive(Clone, Debug)]
struct Canon {
    class: XfK,
    rot: u32,
    scale: f64,
    /// Valid when `class == Affine`.
    lin: [f64; 4],
    /// `(tx, ty, rot_at, k)` while a similarity: `k` is the product of the
    /// scales applied after the translation, kept per entry so a zero scale
    /// zeroes only the translations it follows.
    lazy_t: Vec<(f64, f64, u32, f64)>,
    /// Valid when `class == Affine`.
    t: (f64, f64),
    rewrites: u32,
}

fn mat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

fn mat_vec(a: [f64; 4], v: (f64, f64)) -> (f64, f64) {
    (a[0] * v.0 + a[1] * v.1, a[2] * v.0 + a[3] * v.1)
}

fn rot_mat(p: u32, s: f64) -> [f64; 4] {
    if p == 0 {
        return [s, 0.0, 0.0, s];
    }
    let (sn, cs) = sc(rad(p));
    [s * cs, -s * sn, s * sn, s * cs]
}

impl Canon {
    fn new() -> Self {
        Canon {
            class: XfK::Identity,
            rot: 0,
            scale: 1.0,
            lin: [1.0, 0.0, 0.0, 1.0],
            lazy_t: Vec::new(),
            t: (0.0, 0.0),
            rewrites: 0,
        }
    }

    fn of(chain: &[Xf]) -> Self {
        let mut c = Canon::new();
        for x in chain {
            c.push(*x);
        }
        c.settle();
        c
    }

    fn push(&mut self, x: Xf) {
        if self.class == XfK::Affine {
            match x {
                Xf::Rotate(p) => self.affine(rot_mat(p, 1.0)),
                Xf::Scale(s) => self.affine([s, 0.0, 0.0, s]),
                Xf::Linear(m) => self.affine(m),
                Xf::Translate(a, b) => {
                    self.t = (self.t.0 + a, self.t.1 + b);
                }
            }
            return;
        }
        match x {
            // R(a)R(b) = R(a + b), exact in turns.
            Xf::Rotate(p) => {
                if self.class != XfK::Identity {
                    self.rewrites += 1;
                }
                self.rot = self.rot.wrapping_add(p);
                self.class = self.class.max(XfK::Rotation);
            }
            // A negative similarity is a half turn and a positive scale: the
            // rank order of a projection is preserved by construction.
            Xf::Scale(s) => {
                if s < 0.0 {
                    self.rot = self.rot.wrapping_add(1 << 31);
                    self.rewrites += 1;
                }
                self.scale *= s.abs();
                for e in &mut self.lazy_t {
                    e.3 *= s.abs();
                }
                self.class = self.class.max(XfK::Similarity);
            }
            Xf::Translate(a, b) => self.lazy_t.push((a, b, self.rot, 1.0)),
            Xf::Linear(m) => {
                // Leave the similarity form: materialise L and t once.
                self.lin = rot_mat(self.rot, self.scale);
                self.t = self.lazy_translation();
                self.lazy_t.clear();
                self.class = XfK::Affine;
                self.affine(m);
            }
        }
    }

    fn affine(&mut self, m: [f64; 4]) {
        self.lin = mat_mul(m, self.lin);
        self.t = mat_vec(m, self.t);
    }

    /// `R(α)R(−α) = I` and `s = 1`: drop back down the class lattice.
    fn settle(&mut self) {
        if self.class == XfK::Similarity && self.scale == 1.0 {
            self.class = XfK::Rotation;
            self.rewrites += 1;
        }
        if self.class == XfK::Rotation && self.rot == 0 {
            self.class = XfK::Identity;
            self.rewrites += 1;
        }
    }

    /// `Σ k_i R(rot − rot_i) t_i`; a term with no relative rotation costs
    /// no trigonometry.
    fn lazy_translation(&self) -> (f64, f64) {
        let mut t = (0.0, 0.0);
        for &(a, b, r, k) in &self.lazy_t {
            let v = mat_vec(rot_mat(self.rot.wrapping_sub(r), k), (a, b));
            t = (t.0 + v.0, t.1 + v.1);
        }
        t
    }

    fn translation(&self) -> (f64, f64) {
        if self.class == XfK::Affine {
            self.t
        } else {
            self.lazy_translation()
        }
    }

    fn has_translation(&self) -> bool {
        if self.class == XfK::Affine {
            self.t != (0.0, 0.0)
        } else {
            self.lazy_t.iter().any(|&(a, b, ..)| a != 0.0 || b != 0.0)
        }
    }

    /// `‖L‖_F²`, exact for a similarity without trigonometry.
    fn frob2(&self) -> f64 {
        match self.class {
            XfK::Affine => self.lin.iter().map(|v| v * v).sum(),
            _ => 2.0 * self.scale * self.scale,
        }
    }

    /// `(L Lᵀ)₀₁`: zero for every similarity.
    fn llt01(&self) -> f64 {
        match self.class {
            XfK::Affine => self.lin[0] * self.lin[2] + self.lin[1] * self.lin[3],
            _ => 0.0,
        }
    }

    fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        let l = match self.class {
            XfK::Affine => self.lin,
            _ => rot_mat(self.rot, self.scale),
        };
        let v = mat_vec(l, p);
        let t = self.translation();
        (v.0 + t.0, v.1 + t.1)
    }
}

// ───────────────────────── symmetry-reduced order statistics ─────────────────────────

/// For `cos(φ + 2πk/m)`, ties occur only at `φ ≡ 0 mod 1/(2m)` turn, so inside
/// each half-sector the ascending order of `k` is fixed. Two permutations
/// replace every sort.
struct SectorPerm {
    m: u32,
    perm: [Vec<u32>; 2],
}

impl SectorPerm {
    fn new(m: u32) -> Self {
        let mk = |half: f64| {
            let psi = (0.25 + 0.5 * half) / m as f64;
            let mut idx: Vec<u32> = (0..m).collect();
            idx.sort_by(|&a, &b| {
                let va = (TAU * (psi + a as f64 / m as f64)).cos();
                let vb = (TAU * (psi + b as f64 / m as f64)).cos();
                va.total_cmp(&vb)
            });
            idx
        };
        SectorPerm {
            m,
            perm: [mk(0.0), mk(1.0)],
        }
    }

    /// The rank-th smallest of `cos(2π(phi_t + k/m))`, `phi_t` in turns.
    #[inline]
    fn value(&self, rank: usize, phi_t: f64) -> f64 {
        let m = self.m as f64;
        let f = (phi_t * m).rem_euclid(1.0);
        let j = self.perm[usize::from(f >= 0.5)][rank];
        (TAU * ((f + j as f64) / m)).cos()
    }
}

// ───────────────────────── oracle ─────────────────────────

fn center_of(s: &Regular) -> (f64, f64) {
    match s.center {
        Center::Wankel { e } => {
            let (sn, cs) = sc(rad(s.phase.wrapping_mul(3)));
            (e * cs, e * sn)
        }
        Center::Fixed(x, y) => (x, y),
    }
}

fn apply_raw(chain: &[Xf], mut p: (f64, f64)) -> (f64, f64) {
    for x in chain {
        p = match *x {
            Xf::Rotate(a) => {
                let (sn, cs) = sc(rad(a));
                (cs * p.0 - sn * p.1, sn * p.0 + cs * p.1)
            }
            Xf::Scale(s) => (s * p.0, s * p.1),
            Xf::Translate(a, b) => (p.0 + a, p.1 + b),
            Xf::Linear(m) => mat_vec(m, p),
        };
    }
    p
}

/// Materialise every vertex, apply the raw chain to each, evaluate.
fn oracle(s: &Regular, chain: &[Xf], t: &Term) -> Answer {
    let c = center_of(s);
    let pts: Vec<(f64, f64)> = (0..s.m)
        .map(|k| {
            let (sn, cs) = sc(rad(s.phase) + TAU * k as f64 / s.m as f64);
            apply_raw(chain, (c.0 + s.r * cs, c.1 + s.r * sn))
        })
        .collect();
    eval_points(&pts, t)
}

/// The case's magnitude for [`close`].
fn magnitude(s: &Regular, chain: &[Xf]) -> f64 {
    match oracle(s, chain, &Term::SumSqOrigin) {
        Answer::Num(v) => v + s.m as f64,
        _ => unreachable!(),
    }
}

fn eval_points(pts: &[(f64, f64)], t: &Term) -> Answer {
    let n = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / n, sy / n);
    match *t {
        Term::Count => Answer::Num(n),
        Term::Sum => Answer::Pt(sx, sy),
        Term::Centroid => Answer::Pt(mx, my),
        Term::SumSqCentered => Answer::Num(
            pts.iter()
                .map(|p| (p.0 - mx).powi(2) + (p.1 - my).powi(2))
                .sum(),
        ),
        Term::SumSqOrigin => Answer::Num(pts.iter().map(|p| p.0 * p.0 + p.1 * p.1).sum()),
        Term::CrossCentered => Answer::Num(pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum()),
        Term::QuantileX(q) => {
            let mut xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
            xs.sort_by(f64::total_cmp);
            Answer::Num(xs[rank_per_10000(xs.len(), q)])
        }
        Term::Vertex(k) => {
            let p = pts[k as usize];
            Answer::Pt(p.0, p.1)
        }
        Term::Collide(px, py, d) => Answer::Bool(
            pts.iter()
                .any(|p| (p.0 - px).powi(2) + (p.1 - py).powi(2) <= d * d),
        ),
    }
}

// ───────────────────────── planner + evaluator ─────────────────────────

#[derive(Debug)]
struct Plan {
    rid: Rid,
    /// The recipe the table proposed, before guards.
    proposed: Rid,
    trig: u64,
    vertices: u32,
}

/// The centre's image `L c + t`; for a Wankel centre under a similarity, one
/// phasor of `3θ + rot` and no matrix.
fn center_image(s: &Regular, c: &Canon) -> (f64, f64) {
    let lc = match (s.center, c.class) {
        (Center::Wankel { e }, k) if k != XfK::Affine => {
            if e == 0.0 {
                (0.0, 0.0)
            } else {
                let (sn, cs) = sc(rad(s.phase.wrapping_mul(3).wrapping_add(c.rot)));
                (c.scale * e * cs, c.scale * e * sn)
            }
        }
        _ => {
            let c0 = center_of(s);
            if c.class == XfK::Affine {
                mat_vec(c.lin, c0)
            } else {
                mat_vec(rot_mat(c.rot, c.scale), c0)
            }
        }
    };
    let t = c.translation();
    (lc.0 + t.0, lc.1 + t.1)
}

/// `|c|` without a phase when the concept carries it.
fn center_norm(s: &Regular) -> f64 {
    match s.center {
        Center::Wankel { e } => e.abs(),
        Center::Fixed(x, y) => x.hypot(y),
    }
}

/// Guarded evaluation. A guard that fails falls through to `Materialize`.
fn plan_eval(
    s: &Regular,
    chain: &[Xf],
    t: &Term,
    perms: &mut HashMap<u32, SectorPerm>,
) -> (Answer, Plan) {
    let t0 = trig();
    let c = Canon::of(chain);
    let proposed = lookup_array(ShapeK::Regular, c.class, term_k(t));
    let m = s.m as f64;
    let r2 = s.r * s.r;
    // The centred second moment Σ (z−c)(z−c)ᵀ = (m r²/2)·I holds for m ≥ 3
    // only; m = 2 is a rank-one dyad.
    let isotropic = s.m >= 3;
    let mut rid = proposed;
    let mut vertices = 0;
    let ans = match (proposed, t) {
        (Rid::Const, Term::Count) => Answer::Num(m),
        (Rid::CenterPhasor, Term::Sum) => {
            let p = center_image(s, &c);
            Answer::Pt(m * p.0, m * p.1)
        }
        (Rid::CenterPhasor, Term::Centroid) => {
            let p = center_image(s, &c);
            Answer::Pt(p.0, p.1)
        }
        (Rid::IsoInvariant, Term::SumSqCentered) if isotropic => {
            Answer::Num(0.5 * m * r2 * c.frob2())
        }
        (Rid::IsoInvariant, Term::CrossCentered) if isotropic => {
            Answer::Num(0.5 * m * r2 * c.llt01())
        }
        (Rid::OriginNorm, Term::SumSqOrigin) if isotropic => {
            let centred = 0.5 * m * r2 * c.frob2();
            let cn2 = if c.class != XfK::Affine && !c.has_translation() {
                (c.scale * center_norm(s)).powi(2)
            } else {
                let p = center_image(s, &c);
                p.0 * p.0 + p.1 * p.1
            };
            Answer::Num(centred + m * cn2)
        }
        (Rid::SectorQuantile, Term::QuantileX(q)) => {
            // Σ-free: x'_k = c'_x + r·(L₀₀ cos φ_k + L₀₁ sin φ_k)
            //            = c'_x + r·ρ·cos(φ_k − β).
            let (rho, shift_t) = if c.class == XfK::Affine {
                let (a, b) = (c.lin[0], c.lin[1]);
                (a.hypot(b), -b.atan2(a) / TAU)
            } else {
                (c.scale, c.rot as f64 / TURN)
            };
            let cx = center_image(s, &c).0;
            let rank = rank_per_10000(s.m as usize, *q);
            let sp = perms.entry(s.m).or_insert_with(|| SectorPerm::new(s.m));
            TRIG.fetch_add(1, Relaxed);
            let v = sp.value(rank, s.phase as f64 / TURN + shift_t);
            Answer::Num(cx + s.r * rho * v)
        }
        (Rid::VertexPhasor, Term::Vertex(k)) => {
            let c0 = center_of(s);
            let (sn, cs) = sc(rad(s.phase) + TAU * *k as f64 / m);
            let v = c.apply((c0.0 + s.r * cs, c0.1 + s.r * sn));
            Answer::Pt(v.0, v.1)
        }
        _ => {
            rid = Rid::Materialize;
            vertices = s.m;
            oracle(s, chain, t)
        }
    };
    (
        ans,
        Plan {
            rid,
            proposed,
            trig: trig() - t0,
            vertices,
        },
    )
}

/// Relative `1e-9` of the compared values, plus an absolute floor of `1e-13`
/// of the case's magnitude `Σ|p|² + m` so results that cancel to zero are
/// judged against the size of the terms that cancelled.
fn close(a: Answer, b: Answer, mag: f64) -> bool {
    let ok = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(y.abs()) + 1e-13 * mag;
    match (a, b) {
        (Answer::Num(x), Answer::Num(y)) => ok(x, y),
        (Answer::Pt(x0, y0), Answer::Pt(x1, y1)) => ok(x0, x1) && ok(y0, y1),
        (Answer::Bool(x), Answer::Bool(y)) => x == y,
        _ => false,
    }
}

// ───────────────────────── section 1: the Wankel table ─────────────────────────

fn wankel_table() {
    println!("== 1. Wankel concept + transformations: what each terminal needs ==");
    let w = Regular {
        m: 3,
        r: 100.0,
        phase: 0x1234_5678,
        center: Center::Wankel { e: 14.0 },
    };
    let chain = [
        Xf::Rotate(0x2000_0000),
        Xf::Translate(5.0, -3.0),
        Xf::Rotate(0xE000_0000), // inverse of the first: cancels
        Xf::Scale(2.0),
        Xf::Scale(0.5),
    ];
    let mut perms = HashMap::new();
    let queries: [(&str, Term); 9] = [
        ("A count", Term::Count),
        ("B sum of positions", Term::Sum),
        ("  centroid", Term::Centroid),
        ("C centred Σ|z−c|²", Term::SumSqCentered),
        ("  centred Σ(x−x̄)(y−ȳ)", Term::CrossCentered),
        ("D Σ|z|² (translated)", Term::SumSqOrigin),
        ("E apex 1", Term::Vertex(1)),
        ("  x-quantile p=0.5", Term::QuantileX(5000)),
        ("F collide (110, 0) d 5", Term::Collide(110.0, 0.0, 5.0)),
    ];
    println!(
        "  chain: R(+1/8) · T(5, −3) · R(−1/8) · S(2) · S(0.5)  →  canonical {:?}, {} rewrites",
        Canon::of(&chain).class,
        Canon::of(&chain).rewrites
    );
    println!(
        "  {:<26} {:<15} {:<15} {:>5} {:>7} {:>6}   oracle trig/vertices",
        "query", "proposed", "executed", "trig", "verts", "equal"
    );
    for (name, t) in queries {
        let (a, p) = plan_eval(&w, &chain, &t, &mut perms);
        let o0 = trig();
        let o = oracle(&w, &chain, &t);
        let ot = trig() - o0;
        let eq = close(a, o, magnitude(&w, &chain));
        assert!(eq, "{name}: recipe {a:?} vs oracle {o:?}");
        println!(
            "  {:<26} {:<15} {:<15} {:>5} {:>7} {:>6}   {ot} / 3",
            name,
            format!("{:?}", p.proposed),
            format!("{:?}", p.rid),
            p.trig,
            p.vertices,
            eq
        );
    }
    // Untranslated: Σ|z|² = 3 s²(R² + e²) needs no phase at all.
    let chain2 = [Xf::Rotate(0x1111_1111), Xf::Scale(-1.5)];
    let (a, p) = plan_eval(&w, &chain2, &Term::SumSqOrigin, &mut perms);
    assert!(close(
        a,
        oracle(&w, &chain2, &Term::SumSqOrigin),
        magnitude(&w, &chain2)
    ));
    assert_eq!(p.trig, 0, "untranslated Σ|z|² must be phase-free");
    println!(
        "  D' Σ|z|² under R·S(−1.5), no translation: {:?} with {} trig = 3·s²(R²+e²) = {:.3}",
        p.rid,
        p.trig,
        3.0 * 2.25 * (100.0f64.powi(2) + 14.0f64.powi(2))
    );
}

// ───────────────────────── section 2: guards and falsifiers ─────────────────────────

fn guards() {
    println!("\n== 2. guards: every recipe against the oracle, and where a recipe must refuse ==");
    let mut rng = Rng(0x5EED);
    let mut perms = HashMap::new();
    let mut checked = 0usize;
    let mut by_rid: HashMap<Rid, (usize, u64, u64)> = HashMap::new();
    for case in 0..4000 {
        let m = [2u32, 3, 3, 4, 5, 8, 16, 256][case % 8];
        let center = if case % 3 == 0 {
            Center::Fixed(rng.range(-50.0, 50.0), rng.range(-50.0, 50.0))
        } else {
            Center::Wankel {
                e: rng.range(0.0, 20.0),
            }
        };
        let s = Regular {
            m,
            r: rng.range(1.0, 150.0),
            phase: rng.next() as u32,
            center,
        };
        let n = rng.below(5) as usize;
        let chain: Vec<Xf> = (0..n)
            .map(|_| match rng.below(5) {
                0 => Xf::Rotate(rng.next() as u32),
                1 => Xf::Scale(rng.range(-2.0, 2.0)),
                4 => Xf::Scale(0.0),
                2 => Xf::Translate(rng.range(-30.0, 30.0), rng.range(-30.0, 30.0)),
                _ => Xf::Linear([
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                ]),
            })
            .collect();
        let terms = [
            Term::Count,
            Term::Sum,
            Term::Centroid,
            Term::SumSqCentered,
            Term::SumSqOrigin,
            Term::CrossCentered,
            Term::QuantileX(rng.below(10_001) as u32),
            Term::Vertex(rng.below(m as u64) as u32),
            Term::Collide(
                rng.range(-200.0, 200.0),
                rng.range(-200.0, 200.0),
                rng.range(1.0, 100.0),
            ),
        ];
        let mag = magnitude(&s, &chain);
        for t in terms {
            let (a, p) = plan_eval(&s, &chain, &t, &mut perms);
            let o0 = trig();
            let o = oracle(&s, &chain, &t);
            let ot = trig() - o0;
            assert!(
                close(a, o, mag),
                "case {case}: m {m} {chain:?} {t:?}: {:?} gave {a:?}, oracle {o:?}",
                p.rid
            );
            let e = by_rid.entry(p.rid).or_default();
            e.0 += 1;
            e.1 += p.trig;
            e.2 += ot;
            checked += 1;
        }
    }
    println!(
        "  {checked} (concept, chain, terminal) cases equal the oracle (rel. 1e-9 of the values + 1e-13 of Σ|p|²)"
    );
    // A translation applied after a zero scale survives it.
    let z = [Xf::Scale(0.0), Xf::Translate(5.0, -3.0)];
    let w0 = Regular {
        m: 3,
        r: 100.0,
        phase: 0x1234_5678,
        center: Center::Wankel { e: 14.0 },
    };
    let (a, _) = plan_eval(&w0, &z, &Term::Centroid, &mut perms);
    assert_eq!(a, Answer::Pt(5.0, -3.0));
    assert_eq!(oracle(&w0, &z, &Term::Centroid), Answer::Pt(5.0, -3.0));
    println!("  S(0) · T(5, −3): centroid (5, −3) by recipe and oracle");
    let mut rows: Vec<_> = by_rid.into_iter().collect();
    rows.sort_by_key(|r| format!("{:?}", r.0));
    for (rid, (n, tr, ot)) in rows {
        println!(
            "    {:<15} {:>6} cases  trig/case {:>6.2}  oracle trig/case {:>7.2}  {:?}",
            format!("{rid:?}"),
            n,
            tr as f64 / n as f64,
            ot as f64 / n as f64,
            exactness(rid)
        );
    }

    // m = 2 is a dyad, not isotropic: the invariant must refuse.
    let s2 = Regular {
        m: 2,
        r: 10.0,
        phase: 0x0800_0000,
        center: Center::Fixed(0.0, 0.0),
    };
    let lin = [Xf::Linear([2.0, 0.0, 0.0, 1.0])];
    let iso = 0.5 * 2.0 * 100.0 * Canon::of(&lin).frob2();
    let truth = match oracle(&s2, &lin, &Term::SumSqCentered) {
        Answer::Num(v) => v,
        _ => unreachable!(),
    };
    let (_, p) = plan_eval(&s2, &lin, &Term::SumSqCentered, &mut perms);
    assert!((iso - truth).abs() > 1.0, "m = 2 must break isotropy");
    assert_eq!(p.rid, Rid::Materialize);
    println!(
        "  m = 2 under a non-orthogonal map: isotropic formula {iso:.2} vs truth {truth:.2}; recipe refused → {:?}",
        p.rid
    );

    // A non-orthogonal map is not a rotation: the invariant changes with it.
    let w = Regular {
        m: 3,
        r: 10.0,
        phase: 0,
        center: Center::Wankel { e: 1.0 },
    };
    let shear = [Xf::Linear([1.0, 1.5, 0.0, 1.0])];
    let rot_only = 3.0 * 100.0;
    let (a, _) = plan_eval(&w, &shear, &Term::SumSqCentered, &mut perms);
    let Answer::Num(v) = a else { unreachable!() };
    assert!((v - rot_only).abs() > 1.0);
    println!(
        "  shear [1 1.5; 0 1]: Σ|z−c|² = {v:.2}, not the rotation invariant {rot_only:.2} (class {:?})",
        Canon::of(&shear).class
    );

    // Symmetry of the profile: period 1/m turn and reflection, measured.
    let mut worst = 0.0f64;
    for i in 0..20_000 {
        let m = [3u32, 16][i % 2];
        let th = (i as f64 * 0.618_033_988_7).fract();
        let q = |phi: f64| {
            let mut v: Vec<f64> = (0..m)
                .map(|k| (TAU * (phi + k as f64 / m as f64)).cos())
                .collect();
            v.sort_by(f64::total_cmp);
            v
        };
        let (a, b, c) = (q(th), q(th + 1.0 / m as f64), q(-th));
        for k in 0..m as usize {
            worst = worst.max((a[k] - b[k]).abs()).max((a[k] - c[k]).abs());
        }
    }
    assert!(worst < 1e-12);
    println!(
        "  profile symmetry: Q(θ + 1/m) = Q(θ) = Q(−θ) to {worst:.1e} on 20,000 phases; one half-sector (1/(2m) turn) is the whole domain"
    );
    println!(
        "  note: 1/3 turn is not representable in u32 turns (2^32 mod 3 = {}); threefold symmetry holds in reals, to 1 ulp-turn in u32",
        (1u64 << 32) % 3
    );
}

// ───────────────────────── section 3: populations and PowerSums ─────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct S2 {
    n: i128,
    x: i128,
    y: i128,
    xx: i128,
    yy: i128,
    xy: i128,
}

impl From<CrossPowerSums> for S2 {
    fn from(c: CrossPowerSums) -> Self {
        S2 {
            n: c.n as i128,
            x: c.sum_x as i128,
            y: c.sum_y as i128,
            xx: c.sum_x_sq as i128,
            yy: c.sum_y_sq as i128,
            xy: c.sum_xy,
        }
    }
}

/// Integer affine `(a b; c d) p + (tx, ty)`.
#[derive(Clone, Copy, Debug)]
struct IAff {
    m: [i64; 4],
    t: (i64, i64),
}

impl IAff {
    fn then(self, next: IAff) -> IAff {
        let a = next.m;
        let b = self.m;
        IAff {
            m: [
                a[0] * b[0] + a[1] * b[2],
                a[0] * b[1] + a[1] * b[3],
                a[2] * b[0] + a[3] * b[2],
                a[2] * b[1] + a[3] * b[3],
            ],
            t: (
                a[0] * self.t.0 + a[1] * self.t.1 + next.t.0,
                a[2] * self.t.0 + a[3] * self.t.1 + next.t.1,
            ),
        }
    }

    #[inline]
    fn point(&self, x: i64, y: i64) -> (i64, i64) {
        (
            self.m[0] * x + self.m[1] * y + self.t.0,
            self.m[2] * x + self.m[3] * y + self.t.1,
        )
    }

    /// `S' = A S + n t`, `M' = A M Aᵀ + A S tᵀ + t Sᵀ Aᵀ + n t tᵀ`.
    fn summary(&self, s: S2) -> S2 {
        let [a, b, c, d] = self.m.map(i128::from);
        let (tx, ty) = (i128::from(self.t.0), i128::from(self.t.1));
        let lx = a * s.x + b * s.y;
        let ly = c * s.x + d * s.y;
        S2 {
            n: s.n,
            x: lx + s.n * tx,
            y: ly + s.n * ty,
            xx: a * a * s.xx + 2 * a * b * s.xy + b * b * s.yy + 2 * tx * lx + s.n * tx * tx,
            yy: c * c * s.xx + 2 * c * d * s.xy + d * d * s.yy + 2 * ty * ly + s.n * ty * ty,
            xy: a * c * s.xx
                + (a * d + b * c) * s.xy
                + b * d * s.yy
                + tx * ly
                + ty * lx
                + s.n * tx * ty,
        }
    }
}

fn fold_ndarray(xs: &[i32], ys: &[i32], mask: &[u64], keys: &[u32]) -> S2 {
    let mut out = [CrossPowerSums::default(); 1];
    masked_group_cross_power_sums_i32(mask, keys, xs, ys, &mut out);
    out[0].into()
}

fn full_mask(n: usize) -> Vec<u64> {
    let mut m = vec![!0u64; n.div_ceil(64)];
    if !n.is_multiple_of(64) {
        *m.last_mut().unwrap() = (1u64 << (n % 64)) - 1;
    }
    m
}

fn population() {
    println!("\n== 3. populations: affine chains over CrossPowerSums ==");
    let ts = [
        IAff {
            m: [1, 2, 0, 1],
            t: (3, -5),
        },
        IAff {
            m: [0, -1, 1, 0],
            t: (7, 2),
        },
        IAff {
            m: [2, 0, 0, -1],
            t: (-4, 9),
        },
    ];
    let comp = ts[0].then(ts[1]).then(ts[2]);
    println!(
        "  T3∘T2∘T1 composed once: A = {:?}, t = {:?}",
        comp.m, comp.t
    );
    println!(
        "  {:>9}  {:>10} {:>10} {:>10} {:>10} {:>10}   ns/row, then ns per summary step",
        "rows", "A mat+fold", "A2 fused", "B fold+3T", "C fold+1T", "warm 1T"
    );
    for n in [1_000usize, 65_536, 1_000_000] {
        let mut rng = Rng(0xAFF + n as u64);
        let xs: Vec<i32> = (0..n).map(|_| rng.below(201) as i32 - 100).collect();
        let ys: Vec<i32> = (0..n).map(|_| rng.below(201) as i32 - 100).collect();
        let keys = vec![0u32; n];
        let mask = full_mask(n);
        let mut bx = vec![0i32; n];
        let mut by = vec![0i32; n];
        let reps = if n > 100_000 { 7 } else { 31 };
        // A: materialise each step into lanes, then the shipped fold.
        let mut a_res = S2::default();
        let ns_a = median_ns(reps, n, || {
            bx.copy_from_slice(&xs);
            by.copy_from_slice(&ys);
            for t in &ts {
                for i in 0..n {
                    let (x, y) = t.point(i64::from(bx[i]), i64::from(by[i]));
                    bx[i] = x as i32;
                    by[i] = y as i32;
                }
            }
            a_res = fold_ndarray(&bx, &by, &mask, &keys);
            a_res.n as f64
        });
        // A2: no lane, the composed map per row and a scalar fold.
        let mut a2 = S2::default();
        let ns_a2 = median_ns(reps, n, || {
            let mut s = S2::default();
            for i in 0..n {
                let (x, y) = comp.point(i64::from(xs[i]), i64::from(ys[i]));
                let (x, y) = (i128::from(x), i128::from(y));
                s.n += 1;
                s.x += x;
                s.y += y;
                s.xx += x * x;
                s.yy += y * y;
                s.xy += x * y;
            }
            a2 = s;
            s.n as f64
        });
        // B / C: fold X once, transform the summary.
        let mut b_res = S2::default();
        let ns_b = median_ns(reps, n, || {
            let mut s = fold_ndarray(&xs, &ys, &mask, &keys);
            for t in &ts {
                s = t.summary(s);
            }
            b_res = s;
            s.n as f64
        });
        let mut c_res = S2::default();
        let ns_c = median_ns(reps, n, || {
            c_res = comp.summary(fold_ndarray(&xs, &ys, &mask, &keys));
            c_res.n as f64
        });
        let base = fold_ndarray(&xs, &ys, &mask, &keys);
        let ns_w = median_ns(101, 1000, || {
            let mut acc = 0i128;
            for _ in 0..1000 {
                acc = acc.wrapping_add(comp.summary(std::hint::black_box(base)).xy);
            }
            acc as f64
        });
        assert_eq!(a_res, a2);
        assert_eq!(a_res, b_res);
        assert_eq!(a_res, c_res);
        println!(
            "  {n:>9}  {ns_a:>10.2} {ns_a2:>10.2} {ns_b:>10.2} {ns_c:>10.2} {ns_w:>10.2}   all four equal, bitwise (i128)"
        );
    }

    // f64: the same identity is not bitwise.
    let mut rng = Rng(0xF64);
    let n = 65_536;
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|_| (rng.range(-100.0, 100.0), rng.range(-100.0, 100.0)))
        .collect();
    let (a, b, c, d) = (
        0.3f64.cos() * 1.7,
        -(0.3f64.sin()) * 1.7,
        0.3f64.sin() * 1.7,
        0.3f64.cos() * 1.7,
    );
    let (tx, ty) = (12.25, -7.5);
    let fold = |pts: &mut dyn Iterator<Item = (f64, f64)>| {
        pts.fold([0.0f64; 6], |s, (x, y)| {
            [
                s[0] + 1.0,
                s[1] + x,
                s[2] + y,
                s[3] + x * x,
                s[4] + y * y,
                s[5] + x * y,
            ]
        })
    };
    let mat = fold(
        &mut pts
            .iter()
            .map(|&(x, y)| (a * x + b * y + tx, c * x + d * y + ty)),
    );
    let s = fold(&mut pts.iter().copied());
    let (lx, ly) = (a * s[1] + b * s[2], c * s[1] + d * s[2]);
    let sum = [
        s[0],
        lx + s[0] * tx,
        ly + s[0] * ty,
        a * a * s[3] + 2.0 * a * b * s[5] + b * b * s[4] + 2.0 * tx * lx + s[0] * tx * tx,
        c * c * s[3] + 2.0 * c * d * s[5] + d * d * s[4] + 2.0 * ty * ly + s[0] * ty * ty,
        a * c * s[3] + (a * d + b * c) * s[5] + b * d * s[4] + tx * ly + ty * lx + s[0] * tx * ty,
    ];
    let rel = (0..6)
        .map(|i| (mat[i] - sum[i]).abs() / mat[i].abs().max(1.0))
        .fold(0.0f64, f64::max);
    let bitwise = (0..6)
        .filter(|&i| mat[i].to_bits() == sum[i].to_bits())
        .count();
    println!(
        "  f64, 65,536 rows, rotation·1.7 + t: summary path vs materialised fold max rel diff {rel:.1e}; {bitwise}/6 fields bitwise equal (a RealIdentity, not IntExact)"
    );

    // Moments are not sufficient for extrema or collision.
    let p1 = [(0i32, 0i32), (3, 0), (3, 0)];
    let p2 = [(1i32, 0i32), (1, 0), (4, 0)];
    let fold3 = |p: &[(i32, i32)]| {
        let xs: Vec<i32> = p.iter().map(|q| q.0).collect();
        let ys: Vec<i32> = p.iter().map(|q| q.1).collect();
        fold_ndarray(&xs, &ys, &full_mask(3), &[0, 0, 0])
    };
    assert_eq!(fold3(&p1), fold3(&p2));
    let max = |p: &[(i32, i32)]| p.iter().map(|q| q.0).max().unwrap();
    let hit = |p: &[(i32, i32)]| p.iter().any(|q| (q.0 - 4).abs() + q.1.abs() == 0);
    assert_ne!(max(&p1), max(&p2));
    assert_ne!(hit(&p1), hit(&p2));
    for t in [TermK::Quantile, TermK::Vertex, TermK::Collide] {
        for x in ALL_X {
            assert_eq!(lookup_array(ShapeK::Population, x, t), Rid::Materialize);
        }
    }
    println!(
        "  {{0,3,3}} vs {{1,1,4}} on y = 0: identical n, Σx, Σy, Σx², Σy², Σxy ({:?}); max {} vs {}, collide (4,0) {} vs {}; table: Population × {{quantile, vertex, collide}} → Materialize",
        fold3(&p1),
        max(&p1),
        max(&p2),
        hit(&p1),
        hit(&p2)
    );

    // A fold may be pushed through a transform only if the group key does
    // not read a transformed column.
    let n = 4096;
    let mut rng = Rng(0x6E7);
    let xs: Vec<i32> = (0..n).map(|_| rng.below(201) as i32 - 100).collect();
    let ys: Vec<i32> = (0..n).map(|_| rng.below(201) as i32 - 100).collect();
    let ids: Vec<u32> = (0..n as u32).map(|i| i & 1).collect();
    let shift = IAff {
        m: [1, 0, 0, 1],
        t: (50, 0),
    };
    let mask = full_mask(n);
    let group = |xs: &[i32], ys: &[i32], keys: &[u32]| {
        let mut out = [CrossPowerSums::default(); 2];
        masked_group_cross_power_sums_i32(&mask, keys, xs, ys, &mut out);
        [S2::from(out[0]), S2::from(out[1])]
    };
    let tx: Vec<i32> = xs.iter().map(|x| x + 50).collect();
    let key_x = |xs: &[i32]| xs.iter().map(|&x| u32::from(x >= 0)).collect::<Vec<u32>>();
    // key reads x, the transform moves x: pushing the fold down is wrong.
    let pushed = group(&xs, &ys, &key_x(&xs)).map(|s| shift.summary(s));
    let truth = group(&tx, &ys, &key_x(&tx));
    assert_ne!(pushed, truth);
    // key reads an untouched column: pushing down is exact.
    let pushed_ok = group(&xs, &ys, &ids).map(|s| shift.summary(s));
    assert_eq!(pushed_ok, group(&tx, &ys, &ids));
    println!(
        "  group key x ≥ 0 under x + 50: pushed-down summaries ≠ truth (group n {} vs {}); key = id parity: pushed-down = truth",
        pushed[1].n, truth[1].n
    );
}

// ───────────────────────── section 4: projected order statistics ─────────────────────────

#[derive(Clone, Copy)]
struct Inst {
    /// Wankel `e` for m = 3, otherwise the fixed centre's x.
    c: f64,
    r: f64,
    phase: u32,
    rank: u32,
}

/// Arm A: generate every vertex, project, select.
#[inline]
fn arm_a(m: u32, wankel: bool, it: &Inst, buf: &mut [f64; 256]) -> f64 {
    let th = rad(it.phase);
    let cx = if wankel {
        it.c * rad(it.phase.wrapping_mul(3)).cos()
    } else {
        it.c
    };
    let b = &mut buf[..m as usize];
    for (k, v) in b.iter_mut().enumerate() {
        *v = cx + it.r * (th + TAU * k as f64 / m as f64).cos();
    }
    let (_, v, _) = b.select_nth_unstable_by(it.rank as usize, f64::total_cmp);
    *v
}

/// Arm B, m = 3 only: one `sin_cos(θ)`, the ±120° rotations by constants,
/// `cos 3θ = 4c³ − 3c`, a three-element sorting network.
#[inline]
fn arm_b3(it: &Inst) -> f64 {
    const H: f64 = 0.866_025_403_784_438_6; // √3/2
    let (s, c) = rad(it.phase).sin_cos();
    let mut u = [c, -0.5 * c - H * s, -0.5 * c + H * s];
    if u[0] > u[1] {
        u.swap(0, 1);
    }
    if u[1] > u[2] {
        u.swap(1, 2);
    }
    if u[0] > u[1] {
        u.swap(0, 1);
    }
    it.c * (4.0 * c * c * c - 3.0 * c) + it.r * u[it.rank as usize]
}

/// Arm C: a phase-conditioned profile `table[bucket][rank]` over one sector
/// `ψ ∈ [0, 1/m)`, linearly interpolated.
struct Profile {
    m: u32,
    b: usize,
    t: Vec<f32>,
}

impl Profile {
    fn new(m: u32, b: usize) -> Self {
        let mut t = vec![0f32; (b + 1) * m as usize];
        for i in 0..=b {
            let psi = i as f64 / (b as f64 * m as f64);
            let mut v: Vec<f64> = (0..m)
                .map(|k| (TAU * (psi + k as f64 / m as f64)).cos())
                .collect();
            v.sort_by(f64::total_cmp);
            for (k, x) in v.iter().enumerate() {
                t[i * m as usize + k] = *x as f32;
            }
        }
        Profile { m, b, t }
    }

    #[inline]
    fn value(&self, rank: usize, phi_t: f64) -> f64 {
        let f = (phi_t * self.m as f64).rem_euclid(1.0) * self.b as f64;
        let i = (f as usize).min(self.b - 1);
        let w = (f - i as f64) as f32;
        let m = self.m as usize;
        let (a, b) = (self.t[i * m + rank], self.t[(i + 1) * m + rank]);
        f64::from(a + w * (b - a))
    }

    fn bytes(&self) -> usize {
        self.t.len() * 4
    }
}

fn instances(n: usize, m: u32, seed: u64) -> Vec<Inst> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| Inst {
            c: if m == 3 {
                rng.range(0.0, 20.0)
            } else {
                rng.range(-50.0, 50.0)
            },
            r: rng.range(50.0, 150.0),
            phase: rng.next() as u32,
            rank: rng.below(u64::from(m)) as u32,
        })
        .collect()
}

fn quantiles() {
    println!("\n== 4. one projected order statistic per instance: generate+select vs analytic vs profile LUT vs sector permutation ==");
    for m in [3u32, 16, 256] {
        let wankel = m == 3;
        let sp = SectorPerm::new(m);
        let prof = Profile::new(m, 64);
        let prof_odd = Profile::new(m, 63);
        let center = |it: &Inst| {
            if wankel {
                it.c * rad(it.phase.wrapping_mul(3)).cos()
            } else {
                it.c
            }
        };
        let arm_e =
            |it: &Inst| center(it) + it.r * sp.value(it.rank as usize, it.phase as f64 / TURN);
        let arm_c = |p: &Profile, it: &Inst| {
            center(it) + it.r * p.value(it.rank as usize, it.phase as f64 / TURN)
        };
        // Error over a dense phase sweep, every rank, including every tie.
        let mut buf = [0f64; 256];
        let (mut eb, mut ec, mut eco, mut ee) = (0f64, 0f64, 0f64, 0f64);
        let sweep = 200_003u32;
        for i in 0..sweep {
            let phase = (u64::from(i) * (1u64 << 32) / u64::from(sweep)) as u32;
            // The oracle sorts all m projections once; every rank is checked.
            let base = Inst {
                c: 7.0,
                r: 100.0,
                phase,
                rank: 0,
            };
            let th = rad(phase);
            let cx = center(&base);
            let sorted = &mut buf[..m as usize];
            for (k, v) in sorted.iter_mut().enumerate() {
                *v = cx + base.r * (th + TAU * k as f64 / m as f64).cos();
            }
            sorted.sort_by(f64::total_cmp);
            for rank in 0..m {
                let it = Inst { rank, ..base };
                let o = buf[rank as usize];
                if wankel {
                    eb = eb.max((arm_b3(&it) - o).abs());
                }
                ec = ec.max((arm_c(&prof, &it) - o).abs());
                eco = eco.max((arm_c(&prof_odd, &it) - o).abs());
                ee = ee.max((arm_e(&it) - o).abs());
            }
        }
        assert!(ee < 1e-9, "sector permutation is exact up to rounding");
        if wankel {
            assert!(eb < 1e-9);
        }
        println!(
            "  m = {m:<3} max |error| on R = 100 ({sweep} phases × all {m} ranks):  B {}  C(64 even) {ec:.1e}  C(63 odd) {eco:.1e}  E {ee:.1e}   C table {} B, E table {} B",
            if wankel { format!("{eb:.1e}") } else { "—".into() },
            prof.bytes(),
            2 * m as usize * 4
        );
        for n in [65_536usize, 1_000_000] {
            let inst = instances(n, m, 0x1A + u64::from(m) + n as u64);
            let reps = 5;
            let na = if m == 256 { n.min(65_536) } else { n };
            let ns_a = median_ns(reps, na, || {
                let mut buf = [0f64; 256];
                inst[..na]
                    .iter()
                    .map(|it| arm_a(m, wankel, it, &mut buf))
                    .sum()
            });
            let ns_b = if wankel {
                format!(
                    "{:.1}",
                    median_ns(reps, n, || inst.iter().map(arm_b3).sum())
                )
            } else {
                "—".into()
            };
            let ns_c = median_ns(reps, n, || inst.iter().map(|it| arm_c(&prof, it)).sum());
            let ns_e = median_ns(reps, n, || inst.iter().map(arm_e).sum());
            // D: terminals that need no profile at all.
            let ns_d = median_ns(reps, n, || {
                inst.iter()
                    .map(|it| 0.5 * m as f64 * it.r * it.r * 2.0)
                    .sum()
            });
            println!(
                "    {n:>9} instances  ns/instance: A gen+select {ns_a:>7.1}  B analytic {ns_b:>6}  C LUT {ns_c:>6.1}  E sector {ns_e:>6.1}   (D Σ|z−c|², no profile: {ns_d:.2})"
            );
        }
    }
}

// ───────────────────────── section 5: lookup and rewrite cost ─────────────────────────

fn lookup_cost() {
    println!("\n== 5. recipe lookup and canonicalisation cost ==");
    let mut rng = Rng(0x10C);
    let keys: Vec<(ShapeK, XfK, TermK)> = (0..4096)
        .map(|_| {
            (
                ALL_S[rng.below(2) as usize],
                ALL_X[rng.below(4) as usize],
                ALL_T[rng.below(NT as u64) as usize],
            )
        })
        .collect();
    let map: HashMap<(ShapeK, XfK, TermK), Rid> = ALL_S
        .iter()
        .flat_map(|&s| {
            ALL_X.iter().flat_map(move |&x| {
                ALL_T
                    .iter()
                    .map(move |&t| ((s, x, t), lookup_array(s, x, t)))
            })
        })
        .collect();
    let mut bad: HashMap<(ShapeK, XfK, TermK), Rid, Collide> = HashMap::with_hasher(Collide);
    for (k, v) in &map {
        bad.insert(*k, *v);
    }
    for &(s, x, t) in &keys {
        let a = lookup_array(s, x, t);
        assert_eq!(a, lookup_match(s, x, t));
        assert_eq!(a, map[&(s, x, t)]);
        assert_eq!(
            a,
            bad[&(s, x, t)],
            "a colliding hasher must still return the right recipe"
        );
    }
    let n = keys.len();
    let sum = |f: &dyn Fn(ShapeK, XfK, TermK) -> Rid| {
        keys.iter()
            .map(|&(s, x, t)| f(s, x, t) as usize as f64)
            .sum::<f64>()
    };
    let ns_arr = median_ns(31, n, || {
        sum(&|s, x, t| lookup_array(s, std::hint::black_box(x), t))
    });
    let ns_match = median_ns(31, n, || {
        sum(&|s, x, t| lookup_match(s, std::hint::black_box(x), t))
    });
    let ns_map = median_ns(31, n, || sum(&|s, x, t| map[&(s, x, t)]));
    let ns_bad = median_ns(31, n, || sum(&|s, x, t| bad[&(s, x, t)]));
    println!(
        "  ns/lookup over 4,096 random keys: static array {ns_arr:.2}  match {ns_match:.2}  HashMap(SipHash) {ns_map:.2}  HashMap(all keys collide) {ns_bad:.2}; all agree"
    );
    let chain = [
        Xf::Rotate(0x2000_0000),
        Xf::Translate(5.0, -3.0),
        Xf::Rotate(0xE000_0000),
        Xf::Scale(2.0),
        Xf::Scale(0.5),
        Xf::Rotate(7),
        Xf::Translate(1.0, 1.0),
        Xf::Scale(-1.0),
    ];
    let ns_canon = median_ns(31, 1000, || {
        (0..1000)
            .map(|_| Canon::of(std::hint::black_box(&chain)).rewrites as f64)
            .sum()
    });
    let w = Regular {
        m: 3,
        r: 100.0,
        phase: 0x1234_5678,
        center: Center::Wankel { e: 14.0 },
    };
    let mut perms = HashMap::new();
    let ns_plan = median_ns(31, 1000, || {
        (0..1000)
            .map(|_| {
                match plan_eval(
                    &w,
                    std::hint::black_box(&chain),
                    &Term::SumSqCentered,
                    &mut perms,
                )
                .0
                {
                    Answer::Num(v) => v,
                    _ => 0.0,
                }
            })
            .sum()
    });
    let ns_oracle = median_ns(31, 1000, || {
        (0..1000)
            .map(
                |_| match oracle(&w, std::hint::black_box(&chain), &Term::SumSqCentered) {
                    Answer::Num(v) => v,
                    _ => 0.0,
                },
            )
            .sum()
    });
    println!(
        "  8-step chain: canonicalise {ns_canon:.0} ns; canonicalise + lookup + evaluate Σ|z−c|² {ns_plan:.0} ns vs oracle (3 vertices × 8 raw steps) {ns_oracle:.0} ns"
    );
}

// ───────────────────────── 1-D affine quantile guard ─────────────────────────

/// `Quantile_p(a·X + b)`: for `a < 0` the order reverses, so the rank-th
/// smallest of `aX + b` is `a · X[n − 1 − rank] + b`.
fn affine_quantile(sorted: &[i64], a: i64, b: i64, per_10000: u32) -> i64 {
    let r = rank_per_10000(sorted.len(), per_10000);
    let i = if a < 0 { sorted.len() - 1 - r } else { r };
    a * sorted[i] + b
}

fn quantile_guard() {
    let mut rng = Rng(0x0D);
    for _ in 0..2000 {
        let n = 1 + rng.below(40) as usize;
        let mut xs: Vec<i64> = (0..n).map(|_| rng.below(1000) as i64 - 500).collect();
        xs.sort_unstable();
        let a = rng.below(9) as i64 - 4;
        let b = rng.below(100) as i64 - 50;
        let p = rng.below(10_001) as u32;
        let mut ys: Vec<i64> = xs.iter().map(|x| a * x + b).collect();
        ys.sort_unstable();
        assert_eq!(
            affine_quantile(&xs, a, b, p),
            ys[rank_per_10000(n, p)],
            "a {a} b {b} p {p}"
        );
    }
    println!("\n== 6. 1-D affine quantile: t + s·Q(rank) for s ≥ 0, t + s·Q(n−1−rank) for s < 0 — 2,000 random cases equal the sorted oracle ==");
}

fn main() {
    println!(
        "D-ART-1 algebraic recipe probe  avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    wankel_table();
    guards();
    population();
    quantiles();
    lookup_cost();
    quantile_guard();
    println!("\nall equalities and falsifiers hold");
}
