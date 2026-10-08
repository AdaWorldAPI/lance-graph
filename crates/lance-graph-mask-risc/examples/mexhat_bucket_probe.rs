//! D-MHB-1 — Mexican-hat response without a raster: does popcount stacking
//! (rings or bit-sliced weight planes) beat direct per-point geometry, and how
//! much does an exact-bound early exit save on a threshold decision?
//!
//! Nothing here is a production primitive. The question is answered for ONE
//! query shape: the centre-surround response at a query cell `c` of a
//! presence bitmap `P` over a row-major `W × H` grid,
//!
//! ```text
//! S(c) = Σ_{p ∈ P, |p − c|² ≤ R²} w(|p − c|²)
//! ```
//!
//! where `w` is a normalised Difference-of-Gaussians quantised to integers
//! (`w(q) = round(DoG(√q) / DoG(0) · 2^SHIFT)`, evaluated on `q = r²`).
//!
//! Three semantics, never mixed:
//!
//! - **continuous**: `f64` DoG, the reference the quantisation is measured
//!   against (arm A);
//! - **quantised**: integer `w(q)` per lattice offset. Arms Ascan, D, F, G and B
//!   must agree on it EXACTLY, and the probe asserts that they do;
//! - **bucketed**: `K` radius² rings, one representative weight each (arm E).
//!   A further approximation, measured against the quantised answer.
//!
//! | arm | execution |
//! |---|---|
//! | A     | `f64` DoG over present points in the window (continuous reference) |
//! | Ascan | every set bit of the WHOLE grid, `q`, LUT (full candidate scan) |
//! | D     | set bits in the window only, `q = dx² + dy²`, LUT |
//! | E     | `K` ring templates: `Σ_k w_k · popcount(win ∧ ring_k)` |
//! | F     | bit-sliced weight planes: `Σ_b 2^b · (popcnt(win ∧ Pos_b) − popcnt(win ∧ Neg_b))` |
//! | G     | per row, the cheaper of D and F (both exact): one popcount decides |
//! | B     | shipped mask-risc: per-query weight lane over the row band, `MaskedSumI32` over `P` |
//! | H     | early exit for `S ≥ T` over F's rows, with exact suffix bounds `[L, U]` |
//!
//! Timings are printed, never asserted. Every equality is asserted.
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-mask-risc --example mexhat_bucket_probe
//! ```

// The template builders index window rows and columns as 2-D coordinates
// (`dy = yi − R`, `dx = xi − R`) into several tables at once; an iterator per
// table would hide the geometry the loops exist to express.
#![allow(clippy::needless_range_loop)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{Foreign, LaneRef, Operand, Out, Planes, Program, Terminal, Value};

// ───────────────────────── allocation counter ─────────────────────────

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

// ───────────────────────── rng ─────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ───────────────────────── the kernel ─────────────────────────

/// Weight scale: `w(0) = 2^SHIFT`.
const SHIFT: u32 = 12;
/// Bit planes needed for `|w| ≤ 2^SHIFT`.
const PLANES: usize = SHIFT as usize + 1;

/// Normalised DoG at squared radius `q` (no square root needed).
fn dog(q: f64, sc: f64, ss: f64) -> f64 {
    let g = |s: f64| (-q / (2.0 * s * s)).exp() / (2.0 * std::f64::consts::PI * s * s);
    g(sc) - g(ss)
}

/// Analytic zero crossing of the normalised DoG, as a squared radius:
/// `r0² = 2 σc² σs² ln(σs²/σc²) / (σs² − σc²)`.
fn zero_crossing_q(sc: f64, ss: f64) -> f64 {
    let (a, b) = (sc * sc, ss * ss);
    2.0 * a * b * (b / a).ln() / (b - a)
}

struct Kernel {
    sc: f64,
    ss: f64,
    /// Cutoff radius; the window is `2R + 1` cells wide and must fit a `u64`.
    r: usize,
    r2: usize,
    /// Quantised weight per squared radius `0..=R²`.
    lut: Vec<i32>,
    /// Per window row `dy + R`: the disk cells (`q ≤ R²`) as a `u64` over `dx + R`.
    disk: Vec<u64>,
    /// Per window row: bit-sliced positive and negative magnitude planes.
    pos: Vec<[u64; PLANES]>,
    neg: Vec<[u64; PLANES]>,
    /// Per window row: the largest positive / most negative contribution the
    /// row could still make (all its positive / negative cells present).
    row_up: Vec<i64>,
    row_lo: Vec<i64>,
    /// Per window row: how many of its `2 · PLANES` planes are non-zero,
    /// i.e. what arm F pays for the row in popcounts.
    planes_used: Vec<u32>,
}

impl Kernel {
    fn new(sc: f64, kappa: f64) -> Self {
        let ss = sc * kappa;
        let r = (3.0 * ss).ceil() as usize;
        assert!(2 * r < 64, "window must fit one u64 (R = {r})");
        let r2 = r * r;
        let peak = dog(0.0, sc, ss);
        let lut: Vec<i32> = (0..=r2)
            .map(|q| (dog(q as f64, sc, ss) / peak * f64::from(1u32 << SHIFT)).round() as i32)
            .collect();
        let wd = 2 * r + 1;
        let (mut disk, mut pos, mut neg, mut row_up, mut row_lo) = (
            vec![0u64; wd],
            vec![[0u64; PLANES]; wd],
            vec![[0u64; PLANES]; wd],
            vec![0i64; wd],
            vec![0i64; wd],
        );
        for yi in 0..wd {
            let dy = yi as i64 - r as i64;
            for xi in 0..wd {
                let dx = xi as i64 - r as i64;
                let q = (dx * dx + dy * dy) as usize;
                if q > r2 {
                    continue;
                }
                disk[yi] |= 1 << xi;
                let w = lut[q];
                let (planes, mag) = if w >= 0 {
                    (&mut pos[yi], w)
                } else {
                    (&mut neg[yi], -w)
                };
                for (b, p) in planes.iter_mut().enumerate() {
                    if (mag >> b) & 1 == 1 {
                        *p |= 1 << xi;
                    }
                }
                if w > 0 {
                    row_up[yi] += i64::from(w);
                } else {
                    row_lo[yi] += i64::from(w);
                }
            }
        }
        let planes_used = (0..wd)
            .map(|yi| {
                (0..PLANES)
                    .map(|b| u32::from(pos[yi][b] != 0) + u32::from(neg[yi][b] != 0))
                    .sum()
            })
            .collect();
        Kernel {
            sc,
            ss,
            r,
            r2,
            lut,
            disk,
            pos,
            neg,
            row_up,
            row_lo,
            planes_used,
        }
    }

    fn wd(&self) -> usize {
        2 * self.r + 1
    }
}

/// `K` radius² rings over `0..=R²` with one representative weight each: the
/// mean of the quantised weights of the lattice cells the ring contains.
struct Rings {
    /// `ring[k][row]`: ring `k`'s cells in window row `row`.
    ring: Vec<Vec<u64>>,
    w: Vec<i64>,
}

fn rings(k: &Kernel, n: usize) -> Rings {
    // Equal-width rings in q. Edge rule: cell with q goes to ring
    // `min(q * n / (R² + 1), n − 1)` — one ring per q, deterministically.
    let ring_of = |q: usize| (q * n / (k.r2 + 1)).min(n - 1);
    let wd = k.wd();
    let mut ring = vec![vec![0u64; wd]; n];
    let (mut sum, mut cnt) = (vec![0i64; n], vec![0i64; n]);
    for yi in 0..wd {
        let dy = yi as i64 - k.r as i64;
        for xi in 0..wd {
            let dx = xi as i64 - k.r as i64;
            let q = (dx * dx + dy * dy) as usize;
            if q > k.r2 {
                continue;
            }
            let b = ring_of(q);
            ring[b][yi] |= 1 << xi;
            sum[b] += i64::from(k.lut[q]);
            cnt[b] += 1;
        }
    }
    let w = sum
        .iter()
        .zip(&cnt)
        .map(|(s, c)| {
            if *c == 0 {
                0
            } else {
                (*s as f64 / *c as f64).round() as i64
            }
        })
        .collect();
    Rings { ring, w }
}

// ───────────────────────── the grid ─────────────────────────

struct Grid {
    w: usize,
    h: usize,
    /// Row-major presence, `w / 64` words per row.
    p: Vec<u64>,
}

impl Grid {
    fn words_per_row(&self) -> usize {
        self.w / 64
    }
    fn set(&mut self, x: usize, y: usize) {
        let i = y * self.w + x;
        self.p[i / 64] |= 1 << (i % 64);
    }
    /// `len ≤ 63` bits of row `y` starting at column `x0`, bit `j` = column `x0 + j`.
    fn window(&self, x0: usize, y: usize, len: usize) -> u64 {
        let base = y * self.words_per_row();
        let (wi, sh) = (x0 / 64, x0 % 64);
        let lo = self.p[base + wi] >> sh;
        let v = if sh != 0 && sh + len > 64 {
            lo | (self.p[base + wi + 1] << (64 - sh))
        } else {
            lo
        };
        v & ((1u64 << len) - 1)
    }
}

fn random_grid(w: usize, h: usize, rho: f64, seed: u64) -> Grid {
    let mut g = Grid {
        w,
        h,
        p: vec![0; w * h / 64],
    };
    let mut r = Rng(seed);
    for y in 0..h {
        for x in 0..w {
            if r.unit() < rho {
                g.set(x, y);
            }
        }
    }
    g
}

/// A wavefront: one ring of radius `r0` and width ~1 cell around the grid
/// centre, plus `noise` density elsewhere.
fn wavefront_grid(w: usize, h: usize, r0: f64, noise: f64, seed: u64) -> Grid {
    let mut g = random_grid(w, h, noise, seed);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            if (d - r0).abs() < 0.5 {
                g.set(x, y);
            }
        }
    }
    g
}

// ───────────────────────── arms ─────────────────────────

#[derive(Default, Clone, Copy)]
struct Work {
    /// `q` computations (one per visited present point).
    geometry: u64,
    /// `exp` evaluations.
    exps: u64,
    popcounts: u64,
    rows: u64,
}

/// Arm A, continuous: `f64` DoG over present points of the window.
fn arm_a(g: &Grid, k: &Kernel, cx: usize, cy: usize, wk: &mut Work) -> f64 {
    let peak = dog(0.0, k.sc, k.ss);
    let mut s = 0.0;
    for yi in 0..k.wd() {
        let y = cy + yi - k.r;
        let dy = yi as i64 - k.r as i64;
        let mut m = g.window(cx - k.r, y, k.wd());
        while m != 0 {
            let xi = m.trailing_zeros() as i64;
            m &= m - 1;
            let dx = xi - k.r as i64;
            let q = (dx * dx + dy * dy) as usize;
            wk.geometry += 1;
            if q <= k.r2 {
                wk.exps += 2;
                s += dog(q as f64, k.sc, k.ss) / peak * f64::from(1u32 << SHIFT);
            }
        }
    }
    s
}

/// Arm Ascan: every set bit of the whole grid.
fn arm_scan(g: &Grid, k: &Kernel, cx: usize, cy: usize, wk: &mut Work) -> i64 {
    let mut s = 0i64;
    for (wi, word) in g.p.iter().enumerate() {
        let mut m = *word;
        while m != 0 {
            let i = wi * 64 + m.trailing_zeros() as usize;
            m &= m - 1;
            let (x, y) = ((i % g.w) as i64, (i / g.w) as i64);
            let (dx, dy) = (x - cx as i64, y - cy as i64);
            wk.geometry += 1;
            let q = dx * dx + dy * dy;
            if q <= k.r2 as i64 {
                s += i64::from(k.lut[q as usize]);
            }
        }
    }
    s
}

/// Arm D: set bits of the window, `q`, LUT.
fn arm_d(g: &Grid, k: &Kernel, cx: usize, cy: usize, wk: &mut Work) -> i64 {
    let mut s = 0i64;
    for yi in 0..k.wd() {
        let dy = yi as i64 - k.r as i64;
        let mut m = g.window(cx - k.r, cy + yi - k.r, k.wd()) & k.disk[yi];
        while m != 0 {
            let dx = m.trailing_zeros() as i64 - k.r as i64;
            m &= m - 1;
            wk.geometry += 1;
            s += i64::from(k.lut[(dx * dx + dy * dy) as usize]);
        }
    }
    s
}

/// Arm E: ring popcounts times the ring's representative weight.
fn arm_e(g: &Grid, k: &Kernel, rg: &Rings, cx: usize, cy: usize, wk: &mut Work) -> i64 {
    let mut s = 0i64;
    for yi in 0..k.wd() {
        let win = g.window(cx - k.r, cy + yi - k.r, k.wd());
        for (ring, w) in rg.ring.iter().zip(&rg.w) {
            let r = ring[yi];
            if r != 0 {
                wk.popcounts += 1;
                s += w * i64::from((win & r).count_ones());
            }
        }
    }
    s
}

/// One window row of arm F: the row's exact quantised contribution.
#[inline]
fn f_row(k: &Kernel, win: u64, yi: usize, wk: &mut Work) -> i64 {
    let mut s = 0i64;
    for b in 0..PLANES {
        let (p, n) = (k.pos[yi][b], k.neg[yi][b]);
        if p != 0 {
            wk.popcounts += 1;
            s += i64::from((win & p).count_ones()) << b;
        }
        if n != 0 {
            wk.popcounts += 1;
            s -= i64::from((win & n).count_ones()) << b;
        }
    }
    s
}

/// Arm F: bit-sliced weight planes, exact for the quantised semantics.
fn arm_f(g: &Grid, k: &Kernel, cx: usize, cy: usize, wk: &mut Work) -> i64 {
    (0..k.wd())
        .map(|yi| {
            wk.rows += 1;
            f_row(k, g.window(cx - k.r, cy + yi - k.r, k.wd()), yi, wk)
        })
        .sum()
}

/// Arm G: per row, the cheaper of the two exact executions. A row with fewer
/// present disk cells than non-zero planes takes D's per-point path,
/// otherwise F's popcounts. Both are exact, so the choice cannot change the
/// answer; the probe asserts that it does not.
fn arm_g(g: &Grid, k: &Kernel, cx: usize, cy: usize, wk: &mut Work) -> i64 {
    let mut s = 0i64;
    for yi in 0..k.wd() {
        let win = g.window(cx - k.r, cy + yi - k.r, k.wd());
        let mut m = win & k.disk[yi];
        wk.popcounts += 1;
        if m.count_ones() <= k.planes_used[yi] {
            let dy = yi as i64 - k.r as i64;
            while m != 0 {
                let dx = m.trailing_zeros() as i64 - k.r as i64;
                m &= m - 1;
                wk.geometry += 1;
                s += i64::from(k.lut[(dx * dx + dy * dy) as usize]);
            }
        } else {
            wk.rows += 1;
            s += f_row(k, win, yi, wk);
        }
    }
    s
}

/// Present-cell crossover between D and F, in cells per window.
///
/// A pin for this host (D-MHB-1 finding 3: ρ ≈ 0.1 of a 1,009-cell disk),
/// not a derived constant. `crossover_sweep` measures how sensitive arm Q is
/// to it.
const DF_CROSSOVER_CELLS: u32 = 100;

/// One window row of arm D: the row's present disk cells through the LUT.
#[inline]
fn d_row(k: &Kernel, win: u64, yi: usize, wk: &mut Work) -> i64 {
    let dy = yi as i64 - k.r as i64;
    let mut m = win & k.disk[yi];
    let mut s = 0i64;
    while m != 0 {
        let dx = m.trailing_zeros() as i64 - k.r as i64;
        m &= m - 1;
        wk.geometry += 1;
        s += i64::from(k.lut[(dx * dx + dy * dy) as usize]);
    }
    s
}

/// The window's words, fetched once, and its exact present-cell count.
#[inline]
fn window_words(g: &Grid, k: &Kernel, cx: usize, cy: usize, wk: &mut Work) -> ([u64; 64], u32) {
    let mut w = [0u64; 64];
    let mut n = 0u32;
    for (yi, slot) in w.iter_mut().enumerate().take(k.wd()) {
        *slot = g.window(cx - k.r, cy + yi - k.r, k.wd());
        wk.popcounts += 1;
        n += (*slot & k.disk[yi]).count_ones();
    }
    (w, n)
}

/// Arm Q: one choice per query. The window's exact present-cell count picks
/// D below the crossover and F above it; the words are fetched once and
/// shared by the count and the chosen arm.
fn arm_q(g: &Grid, k: &Kernel, cx: usize, cy: usize, cut: u32, wk: &mut Work) -> i64 {
    let (w, n) = window_words(g, k, cx, cy, wk);
    if n <= cut {
        (0..k.wd()).map(|yi| d_row(k, w[yi], yi, wk)).sum()
    } else {
        (0..k.wd())
            .map(|yi| {
                wk.rows += 1;
                f_row(k, w[yi], yi, wk)
            })
            .sum()
    }
}

/// Rows in the order arm H visits them: largest `|contribution|` bound first.
fn h_order(k: &Kernel) -> Vec<usize> {
    let mut o: Vec<usize> = (0..k.wd()).collect();
    o.sort_by_key(|&yi| std::cmp::Reverse(k.row_up[yi] - k.row_lo[yi]));
    o
}

/// Suffix bounds over `order`: `up[i]` / `lo[i]` bound what rows `order[i..]`
/// can still add.
fn h_suffix(k: &Kernel, order: &[usize]) -> (Vec<i64>, Vec<i64>) {
    let n = order.len();
    let (mut up, mut lo) = (vec![0i64; n + 1], vec![0i64; n + 1]);
    for i in (0..n).rev() {
        up[i] = up[i + 1] + k.row_up[order[i]];
        lo[i] = lo[i + 1] + k.row_lo[order[i]];
    }
    (up, lo)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Decision {
    Accept,
    Reject,
}

/// Arm H: decide `S ≥ t` over F's rows, stopping as soon as the exact suffix
/// bounds settle it. Returns the decision and the rows visited.
#[allow(clippy::too_many_arguments)]
fn arm_h(
    g: &Grid,
    k: &Kernel,
    order: &[usize],
    up: &[i64],
    lo: &[i64],
    cx: usize,
    cy: usize,
    t: i64,
    wk: &mut Work,
) -> (Decision, usize) {
    let mut s = 0i64;
    for i in 0..=order.len() {
        // Rows order[i..] can add at most up[i] and at least lo[i].
        if s + up[i] < t {
            return (Decision::Reject, i);
        }
        if s + lo[i] >= t {
            return (Decision::Accept, i);
        }
        if i == order.len() {
            break;
        }
        let yi = order[i];
        wk.rows += 1;
        s += f_row(k, g.window(cx - k.r, cy + yi - k.r, k.wd()), yi, wk);
    }
    unreachable!("with no rows left the bounds are [0, 0], so one of the two tests fired");
}

/// The row evaluator an early exit runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowEval {
    /// F's popcounts on every row (arm H).
    F,
    /// D's per-point path on every row (arm HD).
    D,
    /// One choice per query from the window count (arm HQ).
    Query(u32),
}

/// Arms HD and HQ: arm H's exact suffix-bound early exit over any exact row
/// evaluator. The bounds are the template's, so they hold for every
/// evaluator; only the cost of a visited row changes. HQ pays the full
/// window count up front, which the early exit cannot then save.
#[allow(clippy::too_many_arguments)]
fn arm_hx(
    g: &Grid,
    k: &Kernel,
    order: &[usize],
    up: &[i64],
    lo: &[i64],
    cx: usize,
    cy: usize,
    t: i64,
    ev: RowEval,
    wk: &mut Work,
) -> (Decision, usize) {
    let use_d = match ev {
        RowEval::F => false,
        RowEval::D => true,
        RowEval::Query(cut) => window_words(g, k, cx, cy, wk).1 <= cut,
    };
    let mut s = 0i64;
    for i in 0..=order.len() {
        if s + up[i] < t {
            return (Decision::Reject, i);
        }
        if s + lo[i] >= t {
            return (Decision::Accept, i);
        }
        if i == order.len() {
            break;
        }
        let yi = order[i];
        let win = g.window(cx - k.r, cy + yi - k.r, k.wd());
        wk.rows += 1;
        s += if use_d {
            d_row(k, win, yi, wk)
        } else {
            f_row(k, win, yi, wk)
        };
    }
    unreachable!("with no rows left the bounds are [0, 0], so one of the two tests fired");
}

// ───────────────────────── harness ─────────────────────────

fn centres(g: &Grid, k: &Kernel, n: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| (k.r + r.below(g.w - 2 * k.r), k.r + r.below(g.h - 2 * k.r)))
        .collect()
}

fn time<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut last = f();
    let t = Instant::now();
    for _ in 0..reps {
        last = f();
    }
    (last, t.elapsed().as_nanos() as f64 / reps as f64)
}

fn pct(v: &mut [usize], p: f64) -> usize {
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p) as usize]
}

// ───────────────────────── sections ─────────────────────────

/// Kernel falsifiers: sign, single zero crossing at the analytic radius,
/// single annular minimum, symmetry, ring partition, no overflow.
fn kernel_checks() {
    println!("== kernel falsifiers ==");
    for &(sc, kappa) in &[
        (2.0, 1.5),
        (3.0, 1.5),
        (2.0, 2.0),
        (4.0, 2.0),
        (2.0, 3.0),
        (3.0, 3.0),
    ] {
        let k = Kernel::new(sc, kappa);
        // Positive centre, negative surround.
        assert!(k.lut[0] > 0, "centre must be positive");
        assert!(k.lut.iter().any(|w| *w < 0), "surround must go negative");
        // Exactly one sign change from + to − over q (zeros are a plateau, not a crossing).
        let signs: Vec<i32> = k
            .lut
            .iter()
            .map(|w| w.signum())
            .filter(|s| *s != 0)
            .collect();
        let changes = signs.windows(2).filter(|p| p[0] != p[1]).count();
        assert_eq!(changes, 1, "σc {sc} κ {kappa}: {changes} sign changes");
        // The crossing sits within one q-step of the analytic radius².
        let q0 = zero_crossing_q(k.sc, k.ss);
        let last_pos = k.lut.iter().rposition(|w| *w > 0).unwrap() as f64;
        let first_neg = k.lut.iter().position(|w| *w < 0).unwrap() as f64;
        assert!(
            last_pos <= q0 + 1.0 && q0 - 1.0 <= first_neg,
            "crossing {q0:.2} vs [{last_pos}, {first_neg}]"
        );
        // One annular minimum: non-increasing to the minimum, non-decreasing after.
        let qmin = (0..=k.r2).min_by_key(|q| (k.lut[*q], *q)).unwrap();
        let down = k.lut[..=qmin].windows(2).filter(|p| p[1] > p[0]).count();
        let up = k.lut[qmin..].windows(2).filter(|p| p[1] < p[0]).count();
        assert_eq!((down, up), (0, 0), "σc {sc} κ {kappa}: false extremum");
        // The 8 square symmetries: every template row is a palindrome, and
        // row yi equals column yi.
        let wd = k.wd();
        for yi in 0..wd {
            assert_eq!(k.disk[yi].reverse_bits() >> (64 - wd), k.disk[yi]);
            for xi in 0..wd {
                assert_eq!((k.disk[yi] >> xi) & 1, (k.disk[xi] >> yi) & 1);
            }
        }
        // Ring partition: disjoint, and their union is the disk.
        for n in [4, 8, 16, 32] {
            let rg = rings(&k, n);
            for yi in 0..wd {
                let mut union = 0u64;
                for ring in &rg.ring {
                    assert_eq!(union & ring[yi], 0, "rings overlap");
                    union |= ring[yi];
                }
                assert_eq!(union, k.disk[yi], "rings do not cover the disk");
            }
        }
        // No overflow: the worst case is every positive or every negative cell.
        let up: i64 = k.row_up.iter().sum();
        let lo: i64 = k.row_lo.iter().sum();
        assert!(up < i64::from(i32::MAX) && lo > i64::from(i32::MIN));
        println!(
            "  σc {sc} κ {kappa}: R {:>2}  zero crossing q {q0:6.2} (lattice between {last_pos} and {first_neg})  minimum at q {qmin}  bounds [{lo}, {up}]",
            k.r
        );
    }
    println!("  sign, one crossing at the analytic radius, one minimum, symmetry, ring partition, no overflow: all hold");
}

/// Equality and approximation of every arm on one grid.
fn agreement(name: &str, g: &Grid, k: &Kernel, qs: &[(usize, usize)]) {
    let mut wk = Work::default();
    let rgs: Vec<(usize, Rings)> = [8, 16, 32].iter().map(|n| (*n, rings(k, *n))).collect();
    let (mut err_q, mut norm) = (0.0f64, 0.0f64);
    let mut err_e = vec![0.0f64; rgs.len()];
    for &(cx, cy) in qs {
        let d = arm_d(g, k, cx, cy, &mut wk);
        assert_eq!(
            arm_f(g, k, cx, cy, &mut wk),
            d,
            "F differs from D at ({cx},{cy})"
        );
        assert_eq!(
            arm_g(g, k, cx, cy, &mut wk),
            d,
            "G differs from D at ({cx},{cy})"
        );
        for cut in [0, DF_CROSSOVER_CELLS, u32::MAX] {
            assert_eq!(
                arm_q(g, k, cx, cy, cut, &mut wk),
                d,
                "Q (cut {cut}) differs from D at ({cx},{cy})"
            );
        }
        let a = arm_a(g, k, cx, cy, &mut wk);
        err_q += (a - d as f64).abs();
        norm += a.abs().max(1.0);
        for (i, (_, rg)) in rgs.iter().enumerate() {
            err_e[i] += (arm_e(g, k, rg, cx, cy, &mut wk) - d).abs() as f64;
        }
    }
    // The full scan is O(grid) per query; check it on a few centres.
    for &(cx, cy) in qs.iter().take(8) {
        assert_eq!(
            arm_scan(g, k, cx, cy, &mut wk),
            arm_d(g, k, cx, cy, &mut wk)
        );
    }
    let n = qs.len() as f64;
    println!(
        "  {name}: Ascan = D = F = G = Q on every checked centre; |quantised − continuous| mean {:.2} ({:.3} % of |S|)  |E − D| mean: {}",
        err_q / n,
        100.0 * err_q / norm,
        rgs.iter()
            .zip(&err_e)
            .map(|((n_r, _), e)| format!("K={n_r} {:.1}", e / n))
            .collect::<Vec<_>>()
            .join("  ")
    );
}

struct Line {
    name: String,
    ns: f64,
    wk: Work,
    queries: usize,
}

fn line(l: &Line) {
    let q = l.queries as f64;
    println!(
        "    {:<26} {:>10.0} ns/query  geometry {:>8.1}  exp {:>7.1}  popcount {:>7.1}  rows {:>5.1}",
        l.name,
        l.ns,
        l.wk.geometry as f64 / q,
        l.wk.exps as f64 / q,
        l.wk.popcounts as f64 / q,
        l.wk.rows as f64 / q,
    );
}

fn bench(name: &str, g: &Grid, k: &Kernel, qs: &[(usize, usize)], scan_queries: usize) {
    println!("  -- {name} --");
    let reps = 5;
    let run = |label: &str, f: &mut dyn FnMut(&mut Work, usize, usize) -> i64, nq: usize| {
        let mut wk = Work::default();
        let a0 = allocs();
        let (_, ns) = time(reps, || {
            let mut acc = 0i64;
            let mut w = Work::default();
            for &(cx, cy) in &qs[..nq] {
                acc = acc.wrapping_add(f(&mut w, cx, cy));
            }
            wk = w;
            std::hint::black_box(acc)
        });
        let a1 = allocs();
        assert_eq!(a1, a0, "{label} allocated");
        line(&Line {
            name: label.into(),
            ns: ns / nq as f64,
            wk,
            queries: nq,
        });
    };
    let nq = qs.len();
    run(
        "A   f64 DoG (continuous)",
        &mut |w, x, y| arm_a(g, k, x, y, w) as i64,
        nq,
    );
    run(
        "Ascan full candidate scan",
        &mut |w, x, y| arm_scan(g, k, x, y, w),
        scan_queries,
    );
    run(
        "D   window bits + LUT",
        &mut |w, x, y| arm_d(g, k, x, y, w),
        nq,
    );
    for n in [8, 32] {
        let rg = rings(k, n);
        run(
            &format!("E   rings K={n}"),
            &mut |w, x, y| arm_e(g, k, &rg, x, y, w),
            nq,
        );
    }
    run(
        "F   bit-sliced planes",
        &mut |w, x, y| arm_f(g, k, x, y, w),
        nq,
    );
    run(
        "G   per-row cheaper of D/F",
        &mut |w, x, y| arm_g(g, k, x, y, w),
        nq,
    );
    run(
        "Q   per-query D/F choice",
        &mut |w, x, y| arm_q(g, k, x, y, DF_CROSSOVER_CELLS, w),
        nq,
    );

    // B: the shipped executor. The weight lane is filled per query over the
    // row band `cy − R ..= cy + R` (full width; zero outside the disk), then
    // `MaskedSumI32` over the resident presence plane on that extent.
    let n = g.w * g.h;
    let mut lane = vec![0i32; n];
    let program = Program::new(
        vec![],
        Terminal::MaskedSumI32 {
            mask: Operand::Plane(0),
            lane: 0,
        },
    );
    let mut scratch = Scratch::for_program(&program, n).expect("scratch");
    let mut written = 0u64;
    let a0 = allocs();
    let t = Instant::now();
    for _ in 0..reps {
        written = 0;
        for &(cx, cy) in qs {
            let (y0, y1) = (cy - k.r, cy + k.r + 1);
            for y in y0..y1 {
                let dy = y as i64 - cy as i64;
                for x in 0..g.w {
                    let dx = x as i64 - cx as i64;
                    let q = dx * dx + dy * dy;
                    lane[y * g.w + x] = if q <= k.r2 as i64 {
                        k.lut[q as usize]
                    } else {
                        0
                    };
                }
            }
            written += ((y1 - y0) * g.w * 4) as u64;
            let masks: [&[u64]; 1] = [&g.p];
            let lanes = [LaneRef::I32(&lane)];
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &lanes,
            };
            let v = execute_extent(
                &program,
                &planes,
                &Foreign::NONE,
                &mut scratch,
                Out::None,
                y0 * g.w..y1 * g.w,
            );
            let Ok(Value::SumI64(s)) = v else {
                panic!("B: {v:?}")
            };
            assert_eq!(
                s,
                arm_d(g, k, cx, cy, &mut Work::default()),
                "B differs from D"
            );
        }
    }
    let ns = t.elapsed().as_nanos() as f64 / (reps * qs.len()) as f64;
    let a1 = allocs();
    println!(
        "    {:<26} {:>10.0} ns/query  weight lane written {:>8} B/query  resident lane {} B  allocs/query {:.1}",
        "B   mask-risc MaskedSumI32",
        ns,
        written / qs.len() as u64,
        n * 4,
        (a1 - a0) as f64 / (reps * qs.len()) as f64
    );
}

/// Arm H against the full answer at several selectivities.
fn early_exit(name: &str, g: &Grid, k: &Kernel, qs: &[(usize, usize)]) {
    let order = h_order(k);
    let (up, lo) = h_suffix(k, &order);
    let full: Vec<i64> = qs
        .iter()
        .map(|&(x, y)| arm_f(g, k, x, y, &mut Work::default()))
        .collect();
    let mut sorted = full.clone();
    sorted.sort_unstable();
    println!(
        "  -- {name}: early exit for S ≥ T (rows in |bound| order, {} rows max) --",
        k.wd()
    );
    for p in [0.10, 0.50, 0.90, 0.99] {
        let t = sorted[((sorted.len() - 1) as f64 * p) as usize];
        let mut rows = Vec::with_capacity(qs.len());
        let mut accepted = 0;
        for (&(cx, cy), &s) in qs.iter().zip(&full) {
            let (d, r) = arm_h(g, k, &order, &up, &lo, cx, cy, t, &mut Work::default());
            let want = if s >= t {
                Decision::Accept
            } else {
                Decision::Reject
            };
            assert_eq!(d, want, "H decided {d:?} at ({cx},{cy}), S {s}, T {t}");
            for ev in [RowEval::D, RowEval::Query(DF_CROSSOVER_CELLS)] {
                let (dx, rx) = arm_hx(g, k, &order, &up, &lo, cx, cy, t, ev, &mut Work::default());
                assert_eq!(
                    dx, want,
                    "{ev:?} decided {dx:?} at ({cx},{cy}), S {s}, T {t}"
                );
                assert_eq!(rx, r, "{ev:?} visited a different number of rows");
            }
            accepted += usize::from(d == Decision::Accept);
            rows.push(r);
        }
        let mean = rows.iter().sum::<usize>() as f64 / rows.len() as f64;
        println!(
            "    T at the {:>2.0} % quantile ({t:>7}): accepted {:>5.1} %  rows visited mean {mean:5.1}  p95 {:>2}  of {}",
            p * 100.0,
            100.0 * accepted as f64 / qs.len() as f64,
            pct(&mut rows, 0.95),
            k.wd()
        );
    }
    // Timing at the median threshold, against the full F.
    let t = sorted[sorted.len() / 2];
    let (_, ns_h) = time(5, || {
        qs.iter()
            .map(|&(x, y)| arm_h(g, k, &order, &up, &lo, x, y, t, &mut Work::default()).1)
            .sum::<usize>()
    });
    let (_, ns_f) = time(5, || {
        qs.iter()
            .map(|&(x, y)| arm_f(g, k, x, y, &mut Work::default()))
            .sum::<i64>()
    });
    let mut ns_x = Vec::new();
    for ev in [RowEval::D, RowEval::Query(DF_CROSSOVER_CELLS)] {
        let (_, ns) = time(5, || {
            qs.iter()
                .map(|&(x, y)| arm_hx(g, k, &order, &up, &lo, x, y, t, ev, &mut Work::default()).1)
                .sum::<usize>()
        });
        ns_x.push(ns / qs.len() as f64);
    }
    let (_, ns_d) = time(5, || {
        qs.iter()
            .map(|&(x, y)| arm_d(g, k, x, y, &mut Work::default()))
            .sum::<i64>()
    });
    let (_, ns_q) = time(5, || {
        qs.iter()
            .map(|&(x, y)| arm_q(g, k, x, y, DF_CROSSOVER_CELLS, &mut Work::default()))
            .sum::<i64>()
    });
    let n = qs.len() as f64;
    println!(
        "    median T: H {:.0}  HD {:.0}  HQ {:.0}  ns/query  against full F {:.0}  D {:.0}  Q {:.0}",
        ns_h / n,
        ns_x[0],
        ns_x[1],
        ns_f / n,
        ns_d / n,
        ns_q / n
    );
}

/// Arm Q's sensitivity to its one pin: the cut swept around the crossover on
/// every density, against the better of D and F measured on the same queries.
fn crossover_sweep(k: &Kernel) {
    println!("  cut sweep, 1M grid, 2,048 queries; ns/query (Q / min(D, F))");
    let cuts = [25u32, 50, 100, 200, 400];
    print!("    {:<8}", "ρ");
    for c in cuts {
        print!(" {:>12}", format!("cut {c}"));
    }
    println!("  {:>8} {:>8}", "D", "F");
    for rho in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9] {
        let g = random_grid(1024, 1024, rho, 0xC0 + (rho * 100.0) as u64);
        let qs = centres(&g, k, 2048, 0xC1);
        let med = |f: &dyn Fn() -> i64| {
            let mut v: Vec<f64> = (0..5).map(|_| time(1, f).1).collect();
            v.sort_by(f64::total_cmp);
            v[2] / qs.len() as f64
        };
        let d = med(&|| {
            qs.iter()
                .map(|&(x, y)| arm_d(&g, k, x, y, &mut Work::default()))
                .sum()
        });
        let f = med(&|| {
            qs.iter()
                .map(|&(x, y)| arm_f(&g, k, x, y, &mut Work::default()))
                .sum()
        });
        print!("    {rho:<8}");
        for c in cuts {
            let q = med(&|| {
                qs.iter()
                    .map(|&(x, y)| arm_q(&g, k, x, y, c, &mut Work::default()))
                    .sum()
            });
            print!(" {:>12}", format!("{q:.0} / {:.2}", q / d.min(f)));
        }
        println!("  {d:>8.0} {f:>8.0}");
    }
}

/// Counterexamples the early exit must survive.
fn counterexamples(k: &Kernel) {
    println!("== counterexamples ==");
    let order = h_order(k);
    let (up, lo) = h_suffix(k, &order);
    let (w, h) = (256, 256);
    let (cx, cy) = (128, 128);

    // 1. A partial sum that rises above T and ends below it. The positive
    //    core is present, and negative cells only in rows the visit order
    //    reaches late (|dy| > the zero-crossing radius), so the running sum
    //    climbs first and falls after. The bounds must keep H undecided
    //    while the sum sits above T.
    let q0 = zero_crossing_q(k.sc, k.ss);
    let mut g = Grid {
        w,
        h,
        p: vec![0; w * h / 64],
    };
    for yi in 0..k.wd() {
        for xi in 0..k.wd() {
            let (dx, dy) = (xi as i64 - k.r as i64, yi as i64 - k.r as i64);
            let q = (dx * dx + dy * dy) as usize;
            let core = q <= k.r2 && k.lut[q] > 0;
            let late_surround = q <= k.r2 && k.lut[q] < 0 && (dy * dy) as f64 > q0;
            if core || late_surround {
                g.set(cx + xi - k.r, cy + yi - k.r);
            }
        }
    }
    let s = arm_f(&g, k, cx, cy, &mut Work::default());
    let (mut run, mut peak) = (0i64, i64::MIN);
    for &yi in &order {
        run += f_row(
            k,
            g.window(cx - k.r, cy + yi - k.r, k.wd()),
            yi,
            &mut Work::default(),
        );
        peak = peak.max(run);
    }
    let t = (peak + s) / 2;
    assert!(
        s < t && t < peak,
        "the fixture must rise above T and end below it (S {s}, T {t}, peak {peak})"
    );
    let (d, rows) = arm_h(&g, k, &order, &up, &lo, cx, cy, t, &mut Work::default());
    assert_eq!(d, Decision::Reject);
    println!(
        "  sign change during evaluation: running sum peaks at {peak}, above T = {t}, and ends at S = {s}; H rejects after {rows} of {} rows",
        k.wd()
    );

    // 2. Cancellation: one ring exactly at the zero crossing. S ≈ 0.
    let mut g = Grid {
        w,
        h,
        p: vec![0; w * h / 64],
    };
    for yi in 0..k.wd() {
        for xi in 0..k.wd() {
            let (dx, dy) = (xi as i64 - k.r as i64, yi as i64 - k.r as i64);
            if ((dx * dx + dy * dy) as f64 - q0).abs() < q0.sqrt() {
                g.set(cx + xi - k.r, cy + yi - k.r);
            }
        }
    }
    let s = arm_f(&g, k, cx, cy, &mut Work::default());
    let (_, rows) = arm_h(&g, k, &order, &up, &lo, cx, cy, 1, &mut Work::default());
    println!(
        "  cancellation ring at the zero crossing: S = {s}, decision S ≥ 1 needs {rows} of {} rows",
        k.wd()
    );

    // 3. Dense, threshold at the dense mean: no early exit to be had.
    let g = random_grid(w, h, 0.9, 0xD3);
    let qs = centres(&g, k, 512, 0xC3);
    let mut full: Vec<i64> = qs
        .iter()
        .map(|&(x, y)| arm_f(&g, k, x, y, &mut Work::default()))
        .collect();
    full.sort_unstable();
    let t = full[full.len() / 2];
    let rows: usize = qs
        .iter()
        .map(|&(x, y)| arm_h(&g, k, &order, &up, &lo, x, y, t, &mut Work::default()).1)
        .sum();
    println!(
        "  dense ρ = 0.9, T at the median: rows visited mean {:.1} of {} (the bound cannot help here)",
        rows as f64 / qs.len() as f64,
        k.wd()
    );

    // 4. Disable run, in place: an upper bound that ignores the unvisited
    //    rows must decide wrongly somewhere. Proves the bound is load-bearing.
    let g = random_grid(w, h, 0.2, 0xD4);
    let qs = centres(&g, k, 512, 0xC4);
    let zero_up = vec![0i64; up.len()];
    let mut wrong = 0;
    let s_all: Vec<i64> = qs
        .iter()
        .map(|&(x, y)| arm_f(&g, k, x, y, &mut Work::default()))
        .collect();
    let mut sorted = s_all.clone();
    sorted.sort_unstable();
    let t = sorted[sorted.len() * 9 / 10];
    assert!(t > 0, "the disable run needs a positive threshold (T {t})");
    for (&(x, y), &s) in qs.iter().zip(&s_all) {
        let (d, _) = arm_h(&g, k, &order, &zero_up, &lo, x, y, t, &mut Work::default());
        wrong += usize::from((d == Decision::Accept) != (s >= t));
    }
    assert!(
        wrong > 0,
        "a bound that drops the unvisited rows must be caught"
    );
    println!("  disable run: upper bound without the unvisited rows decides {wrong} of {} queries wrongly", qs.len());
}

/// A centre-surround response over geometry cannot see phase: two coherent
/// sources, detector cells bucketed by (r1, r2). With buckets as wide as λ/2,
/// bright and dark fringes share buckets, so pruning by bucket would drop
/// dark fringes along with "uninteresting" cells.
fn interference() {
    println!("== interference: geometry buckets versus phase ==");
    let lambda = 8.0;
    let (s1, s2) = ((96.0, 128.0), (160.0, 128.0));
    for width in [lambda / 2.0, lambda / 8.0] {
        let mut cls: std::collections::HashMap<(i64, i64), (bool, bool)> =
            std::collections::HashMap::new();
        let (mut dark, mut bright) = (0usize, 0usize);
        for y in 0..256 {
            for x in 0..256 {
                let (xf, yf) = (f64::from(x), f64::from(y));
                let r1 = ((xf - s1.0).powi(2) + (yf - s1.1).powi(2)).sqrt();
                let r2 = ((xf - s2.0).powi(2) + (yf - s2.1).powi(2)).sqrt();
                // Equal-amplitude coherent sources: I / 4I0 = cos²(πΔr/λ).
                let i = (std::f64::consts::PI * (r1 - r2) / lambda).cos().powi(2);
                let key = ((r1 / width) as i64, (r2 / width) as i64);
                let e = cls.entry(key).or_default();
                if i < 0.1 {
                    e.0 = true;
                    dark += 1;
                }
                if i > 0.9 {
                    e.1 = true;
                    bright += 1;
                }
            }
        }
        let mixed = cls.values().filter(|(d, b)| *d && *b).count();
        let either = cls.values().filter(|(d, b)| *d || *b).count();
        println!(
            "  bucket width {:>4.1} (λ = {lambda}): {mixed} of {either} occupied (r1, r2) buckets hold both a dark and a bright cell ({dark} dark, {bright} bright cells)",
            width
        );
        if width >= lambda / 2.0 {
            assert!(mixed > 0, "λ/2 buckets must mix fringes");
        }
    }
}

/// Words a (2R+1)² window touches: row-major against Morton order.
fn locality(k: &Kernel) {
    fn morton(x: u32, y: u32) -> u64 {
        let spread = |v: u32| {
            let mut v = u64::from(v);
            v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
            v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
            v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
            v = (v | (v << 2)) & 0x3333_3333_3333_3333;
            (v | (v << 1)) & 0x5555_5555_5555_5555
        };
        spread(x) | (spread(y) << 1)
    }
    let (w, h) = (1024usize, 1024usize);
    let g = Grid { w, h, p: vec![] };
    let qs = centres(&g, k, 2048, 0x10C);
    let (mut rm, mut mo) = (0usize, 0usize);
    for &(cx, cy) in &qs {
        let mut a = std::collections::HashSet::new();
        let mut b = std::collections::HashSet::new();
        for y in cy - k.r..=cy + k.r {
            for x in cx - k.r..=cx + k.r {
                a.insert((y * w + x) / 64);
                b.insert(morton(x as u32, y as u32) / 64);
            }
        }
        rm += a.len();
        mo += b.len();
    }
    let n = qs.len() as f64;
    println!("== locality: u64 words a {0}×{0} window touches ==", k.wd());
    println!(
        "  row-major {:.1}  Morton {:.1}  (the window cells are {})",
        rm as f64 / n,
        mo as f64 / n,
        k.wd() * k.wd()
    );
}

fn main() {
    println!(
        "D-MHB-1 Mexican-hat bucket probe  avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    kernel_checks();

    let k = Kernel::new(3.0, 2.0);
    println!(
        "\nkernel σc {} σs {} R {} window {}×{}  disk cells {}",
        k.sc,
        k.ss,
        k.r,
        k.wd(),
        k.wd(),
        k.disk.iter().map(|r| r.count_ones()).sum::<u32>()
    );

    println!("\n== agreement (quantised arms are asserted equal) ==");
    for (w, h, label) in [(256usize, 256usize, "64K"), (1024, 1024, "1M")] {
        for rho in [0.01, 0.1, 0.5, 0.9] {
            let g = random_grid(w, h, rho, 0xA0 + (rho * 100.0) as u64);
            let qs = centres(&g, &k, 1024, 0xB0);
            agreement(&format!("{label} ρ {rho}"), &g, &k, &qs);
        }
    }

    println!("\n== speed (ns per query; work counted per query) ==");
    for (w, h, label, scan) in [(256usize, 256usize, "64K", 64usize), (1024, 1024, "1M", 8)] {
        for rho in [0.01, 0.1, 0.5, 0.9] {
            let g = random_grid(w, h, rho, 0xA0 + (rho * 100.0) as u64);
            let qs = centres(&g, &k, 2048, 0xB1);
            bench(&format!("{label} ρ {rho}"), &g, &k, &qs, scan);
        }
        let g = wavefront_grid(w, h, w as f64 / 4.0, 0.01, 0xE0);
        let qs = centres(&g, &k, 2048, 0xB2);
        bench(
            &format!("{label} wavefront ring + 1 % noise"),
            &g,
            &k,
            &qs,
            scan,
        );
    }

    println!("\n== early exit ==");
    for rho in [0.01, 0.1, 0.5, 0.9] {
        let g = random_grid(1024, 1024, rho, 0xA0 + (rho * 100.0) as u64);
        let qs = centres(&g, &k, 4096, 0xB3);
        early_exit(&format!("1M ρ {rho}"), &g, &k, &qs);
    }
    let g = wavefront_grid(1024, 1024, 256.0, 0.01, 0xE0);
    let qs = centres(&g, &k, 4096, 0xB4);
    early_exit("1M wavefront ring + 1 % noise", &g, &k, &qs);

    println!("\n== per-query D/F choice ==");
    crossover_sweep(&k);

    println!();
    counterexamples(&k);
    println!();
    interference();
    println!();
    locality(&k);
    println!("\nall equalities and falsifiers hold");
}
