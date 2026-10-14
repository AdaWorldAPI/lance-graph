//! D-HOLO H6b: symbolic normal forms on the Context × Operation tile.
//!
//! Pre-registered in `.claude/plans/d-holo-address-indexed-algebra-v1.md`
//! (H6b, with the connectedness amendment). The 256 × 256 tile is read as
//! context (row `c`) × operation (column `o`). Five representations of a set
//! of (c, o) pairs are compared:
//!
//! - P, canonical product form: contexts grouped by identical operation mask,
//!   a list of (context mask, operation mask), 2 × 256 bits each.
//! - I, normalized Morton interval unions (as in H6), x = o, y = c.
//! - D, the dense 8 KiB mask, row-major `c * 256 + o`, operated on with
//!   `ndarray::simd::{mask_and, mask_or, mask_not, mask_any}`.
//! - H, hybrid: P while it is smaller than D, else D.
//! - O, the row oracle: the pattern's membership function on every cell.
//!
//! Every arm builds from the same membership function by sweeping the 65,536
//! cells, so construction is the generic case, not a symbolic shortcut.
//! Normalization is part of each build and of each operation: no arm has a
//! separate step that could be skipped.
//!
//! `cargo run --release -p lance-graph-mask-risc --example holo_tile_forms_probe`
//! `cargo test -p lance-graph-mask-risc --example holo_tile_forms_probe`

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::{mask_and, mask_any, mask_not, mask_or};

const SIDE: usize = 256;
const CELLS: usize = SIDE * SIDE;
const N: u32 = CELLS as u32;
const DENSE_WORDS: usize = CELLS / 64;
const DENSE_BYTES: usize = DENSE_WORDS * 8;
/// Bytes of one product-form group: a context mask and an operation mask.
const GROUP_BYTES: usize = 64;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

// ---------------------------------------------------------------- 256-bit masks

type M256 = [u64; 4];
const EMPTY: M256 = [0; 4];

fn bit(m: &M256, i: usize) -> bool {
    m[i / 64] >> (i % 64) & 1 == 1
}
fn set(m: &mut M256, i: usize) {
    m[i / 64] |= 1 << (i % 64);
}
fn and(a: &M256, b: &M256) -> M256 {
    [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
}
fn or(a: &M256, b: &M256) -> M256 {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}
fn not(a: &M256) -> M256 {
    [!a[0], !a[1], !a[2], !a[3]]
}
fn is_zero(a: &M256) -> bool {
    a.iter().all(|&w| w == 0)
}

// ---------------------------------------------------------------- P: product form

/// Canonical product form: groups with non-zero, pairwise disjoint context
/// masks and non-zero, distinct operation masks, sorted by operation mask.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Prod {
    groups: Vec<(M256, M256)>,
}

impl Prod {
    /// Canonicalize cells with disjoint context masks: drop empty ones, merge
    /// equal operation masks, sort.
    fn normalize(mut cells: Vec<(M256, M256)>) -> Prod {
        cells.retain(|(c, o)| !is_zero(c) && !is_zero(o));
        cells.sort_unstable_by_key(|a| a.1);
        let mut groups: Vec<(M256, M256)> = Vec::with_capacity(cells.len());
        for (c, o) in cells {
            match groups.last_mut() {
                Some(last) if last.1 == o => last.0 = or(&last.0, &c),
                _ => groups.push((c, o)),
            }
        }
        Prod { groups }
    }

    fn build(f: &dyn Fn(usize, usize) -> bool) -> Prod {
        let mut cells = Vec::with_capacity(SIDE);
        for c in 0..SIDE {
            let mut ops = EMPTY;
            for o in 0..SIDE {
                if f(c, o) {
                    set(&mut ops, o);
                }
            }
            let mut ctx = EMPTY;
            set(&mut ctx, c);
            cells.push((ctx, ops));
        }
        Prod::normalize(cells)
    }

    /// The groups plus the implicit group of contexts with no operations.
    fn with_rest(&self) -> Vec<(M256, M256)> {
        let mut covered = EMPTY;
        for (c, _) in &self.groups {
            covered = or(&covered, c);
        }
        let mut v = self.groups.clone();
        let rest = not(&covered);
        if !is_zero(&rest) {
            v.push((rest, EMPTY));
        }
        v
    }

    /// `f` over the common refinement of both context partitions.
    fn binop(&self, b: &Prod, f: fn(&M256, &M256) -> M256) -> Prod {
        let (ga, gb) = (self.with_rest(), b.with_rest());
        let mut cells = Vec::with_capacity(ga.len() + gb.len());
        for (ca, oa) in &ga {
            for (cb, ob) in &gb {
                let c = and(ca, cb);
                if !is_zero(&c) {
                    cells.push((c, f(oa, ob)));
                }
            }
        }
        Prod::normalize(cells)
    }

    fn intersect(&self, b: &Prod) -> Prod {
        self.binop(b, and)
    }
    fn union(&self, b: &Prod) -> Prod {
        self.binop(b, or)
    }
    fn complement(&self) -> Prod {
        let cells = self
            .with_rest()
            .into_iter()
            .map(|(c, o)| (c, not(&o)))
            .collect();
        Prod::normalize(cells)
    }
    fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
    fn contains(&self, c: usize, o: usize) -> bool {
        self.groups.iter().any(|(cm, om)| bit(cm, c) && bit(om, o))
    }
    fn bytes(&self) -> usize {
        self.groups.len() * GROUP_BYTES
    }
    fn canonical(&self) -> bool {
        let g = &self.groups;
        let disjoint = g
            .iter()
            .enumerate()
            .all(|(i, a)| g[i + 1..].iter().all(|b| is_zero(&and(&a.0, &b.0))));
        disjoint
            && g.iter().all(|(c, o)| !is_zero(c) && !is_zero(o))
            && g.windows(2).all(|w| w[0].1 < w[1].1)
    }
    fn to_dense(&self) -> Dense {
        let mut d = Dense::zero();
        for (cm, om) in &self.groups {
            for c in 0..SIDE {
                if bit(cm, c) {
                    d.w[c * 4..c * 4 + 4].copy_from_slice(om);
                }
            }
        }
        d
    }
}

// ---------------------------------------------------------------- I: Morton intervals

fn spread(v: u32) -> u32 {
    let mut x = v & 0xFF;
    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    (x | (x << 1)) & 0x5555
}
fn gather(v: u32) -> u32 {
    let mut x = v & 0x5555;
    x = (x | (x >> 1)) & 0x3333;
    x = (x | (x >> 2)) & 0x0F0F;
    (x | (x >> 4)) & 0x00FF
}
/// Morton code with x = operation, y = context.
fn morton(c: usize, o: usize) -> u32 {
    spread(o as u32) | (spread(c as u32) << 1)
}
fn unmorton(code: u32) -> (usize, usize) {
    (gather(code >> 1) as usize, gather(code) as usize)
}

/// A sorted set of disjoint, non-adjacent half-open Morton intervals.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Iv {
    s: Vec<(u32, u32)>,
}

fn push(out: &mut Vec<(u32, u32)>, lo: u32, hi: u32) {
    if lo >= hi {
        return;
    }
    if let Some(last) = out.last_mut() {
        if last.1 >= lo {
            last.1 = last.1.max(hi);
            return;
        }
    }
    out.push((lo, hi));
}

impl Iv {
    fn build(f: &dyn Fn(usize, usize) -> bool) -> Iv {
        let mut s = Vec::new();
        for code in 0..N {
            let (c, o) = unmorton(code);
            if f(c, o) {
                push(&mut s, code, code + 1);
            }
        }
        Iv { s }
    }
    fn union(&self, b: &Iv) -> Iv {
        let (a, b) = (&self.s, &b.s);
        let mut out = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() || j < b.len() {
            let take_a = j >= b.len() || (i < a.len() && a[i].0 <= b[j].0);
            let (lo, hi) = if take_a { a[i] } else { b[j] };
            if take_a {
                i += 1;
            } else {
                j += 1;
            }
            push(&mut out, lo, hi);
        }
        Iv { s: out }
    }
    fn intersect(&self, b: &Iv) -> Iv {
        let (a, b) = (&self.s, &b.s);
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            let lo = a[i].0.max(b[j].0);
            let hi = a[i].1.min(b[j].1);
            push(&mut out, lo, hi);
            if a[i].1 < b[j].1 {
                i += 1;
            } else {
                j += 1;
            }
        }
        Iv { s: out }
    }
    fn complement(&self) -> Iv {
        let mut out = Vec::with_capacity(self.s.len() + 1);
        let mut at = 0;
        for &(lo, hi) in &self.s {
            push(&mut out, at, lo);
            at = hi;
        }
        push(&mut out, at, N);
        Iv { s: out }
    }
    fn is_empty(&self) -> bool {
        self.s.is_empty()
    }
    fn contains(&self, c: usize, o: usize) -> bool {
        let code = morton(c, o);
        let k = self.s.partition_point(|&(_, hi)| hi <= code);
        k < self.s.len() && self.s[k].0 <= code
    }
    fn bytes(&self) -> usize {
        self.s.len() * 8
    }
    fn canonical(&self) -> bool {
        self.s.iter().all(|&(lo, hi)| lo < hi && hi <= N)
            && self.s.windows(2).all(|w| w[0].1 < w[1].0)
    }
}

// ---------------------------------------------------------------- D: dense mask

#[derive(Clone, PartialEq, Eq, Debug)]
struct Dense {
    w: Box<[u64]>,
}

impl Dense {
    fn zero() -> Dense {
        Dense {
            w: vec![0u64; DENSE_WORDS].into_boxed_slice(),
        }
    }
    fn build(f: &dyn Fn(usize, usize) -> bool) -> Dense {
        let mut d = Dense::zero();
        for c in 0..SIDE {
            for o in 0..SIDE {
                if f(c, o) {
                    let i = c * SIDE + o;
                    d.w[i / 64] |= 1 << (i % 64);
                }
            }
        }
        d
    }
    fn intersect(&self, b: &Dense) -> Dense {
        let mut d = Dense::zero();
        mask_and(&self.w, &b.w, &mut d.w);
        d
    }
    fn union(&self, b: &Dense) -> Dense {
        let mut d = Dense::zero();
        mask_or(&self.w, &b.w, &mut d.w);
        d
    }
    fn complement(&self) -> Dense {
        let mut d = Dense::zero();
        mask_not(&self.w, CELLS, &mut d.w);
        d
    }
    fn is_empty(&self) -> bool {
        !mask_any(&self.w)
    }
    fn contains(&self, c: usize, o: usize) -> bool {
        let i = c * SIDE + o;
        self.w[i / 64] >> (i % 64) & 1 == 1
    }
    fn bytes(&self) -> usize {
        DENSE_BYTES
    }
}

// ---------------------------------------------------------------- H: hybrid

#[derive(Clone, Debug)]
enum Hyb {
    P(Prod),
    D(Dense),
}

impl Hyb {
    fn settle(p: Prod) -> Hyb {
        if p.bytes() < DENSE_BYTES {
            Hyb::P(p)
        } else {
            Hyb::D(p.to_dense())
        }
    }
    fn build(f: &dyn Fn(usize, usize) -> bool) -> Hyb {
        Hyb::settle(Prod::build(f))
    }
    fn dense(&self) -> Dense {
        match self {
            Hyb::P(p) => p.to_dense(),
            Hyb::D(d) => d.clone(),
        }
    }
    fn binop(&self, b: &Hyb, pf: fn(&Prod, &Prod) -> Prod, df: fn(&Dense, &Dense) -> Dense) -> Hyb {
        match (self, b) {
            (Hyb::P(x), Hyb::P(y)) => Hyb::settle(pf(x, y)),
            (Hyb::D(x), Hyb::D(y)) => Hyb::D(df(x, y)),
            _ => Hyb::D(df(&self.dense(), &b.dense())),
        }
    }
    fn intersect(&self, b: &Hyb) -> Hyb {
        self.binop(b, Prod::intersect, Dense::intersect)
    }
    fn union(&self, b: &Hyb) -> Hyb {
        self.binop(b, Prod::union, Dense::union)
    }
    fn complement(&self) -> Hyb {
        match self {
            Hyb::P(p) => Hyb::settle(p.complement()),
            Hyb::D(d) => Hyb::D(d.complement()),
        }
    }
    fn is_empty(&self) -> bool {
        match self {
            Hyb::P(p) => p.is_empty(),
            Hyb::D(d) => d.is_empty(),
        }
    }
    fn equal(&self, b: &Hyb) -> bool {
        match (self, b) {
            (Hyb::P(x), Hyb::P(y)) => x == y,
            _ => self.dense() == b.dense(),
        }
    }
    fn contains(&self, c: usize, o: usize) -> bool {
        match self {
            Hyb::P(p) => p.contains(c, o),
            Hyb::D(d) => d.contains(c, o),
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Hyb::P(p) => p.bytes(),
            Hyb::D(d) => d.bytes(),
        }
    }
    fn is_dense(&self) -> bool {
        matches!(self, Hyb::D(_))
    }
}

// ---------------------------------------------------------------- the query

/// The per-query formula: X = (A ∩ ¬B) ∪ (B ∩ ¬A), then emptiness of X and
/// equality of X with A. Exercises ∩, ∪ and ¬, plus both tests.
macro_rules! query {
    ($a:expr, $b:expr, $eq:expr) => {{
        let (a, b) = ($a, $b);
        let x = a
            .intersect(&b.complement())
            .union(&b.intersect(&a.complement()));
        let e = x.is_empty();
        let q = $eq(&x, a);
        (x, e, q)
    }};
}

// ---------------------------------------------------------------- patterns

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn coin(seed: u64, c: usize, o: usize, per_10k: u64) -> bool {
    mix(seed ^ ((c * SIDE + o) as u64)) % 10_000 < per_10k
}
fn in_rect(c: usize, o: usize, c0: usize, c1: usize, o0: usize, o1: usize) -> bool {
    (c0..c1).contains(&c) && (o0..o1).contains(&o)
}

type Cell = fn(usize, usize) -> bool;

struct Pattern {
    name: &'static str,
    structured: bool,
    adversarial: bool,
    a: Cell,
    b: Cell,
}

fn patterns() -> Vec<Pattern> {
    let p = |name, structured, adversarial, a: Cell, b: Cell| Pattern {
        name,
        structured,
        adversarial,
        a,
        b,
    };
    vec![
        p(
            "row",
            true,
            false,
            |c, o| c == 37 && (10..200).contains(&o),
            |c, o| c == 37 && (100..250).contains(&o),
        ),
        p(
            "column",
            true,
            false,
            |c, o| o == 91 && (5..230).contains(&c),
            |c, o| o == 91 && (40..256).contains(&c),
        ),
        p(
            "rectangle",
            true,
            false,
            |c, o| in_rect(c, o, 20, 120, 30, 170),
            |c, o| in_rect(c, o, 60, 200, 100, 240),
        ),
        p(
            "prefix",
            true,
            false,
            |c, o| morton(c, o) >> 12 == 3,
            |c, o| morton(c, o) >> 14 == 0,
        ),
        p("diagonal", false, false, |c, o| c == o, |c, o| c == o + 1),
        p(
            "checkerboard",
            false,
            true,
            |c, o| (c + o) % 2 == 0,
            |c, o| (c ^ o) & 2 == 0,
        ),
        p(
            "random 1%",
            false,
            false,
            |c, o| coin(1, c, o, 100),
            |c, o| coin(2, c, o, 100),
        ),
        p(
            "random 50%",
            false,
            true,
            |c, o| coin(3, c, o, 5_000),
            |c, o| coin(4, c, o, 5_000),
        ),
        p(
            "sparse",
            false,
            false,
            |c, o| coin(5, c, o, 10),
            |c, o| coin(6, c, o, 10),
        ),
        p(
            "dense",
            false,
            false,
            |c, o| !coin(7, c, o, 10),
            |c, o| !coin(8, c, o, 10),
        ),
        p(
            "complementary",
            true,
            false,
            |c, o| in_rect(c, o, 20, 120, 30, 170),
            |c, o| !in_rect(c, o, 20, 120, 30, 170),
        ),
        p(
            "disjoint",
            true,
            false,
            |c, o| in_rect(c, o, 0, 100, 0, 100),
            |c, o| in_rect(c, o, 150, 256, 120, 256),
        ),
        p(
            "fragmented",
            false,
            true,
            |c, o| o < c,
            |c, o| o < c * 3 / 4,
        ),
        p(
            "doughnut",
            false,
            false,
            |c, o| in_rect(c, o, 40, 200, 40, 200) && !in_rect(c, o, 90, 150, 90, 150),
            |c, o| in_rect(c, o, 100, 240, 0, 256),
        ),
    ]
}

/// 4-connected components of a pattern on the 256 × 256 grid.
fn components(f: Cell) -> usize {
    let mut seen = vec![false; CELLS];
    let mut stack = Vec::new();
    let mut n = 0;
    for start in 0..CELLS {
        if seen[start] || !f(start / SIDE, start % SIDE) {
            continue;
        }
        n += 1;
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (c, o) = (i / SIDE, i % SIDE);
            let mut try_cell = |c2: usize, o2: usize| {
                let j = c2 * SIDE + o2;
                if !seen[j] && f(c2, o2) {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if c > 0 {
                try_cell(c - 1, o);
            }
            if c + 1 < SIDE {
                try_cell(c + 1, o);
            }
            if o > 0 {
                try_cell(c, o - 1);
            }
            if o + 1 < SIDE {
                try_cell(c, o + 1);
            }
        }
    }
    n
}

/// The oracle answer for the query: X's membership, emptiness, equality.
fn oracle(p: &Pattern) -> (Vec<bool>, bool, bool) {
    let x: Vec<bool> = (0..CELLS)
        .map(|i| {
            let (c, o) = (i / SIDE, i % SIDE);
            let (a, b) = ((p.a)(c, o), (p.b)(c, o));
            a != b
        })
        .collect();
    let empty = x.iter().all(|&v| !v);
    let equal = (0..CELLS).all(|i| x[i] == (p.a)(i / SIDE, i % SIDE));
    (x, empty, equal)
}

// ---------------------------------------------------------------- timing

fn time<F: FnMut()>(mut f: F) -> f64 {
    f();
    let mut v: Vec<f64> = (0..11)
        .map(|_| {
            let t = Instant::now();
            let mut n = 0u32;
            while n < 3 || t.elapsed().as_micros() < 2_000 {
                f();
                n += 1;
            }
            t.elapsed().as_nanos() as f64 / f64::from(n)
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[5]
}

/// Spearman rank correlation (average ranks for ties).
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0;
            for &k in &idx[i..=j] {
                r[k] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (rx, ry) = (ranks(x), ranks(y));
    let n = x.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let vx: f64 = rx.iter().map(|a| (a - mx).powi(2)).sum();
    let vy: f64 = ry.iter().map(|b| (b - my).powi(2)).sum();
    cov / (vx * vy).sqrt()
}

const ARMS: [&str; 4] = ["P", "I", "D", "H"];
const NS: [f64; 4] = [1.0, 10.0, 100.0, 1000.0];

struct Row {
    name: &'static str,
    structured: bool,
    adversarial: bool,
    density: f64,
    comps: usize,
    intervals: usize,
    groups: usize,
    hyb_dense: bool,
    /// Per arm: build (A + B), query, ∩, ∪, ¬, empty, equal, bytes of A.
    build: [f64; 4],
    query: [f64; 4],
    ops: [[f64; 5]; 4],
    bytes: [usize; 4],
}

fn measure(p: &Pattern) -> Row {
    let fa: &dyn Fn(usize, usize) -> bool = &p.a;
    let fb: &dyn Fn(usize, usize) -> bool = &p.b;
    let (pa, pb) = (Prod::build(fa), Prod::build(fb));
    let (ia, ib) = (Iv::build(fa), Iv::build(fb));
    let (da, db) = (Dense::build(fa), Dense::build(fb));
    let (ha, hb) = (Hyb::build(fa), Hyb::build(fb));

    let build = [
        time(|| {
            black_box((Prod::build(fa), Prod::build(fb)));
        }),
        time(|| {
            black_box((Iv::build(fa), Iv::build(fb)));
        }),
        time(|| {
            black_box((Dense::build(fa), Dense::build(fb)));
        }),
        time(|| {
            black_box((Hyb::build(fa), Hyb::build(fb)));
        }),
    ];
    let query = [
        time(|| {
            black_box(query!(
                black_box(&pa),
                black_box(&pb),
                |x: &Prod, y: &Prod| x == y
            ));
        }),
        time(|| {
            black_box(query!(black_box(&ia), black_box(&ib), |x: &Iv, y: &Iv| x == y));
        }),
        time(|| {
            black_box(query!(
                black_box(&da),
                black_box(&db),
                |x: &Dense, y: &Dense| x == y
            ));
        }),
        time(|| {
            black_box(query!(
                black_box(&ha),
                black_box(&hb),
                |x: &Hyb, y: &Hyb| x.equal(y)
            ));
        }),
    ];
    let ops = [
        [
            time(|| {
                black_box(black_box(&pa).intersect(black_box(&pb)));
            }),
            time(|| {
                black_box(black_box(&pa).union(black_box(&pb)));
            }),
            time(|| {
                black_box(black_box(&pa).complement());
            }),
            time(|| {
                black_box(black_box(&pa).is_empty());
            }),
            time(|| {
                black_box(black_box(&pa) == black_box(&pb));
            }),
        ],
        [
            time(|| {
                black_box(black_box(&ia).intersect(black_box(&ib)));
            }),
            time(|| {
                black_box(black_box(&ia).union(black_box(&ib)));
            }),
            time(|| {
                black_box(black_box(&ia).complement());
            }),
            time(|| {
                black_box(black_box(&ia).is_empty());
            }),
            time(|| {
                black_box(black_box(&ia) == black_box(&ib));
            }),
        ],
        [
            time(|| {
                black_box(black_box(&da).intersect(black_box(&db)));
            }),
            time(|| {
                black_box(black_box(&da).union(black_box(&db)));
            }),
            time(|| {
                black_box(black_box(&da).complement());
            }),
            time(|| {
                black_box(black_box(&da).is_empty());
            }),
            time(|| {
                black_box(black_box(&da) == black_box(&db));
            }),
        ],
        [
            time(|| {
                black_box(black_box(&ha).intersect(black_box(&hb)));
            }),
            time(|| {
                black_box(black_box(&ha).union(black_box(&hb)));
            }),
            time(|| {
                black_box(black_box(&ha).complement());
            }),
            time(|| {
                black_box(black_box(&ha).is_empty());
            }),
            time(|| {
                black_box(black_box(&ha).equal(black_box(&hb)));
            }),
        ],
    ];
    let members = (0..CELLS).filter(|&i| (p.a)(i / SIDE, i % SIDE)).count();
    Row {
        name: p.name,
        structured: p.structured,
        adversarial: p.adversarial,
        density: members as f64 / CELLS as f64,
        comps: components(p.a),
        intervals: ia.s.len(),
        groups: pa.groups.len(),
        hyb_dense: ha.is_dense(),
        build,
        query,
        ops,
        bytes: [pa.bytes(), ia.bytes(), da.bytes(), ha.bytes()],
    }
}

fn total(r: &Row, arm: usize, n: f64) -> f64 {
    r.build[arm] + n * r.query[arm]
}

fn main() {
    println!("D-HOLO H6b: normal forms on the 256 × 256 Context × Operation tile");
    println!("tier: {}", tier());
    println!("timings: ns, median of 11; every arm checked against the row oracle first\n");

    let pats = patterns();
    for p in &pats {
        let (x, e, q) = oracle(p);
        let fa: &dyn Fn(usize, usize) -> bool = &p.a;
        let fb: &dyn Fn(usize, usize) -> bool = &p.b;
        let (pa, pb) = (Prod::build(fa), Prod::build(fb));
        let (ia, ib) = (Iv::build(fa), Iv::build(fb));
        let (da, db) = (Dense::build(fa), Dense::build(fb));
        let (ha, hb) = (Hyb::build(fa), Hyb::build(fb));
        let (px, pe, pq) = query!(&pa, &pb, |x: &Prod, y: &Prod| x == y);
        let (ix, ie, iq) = query!(&ia, &ib, |x: &Iv, y: &Iv| x == y);
        let (dx, de, dq) = query!(&da, &db, |x: &Dense, y: &Dense| x == y);
        let (hx, he, hq) = query!(&ha, &hb, |x: &Hyb, y: &Hyb| x.equal(y));
        assert!(
            px.canonical() && ix.canonical(),
            "{}: not canonical",
            p.name
        );
        assert_eq!((pe, pq), (e, q), "{}: P", p.name);
        assert_eq!((ie, iq), (e, q), "{}: I", p.name);
        assert_eq!((de, dq), (e, q), "{}: D", p.name);
        assert_eq!((he, hq), (e, q), "{}: H", p.name);
        for (i, &want) in x.iter().enumerate() {
            let (c, o) = (i / SIDE, i % SIDE);
            assert_eq!(px.contains(c, o), want, "{}: P cell {i}", p.name);
            assert_eq!(ix.contains(c, o), want, "{}: I cell {i}", p.name);
            assert_eq!(dx.contains(c, o), want, "{}: D cell {i}", p.name);
            assert_eq!(hx.contains(c, o), want, "{}: H cell {i}", p.name);
        }
    }
    println!(
        "oracle: all {} patterns match in all four arms\n",
        pats.len()
    );

    let rows: Vec<Row> = pats.iter().map(measure).collect();

    println!("pattern       density comps intervals groups H-form | bytes P / I / D / H");
    for r in &rows {
        println!(
            "{:<13} {:>7.4} {:>5} {:>9} {:>6} {:>6} | {} / {} / {} / {}",
            r.name,
            r.density,
            r.comps,
            r.intervals,
            r.groups,
            if r.hyb_dense { "D" } else { "P" },
            r.bytes[0],
            r.bytes[1],
            r.bytes[2],
            r.bytes[3]
        );
    }

    println!("\nper-operation ns (A op B): ∩ / ∪ / ¬A / empty(A) / A==B");
    for r in &rows {
        for (k, arm) in ARMS.iter().enumerate() {
            let o = r.ops[k];
            println!(
                "{:<13} {arm} {:>9.0} {:>9.0} {:>9.0} {:>7.1} {:>7.1}",
                r.name, o[0], o[1], o[2], o[3], o[4]
            );
        }
    }

    println!("\nbuild (A + B) and query ns per arm: P / I / D / H");
    for r in &rows {
        println!(
            "{:<13} build {:>9.0} {:>9.0} {:>9.0} {:>9.0} | query {:>9.0} {:>9.0} {:>9.0} {:>9.0}",
            r.name,
            r.build[0],
            r.build[1],
            r.build[2],
            r.build[3],
            r.query[0],
            r.query[1],
            r.query[2],
            r.query[3]
        );
    }

    println!("\ntotal ns = build + N × query, ratio to D at each N: P / I / H");
    for r in &rows {
        let mut line = format!("{:<13}", r.name);
        for &n in &NS {
            let d = total(r, 2, n);
            line += &format!(
                " | N={:<4} {:>5.2} {:>5.2} {:>5.2}",
                n as u32,
                total(r, 0, n) / d,
                total(r, 1, n) / d,
                total(r, 3, n) / d
            );
        }
        println!("{line}");
    }

    println!("\ndecision rules at N = 100 (ratio to D; < 2× either way is undecided)");
    for r in &rows {
        let d = total(r, 2, 100.0);
        let (p, i, h) = (
            total(r, 0, 100.0) / d,
            total(r, 1, 100.0) / d,
            total(r, 3, 100.0) / d,
        );
        let rule = if r.structured {
            format!(
                "structured: P {} (≤ 0.5), H {}",
                verdict_le(p, 0.5),
                verdict_le(h, 0.5)
            )
        } else if r.adversarial {
            format!("adversarial: H {} (≤ 1.25)", verdict_le(h, 1.25))
        } else {
            String::from("reported only")
        };
        let best = [(p, "P"), (i, "I"), (1.0, "D"), (h, "H")]
            .into_iter()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map_or("?", |t| t.1);
        println!(
            "{:<13} P {p:>5.2} I {i:>5.2} H {h:>5.2} cheapest {best:<2} {rule}",
            r.name
        );
    }

    let ratio: Vec<f64> = rows
        .iter()
        .map(|r| (total(r, 0, 100.0).min(total(r, 1, 100.0)) / total(r, 2, 100.0)).ln())
        .collect();
    let comps: Vec<f64> = rows.iter().map(|r| r.comps as f64).collect();
    let dens: Vec<f64> = rows
        .iter()
        .map(|r| r.density.min(1.0 - r.density))
        .collect();
    let ivs: Vec<f64> = rows.iter().map(|r| r.intervals as f64).collect();
    println!(
        "\npredictors of ln(best symbolic / D) at N = 100, Spearman over {} patterns",
        rows.len()
    );
    println!(
        "  components                 {:+.2}",
        spearman(&comps, &ratio)
    );
    println!(
        "  density (min(d, 1 - d))    {:+.2}",
        spearman(&dens, &ratio)
    );
    println!(
        "  Morton intervals           {:+.2}",
        spearman(&ivs, &ratio)
    );

    // Not pre-registered: added after the first run showed that the generic
    // 65,536-cell build dominates every total at N = 100.
    let grp: Vec<f64> = rows.iter().map(|r| r.groups as f64).collect();
    let qratio: Vec<f64> = rows
        .iter()
        .map(|r| (r.query[0] / r.query[2]).ln())
        .collect();
    println!("\npost hoc (not pre-registered): query-only ratio to D, P / I / H");
    for r in &rows {
        println!(
            "{:<13} {:>6.2} {:>6.2} {:>6.2}",
            r.name,
            r.query[0] / r.query[2],
            r.query[1] / r.query[2],
            r.query[3] / r.query[2]
        );
    }
    println!(
        "  Spearman, P groups vs ln(P query / D query)      {:+.2}",
        spearman(&grp, &qratio)
    );
    println!(
        "  Spearman, components vs ln(P query / D query)    {:+.2}",
        spearman(&comps, &qratio)
    );
    println!(
        "  Spearman, density vs ln(P query / D query)       {:+.2}",
        spearman(&dens, &qratio)
    );
}

fn verdict_le(x: f64, bound: f64) -> &'static str {
    if x <= bound {
        "PASS"
    } else if x < 2.0 && x > 0.5 {
        "undecided"
    } else {
        "FAIL"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle_set(f: Cell) -> Vec<bool> {
        (0..CELLS).map(|i| f(i / SIDE, i % SIDE)).collect()
    }

    /// FAILS IF: any arm's ∩, ∪, ¬ or query result differs from the row
    /// oracle on any cell of any pattern, or a P / I result is not canonical.
    #[test]
    fn every_arm_matches_the_oracle_on_every_pattern() {
        for p in patterns() {
            let fa: &dyn Fn(usize, usize) -> bool = &p.a;
            let fb: &dyn Fn(usize, usize) -> bool = &p.b;
            let (pa, pb) = (Prod::build(fa), Prod::build(fb));
            let (ia, ib) = (Iv::build(fa), Iv::build(fb));
            let (da, db) = (Dense::build(fa), Dense::build(fb));
            let (ha, hb) = (Hyb::build(fa), Hyb::build(fb));
            let (pi, pu, pn) = (pa.intersect(&pb), pa.union(&pb), pa.complement());
            let (ii, iu, inn) = (ia.intersect(&ib), ia.union(&ib), ia.complement());
            let (di, du, dn) = (da.intersect(&db), da.union(&db), da.complement());
            let (hi, hu, hn) = (ha.intersect(&hb), ha.union(&hb), ha.complement());
            for x in [&pi, &pu, &pn] {
                assert!(x.canonical(), "{}: P not canonical", p.name);
            }
            for x in [&ii, &iu, &inn] {
                assert!(x.canonical(), "{}: I not canonical", p.name);
            }
            for i in 0..CELLS {
                let (c, o) = (i / SIDE, i % SIDE);
                let (a, b) = ((p.a)(c, o), (p.b)(c, o));
                let want = [a && b, a || b, !a];
                assert_eq!(
                    [pi.contains(c, o), pu.contains(c, o), pn.contains(c, o)],
                    want,
                    "{} P",
                    p.name
                );
                assert_eq!(
                    [ii.contains(c, o), iu.contains(c, o), inn.contains(c, o)],
                    want,
                    "{} I",
                    p.name
                );
                assert_eq!(
                    [di.contains(c, o), du.contains(c, o), dn.contains(c, o)],
                    want,
                    "{} D",
                    p.name
                );
                assert_eq!(
                    [hi.contains(c, o), hu.contains(c, o), hn.contains(c, o)],
                    want,
                    "{} H",
                    p.name
                );
            }
            let (x, e, q) = oracle(&p);
            let (px, pe, pq) = query!(&pa, &pb, |x: &Prod, y: &Prod| x == y);
            let (hx, he, hq) = query!(&ha, &hb, |x: &Hyb, y: &Hyb| x.equal(y));
            assert_eq!((pe, pq, he, hq), (e, q, e, q), "{}: query flags", p.name);
            for (i, &want) in x.iter().enumerate() {
                assert_eq!(px.contains(i / SIDE, i % SIDE), want);
                assert_eq!(hx.contains(i / SIDE, i % SIDE), want);
            }
        }
    }

    /// FAILS IF: P is not canonical, i.e. the same set reached by two routes
    /// gives two different group lists, so equality would lie. A ∩ ⊤ and ¬¬A
    /// must equal A exactly.
    #[test]
    fn product_form_is_canonical_across_routes() {
        let full = Prod::build(&|_, _| true);
        for p in patterns() {
            let a = Prod::build(&p.a);
            assert_eq!(a.intersect(&full), a, "{}: A ∩ ⊤", p.name);
            assert_eq!(a.complement().complement(), a, "{}: ¬¬A", p.name);
            let via_union = a
                .intersect(&Prod::build(&p.b))
                .union(&a.intersect(&Prod::build(&p.b).complement()));
            assert_eq!(via_union, a, "{}: (A∩B) ∪ (A∩¬B)", p.name);
        }
    }

    /// FAILS IF: the fixture is degenerate: some pattern is empty or full, no
    /// query is empty, every query is empty, or the components column carries
    /// no spread (so the connectedness predictor could not discriminate).
    #[test]
    fn fixture_discriminates() {
        let pats = patterns();
        let mut empties = 0;
        let mut comps = Vec::new();
        for p in &pats {
            let a = oracle_set(p.a).iter().filter(|&&v| v).count();
            assert!(a > 0 && a < CELLS, "{}: A is empty or full", p.name);
            if oracle(p).1 {
                empties += 1;
            }
            comps.push(components(p.a));
        }
        assert!(empties < pats.len(), "every query is empty");
        assert!(comps.iter().min() == Some(&1) && *comps.iter().max().unwrap() > 1_000);
        // The doughnut is one component with a hole: connectedness alone does
        // not make it an interval set.
        let d = pats.iter().find(|p| p.name == "doughnut").unwrap();
        assert_eq!(components(d.a), 1);
    }

    /// FAILS IF: the hybrid never switches, or switches everywhere: it must
    /// stay P on a rectangle and fall back to D on a random 50% pattern.
    #[test]
    fn hybrid_switches_both_ways() {
        let pats = patterns();
        let get = |n: &str| pats.iter().find(|p| p.name == n).unwrap();
        assert!(!Hyb::build(&get("rectangle").a).is_dense());
        assert!(Hyb::build(&get("random 50%").a).is_dense());
    }

    /// FAILS IF: the Morton decoding is not the inverse of the encoding.
    #[test]
    fn morton_round_trips() {
        for c in 0..SIDE {
            for o in 0..SIDE {
                assert_eq!(unmorton(morton(c, o)), (c, o));
            }
        }
    }
}
