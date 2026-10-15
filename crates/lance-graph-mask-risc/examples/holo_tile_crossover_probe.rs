//! D-HOLO H6c: the group-count crossover, a group-count switch for the hybrid,
//! and construction from a declaration.
//!
//! Pre-registered in `.claude/plans/d-holo-address-indexed-algebra-v1.md`
//! (§ Next falsifier: H6c), after H6b. The tile, the four arms (P, I, D, H),
//! the query and the row oracle are copied unchanged from
//! `holo_tile_forms_probe.rs`; the hybrid gains a switch policy.
//!
//! - Q1: fixtures with exactly g distinct operation masks, blocked or
//!   interleaved over the contexts; P vs D per query as g grows.
//! - Q2: the hybrid switching on P's group count (G = 2..32) beside the byte
//!   switch of H6b, over the 14 H6b patterns and the Q1 family.
//! - Q3: every arm built from the same declaration (rectangles or a Morton
//!   prefix) instead of a 65,536-cell sweep; H6b's structured rule rerun.
//!
//! `cargo run --release -p lance-graph-mask-risc --example holo_tile_crossover_probe`
//! `cargo test -p lance-graph-mask-risc --example holo_tile_crossover_probe`

// The arms and patterns are copied whole from the H6b probe; this probe does
// not use every method or field of them.
#![allow(dead_code)]

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

/// When the hybrid keeps the product form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Switch {
    /// H6b: P while it takes fewer bytes than the dense mask.
    Bytes,
    /// H6c: P while it has at most this many groups.
    Groups(usize),
}

impl Switch {
    fn keeps(self, p: &Prod) -> bool {
        match self {
            Switch::Bytes => p.bytes() < DENSE_BYTES,
            Switch::Groups(g) => p.groups.len() <= g,
        }
    }
    fn label(self) -> String {
        match self {
            Switch::Bytes => String::from("bytes"),
            Switch::Groups(g) => format!("G={g}"),
        }
    }
}

#[derive(Clone, Debug)]
enum Form {
    P(Prod),
    D(Dense),
}

#[derive(Clone, Debug)]
struct Hyb {
    form: Form,
    switch: Switch,
}

impl Hyb {
    fn settle(p: Prod, switch: Switch) -> Hyb {
        let form = if switch.keeps(&p) {
            Form::P(p)
        } else {
            Form::D(p.to_dense())
        };
        Hyb { form, switch }
    }
    fn build(f: &dyn Fn(usize, usize) -> bool, switch: Switch) -> Hyb {
        Hyb::settle(Prod::build(f), switch)
    }
    fn dense(&self) -> Dense {
        match &self.form {
            Form::P(p) => p.to_dense(),
            Form::D(d) => d.clone(),
        }
    }
    fn binop(&self, b: &Hyb, pf: fn(&Prod, &Prod) -> Prod, df: fn(&Dense, &Dense) -> Dense) -> Hyb {
        let s = self.switch;
        match (&self.form, &b.form) {
            (Form::P(x), Form::P(y)) => Hyb::settle(pf(x, y), s),
            (Form::D(x), Form::D(y)) => Hyb {
                form: Form::D(df(x, y)),
                switch: s,
            },
            _ => Hyb {
                form: Form::D(df(&self.dense(), &b.dense())),
                switch: s,
            },
        }
    }
    fn intersect(&self, b: &Hyb) -> Hyb {
        self.binop(b, Prod::intersect, Dense::intersect)
    }
    fn union(&self, b: &Hyb) -> Hyb {
        self.binop(b, Prod::union, Dense::union)
    }
    fn complement(&self) -> Hyb {
        match &self.form {
            Form::P(p) => Hyb::settle(p.complement(), self.switch),
            Form::D(d) => Hyb {
                form: Form::D(d.complement()),
                switch: self.switch,
            },
        }
    }
    fn is_empty(&self) -> bool {
        match &self.form {
            Form::P(p) => p.is_empty(),
            Form::D(d) => d.is_empty(),
        }
    }
    fn equal(&self, b: &Hyb) -> bool {
        match (&self.form, &b.form) {
            (Form::P(x), Form::P(y)) => x == y,
            _ => self.dense() == b.dense(),
        }
    }
    fn contains(&self, c: usize, o: usize) -> bool {
        match &self.form {
            Form::P(p) => p.contains(c, o),
            Form::D(d) => d.contains(c, o),
        }
    }
    fn is_dense(&self) -> bool {
        matches!(self.form, Form::D(_))
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

// ---------------------------------------------------------------- Q1 fixtures

/// The first adjacent pair of grid points where P/D goes from below 1 to at
/// least 1, as `(g below, g above)`. A ratio above 1 that comes before every
/// point below 1 is not a crossing: noise can put one there.
fn upward_crossing(points: &[(usize, f64)]) -> Option<(usize, usize)> {
    points
        .windows(2)
        .find(|w| w[0].1 < 1.0 && w[1].1 >= 1.0)
        .map(|w| (w[0].0, w[1].0))
}

const GS: [usize; 12] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Blocked,
    Interleaved,
}

/// One operation mask per context: `g` distinct seeded masks of density 0.5,
/// assigned in contiguous blocks or round robin.
fn group_rows(g: usize, layout: Layout, seed: u64) -> Vec<M256> {
    let masks: Vec<M256> = (0..g as u64)
        .map(|k| {
            let mut m = EMPTY;
            for (w, word) in m.iter_mut().enumerate() {
                *word = mix(seed.wrapping_mul(0x1000).wrapping_add(k * 4 + w as u64));
            }
            m
        })
        .collect();
    (0..SIDE)
        .map(|c| match layout {
            Layout::Blocked => masks[c * g / SIDE],
            Layout::Interleaved => masks[c % g],
        })
        .collect()
}

// ---------------------------------------------------------------- Q3 declarations

/// A half-open rectangle: contexts `c0..c1`, operations `o0..o1`.
type Rect = (usize, usize, usize, usize);

/// A pattern as a declaration: disjoint rectangles, or one aligned Morton block.
#[derive(Clone, Debug)]
enum Decl {
    Rects(Vec<Rect>),
    Prefix { shift: u32, value: u32 },
}

/// The operation mask with bits `o0..o1` set.
fn range_mask(o0: usize, o1: usize) -> M256 {
    let mut m = EMPTY;
    for (w, word) in m.iter_mut().enumerate() {
        let (lo, hi) = (w * 64, w * 64 + 64);
        let (a, b) = (o0.max(lo), o1.min(hi));
        if a < b {
            let n = b - a;
            let bits = if n == 64 {
                u64::MAX
            } else {
                ((1u64 << n) - 1) << (a - lo)
            };
            *word |= bits;
        }
    }
    m
}

impl Decl {
    /// The declaration as rectangles. An aligned Morton block of 4^k codes is
    /// a 2^k × 2^k square.
    fn rects(&self) -> Vec<Rect> {
        match self {
            Decl::Rects(r) => r.clone(),
            Decl::Prefix { shift, value } => {
                let (c, o) = unmorton(value << shift);
                let side = 1usize << (shift / 2);
                vec![(c, c + side, o, o + side)]
            }
        }
    }
    fn contains(&self, c: usize, o: usize) -> bool {
        self.rects()
            .iter()
            .any(|&(c0, c1, o0, o1)| in_rect(c, o, c0, c1, o0, o1))
    }
}

/// P from a declaration: cut the contexts at every rectangle edge, OR the
/// operation masks of the rectangles covering each piece, then normalize.
fn prod_declared(d: &Decl) -> Prod {
    let rects = d.rects();
    let mut cuts: Vec<usize> = rects.iter().flat_map(|r| [r.0, r.1]).collect();
    cuts.extend([0, SIDE]);
    cuts.sort_unstable();
    cuts.dedup();
    let mut cells = Vec::with_capacity(cuts.len());
    for w in cuts.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mut ops = EMPTY;
        for &(c0, c1, o0, o1) in &rects {
            if c0 <= lo && hi <= c1 {
                ops = or(&ops, &range_mask(o0, o1));
            }
        }
        cells.push((range_mask(lo, hi), ops));
    }
    Prod::normalize(cells)
}

/// D from a declaration: each covered row gets its operation range word by word.
fn dense_declared(d: &Decl) -> Dense {
    let mut out = Dense::zero();
    for (c0, c1, o0, o1) in d.rects() {
        let m = range_mask(o0, o1);
        for c in c0..c1 {
            for (w, &bits) in m.iter().enumerate() {
                out.w[c * 4 + w] |= bits;
            }
        }
    }
    out
}

/// I from a declaration: each rectangle as its maximal aligned Morton blocks,
/// unioned; a prefix is its one interval.
fn iv_declared(d: &Decl) -> Iv {
    match d {
        Decl::Prefix { shift, value } => Iv {
            s: vec![(value << shift, (value + 1) << shift)],
        },
        Decl::Rects(rects) => rects
            .iter()
            .fold(Iv { s: Vec::new() }, |acc, &r| acc.union(&box_iv(r))),
    }
}

fn box_iv(r: Rect) -> Iv {
    fn rec(r: Rect, c: usize, o: usize, side: usize, base: u32, out: &mut Vec<(u32, u32)>) {
        let (c0, c1, o0, o1) = r;
        if c >= c1 || c + side <= c0 || o >= o1 || o + side <= o0 {
            return;
        }
        if c0 <= c && c + side <= c1 && o0 <= o && o + side <= o1 {
            push(out, base, base + (side * side) as u32);
            return;
        }
        let h = side / 2;
        let q = (h * h) as u32;
        rec(r, c, o, h, base, out);
        rec(r, c, o + h, h, base + q, out);
        rec(r, c + h, o, h, base + 2 * q, out);
        rec(r, c + h, o + h, h, base + 3 * q, out);
    }
    let mut s = Vec::new();
    rec(r, 0, 0, SIDE, 0, &mut s);
    Iv { s }
}

/// The H6b patterns that have a declaration: (name, structured, A, B).
fn declared() -> Vec<(&'static str, bool, Decl, Decl)> {
    let r = |v: &[Rect]| Decl::Rects(v.to_vec());
    // The complement of [20,120) × [30,170) as four disjoint rectangles.
    let not_rect = r(&[
        (0, 20, 0, 256),
        (120, 256, 0, 256),
        (20, 120, 0, 30),
        (20, 120, 170, 256),
    ]);
    vec![
        (
            "row",
            true,
            r(&[(37, 38, 10, 200)]),
            r(&[(37, 38, 100, 250)]),
        ),
        (
            "column",
            true,
            r(&[(5, 230, 91, 92)]),
            r(&[(40, 256, 91, 92)]),
        ),
        (
            "rectangle",
            true,
            r(&[(20, 120, 30, 170)]),
            r(&[(60, 200, 100, 240)]),
        ),
        (
            "prefix",
            true,
            Decl::Prefix {
                shift: 12,
                value: 3,
            },
            Decl::Prefix {
                shift: 14,
                value: 0,
            },
        ),
        ("complementary", false, r(&[(20, 120, 30, 170)]), not_rect),
        (
            "disjoint",
            false,
            r(&[(0, 100, 0, 100)]),
            r(&[(150, 256, 120, 256)]),
        ),
        (
            "doughnut",
            false,
            r(&[
                (40, 90, 40, 200),
                (150, 200, 40, 200),
                (90, 150, 40, 90),
                (90, 150, 150, 200),
            ]),
            r(&[(100, 240, 0, 256)]),
        ),
    ]
}

// ---------------------------------------------------------------- reporting

/// One fixture for Q2: a name and its two membership functions.
struct Fixture {
    name: String,
    a: Box<dyn Fn(usize, usize) -> bool>,
    b: Box<dyn Fn(usize, usize) -> bool>,
}

fn fixtures() -> Vec<Fixture> {
    let mut v: Vec<Fixture> = patterns()
        .into_iter()
        .map(|p| Fixture {
            name: p.name.to_string(),
            a: Box::new(p.a),
            b: Box::new(p.b),
        })
        .collect();
    for &g in &GS {
        for layout in [Layout::Blocked, Layout::Interleaved] {
            let (ra, rb) = (group_rows(g, layout, 11), group_rows(g, layout, 12));
            v.push(Fixture {
                name: format!("g={g} {layout:?}"),
                a: Box::new(move |c, o| bit(&ra[c], o)),
                b: Box::new(move |c, o| bit(&rb[c], o)),
            });
        }
    }
    v
}

const SWITCHES: [Switch; 6] = [
    Switch::Bytes,
    Switch::Groups(2),
    Switch::Groups(4),
    Switch::Groups(8),
    Switch::Groups(16),
    Switch::Groups(32),
];

fn main() {
    println!("D-HOLO H6c: group-count crossover, group-count switch, declared construction");
    println!("tier: {}", tier());
    println!("timings: ns, median of 11\n");

    // ------------------------------------------------------------ Q1
    println!(
        "Q1: P vs D per query as the group count grows (A, B: same g and layout, seeds 11, 12)"
    );
    println!("g    layout       P groups | P query    D query  P/D | P/D ∩    ∪     ¬");
    let mut crossover = Vec::new();
    for layout in [Layout::Blocked, Layout::Interleaved] {
        let mut last_below: Option<usize> = None;
        let mut first_above: Option<usize> = None;
        let mut points: Vec<(usize, f64)> = Vec::new();
        for &g in &GS {
            let (ra, rb) = (group_rows(g, layout, 11), group_rows(g, layout, 12));
            let fa = |c: usize, o: usize| bit(&ra[c], o);
            let fb = |c: usize, o: usize| bit(&rb[c], o);
            let (pa, pb) = (Prod::build(&fa), Prod::build(&fb));
            let (da, db) = (Dense::build(&fa), Dense::build(&fb));
            assert_eq!(pa.groups.len(), g, "g={g} {layout:?}: fixture group count");
            let (px, ..) = query!(&pa, &pb, |x: &Prod, y: &Prod| x == y);
            let (dx, ..) = query!(&da, &db, |x: &Dense, y: &Dense| x == y);
            assert_eq!(px.to_dense(), dx, "g={g} {layout:?}: P and D disagree");
            let pq = time(|| {
                black_box(query!(
                    black_box(&pa),
                    black_box(&pb),
                    |x: &Prod, y: &Prod| x == y
                ));
            });
            let dq = time(|| {
                black_box(query!(
                    black_box(&da),
                    black_box(&db),
                    |x: &Dense, y: &Dense| x == y
                ));
            });
            let pi = time(|| {
                black_box(black_box(&pa).intersect(black_box(&pb)));
            });
            let di = time(|| {
                black_box(black_box(&da).intersect(black_box(&db)));
            });
            let pu = time(|| {
                black_box(black_box(&pa).union(black_box(&pb)));
            });
            let du = time(|| {
                black_box(black_box(&da).union(black_box(&db)));
            });
            let pn = time(|| {
                black_box(black_box(&pa).complement());
            });
            let dn = time(|| {
                black_box(black_box(&da).complement());
            });
            let r = pq / dq;
            points.push((g, r));
            if r < 1.0 {
                last_below = Some(g);
            } else if first_above.is_none() {
                first_above = Some(g);
            }
            println!(
                "{g:<4} {:<12} {:>8} | {pq:>9.0} {dq:>9.0} {r:>5.2} | {:>5.2} {:>5.2} {:>5.2}",
                format!("{layout:?}"),
                pa.groups.len(),
                pi / di,
                pu / du,
                pn / dn
            );
        }
        let up = upward_crossing(&points);
        println!("  {layout:?}: last g with P/D < 1: {last_below:?}; first g with P/D ≥ 1: {first_above:?}; first upward crossing: {up:?}");
        crossover.push((layout, up));
    }
    let in_range = crossover
        .iter()
        .all(|&(_, up)| matches!(up, Some((b, a)) if b >= 4 && a <= 16));
    println!(
        "  prediction (crossover between g = 4 and 16 in both layouts): {}",
        if in_range { "HOLDS" } else { "FALSIFIED" }
    );
    let step = |g: Option<usize>| g.and_then(|g| GS.iter().position(|&x| x == g));
    let agree = match (
        step(crossover[0].1.map(|(_, a)| a)),
        step(crossover[1].1.map(|(_, a)| a)),
    ) {
        (Some(a), Some(b)) => a.abs_diff(b) <= 1,
        _ => false,
    };
    println!("  layouts agree within one step of the g grid: {agree}\n");

    // ------------------------------------------------------------ Q2
    println!("Q2: hybrid switch policies vs D, over 14 H6b patterns + {} Q1 fixtures (N = 100, sweep build)", GS.len() * 2);
    let fx = fixtures();
    let mut dq = Vec::new();
    let mut dt = Vec::new();
    for f in &fx {
        let (da, db) = (Dense::build(&*f.a), Dense::build(&*f.b));
        let q = time(|| {
            black_box(query!(
                black_box(&da),
                black_box(&db),
                |x: &Dense, y: &Dense| x == y
            ));
        });
        let b = time(|| {
            black_box((Dense::build(&*f.a), Dense::build(&*f.b)));
        });
        dq.push(q);
        dt.push(b + 100.0 * q);
    }
    println!("switch  | max total/D  max query/D | worst fixture (query)        | verdict");
    for &s in &SWITCHES {
        let mut tot = Vec::new();
        let mut qr = Vec::new();
        for (k, f) in fx.iter().enumerate() {
            let (ha, hb) = (Hyb::build(&*f.a, s), Hyb::build(&*f.b, s));
            let q = time(|| {
                black_box(query!(
                    black_box(&ha),
                    black_box(&hb),
                    |x: &Hyb, y: &Hyb| x.equal(y)
                ));
            });
            let b = time(|| {
                black_box((Hyb::build(&*f.a, s), Hyb::build(&*f.b, s)));
            });
            tot.push((b + 100.0 * q) / dt[k]);
            qr.push(q / dq[k]);
        }
        let mt = tot.iter().copied().fold(0.0, f64::max);
        let (wk, mq) =
            qr.iter()
                .copied()
                .enumerate()
                .fold((0, 0.0), |a, (k, v)| if v > a.1 { (k, v) } else { a });
        let verdict = if mt <= 1.25 && mq <= 1.25 {
            "safe (≤ 1.25)"
        } else if mt >= 2.0 || mq >= 2.0 {
            "NOT safe (≥ 2× somewhere)"
        } else {
            "undecided (1.25–2)"
        };
        println!(
            "{:<7} | {mt:>11.2} {mq:>12.2} | {:<28} | {verdict}",
            s.label(),
            fx[wk].name
        );
    }
    println!();

    // ------------------------------------------------------------ Q3
    println!("Q3: construction from a declaration (every arm, D included); ns and ratio to D");
    println!("pattern       build P / I / D (swept D)     | query P / I / D      | total N=100 P/D H/D | N where P < D");
    for (name, structured, a, b) in declared() {
        let (pa, pb) = (prod_declared(&a), prod_declared(&b));
        let (ia, ib) = (iv_declared(&a), iv_declared(&b));
        let (da, db) = (dense_declared(&a), dense_declared(&b));
        let s = Switch::Groups(8);
        let (ha, hb) = (Hyb::settle(pa.clone(), s), Hyb::settle(pb.clone(), s));
        let bp = time(|| {
            black_box((prod_declared(black_box(&a)), prod_declared(black_box(&b))));
        });
        let bi = time(|| {
            black_box((iv_declared(black_box(&a)), iv_declared(black_box(&b))));
        });
        let bd = time(|| {
            black_box((dense_declared(black_box(&a)), dense_declared(black_box(&b))));
        });
        let bh = time(|| {
            black_box((
                Hyb::settle(prod_declared(black_box(&a)), s),
                Hyb::settle(prod_declared(black_box(&b)), s),
            ));
        });
        let fa = |c: usize, o: usize| a.contains(c, o);
        let fb = |c: usize, o: usize| b.contains(c, o);
        let bsw = time(|| {
            black_box((Dense::build(&fa), Dense::build(&fb)));
        });
        let qp = time(|| {
            black_box(query!(
                black_box(&pa),
                black_box(&pb),
                |x: &Prod, y: &Prod| x == y
            ));
        });
        let qi = time(|| {
            black_box(query!(black_box(&ia), black_box(&ib), |x: &Iv, y: &Iv| x == y));
        });
        let qd = time(|| {
            black_box(query!(
                black_box(&da),
                black_box(&db),
                |x: &Dense, y: &Dense| x == y
            ));
        });
        let qh = time(|| {
            black_box(query!(
                black_box(&ha),
                black_box(&hb),
                |x: &Hyb, y: &Hyb| x.equal(y)
            ));
        });
        let tot = |b: f64, q: f64| b + 100.0 * q;
        let (rp, rh) = (tot(bp, qp) / tot(bd, qd), tot(bh, qh) / tot(bd, qd));
        let n_star = if qp < qd {
            format!("{:.0}", ((bp - bd) / (qd - qp)).max(0.0).ceil())
        } else if bp <= bd {
            String::from("N ≤ crossover only")
        } else {
            String::from("never")
        };
        let rule = if structured {
            format!(
                " structured: {}",
                if rp.min(rh) <= 0.5 {
                    "PASS (≤ 0.5)"
                } else if rp.min(rh) > 0.5 && rp.min(rh) < 2.0 {
                    "undecided"
                } else {
                    "FAIL"
                }
            )
        } else {
            String::new()
        };
        println!(
            "{name:<13} {bp:>6.0} {bi:>6.0} {bd:>6.0} ({bsw:>7.0}) | {qp:>6.0} {qi:>6.0} {qd:>6.0} | {rp:>5.2} {rh:>5.2} | {n_star}{rule}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ratio_above_one_before_every_point_below_is_not_a_crossing() {
        // Codex example: above 1 at g = 1, then below 1 through g = 64.
        let pts: Vec<(usize, f64)> = GS
            .iter()
            .map(|&g| (g, if g == 1 { 1.2 } else { 0.8 }))
            .collect();
        assert_eq!(upward_crossing(&pts), None);
        // An ordinary crossing between 8 and 12 is found.
        let pts: Vec<(usize, f64)> = GS
            .iter()
            .map(|&g| (g, if g <= 8 { 0.8 } else { 1.2 }))
            .collect();
        assert_eq!(upward_crossing(&pts), Some((8, 12)));
    }

    fn same_cells(a: &dyn Fn(usize, usize) -> bool, b: &dyn Fn(usize, usize) -> bool) -> bool {
        (0..CELLS).all(|i| a(i / SIDE, i % SIDE) == b(i / SIDE, i % SIDE))
    }

    /// FAILS IF: a declaration builds a different set from the swept build of
    /// the same membership function, in any arm (P: group list; D: words; I:
    /// intervals), or a declaration does not describe its H6b pattern.
    #[test]
    fn declared_build_equals_swept_build() {
        let pats = patterns();
        for (name, _, a, b) in declared() {
            let p = pats.iter().find(|p| p.name == name).unwrap();
            for (d, f) in [(&a, p.a), (&b, p.b)] {
                let g = |c: usize, o: usize| d.contains(c, o);
                assert!(
                    same_cells(&g, &f),
                    "{name}: declaration differs from the pattern"
                );
                assert_eq!(prod_declared(d), Prod::build(&f), "{name}: P");
                assert_eq!(dense_declared(d), Dense::build(&f), "{name}: D");
                assert_eq!(iv_declared(d), Iv::build(&f), "{name}: I");
            }
        }
    }

    /// FAILS IF: a Q1 fixture does not have exactly g groups, or is empty or
    /// full, or the two layouts coincide for g > 1.
    #[test]
    fn group_fixtures_have_g_groups() {
        for &g in &GS {
            for layout in [Layout::Blocked, Layout::Interleaved] {
                let rows = group_rows(g, layout, 11);
                let p = Prod::build(&|c, o| bit(&rows[c], o));
                assert_eq!(p.groups.len(), g, "g={g} {layout:?}");
                let n: usize = rows
                    .iter()
                    .map(|m| m.iter().map(|w| w.count_ones() as usize).sum::<usize>())
                    .sum();
                assert!(n > 0 && n < CELLS, "g={g} {layout:?}: empty or full");
            }
            if g > 1 {
                assert_ne!(
                    group_rows(g, Layout::Blocked, 11),
                    group_rows(g, Layout::Interleaved, 11)
                );
            }
        }
    }

    /// FAILS IF: the group-count hybrid never switches or always switches: at
    /// G = 8 it must keep P at g = 1 and fall back to D at g = 64.
    #[test]
    fn group_switch_switches_both_ways() {
        for layout in [Layout::Blocked, Layout::Interleaved] {
            let one = group_rows(1, layout, 11);
            let many = group_rows(64, layout, 11);
            assert!(!Hyb::build(&|c, o| bit(&one[c], o), Switch::Groups(8)).is_dense());
            assert!(Hyb::build(&|c, o| bit(&many[c], o), Switch::Groups(8)).is_dense());
        }
    }

    /// FAILS IF: on a Q1 fixture or an H6b pattern, P, D or a hybrid under any
    /// switch differs from the row oracle on the query, cell by cell.
    #[test]
    fn every_arm_matches_the_oracle() {
        for f in fixtures() {
            let x: Vec<bool> = (0..CELLS)
                .map(|i| (f.a)(i / SIDE, i % SIDE) != (f.b)(i / SIDE, i % SIDE))
                .collect();
            let e = x.iter().all(|&v| !v);
            let q = (0..CELLS).all(|i| x[i] == (f.a)(i / SIDE, i % SIDE));
            let (pa, pb) = (Prod::build(&*f.a), Prod::build(&*f.b));
            let (px, pe, pq) = query!(&pa, &pb, |x: &Prod, y: &Prod| x == y);
            assert_eq!((pe, pq), (e, q), "{}: P flags", f.name);
            for &s in &SWITCHES {
                let (ha, hb) = (Hyb::build(&*f.a, s), Hyb::build(&*f.b, s));
                let (hx, he, hq) = query!(&ha, &hb, |x: &Hyb, y: &Hyb| x.equal(y));
                assert_eq!((he, hq), (e, q), "{} {s:?}: H flags", f.name);
                for (i, &want) in x.iter().enumerate() {
                    assert_eq!(hx.contains(i / SIDE, i % SIDE), want, "{} {s:?}", f.name);
                }
            }
            for (i, &want) in x.iter().enumerate() {
                assert_eq!(px.contains(i / SIDE, i % SIDE), want, "{}: P", f.name);
            }
        }
    }

    /// FAILS IF: a rectangle's Morton decomposition is not exactly the rectangle.
    #[test]
    fn box_iv_is_exact() {
        for r in [
            (0, 256, 0, 256),
            (37, 38, 10, 200),
            (3, 250, 7, 9),
            (20, 120, 30, 170),
        ] {
            let iv = box_iv(r);
            assert!(iv.canonical());
            for c in 0..SIDE {
                for o in 0..SIDE {
                    assert_eq!(
                        iv.contains(c, o),
                        in_rect(c, o, r.0, r.1, r.2, r.3),
                        "{r:?} at ({c}, {o})"
                    );
                }
            }
        }
    }
}
