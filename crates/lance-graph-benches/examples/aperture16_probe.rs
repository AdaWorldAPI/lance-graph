//! D-APERTURE-16-0: can the 64K self-space support mask be read as a
//! `[u16; 4096]` execution aperture, and does 16-self granularity schedule
//! work better than the 64-bit words the production kernels already walk?
//!
//! Universe: `self: u16`, 65 536 ordinals. Support mask: `[u64; 1024]`
//! (8 KiB). The u16 view is NOT a copy and NOT a cast: cell `c` is
//! `(words[c >> 2] >> ((c & 3) * 16)) as u16`, which is the same bits in
//! the same order on any endianness (on little-endian it is also exactly what
//! `slice::align_to::<u16>` would yield; checked in `views_agree`).
//!
//! Modes (all assert every route's answer equal before printing a time):
//! - `a1` : visitation only — the same mask walked as u64 words, as u16
//!   cells, adaptive u16 / u64, and a materialised ordinal list;
//! - `a2` : payload-width ladder (u8 … 128-byte records) — full scans vs
//!   gated visitation;
//! - `occ`: one fixed occupancy per cell (0..=16 live) — set-bit visit vs a
//!   dense masked 16-lane loop, the local crossover;
//! - `a3` : VIA `via[self] -> target` — support (`target |= 1`) and
//!   multiplicity (`K_next[target] += K[self]`) separately;
//! - `a4` : bounded extent × aperture over an ordered lane;
//! - `a5` : K propagation over a CSR edge list, gated by source support;
//! - `a6` : coarse target cells — exact-only vs cell-seen bitmap during the
//!   pass vs a `[u16; 4096]` histogram pre-pass, consumed by a next-frontier
//!   build.
//!
//! ```bash
//! cargo run --release -p lance-graph-benches --example aperture16_probe -- a1
//! ```
//!
//! Output is TSV on stdout, one row per (pattern, payload, route).

// The subject of this probe IS the row-ordinal geometry (`self`, `cell`,
// `word`), so its loops index by ordinal on purpose; an iterator would hide
// the coordinate the measurement is about.
#![allow(clippy::needless_range_loop)]

use std::hint::black_box;
use std::time::Instant;

use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch, Value};
use lance_graph_quack::{lower, Agg, Col, Filter, Mask, Query};

const N: usize = 65_536;
const WORDS: usize = N / 64; // 1024
const CELLS: usize = N / 16; // 4096

// ───────────────────────────── utilities ─────────────────────────────

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn reps() -> usize {
    std::env::var("PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(101)
}

/// Median wall time in ns of `f`, and its last result.
fn time<T: Copy>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = f(); // warm
    let mut v: Vec<u128> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            last = black_box(f());
            t0.elapsed().as_nanos()
        })
        .collect();
    v.sort_unstable();
    (v[v.len() / 2] as f64, last)
}

#[inline(always)]
fn bit(m: &[u64], i: usize) -> u64 {
    (m[i >> 6] >> (i & 63)) & 1
}

/// Zero-copy u16 cell view of the u64 words.
#[inline(always)]
fn cell(m: &[u64], c: usize) -> u16 {
    (m[c >> 2] >> ((c & 3) * 16)) as u16
}

// ───────────────────────────── layouts ─────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Layout {
    Uniform,
    Clustered,
    SingleRange,
    TinyRuns,
    Islands,
    Alternating,
    OnePer16,
    OnePer64,
}

impl Layout {
    fn name(self) -> &'static str {
        match self {
            Layout::Uniform => "uniform",
            Layout::Clustered => "clustered",
            Layout::SingleRange => "single-range",
            Layout::TinyRuns => "tiny-runs",
            Layout::Islands => "islands16",
            Layout::Alternating => "alternating",
            Layout::OnePer16 => "one-per-16",
            Layout::OnePer64 => "one-per-64",
        }
    }
    const DENSITY_LAYOUTS: [Layout; 5] = [
        Layout::Uniform,
        Layout::Clustered,
        Layout::SingleRange,
        Layout::TinyRuns,
        Layout::Islands,
    ];
    const FIXED: [Layout; 3] = [Layout::Alternating, Layout::OnePer16, Layout::OnePer64];
}

const DENSITIES: [f64; 11] = [
    1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001,
];

fn set(m: &mut [u64], i: usize) {
    m[i >> 6] |= 1 << (i & 63);
}

/// Exactly `round(d·N)` live selves (fixed layouts ignore `d`).
fn build(layout: Layout, d: f64, seed: u64) -> Vec<u64> {
    let mut m = vec![0u64; WORDS];
    let k = (d * N as f64).round() as usize;
    match layout {
        Layout::Uniform => {
            // partial Fisher–Yates: exactly k distinct positions
            let mut p: Vec<u32> = (0..N as u32).collect();
            for i in 0..k {
                let j = i + (mix(seed ^ i as u64) as usize % (N - i));
                p.swap(i, j);
                set(&mut m, p[i] as usize);
            }
        }
        Layout::Clustered => {
            // runs of 128..640 selves at random starts until k are live
            let mut live = 0;
            let mut s = seed;
            while live < k {
                s = mix(s);
                let start = (s % N as u64) as usize;
                let len = 128 + (mix(s ^ 7) % 512) as usize;
                for i in start..(start + len).min(N) {
                    if live == k {
                        break;
                    }
                    if bit(&m, i) == 0 {
                        set(&mut m, i);
                        live += 1;
                    }
                }
            }
        }
        Layout::SingleRange => {
            let start = (N - k) / 3;
            for i in start..start + k {
                set(&mut m, i);
            }
        }
        Layout::TinyRuns => {
            let mut live = 0;
            let mut s = seed;
            while live < k {
                s = mix(s);
                let start = (s % N as u64) as usize;
                let len = 2 + (mix(s ^ 3) % 3) as usize;
                for i in start..(start + len).min(N) {
                    if live == k {
                        break;
                    }
                    if bit(&m, i) == 0 {
                        set(&mut m, i);
                        live += 1;
                    }
                }
            }
        }
        Layout::Islands => {
            // whole 16-cells fully dense, chosen at random; k rounded down to
            // a multiple of 16 (a partial island would not be an island)
            let cells = k / 16;
            let mut p: Vec<u32> = (0..CELLS as u32).collect();
            for i in 0..cells {
                let j = i + (mix(seed ^ i as u64) as usize % (CELLS - i));
                p.swap(i, j);
                let c = p[i] as usize;
                for s in 0..16 {
                    set(&mut m, c * 16 + s);
                }
            }
        }
        Layout::Alternating => m.iter_mut().for_each(|w| *w = 0x5555_5555_5555_5555),
        Layout::OnePer16 => {
            for c in 0..CELLS {
                set(&mut m, c * 16 + (mix(seed ^ c as u64) % 16) as usize);
            }
        }
        Layout::OnePer64 => {
            for w in 0..WORDS {
                set(&mut m, w * 64 + (mix(seed ^ w as u64) % 64) as usize);
            }
        }
    }
    m
}

fn patterns() -> Vec<(Layout, f64, Vec<u64>)> {
    let mut v = Vec::new();
    for l in Layout::DENSITY_LAYOUTS {
        for d in DENSITIES {
            v.push((l, d, build(l, d, 0xA9E7 ^ (d.to_bits()))));
        }
    }
    for l in Layout::FIXED {
        let m = build(l, 0.0, 0x51);
        let d = live(&m) as f64 / N as f64;
        v.push((l, d, m));
    }
    v
}

fn live(m: &[u64]) -> usize {
    m.iter().map(|w| w.count_ones() as usize).sum()
}

/// Geometry of one mask, the accounting the brief asks for.
#[derive(Clone, Copy)]
struct Geo {
    live: usize,
    zero_cells: usize,
    full_cells: usize,
    zero_words: usize,
    full_words: usize,
}

fn geo(m: &[u64]) -> Geo {
    let mut g = Geo {
        live: live(m),
        zero_cells: 0,
        full_cells: 0,
        zero_words: 0,
        full_words: 0,
    };
    for c in 0..CELLS {
        match cell(m, c) {
            0 => g.zero_cells += 1,
            0xFFFF => g.full_cells += 1,
            _ => {}
        }
    }
    for &w in m {
        match w {
            0 => g.zero_words += 1,
            u64::MAX => g.full_words += 1,
            _ => {}
        }
    }
    g
}

/// Distinct 64-byte payload lines that hold at least one live self.
fn live_lines(m: &[u64], width: usize) -> usize {
    let mut next_free = 0usize; // first line not yet counted
    let mut n = 0;
    for w in 0..WORDS {
        let mut b = m[w];
        while b != 0 {
            let i = w * 64 + b.trailing_zeros() as usize;
            b &= b - 1;
            let first = (i * width / 64).max(next_free);
            let last = (i * width + width - 1) / 64;
            if last >= first {
                n += last - first + 1;
                next_free = last + 1;
            }
        }
    }
    n
}

// ───────────────────────────── payloads ─────────────────────────────

trait Payload: Copy {
    const W: usize;
    fn make(i: u64) -> Self;
    fn f(&self) -> u64;
}
impl Payload for u8 {
    const W: usize = 1;
    fn make(i: u64) -> Self {
        mix(i) as u8
    }
    #[inline(always)]
    fn f(&self) -> u64 {
        *self as u64
    }
}
impl Payload for u16 {
    const W: usize = 2;
    fn make(i: u64) -> Self {
        mix(i) as u16
    }
    #[inline(always)]
    fn f(&self) -> u64 {
        *self as u64
    }
}
impl Payload for u32 {
    const W: usize = 4;
    fn make(i: u64) -> Self {
        (mix(i) as u32) >> 12 // < 2^20, so an i32 sum is exact
    }
    #[inline(always)]
    fn f(&self) -> u64 {
        *self as u64
    }
}
impl Payload for u64 {
    const W: usize = 8;
    fn make(i: u64) -> Self {
        mix(i)
    }
    #[inline(always)]
    fn f(&self) -> u64 {
        *self
    }
}
macro_rules! rec {
    ($n:literal) => {
        impl Payload for [u64; $n] {
            const W: usize = 8 * $n;
            fn make(i: u64) -> Self {
                core::array::from_fn(|k| mix(i ^ ((k as u64) << 40)))
            }
            /// Touches every byte of the record.
            #[inline(always)]
            fn f(&self) -> u64 {
                self.iter().fold(0u64, |a, x| a.wrapping_add(*x))
            }
        }
    };
}
rec!(2);
rec!(4);
rec!(8);
rec!(16);

// ───────────────────────────── routes ─────────────────────────────

/// Scan every row, branch on its bit (the "consult mask per row" baseline).
fn scan_branch<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for (i, x) in p.iter().enumerate() {
        if bit(m, i) == 1 {
            a = a.wrapping_add(x.f());
        }
    }
    a
}

/// Load every row, AND with the bit (branchless; touches N × W).
fn full_branchless<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for (i, x) in p.iter().enumerate() {
        a = a.wrapping_add(x.f() & bit(m, i).wrapping_neg());
    }
    a
}

/// The production shape (ndarray `group_walk` / `masked_sum_i32`): skip a
/// zero word in one test, then visit set bits.
fn visit64<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for (w, &word) in m.iter().enumerate() {
        let mut b = word;
        while b != 0 {
            let i = w * 64 + b.trailing_zeros() as usize;
            b &= b - 1;
            a = a.wrapping_add(p[i].f());
        }
    }
    a
}

/// The aperture: test each u16 cell, visit set bits inside it.
fn visit16<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for c in 0..CELLS {
        let mut b = cell(m, c);
        while b != 0 {
            let i = c * 16 + b.trailing_zeros() as usize;
            b &= b - 1;
            a = a.wrapping_add(p[i].f());
        }
    }
    a
}

/// u64 word test first (one test skips 64), u16 cells only inside live
/// words — the aperture read hierarchically.
fn visit64_16<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for (w, &word) in m.iter().enumerate() {
        if word == 0 {
            continue;
        }
        for q in 0..4 {
            let mut b = (word >> (q * 16)) as u16;
            let base = w * 64 + q * 16;
            while b != 0 {
                let i = base + b.trailing_zeros() as usize;
                b &= b - 1;
                a = a.wrapping_add(p[i].f());
            }
        }
    }
    a
}

/// Adaptive u16: skip / dense 16 / set-bit.
fn adapt16<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for c in 0..CELLS {
        let b = cell(m, c);
        if b == 0 {
            continue;
        }
        let base = c * 16;
        if b == 0xFFFF {
            for x in &p[base..base + 16] {
                a = a.wrapping_add(x.f());
            }
        } else {
            let mut b = b;
            while b != 0 {
                let i = base + b.trailing_zeros() as usize;
                b &= b - 1;
                a = a.wrapping_add(p[i].f());
            }
        }
    }
    a
}

/// Adaptive u64: skip / dense 64 / set-bit.
fn adapt64<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for (w, &word) in m.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let base = w * 64;
        if word == u64::MAX {
            for x in &p[base..base + 64] {
                a = a.wrapping_add(x.f());
            }
        } else {
            let mut b = word;
            while b != 0 {
                let i = base + b.trailing_zeros() as usize;
                b &= b - 1;
                a = a.wrapping_add(p[i].f());
            }
        }
    }
    a
}

/// Hierarchical adaptive: u64 skip / dense 64; inside a partial word each
/// u16 quarter is skipped, run dense when full, set-bit visited otherwise.
fn adapt64_16<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for (w, &word) in m.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let base = w * 64;
        if word == u64::MAX {
            for x in &p[base..base + 64] {
                a = a.wrapping_add(x.f());
            }
            continue;
        }
        for q in 0..4 {
            let b = (word >> (q * 16)) as u16;
            let qb = base + q * 16;
            if b == 0xFFFF {
                for x in &p[qb..qb + 16] {
                    a = a.wrapping_add(x.f());
                }
            } else {
                let mut b = b;
                while b != 0 {
                    let i = qb + b.trailing_zeros() as usize;
                    b &= b - 1;
                    a = a.wrapping_add(p[i].f());
                }
            }
        }
    }
    a
}

/// u64 schedule with branch-free full-run detection: the full u16 quarters
/// of a word are found without a branch, run dense, and every remaining
/// set bit is walked in ONE loop over the word (no per-quarter split). A
/// word with no full quarter costs one predictable test over `visit64`.
fn adapt64q<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for (w, &word) in m.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let base = w * 64;
        let mut full = 0u64;
        for q in 0..4 {
            let hit = (((word >> (q * 16)) & 0xFFFF) == 0xFFFF) as u64;
            full |= hit.wrapping_neg() & (0xFFFF << (q * 16));
        }
        if full != 0 {
            for q in 0..4 {
                if (full >> (q * 16)) & 1 == 1 {
                    for x in &p[base + q * 16..base + q * 16 + 16] {
                        a = a.wrapping_add(x.f());
                    }
                }
            }
        }
        let mut b = word & !full;
        while b != 0 {
            let i = base + b.trailing_zeros() as usize;
            b &= b - 1;
            a = a.wrapping_add(p[i].f());
        }
    }
    a
}

/// Selection vector: materialise ordinals, then fold. The build is inside
/// the timed region and its bytes are reported.
fn ordinal<P: Payload>(m: &[u64], p: &[P], buf: &mut Vec<u16>) -> u64 {
    buf.clear();
    for (w, &word) in m.iter().enumerate() {
        let mut b = word;
        while b != 0 {
            buf.push((w * 64) as u16 + b.trailing_zeros() as u16);
            b &= b - 1;
        }
    }
    buf.iter()
        .fold(0u64, |a, &i| a.wrapping_add(p[i as usize].f()))
}

/// Pure visitation checksum (no payload): count and ordinal sum.
fn vis_ck(i: usize, a: &mut (u64, u64)) {
    a.0 += 1;
    a.1 += i as u64;
}

// ───────────────────────────── modes ─────────────────────────────

fn mode_a1() {
    let r = reps();
    println!("layout\tdensity\tlive\tzero_cells\tfull_cells\tzero_words\tfull_words\troute\tns\tmask_tests\ttemp_bytes");
    // native production Count over the same mask, for reference
    let count_p = lower(&Query {
        filter: Filter::plane(Mask(0)),
        agg: Agg::Count,
    })
    .unwrap();
    for (l, d, m) in patterns() {
        let g = geo(&m);
        let mut buf: Vec<u16> = Vec::with_capacity(N);
        type Route<'a> = (
            &'static str,
            Box<dyn FnMut() -> (u64, u64) + 'a>,
            usize,
            usize,
        );
        let routes: [Route; 6] = [
            (
                "u64-visit",
                Box::new(|| {
                    let mut a = (0, 0);
                    for (w, &word) in m.iter().enumerate() {
                        let mut b = word;
                        while b != 0 {
                            vis_ck(w * 64 + b.trailing_zeros() as usize, &mut a);
                            b &= b - 1;
                        }
                    }
                    a
                }),
                WORDS,
                0,
            ),
            (
                "u16-visit",
                Box::new(|| {
                    let mut a = (0, 0);
                    for c in 0..CELLS {
                        let mut b = cell(&m, c);
                        while b != 0 {
                            vis_ck(c * 16 + b.trailing_zeros() as usize, &mut a);
                            b &= b - 1;
                        }
                    }
                    a
                }),
                CELLS,
                0,
            ),
            (
                "u64>u16-visit",
                Box::new(|| {
                    let mut a = (0, 0);
                    for (w, &word) in m.iter().enumerate() {
                        if word == 0 {
                            continue;
                        }
                        for q in 0..4 {
                            let mut b = (word >> (q * 16)) as u16;
                            while b != 0 {
                                vis_ck(w * 64 + q * 16 + b.trailing_zeros() as usize, &mut a);
                                b &= b - 1;
                            }
                        }
                    }
                    a
                }),
                WORDS + 4 * (WORDS - g.zero_words),
                0,
            ),
            (
                "per-row-scan",
                Box::new(|| {
                    let mut a = (0, 0);
                    for i in 0..N {
                        if bit(&m, i) == 1 {
                            vis_ck(i, &mut a);
                        }
                    }
                    a
                }),
                N,
                0,
            ),
            (
                "ordinal-list",
                Box::new(|| {
                    buf.clear();
                    for (w, &word) in m.iter().enumerate() {
                        let mut b = word;
                        while b != 0 {
                            buf.push((w * 64) as u16 + b.trailing_zeros() as u16);
                            b &= b - 1;
                        }
                    }
                    let mut a = (0, 0);
                    for &i in buf.iter() {
                        vis_ck(i as usize, &mut a);
                    }
                    a
                }),
                WORDS,
                2 * g.live,
            ),
            (
                "native-count",
                Box::new(|| {
                    let masks: [&[u64]; 1] = [&m];
                    let planes = Planes {
                        n_rows: N,
                        masks: &masks,
                        lanes: &[],
                    };
                    let mut s = Scratch::for_program(&count_p, N).unwrap();
                    match execute_into(&count_p, &planes, &Foreign::NONE, &mut s, Out::None)
                        .unwrap()
                    {
                        Value::Count(c) => (c as u64, u64::MAX),
                        v => panic!("{v:?}"),
                    }
                }),
                WORDS,
                0,
            ),
        ];
        let mut want: Option<(u64, u64)> = None;
        for (name, mut f, tests, temp) in routes {
            let (ns, got) = time(r, &mut f);
            if name == "native-count" {
                assert_eq!(got.0, g.live as u64, "native count {l:?} {d}");
            } else if let Some(w) = want {
                assert_eq!(got, w, "{name} {l:?} {d}");
            } else {
                want = Some(got);
            }
            println!(
                "{}\t{d}\t{}\t{}\t{}\t{}\t{}\t{name}\t{ns:.0}\t{tests}\t{temp}",
                l.name(),
                g.live,
                g.zero_cells,
                g.full_cells,
                g.zero_words,
                g.full_words
            );
        }
    }
}

fn ladder_one<P: Payload>(name: &str, r: usize, pats: &[(Layout, f64, Vec<u64>)]) {
    let p: Vec<P> = (0..N as u64).map(P::make).collect();
    for (l, d, m) in pats {
        let g = geo(m);
        let mut buf: Vec<u16> = Vec::with_capacity(N);
        let lines = live_lines(m, P::W);
        let all_lines = (N * P::W).div_ceil(64);
        let want = scan_branch(m, &p);
        type R<'a, P> = (&'static str, fn(&[u64], &[P]) -> u64);
        let routes: [R<P>; 9] = [
            ("adapt64q", adapt64q::<P>),
            ("adapt64>16", adapt64_16::<P>),
            ("scan-branch", scan_branch::<P>),
            ("full-branchless", full_branchless::<P>),
            ("u64-visit", visit64::<P>),
            ("u16-visit", visit16::<P>),
            ("u64>u16-visit", visit64_16::<P>),
            ("adapt16", adapt16::<P>),
            ("adapt64", adapt64::<P>),
        ];
        for (rn, f) in routes {
            let (ns, got) = time(r, || f(black_box(m), black_box(&p)));
            assert_eq!(got, want, "{rn} {name} {l:?} {d}");
            let (loads, ln) = match rn {
                "scan-branch" | "full-branchless" => (
                    if rn == "scan-branch" { g.live } else { N },
                    if rn == "scan-branch" {
                        lines
                    } else {
                        all_lines
                    },
                ),
                _ => (g.live, lines),
            };
            println!(
                "{}\t{d}\t{}\t{name}\t{}\t{rn}\t{ns:.0}\t{loads}\t{ln}\t0",
                l.name(),
                g.live,
                P::W
            );
        }
        let (ns, got) = time(r, || ordinal(black_box(m), black_box(&p), &mut buf));
        assert_eq!(got, want, "ordinal {name} {l:?} {d}");
        println!(
            "{}\t{d}\t{}\t{name}\t{}\tordinal-list\t{ns:.0}\t{}\t{lines}\t{}",
            l.name(),
            g.live,
            P::W,
            g.live,
            2 * g.live
        );
    }
}

fn mode_a2() {
    let r = reps();
    println!("layout\tdensity\tlive\tpayload\twidth_B\troute\tns\tpayload_loads\tpayload_lines\ttemp_bytes");
    let pats = patterns();
    ladder_one::<u8>("u8", r, &pats);
    ladder_one::<u16>("u16", r, &pats);
    ladder_one::<u32>("u32", r, &pats);
    ladder_one::<u64>("u64", r, &pats);
    ladder_one::<[u64; 2]>("rec16", r, &pats);
    ladder_one::<[u64; 4]>("rec32", r, &pats);
    ladder_one::<[u64; 8]>("rec64", r, &pats);
    ladder_one::<[u64; 16]>("rec128", r, &pats);
}

/// Every cell holds exactly `k` live bits (random positions).
fn occ_mask(k: usize, seed: u64) -> Vec<u64> {
    let mut m = vec![0u64; WORDS];
    for c in 0..CELLS {
        let mut slots: [u8; 16] = core::array::from_fn(|i| i as u8);
        for i in 0..k {
            let j = i + (mix(seed ^ ((c * 16 + i) as u64)) as usize % (16 - i));
            slots.swap(i, j);
            set(&mut m, c * 16 + slots[i] as usize);
        }
    }
    m
}

/// Dense masked 16-lane loop for every non-zero cell (branch-free inside).
fn dense16<P: Payload>(m: &[u64], p: &[P]) -> u64 {
    let mut a = 0u64;
    for c in 0..CELLS {
        let b = cell(m, c) as u64;
        if b == 0 {
            continue;
        }
        let base = c * 16;
        for (s, x) in p[base..base + 16].iter().enumerate() {
            a = a.wrapping_add(x.f() & ((b >> s) & 1).wrapping_neg());
        }
    }
    a
}

fn occ_one<P: Payload>(name: &str, r: usize) {
    let p: Vec<P> = (0..N as u64).map(P::make).collect();
    for k in 0..=16 {
        let m = occ_mask(k, 0x0CC ^ k as u64);
        let want = scan_branch(&m, &p);
        let (a, ga) = time(r, || visit16(black_box(&m), black_box(&p)));
        let (b, gb) = time(r, || dense16(black_box(&m), black_box(&p)));
        let (c, gc) = time(r, || visit64(black_box(&m), black_box(&p)));
        let (e, ge) = time(r, || adapt16(black_box(&m), black_box(&p)));
        assert_eq!((ga, gb, gc, ge), (want, want, want, want), "occ {name} {k}");
        println!("{name}\t{}\t{k}\t{a:.0}\t{b:.0}\t{c:.0}\t{e:.0}", P::W);
    }
}

fn mode_occ() {
    let r = reps();
    println!("payload\twidth_B\tlive_per_cell\tu16_setbit_ns\tu16_dense_masked_ns\tu64_setbit_ns\tadapt16_ns");
    occ_one::<u8>("u8", r);
    occ_one::<u32>("u32", r);
    occ_one::<u64>("u64", r);
    occ_one::<[u64; 4]>("rec32", r);
    occ_one::<[u64; 16]>("rec128", r);
}

// ── VIA: support and multiplicity, separately ──

#[derive(Clone, Copy)]
enum Target {
    Uniform,
    Clustered,
    Hotspot,
    Zipf,
    OneToOne,
    ManyToOne,
}

impl Target {
    const ALL: [Target; 6] = [
        Target::Uniform,
        Target::Clustered,
        Target::Hotspot,
        Target::Zipf,
        Target::OneToOne,
        Target::ManyToOne,
    ];
    fn name(self) -> &'static str {
        match self {
            Target::Uniform => "uniform",
            Target::Clustered => "clustered",
            Target::Hotspot => "hotspot",
            Target::Zipf => "zipf",
            Target::OneToOne => "one-to-one",
            Target::ManyToOne => "many-to-one",
        }
    }
    fn of(self, s: usize) -> u16 {
        let h = mix(s as u64 ^ 0x7A);
        match self {
            Target::Uniform => h as u16,
            // the target stays within 256 of the source: local edges
            Target::Clustered => (s as u64).wrapping_add(h % 256) as u16,
            // 90 % of edges land in 64 hot targets
            Target::Hotspot => {
                if h % 10 < 9 {
                    (h >> 8) as u16 % 64
                } else {
                    (h >> 8) as u16
                }
            }
            // rank ∝ 1/u: inverse-transform of a heavy tail
            Target::Zipf => {
                let u = ((h >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
                ((1.0 / u) as u64 % N as u64) as u16
            }
            Target::OneToOne => (s as u16).wrapping_mul(40_503).wrapping_add(1),
            Target::ManyToOne => (s >> 8) as u16,
        }
    }
}

fn mode_a3() {
    let r = reps();
    println!("target\tsrc_density\tlive\tsemantics\troute\tns\tvia_reads\tk_reads\ttarget_writes");
    let k: Vec<u32> = (0..N as u64).map(|i| (mix(i ^ 0x4B) % 7) as u32).collect();
    for t in Target::ALL {
        let via: Vec<u16> = (0..N).map(|s| t.of(s)).collect();
        for d in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001] {
            let m = build(Layout::Uniform, d, 0x5A ^ d.to_bits());
            let g = geo(&m);
            let mut buf: Vec<u16> = Vec::with_capacity(N);
            // ---- support: target_mask[via[s]] = 1 ----
            let mut tm = vec![0u64; WORDS];
            let sup = |tm: &mut [u64]| -> u64 {
                tm.iter().map(|w| w.count_ones() as u64).sum::<u64>()
                    ^ tm.iter()
                        .fold(0u64, |a, w| a.wrapping_mul(31).wrapping_add(*w))
            };
            let mut want = None;
            for route in ["full-scan", "u64-gate", "u16-gate", "ordinal"] {
                let (ns, got) = time(r, || {
                    tm.iter_mut().for_each(|w| *w = 0);
                    match route {
                        "full-scan" => {
                            for s in 0..N {
                                let v = via[s] as usize;
                                tm[v >> 6] |= bit(&m, s) << (v & 63);
                            }
                        }
                        "u64-gate" => {
                            for (w, &word) in m.iter().enumerate() {
                                let mut b = word;
                                while b != 0 {
                                    let v = via[w * 64 + b.trailing_zeros() as usize] as usize;
                                    b &= b - 1;
                                    tm[v >> 6] |= 1 << (v & 63);
                                }
                            }
                        }
                        "u16-gate" => {
                            for c in 0..CELLS {
                                let mut b = cell(&m, c);
                                while b != 0 {
                                    let v = via[c * 16 + b.trailing_zeros() as usize] as usize;
                                    b &= b - 1;
                                    tm[v >> 6] |= 1 << (v & 63);
                                }
                            }
                        }
                        _ => {
                            buf.clear();
                            for (w, &word) in m.iter().enumerate() {
                                let mut b = word;
                                while b != 0 {
                                    buf.push((w * 64) as u16 + b.trailing_zeros() as u16);
                                    b &= b - 1;
                                }
                            }
                            for &s in buf.iter() {
                                let v = via[s as usize] as usize;
                                tm[v >> 6] |= 1 << (v & 63);
                            }
                        }
                    }
                    sup(&mut tm)
                });
                match want {
                    None => want = Some(got),
                    Some(w) => assert_eq!(got, w, "support {route}"),
                }
                let reads = if route == "full-scan" { N } else { g.live };
                println!(
                    "{}\t{d}\t{}\tsupport\t{route}\t{ns:.0}\t{reads}\t0\t{reads}",
                    t.name(),
                    g.live
                );
            }
            // ---- multiplicity: K_next[via[s]] += K[s] ----
            let mut kn = vec![0u32; N];
            let ck = |kn: &[u32]| {
                kn.iter().enumerate().fold(0u64, |a, (i, x)| {
                    a.wrapping_add((*x as u64) * (i as u64 + 1))
                })
            };
            let mut want = None;
            for route in ["full-scan", "u64-gate", "u16-gate", "ordinal"] {
                let (ns, got) = time(r, || {
                    kn.iter_mut().for_each(|x| *x = 0);
                    match route {
                        "full-scan" => {
                            for s in 0..N {
                                kn[via[s] as usize] += k[s] & (bit(&m, s) as u32).wrapping_neg();
                            }
                        }
                        "u64-gate" => {
                            for (w, &word) in m.iter().enumerate() {
                                let mut b = word;
                                while b != 0 {
                                    let s = w * 64 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    kn[via[s] as usize] += k[s];
                                }
                            }
                        }
                        "u16-gate" => {
                            for c in 0..CELLS {
                                let mut b = cell(&m, c);
                                while b != 0 {
                                    let s = c * 16 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    kn[via[s] as usize] += k[s];
                                }
                            }
                        }
                        _ => {
                            buf.clear();
                            for (w, &word) in m.iter().enumerate() {
                                let mut b = word;
                                while b != 0 {
                                    buf.push((w * 64) as u16 + b.trailing_zeros() as u16);
                                    b &= b - 1;
                                }
                            }
                            for &s in buf.iter() {
                                kn[via[s as usize] as usize] += k[s as usize];
                            }
                        }
                    }
                    ck(&kn)
                });
                match want {
                    None => want = Some(got),
                    Some(w) => assert_eq!(got, w, "mult {route}"),
                }
                let reads = if route == "full-scan" { N } else { g.live };
                println!(
                    "{}\t{d}\t{}\tmultiplicity\t{route}\t{ns:.0}\t{reads}\t{reads}\t{reads}",
                    t.name(),
                    g.live
                );
            }
        }
    }
}

// ── bounded extent × aperture ──

fn mode_a4() {
    let r = reps();
    println!("mask_density\textent_frac\tlive_in_extent\tuniverse\textent_selves\tclosed_cells_in_extent\troute\tns\tts_reads\tpayload_loads");
    // non-decreasing ordered lane: ts = self / 4  (duplicates)
    let ts: Vec<i32> = (0..N as i32).map(|i| i / 4).collect();
    let val: Vec<u32> = (0..N as u64).map(u32::make).collect();
    let span = ts[N - 1] + 1;
    for md in [1.0, 0.5, 0.1, 0.01, 0.001] {
        let m = build(Layout::Uniform, md, 0xE7 ^ md.to_bits());
        for ef in [1.0, 0.5, 0.1, 0.01, 0.001] {
            let a = span / 5;
            let b = a + ((span as f64 * ef) as i32).max(1);
            let want: u64 = (0..N)
                .filter(|&i| bit(&m, i) == 1 && ts[i] >= a && ts[i] < b)
                .map(|i| val[i] as u64)
                .sum();
            let lo = ts.partition_point(|&t| t < a);
            let hi = ts.partition_point(|&t| t < b);
            let live_in: usize = (lo..hi).filter(|&i| bit(&m, i) == 1).count();
            let closed = (lo / 16..hi.div_ceil(16))
                .filter(|&c| cell(&m, c) == 0)
                .count();
            // A: full universe — predicate every row, visit (m ∧ pred)
            let (na, ga) = time(r, || {
                let mut s = 0u64;
                for i in 0..N {
                    let ok = (ts[i] >= a) & (ts[i] < b);
                    s += val[i] as u64 & ((bit(&m, i) & ok as u64).wrapping_neg());
                }
                s
            });
            // B: extent only — binary search, then every row of [lo, hi)
            let (nb, gb) = time(r, || {
                let lo = ts.partition_point(|&t| t < a);
                let hi = ts.partition_point(|&t| t < b);
                let mut s = 0u64;
                for i in lo..hi {
                    s += val[i] as u64 & bit(&m, i).wrapping_neg();
                }
                s
            });
            // C: aperture only — visit live bits of the whole universe,
            // test the ordered lane per live self
            let (nc, gc) = time(r, || {
                let mut s = 0u64;
                for c in 0..CELLS {
                    let mut bb = cell(&m, c);
                    while bb != 0 {
                        let i = c * 16 + bb.trailing_zeros() as usize;
                        bb &= bb - 1;
                        if ts[i] >= a && ts[i] < b {
                            s += val[i] as u64;
                        }
                    }
                }
                s
            });
            // D: extent + aperture — cells covering [lo, hi), edge cells
            // clipped, set bits only
            let (nd, gd) = time(r, || {
                let lo = ts.partition_point(|&t| t < a);
                let hi = ts.partition_point(|&t| t < b);
                let mut s = 0u64;
                if lo < hi {
                    let (c0, c1) = (lo / 16, (hi - 1) / 16);
                    for c in c0..=c1 {
                        let mut bb = cell(&m, c) as u32;
                        if c == c0 {
                            bb &= !0u32 << (lo % 16);
                        }
                        if c == c1 {
                            bb &= (1u32 << ((hi - 1) % 16 + 1)) - 1;
                        }
                        while bb != 0 {
                            let i = c * 16 + bb.trailing_zeros() as usize;
                            bb &= bb - 1;
                            s += val[i] as u64;
                        }
                    }
                }
                s
            });
            // E: extent + u64 words — the same with 64-bit words
            let (ne, ge) = time(r, || {
                let lo = ts.partition_point(|&t| t < a);
                let hi = ts.partition_point(|&t| t < b);
                let mut s = 0u64;
                if lo < hi {
                    let (w0, w1) = (lo / 64, (hi - 1) / 64);
                    for w in w0..=w1 {
                        let mut bb = m[w];
                        if w == w0 {
                            bb &= !0u64 << (lo % 64);
                        }
                        if w == w1 && (hi - 1) % 64 != 63 {
                            bb &= (1u64 << ((hi - 1) % 64 + 1)) - 1;
                        }
                        while bb != 0 {
                            let i = w * 64 + bb.trailing_zeros() as usize;
                            bb &= bb - 1;
                            s += val[i] as u64;
                        }
                    }
                }
                s
            });
            for (n, (ns, g, tsr, loads)) in [
                ("full-universe", (na, ga, N, N)),
                ("extent-only", (nb, gb, 34, hi - lo)),
                ("aperture-only", (nc, gc, live(&m), live(&m))),
                ("extent+aperture16", (nd, gd, 34, live_in)),
                ("extent+words64", (ne, ge, 34, live_in)),
            ] {
                assert_eq!(g, want, "{n} md={md} ef={ef}");
                println!(
                    "{md}\t{ef}\t{live_in}\t{N}\t{}\t{closed}\t{n}\t{ns:.0}\t{tsr}\t{loads}",
                    hi - lo
                );
            }
        }
    }
}

// ── K propagation over CSR edges ──

struct Csr {
    off: Vec<u32>, // N + 1
    dst: Vec<u16>,
    src_of_edge: Vec<u16>, // for the edge-list full scan
}

fn csr(t: Target, fan: usize) -> Csr {
    let mut off = Vec::with_capacity(N + 1);
    let mut dst = Vec::with_capacity(N * fan);
    let mut soe = Vec::with_capacity(N * fan);
    for s in 0..N {
        off.push(dst.len() as u32);
        for e in 0..fan {
            dst.push(
                t.of(s * fan + e)
                    ^ if matches!(t, Target::ManyToOne) {
                        0
                    } else {
                        e as u16
                    },
            );
            soe.push(s as u16);
        }
    }
    off.push(dst.len() as u32);
    Csr {
        off,
        dst,
        src_of_edge: soe,
    }
}

fn mode_a5() {
    let r = reps();
    println!("target\tfanout\tsrc_density\tlive\tedges\troute\tns\tsrc_reads\tvia_reads\tk_reads\ttarget_writes");
    let k: Vec<u32> = (0..N as u64)
        .map(|i| 1 + (mix(i ^ 0x4B) % 7) as u32)
        .collect();
    for t in [
        Target::Uniform,
        Target::Clustered,
        Target::Hotspot,
        Target::ManyToOne,
    ] {
        for fan in [1usize, 4, 16] {
            let g = csr(t, fan);
            let e = g.dst.len();
            for d in [1.0, 0.1, 0.01, 0.001] {
                let m = build(Layout::Uniform, d, 0x6B ^ d.to_bits());
                let lv = live(&m);
                let live_edges: usize = (0..N)
                    .filter(|&s| bit(&m, s) == 1)
                    .map(|s| (g.off[s + 1] - g.off[s]) as usize)
                    .sum();
                let mut kn = vec![0u32; N];
                let mut buf: Vec<u16> = Vec::with_capacity(N);
                let ck = |kn: &[u32]| {
                    kn.iter().enumerate().fold(0u64, |a, (i, x)| {
                        a.wrapping_add((*x as u64) * (i as u64 + 1))
                    })
                };
                let mut want = None;
                for route in ["edge-scan", "src-u64-gate", "src-u16-gate", "ordinal"] {
                    let (ns, got) = time(r, || {
                        kn.iter_mut().for_each(|x| *x = 0);
                        match route {
                            "edge-scan" => {
                                for i in 0..e {
                                    let s = g.src_of_edge[i] as usize;
                                    kn[g.dst[i] as usize] +=
                                        k[s] & (bit(&m, s) as u32).wrapping_neg();
                                }
                            }
                            "src-u64-gate" => {
                                for (w, &word) in m.iter().enumerate() {
                                    let mut b = word;
                                    while b != 0 {
                                        let s = w * 64 + b.trailing_zeros() as usize;
                                        b &= b - 1;
                                        let ks = k[s];
                                        for &v in &g.dst[g.off[s] as usize..g.off[s + 1] as usize] {
                                            kn[v as usize] += ks;
                                        }
                                    }
                                }
                            }
                            "src-u16-gate" => {
                                for c in 0..CELLS {
                                    let mut b = cell(&m, c);
                                    while b != 0 {
                                        let s = c * 16 + b.trailing_zeros() as usize;
                                        b &= b - 1;
                                        let ks = k[s];
                                        for &v in &g.dst[g.off[s] as usize..g.off[s + 1] as usize] {
                                            kn[v as usize] += ks;
                                        }
                                    }
                                }
                            }
                            _ => {
                                buf.clear();
                                for (w, &word) in m.iter().enumerate() {
                                    let mut b = word;
                                    while b != 0 {
                                        buf.push((w * 64) as u16 + b.trailing_zeros() as u16);
                                        b &= b - 1;
                                    }
                                }
                                for &s in buf.iter() {
                                    let s = s as usize;
                                    let ks = k[s];
                                    for &v in &g.dst[g.off[s] as usize..g.off[s + 1] as usize] {
                                        kn[v as usize] += ks;
                                    }
                                }
                            }
                        }
                        ck(&kn)
                    });
                    match want {
                        None => want = Some(got),
                        Some(w) => assert_eq!(got, w, "K {route}"),
                    }
                    let (sr, vr, kr, tw) = if route == "edge-scan" {
                        (e, e, e, e)
                    } else {
                        (lv, live_edges, lv, live_edges)
                    };
                    println!(
                        "{}\t{fan}\t{d}\t{lv}\t{e}\t{route}\t{ns:.0}\t{sr}\t{vr}\t{kr}\t{tw}",
                        t.name()
                    );
                }
            }
        }
    }
}

// ── coarse target cells: does a target aperture pay for itself? ──

fn mode_a6() {
    let r = reps();
    println!("target\tsrc_density\tlive\ttouched_target_cells\tmax_cell_contrib\troute\tns\textra_pass_reads\tconsume_cells");
    let k: Vec<u32> = (0..N as u64)
        .map(|i| 1 + (mix(i ^ 0x4B) % 7) as u32)
        .collect();
    for t in Target::ALL {
        let via: Vec<u16> = (0..N).map(|s| t.of(s)).collect();
        for d in [1.0, 0.1, 0.01, 0.001, 0.0001] {
            let m = build(Layout::Uniform, d, 0x6C ^ d.to_bits());
            let lv = live(&m);
            // reference: touched target cells, max contributions per cell
            let mut hist_ref = vec![0u32; CELLS];
            for s in (0..N).filter(|&s| bit(&m, s) == 1) {
                hist_ref[via[s] as usize >> 4] += 1;
            }
            let touched = hist_ref.iter().filter(|&&h| h > 0).count();
            let maxc = *hist_ref.iter().max().unwrap();
            // Each route: zero K_next, propagate under the source aperture,
            // then CONSUME: build the next frontier mask (K_next > 0).
            // The routes differ only in how much of K_next they zero and scan.
            let mut kn = vec![0u32; N];
            let mut next = vec![0u64; WORDS];
            let mut seen = vec![0u64; CELLS / 64];
            let mut hist = vec![0u16; CELLS];
            let mut want = None;
            for route in [
                "exact+full-consume",
                "exact+cell-seen",
                "hist-prepass+exact",
                "exact+target-mask",
            ] {
                // K_next starts clean for every route; a route that only
                // zeroes touched cells must leave it clean again.
                kn.iter_mut().for_each(|x| *x = 0);
                let (ns, got) = time(r, || {
                    next.iter_mut().for_each(|w| *w = 0);
                    match route {
                        "exact+full-consume" => {
                            kn.iter_mut().for_each(|x| *x = 0);
                            for c in 0..CELLS {
                                let mut b = cell(&m, c);
                                while b != 0 {
                                    let s = c * 16 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    kn[via[s] as usize] += k[s];
                                }
                            }
                            for (v, &x) in kn.iter().enumerate() {
                                next[v >> 6] |= ((x > 0) as u64) << (v & 63);
                            }
                        }
                        "exact+cell-seen" => {
                            seen.iter_mut().for_each(|w| *w = 0);
                            for c in 0..CELLS {
                                let mut b = cell(&m, c);
                                while b != 0 {
                                    let s = c * 16 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    let v = via[s] as usize;
                                    kn[v] += k[s];
                                    seen[v >> 10] |= 1 << ((v >> 4) & 63);
                                }
                            }
                            // consume only touched cells, and re-zero them
                            for (sw, &word) in seen.iter().enumerate() {
                                let mut b = word;
                                while b != 0 {
                                    let tc = sw * 64 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    let mut bits = 0u16;
                                    for j in 0..16 {
                                        let v = tc * 16 + j;
                                        bits |= ((kn[v] > 0) as u16) << j;
                                        kn[v] = 0;
                                    }
                                    next[tc >> 2] |= (bits as u64) << ((tc & 3) * 16);
                                }
                            }
                        }
                        "exact+target-mask" => {
                            // The exact next frontier is written in the same
                            // pass (K(u) ≥ 1 for every live u, so K_next > 0
                            // iff touched); it is then the schedule for
                            // clearing K_next. No coarse object at all.
                            for c in 0..CELLS {
                                let mut b = cell(&m, c);
                                while b != 0 {
                                    let s = c * 16 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    let v = via[s] as usize;
                                    kn[v] += k[s];
                                    next[v >> 6] |= 1 << (v & 63);
                                }
                            }
                            for (w, &word) in next.iter().enumerate() {
                                let mut b = word;
                                while b != 0 {
                                    kn[w * 64 + b.trailing_zeros() as usize] = 0;
                                    b &= b - 1;
                                }
                            }
                        }
                        _ => {
                            hist.iter_mut().for_each(|h| *h = 0);
                            for c in 0..CELLS {
                                let mut b = cell(&m, c);
                                while b != 0 {
                                    let s = c * 16 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    let h = &mut hist[via[s] as usize >> 4];
                                    *h = h.saturating_add(1);
                                }
                            }
                            for c in 0..CELLS {
                                let mut b = cell(&m, c);
                                while b != 0 {
                                    let s = c * 16 + b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    kn[via[s] as usize] += k[s];
                                }
                            }
                            for (tc, &h) in hist.iter().enumerate() {
                                if h == 0 {
                                    continue;
                                }
                                let mut bits = 0u16;
                                for j in 0..16 {
                                    let v = tc * 16 + j;
                                    bits |= ((kn[v] > 0) as u16) << j;
                                    kn[v] = 0;
                                }
                                next[tc >> 2] |= (bits as u64) << ((tc & 3) * 16);
                            }
                        }
                    }
                    next.iter()
                        .fold(0u64, |a, w| a.wrapping_mul(31).wrapping_add(*w))
                });
                match want {
                    None => want = Some(got),
                    Some(w) => assert_eq!(got, w, "a6 {route} {} {d}", t.name()),
                }
                let (extra, consume) = match route {
                    "exact+full-consume" => (0, CELLS),
                    "exact+cell-seen" => (0, touched),
                    "exact+target-mask" => (0, WORDS),
                    _ => (lv, touched),
                };
                println!(
                    "{}\t{d}\t{lv}\t{touched}\t{maxc}\t{route}\t{ns:.0}\t{extra}\t{consume}",
                    t.name()
                );
            }
        }
    }
}

/// The u16 view is the same bits as an in-place u16 reinterpretation.
fn views_agree() {
    let m = build(Layout::Uniform, 0.3, 1);
    // SAFETY: u64 -> u16 is a valid reinterpretation of initialised bytes;
    // `align_to` returns empty prefix/suffix because u64 alignment ≥ u16.
    let (pre, mid, suf) = unsafe { m.align_to::<u16>() };
    assert!(pre.is_empty() && suf.is_empty());
    if cfg!(target_endian = "little") {
        for c in 0..CELLS {
            assert_eq!(mid[c], cell(&m, c), "cell {c}");
        }
    }
    let _ = (Col(0), LaneRef::U32(&[]));
}

fn main() {
    views_agree();
    match std::env::args().nth(1).as_deref() {
        Some("a1") => mode_a1(),
        Some("a2") => mode_a2(),
        Some("occ") => mode_occ(),
        Some("a3") => mode_a3(),
        Some("a4") => mode_a4(),
        Some("a5") => mode_a5(),
        Some("a6") => mode_a6(),
        _ => eprintln!("usage: aperture16_probe a1|a2|occ|a3|a4|a5|a6"),
    }
}
