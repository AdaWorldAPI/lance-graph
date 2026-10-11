//! D-PHOTO-0: exposing 64 resident populations instead of recomputing them.
//!
//! 64 membership bitplanes `P0..P63` over one universe of 2^20 addresses are
//! built once. Each query is a Boolean expression over them ("an exposure"),
//! answered as COUNT, EXISTS, KEEP or SUM. The question is what a NEW
//! expression costs once the populations exist, and what each cached
//! representation costs to build, update, spill and reload.
//!
//! | arm | representation | per-query work |
//! |---|---|---|
//! | B  | 64 bitplanes, Quack `lower_fused` → mask-risc | the words of the planes the expression names |
//! | Bi | same, Quack `lower` (in place) | same words, more passes |
//! | D  | B over the expression's support range only (`execute_extent`) | the words inside the range |
//! | C  | one `u64` signature per address (probe-local transpose) | every address |
//! | E  | signature → count classes (probe-local) | one test per distinct signature |
//!
//! There is no executable "reasoning" producing these populations in the
//! tree, so arm A (recompute the reasoning per query) has no honest
//! baseline; the generator's cost is printed as population-creation cost
//! and labelled as such. C and E are probe-local because nothing in
//! lance-graph or ndarray transposes bitplanes to per-address signatures.
//!
//! Every answer is checked against a per-address oracle that reads the
//! planes bit by bit (on the first queries of each batch; the repeated-query
//! timings reuse checked expressions).
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-quack --example photo_probe
//! ```

use std::collections::HashMap;
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, execute_into, Scratch};
use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Out, Planes, Program, Value};
use lance_graph_quack::{lower, lower_fused, Agg, Col, Filter, Mask, Query};

const N: usize = 1 << 20;
const W: usize = N / 64;
const P: usize = 64;

// ───────────────────────── populations ─────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, m: u64) -> u64 {
        self.next() % m
    }
}

#[derive(Clone, Copy, Debug)]
enum Regime {
    /// Independent populations, each ~1% of addresses.
    Sparse,
    /// Independent populations, each ~50%.
    Dense,
    /// Each population is 4 runs of ~4,096 addresses: mostly empty words.
    Clustered,
    /// Every address copies one of 256 archetype signatures: low diversity,
    /// heavily overlapping populations.
    Archetypes,
}

/// Build the 64 planes; the time this takes stands in for "the reasoning".
fn populations(regime: Regime) -> Vec<Vec<u64>> {
    let mut r = Rng(0xF0_70 + regime as u64);
    let mut planes = vec![vec![0u64; W]; P];
    match regime {
        Regime::Sparse | Regime::Dense => {
            let d = if matches!(regime, Regime::Sparse) {
                1
            } else {
                50
            };
            for p in planes.iter_mut() {
                for a in 0..N {
                    if r.below(100) < d {
                        p[a / 64] |= 1 << (a % 64);
                    }
                }
            }
        }
        Regime::Clustered => {
            for p in planes.iter_mut() {
                for _ in 0..4 {
                    let lo = r.below((N - 4096) as u64) as usize;
                    for a in lo..lo + 4096 {
                        p[a / 64] |= 1 << (a % 64);
                    }
                }
            }
        }
        Regime::Archetypes => {
            let arche: Vec<u64> = (0..256).map(|_| r.next()).collect();
            for a in 0..N {
                let s = arche[r.below(256) as usize];
                for (i, p) in planes.iter_mut().enumerate() {
                    if s >> i & 1 == 1 {
                        p[a / 64] |= 1 << (a % 64);
                    }
                }
            }
        }
    }
    planes
}

fn bit(p: &[u64], a: usize) -> bool {
    p[a / 64] >> (a % 64) & 1 == 1
}

// ───────────────────────── expressions ─────────────────────────

#[derive(Clone, Debug)]
enum Ex {
    Leaf(u16),
    And(Vec<Ex>),
    Or(Vec<Ex>),
    Not(Box<Ex>),
}

impl Ex {
    fn filter(&self) -> Filter {
        match self {
            Ex::Leaf(i) => Filter::Plane(Mask(*i)),
            Ex::And(v) => Filter::And(v.iter().map(Ex::filter).collect()),
            Ex::Or(v) => Filter::Or(v.iter().map(Ex::filter).collect()),
            Ex::Not(e) => Filter::Not(Box::new(e.filter())),
        }
    }
    /// The oracle reads the planes, one bit per leaf per address.
    fn on_planes(&self, planes: &[Vec<u64>], a: usize) -> bool {
        match self {
            Ex::Leaf(i) => bit(&planes[*i as usize], a),
            Ex::And(v) => v.iter().all(|e| e.on_planes(planes, a)),
            Ex::Or(v) => v.iter().any(|e| e.on_planes(planes, a)),
            Ex::Not(e) => !e.on_planes(planes, a),
        }
    }
    fn on_sig(&self, s: u64) -> bool {
        match self {
            Ex::Leaf(i) => s >> i & 1 == 1,
            Ex::And(v) => v.iter().all(|e| e.on_sig(s)),
            Ex::Or(v) => v.iter().any(|e| e.on_sig(s)),
            Ex::Not(e) => !e.on_sig(s),
        }
    }
    fn leaves(&self, out: &mut Vec<u16>) {
        match self {
            Ex::Leaf(i) => {
                if !out.contains(i) {
                    out.push(*i)
                }
            }
            Ex::And(v) | Ex::Or(v) => v.iter().for_each(|e| e.leaves(out)),
            Ex::Not(e) => e.leaves(out),
        }
    }
    /// A row range outside which the expression is certainly false, from
    /// each population's own occupied range: AND intersects, OR takes the
    /// hull, NOT gives up. The same algebra as quack's `fold_ranges`, at the
    /// population level.
    fn support(&self, occ: &[(usize, usize)]) -> (usize, usize) {
        match self {
            Ex::Leaf(i) => occ[*i as usize],
            Ex::And(v) => v
                .iter()
                .map(|e| e.support(occ))
                .fold((0, N), |(a, b), (c, d)| {
                    let lo = a.max(c);
                    (lo, b.min(d).max(lo))
                }),
            Ex::Or(v) => v
                .iter()
                .map(|e| e.support(occ))
                .filter(|(a, b)| a < b)
                .fold((N, 0), |(a, b), (c, d)| (a.min(c), b.max(d)))
                .pipe(|(a, b)| if a > b { (0, 0) } else { (a, b) }),
            Ex::Not(_) => (0, N),
        }
    }
}

trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

/// The canonical example: (P0 | P1 | P2 | P3) & P17 & !P29.
fn canonical() -> Ex {
    Ex::And(vec![
        Ex::Or((0..4).map(Ex::Leaf).collect()),
        Ex::Leaf(17),
        Ex::Not(Box::new(Ex::Leaf(29))),
    ])
}

/// A random expression of 3-7 distinct leaves, mixing AND / OR / NOT.
fn random_ex(r: &mut Rng) -> Ex {
    let k = 3 + r.below(5) as usize;
    let mut leaves: Vec<u16> = Vec::new();
    while leaves.len() < k {
        let l = r.below(P as u64) as u16;
        if !leaves.contains(&l) {
            leaves.push(l);
        }
    }
    let mut terms: Vec<Ex> = leaves
        .into_iter()
        .map(|l| {
            if r.below(4) == 0 {
                Ex::Not(Box::new(Ex::Leaf(l)))
            } else {
                Ex::Leaf(l)
            }
        })
        .collect();
    while terms.len() > 1 {
        let b = terms.pop().unwrap();
        let a = terms.pop().unwrap();
        terms.push(if r.below(2) == 0 {
            Ex::And(vec![a, b])
        } else {
            Ex::Or(vec![a, b])
        });
    }
    terms.pop().unwrap()
}

// ───────────────────────── execution ─────────────────────────

fn run(
    p: &Program,
    planes: &Planes<'_>,
    extent: (usize, usize),
    keep: Option<&mut [u64]>,
) -> Value {
    let mut s = Scratch::for_program(p, N).expect("scratch");
    let out = match keep {
        Some(k) => Out::Mask(k),
        None => Out::None,
    };
    if extent == (0, N) {
        execute_into(p, planes, &Foreign::NONE, &mut s, out).expect("runs")
    } else {
        execute_extent(p, planes, &Foreign::NONE, &mut s, out, extent.0..extent.1).expect("runs")
    }
}

fn count_of(v: Value) -> u64 {
    match v {
        Value::Count(c) => c as u64,
        other => panic!("{other:?}"),
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

// ───────────────────────── main ─────────────────────────

fn main() {
    let vals: Vec<i32> = (0..N).map(|i| (i % 1000) as i32).collect();
    for regime in [
        Regime::Sparse,
        Regime::Dense,
        Regime::Clustered,
        Regime::Archetypes,
    ] {
        section(regime, &vals);
    }
}

fn section(regime: Regime, vals: &[i32]) {
    let t = Instant::now();
    let pl = populations(regime);
    let create_ms = ms(t);
    let masks: Vec<&[u64]> = pl.iter().map(Vec::as_slice).collect();
    let lanes = [LaneRef::I32(vals)];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };

    // Population descriptors: occupied word range per plane (for D).
    let t = Instant::now();
    let occ: Vec<(usize, usize)> = pl
        .iter()
        .map(|p| {
            let lo = p.iter().position(|w| *w != 0).map_or(N, |w| w * 64);
            let hi = p
                .iter()
                .rposition(|w| *w != 0)
                .map_or(0, |w| ((w + 1) * 64).min(N));
            if lo >= hi {
                (0, 0)
            } else {
                (lo, hi)
            }
        })
        .collect();
    let occ_ms = ms(t);

    // C: transpose to one signature per address.
    let t = Instant::now();
    let mut sig = vec![0u64; N];
    for (i, p) in pl.iter().enumerate() {
        for (w, &word) in p.iter().enumerate() {
            let mut x = word;
            while x != 0 {
                let b = x.trailing_zeros() as usize;
                sig[w * 64 + b] |= 1 << i;
                x &= x - 1;
            }
        }
    }
    let transpose_ms = ms(t);

    // E: signature → (count, sum of vals).
    let t = Instant::now();
    let mut classes: HashMap<u64, (u64, i64)> = HashMap::new();
    for (a, &s) in sig.iter().enumerate() {
        let e = classes.entry(s).or_insert((0, 0));
        e.0 += 1;
        e.1 += i64::from(vals[a]);
    }
    let class_ms = ms(t);

    println!(
        "\n== {regime:?}: create {create_ms:.0} ms (stand-in for reasoning) | occupied ranges {occ_ms:.2} ms | transpose {transpose_ms:.0} ms | {} signature classes built in {class_ms:.0} ms ==",
        classes.len()
    );
    println!(
        "   resident: planes {} KiB, signatures {} KiB, classes ~{} KiB",
        P * W * 8 / 1024,
        N * 8 / 1024,
        classes.len() * 24 / 1024
    );

    // Correctness: the canonical expression plus 20 random ones, every
    // representation and every terminal, against the per-address oracle.
    let mut r = Rng(0xE7_905E);
    let mut checked: Vec<Ex> = vec![canonical()];
    checked.extend((0..20).map(|_| random_ex(&mut r)));
    for ex in &checked {
        let want: Vec<usize> = (0..N).filter(|&a| ex.on_planes(&pl, a)).collect();
        let want_sum: i64 = want.iter().map(|&a| i64::from(vals[a])).sum();
        let f = ex.filter();
        let sup = ex.support(&occ);
        for (fused, ext) in [(true, (0, N)), (false, (0, N)), (true, sup)] {
            let lo = |agg| {
                let q = Query {
                    filter: f.clone(),
                    agg,
                };
                if fused { lower_fused(&q) } else { lower(&q) }.expect("lowers")
            };
            let c = if ext.0 >= ext.1 {
                0
            } else {
                count_of(run(&lo(Agg::Count), &planes, ext, None))
            };
            assert_eq!(
                c,
                want.len() as u64,
                "{regime:?} count {ex:?} fused={fused} ext={ext:?}"
            );
            if ext.0 < ext.1 {
                let any = matches!(run(&lo(Agg::Any), &planes, ext, None), Value::Bool(true));
                assert_eq!(any, !want.is_empty());
                let s = match run(&lo(Agg::SumI32(Col(0))), &planes, ext, None) {
                    Value::SumI64(s) => s,
                    other => panic!("{other:?}"),
                };
                assert_eq!(s, want_sum);
                let mut kept = vec![0u64; words_for(N)];
                run(&lo(Agg::Rows), &planes, ext, Some(&mut kept));
                let got: Vec<usize> = (0..N).filter(|&a| bit(&kept, a)).collect();
                assert_eq!(got, want);
            }
        }
        let c_sig = sig.iter().filter(|&&s| ex.on_sig(s)).count();
        assert_eq!(c_sig, want.len());
        let (c_cls, s_cls) = classes
            .iter()
            .filter(|(s, _)| ex.on_sig(**s))
            .fold((0u64, 0i64), |(c, s), (_, v)| (c + v.0, s + v.1));
        assert_eq!((c_cls, s_cls), (want.len() as u64, want_sum));
    }
    println!("   correctness: canonical + 20 random expressions, all arms and terminals equal the oracle");

    // Work and time per new exposure: 1..10,000 fresh expressions, COUNT.
    println!(
        "   {:>6} {:>12} {:>12} {:>12} {:>12} {:>12} | words/query B {:>7} D {:>7}",
        "q", "B fused", "B in-place", "D extent", "C sigs", "E classes", "", ""
    );
    for nq in [1usize, 10, 100, 1_000, 10_000] {
        let mut r = Rng(0x0901 + nq as u64);
        let exs: Vec<Ex> = (0..nq).map(|_| random_ex(&mut r)).collect();
        let mut words_b = 0usize;
        let mut words_d = 0usize;
        let mut times = [0f64; 5];
        let mut answers = [0u64; 5];
        for ex in &exs {
            let f = ex.filter();
            let mut leaves = Vec::new();
            ex.leaves(&mut leaves);
            let sup = ex.support(&occ);
            words_b += leaves.len() * W;
            words_d += leaves.len() * (sup.1.saturating_sub(sup.0)).div_ceil(64);
            let q = Query {
                filter: f,
                agg: Agg::Count,
            };
            let t = Instant::now();
            answers[0] += count_of(run(&lower_fused(&q).unwrap(), &planes, (0, N), None));
            times[0] += ms(t);
            let t = Instant::now();
            answers[1] += count_of(run(&lower(&q).unwrap(), &planes, (0, N), None));
            times[1] += ms(t);
            let t = Instant::now();
            if sup.0 < sup.1 {
                answers[2] += count_of(run(&lower_fused(&q).unwrap(), &planes, sup, None));
            }
            times[2] += ms(t);
            // C and E cost the same per query at every repetition count, so
            // they stop at 1,000 queries; their per-query time is still the
            // mean over the queries they ran.
            if nq <= 1_000 {
                let t = Instant::now();
                answers[3] += sig.iter().filter(|&&s| ex.on_sig(s)).count() as u64;
                times[3] += ms(t);
                let t = Instant::now();
                answers[4] += classes
                    .iter()
                    .filter(|(s, _)| ex.on_sig(**s))
                    .map(|(_, v)| v.0)
                    .sum::<u64>();
                times[4] += ms(t);
            }
        }
        assert!(
            answers[..3].iter().all(|&a| a == answers[0]),
            "{regime:?} {answers:?}"
        );
        if nq <= 1_000 {
            assert!(
                answers[3] == answers[0] && answers[4] == answers[0],
                "{regime:?} {answers:?}"
            );
        }
        if nq > 1_000 {
            // Not run: printed as NaN, never as a time.
            times[3] = f64::NAN;
            times[4] = f64::NAN;
        }
        println!(
            "   {:>6} {:>10.3}ms {:>10.3}ms {:>10.3}ms {:>10.3}ms {:>10.3}ms | per query  {:>7} {:>7}",
            nq,
            times[0] / nq as f64,
            times[1] / nq as f64,
            times[2] / nq as f64,
            times[3] / nq as f64,
            times[4] / nq as f64,
            words_b / nq,
            words_d / nq
        );
    }

    // Incremental update: flip 1,000 addresses of P5.
    let mut r = Rng(0x0DE1);
    let flips: Vec<usize> = (0..1000).map(|_| r.below(N as u64) as usize).collect();
    let mut pl2 = pl.clone();
    let mut sig2 = sig.clone();
    let mut classes2 = classes.clone();
    let t = Instant::now();
    for &a in &flips {
        pl2[5][a / 64] ^= 1 << (a % 64);
    }
    let upd_b = ms(t);
    let t = Instant::now();
    for &a in &flips {
        let old = sig2[a];
        let new = old ^ (1 << 5);
        sig2[a] = new;
        let e = classes2.get_mut(&old).unwrap();
        e.0 -= 1;
        e.1 -= i64::from(vals[a]);
        if e.0 == 0 {
            classes2.remove(&old);
        }
        let e = classes2.entry(new).or_insert((0, 0));
        e.0 += 1;
        e.1 += i64::from(vals[a]);
    }
    let upd_ce = ms(t);
    let t = Instant::now();
    let p = &pl2[5];
    let _ = (
        p.iter().position(|w| *w != 0),
        p.iter().rposition(|w| *w != 0),
    );
    let upd_d = ms(t);
    println!("   update 1,000 addresses of one population: planes {upd_b:.3} ms, signatures+classes {upd_ce:.3} ms, recompute its occupied range {upd_d:.3} ms");

    // Spill and reload: planes vs signatures, to a local file.
    let path =
        std::env::temp_dir().join(format!("photo_probe_{}_{regime:?}.bin", std::process::id()));
    let bytes: Vec<u8> = pl
        .iter()
        .flat_map(|p| p.iter().flat_map(|w| w.to_le_bytes()))
        .collect();
    let t = Instant::now();
    std::fs::write(&path, &bytes).unwrap();
    let spill = ms(t);
    let t = Instant::now();
    let back = std::fs::read(&path).unwrap();
    let reload = ms(t);
    assert_eq!(back.len(), bytes.len());
    let _ = std::fs::remove_file(&path);
    // Sparse spill: only the nonzero words, as (word index, word).
    let nz: usize = pl
        .iter()
        .map(|p| p.iter().filter(|w| **w != 0).count())
        .sum();
    println!(
        "   spill planes {} KiB in {spill:.1} ms, reload {reload:.1} ms (page cache, not cold) | nonzero words {} of {} ({:.1}% → sparse spill ~{} KiB)",
        bytes.len() / 1024,
        nz,
        P * W,
        100.0 * nz as f64 / (P * W) as f64,
        nz * 12 / 1024
    );
}
