//! Per-call cost of the bound NAL operations against a scalar row loop.
//!
//! `cargo run --release --manifest-path crates/lance-graph-nal/Cargo.toml --example nal_cost`
//!
//! Every answer is asserted equal to the row loop before anything is timed.

use std::hint::black_box;
use std::time::Instant;

use lance_graph_nal::{nal, Belief, Beliefs, Copula};

fn fixture(n: usize) -> Vec<Belief> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let r = z ^ (z >> 31);
            Belief {
                s: (r % 4096) as u16,
                copula: Copula::ALL[((r >> 12) % 4) as usize],
                p: ((r >> 16) % 4096) as u16,
                f: (r >> 32) as u8,
                c: (r >> 40) as u8,
            }
        })
        .collect()
}

/// Median nanoseconds per call over 101 batches of `reps` calls.
fn median_ns(reps: u32, mut f: impl FnMut() -> usize) -> f64 {
    let mut v: Vec<f64> = (0..101)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                black_box(f());
            }
            t.elapsed().as_nanos() as f64 / f64::from(reps)
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[50]
}

fn main() {
    let simd = if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else {
        "baseline"
    };
    println!("target: {simd}");
    for n in [65_536usize, 1_048_576] {
        let input = fixture(n);
        let beliefs = Beliefs::seal(0x0E01_0001, &input, 1).unwrap();
        let b = beliefs.bind(&beliefs.witness()).unwrap();
        let st = input[n / 2].statement();
        let s = st.s;

        let q = nal::Question::bind(&b, st).unwrap();
        let mut qs = q.scratch(&b).unwrap();
        let t = nal::TruthAtLeast::bind(&b, st, 128).unwrap();
        let mut ts = t.scratch(&b).unwrap();
        let sup = nal::Support::bind(&b, s).unwrap();
        let mut ss = sup.scratch(&b).unwrap();
        let across = nal::TruthAtLeastAcross::bind(&b, s, 128).unwrap();
        let mut asx = across.scratch(&b).unwrap();

        let same = |x: &Belief| x.statement() == st;
        assert_eq!(
            q.count(&b, &mut qs).unwrap(),
            input.iter().filter(|x| same(x)).count()
        );
        assert_eq!(
            t.count(&b, &mut ts).unwrap(),
            input.iter().filter(|x| same(x) && x.c >= 128).count()
        );
        assert_eq!(
            sup.count(&b, &mut ss).unwrap(),
            input.iter().filter(|x| x.s == s).count()
        );
        assert_eq!(
            across.count(&b, &mut asx).unwrap(),
            input.iter().filter(|x| x.s == s && x.c >= 128).count()
        );

        println!(
            "n = {n}  (subject support = {})",
            sup.count(&b, &mut ss).unwrap()
        );
        let r = |name: &str, ns: f64| println!("  {name:<34} {ns:>12.1} ns");
        r(
            "bind Question (search + procedure)",
            median_ns(100, || {
                nal::Question::bind(&b, st).unwrap().program().ops.len()
            }),
        );
        r(
            "Question::count (kept program)",
            median_ns(10_000, || q.count(&b, &mut qs).unwrap()),
        );
        r(
            "TruthAtLeast::count",
            median_ns(10_000, || t.count(&b, &mut ts).unwrap()),
        );
        r(
            "Support::count",
            median_ns(10_000, || sup.count(&b, &mut ss).unwrap()),
        );
        r(
            "TruthAtLeastAcross::count",
            median_ns(1_000, || across.count(&b, &mut asx).unwrap()),
        );
        r(
            "Choice::of",
            median_ns(10_000, || {
                nal::Choice::of(&b, st).map_or(0, |x| x.c as usize)
            }),
        );
        r(
            "row loop: count statement",
            median_ns(3, || input.iter().filter(|x| same(x)).count()),
        );
        r(
            "row loop: subject with c >= 128",
            median_ns(3, || {
                input.iter().filter(|x| x.s == s && x.c >= 128).count()
            }),
        );
    }
}
