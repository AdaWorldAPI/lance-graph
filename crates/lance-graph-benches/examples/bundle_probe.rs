//! D-BIND-BUNDLE / D-FOLD-CARRIER probe: one bound query, several physical
//! routes, the same answer.
//!
//! The query (already bound: names are gone, only lanes and planes remain):
//!
//! ```text
//! rows where sel < t  AND  dept_fk ∈ allowed_depts      -> count, sum(age)
//! ```
//!
//! `sel` is an i32 lane, `dept_fk` a u32 foreign key into a 1 024-row Dept
//! table whose resident plane marks the allowed departments (half of them).
//! `t` sweeps the density of the first conjunct from 100 % to 0.01 %.
//!
//! Routes ("bundles"), all over the SAME `quack::Query` value:
//! - `gated`   : `quack::lower` — conjuncts chained, each later predicate
//!   evaluated only in 64-row words that still have survivors;
//! - `gated-r` : `quack::lower` with the conjuncts in the opposite order
//!   (semijoin first) — the same meaning, a different skip pattern;
//! - `fused`   : `quack::lower_fused` — per-predicate slots, ternlog skeleton,
//!   no skipping;
//! - `ordinal` : a probe-local sorted-ordinal carrier — build a `Vec<u32>` of
//!   surviving rows for the first conjunct, filter it by dept membership,
//!   sum. This is exactly the selection vector the quack translation matrix
//!   rules ELIMINATE (row R1); it exists here only to measure that ruling.
//!
//! Two layouts: `uniform` (sel is a hash, survivors scattered over every
//! word) and `clustered` (sel = row index, survivors contiguous).
//!
//! Every route's answer is asserted equal to a row-at-a-time oracle before
//! any time is printed.
//!
//! ```bash
//! cargo run --release -p lance-graph-benches --example bundle_probe
//! ```

use std::time::Instant;

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, ForeignPlane, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{
    lower, lower_fused, Agg, Cmp, Col, Filter, ForeignPlane as FP, Mask, Query,
};

const N: usize = 1_000_000;
const DEPTS: usize = 1024;
const SEL: Col = Col(0);
const DEPT: Col = Col(1);
const AGE: Col = Col(2);
const LIVE: Mask = Mask(0);
const DEPT_PLANE: FP = FP(0);

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn ones(n: usize) -> Vec<u64> {
    let mut w = vec![u64::MAX; words_for(n)];
    if !n.is_multiple_of(64) {
        *w.last_mut().unwrap() = (1u64 << (n % 64)) - 1;
    }
    w
}

struct Data {
    sel: Vec<i32>,
    dept: Vec<u32>,
    age: Vec<i32>,
    live: Vec<u64>,
    allowed: Vec<u64>,
}

impl Data {
    fn new(clustered: bool) -> Self {
        let sel = (0..N as u64)
            .map(|i| {
                if clustered {
                    i as i32
                } else {
                    (mix(i) % N as u64) as i32
                }
            })
            .collect();
        let dept = (0..N as u64)
            .map(|i| (mix(i ^ 0xD3) % DEPTS as u64) as u32)
            .collect();
        let age = (0..N as u64)
            .map(|i| 18 + (mix(i ^ 0xA6) % 60) as i32)
            .collect();
        let mut allowed = vec![0u64; words_for(DEPTS)];
        for d in (0..DEPTS).filter(|d| mix(*d as u64 ^ 0x51).is_multiple_of(2)) {
            allowed[d / 64] |= 1 << (d % 64);
        }
        Data {
            sel,
            dept,
            age,
            live: ones(N),
            allowed,
        }
    }
    fn allowed(&self, d: u32) -> bool {
        self.allowed[d as usize / 64] >> (d % 64) & 1 == 1
    }
    fn oracle(&self, t: i32) -> (u64, i64) {
        let (mut c, mut s) = (0u64, 0i64);
        for r in 0..N {
            if self.sel[r] < t && self.allowed(self.dept[r]) {
                c += 1;
                s += self.age[r] as i64;
            }
        }
        (c, s)
    }
    fn run(&self, program: &Program) -> Value {
        let lanes = [
            LaneRef::I32(&self.sel),
            LaneRef::U32(&self.dept),
            LaneRef::I32(&self.age),
        ];
        let masks: [&[u64]; 1] = [&self.live];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        let fp = [ForeignPlane {
            words: &self.allowed,
            rows: DEPTS,
        }];
        let foreign = Foreign {
            planes: &fp,
            lanes: &[],
        };
        let mut scratch = Scratch::for_program(program, N).expect("scratch");
        execute_into(program, &planes, &foreign, &mut scratch, Out::None).expect("executes")
    }
    /// A mask-native emulation of a gated `Gather` (mask-risc has none): the
    /// first conjunct is kept as a MASK (no ordinals), then the dept lookup and
    /// the fold run only inside words of that mask that have survivors. This
    /// is what `MaskOp::Gather { under }` would buy; it materialises no
    /// selection vector.
    fn gated_gather(&self, keep_cmp: &Program, mask: &mut [u64]) -> (u64, i64) {
        let lanes = [
            LaneRef::I32(&self.sel),
            LaneRef::U32(&self.dept),
            LaneRef::I32(&self.age),
        ];
        let masks: [&[u64]; 1] = [&self.live];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        let mut scratch = Scratch::for_program(keep_cmp, N).expect("scratch");
        execute_into(
            keep_cmp,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::Mask(mask),
        )
        .expect("keeps");
        let (mut c, mut s) = (0u64, 0i64);
        for (w, &bits) in mask.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                let r = w * 64 + b.trailing_zeros() as usize;
                b &= b - 1;
                if self.allowed(self.dept[r]) {
                    c += 1;
                    s += self.age[r] as i64;
                }
            }
        }
        (c, s)
    }
    /// The selection-vector carrier: materialise the first conjunct's
    /// survivors as sorted ordinals, then narrow and fold over them.
    fn ordinal(&self, t: i32) -> (u64, i64) {
        let rows: Vec<u32> = (0..N as u32)
            .filter(|&r| self.sel[r as usize] < t)
            .collect();
        let (mut c, mut s) = (0u64, 0i64);
        for &r in &rows {
            if self.allowed(self.dept[r as usize]) {
                c += 1;
                s += self.age[r as usize] as i64;
            }
        }
        (c, s)
    }
}

fn median_ns(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<u128> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_nanos()
        })
        .collect();
    v.sort_unstable();
    v[v.len() / 2] as f64
}

fn main() {
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(31);
    println!(
        "layout\tdensity\tsurvivors\tgated_us\tgated_r_us\tfused_us\tordinal_us\tordinal_bytes\tcmp_only_us\tvia_only_us\tgated_count_us\tgated_gather_us"
    );
    for clustered in [false, true] {
        let d = Data::new(clustered);
        for density in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001] {
            let t = (N as f64 * density) as i32;
            let cmp = Filter::cmp(SEL, Cmp::LtI32(t));
            let via = Filter::semijoin(DEPT, DEPT_PLANE);
            let q = |f: Filter, agg| Query { filter: f, agg };
            let fwd = || Filter::and([Filter::plane(LIVE), cmp.clone(), via.clone()]);
            let rev = || Filter::and([Filter::plane(LIVE), via.clone(), cmp.clone()]);
            let bundles: [(&str, Program, Program); 3] = [
                (
                    "gated",
                    lower(&q(fwd(), Agg::Count)).unwrap(),
                    lower(&q(fwd(), Agg::SumI32(AGE))).unwrap(),
                ),
                (
                    "gated-r",
                    lower(&q(rev(), Agg::Count)).unwrap(),
                    lower(&q(rev(), Agg::SumI32(AGE))).unwrap(),
                ),
                (
                    "fused",
                    lower_fused(&q(fwd(), Agg::Count)).unwrap(),
                    lower_fused(&q(fwd(), Agg::SumI32(AGE))).unwrap(),
                ),
            ];
            let want = d.oracle(t);
            // correctness first: every route, both terminals
            for (name, count_p, sum_p) in &bundles {
                let c = match d.run(count_p) {
                    Value::Count(c) => c as u64,
                    v => panic!("{name}: {v:?}"),
                };
                let s = match d.run(sum_p) {
                    Value::SumI64(s) => s,
                    v => panic!("{name}: {v:?}"),
                };
                assert_eq!((c, s), want, "{name} at density {density}");
            }
            assert_eq!(d.ordinal(t), want, "ordinal at density {density}");
            let keep_cmp = lower(&q(
                Filter::and([Filter::plane(LIVE), cmp.clone()]),
                Agg::Rows,
            ))
            .unwrap();
            let mut mask = vec![0u64; words_for(N)];
            assert_eq!(
                d.gated_gather(&keep_cmp, &mut mask),
                want,
                "gated-gather at density {density}"
            );

            let time = |p: &Program| {
                median_ns(reps, || {
                    let _ = std::hint::black_box(d.run(p));
                }) / 1e3
            };
            let gated = time(&bundles[0].2);
            let gated_r = time(&bundles[1].2);
            let fused = time(&bundles[2].2);
            let ordinal = median_ns(reps, || {
                let _ = std::hint::black_box(d.ordinal(t));
            }) / 1e3;
            let first = (0..N).filter(|&r| d.sel[r] < t).count();
            // controls: which conjunct carries the cost, and the terminal's share
            let cmp_only = time(
                &lower(&q(
                    Filter::and([Filter::plane(LIVE), cmp.clone()]),
                    Agg::Count,
                ))
                .unwrap(),
            );
            let via_only = time(
                &lower(&q(
                    Filter::and([Filter::plane(LIVE), via.clone()]),
                    Agg::Count,
                ))
                .unwrap(),
            );
            let gated_count = time(&bundles[0].1);
            let gg = median_ns(reps, || {
                let _ = std::hint::black_box(d.gated_gather(&keep_cmp, &mut mask));
            }) / 1e3;
            println!(
                "{}\t{density}\t{}\t{gated:.1}\t{gated_r:.1}\t{fused:.1}\t{ordinal:.1}\t{}\t{cmp_only:.1}\t{via_only:.1}\t{gated_count:.1}\t{gg:.1}",
                if clustered { "clustered" } else { "uniform" },
                want.0,
                first * 4,
            );
        }
    }
}
