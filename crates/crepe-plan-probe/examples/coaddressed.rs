//! D-RU-4 (U4): the co-addressed join operator against the star path, over a
//! family of restricted queries on one star. Run from this crate's directory:
//!
//! ```text
//! cargo run --release --example coaddressed    # the benchmark
//! cargo test --example coaddressed             # the correctness fixture
//! ```
//!
//! # The queries
//!
//! Query `q` (one per `q` in `0..QUERIES`):
//!
//! ```text
//! Out(x, c, z) <- R(x, c), S(c, z), T(c, q)
//! ```
//!
//! # Arms
//!
//! * **C** — [`native::prepare`] per query: the restriction `T.k == q` is
//!   applied to the rows before layout, so every query lays out all three
//!   sides again.
//! * **CS** — [`native::prepare_semijoin`] per query: the same, and R and S
//!   keep only the centers the restricted T has.
//! * **O** — [`coaddressed::eval`]: the table is laid out once, without the
//!   restriction; each query runs `T.k == q` as a `Rel` over T's rows and
//!   intersects on the shared domain.
//!
//! Every arm answers the bag count and the distinct-center count of every
//! query, and the answers are compared.

use std::time::{Duration, Instant};

use crepe_plan_probe::coaddressed::{self, CoTable, plan_rels};
use crepe_plan_probe::logic::{RelId, VarId};
use crepe_plan_probe::native::{self, HeadTerm, SideKey, SidePlan, StarPlan, Store};
use mork_pathmap_probe::join::JoinExec;

const R: RelId = RelId(0);
const S: RelId = RelId(1);
const T: RelId = RelId(2);
const GENS: [u64; 3] = [1, 2, 3];

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn plan(q: Option<u64>) -> StarPlan {
    let side = |rel, key_col, restrict, emit| SidePlan {
        key: SideKey {
            rel,
            key_col,
            restrict,
            unary: false,
        },
        emit,
    };
    StarPlan {
        center: VarId(1),
        sides: vec![
            side(R, 1, None, Some(VarId(0))),
            side(S, 0, None, Some(VarId(2))),
            side(T, 0, q, None),
        ],
        head: vec![HeadTerm::Side(0), HeadTerm::Center, HeadTerm::Side(1)],
    }
}

/// `n` rows: column `0` below `a`, column `1` below `b`.
fn rows(seed: u64, n: usize, a: u32, b: u32) -> Vec<[u32; 2]> {
    let mut x = seed;
    let mut next = |m: u32| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % u64::from(m)) as u32
    };
    (0..n).map(|_| [next(a), next(b)]).collect()
}

struct Data {
    r: Vec<[u32; 2]>,
    s: Vec<[u32; 2]>,
    t: Vec<[u32; 2]>,
}

impl Data {
    fn new(n: usize, centers: u32, queries: u32) -> Data {
        Data {
            r: rows(0x9E37_79B9, n, 1_000_000, centers),
            s: rows(0x7F4A_7C15, n, centers, 1_000_000),
            t: rows(0x2545_F491, n, centers, queries),
        }
    }
    fn store(&self) -> Store<'_> {
        Store {
            relations: vec![(R, &self.r, 1), (S, &self.s, 2), (T, &self.t, 3)],
        }
    }
}

/// `(bag count, distinct centers)` per query, rows laid out or read, time.
type ArmRun = (Vec<(u64, u64)>, usize, Duration);

fn cold(data: &Data, queries: u32, semijoin: bool) -> ArmRun {
    let store = data.store();
    let start = Instant::now();
    let mut out = Vec::new();
    let mut rows = 0;
    for q in 0..queries {
        let p = plan(Some(u64::from(q)));
        let prep = if semijoin {
            native::prepare_semijoin(&p, &store)
        } else {
            native::prepare(&p, &store)
        };
        rows += prep.rows_prepared;
        let bag = JoinExec::new(&prep.table)
            .bag_count(&prep.table, &GENS, None)
            .expect("bag");
        let centers = native::distinct_centers(&prep, &p, &GENS).expect("centers");
        out.push((bag, centers));
    }
    (out, rows, start.elapsed())
}

/// `(answers, side rows read, eval time)` plus the one layout's time and rows.
fn operator(data: &Data, queries: u32) -> (ArmRun, Duration, usize) {
    let store = data.store();
    let start = Instant::now();
    let table = CoTable::prepare(&plan(None), &store);
    let prepare = start.elapsed();
    let start = Instant::now();
    let mut out = Vec::new();
    let mut rows = 0;
    for q in 0..queries {
        let p = plan(Some(u64::from(q)));
        let ev = coaddressed::eval(&table, &p, &plan_rels(&p), &GENS).expect("eval");
        assert!(ev.programs.iter().all(|p| !native::has_foreign_hop(p)));
        rows += ev.cost.side_rows_read;
        out.push((ev.bag_count(&table), ev.distinct_centers()));
    }
    ((out, rows, start.elapsed()), prepare, table.rows())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let n = env("BENCH_ROWS", 200_000);
    let centers = env("BENCH_CENTERS", 20_000) as u32;
    let queries = env("BENCH_QUERIES", 64) as u32;
    let reps = env("BENCH_REPS", 3);
    println!(
        "simd: avx512f={}  rows/relation={n}  centers={centers}  queries={queries}  reps={reps}",
        cfg!(target_feature = "avx512f")
    );
    let data = Data::new(n, centers, queries);
    let mut best: [Option<(f64, usize)>; 3] = [None; 3];
    let mut layout = (f64::MAX, 0);
    for _ in 0..reps {
        let c = cold(&data, queries, false);
        let cs = cold(&data, queries, true);
        let ((o, o_rows, o_t), prep, prep_rows) = operator(&data, queries);
        assert_eq!(c.0, o, "C and O disagree");
        assert_eq!(cs.0, o, "CS and O disagree");
        for (slot, (t, r)) in best
            .iter_mut()
            .zip([(ms(c.2), c.1), (ms(cs.2), cs.1), (ms(o_t), o_rows)])
        {
            if slot.is_none_or(|(b, _)| t < b) {
                *slot = Some((t, r));
            }
        }
        layout.0 = layout.0.min(ms(prep));
        layout.1 = prep_rows;
    }
    let bag: u64 = cold(&data, queries, false).0.iter().map(|a| a.0).sum();
    println!("answers equal across arms; total bag rows over all queries: {bag}");
    println!("{:<4} {:>12} {:>16}", "arm", "ms (best)", "rows touched");
    for (name, b) in ["C", "CS", "O"].iter().zip(best) {
        let (t, r) = b.expect("ran");
        println!("{name:<4} {t:>12.2} {r:>16}");
    }
    println!(
        "O layout (once): {:.2} ms, {} rows; O total incl. layout: {:.2} ms",
        layout.0,
        layout.1,
        layout.0 + best[2].expect("ran").0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All three arms give the same per-query answers, and the fixture has
    /// answers to compare.
    #[test]
    fn arms_agree() {
        let data = Data::new(3_000, 200, 8);
        let c = cold(&data, 8, false);
        let cs = cold(&data, 8, true);
        let ((o, _, _), _, _) = operator(&data, 8);
        assert_eq!(c.0, o);
        assert_eq!(cs.0, o);
        assert!(o.iter().filter(|a| a.0 > 0).count() >= 6, "{o:?}");
    }
}
