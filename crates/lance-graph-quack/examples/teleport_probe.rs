//! D-TELEPORT-0: does a fold restricted at the source ELIMINATE work, or only
//! do the same work faster?
//!
//! The question being tested: whether `R(s, o) = K(s, o) AND B(o)` can be
//! answered by addressing only the rows that can survive ("teleportation"),
//! instead of scanning, joining and filtering ("navigation"). A speedup alone
//! does not settle it: AVX-512 or ternlog fusion make a scan faster without
//! skipping any of it. So every arm reports WORK COUNTERS computed from the
//! masks themselves, next to its time.
//!
//! Nothing here is a production primitive. Every arm runs the shipped
//! `lance-graph-mask-risc` executor. The probe adds no op, no lowering and no
//! kernel; the one probe-local loop is the row-at-a-time oracle every count is
//! asserted against. The binary searches that find a range are
//! `slice::partition_point`, the same call `ordered_lane` uses.
//!
//! Edges have an `i32` source lane `s` and a `u32` object lane `o` that holds
//! the node's ROW ORDINAL (the precondition for O(1) address-to-row). `B` is a
//! RESIDENT bitmap over the node table. `K = s < t`.
//!
//! | arm | program | what it is |
//! |---|---|---|
//! | A  | `Pred K`, ungated `Gather(o -> B)`, `And`, `Count` | scan, join, filter |
//! | B  | `Pred K`, `Gather(o -> B) under K`, `Count` | fused: gather only K's survivors |
//! | C  | `Gather(o -> B)`, `Count`, over `execute_extent(0..hi)` | source-restricted: `hi` found by binary search on sorted `s` |
//!
//! C is only defined on the CLUSTERED layout (edges sorted by `s`). On the
//! scattered layout `K` is not an interval, and C does not exist; that is the
//! control.
//!
//! The intersection section ("Schnittpunkt") tests the cheapest restriction of
//! all: two row ranges ANDed together.
//!
//! | arm | program |
//! |---|---|
//! | S1 | `Range r1`, `Range r2`, `And`, `Count`: what a two-range conjunction lowers to today |
//! | S2 | one `Range [max(lo), min(hi))`, `Count`: the intersection computed before execution |
//! | S3 | `Pred s < t` scanned on a sorted lane, `Count` |
//! | S4 | `hi = partition_point(s < t)`, one `Range [0, hi)`, `Count`: the crossing point found by binary search |
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-quack --example teleport_probe
//! ```
//!
//! Timings are printed, never asserted. Every count is asserted. No cache-miss
//! counter is reported: `perf` is not available in the environment this probe
//! was written in, and a number this probe cannot measure is not printed.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, execute_into, Scratch};
use lance_graph_mask_risc::{
    words_for, ExecError, Foreign, ForeignPlane, LaneRef, Lowering, MaskOp, Operand, Out, Planes,
    Pred, Program, Terminal, Value,
};

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

// ───────────────────────── data ─────────────────────────

/// SplitMix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const S_SCALE: i32 = 1_000_000;
const NODES: usize = 65_536;

struct Edges {
    s: Vec<i32>,
    o: Vec<u32>,
}

/// `n` edges. `clustered` sorts by `s`, so `K = s < t` is one leading run.
/// The shuffle for the scattered layout keeps the SAME multiset of edges.
fn edges(n: usize, clustered: bool) -> Edges {
    let mut r = Rng(0x7E1E_9047);
    let mut pairs: Vec<(i32, u32)> = (0..n)
        .map(|_| {
            (
                (r.next() % S_SCALE as u64) as i32,
                (r.next() % NODES as u64) as u32,
            )
        })
        .collect();
    if clustered {
        pairs.sort_unstable();
    }
    Edges {
        s: pairs.iter().map(|p| p.0).collect(),
        o: pairs.iter().map(|p| p.1).collect(),
    }
}

/// `B`: a resident bitmap over the node table, `sel_b` of the nodes set.
fn node_plane(sel_b: f64) -> Vec<u64> {
    let mut r = Rng(0xB0B0);
    let mut w = vec![0u64; words_for(NODES)];
    for i in 0..NODES {
        if (r.next() % 1_000_000) < (sel_b * 1_000_000.0) as u64 {
            w[i / 64] |= 1 << (i % 64);
        }
    }
    w
}

fn bit(w: &[u64], i: usize) -> bool {
    w[i / 64] >> (i % 64) & 1 == 1
}

fn thresh(sel: f64) -> i32 {
    (sel * f64::from(S_SCALE)) as i32
}

/// The oracle: per row, no mask, no executor.
fn oracle(e: &Edges, t: i32, b: &[u64]) -> usize {
    e.s.iter()
        .zip(&e.o)
        .filter(|(s, o)| **s < t && bit(b, **o as usize))
        .count()
}

// ───────────────────────── work counters ─────────────────────────

/// What an arm touched, derived from the masks and the program shape, never
/// from a timer. `words` = mask words of this table the program visits;
/// `fk_reads` = `o` values read; `gathers` = foreign-plane bits read;
/// `scratch_w` = scratch words written.
#[derive(Clone, Copy)]
struct Work {
    words: usize,
    lane_reads: usize,
    fk_reads: usize,
    scratch_w: usize,
}

fn count(v: Result<Value, ExecError>) -> usize {
    match v {
        Ok(Value::Count(c)) => c,
        other => panic!("expected a Count, got {other:?}"),
    }
}

fn lowering_name(p: &Program) -> &'static str {
    match p.compile().lowering() {
        Lowering::Range(_) => "Range(fused)",
        Lowering::Ternlog(_) => "Ternlog",
        Lowering::TernlogKeep(_) => "TernlogKeep",
        Lowering::Tern2(_) => "Tern2",
        Lowering::Tern3(_) => "Tern3",
        Lowering::Tiled => "Tiled",
    }
}

/// Run `p` over `extent` `reps` times; return the count, ns/exec and
/// allocations per exec. The scratch is built before the timer.
fn time(
    p: &Program,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    extent: std::ops::Range<usize>,
    reps: usize,
) -> (usize, f64, f64) {
    let mut scratch = Scratch::for_program(p, planes.n_rows).expect("scratch");
    let full = extent == (0..planes.n_rows);
    let run = |s: &mut Scratch<'_>| {
        if full {
            count(execute_into(p, planes, foreign, s, Out::None))
        } else {
            count(execute_extent(
                p,
                planes,
                foreign,
                s,
                Out::None,
                extent.clone(),
            ))
        }
    };
    let first = run(&mut scratch);
    let a0 = ALLOCS.load(Ordering::Relaxed);
    let mut best = f64::MAX;
    // Median-of-runs would need a buffer per arm; best-of-batches is enough
    // for the question asked and allocates nothing inside the timer.
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            assert_eq!(run(&mut scratch), first);
        }
        best = best.min(t.elapsed().as_nanos() as f64 / reps as f64);
    }
    let allocs = (ALLOCS.load(Ordering::Relaxed) - a0) as f64 / (5 * reps) as f64;
    (first, best, allocs)
}

fn row(name: &str, p: &Program, w: Work, ns: f64, allocs: f64) {
    println!(
        "    {:<3} {:<12} words {:>7}  lane-reads {:>8}  fk-reads {:>8}  scratch-w {:>7}  allocs {:.1}  {:>10.0} ns",
        name,
        lowering_name(p),
        w.words,
        w.lane_reads,
        w.fk_reads,
        w.scratch_w,
        allocs,
        ns
    );
}

// ───────────────────────── arms ─────────────────────────

const S: u16 = 0; // `s` lane
const O: u16 = 1; // `o` lane

fn k(t: i32, dst: u16) -> MaskOp {
    MaskOp::Pred {
        pred: Pred::LtI32 { lane: S, t },
        under: None,
        dst,
    }
}

fn gather(under: Option<Operand>, dst: u16) -> MaskOp {
    MaskOp::Gather {
        lane: O,
        foreign: 0,
        under,
        dst,
    }
}

fn count_of(slot: u16) -> Terminal {
    Terminal::Count {
        mask: Operand::Scratch(slot),
    }
}

fn arm_a(t: i32) -> Program {
    Program::new(
        vec![
            k(t, 0),
            gather(None, 1),
            MaskOp::And {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        count_of(2),
    )
}

fn arm_b(t: i32) -> Program {
    Program::new(
        vec![k(t, 0), gather(Some(Operand::Scratch(0)), 1)],
        count_of(1),
    )
}

fn arm_c() -> Program {
    Program::new(vec![gather(None, 0)], count_of(0))
}

// ───────────────────────── sections ─────────────────────────

fn join_sweep(n: usize, reps: usize) {
    println!("\n== K(s,o) AND B(o): COUNT, n = {n} edges, {NODES} nodes ==");
    let words = words_for(n);
    for clustered in [true, false] {
        let e = edges(n, clustered);
        let lanes = [LaneRef::I32(&e.s), LaneRef::U32(&e.o)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        println!(
            "\n  layout: {}",
            if clustered {
                "CLUSTERED (sorted by s)"
            } else {
                "SCATTERED (same edges, shuffled)"
            }
        );
        for sel_b in [0.5, 0.01] {
            let b = node_plane(sel_b);
            let fp = [ForeignPlane {
                words: &b,
                rows: NODES,
            }];
            let foreign = Foreign {
                planes: &fp,
                lanes: &[],
            };
            for sel_k in [0.5, 0.1, 0.01, 0.001, 0.0001] {
                let t = thresh(sel_k);
                let want = oracle(&e, t, &b);
                let k_rows = e.s.iter().filter(|s| **s < t).count();
                println!("  sel_K {sel_k:<7} sel_B {sel_b:<5} |K| {k_rows:>8}  |R| {want:>8}");

                let pa = arm_a(t);
                let (ca, ns, al) = time(&pa, &planes, &foreign, 0..n, reps);
                assert_eq!(ca, want, "arm A");
                row(
                    "A",
                    &pa,
                    Work {
                        words,
                        lane_reads: n,
                        fk_reads: n,
                        scratch_w: 3 * words,
                    },
                    ns,
                    al,
                );

                let pb = arm_b(t);
                let (cb, ns, al) = time(&pb, &planes, &foreign, 0..n, reps);
                assert_eq!(cb, want, "arm B");
                row(
                    "B",
                    &pb,
                    Work {
                        words,
                        lane_reads: n,
                        fk_reads: k_rows,
                        scratch_w: 2 * words,
                    },
                    ns,
                    al,
                );

                if clustered {
                    // The crossing point of the sorted lane with `t`: every
                    // row before it is in K, none after. `partition_point` is
                    // O(log n) and reads ~log2(n) values of `s`.
                    let hi = e.s.partition_point(|s| *s < t);
                    assert_eq!(hi, k_rows);
                    let pc = arm_c();
                    let (cc, ns, al) = time(&pc, &planes, &foreign, 0..hi, reps);
                    assert_eq!(cc, want, "arm C");
                    let ew = words_for(hi);
                    row(
                        "C",
                        &pc,
                        Work {
                            words: ew,
                            lane_reads: (n as f64).log2().ceil() as usize,
                            fk_reads: hi,
                            scratch_w: ew,
                        },
                        ns,
                        al,
                    );
                }
            }
        }
    }
}

fn range_op(lo: u32, hi: u32, dst: u16) -> MaskOp {
    MaskOp::Pred {
        pred: Pred::Range { lo, hi },
        under: None,
        dst,
    }
}

fn schnittpunkt(n: usize, reps: usize) {
    println!("\n== Schnittpunkt: the intersection of two restrictions, n = {n} ==");
    let words = words_for(n);
    let e = edges(n, true);
    let lanes = [LaneRef::I32(&e.s), LaneRef::U32(&e.o)];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let nn = n as u32;

    // Two row ranges, intersection width swept from wide to narrow.
    println!("  -- two row ranges ANDed --");
    for frac in [0.5, 0.1, 0.01, 0.0001] {
        let width = ((f64::from(nn) * frac) as u32).max(1);
        let (l1, h1) = (nn / 4, nn / 4 + width + width / 2);
        let (l2, h2) = (nn / 4 + width / 2, nn / 4 + width / 2 + width);
        let (lo, hi) = (l1.max(l2), h1.min(h2));
        let want = (hi - lo) as usize;

        let s1 = Program::new(
            vec![
                range_op(l1, h1, 0),
                range_op(l2, h2, 1),
                MaskOp::And {
                    a: Operand::Scratch(0),
                    b: Operand::Scratch(1),
                    dst: 2,
                },
            ],
            count_of(2),
        );
        let (c, ns, al) = time(&s1, &planes, &Foreign::NONE, 0..n, reps);
        assert_eq!(c, want, "S1");
        println!("  intersection {:>8} rows", want);
        row(
            "S1",
            &s1,
            Work {
                words,
                lane_reads: 0,
                fk_reads: 0,
                scratch_w: 3 * words,
            },
            ns,
            al,
        );

        let s2 = Program::new(vec![range_op(lo, hi, 0)], count_of(0));
        let (c, ns, al) = time(&s2, &planes, &Foreign::NONE, 0..n, reps);
        assert_eq!(c, want, "S2");
        row(
            "S2",
            &s2,
            Work {
                // The fused range count reads no lane and writes no slot; it
                // counts the interval from its two endpoints.
                words: 2,
                lane_reads: 0,
                fk_reads: 0,
                scratch_w: 0,
            },
            ns,
            al,
        );
    }

    println!("  -- a monotone predicate on a sorted lane: scan vs crossing point --");
    for sel in [0.5, 0.01, 0.0001] {
        let t = thresh(sel);
        let want = e.s.iter().filter(|s| **s < t).count();

        let s3 = Program::new(vec![k(t, 0)], count_of(0));
        let (c, ns, al) = time(&s3, &planes, &Foreign::NONE, 0..n, reps);
        assert_eq!(c, want, "S3");
        println!("  sel {sel:<7} |K| {want:>8}");
        row(
            "S3",
            &s3,
            Work {
                words,
                lane_reads: n,
                fk_reads: 0,
                scratch_w: words,
            },
            ns,
            al,
        );

        // The binary search is part of the arm: it is timed with it.
        let tm = Instant::now();
        let mut hi = 0;
        for _ in 0..reps {
            hi = std::hint::black_box(&e.s).partition_point(|s| *s < t);
        }
        let search_ns = tm.elapsed().as_nanos() as f64 / reps as f64;
        let s4 = Program::new(vec![range_op(0, hi as u32, 0)], count_of(0));
        let (c, ns, al) = time(&s4, &planes, &Foreign::NONE, 0..n, reps);
        assert_eq!(c, want, "S4");
        row(
            "S4",
            &s4,
            Work {
                words: 2,
                lane_reads: (n as f64).log2().ceil() as usize,
                fk_reads: 0,
                scratch_w: 0,
            },
            ns + search_ns,
            al,
        );
    }
}

/// A fact change flips one bit of `B` and re-runs the SAME program; a rule
/// change builds and compiles a NEW program. Both are timed per change.
fn change_cost(n: usize, reps: usize) {
    println!("\n== fact change vs rule change, n = {n}, arm B shape ==");
    let e = edges(n, false);
    let lanes = [LaneRef::I32(&e.s), LaneRef::U32(&e.o)];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let mut b = node_plane(0.5);
    let t = thresh(0.01);
    let p = arm_b(t);
    let mut scratch = Scratch::for_program(&p, n).expect("scratch");

    let tm = Instant::now();
    let mut got = 0;
    for i in 0..reps {
        let w = (i * 7) % b.len();
        b[w] ^= 1;
        let fp = [ForeignPlane {
            words: &b,
            rows: NODES,
        }];
        let foreign = Foreign {
            planes: &fp,
            lanes: &[],
        };
        got = count(execute_into(&p, &planes, &foreign, &mut scratch, Out::None));
    }
    let fact_ns = tm.elapsed().as_nanos() as f64 / reps as f64;
    // The oracle runs once, outside the timer, on the final state of `B`.
    assert_eq!(got, oracle(&e, t, &b), "fact change");

    let fp = [ForeignPlane {
        words: &b,
        rows: NODES,
    }];
    let foreign = Foreign {
        planes: &fp,
        lanes: &[],
    };
    let tm = Instant::now();
    for i in 0..reps {
        let q = arm_b(t + i as i32);
        let _ = std::hint::black_box(q.compile());
        let mut s = Scratch::for_program(&q, n).expect("scratch");
        let _ = count(execute_into(&q, &planes, &foreign, &mut s, Out::None));
    }
    let rule_ns = tm.elapsed().as_nanos() as f64 / reps as f64;
    println!("    fact change (flip a B bit, re-run):        {fact_ns:>10.0} ns");
    println!("    rule change (new program, compile, scratch, run): {rule_ns:>10.0} ns");
}

fn main() {
    let n: usize = std::env::var("TELEPORT_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20);
    let reps: usize = std::env::var("TELEPORT_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    join_sweep(n, reps);
    schnittpunkt(n, reps);
    change_cost(n, reps);
}
