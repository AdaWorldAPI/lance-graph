//! EXO-290 Ask 2, step 1: where does running tiles on several threads beat
//! the single-threaded tile loop? Measured before any driver is built.
//!
//! The split is the one `execute_extent` already licenses: the row range is
//! cut into chunks of whole tiles (`TILE_WORDS * 64` rows, so no two chunks
//! share a word), each worker runs `execute_compiled` over its chunk with
//! its own scratch and, for a group terminal, its own `Out::I64`, all
//! allocated before the timed loop. The parts merge by the terminal's law:
//! count and sum add, min takes the minimum, a group sink merges with
//! `Terminal::merge_group_sink`. Every parallel answer is compared with the
//! single-threaded one, so a row is only reported if it is bit-for-bit equal.
//!
//! Two ways to get the workers, both std-only (no new dependency):
//!
//! - `par`: spawn per call with `std::thread::scope`. `spawn_us` times an
//!   empty scope with the same number of spawns, the floor any chunk must
//!   pay back.
//! - `pool`: workers spawned once and parked on a `std::sync::Barrier`; one
//!   call is two barrier waits (start, done). `wake_us` times the two waits
//!   with no work.
//!
//! Sizes stop at 2^19 rows: past that the single-threaded group sum crosses
//! 1 ms, which this repository treats as a failed measurement, not a row.
//!
//! `cargo run --release -p lance-graph-mask-risc --example par_tile_probe`

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, Scratch, TILE_WORDS};
use lance_graph_mask_risc::{
    Compiled, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Terminal, Value,
};

const TILE_ROWS: usize = TILE_WORDS * 64;
const GROUPS: usize = 64;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn median(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    ts.sort_by(|a, b| a.total_cmp(b));
    ts[reps / 2]
}

/// `a < 500 AND NOT (b < 200)`, then the terminal. Lane predicates, so the
/// tiled path runs (no fused lowering).
fn program(terminal: Terminal) -> Program {
    Program {
        ops: vec![
            MaskOp::Pred {
                pred: Pred::LtI32 { lane: 0, t: 500 },
                under: None,
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LtI32 { lane: 1, t: 200 },
                under: None,
                dst: 1,
            },
            MaskOp::AndNot {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        terminal,
        scratch_slots: 3,
    }
}

/// `[0, n)` cut into at most `t` chunks of whole tiles.
fn chunks(n: usize, t: usize) -> Vec<(usize, usize)> {
    let per = n.div_ceil(t).div_ceil(TILE_ROWS) * TILE_ROWS;
    (0..t)
        .map(|i| ((i * per).min(n), ((i + 1) * per).min(n)))
        .filter(|(lo, hi)| lo < hi)
        .collect()
}

fn merge(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Count(x), Value::Count(y)) => Value::Count(x + y),
        (Value::SumI64(x), Value::SumI64(y)) => Value::SumI64(x + y),
        (Value::OptI32(x), Value::OptI32(y)) => Value::OptI32(match (x, y) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        }),
        (Value::GroupSummed, Value::GroupSummed) => Value::GroupSummed,
        other => panic!("no merge for {other:?}"),
    }
}

struct Worker {
    scratch: Vec<u64>,
    sink: Vec<i64>,
}

/// Run `c` over `planes` on `parts.len()` workers. Worker 0 runs on the
/// calling thread. Returns the merged value; a group terminal's merged sink
/// is left in `workers[0].sink`.
fn run_parallel(
    c: &Compiled<'_>,
    planes: &Planes<'_>,
    group: bool,
    parts: &[(usize, usize)],
    workers: &mut [Worker],
) -> Value {
    let run = |w: &mut Worker, lo: usize, hi: usize| {
        let mut s = Scratch::over(&mut w.scratch, TILE_WORDS, 3).unwrap();
        let out = if group {
            Out::I64(&mut w.sink)
        } else {
            Out::None
        };
        execute_compiled(c, planes, &Foreign::NONE, &mut s, out, lo..hi).unwrap()
    };
    let (first, rest) = workers.split_first_mut().unwrap();
    let mut vals: Vec<Value> = Vec::with_capacity(parts.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = rest
            .iter_mut()
            .zip(&parts[1..])
            .map(|(w, &(lo, hi))| sc.spawn(move || run(w, lo, hi)))
            .collect();
        vals.push(run(first, parts[0].0, parts[0].1));
        vals.extend(handles.into_iter().map(|h| h.join().unwrap()));
    });
    if group {
        let (first, rest) = workers.split_first_mut().unwrap();
        for w in rest.iter().take(parts.len() - 1) {
            c.program()
                .terminal
                .merge_group_sink(&mut first.sink, &w.sink)
                .unwrap();
        }
    }
    vals.into_iter().reduce(merge).unwrap()
}

/// The pool arm: `parts.len() - 1` workers spawned once and parked on a
/// barrier, worker 0 on the calling thread. Returns the median ns of a whole
/// call and of a call with no work (two barrier waits). Checks every answer.
fn pool_arm(
    c: &Compiled<'_>,
    planes: &Planes<'_>,
    group: bool,
    parts: &[(usize, usize)],
    reps: usize,
    want: Value,
    want_sink: &[i64],
) -> (f64, f64) {
    let t = parts.len();
    let mut workers: Vec<Worker> = (0..t)
        .map(|_| Worker {
            scratch: vec![0; 4 * TILE_WORDS + 64],
            sink: vec![0; GROUPS],
        })
        .collect();
    let start = Barrier::new(t);
    let done = Barrier::new(t);
    let stop = AtomicBool::new(false);
    let work = AtomicBool::new(true);
    let results: Vec<Mutex<Option<Value>>> = (0..t).map(|_| Mutex::new(None)).collect();
    let run = |w: &mut Worker, lo: usize, hi: usize| {
        let mut s = Scratch::over(&mut w.scratch, TILE_WORDS, 3).unwrap();
        let out = if group {
            Out::I64(&mut w.sink)
        } else {
            Out::None
        };
        execute_compiled(c, planes, &Foreign::NONE, &mut s, out, lo..hi).unwrap()
    };
    let (first, rest) = workers.split_first_mut().unwrap();
    let mut timed = (0.0, 0.0);
    std::thread::scope(|sc| {
        for (i, w) in rest.iter_mut().enumerate() {
            let (lo, hi) = parts[i + 1];
            let (start, done, stop, work, slot) = (&start, &done, &stop, &work, &results[i + 1]);
            sc.spawn(move || loop {
                start.wait();
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if work.load(Ordering::Acquire) {
                    *slot.lock().unwrap() = Some(run(w, lo, hi));
                }
                done.wait();
            });
        }
        let mut call = |with_work: bool| {
            work.store(with_work, Ordering::Release);
            start.wait();
            let v0 = with_work.then(|| run(first, parts[0].0, parts[0].1));
            done.wait();
            v0
        };
        let mut check = None;
        timed.0 = median(reps, || {
            let mut v = call(true).unwrap();
            for r in &results[1..] {
                v = merge(v, r.lock().unwrap().take().unwrap());
            }
            check = Some(v);
        });
        timed.1 = median(reps, || {
            call(false);
        });
        stop.store(true, Ordering::Release);
        start.wait();
        assert_eq!(check.unwrap(), want, "pool answer");
    });
    // The workers hold the last WORK run's sinks (the no-work calls write
    // none), so the merged sink is checked here, after the scope.
    if group {
        let mut acc = workers[0].sink.clone();
        for w in &workers[1..] {
            c.program()
                .terminal
                .merge_group_sink(&mut acc, &w.sink)
                .unwrap();
        }
        assert_eq!(acc, want_sink, "pool sink");
    }
    timed
}

fn main() {
    let max_n = 1 << 19;
    let mut seed = 0x2900_u64;
    let a: Vec<i32> = (0..max_n).map(|_| (lcg(&mut seed) % 1000) as i32).collect();
    let b: Vec<i32> = (0..max_n).map(|_| (lcg(&mut seed) % 1000) as i32).collect();
    let k: Vec<u32> = (0..max_n)
        .map(|_| (lcg(&mut seed) % GROUPS as u64) as u32)
        .collect();
    let shapes: [(&str, Terminal, bool); 3] = [
        (
            "count",
            Terminal::Count {
                mask: Operand::Scratch(2),
            },
            false,
        ),
        (
            "sum",
            Terminal::MaskedSumI32 {
                mask: Operand::Scratch(2),
                lane: 1,
            },
            false,
        ),
        (
            "groupsum",
            Terminal::GroupSumI32 {
                mask: Operand::Scratch(2),
                key: 2,
                val: 1,
            },
            true,
        ),
    ];
    println!(
        "shape     n        threads  seq_us   par_us   spawn_us  pool_us  wake_us  par_x  pool_x"
    );
    for (name, term, group) in shapes {
        let p = program(term);
        let c = p.compile();
        for n in [1 << 14, 1 << 16, 1 << 18, 1 << 19] {
            let lanes = [
                LaneRef::I32(&a[..n]),
                LaneRef::I32(&b[..n]),
                LaneRef::U32(&k[..n]),
            ];
            let planes = Planes {
                n_rows: n,
                masks: &[],
                lanes: &lanes,
            };
            let reps = if n <= 1 << 16 { 201 } else { 41 };
            let mut seq = Worker {
                scratch: vec![0; 4 * TILE_WORDS + 64],
                sink: vec![0; GROUPS],
            };
            let mut seq_val = None;
            let seq_ns = median(reps, || {
                seq_val = Some(run_parallel(
                    &c,
                    &planes,
                    group,
                    &[(0, n)],
                    std::slice::from_mut(&mut seq),
                ));
            });
            let (want, want_sink) = (seq_val.unwrap(), seq.sink.clone());
            for t in [2usize, 4] {
                let parts = chunks(n, t);
                let mut workers: Vec<Worker> = (0..t)
                    .map(|_| Worker {
                        scratch: vec![0; 4 * TILE_WORDS + 64],
                        sink: vec![0; GROUPS],
                    })
                    .collect();
                let mut got = None;
                let par_ns = median(reps, || {
                    got = Some(run_parallel(&c, &planes, group, &parts, &mut workers));
                });
                assert_eq!(got.unwrap(), want, "{name} n={n} t={t}");
                if group {
                    assert_eq!(workers[0].sink, want_sink, "{name} sink n={n} t={t}");
                }
                let spawn_ns = median(reps, || {
                    std::thread::scope(|sc| {
                        for _ in 1..parts.len() {
                            sc.spawn(|| std::hint::black_box(0));
                        }
                    })
                });
                let (pool_ns, wake_ns) =
                    pool_arm(&c, &planes, group, &parts, reps, want, &want_sink);
                println!(
                    "{name:<9} {n:<8} {t:<8} {:<8.1} {:<8.1} {:<9.1} {:<8.1} {:<8.1} {:<6.2} {:.2}",
                    seq_ns / 1e3,
                    par_ns / 1e3,
                    spawn_ns / 1e3,
                    pool_ns / 1e3,
                    wake_ns / 1e3,
                    seq_ns / par_ns,
                    seq_ns / pool_ns
                );
            }
        }
    }
}
