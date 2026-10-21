//! D-RU-2 benchmark: a batch of queries sharing one conjunction, run as a
//! [`SetProgram`] (each let once into a scratch plane) against recompute
//! (every query inlined). Reports time, lane reads, ops and materialized
//! bytes. `BENCH_N` rows (default 1M), `BENCH_REPS` repetitions (default 9),
//! `BENCH_Q` queries (default 16).

use std::time::Instant;

use lance_graph_contract::set_algebra::{Cmp, Res, SetExpr};
use lance_graph_contract::set_program::share;
use lance_graph_mask_risc::{words_for, LaneRef, Planes};
use lance_graph_relalg::{run_program, run_recompute, Measure};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let n = env("BENCH_N", 1_000_000);
    let reps = env("BENCH_REPS", 9);
    let nq = env("BENCH_Q", 16);

    let x: Vec<i32> = (0..n).map(|i| (i % 50) as i32 - 20).collect();
    let y: Vec<i32> = (0..n).map(|i| ((i * 7) % 31) as i32).collect();
    let z: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32).collect();
    let v: Vec<i32> = (0..n).map(|i| (i % 13) as i32 - 6).collect();
    let k: Vec<u32> = (0..n).map(|i| (i % 5) as u32).collect();
    let mut p0: Vec<u64> = (0..words_for(n))
        .map(|w| 0x9249_2492_4924_9249u64.rotate_left(w as u32))
        .collect();
    // mask-risc refuses set bits past the last row.
    if !n.is_multiple_of(64) {
        *p0.last_mut().expect("n > 0") &= (1u64 << (n % 64)) - 1;
    }
    let lanes = [
        LaneRef::I32(&x),
        LaneRef::I32(&y),
        LaneRef::I32(&z),
        LaneRef::I32(&v),
        LaneRef::U32(&k),
    ];
    let masks: [&[u64]; 1] = [&p0];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };

    let core = vec![
        SetExpr::i32(0, Cmp::Ge, -5),
        SetExpr::i32(1, Cmp::Lt, 20),
        SetExpr::Res(Res::U32 {
            lane: 4,
            eq: true,
            v: 2,
        }),
    ];
    let queries: Vec<SetExpr> = (0..nq)
        .map(|i| {
            let mut cs = core.clone();
            cs.push(match i % 3 {
                0 => SetExpr::i32(2, Cmp::Lt, i as i32 * 5),
                1 => SetExpr::Sym(0),
                _ => SetExpr::i32(2, Cmp::Gt, i as i32 * 3),
            });
            SetExpr::And(cs)
        })
        .collect();
    let p = share(1, 1, &queries);
    let ms: Vec<Measure> = (0..nq)
        .map(|i| {
            if i % 2 == 0 {
                Measure::Count
            } else {
                Measure::Sum(3)
            }
        })
        .collect();

    let mut none: Vec<&mut [u64]> = Vec::new();
    let mut t_prog = Vec::new();
    let mut t_rec = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        let a = run_program(&p, 1, &planes, &ms, &mut none).expect("program");
        t_prog.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let b = run_recompute(&p, 1, &planes, &ms, &mut none).expect("recompute");
        t_rec.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(a.answers, b.answers, "answers diverged");
        last = Some((a.cost, b.cost));
    }
    let (pc, rc) = last.expect("reps >= 1");
    println!(
        "n_rows={n} queries={nq} lets={} reps={reps} avx512f={} avx2={}",
        p.lets.len(),
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!(
        "program   : {:>8.2} ms  lane_reads={} passes={} programs={} planes={} bytes={} slots={}",
        median(t_prog.clone()),
        pc.lane_reads,
        pc.passes,
        pc.programs,
        pc.planes_materialized,
        pc.plane_bytes,
        pc.max_scratch_slots
    );
    println!(
        "recompute : {:>8.2} ms  lane_reads={} passes={} programs={} planes={} bytes={} slots={}",
        median(t_rec.clone()),
        rc.lane_reads,
        rc.passes,
        rc.programs,
        rc.planes_materialized,
        rc.plane_bytes,
        rc.max_scratch_slots
    );
    println!(
        "ratio     : time {:.2}x  lane_reads {:.2}x",
        median(t_rec) / median(t_prog),
        rc.lane_reads as f64 / pc.lane_reads as f64
    );
}
