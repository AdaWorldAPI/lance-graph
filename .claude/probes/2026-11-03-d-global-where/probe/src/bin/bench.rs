//! LAB: D-GLOBAL-WHERE P0 bench (Part 2 of SPEC.md): our side.
//!
//! Usage: `bench <OUT>`. Writes the datasets as CSV into `<OUT>` and our
//! timings to `<OUT>/ours.json`. Single thread. Every timing is in
//! microseconds; a warm or query value >= 1000 us prints FAILED-GATE; cold
//! preparation is reported and flagged COLD-FLAG when >= 1000 us (reported,
//! not a gate). Exit status 1 if any answer disagrees with the scalar oracle.
//!
//! Dense edges: Bernoulli p = 0.25 for dr64 and dr256, p = 0.03 for multi.

use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use d_global_where_probe::{
    alloc_bytes, bit, coord_of, for_each_bit, gen_edges, gen_lanes, live_bytes, oracle,
    peak_bytes, pick_domain, popcount, project_keep, reset_peak, route_selection, set_bit, stats,
    time_runs, virtual_where, CountingAlloc, Dist, Lanes, Oracle, Plane, Prepared, Rng, A_LT,
    B_LT, SEED, UNIVERSE, UNIVERSE_WORDS,
};
use lance_graph_dt64::dr::{Axis as ResAxis, LocalMap, MapTable, TileId, TileMap};
use lance_graph_quack::{Agg, Cmp};
use ndarray::simd::{checked_binary_i64, mask_and, mask_any, mask_or_assign, I64Op};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const RUNS: usize = 201;

// ---------------------------------------------------------------------------
// records and json
// ---------------------------------------------------------------------------

struct Rec {
    dataset: String,
    size: usize,
    dist: &'static str,
    edges: usize,
    workload: &'static str,
    arm: &'static str,
    answer: i64,
    cold_us: f64,
    warm_med_us: f64,
    warm_p10_us: f64,
    warm_p90_us: f64,
    where_q_us: Option<f64>,
    where_v_us: Option<f64>,
    route_us: Option<f64>,
    fold_us: Option<f64>,
    update_us: Option<f64>,
    alloc_bytes_prep: usize,
    alloc_bytes_query: usize,
    peak_bytes: usize,
    interm: Option<i64>,
}

fn jf(o: Option<f64>) -> String {
    match o {
        Some(v) => format!("{v:.3}"),
        None => "null".to_string(),
    }
}

fn ji(o: Option<i64>) -> String {
    match o {
        Some(v) => v.to_string(),
        None => "null".to_string(),
    }
}

fn write_json(path: &Path, recs: &[Rec]) {
    let mut w = BufWriter::new(File::create(path).expect("ours.json"));
    writeln!(w, "[").unwrap();
    for (i, r) in recs.iter().enumerate() {
        write!(
            w,
            "  {{\"dataset\": \"{}\", \"size\": {}, \"dist\": \"{}\", \"edges\": {}, \"workload\": \"{}\", \"arm\": \"{}\", \"answer\": {}, \"cold_us\": {:.3}, \"warm_med_us\": {:.3}, \"warm_p10_us\": {:.3}, \"warm_p90_us\": {:.3}, \"where_q_us\": {}, \"where_v_us\": {}, \"route_us\": {}, \"fold_us\": {}, \"update_us\": {}, \"alloc_bytes_prep\": {}, \"alloc_bytes_query\": {}, \"peak_bytes\": {}, \"interm_cardinality\": {}}}",
            r.dataset,
            r.size,
            r.dist,
            r.edges,
            r.workload,
            r.arm,
            r.answer,
            r.cold_us,
            r.warm_med_us,
            r.warm_p10_us,
            r.warm_p90_us,
            jf(r.where_q_us),
            jf(r.where_v_us),
            jf(r.route_us),
            jf(r.fold_us),
            jf(r.update_us),
            r.alloc_bytes_prep,
            r.alloc_bytes_query,
            r.peak_bytes,
            ji(r.interm),
        )
        .unwrap();
        writeln!(w, "{}", if i + 1 < recs.len() { "," } else { "" }).unwrap();
    }
    writeln!(w, "]").unwrap();
}

// ---------------------------------------------------------------------------
// csv
// ---------------------------------------------------------------------------

fn write_pairs(path: &Path, header: &str, rows: &[(u32, u32)]) {
    let mut w = BufWriter::new(File::create(path).expect("csv"));
    writeln!(w, "{header}").unwrap();
    for &(a, b) in rows {
        writeln!(w, "{a},{b}").unwrap();
    }
}

fn write_ids(path: &Path, ids: &[u64]) {
    let mut w = BufWriter::new(File::create(path).expect("csv"));
    writeln!(w, "id").unwrap();
    for &g in ids {
        writeln!(w, "{g}").unwrap();
    }
}

fn write_nodes(path: &Path, lanes: &Lanes) {
    let mut w = BufWriter::new(File::create(path).expect("nodes.csv"));
    writeln!(w, "id,a,b").unwrap();
    for g in 0..UNIVERSE {
        writeln!(w, "{g},{},{}", lanes.a[g], lanes.b[g]).unwrap();
    }
}

// ---------------------------------------------------------------------------
// folds over the presence plane (coordinates, masks; no tuples)
// ---------------------------------------------------------------------------

/// W2: any(row[x] & selB) for every x in selA.
fn fold_w2(plane: &Plane, a: &[u64], b: &[u64], tmp: &mut [u64]) -> i64 {
    let mut n = 0i64;
    for_each_bit(a, |x| {
        mask_and(plane.row(x), b, tmp);
        if mask_any(tmp) {
            n += 1;
        }
    });
    n
}

/// W3 `back`: Yb = {y : row[y] & selB != 0}, then count x in A with row[x] & Yb != 0.
fn fold_w3_back(plane: &Plane, a: &[u64], b: &[u64], yb: &mut [u64], tmp: &mut [u64]) -> i64 {
    yb.fill(0);
    for y in 0..plane.coords {
        mask_and(plane.row(y), b, tmp);
        if mask_any(tmp) {
            set_bit(yb, y);
        }
    }
    let mut n = 0i64;
    for_each_bit(a, |x| {
        mask_and(plane.row(x), yb, tmp);
        if mask_any(tmp) {
            n += 1;
        }
    });
    n
}

/// W3 `tile`: per x in A, R2 = OR of the rows of row[x]'s members, test & selB.
fn fold_w3_tile(plane: &Plane, a: &[u64], b: &[u64], acc: &mut [u64], tmp: &mut [u64]) -> i64 {
    let mut n = 0i64;
    for_each_bit(a, |x| {
        acc.fill(0);
        for_each_bit(plane.row(x), |y| mask_or_assign(acc, plane.row(y)));
        mask_and(acc, b, tmp);
        if mask_any(tmp) {
            n += 1;
        }
    });
    n
}

fn adj<'a>(off: &'a [u32], adjv: &'a [u32], x: usize) -> &'a [u32] {
    &adjv[off[x] as usize..off[x + 1] as usize]
}

/// W3 `csr`: DFS over the CSR with early exit.
fn fold_w3_csr(off: &[u32], adjv: &[u32], a: &[u64], b: &[u64]) -> i64 {
    let mut n = 0i64;
    for_each_bit(a, |x| {
        'found: for &y in adj(off, adjv, x) {
            for &z in adj(off, adjv, y as usize) {
                if bit(b, z as usize) {
                    n += 1;
                    break 'found;
                }
            }
        }
    });
    n
}

/// W4 `front`: F = selA; three times F = OR rows of F; popcount(F & selB).
fn fold_w4_front(plane: &Plane, a: &[u64], b: &[u64], f: &mut [u64], g: &mut [u64]) -> i64 {
    f.copy_from_slice(a);
    for _ in 0..3 {
        g.fill(0);
        for_each_bit(f, |x| mask_or_assign(g, plane.row(x)));
        f.copy_from_slice(g);
    }
    mask_and(f, b, g);
    popcount(g) as i64
}

/// W4 `csr`: BFS over the CSR with a visited bitmap per level.
#[allow(clippy::too_many_arguments)]
fn fold_w4_csr(
    off: &[u32],
    adjv: &[u32],
    a: &[u64],
    b: &[u64],
    cur: &mut Vec<u32>,
    next: &mut Vec<u32>,
    visited: &mut [u64],
) -> i64 {
    cur.clear();
    for_each_bit(a, |x| cur.push(x as u32));
    for _ in 0..3 {
        visited.fill(0);
        next.clear();
        for &x in cur.iter() {
            for &y in adj(off, adjv, x as usize) {
                if !bit(visited, y as usize) {
                    set_bit(visited, y as usize);
                    next.push(y);
                }
            }
        }
        std::mem::swap(cur, next);
    }
    cur.iter().filter(|&&z| bit(b, z as usize)).count() as i64
}

/// W5: support = popcount(OR rows of A).
fn fold_w5(plane: &Plane, a: &[u64], acc: &mut [u64]) -> i64 {
    acc.fill(0);
    for_each_bit(a, |x| mask_or_assign(acc, plane.row(x)));
    popcount(acc) as i64
}

/// W6: indegA[y] by per-row bit iteration over x in A; sum of indegA * outdeg, checked i64.
fn fold_w6(plane: &Plane, a: &[u64], outdeg: &[i64], indeg: &mut [i64], prod: &mut [i64]) -> i64 {
    indeg.fill(0);
    for_each_bit(a, |x| {
        for_each_bit(plane.row(x), |y| indeg[y] += 1);
    });
    checked_binary_i64(I64Op::Mul, indeg, outdeg, prod).expect("2-path counts fit i64");
    let mut s = 0i64;
    for &p in prod.iter() {
        s = s.checked_add(p).expect("2-path sum fits i64");
    }
    s
}

/// W7i: popcount of the routed intersection.
fn fold_w7i(a: &[u64], b: &[u64], tmp: &mut [u64]) -> i64 {
    mask_and(a, b, tmp);
    popcount(tmp) as i64
}

// ---------------------------------------------------------------------------
// one arm
// ---------------------------------------------------------------------------

struct Meta {
    dataset: String,
    size: usize,
    dist: &'static str,
    edges: usize,
}

/// Shared preparation cost charged to an arm's cold figure.
#[derive(Clone, Copy, Default)]
struct Cost {
    prep_us: f64,
    alloc_prep: usize,
    peak_prep: usize,
}

fn flag(v: f64) -> &'static str {
    if v >= 1000.0 {
        " FAILED-GATE"
    } else {
        ""
    }
}

/// Run one arm. `f(true)` runs the whole query (WHERE or routing, then the
/// fold); `f(false)` runs the fold alone over the masks the last `f(true)`
/// left in place. Cold = preparation + the first `f(true)`. Warm = 201 runs
/// of `f(warm_flag)`. The fold-only column = 201 runs of `f(false)`.
#[allow(clippy::too_many_arguments)]
fn run_arm(
    m: &Meta,
    cost: Cost,
    workload: &'static str,
    arm: &'static str,
    warm_flag: bool,
    expected: i64,
    cols: (Option<f64>, Option<f64>, Option<f64>),
    f: &mut dyn FnMut(bool) -> i64,
) -> (Rec, bool) {
    reset_peak();
    let live0 = live_bytes();
    let t = Instant::now();
    let answer = black_box(f(true));
    let cold_run_us = t.elapsed().as_nanos() as f64 / 1000.0;
    let run_peak = peak_bytes().saturating_sub(live0);

    let ab = alloc_bytes();
    black_box(f(warm_flag));
    let alloc_query = alloc_bytes() - ab;

    let warm_samples = time_runs(RUNS, || {
        black_box(f(warm_flag));
    });
    let (med, p10, p90) = stats(&warm_samples);
    let fold_samples = time_runs(RUNS, || {
        black_box(f(false));
    });
    let (fold_med, _, _) = stats(&fold_samples);

    let cold_us = cost.prep_us + cold_run_us;
    let ok = answer == expected;
    let mut notes = String::new();
    for v in [med, p10, p90, fold_med, cols.0.unwrap_or(0.0), cols.1.unwrap_or(0.0), cols.2.unwrap_or(0.0)] {
        if v >= 1000.0 {
            notes.push_str(flag(v));
            break;
        }
    }
    if cost.prep_us >= 1000.0 {
        notes.push_str(" COLD-FLAG");
    }
    if !ok {
        notes.push_str(&format!(" MISMATCH(expected {expected})"));
    }
    println!(
        "{:<13} {:<4} {:<10} ans={:<9} warm {:>9.3} us (p10 {:>9.3}, p90 {:>9.3}) fold {:>9.3} us cold {:>11.3} us alloc_query {}{}",
        m.dataset, workload, arm, answer, med, p10, p90, fold_med, cold_us, alloc_query, notes
    );
    (
        Rec {
            dataset: m.dataset.clone(),
            size: m.size,
            dist: m.dist,
            edges: m.edges,
            workload,
            arm,
            answer,
            cold_us,
            warm_med_us: med,
            warm_p10_us: p10,
            warm_p90_us: p90,
            where_q_us: cols.0,
            where_v_us: cols.1,
            route_us: cols.2,
            fold_us: Some(fold_med),
            update_us: None,
            alloc_bytes_prep: cost.alloc_prep,
            alloc_bytes_query: alloc_query,
            peak_bytes: cost.peak_prep.max(run_peak),
            interm: None,
        },
        ok,
    )
}

// ---------------------------------------------------------------------------
// one dataset
// ---------------------------------------------------------------------------

struct Data<'a> {
    meta: Meta,
    lanes: &'a Lanes,
    dom: &'a [u64],
    edges: &'a [(u32, u32)],
    deleted: &'a [(u32, u32)],
    oracle: &'a Oracle,
}

fn push(recs: &mut Vec<Rec>, all_ok: &mut bool, r: (Rec, bool), interm: Option<i64>) {
    let (mut rec, ok) = r;
    rec.interm = interm;
    *all_ok &= ok;
    recs.push(rec);
}

fn median(v: &[f64]) -> f64 {
    stats(v).0
}

fn run_dataset(recs: &mut Vec<Rec>, d: &Data<'_>) -> bool {
    let mut all_ok = true;
    let lanes = d.lanes;
    let size = d.dom.len();
    let live = vec![u64::MAX; UNIVERSE_WORDS];
    let mut keep_a = vec![0u64; UNIVERSE_WORDS];
    let mut keep_b = vec![0u64; UNIVERSE_WORDS];

    // ---- W1 preparation (lower once, scratch carved) -------------------
    let ab = alloc_bytes();
    let t = Instant::now();
    let mut q1 = Prepared::new(Cmp::LtI32(A_LT), Agg::Count, UNIVERSE);
    let w1_prep = Cost {
        prep_us: t.elapsed().as_nanos() as f64 / 1000.0,
        alloc_prep: alloc_bytes() - ab,
        peak_prep: 0,
    };
    let mut qa = Prepared::new(Cmp::LtI32(A_LT), Agg::Rows, UNIVERSE);
    let mut qb = Prepared::new(Cmp::LtI32(B_LT), Agg::Rows, UNIVERSE);

    // ---- cold preparation: tiles, presence plane, outdeg ---------------
    let live_before = live_bytes();
    reset_peak();
    let ab_base = alloc_bytes();
    let t_base = Instant::now();
    let axis = ResAxis::for_survivors(size);
    let tw = axis.tile.positions();
    let ntiles = axis.tiles as usize;
    let coords = ntiles * tw;
    let wpr = coords.div_ceil(64);
    let r_tiles: Vec<TileId> = (0..ntiles as u32).collect();
    let mut table = MapTable::new(1, 1);
    {
        let mut changes: Vec<(TileId, Option<LocalMap>)> = Vec::new();
        for (t, chunk) in d.dom.chunks(tw).enumerate() {
            changes.push((
                t as TileId,
                Some(LocalMap::prepare(axis.tile, chunk).expect("R tile")),
            ));
        }
        table.advance(1, 1, changes);
    }
    let r_us = t_base.elapsed().as_nanos() as f64 / 1000.0;

    // ---- selection tiles (used by W7 only): A descending, B ascending ---
    let ab_sel = alloc_bytes();
    let t_sel = Instant::now();
    let mut a_ids: Vec<u64> = d
        .dom
        .iter()
        .copied()
        .filter(|&g| lanes.a[g as usize] < A_LT)
        .collect();
    a_ids.sort_unstable_by(|p, q| q.cmp(p));
    let mut b_ids: Vec<u64> = d
        .dom
        .iter()
        .copied()
        .filter(|&g| lanes.b[g as usize] < B_LT)
        .collect();
    b_ids.sort_unstable();
    let mut a_tile_ids: Vec<TileId> = Vec::new();
    let mut b_tile_ids: Vec<TileId> = Vec::new();
    {
        let a_axis = ResAxis::for_survivors(a_ids.len());
        let b_axis = ResAxis::for_survivors(b_ids.len());
        let mut changes: Vec<(TileId, Option<LocalMap>)> = Vec::new();
        for (k, chunk) in a_ids.chunks(a_axis.tile.positions()).enumerate() {
            let id = 100 + k as TileId;
            a_tile_ids.push(id);
            changes.push((id, Some(LocalMap::prepare(a_axis.tile, chunk).expect("A tile"))));
        }
        for (k, chunk) in b_ids.chunks(b_axis.tile.positions()).enumerate() {
            let id = 200 + k as TileId;
            b_tile_ids.push(id);
            changes.push((id, Some(LocalMap::prepare(b_axis.tile, chunk).expect("B tile"))));
        }
        table.advance(1, 1, changes);
    }
    let sel_prep_us = t_sel.elapsed().as_nanos() as f64 / 1000.0;
    let sel_alloc = alloc_bytes() - ab_sel;

    // ---- frame, tile views, plane, outdeg -------------------------------
    let t_plane = Instant::now();
    let frame = table.seal();
    let tiles: Vec<TileMap<'_>> = r_tiles
        .iter()
        .map(|&t| table.map(&frame, t).expect("R tile readable"))
        .collect();
    let mut plane = Plane::new(coords);
    for &(s, dd) in d.edges {
        let cx = coord_of(&table, &frame, &r_tiles, tw, u64::from(s)).expect("src in domain");
        let cy = coord_of(&table, &frame, &r_tiles, tw, u64::from(dd)).expect("dst in domain");
        plane.set(cx, cy);
    }
    let outdeg: Vec<i64> = (0..coords).map(|c| popcount(plane.row(c)) as i64).collect();
    let mut sel_a = vec![0u64; wpr];
    let mut sel_b = vec![0u64; wpr];
    let mut tmp = vec![0u64; wpr];
    let mut acc = vec![0u64; wpr];
    let mut yb = vec![0u64; wpr];
    let mut ff = vec![0u64; wpr];
    let mut gg = vec![0u64; wpr];
    let mut visited = vec![0u64; wpr];
    let mut indeg = vec![0i64; coords];
    let mut prod = vec![0i64; coords];
    let mut cur: Vec<u32> = Vec::with_capacity(coords);
    let mut next: Vec<u32> = Vec::with_capacity(coords);
    let plane_us = t_plane.elapsed().as_nanos() as f64 / 1000.0;
    let base = Cost {
        prep_us: r_us + plane_us,
        alloc_prep: (alloc_bytes() - ab_base).saturating_sub(sel_alloc),
        peak_prep: peak_bytes().saturating_sub(live_before),
    };
    let with_sel = Cost {
        prep_us: base.prep_us + sel_prep_us,
        alloc_prep: base.alloc_prep + sel_alloc,
        ..base
    };

    // ---- CSR (cold, only the csr arms are charged) -----------------------
    let ab_csr = alloc_bytes();
    let t_csr = Instant::now();
    let mut csr_off: Vec<u32> = Vec::with_capacity(coords + 1);
    let mut csr_adj: Vec<u32> = Vec::with_capacity(d.edges.len());
    csr_off.push(0);
    for x in 0..coords {
        for_each_bit(plane.row(x), |y| csr_adj.push(y as u32));
        csr_off.push(csr_adj.len() as u32);
    }
    let with_csr = Cost {
        prep_us: base.prep_us + t_csr.elapsed().as_nanos() as f64 / 1000.0,
        alloc_prep: base.alloc_prep + (alloc_bytes() - ab_csr),
        ..base
    };

    if base.prep_us >= 1000.0 {
        println!(
            "{:<13} prep (tiles + plane + outdeg) {:.3} us COLD-FLAG (reported, not gate); selection tiles {:.3} us; csr {:.3} us",
            d.meta.dataset,
            base.prep_us,
            sel_prep_us,
            with_csr.prep_us - base.prep_us
        );
    }

    // ---- WHERE arms, timed separately; masks must agree -------------------
    virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
    virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
    qa.keep(&lanes.a, &live, &mut keep_a);
    qb.keep(&lanes.b, &live, &mut keep_b);
    project_keep(&tiles, tw, &keep_a, &mut tmp);
    let a_agree = tmp == sel_a;
    project_keep(&tiles, tw, &keep_b, &mut tmp);
    let b_agree = tmp == sel_b;
    if !(a_agree && b_agree) {
        println!("{:<13} WHERE arms DISAGREE (A {a_agree}, B {b_agree})", d.meta.dataset);
        all_ok = false;
    }
    let wq_a = median(&time_runs(RUNS, || {
        qa.keep(&lanes.a, &live, &mut keep_a);
        project_keep(&tiles, tw, &keep_a, &mut sel_a);
    }));
    let wq_b = median(&time_runs(RUNS, || {
        qb.keep(&lanes.b, &live, &mut keep_b);
        project_keep(&tiles, tw, &keep_b, &mut sel_b);
    }));
    let wv_a = median(&time_runs(RUNS, || {
        virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
    }));
    let wv_b = median(&time_runs(RUNS, || {
        virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
    }));
    let cols_ab = (Some(wq_a + wq_b), Some(wv_a + wv_b), None);
    let cols_a = (Some(wq_a), Some(wv_a), None);
    for (what, v) in [("where_q A+B", wq_a + wq_b), ("where_v A+B", wv_a + wv_b)] {
        if v >= 1000.0 {
            println!("{:<13} {what} {v:.3} us{}", d.meta.dataset, flag(v));
        }
    }

    let o = d.oracle;
    let meta = &d.meta;

    // ---- W1: SELECT count(*) FROM nodes WHERE a < 50 ----------------------
    {
        let r = run_arm(meta, w1_prep, "W1", "quack_count", true, o.w1, (None, None, None), &mut |_w| {
            q1.count(&lanes.a, &live) as i64
        });
        push(recs, &mut all_ok, r, None);
    }

    // ---- W2 ----------------------------------------------------------------
    {
        let r = run_arm(meta, base, "W2", "fold", true, o.w2, cols_ab, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
                virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
            }
            fold_w2(&plane, &sel_a, &sel_b, &mut tmp)
        });
        push(recs, &mut all_ok, r, Some(o.sel_a as i64));
    }

    // ---- W3: back, tile, csr -----------------------------------------------
    {
        let r = run_arm(meta, base, "W3", "back", true, o.w3, cols_ab, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
                virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
            }
            fold_w3_back(&plane, &sel_a, &sel_b, &mut yb, &mut tmp)
        });
        let yb_n = popcount(&yb) as i64;
        push(recs, &mut all_ok, r, Some(yb_n));
        let r = run_arm(meta, base, "W3", "tile", true, o.w3, cols_ab, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
                virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
            }
            fold_w3_tile(&plane, &sel_a, &sel_b, &mut acc, &mut tmp)
        });
        push(recs, &mut all_ok, r, None);
        let r = run_arm(meta, with_csr, "W3", "csr", true, o.w3, cols_ab, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
                virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
            }
            fold_w3_csr(&csr_off, &csr_adj, &sel_a, &sel_b)
        });
        push(recs, &mut all_ok, r, None);
    }

    // ---- W4: front, csr -----------------------------------------------------
    {
        let r = run_arm(meta, base, "W4", "front", true, o.w4, cols_ab, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
                virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
            }
            fold_w4_front(&plane, &sel_a, &sel_b, &mut ff, &mut gg)
        });
        let front_n = popcount(&ff) as i64;
        push(recs, &mut all_ok, r, Some(front_n));
        let r = run_arm(meta, with_csr, "W4", "csr", true, o.w4, cols_ab, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
                virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
            }
            fold_w4_csr(&csr_off, &csr_adj, &sel_a, &sel_b, &mut cur, &mut next, &mut visited)
        });
        let csr_front = cur.len() as i64;
        push(recs, &mut all_ok, r, Some(csr_front));
    }

    // ---- W5a ------------------------------------------------------------------
    {
        let r = run_arm(meta, base, "W5a", "fold", true, o.w5, cols_a, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
            }
            fold_w5(&plane, &sel_a, &mut acc)
        });
        push(recs, &mut all_ok, r, Some(o.w5));
    }

    // ---- W6 -------------------------------------------------------------------
    {
        let r = run_arm(meta, base, "W6", "fold", true, o.w6, cols_a, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
            }
            fold_w6(&plane, &sel_a, &outdeg, &mut indeg, &mut prod)
        });
        push(recs, &mut all_ok, r, None);
    }

    // ---- W7 / W7i: route the selection tiles into domain coordinates ----------
    {
        let route_us = median(&time_runs(RUNS, || {
            route_selection(&table, &frame, &r_tiles, tw, &a_tile_ids, &mut sel_a);
            route_selection(&table, &frame, &r_tiles, tw, &b_tile_ids, &mut sel_b);
        }));
        if route_us >= 1000.0 {
            println!("{:<13} route A+B {route_us:.3} us{}", meta.dataset, flag(route_us));
        }
        let cols = (None, None, Some(route_us));
        for (arm, warm_flag) in [("route_each", true), ("reuse", false)] {
            let r = run_arm(meta, with_sel, "W7", arm, warm_flag, o.w2, cols, &mut |w| {
                if w {
                    route_selection(&table, &frame, &r_tiles, tw, &a_tile_ids, &mut sel_a);
                    route_selection(&table, &frame, &r_tiles, tw, &b_tile_ids, &mut sel_b);
                }
                fold_w2(&plane, &sel_a, &sel_b, &mut tmp)
            });
            push(recs, &mut all_ok, r, Some(o.sel_a as i64));
            let r = run_arm(meta, with_sel, "W7i", arm, warm_flag, o.w7i, cols, &mut |w| {
                if w {
                    route_selection(&table, &frame, &r_tiles, tw, &a_tile_ids, &mut sel_a);
                    route_selection(&table, &frame, &r_tiles, tw, &b_tile_ids, &mut sel_b);
                }
                fold_w7i(&sel_a, &sel_b, &mut tmp)
            });
            push(recs, &mut all_ok, r, None);
        }
    }

    // ---- W8: W2, delete 1% of the edges, W2 again -------------------------------
    // The update clears the forward presence bits (ids -> coordinates through
    // `locate`, then one bit per edge); outdeg and the CSR are not touched, W2
    // reads neither. Each timing run starts from the full plane (restored
    // untimed), so the median is of 201 clears.
    {
        let mut samples: Vec<f64> = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            for &(s, dd) in d.deleted {
                let cx = coord_of(&table, &frame, &r_tiles, tw, u64::from(s)).expect("src");
                let cy = coord_of(&table, &frame, &r_tiles, tw, u64::from(dd)).expect("dst");
                plane.clear(cx, cy);
            }
            samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
            for &(s, dd) in d.deleted {
                let cx = coord_of(&table, &frame, &r_tiles, tw, u64::from(s)).expect("src");
                let cy = coord_of(&table, &frame, &r_tiles, tw, u64::from(dd)).expect("dst");
                plane.set(cx, cy);
            }
        }
        let update_us = median(&samples);
        for &(s, dd) in d.deleted {
            let cx = coord_of(&table, &frame, &r_tiles, tw, u64::from(s)).expect("src");
            let cy = coord_of(&table, &frame, &r_tiles, tw, u64::from(dd)).expect("dst");
            plane.clear(cx, cy);
        }
        let r = run_arm(meta, base, "W8", "after_delete", true, o.w8, cols_ab, &mut |w| {
            if w {
                virtual_where(&tiles, tw, &lanes.a, A_LT, &mut sel_a);
                virtual_where(&tiles, tw, &lanes.b, B_LT, &mut sel_b);
            }
            fold_w2(&plane, &sel_a, &sel_b, &mut tmp)
        });
        let (mut rec, ok) = r;
        rec.update_us = Some(update_us);
        rec.interm = Some(d.deleted.len() as i64);
        all_ok &= ok;
        if update_us >= 1000.0 {
            println!("{:<13} update {update_us:.3} us{}", meta.dataset, flag(update_us));
        }
        recs.push(rec);
    }

    all_ok
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let out: PathBuf = match std::env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: bench <OUT dir>");
            std::process::exit(2);
        }
    };
    fs::create_dir_all(&out).expect("create OUT");
    println!("dense edges: p = 0.25 for dr64 and dr256, p = 0.03 for multi (4096 nodes)");

    let mut rng = Rng::new(SEED);
    let mut lanes = gen_lanes(&mut rng);
    let sizes: [(&str, usize); 3] = [("dr64", 64), ("dr256", 256), ("multi", 4096)];
    let mut domains: Vec<Vec<u64>> = Vec::new();
    for &(_, size) in &sizes {
        let dom = pick_domain(&mut Rng::new(SEED ^ size as u64), size);
        // Keep every selection non-empty: the first domain ids are in A and B,
        // the third in B (a few ids of 65,536; the lanes stay uniform).
        lanes.a[dom[0] as usize] = 0;
        lanes.b[dom[0] as usize] = 0;
        lanes.a[dom[1] as usize] = 0;
        lanes.b[dom[2] as usize] = 0;
        domains.push(dom);
    }
    write_nodes(&out.join("nodes.csv"), &lanes);

    let mut recs: Vec<Rec> = Vec::new();
    let mut all_ok = true;
    for (si, &(sname, size)) in sizes.iter().enumerate() {
        let dom = &domains[si];
        let p = if size == 4096 { 0.03 } else { 0.25 };
        for dist in Dist::ALL {
            let ds = format!("{sname}_{}", dist.name());
            let mut er = Rng::new(SEED ^ ((size as u64) << 8) ^ (dist as u64 + 1));
            let edges = gen_edges(&mut er, dom, dist, p);
            let k = (edges.len() / 100).max(1);
            let mut idx: Vec<usize> = (0..edges.len()).collect();
            for i in 0..k {
                let j = i + er.below((idx.len() - i) as u64) as usize;
                idx.swap(i, j);
            }
            let deleted: Vec<(u32, u32)> = idx[..k].iter().map(|&i| edges[i]).collect();

            let mut a_ids: Vec<u64> = dom
                .iter()
                .copied()
                .filter(|&g| lanes.a[g as usize] < A_LT)
                .collect();
            a_ids.sort_unstable_by(|x, y| y.cmp(x));
            let mut b_ids: Vec<u64> = dom
                .iter()
                .copied()
                .filter(|&g| lanes.b[g as usize] < B_LT)
                .collect();
            b_ids.sort_unstable();
            write_pairs(&out.join(format!("{ds}_edges.csv")), "src,dst", &edges);
            write_ids(&out.join(format!("{ds}_selA.csv")), &a_ids);
            write_ids(&out.join(format!("{ds}_selB.csv")), &b_ids);
            write_pairs(&out.join(format!("{ds}_delete.csv")), "src,dst", &deleted);

            let orc = oracle(&lanes, dom, &edges, &deleted);
            println!(
                "== {ds}: {} nodes, {} edges, |A|={}, |B|={}, deleting {}",
                size,
                edges.len(),
                orc.sel_a,
                orc.sel_b,
                deleted.len()
            );
            let data = Data {
                meta: Meta {
                    dataset: ds.clone(),
                    size,
                    dist: dist.name(),
                    edges: edges.len(),
                },
                lanes: &lanes,
                dom,
                edges: &edges,
                deleted: &deleted,
                oracle: &orc,
            };
            all_ok &= run_dataset(&mut recs, &data);
        }
    }
    write_json(&out.join("ours.json"), &recs);
    println!("wrote {} records to {}", recs.len(), out.join("ours.json").display());
    if !all_ok {
        eprintln!("bench: at least one answer disagreed with the scalar oracle");
        std::process::exit(1);
    }
}
