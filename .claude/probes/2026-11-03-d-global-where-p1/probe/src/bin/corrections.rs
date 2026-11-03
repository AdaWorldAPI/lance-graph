//! LAB: D-GLOBAL-WHERE P1, three paired corrections (A/B over the SAME data and
//! the SAME answer). Every row is labelled `in-memory`.
//!
//! Lab only. Nothing here is a compiled or run result until the orchestrator
//! builds and runs it. The original probes (falsify_p1, project, compose,
//! where_k) and their timings are the baseline and are unchanged; no operand
//! here replaces an established physical representation.
//!
//!   1. ROUTE ONCE VS PER QUERY: Q in {1, 16, 256} queries over one selection.
//!      `per_query` runs `Binding::prepare` (WorkingSet already built; routing +
//!      CSR + lanes + masks) for every query. `reuse` prepares once, then per
//!      query: `Binding::guard`, the virtual WHERE for A and B, the fold.
//!      Reports amortized per-query cost, the break-even Q, and exercises a
//!      stale reuse after `replace_tile` that `guard` must refuse.
//!   2. NDARRAY KERNELS VS HAND-WRITTEN LOOPS for the ordered u8 compare over a
//!      packed u32 lane (4 x u8 per lane, project.rs layout, unchanged).
//!      `handwritten` = project.rs `fill_lane` shift + mask + compare, copied
//!      verbatim. `kernel_ternary_or` = existing `ndarray::simd`
//!      `ternary_match_u32_to_mask` / `mask_or_assign` / `mask_not` composed
//!      (value < c is the OR of at most 8 prefix matches). A DIRECT ordered
//!      compare over a packed lane has NO-EXISTING-KERNEL (see main()).
//!   3. BORROWED LANE READS VS gather_lane COPIES: `copy` = `gather_lane` into a
//!      Vec then `virtual_where`; `borrowed` = the fold reads the corpus lane in
//!      place through the working set's own coordinate -> global pointer table
//!      (`MapTable::map(..).globals()`), no new Vec of values.
//!
//! ORACLE: scalar loops over the LOGICAL formula; never timed; they read no
//! layout and no working set. Every A/B pair must give equal answers or the
//! binary exits 1.
//!
//! GATES: a warm in-fold value >= 1000 us prints FAILED-GATE (counted,
//! reported, exit code unchanged). Preparation values print COLD.
//!
//! NOT CLAIMED: absolute speed outside this host; that the generated graph or
//! lanes resemble a real corpus; that hand-written vs kernel ranking holds for
//! other shapes; anything on-disk or DuckDB. Preparation and per-query-route
//! costs are COLD (reported, not gates).
//!
//! Usage: `corrections <OUT> [--n 1000000|10000000]... [--sel 20000]`
//! Writes `<OUT>/corrections/n<N>/corrections.json`.

use std::collections::HashSet;
use std::time::Instant;

use d_global_where_p1_probe::*;

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const LABEL: &str = "in-memory";

// ---------------------------------------------------------------------------
// deterministic data
// ---------------------------------------------------------------------------

fn h(id: u64, j: u64) -> u64 {
    let mut z = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ j.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Logical value of the corpus lanes (0..1000).
fn lane_val(id: u64, salt: u64) -> i32 {
    (h(id, salt) % 1000) as i32
}

/// Logical u8 property `k` (0..4) of row `id`.
fn u8_val(id: u64, k: u32) -> u32 {
    ((h(id, 200 + u64::from(k)) >> 8) % 256) as u32
}

struct Fixture {
    n: usize,
    ids: Vec<u64>,
    edges: Vec<(u64, u64)>,
    corpus_a: Vec<i32>,
    corpus_b: Vec<i32>,
}

fn build_fixture(n: usize, sel: usize) -> Fixture {
    let sel = sel.min(MAX_TILES * 256).min(n);
    let stride = (n / sel).max(1) as u64;
    let ids: Vec<u64> = (0..sel as u64).map(|i| i * stride + h(i, 7) % stride).collect();
    let corpus_a: Vec<i32> = (0..n as u64).map(|g| lane_val(g, 100)).collect();
    let corpus_b: Vec<i32> = (0..n as u64).map(|g| lane_val(g, 101)).collect();
    let mut rng = Rng::new(SEED);
    let mut edges: Vec<(u64, u64)> = Vec::with_capacity(sel * 8);
    for i in 0..sel {
        let deg = 4 + rng.below(9) as usize;
        for _ in 0..deg {
            let t = if rng.below(8) == 0 {
                rng.below(n as u64)
            } else {
                ids[(i + rng.below(1024) as usize) % sel]
            };
            edges.push((ids[i], t));
        }
    }
    Fixture {
        n,
        ids,
        edges,
        corpus_a,
        corpus_b,
    }
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Report {
    rows: Vec<String>,
    failed_gates: usize,
    ok: bool,
}

impl Report {
    fn new() -> Report {
        Report {
            ok: true,
            ..Report::default()
        }
    }

    fn check(&mut self, what: &str, cond: bool) {
        if cond {
            println!("  [oracle ok ] {what}");
        } else {
            println!("  [ORACLE FAIL] {what}");
            self.ok = false;
        }
    }

    /// One timed row. `cold` rows are labelled, never gated.
    fn row(&mut self, group: &str, name: &str, samples: &[f64], cold: bool, alloc: (usize, usize), extra: &str) {
        let (med, p10, p90) = stats(samples);
        if med >= 1000.0 && !cold {
            self.failed_gates += 1;
        }
        println!(
            "  [{LABEL}] {group:<14} {name:<30} med {} p10 {p10:.3}us p90 {p90:.3}us alloc {}B/{}calls {extra}",
            fmt_us(med, cold),
            alloc.0,
            alloc.1
        );
        self.rows.push(format!(
            "{{\"label\":\"{LABEL}\",\"group\":\"{group}\",\"name\":\"{name}\",\"median_us\":{med:.3},\
\"p10_us\":{p10:.3},\"p90_us\":{p90:.3},\"cold\":{cold},\"failed_gate\":{},\"alloc_bytes\":{},\"alloc_calls\":{}{}}}",
            med >= 1000.0 && !cold,
            alloc.0,
            alloc.1,
            if extra.is_empty() { String::new() } else { format!(",\"note\":\"{extra}\"") }
        ));
    }

    fn note(&mut self, key: &str, value: String) {
        println!("  [{LABEL}] {key} = {value}");
        self.rows.push(format!("{{\"label\":\"{LABEL}\",\"note\":\"{key}\",\"value\":\"{value}\"}}"));
    }
}

/// Bytes and calls allocated by `f` (one execution).
fn measure<R>(f: impl FnOnce() -> R) -> ((usize, usize), R) {
    let (b0, c0) = (alloc_bytes(), alloc_calls());
    let r = f();
    ((alloc_bytes() - b0, alloc_calls() - c0), r)
}

// ---------------------------------------------------------------------------
// correction 1: route once vs per query
// ---------------------------------------------------------------------------

fn thresholds(q: usize) -> (i32, i32) {
    (100 + ((q * 37) % 800) as i32, 100 + ((q * 53) % 800) as i32)
}

fn oracle_w2(fx: &Fixture, member: &HashSet<u64>, a_lt: i32, b_lt: i32) -> i64 {
    let mut hits: HashSet<u64> = HashSet::new();
    for &(x, y) in &fx.edges {
        if member.contains(&x)
            && member.contains(&y)
            && fx.corpus_a[x as usize] < a_lt
            && fx.corpus_b[y as usize] < b_lt
        {
            hits.insert(x);
        }
    }
    hits.len() as i64
}

fn correction_1(rep: &mut Report, fx: &Fixture) {
    println!("correction 1: route once vs per query (sel {})", fx.ids.len());
    let ws = WorkingSet::build(&fx.ids, 1).expect("working set builds");
    let sizes = [ws.n_coords()];
    let a_of = |g: u64| fx.corpus_a[g as usize];
    let b_of = |g: u64| fx.corpus_b[g as usize];
    let member: HashSet<u64> = fx.ids.iter().copied().collect();
    let mut sc = FoldScratch::new(&sizes);

    // Component costs: P (prepare, cold), w (guard + two wheres), f (fold).
    let (a0, b0) = thresholds(0);
    let prep_samples = time_runs(5, || {
        let b = Binding::prepare(&[&ws], &fx.edges, Arm::B, &a_of, &b_of, a0, b0).expect("prepare");
        std::hint::black_box(&b);
    });
    let (prep_alloc, _) = measure(|| {
        let b = Binding::prepare(&[&ws], &fx.edges, Arm::B, &a_of, &b_of, a0, b0).expect("prepare");
        std::hint::black_box(&b);
    });
    rep.row("route1/part", "prepare (P)", &prep_samples, true, prep_alloc, "COLD");
    let p_us = stats(&prep_samples).0;

    let mut bind = Binding::prepare(&[&ws], &fx.edges, Arm::B, &a_of, &b_of, a0, b0).expect("prepare");
    let (qa, qb) = thresholds(5);
    let w_samples = time_runs(31, || {
        bind.guard(&[&ws], true).expect("fresh binding is current");
        virtual_where(&bind.lane_a[0], qa, &bind.valid[0], &mut bind.a[0]);
        bind.rewhere_b(qb);
    });
    let (w_alloc, _) = measure(|| {
        bind.guard(&[&ws], true).expect("fresh binding is current");
        virtual_where(&bind.lane_a[0], qa, &bind.valid[0], &mut bind.a[0]);
        bind.rewhere_b(qb);
    });
    rep.row("route1/part", "guard+where A+where B (w)", &w_samples, false, w_alloc, "");
    let w_us = stats(&w_samples).0;

    let f_samples = time_runs(31, || {
        std::hint::black_box(w2_and(&bind.graphs, &bind.a, &bind.b, &mut sc));
    });
    let (f_alloc, _) = measure(|| std::hint::black_box(w2_and(&bind.graphs, &bind.a, &bind.b, &mut sc)));
    rep.row("route1/part", "fold w2_and (f)", &f_samples, false, f_alloc, "");

    // Break-even: reuse(Q) = P + Q (w + f), per_query(Q) = Q (P + f)
    // => reuse <= per_query once Q >= P / (P - w).
    let breakeven = if p_us > w_us {
        format!("{:.3} (Q >= {})", p_us / (p_us - w_us), (p_us / (p_us - w_us)).ceil() as u64)
    } else {
        "never (P <= w)".to_string()
    };
    rep.note("route1_break_even_Q", breakeven);

    for &q_count in &[1usize, 16, 256] {
        let runs = match q_count {
            1 => 7,
            16 => 3,
            _ => 1,
        };
        let th: Vec<(i32, i32)> = (0..q_count).map(thresholds).collect();
        let mut ans_pq = vec![0i64; q_count];
        let mut ans_re = vec![0i64; q_count];

        let pq_samples = time_runs(runs, || {
            for (q, &(al, bl)) in th.iter().enumerate() {
                let b = Binding::prepare(&[&ws], &fx.edges, Arm::B, &a_of, &b_of, al, bl).expect("prepare");
                ans_pq[q] = w2_and(&b.graphs, &b.a, &b.b, &mut sc);
            }
        });
        let re_samples = time_runs(runs, || {
            let (al0, bl0) = th[0];
            let mut b = Binding::prepare(&[&ws], &fx.edges, Arm::B, &a_of, &b_of, al0, bl0).expect("prepare");
            for (q, &(al, bl)) in th.iter().enumerate() {
                b.guard(&[&ws], true).expect("reuse is legal while the frame is current");
                virtual_where(&b.lane_a[0], al, &b.valid[0], &mut b.a[0]);
                b.rewhere_b(bl);
                ans_re[q] = w2_and(&b.graphs, &b.a, &b.b, &mut sc);
            }
        });
        // Reuse per-query warm part only (prepare excluded): guard + where + fold.
        let mut b = Binding::prepare(&[&ws], &fx.edges, Arm::B, &a_of, &b_of, th[0].0, th[0].1).expect("prepare");
        let mut ans_warm = vec![0i64; q_count];
        let warm_samples = time_runs(runs.max(3), || {
            for (q, &(al, bl)) in th.iter().enumerate() {
                b.guard(&[&ws], true).expect("current");
                virtual_where(&b.lane_a[0], al, &b.valid[0], &mut b.a[0]);
                b.rewhere_b(bl);
                ans_warm[q] = w2_and(&b.graphs, &b.a, &b.b, &mut sc);
            }
        });
        let (warm_alloc, _) = measure(|| {
            for (q, &(al, bl)) in th.iter().enumerate() {
                b.guard(&[&ws], true).expect("current");
                virtual_where(&b.lane_a[0], al, &b.valid[0], &mut b.a[0]);
                b.rewhere_b(bl);
                ans_warm[q] = w2_and(&b.graphs, &b.a, &b.b, &mut sc);
            }
        });

        let per_q = |s: &[f64]| s.iter().map(|v| v / q_count as f64).collect::<Vec<f64>>();
        // Per-query route includes preparation: COLD. Reuse total includes one
        // preparation: COLD. Reuse warm part is the in-fold figure: gated.
        rep.row("route1", &format!("per_query Q={q_count} (per query)"), &per_q(&pq_samples), true, (0, 0), "includes prepare");
        rep.row("route1", &format!("reuse Q={q_count} (per query)"), &per_q(&re_samples), true, (0, 0), "includes 1 prepare");
        rep.row("route1", &format!("reuse warm Q={q_count} (per query)"), &per_q(&warm_samples), false, (warm_alloc.0 / q_count, warm_alloc.1 / q_count), "no prepare");

        // Oracle on every query (or a stride for Q=256 to bound the oracle time).
        let step = if q_count > 16 { 16 } else { 1 };
        let mut all = ans_pq == ans_re && ans_pq == ans_warm;
        for q in (0..q_count).step_by(step) {
            all &= oracle_w2(fx, &member, th[q].0, th[q].1) == ans_pq[q];
        }
        rep.check(&format!("route1 Q={q_count}: per_query == reuse == oracle (oracle every {step}th query)"), all);
    }

    // Stale reuse must be refused by Binding::guard after replace_tile.
    let small: Vec<u64> = fx.ids.iter().copied().take(2048).collect();
    let small_set: HashSet<u64> = small.iter().copied().collect();
    let small_edges: Vec<(u64, u64)> = fx
        .edges
        .iter()
        .copied()
        .filter(|(x, y)| small_set.contains(x) && small_set.contains(y))
        .collect();
    let mut sws = WorkingSet::build(&small, 9).expect("small set builds");
    let sb = Binding::prepare(&[&sws], &small_edges, Arm::B, &a_of, &b_of, 500, 500).expect("prepare small");
    let mut ssc = FoldScratch::new(&[sws.n_coords()]);
    let before = w2_and(&sb.graphs, &sb.a, &sb.b, &mut ssc);
    rep.check("route1 stale: guard accepts the fresh binding", sb.guard(&[&sws], true).is_ok());
    let frame = sws.seal();
    let mut t0: Vec<u64> = sws.table.map(&frame, 0).expect("tile 0").globals().to_vec();
    t0.reverse();
    sws.replace_tile(0, &t0).expect("replace_tile");
    let refused = matches!(sb.guard(&[&sws], true), Err(BindError::Stale(0)));
    rep.check("route1 stale: guard REFUSES the binding after replace_tile", refused);
    rep.check("route1 stale: disabled guard would NOT refuse (guard is load-bearing)", sb.guard(&[&sws], false).is_ok());
    let fresh = Binding::prepare(&[&sws], &small_edges, Arm::B, &a_of, &b_of, 500, 500).expect("re-prepare");
    let mut fsc = FoldScratch::new(&[sws.n_coords()]);
    let after_fresh = w2_and(&fresh.graphs, &fresh.a, &fresh.b, &mut fsc);
    rep.check("route1 stale: a fresh prepare after replace gives the oracle answer", {
        let m: HashSet<u64> = small.iter().copied().collect();
        let mut hits: HashSet<u64> = HashSet::new();
        for &(x, y) in &small_edges {
            if m.contains(&x) && m.contains(&y) && fx.corpus_a[x as usize] < 500 && fx.corpus_b[y as usize] < 500 {
                hits.insert(x);
            }
        }
        hits.len() as i64 == after_fresh && before == after_fresh
    });
    // Informational only: the stale binding's own fold still reads its old coordinates.
    let stale_answer = w2_and(&sb.graphs, &sb.a, &sb.b, &mut ssc);
    rep.note("route1_stale_fold_answer_unguarded", format!("{stale_answer} (reported, equals fresh {after_fresh}: {})", stale_answer == after_fresh));
}

// ---------------------------------------------------------------------------
// correction 2: ndarray kernels vs hand-written packed u8 compare
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Op {
    Lt,
    Ge,
}

/// project.rs:513-522 `fill_lane`, verbatim: one output word per 64 rows;
/// shift + mask + compare.
fn fill_lane(buf: &mut [u64], lane: &[u32], shift: u32, wmask: u32, test: impl Fn(u32) -> bool) {
    for (w, chunk) in lane.chunks(64).enumerate() {
        let mut word = 0u64;
        for (b, &x) in chunk.iter().enumerate() {
            word |= u64::from(test((x >> shift) & wmask)) << b;
        }
        buf[w] = word;
    }
}

/// The PackU8 arm of project.rs:526-533 (`Op::Lt` / `Op::Ge`).
fn handwritten(p: &[u32], shift: u32, op: Op, c: u32, out: &mut [u64]) {
    match op {
        Op::Lt => fill_lane(out, p, shift, 0xFF, |e| e < c),
        Op::Ge => fill_lane(out, p, shift, 0xFF, |e| e >= c),
    }
}

/// `field < c` as the OR of prefix matches over EXISTING ndarray::simd words:
/// for every set bit `b` of `c`, the field's bits `7..=b` equal `c`'s bits
/// above `b` followed by a 0 at `b`. At most 8 `ternary_match_u32_to_mask`
/// passes (`popcount(c)`); `acc` is fully overwritten.
fn kernel_lt(p: &[u32], shift: u32, c: u32, acc: &mut [u64], tmp: &mut [u64]) {
    acc.fill(0);
    let mut first = true;
    for b in (0..8u32).rev() {
        if (c >> b) & 1 == 1 {
            let high = (c >> (b + 1)) << (b + 1);
            let care8 = 0xFFu32 & !((1u32 << b) - 1);
            let (pattern, care) = (high << shift, care8 << shift);
            if first {
                ndarray::simd::ternary_match_u32_to_mask(p, pattern, care, acc);
                first = false;
            } else {
                ndarray::simd::ternary_match_u32_to_mask(p, pattern, care, tmp);
                ndarray::simd::mask_or_assign(acc, tmp);
            }
        }
    }
}

fn correction_2(rep: &mut Report, fx: &Fixture) {
    let n = fx.n;
    println!("correction 2: ndarray kernels vs hand-written packed u8 compare (n {n})");
    let mut p = vec![0u32; n];
    for (id, slot) in p.iter_mut().enumerate() {
        for k in 0..4u32 {
            *slot |= u8_val(id as u64, k) << (8 * k);
        }
    }
    let words = words_for(n);
    let mut hand = vec![0u64; words];
    let mut kern = vec![0u64; words];
    let mut acc = vec![0u64; words];
    let mut tmp = vec![0u64; words];

    // (name, field k, op, constant). Constants chosen by their set-bit count:
    // 64 (1 pass), 170 (4 passes), 255 (8 passes).
    let cases: [(&str, u32, Op, u32); 4] = [
        ("f0 < 64", 0, Op::Lt, 64),
        ("f2 >= 170", 2, Op::Ge, 170),
        ("f3 < 255", 3, Op::Lt, 255),
        ("f1 >= 64", 1, Op::Ge, 64),
    ];
    for &(name, k, op, c) in &cases {
        let shift = 8 * k;
        let oracle = (0..n as u64)
            .filter(|&id| {
                let v = u8_val(id, k);
                match op {
                    Op::Lt => v < c,
                    Op::Ge => v >= c,
                }
            })
            .count() as u64;

        let hand_s = time_runs(11, || {
            handwritten(&p, shift, op, c, &mut hand);
            std::hint::black_box(&hand);
        });
        let (hand_a, _) = measure(|| handwritten(&p, shift, op, c, &mut hand));
        let run_kernel = |acc: &mut [u64], tmp: &mut [u64], kern: &mut [u64]| match op {
            Op::Lt => kernel_lt(&p, shift, c, kern, tmp),
            Op::Ge => {
                kernel_lt(&p, shift, c, acc, tmp);
                ndarray::simd::mask_not(acc, n, kern);
            }
        };
        let kern_s = time_runs(11, || {
            run_kernel(&mut acc, &mut tmp, &mut kern);
            std::hint::black_box(&kern);
        });
        let (kern_a, _) = measure(|| run_kernel(&mut acc, &mut tmp, &mut kern));

        let passes = c.count_ones();
        rep.row("kernel2", &format!("handwritten {name}"), &hand_s, false, hand_a, "project.rs fill_lane");
        rep.row(
            "kernel2",
            &format!("kernel_ternary_or {name}"),
            &kern_s,
            false,
            kern_a,
            &format!("{passes} ternary passes"),
        );
        // The hand-written tail beyond n is never set (chunks of n); kernels clear it too.
        let same = hand == kern;
        rep.check(
            &format!("kernel2 {name}: masks bit-identical, popcount hand {} kernel {} oracle {oracle}", popcount(&hand), popcount(&kern)),
            same && popcount(&hand) == oracle && popcount(&kern) == oracle,
        );
    }
}

// ---------------------------------------------------------------------------
// correction 3: borrowed lane reads vs gather_lane copies
// ---------------------------------------------------------------------------

/// The fold reads the corpus lane IN PLACE through the working set's own
/// coordinate -> global pointer table (`TileMap::globals`): no Vec of values,
/// the only write is the preallocated coordinate mask.
fn where_borrowed(ws: &WorkingSet, frame: &lance_graph_dt64::dr::Frame, corpus: &[i32], lt: i32, out: &mut [u64]) {
    out.fill(0);
    let tw = ws.tw();
    for (o, &t) in ws.tile_ids.iter().enumerate() {
        if let Ok(view) = ws.table.map(frame, t) {
            for (l, &g) in view.globals().iter().enumerate() {
                if corpus[g as usize] < lt {
                    set_bit(out, o * tw + l);
                }
            }
        }
    }
}

fn correction_3(rep: &mut Report, fx: &Fixture) {
    println!("correction 3: borrowed lane reads vs gather_lane copies (sel {})", fx.ids.len());
    let ws = WorkingSet::build(&fx.ids, 1).expect("working set builds");
    let frame = ws.seal();
    let n_coords = ws.n_coords();
    let valid = ws.valid_mask(&frame); // preparation constant shared by the copy arm; excluded from its timing
    let mut out_copy = vec![0u64; words_for(n_coords)];
    let mut out_borrow = vec![0u64; words_for(n_coords)];
    let corpus = &fx.corpus_a;

    for &lt in &[100i32, 500, 900] {
        let oracle = fx.ids.iter().filter(|&&g| lane_val(g, 100) < lt).count() as u64;

        let copy_s = time_runs(21, || {
            let lane = ws.gather_lane(&frame, |g| corpus[g as usize]);
            virtual_where(&lane, lt, &valid, &mut out_copy);
            std::hint::black_box(&out_copy);
        });
        let (copy_alloc, _) = measure(|| {
            let lane = ws.gather_lane(&frame, |g| corpus[g as usize]);
            virtual_where(&lane, lt, &valid, &mut out_copy);
        });
        let borrow_s = time_runs(21, || {
            where_borrowed(&ws, &frame, corpus, lt, &mut out_borrow);
            std::hint::black_box(&out_borrow);
        });
        let (borrow_alloc, _) = measure(|| where_borrowed(&ws, &frame, corpus, lt, &mut out_borrow));

        rep.row(
            "borrow3",
            &format!("copy gather_lane lt={lt}"),
            &copy_s,
            false,
            copy_alloc,
            &format!("bytes_copied={}", n_coords * 4),
        );
        rep.row("borrow3", &format!("borrowed in-place lt={lt}"), &borrow_s, false, borrow_alloc, "bytes_copied=0");
        rep.check(
            &format!("borrow3 lt={lt}: masks identical, popcount copy {} borrowed {} oracle {oracle}", popcount(&out_copy), popcount(&out_borrow)),
            out_copy == out_borrow && popcount(&out_copy) == oracle,
        );
    }
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

fn run_size(out_dir: &str, n: usize, sel: usize) -> bool {
    println!("== n = {n}  sel = {sel}  ({LABEL}) ==");
    let t = Instant::now();
    let fx = build_fixture(n, sel);
    println!("  fixture built in {:.3}us (COLD): {} ids, {} edges", t.elapsed().as_nanos() as f64 / 1000.0, fx.ids.len(), fx.edges.len());
    let mut rep = Report::new();
    correction_1(&mut rep, &fx);
    correction_2(&mut rep, &fx);
    correction_3(&mut rep, &fx);

    let dir = format!("{out_dir}/corrections/n{n}");
    std::fs::create_dir_all(&dir).expect("create output dir");
    let json = format!(
        "{{\"lab\":true,\"label\":\"{LABEL}\",\"n\":{n},\"sel\":{},\"ok\":{},\"failed_gates\":{},\
\"no_existing_kernel\":\"direct ordered u8 compare over a packed u32 lane\",\"rows\":[\n{}\n]}}\n",
        fx.ids.len(),
        rep.ok,
        rep.failed_gates,
        rep.rows.join(",\n")
    );
    std::fs::write(format!("{dir}/corrections.json"), json).expect("write json");
    println!("  failed gates (reported, not exit code): {}", rep.failed_gates);
    rep.ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!("usage: corrections <OUT> [--n 1000000|10000000]... [--sel 20000]");
        std::process::exit(2);
    }
    let out = args[1].clone();
    let mut sizes: Vec<usize> = Vec::new();
    let mut sel = 20_000usize;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--n" if i + 1 < args.len() => {
                sizes.push(args[i + 1].parse().expect("--n takes an integer"));
                i += 2;
            }
            "--sel" if i + 1 < args.len() => {
                sel = args[i + 1].parse().expect("--sel takes an integer");
                i += 2;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if sizes.is_empty() {
        sizes = vec![1_000_000, 10_000_000];
    }
    println!(
        "NO-EXISTING-KERNEL: a DIRECT ordered u8 compare over a packed u32 lane. ndarray::simd lt_u8_to_mask/ge_u8_to_mask \
read contiguous &[u8] (would compare all 4 fields per lane, interleaved); Quack Cmp has EqU32/NeU32/MatchU32 and \
signed I32 orderings only. Arm kernel_ternary_or composes existing words (ternary_match_u32_to_mask, mask_or_assign, mask_not); nothing new is implemented."
    );
    let mut ok = true;
    for n in sizes {
        ok &= run_size(&out, n, sel);
    }
    if !ok {
        println!("CORRECTIONS FAILED: answer mismatch");
        std::process::exit(1);
    }
    println!("CORRECTIONS ALL ANSWERS MATCH THE ORACLE");
}
