//! D-HOLO-H14d: C's own operation, in four stages, C only
//! (plan `.claude/plans/d-holo-h14d-logic-mask-read-v1.md`).
//!
//! 1. logic: the thought's set expression is reduced symbolically
//!    (`lance_graph_relalg::normalize` / `symbolic`); no data is touched.
//! 2. address materialization: the reduced expression becomes the address set
//!    `R` over the thought's presence planes `P_l` and the aperture `A`, as one
//!    mask-risc program (`relalg::run`, `Measure::Keep`); the COUNT terminal is
//!    `Measure::Count`.
//! 3. masked read: `R` prunes a Lance 13 read at the source
//!    (`with_row_addr_prefilter`).
//! 4. terminal: per survivor `i` of `R`, the presence bit of each requested
//!    lane gates a positional load of `T_l[i]`, then the operation runs.
//!
//! Only addresses are materialized before a terminal. Absent values hold
//! `POISON`, so a read the presence bits did not admit breaks every result.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::{Array, Int32Array, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_graph_mask_risc::{words_for, LaneRef, Planes};
use lance_graph_relalg::{normalize, run, run_as_written, symbolic, Answer, Measure, Query, Rel};

/// Addresses in one tile.
const N: usize = 65_536;
/// 64-address words in one tile.
const W: usize = N / 64;
/// Lanes the thought can request.
const LANES: usize = 8;
/// The plane index of the aperture in `Planes::masks`.
const APERTURE: u16 = LANES as u16;
/// The value an absent address holds; reading it breaks every result.
const POISON: i32 = i32::MIN;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else {
        "below v4: timings not reportable"
    }
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    /// Each address independently.
    Uniform,
    /// Runs of 64 addresses: one draw per word.
    Clustered,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Pattern::Uniform => "uniform",
            Pattern::Clustered => "clustered",
        }
    }
}

/// Whether lane `l` holds a value at address `i` (closed form; the oracle).
fn present(p: Pattern, d: f64, l: usize, i: usize) -> bool {
    let key = match p {
        Pattern::Uniform => i as u64,
        Pattern::Clustered => (i / 64) as u64,
    };
    let h = mix((l as u64) << 40 ^ key ^ 0x5151);
    (h >> 11) as f64 / (1u64 << 53) as f64 < d
}

/// The value lane `l` holds at address `i` when present: small, so sums fit.
fn value(l: usize, i: usize) -> i32 {
    (mix((l as u64) << 32 | i as u64) % 2001) as i32 - 1000
}

fn in_aperture(width: usize, i: usize) -> bool {
    i < width
}

/// One thought's resident data: presence planes, aperture, positional lanes.
struct Fx {
    p: Vec<Vec<u64>>,
    a: Vec<u64>,
    t: Vec<Vec<i32>>,
    pattern: Pattern,
    density: f64,
    width: usize,
}

fn plane(f: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(N)];
    for i in (0..N).filter(|&i| f(i)) {
        w[i / 64] |= 1 << (i % 64);
    }
    w
}

fn fixture(pattern: Pattern, density: f64, width: usize) -> Fx {
    let p = (0..LANES)
        .map(|l| plane(|i| present(pattern, density, l, i)))
        .collect();
    let t = (0..LANES)
        .map(|l| {
            (0..N)
                .map(|i| {
                    if present(pattern, density, l, i) {
                        value(l, i)
                    } else {
                        POISON
                    }
                })
                .collect()
        })
        .collect();
    Fx {
        p,
        a: plane(|i| in_aperture(width, i)),
        t,
        pattern,
        density,
        width,
    }
}

impl Fx {
    /// Run `f` with the planes relalg reads: `P_0 … P_7`, then `A`.
    fn with_planes<T>(&self, f: impl FnOnce(&Planes<'_>) -> T) -> T {
        let lanes: Vec<LaneRef<'_>> = self.t.iter().map(|t| LaneRef::I32(t)).collect();
        let mut masks: Vec<&[u64]> = self.p.iter().map(Vec::as_slice).collect();
        masks.push(&self.a);
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        f(&planes)
    }

    /// Whether `i` belongs to the form's population (the oracle).
    fn member(&self, form: Form, k: usize, i: usize) -> bool {
        let lanes = (0..k).map(|l| present(self.pattern, self.density, l, i));
        let hit = match form {
            Form::And => lanes.fold(true, |a, b| a && b),
            Form::Or => lanes.fold(false, |a, b| a || b),
        };
        hit && in_aperture(self.width, i)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// `P_0 ∧ … ∧ P_{k-1} ∧ A`
    And,
    /// `(P_0 ∨ … ∨ P_{k-1}) ∧ A`
    Or,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::And => "and",
            Form::Or => "or",
        }
    }
}

/// The thought's set expression over its first `k` lanes and the aperture.
fn expr(form: Form, k: usize) -> Rel {
    let lanes: Vec<Rel> = (0..k as u16).map(Rel::Plane).collect();
    match form {
        Form::And => Rel::And(lanes.into_iter().chain([Rel::Plane(APERTURE)]).collect()),
        Form::Or => Rel::And(vec![Rel::Or(lanes), Rel::Plane(APERTURE)]),
    }
}

// ── stage 2: the address mask ──────────────────────────────────────────────

/// `R`, materialized by one mask-risc program.
fn materialize(fx: &Fx, rel: &Rel, r: &mut [u64]) {
    let q = Query {
        rel: rel.clone(),
        measure: Measure::Keep,
    };
    fx.with_planes(|pl| run(&q, pl, Some(r)).expect("lowering"));
}

/// The COUNT terminal: no payload is read.
fn count(fx: &Fx, rel: &Rel) -> u64 {
    let q = Query {
        rel: rel.clone(),
        measure: Measure::Count,
    };
    match fx.with_planes(|pl| run(&q, pl, None).expect("lowering")).answer {
        Answer::Count(c) => c,
        a => panic!("count answered {a:?}"),
    }
}

/// Σ of lane `l` over `R`, fused into the mask-risc program.
fn fused_sum(fx: &Fx, rel: &Rel, l: u16) -> i64 {
    let q = Query {
        rel: rel.clone(),
        measure: Measure::Sum(l),
    };
    match fx.with_planes(|pl| run(&q, pl, None).expect("lowering")).answer {
        Answer::Sum(s) => s,
        a => panic!("sum answered {a:?}"),
    }
}

// ── stage 4: the payload terminal ──────────────────────────────────────────

/// What the payload terminal produced, and how many values it loaded.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Out {
    /// Σ over survivors of the present requested values.
    sum: i64,
    /// Survivors where lanes 0 and 1 are both present and `T_0 > T_1`.
    greater: u64,
    /// Values loaded.
    reads: u64,
}

/// Per survivor `i` of `R`: lane `l`'s presence bit gates the load of
/// `T_l[i]`, then the operation runs (add, and compare lanes 0 and 1).
fn payload(fx: &Fx, r: &[u64], k: usize) -> Out {
    let mut o = Out::default();
    for (w, &bits) in r.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let i = w * 64 + b.trailing_zeros() as usize;
            b &= b - 1;
            let mut v = [None; LANES];
            for l in 0..k {
                if fx.p[l][w] >> (i % 64) & 1 == 1 {
                    v[l] = Some(fx.t[l][i]);
                    o.reads += 1;
                }
            }
            for x in v.iter().take(k).flatten() {
                o.sum += i64::from(*x);
            }
            if let (Some(a), Some(b)) = (v[0], v[1]) {
                o.greater += u64::from(a > b);
            }
        }
    }
    o
}

/// One address: the aperture bit, then each requested lane's presence bit,
/// then the load. `None` = absent; `Some(None)` = outside the aperture.
fn point(fx: &Fx, i: usize, k: usize) -> Option<[Option<i32>; LANES]> {
    if fx.a[i / 64] >> (i % 64) & 1 == 0 {
        return None;
    }
    let mut v = [None; LANES];
    for (l, slot) in v.iter_mut().enumerate().take(k) {
        if fx.p[l][i / 64] >> (i % 64) & 1 == 1 {
            *slot = Some(fx.t[l][i]);
        }
    }
    Some(v)
}

// ── stage 1: the corpus ────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = mix(self.0);
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A random expression over the nine planes, with redundancy and
/// contradiction mixed in.
fn random_rel(rng: &mut Rng, depth: u32) -> Rel {
    let leaf = |rng: &mut Rng| Rel::Plane(rng.below(u64::from(APERTURE) + 1) as u16);
    if depth == 0 {
        return leaf(rng);
    }
    let a = random_rel(rng, depth - 1);
    match rng.below(9) {
        0 => Rel::And(vec![a.clone(), Rel::Not(Box::new(a))]),
        1 => Rel::Or(vec![a.clone(), Rel::Not(Box::new(a))]),
        2 => {
            let b = random_rel(rng, depth - 1);
            Rel::And(vec![a.clone(), Rel::Or(vec![a, b])])
        }
        3 => Rel::And(vec![a.clone(), a]),
        4 => Rel::Or(vec![a, Rel::Empty]),
        5 => Rel::And(vec![a, Rel::All]),
        6 => Rel::Not(Box::new(a)),
        7 => Rel::Or(vec![a, random_rel(rng, depth - 1)]),
        _ => Rel::And(vec![a, random_rel(rng, depth - 1)]),
    }
}

fn corpus(n: usize) -> Vec<Rel> {
    let mut rng = Rng(0x00C0_FFEE);
    (0..n)
        .map(|_| {
            let d = 1 + rng.below(3) as u32;
            random_rel(&mut rng, d)
        })
        .collect()
}

/// Evaluate `rel` at address `i` over the fixture's bits (the oracle).
fn holds(fx: &Fx, rel: &Rel, i: usize) -> bool {
    match rel {
        Rel::All => true,
        Rel::Empty => false,
        Rel::Plane(p) => {
            let w = if *p == APERTURE { &fx.a } else { &fx.p[*p as usize] };
            w[i / 64] >> (i % 64) & 1 == 1
        }
        Rel::Not(c) => !holds(fx, c, i),
        Rel::And(cs) => cs.iter().all(|c| holds(fx, c, i)),
        Rel::Or(cs) => cs.iter().any(|c| holds(fx, c, i)),
        Rel::Atom(_) => unreachable!("the corpus holds planes only"),
    }
}

/// Stage 1 for one program: answered without running (`programs == 0`), or run.
fn stage1(fx: &Fx, rel: &Rel) -> (u64, usize, usize) {
    let q = Query {
        rel: rel.clone(),
        measure: Measure::Count,
    };
    let r = fx.with_planes(|pl| run(&q, pl, None).expect("lowering"));
    match r.answer {
        Answer::Count(c) => (c, r.programs, r.passes),
        a => panic!("{a:?}"),
    }
}

// ── stage 3: Lance ─────────────────────────────────────────────────────────

fn lane_name(l: usize) -> String {
    format!("lane_{l}")
}

fn schema() -> Arc<Schema> {
    let mut f = vec![Field::new("coord", DataType::UInt32, false)];
    f.extend((0..LANES).map(|l| Field::new(lane_name(l), DataType::Int32, false)));
    Arc::new(Schema::new(f))
}

fn batch(fx: &Fx) -> RecordBatch {
    let mut cols: Vec<Arc<dyn Array>> = vec![Arc::new(UInt32Array::from_iter_values(0..N as u32))];
    for l in 0..LANES {
        cols.push(Arc::new(Int32Array::from(fx.t[l].clone())));
    }
    RecordBatch::try_new(schema(), cols).unwrap()
}

async fn write(uri: &str, fx: &Fx) -> Dataset {
    let reader = RecordBatchIterator::new([Ok(batch(fx))], schema());
    let params = WriteParams {
        mode: WriteMode::Create,
        max_rows_per_file: N,
        ..Default::default()
    };
    Dataset::write(reader, uri, Some(params)).await.unwrap()
}

/// `_rowid` of every stored coordinate, positionally (coordinate `c` at `[c]`).
async fn row_ids(ds: &Dataset) -> Vec<u64> {
    let mut s = ds.scan();
    s.project(&["coord"]).unwrap();
    s.with_row_id();
    let b = s.try_into_batch().await.unwrap();
    let coord = b.column_by_name("coord").unwrap().as_any().downcast_ref::<UInt32Array>().unwrap();
    let id = b.column_by_name("_rowid").unwrap().as_any().downcast_ref::<UInt64Array>().unwrap();
    let mut out = vec![u64::MAX; N];
    for j in 0..b.num_rows() {
        out[coord.value(j) as usize] = id.value(j);
    }
    out
}

#[derive(Debug, Default, Clone)]
struct Io {
    bytes_read: usize,
    rows_scanned: usize,
    rows: usize,
}

/// The read of lanes `0..k`; pruned by `mask` when given. Returns
/// `coord → values` and Lance's own statistics.
async fn read(
    ds: &Dataset,
    mask: Option<RowAddrMask>,
    k: usize,
) -> (BTreeMap<u32, Vec<i32>>, Io) {
    let mut names = vec!["coord".to_string()];
    names.extend((0..k).map(lane_name));
    let stats: Arc<Mutex<Io>> = Arc::default();
    let sink = stats.clone();
    let mut s = ds.scan();
    s.project(&names).unwrap();
    if let Some(m) = mask {
        s.with_row_addr_prefilter(m);
    }
    s.scan_stats_callback(Arc::new(move |c| {
        let mut io = sink.lock().unwrap();
        io.bytes_read = c.bytes_read;
        io.rows_scanned = c.all_counts.get("rows_scanned").copied().unwrap_or(0);
    }));
    let b = s.try_into_batch().await.unwrap();
    let mut io = stats.lock().unwrap().clone();
    io.rows = b.num_rows();
    let coord = b.column_by_name("coord").unwrap().as_any().downcast_ref::<UInt32Array>().unwrap();
    let lanes: Vec<&Int32Array> = (0..k)
        .map(|l| b.column_by_name(&lane_name(l)).unwrap().as_any().downcast_ref::<Int32Array>().unwrap())
        .collect();
    let mut out = BTreeMap::new();
    for j in 0..b.num_rows() {
        out.insert(coord.value(j), lanes.iter().map(|a| a.value(j)).collect());
    }
    (out, io)
}

/// `R`'s survivors as Lance row ids.
fn row_mask(r: &[u64], ids: &[u64]) -> RowAddrMask {
    let mut set = Vec::new();
    for (w, &bits) in r.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            set.push(ids[w * 64 + b.trailing_zeros() as usize]);
            b &= b - 1;
        }
    }
    RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(set))
}

// ── timing ─────────────────────────────────────────────────────────────────

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median of 7 batches of `reps` calls, in ns per `units`; `flush` runs
/// before each batch (empty for hot).
fn time(reps: usize, units: f64, mut flush: impl FnMut(), mut f: impl FnMut()) -> f64 {
    let mut out = Vec::new();
    for _ in 0..7 {
        flush();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        out.push(t.elapsed().as_nanos() as f64 / reps as f64 / units);
    }
    median(out)
}

fn evict(buf: &mut [u64]) {
    for (j, x) in buf.iter_mut().enumerate() {
        *x = x.wrapping_add(j as u64);
    }
    std::hint::black_box(&buf);
}

fn csv(fields: &[String]) {
    println!("csv,{}", fields.join(","));
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    println!("lance: {} (resolved at build time; no lockfile is tracked)", env!("PROBE_LANCE_VERSION"));
    println!("tier: {}", tier());
    println!("csv,stage,pattern,density,width,form,k,cache,metric,value");
    let mut ev = vec![0u64; 8 << 20];

    // Stage 1: the corpus over one fixture.
    {
        let fx = fixture(Pattern::Uniform, 0.1, N);
        let progs = corpus(2000);
        let mut decided = 0;
        let mut passes_rw = 0usize;
        let mut passes_as = 0usize;
        for rel in &progs {
            let (_, programs, passes) = stage1(&fx, rel);
            decided += usize::from(programs == 0);
            passes_rw += passes;
            let q = Query { rel: rel.clone(), measure: Measure::Count };
            passes_as += fx.with_planes(|pl| run_as_written(&q, pl, None).unwrap()).passes;
        }
        let ns = time(1, progs.len() as f64, || {}, || {
            for rel in &progs {
                std::hint::black_box(symbolic(&normalize(rel), Measure::Count, N));
            }
        });
        let row = |m: &str, v: String| csv(&["1".into(), "uniform".into(), "0.1".into(), N.to_string(), "-".into(), "-".into(), "-".into(), m.into(), v]);
        row("programs", progs.len().to_string());
        row("decided_symbolically", decided.to_string());
        row("passes_rewritten", passes_rw.to_string());
        row("passes_as_written", passes_as.to_string());
        row("reduce_ns_per_program", format!("{ns:.1}"));
    }

    let dir = tempfile::tempdir().unwrap();
    for pattern in [Pattern::Uniform, Pattern::Clustered] {
        for &d in &[0.001, 0.01, 0.1, 0.5] {
            for &width in &[1024usize, 8192, N] {
                let fx = fixture(pattern, d, width);
                let uri = dir.path().join(format!("{}_{d}_{width}", pattern.name()));
                let ds = write(uri.to_str().unwrap(), &fx).await;
                let ids = row_ids(&ds).await;
                for form in [Form::And, Form::Or] {
                    for &k in &[1usize, 2, 4, 8] {
                        if form == Form::Or && k == 1 {
                            continue;
                        }
                        let rel = expr(form, k);
                        let mut r = vec![0u64; W];
                        materialize(&fx, &rel, &mut r);
                        let survivors: u64 = r.iter().map(|w| u64::from(w.count_ones())).sum();
                        let row = |cache: &str, m: &str, v: String| {
                            csv(&["2-4".into(), pattern.name().into(), d.to_string(), width.to_string(), form.name().into(), k.to_string(), cache.into(), m.into(), v])
                        };
                        row("-", "survivors", survivors.to_string());
                        // stage 2 and the COUNT terminal
                        for (cache, cold) in [("hot", false), ("cold", true)] {
                            let mut sink = vec![0u64; W];
                            let ns = time(if cold { 1 } else { 50 }, W as f64, || if cold { evict(&mut ev) }, || materialize(&fx, &rel, &mut sink));
                            row(cache, "materialize_ns_per_word", format!("{ns:.3}"));
                            let ns = time(if cold { 1 } else { 50 }, W as f64, || if cold { evict(&mut ev) }, || { std::hint::black_box(count(&fx, &rel)); });
                            row(cache, "count_ns_per_word", format!("{ns:.3}"));
                            // stage 4: payload terminal over the survivors
                            let o = payload(&fx, &r, k);
                            let ns = time(if cold { 1 } else { 50 }, survivors.max(1) as f64, || if cold { evict(&mut ev) }, || { std::hint::black_box(payload(&fx, &r, k)); });
                            row(cache, "payload_ns_per_survivor", format!("{ns:.3}"));
                            if !cold {
                                row("-", "payload_reads", o.reads.to_string());
                            }
                            let ns = time(if cold { 1 } else { 50 }, W as f64, || if cold { evict(&mut ev) }, || { std::hint::black_box(fused_sum(&fx, &rel, 0)); });
                            row(cache, "fused_sum_ns_per_word", format!("{ns:.3}"));
                            // a single-address query at 1,024 addresses spread over the tile
                            let pts: Vec<usize> = (0..1024).map(|j| (mix(j as u64) as usize) % N).collect();
                            let ns = time(if cold { 1 } else { 20 }, pts.len() as f64, || if cold { evict(&mut ev) }, || {
                                for &i in &pts {
                                    std::hint::black_box(point(&fx, i, k));
                                }
                            });
                            row(cache, "point_ns_per_query", format!("{ns:.3}"));
                        }
                        // stage 3: Lance, form ∧ only (the read projection is lanes 0..k)
                        if form == Form::And {
                            let (_, plain) = read(&ds, None, k).await;
                            let mask = row_mask(&r, &ids);
                            let t = Instant::now();
                            let (got, masked) = read(&ds, Some(mask), k).await;
                            let wall = t.elapsed().as_nanos() as f64;
                            row("-", "lance_rows_returned", masked.rows.to_string());
                            row("-", "lance_rows_scanned_masked", masked.rows_scanned.to_string());
                            row("-", "lance_rows_scanned_unmasked", plain.rows_scanned.to_string());
                            row("-", "lance_bytes_masked", masked.bytes_read.to_string());
                            row("-", "lance_bytes_unmasked", plain.bytes_read.to_string());
                            row("-", "lance_ns_per_survivor_wall", format!("{:.1}", wall / survivors.max(1) as f64));
                            assert_eq!(got.len() as u64, survivors, "lance returned other rows than R");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELLS: [(Pattern, f64, usize); 4] = [
        (Pattern::Uniform, 0.5, N),
        (Pattern::Uniform, 0.1, 8192),
        (Pattern::Clustered, 0.5, 8192),
        (Pattern::Clustered, 0.1, N),
    ];

    fn reference_r(fx: &Fx, form: Form, k: usize) -> Vec<u64> {
        plane(|i| fx.member(form, k, i))
    }

    /// Check 1: `R` equals the per-address reference.
    #[test]
    fn r_equals_the_reference() {
        for (p, d, w) in CELLS {
            let fx = fixture(p, d, w);
            for form in [Form::And, Form::Or] {
                for k in [1, 2, 4, 8] {
                    let mut r = vec![0u64; W];
                    materialize(&fx, &expr(form, k), &mut r);
                    assert_eq!(r, reference_r(&fx, form, k), "{p:?} {d} {w} {form:?} k {k}");
                }
            }
        }
    }

    /// Check 2: rewritten == as written == reference on the corpus; every
    /// program that reduces to `Empty` or to `All` runs no program.
    #[test]
    fn stage1_answers_agree_and_contradictions_run_nothing() {
        let fx = fixture(Pattern::Uniform, 0.1, N);
        let progs = corpus(2000);
        let mut contradictions = 0;
        for rel in &progs {
            let (c, programs, _) = stage1(&fx, rel);
            let q = Query { rel: rel.clone(), measure: Measure::Count };
            let as_written = fx.with_planes(|pl| run_as_written(&q, pl, None).unwrap());
            assert_eq!(as_written.answer, Answer::Count(c));
            let want = (0..N).filter(|&i| holds(&fx, rel, i)).count() as u64;
            assert_eq!(c, want, "{rel:?}");
            let n = normalize(rel);
            if n == Rel::Empty || n == Rel::All {
                contradictions += 1;
                assert_eq!(programs, 0, "{rel:?} reduced to {n:?} but ran");
            }
        }
        // anti-vacuity: the corpus really contains decidable programs
        assert!(contradictions > 100, "only {contradictions} decidable programs");
        // and programs that must run
        assert!(contradictions * 2 < progs.len(), "{contradictions} of {}", progs.len());
    }

    /// Check 3: the COUNT terminal == popcount(R) == reference.
    #[test]
    fn count_terminal_equals_popcount() {
        for (p, d, w) in CELLS {
            let fx = fixture(p, d, w);
            for form in [Form::And, Form::Or] {
                for k in [1, 2, 4] {
                    let r = reference_r(&fx, form, k);
                    let pop: u64 = r.iter().map(|w| u64::from(w.count_ones())).sum();
                    assert_eq!(count(&fx, &expr(form, k)), pop);
                }
            }
        }
    }

    fn reference_out(fx: &Fx, form: Form, k: usize) -> Out {
        let mut o = Out::default();
        for i in (0..N).filter(|&i| fx.member(form, k, i)) {
            let v: Vec<Option<i32>> = (0..k)
                .map(|l| present(fx.pattern, fx.density, l, i).then(|| value(l, i)))
                .collect();
            for x in v.iter().flatten() {
                o.sum += i64::from(*x);
                o.reads += 1;
            }
            if k >= 2 {
                if let (Some(a), Some(b)) = (v[0], v[1]) {
                    o.greater += u64::from(a > b);
                }
            }
        }
        o
    }

    /// Check 4: payload terminal == reference, reads == Σ_l |R ∧ P_l|.
    #[test]
    fn payload_terminal_reads_exactly_the_present_survivors() {
        for (p, d, w) in CELLS {
            let fx = fixture(p, d, w);
            for form in [Form::And, Form::Or] {
                for k in [1, 2, 4, 8] {
                    let mut r = vec![0u64; W];
                    materialize(&fx, &expr(form, k), &mut r);
                    let got = payload(&fx, &r, k);
                    assert_eq!(got, reference_out(&fx, form, k), "{p:?} {d} {w} {form:?} k {k}");
                    let want: u64 = (0..k)
                        .map(|l| r.iter().zip(&fx.p[l]).map(|(a, b)| u64::from((a & b).count_ones())).sum::<u64>())
                        .sum();
                    assert_eq!(got.reads, want);
                    if form == Form::And {
                        let pop: u64 = r.iter().map(|w| u64::from(w.count_ones())).sum();
                        assert_eq!(got.reads, pop * k as u64);
                    }
                }
            }
        }
    }

    /// Check 5: the fused Σ equals the explicit payload terminal (∧ form,
    /// lane 0 alone).
    #[test]
    fn fused_sum_equals_the_payload_terminal() {
        for (p, d, w) in CELLS {
            let fx = fixture(p, d, w);
            for k in [1, 2, 4] {
                let rel = expr(Form::And, k);
                let mut r = vec![0u64; W];
                materialize(&fx, &rel, &mut r);
                let o = payload(&fx, &r, 1);
                assert_eq!(fused_sum(&fx, &rel, 0), o.sum);
            }
        }
    }

    /// Check 6: Lance returns exactly `R`'s survivors with reference values.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lance_reads_exactly_the_survivors() {
        let dir = tempfile::tempdir().unwrap();
        for (j, (p, d, w)) in CELLS.into_iter().enumerate() {
            let fx = fixture(p, d, w);
            let uri = dir.path().join(format!("c{j}"));
            let ds = write(uri.to_str().unwrap(), &fx).await;
            let ids = row_ids(&ds).await;
            assert!(ids.iter().all(|&x| x != u64::MAX), "every coordinate stored");
            for k in [1, 2, 4] {
                let mut r = vec![0u64; W];
                materialize(&fx, &expr(Form::And, k), &mut r);
                let (got, io) = read(&ds, Some(row_mask(&r, &ids)), k).await;
                let want: Vec<usize> = (0..N).filter(|&i| fx.member(Form::And, k, i)).collect();
                assert_eq!(io.rows, want.len(), "{p:?} {d} {w} k {k}");
                assert_eq!(got.len(), want.len());
                for &i in &want {
                    let vals = &got[&(i as u32)];
                    for (l, &v) in vals.iter().enumerate() {
                        assert_eq!(v, value(l, i));
                    }
                }
            }
        }
    }

    /// Check 7: the single-address query equals the reference, absent and
    /// outside the aperture included.
    #[test]
    fn point_query_equals_the_reference() {
        for (p, d, w) in CELLS {
            let fx = fixture(p, d, w);
            let mut absent = 0;
            for j in 0..4096u64 {
                let i = mix(j) as usize % N;
                let got = point(&fx, i, 4);
                if !in_aperture(w, i) {
                    assert_eq!(got, None);
                    continue;
                }
                let got = got.unwrap();
                for l in 0..4 {
                    let want = present(p, d, l, i).then(|| value(l, i));
                    absent += usize::from(want.is_none());
                    assert_eq!(got[l], want, "{p:?} {d} {w} i {i} l {l}");
                }
            }
            assert!(absent > 0, "the sample holds absent values");
        }
    }

    /// Check 8: the fixture has the stated shape.
    #[test]
    fn fixture_has_the_stated_shape() {
        for (p, d, w) in CELLS {
            let fx = fixture(p, d, w);
            for l in 0..LANES {
                let n = fx.p[l].iter().map(|x| x.count_ones() as f64).sum::<f64>();
                let want = d * N as f64;
                assert!((n - want).abs() <= 0.1 * want + 64.0, "{p:?} {d} lane {l}: {n}");
                for i in 0..N {
                    let is_p = fx.p[l][i / 64] >> (i % 64) & 1 == 1;
                    assert_eq!(is_p, fx.t[l][i] != POISON, "poison exactly at absent");
                }
            }
            let a: u32 = fx.a.iter().map(|x| x.count_ones()).sum();
            assert_eq!(a as usize, w);
        }
    }
}
