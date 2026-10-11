//! D-TELEPORT-2: can Lance prune source reads before mask-risc folds, without
//! a row-sized intermediate population?
//!
//! Every arm answers the same query over a real Lance 13 dataset:
//!
//! ```text
//! SELECT COUNT(*), SUM(v) WHERE x < t
//! ```
//!
//! and is checked against a row-by-row oracle computed from the generator.
//! The probe adds no op, no lowering and no Lance-side code: it uses the
//! public scanner (`filter`, `project`, `with_row_addr_prefilter`,
//! `scan_stats_callback`) and folds the returned Arrow columns in place
//! through mask-risc.
//!
//! | arm | what Lance does | what mask-risc does |
//! |---|---|---|
//! | A  | full scan of `x`, `v` | `x < t` per batch, then COUNT and SUM(v) |
//! | C  | `with_row_addr_prefilter` of the row range the sorted fixture puts `x < t` in | SUM(v) only; the predicate is already the range |
//! | Dz | `filter("xz < t")` through a zone-map index (approximate; Lance rechecks) | SUM(v) only |
//! | Db | `filter("xb < t")` through a B-tree index (exact) | SUM(v) only |
//! | N  | `Db` with an empty projection: COUNT only | nothing |
//!
//! `xz` and `xb` hold the same values as `x`, so each index sits on its own
//! column and the planner cannot pick the other one.
//!
//! Neither D arm rechecks `x < t` in mask-risc: Lance's scan has evaluated it
//! (exactly for the B-tree, by its own recheck for the zone map), so the fold
//! reads only `v`. The folds borrow the Arrow buffers; the only per-batch
//! allocation is the validity bitmap for the nullable fixtures (one word per
//! 64 rows, into a buffer reused across batches).
//!
//! I/O is read from Lance's own `ExecutionSummaryCounts` (`bytes_read`,
//! `iops`), so physical reduction is reported apart from latency. Cold runs
//! open a fresh `Dataset` (empty index cache) and try to drop the OS page
//! cache; whether that succeeded is printed, never assumed.
//!
//! ```text
//! cargo run --release -p lance-graph-benches --example lance_source_probe
//! PROBE_N=16777216 cargo run --release -p lance-graph-benches --example lance_source_probe
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::AsArray;
use arrow::datatypes::Int32Type;
use arrow_array::{Array, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{
    scratch_words_for, tile_words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Terminal, Value,
};
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_index::IndexType;

const SCALE: i32 = 1_000_000;
const FRAG_ROWS: usize = 131_072;
const BATCH: usize = 65_536;

// ───────────────────────── data ─────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    /// `x` ascending with the row: `x < t` is one leading run.
    Sorted,
    /// The same values, shuffled: every zone spans the whole domain.
    Shuffled,
    /// Sorted, then shuffled inside windows of 65,536 rows.
    Partial,
    /// 16 ascending ramps: `x < t` is 16 disjoint runs.
    Periodic,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            s.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
}

struct Data {
    x: Vec<i32>,
    v: Vec<i32>,
    /// `false` where `x` is NULL.
    valid: Vec<bool>,
}

fn data(n: usize, layout: Layout, nulls: bool) -> Data {
    let mut x: Vec<i32> = match layout {
        Layout::Periodic => {
            let p = n / 16;
            (0..n)
                .map(|i| ((i % p) as i64 * i64::from(SCALE) / p as i64) as i32)
                .collect()
        }
        _ => (0..n)
            .map(|i| (i as i64 * i64::from(SCALE) / n as i64) as i32)
            .collect(),
    };
    let mut r = Rng(0x05EE_D0F1_A9CE);
    match layout {
        Layout::Shuffled => r.shuffle(&mut x),
        Layout::Partial => x.chunks_mut(65_536).for_each(|c| r.shuffle(c)),
        _ => {}
    }
    let v = (0..n).map(|_| (r.next() % 1_000) as i32).collect();
    let valid = (0..n).map(|i| !(nulls && i % 17 == 0)).collect();
    Data { x, v, valid }
}

/// The oracle: per row, no Lance, no mask.
fn oracle(d: &Data, t: i32) -> (u64, i64) {
    let (mut c, mut s) = (0u64, 0i64);
    for i in 0..d.x.len() {
        if d.valid[i] && d.x[i] < t {
            c += 1;
            s += i64::from(d.v[i]);
        }
    }
    (c, s)
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("xz", DataType::Int32, true),
        Field::new("xb", DataType::Int32, true),
        Field::new("v", DataType::Int32, false),
    ]))
}

/// Write `d` as fragments of `FRAG_ROWS`, then build a zone map on `xz` and a
/// B-tree on `xb`. Returns the dataset and the two build times.
async fn write(d: &Data, path: &str) -> (Dataset, f64, f64) {
    let opt = |i: usize| d.valid[i].then_some(d.x[i]);
    let col = || Arc::new(Int32Array::from_iter((0..d.x.len()).map(opt)));
    let batch = RecordBatch::try_new(
        schema(),
        vec![col(), col(), col(), Arc::new(Int32Array::from(d.v.clone()))],
    )
    .expect("batch");
    let params = WriteParams {
        mode: WriteMode::Create,
        max_rows_per_file: FRAG_ROWS,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema());
    let mut ds = Dataset::write(reader, path, Some(params))
        .await
        .expect("write");

    let t = Instant::now();
    ds.create_index_builder(
        &["xz"],
        IndexType::ZoneMap,
        &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
    )
    .name("xz_zone".into())
    .await
    .expect("zone map");
    let zone_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    ds.create_index_builder(&["xb"], IndexType::BTree, &ScalarIndexParams::default())
        .name("xb_btree".into())
        .await
        .expect("btree");
    let btree_ms = t.elapsed().as_secs_f64() * 1e3;
    (ds, zone_ms, btree_ms)
}

// ───────────────────────── folds ─────────────────────────

/// Reused fold state: one scratch arena and one validity buffer, sized for
/// the largest batch, allocated once per probe.
struct Folder {
    scratch: Vec<u64>,
    validity: Vec<u64>,
}

impl Folder {
    fn new() -> Self {
        Folder {
            scratch: vec![0; scratch_words_for(tile_words_for(BATCH), 2).expect("small")],
            validity: vec![0; BATCH.div_ceil(64)],
        }
    }

    fn run(&mut self, p: &Program, planes: &Planes<'_>) -> Value {
        let mut s = Scratch::over(
            &mut self.scratch,
            tile_words_for(planes.n_rows),
            p.scratch_slots as usize,
        )
        .expect("scratch");
        execute_into(p, planes, &Foreign::NONE, &mut s, Out::None).expect("runs")
    }

    /// Arm A: `x < t` (gated by validity when `x` has NULLs), then COUNT and
    /// SUM(v), over the batch's own buffers.
    fn predicate(&mut self, b: &RecordBatch, t: i32) -> (u64, i64) {
        let n = b.num_rows();
        let x = b
            .column_by_name("x")
            .expect("x")
            .as_primitive::<Int32Type>();
        let v = b
            .column_by_name("v")
            .expect("v")
            .as_primitive::<Int32Type>();
        let lanes = [LaneRef::I32(x.values()), LaneRef::I32(v.values())];
        let gated = x.null_count() > 0;
        if gated {
            let words = n.div_ceil(64);
            self.validity[..words].iter_mut().for_each(|w| *w = 0);
            let nb = x.nulls().expect("nulls");
            for i in 0..n {
                if nb.is_valid(i) {
                    self.validity[i / 64] |= 1 << (i % 64);
                }
            }
        }
        let validity = std::mem::take(&mut self.validity);
        let masks: [&[u64]; 1] = [&validity[..n.div_ceil(64)]];
        let planes = Planes {
            n_rows: n,
            masks: if gated { &masks } else { &[] },
            lanes: &lanes,
        };
        let pred = MaskOp::Pred {
            pred: Pred::LtI32 { lane: 0, t },
            under: gated.then_some(Operand::Plane(0)),
            dst: 0,
        };
        let m = Operand::Scratch(0);
        let c = match self.run(
            &Program::new(vec![pred], Terminal::Count { mask: m }),
            &planes,
        ) {
            Value::Count(c) => c as u64,
            other => panic!("{other:?}"),
        };
        let s = match self.run(
            &Program::new(vec![pred], Terminal::MaskedSumI32 { mask: m, lane: 1 }),
            &planes,
        ) {
            Value::SumI64(s) => s,
            other => panic!("{other:?}"),
        };
        self.validity = validity;
        (c, s)
    }

    /// Arms C and D: every returned row is a survivor, so COUNT is the row
    /// count and SUM(v) folds `v` under an all-rows range (no lane read for
    /// the mask).
    fn survivors(&mut self, b: &RecordBatch) -> (u64, i64) {
        let n = b.num_rows();
        let v = b
            .column_by_name("v")
            .expect("v")
            .as_primitive::<Int32Type>();
        let lanes = [LaneRef::I32(v.values())];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::Range {
                    lo: 0,
                    hi: n as u32,
                },
                under: None,
                dst: 0,
            }],
            Terminal::MaskedSumI32 {
                mask: Operand::Scratch(0),
                lane: 0,
            },
        );
        match self.run(&p, &planes) {
            Value::SumI64(s) => (n as u64, s),
            other => panic!("{other:?}"),
        }
    }
}

// ───────────────────────── arms ─────────────────────────

#[derive(Default, Clone, Copy)]
struct Io {
    bytes: usize,
    iops: usize,
}

#[derive(Clone, Copy)]
enum Arm {
    A,
    C,
    Dz,
    Db,
    N,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::A => "A  full scan + fold",
            Arm::C => "C  row-range prefilter",
            Arm::Dz => "Dz zone map filter",
            Arm::Db => "Db btree filter",
            Arm::N => "N  btree COUNT, no columns",
        }
    }
}

struct Run {
    count: u64,
    sum: Option<i64>,
    rows_read: u64,
    io: Io,
    ms: f64,
    /// Time to build the row-address mask (arm C only).
    mask_ms: f64,
}

/// The row addresses holding `x < t` on the sorted fixture: rows `0..k`,
/// split across fragments of `FRAG_ROWS` (fragment ids 0, 1, … in write
/// order, checked at setup).
fn sorted_prefix_mask(k: usize) -> RowAddrMask {
    let mut m = RowAddrTreeMap::new();
    let mut row = 0usize;
    let mut frag = 0u64;
    while row < k {
        let take = (k - row).min(FRAG_ROWS);
        m.insert_range((frag << 32)..((frag << 32) + take as u64));
        row += take;
        frag += 1;
    }
    RowAddrMask::from_allowed(m)
}

async fn run(ds: &Dataset, arm: Arm, t: i32, k_sorted: Option<usize>, f: &mut Folder) -> Run {
    let io = Arc::new((AtomicUsize::new(0), AtomicUsize::new(0)));
    let sink = Arc::clone(&io);
    let start = Instant::now();
    let mut sc = ds.scan();
    sc.batch_size(BATCH);
    sc.scan_stats_callback(Arc::new(move |c| {
        sink.0.fetch_add(c.bytes_read, Ordering::Relaxed);
        sink.1.fetch_add(c.iops, Ordering::Relaxed);
    }));
    let mut mask_ms = 0.0;
    match arm {
        Arm::A => {
            sc.project(&["x", "v"]).expect("project");
        }
        Arm::C => {
            let tm = Instant::now();
            let mask = sorted_prefix_mask(k_sorted.expect("sorted only"));
            mask_ms = tm.elapsed().as_secs_f64() * 1e3;
            sc.with_row_addr_prefilter(mask);
            sc.project(&["v"]).expect("project");
        }
        Arm::Dz => {
            sc.filter(&format!("xz < {t}")).expect("filter");
            sc.project(&["v"]).expect("project");
        }
        Arm::Db => {
            sc.filter(&format!("xb < {t}")).expect("filter");
            sc.project(&["v"]).expect("project");
        }
        Arm::N => {
            sc.filter(&format!("xb < {t}")).expect("filter");
            sc.project::<&str>(&[]).expect("project");
        }
    }
    let mut stream = sc.try_into_stream().await.expect("stream");
    let (mut count, mut sum, mut rows) = (0u64, 0i64, 0u64);
    while let Some(b) = stream.try_next().await.expect("batch") {
        rows += b.num_rows() as u64;
        let (c, s) = match arm {
            Arm::A => f.predicate(&b, t),
            Arm::N => (b.num_rows() as u64, 0),
            _ => f.survivors(&b),
        };
        count += c;
        sum += s;
    }
    drop(stream);
    let ms = start.elapsed().as_secs_f64() * 1e3;
    Run {
        count,
        sum: (!matches!(arm, Arm::N)).then_some(sum),
        rows_read: rows,
        io: Io {
            bytes: io.0.load(Ordering::Relaxed),
            iops: io.1.load(Ordering::Relaxed),
        },
        ms,
        mask_ms,
    }
}

fn drop_page_cache() -> bool {
    std::fs::write("/proc/sys/vm/drop_caches", "3").is_ok()
}

// ───────────────────────── main ─────────────────────────

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let n: usize = std::env::var("PROBE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20);
    let sels = [0.5, 0.1, 0.01, 0.001, 0.0001];
    let dir = tempfile::tempdir().expect("tempdir");
    let can_drop = drop_page_cache();
    println!(
        "n = {n}, fragments of {FRAG_ROWS}, batch {BATCH}; OS page-cache drop {}",
        if can_drop {
            "WORKS (cold = dropped)"
        } else {
            "NOT PERMITTED (cold = fresh Dataset only)"
        }
    );

    let cases = [
        (Layout::Sorted, false),
        (Layout::Sorted, true),
        (Layout::Partial, false),
        (Layout::Periodic, false),
        (Layout::Shuffled, false),
        (Layout::Shuffled, true),
    ];
    let mut f = Folder::new();
    for (layout, nulls) in cases {
        let d = data(n, layout, nulls);
        let path = dir
            .path()
            .join(format!("{layout:?}-{nulls}"))
            .to_string_lossy()
            .to_string();
        let tw = Instant::now();
        let (_, zone_ms, btree_ms) = write(&d, &path).await;
        let write_ms = tw.elapsed().as_secs_f64() * 1e3 - zone_ms - btree_ms;
        let ds = Dataset::open(&path).await.expect("open");

        // The fragment ids `sorted_prefix_mask` assumes.
        let ids: Vec<u64> = ds.get_fragments().iter().map(|fr| fr.id() as u64).collect();
        assert_eq!(
            ids,
            (0..ids.len() as u64).collect::<Vec<_>>(),
            "fragment ids"
        );

        println!(
            "\n== {layout:?}, nulls {nulls}: write {write_ms:.0} ms, zone-map build {zone_ms:.0} ms, btree build {btree_ms:.0} ms =="
        );
        println!(
            "  {:<7} {:<28} {:>9} {:>11} {:>7} {:>10} {:>10} {:>9}",
            "sel", "arm", "rows", "bytes", "iops", "cold ms", "warm ms", "mask ms"
        );
        for sel in sels {
            let t = (sel * f64::from(SCALE)) as i32;
            let (want_c, want_s) = oracle(&d, t);
            let k_sorted =
                (layout == Layout::Sorted && !nulls).then(|| d.x.partition_point(|&x| x < t));
            for arm in [Arm::A, Arm::C, Arm::Dz, Arm::Db, Arm::N] {
                if matches!(arm, Arm::C) && k_sorted.is_none() {
                    continue;
                }
                // Cold: a fresh Dataset (empty index cache), page cache dropped
                // when permitted.
                drop_page_cache();
                let cold_ds = Dataset::open(&path).await.expect("open");
                let cold = run(&cold_ds, arm, t, k_sorted, &mut f).await;
                // Warm: best of three on one handle.
                let mut warm = run(&ds, arm, t, k_sorted, &mut f).await;
                for _ in 0..2 {
                    let w = run(&ds, arm, t, k_sorted, &mut f).await;
                    if w.ms < warm.ms {
                        warm = w;
                    }
                }
                for r in [&cold, &warm] {
                    assert_eq!(
                        r.count,
                        want_c,
                        "{layout:?} {nulls} {sel} {}: count",
                        arm.name()
                    );
                    if let Some(s) = r.sum {
                        assert_eq!(s, want_s, "{layout:?} {nulls} {sel} {}: sum", arm.name());
                    }
                }
                println!(
                    "  {:<7} {:<28} {:>9} {:>11} {:>7} {:>10.2} {:>10.2} {:>9.3}",
                    sel,
                    arm.name(),
                    warm.rows_read,
                    warm.io.bytes,
                    warm.io.iops,
                    cold.ms,
                    warm.ms,
                    warm.mask_ms
                );
            }
        }
    }

    deletion_aware_count(n, dir.path()).await;
}

/// The COUNT-only arm against deletions: does an exact index selection give a
/// deletion-aware cardinality, and what does it read?
async fn deletion_aware_count(n: usize, root: &std::path::Path) {
    let d = data(n, Layout::Sorted, false);
    let path = root.join("deleted").to_string_lossy().to_string();
    let (mut ds, _, _) = write(&d, &path).await;
    // Delete every 10th row by `v`-independent position: rows whose x is a
    // multiple of 10 in the sorted fixture.
    ds.delete("x % 10 = 0").await.expect("delete");
    let ds = Dataset::open(&path).await.expect("open");
    println!("\n== COUNT after deleting rows with x % 10 = 0 (sorted, btree) ==");
    let mut f = Folder::new();
    for sel in [0.5, 0.01, 0.0001] {
        let t = (sel * f64::from(SCALE)) as i32;
        let want = d.x.iter().filter(|&&x| x < t && x % 10 != 0).count() as u64;
        let r = run(&ds, Arm::N, t, None, &mut f).await;
        assert_eq!(r.count, want, "deletion-aware count at {sel}");
        let full = run(&ds, Arm::Db, t, None, &mut f).await;
        assert_eq!(full.count, want);
        println!(
            "  sel {sel:<7} COUNT {:>9}  N bytes {:>9} iops {:>4} {:>8.2} ms | Db bytes {:>9} {:>8.2} ms",
            r.count, r.io.bytes, r.io.iops, r.ms, full.io.bytes, full.ms
        );
    }
}
