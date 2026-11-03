//! LAB. One integration test of the chain
//!
//! ```text
//! ClassView -> WideFieldMask (demanded property positions)
//!   -> WHERE (logical predicate) / DR selection (identities)
//!   -> Lance row-address mask (physical row eligibility)
//!   -> Scanner::with_row_addr_prefilter -> Scanner::project (resolved columns only)
//!   -> fold
//! ```
//!
//! Run in release from this crate's directory:
//!
//! ```text
//! cargo run --release --example where_lance_arm -- <OUT> [--n 1000000] [--reps 5]
//! ```
//!
//! Four masks, kept distinct in type, name and output:
//! - `demanded: WideFieldMask`   which PROPERTIES (logical positions of the class);
//! - `where_bits: Vec<u64>`      which ROWS satisfy the logical predicate (arm A, B);
//! - `dr_ids: Vec<u32>`          the DR-side identities (arm C; given, not derived);
//! - `RowAddrMask`               which PHYSICAL rows Lance may read.
//!
//! Arms (every row labelled "Lance-backed"):
//! - A FULL: one scan of where-columns + resolved demanded columns, WHERE in Rust.
//! - B MASKED PREFILTER: scan the where-columns with `_rowaddr`, WHERE -> row list,
//!   build `RowAddrMask` (MASK INGRESS: this COPIES the addresses into Lance's
//!   `RowAddrTreeMap`, it is not zero-copy), re-scan with `with_row_addr_prefilter`
//!   projecting the resolved demanded columns only. ALLOW always; BLOCK as well at 50%.
//! - C ATTESTED ADDRESSED: `SealedFacetLane::attest_sorted` once per version (cold),
//!   ids -> lane ranges with `seek_from`, `addressed::plan_read` under `EXACT` and
//!   `DEFAULT`, a range `RowAddrMask`, prefilter scan, fold.
//!
//! WHAT IS CLAIMED: only what the printed table and the oracle checks show on this
//! fixture, on this machine, in this run. The answers of all arms equal a scalar
//! oracle over the generator formula and a full-projection read.
//!
//! WHAT IS NOT CLAIMED: no speed-up in general; no zero-copy for the mask ingress;
//! no DR -> Lance coordinate authority (arm C's ids are GIVEN); no claim that the
//! ClassView resolution covers a production ontology (a 16-field lab class).
//!
//! OWNER GATES, both open:
//! - S-INGRESS (#1788): the mask ingress copies; which coordinate space is the
//!   authority for a row address is an operator decision, not made here.
//! - D4 / the joint coordinate witness: no witness binds a DR Frame to a Lance
//!   dataset version (see `d4_boundary`, printed as BLOCKED_D4). Nothing here
//!   composes DR ordinals with `_rowid` in a stored table.
//!
//! No-ms rule: a warm in-fold value >= 1000 us (WHERE eval, ClassView resolution,
//! DR lane lookup, fold) prints FAILED-GATE (reported). Lance I/O, mask ingress and
//! cold phases print COLD/IO (reported, not gated).
//!
//! Drafted without being compiled; the orchestrator compiles centrally.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use crepe_plan_probe::addressed::{
    AddressedRelation, Read, ReadPolicy, check_fresh, key, leading, plan_read, range_tree, ranges,
};
use crepe_plan_probe::logic::RelId;
use lance::dataset::scanner::{ExecutionStatsCallback, RowAddrMask, RowAddrTreeMap};
use lance::dataset::{Dataset, WriteParams};
use lance_graph_contract::class_view::{ClassId, ClassView, FieldMask, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::ordered_lane::{OrderedLaneWitness, SealedFacetLane};
use mork_pathmap_probe::join::for_each_set;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const NPROPS: usize = 16;
const CLASS: ClassId = 1;
const IRI_KEY: &str = "predicate_iri";

// ---------------------------------------------------------------- generator

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The value of property `j` on row `id`; the oracle's only source of truth.
fn gen_val(id: u32, j: usize) -> i32 {
    let x = (u64::from(id) * 2_654_435_761 + (j as u64) * 40_503) % 2_147_483_648;
    ((x * 1_103_515_245 % 2_147_483_648) % 1000) as i32
}

// ------------------------------------------------------------------ fixture

/// Physical ordinal (0-based, after `id`) of logical property `j`: a permutation.
fn phys_ordinal(j: usize) -> usize {
    (j * 5 + 3) % NPROPS
}
/// Logical property stored at physical ordinal `ord` (inverse: 5 * 13 = 65 = 1 mod 16).
fn logical_at(ord: usize) -> usize {
    ((ord + 13) * 13) % NPROPS
}
/// Physical column name: deliberately unlike `p{j}` and "Property {j}".
fn phys_name(ord: usize) -> String {
    format!("m{:03}", 417 + ord * 37)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Permuted,
    /// Permuted plus `p_dup`, a second column carrying the iri of `p15`.
    Ambiguous,
}

fn fixture_schema(layout: Layout) -> Arc<Schema> {
    let mut fields = vec![Field::new("id", DataType::UInt32, false)];
    for ord in 0..NPROPS {
        let j = logical_at(ord);
        let md = HashMap::from([(IRI_KEY.to_string(), format!("lab:prop/p{j}"))]);
        fields.push(Field::new(phys_name(ord), DataType::Int32, false).with_metadata(md));
    }
    if layout == Layout::Ambiguous {
        let md = HashMap::from([(IRI_KEY.to_string(), "lab:prop/p15".to_string())]);
        fields.push(Field::new("p_dup", DataType::Int32, false).with_metadata(md));
    }
    Arc::new(Schema::new(fields))
}

fn make_batch(layout: Layout, ids: std::ops::Range<u32>) -> RecordBatch {
    let schema = fixture_schema(layout);
    let mut cols: Vec<arrow_array::ArrayRef> =
        vec![Arc::new(UInt32Array::from_iter_values(ids.clone()))];
    for ord in 0..NPROPS {
        let j = logical_at(ord);
        cols.push(Arc::new(Int32Array::from_iter_values(
            ids.clone().map(|id| gen_val(id, j)),
        )));
    }
    if layout == Layout::Ambiguous {
        cols.push(Arc::new(Int32Array::from_iter_values(
            ids.clone().map(|id| gen_val(id, 15)),
        )));
    }
    RecordBatch::try_new(schema, cols).expect("fixture batch")
}

fn reader(
    layout: Layout,
    range: std::ops::Range<u32>,
) -> RecordBatchIterator<impl Iterator<Item = Result<RecordBatch, arrow_schema::ArrowError>>> {
    const CHUNK: u32 = 1_000_000;
    let (lo, hi) = (range.start, range.end);
    let chunks = (hi - lo).div_ceil(CHUNK);
    let it = (0..chunks).map(move |c| {
        let a = lo + c * CHUNK;
        let b = (a + CHUNK).min(hi);
        Ok(make_batch(layout, a..b))
    });
    RecordBatchIterator::new(it, fixture_schema(layout))
}

/// Written in ascending id order as ONE fragment (`max_rows_per_file` lifted).
///
/// The limit must fit in `u32`: Lance 13 compares it as
/// `params.max_rows_per_file as u32` (`dataset/write.rs:1077`), so `1 << 40`
/// truncates to 0 and closes a file after EVERY batch. At 1M rows the reader
/// emitted one batch and hid that; at 10M it produced many fragments and the
/// attest guard refused the table as `NotContiguous`.
async fn write_fixture(uri: &str, layout: Layout, n: u32) -> Result<Dataset, String> {
    let params = WriteParams {
        max_rows_per_file: u32::MAX as usize,
        ..Default::default()
    };
    Dataset::write(reader(layout, 0..n), uri, Some(params))
        .await
        .map_err(|e| e.to_string())
}

// ----------------------------------------------------------------- ClassView

struct LabClass {
    fields: Vec<FieldRef>,
}
impl LabClass {
    fn with_fields(n: usize) -> Self {
        Self {
            fields: (0..n)
                .map(|j| FieldRef::new(format!("lab:prop/p{j}"), format!("Property {j}")))
                .collect(),
        }
    }
}
impl ClassView for LabClass {
    fn fields(&self, _class: ClassId) -> &[FieldRef] {
        &self.fields
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Refusal {
    AbsentField {
        pos: u8,
    },
    Unauthorized {
        pos: u8,
    },
    UnknownProperty {
        pos: u8,
        iri: String,
    },
    AmbiguousProperty {
        pos: u8,
        iri: String,
        columns: Vec<String>,
    },
    StaleSchema {
        resolved: u64,
        current: u64,
    },
}

/// The only path from demanded positions to a column list. On any refusal there
/// is no column list at all: the caller cannot widen to "all columns".
struct Resolution {
    version: u64,
    positions: Vec<u8>,
    columns: Vec<String>,
}

impl Resolution {
    fn check_current(&self, latest: u64) -> Result<(), Refusal> {
        if latest != self.version {
            return Err(Refusal::StaleSchema {
                resolved: self.version,
                current: latest,
            });
        }
        Ok(())
    }
}

/// position -> `FieldRef.predicate_iri` -> exactly one schema field whose
/// metadata `predicate_iri` matches. `at` is the handle pinned at the attested
/// version; its schema is the one read.
fn resolve(
    view: &dyn ClassView,
    class: ClassId,
    demanded: &WideFieldMask,
    presence: FieldMask,
    permit: &WideFieldMask,
    at: &Dataset,
) -> Result<Resolution, Refusal> {
    let fields = view.fields(class);
    let schema = at.schema();
    let mut out = Resolution {
        version: at.version().version,
        positions: Vec::new(),
        columns: Vec::new(),
    };
    for p in 0..=255u8 {
        if !demanded.has(p) {
            continue;
        }
        if usize::from(p) >= fields.len() || !presence.has(p) {
            return Err(Refusal::AbsentField { pos: p });
        }
        if !permit.has(p) {
            return Err(Refusal::Unauthorized { pos: p });
        }
        let iri = &fields[usize::from(p)].predicate_iri;
        let hits: Vec<&str> = schema
            .fields
            .iter()
            .filter(|f| f.metadata.get(IRI_KEY).map(String::as_str) == Some(iri.as_str()))
            .map(|f| f.name.as_str())
            .collect();
        match hits.len() {
            0 => {
                return Err(Refusal::UnknownProperty {
                    pos: p,
                    iri: iri.clone(),
                });
            }
            1 => {
                out.positions.push(p);
                out.columns.push(hits[0].to_string());
            }
            _ => {
                return Err(Refusal::AmbiguousProperty {
                    pos: p,
                    iri: iri.clone(),
                    columns: hits.iter().map(|s| s.to_string()).collect(),
                });
            }
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------- scanning

#[derive(Debug, Clone, Copy, Default)]
struct Io {
    rows_scanned: usize,
    bytes_read: usize,
}

/// Same callback pattern as `lance_store.rs:162-175`.
fn stats_sink() -> (Arc<Mutex<Io>>, ExecutionStatsCallback) {
    let stats: Arc<Mutex<Io>> = Arc::default();
    let sink = stats.clone();
    let cb: ExecutionStatsCallback = Arc::new(move |c| {
        let mut io = sink.lock().expect("stats lock");
        io.bytes_read = c.bytes_read;
        io.rows_scanned = c.all_counts.get("rows_scanned").copied().unwrap_or(0);
    });
    (stats, cb)
}

async fn scan(
    ds: &Dataset,
    cols: &[String],
    with_addr: bool,
    mask: Option<RowAddrMask>,
) -> Result<(RecordBatch, Io), String> {
    let (stats, cb) = stats_sink();
    let mut s = ds.scan();
    s.project(cols).map_err(|e| e.to_string())?;
    if with_addr {
        s.with_row_address();
    }
    if let Some(m) = mask {
        s.with_row_addr_prefilter(m);
    }
    s.scan_stats_callback(cb);
    let b = s.try_into_batch().await.map_err(|e| e.to_string())?;
    let io = *stats.lock().expect("stats lock");
    Ok((b, io))
}

fn i32col<'a>(b: &'a RecordBatch, name: &str) -> &'a [i32] {
    let a = b
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
        .unwrap_or_else(|| panic!("i32 column {name}"));
    &a.values()[..]
}
fn u32col<'a>(b: &'a RecordBatch, name: &str) -> &'a [u32] {
    let a = b
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .unwrap_or_else(|| panic!("u32 column {name}"));
    &a.values()[..]
}
fn addrcol(b: &RecordBatch) -> &[u64] {
    let a = b
        .column_by_name("_rowaddr")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .expect("_rowaddr");
    &a.values()[..]
}

fn push_unique(v: &mut Vec<String>, s: &str) {
    if !v.iter().any(|x| x == s) {
        v.push(s.to_string());
    }
}

// -------------------------------------------------------------------- WHERE

#[derive(Clone, Copy)]
enum Where {
    /// `p0 < t AND p1 < u` (scattered).
    Scatter { t: i32, u: i32 },
    /// `id in [lo, hi)` (clustered).
    Range { lo: u32, hi: u32 },
}
impl Where {
    fn positions(&self) -> WideFieldMask {
        match self {
            Where::Scatter { .. } => WideFieldMask::from_positions(&[0, 1]),
            Where::Range { .. } => WideFieldMask::EMPTY,
        }
    }
}

fn eval_scatter(p0: &[i32], p1: &[i32], t: i32, u: i32) -> Vec<u64> {
    let mut out = vec![0u64; p0.len().div_ceil(64)];
    for (wi, (a, b)) in p0.chunks(64).zip(p1.chunks(64)).enumerate() {
        let mut word = 0u64;
        for (bit, (&x, &y)) in a.iter().zip(b).enumerate() {
            word |= u64::from((x < t) & (y < u)) << bit;
        }
        out[wi] = word;
    }
    out
}
fn eval_range(id: &[u32], lo: u32, hi: u32) -> Vec<u64> {
    let mut out = vec![0u64; id.len().div_ceil(64)];
    for (wi, a) in id.chunks(64).enumerate() {
        let mut word = 0u64;
        for (bit, &x) in a.iter().enumerate() {
            word |= u64::from((x >= lo) & (x < hi)) << bit;
        }
        out[wi] = word;
    }
    out
}
fn eval_where(w: &Where, b: &RecordBatch, res_w: &Resolution) -> Vec<u64> {
    match *w {
        Where::Scatter { t, u } => eval_scatter(
            i32col(b, &res_w.columns[0]),
            i32col(b, &res_w.columns[1]),
            t,
            u,
        ),
        Where::Range { lo, hi } => eval_range(u32col(b, "id"), lo, hi),
    }
}
fn where_cols(w: &Where, res_w: &Resolution) -> Vec<String> {
    match w {
        Where::Scatter { .. } => res_w.columns.clone(),
        Where::Range { .. } => vec!["id".to_string()],
    }
}

// --------------------------------------------------------------------- fold

fn fold_mask(bits: &[u64], n: usize, cols: &[&[i32]]) -> (u64, Vec<i64>) {
    let count = bits.iter().map(|w| u64::from(w.count_ones())).sum();
    let sums = cols
        .iter()
        .map(|c| {
            let mut s = 0i64;
            for_each_set(bits, n, |i| s += i64::from(c[i]));
            s
        })
        .collect();
    (count, sums)
}
fn fold_all(cols: &[&[i32]]) -> Vec<i64> {
    cols.iter()
        .map(|c| c.iter().map(|&x| i64::from(x)).sum())
        .collect()
}

// ------------------------------------------------------------------- oracle

struct Oracle {
    bits: Vec<u64>,
    count: u64,
    sums: Vec<i64>,
    ids: Vec<u32>,
}

/// Scalar loop over the generator formula; never reads Lance.
fn oracle(n: u32, w: &Where, dem: &[u8]) -> Oracle {
    let mut bits = vec![0u64; (n as usize).div_ceil(64)];
    let mut count = 0u64;
    let mut sums = vec![0i64; dem.len()];
    for id in 0..n {
        let sel = match *w {
            Where::Scatter { t, u } => gen_val(id, 0) < t && gen_val(id, 1) < u,
            Where::Range { lo, hi } => id >= lo && id < hi,
        };
        if sel {
            bits[(id / 64) as usize] |= 1u64 << (id % 64);
            count += 1;
            for (s, &j) in sums.iter_mut().zip(dem) {
                *s += i64::from(gen_val(id, usize::from(j)));
            }
        }
    }
    let mut ids = Vec::with_capacity(count as usize);
    for_each_set(&bits, n as usize, |i| ids.push(i as u32));
    Oracle {
        bits,
        count,
        sums,
        ids,
    }
}

/// Smallest-error `T` (with `U = T`) per target fraction, from the exact 2-D
/// histogram of (p0, p1) over all rows.
fn calibrate(n: u32, targets: &[f64]) -> Vec<(i32, f64)> {
    const W: usize = 1001;
    let mut pre = vec![0u64; W * W];
    for id in 0..n {
        let (a, b) = (gen_val(id, 0) as usize, gen_val(id, 1) as usize);
        pre[(a + 1) * W + (b + 1)] += 1;
    }
    for a in 1..W {
        for b in 1..W {
            pre[a * W + b] += pre[(a - 1) * W + b] + pre[a * W + b - 1] - pre[(a - 1) * W + b - 1];
        }
    }
    targets
        .iter()
        .map(|&f| {
            let best = (1..=1000usize)
                .min_by(|&x, &y| {
                    let ex = (pre[x * W + x] as f64 / f64::from(n) - f).abs();
                    let ey = (pre[y * W + y] as f64 / f64::from(n) - f).abs();
                    ex.partial_cmp(&ey).expect("finite")
                })
                .expect("range");
            (best as i32, pre[best * W + best] as f64 / f64::from(n))
        })
        .collect()
}

// ------------------------------------------------------------ attested lane

/// Reads the keys once (the cold cost) and attests their order, as
/// `addressed::attest` does (`addressed.rs:101-143`), over the `id` column.
async fn attest(ds: &Dataset) -> Result<(SealedFacetLane, OrderedLaneWitness), String> {
    let (b, _) = scan(ds, &["id".to_string()], true, None).await?;
    let id = u32col(&b, "id");
    let addr = addrcol(&b);
    if !(0..b.num_rows()).all(|i| addr[i] == i as u64) {
        return Err("NotContiguous: table is not one contiguous fragment".into());
    }
    // The prefilter is keyed in `_rowid` space; the lane positions are
    // `_rowaddr`. They coincide only without stable row ids. Assert it.
    let mut sc = ds.scan();
    sc.project::<&str>(&[]).map_err(|e| e.to_string())?;
    sc.with_row_id();
    sc.with_row_address();
    let rb = sc.try_into_batch().await.map_err(|e| e.to_string())?;
    let rid = rb
        .column_by_name("_rowid")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or("no _rowid column")?;
    let rad = rb
        .column_by_name("_rowaddr")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or("no _rowaddr column")?;
    if rid.values() != rad.values() {
        return Err(
            "RowIdSpace: _rowid != _rowaddr (stable row ids?); the lane cannot key the prefilter"
                .into(),
        );
    }
    let keys = id.iter().map(|&v| key(v, 0)).collect();
    let lane = SealedFacetLane::attest_sorted(keys, ds.version().version)
        .map_err(|e| format!("Witness({e:?})"))?;
    let w = lane.witness();
    Ok((lane, w))
}

/// Bits over the returned rows (by position) whose address lies in `exact`.
fn keep_by_ranges(addr: &[u64], exact: &[(u32, u32)]) -> Result<Vec<u64>, String> {
    let mut bits = vec![0u64; addr.len().div_ceil(64)];
    let mut r = 0usize;
    let mut prev = 0u64;
    for (i, &a) in addr.iter().enumerate() {
        if i > 0 && a < prev {
            return Err("returned rows are not in ascending address order".into());
        }
        prev = a;
        while r < exact.len() && u64::from(exact[r].1) <= a {
            r += 1;
        }
        if r < exact.len() && a >= u64::from(exact[r].0) {
            bits[i / 64] |= 1u64 << (i % 64);
        }
    }
    Ok(bits)
}

// --------------------------------------------------------------------- arms

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    A,
    BAllow,
    BBlock,
    CExact,
    CDefault,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::A => "A-full",
            Arm::BAllow => "B-allow",
            Arm::BBlock => "B-block",
            Arm::CExact => "C-exact",
            Arm::CDefault => "C-default",
        }
    }
}

// indices into the timing array (microseconds)
const T_WHERE: usize = 0;
const T_RESOLVE: usize = 1;
const T_ROUTE: usize = 2;
const T_INGRESS: usize = 3;
const T_LANCE: usize = 4;
const T_FOLD: usize = 5;
const T_TOTAL: usize = 6;

struct Out {
    t: [f64; 7],
    io: Io,
    count: u64,
    sums: Vec<i64>,
    mask_bytes: usize,
    read: String,
}

struct Env {
    ds: Dataset,
    /// The #1633 addressed relation: dataset uri + attested lane + witness.
    rel: AddressedRelation,
    attest_cold_us: f64,
    view: LabClass,
    presence: FieldMask,
    permit_all: WideFieldMask,
    /// Every column, every row, read once: the projection oracle.
    full: RecordBatch,
}

struct Case {
    layout: &'static str,
    frac_target: f64,
    w: Where,
    /// The DR-side identities (sorted): GIVEN, taken from the scalar oracle.
    ids: Vec<u32>,
    demanded: WideFieldMask,
}

fn us(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e6
}

async fn run_once(env: &Env, c: &Case, arm: Arm) -> Result<Out, String> {
    let t_all = Instant::now();
    let mut t = [0f64; 7];

    // ClassView resolution (where positions, then demanded positions).
    let s = Instant::now();
    let res_w = resolve(
        &env.view,
        CLASS,
        &c.w.positions(),
        env.presence,
        &env.permit_all,
        &env.ds,
    )
    .map_err(|e| format!("{e:?}"))?;
    let res_d = resolve(
        &env.view,
        CLASS,
        &c.demanded,
        env.presence,
        &env.permit_all,
        &env.ds,
    )
    .map_err(|e| format!("{e:?}"))?;
    t[T_RESOLVE] = us(s);

    // The attested version must still be the current one (an I/O call).
    let s = Instant::now();
    let latest = env
        .ds
        .latest_version_id()
        .await
        .map_err(|e| e.to_string())?;
    res_w.check_current(latest).map_err(|e| format!("{e:?}"))?;
    res_d.check_current(latest).map_err(|e| format!("{e:?}"))?;
    t[T_LANCE] += us(s);

    let mut io = Io::default();
    let add_io = |io: &mut Io, x: Io| {
        io.rows_scanned += x.rows_scanned;
        io.bytes_read += x.bytes_read;
    };
    let mut mask_bytes = 0usize;
    let read: String;
    let (count, sums);

    match arm {
        Arm::A => {
            let mut cols = where_cols(&c.w, &res_w);
            for d in &res_d.columns {
                push_unique(&mut cols, d);
            }
            let s = Instant::now();
            let (b, x) = scan(&env.ds, &cols, false, None).await?;
            t[T_LANCE] += us(s);
            add_io(&mut io, x);
            let s = Instant::now();
            let where_bits = eval_where(&c.w, &b, &res_w);
            t[T_WHERE] = us(s);
            let s = Instant::now();
            let dcols: Vec<&[i32]> = res_d.columns.iter().map(|d| i32col(&b, d)).collect();
            let (cn, sm) = fold_mask(&where_bits, b.num_rows(), &dcols);
            t[T_FOLD] = us(s);
            count = cn;
            sums = sm;
            read = "full".into();
        }
        Arm::BAllow | Arm::BBlock => {
            // Phase 1: the where-columns with their addresses.
            let wc = where_cols(&c.w, &res_w);
            let s = Instant::now();
            let (b1, x1) = scan(&env.ds, &wc, true, None).await?;
            t[T_LANCE] += us(s);
            add_io(&mut io, x1);
            let s = Instant::now();
            let where_bits = eval_where(&c.w, &b1, &res_w);
            t[T_WHERE] = us(s);

            // Phase 2a: MASK INGRESS (copies addresses into RowAddrTreeMap).
            // Timed: building the one tree Lance receives. Not timed: its
            // serialized size, which is reporting, not ingress.
            let s = Instant::now();
            let addr = addrcol(&b1);
            let (tree, allow) = if arm == Arm::BAllow {
                let mut list = Vec::new();
                for_each_set(&where_bits, b1.num_rows(), |i| list.push(addr[i]));
                (RowAddrTreeMap::from_iter(list), true)
            } else {
                let mut inv: Vec<u64> = where_bits.iter().map(|w| !w).collect();
                if let Some(last) = inv.last_mut() {
                    let rem = b1.num_rows() % 64;
                    if rem != 0 {
                        *last &= (1u64 << rem) - 1;
                    }
                }
                let mut list = Vec::new();
                for_each_set(&inv, b1.num_rows(), |i| list.push(addr[i]));
                (RowAddrTreeMap::from_iter(list), false)
            };
            t[T_INGRESS] = us(s);
            mask_bytes = tree.serialized_size();
            let mask = if allow {
                read = "allow".into();
                RowAddrMask::from_allowed(tree)
            } else {
                read = "block".into();
                RowAddrMask::from_block(tree)
            };

            // Phase 2b: prefilter scan, resolved demanded columns only.
            let s = Instant::now();
            let (b2, x2) = scan(&env.ds, &res_d.columns, false, Some(mask)).await?;
            t[T_LANCE] += us(s);
            add_io(&mut io, x2);
            let s = Instant::now();
            let dcols: Vec<&[i32]> = res_d.columns.iter().map(|d| i32col(&b2, d)).collect();
            sums = fold_all(&dcols);
            count = b2.num_rows() as u64;
            t[T_FOLD] = us(s);
        }
        Arm::CExact | Arm::CDefault => {
            let policy = if arm == Arm::CExact {
                ReadPolicy::EXACT
            } else {
                ReadPolicy::DEFAULT
            };
            // The ids are GIVEN (DR -> Lance witness BLOCKED_D4): the oracle's
            // identities, sorted, standing in for the DR selection.
            let ids = &c.ids;
            check_fresh(&env.rel, env.ds.uri(), latest).map_err(|e| format!("{e:?}"))?;
            let s = Instant::now();
            let (exact, _rows) = ranges(&env.rel, ids).map_err(|e| format!("{e:?}"))?;
            let (how, planned) = plan_read(&exact, env.rel.lane.n_rows(), policy);
            t[T_ROUTE] = us(s);

            // Timed: building the ONE tree Lance receives (`range_tree`, which
            // `addressed::range_mask` also uses). Not timed: its serialized
            // size, which is reporting, not ingress. Same split as the B arms.
            let s = Instant::now();
            let tree = match how {
                Read::Full => None,
                Read::Ranges | Read::Coalesced => Some(range_tree(&planned)),
            };
            t[T_INGRESS] = us(s);
            let mask = tree.map(|tree| {
                mask_bytes = tree.serialized_size();
                RowAddrMask::from_allowed(tree)
            });
            read = format!("{how:?}/{}r", planned.len());

            let need_addr = how != Read::Ranges;
            let s = Instant::now();
            let (b, x) = scan(&env.ds, &res_d.columns, need_addr, mask).await?;
            t[T_LANCE] += us(s);
            add_io(&mut io, x);

            let s = Instant::now();
            let dcols: Vec<&[i32]> = res_d.columns.iter().map(|d| i32col(&b, d)).collect();
            if need_addr {
                let keep = keep_by_ranges(addrcol(&b), &exact)?;
                let (cn, sm) = fold_mask(&keep, b.num_rows(), &dcols);
                count = cn;
                sums = sm;
            } else {
                sums = fold_all(&dcols);
                count = b.num_rows() as u64;
            }
            t[T_FOLD] = us(s);
        }
    }
    t[T_TOTAL] = us(t_all);
    Ok(Out {
        t,
        io,
        count,
        sums,
        mask_bytes,
        read,
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

fn gate(t: &[f64; 7]) -> String {
    let mut bad = Vec::new();
    for (i, nm) in [
        (T_WHERE, "where"),
        (T_RESOLVE, "resolve"),
        (T_ROUTE, "route"),
        (T_FOLD, "fold"),
    ] {
        if t[i] >= 1000.0 {
            bad.push(nm);
        }
    }
    if bad.is_empty() {
        "in-fold<1000us; lance/ingress/cold=COLD/IO(reported)".into()
    } else {
        format!(
            "FAILED-GATE(reported):{}; lance/ingress/cold=COLD/IO(reported)",
            bad.join(",")
        )
    }
}

struct Row {
    arm: &'static str,
    layout: &'static str,
    frac_target: f64,
    frac_actual: f64,
    k: usize,
    t: [f64; 7],
    cold_total: f64,
    io: Io,
    selected: u64,
    bytes_vs_a: f64,
    mask_bytes: usize,
    read: String,
    gate: String,
}

// ------------------------------------------------------ refusals & traps

fn check(name: &str, ok: bool, fails: &mut usize) {
    println!("  [{}] {name}", if ok { "ok  " } else { "FAIL" });
    if !ok {
        *fails += 1;
    }
}

/// Every refusal paired with a success control; no refusal yields a column list.
async fn refusal_tests(
    small: &Dataset,
    amb: &Dataset,
    small_uri: &str,
    fails: &mut usize,
) -> Result<(), String> {
    println!("\n== ClassView refusals (each with a paired success control) ==");
    let view = LabClass::with_fields(NPROPS);
    let presence = (0..NPROPS as u8).fold(FieldMask::EMPTY, |m, p| m.with(p));
    let all = WideFieldMask::full_for(NPROPS);
    let dem = WideFieldMask::from_positions(&[2, 15]);
    let width = small.schema().fields.len();

    // Control: the success path projects exactly the demanded columns, not all.
    let ok = resolve(&view, CLASS, &dem, presence, &all, small);
    let ok_cols = ok.as_ref().map(|r| r.columns.clone()).unwrap_or_default();
    check(
        "control: resolve -> exactly the 2 demanded columns (schema has more)",
        ok.is_ok() && ok_cols.len() == 2 && ok_cols.len() < width,
        fails,
    );
    check(
        "control: resolved names equal the fixture truth (not label, not ordinal)",
        ok_cols == vec![phys_name(phys_ordinal(2)), phys_name(phys_ordinal(15))],
        fails,
    );

    // AbsentField: position not in the class's presence mask.
    let no5 = (0..NPROPS as u8)
        .filter(|&p| p != 5)
        .fold(FieldMask::EMPTY, |m, p| m.with(p));
    let r = resolve(
        &view,
        CLASS,
        &WideFieldMask::from_positions(&[5]),
        no5,
        &all,
        small,
    );
    check(
        "AbsentField refused, no column list",
        matches!(r, Err(Refusal::AbsentField { pos: 5 })),
        fails,
    );
    let r = resolve(
        &view,
        CLASS,
        &WideFieldMask::from_positions(&[5]),
        presence,
        &all,
        small,
    );
    check(
        "AbsentField control: present position resolves",
        r.is_ok(),
        fails,
    );

    // Unauthorized: position not in the caller's permit mask.
    let permit_no7 = WideFieldMask::from_positions(&[0, 1, 2, 3, 4, 5, 6, 8, 9]);
    let r = resolve(
        &view,
        CLASS,
        &WideFieldMask::from_positions(&[7]),
        presence,
        &permit_no7,
        small,
    );
    check(
        "Unauthorized refused, no column list",
        matches!(r, Err(Refusal::Unauthorized { pos: 7 })),
        fails,
    );
    let r = resolve(
        &view,
        CLASS,
        &WideFieldMask::from_positions(&[6]),
        presence,
        &permit_no7,
        small,
    );
    check(
        "Unauthorized control: permitted position resolves",
        r.is_ok(),
        fails,
    );

    // UnknownProperty: the class names p16, no column carries its iri.
    let view17 = LabClass::with_fields(NPROPS + 1);
    let presence17 = (0..=NPROPS as u8).fold(FieldMask::EMPTY, |m, p| m.with(p));
    let all17 = WideFieldMask::full_for(NPROPS + 1);
    let r = resolve(
        &view17,
        CLASS,
        &WideFieldMask::from_positions(&[16]),
        presence17,
        &all17,
        small,
    );
    check(
        "UnknownProperty refused, no column list",
        matches!(&r, Err(Refusal::UnknownProperty { pos: 16, .. })),
        fails,
    );
    let r = resolve(
        &view17,
        CLASS,
        &WideFieldMask::from_positions(&[15]),
        presence17,
        &all17,
        small,
    );
    check(
        "UnknownProperty control: known position resolves",
        r.is_ok(),
        fails,
    );

    // AmbiguousProperty: the second dataset carries p_dup with p15's iri.
    let r = resolve(
        &view,
        CLASS,
        &WideFieldMask::from_positions(&[15]),
        presence,
        &all,
        amb,
    );
    check(
        "AmbiguousProperty refused (2 columns), no column list",
        matches!(&r, Err(Refusal::AmbiguousProperty { columns, .. }) if columns.len() == 2),
        fails,
    );
    let r = resolve(
        &view,
        CLASS,
        &WideFieldMask::from_positions(&[14]),
        presence,
        &all,
        amb,
    );
    check(
        "AmbiguousProperty control: unambiguous position resolves",
        r.is_ok(),
        fails,
    );

    // The trap check: ordinal / label mapping gives a DIFFERENT answer.
    println!("\n== Mapping traps (the test would catch a positional or label mapping) ==");
    let names: Vec<String> = small
        .schema()
        .fields
        .iter()
        .map(|f| f.name.clone())
        .collect();
    let (b, _) = scan(small, &names, false, None).await?;
    let n_small = b.num_rows() as u32;
    let mut ordinal_wrong = 0usize;
    let mut label_misses = 0usize;
    for j in 0..NPROPS {
        let truth: i64 = (0..n_small).map(|id| i64::from(gen_val(id, j))).sum();
        // position == physical column ordinal (after id): the forbidden shortcut.
        let by_ordinal: i64 = i32col(&b, &names[1 + j])
            .iter()
            .map(|&x| i64::from(x))
            .sum();
        if by_ordinal != truth {
            ordinal_wrong += 1;
        }
        // label == column name: the other forbidden shortcut.
        let label = view.field_label(CLASS, j as u8).expect("label");
        if !names.iter().any(|nm| nm == label) {
            label_misses += 1;
        }
        // the resolver's answer
        let r = resolve(
            &view,
            CLASS,
            &WideFieldMask::from_positions(&[j as u8]),
            presence,
            &all,
            small,
        )
        .map_err(|e| format!("{e:?}"))?;
        let via: i64 = i32col(&b, &r.columns[0])
            .iter()
            .map(|&x| i64::from(x))
            .sum();
        if via != truth {
            return Err(format!(
                "resolver answer differs from the generator at p{j}"
            ));
        }
    }
    println!(
        "  ordinal mapping wrong on {ordinal_wrong}/{NPROPS} positions; label mapping finds no column on {label_misses}/{NPROPS}"
    );
    check(
        "ordinal mapping gives a different answer (trap is live)",
        ordinal_wrong > 0,
        fails,
    );
    check(
        "label mapping finds no physical column",
        label_misses == NPROPS,
        fails,
    );

    // StaleSchema and the stale lane, on the small dataset (mutated last).
    println!("\n== Staleness (dataset appended after resolution / attest) ==");
    let res = resolve(&view, CLASS, &dem, presence, &all, small).map_err(|e| format!("{e:?}"))?;
    let (lane, w) = attest(small).await?;
    let rel = AddressedRelation {
        rel: RelId(0),
        uri: small.uri().to_string(),
        lane,
        witness: w,
    };
    let (lane, w) = (&rel.lane, &rel.witness);
    let latest = small.latest_version_id().await.map_err(|e| e.to_string())?;
    check(
        "control: schema resolution current before append",
        res.check_current(latest).is_ok(),
        fails,
    );
    check(
        "control: lane fresh before append",
        check_fresh(&rel, small.uri(), latest).is_ok(),
        fails,
    );
    check(
        "control: seek under the real witness runs",
        lane.seek_from(w, 0, &leading(3).lo_key()).is_ok(),
        fails,
    );
    let mut grown = small.clone();
    grown
        .append(reader_range_fixture(n_small, n_small + 16), None)
        .await
        .map_err(|e| e.to_string())?;
    let latest2 = grown.latest_version_id().await.map_err(|e| e.to_string())?;
    let r = res.check_current(latest2);
    check(
        "StaleSchema refused after append",
        matches!(r, Err(Refusal::StaleSchema { resolved, current }) if current == resolved + 1),
        fails,
    );
    check(
        "stale lane refused after append",
        check_fresh(&rel, grown.uri(), latest2).is_err(),
        fails,
    );
    check(
        "other dataset refused",
        check_fresh(&rel, "memory://elsewhere", latest).is_err(),
        fails,
    );
    let forged = OrderedLaneWitness::forged(latest2, lane.n_rows(), w.digest());
    check(
        "forged (newer-version) witness refused by seek_from",
        lane.seek_from(&forged, 0, &leading(3).lo_key()).is_err(),
        fails,
    );
    let _ = small_uri;
    Ok(())
}

fn reader_range_fixture(
    lo: u32,
    hi: u32,
) -> RecordBatchIterator<impl Iterator<Item = Result<RecordBatch, arrow_schema::ArrowError>>> {
    reader(Layout::Permuted, lo..hi)
}

/// D4: no type binds a DR `Frame` to a Lance dataset version. Source-backed
/// trace (P2 Lance trace, `results/.../REPORT.md`): `lance-graph-dt64` carries
/// no `OrderedLaneWitness`/`SealedFacetLane`/Lance version, and the #1788
/// single-authority decision (S-INGRESS) is still open. So arm C stops at the
/// boundary: its ids are GIVEN, never composed from a DR ordinal.
fn d4_boundary() {
    println!(
        "BLOCKED_D4: no witness binds a DR Frame to a Lance dataset version (S-INGRESS / single authority open, #1788); \
         arm C's ids are given directly; no DR ordinal is composed with _rowid"
    );
}

// --------------------------------------------------------------------- main

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: where_lance_arm <OUT> [--n N] [--reps R]");
        std::process::exit(2);
    }
    let out_dir = args[1].clone();
    let mut n: u32 = 1_000_000;
    let mut reps: usize = 5;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                n = args[i + 1].replace('_', "").parse().expect("--n");
                i += 2;
            }
            "--reps" => {
                reps = args[i + 1].parse().expect("--reps");
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    if let Err(e) = real_main(&out_dir, n, reps.max(1)).await {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}

async fn real_main(out_dir: &str, n: u32, reps: usize) -> Result<(), String> {
    println!("# where_lance_arm LAB, Lance-backed, n={n}, warm reps={reps}, seed={SEED:#x}");
    println!("# mask ingress COPIES addresses into RowAddrTreeMap (S-INGRESS open, not zero-copy)");
    d4_boundary();

    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let uri = |nm: &str| tmp.path().join(nm).to_str().expect("utf8 path").to_string();

    // Fixtures.
    let main_uri = uri("main.lance");
    let small_uri = uri("small.lance");
    let amb_uri = uri("amb.lance");
    write_fixture(&main_uri, Layout::Permuted, n).await?;
    write_fixture(&small_uri, Layout::Permuted, 4096).await?;
    write_fixture(&amb_uri, Layout::Ambiguous, 4096).await?;
    let ds = Dataset::open(&main_uri).await.map_err(|e| e.to_string())?;
    let small = Dataset::open(&small_uri).await.map_err(|e| e.to_string())?;
    let amb = Dataset::open(&amb_uri).await.map_err(|e| e.to_string())?;

    // Refusals first: they mutate only the small dataset.
    let mut fails = 0usize;
    refusal_tests(&small, &amb, &small_uri, &mut fails).await?;

    // Attest the main dataset (cold) and read the projection oracle once.
    let s = Instant::now();
    let (lane, witness) = attest(&ds).await?;
    let attest_cold_us = us(s);
    println!("\n# attest (cold, once per version): {attest_cold_us:.1} us  [COLD/IO, reported]");
    let names: Vec<String> = ds.schema().fields.iter().map(|f| f.name.clone()).collect();
    let (full, _) = scan(&ds, &names, false, None).await?;
    let presence = (0..NPROPS as u8).fold(FieldMask::EMPTY, |m, p| m.with(p));
    let uri_main = ds.uri().to_string();
    let env = Env {
        ds,
        rel: AddressedRelation {
            rel: RelId(0),
            uri: uri_main,
            lane,
            witness,
        },
        attest_cold_us,
        view: LabClass::with_fields(NPROPS),
        presence,
        permit_all: WideFieldMask::full_for(NPROPS),
        full,
    };

    // Cases.
    let fracs = [0.001, 0.01, 0.1, 0.5];
    let cal = calibrate(n, &fracs);
    let mut sm = SplitMix64(SEED);
    let ks: [(usize, &[u8]); 3] = [
        (1, &[7u8][..]),
        (4, &[0u8, 5, 10, 15][..]),
        (
            16,
            &[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15][..],
        ),
    ];
    let mut rows: Vec<Row> = Vec::new();
    println!("\n== Measurements (us, median of {reps} warm runs; cold = first run) ==");
    for (fi, &f) in fracs.iter().enumerate() {
        let (t_thr, _) = cal[fi];
        let len = ((f * f64::from(n)).round() as u32).max(1);
        let lo = (sm.next() % u64::from(n - len + 1)) as u32;
        for (layout, w) in [
            ("scattered", Where::Scatter { t: t_thr, u: t_thr }),
            ("clustered", Where::Range { lo, hi: lo + len }),
        ] {
            for &(k, dem) in &ks {
                let orc = oracle(n, &w, dem);
                if orc.count == 0 {
                    println!("skip {layout} f={f}: empty selection");
                    continue;
                }
                let case = Case {
                    layout,
                    frac_target: f,
                    w,
                    ids: orc.ids.clone(),
                    demanded: WideFieldMask::from_positions(dem),
                };
                let frac_actual = orc.count as f64 / f64::from(n);

                // Projection oracle: full read of all columns, fixture-truth names.
                let proj_sums: Vec<i64> = dem
                    .iter()
                    .map(|&j| {
                        let col = i32col(&env.full, &phys_name(phys_ordinal(usize::from(j))));
                        let mut s = 0i64;
                        for_each_set(&orc.bits, n as usize, |i| s += i64::from(col[i]));
                        s
                    })
                    .collect();
                if proj_sums != orc.sums {
                    return Err(format!(
                        "projection oracle != generator oracle ({layout} f={f} k={k})"
                    ));
                }

                let mut arms = vec![Arm::A, Arm::BAllow];
                if (f - 0.5).abs() < 1e-9 {
                    arms.push(Arm::BBlock);
                }
                arms.push(Arm::CExact);
                arms.push(Arm::CDefault);

                let mut a_bytes = 0usize;
                let mut first_line = true;
                for arm in arms {
                    let cold = run_once(&env, &case, arm).await?;
                    let mut runs: Vec<[f64; 7]> = Vec::new();
                    let mut last = cold;
                    let cold_total = last.t[T_TOTAL];
                    for _ in 0..reps {
                        last = run_once(&env, &case, arm).await?;
                        runs.push(last.t);
                    }
                    // Oracle checks on the last warm run.
                    if last.count != orc.count || last.sums != orc.sums {
                        eprintln!(
                            "MISMATCH {} {layout} f={f} k={k}: count {} vs {}, sums {:?} vs {:?}",
                            arm.name(),
                            last.count,
                            orc.count,
                            last.sums,
                            orc.sums
                        );
                        fails += 1;
                    }
                    let mut t = [0f64; 7];
                    for (idx, slot) in t.iter_mut().enumerate() {
                        *slot = median(runs.iter().map(|r| r[idx]).collect());
                    }
                    if arm == Arm::A {
                        a_bytes = last.io.bytes_read;
                    }
                    let bytes_vs_a = if a_bytes == 0 {
                        0.0
                    } else {
                        100.0 * last.io.bytes_read as f64 / a_bytes as f64
                    };
                    if first_line {
                        println!(
                            "-- {layout} target {:.1}% actual {:.3}% k={k} | demanded(WideFieldMask)={} where_bits(popcount)={} dr_ids={} (ids given, BLOCKED_D4)",
                            f * 100.0,
                            frac_actual * 100.0,
                            case.demanded.count(),
                            orc.count,
                            orc.ids.len()
                        );
                        first_line = false;
                    }
                    let g = gate(&t);
                    println!(
                        "  {:<9} Lance-backed | where {:>9.1} resolve {:>7.1} route {:>9.1} ingress {:>10.1} lance {:>10.1} fold {:>9.1} total {:>10.1} cold {:>10.1} | scanned {:>9} sel {:>8} bytes {:>10} ({:>5.1}% of A) row_addr_mask {:>9}B read {} | {}",
                        arm.name(),
                        t[T_WHERE],
                        t[T_RESOLVE],
                        t[T_ROUTE],
                        t[T_INGRESS],
                        t[T_LANCE],
                        t[T_FOLD],
                        t[T_TOTAL],
                        cold_total,
                        last.io.rows_scanned,
                        last.count,
                        last.io.bytes_read,
                        bytes_vs_a,
                        last.mask_bytes,
                        last.read,
                        g
                    );
                    rows.push(Row {
                        arm: arm.name(),
                        layout: case.layout,
                        frac_target: case.frac_target,
                        frac_actual,
                        k,
                        t,
                        cold_total,
                        io: last.io,
                        selected: last.count,
                        bytes_vs_a,
                        mask_bytes: last.mask_bytes,
                        read: last.read.clone(),
                        gate: g,
                    });
                }
            }
        }
    }

    // JSON.
    let dir = std::path::Path::new(out_dir)
        .join("lance_arm")
        .join(format!("n{n}"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut js = String::new();
    js.push_str(&format!(
        "{{\n \"label\": \"Lance-backed LAB\", \"n\": {n}, \"reps\": {reps}, \"attest_cold_us\": {:.1},\n \"owner_gates\": [\"S-INGRESS\", \"D4\"],\n \"rows\": [\n",
        env.attest_cold_us
    ));
    for (i, r) in rows.iter().enumerate() {
        js.push_str(&format!(
            "  {{\"arm\":\"{}\",\"layout\":\"{}\",\"frac_target\":{},\"frac_actual\":{:.6},\"k\":{},\"us\":{{\"where\":{:.1},\"resolve\":{:.1},\"route\":{:.1},\"ingress\":{:.1},\"lance\":{:.1},\"fold\":{:.1},\"total\":{:.1},\"cold_total\":{:.1}}},\"rows_scanned\":{},\"rows_selected\":{},\"bytes_read\":{},\"bytes_vs_A_pct\":{:.1},\"row_addr_mask_serialized_bytes\":{},\"read\":\"{}\",\"gate\":\"{}\"}}{}\n",
            r.arm, r.layout, r.frac_target, r.frac_actual, r.k,
            r.t[T_WHERE], r.t[T_RESOLVE], r.t[T_ROUTE], r.t[T_INGRESS], r.t[T_LANCE], r.t[T_FOLD], r.t[T_TOTAL], r.cold_total,
            r.io.rows_scanned, r.selected, r.io.bytes_read, r.bytes_vs_a, r.mask_bytes, r.read, r.gate,
            if i + 1 == rows.len() { "" } else { "," }
        ));
    }
    js.push_str(" ]\n}\n");
    let path = dir.join("lance_arm.json");
    std::fs::write(&path, js).map_err(|e| e.to_string())?;
    println!("\n# wrote {}", path.display());

    if fails > 0 {
        eprintln!("FAILED: {fails} check(s)/oracle mismatch(es)");
        std::process::exit(1);
    }
    println!("# all oracle and refusal checks passed");
    Ok(())
}
