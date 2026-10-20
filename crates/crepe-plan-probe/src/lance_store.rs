//! Stage 3 of D-HOLO-H14d for the star: the relations live column-major in
//! Lance (one table per relation, columns `c0` and `c1`, written through
//! lancedb 0.40, read through lance 13), and every side is read under an
//! address mask (plan §2d).
//!
//! Per side:
//!
//! 1. the one column the mask needs is read with `_rowaddr`;
//! 2. the address mask `R` is built by relalg over that column as a `U32`
//!    lane (`eq_u32` for a restriction, `Or` of `eq_u32` for the semijoin
//!    centers), lowered by mask-risc onto `ndarray::simd`;
//! 3. the side is read with `with_row_addr_prefilter(R)`;
//! 4. the rows go to `JoinTable::prepare_star`, as on the in-memory path.
//!
//! A side's generation is its table's Lance version.

use std::sync::{Arc, Mutex};

use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::Dataset;
use lance::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
use lance_graph_mask_risc::{LaneRef, Planes, words_for};
use lance_graph_relalg::{Measure, Query, Rel, RelError, run_with_program};
use mork_pathmap_probe::join::for_each_set;

use crate::logic::RelId;
use crate::native::{PreparedStar, StarPlan, prepare_from_rows};

/// Up to this many semijoin centers, the address mask is an `Or` of `eq_u32`
/// atoms through relalg; above it, the scalar membership test builds it.
///
/// The limit comes from a setting, not from the algebra and not from a speed
/// crossover. relalg lowers `Or(a1..ak)` left-associatively, and each atom
/// and each `OR` takes a scratch slot of its own (slots are not reused
/// inside a fold), so `k` centers take `2k - 1` slots. One fold may use
/// `lance_graph_r2il::MAX_SLOTS` (8, "sized with headroom") of them. Measured:
/// 4 centers fit (7 slots); 5 need 9 and fail with
/// `ScratchBufferTooSmall { need_words: 145, have_words: 129 }` at 1,000 rows.
/// mask-risc itself addresses up to 65,536 slots. A larger arena, or slot
/// reuse in the fold, lifts this. (The 64-needle crossover in plan §2c is
/// ndarray's row filter, a different mechanism.)
pub const SEMIJOIN_ATOMS_MAX: usize = 4;

#[derive(Debug)]
pub enum LanceStoreError {
    Lance(String),
    UnknownRelation(RelId),
    Relalg(RelError),
}

fn lance_err(e: impl std::fmt::Display) -> LanceStoreError {
    LanceStoreError::Lance(e.to_string())
}

/// One relation per table.
pub struct LanceStore {
    tables: Vec<(RelId, lancedb::Table)>,
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("c0", DataType::UInt32, false),
        Field::new("c1", DataType::UInt32, false),
    ]))
}

fn batch(rows: &[[u32; 2]]) -> RecordBatch {
    let c0 = UInt32Array::from_iter_values(rows.iter().map(|r| r[0]));
    let c1 = UInt32Array::from_iter_values(rows.iter().map(|r| r[1]));
    RecordBatch::try_new(schema(), vec![Arc::new(c0), Arc::new(c1)]).expect("two u32 columns")
}

impl LanceStore {
    /// Creates one table per relation under `uri`, each written in key order
    /// (`(c0, c1)` ascending, which is the numeric projection order of the
    /// keys `addressed::key` gives the rows). The write path orders; readers
    /// attest the order, they never sort.
    pub async fn write(
        uri: &str,
        relations: &[(RelId, &str, &[[u32; 2]])],
    ) -> Result<Self, LanceStoreError> {
        let ordered: Vec<(RelId, &str, Vec<[u32; 2]>)> = relations
            .iter()
            .map(|&(rel, name, rows)| {
                let mut r = rows.to_vec();
                r.sort_unstable();
                (rel, name, r)
            })
            .collect();
        let views: Vec<(RelId, &str, &[[u32; 2]])> = ordered
            .iter()
            .map(|(rel, name, r)| (*rel, *name, r.as_slice()))
            .collect();
        Self::write_as_given(uri, &views).await
    }

    /// Creates the tables with the rows in the order given (for the test that
    /// an unordered table is refused at attest).
    pub async fn write_as_given(
        uri: &str,
        relations: &[(RelId, &str, &[[u32; 2]])],
    ) -> Result<Self, LanceStoreError> {
        let db = lancedb::connect(uri).execute().await.map_err(lance_err)?;
        let mut tables = Vec::new();
        for &(rel, name, rows) in relations {
            let t = db
                .create_table(name, batch(rows))
                .execute()
                .await
                .map_err(lance_err)?;
            tables.push((rel, t));
        }
        Ok(Self { tables })
    }

    fn table(&self, rel: RelId) -> Result<&lancedb::Table, LanceStoreError> {
        self.tables
            .iter()
            .find(|(r, _)| *r == rel)
            .map(|(_, t)| t)
            .ok_or(LanceStoreError::UnknownRelation(rel))
    }

    /// Appends rows to a relation; returns its new version.
    pub async fn append(&self, rel: RelId, rows: &[[u32; 2]]) -> Result<u64, LanceStoreError> {
        let t = self.table(rel)?;
        t.add(batch(rows)).execute().await.map_err(lance_err)?;
        t.version().await.map_err(lance_err)
    }

    /// The relation's current Lance version.
    pub async fn version(&self, rel: RelId) -> Result<u64, LanceStoreError> {
        self.table(rel)?.version().await.map_err(lance_err)
    }

    /// The current version of each side of `plan`.
    pub async fn generations(&self, plan: &StarPlan) -> Result<Vec<u64>, LanceStoreError> {
        let mut out = Vec::with_capacity(plan.sides.len());
        for s in &plan.sides {
            out.push(self.version(s.key.rel).await?);
        }
        Ok(out)
    }

    pub(crate) async fn dataset(&self, rel: RelId) -> Result<Arc<Dataset>, LanceStoreError> {
        let t = self.table(rel)?;
        let wrapper = t
            .dataset()
            .ok_or_else(|| LanceStoreError::Lance("not a native table".into()))?;
        wrapper.get().await.map_err(lance_err)
    }
}

/// Lance's own statistics for one read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Io {
    pub rows_scanned: usize,
    pub rows_returned: usize,
    pub bytes_read: usize,
}

/// What preparing one side read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SideIo {
    /// The read of the column the mask was built from; `None` when the side
    /// was read without a mask.
    pub mask_column: Option<Io>,
    /// `|R|`, when a mask was built.
    pub mask_rows: Option<usize>,
    /// The read of the side itself (both columns).
    pub rows: Io,
}

fn stats_sink() -> (
    Arc<Mutex<Io>>,
    lance::dataset::scanner::ExecutionStatsCallback,
) {
    let stats: Arc<Mutex<Io>> = Arc::default();
    let sink = stats.clone();
    let cb: lance::dataset::scanner::ExecutionStatsCallback = Arc::new(move |c| {
        let mut io = sink.lock().expect("stats lock");
        io.bytes_read = c.bytes_read;
        io.rows_scanned = c.all_counts.get("rows_scanned").copied().unwrap_or(0);
    });
    (stats, cb)
}

fn u32_col<'b>(b: &'b RecordBatch, name: &str) -> &'b UInt32Array {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .expect("u32 column")
}

/// One column, every row, with each row's address.
async fn read_column(
    ds: &Dataset,
    col: usize,
) -> Result<(Vec<u32>, Vec<u64>, Io), LanceStoreError> {
    let name = format!("c{col}");
    let (stats, cb) = stats_sink();
    let mut s = ds.scan();
    s.project(&[name.as_str()]).map_err(lance_err)?;
    s.with_row_address();
    s.scan_stats_callback(cb);
    let b = s.try_into_batch().await.map_err(lance_err)?;
    let values = u32_col(&b, &name).values().to_vec();
    let addr = b
        .column_by_name("_rowaddr")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .expect("_rowaddr")
        .values()
        .to_vec();
    let mut io = *stats.lock().expect("stats lock");
    io.rows_returned = b.num_rows();
    Ok((values, addr, io))
}

/// Both columns, under `mask` when given.
pub(crate) async fn read_rows(
    ds: &Dataset,
    mask: Option<RowAddrMask>,
) -> Result<(Vec<[u32; 2]>, Io), LanceStoreError> {
    let (stats, cb) = stats_sink();
    let mut s = ds.scan();
    s.project(&["c0", "c1"]).map_err(lance_err)?;
    if let Some(m) = mask {
        s.with_row_addr_prefilter(m);
    }
    s.scan_stats_callback(cb);
    let b = s.try_into_batch().await.map_err(lance_err)?;
    let c0 = u32_col(&b, "c0");
    let c1 = u32_col(&b, "c1");
    let rows = (0..b.num_rows())
        .map(|i| [c0.value(i), c1.value(i)])
        .collect();
    let mut io = *stats.lock().expect("stats lock");
    io.rows_returned = b.num_rows();
    Ok((rows, io))
}

/// `rel` over `values` as lane 0: the relalg description, lowered by
/// mask-risc, run over the column.
pub fn address_mask(values: &[u32], rel: &Rel) -> Result<Vec<u64>, RelError> {
    let mut keep = vec![0u64; words_for(values.len())];
    let lanes = [LaneRef::U32(values)];
    let planes = Planes {
        n_rows: values.len(),
        masks: &[],
        lanes: &lanes,
    };
    run_with_program(
        &Query {
            rel: rel.clone(),
            measure: Measure::Keep,
        },
        &planes,
        Some(&mut keep),
    )?;
    Ok(keep)
}

/// Rows whose value is one of `centers` (sorted, deduplicated).
pub fn membership_mask(values: &[u32], centers: &[u32]) -> Result<Vec<u64>, RelError> {
    if centers.is_empty() {
        return Ok(vec![0; words_for(values.len())]);
    }
    if centers.len() <= SEMIJOIN_ATOMS_MAX {
        let rel = Rel::Or(centers.iter().map(|&c| Rel::eq_u32(0, c)).collect());
        return address_mask(values, &rel);
    }
    let mut keep = vec![0u64; words_for(values.len())];
    for (i, v) in values.iter().enumerate() {
        if centers.binary_search(v).is_ok() {
            keep[i / 64] |= 1 << (i % 64);
        }
    }
    Ok(keep)
}

fn row_addr_mask(mask: &[u64], n: usize, addr: &[u64]) -> (RowAddrMask, usize) {
    let mut set = Vec::new();
    for_each_set(mask, n, |i| set.push(addr[i]));
    let k = set.len();
    (RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(set)), k)
}

/// Prepares `plan` from the Lance tables, reading each side under its
/// address mask. With `semijoin`, an unrestricted side is read only for the
/// centers every restricted side has (as `native::prepare_semijoin`).
pub async fn prepare_lance(
    plan: &StarPlan,
    store: &LanceStore,
    semijoin: bool,
) -> Result<(PreparedStar, Vec<SideIo>), LanceStoreError> {
    let n_sides = plan.sides.len();
    let mut rows: Vec<Option<(Vec<[u32; 2]>, u64)>> = vec![None; n_sides];
    let mut io = vec![SideIo::default(); n_sides];
    let restricted = |i: usize| {
        let k = plan.sides[i].key;
        (!k.unary).then_some(k.restrict).flatten()
    };

    // Restricted sides first: their rows name the centers.
    for i in (0..n_sides).filter(|&i| restricted(i).is_some()) {
        let key = plan.sides[i].key;
        let ds = store.dataset(key.rel).await?;
        let other = 1 - key.key_col as usize;
        let (values, addr, col_io) = read_column(&ds, other).await?;
        // A constant above `u32::MAX` matches no stored value: an empty mask,
        // as the memory path finds no row.
        let mask = match u32::try_from(restricted(i).expect("restricted")) {
            Ok(k) => address_mask(&values, &Rel::eq_u32(0, k)).map_err(LanceStoreError::Relalg)?,
            Err(_) => vec![0; words_for(values.len())],
        };
        let (m, r) = row_addr_mask(&mask, values.len(), &addr);
        let (got, rows_io) = read_rows(&ds, Some(m)).await?;
        io[i] = SideIo {
            mask_column: Some(col_io),
            mask_rows: Some(r),
            rows: rows_io,
        };
        rows[i] = Some((got, ds.version().version));
    }

    let centers: Option<Vec<u32>> = if semijoin {
        let mut acc: Option<Vec<u32>> = None;
        for i in (0..n_sides).filter(|&i| restricted(i).is_some()) {
            let col = plan.sides[i].key.key_col as usize;
            let mut c: Vec<u32> = rows[i]
                .as_ref()
                .expect("read")
                .0
                .iter()
                .map(|r| r[col])
                .collect();
            c.sort_unstable();
            c.dedup();
            acc = Some(match acc {
                None => c,
                Some(prev) => prev
                    .into_iter()
                    .filter(|v| c.binary_search(v).is_ok())
                    .collect(),
            });
        }
        // A restricted side keeps only the centers every restricted side has,
        // as in `native::prepare_semijoin`. Its rows were already read by its
        // restriction; this trims them, it reads nothing.
        if let Some(c) = &acc {
            for i in (0..n_sides).filter(|&i| restricted(i).is_some()) {
                let col = plan.sides[i].key.key_col as usize;
                if let Some((r, _)) = rows[i].as_mut() {
                    r.retain(|row| c.binary_search(&row[col]).is_ok());
                }
            }
        }
        acc
    } else {
        None
    };

    for i in (0..n_sides).filter(|&i| restricted(i).is_none()) {
        let key = plan.sides[i].key;
        let ds = store.dataset(key.rel).await?;
        let version = ds.version().version;
        match &centers {
            Some(c) => {
                let (values, addr, col_io) = read_column(&ds, key.key_col as usize).await?;
                let mask = membership_mask(&values, c).map_err(LanceStoreError::Relalg)?;
                let (m, r) = row_addr_mask(&mask, values.len(), &addr);
                let (got, rows_io) = read_rows(&ds, Some(m)).await?;
                io[i] = SideIo {
                    mask_column: Some(col_io),
                    mask_rows: Some(r),
                    rows: rows_io,
                };
                rows[i] = Some((got, version));
            }
            None => {
                let (got, rows_io) = read_rows(&ds, None).await?;
                io[i] = SideIo {
                    mask_column: None,
                    mask_rows: None,
                    rows: rows_io,
                };
                rows[i] = Some((got, version));
            }
        }
    }

    let owned = rows
        .into_iter()
        .map(|r| r.expect("every side read"))
        .collect();
    Ok((prepare_from_rows(plan, owned), io))
}
