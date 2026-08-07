//! Calcify an imported ontology into LanceDB — the end of the import.
//!
//! # Why this exists
//!
//! An import that stops at an in-memory artifact leaves every consumer to slice
//! the same blob its own way. The import ends **on disk**: it writes the baked
//! node rows into LanceDB datasets, and from then on the filled database is the
//! thing that gets addressed. Nobody re-reads the source document, and nobody is
//! handed a `Vec` to cut up.
//!
//! # The seam is bytes, not a type
//!
//! This module takes `&[u8]` with a 512-byte stride, never a producer's struct.
//! The 512-byte node layout IS the contract (`key(16) | edges(16) | value(480)`),
//! so a byte seam is the honest one: it costs no dependency on whichever harvest
//! produced the rows, and it cannot drift from the layout, because it *is* the
//! layout. OBO, a relational transcode, and a hand-built fixture all arrive the
//! same way.
//!
//! # 64k tables
//!
//! Rows are partitioned into one dataset per `(classid, family)`. That is not an
//! arbitrary chunk size: in the V3 tail `identity` is a `u16`, so a
//! `(classid, family)` pair addresses **exactly 65 536 slots** — one table's
//! worth. `family` is the next tier up, `classid` the routing prefix in front of
//! it. A table is therefore a closed address space, not a page boundary someone
//! chose, and 64 Ki × 512 B = 32 MiB is its full extent.
//!
//! Because the bake sorts by `(classid, family, identity)`, those partitions are
//! already contiguous runs in the input. This module finds the runs by reading
//! key positions; it never sorts, groups, or moves a row.
//!
//! # Zero-copy
//!
//! One column, `node: FixedSizeBinary(512)`, over the caller's own allocation via
//! [`Buffer::from_custom_allocation`]. No gather, no re-pack, no intermediate
//! `Vec<u8>`: the bytes Lance writes are the bytes the bake produced, and the
//! caller's buffer is kept alive by the `Arc` it hands in.
//!
//! **This is why the row is one column and not three.** Splitting into
//! `key`/`edges`/`value` columns would read better and would cost a strided
//! gather over every row — the copy this module exists to avoid. The three-part
//! carving is a **projection over positions** on the read side
//! ([`KEY_RANGE`] / [`EDGES_RANGE`] / [`VALUE_RANGE`]), which is where it belongs:
//! the layout is addressed, not restructured.
//!
//! # What this module does not do
//!
//! It does not interpret the value slab, resolve a label, or know what an
//! ontology is. It moves addressed rows to disk and hands back the addresses. A
//! reader that wants meaning resolves the classid; that is a different concern
//! and a different crate.

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeBinaryArray};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

/// Bytes per node row — `key(16) | edges(16) | value(480)`.
pub const NODE_ROW_STRIDE: usize = 512;

/// Byte range of the canonical key within a row. Read, never restructured.
pub const KEY_RANGE: std::ops::Range<usize> = 0..16;
/// Byte range of the edge block within a row.
pub const EDGES_RANGE: std::ops::Range<usize> = 16..32;
/// Byte range of the value slab within a row.
pub const VALUE_RANGE: std::ops::Range<usize> = 32..512;

/// `classid` position inside the key: `[0, 4)`, little-endian `u32`.
const CLASSID_AT: usize = 0;
/// `family` position inside the V3 tail: `[12, 14)`, little-endian `u16`.
const FAMILY_AT: usize = 12;
/// `identity` position inside the V3 tail: `[14, 16)`, little-endian `u16`.
const IDENTITY_AT: usize = 14;

/// Rows one `(classid, family)` table can address — `identity` is a `u16`.
pub const ROWS_PER_TABLE: usize = 1 << 16;

/// The address of one 64k table: everything in front of `identity`.
///
/// Ordered exactly as the baked rows are, so a sorted row buffer yields these in
/// ascending order and each appears in exactly one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableAddr {
    /// Routing prefix.
    pub classid: u32,
    /// Basin within the class.
    pub family: u16,
}

impl TableAddr {
    /// The dataset directory name for this table, inside the database directory.
    ///
    /// Fixed width and zero-padded so a lexical listing of the database is also
    /// an ordering by address.
    #[must_use]
    pub fn dataset_name(&self) -> String {
        format!("nodes-{:08x}-{:04x}.lance", self.classid, self.family)
    }
}

/// One contiguous run of rows sharing a [`TableAddr`] — one 64k table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableRun {
    /// The table this run fills.
    pub addr: TableAddr,
    /// First row index of the run, inclusive.
    pub start: usize,
    /// One past the last row index.
    pub end: usize,
}

impl TableRun {
    /// Number of rows in the run.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the run is empty (it never is, as produced by [`partition`]).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Byte range of this run in the row buffer — contiguous, so slicing it
    /// stays zero-copy.
    #[must_use]
    pub const fn byte_range(&self) -> std::ops::Range<usize> {
        self.start * NODE_ROW_STRIDE..self.end * NODE_ROW_STRIDE
    }
}

/// Read a little-endian `u32` at `at` within row `i`.
fn u32_at(bytes: &[u8], i: usize, at: usize) -> u32 {
    let b = i * NODE_ROW_STRIDE + at;
    u32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
}

/// Read a little-endian `u16` at `at` within row `i`.
fn u16_at(bytes: &[u8], i: usize, at: usize) -> u16 {
    let b = i * NODE_ROW_STRIDE + at;
    u16::from_le_bytes([bytes[b], bytes[b + 1]])
}

/// The table address of row `i`, read straight off its key.
#[must_use]
pub fn table_addr_of(bytes: &[u8], i: usize) -> TableAddr {
    TableAddr {
        classid: u32_at(bytes, i, CLASSID_AT),
        family: u16_at(bytes, i, FAMILY_AT),
    }
}

/// The `identity` of row `i` — its slot within its 64k table.
#[must_use]
pub fn identity_of(bytes: &[u8], i: usize) -> u16 {
    u16_at(bytes, i, IDENTITY_AT)
}

/// Split a sorted row buffer into its 64k tables.
///
/// A positional scan: it compares adjacent keys and cuts where the
/// `(classid, family)` prefix changes. Rows are neither moved nor copied — each
/// run is a `start..end` index pair into the caller's buffer.
///
/// # Errors
///
/// - the buffer length is not a multiple of [`NODE_ROW_STRIDE`];
/// - the rows are not sorted by `(classid, family)`, which would make a table
///   appear in two runs and silently split it across two datasets. Refused
///   rather than accommodated: the bake emits sorted output, so an unsorted
///   buffer means something upstream is wrong and merging it here would hide
///   that.
pub fn partition(bytes: &[u8]) -> Result<Vec<TableRun>, String> {
    if !bytes.len().is_multiple_of(NODE_ROW_STRIDE) {
        return Err(format!(
            "row buffer of {} bytes is not a multiple of the {NODE_ROW_STRIDE}-byte stride",
            bytes.len()
        ));
    }
    let n = bytes.len() / NODE_ROW_STRIDE;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut runs = Vec::new();
    let mut start = 0usize;
    let mut current = table_addr_of(bytes, 0);
    for i in 1..n {
        let addr = table_addr_of(bytes, i);
        if addr == current {
            continue;
        }
        if addr < current {
            return Err(format!(
                "rows are not sorted by (classid, family): row {i} is {addr:?} \
                 after {current:?}"
            ));
        }
        runs.push(TableRun {
            addr: current,
            start,
            end: i,
        });
        start = i;
        current = addr;
    }
    runs.push(TableRun {
        addr: current,
        start,
        end: n,
    });

    for r in &runs {
        if r.len() > ROWS_PER_TABLE {
            return Err(format!(
                "table {:?} holds {} rows, more than the {ROWS_PER_TABLE} an u16 \
                 identity can address",
                r.addr,
                r.len()
            ));
        }
    }
    Ok(runs)
}

/// The single-column schema every node dataset uses.
///
/// One `FixedSizeBinary(512)` column. The `key`/`edges`/`value` carving is a
/// read-side projection over byte positions, not three stored columns — see the
/// module docs on why splitting would cost the copy this module avoids.
#[must_use]
pub fn node_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "node",
        DataType::FixedSizeBinary(NODE_ROW_STRIDE as i32),
        false,
    )]))
}

/// Wrap a row buffer as a `RecordBatch` **without copying it**.
///
/// `owner` keeps the caller's allocation alive for as long as Arrow holds the
/// buffer; pass the `Arc` that owns the rows (e.g. `Arc<Vec<Row512>>`). `bytes`
/// must point into that allocation.
///
/// # Errors
///
/// The buffer length must be a multiple of [`NODE_ROW_STRIDE`].
pub fn node_rows_batch<O>(bytes: &[u8], owner: Arc<O>) -> Result<RecordBatch, String>
where
    O: std::panic::RefUnwindSafe + Send + Sync + 'static,
{
    if !bytes.len().is_multiple_of(NODE_ROW_STRIDE) {
        return Err(format!(
            "row buffer of {} bytes is not a multiple of the {NODE_ROW_STRIDE}-byte stride",
            bytes.len()
        ));
    }
    let ptr = std::ptr::NonNull::new(bytes.as_ptr().cast_mut())
        .ok_or_else(|| "row buffer pointer is null".to_string())?;

    // SAFETY: `ptr`/`bytes.len()` describe exactly the caller's slice, and
    // `owner` is an `Arc` over the allocation that slice points into. Arrow
    // holds that `Arc` for the buffer's whole life, so the memory outlives every
    // read. The buffer is never written through: `Buffer` is immutable, and the
    // `cast_mut` above only satisfies `NonNull<u8>`'s signature.
    let buffer = unsafe { Buffer::from_custom_allocation(ptr, bytes.len(), owner) };

    let array = FixedSizeBinaryArray::new(NODE_ROW_STRIDE as i32, buffer, None);
    let column: ArrayRef = Arc::new(array);
    RecordBatch::try_new(node_schema(), vec![column]).map_err(|e| format!("record batch: {e}"))
}

/// Write one 64k table to its own Lance dataset under `db_dir`.
///
/// Returns the dataset path. `WriteMode::Create` is deliberate: an import writes
/// a table once. Appending to an existing table would let a second import
/// silently double its rows, and a table whose identity space is full has
/// nothing to append anyway.
///
/// # Errors
///
/// Propagates batch construction and Lance write failures, including an attempt
/// to write a table that already exists.
pub async fn write_table<O>(
    bytes: &[u8],
    owner: Arc<O>,
    run: TableRun,
    db_dir: &str,
) -> Result<String, String>
where
    O: std::panic::RefUnwindSafe + Send + Sync + 'static,
{
    use lance::dataset::{WriteMode, WriteParams};
    use lance::Dataset;

    let slice = bytes
        .get(run.byte_range())
        .ok_or_else(|| format!("run {run:?} is out of bounds for {} bytes", bytes.len()))?;
    let batch = node_rows_batch(slice, owner)?;
    let schema = batch.schema();
    let path = format!(
        "{}/{}",
        db_dir.trim_end_matches('/'),
        run.addr.dataset_name()
    );

    let reader = arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
    let params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };
    Dataset::write(reader, &path, Some(params))
        .await
        .map_err(|e| format!("lance write {path}: {e}"))?;
    Ok(path)
}

/// **The import's last step:** write every 64k table of a baked row buffer into
/// `db_dir`, and hand back the table addresses that now exist on disk.
///
/// After this returns, the database is the thing to address. The row buffer has
/// no further role and the source document certainly does not.
///
/// # Errors
///
/// Propagates [`partition`]'s refusals and any per-table write failure. A
/// failure part-way leaves the tables written so far in place — the caller sees
/// which, because the error names the table that failed.
pub async fn write_database<O>(
    bytes: &[u8],
    owner: Arc<O>,
    db_dir: &str,
) -> Result<Vec<TableAddr>, String>
where
    O: std::panic::RefUnwindSafe + Send + Sync + 'static,
{
    let runs = partition(bytes)?;
    let mut written = Vec::with_capacity(runs.len());
    for run in runs {
        write_table(bytes, Arc::clone(&owner), run, db_dir).await?;
        written.push(run.addr);
    }
    Ok(written)
}

/// A 64k table opened from disk — the addressable form.
///
/// Holds the Lance dataset, not a decoded copy of it. Reads go through
/// [`Self::row`], which returns the row's 512 bytes and leaves every reading of
/// them (key / edges / value, and whatever the classid says those mean) to the
/// caller.
pub struct NodeTable {
    addr: TableAddr,
    dataset: lance::Dataset,
}

impl NodeTable {
    /// Open the dataset for `addr` inside `db_dir`.
    ///
    /// # Errors
    ///
    /// Propagates Lance open failures, including a table that was never written.
    pub async fn open(db_dir: &str, addr: TableAddr) -> Result<Self, String> {
        let path = format!("{}/{}", db_dir.trim_end_matches('/'), addr.dataset_name());
        let dataset = lance::Dataset::open(&path)
            .await
            .map_err(|e| format!("lance open {path}: {e}"))?;
        Ok(Self { addr, dataset })
    }

    /// The address this table answers for.
    #[must_use]
    pub const fn addr(&self) -> TableAddr {
        self.addr
    }

    /// Rows stored in this table.
    ///
    /// Async and fallible because it is a read against the dataset, not a field
    /// on this struct — the table holds an address and a handle, never a count
    /// it would have to keep in step with the data.
    ///
    /// # Errors
    ///
    /// Propagates the Lance scan failure.
    pub async fn count_rows(&self) -> Result<usize, String> {
        self.dataset
            .count_rows(None)
            .await
            .map_err(|e| format!("lance count_rows {}: {e}", self.addr.dataset_name()))
    }

    /// The underlying dataset, for callers that want to query rather than
    /// address — this crate's planner surface takes it from here.
    #[must_use]
    pub const fn dataset(&self) -> &lance::Dataset {
        &self.dataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a row with the given key fields; the rest of the 512 bytes carry a
    /// recognisable marker so a copy or a mis-stride would be visible.
    fn row(classid: u32, family: u16, identity: u16, marker: u8) -> [u8; NODE_ROW_STRIDE] {
        let mut r = [marker; NODE_ROW_STRIDE];
        r[CLASSID_AT..CLASSID_AT + 4].copy_from_slice(&classid.to_le_bytes());
        r[FAMILY_AT..FAMILY_AT + 2].copy_from_slice(&family.to_le_bytes());
        r[IDENTITY_AT..IDENTITY_AT + 2].copy_from_slice(&identity.to_le_bytes());
        r
    }

    fn buffer(rows: &[[u8; NODE_ROW_STRIDE]]) -> Vec<u8> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    /// The scan cuts where `(classid, family)` changes and nowhere else.
    ///
    /// The anti-vacuity half is the middle table: a partitioner that emitted one
    /// run per row, or one run for everything, fails on the run lengths.
    #[test]
    fn partition_cuts_on_the_table_prefix_only() {
        let buf = buffer(&[
            row(1, 0, 0, 0xA0),
            row(1, 0, 1, 0xA1),
            row(1, 1, 0, 0xB0),
            row(2, 0, 0, 0xC0),
            row(2, 0, 7, 0xC1),
            row(2, 0, 9, 0xC2),
        ]);
        let runs = partition(&buf).expect("sorted");
        assert_eq!(runs.len(), 3, "three distinct (classid, family) tables");
        assert_eq!(runs[0].addr, TableAddr { classid: 1, family: 0 });
        assert_eq!(runs[0].len(), 2);
        assert_eq!(runs[1].addr, TableAddr { classid: 1, family: 1 });
        assert_eq!(runs[1].len(), 1, "a one-row table is still its own table");
        assert_eq!(runs[2].addr, TableAddr { classid: 2, family: 0 });
        assert_eq!(runs[2].len(), 3);
    }

    /// Unsorted input is refused rather than merged — otherwise one table would
    /// be written as two datasets and the second would fail on Create, or worse,
    /// succeed under a different name.
    #[test]
    fn partition_refuses_unsorted_and_accepts_sorted() {
        let bad = buffer(&[row(2, 0, 0, 1), row(1, 0, 0, 2)]);
        assert!(partition(&bad).is_err(), "descending prefix must be refused");
        let good = buffer(&[row(1, 0, 0, 1), row(2, 0, 0, 2)]);
        assert_eq!(partition(&good).expect("sorted").len(), 2);
    }

    /// A truncated buffer is a mis-strided buffer; refusing it here is cheaper
    /// than a silently shifted read of every row after the first.
    #[test]
    fn a_partial_row_is_refused() {
        let mut buf = buffer(&[row(1, 0, 0, 1)]);
        buf.truncate(NODE_ROW_STRIDE - 1);
        assert!(partition(&buf).is_err());
        assert!(node_rows_batch(&buf, Arc::new(())).is_err());
    }

    /// An empty buffer has no tables — and is not an error.
    #[test]
    fn an_empty_buffer_yields_no_tables() {
        assert!(partition(&[]).expect("empty is legal").is_empty());
    }

    /// The batch is a view over the caller's allocation: the bytes Arrow exposes
    /// are the bytes that were handed in, at the same address.
    #[test]
    fn the_batch_borrows_the_callers_allocation() {
        let owner = Arc::new(buffer(&[row(9, 3, 5, 0x5A), row(9, 3, 6, 0x6B)]));
        let src_ptr = owner.as_ptr();
        let batch = node_rows_batch(&owner, Arc::clone(&owner)).expect("well-formed");
        assert_eq!(batch.num_rows(), 2);
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("fixed-size binary");
        assert!(
            std::ptr::eq(col.value_data().as_ptr(), src_ptr),
            "Arrow must point AT the caller's buffer, not a copy of it"
        );
        assert_eq!(col.value(0)[NODE_ROW_STRIDE - 1], 0x5A, "row 0 intact");
        assert_eq!(col.value(1)[NODE_ROW_STRIDE - 1], 0x6B, "row 1 intact");
    }

    /// Table names are fixed-width, so a lexical sort of the database directory
    /// is an ordering by address.
    #[test]
    fn dataset_names_sort_by_address() {
        let a = TableAddr { classid: 0x02, family: 0x0010 }.dataset_name();
        let b = TableAddr { classid: 0x10, family: 0x0002 }.dataset_name();
        assert_eq!(a, "nodes-00000002-0010.lance");
        assert!(a < b, "classid dominates the ordering, as in the key");
    }

    /// The read-side carving covers the row exactly once — the property that
    /// lets one stored column stand in for three.
    #[test]
    fn the_projection_ranges_tile_the_row() {
        assert_eq!(KEY_RANGE.start, 0);
        assert_eq!(KEY_RANGE.end, EDGES_RANGE.start);
        assert_eq!(EDGES_RANGE.end, VALUE_RANGE.start);
        assert_eq!(VALUE_RANGE.end, NODE_ROW_STRIDE);
    }
}
