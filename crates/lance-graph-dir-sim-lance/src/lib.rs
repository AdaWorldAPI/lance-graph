//! The dir-sim history log stored in a Lance dataset (plan
//! `dir-sim-persist-v1`, D-DSP-2b).
//!
//! [`lance_graph_dir_sim::log`] decides what a save writes: a batch of
//! records holding what an image has beyond the log, with a checksummed
//! header. This crate only stores those records and reads them back.
//!
//! - **One batch is one Lance append.** Every record of a batch lands in one
//!   commit, so a batch is in the dataset whole or not at all. The log state
//!   advances ([`LogState::committed`]) only after that commit succeeded.
//! - **Two clocks, never mixed.** The history's logical versions travel in
//!   the records. The Lance dataset version is only the physical commit
//!   counter, used here to refuse a stale writer and to detect a race; it is
//!   never written into, or read as, a logical version.
//! - **Read once at restore.** [`LanceLog::restore`] reads every record in
//!   one scan and hands them to [`replay`], which refuses the whole log at
//!   the first damaged batch. Nothing is repaired.
//!
//! A writer holds the dataset version it last saw. [`LanceLog::append`]
//! refuses to write when the dataset has moved since
//! ([`StoreError::StaleWriter`]): Lance would rebase an append onto a newer
//! version, which would interleave two writers' batches. A writer that
//! slips in between that check and the commit makes the new version
//! something other than the next one; that is reported as
//! [`StoreError::Race`] after the fact, since the commit has landed.

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, RecordBatch, UInt32Array, UInt64Array, UInt8Array};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use lance::Dataset;
use lance_graph_dir_sim::image::StoredImage;
use lance_graph_dir_sim::log::{replay, Batch, LogError, LogState, Record, ReplayError};

pub mod append;
pub use append::{AppendError, VersionedAppend};

/// Why a save or a restore failed.
#[derive(Debug)]
pub enum StoreError {
    /// Lance refused an operation.
    Lance(lance::Error),
    /// The stored records do not form a valid log.
    Replay(ReplayError),
    /// The batch does not follow the log state it was built against.
    Log(LogError),
    /// The dataset is not at the version this writer last saw (`None`: no
    /// dataset). Nothing was written.
    StaleWriter {
        /// The version this writer last saw.
        expected: Option<u64>,
        /// The version the dataset is at.
        found: Option<u64>,
    },
    /// The commit landed, but not as the next version: another writer
    /// committed between this writer's check and its commit. The batch is
    /// stored; the log state was not advanced.
    Race {
        /// The version the commit should have made.
        expected: u64,
        /// The version it made.
        found: u64,
    },
    /// A stored column is missing or has another type.
    Schema(String),
}

impl From<lance::Error> for StoreError {
    fn from(e: lance::Error) -> Self {
        StoreError::Lance(e)
    }
}

impl From<AppendError> for StoreError {
    fn from(e: AppendError) -> Self {
        match e {
            AppendError::Lance(e) => StoreError::Lance(e),
            AppendError::StaleWriter { expected, found } => {
                StoreError::StaleWriter { expected, found }
            }
            AppendError::Race { expected, found } => StoreError::Race { expected, found },
        }
    }
}

/// The record columns: `batch`, `seq`, `kind`, `body`.
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("batch", DataType::UInt64, false),
        Field::new("seq", DataType::UInt32, false),
        Field::new("kind", DataType::UInt8, false),
        Field::new("body", DataType::Binary, false),
    ]))
}

fn to_batch(records: &[Record]) -> Result<RecordBatch, ArrowError> {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(UInt64Array::from_iter_values(
                records.iter().map(|r| r.batch),
            )),
            Arc::new(UInt32Array::from_iter_values(records.iter().map(|r| r.seq))),
            Arc::new(UInt8Array::from_iter_values(records.iter().map(|r| r.kind))),
            Arc::new(BinaryArray::from_iter_values(
                records.iter().map(|r| r.body.as_slice()),
            )),
        ],
    )
}

fn column<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T, StoreError> {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| StoreError::Schema(format!("column `{name}`")))
}

fn from_batch(b: &RecordBatch) -> Result<Vec<Record>, StoreError> {
    let batch: &UInt64Array = column(b, "batch")?;
    let seq: &UInt32Array = column(b, "seq")?;
    let kind: &UInt8Array = column(b, "kind")?;
    let body: &BinaryArray = column(b, "body")?;
    if [
        batch.null_count(),
        seq.null_count(),
        kind.null_count(),
        body.null_count(),
    ]
    .iter()
    .any(|&n| n != 0)
    {
        return Err(StoreError::Schema("null in a record column".into()));
    }
    Ok((0..b.num_rows())
        .map(|i| Record {
            batch: batch.value(i),
            seq: seq.value(i),
            kind: kind.value(i),
            body: body.value(i).to_vec(),
        })
        .collect())
}

fn is_not_found(e: &lance::Error) -> bool {
    matches!(e, lance::Error::DatasetNotFound { .. })
}

/// A writer's handle on the log dataset at `uri`.
///
/// The commit discipline (stale writer refused before writing, race reported after) is
/// [`VersionedAppend`]'s; this type adds the log's own rule, that a batch is the next one.
#[derive(Debug, Clone)]
pub struct LanceLog {
    writer: VersionedAppend,
}

impl LanceLog {
    /// A writer for a log that does not exist yet. The first
    /// [`append`](Self::append) creates the dataset and refuses if one is
    /// already there.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            writer: VersionedAppend::at(uri, None),
        }
    }

    /// The dataset version this writer last saw (the physical commit
    /// counter, not a logical version).
    pub fn version(&self) -> Option<u64> {
        self.writer.version()
    }

    /// Read the whole log at `uri` once and replay it: the stored image,
    /// the log state to append after it, and a writer at the dataset's
    /// current version. A missing dataset is an empty log.
    ///
    /// # Errors
    ///
    /// [`StoreError::Replay`] at the first damaged batch;
    /// [`StoreError::Lance`] / [`StoreError::Schema`] when the dataset
    /// cannot be read as records.
    pub async fn restore(
        uri: impl Into<String>,
    ) -> Result<(StoredImage, LogState, LanceLog), StoreError> {
        let uri = uri.into();
        let ds = match Dataset::open(&uri).await {
            Ok(ds) => ds,
            Err(e) if is_not_found(&e) => {
                let (image, state) = replay(Vec::new()).map_err(StoreError::Replay)?;
                return Ok((image, state, LanceLog::new(uri)));
            }
            Err(e) => return Err(e.into()),
        };
        let all = ds.scan().try_into_batch().await?;
        let records = from_batch(&all)?;
        let (image, state) = replay(records).map_err(StoreError::Replay)?;
        let version = Some(ds.version().version);
        Ok((
            image,
            state,
            LanceLog {
                writer: VersionedAppend::at(uri, version),
            },
        ))
    }

    /// Store `batch` as one Lance commit, then advance `log` past it.
    ///
    /// Returns the new dataset version.
    ///
    /// # Errors
    ///
    /// - [`StoreError::Log`] if `batch` is not the next batch of `log`;
    ///   nothing is written.
    /// - [`StoreError::StaleWriter`] if the dataset is not at the version
    ///   this writer last saw; nothing is written.
    /// - [`StoreError::Race`] if the commit landed as another version than
    ///   the next one; the batch is stored, `log` is not advanced.
    pub async fn append(&mut self, log: &mut LogState, batch: &Batch) -> Result<u64, StoreError> {
        if batch.number() != log.next_batch() {
            return Err(StoreError::Log(LogError::StaleBatch {
                expected: log.next_batch(),
                found: batch.number(),
            }));
        }
        let rb = to_batch(&batch.records).map_err(|e| StoreError::Schema(e.to_string()))?;
        let made = self.writer.append(rb).await?;
        log.committed(batch).map_err(StoreError::Log)?;
        Ok(made)
    }
}

#[cfg(test)]
mod tests {
    use super::append::next_version;
    use super::*;

    #[test]
    fn records_round_trip_through_a_record_batch() {
        let records = vec![
            Record {
                batch: 3,
                seq: 0,
                kind: 9,
                body: vec![1, 2, 3],
            },
            Record {
                batch: 3,
                seq: 1,
                kind: 2,
                body: vec![],
            },
        ];
        let rb = to_batch(&records).unwrap();
        assert_eq!(from_batch(&rb).unwrap(), records);
    }

    #[test]
    fn the_next_version_is_one_more_or_one_for_a_new_dataset() {
        assert_eq!(next_version(None), 1);
        assert_eq!(next_version(Some(4)), 5);
    }
}
