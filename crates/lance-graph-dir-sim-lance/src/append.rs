//! One stale-checked Lance append, over any caller's records (EXO-483).
//!
//! [`VersionedAppend`] is the commit discipline [`crate::LanceLog`] uses, with nothing of
//! the dir-sim log in it, so a consumer with its own records (a document version log, a
//! run record) can write durably under the same rules:
//!
//! - **One batch is one Lance commit.** A batch lands whole or not at all.
//! - **A stale writer writes nothing.** A writer holds the dataset version it last saw.
//!   If the dataset has moved since, [`VersionedAppend::append`] refuses before writing
//!   ([`AppendError::StaleWriter`]): Lance would rebase the append onto the newer version,
//!   which would interleave two writers' batches.
//! - **One writer per dataset.** The supported deployment has a single batch writer
//!   publishing to a dataset, and who may read or write what is decided by the read/write
//!   allow and deny masks, not by racing commits. So there is no concurrent publication
//!   to arbitrate, and this type adds no lock, fence or conditional commit. Lance 14 would
//!   not provide one for appends anyway: an `Append` whose read version is behind is
//!   rebased onto the newer one, because `Append` is compatible with `Append`
//!   (lance 14.0.3 `src/io/commit/conflict_resolver.rs`, `check_append_txn`).
//! - **A second writer is a deployment fault, reported after the fact.** If one appears
//!   anyway and commits between the stale check and this commit, the new version is not
//!   the next one. The batch HAS landed by then: [`AppendError::Race`] reports a landed
//!   batch, never a refusal, and must not be retried as if nothing was written. The
//!   writer's version is not advanced; the caller reopens and reads what landed.
//! - **Two clocks, never mixed.** The Lance dataset version is the physical commit counter,
//!   used only for the two checks above. The caller's logical version (a document's `n`, a
//!   log's batch number) lives in its own records, and this type never reads or writes it.
//!
//! What it is not: an idempotence rule, a receipt, or a log format. Whether a record may be
//! appended at all (its parent is the head, its batch is the next one) is the caller's own
//! law and is checked by the caller before it calls `append`. Nothing here is written for
//! a read.

use arrow_array::{RecordBatch, RecordBatchIterator};
use lance::dataset::{WriteMode, WriteParams};
use lance::Dataset;

/// Why an append failed.
#[derive(Debug)]
pub enum AppendError {
    /// Lance refused an operation.
    Lance(lance::Error),
    /// The dataset is not at the version this writer last saw (`None`: no dataset).
    /// Nothing was written.
    StaleWriter {
        /// The version this writer last saw.
        expected: Option<u64>,
        /// The version the dataset is at.
        found: Option<u64>,
    },
    /// The commit LANDED, but not as the next version: a second writer, outside the
    /// one-writer contract, committed between this writer's check and its commit. The
    /// batch is stored, so this is not a refusal and a retry would write it twice. The
    /// writer's version was not advanced.
    Race {
        /// The version the commit should have made.
        expected: u64,
        /// The version it made.
        found: u64,
    },
}

impl From<lance::Error> for AppendError {
    fn from(e: lance::Error) -> Self {
        AppendError::Lance(e)
    }
}

fn is_not_found(e: &lance::Error) -> bool {
    matches!(e, lance::Error::DatasetNotFound { .. })
}

/// The dataset version a commit on top of `seen` makes: Lance numbers a new dataset 1 and
/// each later commit one higher.
pub(crate) fn next_version(seen: Option<u64>) -> u64 {
    seen.map_or(1, |v| v + 1)
}

/// The version of the dataset at `uri`, or `None` when there is none.
async fn current(uri: &str) -> Result<Option<u64>, lance::Error> {
    match Dataset::open(uri).await {
        Ok(ds) => Ok(Some(ds.latest_version_id().await?)),
        Err(e) if is_not_found(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// A writer's handle on one dataset, holding the version it last saw.
#[derive(Debug, Clone)]
pub struct VersionedAppend {
    uri: String,
    /// The dataset version this writer last saw; `None` while there is no dataset.
    version: Option<u64>,
}

impl VersionedAppend {
    /// A writer that has seen `version` of the dataset at `uri` (`None`: no dataset yet,
    /// so the first append creates it and refuses if one is already there).
    pub fn at(uri: impl Into<String>, version: Option<u64>) -> Self {
        Self {
            uri: uri.into(),
            version,
        }
    }

    /// A writer at the dataset's current version: what a writer holds after it has read
    /// the dataset (for example to find its logical head).
    ///
    /// # Errors
    ///
    /// [`AppendError::Lance`] when the dataset exists but cannot be opened.
    pub async fn open(uri: impl Into<String>) -> Result<Self, AppendError> {
        let uri = uri.into();
        let version = current(&uri).await?;
        Ok(Self { uri, version })
    }

    /// The dataset's location.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// The dataset version this writer last saw (the physical commit counter, never a
    /// logical version).
    pub fn version(&self) -> Option<u64> {
        self.version
    }

    /// Store `batch` as one Lance commit and return the version it made.
    ///
    /// # Errors
    ///
    /// - [`AppendError::StaleWriter`] if the dataset is not at the version this writer
    ///   last saw; nothing is written.
    /// - [`AppendError::Race`] if the commit landed as another version than the next one;
    ///   the batch is stored, the writer's version is not advanced.
    /// - [`AppendError::Lance`] if Lance refuses the write (for example a batch whose
    ///   schema differs from the dataset's).
    pub async fn append(&mut self, batch: RecordBatch) -> Result<u64, AppendError> {
        let found = current(&self.uri).await?;
        if found != self.version {
            return Err(AppendError::StaleWriter {
                expected: self.version,
                found,
            });
        }
        let schema = batch.schema();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let mode = if found.is_some() {
            WriteMode::Append
        } else {
            WriteMode::Create
        };
        let params = WriteParams {
            mode,
            ..WriteParams::default()
        };
        let ds = match Dataset::write(reader, self.uri.as_str(), Some(params)).await {
            Ok(ds) => ds,
            // Another writer created the dataset after the check above.
            Err(lance::Error::DatasetAlreadyExists { .. }) => {
                return Err(AppendError::StaleWriter {
                    expected: None,
                    found: current(&self.uri).await.ok().flatten(),
                })
            }
            Err(e) => return Err(e.into()),
        };
        let made = ds.version().version;
        let next = next_version(found);
        if made != next {
            return Err(AppendError::Race {
                expected: next,
                found: made,
            });
        }
        self.version = Some(made);
        Ok(made)
    }
}
