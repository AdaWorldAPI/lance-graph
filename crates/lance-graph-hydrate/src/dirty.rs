//! The `Hydrated -> Dirty` detector named in
//! `.claude/knowledge/s3-hydration-lifecycle.md` and measured (as the "§4
//! gate") in `crates/lance-graph/examples/hydration_probe.rs`: a Lance
//! dataset version read on an already-open handle is cheap relative to a
//! full dataset open, so comparing the CURRENT local version against a
//! caller-supplied prior version is a viable dirty check — never a content
//! hash of the whole directory.
//!
//! **Correction (5+3 council, 2026-08-17):** neither this module nor
//! anything else in the crate RECORDS a hydration version — `hydrate_dir`'s
//! [`crate::copy::HydrationReport`] carries only object/byte counts. The
//! caller obtains `hydrated_at_version` itself (typically: open the dataset
//! once, right after hydrating, and keep its `version_id()`); this function
//! only compares.

use crate::lifecycle::LifecycleState;
use lance::dataset::Dataset;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DirtyCheckError {
    #[error("path is not valid UTF-8: {0}")]
    InvalidPath(std::path::PathBuf),
    #[error("lance error: {0}")]
    Lance(#[from] lance::Error),
}

/// True iff the LOCAL dataset at `local_path` has a version different from
/// `hydrated_at_version` (a version the CALLER obtained and is supplying —
/// see the module doc; this crate does not track it). Per
/// `LifecycleState::can_flush`, a caller MUST treat `true` here as
/// `LifecycleState::Dirty` and refuse to flush.
pub async fn is_dirty(
    local_path: &Path,
    hydrated_at_version: u64,
) -> Result<bool, DirtyCheckError> {
    let path_str = local_path
        .to_str()
        .ok_or_else(|| DirtyCheckError::InvalidPath(local_path.to_path_buf()))?;
    let ds = Dataset::open(path_str).await?;
    Ok(ds.version_id() != hydrated_at_version)
}

/// [`is_dirty`], mapped onto [`LifecycleState`] (`Hydrated` or `Dirty` only —
/// `Absent`/`Flushed` are determined by presence, not by this function).
/// Closes the gap a 5+3 council found: `LifecycleState` appeared in no
/// function signature anywhere in the crate, so its guards could only ever
/// be consulted by a caller's own discipline, never actually produced by the
/// crate's own dirty-check. This is additive — [`is_dirty`]'s simpler bool
/// form is unchanged and still the right tool when a caller just wants the
/// boolean.
pub async fn lifecycle_of(
    local_path: &Path,
    hydrated_at_version: u64,
) -> Result<LifecycleState, DirtyCheckError> {
    Ok(if is_dirty(local_path, hydrated_at_version).await? {
        LifecycleState::Dirty
    } else {
        LifecycleState::Hydrated
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow::datatypes::{DataType, Field, Schema};
    use lance::dataset::{WriteMode, WriteParams};
    use std::sync::Arc;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
    }

    fn batch(schema: &Arc<Schema>, n: i64) -> RecordBatch {
        let ids: Int64Array = (0..n).collect::<Vec<_>>().into();
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids)]).expect("batch")
    }

    #[tokio::test]
    async fn matching_version_is_not_dirty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ds.lance");
        let schema = schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch(&schema, 5))].into_iter(), schema);
        Dataset::write(
            reader,
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .expect("write");

        let ds = Dataset::open(path.to_str().unwrap()).await.expect("open");
        let v = ds.version_id();

        assert!(
            !is_dirty(&path, v).await.expect("dirty check"),
            "a version read immediately after hydration must not be dirty"
        );
    }

    #[tokio::test]
    async fn a_later_local_write_makes_the_recorded_hydration_version_dirty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ds.lance");
        let schema = schema();
        let reader =
            RecordBatchIterator::new(vec![Ok(batch(&schema, 5))].into_iter(), schema.clone());
        Dataset::write(
            reader,
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .expect("write v1");

        let ds = Dataset::open(path.to_str().unwrap())
            .await
            .expect("open v1");
        let hydrated_at = ds.version_id();

        // A local-only append, simulating drift since hydration.
        let reader2 = RecordBatchIterator::new(vec![Ok(batch(&schema, 3))].into_iter(), schema);
        Dataset::write(
            reader2,
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .expect("append v2");

        assert!(
            is_dirty(&path, hydrated_at).await.expect("dirty check"),
            "a local append after hydration must be reported dirty"
        );
    }

    #[tokio::test]
    async fn lifecycle_of_maps_to_hydrated_and_dirty_correctly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ds.lance");
        let schema = schema();
        let reader =
            RecordBatchIterator::new(vec![Ok(batch(&schema, 5))].into_iter(), schema.clone());
        Dataset::write(
            reader,
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .expect("write v1");
        let hydrated_at = Dataset::open(path.to_str().unwrap())
            .await
            .expect("open v1")
            .version_id();

        assert_eq!(
            lifecycle_of(&path, hydrated_at).await.expect("lifecycle"),
            LifecycleState::Hydrated
        );

        let reader2 = RecordBatchIterator::new(vec![Ok(batch(&schema, 1))].into_iter(), schema);
        Dataset::write(
            reader2,
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .expect("append v2");

        assert_eq!(
            lifecycle_of(&path, hydrated_at).await.expect("lifecycle"),
            LifecycleState::Dirty
        );
    }
}
