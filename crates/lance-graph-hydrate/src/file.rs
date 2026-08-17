//! Single-file hydration: checksum-pinned download with the same
//! hydrate-aside/publish-by-rename discipline as [`crate::copy::hydrate_dir`]
//! — shared via `crate::publish::publish_by_rename`, see that module's
//! doc for the mechanism's doctrine citation, provenance, and why this
//! function's pre-rename re-check (its real defense against POSIX's
//! file-onto-file silent-clobber rename semantics) now lives there —
//! generalized from q2's `osm_slab_hydrate.rs::download_verified` (the
//! `.part` + atomic-rename sidecar pattern for one artifact rather than a
//! whole Lance directory — e.g. a `SHA256SUMS` manifest, a single baked
//! `.soa`/`.chains`/`.books` sidecar).
//!
//! The SHA-256 pin passed here IS the doctrine's idempotency-boundary
//! condition (a) (a pinned source version) for the single-file case —
//! [`crate::copy::hydrate_dir`] has no equivalent and names condition (a) as
//! the caller's responsibility instead.

use crate::publish::{publish_by_rename, remove_staging, PublishError, StagingKind};
use crate::staging::staging_suffix;
use futures::TryStreamExt;
use object_store::{path::Path as ObjPath, ObjectStore};
use sha2::{Digest, Sha256};
use std::path::{Path as FsPath, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HydrateFileError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("object store error: {0}")]
    Store(#[from] object_store::Error),
    #[error("destination already exists, refusing to overwrite: {0}")]
    AlreadyPublished(PathBuf),
    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
}

/// Downloads ONE object to `publish_path`, verifying its SHA-256 against
/// `expected_sha256_hex` before publishing. Uses a `.part` sibling file +
/// atomic rename: a reader of `publish_path` never observes a partially
/// written or unverified file.
///
/// Returns `Err(AlreadyPublished)` without touching the network if
/// `publish_path` already exists — same idempotency-boundary rule as
/// [`crate::copy::hydrate_dir`].
pub async fn hydrate_file(
    store: &dyn ObjectStore,
    remote_object: &ObjPath,
    publish_path: &FsPath,
    expected_sha256_hex: &str,
) -> Result<(), HydrateFileError> {
    if publish_path.exists() {
        return Err(HydrateFileError::AlreadyPublished(
            publish_path.to_path_buf(),
        ));
    }
    let parent = publish_path.parent().unwrap_or_else(|| FsPath::new("."));
    tokio::fs::create_dir_all(parent).await?;

    let part_path = parent.join(format!(
        "{}.part-{}",
        publish_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("artifact"),
        staging_suffix(),
    ));

    // Every path from here on cleans up `part_path` before returning — a
    // 5+3 council found the fetch/write I/O-error path (as opposed to the
    // already-handled checksum-mismatch path) leaked it (2026-08-17).
    let fetch_result: Result<Sha256, HydrateFileError> = async {
        let mut hasher = Sha256::new();
        let mut file = tokio::fs::File::create(&part_path).await?;
        use tokio::io::AsyncWriteExt;
        let get_result = store.get(remote_object).await?;
        let mut stream = get_result.into_stream();
        while let Some(chunk) = stream.try_next().await? {
            hasher.update(&chunk);
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        Ok(hasher)
    }
    .await;

    let hasher = match fetch_result {
        Ok(h) => h,
        Err(e) => {
            let _ = remove_staging(&part_path, StagingKind::File).await;
            return Err(e);
        }
    };

    let actual = hex_lower(&hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_sha256_hex) {
        let _ = remove_staging(&part_path, StagingKind::File).await;
        return Err(HydrateFileError::ChecksumMismatch {
            expected: expected_sha256_hex.to_string(),
            actual,
        });
    }

    match publish_by_rename(&part_path, publish_path, StagingKind::File).await {
        Ok(()) => Ok(()),
        Err(PublishError::AlreadyPublished) => Err(HydrateFileError::AlreadyPublished(
            publish_path.to_path_buf(),
        )),
        Err(PublishError::Io(e)) => Err(e.into()),
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;

    fn store_at(root: &FsPath) -> Arc<dyn ObjectStore> {
        std::fs::create_dir_all(root).expect("remote root");
        Arc::new(LocalFileSystem::new_with_prefix(root).expect("local object store"))
    }

    fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex_lower(&hasher.finalize())
    }

    #[tokio::test]
    async fn hydrate_file_publishes_a_correctly_checksummed_object() {
        let remote_tmp = tempfile::tempdir().expect("remote tempdir");
        let store = store_at(remote_tmp.path());
        let payload = b"berlin.soa payload bytes".to_vec();
        store
            .put(&ObjPath::from("berlin.soa"), payload.clone().into())
            .await
            .expect("put");

        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let publish_path = local_tmp.path().join("berlin.soa");

        hydrate_file(
            &*store,
            &ObjPath::from("berlin.soa"),
            &publish_path,
            &sha256_hex(&payload),
        )
        .await
        .expect("hydrate_file");

        assert_eq!(
            std::fs::read(&publish_path).expect("read published file"),
            payload
        );
        // No stray `.part-*` sidecar left behind.
        let leftovers: Vec<_> = std::fs::read_dir(local_tmp.path())
            .expect("read local tempdir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect();
        assert_eq!(leftovers.len(), 1, "only the published file should remain: {leftovers:?}");
    }

    #[tokio::test]
    async fn hydrate_file_rejects_a_checksum_mismatch_and_leaves_no_partial_file() {
        let remote_tmp = tempfile::tempdir().expect("remote tempdir");
        let store = store_at(remote_tmp.path());
        let payload = b"corrupted-in-transit".to_vec();
        store
            .put(&ObjPath::from("x.soa"), payload.into())
            .await
            .expect("put");

        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let publish_path = local_tmp.path().join("x.soa");

        let err = hydrate_file(
            &*store,
            &ObjPath::from("x.soa"),
            &publish_path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        )
        .await
        .expect_err("must reject the wrong checksum");
        assert!(matches!(err, HydrateFileError::ChecksumMismatch { .. }));
        assert!(!publish_path.exists(), "no file should be published on mismatch");
        let leftovers: Vec<_> = std::fs::read_dir(local_tmp.path())
            .expect("read local tempdir")
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            leftovers.is_empty(),
            "the rejected .part file must be cleaned up: {leftovers:?}"
        );
    }

    #[tokio::test]
    async fn hydrate_file_refuses_to_overwrite_an_existing_destination() {
        let remote_tmp = tempfile::tempdir().expect("remote tempdir");
        let store = store_at(remote_tmp.path());
        store
            .put(&ObjPath::from("x.soa"), b"v1".to_vec().into())
            .await
            .expect("put");

        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let publish_path = local_tmp.path().join("x.soa");
        std::fs::write(&publish_path, b"already-here").expect("pre-create");

        let err = hydrate_file(
            &*store,
            &ObjPath::from("x.soa"),
            &publish_path,
            &sha256_hex(b"v1"),
        )
        .await
        .expect_err("must refuse an existing destination");
        assert!(matches!(err, HydrateFileError::AlreadyPublished(_)));
        assert_eq!(
            std::fs::read(&publish_path).expect("unchanged"),
            b"already-here"
        );
    }
}
