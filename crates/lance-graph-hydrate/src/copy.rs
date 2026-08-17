//! The `absent -> hydrated` edge: a raw, byte-for-byte object-store copy.
//!
//! **Provenance, split honestly (5+3 council correction, 2026-08-17):** the
//! byte-copy — never a `Dataset` scan-and-rewrite, which silently drops
//! deletion vectors, indexes, and multi-version history — is inherited from
//! and proven by this repo's own `crates/lance-graph/examples/
//! hydration_probe.rs` (its T10 gate). **The hydrate-aside/publish-by-rename
//! mechanism below is NOT from that probe** — the probe writes directly into
//! its target with no staging step at all. That mechanism is new code,
//! implementing the doctrine's §4a for the first time, proven only by this
//! crate's own tests.
//!
//! Publication follows the doctrine's **hydrate-aside, publish-by-rename**
//! mechanism (`.claude/knowledge/s3-hydration-lifecycle.md` §4a; already
//! stated in `EPIPHANIES.md`
//! `E-A-REPEATABLE-TRANSFER-IS-NOT-IDEMPOTENCE-OVER-A-MULTI-FILE-DIRECTORY-1`,
//! cited rather than restated as new): objects land in a private sibling
//! staging directory first, then ONE atomic directory rename publishes the
//! complete artifact — via `crate::publish::publish_by_rename`, shared
//! with [`crate::file::hydrate_file`] (see that module's doc for the merge
//! and why the TOCTOU-narrowing details live there now, not restated here).
//! A caller observing the publish path therefore either sees nothing (not
//! yet hydrated) or the complete artifact (hydrated) — never a partial one.
//! This is a filesystem-atomicity boundary, deliberately NOT a lock/lease
//! protocol.
//!
//! **The idempotency boundary has TWO conditions, not one** (doctrine §4a):
//! (a) a pinned source version and (b) an empty/uncontested destination.
//! This function enforces (b) (the `AlreadyPublished` refusal, including at
//! rename time, not only at entry). Condition (a) is structurally the
//! CALLER's: this function copies whatever exists under `remote_root` at
//! call time and has no way to know whether that reflects one stable source
//! snapshot or an in-flux one. (For the single-object case,
//! [`crate::file::hydrate_file`]'s checksum pin IS condition (a).)

use crate::publish::{publish_by_rename, remove_staging, PublishError, StagingKind};
use crate::staging::staging_suffix;
use futures::TryStreamExt;
use object_store::{path::Path as ObjPath, ObjectStore};
use std::path::{Path as FsPath, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HydrateError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("object store error: {0}")]
    Store(#[from] object_store::Error),
    /// The idempotency-boundary condition from the doctrine: `hydrate_dir`
    /// is sound only against an empty/uncontested destination. A caller that
    /// wants "hydrate only if absent" should check `publish_dir.exists()`
    /// first, or match on this variant.
    #[error("destination already exists, refusing to overwrite: {0}")]
    AlreadyPublished(PathBuf),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HydrationReport {
    pub objects_copied: usize,
    pub bytes_copied: u64,
}

/// Copies every object under `remote_root` to `publish_dir`, byte-for-byte.
///
/// Returns `Err(HydrateError::AlreadyPublished)` without touching the
/// filesystem if `publish_dir` already exists — this function performs the
/// `Absent -> Hydrated` transition only; it never merges into or overwrites
/// an existing local copy.
///
/// If `remote_root` has no objects, returns `Ok(HydrationReport::default())`
/// and leaves NOTHING at `publish_dir` (no empty directory) — an empty
/// directory would read as "hydrated" to a caller checking
/// `publish_dir.exists()`, which is not true of zero source objects.
pub async fn hydrate_dir(
    store: &dyn ObjectStore,
    remote_root: &ObjPath,
    publish_dir: &FsPath,
) -> Result<HydrationReport, HydrateError> {
    if publish_dir.exists() {
        return Err(HydrateError::AlreadyPublished(publish_dir.to_path_buf()));
    }
    let parent = publish_dir.parent().unwrap_or_else(|| FsPath::new("."));
    tokio::fs::create_dir_all(parent).await?;

    let leaf = publish_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("artifact");
    let staging = parent.join(format!(".hydrating-{leaf}-{}", staging_suffix()));
    tokio::fs::create_dir_all(&staging).await?;

    // Every path from here on cleans up `staging` before returning — a
    // 5+3 council found every prior error path here leaked it (2026-08-17).
    let report = match fetch_all(store, remote_root, &staging).await {
        Ok(r) => r,
        Err(e) => {
            let _ = remove_staging(&staging, StagingKind::Dir).await;
            return Err(e);
        }
    };

    if report.objects_copied == 0 {
        // On the Ok path: a cleanup failure here must NOT be swallowed — an
        // `Ok` return promises "leaves NOTHING at publish_dir", and silently
        // discarding this error would let that promise be false (a council
        // finding, 2026-08-17).
        remove_staging(&staging, StagingKind::Dir).await?;
        return Ok(report);
    }

    match publish_by_rename(&staging, publish_dir, StagingKind::Dir).await {
        Ok(()) => Ok(report),
        Err(PublishError::AlreadyPublished) => {
            Err(HydrateError::AlreadyPublished(publish_dir.to_path_buf()))
        }
        Err(PublishError::Io(e)) => Err(e.into()),
    }
}

async fn fetch_all(
    store: &dyn ObjectStore,
    remote_root: &ObjPath,
    staging: &FsPath,
) -> Result<HydrationReport, HydrateError> {
    let prefix = format!("{remote_root}/");
    let mut report = HydrationReport::default();
    let mut objects = store.list(Some(remote_root));
    while let Some(meta) = objects.try_next().await? {
        let rel = meta
            .location
            .as_ref()
            .strip_prefix(&prefix)
            .unwrap_or_else(|| meta.location.as_ref())
            .to_string();
        let bytes = store.get(&meta.location).await?.bytes().await?;
        let dest = staging.join(&rel);
        if let Some(dest_parent) = dest.parent() {
            tokio::fs::create_dir_all(dest_parent).await?;
        }
        tokio::fs::write(&dest, &bytes).await?;
        report.objects_copied += 1;
        report.bytes_copied += bytes.len() as u64;
    }
    Ok(report)
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

    #[tokio::test]
    async fn hydrate_dir_copies_every_object_byte_for_byte_including_nested_paths() {
        let remote_tmp = tempfile::tempdir().expect("remote tempdir");
        let store = store_at(remote_tmp.path());
        store
            .put(&ObjPath::from("ds/data.lance"), b"row-bytes".to_vec().into())
            .await
            .expect("put data file");
        store
            .put(
                &ObjPath::from("ds/_transactions/0-uuid.txn"),
                b"txn-bytes".to_vec().into(),
            )
            .await
            .expect("put nested txn file");

        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let publish_dir = local_tmp.path().join("hydrated.lance");

        let report = hydrate_dir(&*store, &ObjPath::from("ds"), &publish_dir)
            .await
            .expect("hydrate");

        assert_eq!(report.objects_copied, 2);
        assert_eq!(
            std::fs::read(publish_dir.join("data.lance")).expect("read copied data file"),
            b"row-bytes"
        );
        assert_eq!(
            std::fs::read(publish_dir.join("_transactions/0-uuid.txn"))
                .expect("read copied nested file"),
            b"txn-bytes"
        );
        // No staging directory left behind.
        let siblings: Vec<_> = std::fs::read_dir(local_tmp.path())
            .expect("read local tempdir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect();
        assert_eq!(siblings.len(), 1, "only the published dir should remain: {siblings:?}");
    }

    #[tokio::test]
    async fn hydrate_dir_refuses_to_overwrite_an_existing_destination() {
        let remote_tmp = tempfile::tempdir().expect("remote tempdir");
        let store = store_at(remote_tmp.path());
        store
            .put(&ObjPath::from("ds/data.lance"), b"v1".to_vec().into())
            .await
            .expect("put");

        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let publish_dir = local_tmp.path().join("hydrated.lance");
        std::fs::create_dir_all(&publish_dir).expect("pre-create destination");
        std::fs::write(publish_dir.join("sentinel"), b"do-not-touch").expect("sentinel");

        let err = hydrate_dir(&*store, &ObjPath::from("ds"), &publish_dir)
            .await
            .expect_err("must refuse an existing destination");
        assert!(matches!(err, HydrateError::AlreadyPublished(_)));
        assert_eq!(
            std::fs::read(publish_dir.join("sentinel")).expect("sentinel survives"),
            b"do-not-touch",
            "an existing destination must be left completely untouched"
        );
    }

    #[tokio::test]
    async fn hydrate_dir_of_an_empty_prefix_publishes_nothing() {
        let remote_tmp = tempfile::tempdir().expect("remote tempdir");
        let store = store_at(remote_tmp.path());

        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let publish_dir = local_tmp.path().join("hydrated.lance");

        let report = hydrate_dir(&*store, &ObjPath::from("nothing-here"), &publish_dir)
            .await
            .expect("hydrate of empty prefix");
        assert_eq!(report.objects_copied, 0);
        assert!(
            !publish_dir.exists(),
            "zero source objects must not produce an empty published dir"
        );
        let leftovers: Vec<_> = std::fs::read_dir(local_tmp.path())
            .map(|rd| rd.filter_map(|e| e.ok()).collect::<Vec<_>>())
            .unwrap_or_default();
        assert!(
            leftovers.is_empty(),
            "no staging directory should survive an empty hydrate: {leftovers:?}"
        );
    }
}
