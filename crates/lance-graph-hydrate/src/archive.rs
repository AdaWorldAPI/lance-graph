//! The `absent -> hydrated` edge for a dataset shipped as ONE zip object.
//!
//! # Why a single archive, and why zip
//!
//! [`crate::copy::hydrate_dir`] hydrates a dataset that lives in the object
//! store as a TREE of objects: list, then get each one. That is the right
//! shape when the store IS the dataset's home. It is the wrong shape for
//! *distribution*, where the artifact is a versioned, checksum-pinned release:
//! a tree has no single identity to pin, no atomic publish upstream, and its
//! listing can interleave with a producer's write.
//!
//! Operator ruling, 2026-08-22: *"bitte als zip, nicht dass wir ein Verzeichnis
//! mit einzelnen Dateien shippen"* — a shipped dataset travels as one
//! container, not as loose files.
//!
//! Zip rather than tar, concretely: a zip ends with a **central directory**, so
//! a reader can enumerate entries and seek to any one of them without touching
//! the rest. A tar has no index; every lookup is a sequential scan. For a
//! multi-hundred-megabyte dataset that is the difference between reading one
//! file and reading all of them — and it is what makes a *partial* or
//! *verifying* read possible at all.
//!
//! # What this composes, and what it adds
//!
//! Nothing here re-implements a mechanism this crate already has:
//!
//! - the checksum-pinned single-object fetch is [`crate::file::hydrate_file`]
//!   (stream + hash + `.part` + rename), so the archive's bytes are verified
//!   before a single entry is read;
//! - the publish is [`crate::publish::publish_by_rename`] with
//!   [`StagingKind::Dir`], so the directory lands with one atomic rename and a
//!   concurrent reader sees either nothing or the whole dataset.
//!
//! What is new is the middle: expanding one verified container into a
//! directory tree inside private staging, under a **containment rule** —
//! every entry must live under the declared `root`, with no `..` and no
//! absolute path. A zip is an untrusted index of paths; an extractor that
//! trusts it writes wherever the archive says (Zip Slip). This one refuses the
//! whole archive on the first entry that leaves `root`, before anything is
//! published, because a dataset with one file missing is not a partial
//! success — it is an unopenable dataset that `count_rows` would report as a
//! wrong-sized table.
//!
//! # The idempotency boundary, unchanged
//!
//! Same two conditions as the rest of the crate (doctrine
//! `.claude/knowledge/s3-hydration-lifecycle.md` §4a): (a) a pinned source —
//! here the `expected_sha256_hex` argument IS condition (a), since one object
//! with one digest is exactly a pinned version; (b) an empty/uncontested
//! destination, enforced by the entry check and again at rename time.

use crate::file::{hydrate_file, HydrateFileError};
use crate::publish::{publish_by_rename, remove_staging, PublishError, StagingKind};
use crate::staging::staging_suffix;
use object_store::{path::Path as ObjPath, ObjectStore};
use std::path::{Component, Path as FsPath, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HydrateArchiveError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("object store error: {0}")]
    Store(#[from] object_store::Error),
    #[error("destination already exists, refusing to overwrite: {0}")]
    AlreadyPublished(PathBuf),
    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("archive error: {0}")]
    Archive(String),
    /// A Zip-Slip refusal: the named entry does not live under the declared
    /// root. Nothing was published.
    #[error("archive entry {entry} is outside {root}/ — refusing to unpack")]
    EscapingEntry { entry: String, root: String },
    /// The archive unpacked without error but carried no FILES under `root` —
    /// a tree of empty directories would publish an unopenable dataset.
    #[error("archive carries no files under {root}/")]
    NoFiles { root: String },
}

/// What a hydration moved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArchiveReport {
    /// Files extracted (directory entries are not counted).
    pub files: usize,
    /// Uncompressed bytes across those files.
    pub bytes: u64,
}

/// Hydrate a dataset directory from ONE checksum-pinned zip object.
///
/// `root` is the single top-level directory the archive is expected to
/// contain (e.g. `"all_lanes.lance"`); it is what gets published AS
/// `publish_dir`, so the caller controls the local name independently of the
/// archive's.
///
/// Returns `Err(AlreadyPublished)` **without touching the network** when
/// `publish_dir` already exists — this function performs the
/// `Absent -> Hydrated` transition only; it never merges into or overwrites an
/// existing local dataset.
pub async fn hydrate_archive(
    store: &dyn ObjectStore,
    remote_object: &ObjPath,
    publish_dir: &FsPath,
    expected_sha256_hex: &str,
    root: &str,
) -> Result<ArchiveReport, HydrateArchiveError> {
    if publish_dir.exists() {
        return Err(HydrateArchiveError::AlreadyPublished(
            publish_dir.to_path_buf(),
        ));
    }
    let parent = publish_dir.parent().unwrap_or_else(|| FsPath::new("."));
    tokio::fs::create_dir_all(parent).await?;

    // Staging sits beside the destination, so the publish rename never
    // crosses a filesystem boundary. One suffix for both, so a crashed run
    // leaves at most one identifiable pair behind.
    let suffix = staging_suffix();
    let archive_path = parent.join(format!(".hydrate-archive-{suffix}.zip"));
    let staging_dir = parent.join(format!(".hydrate-staging-{suffix}"));

    // 1. Verified bytes on disk. `hydrate_file` owns the pin.
    if let Err(e) = hydrate_file(store, remote_object, &archive_path, expected_sha256_hex).await {
        return Err(match e {
            HydrateFileError::Io(e) => HydrateArchiveError::Io(e),
            HydrateFileError::Store(e) => HydrateArchiveError::Store(e),
            HydrateFileError::AlreadyPublished(p) => HydrateArchiveError::AlreadyPublished(p),
            HydrateFileError::ChecksumMismatch { expected, actual } => {
                HydrateArchiveError::ChecksumMismatch { expected, actual }
            }
        });
    }

    // 2. Expand into private staging. Every failure from here on removes both
    //    the archive and the staging tree before returning.
    let unpack = {
        let archive_path = archive_path.clone();
        let staging_dir = staging_dir.clone();
        let root = root.to_string();
        tokio::task::spawn_blocking(move || unpack_zip(&archive_path, &staging_dir, &root)).await
    };
    let cleanup = |e: HydrateArchiveError| async {
        let _ = remove_staging(&staging_dir, StagingKind::Dir).await;
        let _ = remove_staging(&archive_path, StagingKind::File).await;
        e
    };
    let report = match unpack {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return Err(cleanup(e).await),
        Err(join) => {
            return Err(cleanup(HydrateArchiveError::Archive(format!(
                "unpack task failed: {join}"
            )))
            .await)
        }
    };

    // 3. One rename publishes. `publish_by_rename` removes `staging` itself on
    //    failure; the archive is ours to clean either way.
    let staged_root = staging_dir.join(root);
    let published = publish_by_rename(&staged_root, publish_dir, StagingKind::Dir).await;
    let _ = remove_staging(&archive_path, StagingKind::File).await;
    match published {
        Ok(()) => {
            // The now-empty staging parent is not the published artifact.
            let _ = remove_staging(&staging_dir, StagingKind::Dir).await;
            Ok(report)
        }
        Err(PublishError::AlreadyPublished) => {
            let _ = remove_staging(&staging_dir, StagingKind::Dir).await;
            Err(HydrateArchiveError::AlreadyPublished(
                publish_dir.to_path_buf(),
            ))
        }
        Err(PublishError::Io(e)) => {
            let _ = remove_staging(&staging_dir, StagingKind::Dir).await;
            Err(HydrateArchiveError::Io(e))
        }
    }
}

/// Expand every entry into `staging`, refusing the archive on the first entry
/// that does not live under `root`.
///
/// Deliberately NOT `ZipArchive::extract`: that method writes what the archive
/// names, and this function's contract is that it writes only what lives under
/// one declared root. The check runs on EVERY entry before any of it is
/// written, so a refusal leaves nothing half-placed.
fn unpack_zip(
    archive: &FsPath,
    staging: &FsPath,
    root: &str,
) -> Result<ArchiveReport, HydrateArchiveError> {
    let f = std::fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(f)
        .map_err(|e| HydrateArchiveError::Archive(format!("open zip: {e}")))?;

    // Validate the whole index FIRST, off the central directory — the reason
    // a zip is worth shipping. A tar would have to be scanned to learn this,
    // by which point entries are already streaming past.
    for i in 0..zip.len() {
        let entry = zip
            .by_index_raw(i)
            .map_err(|e| HydrateArchiveError::Archive(format!("read entry {i}: {e}")))?;
        let name = entry.name().to_string();
        // `enclosed_name` is zip's own traversal guard; the root check is the
        // narrower contract on top of it. Both must hold.
        let ok = entry
            .enclosed_name()
            .map(|p| under(&p, root))
            .unwrap_or(false);
        if !ok {
            return Err(HydrateArchiveError::EscapingEntry {
                entry: name,
                root: root.to_string(),
            });
        }
    }

    std::fs::create_dir_all(staging)?;
    let mut files = 0usize;
    let mut bytes = 0u64;
    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| HydrateArchiveError::Archive(format!("read entry {i}: {e}")))?;
        let rel = entry
            .enclosed_name()
            .ok_or_else(|| HydrateArchiveError::EscapingEntry {
                entry: entry.name().to_string(),
                root: root.to_string(),
            })?;
        let dest = staging.join(&rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&dest)?;
            continue;
        }
        if let Some(p) = dest.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut out = std::fs::File::create(&dest)?;
        bytes += std::io::copy(&mut entry, &mut out)?;
        files += 1;
    }
    if files == 0 {
        return Err(HydrateArchiveError::NoFiles {
            root: root.to_string(),
        });
    }
    Ok(ArchiveReport { files, bytes })
}

/// True iff `path` is a relative path whose first component is exactly `root`
/// and which contains no `..`.
fn under(path: &FsPath, root: &str) -> bool {
    let mut comps = path.components();
    if comps.next() != Some(Component::Normal(root.as_ref())) {
        return false;
    }
    comps.all(|c| matches!(c, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::local::LocalFileSystem;
    use sha2::{Digest, Sha256};
    use std::io::Write as _;
    use std::sync::Arc;

    fn store_at(root: &FsPath) -> Arc<dyn ObjectStore> {
        Arc::new(LocalFileSystem::new_with_prefix(root).expect("local object store"))
    }

    /// Build a zip from `(name, contents)` pairs; `None` means a directory.
    fn zip_with(entries: &[(&str, Option<&[u8]>)]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts: zip::write::FileOptions<'_, ()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, body) in entries {
            match body {
                Some(data) => {
                    w.start_file(*name, opts).unwrap();
                    w.write_all(data).unwrap();
                }
                None => w.add_directory(*name, opts).unwrap(),
            }
        }
        w.finish().unwrap().into_inner()
    }

    /// The shape a real Lance dataset has — dataset dir, transaction log,
    /// versions, one data file — so the happy path is not proven on a toy.
    fn realistic_zip() -> Vec<u8> {
        zip_with(&[
            ("all_lanes.lance/", None),
            ("all_lanes.lance/_transactions/", None),
            ("all_lanes.lance/_transactions/0-abc.txn", Some(b"txn0")),
            ("all_lanes.lance/_versions/", None),
            ("all_lanes.lance/_versions/1.manifest", Some(b"manifest")),
            ("all_lanes.lance/data/", None),
            ("all_lanes.lance/data/0001.lance", Some(&[7u8; 4096])),
        ])
    }

    fn sha_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Put `bytes` in the remote root as `artifact.zip`; return (store, path).
    fn remote_with(remote_root: &FsPath, bytes: &[u8]) -> (Arc<dyn ObjectStore>, ObjPath) {
        std::fs::write(remote_root.join("artifact.zip"), bytes).unwrap();
        (store_at(remote_root), ObjPath::from("artifact.zip"))
    }

    fn residue(dir: &FsPath) -> Vec<String> {
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                    .filter(|n| n.starts_with(".hydrate-"))
                    .collect()
            })
            .unwrap_or_default()
    }

    // ── CAN FIRE ────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn hydrates_the_whole_dataset_from_one_zip() {
        let remote = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        let bytes = realistic_zip();
        let (store, obj) = remote_with(remote.path(), &bytes);
        let dest = local.path().join("hydrated.lance");

        let report = hydrate_archive(
            store.as_ref(),
            &obj,
            &dest,
            &sha_hex(&bytes),
            "all_lanes.lance",
        )
        .await
        .expect("hydrate");

        assert_eq!(
            report,
            ArchiveReport {
                files: 3,
                bytes: 4 + 8 + 4096
            },
            "files counted, directories not"
        );
        assert_eq!(
            std::fs::read(dest.join("data/0001.lance")).unwrap(),
            vec![7u8; 4096],
            "the data file arrives byte-for-byte"
        );
        assert_eq!(
            std::fs::read_to_string(dest.join("_transactions/0-abc.txn")).unwrap(),
            "txn0",
            "the transaction log arrives — a dataset without it is not openable"
        );
        assert!(dest.join("_versions/1.manifest").is_file());
        // The caller's local name wins over the archive's root name.
        assert!(!local.path().join("all_lanes.lance").exists());
        assert!(
            residue(local.path()).is_empty(),
            "no staging or archive left behind: {:?}",
            residue(local.path())
        );
    }

    // ── CAN STAY SILENT — and prove it never reached the network ────────────
    #[tokio::test]
    async fn an_existing_destination_is_refused_without_fetching() {
        let remote = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        // The remote object DOES NOT EXIST. If this function fetched before
        // checking, it would fail with a store error, not AlreadyPublished.
        let store = store_at(remote.path());
        let dest = local.path().join("hydrated.lance");
        std::fs::create_dir_all(dest.join("data")).unwrap();
        std::fs::write(dest.join("data/live.lance"), b"grown-since-hydrate").unwrap();

        let err = hydrate_archive(
            store.as_ref(),
            &ObjPath::from("artifact.zip"),
            &dest,
            "00",
            "all_lanes.lance",
        )
        .await
        .expect_err("must refuse");
        assert!(
            matches!(err, HydrateArchiveError::AlreadyPublished(_)),
            "got {err:?}"
        );
        assert_eq!(
            std::fs::read_to_string(dest.join("data/live.lance")).unwrap(),
            "grown-since-hydrate",
            "a live dataset is never overwritten"
        );
    }

    // ── REFUSALS ────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn an_entry_outside_the_root_is_refused_and_publishes_nothing() {
        for escape in [
            "../escape.txt",
            "/etc/passwd",
            "other_table.lance/data/0001.lance",
            "all_lanes.lance/../../escape.txt",
        ] {
            let remote = tempfile::tempdir().unwrap();
            let local = tempfile::tempdir().unwrap();
            let bytes = zip_with(&[
                ("all_lanes.lance/", None),
                ("all_lanes.lance/data/0001.lance", Some(b"real")),
                (escape, Some(b"pwned")),
            ]);
            let (store, obj) = remote_with(remote.path(), &bytes);
            let dest = local.path().join("hydrated.lance");

            let err = hydrate_archive(
                store.as_ref(),
                &obj,
                &dest,
                &sha_hex(&bytes),
                "all_lanes.lance",
            )
            .await
            .expect_err("must refuse");
            assert!(
                matches!(err, HydrateArchiveError::EscapingEntry { .. }),
                "{escape}: got {err:?}"
            );
            assert!(!dest.exists(), "{escape}: nothing published");
            assert!(
                !local.path().join("escape.txt").exists()
                    && !local.path().parent().unwrap().join("escape.txt").exists(),
                "{escape}: the escaping entry never lands"
            );
            assert!(
                residue(local.path()).is_empty(),
                "{escape}: residue {:?}",
                residue(local.path())
            );
        }
    }

    #[tokio::test]
    async fn a_wrong_checksum_publishes_nothing() {
        let remote = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        let bytes = realistic_zip();
        let (store, obj) = remote_with(remote.path(), &bytes);
        let dest = local.path().join("hydrated.lance");

        // A VALID archive with the WRONG pin — so what refuses is the pin,
        // not the content.
        let err = hydrate_archive(
            store.as_ref(),
            &obj,
            &dest,
            &sha_hex(b"other"),
            "all_lanes.lance",
        )
        .await
        .expect_err("must refuse");
        assert!(
            matches!(err, HydrateArchiveError::ChecksumMismatch { .. }),
            "got {err:?}"
        );
        assert!(!dest.exists());
        assert!(
            residue(local.path()).is_empty(),
            "{:?}",
            residue(local.path())
        );

        // …and with the RIGHT pin the same archive DOES hydrate — so the
        // refusal above is the pin check, not an inert path.
        hydrate_archive(
            store.as_ref(),
            &obj,
            &dest,
            &sha_hex(&bytes),
            "all_lanes.lance",
        )
        .await
        .expect("hydrate with the correct pin");
        assert!(dest.join("data/0001.lance").is_file());
    }

    #[tokio::test]
    async fn an_archive_of_only_directories_is_refused() {
        // Anti-vacuity for the file counter: an all-directory tree extracts
        // without error and would publish an unopenable dataset.
        let remote = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        let bytes = zip_with(&[("all_lanes.lance/", None), ("all_lanes.lance/data/", None)]);
        let (store, obj) = remote_with(remote.path(), &bytes);
        let dest = local.path().join("hydrated.lance");

        let err = hydrate_archive(
            store.as_ref(),
            &obj,
            &dest,
            &sha_hex(&bytes),
            "all_lanes.lance",
        )
        .await
        .expect_err("must refuse");
        assert!(
            matches!(err, HydrateArchiveError::NoFiles { .. }),
            "got {err:?}"
        );
        assert!(!dest.exists());
        assert!(
            residue(local.path()).is_empty(),
            "{:?}",
            residue(local.path())
        );
    }

    #[test]
    fn under_accepts_the_root_and_rejects_traversal() {
        assert!(under(
            FsPath::new("all_lanes.lance/data/x"),
            "all_lanes.lance"
        ));
        assert!(under(FsPath::new("all_lanes.lance"), "all_lanes.lance"));
        assert!(!under(
            FsPath::new("all_lanes.lance/../x"),
            "all_lanes.lance"
        ));
        assert!(!under(FsPath::new("/all_lanes.lance/x"), "all_lanes.lance"));
        assert!(!under(FsPath::new("crystal.lance/x"), "all_lanes.lance"));
        assert!(!under(FsPath::new(""), "all_lanes.lance"));
    }
}
