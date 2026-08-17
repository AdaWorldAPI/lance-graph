//! The shared hydrate-aside/publish-by-rename TAIL for [`crate::copy::hydrate_dir`]
//! and [`crate::file::hydrate_file`].
//!
//! Extracted during a follow-up to the 2026-08-17 5+3 hardening council,
//! which found (and deliberately deferred, `ISS-HYDRATE-DIR-AND-FILE-
//! DUPLICATE-THEIR-STAGING-BODIES`, `.claude/board/ISSUES.md`) that both
//! functions carried near-identical staging → publish bodies: each with its
//! own TOCTOU window, its own cleanup ladder, and its own rename-race remap.
//! That duplication was the audit surface at risk — a fix to any of the
//! three had to be applied twice or would silently drift. This module is
//! that fix, applied once.
//!
//! **What is shared and what stays per-caller.** The FETCH half (list+get
//! many objects for a directory vs stream+hash one object for a file) is
//! genuinely different in shape and stays in `copy.rs`/`file.rs` — merging
//! it would obscure more than it clarifies. The PUBLISH half — re-check the
//! destination immediately before the rename, attempt the rename, and on
//! failure decide whether that failure IS the doctrine's idempotency-
//! boundary condition (a concurrent publisher won the race) or a genuine
//! I/O error — is identical in intent between the two callers and is what
//! lives here.
//!
//! **The two rename semantics this unifies to, uniformly (a real
//! improvement, not just deduplication).** Directory-onto-directory rename
//! on POSIX fails loudly (`ENOTEMPTY`) when the destination is non-empty;
//! file-onto-file rename SILENTLY CLOBBERS an existing destination instead
//! of failing. Before this module, `hydrate_dir` only remapped the loud
//! failure AFTER attempting the rename (no pre-check), and `hydrate_file`
//! only guarded with a pre-check (its danger case never manifests as a
//! rename error to remap). [`publish_by_rename`] does BOTH for both
//! callers: a pre-rename re-check (the file case's real defense, now also
//! narrowing the directory case's window further) and a post-rename remap
//! on failure (the directory case's real defense, harmless as a no-op
//! safety net for the file case). This strictly narrows both windows; it
//! weakens neither.
//!
//! Still a filesystem-atomicity boundary, deliberately NOT a coordination
//! protocol — no lock, no lease (doctrine `.claude/knowledge/
//! s3-hydration-lifecycle.md` §4a). The residual race (a publish landing in
//! the instant between the pre-check and the OS rename call itself) is
//! narrowed, not closed, exactly as the doctrine's own framing describes it.

use std::path::Path as FsPath;

/// Which removal (and therefore which rename failure mode) applies to a
/// given publish attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StagingKind {
    /// A staging directory (`hydrate_dir`): removed via `remove_dir_all`;
    /// a rename onto a non-empty destination fails loudly (`ENOTEMPTY`).
    Dir,
    /// A staging file (`hydrate_file`): removed via `remove_file`; a
    /// rename onto an existing destination SILENTLY CLOBBERS it on POSIX
    /// rather than failing — the pre-rename re-check is this case's real
    /// defense, not the post-rename remap.
    File,
}

/// Best-effort removal is the caller's choice, not this function's — some
/// callers must swallow a cleanup failure (the primary error already
/// dominates) and at least one must propagate it (an `Ok` return promising
/// "leaves nothing behind" must not be true only optimistically). This
/// returns the real `io::Result` so both are expressible: `let _ =
/// remove_staging(..).await;` to swallow, `remove_staging(..).await?;` to
/// propagate.
pub(crate) async fn remove_staging(staging: &FsPath, kind: StagingKind) -> std::io::Result<()> {
    match kind {
        StagingKind::Dir => tokio::fs::remove_dir_all(staging).await,
        StagingKind::File => tokio::fs::remove_file(staging).await,
    }
}

/// Why [`publish_by_rename`] failed. Both callers map this onto their own
/// error enum's `AlreadyPublished(PathBuf)` / `Io(io::Error)` variants —
/// deliberately not a shared error type, since each caller's variant
/// already carries the path (this function doesn't need to duplicate it).
#[derive(Debug)]
pub(crate) enum PublishError {
    /// `publish_path` was occupied — either at the pre-rename re-check, or
    /// the rename itself failed in a way consistent with a concurrent
    /// publisher having won the race (directory case: `ENOTEMPTY`; file
    /// case: caught by the pre-check, so this arm is defense-in-depth
    /// there). `staging` has already been removed (best-effort).
    AlreadyPublished,
    /// A genuine I/O error unrelated to the destination being occupied.
    /// `staging` has already been removed (best-effort).
    Io(std::io::Error),
}

/// The shared publish tail: re-check `publish_path` immediately before the
/// rename, rename `staging` onto it, and on any failure clean up `staging`
/// and classify the failure per [`PublishError`]. See the module doc for
/// why both the pre-check and the post-rename remap run for both
/// [`StagingKind`]s.
pub(crate) async fn publish_by_rename(
    staging: &FsPath,
    publish_path: &FsPath,
    kind: StagingKind,
) -> Result<(), PublishError> {
    if publish_path.exists() {
        let _ = remove_staging(staging, kind).await;
        return Err(PublishError::AlreadyPublished);
    }
    if let Err(e) = tokio::fs::rename(staging, publish_path).await {
        let _ = remove_staging(staging, kind).await;
        return Err(if publish_path.exists() {
            PublishError::AlreadyPublished
        } else {
            PublishError::Io(e)
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publish_by_rename_publishes_a_directory() {
        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let staging = local_tmp.path().join(".staging-dir");
        std::fs::create_dir_all(&staging).expect("staging dir");
        std::fs::write(staging.join("data.lance"), b"row-bytes").expect("staged file");
        let publish_dir = local_tmp.path().join("hydrated.lance");

        publish_by_rename(&staging, &publish_dir, StagingKind::Dir)
            .await
            .expect("publish");

        assert_eq!(
            std::fs::read(publish_dir.join("data.lance")).expect("published file"),
            b"row-bytes"
        );
        assert!(!staging.exists(), "staging dir must be gone after publish");
    }

    #[tokio::test]
    async fn publish_by_rename_publishes_a_file() {
        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let staging = local_tmp.path().join("x.part");
        std::fs::write(&staging, b"payload").expect("staged file");
        let publish_path = local_tmp.path().join("x.soa");

        publish_by_rename(&staging, &publish_path, StagingKind::File)
            .await
            .expect("publish");

        assert_eq!(std::fs::read(&publish_path).expect("published file"), b"payload");
        assert!(!staging.exists(), "staging file must be gone after publish");
    }

    #[tokio::test]
    async fn publish_by_rename_directory_race_is_reported_as_already_published_and_cleans_up() {
        // Simulates the exact race the pre-check narrows: a competing
        // publisher lands at `publish_dir` between this caller's fetch
        // completing and its own publish attempt.
        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let staging = local_tmp.path().join(".staging-dir");
        std::fs::create_dir_all(&staging).expect("staging dir");
        std::fs::write(staging.join("data.lance"), b"loser").expect("staged file");
        let publish_dir = local_tmp.path().join("hydrated.lance");
        std::fs::create_dir_all(&publish_dir).expect("winner already published");
        std::fs::write(publish_dir.join("data.lance"), b"winner").expect("winner content");

        let err = publish_by_rename(&staging, &publish_dir, StagingKind::Dir)
            .await
            .expect_err("must detect the race");
        assert!(matches!(err, PublishError::AlreadyPublished));
        assert!(!staging.exists(), "loser's staging dir must be cleaned up");
        assert_eq!(
            std::fs::read(publish_dir.join("data.lance")).expect("winner survives"),
            b"winner",
            "the winner's published content must be untouched"
        );
    }

    #[tokio::test]
    async fn publish_by_rename_file_race_is_reported_as_already_published_and_cleans_up() {
        let local_tmp = tempfile::tempdir().expect("local tempdir");
        let staging = local_tmp.path().join("x.part");
        std::fs::write(&staging, b"loser").expect("staged file");
        let publish_path = local_tmp.path().join("x.soa");
        std::fs::write(&publish_path, b"winner").expect("winner already published");

        let err = publish_by_rename(&staging, &publish_path, StagingKind::File)
            .await
            .expect_err("must detect the race");
        assert!(matches!(err, PublishError::AlreadyPublished));
        assert!(!staging.exists(), "loser's staging file must be cleaned up");
        assert_eq!(
            std::fs::read(&publish_path).expect("winner survives"),
            b"winner",
            "the winner's published content must be untouched"
        );
    }
}
