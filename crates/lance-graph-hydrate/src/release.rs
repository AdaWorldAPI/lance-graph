//! Idle RAM/page-cache release, generalized from q2's own
//! `osm_slab_hydrate.rs::advise_dontneed` (also not present anywhere in
//! lance-graph's own doctrine before this crate). This releases the OS
//! page-cache pages backing a hydrated artifact WITHOUT deleting the file —
//! the artifact stays `Hydrated` on disk; only the kernel's readahead cache
//! for it is dropped. Deleting the file is a different operation and is
//! deliberately out of scope here (a caller wanting eviction-by-deletion
//! should reason about `LifecycleState::can_release` first, then delete via
//! ordinary filesystem calls).
//!
//! **Safe on an in-flight [`crate::copy::hydrate_dir`] staging directory**
//! (a caller releasing a parent directory will touch `.hydrating-*`
//! siblings): `POSIX_FADV_DONTNEED` only drops clean page-cache pages, so it
//! cannot corrupt or lose a concurrent write in progress. Expected, not an
//! error, if it happens.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Advises the kernel that the given open file's pages are no longer needed
/// in the page cache. A no-op hint — `POSIX_FADV_DONTNEED` can only be
/// declined, never fail unsafely, so the return value is intentionally not
/// surfaced.
#[cfg(unix)]
fn advise_dontneed_file(f: &fs::File) {
    use std::os::unix::io::AsRawFd;
    let fd = f.as_raw_fd();
    // SAFETY: `fd` is a valid, open file descriptor borrowed from `f` for the
    // duration of this call. `POSIX_FADV_DONTNEED` only advises the kernel's
    // page-cache policy — it cannot invalidate memory this process holds, and
    // a nonzero return is just the kernel declining the hint. `len = 0` means
    // "to the end of the file" per POSIX.
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// unix: open the file and advise DONTNEED; only counts on successful open
/// (matches the prior behavior this council hardened).
#[cfg(unix)]
fn release_one_file(path: &Path) -> bool {
    if let Ok(f) = fs::File::open(path) {
        advise_dontneed_file(&f);
        true
    } else {
        false
    }
}

/// non-unix: there is no portable fadvise equivalent, so opening the file
/// would be pure syscall cost for zero benefit — a council-found waste
/// (2026-08-17). Count the entry without opening it.
#[cfg(not(unix))]
fn release_one_file(_path: &Path) -> bool {
    true
}

/// Walks `dir` and advises DONTNEED for every regular file under it. Returns
/// the number of files the hint was attempted on (not a success count on
/// unix — the hint itself has no observable success/failure signal by
/// design; only the `File::open` that precedes it can fail, and does not
/// count when it does).
///
/// **Error surface (5+3 council correction, 2026-08-17):** a missing `dir`
/// is `Ok(0)` — a legitimate state (nothing to release), not an error. Any
/// OTHER failure reading `dir` itself (permission denied, `dir` is actually
/// a file, …) now propagates as `Err` — the prior version swallowed every
/// failure uniformly, making `Ok(0)` ambiguous between "genuinely empty" and
/// "couldn't even read the tree." A nested subdirectory disappearing or
/// becoming unreadable MID-WALK still degrades to skip-and-continue (a
/// legitimate race to tolerate, unlike the root itself being unreadable).
pub fn release_dir(dir: &Path) -> io::Result<usize> {
    let root_entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };

    let mut count = 0usize;
    let mut stack: Vec<PathBuf> = Vec::new();
    visit(root_entries, &mut stack, &mut count);

    while let Some(d) = stack.pop() {
        // Nested-directory read failures are tolerated (a legitimate race —
        // see the doc comment above); only the ROOT's own read_dir, handled
        // above, is treated as a caller-facing error.
        let Ok(read_dir) = fs::read_dir(&d) else {
            continue;
        };
        visit(read_dir, &mut stack, &mut count);
    }
    Ok(count)
}

fn visit(read_dir: fs::ReadDir, stack: &mut Vec<PathBuf>, count: &mut usize) {
    for entry in read_dir.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            stack.push(entry.path());
            continue;
        }
        if release_one_file(&entry.path()) {
            *count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_dir_counts_nested_regular_files_and_leaves_them_in_place() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("a.lance"), b"data-a").expect("write a");
        let nested = dir.path().join("_transactions");
        fs::create_dir_all(&nested).expect("nested dir");
        fs::write(nested.join("0-uuid.txn"), b"data-b").expect("write b");

        let count = release_dir(dir.path()).expect("release");
        assert_eq!(count, 2, "must visit both the top-level and nested file");

        // The hint never deletes anything — content must be unchanged.
        assert_eq!(
            fs::read(dir.path().join("a.lance")).expect("re-read a"),
            b"data-a"
        );
        assert_eq!(
            fs::read(nested.join("0-uuid.txn")).expect("re-read b"),
            b"data-b"
        );
    }

    #[test]
    fn release_dir_on_a_missing_directory_returns_zero_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("does-not-exist");
        let count = release_dir(&missing).expect("release of missing dir");
        assert_eq!(count, 0);
    }

    /// The two-sided pair the missing-directory test above needed (a council
    /// finding, 2026-08-17): before this fix `release_dir` could NEVER
    /// return `Err`, so the missing-directory test above could not
    /// distinguish "correctly handles absence" from "swallows every
    /// failure." This proves the function CAN fail on a real, different
    /// kind of unreadable root.
    #[test]
    fn release_dir_on_a_path_that_is_a_file_not_a_directory_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("not-a-directory.txt");
        fs::write(&file_path, b"x").expect("write");

        let err = release_dir(&file_path)
            .expect_err("a file path is not a missing directory and must error");
        assert_ne!(
            err.kind(),
            io::ErrorKind::NotFound,
            "must be a real read_dir failure, not silently mapped to the missing-dir case"
        );
    }
}
