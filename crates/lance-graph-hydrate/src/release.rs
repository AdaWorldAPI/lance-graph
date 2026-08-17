//! Idle RAM/page-cache release, generalized from q2's own
//! `osm_slab_hydrate.rs::advise_dontneed` (also not present anywhere in
//! lance-graph's own doctrine before this crate). This releases the OS
//! page-cache pages backing a hydrated artifact WITHOUT deleting the file —
//! the artifact stays `Hydrated` on disk; only the kernel's readahead cache
//! for it is dropped. Deleting the file is a different operation and is
//! deliberately out of scope here (a caller wanting eviction-by-deletion
//! should reason about `LifecycleState::can_release` first, then delete via
//! ordinary filesystem calls).

use std::fs;
use std::io;
use std::path::Path;

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

#[cfg(not(unix))]
fn advise_dontneed_file(_f: &fs::File) {
    // No-op on non-unix platforms — there is no portable fadvise equivalent,
    // and declining the hint is always a safe fallback (see cross-platform.md).
}

/// Walks `dir` and advises DONTNEED for every regular file under it. Returns
/// the number of files the hint was attempted on (not a success count — the
/// hint has no observable success/failure signal by design).
pub fn release_dir(dir: &Path) -> io::Result<usize> {
    let mut count = 0usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(read_dir) = fs::read_dir(&d) else {
            continue;
        };
        for entry in read_dir.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if let Ok(f) = fs::File::open(entry.path()) {
                advise_dontneed_file(&f);
                count += 1;
            }
        }
    }
    Ok(count)
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
}
