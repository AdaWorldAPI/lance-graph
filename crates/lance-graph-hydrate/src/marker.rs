//! The warm-marker skip-rehash mechanism, generalized from q2's own
//! invention (`cockpit-server/src/osm_slab_hydrate.rs`'s
//! `trusted_via_marker`/`write_marker`/`stat_identity`) — not present
//! anywhere in lance-graph's own doctrine before this crate.
//!
//! A hydrated artifact is expensive to re-verify (checksum a multi-GB
//! directory) but cheap to *identify* (mtime + length of its files). The
//! marker records the identity a hydration validated; a later caller trusts
//! the local copy without re-hashing as long as the identity is unchanged.
//! This is a PERFORMANCE optimization, not a correctness mechanism — if the
//! marker is stale or missing, the caller falls back to a real re-verify
//! (never to "assume trusted").

use std::fs;
use std::io;
use std::path::Path;
use std::time::SystemTime;

/// The (mtime, length) identity of a file, used as a cheap stand-in for a
/// full re-hash. Two reads of the same untouched file always agree; any
/// write, truncate, or replace changes at least one field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatIdentity {
    pub mtime_nanos: i128,
    pub len: u64,
}

/// Reads the (mtime, length) identity of a single file.
pub fn stat_identity(path: &Path) -> io::Result<StatIdentity> {
    let meta = fs::metadata(path)?;
    let mtime = meta.modified()?;
    let nanos = mtime
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or_else(|e| -(e.duration().as_nanos() as i128));
    Ok(StatIdentity {
        mtime_nanos: nanos,
        len: meta.len(),
    })
}

/// A marker file recording the identity a hydration last validated, so a
/// later caller can skip re-hashing an unchanged local copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarmMarker {
    pub identity: StatIdentity,
}

impl WarmMarker {
    /// Writes a marker recording `identity` at `marker_path`. The format is
    /// a single line, `<mtime_nanos> <len>` — deliberately not a serde
    /// format, since this crate has no serde dependency and the shape never
    /// needs to be anything but two integers.
    pub fn write(marker_path: &Path, identity: StatIdentity) -> io::Result<()> {
        fs::write(
            marker_path,
            format!("{} {}\n", identity.mtime_nanos, identity.len),
        )
    }

    /// Reads a previously-written marker, if any. `None` on any parse or
    /// I/O failure — a corrupt or missing marker is "not trusted", never an
    /// error a caller must handle specially.
    pub fn read(marker_path: &Path) -> Option<Self> {
        let text = fs::read_to_string(marker_path).ok()?;
        let mut parts = text.trim().split_whitespace();
        let mtime_nanos: i128 = parts.next()?.parse().ok()?;
        let len: u64 = parts.next()?.parse().ok()?;
        Some(Self {
            identity: StatIdentity { mtime_nanos, len },
        })
    }

    /// True iff a marker exists at `marker_path` AND its recorded identity
    /// matches `path`'s CURRENT on-disk identity. A caller uses this to skip
    /// an expensive re-verify; on `false`, the caller must fall back to a
    /// real check (checksum, remote round-trip) — never assume trust.
    pub fn is_trusted(marker_path: &Path, path: &Path) -> bool {
        let Some(marker) = Self::read(marker_path) else {
            return false;
        };
        let Ok(current) = stat_identity(path) else {
            return false;
        };
        marker.identity == current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn stat_identity_agrees_on_two_reads_of_an_untouched_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("f.bin");
        fs::write(&path, b"hello world").expect("write");
        let a = stat_identity(&path).expect("stat 1");
        let b = stat_identity(&path).expect("stat 2");
        assert_eq!(a, b);
    }

    #[test]
    fn stat_identity_changes_when_the_file_is_rewritten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("f.bin");
        fs::write(&path, b"v1").expect("write v1");
        let a = stat_identity(&path).expect("stat v1");
        // Ensure the mtime tick is observable on coarse-grained filesystems.
        sleep(Duration::from_millis(10));
        fs::write(&path, b"v2-longer").expect("write v2");
        let b = stat_identity(&path).expect("stat v2");
        assert_ne!(a, b, "length changed at minimum, so identity must differ");
    }

    #[test]
    fn is_trusted_true_after_write_then_read_with_no_mutation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("artifact.lance");
        let marker_path = dir.path().join("artifact.lance.warm");
        fs::write(&path, b"payload").expect("write artifact");
        let id = stat_identity(&path).expect("stat");
        WarmMarker::write(&marker_path, id).expect("write marker");
        assert!(WarmMarker::is_trusted(&marker_path, &path));
    }

    #[test]
    fn is_trusted_false_after_the_artifact_changes_underneath_the_marker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("artifact.lance");
        let marker_path = dir.path().join("artifact.lance.warm");
        fs::write(&path, b"payload-v1").expect("write v1");
        let id = stat_identity(&path).expect("stat v1");
        WarmMarker::write(&marker_path, id).expect("write marker");

        sleep(Duration::from_millis(10));
        fs::write(&path, b"payload-v2-different-length").expect("write v2");
        assert!(
            !WarmMarker::is_trusted(&marker_path, &path),
            "a changed artifact must never read as trusted"
        );
    }

    #[test]
    fn is_trusted_false_when_no_marker_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("artifact.lance");
        let marker_path = dir.path().join("artifact.lance.warm");
        fs::write(&path, b"payload").expect("write");
        assert!(!WarmMarker::is_trusted(&marker_path, &path));
    }

    #[test]
    fn read_returns_none_for_a_corrupt_marker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let marker_path = dir.path().join("corrupt.warm");
        fs::write(&marker_path, b"not a valid marker at all").expect("write junk");
        assert!(WarmMarker::read(&marker_path).is_none());
    }
}
