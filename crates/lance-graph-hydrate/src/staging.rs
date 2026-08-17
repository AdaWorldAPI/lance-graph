//! The shared staging-name uniqueness helper for [`crate::copy::hydrate_dir`]
//! and [`crate::file::hydrate_file`].
//!
//! Extracted during a 5+3 hardening council (2026-08-17): both functions
//! independently derived a staging suffix from `pid + SystemTime::now().
//! as_nanos()`, which is NOT unique within one process — two async tasks in
//! the same process reaching the derivation in the same clock tick (coarse
//! on some platforms) would compute the identical staging path and interleave
//! writes into what each believes is its own private directory, which is
//! then published as a "complete" artifact. A per-process, monotonically
//! increasing counter closes this regardless of clock resolution.

use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A staging-name suffix unique within this process, for the lifetime of the
/// process: `<pid>-<counter>-<nanos>`. The counter alone would suffice for
/// uniqueness; the pid and nanos are kept so a leaked `.hydrating-*`/`.part-*`
/// directory found later on disk is still identifiable (which process, roughly
/// when) without needing the counter's meaning explained.
pub(crate) fn staging_suffix() -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{n}-{nanos}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_suffix_is_unique_across_many_calls_in_one_tick() {
        // Falsifies the exact defect the council found: pid+nanos alone can
        // collide within one clock tick. A tight loop is the adversarial case.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let s = staging_suffix();
            assert!(seen.insert(s.clone()), "duplicate staging suffix: {s}");
        }
    }
}
