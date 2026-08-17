//! The four-state hydration lifecycle from
//! `.claude/knowledge/s3-hydration-lifecycle.md`: `Absent -> Hydrated ->
//! {Dirty | Flushed}`, with the single hard rule the doctrine names —
//! **flush only from Hydrated, never Dirty** — encoded as a transition
//! guard rather than left to caller discipline.

/// Where a local artifact stands relative to its remote (object-store) copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleState {
    /// No local copy exists. The only state `hydrate_dir` may start from.
    Absent,
    /// A local copy exists and (as far as the caller has checked) matches
    /// what was hydrated. The only state a flush may start from.
    Hydrated,
    /// The local copy has diverged from the version it was hydrated at
    /// (e.g. a newer Lance version was written locally since hydration).
    /// A flush from here would silently overwrite remote history with a
    /// state the doctrine never validated — the hard rule this type exists
    /// to enforce.
    Dirty,
    /// The local copy has been durably written back to the remote store.
    Flushed,
}

impl LifecycleState {
    /// `hydrate_dir` (the `Absent -> Hydrated` edge) is only sound against an
    /// empty/uncontested destination — the idempotency-boundary condition
    /// from the doctrine.
    pub fn can_hydrate(self) -> bool {
        matches!(self, LifecycleState::Absent)
    }

    /// The one hard rule: flush only from `Hydrated`, never `Dirty`.
    pub fn can_flush(self) -> bool {
        matches!(self, LifecycleState::Hydrated)
    }

    /// Idle release (fadvise DONTNEED / directory eviction) is safe once the
    /// local copy is known-durable somewhere other than page cache: either it
    /// still matches remote (`Hydrated`, nothing to lose) or it has just been
    /// written back (`Flushed`). Releasing a `Dirty` copy would drop the only
    /// durable record of a local-only change.
    pub fn can_release(self) -> bool {
        matches!(self, LifecycleState::Hydrated | LifecycleState::Flushed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_absent_can_hydrate() {
        assert!(LifecycleState::Absent.can_hydrate());
        assert!(!LifecycleState::Hydrated.can_hydrate());
        assert!(!LifecycleState::Dirty.can_hydrate());
        assert!(!LifecycleState::Flushed.can_hydrate());
    }

    #[test]
    fn only_hydrated_can_flush_dirty_is_explicitly_blocked() {
        assert!(LifecycleState::Hydrated.can_flush());
        assert!(
            !LifecycleState::Dirty.can_flush(),
            "flushing a Dirty copy would silently overwrite remote with unvalidated state"
        );
        assert!(!LifecycleState::Absent.can_flush());
        assert!(!LifecycleState::Flushed.can_flush());
    }

    #[test]
    fn release_is_safe_from_hydrated_and_flushed_but_not_dirty_or_absent() {
        assert!(LifecycleState::Hydrated.can_release());
        assert!(LifecycleState::Flushed.can_release());
        assert!(
            !LifecycleState::Dirty.can_release(),
            "releasing a Dirty copy would drop the only durable record of a local-only change"
        );
        assert!(!LifecycleState::Absent.can_release());
    }
}
