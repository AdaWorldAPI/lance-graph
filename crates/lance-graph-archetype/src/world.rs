//! `World` — archetype meta-state carrier.
//!
//! Per ADR-0001 Decision 1, `World` is the archetype-side meta-state
//! that pairs a Lance dataset URI with a logical tick counter. This is
//! DISTINCT from the runtime blackboard
//! (`lance_graph_contract::a2a_blackboard::Blackboard`), which carries
//! per-round expert entries. The pairing is intentional — see the
//! mapping table in the plan's Architecture notes section.
//!
//! Stub-only at this stage: `fork` and `at_tick` return
//! `ArchetypeError::Unimplemented` until DU-2.8 wires them to
//! `lance::checkout(branch)` / dataset version pinning respectively.
//! Docstring anchors tie to ADR-0001 §61-72 (dataset branching) and
//! §95 (tick semantics).

use crate::error::ArchetypeError;

/// Archetype meta-state: a dataset URI and a monotonic tick counter.
///
/// The dataset URI points at the Lance dataset that backs this world's
/// archetype storage; it is a `String` placeholder on purpose — wiring
/// to an actual `lance::Dataset` is DU-2.8, deliberately not on this
/// PR (see plan's Non-goals section).
#[derive(Debug, Clone)]
pub struct World {
    /// Logical tick counter. Starts at 0 and advances by 1 per `tick()`.
    /// Not related to wall-clock time; archetype Processors may fire
    /// multiple times within a single host cycle.
    tick: u64,

    /// URI of the backing Lance dataset (scheme + path). Today this is
    /// opaque — the `fork`/`at_tick` methods that would interpret it
    /// are stubs. Kept pub(crate)-readable via `dataset_uri()` so that
    /// downstream tests can assert round-trips.
    dataset_uri: String,
}

impl World {
    /// Construct a new world at tick 0 pinned to the given dataset URI.
    /// No I/O is performed; the URI is stored verbatim.
    pub fn new(dataset_uri: impl Into<String>) -> Self {
        Self {
            tick: 0,
            dataset_uri: dataset_uri.into(),
        }
    }

    /// Advance the tick counter by 1. Returns the new tick value.
    /// Overflow is not checked at the scaffold stage — any realistic
    /// workload will rotate ticks via `at_tick` well before u64 wraps.
    pub fn tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    /// Read the current tick without advancing.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Read the dataset URI this world is pinned to.
    pub fn dataset_uri(&self) -> &str {
        &self.dataset_uri
    }

    /// Fork this world onto a new dataset branch. Per ADR-0001 §61-72,
    /// this will call `lance::checkout(branch)` and return a fresh
    /// `World` pinned to the branch HEAD.
    ///
    /// **Not implemented yet.** DU-2.8 will wire the Lance call; today
    /// returns `ArchetypeError::Unimplemented { method: "World::fork" }`.
    pub fn fork(&self, _branch: &str) -> Result<World, ArchetypeError> {
        Err(ArchetypeError::Unimplemented { method: "World::fork" })
    }

    /// Rewind (or fast-forward) this world to a specific tick. Per
    /// ADR-0001 §95, this will pin the Lance dataset version that
    /// corresponds to `tick`.
    ///
    /// **Not implemented yet.** DU-2.8 will wire the dataset-version
    /// lookup; today returns
    /// `ArchetypeError::Unimplemented { method: "World::at_tick" }`.
    pub fn at_tick(&self, _tick: u64) -> Result<World, ArchetypeError> {
        Err(ArchetypeError::Unimplemented { method: "World::at_tick" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_world_is_at_tick_zero() {
        let w = World::new("lance://tmp/archetype");
        assert_eq!(w.current_tick(), 0);
        assert_eq!(w.dataset_uri(), "lance://tmp/archetype");
    }

    #[test]
    fn tick_increments() {
        let mut w = World::new("lance://tmp/archetype");
        assert_eq!(w.tick(), 1);
        assert_eq!(w.tick(), 2);
        assert_eq!(w.current_tick(), 2);
    }

    #[test]
    fn fork_returns_unimplemented() {
        let w = World::new("lance://tmp/archetype");
        let err = w.fork("experiment").unwrap_err();
        match err {
            ArchetypeError::Unimplemented { method } => {
                assert_eq!(method, "World::fork");
            }
            other => panic!("expected Unimplemented, got {other:?}"),
        }
    }

    #[test]
    fn at_tick_returns_unimplemented() {
        let w = World::new("lance://tmp/archetype");
        let err = w.at_tick(42).unwrap_err();
        match err {
            ArchetypeError::Unimplemented { method } => {
                assert_eq!(method, "World::at_tick");
            }
            other => panic!("expected Unimplemented, got {other:?}"),
        }
    }
}
