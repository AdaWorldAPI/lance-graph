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
    /// the substrate call is `lance::checkout(branch)` followed by
    /// writing to a new dataset path.
    ///
    /// This crate intentionally does NOT depend on `lance` directly
    /// (would force every archetype consumer to pull arrow + lance +
    /// datafusion). Instead, we return a descriptor of the fork
    /// (new dataset URI = parent URI + `?branch=<name>`) plus a
    /// reset tick counter. The downstream consumer (typically
    /// `lance-graph-cognitive::world::ScenarioWorldImpl`) is
    /// responsible for invoking `VersionedGraph::tag_version(name, ...)`
    /// against Lance to materialize the branch.
    ///
    /// The naming convention `<uri>?branch=<name>` is opaque to this
    /// crate and only meaningful to the downstream resolver. This keeps
    /// the archetype crate a thin meta-state carrier per ADR-0001.
    pub fn fork(&self, branch: &str) -> Result<World, ArchetypeError> {
        if branch.is_empty() {
            return Err(ArchetypeError::InvalidBranch);
        }
        let separator = if self.dataset_uri.contains('?') { "&" } else { "?" };
        Ok(World {
            tick: 0,
            dataset_uri: format!("{}{}branch={}", self.dataset_uri, separator, branch),
        })
    }

    /// Rewind (or fast-forward) this world to a specific tick. Per
    /// ADR-0001 §95, the substrate call is
    /// `Dataset::checkout_version(tick)`.
    ///
    /// Same dependency-decoupling argument as `fork`: this crate stays
    /// lance-free. Returns a new `World` with the tick set; the
    /// downstream resolver translates `tick → Lance version` via
    /// `VersionedGraph::at_version`.
    ///
    /// Returns `InvalidTick` if `tick > self.tick` (cannot fast-forward
    /// past the current observation).
    pub fn at_tick(&self, tick: u64) -> Result<World, ArchetypeError> {
        if tick > self.tick {
            return Err(ArchetypeError::InvalidTick { requested: tick, current: self.tick });
        }
        Ok(World {
            tick,
            dataset_uri: self.dataset_uri.clone(),
        })
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
    fn fork_appends_branch_query() {
        let w = World::new("lance://tmp/archetype");
        let forked = w.fork("experiment").expect("fork should succeed");
        assert_eq!(forked.dataset_uri(), "lance://tmp/archetype?branch=experiment");
        assert_eq!(forked.current_tick(), 0);
    }

    #[test]
    fn fork_uses_ampersand_when_query_already_present() {
        let w = World::new("lance://tmp/archetype?tenant=acme");
        let forked = w.fork("scenario_a").expect("fork should succeed");
        assert_eq!(
            forked.dataset_uri(),
            "lance://tmp/archetype?tenant=acme&branch=scenario_a"
        );
    }

    #[test]
    fn fork_rejects_empty_branch_name() {
        let w = World::new("lance://tmp/archetype");
        let err = w.fork("").unwrap_err();
        assert!(matches!(err, ArchetypeError::InvalidBranch));
    }

    #[test]
    fn at_tick_rewinds_within_range() {
        let mut w = World::new("lance://tmp/archetype");
        w.tick();
        w.tick();
        w.tick(); // tick = 3
        let past = w.at_tick(1).expect("at_tick should succeed");
        assert_eq!(past.current_tick(), 1);
        assert_eq!(past.dataset_uri(), "lance://tmp/archetype");
        // Original is untouched.
        assert_eq!(w.current_tick(), 3);
    }

    #[test]
    fn at_tick_at_current_is_identity() {
        let mut w = World::new("lance://tmp/archetype");
        w.tick();
        let same = w.at_tick(1).expect("at_tick(current) should succeed");
        assert_eq!(same.current_tick(), 1);
    }

    #[test]
    fn at_tick_rejects_future() {
        let w = World::new("lance://tmp/archetype");
        let err = w.at_tick(42).unwrap_err();
        match err {
            ArchetypeError::InvalidTick { requested, current } => {
                assert_eq!(requested, 42);
                assert_eq!(current, 0);
            }
            other => panic!("expected InvalidTick, got {other:?}"),
        }
    }
}
