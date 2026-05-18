//! Backend provider marker traits.
//!
//! # Scope
//!
//! Markers for backend-specific Arrow providers used by the federated
//! planner. The concrete `TableProvider` impls (DataFusion) live in
//! their owner crates (e.g. `lance-graph-tikv-provider`); this module
//! provides the **zero-dep marker vocabulary** the contract crate uses
//! to talk about them without pulling in DataFusion or Arrow as deps.
//!
//! # Additive contract
//!
//! Added in 0.2.0. Pure addition — no existing surface is touched.
//!
//! # Pattern
//!
//! ```rust,ignore
//! // In lance-graph-tikv-provider:
//! use datafusion::catalog::TableProvider;
//! use lance_graph_contract::provider::{BackendId, MvccProvider};
//!
//! pub struct TikvNodeTableProvider { /* ... */ }
//!
//! impl TableProvider for TikvNodeTableProvider { /* DataFusion impl */ }
//!
//! impl MvccProvider for TikvNodeTableProvider {
//!     fn backend(&self) -> BackendId { BackendId::Tikv }
//!     fn snapshot_ts(&self) -> Option<u64> { self.snapshot_ts }
//! }
//! ```
//!
//! The federated planner then uses `MvccProvider::snapshot_ts()` to
//! propagate a consistent snapshot across engines, regardless of which
//! concrete `TableProvider` is in play.

/// Which storage backend an Arrow provider sources from.
///
/// Adding a new backend is an additive enum variant — `#[non_exhaustive]`
/// means consumers must use `_ =>` in exhaustive matches, so a new
/// variant does not break their code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendId {
    /// Embedded LSM KV (RocksDB / SurrealKV).
    LocalKv,
    /// Distributed transactional KV (TiKV).
    Tikv,
    /// Lance dataset (columnar projection or primary store).
    Lance,
    /// In-memory Arrow table (test fixture or hot cache).
    InMemory,
}

/// A provider that exposes a MVCC snapshot timestamp.
///
/// Implementors are `TableProvider`s (DataFusion) that source from an
/// engine with a notion of a read-snapshot — TiKV (Percolator HLC),
/// Lance (dataset version), or surrealdb (KV generation).
///
/// The planner uses this to compose snapshot-consistent reads across
/// engines: a single `u64` timestamp threads through every provider
/// in the plan.
pub trait MvccProvider {
    /// Identifies the backend this provider sources from.
    fn backend(&self) -> BackendId;

    /// The MVCC snapshot timestamp this provider is bound to.
    /// `None` means "read latest at scan time" (less consistent across
    /// engines but cheaper).
    fn snapshot_ts(&self) -> Option<u64>;
}

/// A provider that exposes a TiKV snapshot. Refinement of [`MvccProvider`]
/// for the TiKV case, used as a marker so the planner can detect TiKV
/// providers without downcasting through DataFusion's trait objects.
pub trait TikvBackedProvider: MvccProvider {
    /// Always returns [`BackendId::Tikv`].
    fn backend(&self) -> BackendId {
        BackendId::Tikv
    }
}

/// A provider that exposes a Lance snapshot. Refinement of [`MvccProvider`]
/// for the Lance case.
pub trait LanceBackedProvider: MvccProvider {
    /// Always returns [`BackendId::Lance`].
    fn backend(&self) -> BackendId {
        BackendId::Lance
    }

    /// The Lance dataset version this provider is bound to.
    ///
    /// Defaults to `MvccProvider::snapshot_ts` — Lance uses `u64`
    /// dataset versions, which fit the same number space. Override
    /// only if a provider distinguishes "snapshot" (for cross-engine
    /// consistency) from "dataset version" (for time-travel).
    fn dataset_version(&self) -> Option<u64> {
        <Self as MvccProvider>::snapshot_ts(self)
    }
}

/// Compose two snapshot timestamps into the one a join must read at.
///
/// Rule: the older of the two (smaller `u64`), since reading both at
/// max(a, b) would give a non-snapshot-consistent view if one engine
/// hasn't caught up. `None` propagates as "latest" → pick whichever
/// is concrete.
pub fn min_snapshot_ts(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubTikv {
        ts: Option<u64>,
    }
    impl MvccProvider for StubTikv {
        fn backend(&self) -> BackendId {
            BackendId::Tikv
        }
        fn snapshot_ts(&self) -> Option<u64> {
            self.ts
        }
    }
    impl TikvBackedProvider for StubTikv {}

    struct StubLance {
        ts: Option<u64>,
    }
    impl MvccProvider for StubLance {
        fn backend(&self) -> BackendId {
            BackendId::Lance
        }
        fn snapshot_ts(&self) -> Option<u64> {
            self.ts
        }
    }
    impl LanceBackedProvider for StubLance {}

    #[test]
    fn tikv_marker_reports_backend() {
        let p = StubTikv { ts: Some(42) };
        assert_eq!(MvccProvider::backend(&p), BackendId::Tikv);
        assert_eq!(p.snapshot_ts(), Some(42));
    }

    #[test]
    fn lance_dataset_version_defaults_to_snapshot_ts() {
        let p = StubLance { ts: Some(7) };
        assert_eq!(p.dataset_version(), Some(7));
        assert_eq!(MvccProvider::backend(&p), BackendId::Lance);
    }

    #[test]
    fn min_snapshot_picks_older() {
        assert_eq!(min_snapshot_ts(Some(10), Some(5)), Some(5));
        assert_eq!(min_snapshot_ts(Some(10), None), Some(10));
        assert_eq!(min_snapshot_ts(None, Some(5)), Some(5));
        assert_eq!(min_snapshot_ts(None, None), None);
    }
}
