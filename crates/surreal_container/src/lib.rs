//! # `surreal_container` — embedded SurrealDB-on-Lance cognitive store
//!
//! This crate wires an **in-process** [`surrealdb`] [`Datastore`] backed by the
//! `kv-lance` storage engine (the `AdaWorldAPI/surrealdb` fork feature — NOT
//! upstream surrealdb on crates.io).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  surreal_container                       │
//! │                                          │
//! │  SurrealStore (this module)              │
//! │   └── surrealdb::Datastore              │
//! │          └── kv-lance backend           │
//! │                 └── Lance dataset        │
//! │                      (append-only)      │
//! │                                          │
//! │  write  (task 04) ── epoch container     │
//! │  read   (task 05) ── container fetch     │
//! │  fold   (task 06) ── read-time fold      │
//! │  catalog (task 07) ── goal/ptr/edge KV   │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Blocked items (must be resolved before this crate compiles)
//!
//! - **BLOCKED(A)**: Lance 6 semver not confirmed; current pins are `lance = "=4.0.0"`.
//! - **BLOCKED(B)**: LanceDB 0.28 semver not confirmed; current pins are `lancedb = "=0.27.2"`.
//! - **BLOCKED(C)**: The `surrealdb` fork git URL, branch/tag, and exact `kv-lance`
//!   feature flag name must be provided by a fork-access human before the
//!   `[dependencies]` in `Cargo.toml` can be filled in.
//! - **BLOCKED(D)**: If Lance 6 pulls `lance-index` which depends on `ndarray 0.16`
//!   from crates.io, a `[patch.crates-io]` entry must alias it to the
//!   `AdaWorldAPI/ndarray` fork path.  Requires Lance 6 `Cargo.lock` inspection.
//!
//! ## Embedded Datastore init pattern (BLOCKED — do not call until unblocked)
//!
//! Once BLOCKED(C) is resolved, the init will look approximately like:
//!
//! ```rust,ignore
//! // BLOCKED(C): surrealdb Datastore API for kv-lance is not confirmed.
//! // The pattern below is a DESIGN SKETCH derived from cognitive-substrate.md
//! // (which describes core/src/kvs/lance/{wal,memtable,flusher,mod}.rs in the fork).
//! // Do NOT ship this as real code until a fork-access human validates the API.
//! //
//! // use surrealdb::Datastore;
//! //
//! // let ds = Datastore::new("lance://path/to/store").await?;
//! // let ses = Session::owner().with_ns("lance_graph").with_db("containers");
//! // ds.execute("DEFINE TABLE container SCHEMAFULL;", &ses, None).await?;
//! ```
//!
//! ## Usage (once unblocked)
//!
//! ```rust,ignore
//! use surreal_container::SurrealStore;
//!
//! let store = SurrealStore::open("/path/to/lance/store").await?;
//! // write/read/catalog modules build on `store.ds()`.
//! ```

// ── BLOCKED(C): surrealdb is not yet a real dependency (see Cargo.toml) ─────
// When BLOCKED(C) is resolved, replace the stub import below with:
//   use surrealdb::{Datastore, Session};
//
// For now the module compiles as a skeleton without the surrealdb dep so
// the workspace member resolves cleanly.

/// Embedded SurrealDB-on-Lance Datastore handle.
///
/// Wraps a `surrealdb::Datastore` opened with the `kv-lance` storage engine.
/// All other modules in this crate receive a reference to this struct rather
/// than constructing their own Datastore instances.
///
/// # Panics
///
/// Does not panic. All fallible operations return [`SurrealContainerError`].
///
/// # Blocked
///
/// `SurrealStore::open` cannot be implemented until BLOCKED(C) is resolved.
/// The struct definition is present so downstream module stubs can reference
/// the type.
pub struct SurrealStore {
    // BLOCKED(C): field type `surrealdb::Datastore` requires the fork dep.
    // Replace the placeholder below with the real field once unblocked:
    //
    //   ds: surrealdb::Datastore,
    //
    _placeholder: std::marker::PhantomData<()>,
}

impl SurrealStore {
    /// Test-only constructor for an uninitialised `SurrealStore`. Lets
    /// downstream tests exercise stub code paths (like
    /// [`view::read_via_kv_lance`]) without going through the
    /// `BLOCKED(C)`-gated `open()`. Not part of the public surface; gated
    /// by `cfg(test)` to keep the prod constructor singular.
    #[cfg(test)]
    pub(crate) fn test_placeholder() -> Self {
        Self {
            _placeholder: std::marker::PhantomData,
        }
    }
}

impl SurrealStore {
    /// Open (or create) an embedded `kv-lance` Datastore at `lance_path`.
    ///
    /// `lance_path` is the filesystem path to the Lance dataset directory that
    /// `surrealdb`'s `kv-lance` backend will manage.  The directory is created
    /// if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`SurrealContainerError::Blocked`] until BLOCKED(C) is resolved.
    ///
    /// # Example (once unblocked)
    ///
    /// ```rust,ignore
    /// let store = SurrealStore::open("/data/cognitive-store").await?;
    /// ```
    pub async fn open(_lance_path: &str) -> Result<Self, SurrealContainerError> {
        // BLOCKED(C): cannot construct a real Datastore until the surrealdb
        // fork dep (with kv-lance feature) is wired into Cargo.toml.
        //
        // Once unblocked, implementation sketch (API NOT confirmed — sketch only):
        //
        //   let ds = surrealdb::Datastore::new(
        //       &format!("lance://{}", _lance_path)
        //   ).await
        //   .map_err(|e| SurrealContainerError::Init { source: e })?;
        //
        //   Ok(Self { ds })
        //
        Err(SurrealContainerError::Blocked {
            reason: "BLOCKED(C): surrealdb kv-lance fork dep not wired; \
                     see Cargo.toml and workspace Cargo.toml comments",
        })
    }

    /// Return a reference to the inner `surrealdb::Datastore`.
    ///
    /// Modules `write`, `read`, `catalog`, etc. receive this reference and
    /// execute SurrealDB statements through it.
    ///
    /// # Blocked
    ///
    /// Returns `None` until BLOCKED(C) is resolved.
    ///
    /// ```rust,ignore
    /// // Once unblocked, return type will be `&surrealdb::Datastore`
    /// pub fn ds(&self) -> &surrealdb::Datastore { &self.ds }
    /// ```
    pub fn ds(&self) -> Option<()> {
        // BLOCKED(C): real return type is `&surrealdb::Datastore`
        None
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by `surreal_container`.
#[derive(Debug)]
pub enum SurrealContainerError {
    /// A required dependency or API is not yet wired.
    ///
    /// Remove this variant once all `BLOCKED` items are resolved.
    Blocked {
        /// Human-readable description of what is missing.
        reason: &'static str,
    },

    /// The surrealdb fork dep is commented in `Cargo.toml` to keep default
    /// builds fast (cold surrealdb build is heavy). The caller pattern-matches
    /// this to fall back to a direct lance-graph read, or instructs the user
    /// to uncomment the dep and take the cold-build cost.
    ///
    /// Distinct from [`SurrealContainerError::Blocked`] (which signals an
    /// unresolved coordinate / API gap): `BlockedColdBuild` means everything
    /// is *known* and *aligned*; only the cold-build flip-on is pending.
    BlockedColdBuild {
        /// Human-readable pointer to the Cargo.toml note and integration site.
        reason: &'static str,
    },
    // BLOCKED(C): add `Init { source: surrealdb::Error }` once fork dep lands.
    // BLOCKED(A)/(B): add `Lance { source: lance::Error }` once Lance 6 is pinned.
}

impl std::fmt::Display for SurrealContainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blocked { reason } => write!(f, "surreal_container blocked: {reason}"),
            Self::BlockedColdBuild { reason } => {
                write!(f, "surreal_container blocked (cold-build gate): {reason}")
            }
        }
    }
}

impl std::error::Error for SurrealContainerError {}

// ── Module stubs (tasks 04-07 + future tasks) ─────────────────────────────

/// Write path: one epoch-container → one `kv-lance` record (task 04).
///
/// Append-only; one record per epoch, never update-in-place.
// TODO task 04
pub mod write;

/// Read path: fetch a container by id from `kv-lance` (task 05).
// TODO task 05
pub mod read;

/// Read-time fold: merge multiple epoch fragments into one view (task 06).
// TODO task 06
pub mod fold;

/// Control-plane KV: goal-state, container pointers, edges (task 07).
///
/// Small mutable records only — no bulk data stored here.
// TODO task 07
pub mod catalog;

/// L2 moka cache over hot container reads (task 08).
// TODO task 08
pub mod cache;

/// Epoch tick + harvest loop (task 09).
// TODO task 09
pub mod epoch;

/// Lock-free handoff ring between OS-thread cores and the ractor membrane (task 10).
// TODO task 10
pub mod ring;

/// WAL log compaction / snapshot fold (task 11).
// TODO task 11
pub mod compaction;

/// Clean-writer invariants: append-only, single-writer, no LWW collision (task 12).
// TODO task 12
pub mod writer_invariants;

/// SurrealQL read-glove: `MailboxSoaView` adapter over a kv-lance-backed
/// mailbox row (D-PG-6 contract slice). Pairs with
/// `lance-graph::graph::scheduler::LanceVersionScheduler` to close the
/// IN-direction of the bidirectional kanban subscription
/// (`E-SUBSTRATE-IS-THE-SCHEDULER`). The actual SurrealQL projection +
/// kv-lance scan is the cold-build follow-on; the contract surface is
/// available today.
pub mod view;
