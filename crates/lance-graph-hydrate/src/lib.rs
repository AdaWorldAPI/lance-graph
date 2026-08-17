//! `lance-graph-hydrate` — the generic SoA -> S3 -> local volume -> Lance
//! hydration pattern, minted here so any downstream crate (OGAR, q2, or any
//! future consumer) inherits it as a plug-and-play dependency instead of
//! re-implementing it locally.
//!
//! # Why this crate exists
//!
//! Two consumers (q2's `cockpit-server` and this repo's own doctrine) had
//! independently converged on the same shape — hydrate a remote artifact to
//! a local volume once, serve hot, release idle — but the mechanism lived
//! only as repo-local code (q2's `osm_slab_hydrate.rs` / `osm_lance.rs`) and
//! as a doctrine document with one explicitly named, unbuilt gap:
//! `ISS-REMOTE-URI-CONSTRUCTORS-PREDATE-THE-HYDRATION-DOCTRINE`
//! (`.claude/board/ISSUES.md`) — *"Something of the shape `hydrate_from(remote)
//! -> VersionedGraph` ... is the missing piece."* This crate is that piece,
//! generalized past one type.
//!
//! # The doctrine, briefly (full version:
//! `.claude/knowledge/s3-hydration-lifecycle.md`)
//!
//! - **Three layers**: the object store is the hydration SOURCE; a local
//!   mmap-capable directory is THE STORE (always required — never a
//!   page-fault backing store); a persistent volume is a pure
//!   hydration-frequency optimization, never a correctness requirement.
//! - **Four states**: [`LifecycleState::Absent`] → [`LifecycleState::Hydrated`]
//!   → ([`LifecycleState::Dirty`] | [`LifecycleState::Flushed`]). The one hard
//!   rule: **flush only from Hydrated, never Dirty** —
//!   [`LifecycleState::can_flush`] encodes it as a guard, not caller
//!   discipline.
//! - **Hydrate-aside, publish-by-rename**: fetch to a private staging
//!   directory, then ONE atomic rename publishes ([`copy::hydrate_dir`],
//!   [`file::hydrate_file`]) — a filesystem-atomicity boundary, deliberately
//!   not a lock/lease protocol.
//! - **Hydration is a byte copy**, never a Dataset scan-and-rewrite (which
//!   silently drops deletion vectors, indexes, and multi-version history) —
//!   proven in this repo's own `crates/lance-graph/examples/hydration_probe.rs`.
//! - **Warm-marker skip-rehash** ([`marker::WarmMarker`]) and **idle release**
//!   ([`release::release_dir`]) are both q2's own inventions, generalized
//!   here — not present anywhere in lance-graph's doctrine before this crate.
//!
//! # What this crate deliberately does NOT do
//!
//! It does not implement the automatic age+footprint-driven eviction SWEEP
//! policy from `.claude/plans/idle-flush-dataset-eviction-v1.md` — that plan
//! is still a PROPOSAL. This crate ships the mechanisms the policy would
//! call (hydrate, dirty-check, flush-gate, release); the scheduling policy
//! is deliberately out of scope for v1.

pub mod copy;
pub mod dirty;
pub mod env;
pub mod file;
pub mod lifecycle;
pub mod marker;
pub mod release;

pub use copy::{hydrate_dir, HydrateError, HydrationReport};
pub use dirty::{is_dirty, DirtyCheckError};
pub use env::{env as env_var, HydrationSource};
pub use file::{hydrate_file, HydrateFileError};
pub use lifecycle::LifecycleState;
pub use marker::{stat_identity, StatIdentity, WarmMarker};
pub use release::release_dir;
