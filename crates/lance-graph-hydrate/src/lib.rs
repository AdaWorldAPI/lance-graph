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
//!   rule: **flush only from Hydrated, never Dirty** — see
//!   [`LifecycleState`]'s own doc for the honest scope of what
//!   [`LifecycleState::can_flush`] actually enforces today (a checkable
//!   predicate, not yet wired into any flush API).
//! - **Hydrate-aside, publish-by-rename**: fetch to a private staging
//!   directory, then ONE atomic rename publishes ([`copy::hydrate_dir`],
//!   [`file::hydrate_file`]) — a filesystem-atomicity boundary, deliberately
//!   not a lock/lease protocol. This exact mechanism (incl. the three named
//!   corruption modes it prevents) is already stated in this repo's own
//!   `EPIPHANIES.md`
//!   (`E-A-REPEATABLE-TRANSFER-IS-NOT-IDEMPOTENCE-OVER-A-MULTI-FILE-DIRECTORY-1`)
//!   — cited here rather than restated as if new.
//! - **Hydration is a byte copy**, never a Dataset scan-and-rewrite (which
//!   silently drops deletion vectors, indexes, and multi-version history) —
//!   proven in this repo's own `crates/lance-graph/examples/hydration_probe.rs`.
//!   That probe proves the byte-copy half; the staging/publish-by-rename
//!   mechanism itself is new code in this crate (see [`copy`]'s module doc).
//! - **Warm-marker skip-rehash** ([`marker::WarmMarker`]) and **idle release**
//!   ([`release::release_dir`]) are both q2's own inventions, generalized
//!   here — not present anywhere in lance-graph's doctrine before this crate.
//!   A checksum-axis answer to the same "is my local copy still valid?"
//!   question already exists at `lance-graph-ontology/src/lance_cache.rs`
//!   (`ttl_root_checksum`) — a different axis, not a duplicate.
//!
//! # Naming, disambiguated
//!
//! Two names in this crate already mean something else elsewhere in the
//! workspace — noted so a workspace-wide search doesn't land on the wrong
//! definition: [`dirty::is_dirty`] is unrelated to
//! `lance-graph-cognitive`'s `ContainerCache` per-slot dirty bitmap (also
//! named `is_dirty`); this crate's "hydrate" (remote → local) is the
//! opposite direction from `lance_graph::graph::hydrate::hydrate_bgz7`
//! (local weights → LanceDB ingest).
//!
//! # What this crate deliberately does NOT do
//!
//! It does not implement the automatic age+footprint-driven eviction SWEEP
//! policy from `.claude/plans/idle-flush-dataset-eviction-v1.md` — that plan
//! is still a PROPOSAL. This crate ships the mechanisms the policy would
//! call (hydrate, dirty-check, flush-gate, release); the scheduling policy
//! is deliberately out of scope for v1.
//!
//! **Follow-up landed:** [`copy::hydrate_dir`] and [`file::hydrate_file`]'s
//! staging/publish bodies were merged into `publish::publish_by_rename` —
//! a 5+3 hardening council on this crate (2026-08-17) named the duplication
//! as `ISS-HYDRATE-DIR-AND-FILE-DUPLICATE-THEIR-STAGING-BODIES`
//! (`.claude/board/ISSUES.md`) and deferred it deliberately, since it was
//! cheap only while this crate had zero consumers; that window was still
//! open, so this follow-up closes it. See the `publish` module's doc for
//! what merged and what stayed per-caller.

pub mod copy;
pub mod dirty;
pub mod env;
pub mod file;
pub mod lifecycle;
pub mod marker;
mod publish;
pub mod release;
mod staging;

pub use copy::{hydrate_dir, HydrateError, HydrationReport};
pub use dirty::{is_dirty, lifecycle_of, DirtyCheckError};
pub use env::{env as env_var, HydrationSource};
pub use file::{hydrate_file, HydrateFileError};
pub use lifecycle::LifecycleState;
pub use marker::{stat_identity, StatIdentity, WarmMarker};
pub use release::release_dir;
