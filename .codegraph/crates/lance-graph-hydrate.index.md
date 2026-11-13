# lance-graph-hydrate

crate: lance-graph-hydrate
path: crates/lance-graph-hydrate
workspace: member
digest: 9d14e9cc5ba8b211
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep arrow dev — crates/lance-graph-hydrate/Cargo.toml:52 CONFIRMED
- dep futures normal — crates/lance-graph-hydrate/Cargo.toml:30 CONFIRMED
- dep lance normal — crates/lance-graph-hydrate/Cargo.toml:28 CONFIRMED
- dep libc normal target=cfg(unix) — crates/lance-graph-hydrate/Cargo.toml:47 CONFIRMED
- dep object_store normal — crates/lance-graph-hydrate/Cargo.toml:29 CONFIRMED
- dep sha2 normal — crates/lance-graph-hydrate/Cargo.toml:32 CONFIRMED
- dep tempfile dev — crates/lance-graph-hydrate/Cargo.toml:51 CONFIRMED
- dep thiserror normal — crates/lance-graph-hydrate/Cargo.toml:44 CONFIRMED
- dep tokio dev — crates/lance-graph-hydrate/Cargo.toml:50 CONFIRMED
- dep tokio normal — crates/lance-graph-hydrate/Cargo.toml:31 CONFIRMED
- dep zip normal — crates/lance-graph-hydrate/Cargo.toml:43 CONFIRMED

## Depended on by

- rdep lance-graph normal — crates/lance-graph/Cargo.toml:25 CONFIRMED

## Public items

- enum HydrateArchiveError [pub] — crates/lance-graph-hydrate/src/archive.rs:60 CONFIRMED
- struct ArchiveReport [pub] — crates/lance-graph-hydrate/src/archive.rs:83 CONFIRMED
- fn hydrate_archive [pub] — crates/lance-graph-hydrate/src/archive.rs:101 CONFIRMED
- enum HydrateError [pub] — crates/lance-graph-hydrate/src/copy.rs:47 CONFIRMED
- struct HydrationReport [pub] — crates/lance-graph-hydrate/src/copy.rs:61 CONFIRMED
- fn hydrate_dir [pub] — crates/lance-graph-hydrate/src/copy.rs:77 CONFIRMED
- enum DirtyCheckError [pub] — crates/lance-graph-hydrate/src/dirty.rs:22 CONFIRMED
- fn is_dirty [pub] — crates/lance-graph-hydrate/src/dirty.rs:34 CONFIRMED
- fn lifecycle_of [pub] — crates/lance-graph-hydrate/src/dirty.rs:53 CONFIRMED
- fn env [pub] — crates/lance-graph-hydrate/src/env.rs:25 CONFIRMED
- struct HydrationSource [pub] — crates/lance-graph-hydrate/src/env.rs:37 CONFIRMED
- fn HydrationSource::from_env [pub] — crates/lance-graph-hydrate/src/env.rs:47 CONFIRMED
- fn HydrationSource::options [pub] — crates/lance-graph-hydrate/src/env.rs:65 CONFIRMED
- fn HydrationSource::uri [pub] — crates/lance-graph-hydrate/src/env.rs:72 CONFIRMED
- enum HydrateFileError [pub] — crates/lance-graph-hydrate/src/file.rs:29 CONFIRMED
- fn hydrate_file [pub] — crates/lance-graph-hydrate/src/file.rs:48 CONFIRMED
- mod archive [pub] — crates/lance-graph-hydrate/src/lib.rs:78 CONFIRMED
- mod copy [pub] — crates/lance-graph-hydrate/src/lib.rs:79 CONFIRMED
- mod dirty [pub] — crates/lance-graph-hydrate/src/lib.rs:80 CONFIRMED
- mod env [pub] — crates/lance-graph-hydrate/src/lib.rs:81 CONFIRMED
- mod file [pub] — crates/lance-graph-hydrate/src/lib.rs:82 CONFIRMED
- mod lifecycle [pub] — crates/lance-graph-hydrate/src/lib.rs:83 CONFIRMED
- mod marker [pub] — crates/lance-graph-hydrate/src/lib.rs:84 CONFIRMED
- mod release [pub] — crates/lance-graph-hydrate/src/lib.rs:86 CONFIRMED
- enum LifecycleState [pub] — crates/lance-graph-hydrate/src/lifecycle.rs:18 CONFIRMED
- fn LifecycleState::can_hydrate [pub] — crates/lance-graph-hydrate/src/lifecycle.rs:38 CONFIRMED
- fn LifecycleState::can_flush [pub] — crates/lance-graph-hydrate/src/lifecycle.rs:43 CONFIRMED
- fn LifecycleState::can_release [pub] — crates/lance-graph-hydrate/src/lifecycle.rs:52 CONFIRMED
- struct StatIdentity [pub] — crates/lance-graph-hydrate/src/marker.rs:40 CONFIRMED
- fn stat_identity [pub] — crates/lance-graph-hydrate/src/marker.rs:46 CONFIRMED
- struct WarmMarker [pub] — crates/lance-graph-hydrate/src/marker.rs:62 CONFIRMED
- fn WarmMarker::write [pub] — crates/lance-graph-hydrate/src/marker.rs:79 CONFIRMED
- fn WarmMarker::read [pub] — crates/lance-graph-hydrate/src/marker.rs:95 CONFIRMED
- fn WarmMarker::is_trusted [pub] — crates/lance-graph-hydrate/src/marker.rs:115 CONFIRMED
- enum StagingKind [pub(crate)] — crates/lance-graph-hydrate/src/publish.rs:48 CONFIRMED
- fn remove_staging [pub(crate)] — crates/lance-graph-hydrate/src/publish.rs:66 CONFIRMED
- enum PublishError [pub(crate)] — crates/lance-graph-hydrate/src/publish.rs:78 CONFIRMED
- fn publish_by_rename [pub(crate)] — crates/lance-graph-hydrate/src/publish.rs:95 CONFIRMED
- fn release_dir [pub] — crates/lance-graph-hydrate/src/release.rs:82 CONFIRMED
- fn staging_suffix [pub(crate)] — crates/lance-graph-hydrate/src/staging.rs:22 CONFIRMED

## Imports from in-scope crates

- none

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n34["lance-graph"]
  n51["lance-graph-hydrate"]
  n34 --> n51
```
