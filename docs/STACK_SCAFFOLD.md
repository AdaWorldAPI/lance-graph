# Stack scaffold — surrealdb + ractor + ndarray (so a session doesn't have to guess)

> **Purpose:** the known-good wiring for the three-pillar runtime — **ndarray**
> (SIMD/HPC substrate), **ractor** (actor runtime / mailbox owner), **surrealdb**
> (KV-lance storage + SurrealQL AR-API surface). Copy the fragments below; do not
> re-derive the coordinates each session.
> **P0 (CLAUDE.md):** every crate with an `AdaWorldAPI/<name>` fork is wired via
> the **fork** (`path`/`git`), NEVER crates.io. All three here are forks.

## Status (honest, 2026-06-18)

| Pillar | Fork | Local path | Ready? |
|---|---|---|---|
| **ndarray** | `AdaWorldAPI/ndarray` | `/home/user/ndarray` | ✅ ready (HPC, 880+ tests) |
| **ractor** | `AdaWorldAPI/ractor` | `/home/user/ractor` | ✅ ready |
| **surrealdb** | `AdaWorldAPI/surrealdb` | `/home/user/surrealdb` | ⚠️ **KV-lance: backend MODULE implemented, not yet feature-wired.** The Lance KV backend is real code at `crates/core/src/kvs/lance/` (`mod`/`schema`/`tx_buffer`/`timeline`/`background_optimizer`/`tests` — the 19-method `Transactable` scaffold), NOT a sketch. But it is **not yet exposed as a storage feature** (no `kv-lance` in `crates/core/Cargo.toml`, not `mod`'d into `kvs/mod.rs`, no `lance` dep wired) and the `TODO(lance-integration)` Lance-API call sites remain (the `.claude/lance-backend/DAY_BY_DAY.md` 12-day item). Use `storage-mem` until the feature + integration land; then switch the `surrealdb` feature line to `kv-lance`. |

**Footprint (resolved-crate proxy):** lance-graph `Cargo.lock` ≈ **889**;
surrealdb (all backends) ≈ **1148**. Both share the lance+arrow base; surreal's
marginal cost is the SurrealQL engine + multi-protocol server (~260 crates),
NOT the storage backend. Dropping `storage-rocksdb` removes the RocksDB C++
native build (compile-time + tens of MB) but not the engine. Prefer pulling the
surreal **KV-lance + AR-API surface** selectively (where `ExecTarget::SurrealQl`
is wanted), not the full engine — lance-graph already covers storage+query+TS
(lance versioning = time-series, datafusion = query).

## `Cargo.toml` (reference — paths assume siblings under the same parent dir)

```toml
[package]
name = "ada-stack-app"
version = "0.1.0"
edition = "2021"
rust-version = "1.94"

[dependencies]
# ── ndarray: SIMD / HPC substrate (fork; default-features off → pick HPC) ──
ndarray = { path = "../ndarray", default-features = false, features = ["std"] }

# ── ractor: actor runtime / mailbox owner (fork) ──
ractor = { path = "../ractor/ractor", default-features = false, features = ["tokio_runtime"] }
# ractor_cluster = { path = "../ractor/ractor_cluster" }  # only if distributing

# ── lance-graph spine + the contract (fork; the OGAR Core + DO arm live here) ──
lance-graph-contract = { path = "../lance-graph/crates/lance-graph-contract" }
# lance-graph        = { path = "../lance-graph/crates/lance-graph" }  # full spine (heavy: datafusion)

# ── surrealdb: storage + SurrealQL AR-API (fork) ──
# PENDING: replace `storage-mem` with `kv-lance` once the fork's lance-backend
# (surrealdb/.claude/lance-backend) lands. Keep default-features = false to avoid
# pulling rocksdb/tikv/scripting unless needed.
surrealdb = { path = "../surrealdb/surrealdb", default-features = false, features = ["kv-mem"] }

[dependencies.tokio]
version = "1"
features = ["rt-multi-thread", "macros"]

# Fork pins also expressible as git, e.g.:
#   ndarray = { git = "https://github.com/AdaWorldAPI/ndarray", branch = "main" }
# Use path when the sibling clone exists (this workspace); git for CI without it.
```

> Verify the exact surreal crate path/feature names against
> `/home/user/surrealdb/Cargo.toml` before relying on this — the surreal
> workspace lays its public crate under `surrealdb/` and gates storage via
> `surrealdb-server` features (`storage-mem` → `kv-mem` mapping). The line above
> is the intended shape; it is NOT yet build-verified (kv-lance pending).

## `Dockerfile` (reference — Rust 1.94, mold linker, no debuginfo for fast link)

```dockerfile
# syntax=docker/dockerfile:1
FROM rust:1.94-bookworm AS build

# mold avoids the rust-lld SIGBUS link-cliff seen in this workspace.
RUN apt-get update && apt-get install -y --no-install-recommends \
        mold clang cmake pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

ENV CARGO_PROFILE_DEV_DEBUG=0 \
    RUSTFLAGS="-C link-arg=-fuse-ld=mold -C target-cpu=x86-64-v3"
# AVX-512 stack (ndarray HPC, x86-64-v4): set target-cpu=x86-64-v4 on capable silicon.

WORKDIR /build
# Sibling forks must be present in the build context (copy or git-clone them):
#   /build/ndarray /build/ractor /build/surrealdb /build/lance-graph /build/ada-stack-app
COPY . .
WORKDIR /build/ada-stack-app
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/ada-stack-app/target \
    cargo build --release && cp target/release/ada-stack-app /ada-stack-app

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /ada-stack-app /usr/local/bin/ada-stack-app
ENTRYPOINT ["ada-stack-app"]
```

Notes:
- **No RocksDB** in this image — `surrealdb` is `default-features = false` so no
  C++ rocksdb/speedb build. When `kv-lance` lands it adds no native toolchain
  beyond what lance already needs.
- **mold + `CARGO_PROFILE_DEV_DEBUG=0`** mirror the link-cliff workaround this
  workspace uses for the contract/driver crates.
- The build context must contain the sibling fork checkouts (path deps); for a
  CI image without them, switch the `Cargo.toml` lines to `git =` fork pins.

## What this deliberately is NOT

- NOT a lance-graph workspace member (it would bloat the spine's build with the
  surreal engine + datafusion together). It is a **copy-paste reference**.
- NOT build-verified for the surreal pillar until `kv-lance` lands — the ndarray
  + ractor + lance-graph-contract lines are the ready core.
