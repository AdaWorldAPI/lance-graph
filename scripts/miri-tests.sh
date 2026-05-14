#!/bin/sh
#
# Miri test runner for lance-graph — ephemeral nightly, scoped to this
# script ONLY.
#
# Rules of the road (mirrors ndarray/scripts/miri-tests.sh):
#   * Default toolchain is stable 1.95.0 (rust-toolchain.toml).
#     `cargo build`, `cargo test`, `cargo clippy`, CI's clippy / tests
#     jobs all use stable. Nothing else opts into nightly.
#   * Miri ships nightly-only. This script invokes `cargo +nightly miri`,
#     an ephemeral per-invocation switch — does NOT change the default.
#   * The lance-graph workspace contains FFI-heavy crates that Miri
#     CANNOT enter: `lance`, `arrow`, `datafusion`, BLAS, jitson/Cranelift.
#     Those crates are skipped entirely. Miri's value is on the zero-dep
#     contract crate, the planner's pure-Rust paths, and the small
#     standalone codec/rbac/debug crates.
#   * `lance-graph-ontology` has a `lance-cache` feature gating the
#     Lance dataset path; this script runs it WITHOUT that feature so
#     the registry / namespace / TTL parsing paths get checked under
#     Miri.
#
# If this stays clean, the miri job in `.github/workflows/ci.yaml` can
# promote from optional → required for these crates.

set -x
set -e

# Idempotent install of miri + nextest. No-op when already present.
rustup component add miri --toolchain nightly >/dev/null 2>&1 || \
    rustup +nightly component add miri

# Layout randomisation catches missing `#[repr(transparent)]` and similar.
export RUSTFLAGS="-Zrandomize-layout"

# -Zmiri-ignore-leaks: lance-graph-ontology's test helpers do
# `Box::leak(name.into_boxed_str())` to fabricate `&'static str` values
# for `Schema::builder` (which intentionally takes `&'static str` for
# cache-friendly storage). Each test leaks ~10 bytes of namespace/entity
# names by design — Miri's process-exit leak detector flags every one.
# We accept the signal loss because production code does not use
# `Box::leak` anywhere; the leaks live only in test helpers. If real
# leaks creep into prod, clippy's `mem_forget` lint + the regular
# alloc-tracking would surface them on stable before they got near Miri.
export MIRIFLAGS="-Zmiri-ignore-leaks"

# Crates that Miri can actually enter — no lance / arrow / datafusion
# / cblas / inline-asm cpuid paths in their default dependency closure.
MIRI_SAFE_CRATES="
    -p lance-graph-contract
    -p lance-graph-rbac
    -p neural-debug
"

# Crates that build under Miri but skip their Lance-backed feature paths.
# `lance-graph-ontology` default (no lance-cache) is safe; the test
# `tests/round_trip_ttl.rs` exercises the registry / TTL pipeline.
MIRI_SAFE_NO_DEFAULT="
    -p lance-graph-ontology --no-default-features
"

# Run via `cargo +nightly miri test` (not nextest — lance-graph CI doesn't
# wire nextest, and plain `cargo miri test` is sufficient for the targeted
# crate scope here).
cargo +nightly miri test $MIRI_SAFE_CRATES
cargo +nightly miri test $MIRI_SAFE_NO_DEFAULT
