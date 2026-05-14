# S9-W1 agent scratchpad — zone_serialize_check_compile_fail rewrite

**Started:** 2026-05-13
**Goal:** Replace `assert!(true, ...)` smoke with real subprocess compile-fail probe (FIX-1 from PR #355).

## Files touched

- `crates/lance-graph-callcenter/tests/zone_serialize_check_compile_fail.rs` — REWRITTEN (112 LOC)
  * Removed: `assert!(true)` smoke + `_internal_test_serialize_poison` gating
  * Added: `build_script_aborts_on_serialize_derive_in_zone2` test that runs `cargo build` on fixture as subprocess and asserts non-zero exit + abort signature in combined output
  * Kept: `poison_pill_inert_without_feature` inert test (no feature)

- `crates/lance-graph-callcenter/tests/zone-poison-fixtures/Cargo.toml` — NEW (~18 LOC)
  * `[workspace]` table to prevent parent workspace walkup
  * `[build-dependencies] syn = "2"` only dep

- `crates/lance-graph-callcenter/tests/zone-poison-fixtures/build.rs` — NEW (~70 LOC)
  * Mirrors lance-graph-callcenter/build.rs zone-serialize scan
  * Scans src/external_intent.rs; emits `cargo::error=D-CASCADE-V1-1 zone_serialize_check:` + exit 1

- `crates/lance-graph-callcenter/tests/zone-poison-fixtures/src/lib.rs` — NEW (4 LOC)
- `crates/lance-graph-callcenter/tests/zone-poison-fixtures/src/external_intent.rs` — NEW (~18 LOC)
  * POISONED: `#[derive(Clone, Debug, Default, Serialize)]` on `pub struct PoisonExternalIntent`

**NO changes** to `build.rs` (lance-graph-callcenter's real build script) or `Cargo.toml`.

## Abort signature asserted
```
D-CASCADE-V1-1 zone_serialize_check:
```

## Decision: subprocess over trybuild
`trybuild` intercepts rustc errors. The zone check fires in the BUILD SCRIPT via `cargo::error=` + `std::process::exit(1)`, which is a build-script abort — not a rustc compile error. trybuild cannot intercept this. Subprocess `cargo build` on an isolated fixture is the correct tool.

## Pre-existing blocker: ndarray/blake3
`cargo test -p lance-graph-callcenter --test zone_serialize_check_compile_fail` fails because `thinking-engine` depends on `ndarray`, and `ndarray/src/hpc/plane.rs` + `vsa.rs` + `seal.rs` + `merkle_tree.rs` use `blake3` unconditionally (missing `#[cfg(feature = "hpc-extras")]` gate). This is a pre-existing workspace bug unrelated to our changes. The same failure blocks the existing `zone_serialize_check.rs` test too. Implementation is correct; ndarray/blake3 fix is out of scope.

## Fixture verification (standalone)
```
cd crates/lance-graph-callcenter/tests/zone-poison-fixtures && cargo build
→ exit 101 (non-zero)
→ stderr: "D-CASCADE-V1-1 zone_serialize_check: `PoisonExternalIntent` in ... (Zone 2) carries `#[derive(Serialize)]`"
```
Fixture works correctly as a standalone cargo build.

## Cross-file invariant
Only touched: `zone_serialize_check_compile_fail.rs` + new fixture files.
Did NOT touch: `build.rs` (real), `zone_serialize_check.rs`, `Cargo.toml`.
