//! D-CASCADE-V1-1 — subprocess compile-fail probe for the Zone 1/2 check.
//!
//! Closes FIX-1 deferred by PR #355.
//!
//! # What this test does
//!
//! Runs `cargo build` on the standalone fixture project at
//! `tests/zone-poison-fixtures/` as a subprocess and asserts:
//!
//! 1. The process exits **non-zero** (build aborted).
//! 2. The combined stdout+stderr contains the exact abort signature
//!    `"D-CASCADE-V1-1 zone_serialize_check:"` emitted by the fixture's
//!    build script via `cargo::error=`.
//!
//! # Why subprocess, not trybuild
//!
//! The gate fires in the **build script** of `lance-graph-callcenter`, not in
//! the Rust source. `trybuild` intercepts rustc errors; it does not intercept
//! `cargo::error=` from a build script that calls `std::process::exit(1)`.
//! A subprocess `cargo build` is the correct tool for testing build-script
//! aborts — it is equivalent rigour (non-zero exit + expected stderr) with
//! simpler mechanics.
//!
//! # Fixture layout
//!
//! ```text
//! tests/zone-poison-fixtures/
//!   Cargo.toml          — standalone crate, NOT in workspace members
//!   build.rs            — mirrors the real build.rs zone-serialize scan
//!   src/
//!     lib.rs
//!     external_intent.rs  — POISONED: pub struct with #[derive(Serialize)]
//! ```
//!
//! The fixture's `build.rs` scans `src/external_intent.rs`, finds the
//! `Serialize` derive on `PoisonExternalIntent`, and emits:
//!
//! ```text
//! cargo::error=D-CASCADE-V1-1 zone_serialize_check: `PoisonExternalIntent` in …
//! ```
//!
//! then exits 1 — which is exactly the same abort path as the real build.rs.

use std::path::PathBuf;
use std::process::Command;

/// Returns the path to the zone-poison-fixtures directory.
fn fixture_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is set by cargo when running integration tests; it
    // points to the crate root (lance-graph-callcenter/).
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set when running under cargo test");
    PathBuf::from(manifest)
        .join("tests")
        .join("zone-poison-fixtures")
}

#[test]
fn build_script_aborts_on_serialize_derive_in_zone2() {
    let fixture = fixture_dir();
    assert!(
        fixture.join("Cargo.toml").is_file(),
        "fixture Cargo.toml not found at {}",
        fixture.display()
    );

    // Use the same cargo binary that built this test to avoid version skew.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());

    let output = Command::new(&cargo)
        .args(["build", "--manifest-path"])
        .arg(fixture.join("Cargo.toml"))
        // Route build artefacts into the fixture's own target/ so we don't
        // pollute the parent workspace's target directory.
        .args(["--target-dir"])
        .arg(fixture.join("target"))
        .output()
        .expect("failed to spawn cargo build for zone-poison fixture");

    // 1. Must fail.
    assert!(
        !output.status.success(),
        "expected `cargo build` of zone-poison fixture to fail (build script abort), \
         but it succeeded (exit {:?})",
        output.status.code()
    );

    // 2. Combined output must contain the abort signature.
    let combined = {
        let mut v = output.stdout.clone();
        v.extend_from_slice(&output.stderr);
        String::from_utf8_lossy(&v).into_owned()
    };

    const ABORT_SIGNATURE: &str = "D-CASCADE-V1-1 zone_serialize_check:";
    assert!(
        combined.contains(ABORT_SIGNATURE),
        "expected cargo::error= abort signature {:?} in build output, got:\n{}",
        ABORT_SIGNATURE,
        combined
    );
}

#[cfg(not(feature = "_internal_test_serialize_poison"))]
#[test]
fn poison_pill_inert_without_feature() {
    // Default build: the violating struct in the test source is not compiled.
    // Reaching this point confirms the feature gate is OFF in the default build.
    // The real compile-fail proof is `build_script_aborts_on_serialize_derive_in_zone2`
    // above, which runs unconditionally.
}
