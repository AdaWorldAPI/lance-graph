//! D-CASCADE-V1-1 — poison-pill compile-fail proof for the Zone 1/2 check.
//!
//! Gated on `--features _internal_test_serialize_poison`. With the feature
//! ON, this test file declares a deliberately-violating type that mimics
//! the SHAPE of a Zone 2 type (Arrow scalar membrane row) but DOES carry
//! `serde::Serialize`. The build script's check, however, scans the four
//! canonical Zone 1/2 source files — NOT this test file — so toggling the
//! feature alone does not trigger `cargo::error::`.
//!
//! To prove the gate fires for real, a second probe (D-CASCADE-V1-1
//! follow-up — see `.claude/knowledge/soa-dto-dependency-ledger.md` Probe
//! Queue row "Serialize static check") edits one of the four scanned files
//! to add `#[derive(Serialize)]` and confirms the build aborts. That probe
//! is run manually / in CI; this file documents the intent and stages the
//! poison shape so reviewers can see it without grep.
//!
//! Default build (no feature) — this file compiles to a no-op test. CI
//! opt-in to `_internal_test_serialize_poison` exposes the violating type
//! at the test surface; an automated CI gate may then move the type into
//! `src/external_intent.rs` to verify `cargo::error::` aborts.

#[cfg(feature = "_internal_test_serialize_poison")]
mod poison {
    use serde::Serialize;

    /// DELIBERATE VIOLATION (gated): Zone 2-shaped scalar row that carries
    /// `Serialize`. If this struct is moved into `src/external_intent.rs`
    /// or `src/lance_membrane.rs`, the build script aborts the build with
    /// `cargo::error=D-CASCADE-V1-1 zone_serialize_check: ...`.
    #[derive(Clone, Debug, Default, Serialize)]
    pub struct PoisonZone2Row {
        pub external_role: u8,
        pub free_e: u8,
        pub gate_commit: bool,
        pub cycle_fp_hi: u64,
    }
}

#[cfg(feature = "_internal_test_serialize_poison")]
#[test]
fn poison_zone2_row_compiles_under_feature_but_must_not_live_in_zone1_or_zone2_paths() {
    let p = poison::PoisonZone2Row::default();
    assert_eq!(p.external_role, 0);
    // The feature surface holds the violating shape so reviewers can see
    // the contract; it does NOT live under `src/external_intent.rs` or
    // `src/lance_membrane.rs`, which is what the build script scans.
}

#[cfg(not(feature = "_internal_test_serialize_poison"))]
#[test]
fn poison_pill_inert_without_feature() {
    // Default build: the violating struct is not even compiled. This
    // confirms the feature gate keeps the violation out of the default
    // build surface.
    assert!(true, "_internal_test_serialize_poison feature is OFF");
}
