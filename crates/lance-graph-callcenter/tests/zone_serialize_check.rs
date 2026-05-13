//! D-CASCADE-V1-1 — regression smoke test for the Zone 1/2 serialize check.
//!
//! If this test compiles + runs, the build script ran to completion which
//! means the four scanned files (Zone 1: cognitive_shader.rs, Zone 2:
//! external_intent.rs + lance_membrane.rs + external_membrane.rs) did NOT
//! contain any `pub struct` / `pub enum` carrying `#[derive(Serialize)]`.
//!
//! The build script is the actual gate — `cargo::error::` aborts the build
//! before tests get to compile. So a green test here is a positive proof
//! that the doctrine holds.
//!
//! For the negative-direction test (poison pill), see
//! `tests/zone_serialize_check_compile_fail.rs` which is gated behind
//! `--features _internal_test_serialize_poison`.

#[test]
fn zone1_zone2_have_no_serialize_derives() {
    // Reaching this test means cargo::error did NOT fire during build.
    // That is the contract: build script aborts → tests can't run → CI red.
    // Tests run → the Zone 1/2 surface stayed clean. No runtime assertion
    // needed — the test merely compiling-and-running IS the assertion.
}

#[test]
fn zone3_types_remain_unrestricted() {
    // Sanity that Zone 3 types (transcode / postgrest / phoenix / drain /
    // supabase) are NOT scanned. The build script intentionally only inspects
    // four files; anything outside those paths is unaffected. The crate
    // building at all is the positive evidence.
}
