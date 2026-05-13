// Zone-poison fixture: deliberately violating Zone 2 file.
// This file is ONLY used by the subprocess compile-fail probe in
// zone_serialize_check_compile_fail.rs — it is never part of the main build.
//
// The struct below carries `#[derive(Serialize)]` on a public Zone 2 type,
// which the fixture's build.rs detects and aborts with cargo::error=D-CASCADE-V1-1.
// NOTE: serde is not in [dependencies] — the build.rs scans the AST only; the
// file is never compiled, so the missing import does not matter.

/// POISON: Zone 2-shaped scalar row that carries Serialize.
/// The fixture build.rs detects this and emits cargo::error=.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PoisonExternalIntent {
    pub external_role: u8,
    pub free_e: u8,
    pub gate_commit: bool,
    pub cycle_fp_hi: u64,
}
