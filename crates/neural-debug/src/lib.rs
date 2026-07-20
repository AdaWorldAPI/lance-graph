//! Neural Debugger — static analysis + runtime diagnosis for the graph engine stack.
//!
//! Scans .rs files across all repos, detects dead neurons (todo!(), unimplemented!()),
//! stubs (Default::default() returns), NaN risks (f32::NAN, division patterns),
//! and produces a full function registry with neuron states.
//!
//! The 16 planning strategies use this to self-check their dependency chains.

pub mod diagnosis;
pub mod registry;
pub mod scanner;

// Producer / consumer convenience: `neural_debug::registry()` returns the
// process-wide `RuntimeRegistry`. Producers call `record_row(row, state)`;
// consumers call `diag()` or `snapshot_rows()` to surface state.
pub use diagnosis::NeuronState;
pub use registry::{registry, RuntimeDiag, RuntimeRegistry};
// Bit-49 disunification rule: flag consumers still on the deprecated
// `CausalEdge64::inference_type()` (v1 3-bit read) instead of the unified
// `inference_mantissa()` + `from_mantissa()` (v2 4-bit signed mantissa).
pub use scanner::{find_legacy_inference_in_source, scan_legacy_inference, LegacyInferenceHit};
