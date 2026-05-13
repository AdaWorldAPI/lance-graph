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
