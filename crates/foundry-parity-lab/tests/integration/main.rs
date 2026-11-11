//! D-FOUNDRY-PARITY-1 integration tests.
// Store rows index the oracle, arm A and arm B at once; a range loop is clearer.
#![allow(clippy::needless_range_loop)]
pub mod common;
pub mod controls;
pub mod cost;
pub mod method;
pub mod parity;
pub mod refusals;
