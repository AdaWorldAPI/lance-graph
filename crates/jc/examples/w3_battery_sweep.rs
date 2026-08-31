//! W3 pre-registration: solver-order advantage + carrier-fidelity law.
//!
//! Reproduces the D-SK-A/B probe numbers in the shape a jc pillar can gate,
//! and MEASURES whether the kernel-scalar cancellation trap actually fires at
//! these settings — a trap that cannot be demonstrated must not be asserted.
use jc::solver_order::probe_tables;

fn main() {
    probe_tables();
}
