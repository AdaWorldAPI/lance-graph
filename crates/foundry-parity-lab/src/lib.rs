//! D-FOUNDRY-PARITY-1 lab: a 64-store operating network.
//!
//! The question: can source-owned planes, one DT64 lane binding and
//! mask-RISC programs answer operational questions — `CanCook`,
//! `CanFulfill`, `NetworkCapacity`, network ingredient cost — over 64 stores
//! × 64 inventory categories with exact parity against an independent
//! oracle, and refuse correctly when a frame is stale, a row order differs,
//! a lane serves another scope, a store is not visible, or stock is unknown?
//!
//! * [`fixture`] is the only definition of values.
//! * [`source`] owns the planes and implements DT64's `LaneSource`.
//! * [`bind`] is the class: its 211-field basis, route and demanded binding.
//! * [`arm_b`] answers through `borrow_all` + mask-RISC programs.
//! * [`arm_c`] answers per-store cost as one INT8 GEMM over staged A.
//! * [`arm_a`] is the materialized-object null model.
//! * [`oracle`] answers from the fixture alone, and also evaluates the one
//!   scenario (Level S) the DR path cannot: no overlay contract exists.
//!
//! Read-only (Level R). No writeback, no actions, no Foundry comparator:
//! nothing here measures or claims anything about Foundry itself.

pub mod arm_a;
pub mod arm_b;
pub mod arm_c;
pub mod bind;
pub mod fixture;
pub mod oracle;
pub mod source;
