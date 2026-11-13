//! LAB, D-VIRTUAL-ADDRESS-ALGEBRA-1: one address law shared by order and
//! geometry.
//!
//! Two transformations are kept apart:
//!
//! - **Ordered row selection** says which source-local rows are visited. It
//!   already exists: `OrderIndex` / `SealedOrder` (lab), `SelectionSource`,
//!   `LaneBinding::selected`, `fold_bound_selection`, `execute_sel`,
//!   `ndarray::simd::sum_i32_at`. This crate adds nothing to it.
//! - **Geometric address recipe** ([`recipe`]) says how a view cell
//!   `(row, property)` resolves to `(field, position)`. Three bits.
//!
//! [`view`] joins them at the one place they meet: a recipe chain is
//! normalised, and then only chooses the `field` argument of the existing
//! selection fold. [`grid`] is a fixture source behind the two existing
//! sockets. [`physical`] is the oracle that moves data. [`cube`] is the
//! 256^3 address space.
//!
//! Pass conditions: `.claude/plans/d-virtual-address-algebra-1.md`.

#![forbid(unsafe_code)]

pub mod cube;
pub mod grid;
pub mod owner;
pub mod physical;
pub mod recipe;
pub mod view;
