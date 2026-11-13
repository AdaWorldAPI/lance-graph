//! LAB, D-VIRTUAL-ADDRESS-ALGEBRA-1: a transformation of a source-owned SoA
//! tile as an immutable, composable address recipe.
//!
//! - [`recipe`]: the eight symmetries of a square tile as three bits, with
//!   composition, inverse and normalisation. Nothing is read.
//! - [`grid`]: the source owner behind the DT64 lane socket.
//! - [`view`]: bound lanes seen through a recipe; row objects and column
//!   descriptors that read nothing until a value is demanded.
//! - [`route`]: folds over view columns through Quack and mask-RISC, with a
//!   row reversal dropped only where the terminal ignores row order.
//! - [`perm`]: a row bijection and the Boolean rewrite it licenses.
//! - [`physical`]: the reference that moves data, one matrix operation at a
//!   time; the oracle and the expensive baseline.
//! - [`cube`]: the 256^3 address space and three relation families.
//!
//! Nothing here is an owner, a store or an executor, and no contract type is
//! added. Pass conditions: `.claude/plans/d-virtual-address-algebra-1.md`.

#![forbid(unsafe_code)]

pub mod cube;
pub mod grid;
pub mod perm;
pub mod physical;
pub mod recipe;
pub mod route;
pub mod view;
