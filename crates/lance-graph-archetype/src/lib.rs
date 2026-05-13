//! # lance-graph-archetype
//!
//! Archetype transcode scaffold. Per ADR-0001 Decision 1, this crate
//! defines its OWN Rust interface for ECS-style components, processors,
//! a world, and a command broker — it is NOT a mirror of the Python
//! `VangelisTech/archetype` API. The Python repository is a design
//! spec; there is no runtime dependency on it.
//!
//! Per ADR-0001 Decision 3, every type defined here is INSIDE-BBB:
//! none of them cross the external membrane onto
//! `CognitiveEventRow`. The scalar projection for "an archetype tick
//! happened" is already carried by
//! `CognitiveEventRow.cycle_fp_hi/lo` + `MetaWord`.
//!
//! ## Module layout
//!
//! - [`component`] — the `Component` trait (Arrow-field projection).
//! - [`processor`] — the `Processor` trait (RecordBatch transform).
//! - [`world`] — `World` meta-state (tick + dataset URI; `fork` / `at_tick`
//!   are stubs pending DU-2.8).
//! - [`command_broker`] — `CommandBroker` + `Command` (deferred world
//!   mutations, drained at tick boundaries).
//! - [`error`] — `ArchetypeError` (thiserror-backed).
//!
//! ## Status
//!
//! Scaffold only (DU-2.1..2.6). No runtime behaviour yet. See
//! `.claude/plans/archetype-scaffold-v1.md` for scope.

#![deny(missing_docs)]

pub mod command_broker;
pub mod component;
pub mod error;
pub mod processor;
pub mod world;

pub use command_broker::{Command, CommandBroker};
pub use component::Component;
pub use error::ArchetypeError;
pub use processor::Processor;
pub use world::World;
