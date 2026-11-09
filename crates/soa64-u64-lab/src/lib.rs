//! SOA64-U64 R1 lab.
//!
//! Compares five physical representations of the same 64-field logical row
//! set (array-of-structs, u64-lane SoA, narrow-lane SoA, bit-packed SoA, and
//! a per-element identity-lookup null) under demand-shaped queries, with the
//! DT64 lane socket as the coordinate-space witness.
//!
//! * [`fixture`] is the only source of truth for values and presence.
//! * [`arms`] holds the representations; all answer the same [`arms::Query`].
//! * [`oracle`] computes every query straight from the fixture, never from an arm.
//! * [`socket`] binds demanded lanes through the DT64 lane socket.

pub mod arms;
pub mod fixture;
pub mod oracle;
pub mod socket;
