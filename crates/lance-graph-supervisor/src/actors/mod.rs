//! Per-consumer actor implementations.
//!
//! Each active G slot has one actor. The `StubConsumerActor` (in `supervisor.rs`)
//! serves as the skeleton. Concrete implementations live here:
//!
//! - `medcare_actor.rs` — `MedcareConsumerActor` (G=2,  HEALTHCARE_V1, proof-of-concept)
//! - `odoo_actor.rs`    — `OdooConsumerActor`    (G=50, ODOO_V1,       proof-of-concept)
//!
//! Future:
//! - `ogit_actor.rs`  — OgitBridge actor (G=4, SMB_V1)
//! - `woa_actor.rs`   — WoaBridge actor   (G=3, GOTHAM_V1)
//!
//! # BBB invariant
//!
//! Actors receive `ConsumerEnvelope` and return `ConsumerReply`. Internal
//! substrate types (`Vsa10k`, `Vsa16kF32`, `RoleKey`, `SemiringChoice`,
//! Arrow scalars) never cross the actor mailbox boundary.

pub mod medcare_actor;
pub mod odoo_actor;

pub use medcare_actor::MedcareConsumerActor;
pub use odoo_actor::OdooConsumerActor;
