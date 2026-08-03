//! ⚠ STATUS: NO CONCRETE ACTOR SHIPS HERE YET — the one module below is an
//! unwired stub. Every G slot the supervisor actually spawns gets
//! `supervisor::StubConsumerActor`; nothing in this directory is reachable at
//! runtime. Treat the contents as shape/reference, not as consumer wiring.
//!
//! Per-consumer actor implementations. Each active G slot has one actor.
//!
//! - `medcare_actor.rs` — `MedcareConsumerActor` (G=2, HEALTHCARE_V1) —
//!   UNWIRED stub; the generalization candidate for `ConsumerActor<P: PortSpec>`
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

pub use medcare_actor::MedcareConsumerActor;
