//! Map and teleport for the 64k ordered lane.
//!
//! This crate does not execute a `Program`. It draws the map and names the
//! landing. Quack still lowers. Mask-risc still executes. Do not route this
//! through the cognitive planner.

#![forbid(unsafe_code)]

mod query;
mod stars;

pub use query::{
    Aperture, ApertureId, BestQuery, Hom, LaneId, Query, Refuse, Request, Row, Terminal, N,
    MASK_WORDS,
};
pub use stars::{Landing, Map, MapOp, Star, Teleport};
