//! AutocompleteCache: 4096 interdependent attention heads as cognitive substrate.
//!
//! HHTL Resolution:
//!   HEEL:  8×8    = 64 super-blocks     (routing)
//!   HIP:   64×64  = 4096 heads          (attention topology)
//!   TWIG:  256×256 = 65536 heads        (fine-grain)

/// D-BBB-NARS-2 cross-crate fuse: `contract::assertion_wire` byte positions and
/// vocabularies against `causal_edge` — measurement only, compiled out of every
/// non-test build.
#[cfg(test)]
mod assertion_wire_parity;
pub mod candidate_pool;
pub mod convergence;
pub mod kv_bundle;
pub mod lane_eval;
pub mod nars_engine;
/// Stage-2.6a V3 representation-parity census — measurement only, compiled out
/// of every non-test build.
#[cfg(test)]
mod stage26_v3_parity;
pub mod triple_model;
