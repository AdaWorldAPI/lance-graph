//! AutocompleteCache: 4096 interdependent attention heads as cognitive substrate.
//!
//! HHTL Resolution:
//!   HEEL:  8×8    = 64 super-blocks     (routing)
//!   HIP:   64×64  = 4096 heads          (attention topology)
//!   TWIG:  256×256 = 65536 heads        (fine-grain)

pub mod candidate_pool;
pub mod convergence;
pub mod kv_bundle;
pub mod lane_eval;
pub mod nars_engine;
pub mod triple_model;
