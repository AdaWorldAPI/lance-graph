//! # CausalEdge64: The 64-bit Causal Neuron
//!
//! One atomic u64 encoding the complete causal unit:
//! SPO palette indices, NARS truth value, Pearl's causal hierarchy,
//! direction diagnostic, inference type, plasticity state, and temporal index.
//!
//! ## Bit Layout
//!
//! ```text
//! 63        52 51   49 48   46 45   43 42  40 39      32 31      24 23  16 15   8 7    0
//! ┌──────────┬───────┬───────┬───────┬──────┬──────────┬──────────┬──────┬──────┬──────┐
//! │ temporal  │plasti │infer  │direct │causal│  NARS c  │  NARS f  │  O   │  P   │  S   │
//! │  12 bit   │3 bit  │3 bit  │3 bit  │3 bit │  8 bit   │  8 bit   │ 8bit │ 8bit │ 8bit │
//! └──────────┴───────┴───────┴───────┴──────┴──────────┴──────────┴──────┴──────┴──────┘
//! ```
//!
//! ## Why 64 bits
//!
//! - One register on ARM64 and x86_64. Atomic read/write.
//! - One cache line holds 8 edges (64 bytes). Graph traversal = sequential reads.
//! - Compare two edges: one XOR + popcount gives structural + epistemic distance.
//! - Sort edges: native u64 sort gives temporal ordering for free (temporal in MSBs).
//!
//! ## Pearl's Causal Hierarchy as 3-bit Mask
//!
//! ```text
//! mask = 0b000 = ___  → prior (no conditioning)
//! mask = 0b100 = S__  → subject marginal
//! mask = 0b010 = _P_  → predicate marginal
//! mask = 0b001 = __O  → outcome marginal
//! mask = 0b110 = SP_  → confounder detection
//! mask = 0b101 = S_O  → Level 1: Association   P(Y|X)
//! mask = 0b011 = _PO  → Level 2: Intervention  P(Y|do(X))
//! mask = 0b111 = SPO  → Level 3: Counterfactual P(Y_x|X',Y')
//! ```
//!
//! ## NARS in Register
//!
//! Frequency (u8): f = val / 255.0, 256 quantization levels.
//! Confidence (u8): c = val / 255.0, 256 quantization levels.
//! At u8 resolution, revision/deduction/induction/abduction can be
//! precomputed as 256×256 lookup tables (64 KB each, fits L1 cache).
//!
//! ## Plasticity
//!
//! 3 bits encode per-plane plasticity state:
//! - bit 0: S-plane (0=frozen/cold, 1=hot)
//! - bit 1: P-plane
//! - bit 2: O-plane
//!
//! Hot planes accept palette reassignment under evidence pressure.
//! Frozen planes are established clinical patterns.

pub mod edge;
pub mod tables;
pub mod pearl;
pub mod plasticity;
pub mod network;

pub use edge::CausalEdge64;
pub use pearl::CausalMask;
pub use plasticity::PlasticityState;
