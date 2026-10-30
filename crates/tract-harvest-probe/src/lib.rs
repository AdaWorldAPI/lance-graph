//! D-TRACT-HARVEST-0 lab probe: an ONNX graph as a DR#3 graph, resolved
//! through the DT64 ON × WHEN × DO registry, with ONNX Runtime as the
//! independent oracle. Plan: `.claude/plans/d-tract-harvest-0-native-algebra-bridge-v1.md`.
//!
//! ```text
//! .onnx bytes ──► source::SourceInventory     (raw protobuf: names, domain, opset, attributes, constants)
//!      │                    │
//!      └──► Tract (cold) ──►│ dr3::tract_facts (dtype, shape per value)
//!                           ▼
//!            dr3::OnnxGraph   nodes = Identity, values = Property,
//!                             on / do_ = 64x64 relations, dep = do_ ∘ read_by
//!                           ▼
//!            plan::resolve    op signature -> codebook item (lens, ordinal),
//!                             WHEN (K3) over the facts, schedule from dep, Frame sealed
//!                           ▼
//!            Plan::run        each node routed to its DO (lance-graph-onnx / ndarray)
//! ```
//!
//! Execution never calls Tract. No native crate depends on this crate.

pub mod bind;
pub mod dr3;
pub mod fold;
pub mod gather;
pub mod plan;
pub mod provenance;
pub mod registry;
pub mod source;
pub mod tract_index;
pub mod triage;
pub mod value;
