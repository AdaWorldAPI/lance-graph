//! D-TRACT-HARVEST-0 lab probe: Tract as a cold typed ONNX frontend, admission
//! onto existing native methods, ONNX Runtime as the independent oracle.
//! Plan: `.claude/plans/d-tract-harvest-0-native-algebra-bridge-v1.md`.
//!
//! ```text
//! .onnx bytes ──► source::SourceInventory      (raw protobuf: domain, opset, initializers)
//!      │
//!      └──► Tract InferenceModel ─► TypedModel  (cold; shapes, dtypes, constants)
//!                                     │
//!                       admit::admit_*(source, typed) ─► Recipe + Receipt | Refusal
//!                                                         │
//!                               Recipe::run ─► lance-graph-onnx / ndarray only
//! ```
//!
//! The recipes never call Tract. No native crate depends on this crate.

pub mod admit;
pub mod gather;
pub mod provenance;
pub mod source;
