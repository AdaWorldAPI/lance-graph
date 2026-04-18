//! # cognitive-shader-driver — the shader IS the driver
//!
//! End-to-end wiring of the four components that were previously isolated:
//!
//! ```text
//! lance-graph-contract::cognitive_shader  ←  zero-dep DTOs
//!              │
//!              ▼
//! cognitive-shader-driver (THIS CRATE)
//!    ├── BindSpace (struct-of-arrays, genius packed layout)
//!    ├── p64_bridge::CognitiveShader (8 planes × bgz17 O(1) distance)
//!    └── thinking-engine hook (optional, behind `with-engine` feature)
//!                                            │
//!                                            ▼
//!              cycle_fingerprint emitted through ShaderSink
//! ```
//!
//! ## Role Reversal (the genius DTO API)
//!
//! The shader no longer sits below the engine. It holds the BindSpace columns,
//! reads the packed u32 MetaColumn FIRST (cheapest prefilter), then loads only
//! the Fingerprint rows that passed. Each cycle emits exactly one
//! `Fingerprint<256>` (the cycle_fingerprint), which is simultaneously:
//!
//! - a cache key (A2A blackboard lookup),
//! - a retrieval key (BindSpace similarity sweep),
//! - a replay key (deterministic given the same dispatch),
//! - a cursor (next-cycle seed).
//!
//! ## EmbedAnything Patterns
//!
//! - **Builder** (`CognitiveShaderBuilder::new().rows(..).style(..).sink(..).build()`)
//! - **Auto-detect** (`StyleSelector::Auto` routes from qualia column)
//! - **Commit sinks** (`ShaderSink` trait; `NullSink` as zero-cost default)
//! - **Feature gates** (`with-engine` pulls thinking-engine; core stays lean)
//! - **No forward pass at runtime** — bgz17 distance is precomputed

#![warn(rust_2018_idioms)]

pub mod bindspace;
pub mod driver;
pub mod auto_style;

pub use lance_graph_contract::cognitive_shader::{
    CognitiveShaderDriver, ColumnWindow, EmitMode, MetaFilter, MetaSummary, MetaWord,
    NullSink, RungLevel, ShaderBus, ShaderCrystal, ShaderDispatch, ShaderHit,
    ShaderResonance, ShaderSink, StyleSelector,
};

pub use lance_graph_contract::collapse_gate::{GateDecision, MergeMode};

pub use bindspace::{BindSpace, BindSpaceBuilder, EdgeColumn, FingerprintColumns,
                     MetaColumn, QualiaColumn};
pub use driver::{CognitiveShaderBuilder, ShaderDriver};
