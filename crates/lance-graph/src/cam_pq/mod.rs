//! CAM-PQ DataFusion integration for lance-graph.
//!
//! This module wires ndarray's `cam_pq` codec into lance-graph's
//! DataFusion execution engine and Lance storage layer.
//!
//! # Architecture
//!
//! ```text
//! ndarray::hpc::cam_pq   →  THE CODEC  (encode, decode, distance, AVX-512)
//! lance_graph::cam_pq    →  THE WIRING (UDF, storage schema, IVF, jitson)
//! ```
//!
//! # Modules
//!
//! - `udf`: DataFusion scalar UDF `cam_distance(query, cam_column)`
//! - `storage`: Lance table schema for CAM fingerprints + codebooks
//! - `ivf`: IVF coarse partitioning for billion-scale search
//! - `jitson_kernel`: JITSON templates for compiled ADC scan pipelines

pub mod ivf;
pub mod jitson_kernel;
pub mod storage;
pub mod udf;
