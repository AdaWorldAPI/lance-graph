//! # lance-graph-tikv-provider — Glue #2
//!
//! This crate is the **Glue #2** bridge described in
//! `integration-plan.md §5`: it implements DataFusion's `TableProvider`
//! trait over TiKV key ranges, producing Arrow `RecordBatch` output that
//! drops directly into lance-graph's Cypher executor with zero copy.
//!
//! ## Role in the stack
//!
//! TiKV stores graph elements as encoded key-value ranges. This crate
//! translates a DataFusion scan request (filters + projection + limit)
//! into a TiKV range scan, decodes the raw bytes into Arrow columns
//! using the schema declared in `lance-graph-catalog`, and returns an
//! `ExecutionPlan` that streams `RecordBatch` rows back to the planner.
//!
//! The MVCC snapshot timestamp (`snapshot_ts: u64`) is the same number
//! as surrealdb-core's `version` column, TiKV's native HLC timestamp, and
//! the version a Lance projection refresh commits at — **one clock, all
//! storage targets** (plan §5 "Snapshot integration").
//!
//! ## What is stubbed vs implemented
//!
//! **Everything in this crate is a Sprint 0 surface lock.** All method
//! bodies contain `unimplemented!()` with a Sprint 1 TODO comment.
//! The public API (struct fields, method signatures, trait impls) is the
//! stable surface; the implementation lands in Sprint 1 per plan §7.
//!
//! ## Modules
//!
//! - [`node`] — `TikvNodeTableProvider` for node shapes.
//! - [`edge`] — `TikvEdgeTableProvider` for edge shapes.
//! - [`scan`] — `TikvScanExec` physical execution plan stub.
//! - [`error`] — crate-local error type.

pub mod edge;
pub mod error;
pub mod node;
pub mod scan;
