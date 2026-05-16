// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AriGraph: OSINT knowledge graph with episodic memory.
//!
//! Transcoded from Python AriGraph — a memory architecture for LLM agents.

pub mod episodic;
pub mod language;
pub mod orchestrator;
pub mod retrieval;
pub mod sensorium;
pub mod spo_bridge;
pub mod triplet_graph;
pub mod witness_corpus;
pub mod xai_client;

pub use witness_corpus::{CamPqWitnessIndex, WitnessCorpus, WitnessEntry, WitnessId};
