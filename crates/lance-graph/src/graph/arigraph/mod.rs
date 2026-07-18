// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AriGraph: OSINT knowledge graph with episodic memory.
//!
//! Transcoded from Python AriGraph — a memory architecture for LLM agents.

pub mod bm25;
pub mod community;
pub mod episodic;
pub mod language;
pub mod markov_soa;
pub mod orchestrator;
pub mod ppr;
pub mod retrieval;
pub mod sensorium;
pub mod spo_bridge;
pub mod triplet_graph;
pub mod witness_corpus;
pub mod xai_client;

pub use bm25::Bm25Index;
pub use community::Communities;
pub use episodic::EpisodicBasins;
pub use ppr::PersonalizedPageRank;
pub use witness_corpus::{WitnessCorpus, WitnessEntry, WitnessId, WitnessIndexHashMap};

#[cfg(feature = "with-cam-pq")]
pub use witness_corpus::{spo_to_fingerprint, CamPqState, WitnessIndexCamPq};
