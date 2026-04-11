//! # Reader LM 1.5B — RESEARCH-ONLY (Qwen2 legacy lineage)
//!
//! ## RESEARCH-ONLY — do not use for production
//!
//! This crate targets the **Reader-LM 1.5B v1/v2 Qwen2 variant** — the
//! PRE-v5-era lineage. The current production Reader-LM target is
//! **Reader-LM v3 = Jina v5** (BERT 3.x architecture, see
//! `ndarray::hpc::jina::runtime::ModelSource::JinaV5`). The "v3" in
//! "Reader-LM v3" names a NEW lineage (Jina v5 BERT 3.x), not an iteration
//! on the Qwen2 v1/v2 line this crate covers.
//!
//! Keep this crate for **behavioral diffing**: when a Jina v5 forward pass
//! produces a surprising result, comparing against this older pipeline
//! helps isolate what architectural change between Qwen2 and Jina v5 BERT 3.x
//! is responsible. Do NOT reach for this crate when building new production
//! wiring — use Jina v5 directly instead.
//!
//! See `lance-graph/CLAUDE.md` → `Model Registry` → `Research-only /
//! diagnostic fallback` for the canonical policy.
//!
//! ## Original design notes (valid for the Qwen2 1.5B target)
//!
//! Transcoded from jinaai/reader-lm-1.5b. Qwen2 architecture:
//! RoPE + GQA (grouped query attention) + SwiGLU FFN.
//!
//! Two modes:
//! - **Palette mode**: bgz7 weight fingerprints → O(1) structure classification
//! - **Inference mode**: full forward pass → token-by-token Markdown generation
//!
//! The palette mode is instant (17K tok/sec). The inference mode is accurate
//! but slower (~30 tok/sec on CPU). Use palette for routing decisions,
//! inference for actual HTML→Markdown conversion.

pub mod weights;
pub mod inference;
pub mod classifier;
pub mod tokenizer;
