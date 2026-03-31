//! # Reader LM: HTML→Markdown via Qwen2 1.5B Inference
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
