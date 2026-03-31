//! # BGE-M3: Multilingual Embedding via bgz-tensor Compiled Attention
//!
//! XLM-RoBERTa architecture. 100+ languages. No API.
//! Inference via bgz-tensor: attention as table lookup, not matmul.

pub mod weights;
pub mod embed;
pub mod tokenizer;
