// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DeepNSM-CAM: Semantic transformer for the thinking pipeline.
//!
//! 4,096 words x 12 bits. O(1) per word, O(n) per sentence.
//! No transformers, no GPU, no regex. Exact. Deterministic. Bit-reproducible.
//!
//! Pipeline: text -> tokenize -> parse SPO -> encode -> compare via calibrated similarity.

pub mod tokenizer;
pub mod parser;
pub mod encoder;
pub mod similarity;
pub mod nsm_word;
