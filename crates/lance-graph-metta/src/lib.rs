//! MeTTa (Hyperon) as a declaration front end over the masking-algebra
//! polyfill. Nothing is shipped here yet: D-MTP-1 is a probe
//! (`examples/join_probe.rs`) that lowers one MeTTa conjunctive query by hand
//! through `lance-graph-quack` into one mask-risc program and checks it
//! against the datafrog oracle. The `Space` adapter is D-MTP-2.
//!
//! Plan: `.claude/plans/metta-polyfill-v1.md`.
