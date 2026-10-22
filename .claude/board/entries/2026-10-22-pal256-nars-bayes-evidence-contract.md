# NARS revision and Bayesian updates: one measurable evidence basis

Status: **lab-only executable mathematics**, not a replacement for `causal-edge` truth semantics.

Existing measured result: `.claude/board/entries/2026-10-10-d-rpf-p5-nars-beta-correspondence.md`. NARS expectation with horizon k=1 equals the Jeffreys Beta(1/2,1/2) posterior mean for Bernoulli counts **before u8 information loss**; that is a special correspondence, not universal Bayesian equivalence.

Proposed cold typed algebra: `EvidenceCounts {positive, negative, independent_sources}`. Allow exact count sum only when source roots are disjoint; preserve n with a provenance witness outside the u8 F/C lanes. A future `BayesianLogLikelihood` type is distinct and may be summed only under its own independence/model assumptions. Absolutely no `sum(atanh(cosine)) == NARS revision` rule.

Falsifiers: same f at very different n has different c; duplicate roots refuse; two groups with 7/10 produce 14/20, *not* tanh(2*atanh(0.4)).

Run: `cargo test --manifest-path crates/causal-edge/Cargo.toml --test palette256_nars_bayes_evidence_probe`.
