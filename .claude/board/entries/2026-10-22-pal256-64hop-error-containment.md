# Palette256: 64-hop containment is a falsifiable hypothesis, not a hallucination guarantee

Existing reference surfaces: `bgz-tensor::fisher_z::FisherZTable` immutable family LUT; `ndarray::hpc` statistical moments; `lance-graph-nal` evidence-source stamping; Math Foundry cold proof/extraction. No rewriting of these engines.

Test 1: repeated lookup by stable references has no *repeated encode/decode*. Test 2: a **specified** contraction (`L=0.7`) with a bounded rounding term satisfies a 64-hop numerical envelope. Negative control: a bounded but expansive recurrence grows a false belief; 32 derived paths from one root are still one independent source.

Next experiment must use real `FisherZTable` pairwise score compositions, 1/2/4/8/16/32/64 hops, exact f64 oracle, held-out path labels, source disjointness, min-max/mean/p95/p99 numerical error, calibration, false-positive and abstention rates, performance and adversarial correlated roots. Count sample size per *independent original observation*. A rho 0.9998 on 2000 single pairs proves neither 64-hop rank preservation nor semantic validity.

Run: `cargo test --manifest-path crates/bgz-tensor/Cargo.toml --test palette256_multihop_error_probe`.
