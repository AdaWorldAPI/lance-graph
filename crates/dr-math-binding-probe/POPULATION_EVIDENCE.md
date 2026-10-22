# D-DRMC-2 — Population evidence is not proof of universal mathematics

This stage adds a deliberately small independent **audit/reference** implementation for one 64×64 focus, plus scalar-per-pair differential fixtures. It has zero data storage and does not run in a hot application path.

`Gamma` (constraints, two-valued) and `R` (proposed relationship) are bound through Stage 1: same row/column coordinate domains, the same issue of the same store-sealed frame (still current), and the same focus. A three-valued `R` carries its own validity bits; there is no separately bound validity plane. `UNKNOWN` is not FALSE, and a counterexample is reported before any unknown pair: one observed FALSE refutes the claim whatever the unknown pairs hold. The fixed 4096-pair focus is not the global population, and a DR bit does not encode multiplicity, confidence, observational provenance, or independent data quality.

`universal_on_focus` checks, in order, `eligible > 0`, `false == 0`, `unknown == 0` over *these supplied bits*. This is a **finite-focus result**, not a generalized theorem, a causal discovery, or source attestation. Caller-supplied bits and revision numbers can be fabricated. In particular, no production method may call this a `CertifiedLaw` until an independently sealed source snapshot, completeness proof, interpretation of the constraints, and independent oracle have been checked.

Next falsifiers for an actual discovery pipeline:
1. Compare against authoritative source observation IDs and a scalar/SQL/datafrog oracle; measure distinct/bag derivations separately.
2. Add a source seal and valid coverage receipt, distinguish a selected focus from exhaustive whole-population enumeration, and revalidate any learned threshold/rule against a held-out population.
3. For candidate algebraic rewrites, check the statement under formal mathematical assumptions through an independent proof checker. egglog e-graph extraction alone cannot prove it holds in nature.
4. Preserve negative controls: empty sample, unknown observations, one contradicting member, stale revision, changed authorization, out-of-focus true member, cyclic/self-supporting derivations.

Runtime masks/folds remain Quack/R2IL/mask-risc or the established direct resident SIMD fast path. The `observe_focus` loop is test/reference checking only and not an alternative production evaluator.
