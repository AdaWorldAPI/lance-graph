# PR sweep — lance-graph #1145–#1136 (verbatim, 2026-09-05)

- #1145 — D-POP-2 `elect_and_bind`/`WitnessLens::bind_election` writes elected Quorum/Contradiction loci into the row's own CausalWitness register. STRONG. Ruling: "the fabric never reads what it computes".
- #1144 — D-TEH-3 fate probes: `semantic_chunker`, `spiral_segment` both KILL, stay LAB; 25 clippy fixes. WEAK.
- #1143 — calibration math → `jc::{drift, quorum, reliability}` (`pairwise_agreement_u8`/`QuorumLevel`, `reencode_drift`); lab `cronbach.rs` deleted. STRONG. Ruling `E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1`.
- #1142 — `nars::ghost_prior::GhostPrior` in planner; floor gate default `Marker` (0.1 clamp) over `Trace` (0.001 prune). STRONG. Ruling: "Lingering trace ≠ the −6 counterfactual lane".
- #1141 — PROBE-HOUSE-DIFFERENTIAL-1; variant 1 KILL, variant 2 base PASS. WEAK. Ruling: "The periphery of a stratum is the other strata".
- #1140 — Pillar-11 closure: guard correction (arXiv-e → Annals-2e), ndarray SIMD substrate debt, A0/A1/A2 Goursat SIMD bench (F64x8 FMA), AMX/SPR-vs-EMR note. STRONG. Ruling (TD-PILLAR11): "call the named algorithm; if absent, compose from ndarray::simd::{F64x8,F32x16,I16x32}::*; never a consumer-local arithmetic path".
- #1139 — `bridge_gate` moved to contract; callcenter drops thinking-engine dep. NONE.
- #1138 — four operator rulings recorded. WEAK. "jc is the home of all scientifically calibrated math".
- #1137 — plans only. NONE.
- #1136 — PROBE-POP-READOUT-1 KILL (p@10 0.289→0.011); `curiosity_gestalt` rank-inert (ρ=1.000000 vs magnitude); frequency beats all cognitive arms. STRONG. Ruling: "any future frontier-ranking claim clears the frequency control first".
- Cross-cutting: none of the ten mention ternlogq, u64 row masks, version-keyed slab cache, Mexican hat, Shannon, proprioception, EWA/SPD by name.
