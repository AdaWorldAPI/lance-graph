# D-STAUNEN-2 | Biophysical/hyperdimensional execution | draft
**Docs only. No new simulator, copied topology, alternate event schedule or unverified hardware claim.**
## First pass
- Pin Atlas source (paper `ae649a2...`, evidence archive biorxiv-v1) and select one existing STDP fixture, with definition/instance/run identity and complete spike/trace/weight outputs.
- Treat AtlasIR, its reference executor and effect algebra as upstream semantic owner. Study adapter at validated event/observation level; do not alter core Atlas kernels.
- HDC: fixed seeded neuron/time role codebooks -> bind/bundle/permutation -> binary xor/popcount -> prediction residual / novelty signal. Keep provenance from spike event to hypervector: lossy bundling alone cannot assign synaptic credit.
- Moore8 is a bounded *view* onto actual Atlas/DT64 source-to-target synaptic adjacency; track edge identity and delay; no spatial-neighbor equivalence by assumption.
- Three-factor learning e_ij(t) × signed M(t), event-driven traces/conditional STDP; novelty only a modulator. Distinguish functional weight update, fixed-candidate activation, true new edge creation, and myelination.
- CPU costs: per primitive in ns, batch in µs; complete simulation and preparations separately costed. Preserve reference-f64 semantic profile, declare any approximate f32/LUT profile.

## Research gates
A0 source/feature/license inventory; A1 frozen traces and scheduling oracle; A2 binary HDC coding parity; A3 correct identities from Moore; A4 novelty ablation; A5 exact vs approximate numeric; A6 fixed-candidate structural plasticity; A7 transfer and performance.
**Prereq:** D-STAUNEN-1 for admission; AtlasIR validated fixtures; no hardware requirements.
