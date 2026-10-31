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

## Round 2 | Concrete Atlas/HDC integration probe sequence for CC

### Upstream acceptance boundary
Atlas simulation is defined by **Definition/Instance/Run identities**, ordered effect graph (RAW/WAR/WAW), SI dimensions, scalar types, RNG streams and original synapse IDs. Its separately versioned *training plan* implements bounded surrogate VJP, BPTT and Adam/SGD; do not pass training through simulation AtlasIR expecting gradients. Brian2-Atlas's imported Brian2 files are CeCILL-2.1; new Atlas engine original contributions Apache-2.0. Source-paper test cohorts belong to frozen revisions, not necessarily current `dev`.
- Primary: https://github.com/RockLi/brian2-atlas-preprint/blob/main/paper/MANUSCRIPT.md, §§2.2–2.5, 3.7, 3.8, 4.2, 5.7, 6.4.
- AtlasIR frozen schema: https://github.com/RockLi/brian2-atlas/blob/dev/brian2-rust/AtlasIR.md
- Code: https://github.com/RockLi/brian2-atlas/tree/v0.1.0

### A0: get complete biological oracle
Pick a *supported existing* Pair/Triplet STDP small ring or a bounded LIF neuron/synapse fixture. Pin code and values: `definition_hash`, `instance_hash`, `run_hash`, clocks, source-synapse creation order, spike IDs and ticks, pending event queue, neuron voltages, plasticity traces, final weights, RNG stream, execution profile. Baseline is the independently validated Atlas reference; Brian NumPy/C++ when a supported same-model path exists. A nonrecorded field may not be declared equivalent.
### A1: event projection (no semantic mutation)
Adapt `NamedEventStream` or result records into a *borrowed* inspection lens over DR-mapped neuron/synapse identities; preserve (population, global neuron, edge, timestep/tick, generation). Join to existing relationship topology by typed source+target mapping, not Moore index. No new persistent event store and no per-spike owned tuple materialization on native fold.
### A2: reproducible HDC codebooks
Declare vector dimension as multiples of 64 (e.g. 1,024 / 4,096), seed, encoder version, temporal buckets and binning method. `bind(neuron_code, time_role)`, signed/count-aware bundle, permute for order, `xor`/`popcount` for Hamming. Compare to scalar oracle. Test one spike, repeated spike, simultaneous events, reordered spikes, time shift, unknown neuron, overflow, empty window, random-code collisions and deterministic rebuild. Do not infer synapse identity by inverting a lossy bundle.
### A3: Moore neighborhood is a view
Use exact actual input/output edge identities. Eight Moore slots = local geometric/palette projection for candidate *focus*, not eight outgoing connections. Compare Moore-restricted vs true adjacency, topology-shuffled, direction-shuffled and full-neighbor baseline. Conditional path only, with correct missing-neighbor and high-degree refusals.
### A4: novelty-based eligibility
Maintain in Atlas physics `e_ij(t)` for nominated edges; let a signed, calibrated third factor `M(t)` depend on held-out prediction error, reward or novelty with recorded source. For a simple declared rule compare `delta_w = eta * M * e_ij`. Strong baseline = Atlas STDP without novelty; negative = shuffled / constant novelty; extra = same number of candidate updates under random selection. A high prediction error cannot directly authorize a new anatomical edge.
### A5: structural scope
First propose **fixed-candidate** dormant synapses, legal under Atlas's bounded structural-plasticity profile. Test activate, deactivate, generation guard on delayed events, weight/trace reset/restore, duplicates and invalid target refusal. True synaptogenesis and adaptive myelination require new calibrated physiological models and separate studies, not renamed graph writes.
### A6: precision and scheduling
Reference f64 scalar results vs supported native/f32 paths *only* under declared numeric profile/tolerances; compare exact spike times when the baseline declares exactness. Reject reorder of synaptic reductions that modifies later spikes; cache/LUT exponential may require distinct profile and scientific revalidation. Exploit event sparsity only with old-state/effect-safe eligibility proof.
### A7: econometrics and reporting
Hot primitive `xor`, `popcount`, local map lookup and one eligible delta: ns individually; bounded batch: µs. Charge admission, routing, input/output, state preparation, recording and compilation separately. Compare per active edge, per simulated tick and full trajectory in *distinct* tables; be explicit that biological timestep speed is not implied by 1 ns register ops.

### CC deliverables and stop tests
- One pinned reference fixture; one independent HDC scalar oracle; one source-to-neighborhood mapping oracle; six disable-to-red probes; lossless source ID/event provenance; same-seed replay; performance gate only after parity.
- A pass does **not** establish brainwide anatomy, virtual consciousness, physiological white-matter model, cross-platform bit-identical trajectories, or real chip performance.
- If no public, stable Atlas event adapter exists, stop and file an owner question instead of vendor-copying the executor.

### Science and context
- Gerstner et al. 2018 three-factor traces: https://doi.org/10.3389/fncir.2018.00053
- Frey & Morris 1997 synaptic tags: https://doi.org/10.1038/385533a0
- Lisman & Grace 2005 novelty/LTP: https://doi.org/10.1016/j.neuron.2005.05.002
- Hedrick et al. 2022 clustered spinogenesis: https://doi.org/10.1038/s41593-022-01086-6
- McKenzie et al. 2014 myelin learning: https://doi.org/10.1126/science.1254960
- SpikeHD https://doi.org/10.1038/s41598-022-11073-3 ; spiking phasors https://pubmed.ncbi.nlm.nih.gov/39106467/ ; SynapseHD https://doi.org/10.1016/j.neucom.2025.131757 ; hyperdimensional SNN decoding https://doi.org/10.1088/2634-4386/ae4f1e
- Denève 2008 derives a *model-dependent* local Bayesian-learning rule, not general identity with STDP: https://doi.org/10.1162/neco.2008.20.1.118
- Moore-local plans https://github.com/AdaWorldAPI/lance-graph/pull/1405 and https://github.com/AdaWorldAPI/lance-graph/pull/1408

**Owner handoff:** CC may land a *lab-only* branch for A0–A3 with no Atlas changes. A4–A7 require owner approval/semantic admission and a separate scientific PR.
