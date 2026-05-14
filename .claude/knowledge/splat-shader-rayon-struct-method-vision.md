# Splat BLAS Shader Ops + ndarray Struct Methods — Future Vision

> **READ BY:** integration-lead, family-codec-smith, palette-engineer, anyone scoping post-sprint-10 cognitive shader op fleet or ndarray rayon work-stealing wiring
>
> **PREREQUISITES:** `cognitive-shader-driver-thinking-engine-reunification.md`; `splat` references in `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md`, `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`, `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §9
>
> **Status:** CONJECTURE — design vision for sprint-12+; no runtime implementation claimed; foundation in plan documents already in workspace

---

## 1. The Splat Op — Spatial BLAS Over the SPOW Hole Board

Per `tetrahedral-epiphany-splat-integration-v1.md`:

> *"The 4096×4096 field is not primarily a distance ledger. It is a deterministic question surface:*
> *`[A{4096}, B{4096}]::[projection_mask]() -> C{4096}`"*

The 4096×4096 surface (COCA vocabulary × COCA vocabulary) becomes a **question surface** when paired with a projection mask. Two known factors + a hole → the missing factor is computed by projection.

### 1.1 What the splat does

Per `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §9:

```text
splat C across the local SPOW neighborhood
  under the selected cause/effect and palette/attention settings
  then score how the surrounding hole board changes.
```

Inputs:

```text
candidate answer            ← the proposed C
GestaltCause64              ← 64-bit causal/semantic shape of the current hole-state
ThinkingEffect64            ← 64-bit cognitive move selected to transform that state
Palette256                  ← perception palette
Attention256                ← attention aperture
ReasoningWitness64 / NARS   ← truth conditioning
ontology_context_id         ← which G / DOLCE compartment
sigma / theta               ← Σ-tier + sub-tier modulation
Markov history refs         ← ±5..±500 episodic context
replay refs                 ← prior replay-certified candidates
```

The splat produces **pressure over nearby holes** — i.e., it doesn't just answer the immediate question, it **changes the local hole landscape** by deposit/withdrawal of evidence pressure on neighbors.

### 1.2 Why this is a BLAS-class op, not a CAM-PQ op

CAM-PQ answers: "find the top-K nearest neighbors of this query."

Splat answers: "deposit pressure proportional to (similarity × strength × ontology compatibility) over the local 4096×4096 neighborhood, then score downstream hole closure / contradiction / replay coherence."

The first is a **read** (ranked retrieval). The second is a **write** (perturbation deposit). Splat is closer in shape to a BLAS GEMM with a sparse projection mask — and it produces a **field**, not a ranked list.

### 1.3 What "spatial" means here

Per `tetrahedral-epiphany-splat-integration-v1.md`:

```text
2 x 64:
  GestaltCause64    = causal/semantic shape of the current hole-state
  ThinkingEffect64  = cognitive move selected to transform that state

2 x 256:
  AttentionIn256    = perception/search aperture
  AttentionOut256   = action/collapse/fanout aperture

4096:
  COCA meaning basis

The transition:
  GestaltCause64 + AttentionIn256 + meaning4096
    -> ThinkingEffect64 + AttentionOut256
```

The 64×64 + 256×256 + 4096×4096 grid is **the reasoning lattice**. The splat is a **3D Gaussian (or Fisher-Z) impulse** distributed over the lattice. The "spatial" aspect is that nearby cells in the lattice receive proportionally distributed pressure — exactly like a splat in graphics rasterization but in semantic space.

---

## 2. New Cognitive Shader Op Fleet

Beyond the current `cognitive-shader-driver` ops (encode/decode, palette quantize, NARS revise), the splat work suggests a richer op fleet:

| Op | Input | Output | Purpose |
|---|---|---|---|
| **splat_gaussian** | (center, sigma, strength, mask) | field perturbation 4096×4096 | Deposit evidence around a hole answer |
| **splat_fisher_z** | (center, partial correlation, mask) | field perturbation 4096×4096 | Deposit causal-conditioning evidence |
| **score_hole_closure** | (field, hole_locations) | per-hole closure scores | After splat, how much did each surrounding hole close? |
| **score_entropy** | (field) | scalar entropy | How much did splat lower local entropy? |
| **score_contradiction_localization** | (field) | (hole_locations, intensities) | Where did contradictions concentrate? |
| **replay_coherence** | (field, replay_refs) | scalar coherence | How well does this splat align with prior replay-certified candidates? |
| **emit_epiphany_witness** | (field, threshold) | Σ9-Σ10 witness emission | Trigger Pearl-3 if entropy drop crosses threshold |

These are all **field-level operations** on the 4096×4096 surface. They compose: a typical cycle is `splat → score → emit`.

### 2.1 BLAS-class implementations

Each op should be BLAS-class — meaning:
- SIMD-vectorized across the 4096-dim COCA basis
- Cache-line-aware (4096 × f32 = 16 KB = fits in L2 per row; full field 64 MB cold)
- Composable via the AttentionSemiring (per `crates/bgz-tensor/`)
- Parallelizable via rayon work-stealing (see §3 below)

The cycle budget per splat (CONJECTURE, needs profiling):
- splat_gaussian over a 64×64 neighborhood of a 4096×4096 field: ~10-50 µs at f32 SIMD
- score_hole_closure over 7 hole locations: ~1 µs
- score_entropy over full field: ~20 µs (sum-of-logs)
- Total cycle: ~50-100 µs for one splat + score pass

This sits between Zone-1 (ns) and Zone-2 (ms) — call it **Zone-1.5** or "splat-tier."

---

## 3. Rayon Work-Stealing + Struct Methods

The user's framing: "the chance to create ndarray and work stealing of rayon to offer struct methods can simplify the code and make it more meaningful and simpler at same time."

### 3.1 The current problem — free functions over raw arrays

Most workspace SIMD/BLAS code today is free functions that take raw arrays:

```rust
// Anti-pattern (free function on caller's state)
pub fn hamming_distance(a: &[u64; 256], b: &[u64; 256]) -> u32 { ... }
pub fn palette_compose(a: u8, b: u8, table: &[u8; 65536]) -> u8 { ... }
pub fn nars_revise(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) { ... }

// Caller has to assemble:
let dist = hamming_distance(&self.fingerprint, &other.fingerprint);
let composed = palette_compose(self.s_idx, other.s_idx, &PALETTE_TABLES.s_compose);
let (f, c) = nars_revise(self.frequency(), self.confidence(),
                          other.frequency(), other.confidence());
```

The caller has to know about every state slice and every table. The state is **scattered**.

### 3.2 The struct-method pattern (per CLAUDE.md "The Click")

CLAUDE.md is explicit:

> *"The object speaks for itself. `trajectory.resolve(ambiguity)` — not `resolve(trajectory, config, awareness, graph)`. Every method lives on the carrier that has the state to reason with it."*

> *"Litmus tests for any proposed change: Does this add a free function on a carrier's state, or a method on the carrier? → Free function = reject. Method = accept."*

The reunification path is methods on a unified carrier:

```rust
// Pattern: methods on the carrier
impl Think {
    pub fn resolve(&self) -> Resolution { ... }
    pub fn free_energy(&self) -> FreeEnergy { ... }
    pub fn splat(&mut self, candidate: SpoCandidate) -> SplatResult { ... }
    pub fn score_hole_board(&self) -> HoleBoardScore { ... }
    pub fn emit_epiphany(&self) -> Option<EpiphanyWitness> { ... }
}

// Caller:
think.splat(candidate).score_hole_board().emit_epiphany_if_threshold(0.05);
```

The state is **internal**. The carrier knows its trajectory, awareness, graph, codec, episodic memory. Methods reason over those without external assembly.

### 3.3 Rayon work-stealing for parallel struct methods

Where structs hold large internal state (e.g., 4096-atom energy vectors, 4096×4096 distance tables, 1M-row BindSpace SoA columns), the operations can parallelize via rayon:

```rust
impl BindSpace {
    pub fn dispatch_cycle(&mut self, cycle: u32) -> CycleReport {
        use rayon::prelude::*;

        // Parallel SIMD sweep over EdgeColumn — work-stealing scheduler
        // distributes 1M rows across cores.
        let reports: Vec<_> = self
            .edge_column
            .par_chunks_mut(SOA_CHUNK_SIZE)
            .map(|chunk| {
                chunk.iter_mut().for_each(|edge| {
                    edge.forward_inline(&self.palette_tables);
                });
                chunk.summarize()
            })
            .collect();
        CycleReport::merge(reports)
    }
}
```

The point: **rayon's par_iter / par_chunks_mut composes cleanly with struct methods**. The carrier exposes a parallel sweep; the scheduler steals work across cores. No external coordinator needed.

### 3.4 Where struct-methods + rayon land in the codec stack

Per `spo-ontology-format-stack.md`, each format has its own access pattern. Struct-methods + rayon land per-format:

| Format | Struct | Method | Rayon parallelism |
|---|---|---|---|
| **CausalEdge64** (SPO variant) | `BindSpace::EdgeColumn` | `dispatch_cycle()`, `forward_par()` | Row-level work-stealing over 1M-row column |
| **CausalEdge64** (8-channel) | `TierEngine` | `think_par()`, `emit_par()` | Atom-level work-stealing over 4096 atoms |
| **Base17** | `PaletteMatrix` | `compose_par()`, `mxm_par()` | Sparse-matrix block partitioning |
| **CAM-PQ** | `CamPqCorpus` | `query_par()`, `build_index_par()` | Codebook-cell partitioning |
| **bgz-hhtl-d** | `HipCache` | `cascade_par()` | Stage-pipelined cascade across cores |
| **3×16Kbit lossless** | `LosslessSpoStore` | `hamming_par()` | Row-level (each Hamming is itself SIMD-parallel) |

The **computational entropy reduction** comes from collapsing the free-function explosion into a small set of struct methods, each parallelized via rayon when state is amenable.

---

## 4. The Object-Oriented Computational Entropy Argument

The user's closing point: "(computational Entropy through struct object oriented)."

**Computational entropy** here is the information content needed to specify how operations compose. Higher entropy = more free functions, more raw types, more glue code; lower entropy = more methods on carriers, more uniform protocols, less glue.

Worked example — reasoning over a cognitive cycle today:

```rust
// HIGH entropy — caller assembles state from N sources
let energy = thinking_engine.compute_energy(&distance_table, &perturbation);
let top_k = energy_top_k(&energy, 8);
let edges = emit_causal_edges_from_top_k(&top_k, &distance_table);
let triplets = arigraph_lookup_targets(&edges, &entity_index);
let revised = nars_revise_triplets(&triplets, &edges);
let committed = spo_bridge_promote(&revised, &truth_gate);
let updated = bindspace_write(&committed, &mut soa);
```

Worked example — same cycle under struct-method reunification:

```rust
// LOW entropy — the carrier knows its state
let cycle = think.cycle();   // returns a CycleResult carrying all intermediate state
cycle.commit();              // promotes, revises, writes back into self
```

The composition is **internal to the Think struct**. The caller doesn't need to know about distance_table, entity_index, truth_gate, or soa — those are all internal state Think reasons over via methods.

### 4.1 Why this matters operationally

| Metric | Free-function approach | Struct-method approach |
|---|---|---|
| Caller LOC per cycle | ~7-15 lines (state assembly + N calls) | 1-3 lines (cycle + commit) |
| State-passing bugs | High (every parameter has to be right) | Low (state is internal) |
| Test surface | N free functions × M state combinations | M methods × internal invariants |
| Refactoring cost when struct changes | Cascading caller updates | Internal — caller untouched |
| Onboarding cost (new contributor) | Read N free function signatures | Read one struct + its methods |
| Parallelism wiring | Per-call site explicit rayon | Method-internal rayon (transparent) |
| Profiling / observability | Sprinkle metrics in N places | Single struct hooks; CycleResult carries everything |

The **entropy reduction is real and measurable**. Each consolidation removes one or more degrees of freedom from caller code.

### 4.2 Where to apply this — sprint-12+ targets

In order of value:

1. **`Think` struct unification** — collapse thinking-engine + cognitive-shader-driver SoA into one carrier (per `cognitive-shader-driver-thinking-engine-reunification.md` §5)
2. **`CamPqCorpus` struct** — replace free `cam_pq_query(&codebook, &codes, ...)` with `corpus.query(...)`; carrier holds codebook + CLAM tree + IVF inverted index
3. **`PaletteMatrix` already does this for bgz17** — preserve and extend; same pattern works for `BgzHhtlCache`
4. **`AriGraph` already does this for entity_index** — preserve; extend so `arigraph.commit(causal_edge)` is the one-call SPO promotion path
5. **`SplatField` struct** — for the new shader ops above; holds 4096×4096 perturbation field + scoring caches; methods are splat/score/emit

---

## 5. New ndarray Capabilities Worth Surfacing as Methods

Per CLAUDE.md "ndarray Integration Policy" + the `hpc-extras` feature gap (PR #364 / ndarray#116):

ndarray hosts (or will host post-PR-#116 merge):
- `Fingerprint<256>` — 16384-bit fingerprint type
- `SpoDistanceMatrices` — per-plane 256×256 palette distance tables (used by `cache/convergence.rs::PlaneDistance`)
- `cam_pq` codec (M-byte product quantization codes)
- `CLAM tree` (compressed-locally-approximate hierarchical tree for ANN)
- `bf16` SIMD primitives
- `simd_caps()` runtime detection

Each could (and increasingly DOES) expose its operations as struct methods:

```rust
// ndarray methods on Fingerprint<N>
impl<const N: usize> Fingerprint<N> {
    pub fn hamming(&self, other: &Self) -> u32 { ... }
    pub fn bipolar_project(&self) -> [i8; N] { ... }
    pub fn role_slot(&self, role: RoleId) -> &[u64] { ... }
    pub fn xor_bundle(&self, other: &Self) -> Self { ... }
    pub fn cosine(&self, other: &Self) -> f32 { ... }
}

// methods on SpoDistanceMatrices
impl SpoDistanceMatrices {
    pub fn spo_distance(&self, ...) -> u32 { ... }
    pub fn subject_distance(&self, ...) -> u16 { ... }
    pub fn par_pairwise_distance(&self, batch: &[(u8, u8)]) -> Vec<u32> { ... }
}

// methods on CamPqCorpus
impl CamPqCorpus {
    pub fn query(&self, q: &Fingerprint<N>) -> Vec<(usize, f32)> { ... }
    pub fn par_batch_query(&self, queries: &[Fingerprint<N>]) -> Vec<Vec<(usize, f32)>> { ... }
}
```

The pattern: every codec, every index, every persistent store gets a small set of methods on its carrier struct, with `par_*` variants for the rayon-parallel operations.

---

## 6. Splat-Driven Cognitive Cycle (Vision)

Putting it together — the future cognitive cycle under reunification + splat + struct methods:

```rust
// Initialize a Think carrier with all its state
let mut think = Think::new(
    distance_table,
    arigraph,
    palette_tables,
    cam_pq_corpus,
    splat_field,
);

// Run one cycle
let cycle = think.cycle(input_text)
    .splat_gaussian(SplatConfig::default())   // deposit pressure over SPOW board
    .score_hole_closure()                      // how much did surrounding holes close?
    .score_entropy()                            // how much entropy dropped?
    .replay_coherence(&think.episodic)         // does this align with replay-certified history?
    .emit_if_epiphany(EpiphanyThreshold::standard());

// Commit (if cycle resolved to a thought)
if let Some(result) = cycle.resolution() {
    think.commit(result);   // arigraph += new triplets; episodic.append; awareness.revise
}
```

The whole cycle reads like the system is **doing something it knows how to do**, not like a sequence of foreign function calls. That's the computational-entropy win — and it composes with the reunification of thinking-engine + cognitive-shader-driver-SoA into one `Think` carrier.

---

## 7. Recommended Path Forward (sprint-12+)

| Sprint | Target |
|---|---|
| **Sprint-11** | CausalEdge64 v2 lands (per sprint-10 plan); transcoder spec from `cognitive-shader-driver-thinking-engine-reunification.md` §5.2 |
| **Sprint-12** | `Think` carrier struct prototyped; minimum viable struct method set (cycle / commit / resolve) on top of existing thinking-engine + SoA |
| **Sprint-13** | Splat op fleet (`splat_gaussian`, `score_*`, `emit_if_epiphany`) lands as methods on `Think`; ontology-aware via `OntologyFilter` from `ogit-owl-dolce-ontology-compartments.md` §6.3 |
| **Sprint-14** | rayon work-stealing wired through par_* variants of the heavy methods; computational-entropy audit identifies remaining free-function holdouts |
| **Sprint-15+** | Replay/episodic/qualia integration as methods; full `Think` per CLAUDE.md "Thinking is a struct" doctrine |

This is a 5-sprint arc to reach the doctrine-grade architecture. Each sprint preserves backward compatibility (free functions remain as thin wrappers calling the methods) while the canonical surface migrates inward.

---

## 8. Cross-references

- `cognitive-shader-driver-thinking-engine-reunification.md` — the reunification path
- `causal-edge-64-spo-variant.md`, `causal-edge-64-thinking-engine-variant.md`, `causal-edge-64-synergies-and-pr-trajectory.md` — the dual-CausalEdge64 analysis
- `spo-schema-and-mailbox-sidecar.md`, `spo-ontology-format-stack.md`, `ogit-owl-dolce-ontology-compartments.md` — the schema/format/ontology context
- `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md` — splat shader vision (source for §1)
- `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §9 — splat as BLASGraph expansion
- `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md` — splat OSINT use case
- `.claude/plans/thought-cycle-soa-awareness-integration-v1.md` — the related awareness-cycle plan
- CLAUDE.md "Thinking is a struct" doctrine — the foundational litmus test
- CLAUDE.md "The Click" — Markov/role-keys/meaning grounding
- `lab-vs-canonical-surface.md` — UnifiedStep canonical surface (don't add lab endpoints when the bridge is right there)

---

*Authored 2026-05-14. CONJECTURE throughout; sprint-11+ ratification required. Splat references derived from already-shipped plan documents.*
