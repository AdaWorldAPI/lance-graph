# Crystal ↔ Quantum — The Two Blueprints

> **Not contract surface. Two complementary implementation blueprints**
> that already exist in the sibling repos. This doc maps them so the
> lance-graph side doesn't reinvent either.

---

## The Split

| Mode       | Where it lives                             | What it encodes                                             |
|------------|--------------------------------------------|-------------------------------------------------------------|
| Crystal    | ada-consciousness/crystal/ (Python)        | Grammar as a **bundled Markov chain of SPO sentences**      |
| Crystal    | ladybug-rs/src/extensions/hologram/        | 5×5×5 QuorumField + Crystal4K compression                  |
| Quantum    | ladybug-rs/src/extensions/hologram/        | **Holographic memory as embedding residual accumulation**   |
| Quantum    | ladybug-rs/docs/QUANTUM_*_ORCHESTRATOR.md  | 9-op set: CNOT, QFT, phase kickback, decoherence, etc.      |

Same 10K-bit substrate, two operating modes.

---

## Crystal mode — bundled Markov SPO chain

Grammar turned into structure.

```
sentence_i  = SPO triple (subject, verb, object)
verb ∈ {BECOMES, CAUSES, SUPPORTS, CONTRADICTS,
        REFINES, GROUNDS, ABSTRACTS, ENABLES,
        PREVENTS, TRANSFORMS, MIRRORS, DISSOLVES, ...}  (144 total)

sequence    = [s_0, s_1, ..., s_n]
chain       = Markov transitions (verb-labeled edges between SPO nodes)
crystal     = bundle(fingerprint(s_0), fingerprint(s_1), ..., fingerprint(s_n))
facet       = one frozen transition in the chain
```

**Existing types** (ada-consciousness/crystal/markov_crystal.py):
- `Verb` enum (7+ in file, 144 total in taxonomy)
- `Int4State` — 2-bit observer × 2-bit reflection depth → `is_triangulated` when depth == 3
- Crystal7D in `markov_7d.py`: (DN depth, Situation X, Y, Verb, Qualia, Temporal, Markov-P)

**Corresponds to** `lance-graph-contract/src/crystal/`:
- `SentenceCrystal` — one SPO sentence crystallized
- `ContextCrystal` — Markov ±5 window (the chain segment)
- `CrystalFingerprint::Structured5x5` — sandwich cells = facet grid
- Bipolar cells → negative cancellation = VSA consensus on the chain

---

## Quantum mode — holographic residual accumulation

Memory as superposition.

```
memory_i    = embedding + PhaseTag (128-bit)
field       = Σ_i memory_i   (residual accumulation, not replacement)
interference(a, b) = similarity(a, b) × cos(phase_diff(a, b))
cos(phase_diff) = 1.0 − 2.0 × hamming(tag_a, tag_b) / 128

recall(query) = retrieve field components where interference > threshold
```

**Existing types** (ladybug-rs/src/extensions/hologram/):
- `PhaseTag5D` / `PhaseTag7D` — 128-bit phase for signed interference
- `QuantumCell7D { amplitude: Fingerprint, phase: PhaseTag7D }`
- `QuorumField` 5×5×5 lattice of 10K-bit cells
- 9-op set in `quantum_crystal.rs`:
  1. Spatial entanglement (CNOT on lattice)
  2. Quantum Fourier Transform (along axis)
  3. Phase kickback (eigenvalue extraction)
  4. Coherence tracking & decoherence
  5. Surface code error correction
  6. Quantum walk
  7. Adiabatic evolution
  8. Density matrix (mixed-state cells)
  9. State teleportation

**Corresponds to** `lance-graph-contract/src/crystal/`:
- `CrystalFingerprint::Vsa10kF32` — continuous field form
- Each element in the f32 array = amplitude × cos(phase)
- `sandwich_lead` / `sandwich_tail` regions = role-bind for phase keys
- Residual accumulation = `vsa_bundle` without normalization

---

## Why Both Modes Live in the Same Enum

`CrystalFingerprint` is deliberately polymorphic so one crystal can
flip modes during its lifecycle:

```text
fresh sentence       → SentenceCrystal { fingerprint: Structured5x5(...), ... }
                       [Crystal mode: discrete facets, bipolar bundling]

matured into memory  → Vsa10kF32(...)
                       [Quantum mode: continuous field, phase-tagged]
                       residual accumulates across re-encounters
                       unbundle back to Structured5x5 when hardness > 0.8
```

The sandwich layout serves both:

- **Crystal mode**: middle 3,125 cells hold bipolar structure, sandwich
  wings hold cross-role bundles.
- **Quantum mode**: full 10K f32 is the holographic field, phase-tagged
  elsewhere, sandwich wings hold key vectors for bind/unbind.

---

## What NOT to Build in lance-graph

Already exists, pull in via cross-repo path dep or keep out of contract:

- PhaseTag 128-bit, cos-from-hamming → ladybug-rs `hologram/quantum_*.rs`
- QuorumField 5×5×5 → ladybug-rs `hologram/field.rs`
- Crystal4K compression (41:1 via XOR axis) → ladybug-rs `hologram/crystal4k.rs`
- 9-op quantum set → ladybug-rs `hologram/quantum_crystal.rs`
- VSA binary bind/bundle/permute → ndarray `hpc::vsa`
- 144-verb taxonomy, Int4State, Crystal7D → ada-consciousness `crystal/`
- Glyph5B 5-byte archetype addressing → ada-consciousness `universal_grammar/`

lance-graph's crystal contract holds only: shared trait (`Crystal`),
fingerprint variants, sandwich geometry, and hardness thresholds.
The operators live in the crates above.

---

## The Grammar Question — Which Mode for What

Rough policy:

- **New parsed sentences** → Crystal mode (Structured5x5). Markov chain
  bundling via VSA XOR; bipolar cell cancellation does disambiguation.
  Young crystals live here — cheap, structured, queryable.

- **Residual memory accumulation** → Quantum mode (Vsa10kF32 + phase).
  Every recall perturbs the field by a small residual; repeated
  recall hardens specific components (interference-reinforced). Phase
  tags carry temporal/causal distance.

- **Graduation** at hardness ≥ 0.8: unbundle from whichever mode into
  individually-addressable facts in AriGraph episodic memory (already
  wired in the episodic.rs PR).

- **Re-bundling** for cold facts: project back into Quantum mode
  (residual field), where their phase tags mark them as dormant.

---

## Honest State

lance-graph-contract has the **shared surface** and geometry. It does
NOT have operators. All the interesting algebra lives in
`ndarray::hpc::vsa` (binary) or `ladybug-rs::extensions::hologram`
(continuous + phased). The contract types are just the vocabulary
those modules can agree on.

If a consumer wants Crystal mode: use Structured5x5 + `vsa::bundle`.
If a consumer wants Quantum mode: use Vsa10kF32 + phase tags from
ladybug-rs. Contract stays agnostic.
