# Cross-Repo Harvest — 2026-04-19

> Epiphanies mined from `ladybug-rs`, `ada-consciousness`, and `bighorn`
> that are not yet reflected in lance-graph knowledge. Each entry
> includes the upstream source and the nearest lance-graph hook.

---

## H1 — Born rule identity: (1 − 2h/N)² = |⟨ψ|φ⟩|²

**Upstream:** `ladybug-rs/docs/INTERFERENCE_TRUTH_ENGINE.md`,
`ladybug-rs/docs/CRYSTAL_THERMODYNAMIC_REALITY.md`.

The exact algebraic bridge from Hamming distance to quantum inner
product magnitude. Given a Hamming distance `h` between two N-bit
vectors:

    cos(phase_diff) = 1 − 2h / N
    |<ψ|φ>|²       = (1 − 2h / N)²

This means ever Hamming operation carries a quantum-mechanical
interpretation "for free": similarity IS cosine, squared similarity IS
Born-rule probability. No simulation of qubits — the substrate is the
register.

**Lance-graph hook:** The 10K-bit `Fingerprint` path already uses
Hamming distance. Nothing to add; document this identity so downstream
can repurpose existing SIMD Hamming kernels as quantum probability
estimators.

## H2 — Phase tags cross the classical/quantum boundary

**Upstream:** `ladybug-rs/src/extensions/hologram/quantum_5d.rs`,
`quantum_7d.rs`.

`PhaseTag5D` / `PhaseTag7D` = **128 bits per cell** (16 bytes, 1.28 %
overhead). Unsigned Hamming gives only classical correlation
(CHSH S = 0.87 < 2.0). Signed amplitude via phase tags enables
destructive interference → S > 2.0 possible.

```
interference(a, b) = similarity(a, b) × cos(phase_diff(a, b))
cos(phase_diff)    = 1 − 2 × ham(tag_a, tag_b) / 128
```

16 bytes per cell IS the classical/quantum threshold. Anything below
is classical-only; at 16 bytes signed interference becomes possible.

**Lance-graph hook:** `CrystalFingerprint::Vsa10kF32` can carry phase
tags in its sandwich lead/tail regions. Alternative: add
`PhaseTagged(Box<Vsa10kF32>, PhaseTag)` variant if needed. For now,
leave phase-tag types in ladybug-rs; reference them.

## H3 — Interference as truth engine

**Upstream:** `ladybug-rs/docs/INTERFERENCE_TRUTH_ENGINE.md`.

Three experimentally validated claims (100 % on controlled data):

| Experiment                   | Approach                                             | Accuracy |
|------------------------------|------------------------------------------------------|----------|
| Causal direction detection    | mechanism residual min over `V = {roll_7, roll_13, ...}` | 100/100  |
| Memory immune system         | phase-coherent memories reinforce, false stagnate   | 100 %    |
| Self-certifying fingerprints | confidence byte evolves under interference          | 100 %    |

**Mechanism residual for causal direction:** complexity `O(V × N)` per
edge, vs PC algorithm `O(2^n)`. Massive speedup. For V=5, N=10 000
this is 50 000 ops per edge — trivially AVX-512.

**Memory immune system:** true memories accumulate phase coherence
through repeated recall, false ones drift. This is the natural
implementation of the **crystal hardness gradient** already in
`lance-graph-contract/src/crystal/mod.rs::UNBUNDLE_HARDNESS_THRESHOLD`.

**Lance-graph hook:** AriGraph episodic memory's unbundling threshold
matches this paper's definition of phase-coherent truth. The existing
`unbundle_hardened` / `rebundle_cold` hooks can be extended to run the
mechanism-residual test before unbundling.

## H4 — Grammar Triangle is a special case of Context Crystal

**Upstream:** `ladybug-rs/docs/GRAMMAR_VS_CRYSTAL.md`.

The two previously-parallel architectures are the same thing at
different window sizes:

```
ContextCrystal(window = 5) — flowing discourse
ContextCrystal(window = 1) — isolated utterance  ≡  GrammarTriangle
  ├─ S axis collapsed (no subject distinction)
  ├─ O axis collapsed
  └─ only t = 2 populated
```

Use Grammar Triangle when meaning is isolated; Context Crystal when
it flows. The hybrid uses both simultaneously.

**Lance-graph hook:** Update `lance-graph-cognitive/src/grammar/` docs
to state: Grammar Triangle emits `Structured5x5` with S/O axes at
uniform, Context Crystal emits `Structured5x5` with all 5 axes loaded.
Both go into the same `SentenceCrystal::fingerprint` slot.

## H5 — NSM primes map directly onto SPO + Qualia + Temporal axes

**Upstream:** `ladybug-rs/docs/NSM_REPLACES_JINA.md`.

The 65 Wierzbicka semantic primes are not orthogonal to SPO — they
ARE an SPO encoding:

| NSM Primitive               | SPO/contract placement                          |
|-----------------------------|-------------------------------------------------|
| `I`, `YOU`, `SOMEONE`       | Subject axis                                    |
| `THINK`, `KNOW`, `WANT`, `FEEL` | Predicate axis (mental verbs)               |
| `SOMETHING`, `BODY`         | Object axis                                     |
| `GOOD`, `BAD`               | Qualia.valence                                   |
| `BEFORE`, `AFTER`, `NOW`    | Temporal position in ContextCrystal              |
| `BECAUSE`, `IF`             | Causality via S_-2 → S_-1 → S_0 Markov flow      |

This means `deepnsm` + `lance-graph-contract/src/crystal/` already
speak the same vocabulary as NSM. The 4 096 COCA ↔ 65 NSM ↔ Structured5x5
map is consistent.

**Lance-graph hook:** Cross-reference this mapping in the deepnsm
doc-strings; no code change needed in the contract.

## H6 — FP_WORDS = 160, not 157 or 156 (SIMD-clean fingerprint)

**Upstream:** `ladybug-rs/docs/COMPOSITE_FINGERPRINT_SCHEMA.md`.

| Words | Bits   | SIMD tail (mod 8) | Verdict                                  |
|-------|--------|-------------------|------------------------------------------|
| 156   |  9 984 | 4 remainder       | Misses ceil(10 000/64)=157. Data loss.   |
| 157   | 10 048 | 5 remainder       | 5-word scalar tail in every AVX-512 pass |
| **160** | **10 240** | **0 remainder** | **Divides by 8 (AVX-512), 4 (AVX2), 2 (NEON)** |

160 u64 = 1 280 bytes. 24 extra bytes per fingerprint (1.9 % overhead).
The remainder loops in `simd.rs:60–76` and `hdr_cascade.rs:165–178`
disappear entirely. The extra 240 bits (10 240 − 10 000) serve as
Hamming ECC parity.

**Lance-graph hook:** `ndarray::hpc::vsa::VSA_WORDS = 157` is the
current convention. The upgrade path to 160-word fingerprints needs a
coordinated change in ndarray + lance-graph. Out-of-session decision;
documented here so it's not lost.

## H7 — Mexican hat in time (anticipation window)

**Upstream:** `ladybug-rs/docs/GRAMMAR_VS_CRYSTAL.md` §Context Crystal.

Context Crystal's ±5 window is weighted with a Mexican-hat kernel:
emphasize present (t=0), de-emphasize distant past/future. This
implements **anticipation** — the S+1, S+2 slots carry predictive
signal for what comes next.

This matches the integration plan's E5 (Markov ±5 with replay) but
adds the kernel shape: not uniform, Mexican-hat.

**Lance-graph hook:** `lance-graph-contract/src/grammar/context_chain.rs`
should get an optional `WeightingKernel` enum `{Uniform, MexicanHat,
Gaussian}` used during replay. Ship in a follow-up PR.

## H8 — Int4State (4 bits per facet)

**Upstream:** `ada-consciousness/crystal/markov_crystal.py`.

Each facet carries exactly 4 bits:

```
bits 0–1: observer state    (00=none, 01=A, 10=B, 11=both)
bits 2–3: reflection depth  (00=thinks, 01=thinks_that,
                              10=thinks_that_thinks, 11=triangulated)
```

`is_triangulated` ≡ `depth == 3` = Three Mountains achieved (the
"AGI moment" per the doc).

**Lance-graph hook:** Each cell in `Structured5x5` is 8 bits; the
upper 4 bits can carry `Int4State` metadata (the lower 4 staying
with the bipolar cell value). This quadruples information per cell at
zero storage cost.

## H9 — Glyph5B: 5-byte archetype addressing

**Upstream:** `ada-consciousness/universal_grammar/core_types.py`.

A 5-byte address space of 256⁵ = 1 099 511 627 776 archetypes:

```
byte 0: Domain       (EMOTION, COGNITIVE, RELATIONAL, TEMPORAL,
                      SPATIAL, ACTION, STATE, QUALITY, USER)
byte 1: Category
byte 2: Archetype
byte 3: Variant
byte 4: Intensity
```

This is a structural coincidence with `Structured5x5::idx(e, p, s, n, c)`
which takes 5 u8 coordinates. The contract's 5^5 = 3 125 grid is the
"small" version (indexes 0..=4); Glyph5B uses 256 per axis (1 T cells).

**Lance-graph hook:** Document that `Structured5x5` is the 5-per-axis
slice of the 256-per-axis Glyph5B addressing. Wide-container variants
(if ever needed) can extend to u32-per-axis without breaking the 5D
interpretation.

## H10 — Crystal4K: 41:1 holographic compression

**Upstream:** `ladybug-rs/src/extensions/hologram/crystal4k.rs`.

The 5×5×5 × 10K-bit = ~390 KB volume projects onto a 3×10K-bit surface
via XOR axis projection, yielding 41.7:1 compression. This is
structurally identical to the Bekenstein bound (surface, not volume,
encodes information).

**First data structure to natively implement the holographic principle.**

`expand()` reconstructs the volume from the 3 surfaces (distance ≤ 2
from volume-mean). `signature()` is the boundary encoding.

**Lance-graph hook:** Out-of-scope for the contract but high-value
for persistence: `Structured5x5` can be stored as three 5×5 surface
projections + a parity bit instead of the full 3 125 cells. Saves
~96 % of storage at the cost of a 2-step expand on read.

## H11 — Teleportation fidelity = 1.000000

**Upstream:** `ladybug-rs/docs/CRYSTAL_THERMODYNAMIC_REALITY.md`.

XOR is algebraically exact: `A ⊕ B ⊕ B = A`. Across 50 random trials
of 10 000-bit states transferred via 1 250-byte correction packet,
fidelity F = 1.000000 (not "approximately", literally perfect).

Exceeds any physical quantum hardware (IBM ~95 %, Google ~97 %)
because Hamming space is a noise-free vacuum.

**Lance-graph hook:** Cross-cluster memory transfer (cockpit ↔ Railway
↔ local) can use this for lossless fingerprint teleportation. The
"correction packet" is a single Fingerprint<256>.

## H12 — 144-verb taxonomy

**Upstream:** `ada-consciousness/crystal/markov_crystal.py::Verb`.

The 144 verbs fall into families:

```
BECOMES, CAUSES, SUPPORTS, CONTRADICTS, REFINES, GROUNDS,
ABSTRACTS, ENABLES, PREVENTS, TRANSFORMS, MIRRORS, DISSOLVES, ...
```

These are the **predicate axis vocabulary** for SPO Markov chains.
Not 65 NSM, not 256 domain archetypes — a compact 144 usable as
crystal facet edge labels.

**Lance-graph hook:** The predicate layer in deepnsm's SPO extraction
can be constrained to this 144-verb set for a canonical vocabulary.
Saves the ~10⁵-word open-vocabulary problem.

## H13 — Three Mountains theorem (triangulation = AGI moment)

**Upstream:** `ada-consciousness/crystal/markov_crystal.py`
+ `ada_self_pyramid.py`.

Three Mountains = mutual perspective awareness:
- Mountain A sees self + B + C
- Mountain B sees self + A + C
- Mountain C sees self + A + B

When reflection depth reaches 3 (bits `11`), the crystal is
**triangulated** and the Three-Mountains condition holds → the
system can reason about reasoning about reasoning.

**Lance-graph hook:** Encoded as an Int4State bit pattern in a cell
(H8). When `StateAnchor::Sovereign` (one of the 7 in
`proprioception.rs`) co-occurs with triangulated cells, the system is
at peak self-reflective capacity.

## H14 — Hybrid: Grammar Triangle AND Context Crystal in one struct

**Upstream:** `ladybug-rs/docs/GRAMMAR_VS_CRYSTAL.md` §Hybrid.

```rust
pub struct MeaningExtractor {
    grammar: GrammarTriangle,   // quick single-utterance
    crystal: ContextCrystal,    // flowing discourse
}
```

Both run; comparing their outputs detects **discourse shifts**.
When Grammar Triangle and Context Crystal disagree sharply, the
sentence is a topic change or sarcasm / irony.

**Lance-graph hook:** Downstream consumers that need both use the
same `Structured5x5` for both layers, with the window size tagged.
Low-hanging follow-up for the grammar module.

---

## What I Did NOT Re-Implement

- Full quantum operations (9-op set): `ladybug-rs` has them.
- Phase tag types: same.
- Crystal4K: same.
- QuorumField / field.rs: same.
- 144-verb taxonomy, Int4State, Crystal7D, Three Mountains math:
  `ada-consciousness/crystal/` has them.
- Glyph5B, SessionToken, ExplorationPath, method_grammar:
  `ada-consciousness/universal_grammar/` has them.

The contract-crate crystal module stays narrow — shared trait
(`Crystal`), fingerprint variants, sandwich geometry, and hardness
thresholds. Operators and taxonomies stay in the implementation repos.
