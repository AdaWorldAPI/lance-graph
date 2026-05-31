<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright The Lance Authors
-->

# INTEGRATION MAP: the delta-card world-spine — one idea, key and value

## READ BY:
- Anyone implementing the addressing / hydration / delta-card layer of the Wikidata-HHTL spine
- Anyone touching the frozen ontology radix, the Lance fragment GOP, the RISC compose-cache, or the OGIT/DOLCE class deck
- `truth-architect`, `integration-lead`, `palette-engineer`

> **Status: CONVERGED VISION (living), built bottom-up over an 8-turn design
> session. The primitives it composes are SHIPPED (NiblePath, FieldMask,
> ClassView, CausalEdge64+WitnessTable, ComposeTable, CLAM, Lance fragments);
> the consolidation + the delta-card value model are the NEW synthesis. Every
> load-bearing claim is labelled and carries a probe.** Companion:
> `agnostic-lazy-world-spine.md` (the tiered-substrate framing this refines).

---

## The one idea (read this, the rest is derivation)

**A card stores the *surprise*; the deck stores the *expectation*. Meaning =
deck ⊗ delta.** Everything — a recipe, a Wikidata entity, an address, a
sentence-mailbox — is a **small delta over an inherited frozen archetype**,
reconstructed on demand. The deck (class / region / ontology path) is frozen and
shared; the card is a few bits of deviation. This is literally the free-energy
framing (`CLAUDE.md`: `F = (1−likelihood) + kl`): **the archetype is the prior,
the delta is the prediction error, and the bit-width IS the residual surprise.**

It applies to **both halves of a row**: the **key** (address) and the **value**
(content) compress by the *same* delta-over-archetype move.

---

## The on-ramp: a cookbook (the value side)

A recipe card carries only its deltas from an inherited template:

```text
inherited (ZERO bits in the card — it is the deck / the path):
  region  → available ingredients, fat medium, staple   (Italian → olive oil, pasta)
  season  → what is fresh                                (autumn → squash, mushroom)
  persona → diet, heat tolerance, skill                  (vegan, mild)

the card itself (the deltas — the only bits it stores):
  texture  2b (crisp/soft/chewy/creamy)   sweet 2b   sour 2b (none/lemon/vinegar/ferment)
  salty    2b                             veg-axis 2b (mixed/salad/mushroom/Asian)
  ──────────────────────────────────────────────────────────────────────────
  ~10 free bits → a 16-bit card
```

`recipe = (inherited class path) + (8–16 delta bits)`. The 16-bit card is
meaningless alone (*"medium-sour, crisp, mushroom"*) until resolved against
`Italian × autumn × vegan` → reconstructs the full dish. **The box holds the
schema; the card holds 16 bits of flavor-coordinate.**

**Honest boundary (where 16 bits stops being truthful):** the delta carries the
*compressible profile* (dish type / flavor) because region×season×persona already
constrains it. It does NOT carry irreducible specifics — exact quantities, a
novel signature step, or a fusion dish outside any cohort. Those are *new
information* → a wider delta or a fork, never a 2-bit axis. This is the
**generator-vs-derivable split**: profile derives from the template; specifics
are stored values.

---

## The unification: key and value are the same trick

We spent the design compressing the **address (key)**; the cookbook proves the
**content (value)** compresses identically — same delta-over-archetype, same
I/P/B-frame model:

| | KEY side (address) | VALUE side (cookbook / entity content) |
|---|---|---|
| keyframe (I) | frozen ontology radix trie | the archetype (region×season×persona template) |
| delta (P) | appended entity offset | the 8–16-bit flavor/property delta card |
| reconstruct | path → entity identity | template ⊗ delta → full content |
| floor | 27 bits (entropy of 113M) | residual surprise given the deck |

So a row is `[ key-delta-over-frozen-path | value-delta-over-archetype ]` — tiny
both ways, reconstructed against frozen decks held once in OGIT.

---

## The addressing chain (the key side, end-to-end)

Derived across the session; each step grounded in a shipped primitive.

### 1. Partition-as-address, schema-as-deck (the Quartettkarten move)
The address is **location, not a stored column.** A card doesn't carry
"category=Auto"; it's *in the Auto box*. Shard the spine into a 256-ary tree by
nibble-pairs (the OWL/DOLCE `subClassOf` path); *which leaf a row lives in*
encodes the upper bits — stored **once in the directory + OGIT lookup**, never
per-row. Schema (fields/labels/DOLCE) lives in the deck (`ClassView`/`FieldMask`,
resolve-not-store, #441). The card is **pure values + presence mask: zero address
bits, zero schema bits, zero label bits.**

### 2. The 27-bit truthful floor, with a ~0-bit row
113M entities → ⌈log₂⌉ = **27 bits** of irreducible identity entropy (Wikidata
QIDs already run to ~Q130M ≈ 2²⁷ — the QID is a near-optimal flat address;
classes CANNOT make *identity* cheaper). The win is that partition-as-address
makes the 27 bits **free per-row** — `address = (path << offset_bits) |
row_index`, the path in the directory, the offset implicit in file position:
```text
  /0xA7/0x3C/leaf.lance   ← 16 path bits (4 nibbles), held by the directory
    row 0..1724           ← 11 offset bits, implicit (position)
  = 27-bit address, ~0 address bits stored in the row
```

### 3. Sparse radix range-delegation (don't build 256⁴ files)
256⁴ = 4.3B virtual addresses; 113M occupied = **2.6% full**. Never materialize
the empty 97%. The "range register" is a **path-compressed radix/Patricia trie**:
`entry = nibble-range → {Empty | Leaf(file) | Delegate(sub-table)}`. A sparse
DOLCE branch = one `Leaf`; a dense branch (40M scholarly articles) = `Delegate` →
sub-table → many leaves; single-child chains collapse. The register = the
**occupied branch points** (≈ the OWL/DOLCE class count, KB–MB), not 4.3B files.
Skew is absorbed by 38× headroom: cohort ≠ class (giant class → many cohorts,
tiny class → one sparse cohort; sparse cohorts cost nothing — address space is
free, only *resident* memory costs).

### 4. The frozen ISA — no rebalance
The upper ontology (DOLCE/FIBO/GoBD/OGIT + the nibble→class lookup) is a
**compiled constant** — standardized precisely so it can be frozen; zero runtime
churn. Leaves are **append-only** (new entity → new offset; append ≠ move). So
`address = [frozen-path | append-only-offset]` is stable on both halves — a
**compiled perfect hash, not a runtime hash table** → the rebalancer is *deleted*,
not built. A schema bump (DOLCE v1→v2) is a **version-gated, one-time, amortized**
global upgrade carrying an ontology-version byte (the existing
`I-LEGACY-API-FEATURE-GATED` iron rule). The only residual "move" is an
individual *reclassification* — a one-row data correction via the QID↔address
map, a rounding error on 113M.

---

## The frame model (x264/265 — the capstone)

The cold floor IS a keyframe/delta store — and that is **Lance's native
fragment-versioning**, not new machinery:

| video | spine |
|---|---|
| **I-frame** | frozen radix trie + compacted Lance base fragment (self-decodable, exact, rare) |
| **P-frame** | appended entities + CLAM-clustered new arrivals + corrections (cheap, references the keyframe, useless alone) |
| **B-frame** | the RISC compose-cache — multi-hop derived paths, references multiple bases, evictable |
| **GOP** | keyframe + accumulated deltas, periodically re-baselined by **compaction** |

This **resolves the frozen-vs-adaptive tension**: CLAM is adaptive *inside the
delta* (it clusters new arrivals, *proposes* placement as a P-frame); the
keyframe never moves; **compaction = re-emit a fresh keyframe = the amortized
schema upgrade**, the one deliberate version-gated moment where validated
similarity FREEZES into structure. Tradeoff = **read amplification** (resolve =
keyframe + N deltas overlay, the LSM/video-seek cost), bounded by GOP length
(compaction frequency) — a dial, not a flaw. Deltas are *exact* (a P-frame is
lossless); CLAM similarity *decides* the delta, is never stored *as* the address.

---

## RISC: compose, don't materialize (the edge side)

Storing "every human related to every other" = 113M² ≈ 10¹⁶ edges (catastrophe).
RISC move: **store the generators, compute the closure.** Store parent/child/
spouse edges (~N); derive "related to Y in ≤7 hops" on demand via
`bgz-tensor::ComposeTable` (each hop = a u8 table lookup) / blasgraph `mxm`
matrix-power. Six-degrees ⇒ the closure is ≤7 cached hops, not a stored edge and
not a walk.

**This dissolves the hub problem:** *United States*, *human*, *Earth* never store
their millions of inbound back-edges — they're *reached* by composing forward
generators. Hubs were only a problem if you imagined materializing them.

- **generators = `continuant` = permanent/cold** (the DOLCE 1-bit);
- **composed multi-hop paths = `occurrent` = temporary/evictable KV** (the
  B-frame compose-cache). **One eviction policy, derived from the ontology.**
- New design surface: a **per-predicate composability flag** (~12k predicates) —
  "generator (store)" vs "derivable (compose)". Non-composable facts
  (`birth_date`, `population`) are irreducible values, always stored.

---

## The scale identities (why the numbers all rhyme)

Everything lands on the same powers of two:

```text
  6-bit  cohort      = 64           the immediate WitnessTable cohort        [in code]
  16-bit book        = 65,536 SPO   one book/corpus (Bible ~32k = half;       [proposed]
                                     novel ~4-5k ≈ ~4096 SPO mailboxes)
  18-bit hot envelope= 262,144      the CONCURRENT mailbox working set:       [in code: 64K–256K]
                                     both books resident + a Wikidata window
  32-bit world       = 4.3 B        the COLD spine (Wikidata ~115M, lazy)     [in code: mailbox_ref]
```
- **Reasoning = traversing the `CausalEdge64` W-slot → `WitnessTable`/
  `EpisodicWitness64` arc + SPO** — NOT bundling the 16384 VSA fingerprint
  (retired legacy; survives only as the discovery-layer similarity carrier).
- **Reading a text = accumulating SPO mailboxes + their CE64/EW64 arc** (no
  embedding, no forward pass); ambiguity resolved by counterfactual testing
  (`recipe_kernels`: `world ⊗ factual ⊗ counterfactual`, divergence = popcount,
  scenario-only channel).
- **Address vs hot set:** Wikidata is 32-bit-*addressed* (cold), never resident;
  256K is the *concurrent* envelope. You foveate the spine, so 256K holds whole
  corpora + a hydrated Wikidata slice at once — cross-corpus grounded reasoning
  ("Frodo ↔ biblical archetype, grounded in Wikidata") fits in one hot context
  *because* the spine stays cold. **Bounded hot context, unbounded cold spine.**
- The card (8–16 bit delta), the row (~4096 bit), the offset (11 bit), the book
  (16 bit), the address (27–32 bit) are all the same shape: **small delta over a
  frozen inherited archetype.**

---

## Two trees — never confuse them (the iron-rule guard)

| | **frozen ontology radix** (addressing) | **CLAM/CHESS manifold tree** (discovery) |
|---|---|---|
| fan-out | fixed 256/nibble, compiled ISA | adaptive — radii fit to data density |
| shape | frozen (DOLCE/FIBO) | data-derived, shifts as entities arrive |
| role | **the address** (exact, CAM) | **proposes/validates** the partition (offline) + the delta placement |
| rule | addressing = exact | similarity = discovery-only (faiss-homology iron rule) |

CLAM's adaptive radii are *similarity* — brilliant for **deciding** the partition
offline, but must NEVER *be* the runtime address (that would reintroduce the
rebalancing we deleted). **Adaptive proposes (in the delta); frozen ships (the
keyframe).** `aerial`/splat are the same discovery layer; `palette256`/CAM-PQ is
the leaf code (the card's compressed value row).

---

## What's built vs new vs conjecture

- **SHIPPED primitives:** `NiblePath` (#442), `FieldMask`/`ClassView` (#441),
  `CausalEdge64` + `WitnessTable`/`EpisodicWitness64`, `ComposeTable` + blasgraph
  `mxm`, `CLAM` tree, Lance fragment-versioning, `aerial` proposer (#438/#443),
  OGIT/DOLCE cache + DOLCE-from-cache.
- **NEW (the synthesis / design surface):** the sparse radix range-delegation
  register; the delta-card value model; the per-predicate composability flag +
  RISC compose-cache; the `NiblePath`-keyed tiered hydration manager (the one
  missing runtime piece); the I/P/B-frame mapping onto Lance fragments.
- **CONJECTURE (each with a probe, below).**
- **Gate:** D-ARM-7 (Jirak floor, `jc::jirak`) before any hydrated rule writes a
  live store.

## Probes (the falsifiers — measure before freezing)

1. **Partition locality** — `jc/examples/splat_louvain_modularity.rs` (Louvain
   modularity) + CLAM on the real P279+edge graph (e.g. the biology subtree).
   Pass = high modularity ⇒ ~90% local edges ⇒ 16-bit references + the family
   frontier are real. Also yields the natural fan-out (sizes the 4/12/16 split)
   and which hubs to compose-not-store. `clam.rs` itself says CLAM-radii-coincide-
   with-ontology-boundaries is "a TEST, not a fact."
2. **Delta-card truthfulness** — reconstruct content from N delta bits vs ground
   truth; histogram the residual per cohort. Low residual ⇒ the cohort is real &
   8–16 bits suffice; high residual ⇒ wrong cohort or genuinely novel (needs a
   wider delta / fork). This is the entropy the card actually needs.
3. **Compose vs materialize** — measure the ≤7-hop reachability hit-rate +
   compose-cache eviction churn against a stored-edge baseline; confirms the
   N²-avoidance holds and sets the GOP/compaction cadence.

## Cross-references
`agnostic-lazy-world-spine.md` (tiered substrate), `wikidata-hhtl-load.md`
(120→38GB structural compression), `owl-dolce-hhtl-compartments-aerial-fed.md`
(domain compartments), `splat-codebook-aerial-wikidata-compression.md`
(splat→aerial seam). Primitives: `contract::{hhtl::NiblePath, class_view,
witness_table, splat}`, `causal-edge::CausalEdge64`, `bgz-tensor::{attention::
ComposeTable, hhtl_cache::RouteAction}`, `lance-graph::graph::neighborhood::clam`,
`crates/jc` (Louvain example, Jirak floor). Iron rules: `I-VSA-IDENTITIES`,
`I-NOISE-FLOOR-JIRAK`, `I-LEGACY-API-FEATURE-GATED`; `cognitive-risc-classes.md`
N4. `CLAUDE.md` The Click (free-energy = prior + prediction-error).
