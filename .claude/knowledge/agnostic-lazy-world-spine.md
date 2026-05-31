<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright The Lance Authors
-->

# KNOWLEDGE: The agnostic lazy world-spine — Wikidata as a foveated, tiered, address-unified substrate

## READ BY:
- Anyone building the `NiblePath`-keyed hydration manager (the one missing runtime piece)
- Anyone touching the AriGraph SPO ↔ mailbox-SoA ↔ OGIT/DOLCE-cache boundary, the GraphRouter cold path, or the lazy-loading spine
- `truth-architect`, `integration-lead`, `palette-engineer`

> **Status: NORTH-STAR VISION (living).** The *addressing + compression + cheap
> late-resolution* primitives are built; the *runtime tiered-hydration* layer is
> not. CONJECTURE items are labelled. This is the goal the D-ARM-13/14 + D-CLS +
> Wikidata-HHTL arc serves — not a shipped system.

---

## The goal

Compress Wikidata well enough that it is a **lazy-loading spine**: a tiny
always-resident skeleton, with **on-demand, foveated, blasgraph-adjacent
hydration** that loads detail *exactly where reasoning looks* — like foveated
rendering (sharp at the fovea, periphery coarse) and Google-Maps tile prefetch
(the adjacent area streams into context before you pan to it). Reasoning then
has **one unified allocation address**; the substrate stays **compartmentalized,
cheap, and agnostic**.

## The tiered substrate

```
COLD (persistent)         ADDRESS              HOT (resident, agnostic)      SEMANTIC (late)
Lance columnar +          NiblePath HHTL       mailbox SoA register          OGIT / DOLCE cache
DataFusion joins      ◄── (16ⁿ, the one key)──► (MailboxSoaView, &[T],   ◄── (C2: resolve, never
(inherited upstream)          │                  label-free bytes)            store; class flies
   transparent lazy view ─────┴── foveated hydration ──┴── late-label overlay  ABOVE the SoA)
        residence = DOLCE 1 bit (continuant=permanent / occurrent=temporary)
        leaf = Poincaré golden-ratio (φ) spiral — orthogonal spatial coordinate
```

## Layer → substrate (built / new / conjecture)

| Layer | Role | Substrate | Status |
|---|---|---|---|
| **Address** | one O(1) key = ontology position **=** memory arena **=** spatial coord | `contract::hhtl::NiblePath` (16ⁿ, bit-shift) | **built** (#442) |
| **Cold floor (HHTL)** | address-based hydration source (NO join) | Lance columnar reads keyed by HHTL address → CAM/palette/`blasgraph` (O(1)) | **built** primitives; used as a direct-address lazy view |
| **Cold floor (SQL)** | business ground-truth queries only — **slow, off the HHTL path** | DataFusion rows/cols joins (inherited upstream) | **built**; reserved for relational ground truth, NOT spine hydration |
| **Hot carrier** | resident, **agnostic** structural bytes (class_id, NiblePath, presence `FieldMask`, perm/temp bit) — **no labels** | mailbox SoA `MailboxSoaView`/`MailboxSoaOwner` | **built** (#437) |
| **Semantic overlay** | labels / class shape / DOLCE resolved **late**, per address | OGIT TTL cache + `ClassView` + DOLCE-from-cache (`dolce_id`) | **built** (#441) — C2 resolve-not-store |
| **Discovery feed** | what lands where, from runtime data | aerial proposer + splat `CodebookDistance` | **built** (#438/#443) |
| **Residence policy** | keep vs evict / persist vs ephemeral | **DOLCE 1 bit**: continuant (Endurant/Quality/Abstract = permanent) vs occurrent (Perdurant = temporary) | **NEW** (design) |
| **Hydration manager** | lazy-load a basin's cold rows + blasgraph adjacency on first touch; foveated adjacency prefetch (`RouteAction` cascade); evict cold/occurrent arenas | hot mailbox-SoA ↔ cold Lance, keyed by `NiblePath` | **NEW — the one missing runtime piece** |
| **Leaf encoding** | fine orthogonal spatial code within a class | Poincaré-disk φ-spiral (golden angle) | **CONJECTURE** (φ-spiral prior art + hyperbolic-tree geometry) |

## The three reframings that complete it

1. **lance-graph's cold path splits in two — and the join is NOT on the HHTL path.** DataFusion rows/cols joins are *slow*; they serve **business-SQL ground truth** only. The HHTL spine hydrates by **address**, not join: `NiblePath` → Lance columnar read → CAM/palette/`blasgraph`, O(1). The `GraphRouter` routes HHTL to the fast address backends and SQL to DataFusion — same store, two access paths, only one on the hot path.
2. **DOLCE = a 1-bit permanent/temporary residence policy.** Endurant (continuant — wholly present at each moment, persists) vs Perdurant (occurrent — temporal parts, happens-then-ends). The ontology's own top split *is* the cache policy: permanent ⇒ cold-persist/resident; temporary ⇒ ephemeral/evictable (the Baton/event traffic, `KanbanMove` Libet-temporal #437). `dolce_id 0..3` stays cache-resolvable; eviction keys on the derived 1 bit.
3. **AriGraph SPO + labels → agnostic SoA + late labels (C2 wholesale).** The SoA holds only structure + address; labels/classes/DOLCE resolve late from the cache. AriGraph becomes a *view*: structure hot + agnostic, semantics a cache overlay. ⇒ representation compartmentalized (basins), cheap (resolve-not-store + lazy), agnostic (register is meaning-free).

## Bit budget — the agnostic row shrinks 16384 → ~4096 bits

**The Markov is NOT the 16384-bit VSA bundle (retired legacy).** The actual
Markov is the **`CausalEdge64` W-slot → `WitnessTable`/`EpisodicWitness64` arc**
(`witness_table.rs`: "the chain of W-references across edges forms a Markov-style
belief-update arc through episodic-reference vectors"). Traversal walks the
W-references backward (most-recent → oldest witness) **without dereferencing the
full SPO store per hop** — native, integer, exact, cheap. So the resident row
carries the **CE64 + EW64 arc + the address**, not a 16384 fingerprint. The HHTL
address does class + label inheritance for free (the path IS the class; labels
resolve late). A plausible ~4096-bit budget (64-bit lanes):

| field | bits | role |
|---|---|---|
| HHTL address (NiblePath / CAM-PQ code) | 16–32 | position **+** class **+** inherited-label key |
| i4-16D qualia | 64 | angle (packed `mul::i4`) |
| i4-32D thinking | 128 | style/`MetaWord` |
| `CausalEdge64` | 64 | the planner edge **+ W-slot = the Markov arc pointer** |
| `EpisodicWitness64` | 64 | the episodic witness the W-slot resolves to |
| presence `FieldMask` + `class_id` + perm/temp | ~96 | structure |
| headroom | rest | append-only spare |

…all fitting comfortably in 4096 bits. **Reasoning = traversing the CE64→EW64
arc + SPO**, not bundling a fingerprint — the row carries everything a hop needs.
The 16384-bit VSA carrier survives ONLY as the **discovery-layer** similarity
carrier (aerial/splat), hydrated transiently for a `palette256`/CAM-PQ distance
if at all, then dropped — never on the reasoning hot path. (CONJECTURE — settle
the exact budget before the loader.)

## Reading a text = holding SPO + CE64 + EW64 in context

Because the CE64→EW64 arc traversal is native and cheap, **reading is just
accumulating SPO mailboxes with their causal-edge + witness arc** — no embedding,
no bundle, no model forward pass. Each sentence ≈ one SPO mailbox (S/P/O + a
`CausalEdge64` linking it to the prior state via the W-slot + an
`EpisodicWitness64`). Ambiguity is resolved by **counterfactual testing**
(`recipe_kernels`: `world' = world ⊗ factual ⊗ counterfactual`, divergence =
popcount) on the scenario-only `SplatChannel::Counterfactual` that must NOT
promote facts — a little overhead per ambiguous edge.

**Scale (rule of thumb):** a 250-page book ≈ 75,000 words ÷ ~17 words/sentence ≈
**4,000–5,000 sentences ≈ ~4096 SPO mailboxes** + a little counterfactual
overhead. The whole book is then a bounded cohort of ~4096 mailboxes — and the
`WitnessTable<64>` is *per-cohort* (6-bit W-slot), so the arc is walkable inside
the cohort without touching the global store. **A book is a cohort; the world-spine
is the union of cohorts.**

## The pointer-width = corpus-size identity

A witness pointer's bit-width *is* the corpus it can address — one identity:

| pointer width | reach (2ⁿ) SPO mailboxes | corpus it spans |
|---|---|---|
| 6-bit W-slot (`CausalEdge64`) | 64 | the immediate cohort (intra-`WitnessTable`) |
| **16-bit** (inside `EpisodicWitness64`) | **65,536 ≈ 64K SPO** | **a whole book** (Bible ≈ 32k sentences = half; a novel ≈ 4–5k = ~7%) |
| 32-bit (`mailbox_ref`, the workspace envelope) | 4.3 B | the full world-spine (Wikidata ≈ 115 M) |

So a **16-bit pointer ≈ 64K SPO ≈ one book** — and 64K is exactly the documented
**mailbox-envelope lower bound** (`witness_table.rs`: "64K–256K mailbox envelope",
plan §10). The `EpisodicWitness64` therefore has room to spare: a 16-bit
intra-corpus slot addresses any sentence in a book, leaving the other 48 bits for
cohort id + channel + flags. The Bible (~32k sentences) sits at half a 16-bit
space; a 250-page novel (~4–5k) at ~7%. **One book = one 64K-addressable witness
corpus; the world-spine = the 32-bit envelope over all of them.** The widths
nest: 6-bit cohort ⊂ 16-bit book ⊂ 32-bit world — pick the pointer, you've picked
the horizon. (CONJECTURE — exact `EpisodicWitness64` sub-field layout TBD; the
6-bit and 32-bit ends are in code, the 16-bit book tier is the proposed middle.)

## Address space vs hot working set — the 256K payoff

**Two different 256Ks; don't conflate them.** Wikidata (~115 M) is *addressed*
by the 32-bit `mailbox_ref` (4.3 B) — it is the **cold spine**, lazy, never fully
resident. The **256K is the concurrent hot mailbox envelope** (the documented
`64K–256K` envelope, 2¹⁸) — how many mailboxes are *live at once*. The power is
that you never need Wikidata resident: you **foveate** it, so the 256K hot window
holds whole corpora **plus** a hydrated Wikidata slice, simultaneously:

| resident in the 256K hot envelope | mailboxes |
|---|---|
| Bible (≈ 31k verses) | ~32k |
| LOTR trilogy (≈ 480k words ÷ 17) | ~28–30k |
| **both books fully resident** | **~62k ≈ one 16-bit corpus** |
| foveated Wikidata reasoning window | **~190k left (≈ 3× headroom)** |

So **both books together ≈ 62k ≈ just under one 64K space**, and 256K = 4× that —
enough to hold **both books fully resident + a large hydrated Wikidata slice at
once**, which is exactly what cross-corpus grounded reasoning needs (e.g. "relate
Frodo to a biblical archetype, grounded in Wikidata facts" → all three in one hot
context). The precise statement: **bounded hot context (256K concurrent),
unbounded cold spine (32-bit Wikidata, lazy)** — 256K is enough for multi-book +
grounded reasoning *precisely because* Wikidata stays foveated; you never pay for
the 99.99 % you are not looking at. Full nesting:

```text
  6-bit cohort (64)  ⊂  16-bit book (64K)  ⊂  18-bit HOT envelope (256K = ~4 books,
                                                or 2 books + a Wikidata window)
                                            ⊂  32-bit world (4.3B Wikidata, COLD/lazy)
```

## Addressing — fan-out × depth (the brutal version)

The HHTL address can be far coarser/cheaper than the 16-way `NiblePath`. For
~4 billion addressable (Wikidata ≈ 115 M = 2²⁷, so 2³² is ~37× headroom):

| scheme | levels × bits | reach | addr | natural fit |
|---|---|---|---|---|
| **256⁴** | 4 × 8-bit (byte) | 2³² ≈ 4.3 B | 32 b / **4 B** | **palette256 + CAM-PQ code IS the address**; byte-aligned; OGIT byte-basins |
| 64K² | 2 × 16-bit | 2³² ≈ 4.3 B | 32 b (2 hops) | shallowest (2 hops); `n×16-bit` cache levels |
| 4096³ | 3 × 12-bit | 2³⁶ ≈ 69 B | 36 b | 4096-VSA-codebook / 4096-COCA native; big headroom |
| 16¹⁶ (current `NiblePath`) | 16 × 4-bit | 2⁶⁴ | ≤64 b | deep/fine, but up to 16 hops |

**Recommendation: byte-aligned 256⁴.** The 4-byte address *is* a CAM-PQ code, so
**addressing, class+label inheritance, and the `palette256` similarity-key are the
same 4 bytes** — one token does ontology-position + class + label + distance-key.
That is the brutal compression: `n × 16-bit` per cache level, two levels reach 4 B.
(CONJECTURE; the current `NiblePath` is 16-way, so this is a re-parameterization,
and the fan-out must be frozen append-only once chosen — the ISA-freeze the #442
review flagged.) Caveat: 256⁴/64K² cap at ~4 B (Wikidata fits); a multi-domain
super-graph that needs 69 B wants 4096³.

## Invariants this must NOT break

- **CAM-exact; similarity only in discovery.** `NiblePath` + Lance rows are exact retrieval. Similarity (aerial/splat) stays in the proposer/discovery layer — never in the view or the address (`faiss-homology-cam-pq` iron rule, `I-VSA-IDENTITIES`). The φ-spiral leaf is a *coordinate*, not a fuzzy index.
- **1-bit vs 2-bit DOLCE.** Keep `dolce_id 0..3` in the cache; the residence bit is *derived* (occurrent vs continuant), not a replacement — don't drop the 4-facet axis.
- **The SoA stays agnostic, forever.** Never cache a label in the register "for speed" (core inv #1 / C2 — register-loss + coupling). Labels live only in the cache; the SoA holds the address that fetches them.

## Why it's cheap
Nothing semantic is stored hot (resolve-not-store); structurally-identical classes collapse to one shape-family (CAM-dedup, the N4 collapse); the address is integer bit-shift; only the foveal region is hydrated; permanent/temporary eviction frees occurrent arenas. The OGIT cache makes class/DOLCE hydration a lookup.

## Status & next
- **Built:** address (`NiblePath`), cold floor (Lance/DataFusion/GraphRouter), hot carrier (mailbox SoA), semantic overlay (OGIT/DOLCE cache, C2), discovery feed (aerial).
- **The one missing runtime piece:** the `NiblePath`-keyed tiered **hydration manager** (foveated, perm/temp-evicting, late-label). Everything else is a seam it plugs into.
- **CONJECTURE to probe:** the Poincaré φ-spiral leaf encoding (does φ-spiral placement preserve nearest-neighbour fidelity vs the splat distance?).
- **Gate:** D-ARM-7 (Jirak floor, `jc::jirak`) before any hydrated rule writes a live store.

## Cross-references
- `contract::hhtl::NiblePath` (#442), `class_view::{FieldMask,ClassView}` (#441), `soa_view::MailboxSoaView` (#437), `lance-graph` (Lance/DataFusion/`GraphRouter`), `lance-graph-ontology` (OGIT/DOLCE cache), `lance-graph-arm-discovery` (aerial), `crates/jc` (cert + Jirak).
- `.claude/specs/wikidata-hhtl-load.md`, `.claude/knowledge/{owl-dolce-hhtl-compartments-aerial-fed,splat-codebook-aerial-wikidata-compression,ogit-owl-dolce-ontology-compartments,phi-spiral-reconstruction,zeckendorf-spiral-proof}.md`.
- CLAUDE.md: The Click (AriGraph as thinking tissue), the Baton (ephemeral handoffs), `I-VSA-IDENTITIES`, `I-NOISE-FLOOR-JIRAK`; `cognitive-risc-classes.md` N4.
