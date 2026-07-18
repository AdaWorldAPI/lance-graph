# GraphRAG Representations Inventory — 7 papers × the V3 substrate

> **READ BY:** integration-lead, truth-architect, trajectory-cartographer,
> convergence-architect, dto-soa-savant, v3-envelope-auditor — and any session
> designing the OCR→KG→Markov endgame (tesseract `doc.v1` → AriGraph-style KG →
> GraphRAG retrieval → DeepNSM temporal-Markov context).
>
> **Status:** synthesis (2026-07-18). Paper briefs = 6 Opus readers + MDPI
> fetch (this session). Substrate rows = v3 harvest, source-verified against
> `canonical_node.rs` / `class_view.rs` / `facet.rs` / `ocr.rs`. Honesty flags
> [G]/[H]/proposed carried through; nothing silently promoted.

---

## 0 — The probe that lets every representation share ONE SoA

**`PROBE preset-vs-dispatch`** (`.claude/board/EPIPHANIES.md:1184`; reference
pattern `contract/src/ocr.rs:104-115`). "All tenants can be in every SoA" is the
**`ValueSchema::Full`** variant — compile-proven to equal *every* tenant
(`canonical_node.rs:1127`: `assert!(Full.field_mask().count() == VALUE_TENANTS.len())`).
The write path is a pure function of which tenants the class's schema materialises:

```rust
let schema = classid_read_mode(classid).value_schema;   // ocr.rs:105
if schema.has(ValueTenant::EntityType) { write [o..o+2] }   // 108
if schema.has(ValueTenant::Energy)     { write [o..o+4] }   // 112
```

One writer, one 512-B row (`key16|edges16|value480`), each lane written **iff the
live schema carries it**. So the representations below are not separate stores —
they are **tenant lanes / ClassView readings of one SoA**, selected by classid.
"One SoA, never transformed" (`EPIPHANIES:5816` §11.1).

## 1 — Inventory of the 7 arxiv texts

| Paper | Native representation | Format | Witness reference | In workspace? |
|---|---|---|---|---|
| **AriGraph** 2407.04363 | semantic SPO + **episodic edge** (episode → all its triplets) | triplet + episodic incidence | episode vertex `vₑᵗ=oᵗ` | SPO ✓ · episodic edge **MISSING** (string copy) |
| **PersonalAI** 2506.17001 | object SPO + **thesis** hyper-edge (semantic) + **episodic** hyper-edge (temporal) | 3 vertex types + 2 hyper-edges | passage node `vᵢₑ=dᵢ` | object ✓ · thesis **MISSING** · episodic=basins()✓ · temporal=Lance versions |
| **Document GraphRAG** MDPI | Document-KG (chapter/section/chunk URIs) + keyword IKG | hierarchical doc tree + keyword edges | chunk URI | doc-tree = vertical radix walk (planned doc-W4) |
| **SAP Practical** 2507.03226 | dep-parse SPO + **RRF fusion** (vector⊕graph⊕relation) | 3 separate embeddings + rank-fusion | chunk id | SPO ✓ (DeepNSM) · **RRF MISSING** |
| **StepChain** 2510.02827 | sub-question set + **BFS evidence-chain** `Πsᵤ` | decomposed queries + ordered paths | passage (on-the-fly) | BFS ✓ · **decomposition + path-structure MISSING** |
| **GraphRAG-FI** 2503.13804 | filtered set + logits-integration tier | two-tier {coarse,fine} + demote | attention proxy | MUL/DK ✓ (VALIDATES) · filter shape adoptable |
| **GraphRAG under Fire** 2501.14050 | poison relation `r*` (same S+P, diff O) + provenance-trust | relation + per-chunk trust | **source provenance** | source_url ✓ but **trust-wire MISSING** |

## 2 — The facet ladder (le-contract §3) — where each representation addresses

Atom = **16 B** = `4 B prefix (domain|appid|classview-u16) + 96-bit payload`;
ClassView selects the reading (slot purity — labels never in a payload slot).

| # | Shape | Semantics | L4-exact? |
|---|---|---|---|
| L1 | 6×(8:8) | `part_of:is_a` — **episodic basins rail** | — |
| L2 | 6×(8:8) | `memberof:members` | — |
| L3 | 6×(8:8) | `mereology:taxonomy` | — |
| **L4** | 6×(8:8) | **`palette256²`** — each pair indexes 256×256 dist/compose LUT (Fisher-z ρ≥0.999); Morton 2bit×2bit 4×4 perturbation carrier | **YES — similarity = one table read** |
| L5 | 4×(8:8:8) | SPO-style triplets | — |
| L6 | 3×(8:8:8:8) | SPOG quads (Odoo) | — |
| L7 | 2×48-bit | `hhtl ++ helix` — absolute location (q2 FMA) | — |
| L8 | 2×48-bit | `helix ++ CAM_PQ` — analog old style | — |

Extended 6×(8:8) readings: `area:location` (stacked/**vertical**), `basin:relationtype`
(**horizontal**), `relationtype:relationtype_orthogonal` (static-basin).

> **Reading A (operator, this session)** — `2×(basin:identity SPO)`: the 6 pairs
> grouped 3+3 as `[semantic SPO | witness SPO]`, each S/P/O a `(basin:identity)`
> pair. **This is a NOT-YET-SANCTIONED extended reading** of the L1–L4 plane
> (distinct from L5's 4×3-byte triplets — it keeps `u8:u8` pairs, grouped
> semantically). It closes the AriGraph episodic-edge gap in-register. Falsifier:
> does the `basin` byte agree with the Leiden `communities()`/`basins()`
> partition, per-typed-plane (never pooled)? [proposed — probe-gated]

## 3 — The 13 value tenants (canonical_node.rs:830-993) + the 14th

| # | Tenant | Width | Carries | witness·context·basin·time·NARS·edge | Built |
|---|---|---|---|---|---|
| 0 | **Meta** (MetaWord) | 8B | awareness/NARS/free-energy bits | W·—·—·—·**N**·— | ✅ |
| 1 | Qualia | 8B | 16×i4 channels | angle | ✅ |
| 2 | **MaterializedEdges** | 32B | 4× CausalEdge64 | **W·C·—·t·N·E** | ✅ |
| 3 | Fingerprint | 32B | identity print (points-to) | context(id) | ✅ |
| 4 | HelixResidue | 6B | 48-bit place (Signed360) | basin/location | ✅ |
| 5 | TurbovecResidue | 16B | PQ32×4 residue | search | ✅ |
| 6 | Energy | 4B | spatio-temporal accumulator | time(partial) | ✅ |
| 7 | Plasticity | 4B | persisted plasticity | learn | ✅ |
| 8 | EntityType | 2B | OGIT class ordinal | class | ✅ |
| 9 | **Kanban** | 8B | phase\|exec\|cycle(u32) | **time(cycle)·resolution** | ✅ |
| 10-12 | **Style triangle** (Frozen/Learned/Explore) | 3×12B | 12 palette256 policy atoms | policy (NOT a trajectory) | ✅ Full-only |
| 13 | **BoardAggregates** | — | per-mailbox board aggregates | board·time | ❌ **PLANNED** (W2a, offset 188, batched mint) |

`ValueSchema` presets: **Bootstrap**(0) · **Cognitive**(7, 66B) · **Compressed**(4,
56B) · **Full**(13, 156B → 324 B reserve). All layout-preserving.

## 4 — THE MATRIX (every representation × the questions)

Legend: ✓ yes · ✓✓ *is* this · — no · V/H/E = vertical/horizontal/edge axis.

| Representation | Format | Witness ref | Wit | Ctx | Basin | Axis | Time | NARS | Causality-traj? | Wire |
|---|---|---|---|---|---|---|---|---|---|---|
| **SPO triple** | `Triplet{s,p,o,truth}` / spo:u64=3×u8 | WitnessCorpus source_url | ✓ | ✓✓ | ✓ | E | — | ✓ | **✓** owned+wit+NARS | chain episodic_search (Eq.1) |
| **Episodic-witness edge** | `EpisodicEdges64` 4×u16 / WitnessEntry | episode/passage node | ✓✓ | ✓ | ✓✓ | E+H | ✓ | ~ | **✓** the episodic arc | wire `EpisodicEdges64` (unwired) |
| **6×(8:8) facet (Reading A)** | `classid\|2×(basin:id SPO)` | 2nd (witness) SPO | ✓ | ✓ | ✓✓ | V/register | — | — | candidate (static row) | ratify as §3 reading + basin-byte probe |
| **Community/basin partition** | `Communities` + `EpisodicBasins` | — (derived) | — | ✓ | ✓✓ | H | ~ | — | **grounds** trajectories | + thesis (3rd partition) |
| **Temporal standing-wave** | `SoaWavePrimer{±radius}`→`WaveProjection`; temporal.rs stream | Lance version / stream pos | ✓ | ✓✓ | ✓ | **T** | ✓✓ | — (leashed) | **✓✓** the WAVE IDENTIFIES | **D-MTS-1** keystone probe (un-run) |
| **CausalEdge64** | u64: rung\|mantissa\|W-slot\|temporal | W-slot witness arc | ✓✓ | ✓✓ | — | E | ✓ | ✓✓ | **✓✓✓** the trajectory particle | D-MTS-6b shrink; CascadeChannels8 |
| **Palette / Fisher-z metric** | L4 6×(8:8) palette256² / `PaletteDistanceTable` 256² | — (metric) | — | — | ✓ | L4 addr | — | — | — enables matching | D-MTS-2/3 shader cert |
| **COCA-4096 constant** | 4096² u8 matrix / cascade L3 / 96D PQ codebook | IS certification witness | ✓ | ✓ | — | L3 rung | — pinned v3 | — | — reference frame | anchor doc (à la Jina-v5 registry) |
| **Thesis vertex** *(unbuilt)* | community-within-one-episode | episode | ~ | ✓ | ✓✓ | H | — | — | grounds | parallel to `EpisodicBasins` |
| **RRF fusion** *(unbuilt)* | `rrf(lists,k=60)→Vec<ScoredId>` | — | — | ✓ fuses | ✓ fuses | combinator | — | gate | free fn in `retrieval.rs` — **keystone** |
| **Sub-question decomp** *(unbuilt)* | query→{q₁..qₘ} seed sets | — | — | ✓ | — | — | seq | — | LLM/DeepNSM tail | `retrieve_multi` on `OsintRetriever` |
| **Evidence-chain `Πsᵤ`** *(unbuilt)* | pred-map + `A-(r)→B-(r)→C` | the path (audit trail) | ✓ | ✓✓ | — | E-path | ord | ✓ | **✓✓** chain = trajectory | extend `get_associated`→`associated_paths` |
| **Provenance→trust** *(unbuilt)* | source_url→`TruthValue.confidence` | source_url | ✓✓ | — | — | — | — | ✓✓ | enables trust-gated traj | at `promote_to_spo`/`WitnessCorpus::insert` |
| **BeamSearch** *(unbuilt)* | N semantic paths, dedup | — | — | ✓ | — | E-path | — | — | produces path candidates | `retrieve_beam` on `OsintRetriever` |
| **BoardAggregates** *(unbuilt)* | 14th tenant, offset 188 | — | — | board | — | lane | ✓ | — | — board state | W2a batched mint |
| **Temporal hyper-edge** | `QueryReference::at(v,rung)` + `Episode.step` | Lance version | ✓ | ✓ | — | **T** | ✓✓ | — | **✓** temporal carrier | shipped (temporal.rs); D-MTS-1 parity |

## 5 — Causality-trajectory candidates (the qualifying subset)

A representation is a **causality-trajectory candidate** iff **owned**
(`mailbox_owner()`/write-on-behalf) + **witnessed** (a provenance ref) +
**NARS-revised** (truth freq/conf) + reachable by **proper context edges**
(`CausalEdge64`). Grounded verdict (v3 harvest §E):

- **✓✓✓ the trajectory itself:** `CausalEdge64`/MaterializedEdges (edge + NARS
  mantissa + W-slot witness + owner-provenance); the **temporal.rs stream /
  Lance-versions** (episodic memory = the carrier); the standing-wave **proposes**
  onto it (leashed to the CE64 particle that confirms — `markov_soa`).
- **✓✓ trajectory candidates:** SPO triple (with truth), episodic-witness edge
  (once wired), evidence-chain `Πsᵤ` (the assembled path IS a candidate).
- **grounds, not is:** Communities/basins (basins GROUND, SPO ANCHOR, the WAVE
  IDENTIFIES — `E-THINKING-SPINE-CHESS-EVIDENCE-1`); L1–L3 rails; Meta/MetaWord.
- **not a trajectory:** palette/Fisher-z (metric), COCA (constant frame),
  Fingerprint/Helix/Turbovec/EntityType/Plasticity (bare values), the style
  triangle (policy the trajectory dispatches through, explicitly NOT a gestalt).

## 6 — All-tenants-in-one-SoA verdict

Under `ValueSchema::Full` + the `ocr.rs` write-path dispatch, the **tenant-lane**
representations (in-row) are: SPO (via edges+fingerprint), CausalEdge64
(MaterializedEdges), the facet register (L1–L4 readings), Kanban (resolution),
Energy (temporal-partial), the style triangle, and BoardAggregates (once minted).
The **derived / out-of-row** representations are: Communities/basins/thesis
(computed partitions), the temporal standing-wave (a projection over the stream of
rows), COCA/palette (build-time codebooks + LUTs), RRF/decomposition/BeamSearch
(retrieval logic on `OsintRetriever`). Every reader **re-resolves** the substrate
from the classid in the 16-B key — no `ClassRoutingDTO`, nothing crosses a mailbox
boundary (`EPIPHANIES:1184`).

## 7 — Priority wire order (highest ROI first)

1. **RRF fusion** — the keystone; every leg exists, one free fn closes it.
2. **Provenance→trust** — source_url→confidence at promote; core for OCR mixed-trust docs.
3. **Chained episodic_search (Eq.1)** — makes the crate actually AriGraph, not RAG.
4. **Evidence-chain `Πsᵤ`** — surfaces the audit trail the NARS substrate exists for; = causality-trajectory candidates.
5. **Thesis partition** — the most load-bearing memory type (PersonalAI Table 3).
6. **Sub-question decomposition** + **BeamSearch** — the two winning retrieval halves.
7. **D-MTS-1** (Markov-temporal-stream parity) — the probe gating the whole VSA→stream migration.
