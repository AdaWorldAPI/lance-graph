# SYNERGY-MAP-S00-S07 — reuse / transcode / wire, not build

> Pre-S00 synergy map for the Stockfish × Lance/AriGraph × Palette256 × NARS ×
> GraphRAG × OGAR-doc × a2ui convergence. Verified against ground truth
> (git tips + `file:line`) 2026-07-16. Ownership/representation calls are the
> main thread's; mechanical inventory was fanned out and receipted.
>
> **Pinned tips at authoring:** lance-graph `aa8a3a0` (main) · stockfish-rs
> `f3f728a` (main; PR #12 teacher slice merged `8c8d3df`) · a2ui-rs `239d01e`
> (main) · OGAR `f1952a4` (main) · ndarray `95176af` (master) · MedCare-rs
> `f2f56a2` (main).

## 0. Governing rule (operator, this arc)

> **Reuse canonical owners, transcode useful algorithms onto existing
> representations, and add new structures only where a concrete missing
> capability is demonstrated.**

Three operator constraints ride on top:
- **All generic cognition is lance-graph-owned.** NARS revision, StyleFamily,
  tactic recipes, retrieval *reasoning* → lance-graph. Domain repos hand it
  typed values; they never host cognition.
- **No separate relationship table.** Edges live in the canonical node's
  `EdgeBlock` (12 in-family + 4 out-of-family) + the value tenants. CSR/CSC for
  algorithms is a *transient* blasgraph projection, computed then discarded.
- **No fork.** graphrag-rs is a reference to read, not a dependency to link.

## 1. TL;DR — the arc is a WIRING arc, not a BUILD arc

The fork-vs-steal question is settled and, on re-ground, **largely already
executed upstream**:

- **The graphrag operators are already stolen onto the SPO carrier.** RRF, PPR/
  HippoRAG, Leiden communities, BM25, chained episodic search, basins all landed
  on lance-graph main as `D-GR-*` (graphrag plan v1.2), each reading the
  `TripletGraph` — **not** a petgraph/private graph. There is **no fork** and
  **no relationship table**. (§4.A)
- **The "learning curve" tenants already exist in part.** The autopoiesis
  triangle `FrozenStyle` / `LearnedStyle` / `ExploreStyle` landed as
  `VALUE_TENANTS` 10–12 (three palette256 lanes). These *are* the learning-curve
  wired into tenants, and they map 1:1 onto Stockfish's
  Frozen/Teacher(Learned)/External/Exploration `CandidatePolicy` arms. (§4.B)
- **The Stockfish teacher stack is SOLID** on stockfish-rs main — the earlier
  "DecisionEpisodeV1 absent" was a **108-commit-stale-checkout artifact**, now
  corrected. (§4.C)

**So S00–S07 collapses to:** wire the already-landed operators through the
`DocGraphQuery::retrieve` seam on the real carrier; land the one reserved
episodic-witness tenant; and connect Stockfish's `DecisionEpisodeV1` → episodic
tenant → NARS revision (in the triangle tenants) → `ExternalPolicy` return. The
**net-new structure surface is three thin seams, none of them memory** (§6).

## 2. Reconciliations (do not relearn)

- **Stale-checkout law.** Inventories must be taken against `origin/<default>`,
  never a local checkout. The Stockfish false-ABSENT (`DecisionEpisodeV1`) and
  the lance-graph 108-commit gap both came from stale local trees. All rows
  below are re-grounded against the pinned tips.
- **`SearchEpisodeKey` does not exist.** The brief's name is wrong; the episode
  identity type is **`GameEpisodeKey`** (`stockfish-rs src/episode.rs:92`). The
  "five-identity model" is four named identities + Strict/Retro physical split.
- **Proposal, not ratified charter.** `OGAR docs/A2UI-SCREEN-ADDRESSING-PROPOSAL.md`
  is a merged proposal graded `[S]`; council + P-REHOST (C4) pending.
- **Stale docs:** `.claude/v3/soa_layout/tenants.md:41` says "328 B reserved" —
  live code is **292 B** (triangle lanes 10–12 landed; Full slab ends row-offset
  188). Re-check tenants against `canonical_node.rs`, not the doc.

## 3. Ownership matrix (S00 seed)

| Meaning | Canonical owner | Repo |
|---|---|---|
| Exact position / domain identity (chess) | `PositionKey`, `GameEpisodeKey`, `ChessEvidenceRef` | stockfish-rs |
| Falsifiable hindsight teacher (domain) | `DecisionEpisodeV1`, `TeacherTrace/Label`, `SearchEvent`, `CandidatePolicy` | stockfish-rs |
| Canonical memory (nodes/edges/episodes/tenants) | `NodeGuid`/`EdgeBlock`/`NodeRow`, `VALUE_TENANTS`, `MailboxSoA` | lance-graph-contract |
| Generic cognition (NARS, StyleFamily, recipes, retrieval reasoning) | `crystal::TruthValue`, `style_family`, `recipes`, `arigraph/*` | lance-graph |
| Hot routing codes | Palette256 (`bgz17`), CAM-PQ (`ndarray`), `cognitive_palette` 226-atom codebook | bgz17 / ndarray / lance-graph-contract |
| Time-series / version read | `temporal.rs QueryReference::at + deinterlace` | lance-graph-planner |
| Meaning / IR / vocabulary / evidence | `ClassView`(trait in contract), `ogar-doc-ir`, `ActionDef/ActionInvocation`, `nav_witnessed` | OGAR |
| Projection / interaction / edit | `NodeDelta`/`ActionInvoke`(frames, OGAR), `KlickwegEdge`, skins, paint | a2ui-rs |
| Execution orchestration | `rs-graph-llm`/graph-flow (UPSTREAM/external), `ladybug-rs` (design ref) | external — NOT in scope this session |

## 4. Ingredient inventory (receipted)

### A. GraphRAG retrieval — ALREADY on lance-graph main (EXTEND, do not transcode-cold)

| Ingredient | Home `file:line` | Status | Action |
|---|---|---|---|
| RRF (Reciprocal Rank Fusion) | `arigraph/rrf.rs:64` | SOLID (pure fn over `&[&[ScoredId]]`, k=60) | REUSE — fusion keystone |
| Personalized PageRank / HippoRAG | `arigraph/ppr.rs:112` (`TripletGraph::personalized_pagerank`, NARS-weighted) | SOLID | REUSE |
| Louvain + Leiden refinement | `arigraph/community.rs` (`communities()`) | SOLID | REUSE |
| BM25 lexical | `arigraph/bm25.rs:44` | SOLID | REUSE |
| Chained episodic search (AriGraph Eq.1) | `arigraph/episodic.rs:345`; basins `:243` | SOLID | REUSE |
| Causal-weight chain (Pearl ladder) | `lance-graph-cognitive/src/search/causal.rs:187` | SOLID | REUSE |
| Reranker / cross-encoder | `contract/high_heel.rs:1093`; `neighborhood/search.rs:255` | SHAPED (lens/vector rerank; no cross-encoder retrieval op) | ADAPT |
| retrieval-explain / dual-level / lightrag | — | ABSENT (doctrine only) | NEW if demonstrated |
| **Unifying seam** `DocGraphQuery` (default rung-aware `retrieve()`) | `contract/doc_graph.rs:206` (`:249 retrieve`) | SHAPED — impl only on `MockDocGraph` (`:318`); real `impl for OsintRetriever` is a `///` spec (`:189`); live `OsintRetriever::retrieve` (`arigraph/retrieval.rs:235`) still old path | **the S04 wiring point**, gated on the "G0 load-bearing verdict" |

**Verdict A:** the operators exist and read the carrier. S04 = wire the legs
through RRF into `DocGraphQuery::retrieve` on the real `OsintRetriever`. This is
**integration, not harvest** — the harvest already happened (`1306bf6`,
`4ea1f21`, `2d45279`, `b5b0b30`, `5a5bc4c`).

### B. The learning-curve tenants — PARTIALLY LANDED

| Ingredient | Home `file:line` | Status | Action |
|---|---|---|---|
| Autopoiesis triangle: `FrozenStyle`/`LearnedStyle`/`ExploreStyle` value-tenant lanes (10–12, 12-B palette256 each) | `canonical_node.rs:969-992`, VALUE_TENANTS enum `:828` | SOLID (commits `381ba4b`/`b750881`/`04f8cc9`) | **REUSE — this is "learning curve in tenants."** Stockfish `CandidatePolicy` Frozen/Teacher/External/Exploration maps onto Frozen/Learned/Explore |
| 226-atom palette256 FROZEN value codebook (`AtomId(u8)`, `AtomCatalogue::resolve`) | `contract/cognitive_palette.rs:1` (`b26d184`) | SOLID | REUSE — the value-tenant addressing codebook (the "PaletteCodebookId" role) |
| Episodic-witness tenant (96-bit, AriGraph-adjacent) | `soa_view.rs:257-274` (deferred accessor: "`EpisodicWitness64` is NOT YET a code symbol") | SHAPED / reserved (292 B headroom in the 480-B slab) | **NEW_REQUIRED — the one memory structure to land**; mint into reserved headroom, RESERVE-DON'T-RECLAIM |
| temporal.rs version-range / time-series read | `lance-graph-planner/src/temporal.rs:139` (`QueryReference::at`), `:91` deinterlace; `contract/temporal_pov.rs:177` | SOLID | REUSE (migration `temporal-markov…v1`: D-MTS-5/6 GREEN; **D-MTS-1 parity probe still Queued** — VSA cutover not done) |
| AriGraph `EpisodicMemory` / `markov_soa` / `episodes_to_palette_layers` | `graph/arigraph/episodic.rs`, `arigraph/markov_soa.rs`, `planner cache/convergence.rs` | SOLID (cold path) | REUSE |
| Palette256 tables (256×256) + CAM-PQ 6×256 | `bgz17/palette.rs`+`distance_matrix.rs`; `ndarray/src/hpc/cam_pq.rs` | SOLID canonical | REUSE (never re-derive a LUT) |
| `impl CamCodecContract` | — (`contract/cam.rs:182` trait only; impl is in ndarray fork) | ABSENT in-tree | ADAPT — wire ndarray codec to the contract trait (Phase-3 TODO) |
| NARS `TruthValue` | canonical: `contract/crystal/mod.rs:93`; 5+ competing defs elsewhere | SOLID but unconsolidated | REUSE `crystal::TruthValue`; do NOT mint a 6th |
| StyleFamily(12) + 34 tactic recipes + rung ladder | `contract/style_family.rs`, `recipes.rs`, `recipe_kernels.rs` | SOLID | REUSE — the selector that projects component evidence → one ordering |

### C. Stockfish teacher stack — SOLID (domain owner)

All `stockfish-rs` main (`f3f728a`), tested (PR #12: 29 lib tests, 4 hard gates green):
`DecisionEpisodeV1` (`episode.rs:171`), `TeacherTrace` (`trace.rs:169`),
`TeacherLabel` (`trace.rs:234`), `SearchEvent` enum (`trace.rs`), `CandidatePolicy`
+ Frozen/Teacher/External/Exploration (`policy.rs:202/211/225/244/276`),
`search_with_order` (`search.rs:132`), `PositionKey` (`episode.rs:39`),
`GameEpisodeKey` (`episode.rs:92`), `ChessEvidenceRef` (`episode.rs:124`),
Strict/Retro split, golden replay fixture + `examples/expert_iteration_stream.rs`.
**Action: REUSE via the neutral LE codec** — Lance/NARS learn chess by decoding
`DecisionEpisodeV1`, never reopening Stockfish internals. `ExternalPolicy` is the
return seam; its Lance/NARS score *producer* is the unbuilt S05 downstream.

### D. OGAR meaning / evidence / projection

| Ingredient | Home | Status | Action |
|---|---|---|---|
| `ClassView` trait / `WideFieldMask` | `contract/class_view.rs:903/221` (trait+type OWNED by lance-graph-contract); `OgarClassView` impl in OGAR | SOLID (mask retype-in-place NOT yet landed: `rbac.rs:176` still narrow `FieldMask`) | REUSE trait; ADAPT the wide-RBAC seam |
| `ogar-a2ui-frame` (NodeDelta/ActionInvoke) | `OGAR/crates/ogar-a2ui-frame/src/lib.rs:124/137` | SOLID | REUSE (a2ui-core re-exports) |
| `ogar-doc-ir` (doc.v1, closed `RegionKind`, reading order, Provenance, spatial rails) | `OGAR/crates/ogar-doc-ir/src/lib.rs:208` | SOLID — **but NO stable region identity** (region has no id) | REUSE; region-id is a gap for S03 |
| `EvidenceAddress` / `SourceSpanAddress` | ABSENT (nearest `reasoning.rs:39 EvidenceRef` = batch-granularity, wrong phase) | ABSENT | **NEW_REQUIRED (S03)** — see §6 |
| `ProjectionAddress` (object/class_view/field_position/template_region/evidence) | ABSENT (4/5 components exist scattered; evidence + region-id missing) | ABSENT | **NEW_REQUIRED (S03)** — see §6 |
| `ActionDef`/`ActionInvocation` + SPO emit | `ogar-vocab/src/lib.rs:389/508`; `ogar-emitter:774` | SOLID | REUSE |
| `nav_witnessed` (codegen gate ≠ runtime SPO predicate) | gate `ogar-emitter/do_adapter.rs:46`; runtime const ABSENT | SHAPED (OGAR issue #210 OPEN) | OGAR-owned follow-up; a2ui emits a `NavWitness` value and stops |
| Region-grammar → nested ClassView (Odoo layout facts) | OGAR `#211` merged as `docs/…odoo-transpile-arc-closure.md` (doc); harvest external | PROPOSED (doc names the seam; no OGAR code) | ADAPT when built; feed nested ClassViews, WIDE masks only |

### E. a2ui projection / edit (S06)

SOLID @ `239d01e`: `NodeDelta`/`ActionInvoke` (re-exported), `KlickwegEdge`
(`a2ui-server/desktop.rs:56`), `resolve_nested`/`NestedSurface`
(`a2ui-wasm/lib.rs:165/377`), `Skin::{Form,Flow}` (`a2ui-paint/lib.rs:114`),
`PaintLayout`+real wgpu `GpuPainter` (`:187/:498`), resolved-surface accessor,
WideFieldMask fail-closed RBAC (`project.rs:70`).
**ABSENT / NEW for the projectional editor:** `SetField`/write-frame (3rd Frame
variant, OGAR-side), `EditCommand`/operation-journal, `ProjectionAddress`,
Grid/Graph/Timeline/Spatial skins, semantic-LOD, browser glyph-raster present.

### F. External (out of scope this session — do not assume reachable)

`rs-graph-llm` / graph-flow: **not an AdaWorldAPI repo** (org search 0 hits),
absent locally; the operator "resets from upstream". `ladybug-rs`: real GitHub
repo, design-ref only, not in scope. All execution primitives are DOCUMENTED-ONLY.
Any S05 execution work uses these as *design references*, not imports, until the
operator brings them into scope.

### G. automataIA external quarry (EXTERNAL — patterns, not deps)

Confirmed at README/source level (the structured harvest agent hit the
output cap; these are from the operator's own analysis + earlier README fetches —
treat perf numbers as README-level, not proven):
- **graphrag-rs** — retrieval operators (already independently landed as D-GR-*; use as cross-check reference only). TRANSCODE-already-done.
- **wasm-typst-studio-rs** — persistent compiler session + rendered-coord→source-span bidirectional addressing. **The strongest S03/S06 analogue** — the pattern for `ProjectionAddress`↔`EvidenceAddress` round-trip. REUSE pattern.
- **lodviz-rs** — native/WASM algo-core split, LTTB + M4 LOD, linked selection. REUSE for the S06 Graph/Timeline skin — **with semantic-needle pinning** (LTTB must never drop PV-changes / refutations / contradiction events).
- **graph-librarian-rs** — sequential model-lease phase scheduling. REUSE as an rs-graph-llm workflow pattern (S05), not storage.
- **dashboard-studio-rs** — command-pattern undo/redo journal. REUSE ergonomics for `EditCommand` (S06); REJECT its ECharts/JSON renderer + persistence.
- **rust-relations-explorer** — source-graph query catalogue (callers/cycles/centrality). REUSE queries onto Lance; REJECT its JSON KnowledgeGraph persistence.
- **agentic-graphrag-rl-trainer** — component-wise reward + immutable checkpoints. REUSE the reward-decomposition *concept* for S05 (component evidence → NARS predicates).

## 5. Execution order (collapsed)

```
   (this map)
      ↓
S00  ownership + identity contract  ── OGAR doc; reuse existing types, mint nothing yet
      ├─ S01  land the episodic-witness tenant (reserved headroom) + ingest DecisionEpisodeV1
      ├─ S02  Palette256 hot retrieval — REUSE bgz17/cam_pq + cognitive_palette codebook; test dual-lane
      └─ S03  EvidenceAddress + ProjectionAddress (the two real NEW types) + doc-ir region-id
              ↓
S04  WIRE the landed D-GR-* operators through RRF into DocGraphQuery::retrieve (integration, gated on G0)
              ↓
S05  expert-iteration: DecisionEpisodeV1 → episodic tenant → NARS revision in the triangle tenants
     → ExternalPolicy return.  Execution via rs-graph-llm/ladybug DESIGN REF only (out of scope)
              ↓
S06  a2ui projectional editor: SetField write-frame + EditCommand journal + Grid/Graph/Timeline skins
     (Typst-pattern bidirectional addressing; lodviz LOD with needle-pinning)
              ↓
S07  golden vertical slice (Stockfish arm first — everything it needs is SOLID except the episodic
     tenant + the retrieve() wiring)
```

Parallelism after S00: S01/S02/S03 independent. S04 waits on S01+S02 contracts
(but its operators are already coded — it's wiring). S05 designs handles after
S00; live integration waits on S01+S04. S06 audits seams early; canonical
integration waits on S03–S05. S07 needs pinned S01–S06 handovers.

## 6. The ONLY net-new structures (each with nearest equivalent + why insufficient)

1. **Episodic-witness tenant (96-bit, AriGraph-adjacent).**
   Nearest: the `EpisodicWitness64` deferred accessor (`soa_view.rs:257`) + shipped
   seeds `WitnessTable<64>`/`EpisodicEdges64` + `CausalEdge64` W-slot. Insufficient
   because none is a live value-tenant column carrying episodic incidence in the
   hot SoA. **Land it into reserved slab headroom** (RESERVE-DON'T-RECLAIM); it is
   *not* a new table and *not* a new store — one tenant lane beside the triangle.
2. **`EvidenceAddress` (+ `ProjectionAddress`).**
   Nearest: `reasoning.rs:39 EvidenceRef` (Arrow-batch granularity, wrong phase) and
   the scattered projection components (NodeDelta.key + classid→ClassView +
   FieldView.position + template slot). Insufficient because nothing binds
   doc-ir `content_sha256` + page + (missing) region-id + BBoxRail into an
   addressable handle, and nothing composes the five projection components.
   These are the S03 bidirectional-addressing types (the Typst pattern). OGAR-owned.
3. **`RetrievalHit` explanation record** (score components + candidate reason +
   expansion path + episode ids + evidence addresses + codebook/trace ids).
   Nearest: `ScoredId` (id+score only) + the D-GR operators' internal outputs.
   Insufficient because retrieval must be *explainable* end-to-end. A return type
   over the wired `retrieve()`, **not a store**.

Everything else — the graphrag operators, the GraphView-ish seams
(`DocGraphQuery`/`MailboxSoaView`/`TypedGraph`/`graph_router`), palette/CAM-PQ,
temporal.rs, NARS, StyleFamily, the triangle tenants, the Stockfish teacher
stack, the OGAR frames/vocab, the a2ui skins/paint — is **REUSE** or
**TRANSCODE/EXTEND-onto-existing**. No fork. No relationship table. No new memory
architecture.

## 7. Hard gates carried into S07

Stable canonical identities · tenant identity on every row · Strict/Retro leak-
free · **no duplicate canonical graph / no relationship table** · palette codebook
version pinned · exact≠approximate identities separated · evidence resolves both
directions · rare events survive LOD (needle-pinning) · policy checkpoints
immutable · workflow contexts carry handles not memory bodies · RBAC fail-closed
before framing · deterministic restart/replay · benchmarks are measured receipts.

---

*Ground-truth receipts: fan-out inspection (wf_24525178) + lance-graph re-ground @
`aa8a3a0`. Stale-inventory corrections in §2. This map is pre-S00; S00 ratifies
the ownership matrix (§3) in OGAR docs and opens the identity contract.*
