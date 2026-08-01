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

---

## 8. Capstone inspiration — attention headers over the 64×64 (operator-taught, 2026-07-31)

> **⛔ STATUS: RESTING PLAN (operator ruling, 2026-07-31). Too many gaps — do
> not build from ANY §8 subsection.** A 5-savant + 2-reviewer council pass
> (cycle 1 of a planned 2; cycle 2 cancelled on this ruling) graded ~25 claims:
> most of §8.8's floor is CODED (24 witness loci / i4 ∈ [−8,+7] /
> `WITNESS_REGISTER_BYTES = 12` in `contract/src/causal_witness.rs:52-58`;
> certified rung↔pearl↔2³ test; `RECIPES: [Recipe; 34]`; F-triage thresholds
> exact in `grammar/free_energy.rs:28-35`) — but: **§8.6/§8.7's PURPOSE was
> wrong** (operator: they are for torque / semantic pressure / attention
> headers, NEVER grammar resolution — grammar belongs to the deterministic
> floor alone, no residue ladder); §8.7's "interference" is unsigned-energy
> ensembling unless a signed/phase carrier is wired (`ShaderResonance` has no
> sign term) and its falsifier cannot distinguish the two; Kant is the ONLY
> philosopher with a code anchor (`mul/compass.rs:19-39` needle — Wittgenstein/
> Nietzsche/Hegel/Schopenhauer are ABSENT everywhere, incl. as style vectors);
> C4.3's basin↔basin map is ABSENT (`BasinCode` is a different object);
> Valenz is positional-only, German tense labels don't exist (English `Tense`
> enum does), RungLevel is 0–9 not 0–10; φ−1 "permanent humility" is DOC-ONLY.
> Prior art not consulted: `deepnsm-morton-comma-facet-v1.md` +
> `causal-rung-standing-wave-v1.md` (both 2026-07-21) cover adjacent territory
> with their own probes. Full graded table banked in the council scratchpad;
> any revival starts from a fresh Phase-0 spec on the PRESSURE framing, routed
> through `epiphany-brainstorm-council`, and re-grades C7.4 (unverified).

> **Status: INSPIRATION / CONJECTURE.** Appended post-authoring, operator-
> directed ("pay this idea forward"). Nothing here mandates a net-new
> structure (§6 stays closed); it names how already-shipped pieces compose,
> for whichever session runs the S-arc. Each claim carries its receipt or is
> marked CONJECTURE.

### 8.0 ⊘ MEASURED RETRACTION of the original §8.1 (2026-07-31, same day)

**The original §8.1 claimed "the 64×64 = 4096 tile is ONE relation wearing
three hats" (chess/NNUE · gridlake · attention header). Six code-reading
passes (Read-only, grep/sed/head/tail prohibited) falsified every load-bearing
claim.** The retracted text is kept below as §8.1-RETRACTED per append-only
canon; §8.1-MEASURED replaces it. Root cause, named plainly: **numeric
coincidence (4096 appearing in several places) was treated as identity, and a
mechanism recalled from general knowledge (how NNUE accumulators work) was
written with a "[Receipt: …]" label it had not earned.**

| original claim | measured |
|---|---|
| domino sweeps a 64×64/4096 tile | **4×4.** `const TILE: usize = 4; const LANES = TILE*TILE; // 16 lanes/board` (`symbiont/src/domino.rs:29-30`). No `64×64` or `4096` literal anywhere in the file. The 16×16 is the AMX GEMM *batch* shape (16 boards), not a board. |
| domino = NNUE-style accumulator | **ABSENT.** No chess/NNUE/accumulator concept in `domino.rs`, `bridge.rs`, `kanban_loop.rs` (read whole). `Energy` is a plain per-board f32 reduction sum, not an incremental game-state accumulator. |
| lane_j 4096 = a square×square relation | **A hash-bucket grid.** Station name → FNV-1a64 → two axis bytes → Morton interleave → slot in a flat group-by table; cells hold `mins/maxs/sums/counts` (`onebrc-probe/src/lane_j.rs:137-143`) for the 1BRC benchmark's ~400 station groups. The axes are halves of a hashed key and carry no geometry. |
| "~448 Mrows/s, measured" | **Misattributed.** `lane_j.rs:17-23` *cites* ndarray #227's `onebrc_cascade_probe` for that figure. This crate's own best single-thread numbers are ~21.5/23.3 Mrows/s (`onebrc-probe/README.md:286-287`). |
| attention header is O(1) "over exactly this tile" | **k×k palette-archetype, canonically 256×256** (`bgz-tensor/src/attention.rs:19-23`, `lib.rs:26-30`); indices are `q_palette_idx`/`k_palette_idx` = quantized weight-row archetypes, not squares. A 64×64 exists only as the narrow p64-compat export `build_hip` / `as_p64_distances() -> Option<[[u16;64];64]>` (`hhtl_cache.rs:512-538`). |
| stockfish-rs backs the NNUE reading | **Not readable in this environment.** Glob `/home/user/stockfish*/**` and `/home/user/*/stockfish*` → zero matches; SYNERGY-MAP references it only by pinned commit (`f3f728a`, header L8-9). The claim was unverifiable by construction. |
| (§8.3) "Gaussian-splat spatial blasgraph 3DGS" | **No splat/3DGS/point-cloud/render concept in blasgraph** (`mod.rs`/`semiring.rs`/`hdr.rs` read whole). Blasgraph is GraphBLAS-style sparse-matrix algebra over 16384-bit vectors + a Hamming "exposure cascade" where **HDR is a photography metaphor**, integer-only hot path (`hdr.rs:12-13`). |

### 8.1-MEASURED — what the pieces ACTUALLY are, and the ONE real Stockfish seam

Read as separate objects that happen to share round numbers — not one relation:

- **domino** (`symbiont/src/domino.rs`) — a 4×4 Morton-addressed **BF16 tile of
  16 lanes per board**, living in the `Fingerprint` value tenant; 16 boards
  batch into one AMX `16×16` BF16 tile GEMM (`C[16,16] = A[16,32]·W[32,16]`,
  `:134`). One step: gather lanes → tile GEMM → per-board slice summed into
  the `Energy` tenant (f32) and the 16 values re-quantized back to BF16.
  `CognitiveWork` invokes exactly `domino::domino_sweep(&mut self.rows, 3)`
  then `sync_energy()` (`kanban_loop.rs:106-109`). Pay-forward is a fair
  description of the re-quantized feedback; **NNUE is not**.
- **lane_j's 64×64** — a *cache-tier knob*, not an architecture. The operator's
  own question is quoted verbatim in the source: *"or should we assign 8x8 or
  64x64 gridlake soa / what if we match the soa into a grid 64x64 = 4096 xBF16
  = 16kb?"* (`lane_j.rs:1-8`). It sizes a group-by accumulator table to fit
  cache; that is its whole job.
- **bgz-tensor attention** — `table[q_palette_idx][k_palette_idx]` replacing
  `Q·K^T/√d` at O(1) (`attention.rs:6-10`), over **palette archetypes**,
  canonically k=256. Multi-hop compose = `xor_bind` of two Base17 entries then
  nearest-palette lookup (`attention.rs:170-196`).

**The real Stockfish seam — measured, and better than the invented one.**
`bgz-tensor/examples/nnue_palette_cosine.rs` (D-PALETTE-NNUE) takes NNUE
**feature-transformer columns** (a data blob exported by a sibling
`stockfish-rs` example) as a **test corpus**, builds a Fisher-z k×k cosine
table, and gates on whether the certified palette256 cosine-replacement
preserves *pairwise-cosine ranking*: `ρ_all ≥ 0.999 && ρ_mid ≥ 0.99`
(`:197-224`). Its own conclusion line: *"The NNUE FT columns ARE a palette256
tenant: the certified Fisher-z cosine-replacement preserves pairwise-cosine
ranking (one-table-read similarity), no materialization."* So NNUE's role here
is **a demanding dataset that validates a codec**, not a mechanism the
substrate imitates. And per §4C the ratified stockfish-rs surface is a
**teacher stack** — `DecisionEpisodeV1`, `TeacherTrace`, `TeacherLabel`,
`CandidatePolicy`, `search_with_order`, `PositionKey`, `GameEpisodeKey` —
decision-episode plumbing, with no NNUE or accumulator in the named set.

### 8.1-RETRACTED (kept for the record — DO NOT BUILD FROM THIS)

>
> `64×64 = 4096` is simultaneously:
>
> 1. **The chess relation itself** — square×square, the Stockfish/NNUE shape.
>    The teacher stack (§4C) already keys on it; NNUE's efficiently-updatable
>    accumulator over that relation is **pay-forward torque**: per move, deltas
>    are summed FORWARD into the accumulator — never recomputed, and (the
>    mis-reading to avoid) never "held" as a memory. A probe that judges this
>    dynamic by retention criteria is measuring the wrong axis; its job is
>    transfer, not storage. [Receipt: NNUE design; symbiont `domino.rs` runs
>    the same shape — 16-lane Morton tiles, int8/BF16 requant feedback.]
> 2. **The gridlake SoA unit** — `onebrc-probe/lane_j.rs`: 4096 cells ≈ 80 KB
>    batch table, ~448 Mrows/s single-thread, `E-1BRC-GRIDLAKE-SWEETSPOT-1`.
>    [Receipt: measured.]
> 3. **An attention header** — bgz-tensor's attention-as-lookup:
>    `Q·K^T/√d → table[q_idx][k_idx]`, O(1) over exactly this tile.
>    [Receipt: shipped, AttentionSemiring + HHTL cascade.]
>
> Same square-pair relation, three reads. No new type needed to unify them —
> the unification IS that they are already the same tile.

### 8.2 The modulation is the Morton inverse-pyramid perturbation shader

The attention header's weights are NOT a matmul: the weight at `[q,k]` is the
cascade value at that Morton address — coarse→fine over the 2bit×2bit 4×4
walk, deterministic phase (coprime CurveRuler stride, D-QUANTGATE), magnitude
the only stored bits (OGAR perturbation canon; carrier per
`E-MARKOV-TEMPORAL-STREAM-1` = the L4 `6× palette256:palette256` tenant).

> **⚠ Name collision, corrected 2026-07-31 after a code read.** This
> "perturbation" is the **OGAR encoding canon** (exponent/location/phase/
> magnitude). It is NOT `crates/perturbation-sim`, which is a **power-grid
> cascading-failure simulator** — graph-Laplacian low-rank perturbation with
> Weyl / Davis–Kahan / Cheeger / Kron plus DC power flow, validated on the
> real 261-bus Iberian network (`perturbation-sim/PAPER.md:20-24`). The word
> "shader" appears in NEITHER. The one adjacent artifact is
> `perturbation-sim/src/splat.rs` — a **PROTOTYPE** anisotropic-Gaussian (EWA)
> coarsening with a `morton2` Z-order code over an *electrical* 2-D spectral
> embedding; real, but a different crate, a different domain, and not 3DGS.
Trained weights (an NNUE eval head, an OCR kernel) are BOLTED ON above this —
optional skill layers. The muscle memory is the dynamic itself, weight-free.

**Except the Pythagorean comma.** Stacking the coprime stride up the pyramid
is stacking fifths: each level boundary ALMOST closes and leaves an
irreducible residual — (3:2)^12 ≠ (2:1)^7. The canon already names where it
goes: *"the unaligned remainder overflows to the next level or full-residual
escalation."* The comma IS the escalation term — it can be distributed
(tempered) or lumped, never removed. **Falsifier for whoever builds this:**
measure the per-level closure residual of the cascade; it must be nonzero and
comma-sized, and it must be ROUTED (next level or escalation) — a build whose
levels close exactly has silently absorbed the comma somewhere, which is the
bug, not the success.

### 8.3 Orthogonal axes: spatial splat hydration × temporal deinterlacing

**Corrected 2026-07-31 (operator):** the first version of this section
mislabelled both axes; this is the corrected reading.

- **temporal.rs is temporal DEINTERLACING**, not merely "the time axis":
  reads are normalized to the reader's pinned frame the way relativistic
  corrections normalize GPS clocks during orbit — a reader pinned at version
  `v` must never consume knowledge minted after `v` (**hindsight-knowledge
  pollution**, the time-travel anomaly), and temporal.rs prevents it with
  much simpler machinery than relativity: version pins + deinterlace. This
  is the mechanism BEHIND §7's "Strict/Retro leak-free" gate, not a separate
  idea. [Receipt: temporal.rs `QueryReference::at` + deinterlace;
  E-MARKOV-TEMPORAL-STREAM-1.]
- **The spatial axis is the Morton-addressed tile** — and the specific
  claim that this is "Gaussian-splat spatial blasgraph 3DGS" was **NOT
  supported by a code read (2026-07-31)** and is downgraded to
  [CONJECTURE — UNANCHORED]: blasgraph contains no splat / 3DGS /
  point-cloud / rendering concept at all (`mod.rs`/`semiring.rs`/`hdr.rs`
  read whole), and its `hdr.rs` "HDR" is a **photography exposure-meter
  metaphor** with an explicitly integer-only hot path (`:12-13`), not
  imaging. The nearest real artifact is a PROTOTYPE EWA/Morton coarsener in
  the unrelated `perturbation-sim` crate (§8.2 note). What IS measured: the
  spatial carrier is the Morton `palette256:palette256` tenant, and
  domino's concrete instance of it is a 4×4 BF16 16-lane tile (§8.1-MEASURED).
  A splat/hydration framing needs its own probe before it is written as
  architecture.

One episode is then read on two orthogonal normalizations: WHERE (splat
hydration over the tile) × WHEN (deinterlaced to the reader's frame). Never
two stores (§7 gate).

### 8.4 Horizontverschmelzung borrows the header shape [CONJECTURE]

**Corrected 2026-07-31 (operator): the first version modelled the fusion with
`vsa_bundle` + XOR — WRONG for this substrate.** The 2026-07-10 supersession
(E-MARKOV-TEMPORAL-STREAM-1) demoted VSA to its I-VSA-IDENTITIES four-test
niche; the Markov trajectory lives on the temporal.rs sorted stream, and the
carrier is the palette256 tenant. A fusion model built on bundle/XOR imports
the retired substrate. The corrected shape:

- **The fusion op is NARS Revision on `(f, c)`** — two truth values from
  independent evidential bases, met over the DEINTERLACED stream (each
  horizon read in its own frame first, per §8.3, so neither pollutes the
  other with hindsight).
- **The attention-header inspiration** is the spatial half: a palette-tile
  lookup mapping WHICH basins of horizon A attend to WHICH of horizon B, at
  what cascade depth — fusion gets a spatial map instead of a scalar.
- **The residual that never fuses** is a COMMITTED CONTRADICTION — the
  canon's own mechanism ("opinions are committed contradictions preserved,
  not resolved") — the comma-analog routed to preservation, not an XOR
  register and not averaged away.

Probe-first per house rule: pass/fail (does Revision-with-spatial-map
out-predict scalar Revision on held-out belief revision?) before any type
lands.

### 8.5 The Evaluation wire — joining the loop half to the reasoning half [DESIGN]

Two shipped halves, measured this arc, never joined:

- **the loop** — symbiont's kanban arc `Planning → CognitiveWork → Evaluation
  → Commit`, fired synchronously by the writer's own version tick
  (`VersionScheduler::on_version → try_advance_phase`; no bus, no ack —
  E-ACK-ELIMINATED-1). [MEASURED: shipped, green, currently sweeping domino.]
- **the reasoning** — `lance-graph-planner/src/cache/nars_engine.rs`:
  Deduction / Induction / Abduction / Revision / Synthesis / Intervention /
  Counterfactual, fully built, with nothing that fires it. [MEASURED:
  shipped, unwired.]

**The join is one phase wide, and it is NOT a replacement.** An earlier
proposal ("make `CognitiveWork` call `nars_engine` instead of the tile GEMM")
was wrong in one word — *instead*. Domino STAYS (§8.1–§8.3: pay-forward
hydration, attention headers, splat field). The NARS ops attach DOWNSTREAM:

```
writer commits batch → knows its version (sync)
  Planning       MetaFilter sweep — who is in play this cycle
  CognitiveWork  domino, UNCHANGED: pay-forward hydration over the 4096
                 tile → energy, firing rows, and the per-level closure
                 residual (the comma, §8.2)
  Evaluation     ← THE MISSING WIRE. Fired rows' premises route to
                 nars_engine by the SHAPE of what fired:
                   · within-board unknowns → a sudoku-style constraint
                     settle (pinned cells + semiring propagation to a fixed
                     point) — deduction as weight-free constraint closure
                     [CONJECTURE — and the "NaN-autocomplete over blasgraph"
                     phrasing is UNANCHORED: a 2026-07-31 read found NO
                     NaN sentinel/detection/propagation in blasgraph, whose
                     cascade hot path is integer-only by design
                     (`hdr.rs:12-13`). The semiring propagation IS real
                     (7 semirings, `blasgraph/mod.rs:12-20`); the NaN
                     encoding of "unfilled" is an unbuilt proposal, not a
                     shipped mechanism]
                   · two independent bases → Revision over DEINTERLACED
                     frames (§8.3/§8.4 — each horizon read in its own frame
                     first; the independence Revision's confidence-raising
                     math assumes)
                   · the routed comma → Abduction / escalation — the
                     residual no level absorbs is exactly the surprise that
                     warrants a hypothesis or a ticket
  Commit         revised (f,c) edges; the never-fusing residual committed
                 as a PRESERVED CONTRADICTION, per canon
```

**Evaluation is where the sweep's output becomes premises.** Domino surfaces
WHAT to think about and how urgently (energy, residual); nars_engine is HOW
it gets thought. Op dispatch is not new machinery either — it is the rung-3
recipe codebook's job (the 34 NARS tactic runbooks), with the established
dyadic/within-board split: Deduction and the NaN-settle stay inside one
board; Revision / Synthesis / Abduction cross mailboxes.

**Shape of the wire:** an `Outcome → premises → op` adapter in the same
spirit as the shipped D-MBX-A6 `Outcome → KanbanMove` adapter on the move
side — an adapter, not an engine, honoring §6 (no net-new structures beyond
a return-type-shaped seam). Consumers then supply PREMISES, not engines: a
domain's own reasoners (differential, abductive frontier, etc.) feed ops
that already exist instead of growing a parallel engine — which is precisely
why consumer kanbanstep wiring was blocked: `CognitiveWork` had no reasoning
surface to hand premises to. With Evaluation wired, that blocker dissolves
for every consumer at once.

**Status:** loop half + reasoning half [MEASURED-shipped]; the Evaluation
wire itself [DESIGN]; the NaN-autocomplete settle [CONJECTURE — probe
before build, per house rule: pass/fail is that the settle reaches a fixed
point that a held-out pinned-cell mask predicts, and that raising the
pin count monotonically shrinks the unfilled set].

### 8.6 Gadamer loose ends — ambiguity resolution is the live consumer

The loose end §8.4/§8.5 left open: what actually USES the fusion machinery,
cycle by cycle. The answer is already in the supersession that governs this
whole section — E-MARKOV-TEMPORAL-STREAM-1's stated purpose is that
*"grammar-resolver ambiguities are resolved live and granularly against a
version-range read (`QueryReference::at(v, rung)` + deinterlace)."*
**Ambiguity resolution is not an application of the §8 stack; it is what the
stack exists for.** Tying the ends:

- **Every ambiguity is a micro-Horizontverschmelzung.** The reader's horizon
  is the prior — Gadamer's *Vorurteil*, which the substrate already carries
  as the NARS `(f, c)` prior read in the reader's OWN deinterlaced frame
  (§8.3: a prior polluted by hindsight is not a prejudice, it is a leak).
  The utterance's horizon is the candidate-reading set — basins. Resolution
  is their fusion. [Receipt: the deepnsm-v2 `Rel` tag feeding the ±8
  antecedent pointer is a shipped micro-instance: a positional prior meeting
  a candidate referent inside the local window.]
- **The resolution triage IS the shipped F-triage, and it routes like
  §8.5's Evaluation table:**
    · resolved inside the local horizon → **Commit** (one reading, one edge);
    · a near-tie (ΔF small) → **Epiphany: BOTH readings committed + a
      preserved Contradiction** — Gadamer's demand that fusion keep the
      tension, already canon ("both triples + Contradiction"), never a
      forced disambiguation;
    · beyond the local horizon (the ±8 Escalate zone) → **escalation to the
      global graph** — the comma-routing analog at the linguistic level:
      what the local horizon cannot close is ROUTED, not absorbed;
    · genuinely stuck (F high) → **FailureTicket**, the <25% tail.
- **The circle never closes, by construction.** Gadamer: fusion is never
  total — the horizons keep moving. The canon already prices this in as the
  **φ−1 ceiling on awareness revision ("permanent humility")**: no amount of
  resolved ambiguity drives confidence to 1, so every fusion leaves the next
  reading genuinely open. The hermeneutic circle = the kanban cycle loop
  (F descends per cycle; the shader rests when surprise is spent), with a
  floor that guarantees the circle stays a circle.
- **[CONJECTURE] the attention-header borrowing (§8.4), specialized:** the
  candidate readings are basins, and the spatial map says which CONTEXT
  basins attend to which CANDIDATE basin at what cascade depth — ambiguity
  resolution as a palette-tile lookup rather than a scored list. Probe
  pass/fail before any type: on held-out ambiguous tokens, does the
  map-weighted resolution beat the flat prior at picking the reading the
  wider context later confirms?

### 8.7 The four lenses — Wittgenstein · Nietzsche · Kant · Hegel, as Doppelspalt ripples

§8.6 reads an ambiguity through ONE horizon-pair. The operator's extension:
read it through FOUR lenses simultaneously, and let the readings INTERFERE —
the Doppelspalt shape. No new machinery is required; every part is a shipped
shape:

- **A lens is a style vector, already canon.** `atom-basis-inventory.md`:
  *"thinking style = one i4-32D vector... Kant / Schopenhauer = specific
  vectors"* — a lens is an OBJECT (StyleRecipe), never a rung (the
  persona-vs-rung fence holds: lenses are angles/styles, not the 34 NARS
  runbooks). The MUL compass already carries a Kant needle;
  `scientific-kg-substrate-v1.md` already names a Kant/Schopenhauer/Hegel
  validation gate. Wittgenstein · Nietzsche · Kant · Hegel = four specific
  vectors. Orientation (loose, not spec):
    · **Kant** — conditions of what can appear: the reader's frame — the
      deinterlaced pin + rung, the Vorurteil formalized (§8.6);
    · **Wittgenstein** — meaning-as-use: WHICH language-game the token is in
      (the grammar/FSM read of context);
    · **Nietzsche** — perspectivism: WHOSE drive is reading (the
      qualia/angle read);
    · **Hegel** — dialectic/becoming: the temporal stream itself; Aufhebung
      = the preserved contradiction CARRIED and later revised, never erased
      (§8.6's Epiphany path is already Hegelian by construction).
- **Each lens is a slit; each produces a ripple field — the shipped Ψ
  shape.** `ShaderResonance` is literally documented as *"ripple field:
  per-row energy + top-k hits."* One ambiguous event → four Ψ fields over
  the same candidate basins.
- **Interference is the readout.** Superpose the four fields:
  **constructive peaks** = readings supported from genuinely independent
  angles (multi-lens support, the diversity-catches-what-redundancy-cannot
  principle); **destructive nodes** = contradiction sites → §8.6's
  preserved-contradiction path, not averaged away. Collapse happens only at
  **Commit** — the measurement, after the interference pattern has formed,
  never before.
- **Independence is a MEASURED requirement, not an aspiration.** The
  cloned-lane probe (EPIPHANIES, the Pearl audit) showed non-independent
  witnesses produce +94% naive agreement at similarity 1.000000 — a hidden
  common cause faking consensus. A cloned lens adds NOTHING: the Doppelspalt
  only shows a pattern when the slits are genuinely separate. Deinterlacing
  (§8.3) + four genuinely distinct lens priors are what make the
  interference informative.

**[CONJECTURE] with its falsifier PAIR stated before any type lands:**
- CAN IT FIRE — on held-out ambiguous tokens, four-lens interference peaks
  must out-predict the best single lens at picking the reading the wider
  context later confirms;
- CAN IT STAY SILENT — the **cloned-lens control**: replace the four lenses
  with one lens cloned 4×; the gain must VANISH. If it does not, the
  "interference" was a hidden common cause, exactly the defect the
  cloned-lane probe measured — and the lens set carries as much information
  as one lens.

### 8.8 First and foremost — the deterministic grammar floor (DeepNSM-v2)

**Priority ruling (operator): §8.6's fusion and §8.7's lens interference are
for the RESIDUE. The floor runs first, and it is deterministic grammar.**
This is DeepNSM's founding thesis applied to the whole §8 stack: the FSM
resolves the bulk mechanically; only what grammar cannot close escalates
upward. Plan-only note — the deepnsm-v2 CRATE is not to be touched from this
arc; its wiring is already ahead of what outside sessions assume.

- **24 deterministic i4 anaphora pointers.** An i4's range is **−8..+7 —
  exactly the sentence window** deepnsm-v2 already reads. A Relativpronomen
  anaphora pointer is therefore ONE i4: a deterministic relative offset to
  its antecedent, no search, no scoring. Shipped seed: `wave.rs`'s
  `CausalWitnessFacet` (antecedent / kausal / grounding offsets) fed by
  `fsm.rs`'s `Rel` tag ("the relativizer's antecedent IS the matrix
  subject"). And the arithmetic is not a coincidence to wave away:
  **24 × i4 = 96 bits = 12 bytes = the V3 facet payload.** [CONJECTURE with
  a gate: a 24×i4 carving is a NEW projection of the content-blind 12-B
  register — it must be SANCTIONED by the ClassView per le-contract §3
  (which today carves 6×(u8:u8) / 4×(u8:u8:u8) / 3×(u8:u8:u8:u8)), never
  assumed. i4 is a native grain elsewhere (QualiaI4Column), so the ask is
  a sanctioning, not an invention.]
- **Verb heuristics: Valenz + Tempus, both deterministic.**
  *Transitiv/intransitiv* (valence) tells the FSM whether `HaveVerb` should
  expect an object at all — the FSM already closes an intransitive embedded
  clause without a triple; valence lifts that from special case to rule.
  *Plusquamperfekt / Perfekt / Futur I / Futur II* are **frame stamps for
  the deinterlacer (§8.3), supplied by morphology for free**: Plusquamperfekt
  marks an event BEFORE the narrative now; Futur II is a future reference
  point looking BACK (a nested frame). Tense is the linguistic version-
  stamp — grammar hands temporal deinterlacing its ordering hints
  deterministically, before any inference runs.
- **The SPO 2³ rung decomposition ladder.** A triple has 8 fill-states
  (which of S/P/O are bound); the convention rung ↔ pearl-level ↔ 2³ mask is
  ALREADY CERTIFIED in the contract
  (`rung_pearl_levels_and_masks_follow_the_certified_convention`,
  `cognitive_shader.rs`). Unbound slots are exactly §8.5's NaN cells — the
  autocomplete settle fills them from the bound ones.
- **The 34 recipes dispatch BY RUNG, 0–9, deterministically.**
  `RungLevel` spans the ladder (0 = Surface … clamping at Transcendent);
  `recipes::RECIPES: [Recipe; 34]` carries the rung-3 runbooks with SPO-2³
  coverage; the rung arrives WITH the read (`QueryReference::at(v, rung)`).
  So which tactic fires is a function of `(rung, 2³ mask)` — a certified
  convention, not a learned chooser (the ladder doc's own line: a macro
  choosing which tactic fires **does not exist**, and §8 does not create
  one).

**The full resolution order, restated as the ladder it is:**

```
1. grammar floor      pointers (24×i4) · Valenz · Tempus-frame-stamps · FSM
2. 2³/rung dispatch   bound-slot mask + rung level → one of the 34 recipes
3. fusion             §8.6 — Revision over deinterlaced frames (residue only)
4. lens interference  §8.7 — four slits, for what fusion cannot settle
5. tail               FailureTicket / LLM — the <25%, last, never first
```

Each rung up the ladder is strictly more expensive and strictly rarer —
the floor is where the volume lives, which is why it comes first and
foremost.

*Provenance: operator rulings in-session; §8.3/§8.4 corrected same day by
operator review — the axis labels (deinterlacing / splat hydration) and the
removal of the bundle/XOR fusion model (VSA demoted per
E-MARKOV-TEMPORAL-STREAM-1; the doc initially reimported it — an
inconsistency the operator caught, recorded here per corrections-cite-their-
pass). Original rulings: (domino = pay-forward torque; = NNUE
64×64 Stockfish wiring; keeps its place as attention headers modulated by the
Morton cascade inverse-pyramid perturbation shader, other than the Pythagorean
comma; muscle memory is the dynamic, weights are bolted on). A tesseract-side
probe conclusion judging domino by retention ("no bounded-hold regime") is
WITHDRAWN as a claim about domino — wrong axis; its echo-state and
quantization measurements stand on their own.*
