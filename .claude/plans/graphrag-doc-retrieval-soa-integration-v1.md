# graphrag + doc-retrieval on the V3 SoA — integration plan v1

> **Status:** DESIGN — evaluation + wave plan (no bytes land here). **v1.1
> (2026-07-17): expand AriGraph in place — NO standalone `crates/graphrag` (§1).**
> Aligned to
> **#708 / D-TRI-6 — now MERGED (`8d3209c`; `E-RUNG-ASCENT-WIRED-1` SHIPPED)** +
> `E-MASLOW-PYRAMID-OF-COGNITION-1` and the `triangle-tenants-gestalt-separation-v1`
> Maslow pyramid. The rung-ascent loop is landed code (D-GR-2's dependency is
> satisfied). Any OGAR classid touch is **mint-gated** (batched, baton-audited).
> Probe-first: the **graph-load-bearing baseline** is the FIRST deliverable,
> before any Leiden/PPR code.
> **Grounding:** three Opus reader-agents (v3 waves+census · CausalEdge64/witness
> substrate · OGAR↔lance-graph boundary) + `.claude/knowledge/graphrag-rs-inventory.md`
> (`E-V3-GRAPHRAG-INV-1`, the pre-existing automataIA/graphrag-rs audit) + the
> automataIA repo survey (graphrag-rs ★519, graph-librarian-rs, wasm-typst-studio-rs,
> lodviz-rs). Citations are as-of `efd21d2` (#707) — verify before coding.

## §0. The reframe — direction correction (READ FIRST)

The task said "create crates **inside lance-graph feeding into** ogar-doc." Run
through `assembler-vs-storage-substrate.md` + `compilation-vs-runtime-substrate.md`,
that **inverts the dependency** and is WRONG-SHELF. The corrected direction:

- **OGAR = assembler.** It OWNS document ingestion, the `document 0x080B` mint,
  and persistence — already built as `ogar-from-docv1` → `ogar-doc-ir` →
  `ogar-doc` (doc-W4), all in `/home/user/OGAR/crates/`. A lance-graph
  doc-transcode crate would **duplicate `ogar-from-docv1`** and invert the dep.
- **lance-graph = spine.** It provides **contract types** (`ClassView`, facet /
  envelope, masks) that `ogar-doc` builds against — OGAR consumes
  `lance-graph-contract`, **never the reverse** — **plus query capability** over
  the *calcified* substrate.
- **Lance = calcification.** "There is no ingest API." The SoA/Lance write is
  calcification by the batch writer on-behalf of a mailbox owner.
- **The door-knocking test forbids** any build step or typed-value constructor
  asking the running substrate for permission. `doc.v1 → ogar_doc_ir` is a plain
  Rust fn, unit-testable with nothing running.

**Therefore the honest shape of "feeds both ogar-doc and graphrag":** a single
**query/view surface** over the calcified document-graph, consumed by two
readers — `crates/graphrag` (retrieval) and OGAR `ogar-doc`'s
`reconstruct_document` ("re-issue with updated knowledge" / "documents in this
community"). `ogar-doc` calls it **via a `lance-graph-contract` trait**
(`DocGraphQuery`), so OGAR still depends only on the contract. graphrag provides
the impl. **Neither ingests; both read.**

## §1. Evaluation — what to build, what to reject

> **★ v1.1 REVISION (2026-07-17, operator-directed): NO `crates/graphrag`.**
> Expand **AriGraph** in place instead. AriGraph already owns `retrieval.rs`
> (OsintRetriever — BFS+episodic fusion), `witness_corpus.rs`, `episodic.rs`,
> `spo_bridge.rs`, `markov_soa.rs` — the retrieval brain + the episodic-witness
> basins are already there. A separate crate is a **free function on AriGraph's
> state** (the litmus-test reject) and a **parallel retrieval layer** beside an
> existing one. There is **no** community/basin/cluster/Leiden/partition type in
> AriGraph today (grep-confirmed) — so Leiden fills the one *structural-partition*
> gap that **complements** the episodic-witness partition AriGraph owns. This
> dissolves the §10 dep-weight feature-gate: the graph capabilities live where the
> graph lives (core). §4/§10 updated; the #708 rung-ascent alignment (§3), the
> reuse census (§2), the boundary (§0), and the probes (§6) all carry over.

| Candidate | Verdict | Why |
|---|---|---|
| lance-graph doc-transcode / perceptual-IR / `document`-mint crate | **REJECT** | Duplicates OGAR `ogar-from-docv1`/`ogar-doc-ir`/`ogar-doc`; inverts the dep; WRONG-SHELF. Doc ingestion is assembler-side. |
| Monolithic `graphrag` (extraction→graph→retrieval from scratch, à la graphrag-rs) | **REJECT** | Extraction, SPO edges, fact store, vector/CAM-PQ retrieval already EXIST (§2). graphrag-rs = REUSE-AS-REFERENCE only; its `LanceDBStore` is a stub, its `InferenceEngine` is the no-singleton anti-exhibit. |
| **`crates/graphrag`** — thin retrieval orchestrator crate | **REJECT (v1.1)** | Duplicates AriGraph's existing `retrieval.rs`; a free-function-on-carrier's-state (litmus reject); a new *layer* where the rule is new *method*. **Expand AriGraph instead.** |
| **Leiden community → `arigraph/community.rs`** (NEW module) | **BUILD** | The structural partition; complements `witness_corpus`/`episodic` basins. First partition type in AriGraph — fills the gap, doesn't clutter. |
| **PPR / HippoRAG → extend `arigraph/retrieval.rs`** | **BUILD** | A graph-ranking sibling of the existing `find_path`/`get_associated`/BFS+episodic fusion; community = a third fusion signal. Backed by in-core `blasgraph::hdr_pagerank`. |
| **`DocGraphQuery` trait** in `lance-graph-contract` (impl = AriGraph methods) | **BUILD (thin)** | The zero-dep read surface `ogar-doc` consumes; keeps OGAR on the contract, not the impl. |
| BM25 / lexical leg | **BUILD (small, OUT of AriGraph)** | Text-index, not a graph op; a tiny module elsewhere (or reuse existing). |
| New lance-graph OGAR-seam crate | **REJECT** | `lance-graph-ogar` already IS the seam. Extend if needed. |

**Net (v1.1): expand AriGraph** — `arigraph/community.rs` (Leiden) + PPR into
`arigraph/retrieval.rs`, complementing the witness/episodic basins it already
owns; a thin `DocGraphQuery` contract trait (impl = those methods); a small
out-of-graph BM25 leg. **No new crate.** Everything else is wiring.

## §2. Substrate reuse — the census (build only the gaps)

REUSE (do **not** rebuild — the substrate provides these):

- **SPO edge = `causal_edge::CausalEdge64`** (8 B, `#[repr(transparent)]`). v2
  layout (default `causal-edge-v2-layout`): S/P/O palette idx (bits 0-23), NARS
  **frequency**/**confidence** u8 (24-39), Pearl-2³ causal mask (40-42),
  direction triad (43-45), inference mantissa i4 (46-49), plasticity (50-52),
  **W-slot witness handle** (53-58), truth-band lens (59-60), spare (61-63).
  **Disambiguation hazard:** two `CausalEdge64` exist — use `causal_edge` (SPO
  palette), NOT `thinking_engine::layered` (8×u8 strength vector).
- **NARS truth = native `(freq u8, conf u8)`** in the edge (bits 24-39);
  `expectation()`, `evidence_weight()` on `CausalEdge64`. Don't add a
  confidence/weight field.
- **Provenance/witness.** `EpisodicWitness64` is **not a symbol** — it names a
  *queued* SoA column (`soa_view.rs` `episodic_witness()` deferred). Today use
  `EpisodicEdges64` (4×16-bit MRU episodic edges) + `WitnessTable<64>` /
  `WitnessEntry{mailbox_ref, spo_fact_ref}` resolving the CE64 v2 W-slot. Don't
  add a provenance field; **materializing the column is a candidate probe**, not
  an assumption.
- **Fact store = AriGraph `TripletGraph` + SPO-G quad.** `Triplet` + G-slot
  (`ContextTag::{Observation, Intervention}`) = the quad; `CounterfactualSpoG`
  from `TripletGraph::intervene_on` (Pearl rung-2). Promote a CE64 to a graph
  edge via `spo_bridge::promote_to_spo` past the truth gate. Cold, in
  lance-graph core — **not** a SoA column.
- **Read surface = `MailboxSoaView`** (zero-copy; `edges:[CausalEdge64;N]` is the
  EdgeColumn). Don't re-encode to Arrow. Node adjacency = the 16-byte `EdgeBlock`
  (coarse, key-side); identity/class = `NodeGuid` / `class_id` u16 (resolves via
  `lance-graph-ontology`). Three edge encodings — EdgeBlock (adjacency) / CE64
  (causal weight) / EpisodicEdges64+WitnessTable (provenance) — **never conflate**.
- **Extraction (upstream proposer) = `lance-graph-arm-discovery`** (Aerial+
  transcode, integer codebook-distance oracle → `{s,p,o,f,c}` ndjson NARS-truth
  SPO candidates) + `nsm/parser.rs` PoS-FSM→SPO. No-LLM, workspace-native.
- **Vector retrieval = CAM-PQ** (`cam_pq/{ivf,storage,udf}`; codec ICC 0.9999,
  `cam-pq-production-wiring-v1`). *(The exact vector-search entry symbol moved
  since the agent read — verify the current entry point.)*

BUILD (the genuine gaps — each gated on §6 P0):

1. **Hierarchical Leiden community detection** — in-tree is example-only
   (`jc/examples/splat_louvain_modularity.rs`, `splat_lpa_label_propagation.rs`);
   graphrag-rs `leiden.rs` is single-level, REUSE-AS-REFERENCE. Build hierarchical
   over the CE64/SPO-G graph.
2. **HippoRAG-PPR** (personalized PageRank with reset distribution) — plain
   `hdr_pagerank` (`blasgraph/ops.rs`) + `ScentCsr::spmv` exist; the reset/seed
   distribution + dual-signal (passage-weight) is the gap. Reference
   graphrag-rs `hipporag_ppr.rs`.
3. **BM25 keyword arm** — ABSENT; small pure function. Reference graphrag-rs
   `KeywordExtractor`.

## §3. The #708 alignment — retrieval IS rung ascent

**Do not build a bespoke retrieval-escalation ladder.** graphrag retrieval
dispatches through the **`RungElevator`** wired by **#708 (merged `8d3209c`,
D-TRI-6; `E-RUNG-ASCENT-WIRED-1`)** — now landed code, not a landing dependency.
The elevator + ladder type are contract types:
`lance_graph_contract::cognitive_shader::{RungElevator, RungLevel}`
(`cognitive_shader.rs:272` / `:157`), and the widen is
`cognitive-shader-driver::driver::rung_widened_layer_mask`
(`driver.rs:701` — `fn(base, level, req_mask: u8) -> u8`). Per #708: `on_gate()`
advances one rung per dispatch; the current rung selects the cycle's cascade
breadth via `rung_widened_layer_mask` — a UNION over the **8-bit predicate-plane
mask (CAUSES..BECOMES)**, **identity at base (zero regression), superset-monotone
above**. `RungLevel` is now canonical in `lance-graph-contract` (post-#708 dedup;
thinking-engine re-exports; `as_u8()` added). The follow-up review fix `17368ea`
locks the ordering — **advance the elevator BEFORE the sinks** — which graphrag's
`retrieve.rs` must mirror (gate the walk on the *post-advance* rung). **BLOCK
ascends, FLOW relaxes to base.**

Map retrieval onto the Maslow pyramid (`triangle-tenants-gestalt-separation-v1`
§3a; the rung-content ladder in `persona-vs-rung-ladder.md`):

| Rung | Pyramid level | Retrieval action | Predicate-plane mask |
|---|---|---|---|
| 0–1 (base, FLOW) | observation / gestalt | CAM-PQ vector + BM25 surface lookup | identity |
| 2 | SPO 2³ | SPO-G edge hop over `CausalEdge64` | widen (Pearl rung-1 planes) |
| 3 | CE64 NARS candidates | HippoRAG-PPR + community-scoped expansion, NARS-truth-weighted | wider CAUSES..BECOMES union |
| 3–4 | NARS candidate design / revision | community summaries (Rig oracle → compiled template, W3) | full |
| 4 (apex) | counterfactual | Pearl rung-2 intervention (`ContextTag` G-slot) | — |

**The graph is load-bearing precisely because BLOCK ascends the elevator**
(retrieval surprise / low NARS confidence / contradiction widens the traversal).
graphrag never re-decides the level — it reads the driver's `RungLevel` and
supplies the wider graph walk. This is the anti-decorative-graph guarantee (vs
the graph-librarian-rs anti-pattern where the graph is built but never traversed).

## §4. Topology (v1.1 — expand AriGraph, no new crate)

```
crates/lance-graph/src/graph/arigraph/     (EXPAND — the graph owns its own structure)
  community.rs   # NEW: hierarchical Leiden over the TripletGraph adjacency
                 #      (triplets + entity_index; backed by in-core blasgraph CSR).
                 #      The structural partition beside witness_corpus/episodic basins.
  retrieval.rs   # EXTEND OsintRetriever: + PPR/HippoRAG (reset-distribution atop
                 #      blasgraph::hdr_pagerank), + community as a THIRD fusion signal
                 #      (beside BFS + episodic). Driven by the #708 RungElevator:
                 #      detect_contradictions → BLOCK → wider community/PPR walk.
  # reuse in place: triplet_graph.rs (get_associated/intervene_on/detect_contradictions/
  #   revise_with_evidence), spo_bridge.rs (promote_to_spo), witness_corpus.rs,
  #   episodic.rs, markov_soa.rs. NO new Edge/Provenance/Graph type.

crates/lance-graph-contract/               (EXTEND, zero-dep)
  src/doc_graph.rs   # DocGraphQuery trait (impl = AriGraph's retrieval methods) — the
                     #   read surface ogar-doc consumes. NO DocGraphView duplicate carrier.

<small out-of-graph leg>                   (BM25 lexical — NOT in AriGraph)
  a tiny keyword module (or reuse an existing text index); text-index ≠ fact graph.
```

**Dep note (v1.1 — the feature-gate is DISSOLVED):** the community/PPR code lives
in `arigraph/` = lance-graph **core**, which already has the graph + datafusion/
lance/arrow. There is no separate light crate to gate — the graph capabilities
live where the graph lives. The only zero-dep additions are the `DocGraphQuery`
trait (contract) and the BM25 leg. The `RungElevator` (contract, #708) is reached
from `retrieval.rs` in-core; `rung_widened_layer_mask` (still private,
`driver.rs:701`) is either made pub / moved to the contract, or replicated (§10).

`ogar-doc` (OGAR) consumes `DocGraphQuery` **via the contract** — the correct
"feeds ogar-doc": ogar-doc is a **caller** of AriGraph's query surface (through
the contract trait), never fed data by a lance-graph crate.

## §5. Wave sequencing (aligned to v3 W0–W6 + D-TRI)

Neither crate is a new W-wave. Sequenced **after the W1 keystone**
(`mailbox_owner()` shipped #631; batch-writer W1b in-PR):

- **G0 — baseline probe (FIRST, no code): P-GRAPH-LOADBEARING** (§6). Gate on
  truth-architect / measurement-before-synthesis.
- **D-GR-1** — `DocGraphQuery` trait + `DocGraphView` read projection in the
  contract (both consumers build against it). Zero SoA writes.
- **D-GR-2** — `crates/graphrag` scaffold + `retrieve.rs` binding **existing**
  CAM-PQ + SPO-G hops onto the canonical `RungLevel` / `RungElevator` (**#708
  merged `8d3209c` — dependency satisfied**; advance-before-sinks ordering per
  `17368ea`). Mirrors the #708 settlement probe (BLOCK ascends → wider walk;
  FLOW at base).
- **D-GR-3** — BUILD the 3 gaps (`community.rs` Leiden, `ppr.rs` HippoRAG,
  `keyword.rs` BM25), **each individually gated** on G0 showing it beats
  vector-only.
- **D-GR-4** — `summarize.rs` community summaries via the Rig oracle → compiled
  template (depends on W3 `template-runtime`).
- **D-GR-5** — wire `ogar-doc` `reconstruct_document` + "similar/related
  documents" to `DocGraphQuery` (cross-repo; baton-audited classid handshake).

Depends-on: W1b (any persisted result is born-stamped), W2 (retrieval cycles =
kanban lanes if a cycle persists), W3 (summaries = compiled templates). Composes
**atop** `oxigraph-arigraph-cognitive-shader-soa-merge-v1` (that plan builds the
unified SoA context; graphrag is the retrieval layer over it — §8).

## §6. Falsifiers / probes (probe-first)

- **P-GRAPH-LOADBEARING (G0, the gate).** On a document corpus, measure retrieval
  quality **with vs without** graph traversal (vector-only vs vector+SPO-G+PPR).
  KILL condition: if the graph does not beat vector-only on multi-hop/global
  questions, do **not** build Leiden/PPR — the graph would be decorative
  (graph-librarian-rs anti-pattern). Mirrors the 1BRC addressing-tax probe.
- **P-RUNG-RETRIEVAL.** Hard/contradictory query → BLOCK → elevator ascends →
  wider CAUSES..BECOMES mask → higher recall; easy query → FLOW → stays at base
  (identity mask, cheap). Mirrors #708's D-TRI-6 settlement probe.
- **P-PPR-MULTIHOP.** HippoRAG-PPR beats vector-only on 2+-hop questions.
- **P-COMMUNITY-GLOBAL.** Leiden community summaries beat flat chunk-concat on
  global/thematic questions (the LightRAG dual-level claim).
- **jc battery** (ICC / Spearman / Cronbach) before any new lane-reading backs a
  claim (v3 standing gate). **Landed #709** as `jc::reliability::{pearson,
  spearman, cronbach_alpha, icc(IccForm)}` — every probe above computes its
  with-vs-without delta + significance through this crate, not a hand-rolled
  metric (and per `I-NOISE-FLOOR-JIRAK`, cites Jirak's weak-dependence rate for
  σ-claims).

## §7. Alignment hazards (and how this plan avoids each)

1. **write-on-behalf / born-stamped (W1 keystone).** graphrag is read-mostly; the
   only writes (persisted community membership / summaries) route the batch-writer
   `cast(on_behalf = envelope.mailbox_owner(), …)` — never write-as-self (the smb
   `LanceConnector::upsert` ORPHAN-WRITE is the cautionary tale). v3-mailbox-warden
   gate.
2. **No-singleton.** No global `Arc<RwLock<KnowledgeGraph>>`. Community/PPR operate
   over `MailboxSoA` rows + AriGraph; results are ephemeral or calcified
   per-mailbox. graphrag-rs `InferenceEngine` (graph-as-config-struct) is the
   explicit anti-exhibit.
3. **16-byte facet never-widen + classid canon-high.** Reuse existing task
   classids (`0x080B`/`0x0807`/`0x0808`/`0x080A`, hi-u16 = concept, canon-high). A
   persisted community-membership lane = a **new `ValueTenant` discriminant in the
   328 B headroom** (additive-at-end, `ENVELOPE_LAYOUT_VERSION` unchanged),
   field-isolation-matrix gated by v3-envelope-auditor. `u8:u8` never widened to
   u16/u24.
4. **Probe-first / wire-don't-invent.** COMPONENT-MAP + `graphrag-rs-inventory.md`
   ARE the precomputed "does it exist" search; §2 leans on them. G0 gates code.
5. **CausalEdge64 duplication + cross-repo baton.** Always qualify
   `causal_edge::CausalEdge64`. Any OGAR classid for the D-GR-5 seam is a
   **batched** mint + sync-fuse, baton-handoff-auditor gated.

## §8. Relationship to existing plans

- **`graphrag-rs-inventory.md`** (`E-V3-GRAPHRAG-INV-1`) — the component-by-component
  automataIA/graphrag-rs audit. **Mandatory pre-read**; this plan does not re-fork.
- **`oxigraph-arigraph-cognitive-shader-soa-merge-v1`** — builds the unified
  SoA context (Oxigraph RDF + AriGraph episodes + SPO). graphrag is the
  **retrieval layer atop it**; they compose, not compete.
- **`entropy-ladder-spo-rung-v1`** (R1 shipped) — SPO/NARS rung decomposition;
  the rung mapping in §3 must reconcile with its entropy-level = fact-position.
- **`temporal-markov-and-style-classes-v1`** (ACTIVE) — governs the retrieval
  substrate (episodic axis = Lance versions); graphrag rides `temporal.rs`.
- **`triangle-tenants-gestalt-separation-v1`** (D-TRI-1..6) + **#708** — the
  Maslow pyramid graphrag retrieval ascends (§3).
- **`normalized-entity-holy-grail-v1`**, **`episodic-risc-spine-v1`**,
  **`cam-pq-production-wiring-v1`**, **`wikidata-lazy-spine-hydration-v1`** —
  adjacent; entity-resolution + episodic-addressing + codec + KG-hydration inputs.

## §9. Board hygiene · D-ids · gate

- D-ids: **D-GR-1..5** (this plan) — STATUS_BOARD rows on land.
- New plan → INTEGRATION_PLANS.md prepend (this commit); EPIPHANIES entry only
  when a finding lands (probe result), not for the plan itself.
- **Gate:** G0 (P-GRAPH-LOADBEARING) is the first deliverable — **no Leiden/PPR
  code before it is green.** D-GR-5 (OGAR seam) is mint-gated + baton-audited.
- **Open questions:** (O1) does `ogar-doc` `reconstruct_document` genuinely need a
  runtime graph query, or is the "updated knowledge" re-issue a batch re-persist?
  (settles whether D-GR-5 is a live query seam or a re-run). (O2) materialize the
  `EpisodicWitness64` column now (helps PPR recency) or stay on
  `EpisodicEdges64`+`WitnessTable`? (probe). (O3) reconcile the §3 rung mapping
  with `entropy-ladder-spo-rung-v1`'s entropy levels — same ladder or two?

## §10. Feasibility — verified against the code (`8d3209c`, #708 merged)

Checked every reuse/build assumption against the tree. **Verdict: FEASIBLE.**
Load-bearing legs reach from **zero-dep** crates; two small caveats, both with
clean fixes; **no circular deps**.

**Confirmed reachable (pub, verified):**
- **Rung ascent (the #708 core) — zero upstream change.** `RungElevator` is a
  pub struct with **pub fields** + `const fn new(base)` + `on_gate(&mut self,
  GateDecision) -> RungLevel` (pure transition), all in **zero-dep**
  `lance-graph-contract::cognitive_shader` (`:272`). graphrag holds its own
  elevator; retrieval-surprise → `GateDecision` → `on_gate` → `RungLevel`.
- **SPO edge + NARS truth — zero-dep** (`causal_edge::CausalEdge64`).
- **Read surface — zero-dep** (`MailboxSoaView`, contract `soa_view.rs:42`).
- **Fact-store traversal — richer than assumed (closes the §9 arigraph
  question).** `TripletGraph` (arigraph) exposes, all pub: `get_associated(
  entities, steps)` (`:141`, the multi-hop SPO-G walk), `find_path` (`:193`),
  `intervene_on` (`:714`, Pearl-2 apex), `infer_deductions` (`:755`),
  **`detect_contradictions(conf)` (`:805` — the natural BLOCK trigger)**,
  `revise_with_evidence` (`:829`), `with_truth`. Plus `spo_bridge::promote_to_spo`
  (`:110`). The rung loop maps 1:1: `detect_contradictions` → BLOCK → `on_gate`
  ascends → `get_associated` widens the hop → `intervene_on` at the apex.
- **PPR base — pub** (`hdr_pagerank`, `blasgraph/ops.rs:275`; `ScentCsr::spmv`,
  `neighborhood/sparse.rs:98`). **CAM-PQ — pub** (`cam_pq/{ivf,storage}`).

**Caveat 1 — ~~dep-weight split~~ DISSOLVED by v1.1 (expand AriGraph).** Original
finding: the graph legs (`TripletGraph`, `spo_bridge`, `hdr_pagerank`, CAM-PQ)
live in lance-graph core (drags datafusion/lance/arrow), so a standalone crate
would need a light/heavy feature gate. **v1.1 removes the crate** — Leiden/PPR
land as `arigraph/` modules, which are already in core with the graph, so there
is nothing to gate. The only zero-dep additions are the `DocGraphQuery` contract
trait and the BM25 leg. **New design constraint (replaces the gate concern):**
`no-singleton` + `write-on-behalf` — community detection reads the graph the
mailbox owns and must NOT materialize a *global* partition singleton; a persisted
community-membership is a value-tenant lane, **born-stamped** via the batch
writer (v3-mailbox-warden gate). Confirm `TripletGraph`'s ownership model before
adding a persisting method.

**Caveat 2 — `rung_widened_layer_mask` is private** (`driver.rs:701`, bare `fn`)
— the predicate-plane widen can't be called directly. It is a SECONDARY leg (the
rung ASCENT works without it). Fix (small): promote it `pub` / move it to the
contract beside `RungElevator` (~1-line upstream tweak, baton to the
cognitive-shader-driver owner) — recommended; or replicate the pure
`(base, level, mask) -> u8`.

**Genuine BUILD (confirmed):** hierarchical Leiden — `jc`'s louvain is
**example-only** (nothing in `jc/src`), so `arigraph/community.rs` promotes it to
a lib over the `TripletGraph` adjacency; HippoRAG-PPR = reset-distribution atop
`hdr_pagerank`, added to `arigraph/retrieval.rs`; BM25 = a small out-of-graph fn.
All build on reachable primitives. AriGraph fit confirmed: it owns `retrieval.rs`
+ `witness_corpus.rs` + `episodic.rs` + `spo_bridge.rs` + `markov_soa.rs` today,
and has **no** community/partition type — Leiden fills the gap, does not clutter.

**Immediate next step (v1.1, no upstream change):** add `arigraph/community.rs`
(Leiden over `TripletGraph`) + a `cargo check -p lance-graph`; then extend
`arigraph/retrieval.rs` with PPR + community fusion under the #708 `RungElevator`;
then the G0 `P-GRAPH-LOADBEARING` probe (jc::reliability battery). No new crate.
