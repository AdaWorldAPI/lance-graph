# graphrag + doc-retrieval on the V3 SoA — integration plan v1

> **Status:** DESIGN — evaluation + wave plan (no bytes land here). Aligned to
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

| Candidate crate | Verdict | Why |
|---|---|---|
| lance-graph doc-transcode / perceptual-IR / `document`-mint crate | **REJECT** | Duplicates OGAR `ogar-from-docv1`/`ogar-doc-ir`/`ogar-doc`; inverts the dep; WRONG-SHELF. Doc ingestion is assembler-side. |
| Monolithic `graphrag` (extraction→graph→retrieval from scratch, à la graphrag-rs) | **REJECT** | Extraction, SPO edges, fact store, vector/CAM-PQ retrieval already EXIST (§2). Violates *wire-don't-invent*. graphrag-rs = REUSE-AS-REFERENCE only (`graphrag-rs-inventory.md`); its `LanceDBStore` is a stub, its `InferenceEngine` is the no-singleton anti-exhibit. |
| **`crates/graphrag`** — thin retrieval **orchestrator**, query-side only | **BUILD** | Binds existing legs onto the #708 rung-ascent loop; BUILDS only the 3 genuine gaps (§2). Reads calcified substrate; never ingests. |
| **`DocGraphQuery` trait** in `lance-graph-contract` + a `DocGraphView` read projection | **BUILD (thin)** | The shared surface both graphrag and `ogar-doc` consume. Keeps OGAR on the contract, not the impl. |
| New lance-graph OGAR-seam crate | **REJECT** | `lance-graph-ogar` already IS the seam (re-exports ogar-vocab Class/codebook + `ClassView` impl). Extend it if needed. |

**Net: two BUILD targets** — `crates/graphrag` (the orchestrator + the 3 gaps) and
a thin `DocGraphQuery` contract surface. Everything else is wiring.

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

## §4. Crate topology (recommended)

```
crates/graphrag/                 (NEW, lance-graph, W5 query consumer)
  src/
    lib.rs
    retrieve.rs      # rung-ascent orchestrator; consumes contract RungElevator/RungLevel (#708); the load-bearing walk
    community.rs     # BUILD: hierarchical Leiden over CE64/SPO-G
    ppr.rs           # BUILD: HippoRAG reset-distribution PPR (atop hdr_pagerank/spmv)
    keyword.rs       # BUILD: BM25
    summarize.rs     # community summaries via Rig oracle → compiled template (W3 dep)
  deps: lance-graph-contract (RungElevator/RungLevel/cognitive_shader — #708),
        causal-edge, cognitive-shader-driver (the rung_widened_layer_mask widen),
        lance-graph (arigraph query surface), bgz-tensor/cam_pq (vector), ndarray.
        NO ingest; NO OGAR dep; reads calcified substrate.

lance-graph-contract/            (EXTEND)
  src/doc_graph.rs   # DocGraphQuery trait + DocGraphView read projection (the shared surface)

lance-graph-ogar/                (EXISTING seam — extend only if the trait needs OGAR ClassView glue)
```

`ogar-doc` (OGAR) consumes `DocGraphQuery` **via the contract** — that is the
correct "feeds ogar-doc": ogar-doc is a **caller** of graphrag's query surface,
never fed data by a lance-graph crate.

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
  claim (v3 standing gate).

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
