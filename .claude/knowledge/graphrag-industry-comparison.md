# GraphRAG Industry Comparison — our stack vs the field (representation × methods)

> **READ BY:** integration-lead, truth-architect, convergence-architect,
> workspace-primer — and any session positioning lance-graph/AriGraph against
> external GraphRAG / graph-DB / causality frameworks.
>
> **Status:** synthesis (2026-07-18). Industry rows = web-verified (sources
> §5, 2025–2026 tips). "Ours" rows = file:line-verified against
> `crates/lance-graph/src/graph/arigraph/*` + `crates/causal-edge` +
> `crates/thinking-engine` + `crates/cognitive-shader-driver`. Companions:
> `graphrag-representations-inventory.md` (7 papers × substrate) +
> `graphrag-rs-inventory.md` (the automataIA fit inventory).

---

## 0 — The reference stack (what everything is compared against)

lance-graph/AriGraph is the **no-LLM, exact-distance, SoA-tenant** inversion of
mainstream GraphRAG:

- **Extraction** — DeepNSM 6-state PoS FSM (4096-word COCA, 512-bit VSA), **no
  transformer inference** (<10 µs/sentence), vs LLM entity/relation extraction.
- **Similarity** — `palette256²` **exact u8 distance table** (Fisher-z ρ≥0.999,
  one table read), vs float vector embeddings + ANN.
- **Graph engine** — GraphBLAS `blasgraph` with **7 HDR semirings** over
  CSR/CSC/COO/HyperCSR, vs single-semiring / property-graph traversal.
- **Reasoning** — NARS truth-revision + Pearl do-operator on the `TripletGraph`,
  vs LLM-at-query-time.
- **Memory** — SoA tenants (one 512-B row, `key16|edges16|value480`), **computed
  once, never re-transformed**, vs re-indexed parquet / managed vector store.

## 1 — The field (representation × methods × LLM-dependency)

| Framework | Representation | Key methods / pipeline | LLM (index / query) | Upstream status | Maps to ours |
|---|---|---|---|---|---|
| **Standard RAG** | chunks + float embeddings | chunk → embed → vector top-k → stuff → generate | — / gen | baseline | the thing we replace whole |
| **MS GraphRAG** | LLM-extracted KG + Leiden communities + **LLM community reports** + embeddings | extract entities/rels → Leiden → **LLM report per community** → embed; **global** (map-reduce over reports) / **local** / DRIFT / LazyGraphRAG | **heavy / heavy** | active | same graph→community skeleton; we replace LLM extraction (DeepNSM) + LLM reports (**implicit summaries**, §3) + embeddings (palette256²) |
| **MS Semantic Kernel** | none (orchestration SDK) | plugins + planners + function-calling + BYO vector connectors | orchestration | **superseded** → Microsoft Agent Framework (AutoGen merge, GA 2026) | analog of our `OrchestrationBridge`/`UnifiedStep` + thinking-style planner — **not** the retrieval analog |
| **Google Vertex/Gemini** | managed float embeddings + Google-Search grounding + long-context | RAG Engine (import→chunk→embed→retrieve→generate), grounding, high-fidelity cited answers | embed / Gemini | active | maximal opposite: cloud, float, LLM-gen. Long-context is a **complement** to RAG (Google's own positioning), not a substitute |
| **RedisGraph** | property graph as **sparse adjacency matrices** (GraphBLAS, CSR+CSC) | Cypher via `GRAPH.QUERY`; traversal = **matrix multiplication** over GraphBLAS | none | **EOL 2025-01-31** (→ FalkorDB) | **direct ancestor of `blasgraph`** — same GraphBLAS idea, but single Boolean semiring; ours has 7 HDR semirings and is alive |
| **KuzuDB** | embedded **columnar** property graph, CSR adjacency, built-in vector/FTS | Cypher; vectorized exec + **factorized query** + worst-case-optimal joins | none | **archived 2025-10-10** (Apple) | the explicit model for our `adjacency/` column-grouped CSR/CSC + batch intersection — embedded *inside* the cognitive stack, and alive |
| **MIT causality (Uhler et al.)** | latent **causal-variable DAG** | causal **disentanglement from observational data** (Jacobian-variance pruning, bottom-up); identifiability guarantees (NeurIPS 2024) | — | research | **upstream complement, not competitor** — it *discovers* structure (Rung-1); `CausalEdge64` *carries + NARS-revises* already-typed edges. Consume, don't rebuild |
| **graphrag-rs** (automataIA) | KG + float embeddings + text units (Rust) | `GraphRAG::{new,builder}`, `add_document`, `ask`/`query`/`ask_explained`; PageRank/Leiden/BM25/LightRAG/ROGRAG retrieval; Candle/Ollama backends | **heavy / heavy** | external repo (inventoried, **not vendored**; `LanceDBStore` is a `NotImplemented` stub) | closest public *peer* (a Rust GraphRAG) — but LLM+float; we are its no-LLM inversion. REUSE-as-reference only |
| **PersonalAI** (2506.17001) | hyper-graph: **object + thesis + episodic** vertices; object/thesis-hyper/episodic-hyper edges | LLM "Memorize" (extract triples → thesis+episodic hyper-edges) → 4 retrievers (**A\*, WaterCircles, BeamSearch, Mixed**) → LLM **Knowledge-Update** prune | **heavy / heavy** | research (AriGraph successor) | closest conceptual sibling; its hyper-edges ↔ our HyperCSR + episodic-witness; its LLM update ↔ our NARS revision. §2 diff |
| **Ours (lance-graph/AriGraph)** | `TripletGraph` SPO + episodic memory + `palette256²` facets + SoA tenants | FSM extract; `rrf`/`ppr`/`bm25`/`communities`/`episodic_search`; `infer_deductions`/`detect_contradictions`/`revise_with_evidence`/`intervene_on` (Pearl do) | **none / none** (LLM tail only) | active | — |

## 2 — Method diff: graphrag-rs vs PersonalAI vs Ours

| Operation | graphrag-rs | PersonalAI | **Ours (file:line)** |
|---|---|---|---|
| Extraction | LLM / GLiNER / pattern | LLM triple extraction | **DeepNSM FSM (no LLM)** → `TripletGraph::add_triplets` (`triplet_graph.rs:106`) |
| Higher-order unit | — | **thesis hyper-edge** (atomic thought) | `EpisodeTheses` (`episodic.rs:120`, `.theses()` `:384`) — the thesis analog, derived not LLM-built |
| Episodic link | text units | **episodic hyper-edge** (passage → all its vertices) | `EpisodicBasins` (`episodic.rs:79`) + `EpisodicEdges64` (family:local) + `SpoFacet` 3+3 register |
| Community | Leiden | — | `TripletGraph::communities()` (`community.rs:108`) — **no LLM summary** (§3) |
| Retrieval | vector/BM25/PageRank/LightRAG/ROGRAG | **A\* / WaterCircles / BeamSearch / Mixed** | `OsintRetriever::retrieve` (`retrieval.rs:235`) + `rrf` + `ppr` + `associated_paths` (`triplet_graph.rs:208`); BeamSearch = an unbuilt candidate (my inventory §7) |
| Fusion | hybrid | union of retrievers | **RRF** `reciprocal_rank_fusion` (`rrf.rs:64`, k=60) — the keystone |
| Reasoning | — (LLM at gen) | — (LLM at gen) | **`infer_deductions`** (`:830`), **`detect_contradictions`** (`:880`), **`intervene_on`** Pearl do-operator (`:789`, `ContextTag::INTERVENTION_RAW_G`) |
| Update/revision | re-index | **LLM Knowledge-Update prune** | **`revise_with_evidence`** (`:904`) — NARS truth-revision, no LLM; `WitnessCorpus::evict_stale_before` (`:392`) |
| Memory | in-mem / LanceDB(stub) / Qdrant | LLM-maintained KG | **SoA tenants** (one 512-B row, never re-transformed) + `WitnessCorpus` provenance |

**The spine of the diff:** graphrag-rs and PersonalAI are the two closest peers, and
both are **LLM-maintained at index AND query time**. We do the same *operations*
(triples, communities, multi-hop, thesis/episodic grouping, prune) with **zero LLM
in the hot path** — FSM extraction, exact palette distance, NARS revision.

## 3 — The four headlines

**H1 — Implicit summaries.** Microsoft GraphRAG's costliest, lossiest step is the
**LLM community report** (an LLM writes prose per Leiden community; global search
map-reduces over prose). The **horizontal 6×(8:8)** reading (`basin:relationtype`,
le-contract §3; inventory §2) makes that summary **implicit and free**: a node's
`basin` byte *is* its community membership, the horizontal `relationtype` pairs
*are* its relational profile — "summarize the community" collapses to a
**palette256² table read**. Microsoft *generates* the summary with an LLM at
index-time; we *address* it with a byte at query-time. This is why our
`communities()` has **no** summarization pass by design.

**H2 — We are the living union of two dead graph DBs.** RedisGraph (GraphBLAS
sparse-adjacency, **EOL 2025-01-31**) + KuzuDB (columnar CSR, **archived
2025-10-10**) are the two closest architectural analogs — and both are abandoned
upstream. `blasgraph` continues RedisGraph's GraphBLAS design with **7 HDR
semirings** (vs one Boolean); `adjacency/` continues Kuzu's column-grouped CSR/CSC.
The positioning is not "us vs them" — it's **us = RedisGraph's algebra +
Kuzu's columns + GraphRAG's operators, over a no-LLM exact-distance substrate.**

**H3 — MIT causality is upstream, not a competitor.** Uhler et al.'s observational
causal disentanglement (NeurIPS 2024) *discovers* a latent causal DAG (Rung-1,
Jacobian-variance pruning). Our `CausalEdge64` + `intervene_on` (Pearl do-operator)
*carry and revise* already-typed edges (Rung-2/3). Their output is our input — a
**consume-don't-rebuild** seam for structure discovery, not overlap.

**H4 — Reasoning is the V3 substrate, not `CausalEdge64`.** The one-liner "NARS =
CausalEdge64" is stale. Awareness-cramming into the 64-bit register is **let go**
(M20; le-contract §3), empirically shown oversized (D-MTS-6 GREEN: k\*=1, 2 bits/edge
matches all three awareness proxies vs baseline-16). Reasoning now = the **thinking-
engine** running it (`cognitive_stack`, `qualia`, `think`), the **cognitive-shader-
driver** dispatching it over `MailboxSoA`, **p64** converging it (call-site shim,
zero-dep), and the **P4 ancestry pipeline** wiring awareness into **facet tenants**
(the autopoiesis triangle + `SpoFacet`), not a wider edge. `CausalEdge64` is
**demoted to three concrete survivors** (SoA baton edge, perturbation baseline, p64
address), not deleted. See `.claude/v3/*` (this session's gem update).

## 4 — Honesty flags (carried, not hidden)

- Gemini long-context = **complement** to RAG per Google's own positioning (my
  earlier "substitute" framing was wrong); exact 1M/2M token number not re-pinned.
- graphrag-rs perf figures (6000× token reduction, 27× PageRank) are **self-reported**
  README/Medium claims, not independently verified.
- "MIT causality learning" has **no single canonical proposal** — Uhler
  observational-disentanglement (NeurIPS 2024) is the best graph-causality fit;
  soft-interventions sibling (NeurIPS 2023) is the interventional alternate.
- PersonalAI is **June 2025** (`2506` = 2025-06), not 2026; its update step is
  discrete/LLM-triggered, not "continuous."
- RedisGraph + KuzuDB are **discontinued upstream** — relevant precisely because
  they are our two closest living analogs.

## 5 — Sources

MS GraphRAG (research.microsoft.com Local-to-Global, DRIFT, Lazy; microsoft.github.io/graphrag outputs) ·
Semantic Kernel → MAF (github.com/microsoft/semantic-kernel; Agent Framework announce) ·
Vertex AI RAG + grounding (cloud.google.com/blog) ·
RedisGraph EOL (redis.io/blog/redisgraph-eol; FalkorDB migration) ·
KuzuDB (github.com/kuzudb/kuzu releases; theregister.com 2025-10-14) ·
MIT causality (news.mit.edu 2024-11-07; arXiv 2307.06250) ·
graphrag-rs (github.com/automataIA/graphrag-rs) ·
PersonalAI (arXiv 2506.17001) + AriGraph (2407.04363).
