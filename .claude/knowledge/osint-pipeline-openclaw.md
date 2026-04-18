# OSINT Pipeline — The Openclaw-Style Open Source Intel Platform

> **Third substrate demo** alongside chess (`chess-database-reencoding.md`)
> and Wikidata (`wikidata-spo-nars-at-scale.md`). Where chess proves
> precision and Wikidata proves knowledge-graph scale, OSINT proves
> applied intelligence: harvesting the open web into a Palantir-class
> queryable graph on a laptop.
>
> Name rationale: openclaw is the open-source recreation of a proprietary
> platform game. An "OSINT openclaw" is the open-source recreation of a
> proprietary intelligence platform (Palantir Gotham, Maltego, Bellingcat's
> custom tooling). Same pattern: rebuild the closed system from public
> components, let anyone run it.

---

## The Existing Stack (Mostly Already Crates)

Everything in the inner pipeline is already a crate in the
`lance-graph` workspace. External integrations are limited to two
repos (`spider` for crawling, `aiwar` for the force-directed viz
frontend) and the `aiwar-neo4j-harvest` Cypher pattern library (which
is text files, not code).

| Component | Location | Status | Weights/Codebook |
|-----------|----------|--------|------------------|
| Crawler | `AdaWorldAPI/spider` | External repo, add as workspace dep | — |
| URL→text (optional alt) | `AdaWorldAPI/reader` | External Jina tool, optional | — |
| Reader-LM (Qwen2 1.5B) | `crates/reader-lm/` | **Internal crate (workspace-excluded = standalone)** | Qwen2 weights in Release if any |
| Reader-LM v3 (production) | `ndarray::hpc::jina::runtime::ModelSource::JinaV5` | **Internal in ndarray** | Jina v5 weights in Release |
| Semantic decomposition | `crates/deepnsm/` | **Internal crate (2200 LOC, 0 deps)** | COCA codebook from Release if any |
| Triplet store + episodic memory | `crates/lance-graph/src/graph/arigraph/` | **Embedded in lance-graph core (4696 LOC)** | — |
| OSINT pipeline shell | `crates/lance-graph-osint/` | **Internal crate (crawler + extractor + pipeline + reader)** | — |
| Cypher enrichment library | `AdaWorldAPI/aiwar-neo4j-harvest/cypher/` | External repo, 29 `.cypher` files | Text only |
| Force-directed viz | `AdaWorldAPI/aiwar` | External frontend (d3/three.js) | — |
| Theoretical frame | `AdaWorldAPI/war-test` | External (research paper impl) | — |
| Cockpit UI shell | `AdaWorldAPI/q2` | External, deployed at cubus.up.railway.app | — |

**What's internal already:** 95% of the inference + storage pipeline.
**What needs to be pulled in:** spider (crawler) and aiwar (viz
frontend). Both are one-line adds.

Codebook note: DeepNSM's 4096 COCA distance matrix, Reader-LM palette
weights, and Jina v5 model weights all follow the project's standard
pattern — released as GitHub Release assets, downloaded/mmapped by the
crate at load time. No weights in-tree.

---

## The Pipeline

```
[Targets: URLs, search queries, monitored feeds]
       │
       ▼
┌───────────────────────────────────────────────────────┐
│  spider (our fork)                                    │
│    HTTP fetch + respect robots.txt + rate limit       │
│    Google search scraping, JSON API pull              │
│    Outputs: raw HTML pages, JSON blobs, metadata      │
└────────────────────┬──────────────────────────────────┘
                     │ raw HTML / JSON
                     ▼
┌───────────────────────────────────────────────────────┐
│  Reader-LM PALETTE mode (bgz7 fingerprint, 17K tok/s) │
│    First pass: is this page relevant?                 │
│    Route high-signal pages to deep extraction         │
│    Drop low-signal pages at the door                  │
└────────────────────┬──────────────────────────────────┘
                     │ routed pages
                     ▼
┌───────────────────────────────────────────────────────┐
│  Reader-LM INFERENCE mode OR Jina v5 production       │
│    HTML → clean Markdown                              │
│    Boilerplate removal, article body extraction       │
└────────────────────┬──────────────────────────────────┘
                     │ clean text
                     ▼
┌───────────────────────────────────────────────────────┐
│  DeepNSM (decomposed for standalone use)              │
│    Text → SPO triples via 6-state PoS FSM             │
│    Entity words → 4096 COCA addresses                 │
│    Unknown words → 20K scientific extension or hash   │
│    Cross-reference to Wikidata palette (entity IDs)   │
└────────────────────┬──────────────────────────────────┘
                     │ SPO triples
                     ▼
┌───────────────────────────────────────────────────────┐
│  lance-graph-osint pipeline (already shipped)         │
│    crawler.rs: orchestrates spider                    │
│    reader.rs:  reader-lm / Jina v5 dispatch           │
│    extractor.rs: DeepNSM → triplets                   │
│    pipeline.rs:  flow orchestration                   │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────┐
│  AriGraph triplet_graph + episodic memory             │
│    Triplet.insert() with NARS truth                   │
│    Episode per source document, Hamming retrieval     │
│    BFS association for multi-hop OSINT queries        │
└────────────────────┬──────────────────────────────────┘
                     │ enriched graph
                     ▼
┌───────────────────────────────────────────────────────┐
│  aiwar-neo4j-harvest Cypher enrichment patterns       │
│    29 pre-authored investigative queries:             │
│      pattern_zero, operational_killchain,             │
│      surveillance_media, compliance_theater,          │
│      hemisphere_china, temporal_opacity, …            │
│    Composable; each pattern is one .cypher file       │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────┐
│  aiwar visualization (d3 + three.js)                  │
│    Force-directed graph layout                        │
│    Timeline axis for temporal patterns                │
│    Entity cards, relationship weights                 │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
[q2 cockpit at cubus.up.railway.app]
   Same UI that serves chess + Wikidata demos.
   OSINT mode selector; same contract, different ingestor.
```

---

## Critical Step: DeepNSM Decomposition

DeepNSM currently lives at `lance-graph/crates/deepnsm/` wired to the
DataFusion query layer (`lance-graph/src/nsm/`). For OSINT use we need a
**decomposed, standalone extractor** — no DataFusion dependency, just:

```rust
// Target API
use deepnsm_extract::{Extractor, Triplet, TruthValue};

let extractor = Extractor::new();  // loads 4096 COCA + distance matrix
let text = "The Pentagon contracted OpenAI in December 2024.";
let triples: Vec<Triplet> = extractor.extract(text);
// [
//   Triplet { s: "Pentagon", p: "contracted", o: "OpenAI", truth: ... },
//   Triplet { s: "contract", p: "occurred_at", o: "2024-12", truth: ... },
// ]
```

Decomposition work (~300 LOC):
1. Extract the 6-state PoS FSM from `nsm/parser.rs`
2. Extract the COCA codebook from `nsm/encoder.rs`
3. Pure function: `&str → Vec<Triplet>` with no DataFusion context
4. Add to the OSINT pipeline as the text → SPO stage

The DeepNSM research paper claim is real: **680GB model → 16.5MB**,
**50ms/token → <10μs/sentence**. At that speed, 5 billion web pages
finish extraction in ~14 hours on a workstation.

---

## Why This Beats Palantir / Maltego / Bellingcat's Stack

### Cost structure

| Tool | Entry cost | Annual cost | Scale limit |
|------|-----------|-------------|-------------|
| Palantir Gotham | $M+ deploy | $M/year | Any (Palantir's cost) |
| Maltego XL | $10K+ license | $5K+/year | ~10M entities |
| Bellingcat's custom | free tooling patchwork | engineer hours | Per-investigation scope |
| **Our stack** | **$0** | **$0 hosting on a VPS** | **Billions of triples** |

### Query expressiveness

- Palantir: proprietary DSL, exposed as forms in the UI.
- Maltego: transform DSL, acyclic.
- Bellingcat: SQL + Python + Python.
- **Ours: Cypher / GQL / Gremlin / SPARQL on the same graph**, with NARS
  confidence propagation and CAM-PQ semantic distance as UDFs.

### Data ownership

- Palantir: their servers.
- Maltego: their transform servers call out.
- Bellingcat: runs on your machine but you're rolling your own.
- **Ours: your workstation, your data. Deploy to a private VPS if you
  want remote collaboration.**

### The positioning sentence

> "Palantir for the rest of us. Cypher-queryable OSINT graph, 29 pre-
> built investigative patterns, runs on a laptop, reads the whole open
> web at 17K tokens per second."

---

## Sample Investigation Flow

Using `aiwar_enrichment_pattern_zero.cypher` as an example (already in
the repo):

1. **Seed:** "Investigate the AI-intelligence overlap pattern."
2. **Crawl:** spider fetches news articles mentioning OpenAI, Anthropic,
   Palantir, Pentagon contracts.
3. **Route:** Reader-LM palette flags relevant pages (news, not product
   pages).
4. **Extract:** Reader-LM inference → clean Markdown → DeepNSM → SPO
   triples.
5. **Ingest:** AriGraph stores each triple with NARS truth (confidence
   from source reliability, frequency from repetition).
6. **Enrich:** apply `aiwar_enrichment_pattern_zero.cypher` — this file
   MERGEs the historical paradigm entities (Surveillance / OSINT /
   AI Cognitive Access), connects events, assigns opacity tags.
7. **Visualize:** aiwar d3 force layout shows three paradigm clusters
   and the companies straddling them.
8. **Query:** `MATCH (c:Company)-[:CONTRACTS_WITH]->(g:GovEntity) WHERE
   c.ai_capability = 'frontier' AND g.intel_agency = true RETURN c, g`
   — see who is actively contracting, when, for what.

Each enrichment pattern is one .cypher file. Composable.

---

## Tier 0 — Real Effort Estimate

Revised after confirming most components are already crates:

| # | Task | What it actually takes |
|---|------|------------------------|
| T0.1 | Decompose DeepNSM's `extract(&str) -> Vec<Triplet>` surface so `lance-graph-osint::extractor` can call it without the DataFusion plumbing | ~150 LOC shim + tests, **1 day** if the PoS FSM is already isolated, longer if it's entangled with the NSM query module |
| T0.2 | Add `spider` as workspace dep; wire its output into `lance-graph-osint::crawler` | Mostly configuration + trait-impl glue, **half a day** |
| T0.3 | Reader-LM palette gate in front of inference — route-or-drop | The reader-lm crate already supports both modes; **a few hundred lines** of orchestration in the osint pipeline |
| T0.4 | Cypher pattern registry: load `aiwar-neo4j-harvest/cypher/*.cypher` into the cockpit as named patterns | **100-200 LOC**. Cypher execution is already imported into q2 via the `lance-graph` workspace dep (the parser is in the core crate). Only the *registry indexing* of the 29 pre-authored patterns is new work — walk the directory, parse each file, expose by name. |
| T0.5 | Cockpit endpoint `POST /api/osint/investigate { seed, pattern? }` | Axum handler, follows the existing `/api/debug/osint` pattern in cockpit-server (which is already stubbed) — **~200 LOC** |
| T0.6 | Frontend: force-directed view | Either embed `aiwar` as a standalone page or port its d3 force layout into the q2 cockpit React bundle. **1-3 days** of frontend work depending on polish level |
| T0.7 | One end-to-end investigation demo (pick one existing `.cypher` pattern and walk it) | Content work, not code — **1 day** |
| **Total** | | **~5-8 days** if DeepNSM decomposes cleanly, **~10 days** if the extractor needs more work |

Codebook loading is not an explicit task line — the existing crates
(deepnsm, reader-lm, jina v5 via ndarray) already know how to fetch
their Release assets. As long as the Releases are published, this is a
runtime detail, not build-time work.

---

## Tier 1 — After Tier 0 Lands

| # | Extension | Value |
|---|-----------|-------|
| T1.1 | Cross-reference with Wikidata palette (load Wikidata per the `wikidata-spo-nars-at-scale.md` plan) | Entity linking: "OpenAI" in a news article → Wikidata Q63989 with full provenance |
| T1.2 | Live RSS / news feed tap | Continuous ingestion, not just one-shot investigations |
| T1.3 | Timeline view with temporal patterns | `aiwar_enrichment_temporal_opacity.cypher` already authored |
| T1.4 | Multi-investigator blackboard | `a2a_blackboard` contract; multiple analysts converge on same graph |
| T1.5 | xAI / Claude enrichment on novel connections | AriGraph's `xai_client.rs` is already there |
| T1.6 | Export to PDF/QMD report | Quarto roots make this natural — OSINT as publishable artifact |

---

## Risks

- **DeepNSM extraction quality.** The 6-state PoS FSM is fast but may
  miss nuanced extractions ("suspected", "allegedly", "reportedly" all
  matter in OSINT). Needs a confidence tag in the extractor that feeds
  NARS confidence downstream.
- **Cypher enrichment authorship.** The 29 existing patterns are a
  starting library; new investigations need new patterns. Consider a
  "pattern composer" UI later that lets analysts author Cypher from
  templates.
- **Source reliability scoring.** NARS confidence can't fix
  disinformation — garbage in, garbage out. Pair with a provenance
  layer: track source URL, author, publication. Let the analyst set
  confidence multipliers per source.
- **Legal surface.** Web crawling at scale intersects with terms of
  service, robots.txt, copyright law. For the open-source release,
  document that users are responsible for legality of their crawls.
  Consider a "respect robots" default and an explicit override.
- **Political surface.** The existing Cypher patterns (aiwar-neo4j-harvest)
  are investigative and editorial. Shipping them as a demo with
  specific name-and-shame queries is a different posture from shipping
  a neutral tool. Separate the **tool** (our stack) from the
  **application** (specific investigations); let users author their
  own patterns or import community-shared ones.

---

## Why This Is the Highest-Leverage Demo

- **Largest audience.** Journalists, researchers, intelligence
  analysts, policy shops, legal discovery — millions of people, all
  underserved by existing tools.
- **Hardest competitor moat.** Palantir's entire business is "we
  have the only good UI for this." A free, faster, Cypher-queryable
  alternative is an existential threat to them — which means it's
  also a high-attention release.
- **Validates every substrate bet at once.** OSINT needs:
  - Scale (billions of documents) — Wikidata argument applies.
  - Semantic similarity (find related but not identical mentions) — CAM-PQ.
  - NARS confidence (source reliability propagation) — contract pillar.
  - Episodic memory ("this is similar to that case") — AriGraph.
  - Pluggable framing (same graph, multiple investigator perspectives) — WorldMapRenderer.
  - Live 3D viz (make relationships visceral) — aiwar + q2 cockpit.
  - All four pillars (self-state during search, empathy for actor modeling) — world_model.

  Every piece of the session-capstone substrate earns its keep in OSINT.

- **Demonstrates the "incidentally" strategy at its strongest.** The
  product positioning (`positioning-quarto-4d.md`) says: sell
  "fast Cypher + graph notebook." The OSINT use case is the killer
  **example** of why that matters — not abstract benchmarks but
  "journalism at scale."

---

## Relationship to Chess and Wikidata Demos

| Demo | Dataset | Proves | Target audience |
|------|---------|--------|-----------------|
| Chess vertical | 5B Lichess games | Precision + live telemetry | Chess/AI researchers, enthusiasts |
| Wikidata scale | 1.2B triples | Knowledge-graph compression at scale | Graph DB evaluators, data engineers |
| **OSINT pipeline** | **Live open web** | **Applied intelligence, continuous** | **Journalists, analysts, policy, legal** |

All three share:
- Same lance-graph substrate
- Same 4-pillar contract
- Same q2 cockpit
- Same Cypher query surface
- Same compression stack
- Same BindSpace + AriGraph + episodic memory

Only the ingestor differs. That's the validation: one substrate handles
three distinct domains at billion-entity scale, each with a killer
demo in the same UI.
