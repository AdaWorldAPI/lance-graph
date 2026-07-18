# graphrag + doc-retrieval on the V3 SoA — integration plan v1

> **Status:** DESIGN + FIRST CODE. **v1.2 (2026-07-17):** the base is a *meaning
> measurement* — COCA + NARS are co-equal core faculties, agnosticism is scoped to
> raw data + consumer modelling only (§3a); Leiden community synergies (§3b); the
> DocumentID-KV / witness-handle seam (§4a). **v1.1:** expand AriGraph in place —
> NO standalone `crates/graphrag` (§1). **First code shipped: `arigraph/community.rs`
> (D-GR-3a) — `TripletGraph::communities()`, multi-level Louvain, deterministic.**
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

## §3a. The base — a meaning MEASUREMENT: tokenization, ranking (256), distribution (256²)

The substrate is a **meaning substrate, not a meaning-agnostic one** (operator,
2026-07-17). Two **co-equal core faculties** carry it: **COCA** (distributional
semantics / syntax / pragmatics — the meaning layer) and **NARS** (truth /
reasoning). Agnosticism is scoped ONLY to **raw data** (bytes → the KV, §4a) and
**consumer modelling** (app schema); the graph is the *opposite* of agnostic
about meaning — it **measures** it. The `markov_soa` "vocabulary-agnostic" hot
loop is codebook-*parametric* for cross-domain SIMD reuse (an injected
`Fn(u16,u16)->u8`); for language/document content the injected distance IS the
core COCA meaning — correct, not a "language lens." *(Correction target: the
shipped `markov_soa.rs:16,28` "keep language out" comment reflects the older
framing; the COCA meaning codebook wants a core-reachable home, not quarantine in
the downstream `deepnsm` crate — board follow-up.)*

**Tokenization = the L4 facet tenant.** CAM-PQ = 6 subspaces × 256 centroids
(`cam_pq/udf.rs:27,37`); DeepNSM IS that same PQ, trained from the 96D **f32**
COCA vectors *at build time* (`deepnsm/codebook.rs:3-4` — `NUM_SUBSPACES=6`, 256
centroids) but emitting the **8-bit** code. Content → a `6×2×8bit` code = the
node's **L4 `6×(8:8)` facet tenant** (`le-contract.md:59`).
Quantizing to one centroid per subspace *is* tokenization; the token IS the
facet, no separate index. COCA ≈ 98.4 % English → the code is "as good as
agnostic" (a near-universal meaning basis, a *measurement*, not a narrowing bias).

**Everything is 8-bit (CLAM), never f32.** The 12-byte facet is **`6×2×8bit`** —
six subspaces, each a two-byte pair — and the CLAM tree indexes those 8-bit codes;
**f32 is a build-time source only** (the codebook is trained from f32, the runtime
register is not). The same `6×(8:8)` register is **polymorphic** (classview-selected):
read as **`part_of:is_a` family identity** (L1 — the HHTL family/basin rail) OR as
the **palette256² centroid** (L4 CAM-PQ). So "6×256:256" is the is_a family
identity AND the centroid — one register, two lenses; this is *why* communities
(structural) and is_a basins (family) couple so tightly (§3b.1).

**256 = ranking; 256² = distribution** (the le-contract distance-vs-compose split,
`:163-167`):

| | **256 (single 8-bit) = ranking** | **256² (2×8bit pair) = distribution** |
|---|---|---|
| shape | one monotone axis | the full joint pairwise |
| accuracy | the cosine axis only — **REQUIRES the cosine-replacement** (Fisher-z: cosine→`atanh`→i8; a raw byte is not cosine) | **as accurate as content-blind f32** — the pair recovers full f32 fidelity at 8-bit (the claim P-PQ-RANK certifies) |
| read | `\|Δi8\|·(z_range/254)`, table-free | the semiring COMPOSE table (kept — "z-addition does not compose cosines", `:167`) |
| answers | "how similar / what order" | "how it relates / composes / transitions" |
| a word/doc | a scalar position | its **relational profile = its distribution** |

**Distance is the stacked cosine-replacement, NOT popcount — and it ALREADY EXISTS
(operator, 2026-07-17: "grep it, don't build it").** The cosine-replacement is the
Fisher-Z `arctanh` map `z = ½(ln(1+s)−ln(1−s))`, shipped and **certified** as
`bgz-tensor::fisher_z::{FamilyGamma, Base17Fz}` (ρ≥0.999, 21 roles, the `arctanh→i8`
table), with the clean-room `helix::fisher_z::Similarity` + `helix`'s place/residue
256-palette ladder (`crates/helix/src/{fisher_z,placement,residue,distance,simd}.rs`
+ `KNOWLEDGE.md`) and the `lance-graph-contract::distance` surface; the map is
`.claude/DISTANCE_METRIC_INVENTORY.md`. A node-to-node distance is **6 of those
cosine-replacement rankings stacked** — one 256-axis per subspace, `atanh`→i8,
L1-metric-safe over the 256×256 LUT — assembled into the HHTL family identity
(`6×256`; the stacking IS the HHTL cascade / `bgz-tensor::hhtl_d`, §3b's hierarchy).
That *is* the CAM-PQ ADC, and it **retires the Hamming/popcount path**:
XOR-then-popcount (`blasgraph/ndarray_bridge.rs` `hamming_*`/`popcount_*`,
dead-code) is brute-force bit-counting — a coarse tally where the certified `atanh`
axis is cheaper and monotone-exact. So the distance work is **WIRING the existing
certified cosine-replacement into the graph/PPR distance path** (helix graduates
from clean-room per its KNOWLEDGE.md §Consolidation: re-export `bgz-tensor`'s
`FamilyGamma`), gated by the encoding-ecosystem naive-u8 floor (≥0.9980 Pearson) /
P-PQ-RANK — **not building a new kernel.**

**This is a MEASUREMENT, not a judgment — and that is the point.** COCA ×
6×256:256 is **deterministic** (integer/table, bit-reproducible, 0 learned
params, <10 µs/sentence; an LLM is stochastic even at T=0), **agnostic** as a
*fixed, transparent, empirical* basis (vs an LLM's opaque, RLHF-shaped,
version-drifting one), and **auditable/falsifiable** — a fixed inspectable table
you can *certify* with the jc battery (ρ≥0.999). Determinism is the
**precondition** for the probe-first spine: you cannot run a reproducible
falsification against a stochastic oracle. The architecture therefore **demotes
the LLM to the escalation tail** (the "<25 %" the stack routes to an oracle only
when measurement runs out); everything in the hot loop is a certifiable
measurement.

**The rung ascent IS ranking → distribution.** Rung 0–1 = 256-ranking; rung 2+ =
256²-distribution (SPO-G compose, PPR, community). BLOCK/surprise escalates from
"sort by similarity" to "reason over the relational field." **Exactness escape
hatch:** the bgz-tensor **index+residual ladder** (`adaptive_codec.rs`,
`:189-207`) — centroid PLACES, a Hadamard residual CORRECTS in a 3-tier LFD
split, hardest ~10 % → `Passthrough` (exact vector).

**DeepNSM is the no-LLM tokenizer + SPO sensor — an UPSTREAM leg.** text →
COCA-4096 tokenize → PoS-FSM → 36-bit SPO → ±5 VSA context (`deepnsm/pipeline.rs`).
**Crate-layering constraint (mechanical, not semantic):** AriGraph core cannot
Cargo-depend on the `deepnsm` crate (downstream). So `community.rs`/`retrieval.rs`
consume the **in-core `crates/lance-graph/src/nsm/` copy** or receive SPO
produced upstream (osint) — the *meaning* (COCA) is core; only the *sensor
packaging* is downstream. DeepNSM's `BasinClassification` (`episodic_spo.rs:216`)
is a per-sentence basin proposer — the design reference for §3b's
community/basin cross-validation.

## §3b. Leiden community synergies (why AriGraph, not a crate)

Communities are the **structural partition** completing the partitions AriGraph
already carries, and they are **distributional-meaning modes, NARS-truth-weighted**
— not agnostic clustering of opaque ranks. Six synergies:

1. **× episodic/family basins — orthogonal partitions that cross-validate.** The
   basin = `part_of:is_a` rail / `EpisodicEdges64` family (`family==0`
   intra-basin) — the *inherited/experiential* grouping; a community = the
   *computed structural* grouping. Coincide ⇒ high-confidence structure; cross ⇒
   a discovered bridge; basin-without-community ⇒ a revision candidate. A NEW
   partition beside the family rail, never a re-carving of it (measure agreement:
   P-COMMUNITY-BASIN-AGREE).
   **Sharpening (operator, 2026-07-17) — the identity hypothesis.** "Part of"
   presupposes a *category* to be part **of**, and that category IS the community
   = the basin = the HHTL family identity (the same `6×(8:8)` register read as the
   L1 `is_a` rail, §3a). The two partitions are *computed* differently (Leiden over
   structure vs the experiential family rail) but may resolve to the **same
   categories** — community detection is **constitutive** of the `is_a` category,
   not a decoration on it: you detect the community/basin first, then `part_of`
   points into it and membership **is** `is_a`. So P-COMMUNITY-BASIN-AGREE tests an
   **identity**, not merely a correlation — community ≡ basin ⇒ agreement → 1.0;
   high-but-<1.0 ⇒ correlated-but-distinct, and the disagreements are exactly the
   bridges / revision candidates. Either way a finding; the identity is the
   *hypothesis*, the probe is the *falsifier* (the "same concept at the same time"
   claim made precise). **[S1 harness SHIPPED #720** —
   `examples/p_community_basin_agree.rs`, φ via `jc::pearson`: aligned 1.0000 /
   bridged 0.5500 with the bridge named; real-corpus verdict OPEN.]
   **Orientation generalization (operator, 2026-07-17 — captured in
   `.claude/knowledge/context-role-traversal-tissue.md`):** `part_of:is_a` ≡
   **context:role**, one polymorphic `(8:8)` register with two orientations —
   **vertical** (stacked exactness = HHTL family identity; falsifier =
   P-HIER-LEIDEN-HHTL over `Communities.levels`) and **horizontal** (6-context
   episodic-witness `basin:role`; falsifier = the shipped S1 harness re-fed from
   the 6-slot frame). The S1 identity is the register-level invariant both
   orientations test; the same tissue is the reuse spine for screens
   (`menu_address`/`WideFieldMask`), documents, and time series.
2. **× the 256² distribution — Leiden clusters the distribution, not the
   ranking.** Modularity runs over the compose/distribution graph (256²
   palette-compose + `CausalEdge64` SPO edges), NOT the 256 rank. A community IS
   a mode of the relational field — inexpressible as a scalar rank; this is why
   §3a's pairwise is load-bearing.
3. **× COCA meaning / NARS truth — communities = topics, truth-weighted, no
   LLM.** A word/doc's meaning IS its distribution (its 256²/4096² row); Leiden
   over it = semantic communities = topics — the **no-LLM community-summary** leg
   (vs graphrag-rs's LLM summaries). Edges are NARS-weighted (`community.rs` uses
   `truth.confidence`; expectation-weighting is the refinement) — a community is
   a *truth-weighted meaning mode*, COCA + NARS together.
4. **× the rung ascent — community IS the rung-3 layer.** 0–1 ranking → 2 SPO-hop
   → **3 community-scoped expansion** (community bounds the PPR restart set; the
   summary is the coarse answer) → 4 `intervene_on` apex. `detect_contradictions`
   → BLOCK widens the community scope.
5. **× PPR — community-scoped personalized PageRank.** Restart-within-community =
   HippoRAG passage-community scoping: cheaper (bounded) + more relevant. Community
   picks the subgraph; PPR ranks within it.
6. **× the witness/KV/document layer — communities of DOCUMENTS.** Documents are
   witnessed nodes (DocumentID handles, §4a). Leiden over the document-graph =
   document communities = topic clusters — the "documents in this community" seam
   `ogar-doc`'s reconstruct/related-docs path consumes.

**Hierarchical Leiden ↔ the existing hierarchy.** Leiden is hierarchical; the
substrate is already hierarchical (256 = 4⁴ centroid tree; HHTL cascade;
`part_of:is_a` taxonomy). A super-community SHOULD align with a coarse HHTL tier
/ taxonomy parent — **register and measure agreement** (P-HIER-LEIDEN-HHTL), not
bolt on a foreign hierarchy; disagreement = discovered structure. **The LFD-hard
tier = community boundaries** — the ~10 % Passthrough nodes are the bridges where
assignment is ambiguous; the exactness ladder and the community geometry are the
same object seen twice.

**Shipped (D-GR-3a):** `arigraph/community.rs` — multi-level Louvain over the
`TripletGraph` adjacency (NARS-confidence-weighted), the carrier method
`TripletGraph::communities()`, deterministic (BTreeMap-ordered moves,
sorted-entity index), 5 inline tests (two-triangle→2, clique→1, determinism,
empty-safe, weighted-cohesion).

**Shipped (D-GR-3b, this PR — all pure / reversible / no-write-path, so they
land ahead of G0 exactly as D-GR-3a did):**
- `arigraph/ppr.rs` — `TripletGraph::personalized_pagerank(seeds, damping, iters)`,
  HippoRAG spread over the confidence-weighted graph, deterministic, unit-sum,
  6 tests (near-triangle-outranks-far, sum≈1, determinism, empty-safe,
  unmatched-seed-fallback, seed-top-ranked).
- `community.rs` **Leiden connectivity refinement** (`refine_connected`) — splits
  any internally-disconnected Louvain community into its connected components
  (Leiden's guarantee), deterministic BFS over intra-community edges, +2 tests;
  `labels` is now the refined coarsest partition, `levels.last()` the raw Louvain.
- `arigraph/bm25.rs` — Okapi BM25 lexical leg (`Bm25Index`, k1=1.2/b=0.75),
  the rung-0/1 baseline beside CAM-PQ ranking, deterministic, 5 tests.
- **D-GR-1** `lance-graph-contract/src/doc_graph.rs` — the zero-dep `DocGraphQuery`
  trait + `ScoredId`, carrying the rung→walk dispatch as a default method; 9 tests.
  The D-GR-2 design (OsintRetriever ↔ #708 RungElevator) is embedded as its module-doc.
- **G0** `examples/g0_graph_loadbearing.rs` — the P-GRAPH-LOADBEARING harness:
  vector-only (BM25) vs vector+PPR+community on a synthetic multi-hop fixture,
  prints the with-vs-without delta (gold `turbines`: vector-only rank 8/8 @ 0.0,
  graph rank 6 @ 0.146). Synthetic **scaffold**, not the verdict — the real
  KILL/PASS needs a labeled corpus + jc::reliability.

**Still gated on G0:** the **D-GR-2 wiring** — fusing CAM-PQ + SPO-G + PPR +
community into `retrieval.rs` under the RungElevator (design done, impl gated).
The cosine-replacement distance that retires popcount is a separate
P-PQ-RANK-gated **wiring** (the primitive already exists — certified
`bgz-tensor::fisher_z::{FamilyGamma, Base17Fz}` ρ≥0.999 + `helix`, §3a — route it
into the graph/PPR distance path; do not rebuild it).

## §4. Topology (v1.1 — expand AriGraph, no new crate)

```
crates/lance-graph/src/graph/arigraph/     (EXPAND — the graph owns its own structure)
  community.rs   # SHIPPED (D-GR-3a): TripletGraph::communities() — multi-level Louvain
                 #      over the triplet adjacency (NARS-confidence-weighted), deterministic,
                 #      5 tests. The structural partition beside witness_corpus/episodic basins.
                 #      Next: Leiden refinement pass; PPR/community fusion (D-GR-3b).
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

## §4a. The document-identity KV seam — witness holds a handle, never raw

Operator threads (v1.2): a separate KV so the episodic witness doesn't carry raw
data, keyed by DocumentPath/DocumentID, related to `ogar-doc`/tesseract-rs
rendering over ClassView×WideFieldMask. Grounded:

- **DocumentID/DocumentPath is NEW and the RIGHT key.** It doesn't exist yet (no
  KV table; no `DocumentPort` in tesseract-rs). Do NOT reuse `content_sha256`
  (per-acquisition DEDUP key — scan vs HTML of the same invoice differ,
  `ogar-doc-ir/lib.rs:308`) or expose `kv_key` (storage-opaque). A stable
  DocumentID gives the **witness ↔ node ↔ KV** triangle one handle independent of
  retina and blob store — make the raw-ref's `kv_key` field CARRY the DocumentID.
- **The three-tier handle chain (no raw embedding):** (1) AriGraph **witness** =
  `WitnessEntry{mailbox_ref, spo_fact_ref}` (hot path) — already **handle-only**
  ✅; (2) **`document 0x080B` node value** = raw-ref `{sha256 digest,
  kv_key=DocumentID, mime, counts}` — the SINGLE place DocumentID↔kv_key lives,
  calcifies to Lance as SoA value bits; (3) **consumer KV** keyed by DocumentID =
  the actual bytes/raster (S3 / MedCare `file_filelist` / a *passive* Lance blob
  column). The render leg **late-binds** the raster only at render time.
- **The change your KV idea targets is real + localized (D-GR-6).** The hot-path
  witness is already raw-free; but the **cold-path `witness_corpus::WitnessEntry`
  embeds `evidence_blob: Bytes`** and **`episodic::Episode` embeds `observation:
  String`** — replace those with a DocumentID handle → KV. A scoped AriGraph
  change, not a rewrite.
- **Boundary + rendering:** the KV is **consumer/blob-store side, never a Lance
  ingest endpoint**. OGAR assembles; lance calcifies handles; tesseract-rs stays
  untouched (`doc.v1` seam). Rendering reuses the shipped `ClassView ×
  WideFieldMask` brick (`render_field_view` / `render_class_with_methods_wide` /
  a2ui `project_node`; PDF = the unbuilt `ogar-doc reconstruct_document`), RBAC
  `template ∩ role` fail-closed. **Sequencing:** the `0x080B`/`0x080A` mints +
  persist/reconstruct are BLUEPRINT-ONLY, council/mint-gated — D-GR-5/6 sequence
  AFTER the doc-W4 council.

## §5. Wave sequencing (aligned to v3 W0–W6 + D-TRI)

Neither crate is a new W-wave. Sequenced **after the W1 keystone**
(`mailbox_owner()` shipped #631; batch-writer W1b in-PR):

- **G0 — P-GRAPH-LOADBEARING — HARNESS SHIPPED** (`examples/g0_graph_loadbearing.rs`).
  The with-vs-without measurement *mechanism* runs (synthetic multi-hop scaffold,
  prints the delta); the real KILL/PASS **verdict** still needs a labeled corpus +
  jc::reliability. Gate on truth-architect / measurement-before-synthesis — the
  **D-GR-2 wiring stays blocked on the real-corpus verdict**, not on the scaffold.
- **D-GR-3a — SHIPPED** — `arigraph/community.rs` (`TripletGraph::communities()`,
  multi-level Louvain, deterministic, 5 tests). Landed ahead of G0 as a *pure,
  reversible* capability with no write path — it computes a partition, gates
  nothing. G0 still gates whether it is *wired into retrieval*.
- **D-GR-1 — SHIPPED** — `DocGraphQuery` trait + `ScoredId` in `lance-graph-contract`
  (`doc_graph.rs`, zero-dep, 9 tests, rung→walk dispatch as a default method; the
  D-GR-2 design lives in its module-doc). Zero SoA writes.
- **D-GR-2** — extend `arigraph/retrieval.rs` to bind **existing** CAM-PQ + SPO-G
  hops onto the canonical `RungLevel`/`RungElevator` (**#708 merged `8d3209c`**;
  advance-before-sinks ordering per `17368ea`). Mirrors the #708 settlement probe
  (BLOCK ascends → wider walk; FLOW at base).
- **D-GR-3b** — the remaining BUILD gaps: **Leiden refinement** (well-connected
  communities) on `community.rs`; **`ppr.rs` HippoRAG** (reset-distribution atop
  `blasgraph::hdr_pagerank`) + **community-scoped PPR** fused into `retrieval.rs`
  as a third signal beside BFS+episodic; **BM25** (small, out-of-graph). Each
  gated on G0 beating vector-only.
- **D-GR-4** — community summaries via the Rig oracle → compiled template (W3
  `template-runtime`). No-LLM path preferred (DeepNSM distributional summary);
  LLM only at the escalation tail.
- **D-GR-5** — wire `ogar-doc` `reconstruct_document` + "documents in this
  community" to `DocGraphQuery` (cross-repo; baton-audited; AFTER doc-W4 council).
- **D-GR-6** — the witness-KV separation (§4a): replace the **cold-path**
  `witness_corpus::WitnessEntry.evidence_blob` + `episodic::Episode.observation`
  raw embeds with a **DocumentID handle → consumer KV**. Scoped AriGraph change;
  born-stamped if it persists; AFTER doc-W4 council (DocumentID is the raw-ref key).

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
- **P-COMMUNITY-BASIN-AGREE (new, v1.2).** Leiden communities vs the
  `part_of:is_a`/`EpisodicEdges64` family basins — measure agreement
  (jc::reliability). High agreement ⇒ structure confirms experience; the
  disagreements are the discovered bridges / revision candidates (§3b.1).
- **P-HIER-LEIDEN-HHTL (new, v1.2).** Hierarchical Leiden super-communities vs the
  coarse HHTL tiers / taxonomy parents — measure agreement; disagreement =
  structure the taxonomy lacks (§3b hierarchy note).
- **P-MEASUREMENT-DETERMINISM (new, v1.2).** `community.rs` (and every base read)
  is bit-reproducible: same graph → identical partition across runs/machines
  (the property that makes the jc certifications *possible*, and that an
  LLM-based clustering cannot offer). Asserted by the `deterministic` unit test;
  the substrate-wide claim of §3a.
- **P-PQ-RANK (new, v1.2).** Spearman ρ of the `6×256:256` node-to-node distance
  vs the `4096²` reference on COCA pairs (jc::reliability). ρ≥0.99 ⇒ 256 centroids
  suffice for the base; else escalate hard pairs to the index+residual /
  Passthrough tier (§3a).
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
6. **Meaning-substrate, NOT meaning-agnostic (v1.2 correction).** Do NOT
   reintroduce the "keep language out of the agnostic graph" framing. COCA
   distributional meaning is a **core faculty** (co-equal with NARS); the
   `markov_soa` hot loop is codebook-*parametric* for cross-domain reuse, and for
   language content the injected COCA distance is correct, not a "language lens."
   Agnosticism = raw data + consumer modelling only (§3a). Only the *crate
   layering* is a real constraint: AriGraph core can't Cargo-dep the downstream
   `deepnsm` crate — use the in-core `nsm/` copy. (The shipped
   `markov_soa.rs:16,28` comment is a correction target, not a rule to obey.)

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

**Next step (v1.2):** `arigraph/community.rs` is **DONE (D-GR-3a)** —
`TripletGraph::communities()`, multi-level Louvain, deterministic, 5 tests
(Louvain core verified standalone before the core build). Remaining, in order:
(1) `ppr.rs` HippoRAG + community-scoped PPR fused into `arigraph/retrieval.rs`
under the #708 `RungElevator` (D-GR-3b); (2) the `DocGraphQuery` contract trait
(D-GR-1); (3) the G0 `P-GRAPH-LOADBEARING` probe + P-PQ-RANK / P-COMMUNITY-BASIN-AGREE
(jc::reliability). No new crate; no upstream change for (1)-(3).
