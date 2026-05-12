# anatomy-realtime-v1 — The Proof-of-Vision Plan

> **Owner:** Worker Agent W12 (12-agent unified-OGIT architecture-synthesis sprint).
> **Branch:** `claude/unified-ogit-architecture-synthesis`.
> **Master plan:** `.claude/plans/unified-ogit-architecture-v1.md` (W1).
> **Sub-plans referenced:** `ogit-g-context-bundle-v1.md` (W10),
> `compile-time-consumer-binding-v1.md` (W11).
> **Tech-debt anchor:** `TD-ANATOMY-DEMO-8` in `.claude/board/TECH_DEBT.md`.
> **Cross-repo touchpoints:** `lance-graph/`, `MedCare-rs/`, `q2/cockpit-server/`.

---

## 0. Why this demo exists

The unified-OGIT architecture has fifteen named patterns (A-O). Eleven of
them are partially shipped substrate (CAM-PQ codebook, deepnsm, EWA-Sandwich,
OGIT-G u32 slot, GenericBridge admit/route, medcare-rs RBAC, LanceAuditSink,
SplatShaderBlas + Pillar-7 alpha-saturation, Q2 cockpit-server, qualia17D,
p64-bridge::CognitiveShader). Four are wiring (Tier-1/Tier-2 deliverables).

The danger of a fifteen-pattern architecture is the **demo gap**: code that
compiles, tests that pass, no single artifact in which a user watches the
whole thing breathe. This plan defines that artifact.

**The filter.** Every architectural primitive in the workspace earns its
place in this demo by appearing on the screen, or gets exposed as
not-yet-needed (and is dropped or deferred). The demo is the live filter
between "shipped infrastructure that funds the vision" and "speculative
crates that should be excluded from the workspace."

**The user.** A radiologist or clinician loads a CT scan in Q2's cockpit-
server UI and works through ten steps. Each step exercises one or more
architectural patterns. When the demo works end-to-end, the architecture
is proven; when a step fails, the corresponding pattern has a wiring gap.

---

## 1. The ten-step demo (what the radiologist sees)

| # | User action | Architectural pattern(s) exercised |
|---|---|---|
| 1 | Load CT scan; voxels stream into BindSpace (~10^9 rows for full-body CT). | Pattern C (GenericBridge admit), Pattern I (CycleAccumulator pre-warm). |
| 2 | Each voxel gets a tissue class via 256-entry CAM-PQ palette lookup (already shipped in `cam_pq/`). | Pattern N (codebook similarity, fingerprint to centroid in O(1)). |
| 3 | Tissue class to FMA anatomical class via u32 to ContextBundle lookup. | Pattern A (SPO-G u32 quads), Pattern B (`G=FMA_V1` ContextBundle). |
| 4 | Connected anatomical regions emerge via L4 SoA sweep + alpha-saturation. | SplatShaderBlas (PR #346), Pillar-7 (PR #347). |
| 5 | Sigma-propagation along `part-of` / `supplies-blood-to` / `innervates` edges. | EWA-Sandwich (PR #289, SPD-preserved). |
| 6 | User clicks "show everything connected to the heart" -> Cypher / SPARQL query over SPO-G with `G=FMA_V1`. | Patterns A + L (graph-router cross-modal). |
| 7 | Realtime 3D render with FMA labels overlaid + cross-section. | Pattern H (`p64-bridge::CognitiveShader` dispatches per-G program), Q2 cockpit-server (q2 PR #35). |
| 8 | Radiologist adds finding: "tumor in left lung, suspicious for adenocarcinoma" -> GenericBridge admits write via medcare-rs ConsumerPointer; RBAC gates; LanceAuditSink emits trail. | Patterns C, E, F (compile-time admit + RBAC POLICY-1 + audit). |
| 9 | Cross-modal: "find scans where this same anatomical pattern appears" -> DeepNSM medical-terminology vocab + SplatShaderBlas-Bitpacked Jaccard probe. | Pattern N (codebook similarity), Pillar-7 (PR #347). |
| 10 | Phenomenological feedback: each step generates Qualia17D + CausalEdge64 + ResonanceDTO; meta-awareness fingerprint accumulates ("I just understood the lung anatomy") into AriGraph core memory. | Pattern O (qualia.rs), Pattern G (ThinkingStyle inheritance). |

Patterns J, K, M are exercised implicitly by the substrate (J = Lance MVCC
underneath every write; K = the SoA invariant the columns enforce;
M = wave-mode bundle similarity in Step 9 co-active with particle-mode
graph traversal in Step 6).

---

## 2. What is already shipped (substrate inventory)

This is the floor the demo stands on. Each line is a citable PR or
crate-status.

| Substrate | Citation | Why it matters here |
|---|---|---|
| Splat math: 20K x 20K Gaussian-splat lab precedent (zero errors). | jc lab precedent. | Voxel batch math at scale. |
| 1000-path x 10-hop SPD-preserved propagation. | EWA-Sandwich PR #289. | Sigma-propagation along anatomical edges (Step 5). |
| L1+L2 popcount-AND ~ 5.8x CSR (Triangle Count probe). | PR #346. | Fast anatomical-neighborhood graph traversal. |
| Pillar 6 SPD-preservation 1000/1000. | PR #289. | EWA-Sandwich correctness gate. |
| Pillar 7 alpha-saturation 100/100 convergence. | LPA probe PR #346. | Connected-region emergence (Step 4). |
| O(1) ontology lookup 2554x SPARQL-proxy. | PR #355 D-CASCADE-V1-11. | FMA class lookup is fingerprint to codebook to O(1). |
| Lance MVCC + audit + RBAC seams closed. | PRs #29, #98, #337. | Step 8's audit trail; POLICY-1 gating. |
| `p64-bridge::CognitiveShader` (Pattern H). | shipped. | Per-G program dispatch (Step 7). |
| `qualia.rs` (Pattern O). | shipped. | Qualia17D fingerprint generation (Step 10). |
| thinking-engine cognitive substrate (Patterns M, N). | shipped. | Wave + particle modes (Steps 6 + 9). |
| Q2 cockpit-server. | q2 PR #35. | The UI surface itself (Step 7). |
| medcare-rs RBAC + audit sink. | shipped. | POLICY-1 gating + LanceAuditSink (Step 8). |
| CAM-PQ codebook tissue classifier. | `cam_pq/` shipped. | Voxel to tissue-class palette lookup (Step 2). |

What is **not** yet shipped is named in section 3.

---

## 3. What is needed - 5-7 PRs over weeks

Effort estimates are LOC after tests, on the assumption that W10's Tier-1
slot (SPO-G u32 + ContextBundle) and W11's Tier-2 manifest are merged
first. Every PR has a hard acceptance criterion and a dependency arrow.

### PR-ANATOMY-1: OWL hydrator for FMA

**Goal.** Hydrate FMA (Foundational Model of Anatomy: 75K anatomical
classes + 168 properties) as `G=FMA_V1` in OGIT.

**Where.**
- New file: `crates/lance-graph-ontology/src/hydrators/owl.rs`.
- New data: `data/ontologies/fma.ttl` (downloaded at build time; not
  committed; SHA-pinned in manifest).

**What.**
- Parse FMA `.ttl` using `rio_xml` or equivalent OWL/RDF parser.
- Map each `owl:Class` to a `u32 entity_id` under `G=FMA_V1`.
- Map `rdfs:subClassOf`, `BFO:part_of`, and the FMA-specific
  properties (`supplies_blood_to`, `innervates`, `regional_part_of`,
  ...) to SPO-G triples in the `(S, P, O, G=FMA_V1)` quad store.
- Index by `(G, version, entity_id)` for O(1) lookup.

**Acceptance.** The SPARQL query
`SELECT ?cls WHERE { ?cls rdfs:subClassOf <http://purl.org/sig/ont/fma/fma_heart> }`
returns all 1500+ anatomical descendants of the heart in <10 ms,
measured cold on a developer laptop.

**Effort.** ~600 LOC (parser + hydrator + tests + golden TTL fixture).

**Dependencies.** W10's Tier-1 (SPO-G u32 slot + ContextBundle) merged.

---

### PR-ANATOMY-2: DICOM hydrator for medical scans

**Goal.** Ingest CT / MRI / PET scans from DICOM files into BindSpace as
voxel rows.

**Where.** New file:
`crates/medcare-rs/crates/medcare-realtime/src/dicom_hydrator.rs`.

**What.**
- Read DICOM headers (patient ID, modality, slice thickness, study UID,
  series UID) -> map to `MappingRow`.
- For each voxel: compute CAM-PQ palette index using the tissue-class
  classifier already shipped in `cam_pq/` -> write to BindSpace.
- Position index = `(study_uid, series_uid, slice_z, voxel_xy)`.
- ~10^9 voxels for full-body CT -> batched into `CycleAccumulator` ->
  flushed to Lance.

**Acceptance.** A 30 GB full-body CT ingests into BindSpace in <60 s on a
4-core developer machine; the query "voxels at slice 47" returns in
<100 ms.

**Effort.** ~400 LOC + tests.

**Dependencies.**
- `dicom-rs` crate (third-party).
- medcare-rs `ConsumerPointer` entry from W11's manifest.

---

### PR-ANATOMY-3: Anatomical-adjacency edge writer (FMA `part_of` -> SPO-G)

**Goal.** Per voxel, write SPO-G edges for FMA anatomical relationships
(`part_of`, `supplies_blood_to`, `innervates`, etc.) so that graph
traversal can answer "everything connected to the heart" in <50 ms.

**Where.**
- Extension to PR-ANATOMY-1's hydrator.
- New file: `crates/lance-graph/src/graph/spo/anatomy.rs`.

**What.**
- Read FMA's anatomical relationship triples.
- For each relationship type, write SPO-G edges with `G=FMA_V1`.
- AriGraph indexes (PR #355 SPO-1 bridge) provide string-keyed warm
  cache; fingerprint-keyed cold store provides O(1) graph traversal.

**Acceptance.** Query "everything connected to the heart" returns the
full 3-hop anatomical neighborhood in <50 ms.

**Effort.** ~300 LOC + tests.

**Dependencies.** PR-ANATOMY-1 merged.

---

### PR-ANATOMY-4: Q2 cockpit-server 3D voxel render view

**Goal.** Add a 3D canvas to cockpit-server's UI that renders a voxel cube
colored by FMA anatomical class.

**Where.**
- New file: `crates/cockpit-server/src/views/anatomy_3d.rs`.
- New frontend file: `cockpit/src/components/Anatomy3DView.tsx`.

**What.**
- Three.js or WebGPU-based 3D voxel renderer.
- Color voxels by FMA class (look up `G=FMA_V1 -> bundle.labels` for the
  display name).
- Streamed via SSE per topology I-3 BBB (scalar-only wire shape; no
  `Vsa16kF32` leaking through the wire).
- Click-to-query: clicking a voxel emits a Cypher query that
  GenericBridge admits via medcare-rs `ConsumerPointer`.

**Acceptance.** Loading a scan + clicking on the heart highlights all
1500+ anatomical descendants in <500 ms total user-perceived latency
(measured wall-clock from click to last voxel highlighted).

**Effort.** ~800 LOC server-side + ~600 LOC frontend.

**Dependencies.** PR-ANATOMY-1 / 2 / 3 shipped first.

---

### PR-ANATOMY-5: DeepNSM medical-vocabulary bundle slot

**Goal.** Per-G vocabulary slot for medical terminology (anatomical terms,
ICD-10 disease names, drug names) so that free-text findings parse to
SPO triples with correctly resolved anatomy.

**Where.**
- New data: `data/vocabularies/medcare.csv`,
  `data/vocabularies/fma_anatomy.csv`.
- Extension to W10's `ContextBundle.vocabulary` slot.

**What.**
- Curate ~few hundred medical terms with COCA-style frequency tagging.
- Map terms to u32 token IDs.
- Hydrate into `G=Healthcare` bundle's vocabulary slot at compile time
  via the build-script defined in W11's plan.

**Acceptance.** A clinician's free-text finding ("tumor in left lung,
suspicious for adenocarcinoma") parses to SPO triples with anatomical
terms correctly resolved via the medical vocab (specifically:
`lung_left` resolves to the FMA class, not the COCA generic
`lung_n_001`).

**Effort.** ~200 LOC + curation + tests.

**Dependencies.** W10's `ContextBundle.vocabulary` slot present; W11's
build-script manifest in place.

---

### PR-ANATOMY-6 (optional): Wave-mode similarity probe

**Goal.** "Find scans where this same anatomical pattern appears" -
Pattern N similarity via SplatShaderBlas-Bitpacked Jaccard.

**Where.** New file: `crates/jc/examples/anatomy_similarity_probe.rs`.

**What.**
- Compute scan-level `AwarenessPlane16K` from anatomy-class voxel
  statistics.
- Jaccard similarity (popcount-AND) across all scans in the dataset.
- Top-K nearest scans by anatomical signature.

**Acceptance.** Querying "patients with similar lung-anatomy pattern"
returns clinically-relevant matches (validated against a manually
labelled holdout set of ~50 scans).

**Effort.** ~300 LOC.

**Dependencies.** PR-ANATOMY-1 / 2 / 3 shipped. SplatShaderBlas-Bitpacked
already shipped (PR #347).

---

### PR-ANATOMY-7 (optional): Qualia17D feedback loop

**Goal.** When the radiologist interacts with the system, record their
cognitive state (Qualia17D, derived from convergence patterns of the
clinical-reasoning ThinkingStyle subset) into AriGraph core memory.

**Where.**
- Extension to the medcare-rs actor (the supervisor introduced in W11's
  plan).
- New entries in `EPIPHANIES.md` per radiologist session.

**What.**
- `ThinkingStyle` inherits per Pattern G: Healthcare =
  `Differential (+) EvidenceBased (+) RiskStratified`.
- Each query+result pair generates a Qualia17D fingerprint.
- High-salience patterns crystallize as epiphanies (e.g., "this is the
  third time I've seen this specific anatomic-symptom pattern -> flag
  for cohort study").

**Acceptance.** A clinical-research-mode flag in cockpit-server surfaces
high-salience cross-patient patterns; the third occurrence of a
specific anatomic-symptom pattern triggers a visible UI flag.

**Effort.** ~400 LOC.

**Dependencies.** PR-ANATOMY-4 (UI surface for the flag); qualia.rs
shipped (Pattern O); thinking-engine shipped (Pattern G).

---

## 4. Dependencies graph

```
W10 Tier-1 (SPO-G u32 + ContextBundle + GenericBridge)
  v
W11 Tier-2 (Manifest schema + Ractor supervisor)
  v
PR-ANATOMY-1 (FMA OWL hydrator) --------+
PR-ANATOMY-2 (DICOM hydrator) ----------|
PR-ANATOMY-3 (FMA SPO-G edges) ---------+
  v
PR-ANATOMY-4 (Q2 3D render view) -------- PR-ANATOMY-5 (Medical vocab)
                                              (parallel; no blocking arrow)
  v
PR-ANATOMY-6 (Anatomy similarity)
PR-ANATOMY-7 (Qualia feedback loop)
```

The two optional PRs at the bottom can be deferred without losing the
demo-able state. The minimum demo-able cut is PR-1 + PR-2 + PR-3 + PR-4.
PR-5 is highly recommended because Step 8 in section 1 depends on free-text
resolution; without PR-5, the radiologist types in clinical English and
the system can only recognize whatever DeepNSM's default COCA vocab
covers, which is generic and clinically inadequate.

---

## 5. Timeline (calendar, not effort)

| Phase | Weeks | Deliverables |
|---|---|---|
| Phase A - Tier-1 | Weeks 1-2 | W10 ships (SPO-G u32 + ContextBundle); W11 ships (manifest + ractor supervisor). |
| Phase B - Hydrators | Weeks 3-4 | PR-ANATOMY-1 + PR-ANATOMY-2 + PR-ANATOMY-3 merge in sequence. |
| Phase C - First demo-able cut | Weeks 5-6 | PR-ANATOMY-4 (Q2 3D view) + PR-ANATOMY-5 (medical vocab). **End of this phase = the system is demoable end-to-end.** |
| Phase D - Full vision | Weeks 7+ | PR-ANATOMY-6 (similarity probe) + PR-ANATOMY-7 (qualia feedback loop). |

The Phase-C boundary is the discrete "go/no-go" gate. If Phase A-C lands
on schedule, the architecture has been proven by a working artifact. If
Phase A-C slips by more than two weeks, the entropy ledger gets a row
acknowledging that some upstream pattern is heavier than estimated.

---

## 6. What the demo proves (pattern coverage matrix)

A clinician using the system end-to-end exercises:

| Pattern | What this demo exercises | Step |
|---|---|---|
| A (SPO-G u32 quads) | FMA anatomical edges as `(S,P,O,G=FMA_V1)`. | 3, 6, 8 |
| B (ContextBundle resolution) | `G=FMA_V1` and `G=Healthcare` bundles resolved per click. | 2, 3, 6 |
| C (GenericBridge admit) | Write of radiologist finding admitted via medcare-rs ConsumerPointer. | 1, 8 |
| D (OWL hydrator) | FMA TTL to bundle, ~0 hand-written Rust LOC for the ontology data. | 3, 6 (hydration step) |
| E (Compile-time admit) | medcare-rs compiled in; `G=Healthcare` active. | 1, 8 |
| F (Ractor supervisor) | Radiologist-actor messages routed via supervisor tree. | 1, 8 |
| G (ThinkingStyle inheritance) | Healthcare styles inherited from DOLCE: Differential + EvidenceBased + RiskStratified. | 10 |
| H (CognitiveShader dispatch) | `p64-bridge::CognitiveShader` dispatches per-G program. | 7 |
| I (CycleAccumulator pre-warm) | Background pre-warms answers as voxels stream. | 1, 4 |
| L (Graph-router cross-modal) | Cypher / SPARQL query co-resolves over SPO + Lance. | 6 |
| M (Wave-mode similarity) | Scan similarity via Jaccard popcount-AND. | 9 |
| N (Codebook similarity) | FMA class lookup = fingerprint to codebook to O(1). | 2, 3 |
| O (Qualia17D feedback) | Qualia17D records cognitive state per query. | 10 |

Coverage gaps that are **intentional** (not pretending the demo proves
what it doesn't):

- **Pattern J (Lance MVCC).** Exercised implicitly by every write; never
  surfaced to the user. Correct - Pattern J is plumbing.
- **Pattern K (SoA invariant).** Enforced by the column store; never
  surfaced. Correct - Pattern K is a code-review-time invariant, not
  a runtime UI feature.

---

## 7. Honest self-review

Three bullets, brutally honest.

- **The 10^9-voxel claim for a 30 GB full-body CT is a back-of-envelope
  number.** A real 30 GB CT is closer to 5x10^8 voxels depending on
  modality and resolution; the order of magnitude is right but the
  specific figure should be re-measured against the first real DICOM
  fixture in PR-ANATOMY-2. If the ingest budget of <60 s slips,
  PR-ANATOMY-2 might need a streaming-with-backpressure variant rather
  than a single batched ingest. Flagging this rather than burying it.

- **PR-ANATOMY-4's effort estimate (800 LOC server + 600 LOC frontend)
  is the highest-risk number in this plan.** Three.js / WebGPU voxel
  rendering at the 10^8-voxel scale is non-trivial; many real-world
  implementations resort to octree LOD which doubles the LOC. If the
  frontend grows past ~1.5K LOC, the PR should be split into
  PR-ANATOMY-4a (server-side anatomy_3d module + SSE wire) and
  PR-ANATOMY-4b (cockpit Anatomy3DView component). Calling this out
  now so it doesn't become a surprise.

- **The optional PRs (6 and 7) are flagged optional for a reason - they
  are research-grade, not infrastructure-grade.** PR-6's "clinically-
  relevant matches" acceptance criterion is the softest in this plan;
  the labelled holdout of ~50 scans does not exist yet and curating it
  is itself a multi-day task. PR-7's "high-salience pattern" definition
  is qualitative and depends on whether the radiologist actually
  agrees with what the system surfaces. Both are valuable for the
  vision but neither should block the Phase-C go/no-go gate.

---

## 8. Cross-references

- **Master plan.** `.claude/plans/unified-ogit-architecture-v1.md` (W1)
  - Section 5 ("Proof of vision") names this plan as the north-star
  demo; this document is the realization of that section.
- **Tier-1 sub-plan.** `.claude/plans/ogit-g-context-bundle-v1.md` (W10)
  - defines the SPO-G u32 slot, ContextBundle struct, and the
  GenericBridge admit contract that PR-ANATOMY-1 / 2 / 3 build on.
- **Tier-2 sub-plan.** `.claude/plans/compile-time-consumer-binding-v1.md`
  (W11) - defines the manifest schema and ractor supervisor that
  PR-ANATOMY-1 (manifest entry for `fma.ttl`) and PR-ANATOMY-2
  (`ConsumerPointer` for the DICOM hydrator) consume.
- **Tech-debt anchor.** `TD-ANATOMY-DEMO-8` in
  `.claude/board/TECH_DEBT.md` - this plan resolves that row.
- **Entropy ledger.** Once Phase-C ships, the ledger gets a row
  acknowledging that the architecture is end-to-end visible in a
  single binary; pattern-level entropy drops by one level for each
  pattern listed in section 6's coverage matrix.
- **Epiphanies.** Each radiologist session in Phase D generates
  candidate `EPIPHANIES.md` entries via PR-ANATOMY-7; these are gated
  on human review before commit.

---

*End of `anatomy-realtime-v1.md`. Sole deliverable of Worker Agent W12.*
