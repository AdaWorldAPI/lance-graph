# Genetic Research Headstone Exploration — lance-graph

## Purpose

This document is a headstone exploration for the full line of thought connecting:

```text
upstream domain corpora (FASTA / VCF / BAM / GFF / 1000-Genomes / ClinVar / GO / Reactome / htslib)
  ndarray::hpc::clam + CHAODA   (3-level 16-way clustering + LFD anomaly)
  ndarray::hpc::cam_pq           (6 × 256 = 48-bit Lloyd-Max fingerprint; 94 % of Jina 1024-D)
  ndarray::hpc::activations      (exp / log / softmax / matrix exp / Lie-algebra Lyndon log-signature)
  ndarray::hpc::amx_matmul       (197 GMAC/s int8 GEMM on Emerald Rapids)
  bgz17                          (11/17 X-Trans stride for anti-moiré k-mer sampling)
  lance-graph-contract           (canonical NodeGuid · EdgeBlock · NodeRow; HHTL nibble-trie; MailboxSoA<N>; CounterfactualMailbox)
  lance-graph-ontology           (OntologyRegistry + TtlHydrators + 47 KB Lance dictionary cache + wikidata_hhtl)
  lance-graph-arm-discovery      (reliability suite: Pearson / Spearman / Cronbach α / ICC(2,1))
  deepnsm                        (sentence-level AriGraph reader; P64 / Cam4096 / Crystal4096)
  rubicon                        (§14 oracle: compare_normalised with provenance fields)
  adapter-genetics-experimental  (NEW — thin domain wiring; not yet built)
```

The goal is to preserve the architectural synthesis for genetic research *before* implementation details scatter it into separate plans — so a domain expert walking up to the substrate lands on the destination shape, not on the next tactical PR.

---

## Capstone thesis

```text
Bioinformatics ships pipelines.
Each pipeline is a tool with its own grammar, its own confidence calibration,
its own provenance discipline (usually informal), and its own failure modes.

The substrate ships a SHAPE.
Every variant, every annotation, every counterfactual, every cohort summary
takes the same shape: a row in MailboxSoA<N>, with an entropy×energy plane
classification derived from its Hebbian plasticity counter and its
last-active-cycle stamp, with adjacency in EdgeBlock, with class-resolved
value tenants in the 480-byte slab, with NARS truth (frequency, confidence)
calibrating its provenance, with Pearl-2³ subset addressing in CausalEdge64
for do-calculus, with counterfactual minority poles preserved in a separate
lane via InferenceType::Counterfactual.

One graph, every domain layer composes:
  sequence sketches as CAM-PQ fingerprints;
  genomic coordinates as HHTL cascade addresses;
  novel variants as CHAODA anomaly scores against LFD distribution;
  pathway propagation as MailboxSoA active-inference fan-out;
  counterfactual driver-mutation histories as CounterfactualMailbox lanes;
  causality as Pearl-2³ subset queries;
  literature evidence as DeepNSM SPO triples;
  cross-pipeline equivalence as §14 oracle verdicts.

The genetic researcher's job is the domain mint:
  which classid for which entity?
  which ontology hydrator for which vocabulary?
  which counterfactual gates the next study question?

Everything below the domain mint is shipped.
```

---

## The four-layer architecture (from a geneticist's vantage)

### Layer 0 — Domain corpora (upstream, never vendored)

FASTA / FASTQ / VCF / BAM / GFF / annotation databases. They live at their canonical homes:
- 1000-Genomes / GIAB at NCBI / EBI.
- ClinVar at NCBI ClinVar releases.
- GO / Reactome / Sequence Ontology at OBO Foundry.
- Reference genome assemblies at UCSC / Ensembl / NCBI.

The adapter `TtlHydrator`s point at canonical releases and pin a version; the corpora do not move into the substrate.

This layer answers:

```text
which release of which reference is pinned
where the truth set / ontology / annotation lives
who owns its evolution
how the substrate consumer treats version cadence (GO monthly, ClinVar weekly, Reactome quarterly)
```

### Layer 1 — Adapter (`adapter-genetics-experimental`, proposed)

Thin domain wiring. **Zero new substrate primitives.** Mirrors `OcrProvider` engine-agnostic boundary from `lance-graph` #498's `LayoutBlock::to_node_row` transcode. Defines:

- `GenomicSubstrate` trait (the seam).
- Parsers (host `noodles-fasta`, `noodles-vcf`, `noodles-bam` — pure-Rust bioinformatics ecosystem).
- `Cam6x8` k-mer fingerprint function (calls into shipped CAM-PQ codec).
- `VcfRecordTranscoder` (mirrors OCR `LayoutBlock → NodeRow`).
- Class-mint registry sync with `lance-graph-ontology`.
- Per-class `ClassView::value_schema` selection (rides existing `Full` / `Compressed` presets — no new variant per #500's contract test).

This layer answers:

```text
how a VCF record becomes a NodeRow
how a k-mer becomes a 48-bit fingerprint
which classid identifies a Variant / Gene / Pathway / Cell / IntegrationSite
which ValueSchema preset materialises which tenant for which class
```

### Layer 2 — Substrate primitives (shipped)

The CAM-PQ codec, CLAM tree, CHAODA anomaly scoring, bgz17 stride, ndarray AMX int8 GEMM, lance-graph-contract canonical NodeGuid + EdgeBlock + NodeRow, MailboxSoA<N>, CounterfactualMailbox, Pearl-2³ in CausalEdge64, OntologyRegistry, DeepNSM sentence reader, §14 oracle. All file:line-grounded in `docs/GENETIC_RESEARCH_VIA_STACK.md` §1 + §3.

This layer answers:

```text
what every substrate primitive provides at the kernel level
which file:line carries the canonical implementation
which test suite proves green on main
which gating probe has been run (where green) vs. is queued (where speculative)
```

### Layer 3 — Research consumer

The geneticist's queries, in the substrate's native vocabulary:

- *"Find every variant in gene X within population Y in the bootstrap basin."* → one HHTL prefix scan.
- *"Score variant V against population local manifold for novelty."* → one CHAODA call on the CLAM tree.
- *"Sketch a 100 kb genomic region for cohort-wide similarity search."* → one bgz17-strided CAM-PQ fingerprint.
- *"Compare GATK calls vs. DeepVariant calls for sample S with provenance preserved."* → one §14 oracle invocation.
- *"Simulate KRAS-G12D-vs-WT counterfactual propagation in a 1024-cell tumor lattice."* → one `MailboxSoA<1024>` instantiation + `CounterfactualMailbox` for the G12D-vs-WT split.
- *"Extract gene-disease associations from PubMed abstracts."* → DeepNSM sentence reader over the corpus.

This layer answers:

```text
what queries the geneticist asks
which substrate primitive each query consumes
which ontology hydrator each query references
where the falsifiable certificate for each query result lives
```

---

## Why bioinformatics pipelines alone are not enough

Each existing tool answers part of the question. None compose into a single counterfactual-preserving graph.

| Tool family | What it gives | What it doesn't |
|---|---|---|
| GATK / DeepVariant / bcftools | Per-sample variant calls with caller-specific confidence | No cross-caller provenance reconciliation; no counterfactual lane; no entropy×energy substrate-state calibration |
| BLAST / Diamond / minimap2 | Sequence similarity rankings | No fingerprint-substrate integration; no SPO emission; no graph-native composition |
| Reactome / WikiPathways / KEGG | Annotated pathway membership | Static; no counterfactual propagation; no Friston-FEP evidence calibration |
| CADD / REVEL / AlphaMissense | Per-variant deleteriousness scores | Trained classifier required; no unsupervised novel-variant flag; no LFD anomaly grounding |
| CellNOpt / SCENIC / etc. | Network inference from expression | No counterfactual lane preservation; no Pearl-2³ do-calculus addressing |
| nf-core / Snakemake pipelines | Reproducibility via workflow management | Workflow-level, not graph-native; no §14 oracle equivalence checking; no provenance-normalised cross-pipeline comparison |

The substrate's composition gives the missing piece: **one SPO graph, accumulating evidence across all of these consumers, with counterfactual lanes preserved, with entropy×energy quadrant classification per variant, with Pearl-2³ do-calculus addressing, and with §14 oracle equivalence as the falsifiable cross-tool benchmark.**

---

## Why building genomics tooling from scratch is not enough

You'd reach for the shipped substrate even if you started from scratch, because:

- The CAM-PQ codec is mature (PR #482 ratified the canon; ndarray PR #218 measured fidelity).
- The CLAM tree + CHAODA scoring is mature (~1600 lines, validated probes).
- The AMX int8 GEMM is real silicon performance (197 GMAC/s, ndarray PR #217 measured).
- The entropy × energy plane is empirically validated (ρ(entropy, prediction accuracy) = −0.78 measured, ndarray PR #218).
- The reliability stats (Pearson / Spearman / Cronbach α / ICC) are shipped (ndarray PR #218).
- The CounterfactualMailbox is shipped with its iron invariant mechanically enforced.
- The §14 oracle is in production use for OCR caller comparison (post-#498).
- The OntologyRegistry has TTL hydrators for SKOS / FIBO / Odoo / ZUGFeRD as proven patterns.

Building these from scratch is N person-years of work. The lift to genetic-research-via-substrate is the **domain wiring** — measured in days to weeks per deliverable in `genetic-research-substrate-integration-v1.md`.

---

## Invariants

These are what the substrate enforces; the genetic-research adapter inherits them.

1. **§0 anti-invention guardrail** (lance-graph #496): no new `ValueSchema` variant; no new substrate types. Genetic-research-specific work is *wiring*, not new substrate.
2. **No-new-enum-variant contract test** (lance-graph #500): genomic classes ride existing `Full` / `Compressed` presets via `classid → ClassView`. **Do not propose a `ValueSchema::Genetic`.**
3. **Counterfactuals stay in their own lane** (`counterfactual.rs` iron invariant): `InferenceType::Counterfactual` mantissa = -6; never written as observed SPO.
4. **Closed-vocab discipline** (ruff PR #5 `predicate_count_locked_at_N`): new genetic predicates land in `ruff_spo_triplet::Predicate` under the locked-count gate.
5. **No C++ source vendored into Rust-target crates.** htslib stays upstream; if a transcoded version is wanted, route through `ruff_cpp_spo` (cross-repo handover at `AdaWorldAPI/ruff`).
6. **Five-specialist drift-catching pass** (lance-graph #500): `cascade-architect` / `family-codec-smith` / `palette-engineer` / `dto-soa-savant` / `truth-architect` review before any FINDING-grade claim.
7. **Gating probes before FINDING**: `PROBE-CHAODA-1000G`, `PROBE-KRAS-COUNTERFACTUAL-DET`, `PROBE-CAM-PQ-VS-BLAST` gate the substrate's claims to bioinformatics audiences.
8. **Boundary: representation + research tooling only.** No medical/diagnostic claims (per the predecessor `3DGS-genetics-4x4-fanout-plan.md`).

---

## What "complete" looks like

The headstone is reached when:

1. **`adapter-genetics-experimental` compiles** and the locked-shape test passes (the *"shape locked"* milestone analogous to `ruff_ruby_spo` PR #4).
2. **FASTA + VCF round-trip into `NodeRow`** via the `VcfRecordTranscoder`. The first measurable artifact: load 1000-Genomes Phase 3 chromosome 22 into the substrate and round-trip back to VCF, byte-identical for the chr 22 subset.
3. **CHAODA on 1000-Genomes feature vectors** produces ROC-AUC ≥ 0.85 on the held-out novel-singleton test (PROBE-CHAODA-1000G green).
4. **CAM-PQ-vs-BLAST agreement** measured: Spearman ρ ≥ 0.7 on top-100 RefSeq similarity rankings (PROBE-CAM-PQ-VS-BLAST green).
5. **KRAS G12D 1024-cell counterfactual fan-out** simulation runs deterministically (PROBE-KRAS-COUNTERFACTUAL-DET bit-exact across runs), and the observed-lane oncogenic-transformation rate matches published outcomes within tolerance.
6. **GO / Reactome / ClinVar TTL hydrators** load into `OntologyRegistry` and ontology cache invalidation works on Lance version bump.
7. **§14 oracle benchmarks GATK vs. DeepVariant** against GIAB HG002 truth set with F1 meeting published minima.
8. **DeepNSM genetic-language reader probe** demonstrates `P64` projection consistency on protein-coding sequences (versus structured-noise on non-coding).
9. **Histology splat extension** carries per-splat genomic profile via `Full` ValueSchema with `Fingerprint` + `HelixResidue` tenants populated.

When these nine hold, the substrate has fulfilled its purpose as the genetic-research foundation: one graph, every domain layer composes, counterfactual lanes preserved, falsifiable certificates everywhere.

---

## Headstone state — what the era closes

```text
The era that closes:
  - Per-tool pipeline silos with no cross-tool provenance reconciliation.
  - Variant calling and pathway annotation as separate worlds with no
    shared substrate.
  - Counterfactual driver-mutation histories thrown away because no tool
    preserves them.
  - Reproducibility via workflow managers rather than substrate-native
    provenance + §14 oracle equivalence.
  - Outlier detection requiring trained classifiers (CADD / REVEL etc.)
    because no unsupervised LFD-based novel-variant detector existed at
    bioinformatics scale.
  - "Bioinformatics builds its own tools" as the default assumption.

The era that opens:
  - One SPO graph accumulating cross-tool, cross-paper, cross-cohort
    variant evidence with provenance preserved.
  - Counterfactual driver-mutation lanes queryable for retrospective
    analysis ("what if KRAS had mutated at codon 13 instead of 12?").
  - CHAODA unsupervised novel-variant detection on the same CLAM tree
    the substrate uses for language retrieval.
  - Pearl-2³ do-calculus native in CausalEdge64 for cancer-pathway
    causal queries.
  - The Friston entropy×energy substrate-state plane as the calibrated
    confidence axis for every variant in the store.
  - The same Gaussian-splat math at cm (organ), mm (lesion), and µm
    (cell / histology slide) scales, with the splat carrier extending
    to per-cell genomic profile.
  - Cross-pipeline equivalence checking as a substrate-native operation
    (§14 oracle), not a workflow manager's afterthought.
  - Domain experts wiring the genetic-research adapter, not rebuilding
    the substrate.
```

The capstone thesis at the top of this doc is the one-line restatement of the open-era state.

---

## Cross-references

### This repo (`AdaWorldAPI/lance-graph`)
- `docs/GENETIC_RESEARCH_VIA_STACK.md` — the *why* doc for a domain expert, file:line-grounded.
- `.claude/plans/genetic-research-substrate-integration-v1.md` — the implementation plan (10 deliverables + 3 probes).
- `.claude/plans/3DGS-genetics-4x4-fanout-plan.md` — predecessor static-representation plan.
- `.claude/plans/3DGS-cross-pollination-raw-field-plan.md` — sibling cross-domain plan (ultrasound + neuronal + genetics share the raw-field backbone).
- `crates/lance-graph-contract/src/canonical_node.rs` — `NodeGuid` / `EdgeBlock` / `NodeRow` (the row substrate).
- `crates/lance-graph-contract/src/counterfactual.rs` — `SplitPoles` / `CounterfactualMailbox` / `revise_if_minority_wins`.
- `crates/lance-graph-contract/src/hhtl.rs` — `NiblePath` HHTL nibble-trie.
- `crates/lance-graph-contract/src/high_heel.rs:202` — `CausalEdge64` bit layout (Pearl mask included).
- `crates/lance-graph-ontology/` — `OntologyRegistry` + TTL hydrators.
- `crates/lance-graph/src/cam_pq/storage.rs` — CAM-PQ 48-bit fingerprint storage.
- `crates/lance-graph-turbovec/KNOWLEDGE.md` — TurboQuant + LUT-ADC kernel (76 µs/query measured).
- `crates/cognitive-shader-driver/src/mailbox_soa.rs` — `MailboxSoA<N>` + `consume_firing`.
- `crates/deepnsm/` — sentence-level AriGraph reader.

### Sibling repo (`AdaWorldAPI/ndarray`)
- `src/hpc/clam.rs:1493-1560` — CHAODA Phase 4 anomaly scoring on LFD distribution.
- `src/hpc/amx_matmul.rs` — int8 GEMM at 197 GMAC/s.
- `src/hpc/activations.rs` — exp / log / softmax / softmax-backward.
- `src/hpc/linalg/mat_exp.rs` — matrix exponential via Padé.
- `crates/bgz17/src/lib.rs:53-60` — 11/17 X-Trans stride constants.

### Upstream substrate context (`AdaWorldAPI/lance-graph` PRs)
- PR #491 — entropy × energy framing; SoA migration diff resolution.
- PR #494 — `EntropyRung` + `Quadrant` + `nars_entropy(f, c)`.
- PR #495 — 3-byte EdgeRef witness; reliability ⊥ causality empirically.
- PR #496 — `ValueSchema` presets + §0 anti-invention guardrail.
- PR #498 — GUID decode→read-mode keystone; helix `Signed360`; OCR→NodeRow transcode template.
- PR #500 — rebaseline + no-new-variant contract test + gating-probes pattern.

### External cross-repo
- `AdaWorldAPI/bardioc/SUBSTRATE_STATE_FRAMINGS.md` — entropy × energy plane framing (durable bardioc-side architectural doc).
- `AdaWorldAPI/bardioc/.claude/handovers/2026-06-16-session-handover.md` — bardioc session handover (covers the preceding session's confabulation pattern + discipline lessons).
- `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-headstone-exploration.md` — sibling headstone for the C++ harvester (relevant if htslib transcoding wanted).
- `OGAR/docs/CASCADE-SYNERGIES-EPIPHANY.md` — Morton-cascade × palette256 × golden-helix synthesis (foundational to the genetics fanout).
- `OGAR/docs/DISCOVERY-MAP.md` — D-CASCADE / D-MOIRE / D-BGZ17 ledger entries.

### Workspace headstones (for shape reference)
- `lance-graph/.claude/plans/3DGS-Cesium-BindSpace4-headstone-exploration.md` — the headstone shape this document follows.
- `bardioc/ROADMAP_RUST_PRIMARY_HEADSTONE.md` — Phase A→I migration headstone.
- `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-headstone-exploration.md` — sibling headstone (C++ harvester).
- `AdaWorldAPI/tesseract-rs/.claude/handovers/2026-06-16-tesseract-rs-headstone-exploration.md` — sibling headstone (Rust target).

---

_Authored 2026-06-16 by external session `AdaWorldAPI/bardioc` `session_01VysoWJ6vsyg3wEGc5v7T5v`. Headstone shape — preserves the architectural synthesis for what genetic-research-via-substrate IS when complete. Companion pattern-recognition hand-off at `docs/GENETIC_RESEARCH_VIA_STACK.md` carries the *why*; companion implementation plan at `.claude/plans/genetic-research-substrate-integration-v1.md` carries the *how*. No code, no PR for substrate changes — synthesis-preservation only._
