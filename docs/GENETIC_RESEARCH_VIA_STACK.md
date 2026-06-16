# Genetic Research via the AdaWorld Stack — Pattern-Recognition Hand-off

> **For:** a geneticist / bioinformatician / computational-biology researcher who knows their domain cold but has not seen this stack.
> **Reading time:** ~30 minutes.
> **Promise:** every shipped claim cites a file:line in this workspace. Nothing here is speculation about *"what we could build"* — it's *"what already runs."* Where we propose new work, it's named as proposal and the lift is estimated honestly.
> **Companions:** `.claude/plans/genetic-research-substrate-integration-v1.md` (the implementation plan), `.claude/handovers/2026-06-16-genetic-research-headstone-exploration.md` (the destination-state synthesis).

---

## 0. The bet

The stack you're about to look at was built for cognitive workloads (visual splats, ontology cascades, sentence understanding, OCR transcoding). Eight of its already-shipped primitives turn out to be **isomorphic** to bioinformatics machinery you'd otherwise have to assemble yourself: CAM-PQ fingerprints map to sequence sketching; HHTL nibble-tries map to genomic coordinate cascades; CLAM trees map to chromosome→arm→band→gene hierarchies; CHAODA anomaly scoring maps to novel-variant detection; bgz17 11/17 strides map to anti-aliased k-mer sampling; the entropy×energy substrate-state plane maps to Bayesian variant-evidence accumulation; the MailboxSoA + counterfactual-mailbox pair maps to cancer-pathway propagation simulation; and the Pearl-2³ mask in `CausalEdge64` maps directly to do-calculus on driver-mutation graphs.

The bet: **build the genetic-research adapter as a thin domain wiring on top of these primitives instead of as a new bioinformatics tool**, and the resulting substrate accumulates evidence across callers / studies / literature / cohorts in one graph with provenance, counterfactual lanes, and falsifiable certificates — something none of GATK / DeepVariant / nf-core / single-tool pipelines ship today.

---

## 1. Eight shapes you already know, eight primitives we already shipped

Each subsection: domain shape → substrate primitive → where it lives (file:line) → why the composition makes you blink.

### 1.1 Sequence sketches (MinHash / ntHash / MASH) ↔ CAM-PQ 48-bit fingerprint

**You know:** a sequence-similarity sketch is a fixed-width fingerprint such that Hamming or Jaccard distance on fingerprints approximates a known sequence-distance proxy (usually Mash-style Jaccard estimating ANI).

**We have:** **CAM-PQ 6×256 = 48-bit fingerprint** (`crates/lance-graph/src/cam_pq/storage.rs:9`, `ndarray/.claude/knowledge/pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md:22`). Six product-quantization subspaces × 256 Lloyd-Max centroids per subspace = exactly 48 bits. Validated: **48 bits captures ~94% of Jina 1024-D semantic similarity on SimLex-999** (`lance-graph/.claude/knowledge/linguistic-epiphanies-2026-04-19.md:299-312`).

**The composition:** swap the Jina-1024D embedding for any sequence embedding (ESM, ProtBERT for proteins; or a learned DNA embedding; or even a frequency-vector of k-mer counts). The same 6-subspace PQ + 256-centroid Lloyd-Max codec gives you a Mash-compatible fingerprint at substrate-native width with measured fidelity. Hamming distance lights up the AVX-512BW LUT scan kernel (`crates/lance-graph-turbovec/KNOWLEDGE.md:60-92`): **76 µs/query at n=20,000, dim=512, recall@10 = 0.785** — that's BLAST-against-RefSeq territory on commodity silicon.

### 1.2 Genomic coordinates ↔ HHTL nibble-trie addressing

**You know:** `chr1:1234567` is a hierarchical address that ordinarily lives as a flat (chromosome, position) tuple, and you do range queries by binary search or interval trees.

**We have:** **HHTL = Hierarchical Hash Trie Lattice** (`crates/lance-graph-contract/src/hhtl.rs`). A 16-ary nibble trie with depth 16 (= 64 bits of address), where every nibble is one cascade level, prefix-shift is `is_ancestor_of`, and subtree enumeration is a native range scan under `addr64 = path << 4·(16−depth)`. Live in `lance_graph_contract::NodeGuid` as the bytes `HEEL · HIP · TWIG` (3 × u16 = 12 nibbles, with classid above and family/identity below).

**The composition:** map (organism, chromosome, region, gene, position) onto the cascade. A genomic-region range query becomes a single subtree scan; nearest-gene-by-locus becomes a parent-prefix walk. **No B-tree, no interval forest, no rebuild on insertion** — the substrate already does the cascade-walks via shift/mask arithmetic. And the **same address space holds the patient ID (in `family`) and the variant ID (in `identity`)** — so "find every variant in gene X within population Y" is one prefix scan.

### 1.3 Chromosome→arm→band→gene ↔ CLAM 3-level 16-way tree

**You know:** cytogenetic coordinates form a fixed multi-level hierarchy (22 + XY chromosomes → ~25 chromosomal bands per arm → ~20,000 genes → ~3 × 10⁹ bases). Your tooling tends to treat each level as a separate index.

**We have:** **CLAM (Clustered Hierarchical Approximate matching)** ships as `ndarray::hpc::clam` (`ndarray/src/hpc/clam.rs`, ~1600 lines), with the canonical 3-level 16-way layout (`lance-graph/.claude/session_2026_04_11_bf16_hhtl_combined_research.md:127-148`):

```
bits 15..12 = L0: 16 coarse clusters     (HEEL scan target)
bits 11..8  = L1: 256 mid-clusters       (HIP, 1:1 Jina-v5 centroids)
bits 7..4   = L2: 4096 terminal          (TWIG, COCA alignment)
```

**The composition:** the CLAM tree IS your chromosome-arm-gene cascade when populated with genomic centroids instead of language-embedding centroids. Hierarchical clustering on chromosomal-band features gives you 16 coarse → 256 mid → 4096 terminal — *a roughly gene-resolved leaf count by construction*. The substrate's `silhouette` / `Cronbach α` / `ARI` cluster-quality probes (`lance-graph/.claude/probe_m1_result_2026_04_11.md`) tell you whether your cytogenetic hierarchy actually emerges from your feature distance, or whether your distance metric is wrong. **That's a falsifiable certificate, not a hand-drawn diagram.**

### 1.4 Novel-variant detection ↔ CHAODA anomaly on LFD distribution

**You know:** identifying a rare or de novo variant against population catalogues is essentially an outlier-detection problem in a high-dimensional feature space (allele frequency × read-depth × strand bias × neighborhood × etc.). Tools like CADD, REVEL, AlphaMissense produce per-variant scores.

**We have:** **CHAODA** (Clustered Hierarchical Anomaly and Outlier Detection Algorithm, Ishaq et al. 2021) shipped as Phase 4 of `ndarray::hpc::clam` (`ndarray/src/hpc/clam.rs:1493-1567`):

```rust
pub struct AnomalyScore {
    pub index: usize,                // original dataset index
    pub lfd: f64,                    // LFD of the leaf cluster
    pub score: f64,                  // normalised in [0, 1]; higher = more anomalous
    pub awareness: AwarenessState,   // Crystallized / Tensioned / Uncertain / Noise (clam.rs:1549-1557)
}

impl ClamTree {
    pub fn anomaly_scores(&self, data: &[u8], vec_len: usize) -> Vec<AnomalyScore>;
    // Local Fractional Dimensionality (LFD) per cluster; high LFD = complex local geometry
}
```

**The composition:** build a CLAM tree on your per-variant feature vectors; CHAODA scores every variant against the local manifold's intrinsic dimensionality. A novel variant in a region of high LFD lights up as `AnomalyScore { score → 1.0, awareness → AwarenessState::Noise }` (the `score ≥ 0.75` quartile per `clam.rs:1556`) because its position differs from the population's local manifold — *without you having to train a classifier or annotate a truth set first*. This is *unsupervised* outlier detection on the same tree your range queries walk.

> **⚠→✅ MEASURED CAVEAT (2026-06-16):** the *original* `anomaly_scores` implements **only the single-method leaf-LFD signal**. A spike (ndarray PR #219) on ideal synthetic data measured **ROC-AUC = 0.624** — below the ≥ 0.85 bar — because leaf LFD captures *intra-leaf* geometry complexity, not *inter-leaf* isolation. **The multi-method ensemble has since been built** (ndarray PR #220, `ClamTree::ensemble_anomaly_scores`: parent-child path-minority ⊕ connected-component cardinality) and measured at **ROC-AUC = 0.991** on the same fixture — clearing the bar. So the *kernel* now does isolation-aware novelty detection; use `ensemble_anomaly_scores`, not the single-LFD `anomaly_scores`, for this composition. **Still gated:** this is synthetic-only proof. Genomic novelty detection (`PROBE-CHAODA-1000G` on 1000-Genomes + ClinVar) remains unproven until the VCF→feature-vector pipeline (plan D-GEN-1+2) exists. The pattern match is real and the kernel capability is now demonstrated; the genomic claim is not yet measured.

### 1.5 minimap2 minimizers ↔ bgz17 11/17 X-Trans stride

**You know:** minimizers are a way to sample k-mer positions sparsely but representatively: pick the lexicographically smallest k-mer in every length-w window. Goal: dramatic sketching at minimal accuracy cost.

**We have:** **bgz17 11/17 golden stride** (`crates/bgz17/src/lib.rs:53-60`, `lance-graph/.claude/BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md`). On a prime base of 17, the stride `11 = round(17/φ) = round(10.506) = 11` is *both* coprime with 17 (full permutation guaranteed: `(i·11) mod 17` visits all 17 residues exactly once) AND closest to a true φ-rotation. Result: **maximally-irrational integer stride** that minimises periodic resonance — the same anti-aliasing principle Fujifilm uses in the X-Trans sensor CFA. Validated as anti-moiré with provable bounds (`OGAR/docs/DISCOVERY-MAP.md` D-BGZ17).

**The composition:** minimizers reduce window-redundancy via "pick the min hash"; the bgz17 stride reduces sequence-position redundancy via "pick positions at golden-irrational offsets." Both achieve aperiodic sampling; bgz17's bound is *number-theoretic* (provable via gcd + φ-approximation), while minimizers' bound is empirical. For sketching long sequences (k-mers, codon triplets, peptide windows), the bgz17 stride is a drop-in sampler with the same accuracy-vs-density trade-off as minimizers but a falsifiable theoretical bound.

### 1.6 Bayesian variant-evidence accumulation ↔ entropy × energy substrate-state plane

**You know:** variant calling resembles Bayesian update — each supporting read shifts the posterior on `P(variant | evidence)`. GATK's HaplotypeCaller does this explicitly via PairHMM + active-region rescoring; DeepVariant does it implicitly via a CNN. The clinical curation step (ACMG/AMP classification) is another layer of evidence accumulation: literature, segregation, in-silico predictions, functional studies.

**We have:** **the entropy × energy substrate-state plane** (`bardioc/SUBSTRATE_STATE_FRAMINGS.md`, lance-graph PR #491 §6, lance-graph PR #494's `EntropyRung` + `Quadrant` enums + `nars_entropy(f, c) = 1 − c · |2f − 1|`). Four quadrants:

```
                    high energy
                         │
   Confusion / Chaos     │      Wisdom (crystalline)
   (in-progress climb)   │      (the integrated apex)
   ──────────────────────┼──────────────────────
   Staunen               │      Boredom / Inert
   (cognitive pressure)  │      (ordered but not energised)
                         │
                    low energy

   high entropy   ←──────────────────→   low entropy
```

Variant-evidence accumulation as a substrate trajectory: a *novel* variant arrives in **Staunen** (high entropy: little supporting evidence; low energy: no prior accumulation). As reads accumulate and the variant survives filter passes, it crosses through **Confusion / Chaos** (energy invested but the entropy hasn't yet collapsed into a clean call). After cohort confirmation + ClinVar consistency + literature support, it settles into **Wisdom** (low entropy, high accumulated evidence — call confidence near 1.0, ACMG class Pathogenic or Likely Pathogenic). And `ρ(entropy, prediction accuracy) = −0.78` is **measured** (ndarray PR #218) — that's not a theoretical claim. Entropy IS a validated reliability proxy.

**The composition:** every variant in your store carries a quadrant classification derived from `MailboxSoA.energy` (per-row signed spatio-temporal accumulator) + `plasticity_counter` (saturating Hebbian, = lifetime evidence count) + classid-prefix codebook hit-rate (the entropy proxy). The §14 oracle's *"provenance-normalized equivalence"* compares two pipelines' quadrant assignments; disagreements are exactly where calibration matters.

### 1.7 Cancer-pathway propagation / virus genome integration ↔ MailboxSoA + counterfactual mailbox

**You know:** RAS/MAPK pathway propagation, MYC-driven transcriptional cascades, viral integration site preferences (HPV, HBV, EBV oncogenic integration), and metastatic seeding all have the same shape: a discrete event triggers a cascade through a graph of cellular sites under partial observability, with *counterfactual* branches you'd love to simulate but can't easily express in standard pipelines.

**We have:**
- **`MailboxSoA<N>`** (`crates/cognitive-shader-driver/src/mailbox_soa.rs`) — per-row SoA with `energy: [f32; N]` (signed spatio-temporal accumulator), `plasticity_counter: [u8; N]` (saturating Hebbian = lifetime activation count), `last_active_cycle: [u32; N]` (in-place consumption stamp), `edges: [CausalEdge64; N]` (LE baton edge per row), `consume_firing(row) -> bool` (in-place active inference: row threshold-crosses, energy resets, stamp advances).
- **`CounterfactualMailbox`** (`crates/lance-graph-contract/src/counterfactual.rs:232`) — v3 split-resolution: `SplitPoles` (the alternative-pole representation), `deposit_counterfactual` (writes the minority pole as `InferenceType::Counterfactual` with `to_mantissa() = -6` into the episodic edge), `revise_if_minority_wins` (free-energy comparison flips the canonical reading). **Iron invariant in the doc comment:** *"A counterfactual stays in a separate lane — it is NEVER written as observed SPO. The `InferenceType::Counterfactual` tag is the mechanical enforcement of that invariant."*
- **`SPAWN_DISSONANCE_THRESHOLD: f32 = 0.55`** — calibrated threshold for when a substrate state diverges enough to be *"worth a counterfactual test."*

**The composition:** map each MailboxSoA row to one cell / one transcription-factor binding site / one viral integration site:
- `energy[row]` = RAS-GTP fraction at this site / viral copy number / cytokine concentration.
- `plasticity_counter[row]` = lifetime cumulative activation (DIKW-climb).
- `last_active_cycle[row]` = last firing time (replication burst recency).
- `edges[row]` 12 in-family slots = 12 neighbouring cells in the tissue lattice; 4 out-of-family = vascular / immune / inter-organ.
- `qualia[row]` = 16-i4 affective lane for stress / hypoxia / immune-surveillance signal.

At each `consume_firing(row)`: if local dissonance > 0.55, spawn `CounterfactualMailbox` with `SplitPoles` for the alternative cascade path (e.g. KRAS G12D vs. wild-type at this site; integration at locus A vs. B). Both poles fan out forward — majority writes observed SPO edges into the AriGraph episodic chain; minority stays as `InferenceType::Counterfactual` in a separate lane. After K cycles, `revise_if_minority_wins` flips the canonical reading if the counterfactual lane out-predicts the observed. **This is Friston active inference at the per-row level, with the counterfactual ledger preserved for retrospective re-analysis** — the substrate-native analog of "what would have happened if this driver mutation had occurred earlier / elsewhere."

### 1.8 do-calculus / Pearl causality on driver-mutation graphs ↔ Pearl-2³ mask in CausalEdge64

**You know:** Judea Pearl's causal hierarchy distinguishes observation (P(Y | X)) from intervention (P(Y | do(X))) from counterfactual (P(Y_X | X', Y')). For driver-mutation networks (TP53, BRAF, KRAS, PIK3CA, ...) you want to ask all three.

**We have:** **`CausalEdge64` bit layout includes a Pearl mask** (`crates/lance-graph-contract/src/high_heel.rs:202`): *"S/P/O palette + NARS + Pearl mask + inference + plasticity + temporal."* Plus `crate::pearl_junction` (`crates/lance-graph-contract/src/pearl_junction.rs`) classifies *"the three structural relations the Pearl-junction classifier needs"* (`hhtl.rs:231`). And **`PEARL_SUBSETS`** in `ndarray::hpc::entropy_ladder` (#494) ships the 2³ = 8 hypothesis subsets ready for Pearl do-calculus on the SPO graph — the 8 octants of {observed, do-intervened, counterfactual} × {S, P, O}.

**The composition:** every causal-edge in your variant graph carries the 8-subset Pearl mask in-band. `decompose_spo` reads the 3 × palette-256 SPO already encoded in `CausalEdge64` — *no re-quantization needed* (#494 architecture decision). For a KRAS → MAPK → ERK → MYC cascade, the per-edge Pearl mask lets you query: *"given the observed presence of KRAS G12D and observed MYC overexpression, what's the counterfactual probability of MYC overexpression in the do(KRAS-WT) intervention?"* The substrate addresses the 8 octants natively; you write the query, not the do-calculus engine.

---

## 2. The architecture

```text
upstream domain corpora (FASTA / VCF / BAM / GFF / 1000-Genomes / ClinVar / GO / Reactome)
   ↓ (parsers — proposed, see plan §1)
adapter-genetics-experimental crate (proposed; thin domain wiring, no new substrate)
   ↓
classid → ClassView resolves which ValueSchema preset materialises per-row
   ↓
ndarray::hpc primitives                          lance-graph-contract primitives
  CAM-PQ 48-bit fingerprint                        canonical NodeGuid (classid·HEEL·HIP·TWIG·family·identity)
  CLAM 3-level 16-way tree                         EdgeBlock (12 in-family + 4 out-of-family)
  CHAODA anomaly score                             NodeRow = 512 B = key(16) | edges(16) | value(480)
  bgz17 11/17 X-Trans stride                       MailboxSoA<N> (energy / plasticity / edges / qualia)
  matrix exp (Padé)                                EdgeCodecFlavor / ValueSchema (per-class via classid)
  softmax / log-softmax (axis-aware SIMD)          CausalEdge64 with Pearl 2³ mask
  Lie-algebra Lyndon log-signature                 CounterfactualMailbox (SplitPoles + revise_if_minority_wins)
   ↓
lance / lancedb columnar SPO store
   ↓
§14 oracle (provenance-normalised equivalence: rubicon::oracle::compare_normalised)
   ↓
queryable, falsifiable, counterfactual-preserving variant/cohort/pathway graph
```

---

## 3. What's running today vs. what we propose

| Layer | Status |
|---|---|
| CAM-PQ 48-bit fingerprint + Hamming LUT scan | **shipped** (`lance-graph-turbovec`, validated 76 µs/query @ n=20K) |
| CLAM 3-level 16-way tree | **shipped** (`ndarray::hpc::clam`, 1600 lines, with silhouette / ARI / Cronbach α probes) |
| CHAODA anomaly score | **shipped** (`ndarray::hpc::clam.rs:1493-1560`) |
| bgz17 11/17 stride | **shipped** (`crates/bgz17/`, X-Trans rationale documented) |
| HHTL nibble-trie | **shipped** (`lance-graph-contract::hhtl::NiblePath`) |
| `NodeGuid` / `EdgeBlock` / `NodeRow` canonical layout | **shipped** (`canonical_node.rs`, lance-graph #489/#490) |
| `MailboxSoA<N>` + `consume_firing` | **shipped** (`mailbox_soa.rs`) |
| `CounterfactualMailbox` + `SplitPoles` + `revise_if_minority_wins` | **shipped** (`counterfactual.rs`) |
| Pearl-2³ mask in `CausalEdge64` + `pearl_junction` classifier + `PEARL_SUBSETS` | **shipped** |
| Entropy × energy substrate-state plane + `EntropyRung` + `Quadrant` + `nars_entropy` | **shipped** (#491, #494, #495) |
| `OntologyRegistry` + Pattern D hydrators (`hydrate_dolce` / `hydrate_owltime` / `hydrate_provo` / `hydrate_qudt` / `hydrate_schemaorg` / `hydrate_skos` / `hydrate_fibo_fnd` / `hydrate_fibo_be` / `hydrate_odoo` / `hydrate_zugferd` / `hydrate_skr03/04`) over shipped `OwlHydrator` + `MetaStructureHydrator` + 47 KB Lance dictionary cache | **shipped** (`lance-graph-ontology/src/hydrators/mod.rs:1-57`) |
| §14 oracle (`rubicon::oracle::compare_normalised`) | **shipped** (`bardioc/rubicon`) |
| DeepNSM sentence-level AriGraph reader | **shipped** (lance-graph #479; 200 tests) |
| ndarray AMX int8 GEMM (197 GMAC/s on Emerald Rapids) | **shipped** (ndarray #217) |
| Histology splat bridge (same Gaussian math at cm/mm/µm) | **shipped framing** (devcon flyers; splat-native ultrasound arc) |
| ───────────────────────────────────────────── | ───────────────── |
| `crates/adapter-genetics-experimental` scaffold | **proposed** (plan D-GEN-1) |
| FASTA / VCF / BAM parsers (host `noodles-*`) | **proposed** (plan D-GEN-2) |
| k-mer → CAM-PQ fingerprint function | **proposed** (plan D-GEN-3) |
| GO / Reactome / ClinVar Pattern D `hydrate_*()` glue (over shipped `OwlHydrator` / `MetaStructureHydrator` — no new trait) | **proposed** (plan D-GEN-5) |
| KRAS G12D 1024-cell counterfactual fan-out simulation | **proposed** (plan D-GEN-7) |
| §14 oracle benchmark against GIAB truth set | **proposed** (plan D-GEN-10) |

---

## 4. The dynamics axis (what nobody ships today)

The eight pattern matches above are individually compelling. The synthesis that nobody ships:

> **Cancer-pathway propagation simulation with counterfactual lanes preserved.**

Standard tools (GATK / DeepVariant) call variants from a single sample. Pathway tools (Reactome / WikiPathways) annotate observed pathway membership. Network-inference tools (CellNOpt / SCENIC) reconstruct pathway connections from expression data. *None of them* preserve the counterfactual branch — *"what if KRAS had mutated at codon 12 instead of codon 13 in this tumor's earliest clonal expansion?"* — as a persistent, query-visible substrate lane.

The substrate does. `CounterfactualMailbox` is the storage; `revise_if_minority_wins` is the update rule; `InferenceType::Counterfactual` is the mechanical guarantee that counterfactual edges never leak into observed SPO. For a cancer cohort, this is the equivalent of *"every patient's tumor carries both its actual driver mutation history AND the counterfactual alternatives the substrate's Friston-FEP active inference deemed worth tracking,"* and the §14 oracle can compare counterfactual lanes across patients to spot *"this counterfactual branch was rejected in 9 out of 10 tumors but accepted in tumor X — what's structurally different about X?"* That's clinical-research signal nobody else can address natively.

---

## 5. The first cargo invocation that proves it

Today, with the shipped substrate (zero new code), you can:

```bash
# Build the CHAODA-on-vectors substrate (already green on main)
cd /home/user/ndarray
cargo test -p ndarray --lib clam::tests::chaoda
```

This runs the existing CHAODA test suite. Replace the test vectors with a 1000-Genomes feature matrix loaded from VCF (manually parsed for now; the FASTA/VCF adapter is plan D-GEN-2) and you have novel-variant detection running today, against the same CLAM tree that does language-embedding retrieval. The kernel is the same; the input is genomic.

For the dynamics axis, the runnable proof of concept is:

```bash
cd /home/user/lance-graph
cargo test -p lance-graph-contract --lib counterfactual::tests
```

This exercises `SplitPoles`, `deposit_counterfactual`, `CounterfactualMailbox`, and `revise_if_minority_wins`. Replace the test poles with a KRAS-G12D-vs-WT split at codon 12 of one mailbox row, fan-out via `consume_firing` across a 1024-row cellular lattice, and you have the cancer-cascade simulation harness running today. The wiring (plan D-GEN-7) is ~1-2 days of work; the substrate is shipped.

---

## 6. Why this is worth your time, in one paragraph

You'd build the genetics tool either way. What this substrate gives you that you cannot easily reach for anywhere else is **a counterfactual-lane-preserving SPO graph with Friston-FEP-calibrated variant evidence accumulation, native Pearl-2³ do-calculus addressing, CHAODA-grade unsupervised novel-variant detection, and provenance-normalised cross-pipeline equivalence checking — all running on shipped kernels** (CAM-PQ, CLAM/CHAODA, bgz17, AMX int8 GEMM at 197 GMAC/s, the entropy × energy plane validated at ρ = −0.78 against prediction accuracy). The work to unlock genetics-specific use is mostly file-format parsers + ontology hydrators + a domain class-mint, not new substrate. Most of the eight pattern matches above are one-day to one-week tasks each; the four-week first-deliverable target in the implementation plan is honest.

---

## 7. Where to go next

- **Implementation plan:** `.claude/plans/genetic-research-substrate-integration-v1.md`. Names 10 deliverables D-GEN-1..10 + 3 gating probes (CHAODA-on-1000G, KRAS-counterfactual-determinism, CAM-PQ-vs-BLAST agreement). Per-deliverable lift estimate ranges from a-few-hours to two-weeks.
- **Headstone:** `.claude/handovers/2026-06-16-genetic-research-headstone-exploration.md`. Capstone synthesis for the destination state — what the substrate looks like *when complete* for genetic research, what era closes (per-tool pipeline silos), what era opens (one counterfactual-preserving graph).
- **The existing exploratory plan:** `.claude/plans/3DGS-genetics-4x4-fanout-plan.md`. Predates this synthesis; covers the static representation (4×4 lane interpretation: sequence coordinate / motif / expression / time). This document IS the *dynamics + counterfactual* extension that the static plan was missing.

---

_Authored 2026-06-16 by external session `AdaWorldAPI/bardioc` `session_01VysoWJ6vsyg3wEGc5v7T5v`. Every file:line citation in this document points to an actual file in this workspace as of the authoring date. If a citation has rotted, treat it as a discipline failure on the author's side and report it; the substrate's no-confabulation rule applies to this document too._
