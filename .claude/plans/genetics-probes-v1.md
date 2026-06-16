# Genetics Substrate — Gating Probes v1

> **Type:** plan (probe queue for the `adapter-genetics-experimental` family).
> **Status:** PLANTED 2026-06-16 — promotes the three probes named in
>   `.claude/plans/genetic-research-substrate-integration-v1.md` §2 from "named"
>   to "fully-specified with file:line citations + pass/fail criteria locked."
> **Why:** the genetic-research plan makes load-bearing claims — *"CHAODA detects
>   novel variants without trained classifier"* / *"KRAS counterfactual fan-out
>   is deterministic"* / *"CAM-PQ 48-bit fingerprint approximates sequence
>   similarity"* — that gate the entire D-GEN-1..10 spend (~9 weeks). Per the
>   workspace insight-update cycle (CLAUDE.md: Claim → Probe → Run →
>   FINDING/correct), these probes settle each claim BEFORE the adapter crate
>   is funded.
> **Cross-ref:** `.claude/plans/genetic-research-substrate-integration-v1.md`,
>   `docs/GENETIC_RESEARCH_VIA_STACK.md`,
>   `.claude/handovers/2026-06-16-genetic-research-headstone-exploration.md`.

---

## Sequencing

| Phase | Probe | Cost | Status | Gates |
|---|---|---|---|---|
| **P0** | PROBE-CHAODA-1000G | ~3 days (after D-GEN-1+2) | ⚠ **spike RUN — AUC 0.624, BELOW bar** (ndarray #219) | The "CHAODA-as-novelty-detector" line of the entire plan |
| **P1** | PROBE-KRAS-COUNTERFACTUAL-DET | ~2 days (included in D-GEN-7) | queued | D-GEN-7 flagship dynamics-axis claim |
| **P2** | PROBE-CAM-PQ-VS-BLAST | ~1 week | queued | D-GEN-3 sequence-fingerprint claim |

**⚠ Blocker surfaced 2026-06-16:** the P0 spike (ndarray #219) shows the shipped
single-method leaf-LFD `anomaly_scores` reaches only AUC 0.624 on ideal synthetic
data. Porting the multi-method CHAODA ensemble (Ishaq et al. 2021) is now a
**prerequisite for PROBE-CHAODA-1000G**, ahead of any genomic-fixture work. See
the ⚠ FINDING under PROBE-CHAODA-1000G below.

**Critical-path note:** PROBE-CHAODA-1000G is the single highest-leverage probe.
If it fails (AUC < 0.85 on novel-variant detection against ClinVar Pathogenic
held-out), the unsupervised-novel-variant story collapses regardless of every
other adapter deliverable. PROBE-CAM-PQ-VS-BLAST is the next-most-load-bearing
because it gates the entire sequence-similarity composition (D-GEN-3 → D-GEN-10
benchmark relies on it). PROBE-KRAS-COUNTERFACTUAL-DET is the cheapest of the
three (substrate's no-randomness invariant should make this near-trivial; the
probe is a regression gate, not a discovery gate).

---

## PROBE-CHAODA-1000G — unsupervised novel-variant detection

### Claim under test

> *"CHAODA detects novel variants without trained classifier"* —
> `docs/GENETIC_RESEARCH_VIA_STACK.md` §1.4. The substantive form: the
> shipped CLAM tree + LFD-based CHAODA anomaly scoring at
> `ndarray/src/hpc/clam.rs:1498` (`AnomalyScore` struct) /
> `:1517` (`anomaly_scores` method) separates known-Pathogenic novel
> singletons from common population variants at ROC-AUC ≥ 0.85 on a
> held-out test fold drawn from 1000-Genomes Phase 3 + ClinVar.

### ⚠ FINDING (spike substitute RUN 2026-06-16) — single-method LFD is BELOW the bar

The 1-day spike substitute (see §Cost below) has been **run** against the
shipped kernel (`ndarray` PR #219, `test_chaoda_flags_novel_outliers_in_genetics_like_mixture`).
On a 5-lane Gaussian mixture — three tight "common" clusters + eight
deliberately extreme "novel" outliers, thermometer-encoded so Hamming
distance is monotone in per-lane L1 magnitude — the shipped single-method
leaf-LFD `anomaly_scores` measured:

| metric | value |
|---|---|
| mean cluster score | 0.6749 |
| mean outlier score | 0.7500 |
| frac cluster ≥ 0.5 | 0.733 |
| frac outlier ≥ 0.5 | 0.750 |
| **ROC-AUC** | **0.6240** |

**AUC 0.624 on the easiest possible case is well below the ≥ 0.85 bar.**
The cause is mechanical and now proven: leaf `LFD = log₂(|B(c,r)|/|B(c,r/2)|)`
measures *intra-leaf* geometry complexity, not *inter-leaf* isolation, so an
isolated singleton lands in a leaf whose LFD is comparable to a dense
cluster's, and the global min-max normalisation compresses both into the
same score band. The CHAODA ensemble of Ishaq et al. 2021 combines several
graph-based signals (relative/component cardinality, graph neighbourhood,
random-walk stationary distribution, vertex degree); **only the LFD signal is
shipped today.**

**Consequence for this probe:** `PROBE-CHAODA-1000G` as specified — using
the shipped single-method `anomaly_scores` — would NOT pass even with perfect
genomic fixtures. Before any D-GEN spend on the CHAODA-1000G fixture pipeline,
the substrate needs **the multi-method CHAODA ensemble ported** (or an
augmented anomaly signal). The §1.4 *"unsupervised novel-variant detection"*
claim in `docs/GENETIC_RESEARCH_VIA_STACK.md` is **NOT supported by the
shipped kernel alone** and is now caveated there. This is the
evidence-before-build payoff: the gap is caught before the adapter is funded.

### Current evidence (CONJECTURE → partially FINDING, see ⚠ above)

- The CHAODA kernel is shipped and validated for language-embedding
  anomaly scoring (`ndarray/src/hpc/clam.rs:1493-1567`, Phase 4 section).
- The normalisation is `score = (lfd - lfd_min) / lfd_range`, mapped to
  `AwarenessState` quartiles `Crystallized`/`Tensioned`/`Uncertain`/`Noise`
  (`clam.rs:1549-1557`).  **Note:** the `awareness` field is one of these
  four shipped states; *"Salient"* (mentioned in the merged
  `GENETIC_RESEARCH_VIA_STACK.md` §1.4) is **not** a shipped variant —
  see the §A1 cite-rot fix in this PR.
- The kernel has **never been run against genomic feature vectors**. The
  claim "CHAODA at the same kernel works for variants too" is the bet; the
  probe is the falsifier.
- The CLAM tree's silhouette / Cronbach α / ICC reliability probes
  (ndarray PR #218) establish that the tree converges *when the distance
  metric matches the feature manifold* — that's the conditional this
  probe must measure on genomic features.

### Probe

**Step 1 — Feature vector definition (5-dim per variant).**

| Lane | Field | Source |
|---|---|---|
| 0 | Allele frequency | VCF `INFO/AF` |
| 1 | Total read depth | VCF `INFO/DP` |
| 2 | Strand bias (Fisher) | VCF `INFO/FS` (GATK convention) |
| 3 | 100bp-window Shannon entropy | Computed from reference k-mer counts; bgz17 11/17 sampling (`crates/bgz17/`) |
| 4 | Conservation score | phyloP100way (UCSC track, release-pinned) |

Each lane normalised to `[0, 1]` against its empirical CDF on the training fold.

**Step 2 — Corpus pin.**

- **Training:** 1000-Genomes Phase 3 release `20130502` (NCBI GRCh37) common
  variants (AF ≥ 0.01) on chromosomes 1, 7, 17 (~12 M variants).
- **Held-out test:** 50/50 mix of common variants (AF ≥ 0.01, *expected
  benign by manifold-proximity*) and ClinVar release `2024-12` Pathogenic /
  Likely-Pathogenic singletons (AF < 0.001, *expected anomalous*) on
  chromosomes 22, X (~80 K variants).
- **Ground-truth label:** Pathogenic/Likely-Pathogenic = 1 (positive class),
  common-benign = 0 (negative class).

**Step 3 — Run.**

1. Build CLAM tree on training fold (`ClamTree::build`) with the 3-level 16-way
   layout (`ndarray/src/hpc/clam.rs`, HEEL=16 / HIP=256 / TWIG=4096 per
   `lance-graph/.claude/session_2026_04_11_bf16_hhtl_combined_research.md`).
2. Project held-out vectors through the tree (assign to leaf cluster).
3. Compute anomaly scores via the **ported multi-method ensemble**
   (`D-GEN-CHAODA-ENSEMBLE`, see DAG-honesty below) — NOT the single-method
   `anomaly_scores`. The shipped `anomaly_scores(held_out_bytes, vec_len=5)` →
   `Vec<AnomalyScore>` (single leaf-LFD signal) is the **known-bad baseline**
   that the ⚠ FINDING measured at AUC 0.624; run it too, but only as the
   baseline column the ensemble must beat. The probe's accept/reject decision
   reads the **ensemble** score, not `AnomalyScore.score`.
4. Compute ROC-AUC for BOTH score columns against ground-truth label:
   (a) the ensemble score (the gated number), (b) the single-LFD
   `AnomalyScore.score` baseline (expected ≈ 0.62, the regression floor).
5. Compute per-quartile confusion matrix on the ensemble score to characterise
   *where* the discriminative signal lives.

### Pass condition

- **Ensemble ROC-AUC ≥ 0.85** on the held-out fold. (The single-LFD baseline
  is NOT gated — it is recorded only to confirm the ensemble's lift over the
  known AUC ≈ 0.62 floor.)
- **Per-quartile separation (ensemble score):** Pathogenic-class fraction in
  the top score quartile ≥ 3× the Pathogenic-class fraction in the bottom
  quartile. (Sanity check that the signal is in the anomalous tail, not noise.)
- Tree-quality probes from ndarray PR #218 stay green
  (silhouette ≥ 0.4 on training fold, Cronbach α ≥ 0.7 across the 5 lanes).

### Fail mode → what it means

- Ensemble AUC < 0.85 ⇒ even the multi-method CHAODA ensemble on genomic
  features does NOT recover supervised-classifier discrimination. The whole
  "unsupervised novel-variant detection" claim in
  `GENETIC_RESEARCH_VIA_STACK.md` §1.4 collapses. Either the feature vector is
  underdetermined (add more lanes), or the LFD/graph-anomaly framing doesn't
  capture biological-novelty geometry (rethink composition).
- Ensemble AUC ≈ single-LFD baseline (≈ 0.62) ⇒ the ensemble port added no
  lift; the graph-based signals are not separating on this manifold either —
  escalate before any genomic-fixture spend.
- Ensemble AUC ≥ 0.85 but top-quartile Pathogenic fraction ≤ bottom-quartile ⇒
  the signal is real but **inverted** — common variants land in the anomalous
  band (perhaps from greater linkage / regulatory complexity). Useful but the
  score polarity must be re-documented before publication.

### Cost

- ~3 days **after** D-GEN-1 (adapter scaffold) + D-GEN-2 (VCF parser) ship —
  this probe is NOT runnable in this checkout because there is no VCF → CLAM
  feature-vector pipeline today.
- A spike substitute (~1 day, runnable today): build a CLAM tree on
  synthesised 5-dim Gaussian-mixture data with one "outlier" component;
  verify the anomaly_scores fire on the outlier component. This is a
  smoke test for the kernel, NOT for the genomic-novelty claim.
  **DONE 2026-06-16 — `ndarray` PR #219.** Result: AUC 0.624 (see the ⚠
  FINDING above). The kernel runs deterministically and the polarity is
  correct (outliers ≥ cluster mean) but the single-method LFD signal is
  far too weak — the multi-method ensemble is the prerequisite, not the
  genomic fixtures.

---

## PROBE-KRAS-COUNTERFACTUAL-DET — substrate determinism

### Claim under test

> D-GEN-7's KRAS G12D 1024-cell counterfactual fan-out simulation is
> bit-deterministic across runs with identical seeds.

### Current evidence (CONJECTURE)

- The substrate's no-randomness invariant is documented but never tested
  for the `MailboxSoA<1024>` + `CounterfactualMailbox` composition under
  fan-out load.
- `consume_firing` is integer-state at the threshold-crossing decision; the
  `energy` accumulator is f32 (`crates/cognitive-shader-driver/src/mailbox_soa.rs`).
  Any f32 reduction across cycle boundaries (sum-of-firings) is the candidate
  drift point.

### Probe

1. Run D-GEN-7's KRAS-G12D-vs-WT fan-out twice with identical seed material,
   identical mailbox capacity, identical cycle count (N=100).
2. Bit-compare:
   - Final `MailboxSoA.energy[0..1024]` f32 arrays (memcmp).
   - `plasticity_counter[0..1024]` u8 arrays.
   - `last_active_cycle[0..1024]` u32 arrays.
   - `CounterfactualMailbox` edge counts per split-pole.
3. If any divergence: bisect by cycle count to find first-diverging cycle.

### Pass condition

Bit-exact match across two runs, all four arrays.

### Fail mode → what it means

Any divergence pinpoints an unmarked f32 nondeterminism on the fan-out
critical path. Either fix (port to integer accumulator at the divergence
point) or carve out (mark the divergent stage with `f32_drift_acknowledged`
in the simulation harness, document the tolerance bound).

### Cost

~2 days, included in D-GEN-7's test scope. Not runnable in this checkout
because D-GEN-7 itself is unstarted.

---

## PROBE-CAM-PQ-VS-BLAST — sequence-fingerprint fidelity

### Claim under test

> *"CAM-PQ 48-bit fingerprint approximates sequence similarity"* —
> `docs/GENETIC_RESEARCH_VIA_STACK.md` §1.1. Substantively: Hamming-distance
> rankings on CAM-PQ 6×256 = 48-bit fingerprints of protein sequences track
> BLAST e-value rankings on the same query-vs-target pairs at Spearman
> ρ ≥ 0.7 and ICC ≥ 0.6.

### Current evidence (CONJECTURE)

- CAM-PQ codec ships at `crates/lance-graph/src/cam_pq/storage.rs:9`.
- Σ₁ SEED preserves 94% of Jina 1024-D semantic similarity on SimLex-999
  (`.claude/knowledge/linguistic-epiphanies-2026-04-19.md:299-312`) —
  **language**, not biology.
- The claim "the same 6-subspace PQ + 256-centroid Lloyd-Max codec gives a
  Mash-compatible fingerprint at substrate-native width" is the bet; the
  probe is the falsifier.
- The ESM/ProtBERT embedding pipeline (ndarray AMX int8 GEMM at 197 GMAC/s,
  ndarray PR #217) is the upstream embedding source; CAM-PQ rides those
  vectors.

### Probe

**Step 1 — Corpus.** Held-out RefSeq protein subset: 10 000 sequences chosen
to span the BLAST identity bins (10–30%, 30–50%, 50–70%, 70–90%, 90–100%)
evenly.

**Step 2 — Embed.** ESM-2 small (`esm2_t6_8M_UR50D`) → 320-D protein
embeddings via the existing GGUF loader + ndarray AMX int8 GEMM path.

**Step 3 — Fingerprint.** CAM-PQ encode each 320-D embedding → 48-bit
`Cam6x8`. (Note: SimLex-999 fidelity was measured on 1024-D Jina; ESM-2
small is 320-D — the embedding dimension matters less than the manifold
quality, but document the difference.)

**Step 4 — Rank.** For each of 100 query sequences (1% sample): compute
top-100 nearest-neighbour rankings under (a) Hamming distance on CAM-PQ
fingerprints, (b) BLAST e-value.

**Step 5 — Compare.** Spearman ρ + Cronbach α + ICC(2,1) on the top-100
rankings, using the reliability suite from `lance-graph-arm-discovery`
(ndarray PR #218).

### Pass condition

- **Spearman ρ ≥ 0.7** across the 100 queries, median.
- **ICC ≥ 0.6** on the agreement between Hamming-rank and BLAST-rank.
- Per-identity-bin breakdown: Spearman ρ ≥ 0.5 in each bin (sanity that
  the signal is not driven only by easy >90% hits).

### Fail mode → what it means

- ρ < 0.7 (median) ⇒ the Σ₁ SEED 94% fidelity that holds for language does
  not transfer to protein sequence similarity at this PQ width. Either the
  embedding (ESM-2 small → 320-D) is too compressed, or the 6×256 codec is
  miscalibrated for protein-manifold geometry. The composition needs
  re-quantization at a different (subspace count, centroid count) before
  the sequence-similarity story can be claimed.
- ρ ≥ 0.7 in high-identity bins only ⇒ the fingerprint is a good
  *near-duplicate* detector but **not** a homology detector — important
  scope reduction for the published claim.

### Cost

~1 week. Includes the ESM-2 embedding pipeline integration (GGUF load
exists; the protein-embedding adapter does not). Not runnable in this
checkout until the embedding adapter is wired.

---

## §A1 — cite-rot fix folded into this PR

`docs/GENETIC_RESEARCH_VIA_STACK.md` §1.4 in the merged PR #501 reads:

> *"A novel variant in a region of high LFD lights up as `AnomalyScore
> { score → 1.0, awareness → Salient }`"*

There is **no `AwarenessState::Salient`**. The shipped variants per
`ndarray/src/hpc/clam.rs:1549-1557` are `Crystallized` / `Tensioned` /
`Uncertain` / `Noise`. The high-LFD tail (`score ≥ 0.75`) maps to
`AwarenessState::Noise`. This document fixes that citation as part of the
same commit so future probe-runners are not chasing a non-existent variant.

Also corrected: the `score` field is `f64`, not `f32` (clam.rs:1504).

---

## DAG honesty

The genetic-research plan's `4-week first-deliverable target` (P1 in §3 of
the plan) assumes PROBE-CHAODA-1000G's claim is recoverable. If the probe
fails, the adapter scaffold (D-GEN-1..4) still has value — VCF round-trip,
CAM-PQ k-mer fingerprints, classid mint — but the §1.4 novelty-detection
story must be retracted and the GENETIC_RESEARCH_VIA_STACK.md hand-off
re-shaped before further external-audience use.

**Update 2026-06-16 — the spike already fired the first warning.** The P0
spike (ndarray #219) measured AUC 0.624 with the *shipped single-method
leaf-LFD* signal. This does NOT retract the whole probe — it relocates the
prerequisite: **before** PROBE-CHAODA-1000G can pass, the multi-method
CHAODA ensemble (Ishaq et al. 2021: relative/component cardinality, graph
neighbourhood, random-walk stationary distribution, vertex degree) must be
ported into `ndarray::hpc::clam`. That ensemble port is now the true P0
work item; the genomic-fixture pipeline (which depends on D-GEN-1+2) is
gated behind it. The §1.4 hand-off has been caveated rather than retracted —
the pattern match is sound, the single shipped signal is not yet sufficient,
and the honest path is "port the ensemble, then re-run the spike, then build
the fixture." A new candidate deliverable falls out of this:

> **D-GEN-CHAODA-ENSEMBLE (new, prerequisite to PROBE-CHAODA-1000G):** add the
> multi-method CHAODA anomaly ensemble to `ndarray::hpc::clam` as a **new
> scoring entry point** (e.g. `ensemble_anomaly_scores(...) -> Vec<AnomalyScore>`,
> name TBD at implementation), combining the graph-based signals of Ishaq et
> al. 2021. The existing single-method `anomaly_scores` is **kept unchanged as
> the documented baseline / regression** (the ndarray #219 spike's `auc < 0.85`
> tripwire stays green on it). **`PROBE-CHAODA-1000G` Step 3 must call the new
> ensemble entry point, not `anomaly_scores`** — that wiring is part of this
> deliverable, otherwise the genomic probe would re-measure the known-bad
> AUC-0.624 path. Re-run the ndarray #219 spike against the ensemble; gate at
> AUC ≥ 0.85 on the synthetic mixture *before* genomic fixtures are built.
> Lift: ~1 week (the graph-construction primitives — cluster cardinality,
> neighbourhood, random-walk — are mostly present in the CLAM tree already;
> the ensemble combination + per-method scoring + the probe-API wiring is the
> new code).

**PROBE-CHAODA-1000G fires first, even though chronologically D-GEN-1..2 must
ship first.** That ordering is a substrate-economic decision (cheaper to
build the adapter than to abandon a year of plan), but the probe gating
discipline (CLAUDE.md cycle) demands it runs the moment it CAN run, not
later when more is sunk.

---

_Planted 2026-06-16 by external session `AdaWorldAPI/bardioc`
`session_01VysoWJ6vsyg3wEGc5v7T5v`. Mirrors the probe-spec discipline of
`ocr-probes-v1.md` (lance-graph PR #500). No probe is RUN in this PR —
each is gated on adapter-genetics-experimental scaffold + corpus + embedding
pipeline arriving._
