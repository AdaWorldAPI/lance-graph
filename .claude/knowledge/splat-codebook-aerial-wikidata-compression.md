<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright The Lance Authors
-->

# KNOWLEDGE: Splat codebook → aerial discovery → deterministic Wikidata compression

## READ BY:
- Any worker wiring `lance-graph-arm-discovery::aerial::CodebookDistance` to a production oracle
- Any worker touching `crates/jc` (Jirak-Cartan) pillars 5/9/9b/10 in an ARM/compression context
- Any worker on the Wikidata-HHTL load (`specs/wikidata-hhtl-load.md`) skeleton/basin pass
- `truth-architect`, `integration-lead`, `palette-engineer`

## P0 TRIGGERS:
- About to give `aerial` a real distance oracle → the oracle is jc's certified palette256 table, read this
- About to implement the D-ARM-7 Jirak significance floor → it derives from `jc::jirak`, read this
- About to add a float similarity to the ARM/discovery path → STOP, the float lives OFFLINE in jc, not here

---

## The convergence

`lance-graph-arm-discovery` (the Aerial+ transcode) shipped with two open seams:

1. a **production `CodebookDistance` oracle** (the de-float replaced the autoencoder with an injected integer distance — E-ARM-PROBE-IS-CODEBOOK-TOPK), and
2. the **D-ARM-7 Jirak significance floor** (the still-unimplemented gate above the classical ARM support/confidence floors — ISSUE ARM-JIRAK-FLOOR).

**Both resolve to the same crate: `crates/jc` (Jirak-Cartan).** jc *proves the
codebook is sound*; aerial *uses it to discover the ontology skeleton* that
drives deterministic Wikidata compression. The user's framing —
"gaussian-splat spatial blasgraph top-k 10000×10000 … for OWL/DOLCE+ SPO HHTL
classes and basin via aerial+ to deterministically compress Wikidata … adjacent
to JC Jirak[-Cartan] with EWA-sandwich gaussian splat" — names this end to end.

```text
  OFFLINE  (build + certify the codebook — f64 is allowed here)
  ───────────────────────────────────────────────────────────────────────
  ndarray::hpc::splat3d            jc::ewa_sandwich{,_3d}  (Pillars 9/9b)
  Gaussian-splat spatial top-k  →  Σ-push-forward J·W·Σ·Wᵀ·Jᵀ certified
  (10000×10000 BLASGraph;          jc::sigma_codebook_probe: 256-codebook
   jc splat_* graph algos:           captures the Σ-distribution at
   Louvain / triangle / LPA /         ρ=0.9973 (R² ≥ 0.99, log-Euclidean)
   Jaccard-Adamic-Adar)            jc::pflug (Pillar 10): CAM-PQ tree
        │                            quantization is Lε-faithful ⇒ the
        ▼                            HHTL cascade preserves FreeEnergy
  256×256 integer [a,b] distance table  ──────────────────────────────┐
                                                                       │
  ONLINE  (use the frozen table — integer ONLY)                        │
  ───────────────────────────────────────────────────────────────────│───
  aerial::CodebookDistance  ◄── MatrixDistance::new(spec, table) ◄─────┘
        │   (the codebook probe: nearest consequents within θ)
        ▼
  aerial discovers OWL/DOLCE+ SPO HHTL classes + basins from Wikidata triples
        │   gated by:  classical support/confidence (ppm)
        │         AND  D-ARM-7 Jirak floor  ◄── derives its rate from jc::jirak
        ▼                                        (n^(p/2-1), weak dependence)
  skeleton (P279/P31 class DAG + basin assignment)  →  codebook-HHTL compression
  (specs/wikidata-hhtl-load.md: skeleton + basins + CAM-dedup + thin rows)
```

## The two seams, precisely

| Seam | aerial side | jc side |
|---|---|---|
| Similarity oracle | `aerial::CodebookDistance::distance(a,b) -> u32` (integer, frozen). `MatrixDistance` already consumes a `[u32; dim²]` table. | jc builds + certifies that table: `ewa_sandwich` (splat Σ-push-forward correct) + `sigma_codebook_probe` (256-codebook viable at ρ=0.9973). |
| Significance floor (D-ARM-7) | `CandidateRule::passes()` gates classical support/confidence; the Jirak floor is the stricter, unimplemented gate above it. | `jc::jirak` supplies the correct rate (`n^(p/2-1)` for weakly-dependent data) — classical IID Berry-Esseen is wrong here (I-NOISE-FLOOR-JIRAK). The floor threshold derives from this rate. |

## The float boundary (why this is doctrine-clean)

The de-float removed float from aerial's **online** path. This pipeline keeps
that invariant: **float lives only in jc's OFFLINE certification** — k-means in
log-Euclidean space (`sigma_codebook_probe`), the SPD/covariance EWA math
(`ewa_sandwich`), the Berry-Esseen sup-error (`jirak`). All of that runs once,
produces a frozen integer artifact (the 256×256 `[a,b]` table + a derived
significance threshold), and is never on aerial's hot path. This is exactly the
CAM-PQ doctrine (`faiss-homology-cam-pq.md`, I-VSA-IDENTITIES): **build the
codebook offline (float OK), address it online with integer codes.** jc is the
"build + prove"; aerial is the "use".

## jc pillar → role in this pipeline

| jc module | Pillar | Role here |
|---|---|---|
| `ewa_sandwich`, `ewa_sandwich_3d` | 9, 9b | Certifies the Gaussian-splat Σ-push-forward (`ndarray::hpc::splat3d`) — the splat top-k that *builds* the codebook is correct. |
| `sigma_codebook_probe` | (probe) | The ρ=0.9973 / R²≥0.99 viability measurement — *why* a 256-codebook may replace a float similarity. |
| `pflug` | 10 | Pflug-Pichler nested-distance Lipschitz — the CAM-PQ/HHTL tree quantization preserves FreeEnergy within Lε, so codebook-HHTL compression is faithful, not lossy-by-surprise. |
| `jirak` | 5 | The weak-dependence Berry-Esseen rate — the engine the D-ARM-7 significance floor derives from. |

## Wikidata / DOLCE mapping (specs/wikidata-hhtl-load.md, ogit-owl-dolce)

- **Skeleton (Pass 1):** P279 (subClassOf) DAG + P31 (instance-of) → classes + basins (HHTL levels). aerial's discovered `(X → Y)` rules over the class/property items ARE candidate skeleton edges; the Jirak floor decides which are significant enough to persist.
- **DOLCE as axis template:** Endurant/Perdurant/Quality/Abstract (≈ Object/Process/Quality/Region) define WHICH HHTL axes exist; Wikidata properties fill WHAT occurs. aerial discovers the fill against the DOLCE scaffold.
- **Compression is structural:** classes + masks + refs + CAM-deduped shapes, not a gzip trick (~120GB → ~38GB). aerial supplies the class/basin structure; codebook-HHTL supplies the bucket router (16ⁿ nibble addressing).

## Status & open questions

- **Phase 1 SHIPPED** (D-ARM-14, branch `claude/jolly-cori-clnf9-darm14`): the two aerial-side seams are concrete code, not just a trait — `aerial::TopKDistance` (the sparse per-node splat-top-k `CodebookDistance` the 10000² splat actually emits — top-k per node, not a dense `dim²` table) and `aerial::ontology::{DolceCategory, OntologyProjector}` (the DOLCE 4-facet skeleton → `rdfs:subClassOf`/`rdf:type` SPO output). An end-to-end test runs splat-top-k → discovery → skeleton projection. Standalone, integer, 41/41.
- **CONJECTURE** (architecture): the splat→aerial→Wikidata pipeline *as a whole* — the real jc/blasgraph splat producing the neighbour lists, and the Wikidata loader, are not yet built. The seams they plug into now exist.
- **Hard prerequisite:** D-ARM-7 (the Jirak floor) must land before aerial promotes any rule to a live skeleton — ISSUE ARM-JIRAK-FLOOR. jc::jirak is the engine; the *gate function* (rule → significant?) is the deliverable.
- **No new aerial dependency needed:** jc emits a frozen integer table; aerial consumes it through the existing `MatrixDistance`/`CodebookDistance` seam. Do NOT make aerial depend on jc — keep the float-free standalone posture; pass the certified table in.
- **Open (from wikidata-hhtl-load):** P279 fan-out is wildly uneven (2…4000 children) — whether it rebalances onto a clean 16ⁿ tree or forces adaptive fan-out is MEASURABLE; measure on a real P279 subtree before fixing the HHTL base.
- **target-cpu:** the splat/SIMD kernels (AVX-512 VPOPCNTQ, Intel AMX) need `-C target-cpu=native` or `x86-64-v4`; otherwise ndarray's correct-but-scalar polyfill runs.

## Cross-references

- `crates/lance-graph-arm-discovery/` — the aerial consumer; `aerial::CodebookDistance` is the seam.
- `crates/jc/src/{ewa_sandwich,ewa_sandwich_3d,sigma_codebook_probe,pflug,jirak}.rs` — the certifier.
- `.claude/specs/wikidata-hhtl-load.md` — the compression pipeline.
- `.claude/knowledge/ogit-owl-dolce-ontology-compartments.md` — the DOLCE scaffold.
- `.claude/plans/3DGS-HHTL-datalake-traversal-plan.md` — the splat-HHTL traversal.
- `.claude/knowledge/faiss-homology-cam-pq.md`, CLAUDE.md `I-VSA-IDENTITIES`, `I-NOISE-FLOOR-JIRAK` — the float/codebook + Jirak iron rules.
- `EPIPHANIES.md` E-ARM-PROBE-IS-CODEBOOK-TOPK (the de-float) + the entry that points here.
