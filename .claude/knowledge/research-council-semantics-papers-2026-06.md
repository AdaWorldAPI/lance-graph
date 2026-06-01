# Research council — 8 semantics/embedding papers, firewall-filtered (2026-06-01)

**READ BY:** integration-lead, truth-architect; anyone mining external research into the substrate.
**Status:** SYNTHESIS (5-agent research council, all papers read IN FULL — no grep/head/tail, per `E-READ-NOT-GREP`). Each verdict is firewall-checked against the deterministic CAM-PQ substrate.
**Method:** 5 Opus reviewers (A1 NSM / A2 interpretable-embeddings / A3 embedding-structure / A4 sequence+decoding / A5 applications+firewall-critic) each read their assigned full-text papers and judged: mechanism, workspace relevance, firewall (does adoption pull LLM/float/language onto the hot path?), verdict.

## The slate (8 papers)

| # | paper (arXiv) | the extractable mechanism | verdict | firewall |
|---|---|---|---|---|
| 1 | 2505.11764 **Towards Universal Semantics / DeepNSM** (OSU) | **Legality Score** `α·(primes−molecules)/words` + lemma-containment circularity; prime-vs-**molecule** ontology | PROBE | gate **PASS** (integer, no LLM); the LLM machinery **CONFLICT** |
| 2 | 2410.03435 **Interpretable Text Embeddings (CQG-MBQA)** | **`cognitive load = ⟨u,v⟩`** (count of shared named axes) — self-explaining similarity; contrastive cluster-vs-hard-negative axis design (offline) | PROBE | metric **PASS** (integer hot-path); learned encoder **CONFLICT** |
| 3 | 2512.00852 **One Swallow / SAFARI** (NUS) | **Semantic-Shift basin-boundary detector** w/ Weyl-1912 bound `‖A_y‖₂·σ_max(A_x)` (no SVD, no labels) | PROBE | **PASS** (offline auditor; spectral-norm on ±1 fingerprints) |
| 4 | 2508.10003 **Semantic Structure in LLM Embeddings** (Kozlowski, U Chicago) | **cosine-proportional interference law**; meaning = continuous ~3-axis (Eval/Potency/Activity) low-rank manifold; **non-orthogonality is signal** | **PROBE (adversarial red-team)** | N-A as component; **CONFLICT as worldview** |
| 5 | 2510.18046 **LaMAR** (sequential recommenders) | **novelty-gated admission** — a near-duplicate signal *hurts*; diversity carries the gain | PROBE | abstracted (offline-frozen symbol + novelty gate) **PASS**; LLM gen **CONFLICT** |
| 6 | 2506.23601 **SemDiD** (semantic-guided diverse decoding) | **repulsion-from-nearest-rival + ε-quality-floor + harmonic (weakest-link) combiner**, k parallel arcs | **ADOPT-NOW** (cosine→Hamming) | **PASS** (training-free; swap float-cosine for VSA-overlap) |
| 7 | 2504.21074 **LLMs for Process Mining** | **footprint fitness** `(A×A)→{→,←,∥,#}`, quality = matching-cell fraction (frequency-free, integer) | PROBE (validator ADOPT-grade) | **SAFE iff quarantined** as proposer; TRAP if LLM-edges land as fact |
| 8 | 2412.08671 **Deep Semantic Segmentation** | neighbor-window (3×3) offset refinement | **SKIP** | **TRAP** — float/learned all the way down, no propose/validate seam |

## The three unifying threads

1. **The firewall is a *useful filter*, externally corroborated.** All 8 papers are LLM/float-based; the council extracted only the deterministic/integer/offline kernel from 7 and cleanly SKIP'd the 1 with no quarantine seam. **The "similarity PROPOSES (float, offline, upstream) / CAM ADDRESSES (integer, hot, deterministic)" doctrine was independently re-derived by 4 of 5 reviewers from 7 different papers** (A1 LLM-proposes-explication / Legality-validates; A2 LLM-authors-axes / ⟨u,v⟩-explains; A4 generate-arcs / repulsion-selects; A5 ARM-or-LLM-proposes-edges / footprint-validates). Strong outside evidence the core architecture is right.

2. **One shared operator across the memory + inference axes (A4):** *"retain a candidate iff its hypervector is far enough from the current incumbents, under a quality floor."* Mount once on **head2head** (arc arbitration — SemDiD) and once on **EW64** (episodic-edge admission — LaMAR's "novelty beats volume"). Integer VSA-overlap vs an incumbent set; no learned weights, no float, no language. The EW64 side refines the just-merged `promote`/`coldest` (#447/#448): admission should gate on **novelty**, not recency alone.

3. **Three deterministic integer VALIDATORS to adopt** (each turns a proposal into a check): **Legality** (A1, prime-reduction purity, DeepNSM-side), **⟨u,v⟩** (A2, similarity self-explanation, CAM-PQ-side), **footprint `{→,←,∥,#}`** (A5, ordering-relation validation, aerial→DOLCE boundary).

## The adversarial finding (highest-value to weigh)

**A3/Kozlowski (#4) is the strongest external challenge to the spine geometry.** Cross-model-replicated: semantic features are **meaningfully non-orthogonal** (whitening *reduces* human-rating fidelity ~20% — "non-orthogonality is a feature, not a bug"), and meaning collapses to a continuous ~3-axis low-rank manifold. A3's key distinction: **role disjointness** (SUBJECT vs PREDICATE binding slots) is standard, correct VSA — *not* challenged. But **content orthogonality** is: if the substrate treats concept-overlap as quantization noise, this says that overlap *is* the carrier of meaning, and the hard 4096-basin + ρ=0.9973 CAM-PQ partition **may be discarding the entangled low-rank structure that is the actual semantics.** A3's complement (#3 SAFARI) hands the *corroborating* tool: the Weyl Semantic-Shift auditor directly tests the #444 98.6%-locality (sharp `µ+3τ` spikes at class boundaries = clean partition; smeared = falsified).

## Prioritized build/probe slate (firewall-filtered)

- **ADOPT-NOW (offline-buildable):** **SemDiD → a head2head `WinnerCriterion::Repulsion`** — repulsion-from-nearest-rival (VSA-overlap) + ε-quality-floor + harmonic combiner. Self-contained in `contract::head2head`; integer; offline-testable. *(A4)*
- **ADOPT-grade VALIDATOR, offline:** **Legality Score** in `deepnsm` — prime/molecule counting + circularity; a float-free certifier of SPO prime-reduction. *(A1)*
- **PROBE (highest-value VALIDATION of the foundation):** the **Kozlowski antonym-direction interference test** — build antonym-difference directions in VSA space; does cross-direction cosine predict CAM-PQ mis-addressing? If yes, the basin model is lossy w.r.t. gradient semantics. Pair with the **SAFARI Weyl basin-coherence auditor** over the OGIT/DOLCE partition (tests #444). *(A3 — both)*
- **PROBE (refines shipped work):** **EW64 novelty-gated admission** — gate `promote` on dissimilarity-to-incumbents, not recency alone. Needs edge-fingerprint access (the co-addressed content), so it rides the gated SoA/hot-path seam, not pure-contract. *(A4/LaMAR)*
- **PROBE (interpretability):** **⟨u,v⟩ self-explanation** for CAM-PQ (list shared codebook IDs/primes behind a match) *(A2)*; **footprint validator** at the aerial→DOLCE landing *(A5)*.
- **SKIP:** segmentation (#8) — unconditional firewall trap.

*Cross-ref:* EPIPHANIES `E-RESEARCH-COUNCIL-PROPOSE-VALIDATE`; `E-READ-NOT-GREP` (the council read full text); the firewall doctrine (`E-ENGLISH-BIFURCATES`, markov_soa SoC); `head2head`, `episodic_edges`, DeepNSM. Papers staged at `/tmp/papers/*.txt`.
