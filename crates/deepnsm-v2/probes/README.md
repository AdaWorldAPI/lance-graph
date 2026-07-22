# deepnsm-v2 probes — meaning vs routing vs frequency, Jina-grounded

Reproducible measurement scripts behind the CAM-PQ 96 architecture decision.
All read `JINA_API_KEY` from env (never committed); Jina calls go through
`curl` (`/root/.ccr/ca-bundle.crt` in the CI sandbox; plain curl elsewhere).
Data inputs (not committed): `coca_tokens.csv` from the repo release
`v0.1.0-coca-data` (CC0), `lemmas_5k.csv` from `crates/deepnsm/word_frequency/`.
Deterministic seeds (`0x9E3779B9`); no clock/rng in the substrate under test.

## Results (2026-07-22, jina-embeddings-v3, 256-d/96-d)

### 1. `crosscheck_full.py` — the three axes vs the Jina meaning oracle

360 tokens ∈ (8-genre table ∩ 18k DocuScope), 64,620 pairs:

| axis | statistic | value | reading |
|---|---|---|---|
| frequency | ρ(\|Δlog perMil\|, Jina-dist) | **−0.067** | ROUTING — orthogonal to meaning |
| 8-genre count-distance | ρ(1−zcos(genre), Jina-dist) | **+0.039** | coarse floor only (curated NSM pairs ρ=0.762, random pairs ~0) |
| DocuScope category | AUC(same>cross Jina cos) | **0.567** | coarse awareness axis |

(A first stratified run, 343 tokens/58,653 pairs: freq ρ=−0.091, category
AUC=0.677, Cohen d=0.694 — same ordering.)

**Determination: frequency = routing (⟂ meaning). Count-derived meaning is a
real but COARSE tier — fine meaning needs a trained codebook.**

### 2. `fidelity_48_vs_96.py` — CAM-PQ 48 POINT vs CAM-PQ 96 DISTRIBUTION

3,000-token vocab, real Jina 96-d embeddings, 299,896 eval pairs; PQ codebooks
trained per subspace (k-means-256, 20 iters):

| substrate | shape | recon MSE | ρ(dist, Jina) |
|---|---|---|---|
| deepnsm 48-bit POINT | 6×16-d | 0.00292 | **0.711** |
| deepnsm_v2 96-bit DISTRIBUTION | 12×8-d | 0.00171 | **0.828** |

**Δρ = +0.117 (+16.5%), 41% lower MSE. The 6×cosine² DISTRIBUTION preserves
more Jina meaning than the 6×cosine POINT.** Caveat: 2× code budget; a 96-bit
point control (structure-vs-budget isolation) is an open follow-up.

### 3. `spo_usedfor_nars.py` — can 0.828 carry SPO 2³ "used for" reasoning?

48 subjects × 15 used_for groups; codebook trained on a SEPARATE 1,500-word
vocab (no fit-to-test):

| representation | kNN@1 function-purity | analogical SP→O |
|---|---|---|
| Jina-full | 0.792 | 0.792 |
| **96-bit DISTRIBUTION** | **0.667** | **0.667** |
| 48-bit POINT | 0.500 | 0.500 |

Predicate-token arithmetic FAILS: mean rank of the true used_for object
worsens 0.56 → 1.02 when adding the "used for" embedding to S.

**Determination: (a) the substrate carries the ANALOGICAL leg of NARS 2³
(similar subjects inherit used_for) and 96 > 48 exactly where it matters
(0.667 vs 0.500); (b) the relation itself must be a STORED SPO edge — it is
not recoverable by vector arithmetic. Division of labor: triples store
relations; the substrate generalizes them; NARS deduces.**

## Honest bounds

Jina-v3 (not v5) oracle; probe-scale N; k-NN purity as reasoning proxy;
Python+numpy k-means (the production codebook trainer is the ndarray-side
producer, `TD-CERTIFIED-DISTANCE-TABLE-UNCONSUMED`). These are measurement
probes gating architecture — not production certification.
