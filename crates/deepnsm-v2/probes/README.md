# deepnsm-v2 probes — meaning vs routing vs frequency, Jina-grounded

Reproducible measurement scripts behind the CAM-PQ 96 architecture decision.
All read `JINA_API_KEY` from env (never committed; passed to curl via stdin
config, never argv). CA bundle: system trust store by default; set
`JINA_CA_BUNDLE=/root/.ccr/ca-bundle.crt` in the proxied CI sandbox.
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

**HELD-OUT protocol (corrected 2026-07-22 per PR #801 review):** codebooks fit
on a disjoint 2,000-token train split; encoded + scored ONLY on the held-out
1,000 tokens (299,699 pairs), real Jina 96-d embeddings:

| substrate | shape | recon MSE | ρ(dist, Jina) |
|---|---|---|---|
| deepnsm 48-bit POINT | 6×16-d | 0.00388 | **0.624** |
| deepnsm_v2 96-bit DISTRIBUTION | 12×8-d | 0.00237 | **0.766** |

**Δρ = +0.142 (+22.7%), 39% lower MSE — out-of-sample.** The correction
LOWERED absolute fidelity (the original in-sample run — 0.711/0.828, +16.5% —
overstated it by scoring the training set) but WIDENED the 96-vs-48 gap: the
DISTRIBUTION generalizes better than the POINT, which strengthens the
architecture conclusion. Caveat unchanged: 2× code budget; a 96-bit point
control (structure-vs-budget isolation) is an open follow-up.

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

### 4. `embed_bible_vocab.py` + `train_codebook.py` — the TRAINED codebook producer

KJV Bible vocabulary (12,543 words, PD, Gutenberg #10 — text not committed),
embedded via Jina-v3 96-d, codebooks k-means-256 per axis. **Held-out**
measurement (train 10,000 / eval 2,543, 299,882 pairs):

| code | shape | ρ(dist, Jina) | MSE |
|---|---|---|---|
| 48-bit POINT | 6×16d PQ | 0.617 | 0.00343 |
| **96-bit POINT control** | 6×(coarse+residual RQ) | **0.786** | 0.00198 |
| 96-bit DISTRIBUTION | 12×8d PQ | 0.774 | 0.00211 |

**Structure-vs-budget isolated:** at equal 96 bits the residual-refined POINT
slightly exceeds the 12-axis DISTRIBUTION on raw fidelity (+0.012). The
96-vs-48 gain is therefore mostly BUDGET; the DISTRIBUTION's justification is
the **facet algebra** — independently-addressable bytes, `6×(u8:u8)` rail
semantics, morton-comma prefix/ancestry routing — which residual coding breaks
(its second byte is meaningless without the first). A measured trade: ~0.012 ρ
for content-blind addressability. GATE (ρ96 ≥ 0.6 AND ρ96 > ρ48): PASS.

Artifacts (NOT committed — hosted as the `AdaWorldAPI/lance-graph` release
`v0.1.0-cam96-data`; fetch per `data/README.md`): `cam96_codebook.bin`
(12×256×8d f32 + d_max, 96 KB), `cam96_codes.bin` (12,543 × 12 B),
`bible_vocab.txt` — loaded at runtime by `src/codebook.rs`; exercised
end-to-end by `examples/bible_wave.rs`.

### 5. `examples/bible_wave.rs` — the WHOLE-BOOK falsifier (Rust, gates in-code)

KJV through the full pipeline (verses → PoS → FSM → SPO stream → TemporalStream
+ trained codebook): **G1** 23,145 verses ≤ 65,536 (one 256×256 tile);
**G3** 31,327 triples / 606 subjects / 1,081 predicates; **G4** trained-codebook
meaning sim(god,lord)=0.625 > sim(god,fish)=0.265. **Long-range: 63.3% of
27,086 same-subject recurrence links are beyond ±5** — the context v1's
fire-and-forget ring structurally forfeits; 55.7% beyond ±8 (the Escalate zone
→ global graph). The whole-book literal read returns all 31,327 triples in one
window — no bundle, no beam, no per-sentence reset.
