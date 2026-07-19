# DeepNSM × Morton-comma facet — design + PROBE-DEEPNSM-FACET spec (v1)

> **Status:** SPEC (design handoff for independent replication).
> **Contains NO measured results** — this doc is the shared design + probe
> specification. Each session implements and runs independently; results are
> held for operator adjudication (see § Replication protocol).
> **Provenance:** the Morton-comma substrate (`E-PERTURBATION-CONVERGENCE-1`,
> merged) applied to `crates/deepnsm`. Cross-ref: OGAR `256×256 CENTROID TILE`
> canon; le-contract L4 `6×palette256:palette256`; the game-matrix reasoning
> in this doc's §2.

## 0. The one-sentence design

A DeepNSM word becomes a **canonical V3 facet**:
`classid(4B) = norm(prefix, frequency, PoS)  |  payload(12B) = 6 × (FisherZ:FisherZ)`
— prefix+frequency are a **routing/attention header** (codebook scoping + LOD),
NOT distance axes; the 6 byte-pairs are the **semantic distribution** (256:256,
Fisher-z cosine-replacement), analytic (48 B γ) or materialized (palette256).

## 1. Why 256:256 over 256 (distribution over point)

DeepNSM today: `96D subgenre-freq → 6 subspaces × 16D → k-means(256) → 6-byte
code`; distance = ADC (16-D L2 per subspace). Each subspace is a **point
estimate** (one centroid). The upgrade: each subspace becomes a **2-D
distributional coordinate** (`FisherZ:FisherZ`) — Fisher-z linearizes cosine, so
L2-in-Fisher-z ≈ cosine, and the pair keeps spread, not just location. Analytic:
`bgz-tensor::fisher_z` (cosine→atanh→8-byte per-family affine→normalized i8;
hydrate via tanh). Materialized alternative: 6 × 256² palette LUT.

## 2. The game-matrix (design theory — the lane decision)

Three players, each choosing a lane. **This section is DESIGN reasoning
(architectural payoffs), not measured results.**

| Player | Strategies |
|---|---|
| **MC** Morton-comma | `Address` (position=key, replayable) · `Flat` |
| **P²** Palette256² | `Dist` (256:256) · `Point` (256 flat) |
| **PF** prefix:frequency | `Header` (classid→codebook-scope+LOD) · `Payload` (distance axis) · `Off` |

**Pairwise structure:** MC×P² = super-additive (P² *is* the L4 tenant on the
Morton cascade — one structure); MC×PF = synergy (PF is the cascade's
longest-prefix routing header); **P²×PF = conditional** — synergy iff PF=`Header`,
**collapse** iff PF=`Payload`.

**Dominant strategy: PF=`Header`.** It weakly dominates `Payload` (which only
adds collapse risk — its attention value is recoverable in the header) and `Off`.
**Equilibrium: (Address, Dist, Header).** The **defection trap** is
(Address, Dist, Payload): folding prefix:frequency into the 6×(8:8) is locally
tempting (one uniform code) but breaks fidelity (see §3 — this is exactly what the
probe measures).

## 3. PROBE-DEEPNSM-FACET — the LANE TEST (the decisive cell)

Full Fisher-z fidelity (hierarchical 256² vs flat 256) is gated on the
gitignored codebook (`codebook_pq.bin` / `subgenres_5k.csv`) — a **v2** leg that
needs the `.bin`. The **runnable decisive cell** — does frequency-in-payload
collapse semantic distance vs frequency-in-header — runs on **committed data**.

### 3.1 Data (committed, reproducible)
- `crates/deepnsm/word_frequency/lemmas_5k.csv` — columns `blogPM,webPM,TVMPM,
  spokPM,ficPM,magPM,newsPM,acadPM` (8 per-million genre profile) + `freq` + `PoS`.
- 5,050 lemmas. Both sessions read THIS file (no regeneration → identical ground
  truth).

### 3.2 Encoding (spec the normalization EXACTLY so impls match)
1. **Semantic vector** `s(w)` = the 8 `*PM` columns → `log1p` → per-dimension
   z-score across the 5,050 vocab (mean 0, std 1 per genre column). 8-D.
2. **Frequency scalar** `f(w)` = `log1p(freq)` → z-score across vocab. 1-D.
3. **Semantic truth distance** `d_truth(a,b) = L2(s(a), s(b))` — this is the
   distributional distance DeepNSM's CAM approximates. It is the reference.

### 3.3 The two lanes
- **HEADER lane** (PF=Header): `d_header(a,b) = d_truth(a,b)` — frequency is a
  separate routing header, NOT in the distance.
- **PAYLOAD lane** (PF=Payload): `d_payload(a,b) = L2([s(a); f(a)], [s(b); f(b)])`
  — the 9-D distance with the normalized log-frequency APPENDED as a coordinate
  (frequency folded into the payload).

### 3.4 Measurements (report all; NO peeking at the other session)
1. **ρ_all** = Spearman ρ between `d_payload` and `d_truth` over a fixed
   deterministic sample of pairs (spec: all pairs among the top-256 lemmas by
   rank, = 32,640 pairs — deterministic, no RNG).
2. **Frequency-shuffle invariance (the orthogonality proof):** permute the `f`
   values across words; recompute `d_header` (unchanged by construction) and
   `d_payload` (changes). Report whether `d_header` is invariant (it must be) and
   the magnitude of `d_payload`'s change.
3. **Neighborhood corruption — the collapse signal (NOT absolute inversion).**
   ⚠ The append is **monotone-non-decreasing**: `d_payload = √(d_truth² + (Δf)²)
   ≥ d_truth` (Pythagoras), so `d_payload < d_truth` is IMPOSSIBLE and must NOT
   be used as a criterion. The real defection is **rank/neighbor corruption**:
   frequency inflates distances by the frequency gap, which pushes
   **semantically-near-but-frequency-far** pairs (near-synonyms with divergent
   frequency) APART and reorders the neighborhood. Measure:
   - (a) **Nearest-neighbor flip rate:** for each of the 256 words, its nearest
     neighbor under `d_header`(=`d_truth`) vs under `d_payload`; report the
     fraction whose nearest neighbor CHANGES.
   - (b) **The corrupted set:** pairs that are semantically **near** (`d_truth`
     bottom decile) but frequency-**far** (`|f(a)−f(b)|` top decile). Report
     their mean inflation `d_payload − d_truth` (> 0) and their mean **downward
     displacement** in the global nearness ranking (they fall relative to
     frequency-near pairs) vs the complement.
4. **Named anchor (computed, not hand-picked):** among the semantically-nearest
   decile of pairs, report the one with the LARGEST frequency gap — its
   `d_truth`, `d_payload`, inflation, and how many rank positions it FALLS in the
   nearness ordering (a true near-pair demoted purely by frequency).

### 3.5 PASS / COLLAPSE criteria (pre-registered — fill with YOUR numbers only)
- **HEADER lane** must be **frequency-shuffle-invariant** (measurement 2) — the
  lane-orthogonality proof (frequency didn't leak into semantics).
- **COLLAPSE CONFIRMED** (⇒ PF=Payload is a defection, equilibrium holds) iff:
  `ρ_all < 1.0` AND the **nearest-neighbor flip rate** (3a) is materially
  non-zero AND the semantically-near-frequency-far set (3b) shows systematic
  positive inflation + downward rank displacement (frequency corrupts the
  semantic neighborhood).
- **COLLAPSE REFUTED** (⇒ reconsider the matrix) iff `ρ_all ≈ 1.0` and ~zero
  nearest-neighbor flips — frequency did NOT corrupt the ordering.
- **Optional stronger variant (fixed-width faithful):** since the real payload is
  fixed-width `6×(8:8)`, "folding in" frequency means it REPLACES a semantic
  axis, not appends. Under REPLACE (overwrite the lowest-variance genre dim with
  `f`), absolute inversions `d_payload < d_truth` DO become possible (a dropped
  axis shrinks some distances) on top of the frequency contamination — a report
  may add this variant, but the neighbor/rank corruption above is the primary,
  append-robust falsifier.

## 3a. Scaling to 20k academic COCA (the regime where the facet earns its keep)

Committed this PR: `crates/deepnsm/word_frequency/academic_20k.csv` (20,845 rows;
cols `ID, band, status, word, Pos, COCA-All, COCA-Acad, ratio, disp, range` —
the 20k most-frequent COCA-Academic words, academicwords.info).

**Why 20k matters:** a dense distance matrix is O(V²). 5k → 25M (fine); **20k →
~400M cells** (borderline); full 151K → 23 GB (infeasible). The facet is O(V)
storage + O(1) render — the substrate's whole point. So 20k is precisely where
"store the metric, render the distance" beats "materialize the matrix."

**The 20k list is STRUCTURAL evidence for the lane separation.** Its columns are
almost ALL header/routing signals — `band` (frequency-LOD stratum), `ratio`
(COCA-Acad / COCA-All = academic-register routing), `disp` (dispersion =
confidence/attention), `Pos` (grammatical routing) — with the distributional
*payload* (subgenre/embedding) held SEPARATELY. The academic list is a **header
table**; it confirms prefix+frequency+register belong in the classid, not the
distance. (No per-genre breakdown here — so the 20k fidelity leg needs the
distributional codebook; the 5k `lemmas_5k.csv` 8-genre proxy carries the §3
lane test.)

**Header packing for 20k (design):** `classid(4B)` = `band` (LOD) · quantized
`ratio` (register axis) · `Pos` · lemma-family prefix. This is the codebook-scope
+ attention header; the 6×(FisherZ:FisherZ) payload is the (separately-sourced)
distributional meaning.

## 3b. Wiring into the 12×12 = 144-verb universal grammar (design hook)

The rung-2 **144 verb-atoms** (`.claude/v3/knowledge/persona-vs-rung-ladder.md`:
"rung 2 = 144 verb atoms") are a universal, interpretable semantic basis — the
grammar analog of the 63 NSM primes DeepNSM already anchors on. Wiring:

- **The coarse centroid level = the 144-verb basis, not anonymous k-means.** Each
  word's coarse byte indexes its nearest verb-atom (144 < 256 → fits one byte),
  making the coarse address **interpretable** (a verb, not centroid #173) and
  **universal** (language-agnostic grammar). The fine byte is the within-atom
  residual. So a subspace's `256:256` = `(verb-atom : residual)`.
- **12×12 = the coarse address is a `12:12` pair** — the two doz­enal axes of the
  universal grammar (e.g. subject-role × predicate-role, or aspect × mood),
  mapping directly onto the `(8:8)`-byte-pair carving (144 lives in the low bits
  of the coarse byte).
- **Anchoring = the NSM-prime pattern, extended.** DeepNSM already gives each of
  63 NSM primes a CAM fingerprint (`nsm_primes.json`); the 144 verb-atoms are the
  same move at grammar scale — they become the coarse codebook, and every word
  resonates onto them. The `Pos='v'` rows in `academic_20k.csv` (thousands of
  verbs) are the population the 144 basis must span.

**Status: DESIRED WIRING (v2 hook), gated on the 144-verb basis being available
as fingerprints.** It does NOT change the §3 lane test (which is basis-agnostic —
it only tests header-vs-payload for `frequency`). It DOES sharpen the equilibrium:
if the coarse anchor is the 144-verb grammar, the "attention header" (§2 PF) is
literally routing a word to its grammatical verb-atom — the header lane and the
universal grammar are the same structure.

## 4. Replication protocol (BLIND — read before running)

- **Implement independently from THIS spec.** Do NOT import the other session's
  probe code. Diverse implementations (e.g. Rust vs Python) strengthen the
  witness — if two independent impls on the same committed CSV agree on the
  collapse count + anchor inversions, the result is real, not a single-session
  artifact.
- **Do NOT publish results** in this PR, in PR comments, in `EPIPHANIES.md`, or
  any shared board file, until the operator confirms both sessions have run.
  Report your numbers to the **operator only**. The operator adjudicates the two
  blind results.
- Determinism: no RNG anywhere (fixed pair sets by rank). Same CSV + same spec ⇒
  the two impls should converge if the design is real.

## 5. If PASS — what lands (v2, AFTER operator adjudication)
- Wire DeepNSM word → `classid[norm(prefix,freq,PoS)] + 6×(FisherZ:FisherZ)`.
- Distance = analytic Fisher-z (48 B γ) with materialized palette256 as the
  optional cache (le-contract "table = cache of the formula").
- The full hierarchical 256²-vs-flat-256 fidelity leg (needs the `.bin` codebook)
  is the named v2.
