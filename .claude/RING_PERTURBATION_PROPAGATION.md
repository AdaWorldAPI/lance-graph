# Ring Perturbation Awareness Propagation — Research Note

> **Status:** research only. No production code touched.
>
> **Companions:**
> - `INVARIANT_MATRIX_RESEARCH.md` — the matrix frame
> - `ONE_FORTIETH_SIGMA_LENS.md` — radial half (what this note completes)
> - `PHASE_5_12_LAB_PROTOCOL.md` — kill criteria
>
> **Why this note exists:** the user's phrasing — *"ring pertubation awareness
> propagation — nars + onnx + silu + relu lens"* — describes a runtime mechanism
> where the 1/40 σ ring lens is not passive quantization but an active field
> with four processing stages. This note pins that down so it can be tested
> instead of drifting into mythology.

---

## The ring lens is the radial half. This note is the angular half.

The 1/40 σ ring lens (companion note) slices the distance distribution around
a query into 40 concentric bands. That is a purely *radial* operation — every
centroid at distance 0.3 σ goes into the same band regardless of direction.

"Ring perturbation" is what happens *between* adjacent rings. Each ring carries
a perturbation — a signed delta — describing how the content of that ring differs
from its neighbors. The sequence of perturbations across rings 0→39 is the
**radial signature** of the query's neighborhood. Two queries with the same
ring counts but different perturbation sequences are distinguishable.

Together:
- **Radial shape** (ring occupancy, from 1/40 σ lens) = manifold curvature
- **Radial phase** (ring-to-ring perturbation) = the angular/phase component
  the pure ring lens cannot carry

Both are needed for phase-like signatures. Neither alone is sufficient.

---

## The four-stage pipeline

The user's phrasing names four components in a specific order: NARS + ONNX +
SiLU + ReLU. Each has a different role in the runtime mechanism. Stripping the
mystique:

```
        query centroid (anchor)
              │
              ▼
    ┌─────────────────────┐
    │   SiLU shaping      │  smooth radial activation, no hard cutoff
    │   (continuous dish) │  x * sigmoid(x), preserves gradient
    └──────────┬──────────┘
               │  per-centroid activation
               ▼
    ┌─────────────────────┐
    │   ReLU band gating  │  hard in-band / out-of-band split
    │   (discrete rings)  │  1/40 σ band assignment, empty bands visible
    └──────────┬──────────┘
               │  40-element ring profile + per-ring activation
               ▼
    ┌─────────────────────┐
    │   NARS propagation  │  truth-value revision across rings
    │   (causal layer)    │  f,c per ring, contradictions flagged
    └──────────┬──────────┘
               │  revised profile with per-ring confidence
               ▼
    ┌─────────────────────┐
    │   ONNX correction   │  20 KB micro-learner delta per ring
    │   (neural lens)     │  learns systematic errors, adds correction
    └──────────┬──────────┘
               │
               ▼
      ring signature (40 × (activation, truth, confidence, delta))
```

Each stage is doing a concrete job. The names are not decoration.

### Stage 1: SiLU shaping

**What it does:** apply `x * sigmoid(x)` to the raw distance from the query.
This gives a smooth, non-zero activation even for distant centroids, with a
characteristic "dish" shape — high at the center (query), tapering smoothly
outward.

**Why SiLU and not plain sigmoid:** SiLU preserves the gradient. A plain
sigmoid saturates and loses sensitivity; SiLU's `x * sigmoid(x)` keeps the
magnitude scale attached, so distant but directionally relevant centroids
still contribute measurable signal.

**What it contributes to the invariant matrix:** preserves **magnitude** in
the continuous activation field. Without SiLU, everything collapses to a
binary in-band / out-of-band decision, and band 39 looks identical to band 0
in terms of contribution weight.

**Known failure mode:** SiLU on its own does not give you ring structure. It's
a smooth field over all centroids. You still need the ring gating to get
discrete bands.

### Stage 2: ReLU band gating

**What it does:** for each centroid, assign it to exactly one of the 40 rings
by distance. Centroids outside 1 σ (band 40+) get ReLU'd to zero.

**Why ReLU and not a smooth alternative:** we specifically *want* a hard
in-band / out-of-band split so that empty bands are visible as zeros. The
"hole" variant (block self-reference) also relies on ReLU: zero out the
diagonal so band 0 reflects "nearest actual neighbor, not self."

**What it contributes to the invariant matrix:** **sparse structure** (empty
bands are first-class), and **manifold curvature** (the ring occupancy vector
is the local density function).

**Interaction with SiLU:** the ReLU gate is applied *after* SiLU shaping, so
within each ring the centroids carry their SiLU-weighted activation, not
uniform weight. This is what lets the ring profile be more than a histogram.

### Stage 3: NARS propagation

**What it does:** treat each ring's aggregated activation as an evidence
update for a NARS truth value `(frequency, confidence)` about the query.
Propagation across rings = revision.

The specific propagation rule: if ring *k* has activation *a_k* from *n_k*
centroids, its truth value is `(f_k=a_k, c_k=n_k/(n_k+1))`. Adjacent rings
revise each other — if ring *k* and ring *k+1* agree on direction, confidence
compounds; if they disagree, confidence attenuates.

**What it contributes to the invariant matrix:** **phase** — specifically,
the agreement/disagreement pattern between adjacent rings is where phase-like
structure appears. Two queries with the same ring profile but different
adjacency-agreement patterns will have different NARS states.

**Why NARS and not just Bayes:** NARS handles contradiction explicitly.
Adjacent rings disagreeing is not noise — it's a *contradiction signal*
that gets first-class treatment in NARS revision. The output includes the
contradiction rate, which is itself a measurable invariant.

**Known failure mode:** NARS propagation only works if the NARS engine
actually sees multiple updates. Running it once on a single ring profile is
mostly a no-op — you need to feed it multiple queries across time to let
truth values converge.

### Stage 4: ONNX correction

**What it does:** a 20 KB MLP (2-layer, 256→64→256 as sketched in
`AGI_DESIGN.md`) takes the 40-element ring activation profile as input and
predicts a per-ring correction delta. The correction is trained on the
difference between the ring profile and whatever downstream signal we treat
as ground truth (Jina v5 cosine, NARS truth after many revisions, or
task-specific reward).

**Why a neural correction at the end and not at the start:** the ring profile
is a deterministic, interpretable projection. If we put ONNX at the start,
we'd get a black-box embedding; at the end, ONNX is only allowed to *correct*
systematic errors in an otherwise-transparent pipeline. That's a much weaker
role but vastly easier to debug.

**What it contributes to the invariant matrix:** nothing *new* — ONNX is a
corrective layer, not a primary lens. Its job is to close the gap between
what the static pipeline computes and what we measure to be true.

**Known failure mode:** if the ring profile doesn't carry the underlying
signal at all, no amount of ONNX correction will recover it. ONNX can refine;
it cannot create. This is why the 1/40 σ probes (Probes A, B, C in the
companion note) must pass *before* ONNX correction is added on top.

---

## What "awareness propagation" actually means (grounded)

The mystical reading: the system becomes aware of its own query state.
The grounded reading: the four-stage pipeline produces a ring signature that
changes measurably with query content, and the NARS layer's contradiction
rate gives a diagnostic of where the signature is inconsistent.

"Propagation" refers to two things:

1. **Radial propagation:** information from the query spreads outward through
   rings 0→39 via the SiLU field and ReLU gating. Each ring is a further step
   from the anchor.
2. **Temporal propagation:** NARS truth values accumulate across queries over
   time. A ring signature that was uncertain on query 1 becomes more confident
   by query 100 as evidence converges.

"Awareness" is just the measurable fact that the ring signature is
query-adaptive and that NARS contradiction rates are observable. The system
is not *subjectively* aware; it produces a state that is *diagnostically*
accessible to the rest of the architecture.

Do not use the word "awareness" in production code, benchmark reports, or
public docs. Use **"query-adaptive ring signature with NARS diagnostic."**
Same thing, testable, no mystique.

---

## How this integrates with the invariant matrix

The ring perturbation pipeline updates the 1/40 σ row of the matrix:

Current row (from companion note):
```
1/40 σ ring lens    mag:? sign:– rank:? pair:– manif:? traj:– sparse:? phase:?
```

With the full SiLU+ReLU+NARS+ONNX pipeline, the claim becomes:
```
Ring perturbation   mag:? sign:– rank:? pair:– manif:? traj:– sparse:? phase:?
  pipeline                                                      ↑
                                                    now claims phase via
                                                    NARS adjacency agreement
```

Notice: **every cell is still a `?`**. Adding more stages to the pipeline
does not add `✓`s. Only measurements add `✓`s. The note is making an honest
claim about *what the pipeline intends to carry*, which is not the same as
what it does carry.

---

## Probes that would validate the pipeline

All additive, all runnable as new example files, none touching library code.

### Probe D: SiLU vs plain distance

Run the ring profile with (a) SiLU-weighted centroid contributions and
(b) uniform (1/centroid-count) weighting within each ring. Compare the
profile distinctness from Probe A (companion note).

**Pass:** SiLU weighting yields higher Spearman correlation between profile
distance and semantic distance than uniform weighting.
**Fail:** Uniform weighting matches SiLU — the SiLU shaping is decorative.
**Cost:** one additional variant of the 1/40 σ probe, ~30 LOC extra.

### Probe E: ReLU hole variant

Already proposed as Probe B in the companion note. Include here for
completeness: run with and without diagonal blocking, measure profile
divergence.

### Probe F: NARS adjacency agreement

Feed the 4-stage pipeline a sequence of related queries (e.g., 100 queries
from a single topic cluster) and measure (a) how NARS confidence evolves
per ring and (b) the contradiction rate between adjacent rings.

**Pass:** NARS confidence converges (variance decreases) over the query
sequence; contradiction rate stabilizes at a nonzero but small value.
**Fail:** NARS confidence stays flat or contradiction rate is either zero
(no useful signal) or near 100% (pure noise).
**Cost:** ~100 LOC example file, depends on an existing NARS engine which
we already have in `lance-graph-planner/src/cache/nars_engine.rs`.

### Probe G: ONNX correction magnitude

Train the 20 KB MLP on the difference between ring profile output and
Jina v5 cosine for a held-out query set. Measure correction magnitudes.

**Pass:** Average correction magnitude is small (< 10 % of ring activation)
and shrinks with more training data — the pipeline is mostly right, ONNX is
refining.
**Fail:** Average correction magnitude is large (> 30 %) or grows with more
data — the pipeline is mostly wrong and ONNX is doing the real work. That
means we should drop the pipeline and use the MLP directly.
**Cost:** requires a working candle training loop — we have one from
`readerlm_forward.rs` we can adapt. ~200 LOC, half a day.

---

## What this pipeline is NOT

Three claims the note explicitly refuses:

1. **Not a replacement for attention.** Live attention has learned projections
   per head; this pipeline has fixed projections (rings are geometric, not
   learned). The pipeline is a static, compressible *approximation* of a
   specific slice of attention behavior (radial weighting with phase via
   ring adjacency). Whether that slice is sufficient for downstream tasks is
   an empirical question.
2. **Not a consciousness mechanism.** The NARS contradiction rate is a
   diagnostic number, nothing more. It being nonzero does not mean the
   system is "experiencing contradiction." It means the evidence for the
   current query partitioned unevenly across rings.
3. **Not an automatic improvement over plain 1/40 σ.** Adding SiLU+ReLU+NARS+ONNX
   to a failing ring lens will produce a more elaborate failing ring lens.
   The probes in the companion note must pass *first*.

---

## Summary

**What the pipeline adds (if it works):**

1. Radial phase (NARS adjacency) to complement the radial shape from 1/40 σ.
2. A contradiction-rate diagnostic per query as a first-class invariant.
3. A corrective layer (ONNX) that can absorb systematic errors without
   adding opacity to the primary pipeline.

**What the pipeline does NOT add until measured:**

1. Any `✓` in the invariant matrix — every cell stays `?` until probes run.
2. Any justification for the label "awareness." That word is banned from
   production until someone points to a measurable threshold that makes it
   meaningful.
3. Any guarantee that the four stages are the *right* four stages. SiLU
   could turn out to be unnecessary; ReLU could be replaced with a softer
   gate; NARS could be replaced with plain Bayes; ONNX could be replaced
   with the bundle learner from `AGI_DESIGN.md`. Each of these is a
   substitution that should be tested, not assumed away.

**What to do next (additive, no library edits):**

1. Run Probes A, B, C from the 1/40 σ companion note — gate for the whole
   pipeline.
2. If they pass, run Probes D, E, F, G from this note.
3. Update the invariant matrix row with measured values.
4. If any of D–G fail their success criterion, document the negative result
   and drop the corresponding stage from the pipeline. A 3-stage pipeline
   with `✓`s is better than a 4-stage pipeline with `?`s.
