# Framing the resilience study as CLAM (hierarchy) + CHAODA (anomaly ensemble)

*The resilience study is not a bespoke method — it is a CLAM cluster tree with a
CHAODA-style anomaly ensemble, on the electrical-distance manifold. Both already
exist in `ndarray::hpc::clam` (cited below); this doc maps the correspondence and
states honestly what is grounded vs what is a conceptual mapping not yet wired.*

> Operator prompt (2026-06-16): "you could even try to frame it — CLAM
> (resilience) / CHAODA". Companion to `PAPER.md`, `COUNTRY_STUDY.md`,
> `src/columns.rs`.

## The correspondence

| resilience-study object | CLAM / CHAODA construct | ndarray reference |
|---|---|---|
| recursive Cheeger/HHTL bisection into basins | **CLAM cluster tree** (`ClamTree::build`) | `hpc/clam.rs` `ClamTree` |
| a basin (compartment) | a **`Cluster`** | `hpc/clam.rs:106` `Cluster { radius, cardinality, lfd }` |
| basin algebraic connectivity λ₂ / mean R | cluster **radius** / spread | `Cluster::radius` |
| basin node count | cluster **cardinality** | `Cluster::cardinality` |
| how fragmented/space-filling a basin is | **local fractal dimension** `Lfd` | `hpc/clam.rs:81` `Lfd::compute(count_r, count_half_r)` |
| fail-first / exposure ranking | **CHAODA anomaly score** | `hpc/clam.rs:1517` `ClamTree::anomaly_scores() -> Vec<AnomalyScore>` |
| "this compartment can't wait" flag | CHAODA **flag threshold** (≥ 0.75) | `hpc/clam.rs` anomaly-flag test |

So the study's machinery is CLAM's machinery on a different metric: instead of a
Hamming/embedding distance, the manifold is the **electrical distance** (effective
resistance `R_ij = (e_i−e_j)ᵀ L⁺ (e_i−e_j)`, the self-inverse `L⁺` reference). The
HHTL tiers ARE the CLAM tree depth; the weakest compartment IS the cluster CHAODA
would score as the outlier.

## Why the three axes ARE a CHAODA ensemble (the load-bearing match)

CHAODA's thesis: **no single graph-anomaly method wins; ensemble several *diverse*
detectors** (relative cardinality, parent/child cardinality ratio, graph
neighbourhood, stationary distribution, …) and the gain comes from their
*non-redundancy*. The resilience study's three axes are exactly such an ensemble:

- **topology** (λ₂ / Kirchhoff) — the connectivity detector,
- **buffer** (inertia storage) — the transient detector,
- **policy** (feed-in / dispatch) — the operational detector,

ensembled into the **exposure** score. And the study's measured **low / negative
Cronbach α** (the axes are distinct facets, `Spearman ≈ 0` between them) is not a
defect — it is *precisely CHAODA's design goal*: low inter-detector correlation is
what makes the ensemble add information rather than restate it. The discriminant
finding and the CHAODA non-redundancy principle are the same statement.

This also re-frames the §4.11 confound cleanly: the modifier `Weyl × (1/Fiedler)`
failed as an independent axis because `1/λ₂` is the dominant Kirchhoff term — i.e.
it was a **redundant detector**, the CHAODA anti-pattern. The buffer axis is the
*orthogonal* detector the ensemble actually needed.

## Honest scope

- **Grounded [G]:** `CLAM`, `Cluster{radius,cardinality,lfd}`, `Lfd`, and
  `ClamTree::anomaly_scores` all exist in `ndarray::hpc::clam` (cited). The
  structural correspondence is exact, not metaphor.
- **Conceptual [H]:** `perturbation-sim` is zero-dep and is **NOT wired** to
  `ndarray::hpc::clam`. The mapping above is read off the APIs, not run. No code
  here calls `ClamTree` or `anomaly_scores`.
- **The falsifiable probe** that would promote [H]→[G]: build a `ClamTree` over the
  contingency factor vectors (or the per-basin `(λ₂, Kf, buffer)` rows), run
  `anomaly_scores`, and correlate the CHAODA ranking against the study's exposure
  ranking (ICC/Spearman, Jirak rate). If they agree, the study *is* CHAODA on the
  electrical manifold; if not, the framing is rhyme and gets retracted. This is the
  gated bridge (crosses perturbation-sim's zero-dep boundary into `ndarray`,
  behind a feature flag) — analogous to the calibration harness, not yet built.

## Tie-in to the calibrated columns (`src/columns.rs`)

CLAM gives two more value members for free, hung off the same HHTL-OGAR key:
`radius` (basin spread) and `lfd` (local fractal dimension). They are helix-residue
value members like the rest — orthogonal to topology by the key/value split — and
the CHAODA `anomaly_score` is the *read* over the column set, the same way
`exposure` is. The substrate that carries the study is therefore literally a CLAM
tree of HHTL-keyed helix value members with a CHAODA read.
