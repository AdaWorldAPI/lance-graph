# METHODS — mathematical grounding & anti-dilution notes

This crate composes several named theorems. They are easy to blur together
("it's all spectral graph stuff") and blurring them produces wrong claims. This
document pins each method to its **object**, its **theorem**, and the
**distinctions that must never collapse**, then shows how all four connect and
how a statistician validates them with ICC / Pearson / Spearman / Cronbach.

---

## 0. The one operator (the grounding that connects everything)

Every method here is a reading of **one object**: the weighted graph Laplacian

```
L = B · diag(b) · Bᵀ          (n×n, SPSD, one zero eigenvalue per component)
```

`B` = node–edge incidence, `b_e = 1/x_e` = line susceptance. The four methods
are four *functions of L*:

| Method | Function of L | Module |
|---|---|---|
| Effective resistance / embedding | `L⁺` (Moore–Penrose pseudo-inverse) | `basin::effective_resistance`, `spectral_embedding` |
| Spectral perturbation (Weyl/Davis–Kahan) | `spec(L)` — the eigenpairs `{λ_k, v_k}` and how a rank-1 `ΔL` moves them | `perturbation.rs` |
| Cheeger exchange rate | `spec(D^{-1/2} L D^{-1/2})` — the **normalized** gap `μ₂` | `basin::cheeger_sweep` |
| Kron / basin tiering | `L / L_II` — the Schur complement | `basin::kron_reduce` |
| Cascade (local collapse) | DC flow `θ = L⁺ p`, `f = b·Bᵀθ`, + threshold trips | `flow.rs`, `cascade.rs` |

The cascade is the only *nonlinear* layer (thresholds); the other four are
linear-algebraic. **The unifying statement: `L` is the operator; the methods are
its pseudo-inverse, its spectrum, its normalized spectrum, and its Schur
complement.** That is why they can be cross-validated against each other (§5) —
they are not independent instruments, they are projections of one object, so
their *disagreements* are informative.

---

## 1. Weyl + Davis–Kahan — spectral perturbation (`perturbation.rs`)

- **Object:** the **combinatorial** Laplacian spectrum; the Fiedler value
  `λ₂` (algebraic connectivity).
- **Line trip = rank-1 perturbation:** `E = L' − L = −b_k(e_a−e_b)(e_a−e_b)ᵀ`,
  `‖E‖₂ = 2·b_k`.
- **Weyl:** `|λᵢ(L+E) − λᵢ(L)| ≤ ‖E‖₂` for every `i`. (tested per line)
- **Davis–Kahan:** Fiedler-vector rotation `sinθ ≤ ‖E‖₂ / gap`, `gap` = the
  separation of `λ₂` from its neighbours. (tested)
- **Reading:** the *Raumgewinn* (territory) shift of one contingency.

## 2. Effective resistance + embedding (`basin.rs`)

- **Object:** `L⁺`. `R_ij = (eᵢ−eⱼ)ᵀ L⁺ (eᵢ−eⱼ) = L⁺_ii + L⁺_jj − 2L⁺_ij`.
- **Theorem:** `R` is a **metric** (resistance distance; triangle inequality
  holds — tested). `spectral_embedding` returns low-eigenvector coordinates,
  giving an *electrical* embedding.
- **Anti-dilution — the geography trap:** a Morton / HHTL tile MUST be built on
  this embedding (or on `R`), **not on lon/lat**. Geographic adjacency ≠
  electrical adjacency. Tiling by geography is a *rhyme*, [S]; tiling by `R` /
  spectral coords is principled, [H→G]. (This is the one correction that makes
  the Morton-cascade idea valid.)

## 3. Cheeger — the field↔cut exchange rate (`basin.rs`)

- **Object:** the **normalized** Laplacian `D^{-1/2} L D^{-1/2}`, gap `μ₂`.
- **Theorem (Cheeger):** `μ₂/2 ≤ h(G) ≤ √(2·μ₂)`, where `h(G)` = minimum
  conductance (the min-cut). The Fiedler **sweep cut** realizes a `φ` with
  `μ₂/2 ≤ h(G) ≤ φ ≤ √(2μ₂)`. (both bounds tested)
- **Reading:** the literal conversion between the *field eigenvalue*
  (Raumgewinn) and the *cut* (infight). A blackout is a cut materializing;
  `μ₂ → 0` ⇒ the board is cheap to cut.
- **Anti-dilution — two different λ₂:** the **combinatorial** `λ₂` (§1, what
  Weyl perturbs) and the **normalized** `μ₂` (§3, what Cheeger bounds) are
  eigenvalues of *different operators*. Both get called "algebraic
  connectivity" loosely. Keep them separate: Weyl needs combinatorial; Cheeger
  needs normalized. The crate stores `fiedler_*` (combinatorial) in
  `SpectralPerturbation` and `mu2` (normalized) in `Cheeger`, never merged.

## 4. Kron reduction — basin tiering & cross-border (`basin.rs`)

- **Object:** the Schur complement `L_red = L_BB − L_BI L_II⁻¹ L_IB`.
- **Theorem (Dörfler–Bullo 2013):** `L_red` is a valid loopy Laplacian on the
  boundary buses that **preserves effective resistance** between them exactly.
  (tested) ⇒ a basin → one super-node with ports.
- **Cauchy interlacing:** the eigenvalues of the principal interior block
  `L_II` interlace `L`'s: `λ_k(L) ≤ λ_k(L_II) ≤ λ_{k+(n−m)}(L)`. (tested) ⇒
  per-tier `λ₂` *bounds* the parent's — the HHTL hierarchy is spectrally
  consistent, so you never eigensolve the whole continent at once.
- **HHTL mapping:** TWIG = buses, HIP = basins (Kron-reduced), HEEL =
  cross-border super-graph of basin equivalents joined by interconnector edges.

---

## The Go-meta (how all four compose into one decision)

| Go | Grid quantity | Method | Tier |
|---|---|---|---|
| **Infight** (local life/death) | cascade trip fraction | §-cascade | TWIG |
| **Raumgewinn** (territory) | `λ₂` / Fiedler shift | §1 | HEEL |
| **Exchange rate** | `μ₂/2 ≤ h ≤ √(2μ₂)` | §3 Cheeger | meta |
| **Reading ahead** (basin as a stone) | Schur complement + interlacing | §4 | HIP↔HEEL |

`basin::infight_vs_raumgewinn` runs the cascade once and classifies the
contingency `Regime::{Infight, Raumgewinn, Balanced}`: a *bridge cut* is
`Raumgewinn` (few trips, big `λ₂` collapse); a *meshed-corridor overload* is
`Infight` (many trips, connectivity holds). (tested)

---

## 5. Statistics — the four methods as a measurement battery

`basin::contingency_features` returns, per seed contingency, a 5-vector of
**different properties of the same operator**:

```
x = [ d_lambda2,   dk_rotation,   d_conductance,   infight,   raumgewinn ]
       Weyl Δλ₂     Davis–Kahan     Cheeger Δφ        cascade     1−λ₂'/λ₂
```

Run it over a set of `m` contingencies → an `m×5` feature matrix `X`, plus an
observed-severity vector `y` (buses lost / energy-not-served, from
ENTSO-E/ESIOS outage data). Then, with `ndarray::hpc::reliability`:

### Reliability — *do the instruments agree / cohere?*
- **Cronbach α** over the 5 columns (z-scored): internal consistency of the
  battery. **High α (>0.9)** ⇒ the five readings load on ONE latent factor
  ("grid stress") → fuse into a single criticality index. **Low α (<0.7)** ⇒
  they capture *distinct facets* (infight vs Raumgewinn really are different) →
  keep the vector, do NOT average. α is the gate on "is there one number or
  four?".
- **ICC(2,1)** treating the five methods as *raters* of each contingency's
  criticality (after z-scoring to a common scale): absolute-agreement
  inter-method reliability. **ICC < Pearson among the methods** is the
  signature of *systematic rater bias* — e.g. the spectral methods (§1, §3)
  over-rate territorial cuts while the cascade (§-cascade) over-rates local
  fights. That bias is exactly the Go duality showing up as a measurable
  ICC-vs-Pearson gap, not noise.

### Validity — *do the instruments predict the truth?*
- **Pearson r(xₖ, y):** criterion validity, linear — which property best
  predicts severity.
- **Spearman ρ(xₖ, y):** the **operational** metric — which property best
  *ranks* the worst contingencies (N−1 screening is a ranking problem; ρ is
  scale-free and robust to the heavy tails of cascade sizes).
- **Convergent vs discriminant validity:** methods that *should* agree
  (`d_lambda2` vs `d_conductance`, both field) should correlate (convergent);
  methods that *should* separate (`infight` vs `raumgewinn`) should de-correlate
  on territorial contingencies (discriminant). Both are testable with the
  correlation matrix of `X`.

### Control variables — partial correlation (the key request)
Because the five are properties of *one* operator, they are natural **mutual
controls**. The unique contribution of each scale is the **partial
correlation**:

```
r(x, y | z) = ( r_xy − r_xz · r_zy ) / √( (1 − r_xz²)(1 − r_zy²) )
```

- *Does local collapse predict severity beyond the global cut?* →
  `r(infight, y | d_conductance)`. If it drops to ~0, the cut already explained
  it (Raumgewinn-dominated regime); if it stays high, infight carries unique
  signal.
- *Does the spectral shift add anything over the cascade?* →
  `r(d_lambda2, y | infight)` — incremental validity of the field eigenvalue.
- Generalize to **partial Cronbach / part correlations** to strip a controlling
  facet before assessing the remainder's consistency.

This is how the four become control variables: each method's property is a
covariate you partial out to isolate another's unique explanatory power.

### Significance — Jirak, never IID
Contingencies on a grid are **weakly dependent** (shared lines, overlapping
cascades, spatial correlation), so every p-value / confidence interval / "N σ
above the noise floor" must use the **Jirak 2016** rate `n^(p/2−1)` (arXiv
1606.01617), **not** classical IID Berry–Esseen — which understates error and
inflates significance. Use **Fisher-Z** (`helix::fisher_z`, `z = arctanh(r)`)
to build correlation CIs, but with the effective sample size deflated for
dependence (the IID `Var ≈ 1/(n−3)` is optimistic; Jirak gives the honest
floor). See `ada-docs/research/JIRAK_MATH_THEOREMS_HARVEST.md` §3 and the
`I-NOISE-FLOOR-JIRAK` iron rule.

### Workflow (turnkey)
```rust
let feats: Vec<[f64;5]> = seeds.iter()
    .map(|&s| perturbation_sim::contingency_features(&grid, &p, s, cfg).as_row())
    .collect();
// columns → ndarray::hpc::reliability:
//   icc_a1(&[col_k, y])           // per-method absolute agreement with truth
//   pearson(col_k, y), spearman(col_k, y)
//   cronbach(&columns)            // battery internal consistency
//   FidelityReport::compute(col_k, y)  // one-shot r/ρ/ICC/α
// partial correlations from the r-matrix via the formula above.
// significance: Jirak n^(p/2−1), Fisher-Z CIs with deflated n.
```

---

## 6. Fast-sketch synergy (`sketch.rs`, PROTOTYPE)

The XOR/bundle/JL machinery *is* a field-tier accelerator — two pieces.

- **Spielman–Srivastava resistance sketch** (`resistance_sketch`). `R_eff(u,v) =
  ‖W^{1/2}B L⁺ (e_u−e_v)‖²` exactly (because `Mᵀ M = L`), so `k = O(log n/ε²)`
  random ±1 projections give an unbiased JL estimate `‖z_u − z_v‖²`. The ±1
  rows are a `vsa_bundle` of sign fingerprints; the distance readout is the
  σ-band/Hamming readout. **[G]** (theorem). **Honest scope:** the prototype
  uses dense `L⁺`, so it demonstrates *accuracy* (tested < 12% rel-err at
  k=6000 on the bridge graph), **not** the asymptotic speed win — that needs a
  fast Laplacian solver. Value is at continental `n` where the exact eigensolve
  dies; at demo `n` the exact path wins. Error bars are **Jirak**, not IID.
- **Walsh/Morton pyramid screen** (`walsh_pyramid_energy`, `fwht`). The WHT of a
  node field, grouped into dyadic (Morton/quadtree) levels: coarse (low-
  sequency) energy = global/**Raumgewinn**/collapse, fine = local/**infight**.
  One `O(N log N)` pass; the sign side (XOR/`bind`) of the pyramid. **[H]** — a
  *screen*: the Walsh basis equals the graph eigenbasis only on hypercube-
  structured graphs, so coarse energy *flags* candidate collapse regions that
  the exact eigensolve (`perturbation.rs`/`basin.rs`) then certifies. Tested:
  smooth field is coarse-dominated, a spike is fine-dominated; `fwht∘fwht = N·I`.

## 7. Gaussian-splat magnitude side (`splat.rs`, PROTOTYPE)

The **magnitude** side of the same pyramid (the `vsa_bundle` algebra; §6 is the
sign side). EWA splatting *is* anisotropic mip-filtering — its reason to exist
is anti-aliasing resampling between pyramid levels.

- **`splat_neighborhood`** fits an SPD covariance `Σ` to a bus's local
  *electrical* neighbourhood (the `spectral_embedding` coords, resistance-
  closeness weighted). `Σ`'s anisotropy is the spread direction ≈ the cut
  normal. Tested symmetric + PSD. **Splats in electrical coordinates, never
  geography** — the one correction that makes the Morton idea valid.
- **`ewa_coarsen` vs `box_coarsen`** coarsen one pyramid level. A z-order seam
  ("Z-jump") groups spatially-distant cells; a hard box-average aliases them in,
  EWA's Gaussian footprint down-weights them. Tested: wide-σ EWA → box (limit);
  tight-σ EWA suppresses a seam outlier (box ≈ 34 → EWA ≈ 1). **[H]** — shows
  the construction + the seam fix; the `Σ` push-forward up the pyramid is the
  certified `jc::ewa_sandwich` / ndarray pillar-12 `J·Σ·Jᵀ`, not re-derived.
- **`morton2`** — the 2-bit/axis Z-order interleave (the 4×4 tile = the EWA
  footprint quantum). Tested against known codes.

Together §6+§7 are the OGAR bipolar-phase pyramid: **sign (Walsh/XOR — the
infight↔Raumgewinn scale) × magnitude (EWA Gaussian splat — the anisotropic
neighbourhood footprint)**. The two-algebra rule (`I-VSA-IDENTITIES`): sign side
= XOR/`bind`, magnitude side = `bundle`/EWA, never mixed.

## 8. The fidelity ladder is iterative, not a fork that limits the design

Physical fidelity is a *sequence of iterations on one extensible substrate*, not
a one-time DC-vs-AC choice. The design must not bake in DC; each rung adds data
to the same `Grid`/`Edge`/cascade and the same `L`-derived methods:

| Iteration | Adds | Reuses unchanged |
|---|---|---|
| **cheap** | `Edge` gains `r` (resistance) → `loss_gradient` = `I²R` field; temporal driver runs the cascade per time slice → test-retest ICC | all of §1–§7 (the methods are functions of `L`; richer weights just change `L`) |
| **medium** | event-driven cascade: relay inverse-time curves + thermal inertia → trip *timing*; tech-debt modifiers (age→derate/R/relay/failure-prior) as control variables | `simulate_outage` becomes a timed variant; Weyl/Cheeger/Kron/sketch/splat untouched |
| **fork** | full **AC** π-model (`R+jX+jB/2`, voltages, reactive Q, Newton–Raphson) → voltage-collapse mode + true losses | the Laplacian generalizes to the complex `Y_bus`; the spectral/effective-resistance/Cheeger/Kron machinery has complex analogues — the field tier carries over |

**Design rule:** keep every method a function of the (possibly complex,
possibly time-indexed) admittance operator, and keep parameters as `Edge`/`Node`
data columns + modifier functions. Then "cheap → medium → fork" is *adding
columns and swapping the solver*, never rewriting the field tier. The data
ceiling (proprietary asset condition) is handled by priors + sensitivity +
disclosure (`n_estimated_*`), never invented numbers.

## Anti-dilution table — the distinctions to never collapse

| Do NOT conflate | Because |
|---|---|
| Weyl *equidistribution* (`jc::weyl`) vs Weyl *eigenvalue perturbation* (`perturbation.rs`) | same surname, different theorem; only the latter bounds `Δλ` |
| Combinatorial `λ₂` (Weyl) vs normalized `μ₂` (Cheeger) | different operators; Cheeger's constants only hold for `μ₂` |
| Geographic adjacency vs effective-resistance distance | spatial ≠ electrical; Morton/HHTL must ride `R`, not lon/lat |
| Infight (cascade, combinatorial, threshold) vs Raumgewinn (`λ₂`, spectral, smooth) | two value systems; Cheeger is the *only* rigorous bridge |
| Reliability (α, ICC: do instruments agree) vs validity (r, ρ vs `y`: do they predict truth) | high reliability with low validity = consistent but wrong |
| ICC vs Pearson | Pearson ignores systematic bias; ICC catches it (the method-bias = the Go duality) |
| IID Berry–Esseen vs Jirak weak-dependence | grid contingencies are dependent; IID inflates significance |
| Estimated `x`/`s_nom` (OSM proxy) vs measured | DC screening proxy, not as-built protection data |
| SS sketch *accuracy* vs *speed* | exact in expectation (JL), but "fast" needs a fast Laplacian solver this crate lacks — at demo `n` the exact path wins |
| Walsh pyramid *screen* vs exact partition | basis = graph eigenbasis only on hypercubes; it flags, the eigensolve certifies |
| Splat in geography vs electrical embedding | a lon/lat splat is a rhyme; a `spectral_embedding` splat is principled |
| Fidelity ladder = iterations vs a limiting fork | cheap→medium→fork adds columns + swaps solver; never rewrites the field tier |
