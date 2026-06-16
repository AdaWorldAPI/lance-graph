# A Three-Axis Spectral Resilience Model for European Transmission Grids
## Ein Drei-Achsen-Spektral-Resilienzmodell für europäische Übertragungsnetze

*A mathematically verifiable construct-validity study. Topology is measured from
the open PyPSA-Eur/OSM network; inertia and policy are transparent, declared
priors. Every number below regenerates from a named example on public data.*

> Companion to `PAPER.md` (the methods paper) and the `perturbation-sim` crate.
> Reproduction commands in Appendix A. Grades: **[G]** proven/verified-in-code,
> **[H]** measured-but-model-dependent, **[S]** prior/illustrative.

---

## Epiphanies first / Kernideen zuerst

This study leads with its findings, then proves them.

1. **Grid resilience is not one number — it is three orthogonal axes.**
   *Topology* (algebraic connectivity `λ₂` / Kirchhoff index), *buffer*
   (transient inertia storage), and *policy* (feed-in / dispatch / import). A grid
   can be strong on one and fail on another. Collapsing them into one metric is the
   error that the **France paradox** exposes.

2. **The France paradox is the proof.** France has the **lowest measured `λ₂`** in
   the eight-country panel (`3.06e-7` — a vast, sparse, long-line grid) yet is among
   the **least exposed**, because its nuclear synchronous **buffer** is the highest.
   A topology-only screen red-flags France; only the separated buffer axis explains
   why it holds. The axes *must* be distinct. **[G] (measured λ₂) + [S] (buffer prior)**

3. **The amplifier we first reached for was a confound.** The modifier
   `m = Weyl × (1/Fiedler)` is not an independent signal: `1/λ₂` is the **dominant
   term of the Kirchhoff index**, so `m` silently re-measured the resistive axis. The
   genuinely independent third axis is the **buffer** (set by inertia, not topology):
   `Spearman(λ₂, buffer/node) ≈ 0`. **[G]**

4. **The self-inverse reference makes resilience readable once, not replayed.** The
   Laplacian pseudoinverse `L⁺` (`λ_k ↔ 1/λ_k`, `(L⁺)⁺=L`) integrates the response
   over *all* perturbations, so `λ₂` and `Kf` are read once from the field — never by
   re-predicting a specific trip. Verified exact in code (Moore-Penrose residual
   `1.3e-13`). **[G]**

5. **Unsupervised, the model localizes the 2025 fault region.** Run blind on the
   Iberian peninsula, the spectral cut isolates a **100-bus all-Spanish sub-region**
   (not the ES–PT border) — the weakest seam is inside Spain. And **Spain tops the
   exposure ranking** of the eight countries, matching the 28 Apr 2025 reality. **[H]**

6. **It is a fail-first investment locator with a named product.** The binding
   constraint per country names the intervention: buffer-bound → synchronous inertia
   (gas turbine / synchronous condenser / pumped storage), topology-bound → corridor,
   policy-bound → curtailment/forecast. Spain's buffer lever cuts modelled exposure
   **−50 %**, the largest in the panel. **[H] structure / [S] magnitude**

**DE (Kurzfassung).** Netz-Resilienz ist *drei orthogonale Achsen* — Topologie
(`λ₂`/Kirchhoff), Puffer (Trägheits-Speicher), Politik. Das **Frankreich-Paradox**
beweist die Trennung: FR hat das niedrigste gemessene `λ₂`, ist aber dank
Nuklear-Puffer kaum exponiert. Der zuerst gewählte Verstärker `Weyl×(1/Fiedler)`
war ein Confound (`1/λ₂` = dominanter Kirchhoff-Term); die unabhängige dritte
Achse ist der **Puffer** (`Spearman(λ₂, Puffer) ≈ 0`). Die selbst-inverse Referenz
`L⁺` macht Resilienz *einmalig lesbar* (verifiziert, `1.3e-13`). Unüberwacht
isoliert das Modell eine 100-Knoten-rein-spanische Region; **Spanien führt das
Exposure-Ranking** an. Es ist ein **Fail-first-Investitionslokator** (Puffer →
Synchron-Trägheit; Spanien **−50 %**). Nur die Topologie ist gemessen; Trägheit/
Politik sind deklarierte Prioren.

---

## 1. The model

### 1.1 The one operator

Every quantity derives from the **susceptance-weighted graph Laplacian**
`L = B · diag(b) · Bᵀ`, where `B` is the bus–line incidence matrix and
`bₑ = 1/xₑ` the line susceptance. `L` is symmetric positive-semidefinite with
ascending eigenpairs `(λ_k, v_k)`, `λ₁ = 0`.

### 1.2 Axis 1 — Topology (MEASURED [G])

Read directly from `L`'s spectrum:

- **Algebraic connectivity** `λ₂` — the worst-case structural margin.
- **Kirchhoff index** `Kf = n · Σ_{k≥2} 1/λ_k = n · trace(L⁺)` — total effective
  resistance, the response integrated over all balanced injections.
- **Mean effective resistance** `R̄ = Kf / C(n,2)` — a size-normalized topology
  density comparable across grids of different `n`.
- **Bisection stability** `s = (λ₃ − λ₂)/λ₂` — the Davis–Kahan gap ratio; `s ≳ 1`
  ⇒ a well-separated, trustworthy Fiedler partition, `s ≪ 1` ⇒ ambiguous.

The self-inverse reference: `L⁺` shares `L`'s eigenvectors with reciprocal
eigenvalues `1/λ_k`, and `(L⁺)⁺ = L` (an involution). So `1/λ₂` is the top
eigenvalue of `L⁺`, and effective resistance
`R_ij = (e_i−e_j)ᵀ L⁺ (e_i−e_j) = Σ_{k≥2} (v_k[i]−v_k[j])²/λ_k`.

### 1.3 Axis 2 — Buffer (DECLARED PRIOR [S])

The transient storage the resistive axis omits. From the swing equation
`RoCoF = f₀·Δp/(2H)`, the largest sudden imbalance a unit with effective inertia
`H` (seconds) absorbs before its frequency crosses a protection band `Δf` is

```
  B(H) = 2 · H · Δf / f₀          (here Δf = 0.2 Hz, f₀ = 50 Hz ⇒ B = 0.008 · H)
```

`B` is set by **inertia, not topology** — orthogonal to `λ₂`/`Kf` by construction.
`H_eff` is assigned per country from its generation mix (nuclear/hydro high; wind/
solar inverter-based, low). The **Ketchup yield** is the sharp threshold: an impulse
below `B` is absorbed elastically; at or above it the cell yields and seeds the
cascade.

### 1.4 Axis 3 — Policy (DECLARED PRIOR [S])

A dimensionless operational multiplier `π` on the impulse: `π < 1` for conservative
regimes (curtailed feed-in, fast imports, pumped storage, good forecasting), `π > 1`
for permissive feed-in.

### 1.5 The exposure index and the fail-first rule

```
  Exposure   E_c = R̄_c · π_c / B(H_c)              (↑ weak topology, ↑ permissive, ↓ buffer)

  Binding constraint = argmax over the three median-normalized factors:
       topology  R̄_c / median(R̄)
       buffer    (1/B_c) / median(1/B)
       policy    π_c / median(π)

  Marginal intervention (one step on the binding axis):
       buffer-bound   → +2 s of H_eff   (a synchronous-inertia asset)
       topology-bound → −20 % R̄         (an inter-basin corridor)
       policy-bound   → −0.2 π           (curtailment + forecast + import)
```

`E` is dimensionless and illustrative; only `R̄` is measured. The binding constraint
names the fail-first investment **type**; the marginal cut is the modelled exposure
reduction from one step on that axis.

---

## 2. Data

Topology: the **PyPSA-Eur / OSM** prebuilt network (Zenodo 13358976, ODbL, © OSM
contributors), per-country largest AC-connected component. The base CSV carries
voltage/length/circuits only; reactance (`x ≈ 0.33 Ω/km · length`) and limits are
estimated (disclosed via `n_estimated_*`). Inertia `H_eff` and policy `π` are
declared priors (operator domain knowledge + generation-mix literature), **not
measured here**. Ground truth for orientation: the ENTSO-E expert-panel final report
on the 28 Apr 2025 Iberian blackout (a *voltage* collapse).

---

## 3. Results — per-country scorecard

Eight countries, largest AC component each. **Axis 1 (λ₂, R̄, stability) MEASURED**;
`H_eff` and `π` are priors [S]. `B = 0.008·H_eff`; `E = R̄·π/B`.

| Country | n (buses) | λ₂ (measured) | R̄ mean-resistance | stability s | H_eff [S] | π [S] | Exposure E | gen mix |
|---|---|---|---|---|---|---|---|---|
| **Spain** | 261 | 3.152e-7 | 5.836e4 | 3.23 | 2.0 | 1.30 | **4.742e6** | wind/solar + old infra |
| Italy | 192 | 4.288e-7 | 6.446e4 | 2.38 | 3.5 | 1.00 | 2.302e6 | gas + solar |
| Norway | 126 | 3.616e-7 | 1.057e5 | 1.06 | 5.0 | 0.80 | 2.115e6 | hydro |
| Portugal | 52 | 1.928e-6 | 5.362e4 | 1.43 | 4.0 | 1.00 | 1.676e6 | hydro + wind |
| Poland | 122 | 8.974e-7 | 5.834e4 | 1.54 | 4.5 | 1.00 | 1.621e6 | coal |
| Britain | 196 | 1.470e-6 | 2.845e4 | 1.27 | 3.0 | 0.90 | 1.067e6 | gas + wind (HVDC island) |
| **France** | 656 | **3.061e-7** | 6.148e4 | 0.37 | 6.0 | 0.80 | **1.025e6** | nuclear |
| Germany | 441 | 1.006e-6 | 2.775e4 | 0.44 | 4.5 | 0.60 | **4.625e5** | mixed + pumped storage |

**Exposure ranking (most exposed first):** Spain (4.74e6) ≫ Italy (2.30e6) >
Norway (2.12e6) > Portugal (1.68e6) > Poland (1.62e6) > Britain (1.07e6) >
France (1.02e6) > Germany (0.46e6).

### 3.1 The France paradox (the headline construct-validity result)

France and Spain have **nearly identical measured topology** — `λ₂(FR)=3.06e-7` vs
`λ₂(ES)=3.15e-7`, `R̄` within 5 %. A topology-only screen ranks them together as the
two most fragile. Yet their exposure differs **4.6×** (FR 1.02e6 vs ES 4.74e6),
entirely because of the buffer axis: `H_eff(FR)=6` (nuclear) vs `H_eff(ES)=2`
(wind/solar) plus permissive Spanish feed-in (`π=1.3`). **The two grids that look
identical to spectral topology are at opposite ends of real-world stability — and
only the separated buffer/policy axes recover that.** This is the model's core
validity claim.

---

## 4. Fail-first investment locator

The binding constraint names the fail-first priority and the product; the marginal
cut is the modelled exposure reduction from one step on that axis.

| Country | binding axis | intervention (product) | E before → after | cut |
|---|---|---|---|---|
| **Spain** | buffer | synchronous inertia — gas turbine / sync-condenser / pumped storage | 4.74e6 → 2.37e6 | **−50 %** |
| Britain | buffer | synchronous inertia | 1.07e6 → 6.40e5 | −40 % |
| Italy | buffer | synchronous inertia | 2.30e6 → 1.46e6 | −36 % |
| Portugal | buffer | synchronous inertia | 1.68e6 → 1.12e6 | −33 % |
| Germany | buffer | synchronous inertia | 4.62e5 → 3.20e5 | −31 % |
| France | topology | transmission corridor (inter-basin) | 1.02e6 → 8.20e5 | −20 % |
| Norway | topology | transmission corridor | 2.11e6 → 1.69e6 | −20 % |
| Poland | policy | feed-in curtailment + forecast + fast-import | 1.62e6 → 1.30e6 | −20 % |

**Reading:** five of eight countries are **buffer-bound** — their fail-first lever is
a synchronous-inertia asset (the gas-turbine / synchronous-condenser case), and
**Spain's −50 % is the largest single resilience lever in the panel**. The structure
(which lever, ranked) is model-determined; the magnitude depends on the `H_eff`/`π`
priors and the `+2 s` step size.

---

## 5. Validity & reliability (on the measured ES core)

Battery over **30 stride-sampled N-1 contingencies × 4 injection raters**, Spearman
ρ with the Jirak weak-dependence significance `|ρ|√n` (a value ≳ 2 clears the noise
floor; classical IID Berry-Esseen does **not** apply — bits are weakly dependent).

| Block | Test | Result | Read |
|---|---|---|---|
| **A. Criterion validity** | structural mediator → cascade size | Fiedler edge-sensitivity ρ=**+0.77** (\|ρ\|√n=4.2); Weyl Δλ₂ ρ=−0.27; eff-resistance ρ=+0.02 | Fiedler sensitivity is the valid predictor; raw Weyl Δλ₂ is **not** (bridges fragment in one step) |
| **B. Reliability** | ranking across raters | ICC(2,1)=**0.71**, Cronbach α=**0.91**, test-retest ρ=**0.86** | vulnerability ranking is injection-independent — a reliable instrument |
| **C. Discriminant** | global vs local collapse | Raumgewinn vs infight ρ=**−0.31** (under stress; ≈0 unstressed) | distinct constructs that trade off (bridge vs loaded-line) |
| **D. Buffer independence** | buffer/node vs λ₂ | Spearman = **−0.45**, \|ρ\|√n=1.3 (**below floor**) | the buffer is a genuinely separate axis from connectivity |
| **E. Modifier** | `m=Δλ₂/λ₂` vs outcomes | vs connectivity-loss ρ=**+0.99** (√n 5.4); vs cascade ρ=−0.29 | the modifier tracks *fragmentation*, not line-count — and is confounded with Kirchhoff (caveat) |

---

## 6. Mathematical verification [G]

The spectral engine and the self-inverse reference are verified, not asserted:

| Check | Result | Meaning |
|---|---|---|
| Moore-Penrose `‖L·L⁺·L − L‖/‖L‖` | **1.3e-13** | `L⁺` is exact |
| `‖L⁺·L·L⁺ − L⁺‖/‖L⁺‖` | 2.4e-13 | (involution) |
| reciprocal spectrum `λ_k(L) ↔ 1/λ_k(L⁺)` | 3.3e-13 | self-inverse reference |
| effective resistance, helper vs eigen-sum | 1.6e-16 | two independent computations agree |
| `λ₂`, `Kf` vs closed forms `K_n`/`C_n`/`P_n` | ~1e-11 | `Kf(K_n)=n−1`, `Kf(C_n)=n(n²−1)/12`, `Kf(P_n)=(n³−n)/6` |
| Cauchy interlacing on every Cheeger split | holds | compartment spectra bound by the global |

Two honest negatives, also measured:
- **Equitability never holds on real grids** — per-node cross-degree coefficient of
  variation is `1.4–9.3` across all eight countries (0 = perfectly equitable). So the
  quotient theorem gives only the interlacing *bound*; compartment certificates are
  valid per-basin but do **not** reproduce the global spectrum exactly. **[G]**
- **Bisection stability is grid-specific and degrades on large/low-`λ₂` grids** —
  France's top cut is ambiguous (`s = 0.37 ≪ 1`), so a single Fiedler partition there
  is unreliable; Spain (`3.23`) and Italy (`2.38`) are well-separated. **[H]**

And the multi-element mode behind the real event:
- **N-2 super-additivity** — on the ES core, 13 of 66 top-line pairs have a *joint*
  algebraic-connectivity loss exceeding the sum of their singles, worst at **3.55×**.
  These correlated pairs are invisible to an N-1 screen. **[H]**

---

## 7. Honest scope & limitations

- **Only Axis 1 (topology) is measured.** `H_eff` (buffer) and `π` (policy) are
  transparent priors [S]; the exposure ranking and the fail-first percentages are
  therefore **structural**, not costed. Feeding real per-bus inertia (ENTSO-E
  publishes system inertia; TSOs hold per-bus generation) and curtailment data turns
  the same machine into a costed ROI per candidate site.
- **This is a structural-vulnerability screen, not a causal blackout forecast.** The
  28 Apr 2025 Iberian event was a *voltage* collapse; the DC/spectral layer screens
  *where* a grid is fragile and *what* lever helps — the voltage trigger is the AC
  fork's domain (`acflow`).
- **Small samples → Jirak rate.** All significance is read at `n^(p/2−1)` for weakly
  dependent contingencies, not classical IID.
- **The reactances and limits are estimated** from line length, not measured.

---

## Appendix A — Reproduction (every number above)

Topology is a Release asset (`perturbation-sim-data-v0.1`, ODbL), not committed.
With `buses.csv`/`lines.csv` in `/tmp/pypsa/`:

```bash
M=crates/perturbation-sim/Cargo.toml
# §3 scorecard + §4 fail-first locator (all 8 countries):
cargo run --release --manifest-path $M --example scorecard -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv
# §5 validity & reliability battery (ES core):
cargo run --release --manifest-path $M --example validate_mediators -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES
# §5E + §6 modifier + self-inverse + interlacing + analytic + N-2:
cargo run --release --manifest-path $M --example explore -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES
# the buffer-axis deconfound (Block D) + Ketchup yield:
cargo run --release --manifest-path $M --example buffer -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES
# the perturbation-agnostic resilience certificate (λ₂, Kf, reinforcement):
cargo run --release --manifest-path $M --example resilience -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES
# unsupervised Iberian localization (the 100-bus all-Spanish cut):
awk -F, 'NR==1||$8=="ES"||$8=="PT"' /tmp/pypsa/buses.csv > /tmp/pypsa/buses_iberia.csv
cargo run --release --manifest-path $M --example explore -- /tmp/pypsa/buses_iberia.csv /tmp/pypsa/lines.csv ALL
# cross-country equitability + stability (replace PT with IT/GB/FR/...):
cargo run --release --manifest-path $M --example explore -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv PT
```

All examples are deterministic (SplitMix64 seeds, cyclic-Jacobi eigensolver), pure
`std` by default, `clippy -D warnings` clean. The exact formulas are in
`src/{resilience,buffer,rolling_floor,timing}.rs`; the closed-form checks and the
self-inverse verification are `examples/explore.rs`.

---

## Appendix B — Grade ledger

| Claim | Grade | Basis |
|---|---|---|
| `λ₂`, `Kf`, `R̄`, stability per country | [G] | measured from PyPSA, engine verified vs closed forms |
| self-inverse `L⁺`, interlacing, analytic | [G] | verified in `explore` to ~1e-13 |
| Fiedler-sensitivity criterion validity, reliability | [H] | measured, N=30, model-dependent sampling |
| buffer ⟂ topology | [G] (structure) | orthogonal by construction; `Spearman≈0` measured |
| equitability never holds; FR cut ambiguous; N-2 3.55× | [G]/[H] | measured across the panel |
| France paradox (topology≈, exposure 4.6× apart) | [G] λ₂ + [S] buffer | measured λ₂; buffer is a prior |
| exposure ranking, fail-first %, Spain −50 % | [H] structure / [S] magnitude | depends on `H_eff`/`π` priors |
| "predicts the blackout" | **not claimed** | structural screen only; voltage trigger = AC fork |
