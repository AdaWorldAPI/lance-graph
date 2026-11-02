# D-PDE-HOLOCUBE-0: Moore26 × Helix perturbation fields × exact discrete conservation × OxiPhysics/OxiLean

**Status:** Research proposal / design-only, 2026-11-03. **NOT an implemented fluid solver, formal proof, production ABI or validated speedup.**

**Mission:** Reuse resident DR64³/DT64³ addressing ideas, the existing `perturbation-sim` and ndarray perturbation work, and the COOLJAPAN physics/proof tools to build the *smallest independently falsifiable* spatial vortex-candidate → conserved-flux → numerical-oracle → formal-certificate path. No production engine, ClassId, CausalEdge64 bits, DR/DT contract, WORLD/THOUGHT admission, or GPU ABI changes are authorized by this document.

## 0. The recognition

There are **three related, non-identical perturbation objects**:

1. **Graph/spectral perturbation** (already implemented): a line outage alters the weighted electrical Laplacian by `E = -b(e_a-e_b)(e_a-e_b)^T`; response is signed bus electrical-angle change and edge-flow shift. The Weyl/Davis–Kahan estimates and grounded M-matrix residual certificate are specific to their mathematical assumptions.
2. **Spatial parametric orientation perturbation** (existing ndarray probe): one 48-bit frame = 24-bit helix principal direction + two 12-bit gradients for a *2D* Morton pyramid. It reconstructs a coherent field on demand. It is not a 48-bit Signed360 pair, nor is the field automatically divergence free.
3. **Physical fluid/force perturbation** (external OxiPhysics): a vortex or turbulent force field, LBM D3Q27 distribution transport, and separate numerical vorticity/divergence diagnostics. This is physical modeling rather than a spectral grid-outage solver.

Their shared shape is **addressed neighborhood + perturbation operator + observable + independently checked error**, not a claim that the three governing equations are equivalent.

### Code anchors, confirmed

| Source | Concrete capability | Boundary |
|---|---|---|
| [`perturbation-sim/src/lib.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/perturbation-sim/src/lib.rs) and [`perturbation.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/perturbation-sim/src/perturbation.rs) | rank-one weighted-Laplacian perturbation, DC-flow and spectral certificates | electrical network only |
| [`MoorePertubationsfeld.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/perturbation-sim/examples/MoorePertubationsfeld.rs), [#1427](https://github.com/AdaWorldAPI/lance-graph/pull/1427) | independent 2D Moore8 Jacobi vs spectral oracle | fixture's edges are all Moore-local; electrical phase is *not* cyclic phasor phase |
| [`moore_observable_first.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/perturbation-sim/examples/moore_observable_first.rs), [#1428](https://github.com/AdaWorldAPI/lance-graph/pull/1428) | residual-bounded early electrical observables, abstention; exact LODF baseline faster in recorded fixture | no nonlinear PDE certification |
| [ndarray `morton_perturbation_probe.rs`](https://github.com/AdaWorldAPI/ndarray/blob/master/examples/morton_perturbation_probe.rs) | on-demand 2D field from helix orientation + gradients | analytic prototype, not a conserved 3D flux |
| [ndarray `phase.rs`](https://github.com/AdaWorldAPI/ndarray/blob/master/src/hpc/phase.rs) | exact mod-2^32 phase arithmetic; error-bounded sin/cos LUT | LUT results approximate, unlike phase integer arithmetic |
| [ndarray `helix_orient.rs`](https://github.com/AdaWorldAPI/ndarray/blob/master/src/hpc/splat3d/helix_orient.rs), [helix Signed360 notes](https://github.com/AdaWorldAPI/ndarray/blob/master/.claude/knowledge/agnostic-surface-cpu-matrix.md) | compressed spherical directions; signed full-sphere variant | directions are not fluxes or proof certificates |
| [ndarray `bgz17_bridge.rs`](https://github.com/AdaWorldAPI/ndarray/blob/master/src/hpc/bgz17_bridge.rs) | 17×i16 compressed signatures and L1 kernels | motif indexing, not cyclic phase or conservation law |
| [OxiPhysics `force_field.rs`](https://github.com/cool-japan/oxiphysics/blob/master/crates/oxiphysics/src/force_field.rs) | `Vortex`, `Turbulent` and other parametric physical forces | force vector is not velocity |
| [OxiPhysics `d3q27.rs`](https://github.com/cool-japan/oxiphysics/blob/master/crates/oxiphysics-lbm/src/d3q27.rs) | 27 discrete velocities, moment transport and weights | 26 moving directions + stationary center; NOT equivalent to 26 independent fluid velocities |
| [OxiPhysics `flow_viz.rs`](https://github.com/cool-japan/oxiphysics/blob/master/crates/oxiphysics-viz/src/flow_viz.rs), [`gpu_fluid_euler.rs`](https://github.com/cool-japan/oxiphysics/blob/master/crates/oxiphysics-gpu/src/gpu_fluid_euler.rs) | 3D vorticity/divergence/Q diagnostics; staggered MAC grid | GPU-Euler source labels a *CPU mock backend*; do not advertise GPU acceleration |
| [OpenAI NavierStokesAndEuler](https://github.com/openai/NavierStokesAndEuler) | Lean 4.34.0-rc2 / Mathlib formalizations claiming forced Navier–Stokes alternatives C/D and unforced Euler blowup | not a proof of unforced Navier–Stokes A/B or independent prize recognition |
| [OxiLean](https://github.com/cool-japan/oxilean) | independent Rust Lean-export checker with `verified/unsupported/rejected` semantics | published export baseline targets older Lean; unsupported is NOT verified |

## 1. Proposed representation (probe only)

**Address:** a canonical Cartesian cell/tile coordinate in a 3D resident DR64³ view; surface cross-tile seams, boundary ghost/halo rules, Morton-index vs actual geometric-neighbor distinction. DT64³ methods are a *prospective routing vocabulary*, not a newly allocated method.

**Parametric hypothesis:** spatially coherent template + tiny residual, never a mandatory dense 3×f32 vector per cell.

- `Phase(u32)` is *cyclic* orientation; it must not be confused with a signed electrical phase angle in `perturbation-sim`.
- `ResidueEdge` (24-bit, hemisphere) and `Signed360` (48-bit, full-sphere signed direction) are alternatives. The planned `2 × Signed360` in/out experiment costs **96 bits for the direction pair alone**, without magnitude/phase/residual. A 48-bit frame of (24-bit orientation + 2×12-bit gradients) is a *different* encoding.
- `in/out` must be defined as **oriented incident flux or directional hypotheses**, *not* independent sources/sinks. Opposite faces/edges share one canonical value and opposite incidence sign.
- BGZ17 may index/retrieve recurring local 3D vortex/tilt motifs. It never supplies a physics invariant.
- Morton/helix frame may reconstruct a potential at any spatial coordinate. A decoder must have an explicit spatial error model, version/pin and escape/residual option.
- No forced on-row 96-bit/48-bit carving or changes to existing ClassView. Benchmark off-row/SoA options against materialized baseline.

### Physical observables, not synonyms

For velocity `u`, vorticity `omega = curl(u)`, strain `S = (grad(u) + grad(u)^T)/2`, helicity density `u · omega`, vortex-stretching `omega^T S omega`, Q, lambda-2, and streamline curvature/torsion are *different observables*. The intuition `spatial torque` means an angular-momentum-like cross product `delta × u`, **not** mechanical torque (`r × F`).

For the 26-offset complete isotropic grid with pitch `h`, define a **diagnostic** linear-exact estimator:

```text
M26 = { delta = h*(i,j,k) | i,j,k in {-1,0,1}, (i,j,k) != (0,0,0) }
curl26(u,x) = (1/(18*h*h)) * sum_{delta in M26} delta × u(x+delta)
div26 (u,x) = (1/(18*h*h)) * sum_{delta in M26} delta · u(x+delta)
```

These are exact for affine fields with complete uniform stencil and exact samples; they are approximate for general smooth fields. Boundaries, anisotropic grids, missing cells and orientation quantization require *separate* rules. Neither formula creates an exactly divergence-free field by itself.

## 2. Conservation: the crucial algebraic path

**Preferred candidate generator**: encode a *vector potential*, not arbitrary independent cell-center velocities. On an oriented cellular complex:

```text
A(edge) := canonical, quantized edge-potential cochain
F(face) := d1 A    (oriented boundary-edge incidence)
D(cell) := d2 F    (oriented boundary-face incidence)
d2 * d1 == 0      (boundary of boundary / exact discrete identity)
```

Interpret `F` as oriented face flux. This gives discrete divergence zero **algebraically**, provided one shared edge ID/value is used by all incident faces and identical incidence conventions apply. To advertise *bit-exact* zero, use sufficiently wide signed integer/fixed-point accumulators without overflow; ordinary floating-point cancellation is only tolerance-bound. Flux boundary conditions, mesh topology and mapping to cell-centered velocities are explicit contracts. Discrete divergence freedom alone does not ensure a Navier–Stokes solution.

Keep Moore26 for **geometric classification** and D3Q27 for an **independent LBM numerical model**; topology-preserving curl/face-divergence uses canonical edge/face incidences (typically 4 edges/face and 6 faces/cell). Do not use a generic 26-neighbor tensor stencil as a substitute for the boundary-of-boundary identity.

### Cheap on-demand evaluator

```text
FieldFrame(anchor, frame_version, orientation_code, cyclic_phase,
           gradients, amplitude, potential_basis, residual_id?)
        |
        +-- decode potential at canonical edge coordinate (or fetch shared edge)
        +-- local discrete curl => face flux F
        +-- face incidence => divergence certificate / local transport
        +-- Moore26 weighted neighborhood => swirl, tilt, Q, helicity candidates
        +-- optional materialization only for independent OxiPhysics oracle
```

The latent templates may use golden-spiral/helix phase, bounded tanh profiles, anisotropic splats/PSD covariance and smooth radial/torsional modifiers. A tanh phase/magnitude template **does not guarantee** incompressibility or blowup. A `2×helix` representation is a candidate-compression scheme until conservation and reconstruction are independently established.

## 3. Physics and mathematical boundaries

1. **Physical equation:** `partial_t u + (u·grad)u - nu*Delta(u) + grad(p) = f`, with `div(u)=0`. Nonlocal pressure, viscosity, forcing regularity, full-domain boundary conditions and continuum-in-scale bounds survive every local stencil optimization.
2. **OpenAI reference:** the Navier–Stokes repo claims forced alternatives C/D with smooth forcing; the Euler result is separately unforced. It is a reference corpus for named proof obligations, not a license to infer an unforced Navier–Stokes theorem or a Clay verdict.
3. **Geometry:** study axisymmetric principal profiles *and* angular Fourier/non-axisymmetric corrections, anisotropic radial/axial scaling, vortex stretching, helicity, curvature and residual cancellation. An axisymmetric candidate family alone may exclude essential correction mechanisms.
4. **Spectral perturbation limits:** `perturbation-sim`'s Weyl and Davis–Kahan certificates apply to symmetric matrices under specified gaps; the early M-matrix residual bounds apply to the grounded electrical Laplacian. Reusing the *workflow* (local perturbation, bound, abstain, independent solve) is legitimate; transferring the *inequality* directly to nonlinear fluid evolution is not.
5. **Statistical uncertainty:** JC/Jirak-dependent-sampling methods may judge stochastic candidate ranking only when dependence conditions are validated. They are not rigorous PDE discretization/continuum error bounds. The known dependent-generator issue must be fixed before dependent inference is promoted.
6. **Logical evidence:** separate `candidate`, `numerically matched`, `rigorously bounded`, `Lean verified`, `independently replayed`, and `unsupported`. Never store these as undocumented CausalEdge64 61..63 codes; no automatic `Causes` or production WORLD admission.
7. **TFLite/ONNX/NNUE:** only optional ranking of candidate profiles, angular modes and proof-obligation priority. Any ML-generated proposal is independent of deterministic certification. Hot path is zero-serialization native ndarray/SoA; measure the conversion and materialization costs separately.

## 4. Staged falsifiers and gates (in order; stop on failure)

| Gate | Smallest experiment | Independent oracle / decisive falsifier | Promotion rule |
|---|---|---|---|
| P0 Source pins | pin Git SHAs/versions, units, grid, boundary, sign/orientation, registered maps and original baseline results | external README/actual source and reproducible smoke build | no unknown inputs or silent version fallback |
| P1 Moore26 algebra | uniform field, rigid rotation `u=(-Ωy, Ωx,0)`, radial expansion `u=α(x,y,z)`, seeded affine fields on interior of 4³/16³ | analytic divergence 0/0/3α and rigid curl `(0,0,2Ω)` | linear-exact test, boundary explicitly excluded or separately defined |
| P2 Conserved generator | quantized canonical edge A → face F → cell D on 4³ and 16³ plus seams | independent face-incidence enumerator, shared-edge mutation, deliberate sign/integer-overflow faults | D==0 bit-exact in integer reference under supported bounds; disable guards must fail |
| P3 Compression vs baseline | 24-bit helix, Signed360 48, 2×Signed360 96, 48-bit orientation+grad prototype, dense 3×f32 or f64; BGZ17 on/off | compare decoded edge potential/flux, angular error, curl/div, measured bytes, allocations, CPU cycles, cold/warm p50/p95 | report full accuracy/latency Pareto; no performance claim before measurements |
| P4 Independent physical agreement | rigid rotation, 2D and 3D Taylor–Green, generated divergence-free potential, an explicitly compressive negative control; optionally a true time-evolution step | OxiPhysics flow_viz and MAC reference; D3Q27 LBM separately with matching model, units and lattice weights | evaluate velocity/flux/vorticity, convergence with mesh refinement; *no* blanket exactness |
| P5 Bound-before-convergence | perturbed and unperturbed vortex observables under fixed error budgets; attempt early abstaining decision | full-resolution reference and independent numerical truncation/error bound | certify only decisions whose *validated* intervals exclude threshold, otherwise abstain |
| P6 Theorem fragment | one formally specified invariant (e.g. d²=0 under shared canonical edge IDs), then algebraic vortex-profile constraints | Lean/Mathlib proof script, independent disable control | named theorem checks with pinned dependencies; numerical profile does not certify continuum PDE |
| P7 Independent kernel | export relevant Lean declarations and replay with OxiLean | three-way `verified/unsupported/rejected` report plus original Lean/Comparator checks | unsupported never passes; Lean 4.34-vs-older-export compatibility is an explicit blocker |

**Negative/ablation battery:** Moore6 vs Moore26 (diagnostic fidelity), odd sign inversions, helix hemisphere sign loss, phase wrap at 2π, flip opposite incidence signs, edge duplicated at Morton seam, non-divergence-free radial expansion, missing global-pressure constraint, stochastic correlated-neighbor sample treated incorrectly as IID, NNUE shuffle/noise permutation. The deliberate wrong version must be detected at the corresponding named gate.

### Suggested scorecard

- **Exact contracts:** algebraic discrete divergence cancellation, canonical shared edge identity, source hash, no overflow on pinned bounds.
- **Numerical error:** max/percentile velocity/flux/curl/strain errors vs independent oracle; mesh convergence and time-step sensitivity.
- **Compute:** bytes per field/tile, hot- and cold-path allocations, cycle profile, p50/p95, SIMD implementation, reconstruction overhead, cache behavior vs dense baseline on same hardware.
- **Scientific status:** model scope, boundary conditions, solver/model disagreement, proof-obligation coverage; all unsupported/unknown outcomes visible.
- **Comparison baseline:** no claims about 1M ops/s, 40M rows/s or Navier–Stokes solver replacement from unrelated graph/mask microbenchmarks.

## 5. Implementation order / ownership

1. **Design only (this PR):** source crosswalk, equations, representation boundaries, falsifiers, no runtime mutation.
2. **Separate probe PR in `lance-graph`:** new opt-in `pde_holocube_probe` or sibling research example; scalar exact-incidence oracle first; no foreign Oxi code copied into production; no forced new crate/dependency.
3. **Optional numerical adapter:** OxiPhysics as pinned independent test-only reference; compare API and units, not just identical-shaped structs.
4. **ndarray/helix optimization PR only after P1–P4:** benchmark `PhaseLut`, signed direction codec and morton-pyramid reconstruction while preserving original codec semantics.
5. **Formal experiment PR:** Lean proof-fragment dependency export with original Lean/Comparator checks and separately reported OxiLean replay; no modification of official OpenAI proof.

**Explicit non-goals:** declaring a Millennium Prize solved; proving fluid singularity from lattice samples; rebranding electrical-phase as circular fluid phase; implicit D3Q27 = Moore26 physics equivalence; using statistical confidence as PDE error bounds; modifying CausalEdge64 reserved codes; quantized in/out pair by itself asserting incompressibility; turning documentation into production authorization.

### Evidence and follow-up

- User-owned lineage: [lance-graph #1427](https://github.com/AdaWorldAPI/lance-graph/pull/1427), [#1428](https://github.com/AdaWorldAPI/lance-graph/pull/1428); [`perturbation-sim`](https://github.com/AdaWorldAPI/lance-graph/tree/main/crates/perturbation-sim); [ndarray perturbation example](https://github.com/AdaWorldAPI/ndarray/blob/master/examples/morton_perturbation_probe.rs).
- External independent targets: [OxiPhysics](https://github.com/cool-japan/oxiphysics), [OxiLean](https://github.com/cool-japan/oxilean), [OpenAI's formalizations](https://github.com/openai/NavierStokesAndEuler), [Clay problem statement](https://www.claymath.org/millennium/navier-stokes-equation/).

**Decision requested:** admit **P0/P1 research probes only**, retaining all subsequent stages as individually gated proposals. Preserve separate original mathematical models, independently check every substantive claim, and prefer the smallest falsifier before adding an abstraction.
