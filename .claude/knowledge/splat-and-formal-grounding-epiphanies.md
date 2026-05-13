# Splat + Formal-Grounding Epiphanies (E1-E6)

> **READ BY:** future Claude sessions investigating Gaussian-splat / CAM-plane
> consolidation, MUL formal grounding, or cross-OGIT-path wisdom merge.
>
> **Status:** Inbox — verification tasks not yet executed. Each entry below
> has explicit verification steps that produce a yes/no signal before
> committing implementation work.
>
> **Authoring context:** Captured 2026-05-13 from external synthesis pass.
> Kept separate from `super-domain-rbac-tenancy-v1.md` so the existing spec
> context doesn't get diluted by speculative tracks. Promote individual
> entries to `.claude/plans/<slug>-v1.md` only after their verification
> task lands a "synergy confirmed" finding.

## Suggested Linear epic structure

- **Epic "Splat substrate consolidation"** → E1 + E2 + E3 + E4
- **Epic "Formal grounding citations"** → E5 + E6 (output: docs, not code)

E1 is cheapest first move (pure grep, no design decisions). E5 produces the
strongest commercial-pitch artifact. E2 is highest leverage if the
channels-as-separate-planes shape lands.

---

## E1 — Triangle-count proves AwarenessPlane16K is dual-reading storage

- **Code:** [`crates/jc/examples/splat_triangle_count.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/00abb94073556af871fee6f011e1ac935fc0356e/crates/jc/examples/splat_triangle_count.rs)
- **Anchor:** `AwarenessPlane16K` in `lance-graph-contract::splat`
- **Verification:** grep every consumer of `AwarenessPlane16K` — does each
  consumer read it as adjacency-bits OR as semantic-activation, or is one
  consumer already doing both? List `file:line` for each.
- **Synergy if confirmed:** same storage serves graph-traversal kernels and
  CAM-distance kernels with no representation conversion; future Cypher
  MATCH lowering can use `popcount-AND` as primitive.
- **Defer if:** only one consumer exists today.

## E2 — Multi-channel split (support / contradiction / forecast / counterfactual / style / source)

- **Doc:** [`.claude/knowledge/gaussian-splat-cam-plane-workaround.md`](https://github.com/AdaWorldAPI/lance-graph/blob/00abb94073556af871fee6f011e1ac935fc0356e/.claude/knowledge/gaussian-splat-cam-plane-workaround.md) §"What gets splatted"
- **Anchor:** `SplatPlaneSet` struct (proposed in §"Data structure sketch", not yet shipped)
- **Verification:** does `AwarenessPlane16K` today carry channel polarity, or
  is it monochannel? If monochannel, what's the minimum diff to add a
  `SplatChannel` enum tag without breaking `splat_triangle_count.rs`?
- **Synergy if confirmed:** triangle-count on `contradiction_plane` becomes
  "how many ambivalent triads exist" — same kernel, different semantic
  question; six free graph metrics from one new field.
- **Defer if:** the six channels turn out to need different bit-density per
  channel.

## E3 — PlanarSplatBundle4096 bands = HHTL on temporal axis

- **Doc:** same workaround md, §"Relationship to planar bundling"
  (local / short / medium / long = 8 / 64 / 512 / 4096 cycles)
- **Anchor:** existing HHTL cascade impl — grep workspace for `HHTL`, `Heel`,
  `Hip`, `Twig`, `Leaf`
- **Verification:** does the existing HHTL impl take the cascade-depth as a
  const generic or a runtime parameter? If const generic, can it be
  re-instantiated with temporal-band-depths without inner-loop changes?
- **Synergy if confirmed:** temporal cascade is free reuse of ontological
  cascade machinery; one code path, two axes.
- **Defer if:** HHTL impl is hardcoded to ontological semantics in its inner
  loop.

## E4 — 3D Gaussian Splat SIMD patterns port to CAM deposition

- **Reference impl:** [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- **Background paper page:** [INRIA 3D-Gaussian-Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) → arXiv [2308.04079](https://arxiv.org/abs/2308.04079)
- **Anchor:** future `cam_plane_splat::deposit` (pseudocode in workaround md §"Deposition rule")
- **Verification:** read gsplat tile-rasterization + alpha-accumulation inner
  loops. Which 3-5 SIMD patterns are structurally identical to CAM-plane
  bit-deposition? Already in `ndarray::simd` or need new ops?
- **Synergy if confirmed:** optimized kernels for billions of splats/sec
  exist as open-source CUDA + reference CPU code; semantic deposition
  inherits hardware-level optimization without re-deriving.
- **Defer if:** the actual deposition op turns out to be `popcount-OR` which
  is already trivially fast.

## E5 — Computational Irreducibility = formal grounding for MUL wisdom-driven sandbox

- **Paper:** [arXiv 2505.04646v1](https://arxiv.org/html/2505.04646v1) — *Computational Irreducibility as Foundation of Agency: Formal Model Connecting Undecidability to Autonomous Behavior*
- **Wikipedia primer:** [Computational irreducibility](https://en.wikipedia.org/wiki/Computational_irreducibility)
- **Background (Wolfram on Post's tag problem):** [arXiv 2103.06931](https://arxiv.org/pdf/2103.06931)
- **Anchor:** MUL gate in `cognitive-shader-driver`; the
  `is_unskilled_overconfident` path; any sandbox-request emission
- **Verification:** read paper §3-5. Map each formal element to one MUL
  primitive: which paper-construct corresponds to `wisdom_marker`? Which to
  `confidence`? Which to the sandbox-request? Document the mapping or note
  where it breaks.
- **Synergy if confirmed:** regulated-industry pitch ("our AI's autonomy is
  mathematically grounded, not emergent illusion") gets a citable formal
  model; differentiates from black-box ML claims of "agency".
- **Defer if:** paper formalism turns out to be too abstract to map onto
  concrete MUL fields.

## E6 — Mahler approximate ideals = algebra for cross-OGIT-path wisdom merge

- **Paper:** [ResearchGate — *The Arithmetic of Diophantine Approximation Groups II: Mahler Arithmetic*](https://www.researchgate.net/publication/301855507_The_Arithmetic_of_Diophantine_Approximation_Groups_II_Mahler_Arithmetic)
- **Anchor:** CAM-PQ in `lance-graph-contract::cam` + OGIT-path inheritance
  in `lance-graph-ontology`
- **Verification:** read §2-3 (tri-filtration structure + partial tensor
  product). Does the OGIT codebook-inheritance chain produce the same
  tri-filtration shape? Specifically: are the three Mahler filtration axes
  homologous to (codebook depth, schema version, namespace inheritance) in
  OGIT?
- **Synergy if confirmed:** cross-customer learning has a formal algebraic
  name; not implementation work, but academic-grade vocabulary for the
  partial-merge property.
- **Defer if:** the filtration axes don't line up; Mahler stays as
  background math.

---

## Deferred — math-deep reader needed

Foliation papers (relevant only if someone fluent in differential geometry
maps them onto HHTL; no actionable Claude Code task without that prior pass):

- [arXiv 2412.01915](https://arxiv.org/pdf/2412.01915) — *Linear Hyperbolic Equations in Double Null Foliation*
- [arXiv 2503.08077](https://arxiv.org/pdf/2503.08077) — *Horizontality of Partially Hyperbolic Foliations*
- [arXiv 2503.14446](https://arxiv.org/pdf/2503.14446) — *Codimension One Foliations on Adjoint Varieties*
- [arXiv 2504.01085](https://arxiv.org/pdf/2504.01085) — *Minimality of Strong Foliations of Anosov and Partially Hyperbolic Diffeomorphisms*
- [arXiv math/9808064](https://arxiv.org/pdf/math/9808064) — *R-covered foliations of hyperbolic 3-manifolds*

Speculative for now:

- [arXiv 2501.00022v1](https://arxiv.org/html/2501.00022v1) — *Algorithmic Idealism II* (too abstract for code-level verification this round)
