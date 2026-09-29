# 2026-09-29 — perturbation-sim: Σθ = L⁺ Σp L⁺ through ndarray CovHighD::sandwich

**STATUS:** MEASURED · **Scope:** `crates/perturbation-sim/src/angle_cov.rs` (feature `pillar`), ndarray `hpc::pillar::cov_high_d`

## What landed
- ndarray `4d4ee17`: `CovHighD::from_symmetric_fn` + public `get`. Before this a consumer
  could not build a CovHighD from data without re-spelling the packed index.
- lance-graph `bb6d044`: `angle_covariance::<N>(eig, Σp, rel_tol)`, the stochastic twin of
  `pseudo_apply`. It is exact for DC flow in a fixed topology. It calls the Pillar-9 sandwich;
  there is no local kernel.

## Measured (6-bus ring + chords)
- Against an f64 dense triple product: rel err 1.2e-7. This is the f32 floor.
- Rank-1 Σp = ppᵀ against `pseudo_apply` outer product: 2.3e-7. That path never forms L⁺.
- 40k-sample Monte Carlo of the deterministic solver: rel err 0.0009.
- Disable runs, each red under exactly its own test:
  - M := I;
  - symmetry guard removed;
  - N-mismatch guard removed.

## Finding in ndarray
Every existing `sandwich` test used M = I. Under a deliberately broken kernel (M·Σ·Σ),
`sandwich_identity_is_identity` stayed green. The new dense non-identity test is the first
that can see an index/transpose defect in the Pillar-9 kernel.

## OPEN
- CovHighD is const-generic N; `Grid::n` is runtime. The caller picks N and it is checked.
  A runtime-sized sandwich is an ndarray change, not taken.
- Line trips are a rank-1 update of L⁺ (LODF), not a sandwich. They are not wired.
- Line-flow covariance has a non-symmetric rectangular Jacobian, which sandwich cannot
  express. Per-line variance is a 2-sparse quadratic form over Σθ, not built.
