//! # perturbation-sim — outage perturbation-shape simulator
//!
//! Models the **shape** of a cascading power-grid failure (the red-edge field a
//! Data-Explorer graph lights up during an outage) by composing the two halves
//! of the eigenvalue-perturbation + edge-propagation method:
//!
//! 1. **Spectral perturbation** ([`perturbation`]). A line trip is a *low-rank
//!    perturbation* `E` of the weighted graph Laplacian `L`:
//!    `L' = L − b_k (e_a − e_b)(e_a − e_b)ᵀ`, a rank-1 update with
//!    `‖E‖₂ = 2·b_k`. We recompute the spectrum and certify **Weyl's
//!    inequality** `|λᵢ(L') − λᵢ(L)| ≤ ‖E‖₂` for every eigenvalue, report the
//!    **Davis–Kahan** Fiedler-vector rotation bound `sinθ ≤ ‖E‖₂ / gap`, and
//!    track the **algebraic connectivity** (Fiedler value `λ₂`) — the drop in
//!    `λ₂` toward 0 is the fragmentation/blackout precursor.
//!
//! 2. **Edge propagation** ([`flow`], [`cascade`]). A DC power-flow model
//!    (`θ = L⁺ p`, line flow `f_e = b_e (θ_a − θ_b)`) redistributes flow when a
//!    line trips. Lines that exceed their limit trip in turn; the recursion is
//!    the cascade. The resulting per-node angle-deviation field and per-edge
//!    flow-shift field are the **perturbation shape**.
//!
//! ## Where this sits in the workspace
//!
//! This is the applied companion to `lance-graph/crates/jc` (the Jirak–Cartan
//! proof-in-code layer). Two honest notes carried over from the math-theorem
//! harvest (`ada-docs/research/JIRAK_MATH_THEOREMS_HARVEST.md`):
//!
//! - `jc::weyl` is Hermann Weyl's **equidistribution** theorem (golden-ratio
//!   low-discrepancy sampling), **not** the eigenvalue-perturbation inequality.
//!   This crate supplies the genuine spectral-perturbation result the harvest
//!   recommended building.
//! - `jc::ewa_sandwich` is a genuine covariance Σ-push-forward along multi-hop
//!   edge paths — the *uncertainty-propagation* sibling of the deterministic
//!   flow cascade here.
//!
//! ## Statistical hand-off
//!
//! [`cascade::PerturbationShape::node_field`] is a per-node magnitude vector
//! ready to be correlated (Pearson / Spearman / ICC via
//! `ndarray::hpc::reliability`) against an *observed* outage footprint —
//! predicted-shape-vs-observed-shape validity. Significance of any such
//! correlation must use the **Jirak 2016** weak-dependence rate `n^(p/2−1)`,
//! not classical IID Berry–Esseen (see `I-NOISE-FLOOR-JIRAK`).
//!
//! ## Zero-dep / determinism
//!
//! Pure `std`. The only numerical engine is a cyclic-Jacobi symmetric
//! eigensolver ([`eigen`]); every result is deterministic and cross-checkable
//! against numpy/scipy/R. Targets modest networks (`n` up to a few hundred
//! buses) — exactly the regime of a regional transmission graph.

pub mod acflow;
pub mod basin;
pub mod buffer;
pub mod cascade;
pub mod chaoda;
pub mod columns;
pub mod eigen;
pub mod flow;
pub mod graph;
pub mod hhtl;
pub mod inertia_data;
pub mod ingest;
pub mod model;
pub mod perturbation;
pub mod resilience;
pub mod rolling_floor;
pub mod sketch;
pub mod splat;
pub mod stats;
pub mod timing;
pub mod witness;

pub use acflow::{AcBus, AcLine, AcSystem, BusKind, PowerFlowResult};
pub use basin::{
    cheeger_sweep, contingency_features, effective_resistance, infight_vs_raumgewinn, kron_reduce,
    laplacian_pinv, spectral_embedding, Cheeger, ContingencyFeatures, GoScore, KronReduced, Regime,
};
pub use buffer::{compartment_buffer, impulse_buffer, inertia_buffer_column, ketchup_yield, Yield};
pub use cascade::{simulate_outage, CascadeConfig, CascadeResult, PerturbationShape};
pub use chaoda::{
    anomaly_ranking, cakes_neighbors, chaoda_scores, resilience_basin_features, CHAODA_FLAG,
};
pub use columns::{
    study_member_specs, study_slot_assignments, Encoding, GuardrailVerdict, InertiaPromotion,
    SoaMemberSpec, INERTIA, INERTIA_PROMOTION, INERTIA_SLOT,
};
pub use eigen::{symmetric_eigen, Eigen};
pub use flow::{dc_flows, lodf};
pub use graph::{Edge, Grid};
pub use hhtl::{basin_lambda2, hhtl_keys, HhtlKey};
pub use inertia_data::{inertia_for_buses, parse_bus_inertia, proxy_inertia, InertiaProvenance};
pub use ingest::{estimate_snom_mva, from_pypsa_csv, PypsaImport};
pub use model::{
    apply_aging, assess_capability, edge_age_factors, scale_susceptance, with_uniform_derate,
    AgeModel, Capability, DataLevel,
};
pub use perturbation::{spectral_perturbation, SpectralPerturbation};
pub use resilience::{algebraic_connectivity, kirchhoff_index, Resilience};
pub use rolling_floor::{weyl_over_fiedler, FloorBand, RollingFloor, StackResult, TierFloors};
pub use sketch::{fwht, resistance_sketch, walsh_pyramid_energy, ResistanceSketch, WalshEnergy};
pub use splat::{box_coarsen, ewa_coarsen, morton2, splat_neighborhood, Splat};
pub use stats::{cronbach_alpha, icc_a1, pearson, spearman, zscore};
pub use timing::{
    cascade_wall_time, collapse_number, implied_dt_per_hop, mechanism_from_timescale, meta_cascade,
    meta_cascade_phase, per_hop_time, rocof_hz_per_s, tier_composite, MetaHop, HHTL_WEIGHTS,
};
pub use witness::{
    field_spectrum, particle_equals_wave, witness_from_spectrum, witness_particle, witness_wave,
};
