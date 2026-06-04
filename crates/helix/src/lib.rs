//! # helix — Place / Residue encoding (golden-spiral hemisphere, Fisher-Z aligned)
//!
//! The orthogonal-residue half of the substrate. **HHTL is the deterministic
//! PLACE** (the trie address — *where*); **helix is the RESIDUE** (the orthogonal
//! edge at that place — the hemispheric angle the place itself does not capture).
//! It is the discrete 2-D companion to `jc::weyl` (the 1-D `{k·φ⁻¹ mod 1}`
//! golden-stride proof).
//!
//! Headline: **8K resolution at Super-8 cost** — full neighbour discrimination
//! (the curve is regenerated from a φ-spiral template, not stored), at
//! 3-bytes-per-edge storage and O(1) 256×256-LUT distance. The resolution lives
//! in the deterministic template (free, regenerable); the cost is only the
//! endpoint pair.
//!
//! ## The object speaks for itself
//!
//! ```
//! use helix::{ResidueEncoder, DistanceLut};
//!
//! let enc = ResidueEncoder::new(4096); // total residue count N
//! let a = enc.encode(0x1234, 1700); // (hhtl place, raw residue n)
//! let b = enc.encode(0x1234, 1701);
//! let lut = DistanceLut::linear();
//! let near = a.distance_adaptive(&b, &lut); // metric-safe L1 on endpoints
//! let far = a.distance_adaptive(&enc.encode(0x1234, 4000), &lut);
//! assert!(near <= far); // adjacent residues are no farther than distant ones
//! ```
//!
//! ## The four-stage pipeline (see `KNOWLEDGE.md`)
//!
//! 1. **Placement** — tomato-rose equal-area hemisphere lift
//!    (`r = √u`, `Y = √(1 − r²)`, azimuth `n·φ`): [`HemispherePoint`].
//! 2. **Place coupling** — stride-4-over-17 arc from the HHTL offset:
//!    [`CurveRuler`].
//! 3. **Fisher-Z alignment** — `arctanh` (= hyperbolic depth `ρ = 2·arctanh(r)`):
//!    [`Similarity`].
//! 4. **Euler hand-off** — the `γ` shove, then quantise into the 256-palette with
//!    a rolling floor: [`RollingFloor`].
//!
//! The result is a [`ResidueEdge`] — a 3-byte endpoint pair whose
//! [`ResidueEdge::distance_adaptive`] is **metric-safe** (L1 on a linear index
//! order, triangle inequality free) and whose [`ResidueEdge::distance_heuristic`]
//! is a non-metric pre-filter (the raw azimuth — never use it for CAKES bounds).
//!
//! ## Relationship to existing workspace primitives (honest overlap)
//!
//! Per the placement check recorded in `KNOWLEDGE.md` § "Overlap & Consolidation",
//! the Fisher-Z/arctanh→int8 quantiser, the golden-spiral azimuth proof, the
//! stride-4 coupling, and the EULER_GAMMA hand-off already exist elsewhere in the
//! workspace (in places certified to ρ ≥ 0.999). helix is a deliberate
//! **clean-room re-derivation** — it re-derives the math rather than reusing those
//! primitives, keeping the cognitive substrate regenerable-from-template; the
//! genuinely new pieces are the equal-area `√u` hemisphere placement and the
//! PLACE/RESIDUE doctrine. Its one dependency is the **mandatory** `ndarray` fork
//! (the SIMD foundation — see `src/simd.rs`). See `KNOWLEDGE.md` for the
//! consolidation path back to the certified primitives.

#![forbid(unsafe_code)]

pub mod constants;
pub mod curve_ruler;
pub mod distance;
pub mod fisher_z;
pub mod placement;
pub mod prove;
pub mod quantize;
pub mod residue;
pub mod simd;

pub use constants::{
    E, EULER_GAMMA, GOLDEN_ANGLE, GOLDEN_RATIO, GOLDEN_RATIO_INV, LN_17, MODULUS, PALETTE_SIZE,
    STRIDE, TRANSIENT_SKIP,
};
pub use curve_ruler::CurveRuler;
pub use distance::DistanceLut;
pub use fisher_z::Similarity;
pub use placement::HemispherePoint;
pub use prove::{prove, ProofResult};
pub use quantize::RollingFloor;
pub use residue::{ResidueEdge, ResidueEncoder};
