//! Atom layer — the **LOCKED 33-dim ThinkingStyleVector** basis.
//!
//! # Source (do NOT re-derive)
//!
//! `EPIPHANIES.md` **E-AGICHAT-DIMENSION-CONTRACT** → agichat's
//! `CANONICAL_DIMENSION_ALLOCATION.md` ("Status: **LOCKED**"). The basis is **not**
//! derived (no ICA/PCA, no "demote the 36 styles") — it is the locked 33-dim
//! allocation, restored on the i4 SoA floor (`E-I4-META-1`, the shipped i4-32 unpack).
//!
//! # Layers (smallest → largest)
//!
//! ```text
//! atom    = ONE lane of the vector (e.g. `deduce`, `R5`, `phi`) — bare-metal, not human-legible.
//! style   = ONE i4 VECTOR over all 33 atoms (a weighting) — the MOLECULE (Kant, Schopenhauer).
//! persona = a composition of styles + thresholds + purpose + β.
//! ```
//!
//! An [`Atom`] is a lane, **not** a [`crate::thinking::ThinkingStyle`]. A style is an
//! `I4x32` vector over the atoms; `ThinkingStyle` is the 6-bit identity that *resolves
//! to* such a vector. The groups below (Pearl/Rung/Σ/Ops/Presence/Meta) are **allocation
//! families**, neither atoms nor molecules.
//!
//! # Execution stack: `atoms → cognitive-shader-driver → SIMD`
//!
//! **Atoms are NOT SIMD.** This module defines the lanes (the catalogue) and the
//! bare-metal carrier (`I4x32` pack/unpack). All composition / affinity / sweep work
//! **dispatches through `cognitive-shader-driver`**, which owns the ndarray i4 SIMD
//! path. There is deliberately no dot-product / SIMD hot path in this layer.
//!
//! # Business is not here
//!
//! No business/FIBU lanes. Business is an **OGIT-inherited sidecar** (`E-OGIT-STAKES-LINCHPIN`):
//! request class → `MappingRow` → `Marking::Financial` → bookkeeping savant. Never an atom.

// ---------------------------------------------------------------------------
// Open layout questions (genuinely unresolved — do NOT guess)
// ---------------------------------------------------------------------------

// BLOCKED: 32-vs-33 reconciliation. The TSV is 33 dims (3+9+5+8+4+4); the shipped
// carrier floor is i4-32 (32 lanes). E-AGICHAT-DIMENSION-CONTRACT says "i4 × 33 (or
// 32 + 1)". Decide: does the carrier hold 32 lanes with the 33rd folded into a spare,
// ride two halves, or is one family trimmed by one? Until decided, the carrier below
// is `I4x32` (32 lanes) and the catalogue lists 33 logical atoms with `dim` 0..33.

// BLOCKED: "8 spare" vs "4 Presence + 4 Meta". STYLE_ENCODING.md describes the last 8
// dims as "8 spare"; the dimension-contract body names them 4 Presence + 4 Meta. The
// catalogue below uses Presence+Meta; confirm which is canonical before wiring.

// BLOCKED: per-group i4 sign/scale. The ordinal ladders (Pearl/Rung/Σ) want unsigned
// magnitude (level along the ladder); the few ± lanes (deduce↔induce in Ops, exploration
// in Meta, authentic↔performance in Presence) want signed. Cite FormatBestPractices.md;
// confirm the per-group convention before implementing pack/unpack scaling.

// ---------------------------------------------------------------------------
// Bare-metal carrier (no SIMD here — dispatch through cognitive-shader-driver)
// ---------------------------------------------------------------------------

/// Packed 32-lane signed-4-bit vector — the bare-metal carrier a **style** rides on.
///
/// 32 signed i4 values in 16 bytes (two nibbles per byte). This holds a thinking-style
/// vector (a weighting over the atom lanes). It is the *bytes*; the cognition is the
/// style/persona **objects** built on it (see `recipe.rs`).
///
/// Composition, affinity, and any vectorized sweep are **not** implemented here — they
/// dispatch through `cognitive-shader-driver` (which owns the ndarray i4-32 SIMD). This
/// type only packs/unpacks the lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct I4x32 {
    bytes: [u8; 16],
}

impl I4x32 {
    /// The all-zero style vector (every lane neutral).
    pub const ZERO: Self = Self { bytes: [0u8; 16] };

    /// Pack 32 signed bytes (one per lane) into the i4 carrier, saturating to [−8, 7].
    ///
    /// Pre-scaling (f32 → i8) is the caller's job per the group convention — see the
    /// `// BLOCKED: per-group i4 sign/scale` note above and `FormatBestPractices.md`.
    pub fn pack(values: &[i8; 32]) -> Self {
        let _ = values;
        todo!("I4x32::pack — bare-metal nibble pack; implement after the sign/scale convention is fixed")
    }

    /// Unpack the 32 lanes to signed bytes (sign-extended i4, range [−8, 7]).
    pub fn unpack(&self) -> [i8; 32] {
        todo!("I4x32::unpack — bare-metal nibble unpack")
    }
}

// ---------------------------------------------------------------------------
// The LOCKED atom catalogue (the 33-dim TSV allocation)
// ---------------------------------------------------------------------------

/// Allocation family of an atom lane. Families are organizational, not atoms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomGroup {
    /// Pearl causal ladder (3): association / intervention / counterfactual.
    Pearl,
    /// Meaning-depth rung ladder (9): R1–R9.
    Rung,
    /// σ-tier chain (5): Ω / Δ / Φ / Θ / Λ.
    Sigma,
    /// Cognitive operations (8): the inference modes + fanout ops.
    Operation,
    /// Presence modes (4).
    Presence,
    /// Meta knobs (4).
    Meta,
}

/// One lane of the LOCKED 33-dim TSV. `dim` is the canonical lane index (0..33).
#[derive(Debug, Clone, Copy)]
pub struct Atom {
    /// Canonical lane index, 0..33, in locked allocation order.
    pub dim: u8,
    /// Allocation family.
    pub group: AtomGroup,
    /// Locked lane name.
    pub name: &'static str,
}

impl Atom {
    const fn new(dim: u8, group: AtomGroup, name: &'static str) -> Self {
        Self { dim, group, name }
    }
}

use AtomGroup::*;

/// The LOCKED 33-dim TSV allocation (E-AGICHAT-DIMENSION-CONTRACT), in canonical order.
///
/// Order is the contract — `CANONICAL_DIMENSION_ALLOCATION.md` rejects arbitrary moves.
pub const CANONICAL_ATOMS: [Atom; 33] = [
    // 3 Pearl — causal ladder (ordinal)
    Atom::new(0, Pearl, "see_association"),
    Atom::new(1, Pearl, "do_intervention"),
    Atom::new(2, Pearl, "imagine_counterfactual"),
    // 9 Rung — meaning-depth ladder (ordinal) 🪜
    Atom::new(3, Rung, "rung_r1"),
    Atom::new(4, Rung, "rung_r2"),
    Atom::new(5, Rung, "rung_r3"),
    Atom::new(6, Rung, "rung_r4"),
    Atom::new(7, Rung, "rung_r5"),
    Atom::new(8, Rung, "rung_r6"),
    Atom::new(9, Rung, "rung_r7"),
    Atom::new(10, Rung, "rung_r8"),
    Atom::new(11, Rung, "rung_r9"),
    // 5 Sigma — σ-tier chain (ordinal)
    Atom::new(12, Sigma, "sigma_omega"),
    Atom::new(13, Sigma, "sigma_delta"),
    Atom::new(14, Sigma, "sigma_phi"),
    Atom::new(15, Sigma, "sigma_theta"),
    Atom::new(16, Sigma, "sigma_lambda"),
    // 8 Operations — inference modes + fanout ops (deduce↔induce is the one ± pair)
    Atom::new(17, Operation, "abduct"),
    Atom::new(18, Operation, "deduce"),
    Atom::new(19, Operation, "induce"),
    Atom::new(20, Operation, "synthesize"),
    Atom::new(21, Operation, "preflight"),
    Atom::new(22, Operation, "escalate"),
    Atom::new(23, Operation, "transcend"),
    Atom::new(24, Operation, "model_other"),
    // 4 Presence — modes (authentic↔performance is a ± pair)  [BLOCKED: "8 spare" alt]
    Atom::new(25, Presence, "authentic"),
    Atom::new(26, Presence, "performance"),
    Atom::new(27, Presence, "protective"),
    Atom::new(28, Presence, "absent"),
    // 4 Meta — knobs (exploration = explore↔exploit / temperature)  [BLOCKED: "8 spare" alt]
    Atom::new(29, Meta, "confidence_threshold"),
    Atom::new(30, Meta, "preflight_depth"),
    Atom::new(31, Meta, "exploration"),
    Atom::new(32, Meta, "verbosity"),
];

/// Look up a locked atom by canonical lane index (0..33).
#[inline]
pub fn atom(dim: u8) -> Option<&'static Atom> {
    CANONICAL_ATOMS.get(dim as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carrier_layout_is_16_bytes() {
        assert_eq!(core::mem::size_of::<I4x32>(), 16);
        assert_eq!(core::mem::align_of::<I4x32>(), 16);
    }

    #[test]
    fn catalogue_is_locked_33_in_order() {
        assert_eq!(CANONICAL_ATOMS.len(), 33);
        for (i, a) in CANONICAL_ATOMS.iter().enumerate() {
            assert_eq!(a.dim as usize, i, "lane dim must equal its index (locked order)");
            assert!(!a.name.is_empty());
        }
    }

    #[test]
    fn group_counts_match_the_contract() {
        let count = |g: AtomGroup| CANONICAL_ATOMS.iter().filter(|a| a.group == g).count();
        assert_eq!(count(Pearl), 3);
        assert_eq!(count(Rung), 9);
        assert_eq!(count(Sigma), 5);
        assert_eq!(count(Operation), 8);
        assert_eq!(count(Presence), 4);
        assert_eq!(count(Meta), 4);
    }
}
