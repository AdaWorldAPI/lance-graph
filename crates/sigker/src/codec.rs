//! Codec route registration for sigker.
//!
//! This module declares sigker's classification under the `CodecRoute`
//! taxonomy from `lance-graph-contract::cam`:
//!
//! ```text
//!   Index    fields  ⇒  Passthrough  (lossless on the carrier)
//!   Index    paths   ⇒  Sigker       (this crate — lossless on tree-quotient)
//!   Argmax   fields  ⇒  CamPq        (codebook quantization, lossy)
//!   Skip     fields  ⇒  Skip         (not stored)
//! ```
//!
//! # Why Index regime
//!
//! Hambly-Lyons 2010 (Annals of Mathematics 171) proves that two paths of
//! bounded variation produce the same signature *iff* they are equal modulo
//! reparametrization and tree-like cancellation. The latter equivalence is
//! the *intended* identification for any path-as-information consumer — a
//! detour-and-return on a graph traversal should not change the encoded
//! information about that traversal.
//!
//! Therefore sigker is **lossless on the natural quotient**, which is the
//! definition of Index regime in the contract taxonomy.
//!
//! # Why this isn't in `lance-graph-contract` directly
//!
//! The contract crate intentionally has a small surface and adding sigker
//! there would force a heavy dep into the contract. Instead this module
//! provides an enum that the contract can adopt (or wrap with `From` impls)
//! when sigker is wired in. The contract change is a one-line addition to
//! `CodecRoute` plus the wiring rule:
//!
//! ```text
//!   match (regime, carrier_kind) {
//!       (Index, Field) => CodecRoute::Passthrough,
//!       (Index, Path)  => CodecRoute::Sigker,    // ← new
//!       (Argmax, _)    => CodecRoute::CamPq,
//!       (Skip, _)      => CodecRoute::Skip,
//!   }
//! ```
//!
//! # Certification dependency
//!
//! This classification is *asserted* until jc Pillar 11 (Hambly-Lyons
//! signature uniqueness on lance-graph paths) lands and verifies it
//! empirically on the actual SPO traversal lengths and carrier widths
//! used in production. See `crates/jc/src/hambly_lyons.rs` (stub).

/// Sigker's view of the `CodecRoute` enum from lance-graph-contract.
///
/// This mirrors the contract enum at the variant level so consumers can
/// pattern-match without taking the contract dep. The `From`/`Into` bridge
/// to the canonical contract enum is added when wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecRouteSigker {
    /// Field is a single value, lossless under direct storage.
    Passthrough,
    /// Field is a *path* (sequence over time/causal ordering); use sigker.
    Sigker,
    /// Field is a high-dim vector with codebook quantization.
    CamPq,
    /// Field is not stored.
    Skip,
}

impl CodecRouteSigker {
    /// Decide the codec route from regime classification + carrier kind.
    pub fn route(regime: Regime, kind: CarrierKind) -> Self {
        match (regime, kind) {
            (Regime::Index, CarrierKind::Field) => CodecRouteSigker::Passthrough,
            (Regime::Index, CarrierKind::Path) => CodecRouteSigker::Sigker,
            (Regime::Argmax, _) => CodecRouteSigker::CamPq,
            (Regime::Skip, _) => CodecRouteSigker::Skip,
        }
    }
}

/// I1 codec regime split (mirrors `lance-graph-contract::cam::Regime`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Identity-bearing — lossless required.
    Index,
    /// Argmax-eligible — lossy quantization OK.
    Argmax,
    /// Not stored.
    Skip,
}

/// Carrier shape — single field vs. ordered path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierKind {
    /// Atomic value (scalar, fixed vector, fingerprint).
    Field,
    /// Ordered sequence (path, traversal, causal trace).
    Path,
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_field_is_passthrough() {
        assert_eq!(
            CodecRouteSigker::route(Regime::Index, CarrierKind::Field),
            CodecRouteSigker::Passthrough
        );
    }

    #[test]
    fn index_path_is_sigker() {
        assert_eq!(
            CodecRouteSigker::route(Regime::Index, CarrierKind::Path),
            CodecRouteSigker::Sigker
        );
    }

    #[test]
    fn argmax_is_campq_regardless_of_kind() {
        assert_eq!(
            CodecRouteSigker::route(Regime::Argmax, CarrierKind::Field),
            CodecRouteSigker::CamPq
        );
        assert_eq!(
            CodecRouteSigker::route(Regime::Argmax, CarrierKind::Path),
            CodecRouteSigker::CamPq
        );
    }

    #[test]
    fn skip_is_skip() {
        assert_eq!(
            CodecRouteSigker::route(Regime::Skip, CarrierKind::Field),
            CodecRouteSigker::Skip
        );
    }
}
