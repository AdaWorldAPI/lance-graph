//! Markov ±5 context chain with replay.
//!
//! The chain carries `2 × MARKOV_RADIUS + 1 = 11` fingerprints of the
//! sentences neighboring a focal point. When a Wechsel (or any
//! ambiguity) is marked, the chain is **replayed** — re-scanned with
//! each candidate branch pinned — and whichever branch preserves
//! coherent NARS confidence across the window wins.

use crate::crystal::fingerprint::CrystalFingerprint;

pub const MARKOV_RADIUS: usize = 5;
pub const CHAIN_LEN: usize = 2 * MARKOV_RADIUS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDirection {
    Forward,
    Backward,
    BothAndCompare,
}

/// A request to replay the chain with a specific branch pinned.
#[derive(Debug, Clone)]
pub struct ReplayRequest {
    /// Which token's ambiguity to resolve.
    pub token_index: u16,
    /// Candidate role id the replay should assume.
    pub candidate_id: u8,
    pub direction: ReplayDirection,
}

/// A ±5 Markov context chain of fingerprints.
#[derive(Debug, Clone)]
pub struct ContextChain {
    /// 11 fingerprints. Position `MARKOV_RADIUS` (index 5) is the focal
    /// sentence; 0..5 are preceding (oldest first); 6..11 are following.
    pub fingerprints: Vec<Option<CrystalFingerprint>>,
}

impl ContextChain {
    pub fn new() -> Self {
        Self { fingerprints: (0..CHAIN_LEN).map(|_| None).collect() }
    }

    pub fn focal_index() -> usize { MARKOV_RADIUS }

    /// Count of filled positions in the chain.
    pub fn filled(&self) -> usize {
        self.fingerprints.iter().filter(|f| f.is_some()).count()
    }

    /// Whether the chain is saturated on both sides of the focal.
    pub fn is_saturated(&self) -> bool {
        self.filled() == CHAIN_LEN
    }

    pub fn focal(&self) -> Option<&CrystalFingerprint> {
        self.fingerprints[Self::focal_index()].as_ref()
    }
}

impl Default for ContextChain {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_length_is_eleven() {
        let c = ContextChain::new();
        assert_eq!(c.fingerprints.len(), CHAIN_LEN);
        assert_eq!(CHAIN_LEN, 11);
    }

    #[test]
    fn focal_is_center() {
        assert_eq!(ContextChain::focal_index(), 5);
    }
}
