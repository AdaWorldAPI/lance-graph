//! Candidate Pool: ranked autocomplete candidates with composition phase tracking.
//!
//! Each candidate comes from one of the 4096 attention heads.
//! The pool tracks what has been said (already_said bundle) and what remains.
//! Composition phase (Exposition→Coda) emerges from surprise/alignment dynamics.

use super::kv_bundle::{HeadPrint, bundle_into};

/// Which composition phase the conversation is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Exposition,    // theme intro (cache full, surprise high, much to say)
    Durchfuehrung, // development (cache depleting, patterns emerging)
    Contrapunkt,   // counter-thesis (contradiction detected, tension rising)
    Bridge,        // convergence (models aligning, tension resolving)
    Pointe,        // resolution (surprise → minimum, insight moment)
    Coda,          // conclusion (cache empty, nothing left)
}

/// Where a candidate came from in the 4096-head matrix.
#[derive(Clone, Copy, Debug)]
pub struct HeadAddress {
    pub row: u8, // 0-63 (or 0-255 at TWIG)
    pub col: u8, // 0-63 (or 0-255 at TWIG)
}

/// One autocomplete candidate.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub head: HeadPrint,
    pub address: HeadAddress,
    pub rank: f32,
    pub confidence: f32, // NARS confidence
    pub frequency: f32,  // NARS frequency
    pub inference: u8,   // 0=deduction, 1=induction, 2=abduction, 3=revision
}

/// The pool of candidates + conversation state.
pub struct CandidatePool {
    candidates: Vec<Candidate>,
    already_said: HeadPrint, // bundle of everything emitted
    emit_count: u32,
    phase: Phase,
    max_candidates: usize,
}

impl CandidatePool {
    pub fn new(max: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(max),
            already_said: HeadPrint::zero(),
            emit_count: 0,
            phase: Phase::Exposition,
            max_candidates: max,
        }
    }

    pub fn add(&mut self, candidate: Candidate) {
        self.candidates.push(candidate);
        self.candidates
            .sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap());
        self.candidates.truncate(self.max_candidates);
    }

    /// Best candidate (highest rank).
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates.first()
    }

    /// Emit best: unbundle from pool, bundle into already_said.
    pub fn emit(&mut self) -> Option<Candidate> {
        if self.candidates.is_empty() {
            return None;
        }
        let best = self.candidates.remove(0);
        bundle_into(&best.head, &mut self.already_said, self.emit_count as f32, 1.0);
        self.emit_count += 1;
        Some(best)
    }

    /// Update composition phase based on dynamics.
    pub fn update_phase(&mut self, surprise: f32, alignment: f32, has_contradiction: bool) {
        self.phase = if self.candidates.is_empty() {
            Phase::Coda
        } else if has_contradiction {
            Phase::Contrapunkt
        } else if surprise < 0.05 && alignment > 0.8 {
            Phase::Pointe
        } else if alignment > 0.6 {
            Phase::Bridge
        } else if self.emit_count > 3 {
            Phase::Durchfuehrung
        } else {
            Phase::Exposition
        };
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn is_done(&self) -> bool {
        self.phase == Phase::Coda
    }
    pub fn already_said(&self) -> &HeadPrint {
        &self.already_said
    }
    pub fn count(&self) -> usize {
        self.candidates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(rank: f32, dim0: i16) -> Candidate {
        let mut dims = [0i16; 17];
        dims[0] = dim0;
        Candidate {
            head: HeadPrint { dims },
            address: HeadAddress { row: 0, col: 0 },
            rank,
            confidence: 0.8,
            frequency: 0.7,
            inference: 0,
        }
    }

    #[test]
    fn test_pool_add_and_rank() {
        let mut pool = CandidatePool::new(3);

        pool.add(make_candidate(0.5, 10));
        pool.add(make_candidate(0.9, 20));
        pool.add(make_candidate(0.7, 30));
        pool.add(make_candidate(0.3, 40)); // should be truncated (4th, max=3)

        assert_eq!(pool.count(), 3);

        // Best should be rank 0.9
        let best = pool.best().unwrap();
        assert_eq!(best.rank, 0.9);
        assert_eq!(best.head.dims[0], 20);
    }

    #[test]
    fn test_emit_updates_already_said() {
        let mut pool = CandidatePool::new(10);

        pool.add(make_candidate(0.9, 100));
        pool.add(make_candidate(0.5, 50));

        // already_said starts as zero
        assert_eq!(pool.already_said().dims[0], 0);

        // Emit best (rank 0.9, dim0=100)
        let emitted = pool.emit().unwrap();
        assert_eq!(emitted.rank, 0.9);
        assert_eq!(emitted.head.dims[0], 100);

        // already_said should now reflect the emitted head
        // First emit: weight_self=0, weight_new=1 → already_said = emitted
        assert_eq!(pool.already_said().dims[0], 100);
        assert_eq!(pool.count(), 1);

        // Emit second
        let emitted2 = pool.emit().unwrap();
        assert_eq!(emitted2.head.dims[0], 50);

        // already_said is now a blend of both
        // weight_self=1, weight_new=1 → average of 100 and 50 = 75
        assert_eq!(pool.already_said().dims[0], 75);
        assert_eq!(pool.count(), 0);
    }

    #[test]
    fn test_phase_transitions() {
        let mut pool = CandidatePool::new(10);
        pool.add(make_candidate(0.5, 10));

        // Initial: Exposition (emit_count=0, no special conditions)
        pool.update_phase(0.5, 0.3, false);
        assert_eq!(pool.phase(), Phase::Exposition);

        // Contradiction → Contrapunkt
        pool.update_phase(0.5, 0.3, true);
        assert_eq!(pool.phase(), Phase::Contrapunkt);

        // High alignment → Bridge
        pool.update_phase(0.5, 0.7, false);
        assert_eq!(pool.phase(), Phase::Bridge);

        // Low surprise + high alignment → Pointe
        pool.update_phase(0.01, 0.9, false);
        assert_eq!(pool.phase(), Phase::Pointe);

        // Emit enough to trigger Durchfuehrung
        for _ in 0..4 {
            pool.add(make_candidate(0.5, 10));
            pool.emit();
        }
        pool.add(make_candidate(0.5, 10)); // need at least one candidate
        pool.update_phase(0.5, 0.3, false);
        assert_eq!(pool.phase(), Phase::Durchfuehrung);
    }

    #[test]
    fn test_coda_when_empty() {
        let mut pool = CandidatePool::new(10);

        // Empty pool → Coda regardless of other params
        pool.update_phase(0.5, 0.5, false);
        assert_eq!(pool.phase(), Phase::Coda);
        assert!(pool.is_done());

        // Add something, no longer Coda
        pool.add(make_candidate(0.5, 10));
        pool.update_phase(0.5, 0.5, false);
        assert_ne!(pool.phase(), Phase::Coda);
        assert!(!pool.is_done());
    }
}
