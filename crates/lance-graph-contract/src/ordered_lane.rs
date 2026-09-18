//! D-DIAMOND-1 R2 — the ordered-lane witness: **storage-attested,
//! planner-consumed.**
//!
//! A [`SemanticPrefix`] over a lane of [`FacetCascade`] keys names a
//! CONTIGUOUS row range only when the lane is in **numeric projection order
//! over the canonical LE image** — lexicographic unsigned order over
//! `(facet_classid, tiers[0].as_u16(), …, tiers[5].as_u16())`
//! ([`FacetCascade::cmp_numeric_projection`]). That is not a property of the
//! carrier (every facet lane has the same bytes), nor of column placement, nor
//! of schema shape: it is a property of the WRITE PATH, and only the storage
//! side that sealed the lane can attest it. So:
//!
//! - the sealer produces an [`OrderedLaneWitness`] when it seals a lane
//!   ([`SealedFacetLane::seal`] sorts then attests; [`attest_sorted`]
//!   attests a lane that is already ordered and REFUSES one that is not);
//! - the planner CONSUMES the witness and may lower a prefix predicate to a
//!   bound (`lower_bound + upper_bound + mask_set_range`) only when the witness
//!   validates against the sealed lane it is about to bound
//!   ([`SealedFacetLane::bound`]);
//! - **no witness → sweep**; **stale or false witness → the bound path is
//!   unavailable** (a [`WitnessError`]), never a plausible wrong mask.
//!
//! The planner must not invent or infer this ordering. Nothing here reads
//! column placement or schema; the witness is minted from the keys themselves
//! and carries the `LanceVersion` it was sealed at, its row count, and an
//! order-sensitive digest of the key sequence — the three things a stale
//! witness gets wrong.
//!
//! This is the REFERENCE implementation for the D-DIAMOND-1 probe: it owns
//! its keys and sorts at seal. A production lane attests in place; the
//! contract (witness fields, validation, the bound) is what is fixed here.
//!
//! [`attest_sorted`]: SealedFacetLane::attest_sorted

use crate::facet::{FacetCascade, SemanticPrefix};
use crate::temporal_pov::LanceVersion;
use core::cmp::Ordering;
use std::vec::Vec;

/// What a sealer attests about one sealed lane: that at `version` the lane
/// holds `n_rows` keys in numeric projection order, whose sequence digests to
/// `digest`. Opaque to the planner — it can only hand it back to
/// [`SealedFacetLane::bound`], which validates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrderedLaneWitness {
    version: LanceVersion,
    n_rows: u32,
    digest: u64,
}

impl OrderedLaneWitness {
    /// The Lance version the lane was sealed at (the reader's `at(version)`).
    #[must_use]
    pub const fn version(&self) -> LanceVersion {
        self.version
    }

    /// Row count attested.
    #[must_use]
    pub const fn n_rows(&self) -> u32 {
        self.n_rows
    }

    /// Order-sensitive FNV-1a digest of the key sequence.
    #[must_use]
    pub const fn digest(&self) -> u64 {
        self.digest
    }

    /// Build a witness from raw fields — **falsifier-only** (F3: a forged or
    /// stale witness must be rejected before any bound runs). Never a
    /// production path: real witnesses come only from sealing.
    #[doc(hidden)]
    #[must_use]
    pub const fn forged(version: LanceVersion, n_rows: u32, digest: u64) -> Self {
        OrderedLaneWitness {
            version,
            n_rows,
            digest,
        }
    }
}

/// Why a lane could not be attested, or why a witness does not validate
/// against the lane it was presented to. Every variant makes the bound path
/// UNAVAILABLE; none lets it run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WitnessError {
    /// [`SealedFacetLane::attest_sorted`] found `keys[i-1] > keys[i]`.
    NotSorted {
        /// The first index `i` whose predecessor is greater.
        first_inversion_at: u32,
    },
    /// More rows than a `u32` ordinal can address.
    TooManyRows,
    /// The witness was minted at a different version than this lane.
    VersionMismatch {
        /// The version the witness claims.
        witnessed: LanceVersion,
        /// The version the lane was sealed at.
        lane: LanceVersion,
    },
    /// The witness attests a different row count.
    RowCountMismatch {
        /// The count the witness claims.
        witnessed: u32,
        /// The lane's actual count.
        lane: u32,
    },
    /// The witness attests a different key sequence.
    DigestMismatch {
        /// The digest the witness claims.
        witnessed: u64,
        /// The lane's sealed digest.
        lane: u64,
    },
}

impl core::fmt::Display for WitnessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WitnessError::NotSorted { first_inversion_at } => {
                write!(f, "lane is not in numeric projection order (first inversion at row {first_inversion_at})")
            }
            WitnessError::TooManyRows => write!(f, "more rows than a u32 ordinal can address"),
            WitnessError::VersionMismatch { witnessed, lane } => {
                write!(
                    f,
                    "witness is for version {witnessed}, lane is sealed at {lane}"
                )
            }
            WitnessError::RowCountMismatch { witnessed, lane } => {
                write!(f, "witness attests {witnessed} rows, lane has {lane}")
            }
            WitnessError::DigestMismatch { witnessed, lane } => {
                write!(f, "witness digest {witnessed:#x} != lane digest {lane:#x}")
            }
        }
    }
}

impl std::error::Error for WitnessError {}

/// A lane of facet keys in numeric projection order, with the witness that
/// says so. Immutable once sealed: readers pin it (by `Arc`, by version) and
/// the open image a writer appends to is a different object entirely.
#[derive(Debug, Clone)]
pub struct SealedFacetLane {
    keys: Vec<FacetCascade>,
    witness: OrderedLaneWitness,
}

impl SealedFacetLane {
    /// Sort `keys` into numeric projection order and attest them at `version`.
    /// The reference seal: **sorting at seal is the one write-side transform**
    /// the fold-first law allows; every read after it is a pure fold.
    ///
    /// # Errors
    ///
    /// [`WitnessError::TooManyRows`] if `keys.len() > u32::MAX`.
    pub fn seal(mut keys: Vec<FacetCascade>, version: LanceVersion) -> Result<Self, WitnessError> {
        // Equal keys are indistinguishable, so an unstable sort is exact.
        keys.sort_unstable_by(FacetCascade::cmp_numeric_projection);
        Self::attest_sorted(keys, version)
    }

    /// Attest a lane that is ALREADY in numeric projection order, refusing one
    /// that is not. This is the gate a writer that lands rows out of order
    /// cannot pass: a shuffled lane gets no witness, so the planner can only
    /// sweep it.
    ///
    /// # Errors
    ///
    /// [`WitnessError::NotSorted`] naming the first inversion;
    /// [`WitnessError::TooManyRows`].
    pub fn attest_sorted(
        keys: Vec<FacetCascade>,
        version: LanceVersion,
    ) -> Result<Self, WitnessError> {
        let n_rows = u32::try_from(keys.len()).map_err(|_| WitnessError::TooManyRows)?;
        if let Some(i) = first_inversion(&keys) {
            return Err(WitnessError::NotSorted {
                first_inversion_at: i as u32,
            });
        }
        let digest = digest_of(&keys);
        Ok(SealedFacetLane {
            keys,
            witness: OrderedLaneWitness {
                version,
                n_rows,
                digest,
            },
        })
    }

    /// The ordered keys — the sealed image, read in place.
    #[must_use]
    pub fn keys(&self) -> &[FacetCascade] {
        &self.keys
    }

    /// The witness this lane was sealed with. Hand it to the planner; the
    /// planner hands it back to [`bound`](Self::bound).
    #[must_use]
    pub const fn witness(&self) -> OrderedLaneWitness {
        self.witness
    }

    /// The version this lane was sealed at.
    #[must_use]
    pub const fn version(&self) -> LanceVersion {
        self.witness.version
    }

    /// Row count.
    #[must_use]
    pub fn n_rows(&self) -> u32 {
        self.witness.n_rows
    }

    /// O(1) validation of a witness the planner holds against THIS lane: the
    /// version, the row count and the digest must all be the ones this lane
    /// was sealed with. A witness from an earlier seal of "the same" lane
    /// (re-sealed after appends), from a different lane, or forged, fails
    /// here — before any bound runs.
    ///
    /// # Errors
    ///
    /// The first of `VersionMismatch` / `RowCountMismatch` / `DigestMismatch`.
    pub fn validate(&self, w: &OrderedLaneWitness) -> Result<(), WitnessError> {
        if w.version != self.witness.version {
            return Err(WitnessError::VersionMismatch {
                witnessed: w.version,
                lane: self.witness.version,
            });
        }
        if w.n_rows != self.witness.n_rows {
            return Err(WitnessError::RowCountMismatch {
                witnessed: w.n_rows,
                lane: self.witness.n_rows,
            });
        }
        if w.digest != self.witness.digest {
            return Err(WitnessError::DigestMismatch {
                witnessed: w.digest,
                lane: self.witness.digest,
            });
        }
        Ok(())
    }

    /// O(n) re-attestation: recompute order and digest from the keys and
    /// compare with what was sealed. Always passes for a lane this type
    /// sealed; exists so a reader attaching to a lane it did not seal can
    /// check storage's claim once, rather than trust it.
    ///
    /// # Errors
    ///
    /// `NotSorted` / `DigestMismatch` if the keys no longer match the seal.
    pub fn verify(&self) -> Result<(), WitnessError> {
        if let Some(i) = first_inversion(&self.keys) {
            return Err(WitnessError::NotSorted {
                first_inversion_at: i as u32,
            });
        }
        let lane = digest_of(&self.keys);
        if lane != self.witness.digest {
            return Err(WitnessError::DigestMismatch {
                witnessed: self.witness.digest,
                lane,
            });
        }
        Ok(())
    }

    /// **The witnessed bound.** Validate `w` against this lane, then locate
    /// the contiguous row range `[lo, hi)` carrying `prefix` with two
    /// `partition_point`s — `lower_bound(prefix.lo_key())` and
    /// `upper_bound(prefix.hi_key())`. The caller paints `[lo, hi)` with
    /// `mask_set_range`; no row is compared after the search.
    ///
    /// # Errors
    ///
    /// Any [`validate`](Self::validate) error — the bound does not run.
    pub fn bound(
        &self,
        w: &OrderedLaneWitness,
        prefix: &SemanticPrefix,
    ) -> Result<(u32, u32), WitnessError> {
        self.validate(w)?;
        Ok(bound_unwitnessed(&self.keys, prefix))
    }
}

/// The raw bound with NO ordering gate: two `partition_point`s over `keys`.
/// Correct only on a slice in numeric projection order; on anything else it
/// returns a PLAUSIBLE WRONG RANGE — which is exactly the failure the witness
/// exists to make unreachable. Exposed for the D-DIAMOND-1 falsifier F2 (a
/// shuffled lane must be caught by the oracle) and for timing the search
/// alone. Not a production path.
#[doc(hidden)]
#[must_use]
pub fn bound_unwitnessed(keys: &[FacetCascade], prefix: &SemanticPrefix) -> (u32, u32) {
    let lo_k = prefix.lo_key();
    let hi_k = prefix.hi_key();
    let lo = keys.partition_point(|k| k.cmp_numeric_projection(&lo_k) == Ordering::Less);
    let hi = keys.partition_point(|k| k.cmp_numeric_projection(&hi_k) != Ordering::Greater);
    (lo as u32, hi as u32)
}

/// Index of the first key whose predecessor is greater, if any.
#[must_use]
pub fn first_inversion(keys: &[FacetCascade]) -> Option<usize> {
    keys.windows(2)
        .position(|w| w[0].cmp_numeric_projection(&w[1]) == Ordering::Greater)
        .map(|i| i + 1)
}

/// Order-sensitive FNV-1a over the LE byte images of `keys`, in sequence.
/// The same multiset in a different order digests differently (tested), so
/// a witness minted before a shuffle does not validate after it.
#[must_use]
pub fn digest_of(keys: &[FacetCascade]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for k in keys {
        for b in k.to_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Skewed keys: small alphabets at the coarse tiles so prefixes at every
    /// depth have non-trivial populations.
    fn skewed_keys(n: usize, seed: u64) -> Vec<FacetCascade> {
        let mut r = SplitMix64(seed);
        (0..n)
            .map(|_| {
                let t = [
                    (r.next() % 3) as u16,
                    (r.next() % 2) as u16,
                    (r.next() % 4) as u16,
                    (r.next() % 3) as u16,
                    (r.next() % 5) as u16,
                    (r.next() % 3) as u16,
                    (r.next() % 7) as u16,
                    (r.next() % 2) as u16,
                ];
                FacetCascade::from_semantic_tiles(t)
            })
            .collect()
    }

    fn shuffle(v: &mut [FacetCascade], seed: u64) {
        let mut r = SplitMix64(seed);
        for i in (1..v.len()).rev() {
            let j = (r.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }

    fn oracle_rows(keys: &[FacetCascade], p: &SemanticPrefix) -> Vec<usize> {
        (0..keys.len()).filter(|&i| p.matches(keys[i])).collect()
    }

    #[test]
    fn seal_sorts_attests_and_validates_its_own_witness() {
        let lane = SealedFacetLane::seal(skewed_keys(2000, 1), 7).unwrap();
        assert!(first_inversion(lane.keys()).is_none());
        assert_eq!(lane.n_rows(), 2000);
        assert_eq!(lane.version(), 7);
        let w = lane.witness();
        assert_eq!(lane.validate(&w), Ok(()));
        assert_eq!(lane.verify(), Ok(()));
        assert_eq!(lane.witness().digest(), digest_of(lane.keys()));
    }

    /// F2 — a shuffled lane cannot be attested, and the UNWITNESSED bound on it
    /// returns a range the oracle rejects. The witness is load-bearing.
    #[test]
    fn f2_shuffled_lane_is_unattestable_and_its_unwitnessed_bound_is_caught() {
        let lane = SealedFacetLane::seal(skewed_keys(4000, 2), 1).unwrap();
        let mut shuffled = lane.keys().to_vec();
        shuffle(&mut shuffled, 99);
        assert!(
            first_inversion(&shuffled).is_some(),
            "shuffle must actually break order"
        );

        match SealedFacetLane::attest_sorted(shuffled.clone(), 1) {
            Err(WitnessError::NotSorted { first_inversion_at }) => assert!(first_inversion_at > 0),
            other => panic!("shuffled lane must be unattestable, got {other:?}"),
        }

        // Pick a prefix with a non-trivial population on the ORDERED lane.
        let probe = lane.keys()[lane.keys().len() / 3];
        let p = SemanticPrefix::of(probe, 3);
        let expect = oracle_rows(lane.keys(), &p);
        assert!(
            !expect.is_empty() && expect.len() * 3 < lane.keys().len(),
            "anti-vacuity"
        );

        let (lo, hi) = bound_unwitnessed(&shuffled, &p);
        let got: Vec<usize> = (lo as usize..hi as usize).collect();
        let truth = oracle_rows(&shuffled, &p);
        assert_ne!(
            got, truth,
            "the oracle must catch the unwitnessed bound on a shuffled lane"
        );
    }

    /// F3 — a forged or stale witness is rejected BEFORE any bound runs. No
    /// range is produced for any of the three kinds of mismatch, nor for a
    /// witness from an earlier seal of the same lane.
    #[test]
    fn f3_forged_or_stale_witness_cannot_execute_the_bound() {
        let keys = skewed_keys(3000, 3);
        let lane = SealedFacetLane::seal(keys.clone(), 10).unwrap();
        let w = lane.witness();
        let p = SemanticPrefix::of(lane.keys()[100], 2);
        assert!(lane.bound(&w, &p).is_ok(), "the real witness works");

        let stale_version = OrderedLaneWitness::forged(w.version() + 1, w.n_rows(), w.digest());
        assert!(matches!(
            lane.bound(&stale_version, &p),
            Err(WitnessError::VersionMismatch { .. })
        ));

        let wrong_rows = OrderedLaneWitness::forged(w.version(), w.n_rows() - 1, w.digest());
        assert!(matches!(
            lane.bound(&wrong_rows, &p),
            Err(WitnessError::RowCountMismatch { .. })
        ));

        let wrong_digest = OrderedLaneWitness::forged(w.version(), w.n_rows(), w.digest() ^ 1);
        assert!(matches!(
            lane.bound(&wrong_digest, &p),
            Err(WitnessError::DigestMismatch { .. })
        ));

        // Stale in the realistic way: the lane was re-sealed after an append.
        let mut grown = keys;
        grown.push(FacetCascade::from_semantic_tiles([9, 9, 9, 9, 9, 9, 9, 9]));
        let resealed = SealedFacetLane::seal(grown, 11).unwrap();
        assert!(
            resealed.bound(&w, &p).is_err(),
            "old witness must not validate on the new seal"
        );
        // And the new witness does not validate on the old lane either.
        assert!(lane.bound(&resealed.witness(), &p).is_err());
    }

    /// F5 — the witnessed bound equals the oracle at every depth 0..=8, on a
    /// skewed lane, for several probes; depth 0 is the whole lane, depth 8 is
    /// the run of keys equal to the probe.
    #[test]
    fn f5_witnessed_bound_equals_oracle_at_every_depth() {
        let lane = SealedFacetLane::seal(skewed_keys(5000, 4), 3).unwrap();
        let w = lane.witness();
        for &pick in &[0usize, 17, 1234, 2500, 4999] {
            let probe = lane.keys()[pick];
            for depth in 0..=8u8 {
                let p = SemanticPrefix::of(probe, depth);
                let (lo, hi) = lane.bound(&w, &p).unwrap();
                let got: Vec<usize> = (lo as usize..hi as usize).collect();
                assert_eq!(
                    got,
                    oracle_rows(lane.keys(), &p),
                    "probe {pick} depth {depth}"
                );
                if depth == 0 {
                    assert_eq!((lo, hi), (0, lane.n_rows()));
                }
                if depth == 8 {
                    assert!(hi > lo, "the probe itself is in its own full-depth range");
                    assert!(lane.keys()[lo as usize..hi as usize]
                        .iter()
                        .all(|k| *k == probe));
                }
            }
        }
    }

    #[test]
    fn digest_is_order_sensitive() {
        let a = skewed_keys(64, 5);
        let mut b = a.clone();
        shuffle(&mut b, 6);
        assert_ne!(a, b);
        assert_ne!(digest_of(&a), digest_of(&b));
        assert_eq!(digest_of(&a), digest_of(&a.clone()));
    }

    #[test]
    fn attest_sorted_accepts_an_ordered_lane_without_sorting() {
        let sealed = SealedFacetLane::seal(skewed_keys(500, 7), 2).unwrap();
        let again = SealedFacetLane::attest_sorted(sealed.keys().to_vec(), 2).unwrap();
        assert_eq!(again.witness(), sealed.witness());
    }
}
