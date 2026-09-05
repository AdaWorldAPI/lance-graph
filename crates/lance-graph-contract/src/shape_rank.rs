//! Payload-law DTO for D-BLW-5 (`.claude/knowledge/observer-effect-tfpn-doctrine.md`
//! §2: "distribution SHAPE × Prozentrang, NEVER the raw statistic"; design
//! `.claude/board/exec-runs/d-blw-5-design-main-thread.md` §c: 16-bucket
//! histogram, `rank₀`, frozen at V₀, remeasure guard keyed
//! `(StatId, Arm, Cohort, Metric, DatasetVersion)`).
//!
//! [`ShapeRankPayload`] cannot carry a raw scalar: it has no `f64` field by
//! construction — that is the Goodhart guard enforced by shape, not by
//! discipline. The producer is `lance_graph_planner::nested_bands::NestedBands::shape_rank`
//! (D-NXG-4); this crate only holds the data that crosses into awareness.
//! Cite E-NXG-21 (NestedBands shipped) and E-NXG-2 (Prozentrang existed only
//! in doctrine before D-NXG-4).

use std::collections::BTreeMap;
use std::fmt;

/// Number of histogram buckets a [`ShapeRankPayload`] carries.
pub const SHAPE_BUCKETS: usize = 16;

/// The shape-of-a-distribution payload: a 16-bucket histogram and a coarse
/// rank into it, frozen at a dataset version.
///
/// There is no `f64` field here — not an oversight, the guard itself. A raw
/// statistic invites Goodhart's law the moment it crosses into awareness
/// (optimize the number, not the distribution it summarized); a shape
/// (histogram + bucket rank) cannot be collapsed back into a single number
/// to chase, so the DTO makes the misuse structurally unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeRankPayload {
    /// 16-bucket histogram counts.
    pub shape: [u64; SHAPE_BUCKETS],
    /// Coarse ladder position, `0..SHAPE_BUCKETS`.
    pub rank: u8,
    /// Dataset version this payload was frozen at (V₀).
    pub version: u64,
}

impl ShapeRankPayload {
    /// Construct a payload. Panics if `rank >= SHAPE_BUCKETS`.
    #[must_use]
    pub fn new(shape: [u64; SHAPE_BUCKETS], rank: u8, version: u64) -> Self {
        assert!((rank as usize) < SHAPE_BUCKETS, "rank out of range");
        Self {
            shape,
            rank,
            version,
        }
    }

    /// Total mass across all buckets, `Σ shape`.
    #[must_use]
    pub fn mass(&self) -> u64 {
        self.shape.iter().sum()
    }

    /// Mass strictly below `rank`, `Σ shape[..rank]`.
    #[must_use]
    pub fn mass_below(&self) -> u64 {
        self.shape[..self.rank as usize].iter().sum()
    }

    /// The percentile proper: `mass_below / mass`, `0.0` when `mass == 0`
    /// (no distribution to be a percentile of).
    #[must_use]
    pub fn prozentrang(&self) -> f32 {
        let mass = self.mass();
        if mass == 0 {
            0.0
        } else {
            self.mass_below() as f32 / mass as f32
        }
    }

    /// The coarse ladder position: `rank / SHAPE_BUCKETS`.
    #[must_use]
    pub fn rank_fraction(&self) -> f32 {
        self.rank as f32 / SHAPE_BUCKETS as f32
    }

    /// Whether this payload was frozen at exactly `version`.
    #[must_use]
    pub fn is_frozen_at(&self, version: u64) -> bool {
        self.version == version
    }
}

// A compile-time guard that the DTO stays scalar-free and small.
const _: () = assert!(core::mem::size_of::<ShapeRankPayload>() <= 144);

/// Remeasure guard key: `(StatId, Arm, Cohort, Metric, DatasetVersion)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RemeasureKey {
    /// Statistic identity.
    pub stat_id: u32,
    /// Experiment arm.
    pub arm: u8,
    /// Cohort identity.
    pub cohort: u32,
    /// Metric identity.
    pub metric: u32,
    /// Dataset version.
    pub dataset_version: u64,
}

/// Error returned when a [`RemeasureLedger`] key is sealed twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemeasureError {
    /// The key was already sealed at `sealed_version`; the new write was
    /// rejected rather than overwriting it.
    AlreadySealed {
        /// The key that was already sealed.
        key: RemeasureKey,
        /// The dataset version the existing payload was sealed at.
        sealed_version: u64,
    },
}

impl fmt::Display for RemeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RemeasureError::AlreadySealed {
                key,
                sealed_version,
            } => write!(
                f,
                "remeasure rejected: stat_id={} already sealed at version {sealed_version}",
                key.stat_id
            ),
        }
    }
}

impl std::error::Error for RemeasureError {}

/// Write-once ledger of sealed [`ShapeRankPayload`]s, keyed by [`RemeasureKey`].
#[derive(Debug, Default, Clone)]
pub struct RemeasureLedger {
    sealed: BTreeMap<RemeasureKey, ShapeRankPayload>,
}

impl RemeasureLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sealed: BTreeMap::new(),
        }
    }

    /// Write-once. `&mut self` here is the ONE sanctioned write path (a
    /// registry/builder, not a compute path — data-flow rule). A second
    /// write to a sealed key ERRORS; it never overwrites.
    pub fn seal(
        &mut self,
        key: RemeasureKey,
        payload: ShapeRankPayload,
    ) -> Result<(), RemeasureError> {
        if let Some(existing) = self.sealed.get(&key) {
            return Err(RemeasureError::AlreadySealed {
                key,
                sealed_version: existing.version,
            });
        }
        self.sealed.insert(key, payload);
        Ok(())
    }

    /// Look up a sealed payload by key.
    #[must_use]
    pub fn get(&self, key: &RemeasureKey) -> Option<&ShapeRankPayload> {
        self.sealed.get(key)
    }

    /// Number of sealed entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sealed.len()
    }

    /// Whether the ledger has no sealed entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sealed.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_has_no_scalar_and_fits_in_144_bytes() {
        assert!(core::mem::size_of::<ShapeRankPayload>() <= 144);
        let payload = ShapeRankPayload::new([1; SHAPE_BUCKETS], 4, 1);
        assert_eq!(payload.mass(), 16);
        assert_eq!(payload.mass_below(), 4);
        assert_eq!(payload.prozentrang(), 0.25);
        assert_eq!(payload.rank_fraction(), 0.25);
    }

    #[test]
    fn prozentrang_is_zero_on_empty_mass() {
        let payload = ShapeRankPayload::new([0; SHAPE_BUCKETS], 4, 1);
        assert_eq!(payload.prozentrang(), 0.0);
        assert!(!payload.prozentrang().is_nan());
    }

    #[test]
    #[should_panic]
    fn new_rejects_rank_out_of_range() {
        let _ = ShapeRankPayload::new([0; SHAPE_BUCKETS], 16, 1);
    }

    #[test]
    fn ledger_seals_once_and_errors_on_remeasure() {
        let mut ledger = RemeasureLedger::new();
        let key = RemeasureKey {
            stat_id: 1,
            arm: 0,
            cohort: 1,
            metric: 1,
            dataset_version: 1,
        };
        let v1 = ShapeRankPayload::new([1; SHAPE_BUCKETS], 4, 1);
        let v2 = ShapeRankPayload::new([2; SHAPE_BUCKETS], 4, 2);

        assert!(ledger.seal(key, v1).is_ok());
        let err = ledger.seal(key, v2).unwrap_err();
        assert_eq!(
            err,
            RemeasureError::AlreadySealed {
                key,
                sealed_version: 1
            }
        );
        assert_eq!(ledger.get(&key), Some(&v1));
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn ledger_distinguishes_every_key_field() {
        let mut ledger = RemeasureLedger::new();
        let base = RemeasureKey {
            stat_id: 1,
            arm: 0,
            cohort: 1,
            metric: 1,
            dataset_version: 1,
        };
        let payload = ShapeRankPayload::new([1; SHAPE_BUCKETS], 4, 1);
        assert!(ledger.seal(base, payload).is_ok());

        let variants = [
            RemeasureKey { stat_id: 2, ..base },
            RemeasureKey { arm: 1, ..base },
            RemeasureKey { cohort: 2, ..base },
            RemeasureKey { metric: 2, ..base },
            RemeasureKey {
                dataset_version: 2,
                ..base
            },
        ];
        for variant in variants {
            assert!(ledger.seal(variant, payload).is_ok());
        }
        assert_eq!(ledger.len(), 6);
    }

    #[test]
    fn error_display_names_the_key() {
        let key = RemeasureKey {
            stat_id: 42,
            arm: 0,
            cohort: 1,
            metric: 1,
            dataset_version: 1,
        };
        let err = RemeasureError::AlreadySealed {
            key,
            sealed_version: 7,
        };
        let msg = format!("{err}");
        assert!(msg.contains("stat_id"));
        assert!(msg.contains('7'));
    }
}
