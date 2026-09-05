//! `NestedBands` — the T2 sealed shape from plan
//! `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md`.
//!
//! A Belichtungsmesser reading over an `i32` value column is a chain of
//! nested row masks `M_0 ⊆ M_1 ⊆ … ⊆ M_{B-1}`. Bucket `i` is
//! `M_i ∧ ¬M_{i-1}` by the NAMED immediate `mask_ternlog::<AND_ANDNOT2>`. A
//! row's Prozentrang is its bucket index — a partition point over nested
//! masks, never a sort. `NestedBands` is built once (`NestedBandsBuilder`),
//! sealed, version-keyed, and NEVER mutated: `split`/`merge` return NEW
//! values under a new version (data-flow rule: no `&mut self` during
//! computation).
//!
//! Cites: E-NXG-1 (a Belichtungsmesser reading IS a nested mask set),
//! E-NXG-17 (PROBE-NXG-HIST-1 GREEN: nested/disjoint/rank-by-mask-walk),
//! E-NXG-18 (**the top band is the universe** — rows above the last
//! boundary must never be lost silently; a ladder whose last mask is not
//! the universe drops rows), E-NXG-19 (the budget/overflow test fires
//! strictly before the entropy/collapse signal on a real epoch shift —
//! the two are not interchangeable), E-NXG-20 (`mu + k·sigma` does not
//! name a rate; a floor read off the rank ladder is the best achievable
//! boundary for a target exceedance rate, and midpoint-estimated sigma
//! misreads the exact sigma, and not even in a stable direction).
//!
//! The three probes this module re-derives as library code (their mask
//! helpers' LOGIC was read and carried over, not their prose):
//! `examples/probe_nxg_hist_1.rs`, `examples/probe_nxg_roll_1.rs`,
//! `examples/probe_nxg_floor_1.rs`.

use lance_graph_contract::shape_rank::{ShapeRankPayload, SHAPE_BUCKETS};
use lance_graph_contract::thought_atoms::normalized_entropy;
use ndarray::simd::ternlog::{AND2, AND_ANDNOT2};
use ndarray::simd::{gt_i32_to_mask, mask_ternlog, popcount_batch_u64};

/// Fixed-point scale for Fisher-2z values on the i32 mask column: 2z ∈
/// roughly [−21, 21] at EPS=1e-9, so ×1024 keeps 3 decimals and stays far
/// inside i32.
pub const Z2_SCALE: f64 = 1024.0;

/// `round(z * Z2_SCALE)` saturated to i32; NaN → 0 (documented).
pub fn quantize_2z(z: f64) -> i32 {
    if z.is_nan() {
        return 0;
    }
    let scaled = (z * Z2_SCALE).round();
    if scaled >= i32::MAX as f64 {
        i32::MAX
    } else if scaled <= i32::MIN as f64 {
        i32::MIN
    } else {
        scaled as i32
    }
}

/// A sealed `NestedBands` value's version. Every `split`/`merge` returns a
/// new `NestedBands` under a caller-supplied new version; the version is
/// never incremented internally.
pub type Version = u64;

/// Per-bucket moment accumulator (`count`, `sum`, `sumsq` of the raw
/// column values that fell in the bucket). Two accumulators per bucket
/// are exactly what is needed to recover an exact sigma from the seal
/// (room-3 correction: histogram-midpoint sigma misreads the true
/// value — see [`NestedBands::sigma_exact`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BucketMoment {
    /// Number of rows in the bucket.
    pub count: u64,
    /// Sum of `value as f64` over the bucket's rows.
    pub sum: f64,
    /// Sum of `(value as f64)^2` over the bucket's rows.
    pub sumsq: f64,
}

/// Builder for a [`NestedBands`]. Configures band count and overflow
/// budget factor, then calibrates (quantile boundaries) or accepts an
/// explicit boundary ladder.
#[derive(Debug, Clone)]
pub struct NestedBandsBuilder {
    bands: usize,
    budget_factor: f64,
}

/// A sealed, nested-mask Belichtungsmesser reading over one `i32` column.
///
/// Invariants (asserted at construction, re-verified by the test suite):
/// bands are nested (`M_i` implies `M_{i+1}`), buckets are pairwise
/// disjoint, bucket popcounts sum to `rows`, and the top band is always
/// the universe (all-ones, tail cleared) so no row is ever silently
/// dropped (E-NXG-18).
#[derive(Debug, Clone)]
pub struct NestedBands {
    version: Version,
    /// Strictly ascending, length `bands - 1`. The top band has NO
    /// boundary entry — it is the universe by construction.
    boundaries: Vec<i32>,
    /// `bands[i]` = rows with value `<= boundaries[i]`; the last entry is
    /// all-ones (tail cleared beyond `rows`), never `le(top_boundary)`.
    bands: Vec<Vec<u64>>,
    /// `buckets[0] = bands[0]`; `buckets[i] = mask_ternlog::<AND_ANDNOT2>(bands[i], bands[i-1], bands[i-1])`.
    buckets: Vec<Vec<u64>>,
    /// Popcount of each bucket, cached at build time.
    popcounts: Vec<u64>,
    rows: usize,
    budget_factor: f64,
}

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

/// `M = rows with value <= boundary`, as `!gt` with the tail beyond `n`
/// cleared. Shared mask-walk logic, carried over from the probes.
fn le_mask(values: &[i32], boundary: i32) -> Vec<u64> {
    let n = values.len();
    let mut m = vec![0u64; words_for(n)];
    gt_i32_to_mask(values, boundary, &mut m);
    for w in m.iter_mut() {
        *w = !*w;
    }
    if !n.is_multiple_of(64) {
        let last = m.len() - 1;
        m[last] &= (1u64 << (n % 64)) - 1;
    }
    m
}

/// The universe mask: all-ones, tail beyond `n` cleared.
fn universe_mask(n: usize) -> Vec<u64> {
    let mut m = vec![u64::MAX; words_for(n)];
    if !n.is_multiple_of(64) {
        let last = m.len() - 1;
        m[last] &= (1u64 << (n % 64)) - 1;
    }
    m
}

fn bit(m: &[u64], row: usize) -> bool {
    (m[row / 64] >> (row % 64)) & 1 == 1
}

/// Build the nested band masks for a boundary ladder (E-NXG-18: the last
/// band is always the universe, never `le(top_boundary)`).
fn build_bands(values: &[i32], boundaries: &[i32]) -> Vec<Vec<u64>> {
    let n = values.len();
    let last = boundaries.len();
    (0..=last)
        .map(|i| {
            if i == last {
                universe_mask(n)
            } else {
                le_mask(values, boundaries[i])
            }
        })
        .collect()
}

/// Bucket `i` = `M_i ∧ ¬M_{i-1}`, by the NAMED immediate (E-NXG-1 / C3 of
/// PROBE-NXG-HIST-1).
fn build_buckets(bands: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let words = bands[0].len();
    (0..bands.len())
        .map(|i| {
            let mut b = vec![0u64; words];
            if i == 0 {
                b.copy_from_slice(&bands[0]);
            } else {
                mask_ternlog::<AND_ANDNOT2>(&bands[i], &bands[i - 1], &bands[i - 1], &mut b);
            }
            b
        })
        .collect()
}

impl NestedBandsBuilder {
    /// Start a builder targeting `bands` bands (before dedup in
    /// [`calibrate`](Self::calibrate)). `bands` must be at least 2 — a
    /// single band carries no rank information.
    pub fn new(bands: usize) -> Self {
        assert!(bands >= 2, "NestedBands needs at least 2 bands");
        Self {
            bands,
            budget_factor: 2.0,
        }
    }

    /// Set the overflow budget factor (a bucket may hold this many times
    /// its equal-mass share before [`NestedBands::overflow`] fires).
    /// Default 2.0.
    pub fn budget_factor(mut self, f: f64) -> Self {
        self.budget_factor = f;
        self
    }

    /// Calibrate quantile boundaries from `column` (the one sort in this
    /// module — a sorted COPY, never mutating the caller's data), dedup
    /// them to strictly ascending, and seal a [`NestedBands`] at
    /// `version`.
    pub fn calibrate(self, column: &[i32], version: Version) -> NestedBands {
        let n = column.len();
        let mut sorted = column.to_vec();
        sorted.sort_unstable();
        let mut boundaries: Vec<i32> = (0..self.bands - 1)
            .map(|i| sorted[((i + 1) * n / self.bands).saturating_sub(1)])
            .collect();
        boundaries.dedup();
        assert!(
            !boundaries.is_empty(),
            "NestedBands::calibrate: column is degenerate (all quantile boundaries collapsed to one value): {column:?}"
        );
        self.with_boundaries(boundaries, column, version)
    }

    /// Equal-WIDTH boundaries over `[min, max]` of `column` (the D-BLW-5
    /// design's "equal-width in 2z ≈ equal-information"), as opposed to
    /// [`calibrate`](Self::calibrate)'s equal-mass quantiles. `bands-1`
    /// boundaries at `min + k*(max-min)/bands`, `k = 1..bands-1`, deduped
    /// strictly ascending; panics if fewer than one boundary survives (a
    /// degenerate column). Then [`with_boundaries`](Self::with_boundaries).
    pub fn calibrate_equal_width(self, column: &[i32], version: Version) -> NestedBands {
        let min = *column
            .iter()
            .min()
            .expect("NestedBandsBuilder::calibrate_equal_width: empty column");
        let max = *column
            .iter()
            .max()
            .expect("NestedBandsBuilder::calibrate_equal_width: empty column");
        let span = (max - min) as f64;
        let bands = self.bands as f64;
        let mut boundaries: Vec<i32> = (1..self.bands)
            .map(|k| min + ((k as f64) * span / bands).round() as i32)
            .collect();
        boundaries.dedup();
        assert!(
            !boundaries.is_empty(),
            "NestedBandsBuilder::calibrate_equal_width: column is degenerate (all equal-width boundaries collapsed to one value): {column:?}"
        );
        self.with_boundaries(boundaries, column, version)
    }

    /// Seal a [`NestedBands`] from an explicit, strictly ascending
    /// boundary ladder. `boundaries` must be non-empty; the builder's own
    /// `bands` field is only consulted by [`calibrate`](Self::calibrate) —
    /// here `band_count` is `boundaries.len() + 1`.
    pub fn with_boundaries(
        self,
        boundaries: Vec<i32>,
        column: &[i32],
        version: Version,
    ) -> NestedBands {
        assert!(
            !boundaries.is_empty(),
            "NestedBands::with_boundaries: empty boundary ladder"
        );
        assert!(
            boundaries.windows(2).all(|w| w[0] < w[1]),
            "NestedBands::with_boundaries: boundaries not strictly ascending: {boundaries:?}"
        );
        let rows = column.len();
        let bands = build_bands(column, &boundaries);
        let buckets = build_buckets(&bands);
        let popcounts = buckets.iter().map(|m| popcount_batch_u64(m)).collect();
        NestedBands {
            version,
            boundaries,
            bands,
            buckets,
            popcounts,
            rows,
            budget_factor: self.budget_factor,
        }
    }
}

impl NestedBands {
    /// This sealed value's version.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Number of rows this reading was built over.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of bands (= `boundaries().len() + 1`).
    pub fn band_count(&self) -> usize {
        self.boundaries.len() + 1
    }

    /// The strictly ascending boundary ladder. Length `band_count() - 1`
    /// — the top band has no boundary entry (it is the universe).
    pub fn boundaries(&self) -> &[i32] {
        &self.boundaries
    }

    /// The `i`-th cumulative band mask (`M_i`).
    pub fn band(&self, i: usize) -> &[u64] {
        &self.bands[i]
    }

    /// The `i`-th bucket mask (`M_i ∧ ¬M_{i-1}`).
    pub fn bucket(&self, i: usize) -> &[u64] {
        &self.buckets[i]
    }

    /// Per-bucket popcounts, cached at build time.
    pub fn popcounts(&self) -> &[u64] {
        &self.popcounts
    }

    /// Alias of [`popcounts`](Self::popcounts) — the payload-law "shape"
    /// of this reading.
    pub fn shape(&self) -> &[u64] {
        &self.popcounts
    }

    /// The bucket index containing `row`, read directly off the bucket
    /// masks (never the value column). `row` must be `< rows()`.
    pub fn rank(&self, row: usize) -> usize {
        debug_assert!(
            row < self.rows,
            "NestedBands::rank: row {row} out of bounds ({})",
            self.rows
        );
        (0..self.buckets.len())
            .find(|&i| bit(&self.buckets[i], row))
            .expect("NestedBands::rank: row is in no bucket — the top band is not the universe")
    }

    /// The bucket index a raw value `v` would fall into, computed
    /// directly from the boundary ladder (never exceeds `band_count() - 1`
    /// by construction, since `partition_point` on a length-`(band_count
    /// - 1)` slice returns at most that length).
    pub fn rank_of_value(&self, v: i32) -> usize {
        self.boundaries.partition_point(|&b| b < v)
    }

    /// D-NXG-4 → D-BLW-5: the payload-law object. `shape` = this ladder's
    /// popcounts (the pooled prior's census), `rank` = `rank_of_value(observed)`
    /// (the observed statistic's Prozentrang bucket), `version` = the V₀ the
    /// caller freezes. Panics unless `band_count() == SHAPE_BUCKETS`.
    pub fn shape_rank(&self, observed: i32, version: Version) -> ShapeRankPayload {
        assert_eq!(
            self.band_count(),
            SHAPE_BUCKETS,
            "NestedBands::shape_rank: band_count() must equal SHAPE_BUCKETS ({SHAPE_BUCKETS}), got {}",
            self.band_count()
        );
        let mut shape = [0u64; SHAPE_BUCKETS];
        shape.copy_from_slice(&self.popcounts);
        let rank = self.rank_of_value(observed) as u8;
        ShapeRankPayload::new(shape, rank, version)
    }

    /// Normalized Shannon entropy of the bucket-popcount histogram (1.0 =
    /// perfectly flat, falling toward 0 as mass concentrates in fewer
    /// buckets). `0.0` if the histogram is degenerate.
    pub fn entropy(&self) -> f32 {
        let weights: Vec<f32> = self.popcounts.iter().map(|&p| p as f32).collect();
        normalized_entropy(&weights).unwrap_or(0.0)
    }

    /// The overflow budget: `budget_factor * rows / band_count`, rounded
    /// down. A bucket past this many rows is holding more than its
    /// allotted multiple of an equal-mass share.
    pub fn budget(&self) -> u64 {
        (self.budget_factor * self.rows as f64 / self.band_count() as f64) as u64
    }

    /// Index of the largest bucket, iff its popcount exceeds
    /// [`budget`](Self::budget) — `None` if every bucket is within
    /// budget (E-NXG-19: this test fires strictly before
    /// [`entropy`](Self::entropy) collapses on a real epoch shift, and
    /// stays silent on the ladder's own calibration epoch).
    pub fn overflow(&self) -> Option<usize> {
        let budget = self.budget();
        let (idx, &max) = self
            .popcounts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &p)| p)
            .expect("NestedBands: at least one bucket");
        (max > budget).then_some(idx)
    }

    /// The merge-on-collapse arm: the smallest adjacent bucket pair
    /// `(i, i+1)` whose combined popcount is under half an equal share of
    /// the rows, iff one exists. `None` on a well-spread ladder (e.g. its
    /// own calibration epoch).
    pub fn collapsed(&self) -> Option<usize> {
        let threshold = self.rows / (2 * self.band_count());
        (0..self.popcounts.len().saturating_sub(1))
            .filter(|&i| self.popcounts[i] + self.popcounts[i + 1] < threshold as u64)
            .min_by_key(|&i| self.popcounts[i] + self.popcounts[i + 1])
    }

    /// Split `bucket` by partial-popcount bisection restricted to that
    /// bucket's own mask, inserting a new boundary strictly inside its
    /// value range and rebuilding under `version`. Returns a NEW value —
    /// `self` is never mutated (data-flow rule: no `&mut self` during
    /// computation). `None` if every candidate value in range gives an
    /// all-ties split (`below == 0` or `below == pop` for every `v`), or
    /// if the chosen value already sits in the boundary ladder.
    pub fn split(&self, bucket: usize, column: &[i32], version: Version) -> Option<NestedBands> {
        assert_eq!(
            column.len(),
            self.rows,
            "NestedBands::split: column length mismatch"
        );
        let band_count = self.band_count();
        // Bucket 0 covers everything <= boundaries[0], which on an i32 column
        // includes negative values (Fisher-2z of a negative r is negative), so
        // the search starts at the column's minimum, never at 0.
        let lo = if bucket == 0 {
            *column
                .iter()
                .min()
                .expect("NestedBands::split: empty column")
        } else {
            self.boundaries[bucket - 1] + 1
        };
        let hi = if bucket == band_count - 1 {
            *column
                .iter()
                .max()
                .expect("NestedBands::split: empty column")
        } else {
            self.boundaries[bucket]
        };
        let bucket_mask = &self.buckets[bucket];
        let pop = popcount_batch_u64(bucket_mask);
        let target = pop / 2;
        let words = bucket_mask.len();
        let mut scratch = vec![0u64; words];
        let (mut a, mut b) = (lo, hi);
        let mut best: Option<(i32, u64, u64)> = None;
        while a < b {
            let mid = a + (b - a) / 2;
            let m = le_mask(column, mid);
            mask_ternlog::<AND2>(bucket_mask, &m, bucket_mask, &mut scratch);
            let below = popcount_batch_u64(&scratch);
            let err = below.abs_diff(target);
            if best.is_none_or(|(_, e, _)| err < e) {
                best = Some((mid, err, below));
            }
            if below < target {
                a = mid + 1;
            } else {
                b = mid;
            }
        }
        let (v, _, below) = best?;
        if below == 0 || below == pop {
            return None;
        }
        if self.boundaries.contains(&v) {
            return None;
        }
        let mut new_boundaries = self.boundaries.clone();
        let pos = new_boundaries.partition_point(|&x| x < v);
        new_boundaries.insert(pos, v);
        Some(
            NestedBandsBuilder::new(2)
                .budget_factor(self.budget_factor)
                .with_boundaries(new_boundaries, column, version),
        )
    }

    /// Remove boundary `lower` (merging buckets `lower` and `lower + 1`
    /// into one) and rebuild under `version`. Returns a NEW value — `self`
    /// is never mutated. `None` if `band_count() <= 2` (nothing left to
    /// merge into) or `lower >= band_count() - 1` (out of range).
    pub fn merge(&self, lower: usize, column: &[i32], version: Version) -> Option<NestedBands> {
        assert_eq!(
            column.len(),
            self.rows,
            "NestedBands::merge: column length mismatch"
        );
        let band_count = self.band_count();
        if band_count <= 2 || lower >= band_count - 1 {
            return None;
        }
        let mut new_boundaries = self.boundaries.clone();
        new_boundaries.remove(lower);
        Some(
            NestedBandsBuilder::new(2)
                .budget_factor(self.budget_factor)
                .with_boundaries(new_boundaries, column, version),
        )
    }

    /// E-NXG-20: bisection over `[min(column), max(column))` using only
    /// `gt_i32_to_mask` + popcount (no sort) — the smallest value `v`
    /// whose exceedance rate (`popcount(gt v) / rows`) is at most `rate`.
    /// Exceedance is non-increasing in `v`, so this is the best achievable
    /// floor for the requested rate: a lower floor's rate can only be
    /// worse (higher), and no strictly smaller value in range beats it.
    /// Returns `(v, exceedance(v))`.
    pub fn best_achievable_floor(&self, column: &[i32], rate: f64) -> (i32, f64) {
        assert_eq!(
            column.len(),
            self.rows,
            "NestedBands::best_achievable_floor: column length mismatch"
        );
        let n = column.len();
        let exceedance = |v: i32| -> f64 {
            let mut m = vec![0u64; words_for(n)];
            gt_i32_to_mask(column, v, &mut m);
            if !n.is_multiple_of(64) {
                let last = m.len() - 1;
                m[last] &= (1u64 << (n % 64)) - 1;
            }
            popcount_batch_u64(&m) as f64 / n as f64
        };
        let mut lo = *column.iter().min().expect("NestedBands: empty column");
        let mut hi = *column.iter().max().expect("NestedBands: empty column");
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if exceedance(mid) <= rate {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        (lo, exceedance(lo))
    }

    /// Per-bucket `(count, sum, sumsq)` of `column`'s values, walked bit
    /// by bit off each bucket mask (word scan + `trailing_zeros` +
    /// `w &= w - 1`).
    pub fn moments(&self, column: &[i32]) -> Vec<BucketMoment> {
        assert_eq!(
            column.len(),
            self.rows,
            "NestedBands::moments: column length mismatch"
        );
        self.buckets
            .iter()
            .map(|bucket| {
                let mut count = 0u64;
                let mut sum = 0.0f64;
                let mut sumsq = 0.0f64;
                for (word_idx, &word) in bucket.iter().enumerate() {
                    let mut w = word;
                    while w != 0 {
                        let bit_idx = w.trailing_zeros() as usize;
                        let row = word_idx * 64 + bit_idx;
                        let v = column[row] as f64;
                        count += 1;
                        sum += v;
                        sumsq += v * v;
                        w &= w - 1;
                    }
                }
                BucketMoment { count, sum, sumsq }
            })
            .collect()
    }

    /// Exact sigma recovered from [`moments`](Self::moments) — the room-3
    /// correction: sigma is recoverable from the seal iff the seal stores
    /// two accumulators per bucket (16 bytes/bucket), not from bucket
    /// midpoints, which misread the true value in a direction that depends
    /// on the arbitrary top-bucket midpoint (E-NXG-17).
    pub fn sigma_exact(&self, column: &[i32]) -> f64 {
        assert_eq!(
            column.len(),
            self.rows,
            "NestedBands::sigma_exact: column length mismatch"
        );
        let moments = self.moments(column);
        let n: u64 = moments.iter().map(|m| m.count).sum();
        let total: f64 = moments.iter().map(|m| m.sum).sum();
        let totsq: f64 = moments.iter().map(|m| m.sumsq).sum();
        let n = n as f64;
        (totsq / n - (total / n).powi(2)).max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::simd::mask_andnot;
    use ndarray::simd::ternlog::AND2 as TEST_AND2;

    const WAV_HEADER: usize = 44;
    const SPEECH: &[u8] = include_bytes!("../../../data/tts-cascade/tts_real_output.wav");
    const SATURATED: &[u8] =
        include_bytes!("../../../data/tts-cascade/cascade_speech_128frames.wav");
    const QUIET: &[u8] = include_bytes!("../../../data/tts-cascade/cascade_output.wav");

    /// Decode a 16-bit mono PCM WAV's samples as `|sample|` (copied from
    /// the probes' `load_abs_samples`, minus the file read).
    fn abs_samples(bytes: &[u8]) -> Vec<i32> {
        assert!(
            bytes.len() > WAV_HEADER && &bytes[0..4] == b"RIFF",
            "not a RIFF file"
        );
        assert_eq!(&bytes[8..12], b"WAVE", "not WAVE");
        let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
        let ch = u16::from_le_bytes([bytes[22], bytes[23]]);
        assert_eq!((bits, ch), (16, 1), "expected 16-bit mono PCM");
        bytes[WAV_HEADER..]
            .chunks_exact(2)
            .map(|c| (i16::from_le_bytes([c[0], c[1]]) as i32).abs())
            .collect()
    }

    fn nested_ok(nb: &NestedBands) -> bool {
        let words = nb.band(0).len();
        let mut scratch = vec![0u64; words];
        for i in 0..nb.band_count() - 1 {
            mask_andnot(nb.band(i), nb.band(i + 1), &mut scratch);
            if popcount_batch_u64(&scratch) != 0 {
                return false;
            }
        }
        true
    }

    fn disjoint_ok(nb: &NestedBands) -> bool {
        let words = nb.bucket(0).len();
        let mut scratch = vec![0u64; words];
        for i in 0..nb.band_count() {
            for j in i + 1..nb.band_count() {
                mask_ternlog::<TEST_AND2>(nb.bucket(i), nb.bucket(j), nb.bucket(i), &mut scratch);
                if popcount_batch_u64(&scratch) != 0 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn hist_partition_and_rank_on_speech() {
        let speech = abs_samples(SPEECH);
        let n = speech.len();
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        assert_eq!(nb.popcounts().iter().sum::<u64>(), n as u64);
        assert!(disjoint_ok(&nb), "buckets must be pairwise disjoint");
        assert!(nested_ok(&nb), "bands must be nested");
        for (row, &v) in speech.iter().enumerate() {
            assert_eq!(
                nb.rank(row),
                nb.rank_of_value(v),
                "rank mismatch at row {row}"
            );
        }
        let nonempty = nb.popcounts().iter().filter(|&&p| p > 0).count();
        assert_eq!(
            nonempty, 16,
            "all 16 buckets must be non-empty on real speech"
        );
        let max_bucket = *nb.popcounts().iter().max().unwrap();
        assert!(
            (max_bucket as usize) * 2 < n,
            "largest bucket must hold less than half the rows"
        );
    }

    #[test]
    fn top_band_is_the_universe_on_a_shifted_epoch() {
        let speech = abs_samples(SPEECH);
        let saturated = abs_samples(SATURATED);
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        let stream: Vec<i32> = speech.iter().chain(saturated.iter()).copied().collect();
        let shifted =
            NestedBandsBuilder::new(16).with_boundaries(nb.boundaries().to_vec(), &stream, 2);
        assert_eq!(
            shifted.popcounts().iter().sum::<u64>(),
            stream.len() as u64,
            "E-NXG-18: top band must be the universe"
        );
        assert_eq!(shifted.overflow(), Some(shifted.band_count() - 1));
    }

    #[test]
    fn overflow_stays_silent_on_its_calibration_epoch() {
        let speech = abs_samples(SPEECH);
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        assert!(
            nb.overflow().is_none(),
            "the budget rule must not fire on its own calibration epoch"
        );
    }

    #[test]
    fn split_halves_the_worst_bucket_and_rewrites_nothing() {
        let speech = abs_samples(SPEECH);
        let saturated = abs_samples(SATURATED);
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        let stream: Vec<i32> = speech.iter().chain(saturated.iter()).copied().collect();
        let shifted =
            NestedBandsBuilder::new(16).with_boundaries(nb.boundaries().to_vec(), &stream, 2);
        let before = shifted.popcounts().to_vec();
        let w = shifted.overflow().unwrap();
        let s = shifted.split(w, &stream, 3).unwrap();
        assert_eq!(s.band_count(), shifted.band_count() + 1);
        assert_eq!(s.version(), 3);
        assert_eq!(s.popcounts().iter().sum::<u64>(), stream.len() as u64);
        assert!(nested_ok(&s));
        assert!(*s.popcounts().iter().max().unwrap() < *before.iter().max().unwrap());
        for b in shifted.boundaries() {
            assert!(
                s.boundaries().contains(b),
                "old boundary {b} lost after split"
            );
        }
        assert_eq!(
            shifted.popcounts(),
            before.as_slice(),
            "split must not mutate the original"
        );
    }

    #[test]
    fn collapse_fires_on_a_quiet_epoch_and_stays_silent_on_calibration() {
        let speech = abs_samples(SPEECH);
        let quiet = abs_samples(QUIET);
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        assert!(
            nb.collapsed().is_none(),
            "collapse must not fire on its own calibration epoch"
        );
        let on_quiet =
            NestedBandsBuilder::new(16).with_boundaries(nb.boundaries().to_vec(), &quiet, 2);
        assert!(
            on_quiet.collapsed().is_some(),
            "collapse must fire when the ladder meets a quiet epoch"
        );
    }

    #[test]
    fn merge_removes_one_boundary_and_sums_the_pair() {
        let speech = abs_samples(SPEECH);
        let quiet = abs_samples(QUIET);
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        let on_quiet =
            NestedBandsBuilder::new(16).with_boundaries(nb.boundaries().to_vec(), &quiet, 2);
        let i = on_quiet.collapsed().unwrap();
        let p = on_quiet.popcounts().to_vec();
        let m = on_quiet.merge(i, &quiet, 5).unwrap();
        assert_eq!(m.band_count(), on_quiet.band_count() - 1);
        assert_eq!(m.popcounts()[i], p[i] + p[i + 1]);
        assert_eq!(m.popcounts().iter().sum::<u64>(), quiet.len() as u64);
        assert!(nested_ok(&m));

        // merge on a 2-band ladder returns None.
        let two_band = NestedBandsBuilder::new(2).with_boundaries(vec![100], &quiet, 9);
        assert!(two_band.merge(0, &quiet, 10).is_none());
    }

    #[test]
    fn split_returns_none_when_a_bucket_is_all_ties() {
        // Synthetic is fine HERE: this is a degenerate-input guard, not a
        // distributional claim, so a real recording adds nothing.
        let column = vec![7i32; 640];
        let nb = NestedBandsBuilder::new(2).with_boundaries(vec![7], &column, 1);
        assert!(nb.split(0, &column, 2).is_none());
    }

    #[test]
    fn best_achievable_floor_pins_the_measured_values() {
        // The floor is the SMALLEST value whose exceedance is <= rate — strictly,
        // no row of slack. PROBE-NXG-FLOOR-1 pinned speech at 15 072, but that
        // rank floor sat ONE ROW over the target (473/94 572 = 0.0050015 > 0.005)
        // inside the probe's ±1/n tolerance; the strict definition lands at
        // 15 077 (472 rows above, 0.004991). Saturated and quiet are unchanged.
        let rate = 0.005;
        for (bytes, expect_floor) in [(SPEECH, 15077), (SATURATED, 32765), (QUIET, 5346)] {
            let column = abs_samples(bytes);
            let nb = NestedBandsBuilder::new(2).with_boundaries(
                vec![*column.iter().max().unwrap() - 1],
                &column,
                1,
            );
            let (floor, achieved) = nb.best_achievable_floor(&column, rate);
            assert_eq!(floor, expect_floor, "floor mismatch");
            assert!(
                achieved <= rate,
                "achieved rate {achieved} exceeds the target {rate}"
            );
            // Reference: the strict definition computed from a sorted copy.
            let mut sorted = column.clone();
            sorted.sort_unstable();
            let n = column.len();
            let max_over = (rate * n as f64).floor() as usize;
            let expect_ref = sorted[n - 1 - max_over];
            assert_eq!(
                floor, expect_ref,
                "floor disagrees with the sorted-copy reference"
            );

            // The next distinct value below the floor overshoots the rate —
            // E-NXG-20's "best achievable".
            let mut sorted = column.clone();
            sorted.sort_unstable();
            if let Some(&next_below) = sorted.iter().rev().find(|&&x| x < floor) {
                let mut m = vec![0u64; column.len().div_ceil(64)];
                gt_i32_to_mask(&column, next_below, &mut m);
                let n = column.len();
                if !n.is_multiple_of(64) {
                    let last = m.len() - 1;
                    m[last] &= (1u64 << (n % 64)) - 1;
                }
                let over = popcount_batch_u64(&m) as f64 / n as f64;
                assert!(
                    over > rate,
                    "value below the floor did not overshoot the rate"
                );
            }
        }
    }

    #[test]
    fn sigma_exact_matches_direct_and_midpoints_do_not() {
        let speech = abs_samples(SPEECH);
        let n = speech.len() as f64;
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);

        let mean_d = speech.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var_d = speech
            .iter()
            .map(|&v| (v as f64 - mean_d).powi(2))
            .sum::<f64>()
            / n;
        let sigma_direct = var_d.sqrt();
        let sigma_exact = nb.sigma_exact(&speech);
        let rel_err = (sigma_exact - sigma_direct).abs() / sigma_direct;
        assert!(
            rel_err < 1e-9,
            "sigma_exact relative error {rel_err} too large"
        );

        // Test-local midpoint estimate: bucket midpoints weighted by popcount.
        let boundaries = nb.boundaries();
        let mids: Vec<f64> = (0..nb.band_count())
            .map(|i| {
                let lo = if i == 0 {
                    0.0
                } else {
                    boundaries[i - 1] as f64
                };
                // The top bucket is open-ended (E-NXG-18) and has no boundary,
                // so it has no midpoint either. Use the column maximum, as the
                // probe effectively did; this arbitrariness is the finding.
                let hi = if i == boundaries.len() {
                    *speech.iter().max().unwrap() as f64
                } else {
                    boundaries[i] as f64
                };
                (lo + hi) / 2.0
            })
            .collect();
        let mean_h = mids
            .iter()
            .zip(nb.popcounts())
            .map(|(m, &p)| m * p as f64)
            .sum::<f64>()
            / n;
        let var_h = mids
            .iter()
            .zip(nb.popcounts())
            .map(|(m, &p)| (m - mean_h).powi(2) * p as f64)
            .sum::<f64>()
            / n;
        let sigma_h = var_h.sqrt();
        // Direction is NOT asserted: with the stale top boundary the probe
        // over-read by 12 %; with the top bucket's midpoint at the last
        // boundary it under-reads by 7 %. The estimator is not even
        // sign-stable, which is why the seal stores moments instead.
        let rel = (sigma_h - sigma_direct).abs() / sigma_direct;
        assert!(
            rel > 0.05,
            "midpoint sigma {sigma_h} is within 5% of direct sigma {sigma_direct} — E-NXG-17's correction would be unnecessary"
        );
    }

    #[test]
    fn entropy_reads_flat_on_calibration_and_collapsed_after_shift() {
        let speech = abs_samples(SPEECH);
        let saturated = abs_samples(SATURATED);
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        assert!(
            nb.entropy() > 0.999,
            "entropy on the calibration epoch should read flat: {}",
            nb.entropy()
        );
        let stream: Vec<i32> = speech.iter().chain(saturated.iter()).copied().collect();
        let shifted =
            NestedBandsBuilder::new(16).with_boundaries(nb.boundaries().to_vec(), &stream, 2);
        assert!(
            shifted.entropy() < 0.9,
            "entropy after the shift should have collapsed: {}",
            shifted.entropy()
        );
    }

    #[test]
    #[should_panic]
    fn builder_rejects_fewer_than_two_bands() {
        let _ = NestedBandsBuilder::new(1);
    }

    #[test]
    fn rank_walk_matches_partition_point_on_all_three_columns() {
        for bytes in [SPEECH, SATURATED, QUIET] {
            let column = abs_samples(bytes);
            let nb = NestedBandsBuilder::new(8).calibrate(&column, 1);
            for (row, &v) in column.iter().enumerate() {
                assert_eq!(
                    nb.rank(row),
                    nb.rank_of_value(v),
                    "rank mismatch at row {row}"
                );
            }
        }
    }

    // ─────────────────────── D-BLW-5 / shape_rank ───────────────────────

    #[test]
    fn equal_width_boundaries_are_equal_width() {
        let speech = abs_samples(SPEECH);
        let nb = NestedBandsBuilder::new(16).calibrate_equal_width(&speech, 1);
        let boundaries = nb.boundaries();
        let diffs: Vec<i32> = boundaries.windows(2).map(|w| w[1] - w[0]).collect();
        let min_d = *diffs.iter().min().expect("at least one diff");
        let max_d = *diffs.iter().max().expect("at least one diff");
        assert!(
            max_d - min_d <= 1,
            "equal-width boundary diffs vary by more than 1: min={min_d} max={max_d} ({diffs:?})"
        );
    }

    #[test]
    fn quantize_2z_saturates_and_maps_nan_to_zero() {
        assert_eq!(quantize_2z(f64::NAN), 0);
        assert_eq!(quantize_2z(0.0), 0);
        assert_eq!(quantize_2z(1.0), 1024);
        assert_eq!(quantize_2z(1e30), i32::MAX);
        assert_eq!(quantize_2z(-1e30), i32::MIN);
        assert_eq!(quantize_2z(f64::INFINITY), i32::MAX);
        assert_eq!(quantize_2z(f64::NEG_INFINITY), i32::MIN);
    }

    /// Lag-1 Pearson autocorrelation of `frame`, `0.0` if either half has
    /// zero variance. Test-local: no generator, real data only.
    fn lag1_r(frame: &[i32]) -> f64 {
        let n = frame.len();
        if n < 2 {
            return 0.0;
        }
        let xs: Vec<f64> = frame[..n - 1].iter().map(|&v| v as f64).collect();
        let ys: Vec<f64> = frame[1..].iter().map(|&v| v as f64).collect();
        let mx = xs.iter().sum::<f64>() / xs.len() as f64;
        let my = ys.iter().sum::<f64>() / ys.len() as f64;
        let mut cov = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for (a, b) in xs.iter().zip(ys.iter()) {
            cov += (a - mx) * (b - my);
            vx += (a - mx) * (a - mx);
            vy += (b - my) * (b - my);
        }
        if vx == 0.0 || vy == 0.0 {
            return 0.0;
        }
        cov / (vx.sqrt() * vy.sqrt())
    }

    #[test]
    fn shape_rank_from_real_lag1_autocorrelations() {
        const FRAME: usize = 1024;
        let speech = abs_samples(SPEECH);
        let saturated = abs_samples(SATURATED);
        let quiet = abs_samples(QUIET);

        // Cut speech into 1024-sample frames (drop the tail), lag-1 r per
        // frame, through jc::stats::fisher_2z, quantized — the pooled prior.
        let frames: Vec<i32> = speech
            .chunks_exact(FRAME)
            .map(|f| quantize_2z(jc::stats::fisher_2z(lag1_r(f))))
            .collect();
        let frame_count = frames.len();

        let nb = NestedBandsBuilder::new(16).calibrate_equal_width(&frames, 7);

        // Can-it-fire, INSIDE the prior's support: the speech frame with the
        // lowest lag-1 r and the one with the highest are real statistics
        // from the same population the ladder was calibrated on, so they
        // must land in different buckets — the ladder discriminates.
        let (lo_q, hi_q) = (*frames.iter().min().unwrap(), *frames.iter().max().unwrap());
        let lo_payload = nb.shape_rank(lo_q, 7);
        let hi_payload = nb.shape_rank(hi_q, 7);
        assert_eq!(
            lo_payload.shape.iter().sum::<u64>(),
            frame_count as u64,
            "shape must sum to the frame count"
        );
        assert_eq!(lo_payload.version, 7);
        assert_eq!(lo_payload.rank, 0, "the minimum of the prior is bucket 0");
        assert_eq!(
            hi_payload.rank, 15,
            "the maximum of the prior is the open-ended top bucket"
        );
        assert!(hi_payload.prozentrang() > lo_payload.prozentrang());

        // FINDING (pinned, not weakened): the two OTHER recordings' whole-file
        // lag-1 r sit far BELOW the speech prior's support (2z ≈ 1.46 and 1.67
        // against a prior spanning ≈ 3.18..5.48). Both therefore rank 0 — and
        // are indistinguishable from the prior's own minimum. shape × rank
        // saturates at the edge bucket: an out-of-support statistic carries
        // no "how far out" beyond rank 0. Recorded as E-NXG-22; the fix, if
        // wanted, is a design decision for the D-BLW-5 loop, not a test edit.
        let whole = |samples: &[i32]| quantize_2z(jc::stats::fisher_2z(lag1_r(samples)));
        let (sat_q, quiet_q) = (whole(&saturated), whole(&quiet));
        assert!(
            sat_q < nb.boundaries()[0] && quiet_q < nb.boundaries()[0],
            "both must be below the ladder"
        );
        assert_eq!(nb.shape_rank(sat_q, 7).rank, 0);
        assert_eq!(nb.shape_rank(quiet_q, 7).rank, 0);
    }

    /// Signed decode of the same recordings: a REAL column with negative values.
    fn signed_samples(bytes: &[u8]) -> Vec<i32> {
        bytes[WAV_HEADER..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
            .collect()
    }

    #[test]
    fn split_handles_a_negative_bucket_zero() {
        // CodeRabbit finding on #1181: bucket 0's bisection started at 0, so a
        // column whose bucket 0 lies below zero could never be split.
        let speech = signed_samples(SPEECH);
        assert!(
            *speech.iter().min().unwrap() < 0,
            "the signed column must go negative"
        );
        let nb = NestedBandsBuilder::new(16).calibrate(&speech, 1);
        assert!(
            nb.boundaries()[0] < 0,
            "bucket 0 must lie entirely below zero for this test to bite"
        );
        let before = nb.popcounts()[0];
        let s = nb
            .split(0, &speech, 2)
            .expect("a negative bucket 0 must be splittable");
        assert_eq!(s.band_count(), nb.band_count() + 1);
        assert!(
            s.boundaries()[0] < nb.boundaries()[0],
            "the new boundary sits inside bucket 0"
        );
        assert!(s.popcounts()[0] < before && s.popcounts()[1] < before);
        assert_eq!(
            s.popcounts()[0] + s.popcounts()[1],
            before,
            "the split partitions bucket 0 exactly"
        );
    }

    #[test]
    #[should_panic]
    fn shape_rank_panics_off_16_bands() {
        let column: Vec<i32> = (0..8).collect();
        let nb = NestedBandsBuilder::new(8).with_boundaries(vec![1, 2, 3, 4, 5, 6, 7], &column, 1);
        let _ = nb.shape_rank(0, 1);
    }

    #[test]
    fn shape_rank_round_trips_through_the_remeasure_ledger() {
        use lance_graph_contract::shape_rank::{RemeasureKey, RemeasureLedger};
        let speech = abs_samples(SPEECH);
        let nb = NestedBandsBuilder::new(16).calibrate_equal_width(&speech, 1);
        let payload = nb.shape_rank(speech[0], 1);

        let mut ledger = RemeasureLedger::new();
        let key = RemeasureKey {
            stat_id: 1,
            arm: 0,
            cohort: 1,
            metric: 1,
            dataset_version: 1,
        };
        assert!(ledger.seal(key, payload).is_ok());
        assert_eq!(ledger.get(&key), Some(&payload));
        assert!(ledger.seal(key, payload).is_err());
    }
}
