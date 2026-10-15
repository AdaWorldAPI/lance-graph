//! `temporal_pov` — a zero-dep, consumer-facing **temporal point-of-view filter**
//! over the Lance version stream.
//!
//! ## What this mirrors (and does NOT reimplement)
//!
//! `lance-graph-planner::temporal` (`crates/lance-graph-planner/src/temporal.rs`)
//! is the canonical epistemology: `QueryReference` (`server_id` + `ref_version` +
//! `hlc_tick` + `EpistemicMode` + `rung`), `TemporalStatus`
//! (`Contemporary`/`Anachronistic`/`Spoiler`/`Unknowable`), `classify`, and
//! `deinterlace` (the merge-sort that turns interlaced per-writer rows into the
//! causally-coherent standing-wave projection a reader deliberates over). That
//! logic **stays in `lance-graph-planner`**, which depends on this crate — never
//! the other way around (a zero-dep crate cannot import a downstream crate's
//! types, and re-deriving `EpistemicMode`/`TemporalStatus` here would be exactly
//! the "4× Fingerprint, 3× ZeckF64" duplication `LATEST_STATE.md`'s Type
//! Duplication table already warns against).
//!
//! This module is the **narrower shape** a caller passes across that boundary:
//! a half-open [`VersionRange`] plus the reader's `rung` (mirroring
//! `QueryReference::rung`, temporal.rs:122–124), packaged as [`TemporalPov`].
//! [`TemporalPov::admits`] answers only the version-range half of admission —
//! "does this row's version fall in the window this reader's `V_ref` covers" —
//! quoting `EpistemicMode::Strict`'s doc (temporal.rs:53–55): *"Only
//! `CONTEMPORARY` rows (`row_version ≤ ref_version`)"*. The richer per-row
//! classification (`knowable_from`, `Spoiler` vs `Anachronistic`, the
//! `DependsClosure` data-axis) needs per-row data this filter does not carry,
//! and is deliberately left to `classify`/`deinterlace` on the planner side.
//!
//! ## Why this exists
//!
//! Operator directive (2026-07-12): a time-range filter for temporal POV, using
//! the `temporal.rs` research. Generalizes `E-MARKOV-TEMPORAL-STREAM-1`
//! (`.claude/board/EPIPHANIES.md`, 2026-07-10 — "the version-range read
//! (`QueryReference::at(v, rung)` + deinterlace) generalizes the VSA ±5 braid:
//! any window width, per-reader epistemic rung, ... auditable/replayable, and
//! still a projection") and instantiates its measured worked example,
//! `D-SF-EPISODIC-1` (`.claude/knowledge/stockfish-nnue-as-perturbation-cascade.md`
//! — a game as a `temporal.rs`-shaped version-stream; "position after ply *v*"
//! is a zero-copy projection via `QueryReference::at(v, rung)` + deinterlace,
//! measured byte-identical GREEN both fresh-from-FEN (34/34 plies) and
//! out-of-order replay (11/11 queries)). [`VersionRange`]/[`TemporalPov`] are
//! the zero-dep vocabulary a consumer (e.g. an OGAR-side reader, or a future
//! non-planner crate) uses to express "which version window, at which rung"
//! without depending on `lance-graph-planner`.

/// A Lance dataset version — the storage frame's clock tick. Zero-dep mirror of
/// `lance_graph_planner::temporal::LanceVersion` (`type LanceVersion = u64`,
/// temporal.rs:47–48).
pub type LanceVersion = u64;

/// A half-open Lance-version interval `[from, to)` — versions `v` with
/// `from <= v < to`. Pure interval algebra; carries no epistemic policy (that
/// is [`TemporalPov`]'s `rung`, and ultimately `EpistemicMode` on the planner
/// side).
///
/// # Examples
///
/// ```
/// use lance_graph_contract::temporal_pov::VersionRange;
///
/// let r = VersionRange::new(10, 20);
/// assert!(r.contains(10));
/// assert!(!r.contains(20)); // half-open: `to` is excluded
/// assert_eq!(r.len(), 10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VersionRange {
    /// Inclusive lower bound.
    pub from: LanceVersion,
    /// Exclusive upper bound.
    pub to: LanceVersion,
}

impl VersionRange {
    /// Build the half-open range `[from, to)`. `from >= to` is a valid,
    /// representable **empty** range (see [`is_empty`](Self::is_empty)) — no
    /// panic, no `Option` wrapper, matching the workspace's "zero means fall
    /// through" ladder convention rather than an error path.
    #[inline]
    #[must_use]
    pub const fn new(from: LanceVersion, to: LanceVersion) -> Self {
        Self { from, to }
    }

    /// The unbounded range `[0, u64::MAX)` — the practical "every version ever
    /// written" window. Mirrors `QueryReference::default()`'s `ref_version:
    /// u64::MAX` "latest" sentinel (temporal.rs:126–137), with one documented
    /// edge case: because this is a **half-open** interval, `contains` excludes
    /// the literal value `u64::MAX` itself (no real Lance version reaches the
    /// `u64` ceiling, so this is a theoretical, not practical, gap — see the
    /// `temporal_pov_at_latest_sentinel_edge_case` test below).
    #[inline]
    #[must_use]
    pub const fn full() -> Self {
        Self::new(0, LanceVersion::MAX)
    }

    /// Whether the range admits no version at all (`from >= to`).
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.from >= self.to
    }

    /// Number of versions the range admits (`0` when [`is_empty`](Self::is_empty)).
    #[inline]
    #[must_use]
    pub const fn len(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            self.to - self.from
        }
    }

    /// Whether `v` falls in `[from, to)`.
    #[inline]
    #[must_use]
    pub const fn contains(&self, v: LanceVersion) -> bool {
        self.from <= v && v < self.to
    }

    /// The overlap of two ranges — `[max(from), min(to))`. May be empty (see
    /// [`is_empty`](Self::is_empty)); never panics (unlike a checked-subtraction
    /// `Range` intersection, an "empty" result here is just `from >= to`, not a
    /// distinguished `None`/panic case).
    #[inline]
    #[must_use]
    pub const fn intersect(&self, other: &Self) -> Self {
        let from = if self.from > other.from {
            self.from
        } else {
            other.from
        };
        let to = if self.to < other.to {
            self.to
        } else {
            other.to
        };
        Self::new(from, to)
    }
}

/// A reader's temporal point-of-view: an admitted [`VersionRange`] plus the
/// reader's `rung` (mirrors `QueryReference::rung`, temporal.rs:122–124, which
/// on the planner side drives `EpistemicMode::for_rung`). This is the
/// consumer-facing filter SHAPE — see the module docs for why the full
/// `EpistemicMode`/`TemporalStatus` classification is deliberately not
/// reimplemented here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporalPov {
    /// The admitted version window.
    pub range: VersionRange,
    /// The reader's rung (`0..=255`; planner-side `EpistemicMode::for_rung`
    /// partitions this into `Strict` (0–4) / `Aware` (5–8) / `Retro` (9+),
    /// temporal.rs:67–73). Carried here as plain data — mode derivation stays
    /// on the planner side.
    pub rung: u8,
}

impl TemporalPov {
    /// Build a point-of-view over an explicit range and rung.
    #[inline]
    #[must_use]
    pub const fn new(range: VersionRange, rung: u8) -> Self {
        Self { range, rung }
    }

    /// Mirrors `QueryReference::at(ref_version, rung)` (temporal.rs:139–151):
    /// a reader pinned at `ref_version`, admitting exactly the contemporary
    /// window `row_version <= ref_version` — expressed here as the half-open
    /// range `[0, ref_version + 1)`. `ref_version == u64::MAX` (the "latest"
    /// sentinel) saturates to [`VersionRange::full`]'s half-open ceiling —
    /// see the same documented edge case as [`VersionRange::full`].
    #[inline]
    #[must_use]
    pub const fn at(ref_version: LanceVersion, rung: u8) -> Self {
        let to = match ref_version.checked_add(1) {
            Some(to) => to,
            None => LanceVersion::MAX,
        };
        Self::new(VersionRange::new(0, to), rung)
    }

    /// Whether `version` falls within this point-of-view's admitted window.
    /// This is the version-range half of admission only — quoting
    /// `EpistemicMode::Strict`'s doc (temporal.rs:53–55): "Only `CONTEMPORARY`
    /// rows (`row_version ≤ ref_version`)". Per-row `knowable_from` /
    /// `Spoiler`-vs-`Anachronistic` classification is the planner's `classify`,
    /// not this filter.
    #[inline]
    #[must_use]
    pub const fn admits(&self, version: LanceVersion) -> bool {
        self.range.contains(version)
    }
}

/// Why [`VersionClock::record`] refused a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockError {
    /// The version is not newer than the clock's head.
    NotNewer {
        /// The clock's head.
        head: LanceVersion,
        /// The refused version.
        version: LanceVersion,
    },
    /// The row count went down: an append-only lane never loses rows.
    RowsWentBack {
        /// Rows at the head.
        head_rows: u64,
        /// The refused count.
        rows: u64,
    },
}

impl core::fmt::Display for ClockError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotNewer { head, version } => {
                write!(f, "version {version} is not newer than head {head}")
            }
            Self::RowsWentBack { head_rows, rows } => {
                write!(f, "row count {rows} is below the head's {head_rows}")
            }
        }
    }
}

impl std::error::Error for ClockError {}

/// The version clock of one append-only lane: which rows each version holds.
///
/// Rows are numbered in append order. Version `v` holds rows
/// `0..rows_through(v)`, so a [`TemporalPov`] resolves to one half-open row
/// range by two array reads. No manifest, no per-version file list and no
/// per-row predicate: eight bytes per version, and the rows themselves stay
/// where they are.
///
/// The tick is any monotone durable coordinate: a Lance version for a
/// one-version-per-commit writer, or a WAL entry position for a MemWAL
/// writer, whose base table does not advance per commit.
///
/// The clock knows versions from its `base` on. A clock seeded on an
/// existing store at its head cannot answer for versions before that head;
/// those reads return `None`, never a guess.
///
/// Rewriting rows (compaction that reorders, deletion) is out of scope:
/// a deleted row stays inside its range and must be masked by the caller;
/// a reordering rewrite needs a new clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionClock {
    base: LanceVersion,
    /// `ends[i]` = rows durable through version `base + i`. Non-decreasing,
    /// never empty.
    ends: Vec<u64>,
}

impl VersionClock {
    /// A clock that knows `base` and its row count.
    #[must_use]
    pub fn new(base: LanceVersion, rows_at_base: u64) -> Self {
        Self {
            base,
            ends: vec![rows_at_base],
        }
    }

    /// The first version this clock can answer for.
    #[inline]
    #[must_use]
    pub fn base(&self) -> LanceVersion {
        self.base
    }

    /// The newest recorded version.
    #[inline]
    #[must_use]
    pub fn head(&self) -> LanceVersion {
        self.base + (self.ends.len() as u64 - 1)
    }

    /// Rows durable at the head.
    #[inline]
    #[must_use]
    pub fn head_rows(&self) -> u64 {
        // `ends` is never empty.
        self.ends[self.ends.len() - 1]
    }

    /// Record that `version` is durable with `rows` rows in total. Versions
    /// skipped between the head and `version` added no rows to this lane
    /// (another operation minted them) and inherit the head's count.
    pub fn record(&mut self, version: LanceVersion, rows: u64) -> Result<(), ClockError> {
        let head = self.head();
        if version <= head {
            return Err(ClockError::NotNewer { head, version });
        }
        let head_rows = self.head_rows();
        if rows < head_rows {
            return Err(ClockError::RowsWentBack { head_rows, rows });
        }
        let gap = (version - head - 1) as usize;
        self.ends.extend(core::iter::repeat_n(head_rows, gap));
        self.ends.push(rows);
        Ok(())
    }

    /// Rows durable through `version`: `0..rows_through(version)`. A version
    /// past the head holds the head's rows (nothing newer exists yet).
    /// `None` for a version before [`Self::base`].
    #[inline]
    #[must_use]
    pub fn rows_through(&self, version: LanceVersion) -> Option<u64> {
        if version < self.base {
            return None;
        }
        let i = (version - self.base).min(self.ends.len() as u64 - 1) as usize;
        Some(self.ends[i])
    }

    /// The rows a point of view admits: the rows that landed in versions
    /// `pov.range.from .. pov.range.to`, as a half-open row range. For
    /// [`TemporalPov::at`]`(v, _)` this is `0..rows_through(v)`.
    #[inline]
    #[must_use]
    pub fn rows_at(&self, pov: &TemporalPov) -> Option<core::ops::Range<u64>> {
        let VersionRange { from, to } = pov.range;
        if from >= to {
            return Some(0..0);
        }
        let start = if from == 0 {
            0
        } else {
            self.rows_through(from - 1)?
        };
        let end = self.rows_through(to - 1)?;
        Some(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> VersionClock {
        // v0: 0 rows, v1: 10, v2: 10 (no rows, e.g. an index op), v3: 25.
        let mut c = VersionClock::new(0, 0);
        c.record(1, 10).unwrap();
        c.record(3, 25).unwrap();
        c
    }

    #[test]
    fn clock_answers_at_and_ranges() {
        let c = clock();
        assert_eq!(c.head(), 3);
        let at = |v| c.rows_at(&TemporalPov::at(v, 0)).unwrap();
        assert_eq!(at(0), 0..0);
        assert_eq!(at(1), 0..10);
        assert_eq!(at(2), 0..10, "the skipped version inherits the head's rows");
        assert_eq!(at(3), 0..25);
        assert_eq!(at(99), 0..25, "a version past the head holds the head's rows");
        assert_eq!(at(u64::MAX), 0..25, "the latest sentinel saturates");
        // Rows landed in versions 2..=3 only.
        let r = c.rows_at(&TemporalPov::new(VersionRange::new(2, 4), 0)).unwrap();
        assert_eq!(r, 10..25);
        let empty = c.rows_at(&TemporalPov::new(VersionRange::new(4, 4), 0)).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn clock_agrees_with_a_per_row_scan() {
        // Oracle: every row carries the version it landed at; the clock range
        // must equal the set of rows the per-row predicate admits.
        let mut c = VersionClock::new(0, 0);
        let mut row_version = Vec::new();
        let mut v = 0u64;
        for step in 1..200u64 {
            v += 1 + step % 3; // gaps of 0..2 skipped versions
            let added = (step * 7) % 5; // some versions add no rows
            for _ in 0..added {
                row_version.push(v);
            }
            c.record(v, row_version.len() as u64).unwrap();
        }
        for from in 0..=v + 2 {
            for to in [from, from + 1, from + 3, v / 2, v + 5] {
                let pov = TemporalPov::new(VersionRange::new(from, to), 0);
                let want: Vec<u64> = (0..row_version.len() as u64)
                    .filter(|&i| pov.admits(row_version[i as usize]))
                    .collect();
                let got: Vec<u64> = c.rows_at(&pov).unwrap().collect();
                assert_eq!(got, want, "from {from} to {to}");
            }
        }
    }

    #[test]
    fn clock_refuses_old_versions_and_shrinking_rows() {
        let mut c = clock();
        assert_eq!(
            c.record(3, 30),
            Err(ClockError::NotNewer { head: 3, version: 3 })
        );
        assert_eq!(
            c.record(4, 24),
            Err(ClockError::RowsWentBack { head_rows: 25, rows: 24 })
        );
        assert_eq!(c.head(), 3, "a refused record changes nothing");
        assert!(c.record(4, 25).is_ok(), "an equal count is a version with no rows");
    }

    #[test]
    fn clock_seeded_at_a_head_does_not_answer_for_earlier_versions() {
        let c = VersionClock::new(100, 5_000);
        assert_eq!(c.rows_through(99), None);
        assert_eq!(c.rows_at(&TemporalPov::at(99, 0)), None);
        assert_eq!(c.rows_at(&TemporalPov::at(100, 0)), Some(0..5_000));
        // A range starting after the base needs rows_through(from - 1).
        assert_eq!(
            c.rows_at(&TemporalPov::new(VersionRange::new(101, 102), 0)),
            Some(5_000..5_000)
        );
        assert_eq!(c.rows_at(&TemporalPov::new(VersionRange::new(100, 101), 0)), None);
    }

    #[test]
    fn version_range_basic_contains_and_len() {
        let r = VersionRange::new(10, 20);
        assert!(r.contains(10));
        assert!(r.contains(19));
        assert!(!r.contains(20), "half-open: `to` is excluded");
        assert!(!r.contains(9));
        assert_eq!(r.len(), 10);
        assert!(!r.is_empty());
    }

    #[test]
    fn version_range_empty_when_from_ge_to() {
        let empty_equal = VersionRange::new(5, 5);
        let empty_inverted = VersionRange::new(5, 3);
        for r in [empty_equal, empty_inverted] {
            assert!(r.is_empty());
            assert_eq!(r.len(), 0);
            assert!(!r.contains(r.from));
            assert!(!r.contains(0));
            assert!(!r.contains(u64::MAX));
        }
    }

    #[test]
    fn version_range_full_covers_practical_range() {
        let full = VersionRange::full();
        assert!(!full.is_empty());
        assert!(full.contains(0));
        assert!(full.contains(1));
        assert!(full.contains(u64::MAX - 1));
        // Documented edge case: half-open excludes the literal ceiling itself.
        assert!(!full.contains(u64::MAX));
    }

    #[test]
    fn version_range_intersect_overlapping() {
        let a = VersionRange::new(0, 100);
        let b = VersionRange::new(50, 150);
        let i = a.intersect(&b);
        assert_eq!(i, VersionRange::new(50, 100));
        // Intersection is commutative.
        assert_eq!(i, b.intersect(&a));
    }

    #[test]
    fn version_range_intersect_disjoint_is_empty() {
        let a = VersionRange::new(0, 10);
        let b = VersionRange::new(20, 30);
        assert!(a.intersect(&b).is_empty());
        assert!(b.intersect(&a).is_empty());
    }

    #[test]
    fn version_range_intersect_one_contains_other() {
        let outer = VersionRange::new(0, 100);
        let inner = VersionRange::new(30, 40);
        assert_eq!(outer.intersect(&inner), inner);
        assert_eq!(inner.intersect(&outer), inner);
    }

    #[test]
    fn version_range_intersect_touching_ranges_is_empty() {
        // [0,10) and [10,20) share no version — the shared boundary is
        // excluded on both sides by the half-open convention.
        let a = VersionRange::new(0, 10);
        let b = VersionRange::new(10, 20);
        assert!(a.intersect(&b).is_empty());
    }

    #[test]
    fn temporal_pov_admits_within_range() {
        let pov = TemporalPov::new(VersionRange::new(10, 20), 0);
        assert!(pov.admits(10));
        assert!(pov.admits(15));
        assert!(!pov.admits(20));
        assert!(!pov.admits(9));
    }

    #[test]
    fn temporal_pov_at_pins_contemporary_window() {
        // Mirrors temporal.rs `classify_time_axis`: a reader `at(100, 0)`
        // admits row_version <= 100 (Contemporary), not row_version > 100
        // (Anachronistic under Strict).
        let pov = TemporalPov::at(100, 0);
        assert!(pov.admits(0));
        assert!(pov.admits(50));
        assert!(pov.admits(100), "ref_version itself must be admitted");
        assert!(!pov.admits(101), "a future-frame row must not be admitted");
    }

    #[test]
    fn temporal_pov_at_latest_sentinel_edge_case() {
        // ref_version == u64::MAX (the QueryReference::default() "latest"
        // sentinel, temporal.rs:126-137) saturates rather than overflowing.
        let pov = TemporalPov::at(u64::MAX, 0);
        assert!(pov.admits(0));
        assert!(pov.admits(u64::MAX - 1));
        // Documented half-open edge case — see VersionRange::full's doc.
        assert!(!pov.admits(u64::MAX));
    }

    #[test]
    fn temporal_pov_rung_boundaries_roundtrip() {
        // Rung is carried verbatim (mode derivation is planner-side); pin the
        // EpistemicMode::for_rung boundary values from temporal.rs:67-73 so a
        // future width change here is caught, without reimplementing the enum.
        for rung in [0u8, 4, 5, 8, 9, 255] {
            let pov = TemporalPov::at(1, rung);
            assert_eq!(pov.rung, rung);
        }
    }
}
