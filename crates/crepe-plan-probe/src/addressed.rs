//! Arm B: the addressed semijoin (plan §2e).
//!
//! A relation's rows are stored in key order: row `(a, b)` carries the key
//! `FacetCascade::from_semantic_tiles([0, 0, a_hi, a_lo, b_hi, b_lo, 0, 0])`,
//! default class (no concept is minted for a probe relation), and the table is
//! written in numeric projection order. The order is attested, never produced
//! here: [`attest`] reads the keys once and hands them to
//! `SealedFacetLane::attest_sorted`, which refuses an unsorted lane.
//!
//! The rows of one leading value `a` are then a contiguous range, located by
//! `SealedFacetLane::seek_from` under the lane's `OrderedLaneWitness`. A side
//! is read with `with_row_addr_prefilter` over the union of its ranges, so the
//! rows Lance visits are the rows of the demanded keys and nothing else. No
//! value is translated to another coordinate space: the addresses are the
//! lane's own positions, in NodeGuid order.
//!
//! What this arm refuses, by name:
//! - a side whose restriction or lookup is on the second column (the lane is
//!   ordered by the first; the reverse order is a second lane, not built). A
//!   side read in full, with no centers to look up, may be keyed either way;
//! - a lane attested on another dataset (`OtherDataset`);
//! - a lane whose table moved on since it was attested (`Stale`);
//! - a table whose rows are not in key order (`NotSorted`, at attest);
//! - a table that is not one contiguous fragment (`NotContiguous`, at attest).

use std::sync::Arc;

use arrow_array::{Array, UInt32Array, UInt64Array};
use lance::dataset::Dataset;
use lance::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
use lance_graph_contract::facet::{FacetCascade, SemanticPrefix};
use lance_graph_contract::ordered_lane::{OrderedLaneWitness, SealedFacetLane, WitnessError};

use crate::lance_store::{Io, LanceStore, LanceStoreError, SideIo, read_rows};
use crate::logic::RelId;
use crate::native::{PreparedStar, StarPlan, prepare_from_rows};

/// The key of row `(a, b)`.
pub fn key(a: u32, b: u32) -> FacetCascade {
    let hi = |v: u32| (v >> 16) as u16;
    let lo = |v: u32| v as u16;
    FacetCascade::from_semantic_tiles([0, 0, hi(a), lo(a), hi(b), lo(b), 0, 0])
}

/// The prefix that fixes the leading value `a` (class tiles and `a`).
pub fn leading(a: u32) -> SemanticPrefix {
    SemanticPrefix::of(key(a, 0), 4)
}

#[derive(Debug)]
pub enum AddressedRefusal {
    Lance(LanceStoreError),
    Witness(WitnessError),
    /// The table is not one contiguous fragment, so lane position `i` is not
    /// row address `i`.
    NotContiguous {
        rel: RelId,
    },
    /// The table changed since its lane was attested.
    Stale {
        rel: RelId,
        attested: u64,
        current: u64,
    },
    /// No lane was attested for this relation.
    NoLane {
        rel: RelId,
    },
    /// The lane was attested on another dataset.
    OtherDataset {
        rel: RelId,
        attested: String,
        current: String,
    },
    /// The side would need the lane ordered by its second column.
    NotLeadingColumn {
        side: usize,
    },
}

impl From<LanceStoreError> for AddressedRefusal {
    fn from(e: LanceStoreError) -> Self {
        AddressedRefusal::Lance(e)
    }
}

/// One relation's attested lane.
pub struct AddressedRelation {
    pub rel: RelId,
    /// The dataset the lane was read from. A lane is only valid against the
    /// table it was attested on, not another table with the same relation
    /// id and version.
    pub uri: String,
    pub lane: SealedFacetLane,
    pub witness: OrderedLaneWitness,
}

/// Reads a relation's keys once and attests their order at the table's
/// current version. This is the cold cost `P` of the arm.
pub async fn attest(store: &LanceStore, rel: RelId) -> Result<AddressedRelation, AddressedRefusal> {
    let ds = store.dataset(rel).await?;
    let (keys, contiguous) = read_keys(&ds).await?;
    if !contiguous {
        return Err(AddressedRefusal::NotContiguous { rel });
    }
    let lane = SealedFacetLane::attest_sorted(keys, ds.version().version)
        .map_err(AddressedRefusal::Witness)?;
    let witness = lane.witness();
    Ok(AddressedRelation {
        rel,
        uri: ds.uri().to_string(),
        lane,
        witness,
    })
}

async fn read_keys(ds: &Dataset) -> Result<(Vec<FacetCascade>, bool), LanceStoreError> {
    let mut s = ds.scan();
    s.project(&["c0", "c1"])
        .map_err(|e| LanceStoreError::Lance(e.to_string()))?;
    s.with_row_address();
    let b = s
        .try_into_batch()
        .await
        .map_err(|e| LanceStoreError::Lance(e.to_string()))?;
    let col = |n: &str| {
        b.column_by_name(n)
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
            .expect("u32 column")
            .clone()
    };
    let (c0, c1) = (col("c0"), col("c1"));
    let addr = b
        .column_by_name("_rowaddr")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .expect("_rowaddr");
    let contiguous = (0..b.num_rows()).all(|i| addr.value(i) == i as u64);
    let keys = (0..b.num_rows())
        .map(|i| key(c0.value(i), c1.value(i)))
        .collect();
    Ok((keys, contiguous))
}

/// What the addressed arm did on one side, beyond Lance's counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lookup {
    /// Distinct leading values sought.
    pub keys: usize,
    /// Rows inside the located ranges.
    pub rows_in_ranges: usize,
    /// How the side's rows were read.
    pub read: Read,
    /// Ranges handed to Lance (0 for a full read).
    pub ranges_read: usize,
}

/// How a side's rows are read once its lane ranges are located.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Read {
    /// Exactly the located ranges.
    #[default]
    Ranges,
    /// The located ranges with short gaps between them read too. The extra
    /// rows are dropped after the read by their own leading value.
    Coalesced,
    /// The whole table without a row filter, then the same drop.
    Full,
}

/// When a side stops reading exact ranges. Many short ranges cost Lance more
/// per row than one sequential read (§2e: from about 10% of centers every
/// masked read loses to the full one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadPolicy {
    /// Two ranges are merged when at most this many rows lie between them.
    pub coalesce_gap: u32,
    /// The table is read in full when the (merged) ranges cover at least this
    /// many rows per 1,000 of the table. Above 1,000, never.
    pub full_permille: u32,
}

impl ReadPolicy {
    /// Exact ranges only: Arm B as first measured.
    pub const EXACT: Self = Self {
        coalesce_gap: 0,
        full_permille: u32::MAX,
    };
    /// The measured switch (`examples/addressed_read.rs`, release, uniform
    /// demand, policy pins rather than derived values):
    /// - Full read at 10% coverage. The crossover moved with table size: at
    ///   50,000 rows exact ranges win up to about 3% and lose from about 10%;
    ///   at 200,000 rows they win up to about 5% and tie at 7–10%. Between
    ///   the two, either choice is within run-to-run noise; above 10% the
    ///   full read is up to 1.9× faster (50% of centers).
    /// - A coalescing gap of 16 rows measured neutral on uniform demand. A
    ///   gap of 64 lost (11.7 vs 6.9 ms at 1% of 200,000 rows). Clustered
    ///   demand, where coalescing should help, is not measured.
    pub const DEFAULT: Self = Self {
        coalesce_gap: 16,
        full_permille: 100,
    };
    /// Always the whole table.
    pub const FULL: Self = Self {
        coalesce_gap: 0,
        full_permille: 0,
    };
}

/// Decides how to read `ranges` (sorted, disjoint, inside `0..n_rows`) and
/// returns the ranges to hand to Lance (empty for [`Read::Full`]). The rule
/// reads only the lane positions, never a row: it is part of preparation.
pub fn plan_read(ranges: &[(u32, u32)], n_rows: u32, p: ReadPolicy) -> (Read, Vec<(u32, u32)>) {
    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for &(lo, hi) in ranges {
        match merged.last_mut() {
            Some(last) if lo - last.1 <= p.coalesce_gap => last.1 = hi,
            _ => merged.push((lo, hi)),
        }
    }
    let span = |r: &[(u32, u32)]| -> u64 { r.iter().map(|&(lo, hi)| u64::from(hi - lo)).sum() };
    let covered = span(&merged);
    if n_rows > 0 && covered * 1000 >= u64::from(n_rows) * u64::from(p.full_permille) {
        return (Read::Full, Vec::new());
    }
    // Merging ranges that touch reads no extra row: still exact.
    let read = if covered > span(ranges) {
        Read::Coalesced
    } else {
        Read::Ranges
    };
    (read, merged)
}

/// The rows of every leading value in `wanted` (sorted, deduplicated) as lane
/// ranges, found with galloping seeks from the previous range's end.
fn ranges(
    a: &AddressedRelation,
    wanted: &[u32],
) -> Result<(Vec<(u32, u32)>, usize), AddressedRefusal> {
    let mut out = Vec::with_capacity(wanted.len());
    let mut from = 0u32;
    let mut rows = 0usize;
    for &v in wanted {
        let lo = a
            .lane
            .seek_from(&a.witness, from, &leading(v).lo_key())
            .map_err(AddressedRefusal::Witness)?;
        let hi = match v.checked_add(1) {
            Some(next) => a
                .lane
                .seek_from(&a.witness, lo, &leading(next).lo_key())
                .map_err(AddressedRefusal::Witness)?,
            None => a.lane.n_rows(),
        };
        if hi > lo {
            out.push((lo, hi));
            rows += (hi - lo) as usize;
        }
        from = hi;
    }
    Ok((out, rows))
}

fn range_mask(ranges: &[(u32, u32)]) -> RowAddrMask {
    let mut m = RowAddrTreeMap::new();
    for &(lo, hi) in ranges {
        m.insert_range(u64::from(lo)..u64::from(hi));
    }
    RowAddrMask::from_allowed(m)
}

fn lane_for(
    lanes: &[AddressedRelation],
    rel: RelId,
) -> Result<&AddressedRelation, AddressedRefusal> {
    lanes
        .iter()
        .find(|l| l.rel == rel)
        .ok_or(AddressedRefusal::NoLane { rel })
}

/// The side's attested lane and dataset, refused when the table moved on.
async fn current_lane<'a>(
    plan: &StarPlan,
    i: usize,
    store: &LanceStore,
    lanes: &'a [AddressedRelation],
) -> Result<(&'a AddressedRelation, Arc<Dataset>), AddressedRefusal> {
    let rel = plan.sides[i].key.rel;
    let a = lane_for(lanes, rel)?;
    let ds = store.dataset(rel).await?;
    if ds.uri() != a.uri {
        return Err(AddressedRefusal::OtherDataset {
            rel,
            attested: a.uri.clone(),
            current: ds.uri().to_string(),
        });
    }
    let current = ds.version().version;
    if current != a.lane.version() {
        return Err(AddressedRefusal::Stale {
            rel,
            attested: a.lane.version(),
            current,
        });
    }
    Ok((a, ds))
}

/// Reads one side's rows for the located `ranges` of the leading values in
/// `wanted` (sorted, deduplicated), as `policy` decides. A coalesced or full
/// read returns rows of other leading values too; they are dropped by the
/// row's own leading value, so the result equals the exact read.
async fn read_side(
    ds: &Dataset,
    a: &AddressedRelation,
    ranges: &[(u32, u32)],
    wanted: &[u32],
    policy: ReadPolicy,
) -> Result<(Read, usize, Vec<[u32; 2]>, Io), AddressedRefusal> {
    let (how, r) = plan_read(ranges, a.lane.n_rows(), policy);
    let mask = match how {
        Read::Full => None,
        Read::Ranges | Read::Coalesced => Some(range_mask(&r)),
    };
    let (mut got, rows_io) = read_rows(ds, mask).await?;
    if how != Read::Ranges {
        got.retain(|row| wanted.binary_search(&row[0]).is_ok());
    }
    Ok((how, r.len(), got, rows_io))
}

/// [`prepare_addressed_with`] under [`ReadPolicy::EXACT`].
pub async fn prepare_addressed(
    plan: &StarPlan,
    store: &LanceStore,
    lanes: &[AddressedRelation],
) -> Result<(PreparedStar, Vec<SideIo>, Vec<Lookup>), AddressedRefusal> {
    prepare_addressed_with(plan, store, lanes, ReadPolicy::EXACT).await
}

/// Prepares `plan` reading only the rows of the demanded keys: restricted
/// sides by their constant, the others by the centers every restricted side
/// has. Without a restricted side every lane is read in full (the arm has no
/// demand to address by). `policy` decides per side whether the located
/// ranges are read exactly, coalesced, or replaced by a full read.
pub async fn prepare_addressed_with(
    plan: &StarPlan,
    store: &LanceStore,
    lanes: &[AddressedRelation],
    policy: ReadPolicy,
) -> Result<(PreparedStar, Vec<SideIo>, Vec<Lookup>), AddressedRefusal> {
    let n_sides = plan.sides.len();
    let mut rows: Vec<Option<(Vec<[u32; 2]>, u64)>> = vec![None; n_sides];
    let mut io = vec![SideIo::default(); n_sides];
    let mut lookups = vec![Lookup::default(); n_sides];
    let restricted = |i: usize| {
        let k = plan.sides[i].key;
        (!k.unary).then_some(k.restrict).flatten()
    };

    for i in (0..n_sides).filter(|&i| restricted(i).is_some()) {
        let key = plan.sides[i].key;
        // The constant must be on the lane's leading column, so the center is
        // the second column.
        if key.key_col != 1 {
            return Err(AddressedRefusal::NotLeadingColumn { side: i });
        }
        let (a, ds) = current_lane(plan, i, store, lanes).await?;
        // A constant above `u32::MAX` matches no stored value: no range.
        let (r, n) = match u32::try_from(restricted(i).expect("restricted")) {
            Ok(k) => ranges(a, &[k])?,
            Err(_) => (Vec::new(), 0),
        };
        let wanted: Vec<u32> = u32::try_from(restricted(i).expect("restricted"))
            .into_iter()
            .collect();
        let (how, read_ranges, got, rows_io) = read_side(&ds, a, &r, &wanted, policy).await?;
        lookups[i] = Lookup {
            keys: 1,
            rows_in_ranges: n,
            read: how,
            ranges_read: read_ranges,
        };
        io[i] = SideIo {
            mask_column: None,
            mask_rows: Some(n),
            rows: rows_io,
        };
        rows[i] = Some((got, a.lane.version()));
    }

    let mut centers: Option<Vec<u32>> = None;
    for i in (0..n_sides).filter(|&i| restricted(i).is_some()) {
        let mut c: Vec<u32> = rows[i]
            .as_ref()
            .expect("read")
            .0
            .iter()
            .map(|r| r[1])
            .collect();
        c.sort_unstable();
        c.dedup();
        centers = Some(match centers {
            None => c,
            Some(prev) => prev
                .into_iter()
                .filter(|v| c.binary_search(v).is_ok())
                .collect(),
        });
    }

    // Restricted sides keep only the intersected centers (rows already read).
    if let Some(c) = &centers {
        for i in (0..n_sides).filter(|&i| restricted(i).is_some()) {
            if let Some((r, _)) = rows[i].as_mut() {
                r.retain(|row| c.binary_search(&row[1]).is_ok());
            }
        }
    }

    for i in (0..n_sides).filter(|&i| restricted(i).is_none()) {
        let key = plan.sides[i].key;
        let (a, ds) = current_lane(plan, i, store, lanes).await?;
        let (n, sought, how, read_ranges, got, rows_io) = match &centers {
            // Without centers the side is read in full, so its key column
            // may be either; only a lookup needs the leading column.
            Some(_) if key.key_col != 0 => {
                return Err(AddressedRefusal::NotLeadingColumn { side: i });
            }
            Some(c) => {
                let (r, n) = ranges(a, c)?;
                let (how, rr, got, rows_io) = read_side(&ds, a, &r, c, policy).await?;
                (n, c.len(), how, rr, got, rows_io)
            }
            None => {
                let (got, rows_io) = read_rows(&ds, None).await?;
                (a.lane.n_rows() as usize, 0, Read::Full, 0, got, rows_io)
            }
        };
        lookups[i] = Lookup {
            keys: sought,
            rows_in_ranges: n,
            read: how,
            ranges_read: read_ranges,
        };
        io[i] = SideIo {
            mask_column: None,
            mask_rows: centers.as_ref().map(|_| n),
            rows: rows_io,
        };
        rows[i] = Some((got, a.lane.version()));
    }

    let owned = rows
        .into_iter()
        .map(|r| r.expect("every side read"))
        .collect();
    Ok((prepare_from_rows(plan, owned), io, lookups))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_FULL: u32 = u32::MAX;

    fn policy(coalesce_gap: u32, full_permille: u32) -> ReadPolicy {
        ReadPolicy {
            coalesce_gap,
            full_permille,
        }
    }

    #[test]
    fn a_gap_of_exactly_the_limit_merges_and_one_more_does_not() {
        let r = [(0, 2), (5, 6)]; // gap of 3 rows
        assert_eq!(
            plan_read(&r, 1_000, policy(3, NO_FULL)),
            (Read::Coalesced, vec![(0, 6)])
        );
        assert_eq!(
            plan_read(&r, 1_000, policy(2, NO_FULL)),
            (Read::Ranges, r.to_vec())
        );
    }

    #[test]
    fn touching_ranges_merge_but_stay_exact() {
        let r = [(0, 2), (2, 4), (9, 10)];
        assert_eq!(
            plan_read(&r, 1_000, ReadPolicy::EXACT),
            (Read::Ranges, vec![(0, 4), (9, 10)])
        );
    }

    #[test]
    fn the_full_read_fires_at_the_threshold_and_not_one_row_below() {
        // 1,000 rows, 100 permille: 100 covered rows switch, 99 do not.
        assert_eq!(plan_read(&[(0, 100)], 1_000, policy(0, 100)).0, Read::Full);
        assert_eq!(plan_read(&[(0, 99)], 1_000, policy(0, 100)).0, Read::Ranges);
    }

    #[test]
    fn the_full_read_counts_the_rows_coalescing_adds() {
        // 50 exact rows in 50 one-row ranges two apart: 99 rows once merged.
        let r: Vec<(u32, u32)> = (0..50).map(|i| (2 * i, 2 * i + 1)).collect();
        assert_eq!(plan_read(&r, 1_000, policy(0, 99)).0, Read::Ranges);
        assert_eq!(plan_read(&r, 1_000, policy(1, 99)).0, Read::Full);
    }

    #[test]
    fn exact_never_reads_in_full_and_full_always_does() {
        assert_eq!(
            plan_read(&[(0, 1_000)], 1_000, ReadPolicy::EXACT).0,
            Read::Ranges
        );
        assert_eq!(plan_read(&[], 1_000, ReadPolicy::FULL).0, Read::Full);
        // An empty table is never read "in full": there is nothing to read.
        assert_eq!(plan_read(&[], 0, ReadPolicy::FULL).0, Read::Ranges);
    }
}
