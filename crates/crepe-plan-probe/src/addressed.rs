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
//! - a side whose restriction or demand is on the second column (the lane is
//!   ordered by the first; the reverse order is a second lane, not built);
//! - a lane whose table moved on since it was attested (`Stale`);
//! - a table whose rows are not in key order (`NotSorted`, at attest);
//! - a table that is not one contiguous fragment (`NotContiguous`, at attest).

use std::sync::Arc;

use arrow_array::{Array, UInt32Array, UInt64Array};
use lance::dataset::Dataset;
use lance::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
use lance_graph_contract::facet::{FacetCascade, SemanticPrefix};
use lance_graph_contract::ordered_lane::{OrderedLaneWitness, SealedFacetLane, WitnessError};

use crate::lance_store::{LanceStore, LanceStoreError, SideIo, read_rows};
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
    Ok(AddressedRelation { rel, lane, witness })
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

/// Prepares `plan` reading only the rows of the demanded keys: restricted
/// sides by their constant, the others by the centers every restricted side
/// has. Without a restricted side every lane is read in full (the arm has no
/// demand to address by).
pub async fn prepare_addressed(
    plan: &StarPlan,
    store: &LanceStore,
    lanes: &[AddressedRelation],
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
        let k = u32::try_from(restricted(i).expect("restricted"))
            .map_err(|_| AddressedRefusal::NotLeadingColumn { side: i })?;
        let (a, ds) = current_lane(plan, i, store, lanes).await?;
        let (r, n) = ranges(a, &[k])?;
        let (got, rows_io) = read_rows(&ds, Some(range_mask(&r))).await?;
        lookups[i] = Lookup {
            keys: 1,
            rows_in_ranges: n,
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
        if key.key_col != 0 {
            return Err(AddressedRefusal::NotLeadingColumn { side: i });
        }
        let (a, ds) = current_lane(plan, i, store, lanes).await?;
        let (mask, n, sought) = match &centers {
            Some(c) => {
                let (r, n) = ranges(a, c)?;
                (Some(range_mask(&r)), n, c.len())
            }
            None => (None, a.lane.n_rows() as usize, 0),
        };
        let (got, rows_io) = read_rows(&ds, mask).await?;
        lookups[i] = Lookup {
            keys: sought,
            rows_in_ranges: n,
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
