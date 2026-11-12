//! The source owner: N rows at one version. It stands in for storage.
//!
//! Every method that produces a selection or stages rows bumps
//! [`Counters::storage`]. Nothing else in the crate may read these lanes.

use crate::bind::Order;
use crate::counters::Counters;

/// The storage version the lanes and the order index were read at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version(pub u64);

/// An inclusive key range, the pushed predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRange {
    pub lo: i32,
    pub hi: i32,
}

impl KeyRange {
    pub fn has(&self, key: i32) -> bool {
        self.lo <= key && key <= self.hi
    }
}

/// What the source hands over: the surviving source rows, the order they
/// are in, and the version they are valid for. Exact, never a candidate set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub version: Version,
    pub rows: Vec<u32>,
    pub order: Order,
}

/// K rows copied out of the source, in the order of the selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Staged {
    pub guid: Vec<u128>,
    pub key: Vec<i32>,
    pub a: Vec<i32>,
    pub b: Vec<i32>,
}

/// N rows, four lanes, and one order index per declared tie rule.
pub struct Source {
    version: Version,
    guid: Vec<u128>,
    key: Vec<i32>,
    a: Vec<i32>,
    b: Vec<i32>,
    /// Index position -> source row, key ascending, ties by source row.
    by_key_row: Vec<u32>,
    /// Index position -> source row, key ascending, ties by guid.
    by_key_guid: Vec<u32>,
}

impl Source {
    /// Takes ownership of the lanes and sorts each order index once. Every
    /// lane has the same length; guids are unique.
    pub fn new(
        version: Version,
        guid: Vec<u128>,
        key: Vec<i32>,
        a: Vec<i32>,
        b: Vec<i32>,
    ) -> Source {
        assert!(guid.len() == key.len() && key.len() == a.len() && a.len() == b.len());
        assert!(u32::try_from(guid.len()).is_ok());
        let mut by_key_row: Vec<u32> = (0..guid.len() as u32).collect();
        by_key_row.sort_unstable_by_key(|&r| (key[r as usize], r));
        let mut by_key_guid = by_key_row.clone();
        by_key_guid.sort_unstable_by_key(|&r| (key[r as usize], guid[r as usize]));
        Source {
            version,
            guid,
            key,
            a,
            b,
            by_key_row,
            by_key_guid,
        }
    }

    pub fn version(&self) -> Version {
        self.version
    }

    pub fn n_rows(&self) -> usize {
        self.guid.len()
    }

    /// The source moved: a selection or a binding made before this is stale.
    pub fn advance(&mut self) {
        self.version = Version(self.version.0 + 1);
    }

    /// The filter: rows whose key is in `range`, in source order. The K
    /// source-row positions it returns are a copy and are counted.
    pub fn select_filter(&self, range: KeyRange, counters: &Counters) -> Selection {
        Counters::bump(&counters.storage, 1);
        let rows: Vec<u32> = (0..self.key.len() as u32)
            .filter(|&r| range.has(self.key[r as usize]))
            .collect();
        Counters::bump(&counters.bytes_copied, (rows.len() * 4) as u64);
        Selection {
            version: self.version,
            rows,
            order: Order::Source,
        }
    }

    /// The index order composed with the filter: rows whose key is in
    /// `range`, in the declared `order`. One contiguous range of that
    /// order's index; nothing is sorted here. The K positions are a copy
    /// and are counted.
    ///
    /// # Panics
    /// If `order` is [`Order::Source`]: that is the filter, not an index.
    pub fn select_ordered(&self, range: KeyRange, order: Order, counters: &Counters) -> Selection {
        let index = match order {
            Order::KeyThenRow => &self.by_key_row,
            Order::KeyThenGuid => &self.by_key_guid,
            Order::Source => panic!("select_ordered needs a key order; use select_filter"),
        };
        Counters::bump(&counters.storage, 1);
        let lo = index.partition_point(|&r| self.key[r as usize] < range.lo);
        let hi = index.partition_point(|&r| self.key[r as usize] <= range.hi);
        let rows = index[lo..hi.max(lo)].to_vec();
        Counters::bump(&counters.bytes_copied, (rows.len() * 4) as u64);
        Selection {
            version: self.version,
            rows,
            order,
        }
    }

    /// Stage the selected rows: one copy of K rows per lane, counted.
    pub(crate) fn take(&self, rows: &[u32], counters: &Counters) -> Staged {
        Counters::bump(&counters.storage, 1);
        Counters::bump(
            &counters.bytes_copied,
            (rows.len() * (16 + 4 + 4 + 4)) as u64,
        );
        let pick = |r: &u32| *r as usize;
        Staged {
            guid: rows.iter().map(|r| self.guid[pick(r)]).collect(),
            key: rows.iter().map(|r| self.key[pick(r)]).collect(),
            a: rows.iter().map(|r| self.a[pick(r)]).collect(),
            b: rows.iter().map(|r| self.b[pick(r)]).collect(),
        }
    }

    /// For the oracle and the resident baseline only: the source lanes as
    /// `(guid, key, a, b)`. Does not count as a storage access; callers
    /// outside `oracle` and the baseline arm must not use it.
    pub fn lanes(&self) -> (&[u128], &[i32], &[i32], &[i32]) {
        (&self.guid, &self.key, &self.a, &self.b)
    }
}
