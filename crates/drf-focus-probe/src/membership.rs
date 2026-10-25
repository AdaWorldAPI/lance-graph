//! D-DRAV-1 AV1: map-only membership, `M[i] = contains(S, DR64[i])`.
//!
//! Value independence is structural: [`membership`] takes a [`BoundMap`] and a
//! [`RowSet`], and neither holds a row, a facet or a gathered column. A caller
//! that holds a `Soa` or a `Focus` cannot pass it here:
//!
//! ```compile_fail
//! use drf_focus_probe::{membership::*, *};
//! fn try_with_rows(soa: &Soa<'_>, set: &RowSet) {
//!     let _ = membership(soa, set); // a source with rows is not a BoundMap
//! }
//! ```
//!
//! All domain, scope, seal and range checks run once, when the map and the set
//! are bound. The slot loop only reads the map and the set. Building a row set
//! from canonical values is preparation and is not part of this claim.

use crate::{Refusal, Scope, SealedMap, SourceId, TILE_WORDS};
use std::sync::Arc;

/// Words in a bitset over the whole u16 ordinal domain.
const SET_WORDS: usize = 1 << 10;

/// A 4096-bit slot mask, 512 bytes.
pub type SlotMask = [u64; TILE_WORDS];

/// A sealed map bound to one attested source domain and scope. Holds no rows.
pub struct BoundMap {
    map: Arc<SealedMap>,
    source: SourceId,
    scope: Scope,
}

impl BoundMap {
    /// Bind once: the map's seal must match the source's index seal, and every
    /// ordinal must lie inside the attested row count. `n_rows` is source
    /// metadata, not a value read.
    pub fn bind(
        map: Arc<SealedMap>,
        source: SourceId,
        scope: Scope,
        n_rows: usize,
    ) -> Result<Self, Refusal> {
        if map.seal != source.index_seal {
            return Err(Refusal::StaleMap {
                map_seal: map.seal,
                source_seal: source.index_seal,
            });
        }
        if let Some((slot, &ord)) = map
            .ord
            .iter()
            .enumerate()
            .find(|&(_, &o)| o as usize >= n_rows)
        {
            return Err(Refusal::OrdinalOutOfRange { slot, ord });
        }
        Ok(Self { map, source, scope })
    }

    pub fn source(&self) -> SourceId {
        self.source
    }
}

/// A set of canonical row ordinals in one attested source domain and scope.
pub struct RowSet {
    bits: Box<[u64; SET_WORDS]>,
    source: SourceId,
    scope: Scope,
}

impl RowSet {
    /// Rows outside the attested row count are refused. Repeated rows are a set
    /// member once; that does not dedupe map slots.
    pub fn new(
        source: SourceId,
        scope: Scope,
        n_rows: usize,
        rows: impl IntoIterator<Item = u16>,
    ) -> Result<Self, Refusal> {
        let mut bits = Box::new([0u64; SET_WORDS]);
        for (slot, r) in rows.into_iter().enumerate() {
            if r as usize >= n_rows {
                return Err(Refusal::OrdinalOutOfRange { slot, ord: r });
            }
            bits[r as usize >> 6] |= 1 << (r & 63);
        }
        Ok(Self {
            bits,
            source,
            scope,
        })
    }

    fn contains(&self, r: u16) -> bool {
        self.bits[r as usize >> 6] >> (r & 63) & 1 == 1
    }
}

/// `M[i] = contains(S, DR64[i])` for every slot. Slots sharing an ordinal all
/// match. Refuses when the map and the set are bound to different domains.
pub fn membership(map: &BoundMap, set: &RowSet) -> Result<SlotMask, Refusal> {
    if map.source != set.source {
        return Err(Refusal::DomainMismatch {
            map: map.source,
            set: set.source,
        });
    }
    if map.scope != set.scope {
        return Err(Refusal::ScopeMismatch {
            map: map.scope,
            set: set.scope,
        });
    }
    let mut m = [0u64; TILE_WORDS];
    for (w, chunk) in m.iter_mut().zip(map.map.ord.as_chunks::<64>().0) {
        let mut word = 0u64;
        for (bit, &o) in chunk.iter().enumerate() {
            word |= (set.contains(o) as u64) << bit;
        }
        *w = word;
    }
    Ok(m)
}
