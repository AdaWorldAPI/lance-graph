//! D-DRF-P1 lab probe. Not a production API.
//!
//! A `Focus` is an immutable 64x64 sealed pointer map plus one gathered i32
//! facet column (the D-DRF-1 §5 derived cache: disposable, never authoritative).
//! The lease begins at `FocusManager::load`, not at the first fold.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "handoff")]
pub mod handoff;

/// Canonical SoA row ABI.
pub const ROW: usize = 512;
/// One SoA row.
pub type Row = [u8; ROW];
/// Pixels in a DR64 focus.
pub const SLOTS: usize = 4096;
/// u64 words in a 4096-bit tile.
pub const TILE_WORDS: usize = 64;

/// Full source witness: never key on a bare ordinal or version (G3, G6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceId {
    pub dataset: u64,
    pub branch: u64,
    pub version: u64,
    pub index_seal: u64,
}

/// Borrowed canonical rows. The probe never owns them.
#[derive(Clone, Copy)]
pub struct Soa<'a> {
    pub rows: &'a [Row],
    pub id: SourceId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MapId(pub u64);

/// Authorized tenant/scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scope(pub u64);

/// Immutable 64x64 u16 pointer map sealed for one row-address seal.
pub struct SealedMap {
    pub id: MapId,
    pub seal: u64,
    pub ord: Box<[u16; SLOTS]>,
}

impl SealedMap {
    pub fn new(id: MapId, seal: u64, ord: Box<[u16; SLOTS]>) -> Self {
        Self { id, seal, ord }
    }
}

/// Cache key: equality on every field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FocusKey {
    pub source: SourceId,
    pub map: MapId,
    pub scope: Scope,
}

/// Declared i32 LE facet at a byte offset inside a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facet {
    offset: usize,
}

impl Facet {
    pub const WIDTH: usize = 4;
    pub fn new(offset: usize) -> Result<Self, Refusal> {
        if offset + Self::WIDTH > ROW {
            return Err(Refusal::FacetOutOfRow);
        }
        Ok(Self { offset })
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    OrdinalOutOfRange {
        slot: usize,
        ord: u16,
    },
    StaleMap {
        map_seal: u64,
        source_seal: u64,
    },
    FacetOutOfRow,
    /// Slot index outside the 64x64 focus.
    SlotOutOfFocus {
        slot: usize,
    },
    /// Two staged writes give one facet of one row different values, or
    /// write overlapping byte ranges of one row (W3).
    Conflict {
        ord: u16,
        offset: usize,
    },
}

fn read_i32(row: &Row, offset: usize) -> i32 {
    i32::from_le_bytes([
        row[offset],
        row[offset + 1],
        row[offset + 2],
        row[offset + 3],
    ])
}

fn check_seal(soa: &Soa<'_>, map: &SealedMap) -> Result<(), Refusal> {
    if map.seal != soa.id.index_seal {
        return Err(Refusal::StaleMap {
            map_seal: map.seal,
            source_seal: soa.id.index_seal,
        });
    }
    Ok(())
}

/// Immutable once built.
pub struct Focus {
    key: FocusKey,
    map: Arc<SealedMap>,
    facet_col: Box<[i32; SLOTS]>,
}

impl Focus {
    fn build(
        soa: &Soa<'_>,
        map: Arc<SealedMap>,
        scope: Scope,
        facet: Facet,
    ) -> Result<Self, Refusal> {
        check_seal(soa, &map)?;
        let mut col = Box::new([0i32; SLOTS]);
        for (slot, &o) in map.ord.iter().enumerate() {
            let row = soa
                .rows
                .get(o as usize)
                .ok_or(Refusal::OrdinalOutOfRange { slot, ord: o })?;
            col[slot] = read_i32(row, facet.offset());
        }
        Ok(Self {
            key: FocusKey {
                source: soa.id,
                map: map.id,
                scope,
            },
            map,
            facet_col: col,
        })
    }

    pub fn key(&self) -> FocusKey {
        self.key
    }
    pub fn map(&self) -> &Arc<SealedMap> {
        &self.map
    }
    pub fn column(&self) -> &[i32; SLOTS] {
        &self.facet_col
    }

    /// Sum of the gathered facet over slots whose mask bit is set.
    pub fn sum_where(&self, mask: &[u64; TILE_WORDS]) -> i64 {
        ndarray::simd::masked_sum_i32(&self.facet_col[..], &mask[..])
    }
}

type Slot = Arc<OnceLock<Result<Arc<Focus>, Refusal>>>;

/// Resident bytes of one built focus: its gathered i32 column. The sealed map
/// is caller-owned and shared through an `Arc`, so it is not counted here.
pub const FOCUS_BYTES: usize = SLOTS * 4;

/// What one [`FocusManager::evict_to_budget`] pass did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetReport {
    /// Entries removed by this pass.
    pub evicted: usize,
    /// Resident bytes after the pass.
    pub resident_bytes: usize,
    /// Of those, bytes held by foci that outside readers still reference.
    pub pinned_bytes: usize,
    /// Still above the budget: only pinned foci remain to evict (W8).
    pub over_budget: bool,
}

struct Entry {
    slot: Slot,
    /// Manager tick of the most recent `load` of this key.
    last: u64,
}

/// No global statics (L3): the lease table is owned by the manager.
#[derive(Default)]
pub struct FocusManager {
    entries: Mutex<HashMap<FocusKey, Entry>>,
    builds: AtomicUsize,
    tick: AtomicU64,
}

impl FocusManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lease begins here. Concurrent loads of one key gather at most once (L1).
    pub fn load(
        &self,
        soa: &Soa<'_>,
        map: &Arc<SealedMap>,
        scope: Scope,
        facet: Facet,
    ) -> Result<Arc<Focus>, Refusal> {
        let key = FocusKey {
            source: soa.id,
            map: map.id,
            scope,
        };
        let now = self.tick.fetch_add(1, Ordering::SeqCst);
        let slot: Slot = {
            let mut g = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let e = g.entry(key).or_insert_with(|| Entry {
                slot: Slot::default(),
                last: now,
            });
            e.last = now;
            e.slot.clone()
        };
        let res = slot.get_or_init(|| {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Focus::build(soa, map.clone(), scope, facet).map(Arc::new)
        });
        if res.is_err() {
            // A refusal is not a lease: do not leave it resident.
            let mut g = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(cur) = g.get(&key)
                && Arc::ptr_eq(&cur.slot, &slot)
            {
                g.remove(&key);
            }
        }
        res.clone()
    }

    /// Drop the manager's entry; holders of the Arc stay valid (L2).
    pub fn retire(&self, key: FocusKey) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&key);
    }

    /// Drop entries no outside holder references (L3). Returns the count.
    pub fn evict_unused(&self) -> usize {
        let mut g = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let before = g.len();
        g.retain(|_, e| match e.slot.get() {
            Some(Ok(f)) => Arc::strong_count(f) > 1,
            _ => true,
        });
        before - g.len()
    }

    /// Bytes of every built focus the manager holds, pinned or not.
    pub fn resident_bytes(&self) -> usize {
        let g = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        g.values()
            .filter(|e| matches!(e.slot.get(), Some(Ok(_))))
            .count()
            * FOCUS_BYTES
    }

    /// Evict unpinned foci, least recently loaded first, until the resident
    /// bytes fit `budget_bytes`. A focus an outside reader still holds is
    /// never evicted, so the manager may stay over budget (W8); a slot still
    /// being built counts as pinned. An evicted focus is rebuilt on its next
    /// `load`.
    pub fn evict_to_budget(&self, budget_bytes: usize) -> BudgetReport {
        let mut g = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let mut resident = 0;
        let mut pinned = 0;
        let mut victims: Vec<(u64, FocusKey)> = Vec::new();
        for (k, e) in g.iter() {
            match e.slot.get() {
                Some(Ok(f)) => {
                    resident += FOCUS_BYTES;
                    if Arc::strong_count(f) > 1 {
                        pinned += FOCUS_BYTES;
                    } else {
                        victims.push((e.last, *k));
                    }
                }
                Some(Err(_)) => {}
                None => pinned += FOCUS_BYTES,
            }
        }
        victims.sort_unstable_by_key(|v| v.0);
        let mut evicted = 0;
        for (_, k) in victims {
            if resident <= budget_bytes {
                break;
            }
            g.remove(&k);
            resident -= FOCUS_BYTES;
            evicted += 1;
        }
        BudgetReport {
            evicted,
            resident_bytes: resident,
            pinned_bytes: pinned,
            over_budget: resident > budget_bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn builds(&self) -> usize {
        self.builds.load(Ordering::SeqCst)
    }
}

/// Strategy A: read rows in place, no cache. `lookahead > 0` prefetches slot
/// `s + lookahead`; the address is rows base + ord*ROW + facet offset with the
/// ordinal bounds-checked first (P3).
pub fn direct_sum_where(
    soa: &Soa<'_>,
    map: &SealedMap,
    facet: Facet,
    mask: &[u64; TILE_WORDS],
    lookahead: usize,
) -> Result<i64, Refusal> {
    check_seal(soa, map)?;
    // Validate every ordinal before any prefetch is issued.
    for (slot, &o) in map.ord.iter().enumerate() {
        if o as usize >= soa.rows.len() {
            return Err(Refusal::OrdinalOutOfRange { slot, ord: o });
        }
    }
    let base = soa.rows.as_ptr() as *const u8;
    let mut acc = 0i64;
    for slot in 0..SLOTS {
        if lookahead > 0
            && let Some(&next) = map.ord.get(slot + lookahead)
        {
            // In range: validated above. wrapping_add keeps this pure address math.
            prefetch(base.wrapping_add(next as usize * ROW + facet.offset()));
        }
        if mask[slot / 64] >> (slot % 64) & 1 == 1 {
            let row = &soa.rows[map.ord[slot] as usize];
            acc = acc.wrapping_add(read_i32(row, facet.offset()) as i64);
        }
    }
    Ok(acc)
}

#[inline(always)]
fn prefetch(p: *const u8) {
    ndarray::simd::prefetch_read_t0(p);
}

/// One staged facet write, addressed by canonical ordinal.
#[derive(Clone, Copy)]
struct Patch {
    ord: u16,
    offset: usize,
    value: i32,
    /// Declared stream position. `None` sorts before every `Some`, so an
    /// unordered write is applied first and can only coexist with ordered
    /// writes of the same value (checked in `flush`).
    pos: Option<u64>,
}

/// The final image of one dirty canonical row, owned for handoff to a writer.
/// `row` is exactly [`ROW`] bytes. It is a `Vec` so the handoff can move it
/// into the writer's payload without a second copy.
pub struct RowImage {
    pub ord: u16,
    pub row: Vec<u8>,
}

/// Writes staged against one focus, coalesced per canonical row at flush
/// (D-DRF W1-W3). The focus itself is never mutated.
pub struct Writeback<'f> {
    focus: &'f Focus,
    patches: Vec<Patch>,
}

impl Focus {
    pub fn writeback(&self) -> Writeback<'_> {
        Writeback {
            focus: self,
            patches: Vec::new(),
        }
    }
}

impl Writeback<'_> {
    /// Stage `value` into `facet` of the row behind `slot`, with no declared
    /// order: a second write to the same facet must carry the same value.
    pub fn stage(&mut self, slot: usize, facet: Facet, value: i32) -> Result<(), Refusal> {
        self.push(slot, facet, value, None)
    }

    /// Stage `value` at declared stream position `pos`. Writes to one facet
    /// apply in position order whatever order they were staged in; two
    /// different values at one position refuse (W3).
    pub fn stage_at(
        &mut self,
        slot: usize,
        facet: Facet,
        value: i32,
        pos: u64,
    ) -> Result<(), Refusal> {
        self.push(slot, facet, value, Some(pos))
    }

    fn push(
        &mut self,
        slot: usize,
        facet: Facet,
        value: i32,
        pos: Option<u64>,
    ) -> Result<(), Refusal> {
        let ord = *self
            .focus
            .map
            .ord
            .get(slot)
            .ok_or(Refusal::SlotOutOfFocus { slot })?;
        self.patches.push(Patch {
            ord,
            offset: facet.offset(),
            value,
            pos,
        });
        Ok(())
    }

    /// Distinct canonical rows touched so far.
    pub fn dirty_rows(&self) -> usize {
        let mut ords: Vec<u16> = self.patches.iter().map(|p| p.ord).collect();
        ords.sort_unstable();
        ords.dedup();
        ords.len()
    }

    /// One image per distinct dirty row, in ordinal order: the sealed source row
    /// with every staged facet applied. Writes to different facets of one row
    /// all survive (W2). Writes to one facet apply in declared position
    /// order; different values with no order between them (equal positions,
    /// or an unordered write), or facets whose byte ranges overlap, refuse (W3).
    pub fn flush(mut self, soa: &Soa<'_>) -> Result<Vec<RowImage>, Refusal> {
        check_seal(soa, &self.focus.map)?;
        if self.focus.key.source != soa.id {
            return Err(Refusal::StaleMap {
                map_seal: self.focus.map.seal,
                source_seal: soa.id.index_seal,
            });
        }
        self.patches.sort_by_key(|p| (p.ord, p.offset, p.pos));
        // Value of the current facet group's unordered writes. `None` sorts
        // first, so they all precede the group's positioned writes; an
        // unordered write has no order against ANY of them, not just the
        // first, so every positioned value must equal it.
        let mut unordered: Option<i32> = None;
        for w in self.patches.windows(2) {
            if (w[0].ord, w[0].offset) != (w[1].ord, w[1].offset) {
                unordered = None;
            } else if w[0].pos.is_none() {
                unordered = Some(w[0].value);
            }
            if w[0].ord != w[1].ord {
                continue;
            }
            // Same facet: different values need two distinct declared
            // positions, else nothing says which wins. Different facets must
            // not overlap. Sorted by offset, so neighbours find every overlap.
            let same = w[0].offset == w[1].offset;
            let ordered = matches!((w[0].pos, w[1].pos), (Some(a), Some(b)) if a != b);
            if (same && w[0].value != w[1].value && !ordered)
                || (same && unordered.is_some_and(|u| u != w[1].value))
                || (!same && w[1].offset < w[0].offset + Facet::WIDTH)
            {
                return Err(Refusal::Conflict {
                    ord: w[1].ord,
                    offset: w[1].offset,
                });
            }
        }
        // Applying in sorted order leaves each facet at its highest position.
        let mut out: Vec<RowImage> = Vec::new();
        for p in &self.patches {
            if out.last().is_none_or(|img| img.ord != p.ord) {
                // The focus was built from this same source, so every ordinal
                // was range-checked then; refuse rather than index if not.
                let row = soa
                    .rows
                    .get(p.ord as usize)
                    .ok_or(Refusal::OrdinalOutOfRange {
                        slot: usize::MAX,
                        ord: p.ord,
                    })?
                    .to_vec();
                out.push(RowImage { ord: p.ord, row });
            }
            let img = out.last_mut().expect("pushed above");
            img.row[p.offset..p.offset + Facet::WIDTH].copy_from_slice(&p.value.to_le_bytes());
        }
        Ok(out)
    }
}
