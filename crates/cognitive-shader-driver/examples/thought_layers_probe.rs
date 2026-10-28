//! **D-HL-5 / D-HL-7 (in memory): up to 64 thought tables over one immutable
//! base, saccades per look, sparse delta writes, merged on read.**
//!
//! Plan: `.claude/plans/ce64-hypothesis-layers-v1.md` (§ Pre-registration).
//! Design (plateau §1 point 4): each thought table is a copy-on-write view of
//! the base table. Any change inside a 512-byte row causes a write, and only
//! the changed bytes are kept: a [`RowDelta`] holds the XOR against the base
//! for the 16-byte slots that differ and nothing else. A read is
//! `base XOR delta`; an untouched row reads the base, zero-copy.
//!
//! **Placement** (operator, 2026-10-14): a thought's delta goes either into a
//! table it owns alone ([`Placement::Table`], the global spend) or into one
//! 16-byte slot of a table it shares with other thoughts
//! ([`Placement::Slot`], the local spend). A local thought may write only its
//! slot, so several thoughts share one table without mixing. Both placements
//! read back identically; they differ only in how many of the 64 tables they
//! use.
//!
//! **Saccades** are the existing `AlphaOverlay` claims, one overlay per
//! thought (not per table, so slot-placed thoughts sharing a table keep
//! separate paths): where a thought looked, in which order, and where it came back. A look is
//! recorded whether or not it changes anything, so "looked and agreed with
//! the base" stays visible.
//!
//! What this probe does not do: persist (D-HL-7's split-tunnel append), or
//! read through Lance. The base is borrowed (`&[NodeRow]`), so no table can
//! write it; that is a compile-time property and has no test (a test of it
//! could not fail).

use std::collections::HashMap;

use lance_graph_contract::alpha::{AlphaAddr, AlphaAllocation, AlphaMask, AlphaOverlay};
use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow, TailVariant};
use lance_graph_contract::class_view::{ClassId, ClassView};
use lance_graph_contract::lane_codec::{EpiLaneClass, EpiLaneSpec, LaneWidth};
use ndarray::hpc::codec9::EpiLane9;
use std::marker::PhantomData;

/// One thought's writes for [`ThoughtTables::write_parallel`].
pub type ThoughtJob = (ThoughtId, Vec<(AlphaAddr, [u8; 512])>);

/// The table budget.
pub const MAX_TABLES: usize = 64;
/// Bytes per slot: one value tenant.
pub const SLOT: usize = 16;
/// Slots per 512-byte row.
pub const SLOTS: usize = 512 / SLOT;
/// The key occupies slot 0. It is the address and never changes.
const KEY_SLOT: u32 = 1 << 0;

/// A thought. Its identity is a number; any name is a label kept elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ThoughtId(pub u8);

/// A table index, `0..MAX_TABLES`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableId(pub u8);

/// Where a thought's deltas go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// A table owned by this thought alone; it may change any non-key slot.
    Table(TableId),
    /// One slot of a shared table; it may change that slot only.
    Slot(TableId, u8),
}

impl Placement {
    fn table(self) -> TableId {
        match self {
            Placement::Table(t) | Placement::Slot(t, _) => t,
        }
    }
}

/// Why a write or a placement was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// More than [`MAX_TABLES`] tables.
    OverBudget,
    /// The thought already has a placement.
    AlreadyPlaced,
    /// The table is owned whole by another thought, or the slot is taken.
    Occupied,
    /// The slot is the key slot or out of range.
    BadSlot,
    /// The thought has no placement.
    Unplaced,
    /// The address is not in the base.
    Unallocated,
    /// The write changes the key: the address is immutable.
    KeyChanged,
    /// A slot-placed thought changed bytes outside its slot.
    OutsideSlot,
    /// The class declares no lane layout, or a different one than the write
    /// was compiled for.
    LaneLayout,
}

/// One row's change against the base: the slots that differ and their XOR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDelta {
    /// Bit `i` set: slot `i` differs from the base.
    pub slots: u32,
    /// The XOR of the changed slots, in slot order.
    pub words: Vec<[u8; SLOT]>,
}

impl RowDelta {
    /// The change from `base` to `new`, or `None` when they are equal.
    #[must_use]
    pub fn between(base: &[u8; 512], new: &[u8; 512]) -> Option<Self> {
        let mut slots = 0u32;
        let mut words = Vec::new();
        for s in 0..SLOTS {
            let mut x = [0u8; SLOT];
            let mut any = 0u8;
            for (i, b) in x.iter_mut().enumerate() {
                *b = base[s * SLOT + i] ^ new[s * SLOT + i];
                any |= *b;
            }
            if any != 0 {
                slots |= 1 << s;
                words.push(x);
            }
        }
        (slots != 0).then_some(Self { slots, words })
    }

    /// `base XOR self`.
    #[must_use]
    pub fn apply(&self, base: &[u8; 512]) -> [u8; 512] {
        let mut out = *base;
        let mut k = 0;
        for s in 0..SLOTS {
            if self.slots & (1 << s) != 0 {
                for i in 0..SLOT {
                    out[s * SLOT + i] ^= self.words[k][i];
                }
                k += 1;
            }
        }
        out
    }

    /// Only the slots in `mask`.
    #[must_use]
    pub fn restrict(&self, mask: u32) -> Option<Self> {
        let mut words = Vec::new();
        let mut k = 0;
        let mut slots = 0u32;
        for s in 0..SLOTS {
            if self.slots & (1 << s) != 0 {
                if mask & (1 << s) != 0 {
                    slots |= 1 << s;
                    words.push(self.words[k]);
                }
                k += 1;
            }
        }
        (slots != 0).then_some(Self { slots, words })
    }

    /// Bytes kept.
    #[must_use]
    pub fn stored_bytes(&self) -> usize {
        self.words.len() * SLOT
    }
}

/// The 512 bytes of a row, little-endian as laid out.
#[must_use]
pub fn row_bytes(r: &NodeRow) -> [u8; 512] {
    let mut b = [0u8; 512];
    b[0..16].copy_from_slice(r.key.as_bytes());
    b[16..32].copy_from_slice(r.edges.as_bytes());
    b[32..512].copy_from_slice(&r.value);
    b
}

/// Who owns a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Owner {
    Whole(ThoughtId),
    Slots([Option<ThoughtId>; SLOTS]),
}

/// One thought table: its deltas. Saccades live per thought.
pub struct Table {
    owner: Owner,
    deltas: HashMap<u32, RowDelta>,
}

/// Whether a lane of `row` is absent (presence 0, value 0) under `spec`. A
/// 9-bit lane is read through `ndarray`'s DR512 codeword, so the probe and the
/// codec share one definition.
fn never_touched(spec: &EpiLaneSpec, row: &[u8; 512], lane: usize) -> bool {
    let (p, m) = (spec.present(row, lane), spec.value(row, lane));
    match spec.width {
        LaneWidth::Bits9 => EpiLane9::from_codeword(u16::from(m) | (u16::from(p) << 8))
            .is_some_and(EpiLane9::never_touched),
        LaneWidth::Bits8 => EpiLaneSpec::absent(m, p),
    }
}

/// Every absent lane of `new` takes the lane of `prior`: its value byte and
/// its presence bit. Returns the lanes kept.
fn keep_unknown(spec: &EpiLaneSpec, prior: &[u8; 512], new: &mut [u8; 512]) -> u64 {
    let mut kept = 0u64;
    for lane in 0..usize::from(spec.lanes) {
        if never_touched(spec, new, lane) {
            new[usize::from(spec.values) + lane] = prior[usize::from(spec.values) + lane];
            kept |= 1 << lane;
        }
    }
    for lane in (0..usize::from(spec.lanes)).filter(|l| kept >> l & 1 == 1) {
        let (byte, bit) = (usize::from(spec.presence) + lane / 8, 1u8 << (lane % 8));
        new[byte] = (new[byte] & !bit) | (prior[byte] & bit);
    }
    kept
}

/// A class bound to the lane layout `L`: [`ThoughtTables::bind_lanes`] checked
/// once that the ClassView declares `L::SPEC` for the class. Writes take the
/// binding and do not look the class up again. A binding shared across
/// threads belongs in a `OnceLock` set when the registry loads; `LazyLock`
/// does not fit, because its initializer cannot take the view.
#[derive(Debug)]
pub struct LaneBinding<L> {
    class: ClassId,
    layout: PhantomData<L>,
}

impl<L> LaneBinding<L> {
    /// The class bound.
    #[must_use]
    pub fn class(&self) -> ClassId {
        self.class
    }
}

/// What a write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    /// The row equals what the table already reads: nothing stored.
    NoChange,
    /// The delta now holds this many changed slots for the row.
    Stored { slots: u32 },
}

/// Up to [`MAX_TABLES`] thought tables over one base.
pub struct ThoughtTables<'a> {
    alloc: &'a AlphaAllocation<'a>,
    tables: Vec<Table>,
    placement: HashMap<ThoughtId, Placement>,
    saccades: HashMap<ThoughtId, AlphaOverlay<'a>>,
    cycle: u32,
}

impl<'a> ThoughtTables<'a> {
    /// No tables yet: reserving costs nothing.
    #[must_use]
    pub fn over(alloc: &'a AlphaAllocation<'a>, cycle: u32) -> Self {
        Self {
            alloc,
            tables: Vec::new(),
            placement: HashMap::new(),
            saccades: HashMap::new(),
            cycle,
        }
    }

    /// Tables in use.
    #[must_use]
    pub fn tables_used(&self) -> usize {
        self.tables.len()
    }

    /// Open a new table, empty.
    ///
    /// # Errors
    /// [`Refusal::OverBudget`] past [`MAX_TABLES`].
    pub fn open_table(&mut self) -> Result<TableId, Refusal> {
        if self.tables.len() >= MAX_TABLES {
            return Err(Refusal::OverBudget);
        }
        let id = TableId(u8::try_from(self.tables.len()).map_err(|_| Refusal::OverBudget)?);
        self.tables.push(Table {
            owner: Owner::Slots([None; SLOTS]),
            deltas: HashMap::new(),
        });
        Ok(id)
    }

    /// Give `thought` a placement.
    ///
    /// # Errors
    /// See [`Refusal`].
    pub fn place(&mut self, thought: ThoughtId, p: Placement) -> Result<(), Refusal> {
        if self.placement.contains_key(&thought) {
            return Err(Refusal::AlreadyPlaced);
        }
        let t = self
            .tables
            .get_mut(p.table().0 as usize)
            .ok_or(Refusal::OverBudget)?;
        match (p, &mut t.owner) {
            (Placement::Table(_), Owner::Slots(slots)) if slots.iter().all(Option::is_none) => {
                t.owner = Owner::Whole(thought);
            }
            (Placement::Slot(_, s), Owner::Slots(slots)) => {
                let s = s as usize;
                if s >= SLOTS || (1u32 << s) & KEY_SLOT != 0 {
                    return Err(Refusal::BadSlot);
                }
                if slots[s].is_some() {
                    return Err(Refusal::Occupied);
                }
                slots[s] = Some(thought);
            }
            _ => return Err(Refusal::Occupied),
        }
        self.placement.insert(thought, p);
        self.saccades
            .insert(thought, AlphaOverlay::over_shared(self.alloc, self.cycle));
        Ok(())
    }

    fn ordinal(&self, addr: AlphaAddr) -> Result<u32, Refusal> {
        self.alloc.ordinal(addr).ok_or(Refusal::Unallocated)
    }

    fn base_row(&self, ord: u32) -> [u8; 512] {
        row_bytes(&self.alloc.base()[ord as usize])
    }

    /// The slots `thought` may change.
    fn allowed(p: Placement) -> u32 {
        match p {
            Placement::Table(_) => !KEY_SLOT,
            Placement::Slot(_, s) => 1 << s,
        }
    }

    /// A look: record the saccade only.
    ///
    /// # Errors
    /// [`Refusal::Unplaced`] or [`Refusal::Unallocated`].
    pub fn look(&mut self, thought: ThoughtId, addr: AlphaAddr, rung: u8) -> Result<(), Refusal> {
        self.saccades
            .get_mut(&thought)
            .ok_or(Refusal::Unplaced)?
            .claim(addr, rung)
            .map(|_| ())
            .map_err(|_| Refusal::Unallocated)
    }

    /// Write what `thought` concludes the row at `addr` is. Records the
    /// saccade, then stores only the slots that differ from the base.
    ///
    /// # Errors
    /// See [`Refusal`]. A refused write records no saccade and stores nothing.
    pub fn write(
        &mut self,
        thought: ThoughtId,
        addr: AlphaAddr,
        rung: u8,
        row: &[u8; 512],
    ) -> Result<Written, Refusal> {
        let p = *self.placement.get(&thought).ok_or(Refusal::Unplaced)?;
        let ord = self.ordinal(addr)?;
        let base = self.base_row(ord);
        let mine = Self::allowed(p);
        let delta = RowDelta::between(&base, row);
        if let Some(d) = &delta {
            if d.slots & KEY_SLOT != 0 {
                return Err(Refusal::KeyChanged);
            }
            if d.slots & !mine != 0 {
                return Err(Refusal::OutsideSlot);
            }
        }
        self.saccades
            .get_mut(&thought)
            .ok_or(Refusal::Unplaced)?
            .claim(addr, rung)
            .map_err(|_| Refusal::Unallocated)?;
        let t = &mut self.tables[p.table().0 as usize];
        // Keep the other slots' owners' bytes; replace only this thought's.
        let others = t.deltas.get(&ord).and_then(|d| d.restrict(!mine));
        let merged = match (others, delta.and_then(|d| d.restrict(mine))) {
            (None, None) => None,
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (Some(a), Some(b)) => {
                let full = b.apply(&a.apply(&base));
                RowDelta::between(&base, &full)
            }
        };
        let before = t.deltas.get(&ord).cloned();
        match merged {
            None => {
                t.deltas.remove(&ord);
            }
            Some(d) => {
                t.deltas.insert(ord, d);
            }
        }
        let after = t.deltas.get(&ord);
        Ok(if before.as_ref() == after {
            Written::NoChange
        } else {
            Written::Stored {
                slots: after.map_or(0, |d| d.slots),
            }
        })
    }

    /// Bind `class` to the lane layout `L`, once.
    ///
    /// # Errors
    /// [`Refusal::LaneLayout`] when `view` declares no layout or another one
    /// for `class`.
    pub fn bind_lanes<L: EpiLaneClass>(
        view: &impl ClassView,
        class: ClassId,
    ) -> Result<LaneBinding<L>, Refusal> {
        if view.epi_lanes(class) != Some(L::SPEC) {
            return Err(Refusal::LaneLayout);
        }
        Ok(LaneBinding {
            class,
            layout: PhantomData,
        })
    }

    /// A sparse write of epistemic lanes laid out as `L` declares. An absent
    /// lane (presence 0, value 0) is not written: it keeps what the thought
    /// read there before, so UNKNOWN never overwrites a known value and the
    /// merge on read returns the earlier value. Every other lane, including a
    /// 9-bit negative (presence 0, magnitude > 0) and +0, is written as given.
    /// Bytes outside the lanes are written as by [`write`](Self::write).
    ///
    /// # Errors
    /// As [`write`](Self::write).
    pub fn write_epi<L: EpiLaneClass>(
        &mut self,
        _binding: &LaneBinding<L>,
        thought: ThoughtId,
        addr: AlphaAddr,
        rung: u8,
        row: &[u8; 512],
    ) -> Result<Written, Refusal> {
        let prior = self.read(thought, addr)?;
        let mut new = *row;
        keep_unknown(&L::SPEC, &prior, &mut new);
        self.write(thought, addr, rung, &new)
    }

    /// Merge on read: the row as `thought` sees it. A slot-placed thought sees
    /// only its own slot over the base.
    ///
    /// # Errors
    /// [`Refusal::Unplaced`] or [`Refusal::Unallocated`].
    pub fn read(&self, thought: ThoughtId, addr: AlphaAddr) -> Result<[u8; 512], Refusal> {
        let p = *self.placement.get(&thought).ok_or(Refusal::Unplaced)?;
        let ord = self.ordinal(addr)?;
        let base = self.base_row(ord);
        let t = &self.tables[p.table().0 as usize];
        Ok(
            match t
                .deltas
                .get(&ord)
                .and_then(|d| d.restrict(Self::allowed(p)))
            {
                None => base,
                Some(d) => d.apply(&base),
            },
        )
    }

    /// The slots in which two thoughts read `addr` differently.
    ///
    /// # Errors
    /// As [`read`](Self::read).
    pub fn differs(&self, a: ThoughtId, b: ThoughtId, addr: AlphaAddr) -> Result<u32, Refusal> {
        let (ra, rb) = (self.read(a, addr)?, self.read(b, addr)?);
        Ok(RowDelta::between(&ra, &rb).map_or(0, |d| d.slots))
    }

    /// Which tables hold a delta for `addr`, one bit per table.
    #[must_use]
    pub fn written_by(&self, addr: AlphaAddr) -> u64 {
        let Ok(ord) = self.ordinal(addr) else {
            return 0;
        };
        self.tables
            .iter()
            .enumerate()
            .filter(|(_, t)| t.deltas.contains_key(&ord))
            .fold(0u64, |w, (i, _)| w | 1 << i)
    }

    /// Where `thought` looked.
    ///
    /// # Errors
    /// [`Refusal::Unplaced`].
    pub fn looked(&self, thought: ThoughtId) -> Result<AlphaMask, Refusal> {
        Ok(self
            .saccades
            .get(&thought)
            .ok_or(Refusal::Unplaced)?
            .attended_mask())
    }

    /// The saccade path of `thought`, in order.
    ///
    /// # Errors
    /// [`Refusal::Unplaced`].
    pub fn scanpath(&self, thought: ThoughtId) -> Result<Vec<AlphaAddr>, Refusal> {
        Ok(self
            .saccades
            .get(&thought)
            .ok_or(Refusal::Unplaced)?
            .scanpath()
            .collect())
    }

    /// Bytes stored across all tables.
    #[must_use]
    pub fn stored_bytes(&self) -> usize {
        self.tables
            .iter()
            .flat_map(|t| t.deltas.values())
            .map(RowDelta::stored_bytes)
            .sum()
    }

    /// Write several whole-table thoughts, one thread per table. Each thread
    /// holds only its own table.
    pub fn write_parallel(&mut self, jobs: &[ThoughtJob]) {
        let alloc = self.alloc;
        let mut by_table: HashMap<usize, (ThoughtId, &RowWrites)> = HashMap::new();
        for (th, rows) in jobs {
            if let Some(Placement::Table(t)) = self.placement.get(th) {
                by_table.insert(t.0 as usize, (*th, rows));
            }
        }
        let mut overlays: HashMap<ThoughtId, &mut AlphaOverlay<'a>> =
            self.saccades.iter_mut().map(|(k, v)| (*k, v)).collect();
        std::thread::scope(|s| {
            for (i, t) in self.tables.iter_mut().enumerate() {
                if let Some((th, rows)) = by_table.get(&i).copied() {
                    let Some(sacc) = overlays.remove(&th) else {
                        continue;
                    };
                    s.spawn(move || {
                        for (addr, row) in rows {
                            let Some(ord) = alloc.ordinal(*addr) else {
                                continue;
                            };
                            let base = row_bytes(&alloc.base()[ord as usize]);
                            let _ = sacc.claim(*addr, 0);
                            match RowDelta::between(&base, row) {
                                Some(d) if d.slots & KEY_SLOT == 0 => {
                                    t.deltas.insert(ord, d);
                                }
                                _ => {
                                    t.deltas.remove(&ord);
                                }
                            }
                        }
                    });
                }
            }
        });
    }
}

/// The rows one thought writes, as (address, new row image).
type RowWrites = Vec<(AlphaAddr, [u8; 512])>;

fn base(n: u32) -> Vec<NodeRow> {
    (0..n)
        .map(|i| {
            let mut value = [0u8; 480];
            for (k, b) in value.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(31).wrapping_add(k as u8);
            }
            NodeRow {
                key: NodeGuid::mint_for(TailVariant::V3, 0x0C0C_0000, 1, 2, 3, 4, 0x22, i + 1),
                edges: EdgeBlock::default(),
                value,
            }
        })
        .collect()
}

fn main() {
    let rows = base(256);
    let alloc = AlphaAllocation::over(&rows);
    let mut tt = ThoughtTables::over(&alloc, 1);
    let shared = tt.open_table().expect("budget");
    // 30 thoughts, each changing one slot of every row, all in one table.
    for s in 2..32u8 {
        tt.place(ThoughtId(s), Placement::Slot(shared, s))
            .expect("free slot");
    }
    for s in 2..32u8 {
        for r in &rows {
            let mut b = row_bytes(r);
            b[usize::from(s) * SLOT] ^= 0xFF;
            tt.write(ThoughtId(s), r.key, 1, &b).expect("own slot");
        }
    }
    println!(
        "thought_layers_probe: 30 thoughts × 256 rows in {} table(s), {} bytes stored (full rows would be {} bytes)",
        tt.tables_used(),
        tt.stored_bytes(),
        30 * 256 * 512
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changed(r: &NodeRow, slot: usize, x: u8) -> [u8; 512] {
        let mut b = row_bytes(r);
        b[slot * SLOT + 3] ^= x;
        b
    }

    /// Negative control: a thought that concludes the base stores nothing, but
    /// its look is still recorded.
    #[test]
    fn agreeing_with_the_base_stores_nothing_but_is_seen() {
        let rows = base(8);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        let t = tt.open_table().unwrap();
        tt.place(ThoughtId(1), Placement::Table(t)).unwrap();
        let w = tt
            .write(ThoughtId(1), rows[3].key, 1, &row_bytes(&rows[3]))
            .unwrap();
        assert_eq!(w, Written::NoChange);
        assert_eq!(tt.stored_bytes(), 0);
        assert_eq!(tt.written_by(rows[3].key), 0);
        assert!(
            tt.looked(ThoughtId(1)).unwrap().contains(3),
            "the look is recorded"
        );
    }

    /// Positive control: one changed slot stores exactly that slot, and the
    /// read returns the written row.
    #[test]
    fn one_changed_slot_stores_one_slot_and_reads_back() {
        let rows = base(8);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        let t = tt.open_table().unwrap();
        tt.place(ThoughtId(1), Placement::Table(t)).unwrap();
        let new = changed(&rows[2], 7, 0x5A);
        let w = tt.write(ThoughtId(1), rows[2].key, 1, &new).unwrap();
        assert_eq!(w, Written::Stored { slots: 1 << 7 });
        assert_eq!(tt.stored_bytes(), SLOT);
        assert_eq!(tt.read(ThoughtId(1), rows[2].key).unwrap(), new);
        assert_eq!(
            tt.read(ThoughtId(1), rows[1].key).unwrap(),
            row_bytes(&rows[1]),
            "an untouched row reads the base"
        );
    }

    /// A second write to a row replaces the first; writing the base back
    /// removes the delta.
    #[test]
    fn rewrites_replace_and_reverting_removes() {
        let rows = base(4);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        let t = tt.open_table().unwrap();
        tt.place(ThoughtId(1), Placement::Table(t)).unwrap();
        tt.write(ThoughtId(1), rows[0].key, 1, &changed(&rows[0], 4, 1))
            .unwrap();
        let second = changed(&rows[0], 9, 2);
        tt.write(ThoughtId(1), rows[0].key, 1, &second).unwrap();
        assert_eq!(tt.read(ThoughtId(1), rows[0].key).unwrap(), second);
        assert_eq!(tt.stored_bytes(), SLOT, "only the slot of the latest write");
        tt.write(ThoughtId(1), rows[0].key, 1, &row_bytes(&rows[0]))
            .unwrap();
        assert_eq!(tt.stored_bytes(), 0, "back to the base: nothing stored");
    }

    /// The key is the address and cannot change.
    #[test]
    fn a_write_that_changes_the_key_is_refused() {
        let rows = base(4);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        let t = tt.open_table().unwrap();
        tt.place(ThoughtId(1), Placement::Table(t)).unwrap();
        let mut b = row_bytes(&rows[1]);
        b[5] ^= 1;
        assert_eq!(
            tt.write(ThoughtId(1), rows[1].key, 1, &b),
            Err(Refusal::KeyChanged)
        );
        assert!(tt.scanpath(ThoughtId(1)).unwrap().is_empty(), "no saccade");
    }

    /// Local placement: thoughts sharing a table each see only their slot,
    /// and a write outside the slot is refused.
    #[test]
    fn slot_placed_thoughts_share_a_table_without_mixing() {
        let rows = base(4);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        let t = tt.open_table().unwrap();
        tt.place(ThoughtId(1), Placement::Slot(t, 22)).unwrap();
        tt.place(ThoughtId(2), Placement::Slot(t, 23)).unwrap();
        assert_eq!(
            tt.place(ThoughtId(3), Placement::Slot(t, 22)),
            Err(Refusal::Occupied)
        );
        let a = changed(&rows[0], 22, 0x11);
        let b = changed(&rows[0], 23, 0x22);
        tt.write(ThoughtId(1), rows[0].key, 1, &a).unwrap();
        tt.write(ThoughtId(2), rows[0].key, 1, &b).unwrap();
        assert_eq!(tt.read(ThoughtId(1), rows[0].key).unwrap(), a);
        assert_eq!(tt.read(ThoughtId(2), rows[0].key).unwrap(), b);
        assert_eq!(tt.tables_used(), 1);
        assert_eq!(
            tt.differs(ThoughtId(1), ThoughtId(2), rows[0].key).unwrap(),
            (1 << 22) | (1 << 23)
        );
        assert_eq!(
            tt.write(ThoughtId(1), rows[0].key, 1, &changed(&rows[0], 23, 9)),
            Err(Refusal::OutsideSlot)
        );
    }

    /// Saccades belong to the thought, not the table: two slot-placed thoughts
    /// sharing a table each see only their own looks, in their own order.
    #[test]
    fn slot_placed_thoughts_keep_their_own_saccades() {
        let rows = base(6);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        let t = tt.open_table().unwrap();
        tt.place(ThoughtId(1), Placement::Slot(t, 22)).unwrap();
        tt.place(ThoughtId(2), Placement::Slot(t, 23)).unwrap();
        tt.look(ThoughtId(1), rows[0].key, 1).unwrap();
        tt.look(ThoughtId(2), rows[4].key, 1).unwrap();
        tt.write(ThoughtId(1), rows[2].key, 1, &changed(&rows[2], 22, 1))
            .unwrap();
        tt.look(ThoughtId(2), rows[1].key, 1).unwrap();
        assert_eq!(
            tt.scanpath(ThoughtId(1)).unwrap(),
            vec![rows[0].key, rows[2].key]
        );
        assert_eq!(
            tt.scanpath(ThoughtId(2)).unwrap(),
            vec![rows[4].key, rows[1].key]
        );
        let (a, b) = (
            tt.looked(ThoughtId(1)).unwrap(),
            tt.looked(ThoughtId(2)).unwrap(),
        );
        assert!(a.contains(0) && a.contains(2) && !a.contains(4) && !a.contains(1));
        assert!(b.contains(4) && b.contains(1) && !b.contains(0) && !b.contains(2));
    }

    /// Global and local placement read back the same rows; local uses fewer
    /// tables.
    #[test]
    fn table_and_slot_placement_read_the_same() {
        let rows = base(16);
        let alloc = AlphaAllocation::over(&rows);
        let mut global = ThoughtTables::over(&alloc, 1);
        let mut local = ThoughtTables::over(&alloc, 1);
        let shared = local.open_table().unwrap();
        for s in 2..12u8 {
            let t = global.open_table().unwrap();
            global.place(ThoughtId(s), Placement::Table(t)).unwrap();
            local
                .place(ThoughtId(s), Placement::Slot(shared, s))
                .unwrap();
            for r in &rows {
                let b = changed(r, usize::from(s), s);
                global.write(ThoughtId(s), r.key, 1, &b).unwrap();
                local.write(ThoughtId(s), r.key, 1, &b).unwrap();
            }
        }
        for s in 2..12u8 {
            for r in &rows {
                assert_eq!(
                    global.read(ThoughtId(s), r.key).unwrap(),
                    local.read(ThoughtId(s), r.key).unwrap()
                );
            }
        }
        assert_eq!(global.tables_used(), 10);
        assert_eq!(local.tables_used(), 1);
        assert_eq!(global.stored_bytes(), local.stored_bytes());
    }

    /// The budget is 64 tables.
    #[test]
    fn the_sixty_fifth_table_is_refused() {
        let rows = base(2);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        for _ in 0..MAX_TABLES {
            tt.open_table().unwrap();
        }
        assert_eq!(tt.open_table(), Err(Refusal::OverBudget));
    }

    /// One word per row says which tables wrote it.
    #[test]
    fn the_written_by_word_names_exactly_the_tables_that_wrote() {
        let rows = base(4);
        let alloc = AlphaAllocation::over(&rows);
        let mut tt = ThoughtTables::over(&alloc, 1);
        for th in 0..5u8 {
            let t = tt.open_table().unwrap();
            tt.place(ThoughtId(th), Placement::Table(t)).unwrap();
        }
        tt.write(ThoughtId(1), rows[2].key, 1, &changed(&rows[2], 3, 1))
            .unwrap();
        tt.write(ThoughtId(4), rows[2].key, 1, &changed(&rows[2], 3, 2))
            .unwrap();
        tt.write(ThoughtId(0), rows[2].key, 1, &row_bytes(&rows[2]))
            .unwrap();
        assert_eq!(tt.written_by(rows[2].key), 0b1_0010);
        assert_eq!(tt.written_by(rows[3].key), 0);
    }

    /// Writing tables in parallel gives the same reads as writing them one
    /// after another.
    #[test]
    fn parallel_and_sequential_writes_read_the_same() {
        let rows = base(64);
        let alloc = AlphaAllocation::over(&rows);
        let jobs: Vec<ThoughtJob> = (0..8u8)
            .map(|th| {
                let rs = rows
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i + usize::from(th)) % 3 != 0)
                    .map(|(i, r)| (r.key, changed(r, 2 + (i + usize::from(th)) % 28, th + 1)))
                    .collect();
                (ThoughtId(th), rs)
            })
            .collect();
        let mut seq = ThoughtTables::over(&alloc, 1);
        let mut par = ThoughtTables::over(&alloc, 1);
        for (th, _) in &jobs {
            let a = seq.open_table().unwrap();
            seq.place(*th, Placement::Table(a)).unwrap();
            let b = par.open_table().unwrap();
            par.place(*th, Placement::Table(b)).unwrap();
        }
        for (th, rs) in &jobs {
            for (addr, row) in rs {
                seq.write(*th, *addr, 0, row).unwrap();
            }
        }
        par.write_parallel(&jobs);
        assert!(seq.stored_bytes() > 0);
        for (th, _) in &jobs {
            assert_eq!(seq.scanpath(*th).unwrap(), par.scanpath(*th).unwrap());
            for r in &rows {
                assert_eq!(seq.read(*th, r.key).unwrap(), par.read(*th, r.key).unwrap());
            }
        }
    }

    // ── Epistemic lanes: NaN is not written, merge on read keeps the prior ──
    //
    // The layout of each class is a compile-time constant (`EpiLaneClass`);
    // the ClassView names the same constant for the classid.

    use lance_graph_contract::lane_codec::{checked, CodeSpace};
    use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};

    const VALUES: u16 = 2 * SLOT as u16;
    const PRESENCE: u16 = 3 * SLOT as u16;
    const CLASS: ClassId = 7;

    /// DR512: 9-bit sign-magnitude, presence bit = sign.
    struct Dr512;
    impl EpiLaneClass for Dr512 {
        const SPEC: EpiLaneSpec = checked(EpiLaneSpec {
            width: LaneWidth::Bits9,
            space: CodeSpace::Sym,
            values: VALUES,
            presence: PRESENCE,
            lanes: 16,
        });
    }

    /// ONNX `QuantizeLinear` uint8 with zero point 128, plus a presence plane.
    struct OnnxU8;
    impl EpiLaneClass for OnnxU8 {
        const SPEC: EpiLaneSpec = checked(EpiLaneSpec {
            width: LaneWidth::Bits8,
            space: CodeSpace::from_zero_point(128),
            values: VALUES,
            presence: PRESENCE,
            lanes: 16,
        });
    }

    /// TFLite symmetric int8 weights (zero point 0).
    struct TfliteSym;
    impl EpiLaneClass for TfliteSym {
        const SPEC: EpiLaneSpec = checked(EpiLaneSpec {
            width: LaneWidth::Bits8,
            space: CodeSpace::from_zero_point(0),
            values: VALUES,
            presence: PRESENCE,
            lanes: 16,
        });
    }

    /// A ClassView that declares one layout (or none) for every class.
    struct View(Option<EpiLaneSpec>);
    impl ClassView for View {
        fn fields(&self, _class: ClassId) -> &[FieldRef] {
            &[]
        }
        fn template(&self, _class: ClassId) -> DisplayTemplate {
            DisplayTemplate::Card
        }
        fn dolce_category_id(&self, _class: ClassId) -> u8 {
            0
        }
        fn epi_lanes(&self, _class: ClassId) -> Option<EpiLaneSpec> {
            self.0
        }
    }

    fn view<L: EpiLaneClass>() -> View {
        View(Some(L::SPEC))
    }

    fn bound<L: EpiLaneClass>() -> LaneBinding<L> {
        ThoughtTables::bind_lanes::<L>(&view::<L>(), CLASS).expect("declared")
    }

    /// Base rows whose lanes all hold value 5 + lane with presence 1.
    fn epi_rows() -> Vec<NodeRow> {
        let (v, p) = (usize::from(VALUES) - 32, usize::from(PRESENCE) - 32);
        let mut rows = base(8);
        for r in &mut rows {
            for lane in 0..16 {
                r.value[v + lane] = 5 + lane as u8;
            }
            r.value[p..p + 2].copy_from_slice(&u16::MAX.to_le_bytes());
        }
        rows
    }

    fn set_lane(row: &mut [u8; 512], lane: usize, present: bool, value: u8) {
        let (v, p) = (usize::from(VALUES), usize::from(PRESENCE));
        row[v + lane] = value;
        let mut w = u16::from_le_bytes([row[p], row[p + 1]]);
        w = (w & !(1 << lane)) | (u16::from(present) << lane);
        row[p..p + 2].copy_from_slice(&w.to_le_bytes());
    }

    fn lane(row: &[u8; 512], lane: usize) -> (bool, u8) {
        let (v, p) = (usize::from(VALUES), usize::from(PRESENCE));
        let w = u16::from_le_bytes([row[p], row[p + 1]]);
        ((w >> lane) & 1 == 1, row[v + lane])
    }

    fn one_thought(rows: &[NodeRow]) -> (AlphaAllocation<'_>, ThoughtId) {
        (AlphaAllocation::over(rows), ThoughtId(1))
    }

    fn placed<'a>(alloc: &'a AlphaAllocation<'a>) -> ThoughtTables<'a> {
        let mut tt = ThoughtTables::over(alloc, 1);
        let t = tt.open_table().unwrap();
        tt.place(ThoughtId(1), Placement::Table(t)).unwrap();
        tt
    }

    /// The contract's never-touched rule and ndarray's DR512 codeword agree on
    /// all 512 codes.
    #[test]
    fn contract_and_codec9_agree_on_never_touched() {
        for code in 0u16..512 {
            let e = EpiLane9::from_codeword(code).unwrap();
            assert_eq!(
                EpiLaneSpec::absent(e.magnitude(), e.presence()),
                e.never_touched(),
                "code {code:#x}"
            );
        }
    }

    /// Binding refuses a class with no declared layout or a different one;
    /// the declared one binds once.
    #[test]
    fn binding_refuses_an_undeclared_or_different_layout() {
        for v in [View(None), view::<OnnxU8>()] {
            assert!(matches!(
                ThoughtTables::bind_lanes::<Dr512>(&v, CLASS),
                Err(Refusal::LaneLayout)
            ));
        }
        let b = ThoughtTables::bind_lanes::<Dr512>(&view::<Dr512>(), CLASS).unwrap();
        assert_eq!(b.class(), CLASS);
    }

    fn nan_over_known<L: EpiLaneClass>() {
        let rows = epi_rows();
        let (alloc, th) = one_thought(&rows);
        let mut tt = placed(&alloc);
        let mut row = row_bytes(&rows[2]);
        set_lane(&mut row, 4, false, 0);
        let w = tt
            .write_epi(&bound::<L>(), th, rows[2].key, 1, &row)
            .unwrap();
        assert_eq!(w, Written::NoChange);
        assert_eq!(tt.stored_bytes(), 0);
        assert_eq!(
            lane(&tt.read(th, rows[2].key).unwrap(), 4),
            (true, 9),
            "base kept"
        );
        assert!(tt.looked(th).unwrap().contains(2), "the look is recorded");
    }

    /// NaN over a known base value: nothing stored, the base value is read.
    #[test]
    fn nan_over_a_known_value_is_not_written() {
        nan_over_known::<Dr512>();
        nan_over_known::<OnnxU8>();
    }

    /// NaN after the thought's own write keeps what the thought wrote.
    #[test]
    fn nan_after_an_own_write_keeps_the_own_value() {
        let rows = epi_rows();
        let (alloc, th) = one_thought(&rows);
        let mut tt = placed(&alloc);
        let b = bound::<Dr512>();
        let mut row = row_bytes(&rows[1]);
        set_lane(&mut row, 3, true, 40);
        tt.write_epi(&b, th, rows[1].key, 1, &row).unwrap();
        set_lane(&mut row, 3, false, 0);
        let w = tt.write_epi(&b, th, rows[1].key, 1, &row).unwrap();
        assert_eq!(w, Written::NoChange);
        assert_eq!(
            lane(&tt.read(th, rows[1].key).unwrap(), 3),
            (true, 40),
            "not the base 8"
        );
    }

    /// 9-bit: presence 0 with magnitude 5 is -5, written and read back.
    #[test]
    fn a_nine_bit_negative_is_written() {
        let rows = epi_rows();
        let (alloc, th) = one_thought(&rows);
        let mut tt = placed(&alloc);
        let mut row = row_bytes(&rows[0]);
        set_lane(&mut row, 6, false, 5);
        let w = tt
            .write_epi(&bound::<Dr512>(), th, rows[0].key, 1, &row)
            .unwrap();
        assert!(matches!(w, Written::Stored { .. }));
        let (p, m) = lane(&tt.read(th, rows[0].key).unwrap(), 6);
        let e = EpiLane9::from_codeword(u16::from(m) | (u16::from(p) << 8)).unwrap();
        assert_eq!(e.value(), Some(-5));
    }

    /// A computed -0.499 is encoded as +0 in the same step and is written.
    #[test]
    fn a_computed_negative_zero_is_written_as_plus_zero() {
        use ndarray::hpc::codec8::{Calibration, TiesEven};
        use ndarray::hpc::codec9::Codec9;
        struct Unit;
        impl Calibration for Unit {
            const SCALE: f32 = 1.0;
            const ZERO_POINT: i32 = 0;
        }
        let code = Codec9::<TiesEven, Unit>::encode_touched(-0.499).unwrap();
        assert_eq!((code.presence(), code.magnitude()), (true, 0));
        let rows = epi_rows();
        let (alloc, th) = one_thought(&rows);
        let mut tt = placed(&alloc);
        let mut row = row_bytes(&rows[5]);
        set_lane(&mut row, 0, code.presence(), code.magnitude());
        let w = tt
            .write_epi(&bound::<Dr512>(), th, rows[5].key, 1, &row)
            .unwrap();
        assert!(
            matches!(w, Written::Stored { .. }),
            "+0 replaces the base 5"
        );
        assert_eq!(lane(&tt.read(th, rows[5].key).unwrap(), 0), (true, 0));
    }

    fn nan_beside_change<L: EpiLaneClass>() {
        let rows = epi_rows();
        let (alloc, th) = one_thought(&rows);
        let mut tt = placed(&alloc);
        let mut row = row_bytes(&rows[4]);
        set_lane(&mut row, 1, false, 0);
        set_lane(&mut row, 2, true, 99);
        let w = tt
            .write_epi(&bound::<L>(), th, rows[4].key, 1, &row)
            .unwrap();
        assert_eq!(w, Written::Stored { slots: 1 << 2 }, "value slot only");
        let read = tt.read(th, rows[4].key).unwrap();
        assert_eq!(lane(&read, 1), (true, 6));
        assert_eq!(lane(&read, 2), (true, 99));
    }

    /// A NaN lane next to a changed lane: only the change is stored.
    #[test]
    fn nan_beside_a_change_stores_only_the_change() {
        nan_beside_change::<Dr512>();
        nan_beside_change::<OnnxU8>();
    }

    /// Asymmetric 8-bit with a presence plane: value 0 with presence 1 is a
    /// code like any other and is written.
    #[test]
    fn an_asym_zero_code_is_written() {
        let rows = epi_rows();
        let (alloc, th) = one_thought(&rows);
        let mut tt = placed(&alloc);
        let mut row = row_bytes(&rows[3]);
        set_lane(&mut row, 5, true, 0);
        let w = tt
            .write_epi(&bound::<OnnxU8>(), th, rows[3].key, 1, &row)
            .unwrap();
        assert!(matches!(w, Written::Stored { .. }));
        assert_eq!(lane(&tt.read(th, rows[3].key).unwrap(), 5), (true, 0));
    }

    /// Symmetric int8: absent is presence 0 and value 0 like every layout;
    /// code -128 with presence 1 is a value and is written.
    #[test]
    fn a_sym_int8_min_code_is_a_value() {
        let rows = epi_rows();
        let (alloc, th) = one_thought(&rows);
        let mut tt = placed(&alloc);
        let b = bound::<TfliteSym>();
        let mut row = row_bytes(&rows[6]);
        set_lane(&mut row, 7, false, 0);
        let w = tt.write_epi(&b, th, rows[6].key, 1, &row).unwrap();
        assert_eq!(w, Written::NoChange, "absent: base kept");
        set_lane(&mut row, 7, true, 0x80);
        let w = tt.write_epi(&b, th, rows[6].key, 1, &row).unwrap();
        assert!(matches!(w, Written::Stored { .. }));
        assert_eq!(lane(&tt.read(th, rows[6].key).unwrap(), 7), (true, 0x80));
    }
}
