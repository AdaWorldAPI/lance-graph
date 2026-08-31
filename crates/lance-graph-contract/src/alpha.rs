//! ⚠ MIGRATED FROM `medcare-rs/crates/medcare-nodesoa/src/alpha.rs` (2026-08-31,
//! operator-ruled: the alpha channel is SUBSTRATE and belongs in lance-graph,
//! not as a consumer-side island). The Arrow/Lance storage glue (`to_batch`,
//! `key_bytes_at`, the `lance` feature module) deliberately stayed with the
//! storage crate — this is the pure overlay algebra over contract types.
//!
//! **The thin-provisioned alpha channel** — a second table at the *same*
//! addresses as an already-baked SoA spine, materialised only where attention
//! actually went.
//!
//! Operator ruling, 2026-08-21: *„ephemer daneben, verwerfbar."* The overlay is
//! **not** a bake: no `data/config/bakes.tsv` row, no digest, no re-pin across
//! four repos. It is discardable **whole** — because it is not a cache of
//! derived truth (which would need invalidation) but a record of *where
//! attention went*. Dropping it costs a re-search, never a correctness
//! question. Ledger: `docs/RAIL_OFFENE_POSTEN.md` Posten 11.
//!
//! # Allocate ≠ claim
//!
//! | | meaning | cost |
//! |---|---|---|
//! | **allocate** | the address space — *every* address the base spine already has | zero rows |
//! | **claim** | materialise ONE row at ONE address | one 512-byte row |
//!
//! An address that was allocated but never claimed reads as
//! [`None`] — **„not attended"**, the zero-fallback ladder one level up. It
//! does **not** fall back to the base row: a plausible value in place of an
//! absent one is exactly the failure the ladder exists to prevent.
//!
//! # Same size, address, hash, index
//!
//! The overlay row is a canonical [`NodeRow`] — the same 512-byte stride, the
//! same `FixedSizeBinary(512)` Arrow column, encoded by the same
//! the FixedSizeBinary(512) node encoder (medcare-nodesoa `node_rows_to_batch`). Its `key` is **copied
//! verbatim** from the base row, never re-minted, so the address is
//! byte-identical by construction and the overlay never has to decode the
//! tail — which is what keeps it out of the V1/V3 trap the sibling lanes
//! document (*„each artifact is read on its own tail"*). What differs is only
//! the value slab: slot 0 carries the [`AlphaStamp`].
//!
//! # One direction
//!
//! The overlay borrows the base spine (`&'a [NodeRow]`) and never holds it
//! mutably. *The overlay reads the graph; the graph never reads the overlay* is
//! therefore a **compile-time** property of this type, not a runtime check —
//! and deliberately not a test, because a test of it could not fail.
//!
//! # What this PoC does NOT do
//!
//! The saccade *direction* is carried as the claim order ([`AlphaStamp::seq`]),
//! not as an edge. The 16-byte [`EdgeBlock`](crate::canonical_node::EdgeBlock)
//! stays **zeroed and reserved**: its one-byte slots are basin-local references
//! that need the sibling codebook of Posten 4a, which is not built. Reserving
//! the block costs nothing and keeps the row canon-shaped for the day it is
//! (RESERVE, DON'T RECLAIM).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::canonical_node::{NodeGuid, NodeRow};

/// An address in the overlay — the base row's [`NodeGuid`], copied verbatim.
///
/// Deliberately the whole 16-byte key and never a decoded `(classid, identity)`
/// pair: the OBO lane is V3-tailed and the ICD/patient lanes are V1-tailed, so
/// any decode here would be right for one artifact and wrong for the next.
pub type AlphaAddr = NodeGuid;

/// Byte offset of the [`AlphaStamp`] inside [`NodeRow::value`] — slot 0 of the
/// 30 available 16-byte value slots (one concern per slot, per the canon's
/// clean-over-packed doctrine).
pub const ALPHA_STAMP_OFFSET: usize = 0;

/// Width of the [`AlphaStamp`] — one 16-byte value slot.
pub const ALPHA_STAMP_BYTES: usize = 16;

/// What one claim records: **that** attention landed here, **when**, in **which
/// order**, and at **which rung**.
///
/// Sixteen bytes, little-endian, in value slot 0:
///
/// ```text
///  0..4   cycle (u32)  — the thinking cycle the claim belongs to
///  4..8   seq   (u32)  — claim order = the saccade's position (the trajectory IS the index)
///  8..9   rung   (u8)  — which rung of attention landed
///  9..11  visits (u16) — how often attention returned here (1 on the first claim)
/// 11..16  reserved      — zeroed; reserved, never reclaimed
/// ```
///
/// It carries **no concept**. A fat concept in every row multiplies the fabric
/// by the concept's size; the overlay row names *where* it looked, and the
/// concept stays in the row the address already resolves to.
///
/// # Why `visits` exists — the regression is the diagnosis
///
/// In eye tracking a **regression** (the gaze jumping back to something already
/// read) is the single most diagnostic event: it says the reader did not
/// integrate the first time. The first cut of this type recorded the revisit
/// only as `fresh: false` and dropped it — throwing away the best signal the
/// channel has. `visits` keeps it *additively*: the first visit's `seq`, `rung`
/// and `cycle` are still never rewritten, so the scanpath's history stays
/// intact; only the counter moves.
///
/// A stored `visits == 0` on an existing row would be self-contradictory (the
/// row exists, so it was claimed at least once). It cannot occur, because the
/// overlay is ephemeral — there are no rows from before this field. That is one
/// concrete thing the *„ephemer daneben, verwerfbar"* ruling buys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphaStamp {
    /// The thinking cycle this claim belongs to.
    pub cycle: u32,
    /// Claim order within the overlay — the saccade's position.
    pub seq: u32,
    /// Which rung of attention landed here.
    pub rung: u8,
    /// How often attention landed here — `1` on the first claim, incremented on
    /// every revisit. Saturating: a hot address stops counting rather than
    /// wrapping to a lie.
    pub visits: u16,
}

impl AlphaStamp {
    /// Encode into the 16-byte value slot (LE), trailing bytes zeroed.
    #[must_use]
    pub fn to_le_slot(self) -> [u8; ALPHA_STAMP_BYTES] {
        let mut b = [0u8; ALPHA_STAMP_BYTES];
        b[0..4].copy_from_slice(&self.cycle.to_le_bytes());
        b[4..8].copy_from_slice(&self.seq.to_le_bytes());
        b[8] = self.rung;
        b[9..11].copy_from_slice(&self.visits.to_le_bytes());
        b
    }

    /// Decode from the 16-byte value slot (LE). Total — every 16-byte pattern
    /// is a readable stamp; the reserved tail is ignored, not validated.
    #[must_use]
    pub fn from_le_slot(b: &[u8; ALPHA_STAMP_BYTES]) -> Self {
        let mut c = [0u8; 4];
        c.copy_from_slice(&b[0..4]);
        let mut s = [0u8; 4];
        s.copy_from_slice(&b[4..8]);
        Self {
            cycle: u32::from_le_bytes(c),
            seq: u32::from_le_bytes(s),
            rung: b[8],
            visits: u16::from_le_bytes([b[9], b[10]]),
        }
    }
}

/// Read the [`AlphaStamp`] out of an overlay row's value slab.
#[must_use]
pub fn stamp_of(row: &NodeRow) -> AlphaStamp {
    let mut slot = [0u8; ALPHA_STAMP_BYTES];
    slot.copy_from_slice(&row.value[ALPHA_STAMP_OFFSET..ALPHA_STAMP_OFFSET + ALPHA_STAMP_BYTES]);
    AlphaStamp::from_le_slot(&slot)
}

/// Why a claim was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaError {
    /// The address is not in the allocation — attention cannot land where the
    /// spine has no address. Refused rather than silently extended: an overlay
    /// that can mint its own addresses is a second spine, not an overlay.
    Unallocated(AlphaAddr),
}

impl std::fmt::Display for AlphaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unallocated(a) => write!(f, "address {a:?} is not in the alpha allocation"),
        }
    }
}

impl std::error::Error for AlphaError {}

/// The outcome of a [`claim`](AlphaOverlay::claim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphaClaim {
    /// The claim's position in the saccade.
    pub seq: u32,
    /// `true` when this call materialised the row; `false` on a revisit, where
    /// the first visit's stamp is kept untouched (attention returning does not
    /// rewrite where it had been).
    pub fresh: bool,
}

/// A population of base ordinals, as words — the execution currency of every
/// set question over the overlay.
///
/// # Reads like WHERE, executes like MASK
///
/// This is lance-graph-java's mask-native invariant applied to the alpha
/// plane: *"a `long[]` of selected row IDs is still a materialised
/// population."* The readings below ([`AlphaOverlay::scanpath`],
/// [`AlphaOverlay::unattended`], [`AlphaOverlay::regressions`]) stay as
/// projections — order and visit counts are genuinely sequential facts a
/// mask cannot carry — but a set QUESTION (which addresses; the
/// expected⊕observed diff; a frontier intersection) executes here, as word
/// ops over base ordinals, never as address-set algebra.
///
/// The diff that motivated this is the cold/hot one, and it is pure algebra —
/// no bespoke method:
///
/// ```text
/// expected.and_not(&attended)   // should have fired, did not  ← the payoff
/// attended.and_not(&expected)   // fired off-book
/// expected.xor(&attended)       // both, as one surprise set
/// ```
///
/// The cold side arrives via [`AlphaAllocation::mask_of`] from an
/// independently-computed route (an ontology walk over the read-only spine),
/// the hot side via [`AlphaOverlay::attended_mask`]. The decorator boundary
/// survives untouched: the walk still never reads alpha — the diff happens
/// HERE, above both planes, on two masks that each side produced blind.
///
/// # The one named materializer
///
/// Per the same law, no unnamed materializer exists: the only way ordinals
/// leave mask form is [`AlphaMask::materialize_ordinals`], O(n) and named as
/// such.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlphaMask {
    words: Box<[u64]>,
    /// Valid bit count. Bits at and past `len` are PHANTOM and every op that
    /// could raise them ([`Self::not`]) must clear them — a complement that
    /// forgets the tail word invents up to 63 addresses the spine never had.
    len: u32,
}

impl AlphaMask {
    /// An all-zero mask over `len` ordinals.
    #[must_use]
    pub fn empty(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)].into_boxed_slice(),
            len: u32::try_from(len).unwrap_or(u32::MAX),
        }
    }

    fn set(&mut self, ordinal: u32) {
        if ordinal < self.len {
            self.words[(ordinal / 64) as usize] |= 1u64 << (ordinal % 64);
        }
    }

    /// Whether `ordinal` is in the population.
    #[must_use]
    pub fn contains(&self, ordinal: u32) -> bool {
        ordinal < self.len && (self.words[(ordinal / 64) as usize] >> (ordinal % 64)) & 1 == 1
    }

    /// Population size — one popcount sweep, no materialization.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Whether the population is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// How many ordinals the mask ranges over (NOT how many are set).
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    fn zip(&self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Self {
        debug_assert_eq!(self.len, other.len, "masks from different allocations");
        Self {
            words: self
                .words
                .iter()
                .zip(other.words.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
            len: self.len,
        }
    }

    /// Set intersection.
    #[must_use]
    pub fn and(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a & b)
    }

    /// Set union.
    #[must_use]
    pub fn or(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a | b)
    }

    /// Symmetric difference — the two-sided surprise set.
    #[must_use]
    pub fn xor(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a ^ b)
    }

    /// `self` minus `other` — the one-sided diff each cold/hot question is.
    #[must_use]
    pub fn and_not(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a & !b)
    }

    /// Complement WITHIN the allocation — the tail word's phantom bits are
    /// cleared, because a complement that forgets them invents addresses.
    #[must_use]
    pub fn not(&self) -> Self {
        let mut out = Self {
            words: self.words.iter().map(|&w| !w).collect(),
            len: self.len,
        };
        let tail = u64::from(self.len % 64);
        if tail != 0 {
            if let Some(last) = out.words.last_mut() {
                *last &= (1u64 << tail) - 1;
            }
        }
        out
    }

    /// **The named materializer** — ordinals out, ascending. O(n), and the
    /// only exit from mask form, per the no-unnamed-materializer rule.
    #[must_use]
    pub fn materialize_ordinals(&self) -> Vec<u32> {
        (0..self.len).filter(|&o| self.contains(o)).collect()
    }
}

/// The **address space** of an overlay, derived from a base spine.
///
/// Holds no rows. The [`HashSet`] is a lookup index over keys the base spine
/// already owns — the allocation itself costs nothing on disk and is never
/// written: it IS the base dataset's key column, read through a borrow.
pub struct AlphaAllocation<'a> {
    base: &'a [NodeRow],
    addrs: HashSet<AlphaAddr>,
    /// The addr → base-ordinal projection, built **lazily, once**.
    ///
    /// The base slice is immutable for the allocation's whole lifetime, so
    /// this index can never go stale — which is exactly what licenses caching
    /// it. The MUTABLE readings (which ordinals are claimed) are never cached;
    /// they are recomputed from the overlay on demand. Cache the immutable
    /// projection, recompute the mutable reading — that split is the whole
    /// invalidation story, because it leaves nothing to invalidate.
    ///
    /// `OnceLock`, not `OnceCell`: the extra `Sync` costs nothing here and
    /// keeps the allocation usable behind a shared reference in async
    /// contexts (`write_alpha_overlay` already crosses an `await`).
    ordinals: OnceLock<HashMap<AlphaAddr, u32>>,
}

impl<'a> AlphaAllocation<'a> {
    /// Allocate over an already-baked spine — e.g.
    /// `medcare_cohorts::obo_store::store().node_rows()`.
    #[must_use]
    pub fn over(base: &'a [NodeRow]) -> Self {
        let addrs = base.iter().map(|r| r.key).collect();
        Self {
            base,
            addrs,
            ordinals: OnceLock::new(),
        }
    }

    /// The base-slice ordinal of `addr`, or [`None`] for a foreign address.
    ///
    /// The ordinal is the address's **position in the base slice** — the one
    /// canonical, re-derivable coordinate this allocation has. It is NOT an
    /// insertion order and NOT a claim order; those are properties of an
    /// overlay, and an ordinal that depended on either would make every mask
    /// meaningless across overlays.
    #[must_use]
    pub fn ordinal(&self, addr: AlphaAddr) -> Option<u32> {
        self.ordinal_index().get(&addr).copied()
    }

    fn ordinal_index(&self) -> &HashMap<AlphaAddr, u32> {
        self.ordinals.get_or_init(|| {
            self.base
                .iter()
                .enumerate()
                .map(|(i, r)| (r.key, u32::try_from(i).unwrap_or(u32::MAX)))
                .collect()
        })
    }

    /// A mask with exactly the given addresses set — how a COLD expected
    /// route enters the mask algebra.
    ///
    /// # Errors
    /// [`AlphaError::Unallocated`] on the first foreign address, mirroring
    /// [`AlphaOverlay::claim`]'s refusal: an expected route naming an address
    /// the spine does not carry is a producer bug, and folding it into an
    /// empty bit would make the later diff silently wrong instead of loudly
    /// refused.
    pub fn mask_of(
        &self,
        addrs: impl IntoIterator<Item = AlphaAddr>,
    ) -> Result<AlphaMask, AlphaError> {
        let mut m = AlphaMask::empty(self.base.len());
        for a in addrs {
            match self.ordinal(a) {
                Some(o) => m.set(o),
                None => return Err(AlphaError::Unallocated(a)),
            }
        }
        Ok(m)
    }

    /// How many addresses exist. Never how many rows the overlay holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.addrs.len()
    }

    /// Whether the allocation is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.addrs.is_empty()
    }

    /// Whether `addr` is an address of this spine.
    #[must_use]
    pub fn contains(&self, addr: AlphaAddr) -> bool {
        self.addrs.contains(&addr)
    }

    /// The base rows, borrowed — read-only, by construction.
    #[must_use]
    pub fn base(&self) -> &'a [NodeRow] {
        self.base
    }
}

/// The **ephemeral overlay**: the claimed rows and nothing else.
///
/// Dropping it discards the whole channel — that is the point, not a caveat.
pub struct AlphaOverlay<'a> {
    alloc: AllocRef<'a>,
    claimed: Vec<NodeRow>,
    at: HashMap<AlphaAddr, usize>,
    cycle: u32,
}

/// How an overlay holds its allocation: **owned** (one overlay, one allocation)
/// or **borrowed** (many overlays, ONE allocation).
///
/// The borrowed arm exists for the parallel fan-out
/// ([`crate::alpha_tunnel`]): ten lanes over the same spine must not mean ten
/// copies of the address set. Allocating is defined as costing *zero rows* —
/// ten `HashSet`s of every address in the base would contradict that in the
/// one place it is supposed to hold.
///
/// Every read goes through [`Deref`], so `AlphaOverlay`'s own code is unchanged
/// and cannot tell the two arms apart — which is what keeps this additive.
enum AllocRef<'a> {
    Owned(AlphaAllocation<'a>),
    Borrowed(&'a AlphaAllocation<'a>),
}

impl<'a> std::ops::Deref for AllocRef<'a> {
    type Target = AlphaAllocation<'a>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(a) => a,
            Self::Borrowed(a) => a,
        }
    }
}

impl<'a> AlphaOverlay<'a> {
    /// A fresh, fully-unclaimed overlay that **borrows** a shared allocation.
    ///
    /// The sibling of [`new`](Self::new) for the fan-out: many overlays, one
    /// allocation, no copy of the address space.
    #[must_use]
    pub fn over_shared(alloc: &'a AlphaAllocation<'a>, cycle: u32) -> Self {
        Self {
            alloc: AllocRef::Borrowed(alloc),
            claimed: Vec::new(),
            at: HashMap::new(),
            cycle,
        }
    }

    /// A fresh, fully-unclaimed overlay over `alloc` for one thinking `cycle`.
    #[must_use]
    pub fn new(alloc: AlphaAllocation<'a>, cycle: u32) -> Self {
        Self {
            alloc: AllocRef::Owned(alloc),
            claimed: Vec::new(),
            at: HashMap::new(),
            cycle,
        }
    }

    /// Materialise the row at `addr` — attention landed here.
    ///
    /// The row's `key` is copied verbatim from the base row (byte-identical
    /// address, no re-mint), its `edges` stay zeroed (Posten 4a), and value
    /// slot 0 gets the [`AlphaStamp`].
    ///
    /// # Errors
    /// [`AlphaError::Unallocated`] when `addr` is not an address of the base
    /// spine. A revisit is **not** an error: it returns the first visit's `seq`
    /// with `fresh: false` and leaves the stored row untouched.
    pub fn claim(&mut self, addr: AlphaAddr, rung: u8) -> Result<AlphaClaim, AlphaError> {
        if let Some(&i) = self.at.get(&addr) {
            // A regression. The first visit's seq/rung/cycle are NOT rewritten —
            // attention returning does not change where it had been — but the
            // return itself is counted, because that is the diagnosis.
            let mut st = stamp_of(&self.claimed[i]);
            st.visits = st.visits.saturating_add(1);
            let slot = st.to_le_slot();
            self.claimed[i].value[ALPHA_STAMP_OFFSET..ALPHA_STAMP_OFFSET + ALPHA_STAMP_BYTES]
                .copy_from_slice(&slot);
            return Ok(AlphaClaim {
                seq: st.seq,
                fresh: false,
            });
        }
        if !self.alloc.contains(addr) {
            return Err(AlphaError::Unallocated(addr));
        }
        let seq = u32::try_from(self.claimed.len()).unwrap_or(u32::MAX);
        let mut row = NodeRow {
            key: addr,
            edges: crate::canonical_node::EdgeBlock::default(),
            value: [0u8; 480],
        };
        let stamp = AlphaStamp {
            cycle: self.cycle,
            seq,
            rung,
            visits: 1,
        };
        row.value[ALPHA_STAMP_OFFSET..ALPHA_STAMP_OFFSET + ALPHA_STAMP_BYTES]
            .copy_from_slice(&stamp.to_le_slot());
        self.at.insert(addr, self.claimed.len());
        self.claimed.push(row);
        Ok(AlphaClaim { seq, fresh: true })
    }

    /// Claim a whole saccade path in visit order — the trajectory IS the index.
    ///
    /// # Errors
    /// The first [`AlphaError::Unallocated`]; claims made before it stand (the
    /// overlay is a record of what happened, not a transaction).
    pub fn claim_path(
        &mut self,
        addrs: impl IntoIterator<Item = AlphaAddr>,
        rung: u8,
    ) -> Result<usize, AlphaError> {
        let mut fresh = 0usize;
        for a in addrs {
            if self.claim(a, rung)?.fresh {
                fresh += 1;
            }
        }
        Ok(fresh)
    }

    /// The overlay row at `addr`, or [`None`] — **„not attended"**.
    ///
    /// Never falls back to the base row.
    #[must_use]
    pub fn get(&self, addr: AlphaAddr) -> Option<&NodeRow> {
        self.at.get(&addr).map(|&i| &self.claimed[i])
    }

    /// How many rows are materialised.
    #[must_use]
    pub fn claimed_len(&self) -> usize {
        self.claimed.len()
    }

    /// How many addresses are allocated.
    #[must_use]
    pub fn allocated_len(&self) -> usize {
        self.alloc.len()
    }

    /// The claimed rows in saccade order — the overlay's whole content.
    #[must_use]
    pub fn rows(&self) -> &[NodeRow] {
        &self.claimed
    }

    /// The allocation this overlay sits on.
    #[must_use]
    pub fn allocation(&self) -> &AlphaAllocation<'a> {
        &self.alloc
    }

    /// The HOT mask: which base ordinals attention has claimed, recomputed
    /// on demand (the mutable reading is never cached — see
    /// [`AlphaAllocation`]'s field doc for the split that licenses caching
    /// only the immutable projection).
    ///
    /// One half of the cold⊕hot diff; the other half comes from
    /// [`AlphaAllocation::mask_of`] over an independently-computed route.
    /// "Independently" is load-bearing: an expected mask derived from the
    /// same traversal that claimed would make expected ≡ observed by
    /// construction and the diff vacuously empty. The parity test pins the
    /// mask against the address-keyed reading so the two representations
    /// cannot drift.
    #[must_use]
    pub fn attended_mask(&self) -> AlphaMask {
        let mut m = AlphaMask::empty(self.alloc.base().len());
        for addr in self.at.keys() {
            if let Some(o) = self.alloc.ordinal(*addr) {
                m.set(o);
            }
        }
        m
    }

    // ── the debugger surface ────────────────────────────────────────────────
    //
    // The overlay is a scanpath recorder, so reading it back IS a thinking
    // debugger: not "what did the code do" (a log answers that) but "where did
    // attention go, in what order, what did it return to — and what did it
    // never look at". The last one is the question no log can answer, because
    // absence leaves no line.

    /// The **scanpath**: the visited addresses in visit order.
    ///
    /// `seq` is the index, so this is simply the claimed rows in order — the
    /// replayable trajectory of one thought.
    pub fn scanpath(&self) -> impl Iterator<Item = AlphaAddr> + '_ {
        self.claimed.iter().map(|r| r.key)
    }

    /// The **blind spot**: allocated addresses attention never landed on.
    ///
    /// This is the half a log cannot report. A thought that reached the wrong
    /// conclusion is often not wrong in what it looked at but in what it never
    /// looked at, and that set is only nameable because the address space is
    /// known independently of the visits.
    pub fn unattended(&self) -> impl Iterator<Item = AlphaAddr> + '_ {
        self.alloc
            .base()
            .iter()
            .map(|r| r.key)
            .filter(move |a| !self.at.contains_key(a))
    }

    /// The **regressions**: addresses attention came back to, with their visit
    /// counts, most-revisited first.
    ///
    /// In reading research a regression marks the point where integration
    /// failed the first time. Here it marks the address a thought could not
    /// settle — the first place to look when a conclusion is wrong.
    #[must_use]
    pub fn regressions(&self) -> Vec<(AlphaAddr, u16)> {
        let mut v: Vec<(AlphaAddr, u16)> = self
            .claimed
            .iter()
            .map(|r| (r.key, stamp_of(r).visits))
            .filter(|(_, n)| *n > 1)
            .collect();
        // Absteigend nach Besuchszahl: der Ort, an den die Aufmerksamkeit am
        // oeftesten zurueckkam, steht vorn (in der Leseforschung das
        // diagnostischste Ereignis).
        v.sort_by_key(|&(_, n)| core::cmp::Reverse(n));
        v
    }

    /// What THIS thought looked at and `other` did not — the diff between two
    /// scanpaths over the same spine.
    ///
    /// Comparing two overlays is the debugging move the channel makes cheap:
    /// two thoughts, one address space, and the difference is a set operation
    /// rather than a re-run.
    pub fn only_in<'b>(&'b self, other: &'b Self) -> impl Iterator<Item = AlphaAddr> + 'b {
        self.claimed
            .iter()
            .map(|r| r.key)
            .filter(move |a| other.get(*a).is_none())
    }

    /// Discard the overlay whole. Costs a re-search, never correctness — which
    /// is why there is no digest and no `bakes.tsv` row to keep in step.
    pub fn discard(self) {
        drop(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic 5-row spine for the mask falsifiers — deliberately NOT the
    /// OBO bake (empty after a container reset) and deliberately `n % 64 != 0`,
    /// because the phantom-bit falsifier is only a falsifier when the tail
    /// word has bits past `len` to wrongly raise.
    fn tiny_base() -> Vec<NodeRow> {
        (0u32..5)
            .map(|i| NodeRow {
                key: NodeGuid::new(0x0B0B_0000 + i, 1, 2, 3, 0x11, i + 1),
                edges: crate::canonical_node::EdgeBlock::default(),
                value: [0u8; 480],
            })
            .collect()
    }

    #[test]
    fn the_mask_reading_and_the_address_reading_cannot_drift() {
        let rows = tiny_base();
        let alloc = AlphaAllocation::over(&rows);
        let mut ov = AlphaOverlay::new(alloc, 1);
        ov.claim(rows[0].key, 1).unwrap();
        ov.claim(rows[3].key, 1).unwrap();

        let attended = ov.attended_mask();
        // mask → addrs must equal the address-keyed reading, both ways.
        let via_mask: HashSet<AlphaAddr> = attended
            .materialize_ordinals()
            .into_iter()
            .map(|o| rows[o as usize].key)
            .collect();
        let via_addrs: HashSet<AlphaAddr> = ov.scanpath().collect();
        assert_eq!(
            via_mask, via_addrs,
            "attended: mask and address reading drifted"
        );

        let un_via_mask: HashSet<AlphaAddr> = attended
            .not()
            .materialize_ordinals()
            .into_iter()
            .map(|o| rows[o as usize].key)
            .collect();
        let un_via_addrs: HashSet<AlphaAddr> = ov.unattended().collect();
        assert_eq!(
            un_via_mask, un_via_addrs,
            "unattended: mask and address reading drifted"
        );
    }

    /// The complement stays INSIDE the allocation. A `not()` that forgets to
    /// clear the tail word raises up to 63 phantom bits past `len` — here 59
    /// of them — and every downstream count/diff silently inflates.
    #[test]
    fn a_complement_never_invents_phantom_addresses() {
        let rows = tiny_base();
        let alloc = AlphaAllocation::over(&rows);
        let mut ov = AlphaOverlay::new(alloc, 1);
        ov.claim(rows[0].key, 1).unwrap();
        ov.claim(rows[1].key, 1).unwrap();

        let un = ov.attended_mask().not();
        assert_eq!(
            un.count(),
            3,
            "5 allocated - 2 claimed = 3; more means phantom bits"
        );
        assert_eq!(un.materialize_ordinals(), vec![2, 3, 4]);
    }

    /// The cold⊕hot diff, two-sided: missed-fire AND off-book-fire must both
    /// be visible, and an identical pair must diff to empty — otherwise the
    /// diff is a guard that fires on everything, which carries exactly as
    /// much information as one that never fires.
    #[test]
    fn the_cold_hot_diff_is_two_sided() {
        let rows = tiny_base();
        let alloc = AlphaAllocation::over(&rows);
        // COLD: the expected route, built WITHOUT touching the overlay.
        let expected = alloc
            .mask_of([rows[0].key, rows[1].key, rows[2].key])
            .unwrap();
        // HOT: what actually fired — B held, D fired off-book.
        let mut ov = AlphaOverlay::new(alloc, 1);
        ov.claim(rows[0].key, 1).unwrap();
        ov.claim(rows[1].key, 1).unwrap();
        ov.claim(rows[3].key, 1).unwrap();
        let attended = ov.attended_mask();

        let missed = expected.and_not(&attended);
        assert_eq!(
            missed.materialize_ordinals(),
            vec![2],
            "C should-have-fired-and-did-not"
        );
        let off_book = attended.and_not(&expected);
        assert_eq!(off_book.materialize_ordinals(), vec![3], "D fired off-book");
        assert_eq!(
            expected.xor(&attended).count(),
            2,
            "the union of both surprises"
        );

        // The silence half: identical expected/observed must diff to EMPTY.
        let same = attended.xor(&attended);
        assert!(same.is_empty(), "identical masks must produce no surprise");
    }

    /// An expected route naming an address the spine does not carry is
    /// refused, mirroring `claim`'s refusal — folding it into an unset bit
    /// would make the later diff silently wrong instead of loudly wrong.
    #[test]
    fn an_expected_route_naming_a_foreign_address_is_refused() {
        let rows = tiny_base();
        let alloc = AlphaAllocation::over(&rows);
        let foreign = NodeGuid::new(
            0xDEAD_BEEF,
            0xFFFF,
            0xFFFF,
            0xFFFF,
            0x00FF_FFFF,
            0x00FF_FFFF,
        );
        assert!(
            !alloc.contains(foreign),
            "premise: the address must be foreign"
        );
        assert_eq!(
            alloc.mask_of([rows[0].key, foreign]),
            Err(AlphaError::Unallocated(foreign))
        );
    }

    /// The ordinal is the BASE POSITION — not an insertion order, not a claim
    /// order. Claiming in reverse must leave every ordinal where the base
    /// slice puts it, or masks from two overlays over the same spine stop
    /// being comparable and the whole algebra is meaningless.
    #[test]
    fn the_ordinal_is_the_base_position_not_a_claim_order() {
        let rows = tiny_base();
        let alloc = AlphaAllocation::over(&rows);
        let mut ov = AlphaOverlay::new(alloc, 1);
        for r in rows.iter().rev() {
            ov.claim(r.key, 1).unwrap();
        }
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(
                ov.allocation().ordinal(r.key),
                Some(u32::try_from(i).unwrap()),
                "ordinal of row {i} must be its base position, claim order be damned"
            );
        }
        // …and the lazily-built index answers identically on a repeat read.
        assert_eq!(ov.allocation().ordinal(rows[2].key), Some(2));
    }
}

#[cfg(test)]
mod claim_semantics {
    use super::*;

    /// A 200-row synthetic spine — big enough that "unattended is the
    /// overwhelming majority" is a real bound, not a rounding artifact.
    fn base() -> Vec<NodeRow> {
        (0u32..200)
            .map(|i| NodeRow {
                key: NodeGuid::new(0x0A0A_0000 + i, 4, 5, 6, 0x33, i + 1),
                edges: crate::canonical_node::EdgeBlock::default(),
                value: [0u8; 480],
            })
            .collect()
    }

    /// Thin provisioning: an allocated-but-unclaimed address reads as
    /// NOT ATTENDED (`None`), never as a plausible value from the base row.
    #[test]
    fn an_unclaimed_address_reads_as_not_attended() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut ov = AlphaOverlay::new(alloc, 7);
        let path: Vec<AlphaAddr> = b.iter().take(64).map(|r| r.key).collect();
        assert_eq!(
            ov.claim_path(path.iter().copied(), 2).expect("allocated"),
            64
        );
        assert!(ov.get(path[0]).is_some(), "claimed = attended");
        let unclaimed = b
            .iter()
            .skip(64)
            .filter(|r| ov.get(r.key).is_none())
            .count();
        assert_eq!(unclaimed, 136, "everything unclaimed reads None");
        assert_eq!(
            ov.claimed_len(),
            64,
            "claiming materialises ONLY the visited rows"
        );
    }

    /// A foreign address is refused; the claims made before it stand (the
    /// overlay is a record, not a transaction).
    #[test]
    fn claiming_an_unallocated_address_is_refused() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut ov = AlphaOverlay::new(alloc, 1);
        let foreign = NodeGuid::new(
            0xDEAD_BEEF,
            0xFFFF,
            0xFFFF,
            0xFFFF,
            0x00FF_FFFF,
            0x00FF_FFFF,
        );
        assert!(
            !ov.allocation().contains(foreign),
            "premise: genuinely foreign"
        );
        assert!(ov.claim(b[0].key, 3).expect("allocated").fresh);
        assert!(matches!(
            ov.claim(foreign, 3),
            Err(AlphaError::Unallocated(_))
        ));
        assert_eq!(ov.claimed_len(), 1, "the prior claim stands");
    }

    /// A revisit keeps the first stamp (seq/rung/cycle) and counts the return
    /// in `visits` — attention returning does not rewrite where it had been.
    #[test]
    fn a_revisit_keeps_the_first_stamp_and_counts_the_return() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut ov = AlphaOverlay::new(alloc, 9);
        assert!(ov.claim(b[3].key, 2).expect("fresh").fresh);
        let again = ov.claim(b[3].key, 5).expect("revisit");
        assert!(!again.fresh);
        assert_eq!(ov.claimed_len(), 1, "the table did not grow");
        let st = stamp_of(ov.get(b[3].key).expect("attended"));
        assert_eq!(st.rung, 2, "first rung kept, not rewritten to 5");
        assert_eq!(st.visits, 2, "the return is counted");
        assert_eq!(st.cycle, 9);
    }
}
