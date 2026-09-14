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
#[derive(Clone, Debug)]
pub struct AlphaMask {
    words: Box<[u64]>,
    /// Valid bit count. Bits at and past `len` are PHANTOM and every op that
    /// could raise them ([`Self::not`]) must clear them — a complement that
    /// forgets the tail word invents up to 63 addresses the spine never had.
    len: u32,
}

/// Equality is over the POPULATION, never the representation: `len` and the
/// words with the tail word masked. A phantom bit a raw writer left past
/// `len` (see [`AlphaMask::words_mut`]) does not make two equal populations
/// compare unequal — the same rule `WideFieldMask` enforces with its
/// canonical `PartialEq`/`Hash`.
impl PartialEq for AlphaMask {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.canonical_words().eq(other.canonical_words())
    }
}

impl Eq for AlphaMask {}

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
        self.canonical_words().map(|w| w.count_ones()).sum()
    }

    /// Whether the population is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.canonical_words().all(|w| w == 0)
    }

    /// The mask for the last word: every bit below `len % 64`, or all ones
    /// when `len` is a multiple of 64 (no phantom positions). The ONE
    /// spelling of the tail law in this file — `not`, `not_assign`,
    /// `from_words`, `clear_tail`, the readers and `PartialEq` all go
    /// through it, so it cannot drift between them.
    fn tail_mask(&self) -> u64 {
        let tail = self.len % 64;
        if tail == 0 {
            u64::MAX
        } else {
            (1u64 << tail) - 1
        }
    }

    /// The words with the tail word masked — what the population IS,
    /// regardless of what a raw writer left past `len`. Readers
    /// (`count`, `is_empty`, `PartialEq`) go through this so a phantom bit
    /// raised through [`Self::words_mut`] can never inflate a count or make
    /// two equal populations compare unequal; [`Self::words`] stays the raw
    /// view for the SIMD seam, which conforming inputs keep zero anyway.
    fn canonical_words(&self) -> impl Iterator<Item = u64> + '_ {
        let last = self.words.len().saturating_sub(1);
        let mask = self.tail_mask();
        self.words
            .iter()
            .enumerate()
            .map(move |(i, &w)| if i == last { w & mask } else { w })
    }

    /// How many ordinals the mask ranges over (NOT how many are set).
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Word-wise combine. **`assert_eq!`, never `debug_assert_eq!`** — the
    /// guard has to hold in release, because that is the build where the
    /// damage is silent.
    ///
    /// Iterator `zip` truncates to the shorter operand while `len` is copied
    /// from `self`, so a mismatch that gets past this line produces a mask
    /// claiming `self.len` addresses over `other.words.len()` words. That is
    /// not a smaller mask, it is an INVALID one, and it fails two ways at a
    /// distance: `count`/`is_empty` under-report silently, and
    /// `contains`/`materialize_ordinals` index past the slice and panic
    /// somewhere far from the call that caused it.
    ///
    /// A length mismatch is a caller mixing two allocations — a programming
    /// error, not a data condition — so it fails closed and immediately,
    /// at the operation that made it, rather than becoming a wrong answer.
    /// The cost is one `u32` comparison against a loop over every word.
    fn zip(&self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Self {
        assert_eq!(self.len, other.len, "masks from different allocations");
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
        out.clear_tail();
        out
    }

    // ── In-place algebra (2026-09-13) ──────────────────────────────────
    //
    // The allocating forms above (`and`/`or`/`xor`/`and_not`/`not`) return a
    // FRESH `Box<[u64]>` per operation — the ownership direction the V3 law
    // forbids: the mailbox/allocation owns the address space and its resident
    // masks; an operation mutates or borrows those planes, it never mints a
    // second one. They are kept as-is (load-bearing for every existing
    // caller) and the forms below are the ones a fused mask program uses.
    // The contract stays zero-dep: these are plain word loops; a consumer
    // with SIMD lowers through `words()`/`words_mut()` onto `ndarray::simd`.

    /// `self &= other`, in place — no allocation. Same allocation-mismatch
    /// law as [`Self::and`]: refuses two masks of different `len`.
    pub fn and_assign(&mut self, other: &Self) {
        assert_eq!(self.len, other.len, "masks from different allocations");
        for (d, &s) in self.words.iter_mut().zip(other.words.iter()) {
            *d &= s;
        }
    }

    /// `self |= other`, in place — no allocation. Unlike `and`, OR and XOR
    /// PROPAGATE a phantom tail bit from a non-conforming operand (one a
    /// raw writer left past `len` through [`Self::words_mut`]), so the tail
    /// is cleared afterwards — one masked AND on the last word.
    pub fn or_assign(&mut self, other: &Self) {
        assert_eq!(self.len, other.len, "masks from different allocations");
        for (d, &s) in self.words.iter_mut().zip(other.words.iter()) {
            *d |= s;
        }
        self.clear_tail();
    }

    /// `self ^= other`, in place — no allocation. Tail cleared for the same
    /// reason as [`Self::or_assign`].
    pub fn xor_assign(&mut self, other: &Self) {
        assert_eq!(self.len, other.len, "masks from different allocations");
        for (d, &s) in self.words.iter_mut().zip(other.words.iter()) {
            *d ^= s;
        }
        self.clear_tail();
    }

    /// `self &= !other`, in place — no allocation. The result is a subset of
    /// the prior `self`, so a conforming tail stays conforming.
    pub fn and_not_assign(&mut self, other: &Self) {
        assert_eq!(self.len, other.len, "masks from different allocations");
        for (d, &s) in self.words.iter_mut().zip(other.words.iter()) {
            *d &= !s;
        }
    }

    /// Complement in place, tail cleared — the in-place [`Self::not`].
    pub fn not_assign(&mut self) {
        for w in self.words.iter_mut() {
            *w = !*w;
        }
        self.clear_tail();
    }

    /// Clear every bit — reuse this allocation as fresh scratch instead of
    /// minting another.
    pub fn clear(&mut self) {
        for w in self.words.iter_mut() {
            *w = 0;
        }
    }

    /// The packed words, mutably — the seam a SIMD consumer WRITES through
    /// (`ndarray::simd::mask_ternlog_assign` takes `&mut [u64]`). The caller
    /// owes the tail invariant: bits at and past `len()` must be left zero.
    /// Every `ndarray::simd` mask writer honours it for conforming inputs
    /// and even ternlog tables; a caller composing an odd table clears the
    /// tail itself ([`Self::clear_tail`]).
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    /// Re-establish the tail invariant after an external writer that may
    /// have raised phantom bits (an odd-table ternlog, a hand-built word).
    pub fn clear_tail(&mut self) {
        let mask = self.tail_mask();
        if let Some(last) = self.words.last_mut() {
            *last &= mask;
        }
    }

    /// **The named materializer** — ordinals out, ascending. O(n), and the
    /// only exit from mask form, per the no-unnamed-materializer rule.
    #[must_use]
    pub fn materialize_ordinals(&self) -> Vec<u32> {
        (0..self.len).filter(|&o| self.contains(o)).collect()
    }

    /// The packed words, one bit per ordinal, tail bits beyond `len` zero.
    ///
    /// A BORROW, not a materializer: the words are the mask. This is the seam
    /// a crate with SIMD (ndarray's `mask_ternlog_assign` takes `&[u64]`) reads
    /// through; the contract itself stays zero-dep.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Rebuild a mask from words produced outside the contract (an `eq_*_to_mask`
    /// sweep, a ternlog result) over an allocation of `len` addresses.
    ///
    /// Refuses, in every build, a word count that does not match `len`
    /// (`words.len() != len.div_ceil(64)` is a caller mixing allocations — the
    /// same law `zip` enforces). Tail bits at and past `len` are CLEARED, never
    /// trusted: a sweep that wrote the phantom tail would otherwise invent up to
    /// 63 addresses the spine never had (the [`Self::not`] rule, applied at the
    /// boundary).
    #[must_use]
    pub fn from_words(words: Box<[u64]>, len: u32) -> Self {
        assert_eq!(
            words.len(),
            (len as usize).div_ceil(64),
            "word count does not match the allocation length"
        );
        let mut mask = Self { words, len };
        mask.clear_tail();
        mask
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
    /// The release-mode length guard. `zip` once guarded its operands with
    /// `debug_assert_eq!`, which is COMPILED OUT of a release build — so a
    /// mismatch reached the iterator `zip`, which truncates to the shorter
    /// operand while `len` was copied from `self`. The product claimed
    /// `self.len` addresses over `other.words.len()` words, which is not a
    /// smaller mask but an INVALID one, and it fails in two different ways
    /// depending on which reading runs first: `count`/`is_empty` silently
    /// under-report, and `contains`/`materialize_ordinals` index past the
    /// slice and panic somewhere far from the call that caused it.
    ///
    /// Two-sided on purpose. The panic half proves the guard fires in
    /// release; [`equal_length_masks_combine_without_panicking`] proves it
    /// stays silent on the ordinary case, so a guard that simply rejected
    /// everything could not pass both.
    #[test]
    #[should_panic(expected = "masks from different allocations")]
    fn combining_masks_of_different_lengths_is_refused_in_every_build() {
        // 200 vs 64: different word counts (4 vs 1), so the truncation is
        // reachable rather than hidden by both operands rounding to one word.
        let wide = AlphaMask::empty(200);
        let narrow = AlphaMask::empty(64);
        let _ = wide.and(&narrow);
    }

    /// `from_words`'s own release-mode length guard — mirrors `zip`'s: a word
    /// count that does not match `len.div_ceil(64)` is a caller mixing
    /// allocations, refused loudly rather than folded into a wrong answer.
    #[test]
    #[should_panic(expected = "word count does not match the allocation length")]
    fn from_words_refuses_a_word_count_that_does_not_match_the_length() {
        // len=200 needs 4 words (200.div_ceil(64) == 4); 3 is short.
        let _ = AlphaMask::from_words(vec![0u64; 3].into_boxed_slice(), 200);
    }

    /// The can-stay-silent half: a genuine word count for a non-multiple-of-64
    /// length round-trips exactly through `words()` / `from_words`.
    #[test]
    fn from_words_round_trips_words_for_a_length_that_is_not_a_multiple_of_64() {
        let mut m = AlphaMask::empty(200);
        m.set(7);
        m.set(130);
        m.set(199);

        assert_eq!(m.words().len(), 4, "200.div_ceil(64) == 4 words");
        let rebuilt = AlphaMask::from_words(m.words().to_vec().into_boxed_slice(), m.len());
        assert_eq!(rebuilt, m, "from_words(m.words(), m.len()) must round-trip");
    }

    /// Tail bits past `len` are CLEARED, never trusted — a sweep that wrote the
    /// phantom tail would otherwise invent addresses the spine never had.
    #[test]
    fn from_words_clears_phantom_tail_bits() {
        // len=200 -> 200 % 64 == 8, so only the low 8 bits of the 4th word are
        // real; the rest of that word plus everything past it is phantom.
        let m = AlphaMask::from_words(vec![u64::MAX; 4].into_boxed_slice(), 200);

        assert_eq!(m.count(), 200, "count must exclude the phantom tail bits");
        assert!(m.contains(199), "the last real ordinal must survive");
        assert!(!m.contains(200), "ordinal 200 is past len and must be gone");
        assert_eq!(
            m.words()[3],
            (1u64 << 8) - 1,
            "200 % 64 == 8; only the low 8 bits of the tail word are real"
        );
    }

    /// The silence half of the guard above — equal lengths must still work,
    /// including the `len % 64 != 0` case where the tail word is partial.
    #[test]
    fn equal_length_masks_combine_without_panicking() {
        let mut a = AlphaMask::empty(200);
        let mut b = AlphaMask::empty(200);
        a.set(7);
        a.set(130);
        b.set(130);
        b.set(199);

        assert_eq!(a.and(&b).materialize_ordinals(), vec![130], "and");
        assert_eq!(a.or(&b).materialize_ordinals(), vec![7, 130, 199], "or");
        assert_eq!(a.xor(&b).materialize_ordinals(), vec![7, 199], "xor");
        assert_eq!(a.and_not(&b).materialize_ordinals(), vec![7], "and_not");
        // Every product must carry a full-width word slice, not a truncated
        // one — the invariant the guard exists to keep.
        assert_eq!(a.and(&b).count() + a.and(&b).not().count(), 200);
    }

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

    // ── In-place algebra falsifiers (PR2, 2026-09-13) ──────────────────
    //
    // The crate is zero-dep, so a seeded xorshift64 stands in for `rand` —
    // deterministic across runs, and unlike a hand-picked bit pattern it
    // exercises the full range of bit positions, including ones straddling
    // a `len % 64 != 0` tail.

    /// Minimal deterministic PRNG. `next_u64` never returns a value whose
    /// choice depends on wall-clock time or any other run-to-run variable —
    /// same seed, same sequence, every run.
    struct Xorshift64(u64);
    impl Xorshift64 {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// A pseudo-random, already-conforming `len`-bit mask: every word is
    /// filled from the seeded stream, then `clear_tail()` re-establishes
    /// the tail invariant so callers start from a valid fixture. Uses
    /// `words_mut()`/`clear_tail()` to build the fixture — their OWN
    /// correctness is pinned separately below, so reusing them here is
    /// fixture plumbing, not circular verification of the ops under test.
    fn random_mask(len: usize, seed: u64) -> AlphaMask {
        let mut rng = Xorshift64(seed | 1); // xorshift is stuck at 0 if seeded with 0
        let mut m = AlphaMask::empty(len);
        for w in m.words_mut() {
            *w = rng.next_u64();
        }
        m.clear_tail();
        m
    }

    /// Are all bits at and past `m.len()` in the last word zero? The
    /// direct falsifier for the tail invariant — distinct from `contains`,
    /// which is len-guarded and can never see a phantom bit regardless of
    /// whether the invariant holds.
    fn tail_is_clear(m: &AlphaMask) -> bool {
        let tail = m.len() % 64;
        if tail == 0 {
            return true;
        }
        match m.words().last() {
            Some(&last) => last >> tail == 0,
            None => true,
        }
    }

    /// FAILS IF: any in-place op computes something other than its
    /// allocating twin — e.g. `and_assign` performing `|=` instead of
    /// `&=`, or leaving `self` untouched instead of combining. Spans
    /// lengths straddling a 64-bit word boundary on both sides (63/64/65)
    /// plus a multi-word 1000, so a bug that only shows up once a mask
    /// needs more than one word cannot hide behind a single-word fixture.
    #[test]
    fn in_place_algebra_matches_the_allocating_forms_across_sizes() {
        for (i, &len) in [1usize, 63, 64, 65, 130, 1000].iter().enumerate() {
            let a0 = random_mask(len, 0x1111_1111_1111_1111 ^ (i as u64));
            let b = random_mask(len, 0x9999_9999_9999_9999 ^ ((i as u64) * 7 + 1));

            let mut and_r = a0.clone();
            and_r.and_assign(&b);
            assert_eq!(and_r, a0.and(&b), "and_assign vs and, len={len}");

            let mut or_r = a0.clone();
            or_r.or_assign(&b);
            assert_eq!(or_r, a0.or(&b), "or_assign vs or, len={len}");

            let mut xor_r = a0.clone();
            xor_r.xor_assign(&b);
            assert_eq!(xor_r, a0.xor(&b), "xor_assign vs xor, len={len}");

            let mut andnot_r = a0.clone();
            andnot_r.and_not_assign(&b);
            assert_eq!(
                andnot_r,
                a0.and_not(&b),
                "and_not_assign vs and_not, len={len}"
            );

            let mut not_r = a0.clone();
            not_r.not_assign();
            assert_eq!(not_r, a0.not(), "not_assign vs not, len={len}");

            // Anti-vacuity: NOT always differs from its input (no single
            // bit equals its own complement), so this holds unconditionally.
            assert_ne!(not_r, a0, "not_assign did nothing, len={len}");
            // AND/OR/XOR/AND_NOT differing from `a0` is only GUARANTEED
            // for a random `b` once there is more than one bit to collide
            // on — at len==1 a genuinely random single bit has a real
            // chance of landing exactly where and/or/xor/and_not are
            // no-ops, which would make this specific assertion flaky
            // rather than falsifying. That length's "did something" claim
            // is instead pinned deterministically in
            // `in_place_ops_actually_change_the_mask` below.
            if len > 1 {
                assert_ne!(and_r, a0, "and_assign did nothing, len={len}");
                assert_ne!(or_r, a0, "or_assign did nothing, len={len}");
                assert_ne!(xor_r, a0, "xor_assign did nothing, len={len}");
                assert_ne!(andnot_r, a0, "and_not_assign did nothing, len={len}");
            }
        }
    }

    /// FAILS IF: any op stops mutating `self` — e.g. `and_assign` becomes
    /// a no-op, or copies `other` wholesale instead of combining. Hand-
    /// built (not random) so each op's "did something" claim holds by
    /// construction rather than by luck: ordinal 0 is in `a` only, 1 in
    /// `b` only, 2 in both.
    #[test]
    fn in_place_ops_actually_change_the_mask() {
        let mut a = AlphaMask::empty(8);
        a.set(0);
        a.set(2);
        let mut b = AlphaMask::empty(8);
        b.set(1);
        b.set(2);

        let mut and_r = a.clone();
        and_r.and_assign(&b);
        assert_ne!(and_r, a, "and_assign: ordinal 0 must drop out");
        assert_eq!(and_r.materialize_ordinals(), vec![2]);

        let mut or_r = a.clone();
        or_r.or_assign(&b);
        assert_ne!(or_r, a, "or_assign: ordinal 1 must appear");
        assert_eq!(or_r.materialize_ordinals(), vec![0, 1, 2]);

        let mut xor_r = a.clone();
        xor_r.xor_assign(&b);
        assert_ne!(
            xor_r, a,
            "xor_assign: ordinal 2 must cancel, ordinal 1 must appear"
        );
        assert_eq!(xor_r.materialize_ordinals(), vec![0, 1]);

        let mut andnot_r = a.clone();
        andnot_r.and_not_assign(&b);
        assert_ne!(andnot_r, a, "and_not_assign: ordinal 2 must drop out");
        assert_eq!(andnot_r.materialize_ordinals(), vec![0]);

        let mut not_r = a.clone();
        not_r.not_assign();
        assert_ne!(not_r, a, "not_assign must flip every real bit");
        assert_eq!(not_r.materialize_ordinals(), vec![1, 3, 4, 5, 6, 7]);
    }

    /// FAILS IF: any op leaves a bit set past `len` on a non-conforming
    /// length. From CONFORMING inputs `not_assign` is the one op that can
    /// raise a phantom bit (flipping the 62 phantom positions in a 130-bit
    /// mask's tail word); `and_assign`/`and_not_assign` cannot; `or_assign`/
    /// `xor_assign` cannot from conforming inputs but DO propagate a
    /// non-conforming operand's phantom bit, which is why they clear the
    /// tail — `or_and_xor_assign_clear_a_phantom_bit_a_non_conforming_operand_carries`
    /// below is the falsifier for that half. This test still checks all
    /// five, because a future refactor that shares code between them could
    /// regress any of them together.
    #[test]
    fn in_place_ops_keep_the_tail_clear_on_a_non_conforming_length() {
        let len = 130; // 130 % 64 == 2: 62 phantom positions in the tail word
        let a0 = random_mask(len, 0xDEAD_BEEF_DEAD_BEEF);
        let b = random_mask(len, 0xFEED_FACE_FEED_FACE);
        assert!(
            tail_is_clear(&a0) && tail_is_clear(&b),
            "fixtures must start conforming or this test measures nothing"
        );

        let mut and_r = a0.clone();
        and_r.and_assign(&b);
        assert!(tail_is_clear(&and_r), "and_assign left phantom tail bits");

        let mut or_r = a0.clone();
        or_r.or_assign(&b);
        assert!(tail_is_clear(&or_r), "or_assign left phantom tail bits");

        let mut xor_r = a0.clone();
        xor_r.xor_assign(&b);
        assert!(tail_is_clear(&xor_r), "xor_assign left phantom tail bits");

        let mut andnot_r = a0.clone();
        andnot_r.and_not_assign(&b);
        assert!(
            tail_is_clear(&andnot_r),
            "and_not_assign left phantom tail bits"
        );

        let mut not_r = a0.clone();
        not_r.not_assign();
        assert!(
            tail_is_clear(&not_r),
            "not_assign left phantom tail bits — the one op that must clear them itself"
        );
    }

    /// `words_mut()` is a raw write seam; nothing re-establishes the tail
    /// invariant automatically, and the RAW word carries what a writer left.
    /// The readers do not: `count()`/`is_empty()`/`PartialEq` mask the tail
    /// word, so a phantom bit can never inflate a population or split two
    /// equal masks. FAILS IF: `clear_tail()` stops masking the tail word
    /// (the raw word keeps bit 2 after the call), OR a reader stops masking
    /// (the phantom would count as a member).
    #[test]
    fn words_mut_needs_clear_tail_and_a_phantom_bit_is_observable_only_in_the_raw_word() {
        let mut m = AlphaMask::empty(130); // 130 % 64 == 2: word 2 has 62 phantom bits
        assert_eq!(m.count(), 0);

        // Hand-raise a phantom bit: ordinal 130 is one past `len`, bit 2 of
        // the 3rd (index-2) word.
        m.words_mut()[2] |= 1 << 2;

        // Can-fire half: the phantom bit IS in the raw word the SIMD seam
        // reads ...
        assert_eq!(
            m.words()[2] >> 2,
            1,
            "the hand-raised phantom bit must be visible in the raw word"
        );
        // ... and NOT in any population reading.
        assert_eq!(m.count(), 0, "count() must mask the tail word");
        assert!(m.is_empty(), "is_empty() must mask the tail word");
        assert!(!m.contains(130), "contains() stays len-guarded");
        assert_eq!(
            m,
            AlphaMask::empty(130),
            "equality is over the population, never the raw tail"
        );

        m.clear_tail();

        // Can-stay-silent half: clear_tail restores raw conformity.
        assert_eq!(
            m.words()[2] >> 2,
            0,
            "every bit at and past len must be zero after clear_tail"
        );
    }

    /// FAILS IF: `or_assign` or `xor_assign` stops clearing the tail — a
    /// phantom bit carried by a NON-conforming operand would then land in
    /// the raw tail word of a mask that never touched `words_mut()` itself.
    /// The can-fire half is the raw word before the op (the operand really
    /// carries the bit); the silent half is the raw word after it.
    #[test]
    fn or_and_xor_assign_clear_a_phantom_bit_a_non_conforming_operand_carries() {
        let mut dirty = AlphaMask::empty(130);
        dirty.words_mut()[2] |= 1 << 5; // ordinal 133, past len
        assert_eq!(
            dirty.words()[2] >> 5 & 1,
            1,
            "fixture: the operand carries the phantom"
        );

        let mut via_or = AlphaMask::empty(130);
        via_or.or_assign(&dirty);
        assert_eq!(
            via_or.words()[2],
            0,
            "or_assign must clear the propagated phantom"
        );

        let mut via_xor = AlphaMask::empty(130);
        via_xor.xor_assign(&dirty);
        assert_eq!(
            via_xor.words()[2],
            0,
            "xor_assign must clear the propagated phantom"
        );

        // and the two ops that cannot raise a bit stay untouched by design
        let mut via_and = AlphaMask::empty(130);
        via_and.set(1);
        via_and.and_assign(&dirty);
        assert_eq!(via_and.words()[2], 0);
    }

    /// `and_assign`'s release-mode length guard, mirroring [`Self::and`]'s
    /// (via `zip`'s `assert_eq!`). FAILS IF: `and_assign` stops checking
    /// `self.len == other.len` before combining — e.g. it started zipping
    /// with `Iterator::zip`'s silent truncation instead.
    #[test]
    #[should_panic(expected = "masks from different allocations")]
    fn and_assign_refuses_masks_of_different_lengths() {
        let mut wide = AlphaMask::empty(200);
        let narrow = AlphaMask::empty(64);
        wide.and_assign(&narrow);
    }

    /// `or_assign`'s release-mode length guard — see `and_assign`'s sibling
    /// test above for the failure mode.
    #[test]
    #[should_panic(expected = "masks from different allocations")]
    fn or_assign_refuses_masks_of_different_lengths() {
        let mut wide = AlphaMask::empty(200);
        let narrow = AlphaMask::empty(64);
        wide.or_assign(&narrow);
    }

    /// `xor_assign`'s release-mode length guard — see `and_assign`'s
    /// sibling test above for the failure mode.
    #[test]
    #[should_panic(expected = "masks from different allocations")]
    fn xor_assign_refuses_masks_of_different_lengths() {
        let mut wide = AlphaMask::empty(200);
        let narrow = AlphaMask::empty(64);
        wide.xor_assign(&narrow);
    }

    /// `and_not_assign`'s release-mode length guard — see `and_assign`'s
    /// sibling test above for the failure mode.
    #[test]
    #[should_panic(expected = "masks from different allocations")]
    fn and_not_assign_refuses_masks_of_different_lengths() {
        let mut wide = AlphaMask::empty(200);
        let narrow = AlphaMask::empty(64);
        wide.and_not_assign(&narrow);
    }

    /// FAILS IF: `clear` leaves any bit set, or changes `len` — a mask
    /// reused as scratch must keep its allocation's address-space width.
    #[test]
    fn clear_zeroes_a_non_empty_mask_and_keeps_its_length() {
        let mut m = random_mask(130, 0xABCD_EF01_ABCD_EF01);
        assert!(
            m.count() > 0,
            "fixture must be non-empty or clear() proves nothing"
        );
        let len_before = m.len();

        m.clear();

        assert_eq!(m.len(), len_before, "clear must not change len");
        assert!(m.is_empty(), "clear must leave the mask empty");
        assert_eq!(m.count(), 0);
        assert_eq!(m.materialize_ordinals(), Vec::<u32>::new());
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
