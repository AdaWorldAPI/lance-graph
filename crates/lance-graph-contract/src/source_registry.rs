//! Evidence-source identity: arbitrary stable ids → dense local slots → stamps.
//!
//! ## The defect this replaces
//!
//! Two crates independently shipped `pub struct Stamp(pub u64)` with
//! `fn source(id: u32) -> Stamp { Stamp(1u64 << (id % 64)) }`. The modulo is
//! **conservative, not unsound** — a collision makes two distinct sources look
//! *overlapping*, so NARS revision refuses to pool them. It loses evidence; it
//! does not fabricate independence. (Both copies documented this correctly; an
//! audit that claimed otherwise was wrong.)
//!
//! What it *does* destroy is everything downstream of knowing which bit is
//! whom: pooling past 64 sources, leave-one-out, withdrawal, and any reading of
//! an evidence count. Hence the rule this module exists to enforce:
//!
//! > **A term id, domain id, witness id, or corpus id must NEVER be silently
//! > interpreted as a bit position.**
//!
//! An external id may be `300`, `50_000`, or a hash. It can still legitimately
//! occupy local slot 0 — if it is the first source registered in the current
//! bounded horizon. The failure condition is not "id ≥ 64", it is "more than
//! 64 *simultaneously represented* identities".
//!
//! ## Ruling: stamps are ARENA-LOCAL, by containment
//!
//! A [`Stamp`] is meaningful only relative to the registry that minted its
//! bits. Registry A may assign source X to slot 0 while registry B assigns
//! source Z to slot 0 — so `Stamp(0b1)` means different evidence in each, and
//! comparing them across registries is nonsense that `disjoint` would answer
//! confidently.
//!
//! Two designs are safe. **This module takes the first:**
//!
//! 1. **Strong containment** — the owning arena holds the registry, mints every
//!    stamp, and performs every `union` / `disjoint`. Callers pass a
//!    [`SourceId`]; no stamp crosses an API boundary, so no stamp can be
//!    compared against a foreign one.
//! 2. *Registry-bearing stamps* — `{registry: SourceRegistryId, bits}`, with
//!    every operation rejecting mismatched registries. Necessary only if stamps
//!    must persist or cross boundaries.
//!
//! **Why (1).** Design (2) exists to make evidence from arena A meaningful
//! inside arena B — a handoff between two independently-owned state containers.
//! That is the shape the substrate already deleted at the mailbox layer (#477:
//! no inter-mailbox carrier at all, one writer per mailbox), reintroduced one
//! layer down under a new name. Containment is not merely cheaper here; it is
//! the option consistent with the ratified ownership model.
//!
//! Enforcement is structural, not advisory: [`Stamp`]'s bits are **private**,
//! it has no `Serialize`, and it is constructible only from a [`SourceSlot`]
//! that a [`SourceRegistry`] issued.
//!
//! **The flip condition, named so it is falsifiable:** if replay must
//! reconstruct evidence from *persisted* state rather than rebuilding the arena,
//! containment breaks. The answer then is still not a registry field on every
//! stamp — it is a **frozen source census**: a versioned artifact from which a
//! deterministic sorted-`SourceId` → `SourceSlot` allocation is regenerated,
//! checked against a registry digest. That keeps mapping identity addressable
//! by epistemic view instead of smuggling it into the hot carrier.

/// A stable, arbitrary, sparse identity for an evidence source — a corpus id, a
/// witness id, a term id, a hash.
///
/// **Never a bit position.** Converting one to a bit index is the defect this
/// module exists to prevent; go through [`SourceRegistry::slot_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct SourceId(pub u64);

/// A dense local slot, `0..64`, issued by one [`SourceRegistry`].
///
/// Meaningful ONLY relative to the registry that issued it. Deliberately not
/// constructible from a raw integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct SourceSlot(u8);

impl SourceSlot {
    /// The slot's index. For diagnostics and dense array indexing within the
    /// owning arena — NOT for reconstructing a stamp elsewhere.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }
}

/// The registry is full: more than [`SourceRegistry::CAPACITY`] distinct
/// sources are simultaneously represented in one evidence horizon.
///
/// A real, reportable condition — never silently folded, which is exactly what
/// `id % 64` did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityExceeded {
    /// The source that could not be admitted.
    pub source: SourceId,
    /// How many slots are already in use (always [`SourceRegistry::CAPACITY`]).
    pub in_use: usize,
}

impl core::fmt::Display for CapacityExceeded {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "source registry full ({} slots in use); cannot admit SourceId({})",
            self.in_use, self.source.0
        )
    }
}

impl core::error::Error for CapacityExceeded {}

/// Maps arbitrary stable [`SourceId`]s onto dense [`SourceSlot`]s for one
/// bounded evidence horizon.
///
/// Insertion-ordered and linear-scanned: the capacity is 64, so a map would
/// cost more than it saves, and the ordering makes the allocation reproducible
/// for a given insertion sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceRegistry {
    slots: Vec<SourceId>,
}

impl SourceRegistry {
    /// Slots available — the width of [`Stamp`]'s bitset.
    pub const CAPACITY: usize = 64;

    /// An empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// The slot for `source`, allocating one if this is its first appearance.
    ///
    /// Idempotent: the same `SourceId` always maps to the same slot within one
    /// registry. Returns [`CapacityExceeded`] rather than wrapping.
    pub fn slot_for(&mut self, source: SourceId) -> Result<SourceSlot, CapacityExceeded> {
        if let Some(slot) = self.lookup(source) {
            return Ok(slot);
        }
        if self.slots.len() >= Self::CAPACITY {
            return Err(CapacityExceeded {
                source,
                in_use: self.slots.len(),
            });
        }
        let idx = u8::try_from(self.slots.len()).expect("len < CAPACITY <= 64");
        self.slots.push(source);
        Ok(SourceSlot(idx))
    }

    /// The slot already held by `source`, if any. Never allocates.
    #[must_use]
    pub fn lookup(&self, source: SourceId) -> Option<SourceSlot> {
        self.slots
            .iter()
            .position(|s| *s == source)
            .map(|i| SourceSlot(u8::try_from(i).expect("index < CAPACITY")))
    }

    /// Which source holds `slot` — the inverse direction, for attribution.
    #[must_use]
    pub fn source_of(&self, slot: SourceSlot) -> Option<SourceId> {
        self.slots.get(slot.0 as usize).copied()
    }

    /// Distinct sources registered so far.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Have any sources been registered?
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Remaining capacity before [`slot_for`](Self::slot_for) starts failing.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        Self::CAPACITY - self.slots.len()
    }

    /// Mint the stamp for `source`, allocating a slot if needed.
    ///
    /// The intended entry point for an owning arena: it never hands a
    /// [`SourceSlot`] to a caller, so the stamp and its meaning stay together.
    pub fn stamp_for(&mut self, source: SourceId) -> Result<Stamp, CapacityExceeded> {
        self.slot_for(source).map(Stamp::from_slot)
    }
}

/// An evidential base: which registry slots contributed to a belief.
///
/// **Arena-local.** The bits are private and there is no way to build one from
/// a raw integer — a `Stamp` can only come from a [`SourceSlot`] its registry
/// issued, or from [`union`](Stamp::union)ing stamps that already exist. Keep
/// it inside the arena that owns the registry; comparing stamps minted by
/// different registries is meaningless and this type deliberately makes that
/// hard to do by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stamp(u64);

impl Stamp {
    /// The empty stamp — "no recorded source".
    ///
    /// Load-bearing as a sentinel: NARS revision must NOT pool two beliefs when
    /// either side is empty, because "no recorded source" is not evidence of
    /// independence. Callers gate on [`is_empty`](Stamp::is_empty).
    pub const EMPTY: Self = Self(0);

    /// The stamp of a single registry slot.
    #[inline]
    #[must_use]
    pub const fn from_slot(slot: SourceSlot) -> Self {
        Self(1u64 << slot.0)
    }

    /// Do these stamps share no source?
    ///
    /// Only meaningful for stamps from the SAME registry — see the module docs.
    #[inline]
    #[must_use]
    pub const fn disjoint(self, other: Self) -> bool {
        self.0 & other.0 == 0
    }

    /// The pooled evidential base.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Is this the no-source sentinel?
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// How many distinct sources this stamp represents — the evidence count
    /// that `id % 64` folding made uninterpretable.
    #[inline]
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Does this stamp include `slot`?
    #[inline]
    #[must_use]
    pub const fn contains(self, slot: SourceSlot) -> bool {
        self.0 & (1u64 << slot.0) != 0
    }

    /// Withdraw one source's contribution — possible only because slots are
    /// stable identities rather than folded hashes.
    #[inline]
    #[must_use]
    pub const fn without(self, slot: SourceSlot) -> Self {
        Self(self.0 & !(1u64 << slot.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The headline property: a LARGE, sparse external id maps cleanly, because
    /// the registry allocates by insertion order and never by arithmetic on the
    /// id. Under `1 << (id % 64)` these three would have collided into slot 0.
    #[test]
    fn sparse_external_ids_get_distinct_slots() {
        let mut reg = SourceRegistry::new();
        let a = reg.slot_for(SourceId(0)).unwrap();
        let b = reg.slot_for(SourceId(64)).unwrap();
        let c = reg.slot_for(SourceId(128)).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);

        // …and the stamps they mint are genuinely disjoint.
        assert!(Stamp::from_slot(a).disjoint(Stamp::from_slot(b)));
        assert!(Stamp::from_slot(b).disjoint(Stamp::from_slot(c)));
    }

    /// A 50_000-valued id legitimately takes slot 0 when it registers first.
    /// The bound is on SIMULTANEOUS identities, not on id magnitude.
    #[test]
    fn a_huge_id_may_hold_slot_zero() {
        let mut reg = SourceRegistry::new();
        let slot = reg.slot_for(SourceId(50_000)).unwrap();
        assert_eq!(slot.index(), 0);
        assert_eq!(reg.source_of(slot), Some(SourceId(50_000)));
    }

    #[test]
    fn slot_allocation_is_idempotent() {
        let mut reg = SourceRegistry::new();
        let first = reg.slot_for(SourceId(7)).unwrap();
        let again = reg.slot_for(SourceId(7)).unwrap();
        assert_eq!(first, again);
        assert_eq!(reg.len(), 1, "no second slot burned");
    }

    /// Capacity exhaustion is REPORTED, not folded. This is the whole
    /// behavioural difference from `id % 64`.
    #[test]
    fn capacity_is_reported_never_wrapped() {
        let mut reg = SourceRegistry::new();
        for i in 0..SourceRegistry::CAPACITY as u64 {
            reg.slot_for(SourceId(i * 1000)).expect("within capacity");
        }
        assert_eq!(reg.remaining(), 0);

        let err = reg.slot_for(SourceId(999_999)).unwrap_err();
        assert_eq!(err.in_use, SourceRegistry::CAPACITY);
        assert_eq!(err.source, SourceId(999_999));

        // An ALREADY-registered source still resolves when full — the registry
        // is out of new slots, not broken.
        assert!(reg.slot_for(SourceId(0)).is_ok());
    }

    /// Evidence counting is interpretable again: N distinct sources ⇒ N bits.
    /// Under modulo folding this number was a lower bound of unknown tightness.
    #[test]
    fn evidence_count_is_exact_over_distinct_sources() {
        let mut reg = SourceRegistry::new();
        let mut pooled = Stamp::EMPTY;
        for i in 0..10u64 {
            pooled = pooled.union(reg.stamp_for(SourceId(i * 64)).unwrap());
        }
        assert_eq!(pooled.count(), 10);

        // Re-pooling the SAME sources adds nothing — idempotent, so a repeated
        // source cannot inflate the count.
        for i in 0..10u64 {
            pooled = pooled.union(reg.stamp_for(SourceId(i * 64)).unwrap());
        }
        assert_eq!(pooled.count(), 10, "repetition is not corroboration");
    }

    /// Withdrawal — the operation folded bits cannot support.
    #[test]
    fn withdrawal_removes_exactly_one_source() {
        let mut reg = SourceRegistry::new();
        let a = reg.slot_for(SourceId(11)).unwrap();
        let b = reg.slot_for(SourceId(22)).unwrap();
        let pooled = Stamp::from_slot(a).union(Stamp::from_slot(b));
        assert_eq!(pooled.count(), 2);

        let minus_a = pooled.without(a);
        assert_eq!(minus_a.count(), 1);
        assert!(!minus_a.contains(a));
        assert!(minus_a.contains(b));
    }

    /// The empty sentinel is not accidentally disjoint-with-everything in a way
    /// that licenses pooling: `disjoint` says true, so callers MUST gate on
    /// `is_empty` separately. Pinned so the guard's necessity stays visible.
    #[test]
    fn empty_stamp_is_disjoint_with_everything_hence_the_separate_guard() {
        let mut reg = SourceRegistry::new();
        let s = reg.stamp_for(SourceId(1)).unwrap();
        assert!(Stamp::EMPTY.disjoint(s));
        assert!(Stamp::EMPTY.is_empty(), "the guard callers must check");
        assert!(!s.is_empty());
    }

    /// Two registries independently assign slot 0 — the concrete reason stamps
    /// must not cross registry boundaries. Documents the hazard the containment
    /// ruling exists to remove; it is a property of the design, not a bug.
    #[test]
    fn slot_zero_means_different_sources_in_different_registries() {
        let mut a = SourceRegistry::new();
        let mut b = SourceRegistry::new();
        let sa = a.stamp_for(SourceId(111)).unwrap();
        let sb = b.stamp_for(SourceId(222)).unwrap();

        assert_eq!(sa, sb, "identical bits…");
        assert_eq!(a.source_of(SourceSlot(0)), Some(SourceId(111)));
        assert_eq!(b.source_of(SourceSlot(0)), Some(SourceId(222)));
        // …yet different evidence. Hence: one registry per arena, stamps stay in.
    }
}
