//! # standing_mask — subscriptions as dirty ∩ interest
//!
//! A subscription is a **standing interest mask**: "notify subscriber `S`
//! when any of THESE field positions of THIS key change." On every write,
//! the writer already produces (or can cheaply produce) a *dirty mask* —
//! the set of field positions that changed. A subscription fires iff
//!
//! ```text
//! dirty ∩ interest ≠ ∅
//! ```
//!
//! **One bitwise AND per write, per candidate subscription.** No query
//! re-execution, no per-field diffing, no re-reading the row to see what
//! changed — the dirty mask is the diff, already computed by the writer.
//! [`WideFieldMask::intersect`] is the AND; [`WideFieldMask::is_empty`] is
//! the "did anything survive" check. [`fires`] is exactly that pair,
//! spelled out once so every call site reads the same thing.
//!
//! Interest masks **compose by union**: [`SubscriptionTable::widen`] folds
//! a new interest onto an existing one via [`WideFieldMask::union`] — a
//! subscriber that wants "notify me on A" and later also wants "notify me
//! on B" ends up with interest `A ∪ B`, not two separate rows.
//!
//! ## Table shape: `Vec` + linear scan is correct here
//!
//! [`SubscriptionTable`] is a flat `Vec<StandingInterest<K>>` scanned
//! linearly by [`SubscriptionTable::notify`]. That is the right choice at
//! *this* layer: the contract is zero-dep and does not know how many
//! subscribers exist system-wide. The consumer is expected to **shard one
//! table per mailbox/tenant** (the V3 ownership doctrine already partitions
//! writes that way), which keeps `N` small — a handful to a few dozen
//! standing interests per shard, not a global subscriber directory. Do not
//! read the linear scan as a gap to "fix" with a hash index or per-field
//! reverse map in this crate; that indexing decision belongs to whichever
//! consumer has enough shard-size data to justify it, if ever.
//!
//! `K` is generic and unconstrained here on purpose: this contract does not
//! name a key type (row key, classid, GUID, ...) — that is the consumer's
//! choice.

use crate::class_view::WideFieldMask;

/// Identifies a subscriber (an interested party) in a [`SubscriptionTable`].
/// Opaque `u64` — the contract does not care what a subscriber *is* (a
/// mailbox id, a session id, a connection handle mapped to an integer,
/// ...), only that it is copyable and comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriberId(pub u64);

/// One standing interest: subscriber `subscriber` wants to be notified
/// whenever a write to `key` dirties any field position in `interest`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandingInterest<K> {
    /// Who is interested.
    pub subscriber: SubscriberId,
    /// Which row/entity this interest is scoped to.
    pub key: K,
    /// The field positions whose change should notify `subscriber`.
    pub interest: WideFieldMask,
}

/// Does a write's dirty-field mask trip a subscription's standing interest?
///
/// `fires(dirty, interest) == !dirty.intersect(interest).is_empty()` — the
/// entire mechanism in one function, so every call site (this module's
/// [`SubscriptionTable::notify`], or a consumer checking a single
/// subscription ad hoc) reads the identical one-AND-one-check shape.
#[inline]
#[must_use]
pub fn fires(dirty: &WideFieldMask, interest: &WideFieldMask) -> bool {
    !dirty.intersect(interest).is_empty()
}

/// A flat table of standing interests, keyed loosely by `(subscriber, key)`
/// pairs. See the module docs for why a linear-scan `Vec` is the correct
/// shape at this layer (shard per mailbox/tenant to keep `N` small).
#[derive(Debug, Clone)]
pub struct SubscriptionTable<K> {
    entries: Vec<StandingInterest<K>>,
}

impl<K: Copy + Eq> Default for SubscriptionTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Copy + Eq> SubscriptionTable<K> {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register `subscriber`'s interest in `key`, exactly as given.
    ///
    /// **Last write wins:** if `(subscriber, key)` already has a standing
    /// interest, this REPLACES it (does not union). Reach for
    /// [`Self::widen`] when the intent is "add more interest on top of what
    /// this subscriber already has."
    pub fn subscribe(&mut self, subscriber: SubscriberId, key: K, interest: WideFieldMask) {
        match self.find_mut(subscriber, key) {
            Some(entry) => entry.interest = interest,
            None => self.entries.push(StandingInterest {
                subscriber,
                key,
                interest,
            }),
        }
    }

    /// Remove `subscriber`'s standing interest in `key`, if any. A no-op if
    /// no such interest is registered.
    pub fn unsubscribe(&mut self, subscriber: SubscriberId, key: K) {
        self.entries
            .retain(|e| !(e.subscriber == subscriber && e.key == key));
    }

    /// Widen `subscriber`'s interest in `key` by unioning `extra` onto
    /// whatever interest it already holds (inserting a fresh interest of
    /// exactly `extra` if it had none). Interest masks only ever grow via
    /// this path — the union is the composition rule described in the
    /// module docs.
    pub fn widen(&mut self, subscriber: SubscriberId, key: K, extra: WideFieldMask) {
        match self.find_mut(subscriber, key) {
            Some(entry) => entry.interest = entry.interest.union(&extra),
            None => self.entries.push(StandingInterest {
                subscriber,
                key,
                interest: extra,
            }),
        }
    }

    /// Given a write to `key` that dirtied field positions `dirty`, return
    /// every subscriber on `key` whose standing interest [`fires`]. Order
    /// matches subscription (insertion) order. `key` unknown to the table
    /// (no subscriptions registered on it) returns an empty `Vec` — no
    /// allocation beyond the output itself.
    #[must_use]
    pub fn notify(&self, key: K, dirty: &WideFieldMask) -> Vec<SubscriberId> {
        self.entries
            .iter()
            .filter(|e| e.key == key && fires(dirty, &e.interest))
            .map(|e| e.subscriber)
            .collect()
    }

    /// Number of standing interests currently registered (across all keys).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the table empty (no standing interests registered)?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn find_mut(&mut self, subscriber: SubscriberId, key: K) -> Option<&mut StandingInterest<K>> {
        self.entries
            .iter_mut()
            .find(|e| e.subscriber == subscriber && e.key == key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_iff_dirty_and_interest_overlap() {
        let interest = WideFieldMask::from_positions(&[1, 3, 5]);
        assert!(fires(&WideFieldMask::from_positions(&[3]), &interest));
        assert!(fires(&WideFieldMask::from_positions(&[0, 5]), &interest));
        assert!(!fires(
            &WideFieldMask::from_positions(&[0, 2, 4]),
            &interest
        ));
        assert!(!fires(&WideFieldMask::EMPTY, &interest));
        assert!(!fires(
            &WideFieldMask::from_positions(&[9]),
            &WideFieldMask::EMPTY
        ));
    }

    #[test]
    fn multiple_subscribers_same_key_each_evaluated_independently() {
        let mut table = SubscriptionTable::new();
        let a = SubscriberId(1);
        let b = SubscriberId(2);
        let key = 42u64;
        table.subscribe(a, key, WideFieldMask::from_positions(&[0]));
        table.subscribe(b, key, WideFieldMask::from_positions(&[1]));

        let dirty = WideFieldMask::from_positions(&[0]);
        assert_eq!(table.notify(key, &dirty), vec![a]);

        let dirty_both = WideFieldMask::from_positions(&[0, 1]);
        assert_eq!(table.notify(key, &dirty_both), vec![a, b]);
    }

    #[test]
    fn same_subscriber_two_keys_are_independent_interests() {
        let mut table = SubscriptionTable::new();
        let s = SubscriberId(7);
        table.subscribe(s, 1u64, WideFieldMask::from_positions(&[0]));
        table.subscribe(s, 2u64, WideFieldMask::from_positions(&[1]));

        let dirty = WideFieldMask::from_positions(&[0]);
        assert_eq!(table.notify(1u64, &dirty), vec![s]);
        assert!(table.notify(2u64, &dirty).is_empty());

        let dirty2 = WideFieldMask::from_positions(&[1]);
        assert!(table.notify(1u64, &dirty2).is_empty());
        assert_eq!(table.notify(2u64, &dirty2), vec![s]);
    }

    #[test]
    fn widen_unions_onto_existing_interest() {
        let mut table = SubscriptionTable::new();
        let s = SubscriberId(1);
        let key = 0u64;
        table.subscribe(s, key, WideFieldMask::from_positions(&[0]));
        table.widen(s, key, WideFieldMask::from_positions(&[5]));

        // Now interested in {0, 5}: a write touching only 5 should fire.
        assert_eq!(
            table.notify(key, &WideFieldMask::from_positions(&[5])),
            vec![s]
        );
        // A write touching neither should not.
        assert!(table
            .notify(key, &WideFieldMask::from_positions(&[9]))
            .is_empty());
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn widen_inserts_when_no_prior_subscription() {
        let mut table = SubscriptionTable::new();
        let s = SubscriberId(1);
        let key = 0u64;
        table.widen(s, key, WideFieldMask::from_positions(&[2]));
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.notify(key, &WideFieldMask::from_positions(&[2])),
            vec![s]
        );
    }

    #[test]
    fn subscribe_replaces_does_not_union() {
        let mut table = SubscriptionTable::new();
        let s = SubscriberId(1);
        let key = 0u64;
        table.subscribe(s, key, WideFieldMask::from_positions(&[0]));
        table.subscribe(s, key, WideFieldMask::from_positions(&[1]));

        // The old interest (0) must be GONE — replaced, not unioned.
        assert!(table
            .notify(key, &WideFieldMask::from_positions(&[0]))
            .is_empty());
        assert_eq!(
            table.notify(key, &WideFieldMask::from_positions(&[1])),
            vec![s]
        );
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn unsubscribe_removes_the_interest() {
        let mut table = SubscriptionTable::new();
        let s = SubscriberId(1);
        let key = 0u64;
        table.subscribe(s, key, WideFieldMask::from_positions(&[0]));
        assert_eq!(table.len(), 1);
        table.unsubscribe(s, key);
        assert!(table.is_empty());
        assert!(table
            .notify(key, &WideFieldMask::from_positions(&[0]))
            .is_empty());

        // Unsubscribing something that was never there is a harmless no-op.
        table.unsubscribe(s, key);
        assert!(table.is_empty());
    }

    #[test]
    fn notify_on_unknown_key_is_empty() {
        let mut table = SubscriptionTable::new();
        table.subscribe(SubscriberId(1), 0u64, WideFieldMask::from_positions(&[0]));
        assert!(table
            .notify(999u64, &WideFieldMask::from_positions(&[0]))
            .is_empty());
    }

    #[test]
    fn wide_tier_masks_work_end_to_end() {
        // Positions >= 64 promote WideFieldMask::Small -> Wide internally;
        // the whole subscribe/widen/notify path must be tier-agnostic.
        let mut table = SubscriptionTable::new();
        let s = SubscriberId(1);
        let key = 0u64;

        table.subscribe(s, key, WideFieldMask::EMPTY.with(70));
        assert!(table
            .notify(key, &WideFieldMask::from_positions(&[3]))
            .is_empty());
        assert_eq!(table.notify(key, &WideFieldMask::EMPTY.with(70)), vec![s]);

        // widen with a low-tier position on top of a wide interest.
        table.widen(s, key, WideFieldMask::from_positions(&[3]));
        assert_eq!(
            table.notify(key, &WideFieldMask::from_positions(&[3])),
            vec![s]
        );
        assert_eq!(table.notify(key, &WideFieldMask::EMPTY.with(70)), vec![s]);
        assert!(table
            .notify(key, &WideFieldMask::from_positions(&[9]))
            .is_empty());
    }

    #[test]
    fn default_table_is_empty() {
        let table: SubscriptionTable<u64> = SubscriptionTable::default();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }
}
