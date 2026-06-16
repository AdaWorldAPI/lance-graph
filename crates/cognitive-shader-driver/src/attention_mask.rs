//! AttentionMask SoA — W6 spec §2 implementation.
//!
//! Implements a flat struct-of-arrays attention mask with LRU eviction policy.
//! Each entry tracks a `(mailbox_id, w_slot)` pair with an activity flag,
//! last-touched cycle counter, and plasticity residual.
//!
//! # Concurrency
//! `AttentionMaskSoA` is `!Send` by design (non-atomic interior mutation).
//! Callers that need cross-thread access must wrap in `tokio::sync::Mutex`.
//!
//! # MailboxId
//! Imports the canonical `MailboxId` alias from
//! `lance_graph_contract::collapse_gate` (also re-exported here for ergonomic
//! `use cognitive_shader_driver::attention_mask::MailboxId` access).

pub use lance_graph_contract::collapse_gate::MailboxId;

// ── SoA row ──────────────────────────────────────────────────────────────────

/// One row of the AttentionMask SoA backing store.
///
/// `last_touched_cycle` is set to `AttentionMaskSoA::current_cycle` on every
/// `touch()` call. Eviction selects the occupied row with the smallest value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionMaskEntry {
    /// Canonical mailbox identity (W-slot corpus-root handle).
    pub mailbox_id: MailboxId,
    /// Physical W-slot index (6-bit palette, 0..64).
    pub w_slot: u8,
    /// Whether this entry currently holds an active attention claim.
    pub active: bool,
    /// Monotonic session cycle at which this entry was last touched.
    /// Eviction policy: argmin over active entries.
    pub last_touched_cycle: u32,
    /// Plasticity residual — reserved for sprint-12+ learning signal.
    pub plasticity_residual: u8,
}

// ── SoA backing store ────────────────────────────────────────────────────────

/// Flat SoA attention mask with LRU eviction.
///
/// Invariant: `active_count() <= max_active` after every `evict_lru` call.
/// Callers are responsible for calling `evict_lru` before inserting when at
/// capacity, or after `touch` returns `true` (newly activated).
pub struct AttentionMaskSoA {
    /// Flat row store. Both active and inactive entries live here.
    pub entries: Vec<AttentionMaskEntry>,
    /// Maximum number of simultaneously active entries before LRU eviction.
    pub max_active: usize,
    /// Monotonic session cycle counter (advanced by `tick()`).
    pub current_cycle: u32,
}

impl AttentionMaskSoA {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create an empty `AttentionMaskSoA` with the given active-entry cap.
    ///
    /// `max_active` must be at least 1; caller may assert this externally.
    pub fn new(max_active: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_active,
            current_cycle: 0,
        }
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Touch `(mailbox_id, w_slot)`.
    ///
    /// * If an active entry already exists for `mailbox_id`, its
    ///   `last_touched_cycle` is bumped and `false` is returned.
    /// * If an inactive entry exists for `mailbox_id`, it is reactivated,
    ///   its `w_slot` updated, `last_touched_cycle` bumped, and `true` returned.
    /// * If no entry exists, a new active entry is appended and `true` returned.
    ///
    /// Returns `true` iff the entry was newly activated (was absent or inactive).
    pub fn touch(&mut self, mailbox_id: MailboxId, w_slot: u8) -> bool {
        let cycle = self.current_cycle;

        // Check for existing entry (active or inactive).
        if let Some(entry) = self.entries.iter_mut().find(|e| e.mailbox_id == mailbox_id) {
            let newly_activated = !entry.active;
            entry.active = true;
            entry.w_slot = w_slot;
            entry.last_touched_cycle = cycle;
            return newly_activated;
        }

        // No existing entry — append a fresh active row.
        self.entries.push(AttentionMaskEntry {
            mailbox_id,
            w_slot,
            active: true,
            last_touched_cycle: cycle,
            plasticity_residual: 0,
        });
        true
    }

    /// Evict the least-recently-touched active entry if `active_count > max_active`.
    ///
    /// Returns the `MailboxId` of the evicted entry, or `None` if no eviction
    /// was needed (active_count ≤ max_active) or if there are no active entries.
    pub fn evict_lru(&mut self) -> Option<MailboxId> {
        if self.active_count() <= self.max_active {
            return None;
        }

        // Find the active entry with the smallest last_touched_cycle.
        let victim_idx = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.active)
            .min_by_key(|(_, e)| e.last_touched_cycle)
            .map(|(i, _)| i)?;

        let evicted_id = self.entries[victim_idx].mailbox_id;
        self.entries[victim_idx].active = false;
        Some(evicted_id)
    }

    /// Advance the session cycle counter (saturating at `u32::MAX`).
    ///
    /// Called once per dispatcher cycle by the owning actor or router.
    pub fn tick(&mut self) {
        self.current_cycle = self.current_cycle.saturating_add(1);
    }

    // ── Inspection ───────────────────────────────────────────────────────────

    /// Number of currently active entries.
    pub fn active_count(&self) -> usize {
        self.entries.iter().filter(|e| e.active).count()
    }

    /// Whether `mailbox_id` currently holds an active attention claim.
    pub fn is_active(&self, mailbox_id: MailboxId) -> bool {
        self.entries
            .iter()
            .any(|e| e.mailbox_id == mailbox_id && e.active)
    }

    /// Read-only view of the entire entry store (active + inactive rows).
    pub fn entries(&self) -> &[AttentionMaskEntry] {
        &self.entries
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A freshly constructed SoA has no entries and active_count == 0.
    #[test]
    fn test_new_empty() {
        let soa = AttentionMaskSoA::new(4);
        assert_eq!(soa.active_count(), 0);
        assert_eq!(soa.entries().len(), 0);
        assert_eq!(soa.current_cycle, 0);
        assert_eq!(soa.max_active, 4);
    }

    /// Touching a new mailbox_id activates it and returns true.
    #[test]
    fn test_touch_activates_new() {
        let mut soa = AttentionMaskSoA::new(4);
        let newly = soa.touch(42, 7);
        assert!(newly, "touch on new entry must return true");
        assert_eq!(soa.active_count(), 1);
        assert!(soa.is_active(42));
        assert_eq!(soa.entries()[0].w_slot, 7);
        assert_eq!(soa.entries()[0].last_touched_cycle, 0);
    }

    /// Touching an already-active entry bumps its cycle but returns false.
    #[test]
    fn test_touch_updates_existing() {
        let mut soa = AttentionMaskSoA::new(4);
        let first = soa.touch(10, 1);
        assert!(first);

        soa.tick(); // cycle = 1
        let second = soa.touch(10, 2);
        assert!(!second, "touch on existing active entry must return false");
        assert_eq!(soa.active_count(), 1);

        let entry = &soa.entries()[0];
        assert_eq!(entry.w_slot, 2, "w_slot should be updated");
        assert_eq!(entry.last_touched_cycle, 1, "cycle should be bumped");
    }

    /// When active_count > max_active, evict_lru deactivates the oldest entry
    /// and returns its mailbox_id.
    #[test]
    fn test_evict_lru_when_over_capacity() {
        let mut soa = AttentionMaskSoA::new(2);

        soa.touch(1, 0); // cycle 0
        soa.tick(); // cycle 1
        soa.touch(2, 0); // cycle 1
        soa.tick(); // cycle 2
        soa.touch(3, 0); // cycle 2 — now active_count = 3 > max_active = 2

        let evicted = soa.evict_lru();
        assert_eq!(
            evicted,
            Some(1),
            "mailbox_id 1 was touched at cycle 0 (oldest)"
        );
        assert_eq!(soa.active_count(), 2);
        assert!(!soa.is_active(1));
        assert!(soa.is_active(2));
        assert!(soa.is_active(3));
    }

    /// evict_lru returns None when active_count <= max_active.
    #[test]
    fn test_evict_lru_returns_none_under_capacity() {
        let mut soa = AttentionMaskSoA::new(4);
        soa.touch(1, 0);
        soa.touch(2, 0);

        let result = soa.evict_lru();
        assert_eq!(result, None, "no eviction needed when under capacity");
        assert_eq!(soa.active_count(), 2);
    }

    /// is_active returns true after touch, false for unknown ids.
    #[test]
    fn test_is_active_after_touch() {
        let mut soa = AttentionMaskSoA::new(4);
        assert!(!soa.is_active(99));
        soa.touch(99, 3);
        assert!(soa.is_active(99));
        assert!(!soa.is_active(100));
    }

    /// tick advances current_cycle by 1 each call.
    #[test]
    fn test_tick_increments_cycle() {
        let mut soa = AttentionMaskSoA::new(4);
        assert_eq!(soa.current_cycle, 0);
        soa.tick();
        assert_eq!(soa.current_cycle, 1);
        soa.tick();
        soa.tick();
        assert_eq!(soa.current_cycle, 3);
    }

    /// evict_lru picks the entry with the globally smallest last_touched_cycle.
    #[test]
    fn test_evict_lru_picks_oldest() {
        let mut soa = AttentionMaskSoA::new(3);

        // Touch three entries at different cycles.
        soa.touch(10, 0); // cycle 0
        soa.tick(); // cycle 1
        soa.touch(20, 0); // cycle 1
        soa.tick(); // cycle 2
        soa.touch(30, 0); // cycle 2
        soa.tick(); // cycle 3

        // Re-touch id=10 to make it fresh; id=20 is now oldest.
        soa.touch(10, 0); // cycle 3

        // Push over capacity by adding a 4th entry.
        soa.touch(40, 0); // cycle 3, active_count = 4 > max_active = 3

        let evicted = soa.evict_lru();
        assert_eq!(
            evicted,
            Some(20),
            "id=20 was last touched at cycle 1 (oldest remaining)"
        );
        assert!(!soa.is_active(20));
        assert!(soa.is_active(10));
        assert!(soa.is_active(30));
        assert!(soa.is_active(40));
    }
}
