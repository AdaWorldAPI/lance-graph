//! # step_mask — the compiled-template live-step selector (D-V3-W3a)
//!
//! The thinking sibling of [`crate::class_view::FieldMask`], per the V3
//! compiled-templates ruling (board `E-COMPILED-THINKING-TEMPLATES`):
//!
//! ```text
//! askama template  ↔  ClassView × FieldMask     (rendering: masked selection over a class)
//! elixir DSL       ↔  Template  × StepMask      (thinking:  masked selection over a plan)
//! ```
//!
//! Bit position `N` = the `N`-th step in a compiled template's **ordered step
//! list** is LIVE for the current style/dispatch. Positions are stable +
//! append-only exactly like `FieldMask`'s N3 rule: once a template version is
//! in the catalogue, a step's bit position never moves and retired positions
//! are never reused (bump the template `version` instead).
//!
//! **Selection, NEVER control flow.** A `StepMask` answers "is step `n` live
//! this dispatch"; it must never encode branch/jump/wait semantics. The
//! honest 1:1 pairings with the executor are `Step ↔ graph_flow::Task` and
//! `OgarAction::ogar_name() ↔ Task::id()` — `NextAction`-shaped control flow
//! (GoTo / End / WaitForInput) is a separate `ControlSignal` surface, NOT
//! bits in this mask (see `.claude/v3/knowledge/compiled-templates.md`, the
//! 2026-07-02 ground-truth correction). Encoding a jump target in a
//! selection mask would silently drop End/WaitForInput semantics.
//!
//! **Standing async plan.** Per the mailbox-kanban ruling, a kanban update
//! *reprioritizes* which template steps are live — it swaps the `StepMask` —
//! it does not wake or gate the thinking cycle. The template keeps executing
//! its live set; a masked-off step is skipped, not awaited.

/// Live-step bitmask over a compiled template's ordered step list — the
/// selection half of `Template × StepMask` (sibling of
/// [`FieldMask`](crate::class_view::FieldMask), same `u64` discipline).
///
/// Zero-dep (`u64`, no `bitflags`); mask width is bounded by the *template's*
/// step count (a pipeline of dozens of steps at most), never a catalogue
/// union. A template that wants more than [`Self::MAX_STEPS`] steps is a
/// decomposition signal (split the template), not a mask-widening use case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct StepMask(pub u64);

impl StepMask {
    /// The empty mask (no step live — the template rests this dispatch).
    pub const EMPTY: Self = Self(0);

    /// The full mask — every addressable step position live. The "no style
    /// narrowing" default for a dispatch that has not selected a lens.
    pub const FULL: Self = Self(u64::MAX);

    /// Maximum addressable step positions in one `u64` mask.
    pub const MAX_STEPS: u32 = 64;

    /// Build a mask from live step positions. Positions `>= MAX_STEPS` (64)
    /// are **ignored** — NOT folded onto a valid bit (folding would alias
    /// position 64 onto bit 0 and silently run the wrong step; same rule as
    /// `FieldMask::from_positions`, Codex P2 on #441).
    pub const fn from_positions(positions: &[u8]) -> Self {
        let mut bits = 0u64;
        let mut i = 0;
        while i < positions.len() {
            if (positions[i] as u32) < Self::MAX_STEPS {
                bits |= 1u64 << positions[i];
            }
            i += 1;
        }
        Self(bits)
    }

    /// The live-set default for a template with `step_count` ordered steps:
    /// bits `0..step_count` set. Counts above [`Self::MAX_STEPS`] saturate to
    /// [`Self::FULL`] (the excess steps are unaddressable — a >64-step
    /// template is a split signal, mirroring `WideFieldMask::full_for`'s
    /// class-conditioned shape).
    pub const fn full_for(step_count: usize) -> Self {
        if step_count >= Self::MAX_STEPS as usize {
            Self::FULL
        } else if step_count == 0 {
            Self::EMPTY
        } else {
            Self((1u64 << step_count) - 1)
        }
    }

    /// Mark step position `n` live. `n >= MAX_STEPS` (64) is a no-op
    /// (NOT folded — see [`from_positions`](StepMask::from_positions)).
    #[inline]
    pub const fn with(self, n: u8) -> Self {
        if (n as u32) < Self::MAX_STEPS {
            Self(self.0 | (1u64 << n))
        } else {
            self
        }
    }

    /// Mask step position `n` off (skip it this dispatch). `n >= MAX_STEPS`
    /// is a no-op.
    #[inline]
    pub const fn without(self, n: u8) -> Self {
        if (n as u32) < Self::MAX_STEPS {
            Self(self.0 & !(1u64 << n))
        } else {
            self
        }
    }

    /// Is step position `n` live this dispatch? `n >= MAX_STEPS` (64) is
    /// always `false` — an out-of-range step is never live (NOT folded).
    #[inline]
    pub const fn is_live(self, n: u8) -> bool {
        (n as u32) < Self::MAX_STEPS && self.0 & (1u64 << n) != 0
    }

    /// Number of live steps.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Is nothing live?
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Bitwise intersection — the steps live in BOTH masks. The fold a
    /// dispatch uses to combine a style lens with a kanban reprioritization
    /// (a step runs only if the style wants it AND the board admits it).
    #[inline]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Bitwise union — the steps live in EITHER mask.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Do the two masks share NO live step? Two style lenses over the same
    /// template are disjoint iff they exercise disjoint step sets.
    #[inline]
    pub const fn is_disjoint(self, other: Self) -> bool {
        self.0 & other.0 == 0
    }

    /// The next live step position `>= from`, or `None` when the rest of the
    /// plan is masked off — the executor's ordered walk primitive
    /// (`while let Some(i) = mask.next_live(cursor) { run step i; cursor = i + 1; }`).
    /// Skipping is O(1) via trailing-zeros, not a per-bit loop.
    #[inline]
    pub const fn next_live(self, from: u8) -> Option<u8> {
        if (from as u32) >= Self::MAX_STEPS {
            return None;
        }
        let masked = self.0 & (u64::MAX << from);
        if masked == 0 {
            None
        } else {
            Some(masked.trailing_zeros() as u8)
        }
    }
}

impl From<u64> for StepMask {
    /// Additive convenience alongside the tuple constructor `StepMask(bits)`.
    #[inline]
    fn from(bits: u64) -> Self {
        Self(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_at_or_above_64_are_ignored_never_folded() {
        // Folding 64 → bit 0 would silently run the wrong step.
        let m = StepMask::from_positions(&[0, 63, 64, 200]);
        assert!(m.is_live(0));
        assert!(m.is_live(63));
        assert!(!m.is_live(64));
        assert_eq!(m.count(), 2);
        // with/without are no-ops out of range, never folds.
        assert_eq!(StepMask::EMPTY.with(64), StepMask::EMPTY);
        assert_eq!(StepMask::FULL.without(64), StepMask::FULL);
        // is_live out of range is always false.
        assert!(!StepMask::FULL.is_live(64));
    }

    #[test]
    fn full_for_matches_step_count_and_saturates() {
        assert_eq!(StepMask::full_for(0), StepMask::EMPTY);
        let three = StepMask::full_for(3);
        assert_eq!(three.count(), 3);
        assert!(three.is_live(0) && three.is_live(1) && three.is_live(2));
        assert!(!three.is_live(3));
        assert_eq!(StepMask::full_for(64), StepMask::FULL);
        assert_eq!(StepMask::full_for(500), StepMask::FULL);
    }

    #[test]
    fn style_lens_intersects_with_board_admission() {
        // A style lens picks steps {0,2,4}; the kanban reprioritization
        // admits {2,3,4}. The dispatch runs the intersection {2,4}:
        // selection composes by AND, never by control flow.
        let style = StepMask::from_positions(&[0, 2, 4]);
        let board = StepMask::from_positions(&[2, 3, 4]);
        let live = style.intersect(board);
        assert_eq!(live, StepMask::from_positions(&[2, 4]));
        assert!(live.is_disjoint(StepMask::from_positions(&[0, 1, 3])));
        assert_eq!(style.union(board), StepMask::from_positions(&[0, 2, 3, 4]));
    }

    #[test]
    fn next_live_walks_the_plan_in_order_skipping_masked_steps() {
        // A 6-step template with steps 1 and 4 masked off: the executor's
        // ordered walk visits 0, 2, 3, 5 — skipped, not awaited.
        let mask = StepMask::full_for(6).without(1).without(4);
        let mut cursor = 0u8;
        let mut visited = Vec::new();
        while let Some(i) = mask.next_live(cursor) {
            visited.push(i);
            cursor = i + 1;
        }
        assert_eq!(visited, vec![0, 2, 3, 5]);
        // Empty rest-of-plan terminates; out-of-range cursor terminates.
        assert_eq!(StepMask::EMPTY.next_live(0), None);
        assert_eq!(StepMask::FULL.next_live(64), None);
        assert_eq!(StepMask::from_positions(&[63]).next_live(63), Some(63));
    }

    #[test]
    fn default_is_empty_and_from_u64_matches_tuple_constructor() {
        assert_eq!(StepMask::default(), StepMask::EMPTY);
        assert_eq!(StepMask::from(0b1010u64), StepMask(0b1010));
        assert_eq!(StepMask::MAX_STEPS, 64);
    }
}
