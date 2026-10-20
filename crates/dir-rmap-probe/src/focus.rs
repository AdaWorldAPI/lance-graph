//! Arm F: a 256-slot focus over selected group identities, compiled over B+.
//!
//! A focus binds up to 256 group ordinals to local slots for ONE chain
//! (every group, the security chain or the mail chain). It then projects the
//! complete B+ closure of that chain onto the slots: 4 words per group,
//! instead of `ceil(groups / 64)`. A user's position in the focus is the OR
//! of its direct groups' projected rows, and a role x location question is a
//! 4-word mask test.
//!
//! **Completeness.** The projection reads the B+ closure, never a closure
//! restricted to the focus. A path that runs through groups outside the
//! focus is therefore represented: `x -> h -> f` with `h` unfocused still
//! sets `f`'s slot in `x`'s row. The test
//! `a_path_through_unfocused_groups_is_represented` checks this, and its
//! disable run (a focus-local closure) goes red. A question F cannot answer
//! completely is refused, never truncated:
//!
//! * a question naming a group outside the focus: [`Refusal::OutOfFocus`];
//! * a focus or B+ built against another version: [`Refusal::Stale`];
//! * more than 256 groups, or a group named twice: [`Refusal::TooLarge`],
//!   [`Refusal::Duplicate`].
//!
//! **Preparation is split** so each phase is timed and counted alone:
//!
//! 1. [`Focus::slots`]: slot -> ordinal and the reverse map (focus
//!    construction);
//! 2. [`Focus::project`]: every group's closure row onto the slots (closure
//!    projection). It walks each focus group's DOWNWARD closure row and sets
//!    that slot in every group the row names, so the cost is the size of the
//!    focus groups' closures, not `groups x 256`. `project_by_test` is the
//!    reference (test every slot of every group's upward row); a test checks
//!    the two agree;
//! 3. per-user rows: [`FocusUsers`] (eager, every user at once) or
//!    [`FocusLazy`] (a user's row on first use).
//!
//! **Refresh.** After a nesting pair is added to B+ ([`Compiled::add_nesting`]),
//! [`Focus::refresh_nesting`] re-projects the child's downward closure (the
//! only groups whose upward rows can have changed). After membership edits
//! the projected rows are unchanged; only the edited users' rows are
//! recomputed ([`FocusUsers::refresh_users`], [`FocusLazy::refresh_users`]).
//! Every refresh takes its witness from the B+ it read, so a focus refreshed
//! against a stale B+ keeps refusing.
//!
//! No per-user 256 x 256 tile is allocated. The projection reads closure
//! rows at the focus's bound ordinals: an address translation done in
//! preparation and counted. Its admission is `ISS-ADMISSION-PREP-BINDING`,
//! still open.
//!
//! Reverse lookup (group ordinal -> slot) is a direct `u16` array over the
//! group ordinal space (slot + 1, 0 = not bound): one indexed read, no hash,
//! no search.

use crate::prep::{Chain, Compiled, Refusal, Witness, bit, ones};

pub const SLOTS: usize = 256;
pub type Row = [u64; 4];

#[inline]
fn or(a: &mut Row, b: &Row) {
    for k in 0..4 {
        a[k] |= b[k];
    }
}

#[inline]
fn set(r: &mut Row, s: usize) {
    r[s >> 6] |= 1 << (s & 63);
}

pub struct Focus {
    pub witness: Witness,
    pub chain: Chain,
    /// Slot -> group ordinal.
    pub ids: Vec<u16>,
    /// Group ordinal -> slot + 1; 0 = not bound.
    slot_of: Vec<u16>,
    /// Group ordinal -> its closure row through `chain`, on the slots.
    proj: Vec<Row>,
}

impl Focus {
    /// Phase 1: bind `groups` to slots. Refuses a stale B+, more than 256
    /// groups, and a group named twice. No projection yet.
    pub fn slots(
        c: &Compiled,
        chain: Chain,
        groups: &[u16],
        current: &Witness,
    ) -> Result<Focus, Refusal> {
        c.b.check(current)?;
        if groups.len() > SLOTS {
            return Err(Refusal::TooLarge);
        }
        let mut slot_of = vec![0u16; c.b.groups];
        for (s, &g) in groups.iter().enumerate() {
            let e = &mut slot_of[g as usize];
            if *e != 0 {
                return Err(Refusal::Duplicate);
            }
            *e = s as u16 + 1;
        }
        Ok(Focus {
            witness: c.b.witness.clone(),
            chain,
            ids: groups.to_vec(),
            slot_of,
            proj: Vec::new(),
        })
    }

    /// Phase 2: project the complete closure of `self.chain` onto the slots.
    pub fn project(&mut self, c: &Compiled) {
        self.proj = vec![[0u64; 4]; c.b.groups];
        for s in 0..self.ids.len() {
            let f = self.ids[s] as usize;
            for g in ones(c.down_row(self.chain, f)) {
                set(&mut self.proj[g], s);
            }
        }
        self.witness = c.b.witness.clone();
    }

    /// The reference projection: test every slot in every group's upward
    /// closure row. Same answer as [`Focus::project`], `groups x slots` work.
    pub fn project_by_test(&mut self, c: &Compiled) {
        self.proj = vec![[0u64; 4]; c.b.groups];
        for (g, row) in self.proj.iter_mut().enumerate() {
            let up = c.row(self.chain, g);
            for (s, &f) in self.ids.iter().enumerate() {
                if bit(up, f as usize) {
                    set(row, s);
                }
            }
        }
        self.witness = c.b.witness.clone();
    }

    /// Phases 1 and 2.
    pub fn bind(
        c: &Compiled,
        chain: Chain,
        groups: &[u16],
        current: &Witness,
    ) -> Result<Focus, Refusal> {
        let mut f = Focus::slots(c, chain, groups, current)?;
        f.project(c);
        Ok(f)
    }

    /// After `c.add_nesting(child, _)`: re-project the groups whose upward
    /// closure can have changed, which are exactly `child`'s downward
    /// closure in this chain (empty when the child does not pass it).
    pub fn refresh_nesting(&mut self, c: &Compiled, child: u16) {
        for g in ones(c.down_row(self.chain, child as usize)) {
            let up = c.row(self.chain, g);
            let mut row = [0u64; 4];
            for (s, &f) in self.ids.iter().enumerate() {
                if bit(up, f as usize) {
                    set(&mut row, s);
                }
            }
            self.proj[g] = row;
        }
        self.witness = c.b.witness.clone();
    }

    /// After membership edits only: the projected rows do not depend on
    /// direct membership, so only the witness moves (taken from `c`).
    pub fn refresh_membership(&mut self, c: &Compiled) {
        self.witness = c.b.witness.clone();
    }

    pub fn check(&self, current: &Witness) -> Result<(), Refusal> {
        if self.witness == *current {
            Ok(())
        } else {
            Err(Refusal::Stale)
        }
    }

    pub fn slot(&self, g: u16) -> Option<usize> {
        match self.slot_of[g as usize] {
            0 => None,
            s => Some(s as usize - 1),
        }
    }

    /// The mask for "in every one of `groups`"; refuses when one is not in
    /// the focus, so a caller falls back to B+ instead of reading a bit that
    /// was never projected.
    pub fn all_of(&self, groups: &[u16]) -> Result<Row, Refusal> {
        let mut m = [0u64; 4];
        for &g in groups {
            set(&mut m, self.slot(g).ok_or(Refusal::OutOfFocus)?);
        }
        Ok(m)
    }

    /// The user's focus row: OR of its direct groups' projected rows.
    /// Unchecked hot path: the caller has checked the witness.
    #[inline]
    pub fn row_of(&self, c: &Compiled, u: u16) -> Row {
        let mut r = [0u64; 4];
        for &d in c.b.user_groups.row(u as usize) {
            or(&mut r, &self.proj[d as usize]);
        }
        r
    }

    /// The projected row of group `g` (tests compare projections).
    pub fn projected(&self, g: usize) -> Row {
        self.proj[g]
    }

    /// A checked question: is `u` in every one of `groups` through the
    /// focus's chain? Refuses rather than answers from a stale or partial
    /// focus.
    pub fn ask(
        &self,
        c: &Compiled,
        current: &Witness,
        u: u16,
        groups: &[u16],
    ) -> Result<bool, Refusal> {
        self.check(current)?;
        c.b.check(current)?;
        let m = self.all_of(groups)?;
        Ok(holds(&self.row_of(c, u), &m))
    }

    /// Bytes retained: slot list, reverse map, projected rows.
    pub fn bytes(&self) -> usize {
        self.ids.len() * 2 + self.slot_of.len() * 2 + self.proj.len() * 32
    }
}

#[inline]
pub fn holds(row: &Row, mask: &Row) -> bool {
    (0..4).all(|k| row[k] & mask[k] == mask[k])
}

/// [`Focus`] plus one precomputed focus row per user (eager).
pub struct FocusUsers {
    pub focus: Focus,
    pub rows: Vec<Row>,
}

impl FocusUsers {
    pub fn from_focus(c: &Compiled, focus: Focus) -> FocusUsers {
        // `users` may be 65,536 (the full u16 ordinal space): iterate as usize.
        let rows = (0..c.b.users).map(|u| focus.row_of(c, u as u16)).collect();
        FocusUsers { focus, rows }
    }

    pub fn bind(
        c: &Compiled,
        chain: Chain,
        groups: &[u16],
        current: &Witness,
    ) -> Result<FocusUsers, Refusal> {
        Ok(FocusUsers::from_focus(
            c,
            Focus::bind(c, chain, groups, current)?,
        ))
    }

    /// Recompute the rows of `users` (after their direct memberships
    /// changed, or after `refresh_nesting` changed projected rows they read).
    pub fn refresh_users(&mut self, c: &Compiled, users: &[u16]) {
        for &u in users {
            self.rows[u as usize] = self.focus.row_of(c, u);
        }
    }

    /// After `c.add_nesting(child, _)`: re-project, then recompute the rows
    /// of the users that read a re-projected row (direct members of the
    /// child's downward closure).
    pub fn refresh_nesting(&mut self, c: &Compiled, child: u16) {
        self.focus.refresh_nesting(c, child);
        let mut users = Vec::new();
        for g in ones(c.down_row(self.focus.chain, child as usize)) {
            users.extend_from_slice(c.b.group_users.row(g));
        }
        self.refresh_users(c, &users);
    }

    pub fn bytes(&self) -> usize {
        self.focus.bytes() + self.rows.len() * 32
    }
}

/// [`Focus`] plus per-user rows computed on first use (lazy). `have` marks
/// the rows that are filled; a refresh clears the marks of the users it
/// names instead of recomputing them.
pub struct FocusLazy {
    pub focus: Focus,
    rows: Vec<Row>,
    have: Vec<u64>,
    pub filled: usize,
}

impl FocusLazy {
    pub fn from_focus(c: &Compiled, focus: Focus) -> FocusLazy {
        FocusLazy {
            focus,
            rows: vec![[0; 4]; c.b.users],
            have: vec![0; c.b.users.div_ceil(64)],
            filled: 0,
        }
    }

    #[inline]
    pub fn row(&mut self, c: &Compiled, u: u16) -> Row {
        let i = u as usize;
        if self.have[i >> 6] >> (i & 63) & 1 == 0 {
            self.rows[i] = self.focus.row_of(c, u);
            self.have[i >> 6] |= 1 << (i & 63);
            self.filled += 1;
        }
        self.rows[i]
    }

    pub fn refresh_users(&mut self, users: &[u16]) {
        for &u in users {
            let i = u as usize;
            if self.have[i >> 6] >> (i & 63) & 1 == 1 {
                self.have[i >> 6] &= !(1 << (i & 63));
                self.filled -= 1;
            }
        }
    }

    /// Drop every cached row (after a nesting refresh, any user may read a
    /// changed projected row).
    pub fn invalidate(&mut self) {
        self.have.fill(0);
        self.filled = 0;
    }

    pub fn bytes(&self) -> usize {
        self.focus.bytes() + self.rows.len() * 32 + self.have.len() * 8
    }
}
