//! Arm F: a 256-slot focus over selected group identities, compiled over B+.
//!
//! The focus binds up to 256 group ordinals to local slots once. Every
//! group's B+ closure row (every group, upward) is then projected onto the
//! focus slots: 4 words per group instead of `ceil(groups / 64)`. A user's
//! position in the focus is the OR of its direct groups' projected rows, and
//! a role x location question is a 4-word mask test.
//!
//! Two variants:
//!
//! * [`Focus`]: per-group projected rows only. Each evaluation ORs the
//!   user's direct groups' rows (4 words each).
//! * [`FocusUsers`]: additionally one 4-word row per user, computed once.
//!   This is the per-user pointer cache the proposal asked to keep out
//!   unless measured; it is here so the measurement exists.
//!
//! No per-user 256 x 256 tile is allocated. The projection reads closure
//! rows at the focus's bound ordinals: an address translation, done once in
//! preparation and counted. Its admission is `ISS-ADMISSION-PREP-BINDING`.
//!
//! Reverse lookup (group ordinal -> slot) is a direct `u16` array over the
//! group ordinal space (slot + 1, 0 = not bound): one indexed read, no hash,
//! no search. At 1,500 groups that is 3 KB.

use crate::prep::{Compiled, Refusal, Witness, bit};

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
    /// Slot -> group ordinal.
    pub ids: Vec<u16>,
    /// Group ordinal -> slot + 1; 0 = not bound.
    slot_of: Vec<u16>,
    /// Group ordinal -> its closure row (every group, upward) on the slots.
    proj: Vec<Row>,
}

impl Focus {
    /// Bind `groups` (distinct, at most 256) and project every group's
    /// closure row onto them.
    pub fn bind(c: &Compiled, groups: &[u16]) -> Focus {
        assert!(groups.len() <= SLOTS, "a focus binds at most 256 groups");
        let n = c.b.groups;
        let mut slot_of = vec![0u16; n];
        for (s, &g) in groups.iter().enumerate() {
            assert_eq!(slot_of[g as usize], 0, "focus groups must be distinct");
            slot_of[g as usize] = s as u16 + 1;
        }
        let mut proj = vec![[0u64; 4]; n];
        for (g, row) in proj.iter_mut().enumerate() {
            let up = c.row(crate::prep::Chain::All, g);
            for (s, &f) in groups.iter().enumerate() {
                if bit(up, f as usize) {
                    set(row, s);
                }
            }
        }
        Focus {
            witness: c.b.witness,
            ids: groups.to_vec(),
            slot_of,
            proj,
        }
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
    #[inline]
    pub fn row_of(&self, c: &Compiled, u: u16) -> Row {
        let mut r = [0u64; 4];
        for &d in c.b.user_groups.row(u as usize) {
            or(&mut r, &self.proj[d as usize]);
        }
        r
    }

    pub fn bytes(&self) -> usize {
        self.ids.len() * 2 + self.slot_of.len() * 2 + self.proj.len() * 32
    }
}

#[inline]
pub fn holds(row: &Row, mask: &Row) -> bool {
    (0..4).all(|k| row[k] & mask[k] == mask[k])
}

/// [`Focus`] plus one precomputed focus row per user.
pub struct FocusUsers {
    pub focus: Focus,
    pub rows: Vec<Row>,
}

impl FocusUsers {
    pub fn bind(c: &Compiled, groups: &[u16]) -> FocusUsers {
        let focus = Focus::bind(c, groups);
        let rows = (0..c.b.users as u16).map(|u| focus.row_of(c, u)).collect();
        FocusUsers { focus, rows }
    }

    pub fn bytes(&self) -> usize {
        self.focus.bytes() + self.rows.len() * 32
    }
}
