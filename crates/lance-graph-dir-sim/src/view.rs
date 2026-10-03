//! A version as `shared base snapshot + overlay`.
//!
//! Structural sharing: every version derived from one observation borrows
//! the same [`Snapshot`] (held by `Arc` in the store). What a version adds is
//! an [`Overlay`] whose size is proportional to its accumulated delta:
//!
//! * added membership rows (a tiny second relation),
//! * a removed-rows bitmap over the base relation — allocated only if
//!   something was removed,
//! * per-attribute override maps `ordinal → (value id, key id)`.
//!
//! Queries run over the base lanes gated by "still live" planes, plus over
//! the overlay rows, and fold the two results (a union is a sum of counts or
//! an OR of masks). The base is never copied.

use crate::snapshot::{bit, clear_bit, set_bit, Dicts, Snapshot, NONE};
use lance_graph_mask_risc::words_for;
use ogar_dir_core::Guid128;
use ogar_dir_sim::{Attribute, Change};
use std::collections::BTreeMap;

/// Why a change does not apply to a version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// `SetAttribute` on a node that does not exist.
    UnknownNode(Guid128),
    /// `SetAttribute`'s `from` no longer holds (stale compare-and-set).
    Stale {
        /// Node.
        node: Guid128,
        /// Attribute.
        attribute: Attribute,
        /// What the change expected.
        expected: Option<String>,
        /// What the version holds.
        actual: Option<String>,
    },
    /// The membership already holds (add) / does not hold (remove).
    NoOp(Change),
    /// A value in the change was never interned (store bug, not input).
    Uninterned,
}

/// The delta-sized part of a version.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Overlay {
    /// Added memberships: identity pair → ordinals (`NONE` if unresolved).
    pub(crate) added: BTreeMap<(Guid128, Guid128), (u32, u32)>,
    /// Removed base membership rows; `None` until the first removal.
    pub(crate) removed: Option<Vec<u64>>,
    /// UPN overrides: ordinal → (value id, key id), `NONE` = cleared.
    pub(crate) upn: BTreeMap<u32, (u32, u32)>,
    /// Primary-SMTP overrides.
    pub(crate) smtp: BTreeMap<u32, (u32, u32)>,
}

impl Overlay {
    /// Number of delta entries held (memberships + overrides).
    pub(crate) fn delta_len(&self) -> usize {
        self.added.len()
            + self
                .removed
                .as_ref()
                .map_or(0, |r| r.iter().map(|w| w.count_ones() as usize).sum())
            + self.upn.len()
            + self.smtp.len()
    }
    fn overrides(&self, a: Attribute) -> &BTreeMap<u32, (u32, u32)> {
        match a {
            Attribute::Upn => &self.upn,
            Attribute::PrimarySmtp => &self.smtp,
        }
    }
}

/// A coherent read of one version: shared snapshot + overlay.
#[derive(Debug)]
pub struct View<'s> {
    pub(crate) snap: &'s Snapshot,
    pub(crate) dicts: &'s Dicts,
    pub(crate) ov: Overlay,
}

impl<'s> View<'s> {
    pub(crate) fn new(snap: &'s Snapshot, dicts: &'s Dicts, ov: Overlay) -> Self {
        Self { snap, dicts, ov }
    }

    /// The shared snapshot (identity check for structural sharing).
    pub fn snapshot(&self) -> &'s Snapshot {
        self.snap
    }
    /// Delta entries this version holds over its snapshot.
    pub fn delta_len(&self) -> usize {
        self.ov.delta_len()
    }
    /// Dense execution ordinal of an identity.
    pub fn ordinal(&self, g: &Guid128) -> Option<u32> {
        self.snap.ordinal(g)
    }
    /// Identity of an ordinal.
    pub fn guid(&self, o: u32) -> Option<Guid128> {
        self.snap.guid(o)
    }
    /// Node count (the population width).
    pub fn len(&self) -> usize {
        self.snap.len()
    }
    /// True if the version has no nodes.
    pub fn is_empty(&self) -> bool {
        self.snap.is_empty()
    }
    /// Active users as a resident bit plane (borrowed).
    pub fn active_users(&self) -> &'s [u64] {
        &self.snap.active_user
    }

    /// Effective membership, by identity. `O(log n)` + overlay lookup.
    pub fn is_member(&self, user: &Guid128, group: &Guid128) -> bool {
        if self.ov.added.contains_key(&(*user, *group)) {
            return true;
        }
        self.snap
            .member_row(user, group)
            .is_some_and(|r| !self.ov.removed.as_ref().is_some_and(|rm| bit(rm, r)))
    }

    /// Effective raw value of an attribute — the one place a value is
    /// resolved to a string (compare-and-set, evidence, plan).
    pub fn attr(&self, node: u32, a: Attribute) -> Option<&'s str> {
        let val = match self.ov.overrides(a).get(&node) {
            Some((v, _)) => *v,
            None => match a {
                Attribute::Upn => *self.snap.upn_val.get(node as usize)?,
                Attribute::PrimarySmtp => *self.snap.smtp_val.get(node as usize)?,
            },
        };
        (val != NONE)
            .then(|| self.dicts.values.resolve(val))
            .flatten()
    }

    /// The base membership rows still live in this version.
    pub(crate) fn live_rows(&self) -> std::borrow::Cow<'s, [u64]> {
        match &self.ov.removed {
            None => std::borrow::Cow::Borrowed(&self.snap.m_all),
            Some(rm) => std::borrow::Cow::Owned(
                self.snap
                    .m_all
                    .iter()
                    .zip(rm)
                    .map(|(a, r)| a & !r)
                    .collect(),
            ),
        }
    }

    /// `active_user` minus the nodes whose attribute `a` is overridden —
    /// the base rows that still own their observed value.
    pub(crate) fn live_owners(&self, a: Attribute) -> std::borrow::Cow<'s, [u64]> {
        let ov = self.ov.overrides(a);
        if ov.is_empty() {
            return std::borrow::Cow::Borrowed(&self.snap.active_user);
        }
        let mut p = self.snap.active_user.clone();
        for o in ov.keys() {
            clear_bit(&mut p, *o as usize);
        }
        std::borrow::Cow::Owned(p)
    }

    /// Apply one change to the overlay. Pure with respect to everything but
    /// `self.ov`; the snapshot is never touched.
    pub(crate) fn apply(&mut self, c: &Change) -> Result<(), ApplyError> {
        match c {
            Change::AddMembership { user, group } => {
                if self.is_member(user, group) {
                    return Err(ApplyError::NoOp(c.clone()));
                }
                match self.snap.member_row(user, group) {
                    Some(r) => {
                        let rm = self
                            .ov
                            .removed
                            .as_mut()
                            .expect("absent base row was removed");
                        clear_bit(rm, r as usize);
                        if rm.iter().all(|w| *w == 0) {
                            self.ov.removed = None;
                        }
                    }
                    None => {
                        let ords = (
                            self.ordinal(user).unwrap_or(NONE),
                            self.ordinal(group).unwrap_or(NONE),
                        );
                        self.ov.added.insert((*user, *group), ords);
                    }
                }
            }
            Change::RemoveMembership { user, group } => {
                if self.ov.added.remove(&(*user, *group)).is_some() {
                    return Ok(());
                }
                match self.snap.member_row(user, group) {
                    Some(r) if self.is_member(user, group) => {
                        let n = self.snap.membership_rows();
                        let rm = self.ov.removed.get_or_insert_with(|| vec![0; words_for(n)]);
                        set_bit(rm, r as usize);
                    }
                    _ => return Err(ApplyError::NoOp(c.clone())),
                }
            }
            Change::SetAttribute {
                node,
                attribute,
                from,
                to,
            } => {
                let o = self.ordinal(node).ok_or(ApplyError::UnknownNode(*node))?;
                let actual = self.attr(o, *attribute);
                if actual != from.as_deref() {
                    return Err(ApplyError::Stale {
                        node: *node,
                        attribute: *attribute,
                        expected: from.clone(),
                        actual: actual.map(str::to_string),
                    });
                }
                let ids = self
                    .dicts
                    .lookup_attr(to.as_deref())
                    .ok_or(ApplyError::Uninterned)?;
                let base = match attribute {
                    Attribute::Upn => {
                        (self.snap.upn_val[o as usize], self.snap.upn_key[o as usize])
                    }
                    Attribute::PrimarySmtp => (
                        self.snap.smtp_val[o as usize],
                        self.snap.smtp_key[o as usize],
                    ),
                };
                let map = match attribute {
                    Attribute::Upn => &mut self.ov.upn,
                    Attribute::PrimarySmtp => &mut self.ov.smtp,
                };
                // Net effect only: setting a value back to the observed one
                // removes the override instead of recording a no-op change.
                if ids.0 == base.0 {
                    map.remove(&o);
                } else {
                    map.insert(o, ids);
                }
            }
        }
        Ok(())
    }
}
