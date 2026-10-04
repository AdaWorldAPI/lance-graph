//! A version as `shared base snapshot + overlay`.
//!
//! Structural sharing: every version derived from one observation borrows
//! the same [`Snapshot`] (held by `Arc` in the store). What a version adds is
//! an [`Overlay`] whose size is proportional to its accumulated delta:
//!
//! * added membership identity pairs,
//! * removed base membership rows (sorted row ids),
//! * per-attribute override maps `ordinal → (value id, key id)`,
//! * created nodes as their own small SoA lanes ([`Created`]),
//! * deleted base nodes (sorted ordinals).
//!
//! Nothing in the overlay is allocated in proportion to the directory: one
//! mutation adds one entry. Queries build their "still live" planes when
//! they run (query scratch, not version state), run over the base lanes and
//! over the overlay's delta rows, and fold the two results. The base is
//! never copied.
//!
//! **Ordinal space of a view.** Base nodes keep their snapshot ordinals
//! `0..n`; created nodes take `n..n + c` in `Guid128` order. A deleted base
//! node keeps its slot but has no ordinal ([`View::ordinal`] returns `None`)
//! and is cleared from every live plane. Ordinals are valid only inside one
//! view and are never stored: the overlay keeps identities wherever an
//! ordinal could shift (added memberships, created nodes).

use crate::snapshot::{bit, clear_bit, set_bit, Dicts, Snapshot, NONE};
use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes};
use lance_graph_quack::{Cmp, Col, Filter, Mask};
use ogar_dir_core::{Guid128, OuHhtl};
use ogar_dir_sim::{Attribute, Change, NodeKind, NodeState};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// Why a change does not apply to a version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// The change names a node that does not exist.
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
    /// `CreateNode` of an identity that already exists.
    NodeExists(Guid128),
    /// `CreateNode` of an observed identity this lineage deleted, with a
    /// different state. Directory identities are not reused; recreating
    /// the observed node unchanged is an undo and is accepted.
    IdentityReused(Guid128),
    /// `DeleteNode`'s expected state no longer holds (stale compare-and-set).
    StaleNode {
        /// Node.
        node: Guid128,
        /// What the change expected to delete.
        expected: Box<NodeState>,
        /// What the version holds.
        actual: Box<NodeState>,
    },
    /// `DeleteNode` of a node that is still a member of, or still has, a
    /// group membership. The change list must remove those edges first.
    NodeHasMemberships(Guid128),
    /// `CreateNode` of a group with `active = false`. Groups have no
    /// enabled flag; their canonical state carries `active = true`.
    InactiveGroup(Guid128),
}

/// Nodes a version created, as SoA lanes sorted by identity. Delta-sized.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Created {
    pub(crate) ids: Vec<Guid128>,
    pub(crate) kind: Vec<NodeKind>,
    pub(crate) active: Vec<bool>,
    pub(crate) upn_val: Vec<u32>,
    pub(crate) upn_key: Vec<u32>,
    pub(crate) smtp_val: Vec<u32>,
    pub(crate) smtp_key: Vec<u32>,
    pub(crate) ou: Vec<Option<OuHhtl>>,
}

impl Created {
    fn find(&self, g: &Guid128) -> Result<usize, usize> {
        self.ids.binary_search(g)
    }
    fn insert(&mut self, at: usize, g: Guid128, s: &NodeState, upn: (u32, u32), smtp: (u32, u32)) {
        self.ids.insert(at, g);
        self.kind.insert(at, s.kind);
        self.active.insert(at, s.active);
        self.upn_val.insert(at, upn.0);
        self.upn_key.insert(at, upn.1);
        self.smtp_val.insert(at, smtp.0);
        self.smtp_key.insert(at, smtp.1);
        self.ou.insert(at, s.ou);
    }
    fn remove(&mut self, at: usize) {
        self.ids.remove(at);
        self.kind.remove(at);
        self.active.remove(at);
        self.upn_val.remove(at);
        self.upn_key.remove(at);
        self.smtp_val.remove(at);
        self.smtp_key.remove(at);
        self.ou.remove(at);
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    fn lanes(&mut self, a: Attribute) -> (&mut Vec<u32>, &mut Vec<u32>) {
        match a {
            Attribute::Upn => (&mut self.upn_val, &mut self.upn_key),
            Attribute::PrimarySmtp => (&mut self.smtp_val, &mut self.smtp_key),
        }
    }
    pub(crate) fn key(&self, a: Attribute) -> &[u32] {
        match a {
            Attribute::Upn => &self.upn_key,
            Attribute::PrimarySmtp => &self.smtp_key,
        }
    }
}

/// The delta-sized part of a version.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Overlay {
    /// Added memberships, by identity (resolved to ordinals per query).
    pub(crate) added: BTreeSet<(Guid128, Guid128)>,
    /// Removed base membership rows.
    pub(crate) removed: BTreeSet<u32>,
    /// UPN overrides of base nodes: ordinal → (value id, key id), `NONE` = cleared.
    pub(crate) upn: BTreeMap<u32, (u32, u32)>,
    /// Primary-SMTP overrides of base nodes.
    pub(crate) smtp: BTreeMap<u32, (u32, u32)>,
    /// Created nodes.
    pub(crate) created: Created,
    /// Deleted base nodes (ordinals).
    pub(crate) deleted: BTreeSet<u32>,
}

impl Overlay {
    /// Number of delta entries held.
    pub(crate) fn delta_len(&self) -> usize {
        self.added.len()
            + self.removed.len()
            + self.upn.len()
            + self.smtp.len()
            + self.created.len()
            + self.deleted.len()
    }
    pub(crate) fn overrides(&self, a: Attribute) -> &BTreeMap<u32, (u32, u32)> {
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
    /// Dense execution ordinal of an existing node. `O(log n + log c)`.
    pub fn ordinal(&self, g: &Guid128) -> Option<u32> {
        match self.snap.ordinal(g) {
            Some(o) if !self.ov.deleted.contains(&o) => Some(o),
            Some(_) => None,
            None => self
                .ov
                .created
                .find(g)
                .ok()
                .map(|i| (self.snap.len() + i) as u32),
        }
    }
    /// Identity of an existing node's ordinal.
    pub fn guid(&self, o: u32) -> Option<Guid128> {
        let n = self.snap.len();
        if (o as usize) < n {
            (!self.ov.deleted.contains(&o)).then(|| self.snap.ids[o as usize])
        } else {
            self.ov.created.ids.get(o as usize - n).copied()
        }
    }
    /// Population width: base slots plus created nodes. A deleted base node
    /// keeps its slot (cleared in every live plane).
    pub fn len(&self) -> usize {
        self.snap.len() + self.ov.created.len()
    }
    /// True if the version has no node slots.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// True if the node exists in this version.
    pub fn exists(&self, g: &Guid128) -> bool {
        self.ordinal(g).is_some()
    }

    /// A base plane widened to [`Self::len`]: deleted base nodes cleared,
    /// created rows set where `created` says so. Borrowed when the overlay
    /// has neither (the common simulation case).
    fn live_plane(&self, base: &'s [u64], created: impl Fn(usize) -> bool) -> Cow<'s, [u64]> {
        let (n, c) = (self.snap.len(), self.ov.created.len());
        if c == 0 && self.ov.deleted.is_empty() {
            return Cow::Borrowed(base);
        }
        let mut p = base.to_vec();
        p.resize(words_for(n + c), 0);
        for &o in &self.ov.deleted {
            clear_bit(&mut p, o as usize);
        }
        for i in (0..c).filter(|&i| created(i)) {
            set_bit(&mut p, n + i);
        }
        Cow::Owned(p)
    }
    /// A base-width plane with deleted base nodes cleared.
    pub(crate) fn base_live(&self, base: &'s [u64]) -> Cow<'s, [u64]> {
        if self.ov.deleted.is_empty() {
            return Cow::Borrowed(base);
        }
        let mut p = base.to_vec();
        for &o in &self.ov.deleted {
            clear_bit(&mut p, o as usize);
        }
        Cow::Owned(p)
    }
    /// Active users as a bit plane over [`Self::len`].
    pub fn active_users(&self) -> Cow<'s, [u64]> {
        let cr = &self.ov.created;
        self.live_plane(&self.snap.active_user, |i| {
            cr.kind[i] == NodeKind::User && cr.active[i]
        })
    }
    /// Existing nodes of `kind` as a bit plane over [`Self::len`].
    pub(crate) fn kind_plane(&self, kind: NodeKind) -> Cow<'s, [u64]> {
        let base = match kind {
            NodeKind::User => &self.snap.user,
            NodeKind::Group => &self.snap.group,
        };
        let cr = &self.ov.created;
        self.live_plane(base, |i| cr.kind[i] == kind)
    }
    /// Effective membership, by identity. `O(log m)` + overlay lookup.
    pub fn is_member(&self, user: &Guid128, group: &Guid128) -> bool {
        if self.ov.added.contains(&(*user, *group)) {
            return true;
        }
        self.snap
            .member_row(user, group)
            .is_some_and(|r| !self.ov.removed.contains(&r))
    }

    fn attr_ids(&self, o: u32, a: Attribute) -> Option<(u32, u32)> {
        let n = self.snap.len();
        if (o as usize) >= n {
            let i = o as usize - n;
            let cr = &self.ov.created;
            return Some(match a {
                Attribute::Upn => (*cr.upn_val.get(i)?, cr.upn_key[i]),
                Attribute::PrimarySmtp => (*cr.smtp_val.get(i)?, cr.smtp_key[i]),
            });
        }
        if let Some(ids) = self.ov.overrides(a).get(&o) {
            return Some(*ids);
        }
        let s = self.snap;
        Some(match a {
            Attribute::Upn => (*s.upn_val.get(o as usize)?, s.upn_key[o as usize]),
            Attribute::PrimarySmtp => (*s.smtp_val.get(o as usize)?, s.smtp_key[o as usize]),
        })
    }

    /// Effective raw value of an attribute — the one place a value is
    /// resolved to a string (compare-and-set, evidence, plan).
    pub fn attr(&self, node: u32, a: Attribute) -> Option<&'s str> {
        let (val, _) = self.attr_ids(node, a)?;
        (val != NONE)
            .then(|| self.dicts.values.resolve(val))
            .flatten()
    }

    /// The canonical semantic state of an existing node (evidence /
    /// compare-and-set boundary: resolves strings).
    pub fn node_state(&self, g: &Guid128) -> Option<NodeState> {
        let o = self.ordinal(g)?;
        let n = self.snap.len();
        let (kind, active, ou) = if (o as usize) < n {
            let s = self.snap;
            let ou = bit(&s.ou_present, o).then(|| s.ou[o as usize]);
            if bit(&s.user, o) {
                (NodeKind::User, bit(&s.active_user, o), ou)
            } else {
                // Groups have no enabled flag (see `ApplyError::InactiveGroup`).
                (NodeKind::Group, true, ou)
            }
        } else {
            let cr = &self.ov.created;
            let i = o as usize - n;
            (cr.kind[i], cr.active[i], cr.ou[i])
        };
        let text = |a| self.attr(o, a).map(str::to_string);
        Some(NodeState {
            kind,
            active,
            upn: text(Attribute::Upn),
            primary_smtp: text(Attribute::PrimarySmtp),
            ou,
        })
    }

    /// The base membership rows still live in this version.
    pub(crate) fn live_rows(&self) -> Cow<'s, [u64]> {
        if self.ov.removed.is_empty() {
            return Cow::Borrowed(&self.snap.m_all);
        }
        let mut p = self.snap.m_all.clone();
        for &r in &self.ov.removed {
            clear_bit(&mut p, r as usize);
        }
        Cow::Owned(p)
    }

    /// Base active users (base width) minus deleted nodes and the nodes
    /// whose attribute `a` is overridden — the base rows that still own
    /// their observed value.
    pub(crate) fn live_owners(&self, a: Attribute) -> Cow<'s, [u64]> {
        let ov = self.ov.overrides(a);
        if ov.is_empty() && self.ov.deleted.is_empty() {
            return Cow::Borrowed(&self.snap.active_user);
        }
        let mut p = self.snap.active_user.clone();
        for o in ov.keys().chain(&self.ov.deleted) {
            clear_bit(&mut p, *o as usize);
        }
        Cow::Owned(p)
    }

    /// Added memberships as delta-sized ordinal lanes (`NONE` = endpoint
    /// does not exist in this version) plus their identities.
    pub(crate) fn added_rows(&self) -> (Vec<u32>, Vec<u32>, Vec<(Guid128, Guid128)>) {
        let mut u = Vec::with_capacity(self.ov.added.len());
        let mut g = Vec::with_capacity(self.ov.added.len());
        for (a, b) in &self.ov.added {
            u.push(self.ordinal(a).unwrap_or(NONE));
            g.push(self.ordinal(b).unwrap_or(NONE));
        }
        (u, g, self.ov.added.iter().copied().collect())
    }

    /// Whether `node` still takes part in a live membership, on either side.
    ///
    /// Base relation: one `Count` program over the two membership lanes
    /// gated by the observed rows (borrowed, no plane built), minus the
    /// removed rows touching the node (delta-sized). The relation is sorted
    /// by user, not by group, so the group side has no index: the work is a
    /// scan of the membership lanes, the allocation is one scratch tile.
    pub(crate) fn has_membership(&self, node: &Guid128) -> bool {
        if self.ov.added.iter().any(|(u, g)| u == node || g == node) {
            return true;
        }
        let s = self.snap;
        let Some(o) = s.ordinal(node) else {
            return false;
        };
        let lanes = [LaneRef::U32(&s.m_user), LaneRef::U32(&s.m_group)];
        let masks: [&[u64]; 1] = [&s.m_all];
        let planes = Planes {
            n_rows: s.membership_rows(),
            masks: &masks,
            lanes: &lanes,
        };
        let observed = crate::exec::count(
            Filter::and([
                Filter::plane(Mask(0)),
                Filter::or([
                    Filter::cmp(Col(0), Cmp::EqU32(o)),
                    Filter::cmp(Col(1), Cmp::EqU32(o)),
                ]),
            ]),
            &planes,
            &Foreign::NONE,
        );
        let removed = self
            .ov
            .removed
            .iter()
            .filter(|&&r| s.m_user[r as usize] == o || s.m_group[r as usize] == o)
            .count();
        observed > removed
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
                        self.ov.removed.remove(&r);
                    }
                    None => {
                        self.ov.added.insert((*user, *group));
                    }
                }
            }
            Change::RemoveMembership { user, group } => {
                if self.ov.added.remove(&(*user, *group)) {
                    return Ok(());
                }
                match self.snap.member_row(user, group) {
                    Some(r) if self.is_member(user, group) => {
                        self.ov.removed.insert(r);
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
                let n = self.snap.len();
                if (o as usize) >= n {
                    let (val, key) = self.ov.created.lanes(*attribute);
                    val[o as usize - n] = ids.0;
                    key[o as usize - n] = ids.1;
                    return Ok(());
                }
                let base = match attribute {
                    Attribute::Upn => self.snap.upn_val[o as usize],
                    Attribute::PrimarySmtp => self.snap.smtp_val[o as usize],
                };
                let map = match attribute {
                    Attribute::Upn => &mut self.ov.upn,
                    Attribute::PrimarySmtp => &mut self.ov.smtp,
                };
                // Net effect only: setting a value back to the observed one
                // removes the override instead of recording a no-op change.
                if ids.0 == base {
                    map.remove(&o);
                } else {
                    map.insert(o, ids);
                }
            }
            Change::CreateNode { node, state } => {
                if state.kind == NodeKind::Group && !state.active {
                    return Err(ApplyError::InactiveGroup(*node));
                }
                // Identity uniqueness: one node per Guid128 in a version.
                if self.exists(node) {
                    return Err(ApplyError::NodeExists(*node));
                }
                if let Some(o) = self.snap.ordinal(node) {
                    // Deleted in this lineage: recreating the observed node
                    // unchanged is an undo; anything else reuses an identity.
                    self.ov.deleted.remove(&o);
                    if self.node_state(node).as_ref() != Some(state) {
                        self.ov.deleted.insert(o);
                        return Err(ApplyError::IdentityReused(*node));
                    }
                    return Ok(());
                }
                let upn = self
                    .dicts
                    .lookup_attr(state.upn.as_deref())
                    .ok_or(ApplyError::Uninterned)?;
                let smtp = self
                    .dicts
                    .lookup_attr(state.primary_smtp.as_deref())
                    .ok_or(ApplyError::Uninterned)?;
                let at = self.ov.created.find(node).unwrap_err();
                self.ov.created.insert(at, *node, state, upn, smtp);
            }
            Change::DeleteNode { node, state } => {
                let actual = self
                    .node_state(node)
                    .ok_or(ApplyError::UnknownNode(*node))?;
                if &actual != state {
                    return Err(ApplyError::StaleNode {
                        node: *node,
                        expected: Box::new(state.clone()),
                        actual: Box::new(actual),
                    });
                }
                // Never strand an edge: the change list removes them first.
                if self.has_membership(node) {
                    return Err(ApplyError::NodeHasMemberships(*node));
                }
                match self.ov.created.find(node) {
                    // Create-then-delete nets out to nothing.
                    Ok(i) => self.ov.created.remove(i),
                    Err(_) => {
                        let Some(o) = self.snap.ordinal(node) else {
                            return Err(ApplyError::UnknownNode(*node));
                        };
                        self.ov.upn.remove(&o);
                        self.ov.smtp.remove(&o);
                        self.ov.deleted.insert(o);
                    }
                }
            }
        }
        Ok(())
    }
}
