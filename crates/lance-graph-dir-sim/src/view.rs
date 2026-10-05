//! A version as `shared base snapshot + overlay`.
//!
//! Structural sharing: every version derived from one observation borrows
//! the same [`Snapshot`] (held by `Arc` in the store). What a version adds is
//! an [`Overlay`] whose size is proportional to its accumulated delta:
//!
//! * added membership identity pairs,
//! * removed membership rows (sorted row ids; unresolved rows by identity),
//! * per population: attribute overrides `ordinal → (value, key)`, created
//!   nodes as their own small SoA lanes ([`Created`]), and deleted base
//!   ordinals.
//!
//! Nothing in the overlay is allocated in proportion to the directory: one
//! mutation adds one entry. Queries build their "still live" planes when
//! they run (query scratch, not version state), run over the base lanes and
//! over the overlay's delta rows, and fold the two results. The base is
//! never copied.
//!
//! **Ordinal spaces of a view.** Users and groups are numbered separately.
//! Base nodes keep their snapshot ordinals `0..n`; created nodes take
//! `n..n + c` in `Guid128` order; a population never exceeds 65,536. A
//! deleted base node keeps its slot but has no ordinal and is cleared from
//! every live plane. Ordinals are valid only inside one view; the overlay
//! keeps identities wherever an ordinal could shift.

use crate::snapshot::{
    bit, clear_bit, set_bit, Dicts, GroupOrdinal, Population, Snapshot, UserOrdinal, MAX_GROUPS,
    MAX_USERS, NONE,
};
use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes};
use lance_graph_quack::{Cmp, Col, Filter, Mask};
use ogar_dir_core::{Dn128, Guid128};
use ogar_dir_sim::{Attribute, Change, NodeKind, NodeState, ValueId};
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
        expected: Option<ValueId>,
        /// What the version holds.
        actual: Option<ValueId>,
    },
    /// The membership already holds (add) / does not hold (remove).
    NoOp(Change),
    /// A value id the store never issued (caller bug, not input).
    Uninterned(ValueId),
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
    /// `CreateNode` into a population already at its bound
    /// ([`MAX_USERS`] / [`MAX_GROUPS`]).
    PopulationFull(NodeKind),
}

/// Nodes a version created in one population, as SoA lanes sorted by
/// identity. Delta-sized.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Created {
    pub(crate) ids: Vec<Guid128>,
    pub(crate) active: Vec<bool>,
    pub(crate) upn_val: Vec<u32>,
    pub(crate) upn_key: Vec<u32>,
    pub(crate) smtp_val: Vec<u32>,
    pub(crate) smtp_key: Vec<u32>,
    pub(crate) dn: Vec<Option<Dn128>>,
}

impl Created {
    fn find(&self, g: &Guid128) -> Result<usize, usize> {
        self.ids.binary_search(g)
    }
    fn insert(&mut self, at: usize, g: Guid128, s: &NodeState, upn: (u32, u32), smtp: (u32, u32)) {
        self.ids.insert(at, g);
        self.active.insert(at, s.active);
        self.upn_val.insert(at, upn.0);
        self.upn_key.insert(at, upn.1);
        self.smtp_val.insert(at, smtp.0);
        self.smtp_key.insert(at, smtp.1);
        self.dn.insert(at, s.dn);
    }
    fn remove(&mut self, at: usize) {
        self.ids.remove(at);
        self.active.remove(at);
        self.upn_val.remove(at);
        self.upn_key.remove(at);
        self.smtp_val.remove(at);
        self.smtp_key.remove(at);
        self.dn.remove(at);
    }
    pub(crate) fn len(&self) -> usize {
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

/// One population's share of a version's delta.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PopOverlay {
    /// UPN overrides of base nodes: ordinal → (value id, key id), `NONE` = cleared.
    pub(crate) upn: BTreeMap<u16, (u32, u32)>,
    /// Primary-SMTP overrides of base nodes.
    pub(crate) smtp: BTreeMap<u16, (u32, u32)>,
    /// Created nodes.
    pub(crate) created: Created,
    /// Deleted base nodes (ordinals).
    pub(crate) deleted: BTreeSet<u16>,
}

impl PopOverlay {
    fn len(&self) -> usize {
        self.upn.len() + self.smtp.len() + self.created.len() + self.deleted.len()
    }
    pub(crate) fn overrides(&self, a: Attribute) -> &BTreeMap<u16, (u32, u32)> {
        match a {
            Attribute::Upn => &self.upn,
            Attribute::PrimarySmtp => &self.smtp,
        }
    }
}

/// The delta-sized part of a version.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Overlay {
    /// Added memberships, by identity (resolved to ordinals per query).
    pub(crate) added: BTreeSet<(Guid128, Guid128)>,
    /// Removed resolved membership rows.
    pub(crate) removed: BTreeSet<u32>,
    /// Removed unresolved (observed, dangling) memberships.
    pub(crate) removed_unresolved: BTreeSet<(Guid128, Guid128)>,
    /// User population delta.
    pub(crate) users: PopOverlay,
    /// Group population delta.
    pub(crate) groups: PopOverlay,
}

impl Overlay {
    /// Number of delta entries held.
    pub(crate) fn delta_len(&self) -> usize {
        self.added.len()
            + self.removed.len()
            + self.removed_unresolved.len()
            + self.users.len()
            + self.groups.len()
    }
}

/// A node position inside a view: its population and its index there.
pub(crate) type Slot = (NodeKind, usize);

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
    /// The store's label/value table (egress formatting only).
    pub fn dicts(&self) -> &'s Dicts {
        self.dicts
    }
    /// Delta entries this version holds over its snapshot.
    pub fn delta_len(&self) -> usize {
        self.ov.delta_len()
    }

    pub(crate) fn pop(&self, kind: NodeKind) -> (&'s Population, &PopOverlay) {
        match kind {
            NodeKind::User => (&self.snap.users, &self.ov.users),
            NodeKind::Group => (&self.snap.groups, &self.ov.groups),
        }
    }
    fn pop_mut(&mut self, kind: NodeKind) -> &mut PopOverlay {
        match kind {
            NodeKind::User => &mut self.ov.users,
            NodeKind::Group => &mut self.ov.groups,
        }
    }
    /// Slot count of a population: base slots plus created nodes. A deleted
    /// base node keeps its slot (cleared in every live plane).
    pub(crate) fn len_in(&self, kind: NodeKind) -> usize {
        let (p, o) = self.pop(kind);
        p.len() + o.created.len()
    }
    /// Index of an existing node within a population.
    pub(crate) fn index_in(&self, kind: NodeKind, g: &Guid128) -> Option<usize> {
        let (p, o) = self.pop(kind);
        match p.ordinal(g) {
            Some(i) if !o.deleted.contains(&i) => Some(usize::from(i)),
            Some(_) => None,
            None => o.created.find(g).ok().map(|i| p.len() + i),
        }
    }
    /// Identity of an existing node's index.
    pub(crate) fn guid_in(&self, kind: NodeKind, i: usize) -> Option<Guid128> {
        let (p, o) = self.pop(kind);
        if i < p.len() {
            (!o.deleted.contains(&(i as u16))).then(|| p.ids[i])
        } else {
            o.created.ids.get(i - p.len()).copied()
        }
    }
    /// Where an existing node lives.
    pub(crate) fn locate(&self, g: &Guid128) -> Option<Slot> {
        [NodeKind::User, NodeKind::Group]
            .into_iter()
            .find_map(|k| self.index_in(k, g).map(|i| (k, i)))
    }

    /// User count, including created users (the user ordinal universe).
    pub fn users_len(&self) -> usize {
        self.len_in(NodeKind::User)
    }
    /// Group count, including created groups.
    pub fn groups_len(&self) -> usize {
        self.len_in(NodeKind::Group)
    }
    /// Ordinal of an existing user. `O(log n + log c)`.
    pub fn user_ordinal(&self, g: &Guid128) -> Option<UserOrdinal> {
        self.index_in(NodeKind::User, g)
            .map(|i| UserOrdinal(i as u16))
    }
    /// Ordinal of an existing group.
    pub fn group_ordinal(&self, g: &Guid128) -> Option<GroupOrdinal> {
        self.index_in(NodeKind::Group, g)
            .map(|i| GroupOrdinal(i as u16))
    }
    /// Identity of a user ordinal.
    pub fn user_guid(&self, o: UserOrdinal) -> Option<Guid128> {
        self.guid_in(NodeKind::User, usize::from(o.0))
    }
    /// Identity of a group ordinal.
    pub fn group_guid(&self, o: GroupOrdinal) -> Option<Guid128> {
        self.guid_in(NodeKind::Group, usize::from(o.0))
    }
    /// True if the node exists in this version.
    pub fn exists(&self, g: &Guid128) -> bool {
        self.locate(g).is_some()
    }

    /// A base plane widened to the population's slot count: deleted base
    /// nodes cleared, created rows set where `created` says so. Borrowed
    /// when the overlay touches neither (the common simulation case).
    fn live_plane(
        &self,
        kind: NodeKind,
        base: &'s [u64],
        created: impl Fn(usize) -> bool,
    ) -> Cow<'s, [u64]> {
        let (p, o) = self.pop(kind);
        let (n, c) = (p.len(), o.created.len());
        if c == 0 && o.deleted.is_empty() {
            return Cow::Borrowed(base);
        }
        let mut plane = base.to_vec();
        plane.resize(words_for(n + c), 0);
        for &d in &o.deleted {
            clear_bit(&mut plane, usize::from(d));
        }
        for i in (0..c).filter(|&i| created(i)) {
            set_bit(&mut plane, n + i);
        }
        Cow::Owned(plane)
    }
    /// Active users as a bit plane over the user ordinals.
    pub fn active_users(&self) -> Cow<'s, [u64]> {
        let cr = &self.ov.users.created;
        self.live_plane(NodeKind::User, &self.snap.users.active, |i| cr.active[i])
    }
    /// Existing nodes of a population as a bit plane over its ordinals.
    pub(crate) fn existing(&self, kind: NodeKind) -> Cow<'s, [u64]> {
        let (p, _) = self.pop(kind);
        self.live_plane(kind, &p.all, |_| true)
    }
    /// A base-width plane with deleted base nodes cleared.
    pub(crate) fn base_live(&self, kind: NodeKind, base: &'s [u64]) -> Cow<'s, [u64]> {
        let (_, o) = self.pop(kind);
        if o.deleted.is_empty() {
            return Cow::Borrowed(base);
        }
        let mut plane = base.to_vec();
        for &d in &o.deleted {
            clear_bit(&mut plane, usize::from(d));
        }
        Cow::Owned(plane)
    }

    /// Effective membership, by identity. `O(log m)` + overlay lookup.
    pub fn is_member(&self, user: &Guid128, group: &Guid128) -> bool {
        if self.ov.added.contains(&(*user, *group)) {
            return true;
        }
        if let Some(r) = self.snap.member_row(user, group) {
            return !self.ov.removed.contains(&r);
        }
        self.snap.is_unresolved_member(user, group)
            && !self.ov.removed_unresolved.contains(&(*user, *group))
    }

    fn attr_ids(&self, (kind, i): Slot, a: Attribute) -> Option<(u32, u32)> {
        let (p, o) = self.pop(kind);
        if i >= p.len() {
            let c = &o.created;
            let j = i - p.len();
            return Some(match a {
                Attribute::Upn => (*c.upn_val.get(j)?, c.upn_key[j]),
                Attribute::PrimarySmtp => (*c.smtp_val.get(j)?, c.smtp_key[j]),
            });
        }
        if let Some(ids) = o.overrides(a).get(&(i as u16)) {
            return Some(*ids);
        }
        Some(match a {
            Attribute::Upn => (*p.upn_val.get(i)?, p.upn_key[i]),
            Attribute::PrimarySmtp => (*p.smtp_val.get(i)?, p.smtp_key[i]),
        })
    }

    /// Effective value of an attribute of an existing node.
    pub fn attr(&self, node: &Guid128, a: Attribute) -> Option<ValueId> {
        self.slot_attr(self.locate(node)?, a)
    }
    pub(crate) fn slot_attr(&self, slot: Slot, a: Attribute) -> Option<ValueId> {
        let (v, _) = self.attr_ids(slot, a)?;
        (v != NONE).then_some(ValueId(v))
    }

    /// The canonical semantic state of an existing node — ids only.
    pub fn node_state(&self, g: &Guid128) -> Option<NodeState> {
        let (kind, i) = self.locate(g)?;
        let (p, o) = self.pop(kind);
        let (active, dn) = if i < p.len() {
            let active = kind == NodeKind::Group || bit(&p.active, i);
            (active, p.dn_of(i))
        } else {
            let j = i - p.len();
            (o.created.active[j], o.created.dn[j])
        };
        Some(NodeState {
            kind,
            active,
            upn: self.slot_attr((kind, i), Attribute::Upn),
            primary_smtp: self.slot_attr((kind, i), Attribute::PrimarySmtp),
            dn,
        })
    }

    /// The resolved membership rows still live in this version.
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

    /// Base active users (base width) minus deleted users and the users
    /// whose attribute `a` is overridden — the base rows that still own
    /// their observed value.
    pub(crate) fn live_owners(&self, a: Attribute) -> Cow<'s, [u64]> {
        let o = &self.ov.users;
        let ov = o.overrides(a);
        if ov.is_empty() && o.deleted.is_empty() {
            return Cow::Borrowed(&self.snap.users.active);
        }
        let mut p = self.snap.users.active.clone();
        for d in ov.keys().chain(&o.deleted) {
            clear_bit(&mut p, usize::from(*d));
        }
        Cow::Owned(p)
    }

    /// The live memberships held by identity — added in the overlay, or
    /// observed with an endpoint the snapshot lacked — split by resolution
    /// against THIS view: delta-sized `(user, group)` ordinal lanes for the
    /// pairs whose endpoints both exist now (an observed pair resolves once
    /// a version creates its missing endpoint), and the identities of the
    /// rest (dangling). Evidence-sized: the snapshot's unresolved table
    /// plus the overlay.
    pub(crate) fn added_rows(&self) -> AddedRows {
        let mut out = AddedRows::default();
        let observed = self
            .snap
            .m_unresolved
            .iter()
            .filter(|p| !self.ov.removed_unresolved.contains(p));
        for &(u, g) in self.ov.added.iter().chain(observed) {
            match (self.user_ordinal(&u), self.group_ordinal(&g)) {
                (Some(uo), Some(go)) => {
                    out.users.push(u32::from(uo.0));
                    out.groups.push(u32::from(go.0));
                }
                _ => out.unresolved.push((u, g)),
            }
        }
        out
    }

    /// Whether `node` still takes part in a live membership, on either side.
    ///
    /// Resolved base rows: one `Count` program over the membership lane of
    /// the node's side, gated by the observed rows (borrowed, no plane
    /// built), minus the removed rows touching the node (delta-sized). The
    /// relation is sorted by user, not by group, so the group side has no
    /// index: the work is a scan of one lane, the allocation one scratch
    /// tile. Unresolved and added rows are delta- or evidence-sized.
    pub(crate) fn has_membership(&self, node: &Guid128) -> bool {
        if self.ov.added.iter().any(|(u, g)| u == node || g == node) {
            return true;
        }
        if self
            .snap
            .m_unresolved
            .iter()
            .any(|p| (p.0 == *node || p.1 == *node) && !self.ov.removed_unresolved.contains(p))
        {
            return true;
        }
        let s = self.snap;
        let (lane, o) = match (s.users.ordinal(node), s.groups.ordinal(node)) {
            (Some(o), _) => (&s.m_user, o),
            (None, Some(o)) => (&s.m_group, o),
            (None, None) => return false,
        };
        let lanes = [LaneRef::U32(lane)];
        let masks: [&[u64]; 1] = [&s.m_all];
        let planes = Planes {
            n_rows: s.membership_rows(),
            masks: &masks,
            lanes: &lanes,
        };
        let observed = crate::exec::count(
            Filter::and([
                Filter::plane(Mask(0)),
                Filter::cmp(Col(0), Cmp::EqU32(u32::from(o))),
            ]),
            &planes,
            &Foreign::NONE,
        );
        let removed = self
            .ov
            .removed
            .iter()
            .filter(|&&r| lane[r as usize] == u32::from(o))
            .count();
        observed > removed
    }

    fn lane_ids(&self, v: Option<ValueId>) -> Result<(u32, u32), ApplyError> {
        self.dicts
            .lane_ids(v)
            .ok_or_else(|| ApplyError::Uninterned(v.expect("None always resolves")))
    }

    /// Apply one change to the overlay. Pure with respect to everything but
    /// `self.ov`; the snapshot is never touched.
    pub(crate) fn apply(&mut self, c: &Change) -> Result<(), ApplyError> {
        match c {
            Change::AddMembership { user, group } => {
                if self.is_member(user, group) {
                    return Err(ApplyError::NoOp(c.clone()));
                }
                if let Some(r) = self.snap.member_row(user, group) {
                    self.ov.removed.remove(&r);
                } else if !self.ov.removed_unresolved.remove(&(*user, *group)) {
                    self.ov.added.insert((*user, *group));
                }
            }
            Change::RemoveMembership { user, group } => {
                if !self.is_member(user, group) {
                    return Err(ApplyError::NoOp(c.clone()));
                }
                if self.ov.added.remove(&(*user, *group)) {
                    return Ok(());
                }
                match self.snap.member_row(user, group) {
                    Some(r) => {
                        self.ov.removed.insert(r);
                    }
                    None => {
                        self.ov.removed_unresolved.insert((*user, *group));
                    }
                }
            }
            Change::SetAttribute {
                node,
                attribute,
                from,
                to,
            } => {
                let slot = self.locate(node).ok_or(ApplyError::UnknownNode(*node))?;
                let actual = self.slot_attr(slot, *attribute);
                if actual != *from {
                    return Err(ApplyError::Stale {
                        node: *node,
                        attribute: *attribute,
                        expected: *from,
                        actual,
                    });
                }
                let ids = self.lane_ids(*to)?;
                let (kind, i) = slot;
                let base_len = self.pop(kind).0.len();
                let base_val = {
                    let p = self.pop(kind).0;
                    match attribute {
                        Attribute::Upn => p.upn_val.get(i).copied(),
                        Attribute::PrimarySmtp => p.smtp_val.get(i).copied(),
                    }
                };
                let o = self.pop_mut(kind);
                if i >= base_len {
                    let (val, key) = o.created.lanes(*attribute);
                    val[i - base_len] = ids.0;
                    key[i - base_len] = ids.1;
                    return Ok(());
                }
                let map = match attribute {
                    Attribute::Upn => &mut o.upn,
                    Attribute::PrimarySmtp => &mut o.smtp,
                };
                // Net effect only: setting a value back to the observed one
                // removes the override instead of recording a no-op change.
                if Some(ids.0) == base_val {
                    map.remove(&(i as u16));
                } else {
                    map.insert(i as u16, ids);
                }
            }
            Change::CreateNode { node, state } => {
                if state.kind == NodeKind::Group && !state.active {
                    return Err(ApplyError::InactiveGroup(*node));
                }
                // Identity uniqueness: one node per Guid128 in a version,
                // across both populations.
                if self.exists(node) {
                    return Err(ApplyError::NodeExists(*node));
                }
                let kind = state.kind;
                if let Some(o) = self.snap.population(kind).ordinal(node) {
                    // Deleted in this lineage: recreating the observed node
                    // unchanged is an undo; anything else reuses an identity.
                    self.pop_mut(kind).deleted.remove(&o);
                    if self.node_state(node).as_ref() != Some(state) {
                        self.pop_mut(kind).deleted.insert(o);
                        return Err(ApplyError::IdentityReused(*node));
                    }
                    return Ok(());
                }
                if self.snap.population(other(kind)).ordinal(node).is_some() {
                    return Err(ApplyError::IdentityReused(*node));
                }
                let max = match kind {
                    NodeKind::User => MAX_USERS,
                    NodeKind::Group => MAX_GROUPS,
                };
                if self.len_in(kind) >= max {
                    return Err(ApplyError::PopulationFull(kind));
                }
                let upn = self.lane_ids(state.upn)?;
                let smtp = self.lane_ids(state.primary_smtp)?;
                let o = self.pop_mut(kind);
                let at = o.created.find(node).unwrap_err();
                o.created.insert(at, *node, state, upn, smtp);
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
                let kind = actual.kind;
                let base = self.snap.population(kind).ordinal(node);
                let o = self.pop_mut(kind);
                match (o.created.find(node), base) {
                    // Create-then-delete nets out to nothing.
                    (Ok(i), _) => o.created.remove(i),
                    (Err(_), Some(b)) => {
                        o.upn.remove(&b);
                        o.smtp.remove(&b);
                        o.deleted.insert(b);
                    }
                    (Err(_), None) => return Err(ApplyError::UnknownNode(*node)),
                }
            }
        }
        Ok(())
    }
}

/// [`View::added_rows`]: resolved delta rows as lanes, unresolved as ids.
#[derive(Debug, Default)]
pub(crate) struct AddedRows {
    pub(crate) users: Vec<u32>,
    pub(crate) groups: Vec<u32>,
    pub(crate) unresolved: Vec<(Guid128, Guid128)>,
}

fn other(kind: NodeKind) -> NodeKind {
    match kind {
        NodeKind::User => NodeKind::Group,
        NodeKind::Group => NodeKind::User,
    }
}
