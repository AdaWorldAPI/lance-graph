//! One observed directory state as SoA lanes — never as objects.
//!
//! | lane / plane            | type        | meaning                                     |
//! |-------------------------|-------------|---------------------------------------------|
//! | `ids`                   | `[Guid128]` | sorted; a node's **ordinal** is its index   |
//! | `user`, `group`         | bit planes  | node kind                                   |
//! | `active_user`           | bit plane   | user ∧ enabled — the recipient population   |
//! | `upn_val` / `smtp_val`  | `[u32]`     | raw value id in [`Dicts::values`]           |
//! | `upn_key` / `smtp_key`  | `[u32]`     | normalized key id in [`Dicts::keys`]        |
//! | `ou`                    | `[OuHhtl]`  | exact OU location (node state, compare-and-set) |
//! | `ou_hi`, `ou_present`   | `[u64]`, plane | OU-HHTL levels 0..4 packed (subtree = prefix match) |
//! | `m_user`, `m_group`     | `[u32]`     | membership relation, sorted by (user, group) |
//!
//! **Ordinal ≠ identity.** An ordinal is valid only inside one snapshot and
//! is never stored in provenance, diffs or plans; those carry [`Guid128`].
//! Ordinals come from sorting by `Guid128`, so they — and every dictionary
//! id assigned while building — are independent of ingestion order.
//!
//! [`Observation`] is the ingestion boundary: it owns its strings once, and
//! [`Snapshot::build`] interns them into the store's dictionaries. After
//! that, execution works on ids and masks; a string is resolved only to
//! report evidence or to compare a compare-and-set value.

use lance_graph_mask_risc::words_for;
use ogar_dir_core::{Guid128, OuHhtl};
use ogar_dir_sim::normalize;
use std::collections::BTreeMap;

/// "No value" / "unresolved endpoint" sentinel. Out of range for every
/// lane, so mask-risc's zero-fallback treats it as matching nothing.
pub const NONE: u32 = u32::MAX;

pub use ogar_dir_sim::NodeKind;

/// Append-only string dictionary. Ids are assignment order; nothing is
/// iterated in hash order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Dict {
    strings: Vec<String>,
    index: BTreeMap<String, u32>,
}

impl Dict {
    /// Id of `s`, interning it if new.
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = u32::try_from(self.strings.len()).expect("dictionary fits u32");
        self.strings.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i
    }
    /// Id of `s` if interned.
    pub fn get(&self, s: &str) -> Option<u32> {
        self.index.get(s).copied()
    }
    /// The string behind an id.
    pub fn resolve(&self, id: u32) -> Option<&str> {
        self.strings.get(id as usize).map(String::as_str)
    }
    /// Number of entries (the group universe of a key `GROUP BY`).
    pub fn len(&self) -> usize {
        self.strings.len()
    }
    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

/// Value storage shared by every snapshot and version of one store.
#[derive(Debug, Default)]
pub struct Dicts {
    /// Raw values exactly as observed.
    pub values: Dict,
    /// Normalized comparison keys.
    pub keys: Dict,
}

impl Dicts {
    /// `(value id, key id)` for an optional raw value.
    pub fn intern_attr(&mut self, s: Option<&str>) -> (u32, u32) {
        match s {
            None => (NONE, NONE),
            Some(s) => (self.values.intern(s), self.keys.intern(&normalize(s))),
        }
    }
    /// `(value id, key id)` without interning; `None` if not yet interned.
    pub fn lookup_attr(&self, s: Option<&str>) -> Option<(u32, u32)> {
        match s {
            None => Some((NONE, NONE)),
            Some(s) => Some((self.values.get(s)?, self.keys.get(&normalize(s))?)),
        }
    }
}

/// One node as read from a source (ingestion staging).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedNode {
    /// Kind.
    pub kind: NodeKind,
    /// Enabled (users only; a group's flag is not stored and reads back
    /// as `true`).
    pub active: bool,
    /// Raw UPN.
    pub upn: Option<String>,
    /// Raw primary SMTP.
    pub primary_smtp: Option<String>,
    /// OU location, if known.
    pub ou: Option<OuHhtl>,
}

impl ObservedNode {
    /// Active user with UPN and primary SMTP.
    pub fn user(upn: &str, smtp: &str) -> Self {
        Self {
            kind: NodeKind::User,
            active: true,
            upn: Some(upn.into()),
            primary_smtp: Some(smtp.into()),
            ou: None,
        }
    }
    /// Group.
    pub fn group() -> Self {
        Self {
            kind: NodeKind::Group,
            active: true,
            upn: None,
            primary_smtp: None,
            ou: None,
        }
    }
}

impl From<ObservedNode> for ogar_dir_sim::NodeState {
    fn from(n: ObservedNode) -> Self {
        Self {
            kind: n.kind,
            active: n.active,
            upn: n.upn,
            primary_smtp: n.primary_smtp,
            ou: n.ou,
        }
    }
}

/// What a source reported. Order of either list is irrelevant.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// Nodes.
    pub nodes: Vec<(Guid128, ObservedNode)>,
    /// `(user, group)` memberships; endpoints need not exist (dangling
    /// observations are representable so validation can report them).
    pub members: Vec<(Guid128, Guid128)>,
}

/// Why an observation could not become a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The same Guid128 was observed twice (which one wins would depend on
    /// ingestion order, so it is refused rather than guessed).
    DuplicateNode(Guid128),
    /// More than `u32::MAX - 1` nodes or memberships.
    TooLarge,
}

pub(crate) fn bit(words: &[u64], i: u32) -> bool {
    i != NONE
        && words
            .get(i as usize / 64)
            .is_some_and(|w| w >> (i % 64) & 1 == 1)
}
pub(crate) fn set_bit(words: &mut [u64], i: usize) {
    words[i / 64] |= 1 << (i % 64);
}
pub(crate) fn clear_bit(words: &mut [u64], i: usize) {
    words[i / 64] &= !(1 << (i % 64));
}
pub(crate) fn ones(n: usize) -> Vec<u64> {
    let mut w = vec![u64::MAX; words_for(n)];
    if !n.is_multiple_of(64) {
        if let Some(last) = w.last_mut() {
            *last = (1u64 << (n % 64)) - 1;
        }
    }
    w
}

/// OU-HHTL levels 0..4 packed into one `u64` (level 0 in the top 16 bits),
/// so "subtree of a prefix of depth ≤ 4" is one ternary match.
pub fn pack_ou(h: &OuHhtl) -> u64 {
    (u64::from(h.0[0]) << 48)
        | (u64::from(h.0[1]) << 32)
        | (u64::from(h.0[2]) << 16)
        | u64::from(h.0[3])
}

/// One observed state. Immutable once built; versions share it by `Arc`.
#[derive(Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub(crate) ids: Vec<Guid128>,
    pub(crate) user: Vec<u64>,
    pub(crate) group: Vec<u64>,
    pub(crate) active_user: Vec<u64>,
    pub(crate) upn_val: Vec<u32>,
    pub(crate) upn_key: Vec<u32>,
    pub(crate) smtp_val: Vec<u32>,
    pub(crate) smtp_key: Vec<u32>,
    pub(crate) ou: Vec<OuHhtl>,
    pub(crate) ou_hi: Vec<u64>,
    pub(crate) ou_present: Vec<u64>,
    pub(crate) m_user: Vec<u32>,
    pub(crate) m_group: Vec<u32>,
    pub(crate) m_all: Vec<u64>,
    /// Membership rows with an endpoint that resolved to no node: row →
    /// the observed identities (rare; kept so evidence never loses identity).
    pub(crate) m_unresolved: BTreeMap<u32, (Guid128, Guid128)>,
}

impl Snapshot {
    /// Build from an observation, interning strings into `d`.
    pub fn build(mut obs: Observation, d: &mut Dicts) -> Result<Self, BuildError> {
        obs.nodes.sort_by_key(|n| n.0);
        if let Some(w) = obs.nodes.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(BuildError::DuplicateNode(w[0].0));
        }
        let n = obs.nodes.len();
        if n >= NONE as usize || obs.members.len() >= NONE as usize {
            return Err(BuildError::TooLarge);
        }
        let mut s = Self {
            ids: Vec::with_capacity(n),
            user: vec![0; words_for(n)],
            group: vec![0; words_for(n)],
            active_user: vec![0; words_for(n)],
            upn_val: Vec::with_capacity(n),
            upn_key: Vec::with_capacity(n),
            smtp_val: Vec::with_capacity(n),
            smtp_key: Vec::with_capacity(n),
            ou: Vec::with_capacity(n),
            ou_hi: Vec::with_capacity(n),
            ou_present: vec![0; words_for(n)],
            m_user: Vec::new(),
            m_group: Vec::new(),
            m_all: Vec::new(),
            m_unresolved: BTreeMap::new(),
        };
        for (i, (id, node)) in obs.nodes.iter().enumerate() {
            s.ids.push(*id);
            match node.kind {
                NodeKind::User => {
                    set_bit(&mut s.user, i);
                    if node.active {
                        set_bit(&mut s.active_user, i);
                    }
                }
                NodeKind::Group => set_bit(&mut s.group, i),
            }
            let (uv, uk) = d.intern_attr(node.upn.as_deref());
            let (sv, sk) = d.intern_attr(node.primary_smtp.as_deref());
            s.upn_val.push(uv);
            s.upn_key.push(uk);
            s.smtp_val.push(sv);
            s.smtp_key.push(sk);
            s.ou.push(node.ou.unwrap_or(OuHhtl::ROOT));
            s.ou_hi.push(node.ou.as_ref().map_or(0, pack_ou));
            if node.ou.is_some() {
                set_bit(&mut s.ou_present, i);
            }
        }
        let mut rows: Vec<(u32, u32, Guid128, Guid128)> = obs
            .members
            .iter()
            .map(|(u, g)| {
                (
                    s.ordinal(u).unwrap_or(NONE),
                    s.ordinal(g).unwrap_or(NONE),
                    *u,
                    *g,
                )
            })
            .collect();
        rows.sort();
        rows.dedup();
        for (r, (uo, go, ug, gg)) in rows.into_iter().enumerate() {
            if uo == NONE || go == NONE {
                s.m_unresolved.insert(r as u32, (ug, gg));
            }
            s.m_user.push(uo);
            s.m_group.push(go);
        }
        s.m_all = ones(s.m_user.len());
        Ok(s)
    }

    /// Node count.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    /// True if no nodes.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// Membership row count.
    pub fn membership_rows(&self) -> usize {
        self.m_user.len()
    }
    /// Dense execution ordinal of an identity (binary search over the sorted id lane).
    pub fn ordinal(&self, g: &Guid128) -> Option<u32> {
        self.ids.binary_search(g).ok().map(|i| i as u32)
    }
    /// Identity of an ordinal.
    pub fn guid(&self, o: u32) -> Option<Guid128> {
        self.ids.get(o as usize).copied()
    }
    /// Identities of membership row `r`.
    pub(crate) fn member_guids(&self, r: u32) -> (Guid128, Guid128) {
        if let Some(p) = self.m_unresolved.get(&r) {
            return *p;
        }
        (
            self.ids[self.m_user[r as usize] as usize],
            self.ids[self.m_group[r as usize] as usize],
        )
    }
    /// Row of membership `(user, group)` in the base relation, if observed.
    pub(crate) fn member_row(&self, user: &Guid128, group: &Guid128) -> Option<u32> {
        match (self.ordinal(user), self.ordinal(group)) {
            (Some(uo), Some(go)) => {
                // Rows are sorted by the full (user, group, …) tuple, so the
                // (user, group) prefix is monotonic over EVERY row — including
                // half-resolved ones (`NONE` sorts after every ordinal).
                let (mut lo, mut hi) = (0usize, self.m_user.len());
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    match (self.m_user[mid], self.m_group[mid]).cmp(&(uo, go)) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => return Some(mid as u32),
                    }
                }
                None
            }
            _ => self
                .m_unresolved
                .iter()
                .find(|(_, p)| **p == (*user, *group))
                .map(|(r, _)| *r),
        }
    }
}
