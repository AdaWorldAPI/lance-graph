//! One observed directory state as SoA lanes — never as objects.
//!
//! Users and groups are two independent populations, each at most
//! [`MAX_USERS`] / [`MAX_GROUPS`] = 65,536 nodes and each with its own dense
//! ordinal space ([`UserOrdinal`], [`GroupOrdinal`], both `u16`; every value
//! of a `u16` is a real ordinal, none is reserved).
//!
//! Per population ([`Population`]):
//!
//! | lane / plane              | type          | meaning                                  |
//! |---------------------------|---------------|------------------------------------------|
//! | `ids`                     | `[Guid128]`   | sorted; a node's **ordinal** is its index |
//! | `active`                  | bit plane     | enabled (users); every group             |
//! | `upn_val` / `smtp_val`    | `[u32]`       | [`ValueId`] in the store's value table    |
//! | `upn_key` / `smtp_key`    | `[u32]`       | [`KeyId`] (comparison form)               |
//! | `dn`                      | `[[u8; 16]]`  | [`Dn128`] codes, read in place (strided)  |
//! | `dn_depth`, `dn_present`  | `[i32]`, plane | hierarchy depth; whether a location is known |
//! | `rcp_rrt` / `rcp_display` | `[u32]`       | Exchange recipient type flags / display type (bit-cast) |
//! | `rcp_details`             | `[u64]`       | recipient type details                    |
//! | `rcp_target`              | `[u32]`       | [`ValueId`] of `targetAddress`            |
//! | `rcp_present`             | `[u8]`        | which of the four were observed           |
//! | `rcp_read`                | bit plane     | whether the recipient attributes were read at all |
//! | `owner`                   | bit plane     | owns its addresses: enabled, or a live mail recipient |
//!
//! The recipient lanes are raw; [`Recipient::from_attributes`] decodes them
//! on read, strictly. An address owner is a user that is enabled **or** a
//! live mail recipient (OGAR `Recipient::is_recipient`): a shared, room or
//! equipment mailbox is a disabled account and still owns its addresses.
//!
//! `proxyAddresses` are many per user, so they are a relation of their own
//! ([`ProxyRelation`]), not a lane.
//!
//! Membership is sparse — `UserOrdinal × GroupOrdinal` rows, never a dense
//! matrix — as two lanes `m_user`, `m_group` sorted by `(user, group)`. The
//! values are `u16` ordinals; the lanes are 32-bit because the substrate has
//! no 16-bit lane. A row whose endpoint does not resolve is kept as
//! identities in `m_unresolved` and never enters the lanes, so no lane value
//! stands for "missing". A group nested in a group has no user ordinal either,
//! so it is held there by identity too; validation tells it from a dangling
//! row by checking that both sides are live groups.
//!
//! **Ordinal ≠ identity.** An ordinal is valid only inside one snapshot and
//! never reaches provenance, diffs or plans; those carry [`Guid128`].
//! Ordinals come from sorting by `Guid128`, so they — and every id interned
//! while building — are independent of ingestion order.
//!
//! [`Observation`] is the ingestion boundary: it owns its strings once, and
//! [`Snapshot::build`] interns them into the store's [`Dicts`] (the cold
//! label/value store). After that, execution works on ids, ordinals and
//! masks; a string is resolved only to report or to actuate.

use crate::proxy::ProxyRelation;
use lance_graph_mask_risc::words_for;
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};
use ogar_dir_sim::{normalize, routing_parts, KeyId, Recipient, RecipientAttributes, ValueId};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub use ogar_dir_sim::NodeKind;

/// Most users one directory population may hold.
pub const MAX_USERS: usize = 65_536;
/// Most groups one directory population may hold.
pub const MAX_GROUPS: usize = 65_536;

/// "No value" in a value or key lane. Value and key ids are store-issued and
/// never reach this value (the store refuses to grow that far).
pub const NONE: u32 = u32::MAX;

/// Dense position of a user inside one snapshot (or one version).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UserOrdinal(pub u16);

/// Dense position of a group inside one snapshot (or one version).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupOrdinal(pub u16);

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
        let i = u32::try_from(self.strings.len())
            .ok()
            .filter(|&i| i != NONE)
            .expect("dictionary below u32::MAX entries");
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
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.strings.len()
    }
    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

/// Boundary counters, as in `lance-graph-report`'s boundary: every text
/// operation is counted, so a test can prove execution did none.
/// Relaxed atomics: diagnostics, not synchronization.
#[derive(Debug, Default)]
pub struct DictCounters {
    /// Text → id, minting if new (ingress).
    pub interns: AtomicU64,
    /// Text → id, never minting (resolving a query literal).
    pub lookups: AtomicU64,
    /// Id → text (egress).
    pub resolutions: AtomicU64,
}

impl DictCounters {
    /// `[interns, lookups, resolutions]`.
    pub fn snapshot(&self) -> [u64; 3] {
        [&self.interns, &self.lookups, &self.resolutions].map(|c| c.load(Ordering::Relaxed))
    }
}

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// The cold label/value store shared by every snapshot and version of one
/// [`VersionStore`](crate::VersionStore): the only place strings live.
/// Append-only, so a [`ValueId`] / [`KeyId`] keeps its meaning for the
/// store's lifetime — across observations, simulations, the desired
/// version, reconciliation and plans.
///
/// Identity, compared with the other boundaries in this workspace: a
/// [`ValueId`] **is** its exact text (a different text is a different
/// value — changing it is a `SetAttribute`, not a relabel), shared by every
/// attribute, valid for the store's lifetime. That is not
/// `lance-graph-report`'s CAM ordinal, which is per field and whose label
/// can be renamed under a fixed ordinal; and not a batch-local dictionary
/// code (`lance-graph-sap`), which cannot appear in a plan.
#[derive(Debug, Default)]
pub struct Dicts {
    /// Raw values exactly as observed or requested.
    values: Dict,
    /// Comparison forms ([`normalize`]).
    keys: Dict,
    /// `key_of[value] = key`, computed once at interning.
    key_of: Vec<u32>,
    /// `routing_alias[value]` = the key of the alias if the value's
    /// comparison form is a routing address (`{alias}@{tenant}.mail.onmicrosoft.com`,
    /// OGAR `ROUTING`), else `NONE`. Parsed once at interning, so
    /// validation compares ids and never reads text.
    routing_alias: Vec<u32>,
    /// Text operations performed.
    pub counters: DictCounters,
}

impl Dicts {
    /// Ingress: the id of a raw value, interning it (and its comparison
    /// key) if new.
    pub fn intern(&mut self, s: &str) -> ValueId {
        bump(&self.counters.interns);
        let v = self.values.intern(s);
        if v as usize == self.key_of.len() {
            let n = normalize(s);
            self.key_of.push(self.keys.intern(&n));
            let alias = routing_parts(&n).map_or(NONE, |(alias, _)| self.keys.intern(alias));
            self.routing_alias.push(alias);
        }
        ValueId(v)
    }
    /// The id of a raw value, if this store ever saw it. Never mints.
    pub fn lookup(&self, s: &str) -> Option<ValueId> {
        bump(&self.counters.lookups);
        self.values.get(s).map(ValueId)
    }
    /// The comparison key of a text (normalized here), if this store ever
    /// saw a value with that key. Never mints. This is how a query literal
    /// such as `smtp = 'Alice@X.de'` becomes a number, once, before lowering.
    pub fn key_lookup(&self, s: &str) -> Option<KeyId> {
        bump(&self.counters.lookups);
        self.keys.get(&normalize(s)).map(KeyId)
    }
    /// Egress: the raw value behind an id.
    pub fn value(&self, v: ValueId) -> Option<&str> {
        bump(&self.counters.resolutions);
        self.values.resolve(v.0)
    }
    /// The comparison key of a value.
    pub fn key_of(&self, v: ValueId) -> Option<KeyId> {
        self.key_of.get(v.0 as usize).map(|&k| KeyId(k))
    }
    /// The alias key of a routing address value; `None` if the value is not
    /// a routing address.
    pub(crate) fn routing_alias(&self, v: ValueId) -> Option<KeyId> {
        let k = *self.routing_alias.get(v.0 as usize)?;
        (k != NONE).then_some(KeyId(k))
    }
    /// Egress: the comparison form behind a key.
    pub fn key_label(&self, k: KeyId) -> Option<&str> {
        bump(&self.counters.resolutions);
        self.keys.resolve(k.0)
    }
    /// Number of distinct comparison keys (the universe of a key `GROUP BY`).
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
    /// `(value, key)` lane entries for an optional value; `None` if the id
    /// was never issued by this store.
    pub(crate) fn lane_ids(&self, v: Option<ValueId>) -> Option<(u32, u32)> {
        match v {
            None => Some((NONE, NONE)),
            Some(v) => Some((v.0, *self.key_of.get(v.0 as usize)?)),
        }
    }
}

/// One node as read from a source (ingestion staging; owns its strings).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedNode {
    /// Kind.
    pub kind: NodeKind,
    /// Enabled; `None` = unknown (no flag was reported). Users only: a
    /// group's flag is not stored and reads back as `Some(true)`.
    pub active: Option<bool>,
    /// Raw UPN.
    pub upn: Option<String>,
    /// Raw primary SMTP.
    pub primary_smtp: Option<String>,
    /// Every other `proxyAddresses` value, raw with its prefix
    /// (`smtp:a@x.de`, `X500:/o=…`). The primary SMTP is `primary_smtp` and
    /// is not repeated here. Becomes rows of the snapshot's
    /// [`ProxyRelation`](crate::proxy::ProxyRelation).
    pub proxies: Vec<String>,
    /// Hierarchy location in the observation's scope, if known.
    pub dn: Option<Dn128>,
    /// The Exchange recipient attributes, raw; `None` = not read (the
    /// source never reported them), distinct from read and all absent
    /// (not mail-enabled).
    pub recipient: Option<ObservedRecipient>,
    /// Raw `mail`: a property on the user's business card, like the
    /// telephone number, shown in the address book and used inside
    /// messages. It follows the user, not the mailbox or the recipient. It
    /// is not the recipient's identity (`PrimarySmtpAddress` is), not an
    /// address anything is received at and not provisioned; kept as
    /// written.
    pub mail: Option<String>,
    /// Raw `mailNickname` (the Exchange Online `Alias`).
    pub alias: Option<String>,
    /// `msExchMailboxGuid`, the mailbox's own GUID (Exchange Online keeps it
    /// as the mailbox's `ExchangeGuid`); `None` = no mailbox, or not read.
    pub exchange_guid: Option<Guid128>,
}

/// The Exchange recipient attributes as a source reported them (strings
/// owned; [`Snapshot::build`] interns the target address).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservedRecipient {
    /// `msExchRemoteRecipientType`.
    pub remote_recipient_type: Option<u32>,
    /// `msExchRecipientDisplayType` (signed).
    pub display_type: Option<i32>,
    /// `msExchRecipientTypeDetails`.
    pub type_details: Option<u64>,
    /// `targetAddress`, the address without its `SMTP:` prefix (stripped
    /// at the ingest boundary, as OGAR's `RecipientAttributes` expects).
    pub target_address: Option<String>,
}

/// Whether a user holds its UPN: enabled, or a live mail recipient (a
/// shared mailbox is a disabled account and keeps its UPN). The UPN is not
/// an Exchange address, so the recipient type alone does not provision it.
/// This is the *claim* (who reserves a value, so no one else may take it),
/// not whether mail is delivered: see [`is_mail_recipient`].
pub(crate) fn is_owner(active: Option<bool>, recipient: Option<Recipient>) -> bool {
    active == Some(true) || recipient.is_some_and(|r| r.is_recipient())
}

/// Whether mail addressed to a user is delivered to it: OGAR's recipient
/// lifecycle decides (`Recipient::is_recipient`) whenever the source read
/// the recipient attributes. A disabled shared, room or equipment mailbox
/// is a recipient; an enabled account that is not mail-enabled is not,
/// whatever stale `mail` or SMTP values it still carries.
///
/// When the source never read the recipient attributes (`None`), only an
/// enabled account counts, the assumption every snapshot without Exchange
/// data was built on; an unknown flag with unknown attributes is not a
/// recipient.
///
/// The same rule provisions a user's mail addresses (its primary SMTP, its
/// secondary `smtp:` proxies and a remote mailbox's routing address): an
/// object holds them exactly when it is a mail recipient, whether or not
/// the account is enabled. Leftover `proxyAddresses` on an account that is
/// not mail-enabled reserve nothing in Exchange.
pub fn is_mail_recipient(active: Option<bool>, recipient: Option<Recipient>) -> bool {
    match recipient {
        Some(r) => r.is_recipient(),
        None => active == Some(true),
    }
}

const RCP_RRT: u8 = 1;
const RCP_DISPLAY: u8 = 2;
const RCP_DETAILS: u8 = 4;
const RCP_TARGET: u8 = 8;

impl ObservedNode {
    /// Active user with UPN and primary SMTP.
    pub fn user(upn: &str, smtp: &str) -> Self {
        Self {
            kind: NodeKind::User,
            active: Some(true),
            upn: Some(upn.into()),
            primary_smtp: Some(smtp.into()),
            proxies: Vec::new(),
            dn: None,
            recipient: None,
            mail: None,
            alias: None,
            exchange_guid: None,
        }
    }
    /// Group.
    pub fn group() -> Self {
        Self {
            kind: NodeKind::Group,
            active: Some(true),
            upn: None,
            primary_smtp: None,
            proxies: Vec::new(),
            dn: None,
            recipient: None,
            mail: None,
            alias: None,
            exchange_guid: None,
        }
    }
}

/// What a source reported. Order of either list is irrelevant.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// The directory (domain / tenant) every [`Dn128`] here belongs to.
    pub scope: DirectoryScope,
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
    /// More users than [`MAX_USERS`].
    TooManyUsers(usize),
    /// More groups than [`MAX_GROUPS`].
    TooManyGroups(usize),
    /// More membership rows than a `u32` row id can address.
    TooManyMemberships,
    /// More proxy-address rows than a `u32` row id can address.
    TooManyProxies,
}

pub(crate) fn bit(words: &[u64], i: usize) -> bool {
    words.get(i / 64).is_some_and(|w| w >> (i % 64) & 1 == 1)
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

/// One node population (all users, or all groups) of a snapshot.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Population {
    pub(crate) ids: Vec<Guid128>,
    pub(crate) all: Vec<u64>,
    /// Rows known enabled (`active == Some(true)`), and every group. An
    /// unknown flag is not in it.
    pub(crate) active: Vec<u64>,
    /// Rows whose flag is known (the validity plane of `active`), and
    /// every group.
    pub(crate) active_known: Vec<u64>,
    pub(crate) upn_val: Vec<u32>,
    pub(crate) upn_key: Vec<u32>,
    pub(crate) smtp_val: Vec<u32>,
    pub(crate) smtp_key: Vec<u32>,
    pub(crate) dn: Vec<[u8; 16]>,
    pub(crate) dn_depth: Vec<i32>,
    pub(crate) dn_present: Vec<u64>,
    pub(crate) rcp_rrt: Vec<u32>,
    pub(crate) rcp_display: Vec<u32>,
    pub(crate) rcp_details: Vec<u64>,
    pub(crate) rcp_target: Vec<u32>,
    pub(crate) rcp_present: Vec<u8>,
    pub(crate) rcp_read: Vec<u64>,
    /// `mail` values as written (`NONE` = absent): a label, never compared.
    pub(crate) mail_val: Vec<u32>,
    /// `mailNickname` comparison keys (`NONE` = absent).
    pub(crate) alias_key: Vec<u32>,
    /// `msExchMailboxGuid` ([`Guid128::NIL`] = absent).
    pub(crate) exchange_guid: Vec<Guid128>,
    /// Holds its UPN ([`is_owner`] of the observed flag and recipient).
    pub(crate) owner: Vec<u64>,
    /// Its mail addresses are provisioned ([`is_mail_recipient`]).
    pub(crate) mail_owner: Vec<u64>,
}

impl Population {
    fn build(nodes: &[&(Guid128, ObservedNode)], d: &mut Dicts) -> Self {
        let n = nodes.len();
        let mut p = Self {
            ids: Vec::with_capacity(n),
            all: ones(n),
            active: vec![0; words_for(n)],
            active_known: vec![0; words_for(n)],
            upn_val: Vec::with_capacity(n),
            upn_key: Vec::with_capacity(n),
            smtp_val: Vec::with_capacity(n),
            smtp_key: Vec::with_capacity(n),
            dn: Vec::with_capacity(n),
            dn_depth: Vec::with_capacity(n),
            dn_present: vec![0; words_for(n)],
            rcp_rrt: Vec::with_capacity(n),
            rcp_display: Vec::with_capacity(n),
            rcp_details: Vec::with_capacity(n),
            rcp_target: Vec::with_capacity(n),
            rcp_present: Vec::with_capacity(n),
            rcp_read: vec![0; words_for(n)],
            mail_val: Vec::with_capacity(n),
            alias_key: Vec::with_capacity(n),
            exchange_guid: Vec::with_capacity(n),
            owner: vec![0; words_for(n)],
            mail_owner: vec![0; words_for(n)],
        };
        for (i, (id, node)) in nodes.iter().enumerate() {
            p.ids.push(*id);
            let active = if node.kind == NodeKind::Group {
                Some(true)
            } else {
                node.active
            };
            if active.is_some() {
                set_bit(&mut p.active_known, i);
            }
            if active == Some(true) {
                set_bit(&mut p.active, i);
            }
            let ids = |s: &Option<String>, d: &mut Dicts| {
                let v = s.as_deref().map(|s| d.intern(s));
                d.lane_ids(v).expect("just interned")
            };
            let (uv, uk) = ids(&node.upn, d);
            let (sv, sk) = ids(&node.primary_smtp, d);
            p.upn_val.push(uv);
            p.upn_key.push(uk);
            p.smtp_val.push(sv);
            p.smtp_key.push(sk);
            p.mail_val.push(ids(&node.mail, d).0);
            p.alias_key.push(ids(&node.alias, d).1);
            p.exchange_guid
                .push(node.exchange_guid.unwrap_or(Guid128::NIL));
            let dn = node.dn.unwrap_or(Dn128::ROOT);
            p.dn.push(dn.bytes());
            p.dn_depth.push(dn.depth() as i32);
            if node.dn.is_some() {
                set_bit(&mut p.dn_present, i);
            }
            let r = node.recipient.clone().unwrap_or_default();
            let target = r.target_address.as_deref().map(|s| d.intern(s));
            let mut present = 0;
            for (bit, on) in [
                (RCP_RRT, r.remote_recipient_type.is_some()),
                (RCP_DISPLAY, r.display_type.is_some()),
                (RCP_DETAILS, r.type_details.is_some()),
                (RCP_TARGET, target.is_some()),
            ] {
                if on {
                    present |= bit;
                }
            }
            p.rcp_rrt.push(r.remote_recipient_type.unwrap_or(0));
            p.rcp_display.push(r.display_type.unwrap_or(0) as u32);
            p.rcp_details.push(r.type_details.unwrap_or(0));
            p.rcp_target.push(target.map_or(NONE, |v| v.0));
            p.rcp_present.push(present);
            if node.recipient.is_some() {
                set_bit(&mut p.rcp_read, i);
            }
            if is_owner(active, p.recipient_of(i)) {
                set_bit(&mut p.owner, i);
            }
            if is_mail_recipient(active, p.recipient_of(i)) {
                set_bit(&mut p.mail_owner, i);
            }
        }
        p
    }

    /// The observed recipient of base node `o`, decoded strictly; `None`
    /// if it was not read.
    pub(crate) fn recipient_of(&self, o: usize) -> Option<Recipient> {
        if !bit(&self.rcp_read, o) {
            return None;
        }
        let has = |b: u8| self.rcp_present[o] & b != 0;
        Some(Recipient::from_attributes(RecipientAttributes {
            remote_recipient_type: has(RCP_RRT).then_some(self.rcp_rrt[o]),
            display_type: has(RCP_DISPLAY).then_some(self.rcp_display[o] as i32),
            type_details: has(RCP_DETAILS).then_some(self.rcp_details[o]),
            target_address: has(RCP_TARGET).then_some(ValueId(self.rcp_target[o])),
        }))
    }

    /// Node count.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// Ordinal of an identity (binary search over the sorted id lane).
    pub(crate) fn ordinal(&self, g: &Guid128) -> Option<u16> {
        self.ids.binary_search(g).ok().map(|i| i as u16)
    }
    pub(crate) fn dn_of(&self, o: usize) -> Option<Dn128> {
        if !bit(&self.dn_present, o) {
            return None;
        }
        let depth = self.dn_depth[o] as usize;
        Dn128::new(&self.dn[o][..depth]).ok()
    }
}

/// One observed state. Immutable once built; versions share it by `Arc`.
#[derive(Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub(crate) scope: DirectoryScope,
    pub(crate) users: Population,
    pub(crate) groups: Population,
    /// `proxyAddresses` of the users, as rows.
    pub(crate) proxies: ProxyRelation,
    pub(crate) m_user: Vec<u32>,
    pub(crate) m_group: Vec<u32>,
    pub(crate) m_all: Vec<u64>,
    /// Observed memberships with an endpoint that is not a node of the
    /// expected kind, by identity, sorted. Never in the lanes.
    pub(crate) m_unresolved: Vec<(Guid128, Guid128)>,
}

impl Snapshot {
    /// Build from an observation, interning strings into `d`.
    pub fn build(mut obs: Observation, d: &mut Dicts) -> Result<Self, BuildError> {
        obs.nodes.sort_by_key(|n| n.0);
        if let Some(w) = obs.nodes.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(BuildError::DuplicateNode(w[0].0));
        }
        let (users, groups): (Vec<_>, Vec<_>) = obs
            .nodes
            .iter()
            .partition(|(_, n)| n.kind == NodeKind::User);
        if users.len() > MAX_USERS {
            return Err(BuildError::TooManyUsers(users.len()));
        }
        if groups.len() > MAX_GROUPS {
            return Err(BuildError::TooManyGroups(groups.len()));
        }
        if obs.members.len() >= u32::MAX as usize {
            return Err(BuildError::TooManyMemberships);
        }
        let user_nodes = users;
        let users = Population::build(&user_nodes, d);
        let proxies = ProxyRelation::build(&user_nodes, &users, d);
        if proxies.len() >= u32::MAX as usize {
            return Err(BuildError::TooManyProxies);
        }
        let groups = Population::build(&groups, d);
        let mut rows: Vec<(u16, u16)> = Vec::with_capacity(obs.members.len());
        let mut unresolved = Vec::new();
        for (u, g) in &obs.members {
            match (users.ordinal(u), groups.ordinal(g)) {
                (Some(uo), Some(go)) => rows.push((uo, go)),
                _ => unresolved.push((*u, *g)),
            }
        }
        rows.sort_unstable();
        rows.dedup();
        unresolved.sort_unstable();
        unresolved.dedup();
        let (m_user, m_group): (Vec<u32>, Vec<u32>) = rows
            .into_iter()
            .map(|(u, g)| (u32::from(u), u32::from(g)))
            .unzip();
        let m_all = ones(m_user.len());
        Ok(Self {
            scope: obs.scope,
            users,
            groups,
            proxies,
            m_user,
            m_group,
            m_all,
            m_unresolved: unresolved,
        })
    }

    /// The directory scope every location in this snapshot belongs to.
    pub fn scope(&self) -> DirectoryScope {
        self.scope
    }
    /// The user population.
    pub fn users(&self) -> &Population {
        &self.users
    }
    /// The group population.
    pub fn groups(&self) -> &Population {
        &self.groups
    }
    /// The users' `proxyAddresses`, as a relation.
    pub fn proxies(&self) -> &ProxyRelation {
        &self.proxies
    }
    /// Resolved membership row count (unresolved rows excluded).
    pub fn membership_rows(&self) -> usize {
        self.m_user.len()
    }
    /// Ordinal of a user.
    pub fn user_ordinal(&self, g: &Guid128) -> Option<UserOrdinal> {
        self.users.ordinal(g).map(UserOrdinal)
    }
    /// Ordinal of a group.
    pub fn group_ordinal(&self, g: &Guid128) -> Option<GroupOrdinal> {
        self.groups.ordinal(g).map(GroupOrdinal)
    }
    /// Identity of a user ordinal.
    pub fn user_guid(&self, o: UserOrdinal) -> Option<Guid128> {
        self.users.ids.get(usize::from(o.0)).copied()
    }
    /// Identity of a group ordinal.
    pub fn group_guid(&self, o: GroupOrdinal) -> Option<Guid128> {
        self.groups.ids.get(usize::from(o.0)).copied()
    }
    /// The population of a kind.
    pub(crate) fn population(&self, kind: NodeKind) -> &Population {
        match kind {
            NodeKind::User => &self.users,
            NodeKind::Group => &self.groups,
        }
    }
    /// Identities of resolved membership row `r`.
    pub(crate) fn member_guids(&self, r: u32) -> (Guid128, Guid128) {
        (
            self.users.ids[self.m_user[r as usize] as usize],
            self.groups.ids[self.m_group[r as usize] as usize],
        )
    }
    /// Row of resolved membership `(user, group)`, if observed. `O(log m)`:
    /// rows are sorted by `(user, group)`.
    pub(crate) fn member_row(&self, user: &Guid128, group: &Guid128) -> Option<u32> {
        let (uo, go) = (self.users.ordinal(user)?, self.groups.ordinal(group)?);
        let key = (u32::from(uo), u32::from(go));
        let (mut lo, mut hi) = (0usize, self.m_user.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match (self.m_user[mid], self.m_group[mid]).cmp(&key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u32),
            }
        }
        None
    }
    /// Whether `(user, group)` is an observed but unresolved membership.
    pub(crate) fn is_unresolved_member(&self, user: &Guid128, group: &Guid128) -> bool {
        self.m_unresolved.binary_search(&(*user, *group)).is_ok()
    }
}
