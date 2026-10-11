//! A history as an append-only log of verified batches (W-3, decision 3(b)).
//!
//! Persisting a [`StoredImage`] writes only what is new since the last save,
//! as one **batch** of records; a storage crate puts one batch in one commit.
//! The logical [`VersionId`]s inside a batch are the history's own clock and
//! never the storage version: one batch may carry many logical versions, or
//! none (a tag moved, a verdict recorded).
//!
//! Every batch starts with a header record giving its number, the logical
//! versions it holds, its record count and a checksum over its records.
//! [`replay`] reads batches in number order and refuses the whole log at the
//! first batch that fails a check: a gap in the numbering, a missing or
//! misplaced header, a record count or checksum that does not match, or a
//! version that does not follow the previous batch. Nothing is repaired and
//! no prefix is returned by [`replay`]; [`replay_prefix`] is the separate,
//! explicit call that reads up to the last good batch and names the damage.
//!
//! Pure: no I/O. A storage crate turns a [`Record`] into a row and back.
//!
//! The encoding is little-endian with fixed widths: `Option` is a `0`/`1`
//! tag, a sequence a `u32` count, a string a `u32` byte length and UTF-8.
//! Enum tags are fixed numbers, never reused. Every struct is encoded by
//! exhaustive destructuring, so a new field fails to compile here instead
//! of being dropped from the log.

use crate::image::{StoredImage, StoredOrigin, StoredRule, StoredVersion};
use crate::snapshot::{Observation, ObservedNode, ObservedRecipient};
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};
use ogar_dir_sim::{
    AddressRole, ArchiveState, Attribute, Change, Endpoint, EvidenceRef, KeyId, MailboxState,
    NodeKind, NodeState, Recipient, RecipientAttributes, RemoteKind, RemoteMailbox, ValueId,
    VersionId, Violation,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------- bytes

/// Why a record body did not decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeError {
    /// Byte offset in the body.
    pub at: usize,
    /// What was expected there.
    pub what: &'static str,
}

#[derive(Default)]
struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }
    fn len(&mut self, n: usize) {
        self.u32(u32::try_from(n).expect("a record part holds fewer than 2^32 items"));
    }
    fn str(&mut self, s: &str) {
        self.len(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn guid(&mut self, g: Guid128) {
        self.0.extend_from_slice(&g.0);
    }
    fn opt<T>(&mut self, v: Option<T>, f: impl FnOnce(&mut Self, T)) {
        match v {
            None => self.u8(0),
            Some(x) => {
                self.u8(1);
                f(self, x);
            }
        }
    }
    fn seq<T>(&mut self, items: &[T], mut f: impl FnMut(&mut Self, &T)) {
        self.len(items.len());
        for x in items {
            f(self, x);
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> R<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, at: 0 }
    }
    fn err<T>(&self, what: &'static str) -> Result<T, DecodeError> {
        Err(DecodeError { at: self.at, what })
    }
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], DecodeError> {
        match self.b.get(self.at..self.at.saturating_add(n)) {
            Some(s) if s.len() == n => {
                self.at += n;
                Ok(s)
            }
            _ => self.err(what),
        }
    }
    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1, "u8")?[0])
    }
    fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(
            self.take(2, "u16")?.try_into().expect("2"),
        ))
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(
            self.take(4, "u32")?.try_into().expect("4"),
        ))
    }
    fn i32(&mut self) -> Result<i32, DecodeError> {
        Ok(i32::from_le_bytes(
            self.take(4, "i32")?.try_into().expect("4"),
        ))
    }
    fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(
            self.take(8, "u64")?.try_into().expect("8"),
        ))
    }
    fn i64(&mut self) -> Result<i64, DecodeError> {
        Ok(i64::from_le_bytes(
            self.take(8, "i64")?.try_into().expect("8"),
        ))
    }
    fn bool(&mut self) -> Result<bool, DecodeError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => self.err("bool"),
        }
    }
    fn len(&mut self) -> Result<usize, DecodeError> {
        let n = self.u32()? as usize;
        // A count can never exceed the bytes left: every item takes one.
        if n > self.b.len() - self.at {
            return self.err("count");
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String, DecodeError> {
        let n = self.len()?;
        let at = self.at;
        let s = self.take(n, "string")?;
        String::from_utf8(s.to_vec()).map_err(|_| DecodeError { at, what: "utf-8" })
    }
    fn guid(&mut self) -> Result<Guid128, DecodeError> {
        Ok(Guid128(self.take(16, "guid")?.try_into().expect("16")))
    }
    fn opt<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Option<T>, DecodeError> {
        match self.u8()? {
            0 => Ok(None),
            1 => f(self).map(Some),
            _ => self.err("option tag"),
        }
    }
    fn seq<T>(
        &mut self,
        mut f: impl FnMut(&mut Self) -> Result<T, DecodeError>,
    ) -> Result<Vec<T>, DecodeError> {
        let n = self.len()?;
        (0..n).map(|_| f(self)).collect()
    }
    fn end(&self) -> Result<(), DecodeError> {
        if self.at == self.b.len() {
            Ok(())
        } else {
            self.err("end of record")
        }
    }
}

// ---------------------------------------------------------------- types

fn w_node_kind(w: &mut W, k: NodeKind) {
    w.u8(match k {
        NodeKind::User => 0,
        NodeKind::Group => 1,
    });
}
fn r_node_kind(r: &mut R<'_>) -> Result<NodeKind, DecodeError> {
    match r.u8()? {
        0 => Ok(NodeKind::User),
        1 => Ok(NodeKind::Group),
        _ => r.err("node kind"),
    }
}

fn w_dn(w: &mut W, d: Dn128) {
    w.seq(d.codes(), |w, c| w.u8(*c));
}
fn r_dn(r: &mut R<'_>) -> Result<Dn128, DecodeError> {
    let at = r.at;
    let codes = r.seq(R::u8)?;
    Dn128::new(&codes).map_err(|_| DecodeError { at, what: "dn" })
}

fn w_archive(w: &mut W, a: ArchiveState) {
    w.u8(match a {
        ArchiveState::None => 0,
        ArchiveState::Provisioned => 1,
        ArchiveState::Deprovisioned => 2,
    });
}
fn r_archive(r: &mut R<'_>) -> Result<ArchiveState, DecodeError> {
    match r.u8()? {
        0 => Ok(ArchiveState::None),
        1 => Ok(ArchiveState::Provisioned),
        2 => Ok(ArchiveState::Deprovisioned),
        _ => r.err("archive state"),
    }
}

fn w_recipient(w: &mut W, rc: Recipient) {
    match rc {
        Recipient::NotMailEnabled => w.u8(0),
        Recipient::OnPremisesMailbox { archive } => {
            w.u8(1);
            w_archive(w, archive);
        }
        Recipient::RemoteMailbox(m) => {
            w.u8(2);
            w.u8(match m.kind() {
                RemoteKind::User => 0,
                RemoteKind::Room => 1,
                RemoteKind::Equipment => 2,
                RemoteKind::Shared => 3,
            });
            w.u8(match m.mailbox() {
                MailboxState::Provisioned => 0,
                MailboxState::Migrated => 1,
                MailboxState::Deprovisioned => 2,
            });
            w_archive(w, m.archive());
            w.opt(m.routing(), |w, v| w.u32(v.0));
        }
        Recipient::Other(a) => {
            w.u8(3);
            let RecipientAttributes {
                remote_recipient_type,
                display_type,
                type_details,
                target_address,
            } = a;
            w.opt(remote_recipient_type, W::u32);
            w.opt(display_type, W::i32);
            w.opt(type_details, W::u64);
            w.opt(target_address, |w, v| w.u32(v.0));
        }
    }
}
fn r_recipient(r: &mut R<'_>) -> Result<Recipient, DecodeError> {
    match r.u8()? {
        0 => Ok(Recipient::NotMailEnabled),
        1 => Ok(Recipient::OnPremisesMailbox {
            archive: r_archive(r)?,
        }),
        2 => {
            let at = r.at;
            let kind = match r.u8()? {
                0 => RemoteKind::User,
                1 => RemoteKind::Room,
                2 => RemoteKind::Equipment,
                3 => RemoteKind::Shared,
                _ => return r.err("remote kind"),
            };
            let mailbox = match r.u8()? {
                0 => MailboxState::Provisioned,
                1 => MailboxState::Migrated,
                2 => MailboxState::Deprovisioned,
                _ => return r.err("mailbox state"),
            };
            let archive = r_archive(r)?;
            let routing = r.opt(|r| r.u32().map(ValueId))?;
            RemoteMailbox::new(kind, mailbox, archive, routing)
                .map(Recipient::RemoteMailbox)
                .ok_or(DecodeError {
                    at,
                    what: "remote mailbox",
                })
        }
        3 => Ok(Recipient::Other(RecipientAttributes {
            remote_recipient_type: r.opt(R::u32)?,
            display_type: r.opt(R::i32)?,
            type_details: r.opt(R::u64)?,
            target_address: r.opt(|r| r.u32().map(ValueId))?,
        })),
        _ => r.err("recipient"),
    }
}

fn w_state(w: &mut W, s: &NodeState) {
    let NodeState {
        kind,
        active,
        upn,
        primary_smtp,
        dn,
        recipient,
    } = *s;
    w_node_kind(w, kind);
    w.opt(active, W::bool);
    w.opt(upn, |w, v| w.u32(v.0));
    w.opt(primary_smtp, |w, v| w.u32(v.0));
    w.opt(dn, w_dn);
    w.opt(recipient, w_recipient);
}
fn r_state(r: &mut R<'_>) -> Result<NodeState, DecodeError> {
    Ok(NodeState {
        kind: r_node_kind(r)?,
        active: r.opt(R::bool)?,
        upn: r.opt(|r| r.u32().map(ValueId))?,
        primary_smtp: r.opt(|r| r.u32().map(ValueId))?,
        dn: r.opt(r_dn)?,
        recipient: r.opt(r_recipient)?,
    })
}

fn w_change(w: &mut W, c: &Change) {
    match *c {
        Change::RemoveMembership { user, group } => {
            w.u8(0);
            w.guid(user);
            w.guid(group);
        }
        Change::DeleteNode { node, ref state } => {
            w.u8(1);
            w.guid(node);
            w_state(w, state);
        }
        Change::SetAttribute {
            node,
            attribute,
            from,
            to,
        } => {
            w.u8(2);
            w.guid(node);
            w.u8(match attribute {
                Attribute::Upn => 0,
                Attribute::PrimarySmtp => 1,
            });
            w.opt(from, |w, v| w.u32(v.0));
            w.opt(to, |w, v| w.u32(v.0));
        }
        Change::SetActive { node, from, to } => {
            w.u8(3);
            w.guid(node);
            w.opt(from, W::bool);
            w.opt(to, W::bool);
        }
        Change::SetLocation { node, from, to } => {
            w.u8(4);
            w.guid(node);
            w.opt(from, w_dn);
            w.opt(to, w_dn);
        }
        Change::SetRecipient { node, from, to } => {
            w.u8(5);
            w.guid(node);
            w.opt(from, w_recipient);
            w.opt(to, w_recipient);
        }
        Change::CreateNode { node, ref state } => {
            w.u8(6);
            w.guid(node);
            w_state(w, state);
        }
        Change::AddMembership { user, group } => {
            w.u8(7);
            w.guid(user);
            w.guid(group);
        }
    }
}
fn r_change(r: &mut R<'_>) -> Result<Change, DecodeError> {
    Ok(match r.u8()? {
        0 => Change::RemoveMembership {
            user: r.guid()?,
            group: r.guid()?,
        },
        1 => Change::DeleteNode {
            node: r.guid()?,
            state: r_state(r)?,
        },
        2 => Change::SetAttribute {
            node: r.guid()?,
            attribute: match r.u8()? {
                0 => Attribute::Upn,
                1 => Attribute::PrimarySmtp,
                _ => return r.err("attribute"),
            },
            from: r.opt(|r| r.u32().map(ValueId))?,
            to: r.opt(|r| r.u32().map(ValueId))?,
        },
        3 => Change::SetActive {
            node: r.guid()?,
            from: r.opt(R::bool)?,
            to: r.opt(R::bool)?,
        },
        4 => Change::SetLocation {
            node: r.guid()?,
            from: r.opt(r_dn)?,
            to: r.opt(r_dn)?,
        },
        5 => Change::SetRecipient {
            node: r.guid()?,
            from: r.opt(r_recipient)?,
            to: r.opt(r_recipient)?,
        },
        6 => Change::CreateNode {
            node: r.guid()?,
            state: r_state(r)?,
        },
        7 => Change::AddMembership {
            user: r.guid()?,
            group: r.guid()?,
        },
        _ => return r.err("change"),
    })
}

fn w_role(w: &mut W, a: AddressRole) {
    w.u8(match a {
        AddressRole::Upn => 0,
        AddressRole::PrimarySmtp => 1,
        AddressRole::SecondarySmtp => 2,
        AddressRole::Mail => 3,
        AddressRole::Routing => 4,
    });
}
fn r_role(r: &mut R<'_>) -> Result<AddressRole, DecodeError> {
    match r.u8()? {
        0 => Ok(AddressRole::Upn),
        1 => Ok(AddressRole::PrimarySmtp),
        2 => Ok(AddressRole::SecondarySmtp),
        3 => Ok(AddressRole::Mail),
        4 => Ok(AddressRole::Routing),
        _ => r.err("address role"),
    }
}

fn w_violation(w: &mut W, v: &Violation) {
    match v {
        Violation::DuplicateSmtp { key, owners } => {
            w.u8(0);
            w.u32(key.0);
            w.seq(owners, |w, g| w.guid(*g));
        }
        Violation::DuplicateUpn { key, owners } => {
            w.u8(1);
            w.u32(key.0);
            w.seq(owners, |w, g| w.guid(*g));
        }
        Violation::AddressConflict { key, holders } => {
            w.u8(2);
            w.u32(key.0);
            w.seq(holders, |w, (g, a)| {
                w.guid(*g);
                w_role(w, *a);
            });
        }
        Violation::RoutingNotInProxies { node } => {
            w.u8(3);
            w.guid(*node);
        }
        Violation::RoutingMismatch { node } => {
            w.u8(4);
            w.guid(*node);
        }
        Violation::DanglingMembership {
            user,
            group,
            missing,
        } => {
            w.u8(5);
            w.guid(*user);
            w.guid(*group);
            w.u8(match missing {
                Endpoint::User => 0,
                Endpoint::Group => 1,
            });
        }
    }
}
fn r_violation(r: &mut R<'_>) -> Result<Violation, DecodeError> {
    Ok(match r.u8()? {
        0 => Violation::DuplicateSmtp {
            key: KeyId(r.u32()?),
            owners: r.seq(R::guid)?,
        },
        1 => Violation::DuplicateUpn {
            key: KeyId(r.u32()?),
            owners: r.seq(R::guid)?,
        },
        2 => Violation::AddressConflict {
            key: KeyId(r.u32()?),
            holders: r.seq(|r| Ok((r.guid()?, r_role(r)?)))?,
        },
        3 => Violation::RoutingNotInProxies { node: r.guid()? },
        4 => Violation::RoutingMismatch { node: r.guid()? },
        5 => Violation::DanglingMembership {
            user: r.guid()?,
            group: r.guid()?,
            missing: match r.u8()? {
                0 => Endpoint::User,
                1 => Endpoint::Group,
                _ => return r.err("endpoint"),
            },
        },
        _ => return r.err("violation"),
    })
}

fn w_version(w: &mut W, v: &StoredVersion) {
    let StoredVersion {
        id,
        parent,
        origin,
        delta,
    } = v;
    w.u64(id.0);
    w.opt(*parent, |w, p| w.u64(p.0));
    match origin {
        StoredOrigin::Observed {
            source,
            observed_at_ms,
        } => {
            w.u8(0);
            w.str(source);
            w.i64(*observed_at_ms);
        }
        StoredOrigin::Simulated { rule, evidence } => {
            let StoredRule { name, version } = rule;
            w.u8(1);
            w.str(name);
            w.u16(*version);
            w.seq(evidence, |w, e| w.str(&e.0));
        }
    }
    w.seq(delta, w_change);
}
fn r_version(r: &mut R<'_>) -> Result<StoredVersion, DecodeError> {
    let id = VersionId(r.u64()?);
    let parent = r.opt(|r| r.u64().map(VersionId))?;
    let origin = match r.u8()? {
        0 => StoredOrigin::Observed {
            source: r.string()?,
            observed_at_ms: r.i64()?,
        },
        1 => StoredOrigin::Simulated {
            rule: StoredRule {
                name: r.string()?,
                version: r.u16()?,
            },
            evidence: r.seq(|r| r.string().map(EvidenceRef))?,
        },
        _ => return r.err("origin"),
    };
    Ok(StoredVersion {
        id,
        parent,
        origin,
        delta: r.seq(r_change)?,
    })
}

fn w_observed(w: &mut W, n: &ObservedNode) {
    let ObservedNode {
        kind,
        active,
        upn,
        primary_smtp,
        proxies,
        dn,
        recipient,
        mail,
        alias,
        exchange_guid,
        security,
    } = n;
    w_node_kind(w, *kind);
    w.opt(*active, W::bool);
    w.opt(upn.as_deref(), W::str);
    w.opt(primary_smtp.as_deref(), W::str);
    w.seq(proxies, |w, p| w.str(p));
    w.opt(*dn, w_dn);
    w.opt(recipient.as_ref(), |w, rc| {
        let ObservedRecipient {
            remote_recipient_type,
            display_type,
            type_details,
            target_address,
        } = rc;
        w.opt(*remote_recipient_type, W::u32);
        w.opt(*display_type, W::i32);
        w.opt(*type_details, W::u64);
        w.opt(target_address.as_deref(), W::str);
    });
    w.opt(mail.as_deref(), W::str);
    w.opt(alias.as_deref(), W::str);
    w.opt(*exchange_guid, W::guid);
    w.opt(*security, W::bool);
}
fn r_observed(r: &mut R<'_>) -> Result<ObservedNode, DecodeError> {
    Ok(ObservedNode {
        kind: r_node_kind(r)?,
        active: r.opt(R::bool)?,
        upn: r.opt(R::string)?,
        primary_smtp: r.opt(R::string)?,
        proxies: r.seq(R::string)?,
        dn: r.opt(r_dn)?,
        recipient: r.opt(|r| {
            Ok(ObservedRecipient {
                remote_recipient_type: r.opt(R::u32)?,
                display_type: r.opt(R::i32)?,
                type_details: r.opt(R::u64)?,
                target_address: r.opt(R::string)?,
            })
        })?,
        mail: r.opt(R::string)?,
        alias: r.opt(R::string)?,
        exchange_guid: r.opt(R::guid)?,
        security: r.opt(R::bool)?,
    })
}

fn w_observation(w: &mut W, o: &Observation) {
    let Observation {
        scope,
        nodes,
        members,
    } = o;
    w.guid(scope.0);
    w.seq(nodes, |w, (g, n)| {
        w.guid(*g);
        w_observed(w, n);
    });
    w.seq(members, |w, (u, g)| {
        w.guid(*u);
        w.guid(*g);
    });
}
fn r_observation(r: &mut R<'_>) -> Result<Observation, DecodeError> {
    Ok(Observation {
        scope: DirectoryScope(r.guid()?),
        nodes: r.seq(|r| Ok((r.guid()?, r_observed(r)?)))?,
        members: r.seq(|r| Ok((r.guid()?, r.guid()?)))?,
    })
}

// ---------------------------------------------------------------- records

/// What a record holds. The numbers are stored and never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum RecordKind {
    /// The batch header: first record of every batch.
    Header = 0,
    /// A dictionary value, by index.
    Value = 1,
    /// A root's observation, by version.
    Root = 2,
    /// A version.
    Version = 3,
    /// A tag set to a version.
    Tag = 4,
    /// A verdict recorded for a version.
    Verdict = 5,
}

impl RecordKind {
    fn from_u8(k: u8) -> Option<Self> {
        Some(match k {
            0 => Self::Header,
            1 => Self::Value,
            2 => Self::Root,
            3 => Self::Version,
            4 => Self::Tag,
            5 => Self::Verdict,
            _ => return None,
        })
    }
}

/// One record as a storage crate stores it: a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// The batch it belongs to.
    pub batch: u64,
    /// Its position in the batch; the header is `0`.
    pub seq: u32,
    /// What it holds ([`RecordKind`] as a number).
    pub kind: u8,
    /// The encoded body.
    pub body: Vec<u8>,
}

/// A batch header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// The batch number; batches are numbered from 0, densely.
    pub batch: u64,
    /// The first logical version the batch holds.
    pub first_version: u64,
    /// How many logical versions it holds.
    pub versions: u64,
    /// How many records follow the header.
    pub records: u32,
    /// FNV-1a 64 over every record after the header, in order.
    pub checksum: u64,
}

fn fnv1a(records: &[Record]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for r in records {
        eat(r.kind);
        for b in u32::try_from(r.body.len())
            .expect("record body under 4 GiB")
            .to_le_bytes()
        {
            eat(b);
        }
        for &b in &r.body {
            eat(b);
        }
    }
    h
}

/// A batch ready to store: its records, header first.
#[derive(Clone, Debug)]
pub struct Batch {
    /// The records, in `seq` order.
    pub records: Vec<Record>,
    next: LogState,
}

impl Batch {
    /// The batch number.
    pub fn number(&self) -> u64 {
        self.records[0].batch
    }
}

// ---------------------------------------------------------------- writer

/// What the log already holds: everything [`LogState::batch`] leaves out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogState {
    next_batch: u64,
    values: usize,
    versions: u64,
    /// A digest of the logged values, roots and versions (`prefix_digest`):
    /// what an image must still hold to continue the log.
    prefix: u64,
    tags: BTreeMap<String, VersionId>,
    verdicts: BTreeMap<VersionId, Vec<Violation>>,
}

/// Why an image cannot be appended to the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogError {
    /// The image does not start with the logged history: it has fewer
    /// values or versions than the log holds, or the logged values, roots
    /// or versions differ in it.
    NotAContinuation,
    /// A tag the log holds is gone from the image; tags are never removed.
    TagRemoved(String),
    /// A verdict the log holds is gone from the image.
    VerdictRemoved(VersionId),
    /// The batch was built against another log state (a batch is committed
    /// once, in number order).
    StaleBatch {
        /// The batch the log expects next.
        expected: u64,
        /// The batch offered.
        found: u64,
    },
}

impl Default for LogState {
    /// An empty log. Its prefix is the digest of the empty history, not 0,
    /// so the first image is checked like every later one.
    fn default() -> Self {
        Self {
            next_batch: 0,
            values: 0,
            versions: 0,
            prefix: prefix_digest(&StoredImage::default(), 0, 0),
            tags: BTreeMap::new(),
            verdicts: BTreeMap::new(),
        }
    }
}

impl LogState {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number the next batch gets.
    pub fn next_batch(&self) -> u64 {
        self.next_batch
    }

    /// The batch holding what `image` has and the log does not, or `None`
    /// when there is nothing new. The log state is unchanged until the
    /// batch is [`committed`](Self::committed).
    ///
    /// # Errors
    ///
    /// [`LogError`] when `image` does not extend the logged history.
    pub fn batch(&self, image: &StoredImage) -> Result<Option<Batch>, LogError> {
        let versions = image.versions.len() as u64;
        if image.values.len() < self.values
            || versions < self.versions
            || prefix_digest(image, self.values, self.versions) != self.prefix
        {
            return Err(LogError::NotAContinuation);
        }
        let tags: BTreeMap<String, VersionId> = image.tags.iter().cloned().collect();
        let verdicts: BTreeMap<VersionId, Vec<Violation>> =
            image.verdicts.iter().cloned().collect();
        if let Some(t) = self.tags.keys().find(|t| !tags.contains_key(*t)) {
            return Err(LogError::TagRemoved(t.clone()));
        }
        if let Some(v) = self.verdicts.keys().find(|v| !verdicts.contains_key(v)) {
            return Err(LogError::VerdictRemoved(*v));
        }

        let mut body: Vec<(RecordKind, Vec<u8>)> = Vec::new();
        for (i, value) in image.values.iter().enumerate().skip(self.values) {
            let mut w = W::default();
            w.u32(u32::try_from(i).expect("fewer than 2^32 values"));
            w.str(value);
            body.push((RecordKind::Value, w.0));
        }
        for (v, obs) in &image.roots {
            if v.0 >= self.versions {
                let mut w = W::default();
                w.u64(v.0);
                w_observation(&mut w, obs);
                body.push((RecordKind::Root, w.0));
            }
        }
        for v in &image.versions[self.versions as usize..] {
            let mut w = W::default();
            w_version(&mut w, v);
            body.push((RecordKind::Version, w.0));
        }
        for (name, v) in &tags {
            if self.tags.get(name) != Some(v) {
                let mut w = W::default();
                w.str(name);
                w.u64(v.0);
                body.push((RecordKind::Tag, w.0));
            }
        }
        for (v, x) in &verdicts {
            if self.verdicts.get(v) != Some(x) {
                let mut w = W::default();
                w.u64(v.0);
                w.seq(x, w_violation);
                body.push((RecordKind::Verdict, w.0));
            }
        }
        if body.is_empty() {
            return Ok(None);
        }

        let batch = self.next_batch;
        let records: Vec<Record> = body
            .into_iter()
            .enumerate()
            .map(|(i, (kind, body))| Record {
                batch,
                seq: u32::try_from(i + 1).expect("fewer than 2^32 records in a batch"),
                kind: kind as u8,
                body,
            })
            .collect();
        let header = Header {
            batch,
            first_version: self.versions,
            versions: versions - self.versions,
            records: u32::try_from(records.len()).expect("fewer than 2^32 records"),
            checksum: fnv1a(&records),
        };
        let mut all = Vec::with_capacity(records.len() + 1);
        all.push(Record {
            batch,
            seq: 0,
            kind: RecordKind::Header as u8,
            body: encode_header(&header),
        });
        all.extend(records);
        Ok(Some(Batch {
            records: all,
            next: LogState {
                next_batch: batch + 1,
                values: image.values.len(),
                versions,
                prefix: prefix_digest(image, image.values.len(), versions),
                tags,
                verdicts,
            },
        }))
    }

    /// Record that `batch` is durably stored. Call only after the storage
    /// commit succeeded.
    ///
    /// # Errors
    ///
    /// [`LogError::StaleBatch`] if `batch` is not the next one.
    pub fn committed(&mut self, batch: &Batch) -> Result<(), LogError> {
        if batch.number() != self.next_batch {
            return Err(LogError::StaleBatch {
                expected: self.next_batch,
                found: batch.number(),
            });
        }
        *self = batch.next.clone();
        Ok(())
    }
}

/// FNV-1a over the first `values` values, the roots of versions below
/// `versions` (in version order) and the first `versions` versions, each
/// item length-prefixed. Two images agree on it exactly when they agree on
/// that prefix (up to hash collision), which is what lets
/// [`LogState::batch`] refuse an image that rewrote logged history.
///
/// It reads the whole prefix, so a save costs time linear in the history;
/// `batch` already walks every root.
fn prefix_digest(image: &StoredImage, values: usize, versions: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |w: W| {
        let len = u32::try_from(w.0.len()).expect("item under 4 GiB");
        for b in len.to_le_bytes().into_iter().chain(w.0) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for value in image.values.iter().take(values) {
        let mut w = W::default();
        w.str(value);
        eat(w);
    }
    let mut roots: Vec<&(VersionId, Observation)> =
        image.roots.iter().filter(|(v, _)| v.0 < versions).collect();
    roots.sort_by_key(|(v, _)| *v);
    for (v, obs) in roots {
        let mut w = W::default();
        w.u64(v.0);
        w_observation(&mut w, obs);
        eat(w);
    }
    for v in image.versions.iter().take(versions as usize) {
        let mut w = W::default();
        w_version(&mut w, v);
        eat(w);
    }
    h
}

fn encode_header(h: &Header) -> Vec<u8> {
    let Header {
        batch,
        first_version,
        versions,
        records,
        checksum,
    } = *h;
    let mut w = W::default();
    w.u64(batch);
    w.u64(first_version);
    w.u64(versions);
    w.u32(records);
    w.u64(checksum);
    w.0
}

fn decode_header(b: &[u8]) -> Result<Header, DecodeError> {
    let mut r = R::new(b);
    let h = Header {
        batch: r.u64()?,
        first_version: r.u64()?,
        versions: r.u64()?,
        records: r.u32()?,
        checksum: r.u64()?,
    };
    r.end()?;
    Ok(h)
}

// ---------------------------------------------------------------- reader

/// Why a log was refused. Every variant names the batch it found damaged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// A batch number is missing (or a batch appears twice).
    BatchGap {
        /// The number expected.
        expected: u64,
        /// The number found.
        found: u64,
    },
    /// A batch whose first record is not a header for that batch.
    MissingHeader(u64),
    /// Record positions are not `0..=count` in order.
    RecordOrder(u64),
    /// The batch holds another number of records than its header says.
    RecordCount {
        /// The batch.
        batch: u64,
        /// The header's count.
        expected: u32,
        /// The records present.
        found: u32,
    },
    /// The records do not hash to the header's checksum.
    Checksum(u64),
    /// A record of an unknown kind.
    UnknownKind(u64, u8),
    /// A record body that does not decode.
    Decode(u64, DecodeError),
    /// A value at another index than the next one.
    ValueOrder(u64),
    /// A version that does not follow the previous one, or a count that
    /// disagrees with the header.
    VersionOrder(u64),
}

/// The image a log holds, and the log state to append after it.
///
/// # Errors
///
/// [`ReplayError`] at the first damaged batch; nothing is returned for the
/// batches before it.
pub fn replay(
    records: impl IntoIterator<Item = Record>,
) -> Result<(StoredImage, LogState), ReplayError> {
    let (image, state, err) = replay_prefix(records);
    match err {
        None => Ok((image, state)),
        Some(e) => Err(e),
    }
}

/// [`replay`] up to the last good batch: the image and log state of every
/// batch before the first damaged one, and the damage, if any.
///
/// For inspecting a damaged log. Appending after a prefix would overwrite
/// the damaged batch's number, so a caller decides that explicitly.
pub fn replay_prefix(
    records: impl IntoIterator<Item = Record>,
) -> (StoredImage, LogState, Option<ReplayError>) {
    let mut records: Vec<Record> = records.into_iter().collect();
    records.sort_by_key(|r| (r.batch, r.seq));
    let mut image = StoredImage::default();
    let mut state = LogState::new();
    let mut tags: BTreeMap<String, VersionId> = BTreeMap::new();
    let mut verdicts: BTreeMap<VersionId, Vec<Violation>> = BTreeMap::new();
    let mut err = None;
    for run in records.chunk_by(|a, b| a.batch == b.batch) {
        let b = run[0].batch;
        // A batch is applied to the image in place and undone on failure by
        // truncating to these lengths (values, roots and versions only ever
        // grow), so replay never copies the accumulated image. Its tags and
        // verdicts collect apart and merge only once the batch is whole.
        let lens = (image.values.len(), image.roots.len(), image.versions.len());
        let (mut ntags, mut nverdicts) = (BTreeMap::new(), BTreeMap::new());
        match apply_batch(
            b,
            state.next_batch,
            run,
            &mut image,
            &mut ntags,
            &mut nverdicts,
        ) {
            Ok(()) => {
                tags.extend(ntags);
                verdicts.extend(nverdicts);
                state.next_batch = b + 1;
            }
            Err(e) => {
                image.values.truncate(lens.0);
                image.roots.truncate(lens.1);
                image.versions.truncate(lens.2);
                err = Some(e);
                break;
            }
        }
    }
    image.tags = tags.clone().into_iter().collect();
    image.verdicts = verdicts.clone().into_iter().collect();
    state.values = image.values.len();
    state.versions = image.versions.len() as u64;
    state.prefix = prefix_digest(&image, state.values, state.versions);
    state.tags = tags;
    state.verdicts = verdicts;
    (image, state, err)
}

fn apply_batch(
    b: u64,
    expected: u64,
    run: &[Record],
    image: &mut StoredImage,
    tags: &mut BTreeMap<String, VersionId>,
    verdicts: &mut BTreeMap<VersionId, Vec<Violation>>,
) -> Result<(), ReplayError> {
    if b != expected {
        return Err(ReplayError::BatchGap { expected, found: b });
    }
    let head = &run[0];
    if head.seq != 0 || head.kind != RecordKind::Header as u8 {
        return Err(ReplayError::MissingHeader(b));
    }
    let header = decode_header(&head.body).map_err(|e| ReplayError::Decode(b, e))?;
    if header.batch != b {
        return Err(ReplayError::MissingHeader(b));
    }
    let rest = &run[1..];
    if rest
        .iter()
        .enumerate()
        .any(|(i, r)| r.seq as usize != i + 1 || r.kind == RecordKind::Header as u8)
    {
        return Err(ReplayError::RecordOrder(b));
    }
    let found = u32::try_from(rest.len()).unwrap_or(u32::MAX);
    if found != header.records {
        return Err(ReplayError::RecordCount {
            batch: b,
            expected: header.records,
            found,
        });
    }
    if fnv1a(rest) != header.checksum {
        return Err(ReplayError::Checksum(b));
    }
    if header.first_version != image.versions.len() as u64 {
        return Err(ReplayError::VersionOrder(b));
    }
    let decode = |e| ReplayError::Decode(b, e);
    let mut added = 0u64;
    for rec in rest {
        let kind = RecordKind::from_u8(rec.kind).ok_or(ReplayError::UnknownKind(b, rec.kind))?;
        let mut r = R::new(&rec.body);
        match kind {
            RecordKind::Header => unreachable!("rejected above"),
            RecordKind::Value => {
                let i = r.u32().map_err(decode)? as usize;
                let s = r.string().map_err(decode)?;
                if i != image.values.len() {
                    return Err(ReplayError::ValueOrder(b));
                }
                image.values.push(s);
            }
            RecordKind::Root => {
                let v = VersionId(r.u64().map_err(decode)?);
                image
                    .roots
                    .push((v, r_observation(&mut r).map_err(decode)?));
            }
            RecordKind::Version => {
                let v = r_version(&mut r).map_err(decode)?;
                if v.id.0 != image.versions.len() as u64 {
                    return Err(ReplayError::VersionOrder(b));
                }
                image.versions.push(v);
                added += 1;
            }
            RecordKind::Tag => {
                let name = r.string().map_err(decode)?;
                tags.insert(name, VersionId(r.u64().map_err(decode)?));
            }
            RecordKind::Verdict => {
                let v = VersionId(r.u64().map_err(decode)?);
                verdicts.insert(v, r.seq(r_violation).map_err(decode)?);
            }
        }
        r.end().map_err(decode)?;
    }
    if added != header.versions {
        return Err(ReplayError::VersionOrder(b));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(n: u8) -> Guid128 {
        Guid128([n; 16])
    }

    // A batch that fails after it already changed the image (here: its
    // header claims a version it does not carry, found only at the end)
    // leaves the image exactly as the batches before it built it.
    #[test]
    fn a_batch_failing_after_it_applied_is_undone() {
        let ab = StoredImage {
            values: vec!["a".into(), "b".into()],
            ..StoredImage::default()
        };
        let mut abc = ab.clone();
        abc.values.push("c".into());
        abc.tags.push(("t".into(), VersionId(0)));

        let mut log = LogState::new();
        let first = log.batch(&ab).unwrap().unwrap();
        log.committed(&first).unwrap();
        let mut second = log.batch(&abc).unwrap().unwrap();
        let mut header = decode_header(&second.records[0].body).unwrap();
        header.versions = 1;
        second.records[0].body = encode_header(&header);

        let (image, state, err) = replay_prefix(first.records.into_iter().chain(second.records));
        assert_eq!(err, Some(ReplayError::VersionOrder(1)));
        assert_eq!(image.values, ab.values, "the failed batch's value is gone");
        assert!(image.tags.is_empty(), "the failed batch's tag is gone");
        assert_eq!(state.next_batch(), 1);
        // The state replay returns continues the log it read.
        assert!(state.batch(&ab).unwrap().is_none());
    }

    fn round<T: PartialEq + std::fmt::Debug>(
        x: &T,
        w: impl Fn(&mut W, &T),
        r: impl Fn(&mut R<'_>) -> Result<T, DecodeError>,
    ) {
        let mut out = W::default();
        w(&mut out, x);
        let mut rd = R::new(&out.0);
        assert_eq!(&r(&mut rd).unwrap(), x);
        rd.end().unwrap();
        // Every strict prefix is refused, never read as something shorter.
        for n in 0..out.0.len() {
            let mut rd = R::new(&out.0[..n]);
            assert!(
                r(&mut rd).and_then(|_| rd.end()).is_err(),
                "prefix {n} of {x:?}"
            );
        }
    }

    fn recipients() -> Vec<Recipient> {
        vec![
            Recipient::NotMailEnabled,
            Recipient::OnPremisesMailbox {
                archive: ArchiveState::Provisioned,
            },
            Recipient::RemoteMailbox(
                RemoteMailbox::new(
                    RemoteKind::Shared,
                    MailboxState::Migrated,
                    ArchiveState::Deprovisioned,
                    Some(ValueId(9)),
                )
                .unwrap(),
            ),
            Recipient::Other(RecipientAttributes {
                remote_recipient_type: Some(4),
                display_type: Some(-2_147_483_642),
                type_details: Some(2_147_483_648),
                target_address: Some(ValueId(3)),
            }),
        ]
    }

    fn state(rc: Option<Recipient>) -> NodeState {
        NodeState {
            kind: NodeKind::User,
            active: Some(false),
            upn: Some(ValueId(1)),
            primary_smtp: None,
            dn: Some(Dn128::new(&[0, 7, 255]).unwrap()),
            recipient: rc,
        }
    }

    #[test]
    fn every_change_round_trips() {
        let mut changes = vec![
            Change::RemoveMembership {
                user: g(1),
                group: g(2),
            },
            Change::DeleteNode {
                node: g(3),
                state: state(None),
            },
            Change::SetAttribute {
                node: g(4),
                attribute: Attribute::PrimarySmtp,
                from: None,
                to: Some(ValueId(5)),
            },
            Change::SetActive {
                node: g(5),
                from: Some(true),
                to: None,
            },
            Change::SetLocation {
                node: g(6),
                from: None,
                to: Some(Dn128::new(&[1]).unwrap()),
            },
            Change::CreateNode {
                node: g(7),
                state: state(Some(Recipient::NotMailEnabled)),
            },
            Change::AddMembership {
                user: g(8),
                group: g(9),
            },
        ];
        for rc in recipients() {
            changes.push(Change::SetRecipient {
                node: g(10),
                from: Some(rc),
                to: None,
            });
        }
        for c in &changes {
            round(c, w_change, r_change);
        }
    }

    #[test]
    fn every_violation_round_trips() {
        for v in [
            Violation::DuplicateSmtp {
                key: KeyId(1),
                owners: vec![g(1), g(2)],
            },
            Violation::DuplicateUpn {
                key: KeyId(2),
                owners: vec![],
            },
            Violation::AddressConflict {
                key: KeyId(3),
                holders: vec![
                    (g(1), AddressRole::Upn),
                    (g(2), AddressRole::Routing),
                    (g(3), AddressRole::Mail),
                ],
            },
            Violation::RoutingNotInProxies { node: g(4) },
            Violation::RoutingMismatch { node: g(5) },
            Violation::DanglingMembership {
                user: g(6),
                group: g(7),
                missing: Endpoint::Group,
            },
        ] {
            round(&v, w_violation, r_violation);
        }
    }

    #[test]
    fn an_observation_round_trips_with_every_field() {
        let node = ObservedNode {
            kind: NodeKind::User,
            active: None,
            upn: Some("a@x.test".into()),
            primary_smtp: Some("A@x.test".into()),
            proxies: vec!["smtp:b@x.test".into(), "X500:/o=ä".into()],
            dn: Some(Dn128::new(&[2, 3]).unwrap()),
            recipient: Some(ObservedRecipient {
                remote_recipient_type: Some(4),
                display_type: Some(-1),
                type_details: None,
                target_address: Some("a@t.mail.onmicrosoft.com".into()),
            }),
            mail: Some("card@x.test".into()),
            alias: Some("a".into()),
            exchange_guid: Some(g(0xEE)),
            security: Some(true),
        };
        round(&node, w_observed, r_observed);
        let obs = Observation {
            scope: DirectoryScope(g(0x5C)),
            nodes: vec![(g(1), node), (g(2), ObservedNode::group())],
            members: vec![(g(1), g(2))],
        };
        let mut w = W::default();
        w_observation(&mut w, &obs);
        let mut r = R::new(&w.0);
        let back = r_observation(&mut r).unwrap();
        r.end().unwrap();
        assert_eq!(back.scope, obs.scope);
        assert_eq!(back.nodes, obs.nodes);
        assert_eq!(back.members, obs.members);
    }

    // A remote mailbox the lifecycle cannot produce is refused on read,
    // not built (a deprovisioned mailbox with a routing address).
    #[test]
    fn an_impossible_remote_mailbox_is_refused() {
        let mut w = W::default();
        w.u8(2);
        w.u8(0);
        w.u8(2);
        w_archive(&mut w, ArchiveState::None);
        w.opt(Some(ValueId(1)), |w, v| w.u32(v.0));
        let err = r_recipient(&mut R::new(&w.0)).unwrap_err();
        assert_eq!(err.what, "remote mailbox");
    }
}
