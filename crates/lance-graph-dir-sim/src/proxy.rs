//! `proxyAddresses` as a relation, never as a field of a node.
//!
//! A user has any number of proxy addresses, so they are rows, not a column:
//!
//! | lane    | type    | meaning                                              |
//! |---------|---------|------------------------------------------------------|
//! | `owner` | `[u32]` | [`UserOrdinal`] (32-bit lane; the substrate has none narrower) |
//! | `key`   | `[u32]` | [`KeyId`] of the normalized address — the comparison identity |
//! | `value` | `[u32]` | [`ValueId`] of the address exactly as observed (egress only) |
//! | `meta`  | `[u32]` | kind (bits 0..2) and the primary flag (bit 2) — [`meta`] |
//! | `smtp_first` | plane | the first SMTP row of each `(owner, key)` run       |
//!
//! The namespace prefix is metadata, not identity: `SMTP:Alice@x.de` and
//! `smtp:alice@X.DE` have one [`KeyId`]. The upper-case `SMTP:` marks the
//! primary address and lives in `meta`; it never reaches the key.
//!
//! The primary row comes from the node's `primary_smtp` (the attribute lane
//! the rules override); every other proxy comes from
//! [`ObservedNode::proxies`](crate::ObservedNode::proxies). Rows are sorted by
//! `(owner, kind, key, primary)` with secondaries first, so `smtp_first`
//! picks one row per owner and address — what makes an owner count once, not
//! once per spelling.

use crate::snapshot::{set_bit, Dicts, ObservedNode, Population, UserOrdinal, NONE};
use lance_graph_mask_risc::words_for;
use ogar_dir_core::Guid128;
use ogar_dir_sim::{KeyId, ValueId};

/// A proxy address namespace. Two bits in [`meta`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ProxyKind {
    /// `SMTP:` / `smtp:` — the only kind uniqueness compares.
    Smtp = 0,
    /// `X500:` — legacy Exchange DN.
    X500 = 1,
    /// `SIP:` — Skype / Teams.
    Sip = 2,
    /// Anything else (`EUM:`, `X400:`, no prefix). Kept, never compared.
    Other = 3,
}

/// The fixed-width `meta` code: `kind | primary << 2`.
pub mod meta {
    use super::ProxyKind;

    /// The primary flag.
    pub const PRIMARY: u32 = 1 << 2;
    /// A secondary SMTP row.
    pub const SMTP_SECONDARY: u32 = ProxyKind::Smtp as u32;
    /// The primary SMTP row.
    pub const SMTP_PRIMARY: u32 = ProxyKind::Smtp as u32 | PRIMARY;

    /// Encode.
    pub const fn encode(kind: ProxyKind, primary: bool) -> u32 {
        kind as u32 | if primary { PRIMARY } else { 0 }
    }
    /// Decode; `None` for a code no row carries.
    pub const fn decode(m: u32) -> Option<(ProxyKind, bool)> {
        let kind = match m & 0b11 {
            0 => ProxyKind::Smtp,
            1 => ProxyKind::X500,
            2 => ProxyKind::Sip,
            _ => ProxyKind::Other,
        };
        let primary = m & PRIMARY != 0;
        if m & !0b111 != 0 || (primary && !matches!(kind, ProxyKind::Smtp)) {
            return None;
        }
        Some((kind, primary))
    }
}

/// Split a raw `proxyAddresses` value into kind, primary flag and address.
///
/// Namespaces are matched case-insensitively; only the exact upper-case
/// `SMTP:` is primary. An unknown namespace (or no `:`) is
/// [`ProxyKind::Other`] and keeps its whole text as the address, so nothing
/// observed is dropped.
pub fn parse_proxy(raw: &str) -> (ProxyKind, bool, &str) {
    let Some((ns, addr)) = raw.split_once(':') else {
        return (ProxyKind::Other, false, raw);
    };
    match ns.to_ascii_lowercase().as_str() {
        "smtp" => (ProxyKind::Smtp, ns == "SMTP", addr),
        "x500" => (ProxyKind::X500, false, addr),
        "sip" => (ProxyKind::Sip, false, addr),
        _ => (ProxyKind::Other, false, raw),
    }
}

/// One row, decoded for a reader (tests, reports). Execution reads lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProxyRow {
    /// Owner.
    pub owner: UserOrdinal,
    /// Comparison identity of the address.
    pub key: KeyId,
    /// The address exactly as observed.
    pub value: ValueId,
    /// Namespace.
    pub kind: ProxyKind,
    /// Upper-case `SMTP:`.
    pub primary: bool,
}

/// The user × proxy relation of one snapshot (see the module docs).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ProxyRelation {
    pub(crate) owner: Vec<u32>,
    pub(crate) key: Vec<u32>,
    pub(crate) value: Vec<u32>,
    pub(crate) meta: Vec<u32>,
    pub(crate) smtp_first: Vec<u64>,
}

impl ProxyRelation {
    /// Build from the users of an observation, in ordinal order (the order
    /// `users` was built in). Interns every proxy address into `d`. Each
    /// node's proxies are interned in sorted order, so ids do not depend on
    /// the order a source listed them in.
    pub(crate) fn build(
        nodes: &[&(Guid128, ObservedNode)],
        users: &Population,
        d: &mut Dicts,
    ) -> Self {
        let mut rows: Vec<(u32, u32, u32, u32)> = Vec::new(); // (owner, meta, key, value)
        for (o, (_, node)) in nodes.iter().enumerate() {
            let o = o as u32;
            if users.smtp_key[o as usize] != NONE {
                rows.push((
                    o,
                    meta::SMTP_PRIMARY,
                    users.smtp_key[o as usize],
                    users.smtp_val[o as usize],
                ));
            }
            let mut raw: Vec<&str> = node.proxies.iter().map(String::as_str).collect();
            raw.sort_unstable();
            for r in raw {
                let (kind, primary, addr) = parse_proxy(r);
                let v = d.intern(addr);
                let k = d.key_of(v).expect("just interned");
                rows.push((o, meta::encode(kind, primary), k.0, v.0));
            }
        }
        // (owner, kind, key, primary) with secondaries first; ties by value.
        rows.sort_unstable_by_key(|&(o, m, k, v)| (o, m & 0b11, k, m & meta::PRIMARY, v));
        let mut rel = Self {
            smtp_first: vec![0; words_for(rows.len())],
            ..Self::default()
        };
        let mut prev = None;
        for (i, &(o, m, k, v)) in rows.iter().enumerate() {
            if m & 0b11 == ProxyKind::Smtp as u32 {
                if prev != Some((o, k)) {
                    set_bit(&mut rel.smtp_first, i);
                }
                prev = Some((o, k));
            }
            rel.owner.push(o);
            rel.meta.push(m);
            rel.key.push(k);
            rel.value.push(v);
        }
        rel
    }

    /// Row count.
    pub fn len(&self) -> usize {
        self.owner.len()
    }
    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.owner.is_empty()
    }
    /// Row `i`, decoded.
    ///
    /// # Panics
    ///
    /// If `i` is out of range.
    pub fn row(&self, i: usize) -> ProxyRow {
        let (kind, primary) = meta::decode(self.meta[i]).expect("built rows decode");
        ProxyRow {
            owner: UserOrdinal(self.owner[i] as u16),
            key: KeyId(self.key[i]),
            value: ValueId(self.value[i]),
            kind,
            primary,
        }
    }
    /// The rows of one owner (rows are sorted by owner).
    pub(crate) fn owner_rows(&self, owner: u32) -> std::ops::Range<usize> {
        let lo = self.owner.partition_point(|&o| o < owner);
        let hi = self.owner.partition_point(|&o| o <= owner);
        lo..hi
    }
    /// Whether `owner` holds `key` as a secondary SMTP address. Delta-sized
    /// callers only: one binary search over the owner lane.
    pub(crate) fn has_secondary_smtp(&self, owner: u32, key: u32) -> bool {
        self.owner_rows(owner)
            .any(|i| self.meta[i] == meta::SMTP_SECONDARY && self.key[i] == key)
    }
}
