//! A version as Active Directory entries, and LDIF.
//!
//! Any version — observed from AD, read from the cloud, or simulated — can be
//! shown as the directory it describes: one entry per OU on a node's path,
//! one per user and one per group, under a naming context the caller names.
//! Every entry carries [`Origin`]: whether it was observed in AD, read from
//! the cloud and mirrored at its on-premises location, placed synthetically
//! because the cloud reported no on-premises location, or created by a
//! simulation. An emulation never passes for the directory it emulates.
//!
//! The projection is read-only: it writes nothing anywhere, and the LDIF it
//! produces is for export and diffing, not for import into AD.
//!
//! What is emitted, per kind:
//!
//! | kind  | attributes |
//! |-------|------------|
//! | OU    | `objectClass: top, organizationalUnit`, `ou` |
//! | user  | `objectClass: top, person, organizationalPerson, user`, `cn`, `objectGUID`, `userPrincipalName`, `mail`, `proxyAddresses`, `userAccountControl`, `msExchMailboxGuid` |
//! | group | `objectClass: top, group`, `cn`, `objectGUID`, `proxyAddresses`, `member` |
//!
//! plus `dirSimOrigin` on every entry. An attribute the version does not
//! hold is absent, never guessed: `userAccountControl` only when the flag is
//! known (512 enabled, 514 disabled), `msExchMailboxGuid` only when
//! observed. A node's relative name is `CN=<objectGUID>`: the version holds
//! no display name, and identity is the GUID.
//!
//! `proxyAddresses` is the effective primary SMTP address (`SMTP:`) followed
//! by the observed non-primary addresses, in their observed spelling. A
//! simulated change of the primary address replaces the `SMTP:` value; the
//! previous primary is not invented as a secondary.

use crate::observe::SYNTHETIC_OU;
use crate::snapshot::NodeKind;
use crate::view::View;
use ogar_dir_core::{Dn, Dn128, Guid128, OuDictionary, OuHhtl, OU_LEVELS};
use std::collections::BTreeSet;

/// Where an entry came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Origin {
    /// Observed in Active Directory.
    Observed,
    /// Read from the cloud, located at its on-premises DN.
    Mirrored,
    /// Read from the cloud with no on-premises location; placed under
    /// [`SYNTHETIC_OU`].
    Synthetic,
    /// Created by a simulation; not observed anywhere.
    Simulated,
}

impl Origin {
    /// The `dirSimOrigin` value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observed => "observed",
            Self::Mirrored => "mirrored",
            Self::Synthetic => "synthetic",
            Self::Simulated => "simulated",
        }
    }
}

/// Where the version's observation came from.
#[derive(Clone, Copy, Debug)]
pub enum Source<'a> {
    /// An AD observation (`observe::from_ad`).
    Ad,
    /// A cloud observation (`observe::from_graph`), with the users it placed
    /// synthetically (`GraphObservation::synthetic`, sorted).
    Cloud {
        /// The synthetically placed users.
        synthetic: &'a [Guid128],
    },
}

/// One attribute value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// UTF-8 text.
    Text(String),
    /// Raw bytes (`objectGUID`, `msExchMailboxGuid`).
    Binary(Vec<u8>),
}

/// One directory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The full DN, RFC 4514 escaped.
    pub dn: String,
    /// Where it came from.
    pub origin: Origin,
    /// The node, for users and groups; `None` for an OU.
    pub node: Option<Guid128>,
    /// Attributes in emission order; a multi-valued attribute repeats.
    pub attrs: Vec<(&'static str, Value)>,
}

impl Entry {
    /// The text values of `name`, in order.
    pub fn text(&self, name: &str) -> Vec<&str> {
        self.attrs
            .iter()
            .filter(|(n, _)| n.eq_ignore_ascii_case(name))
            .filter_map(|(_, v)| match v {
                Value::Text(t) => Some(t.as_str()),
                Value::Binary(_) => None,
            })
            .collect()
    }
}

/// Why a version could not be projected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectError {
    /// The naming context is not a DN made only of `DC=` components.
    NamingContext(String),
    /// A node has no location in this version.
    Unlocated {
        /// The node.
        node: Guid128,
    },
    /// A node's location is not in the OU dictionary (the dictionary is not
    /// the one the observation was read with).
    UnknownOu {
        /// The node.
        node: Guid128,
    },
}

/// The projection of one version.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Projection {
    /// OUs first (parents before children), then users, then groups; each
    /// group of entries sorted by DN.
    pub entries: Vec<Entry>,
    /// Memberships whose user or group does not exist in this version: an
    /// AD `member` cannot name a missing object, so they are counted, not
    /// emitted.
    pub dangling_members: usize,
}

/// Escape one RDN value (RFC 4514 §2.4): the special characters with a
/// backslash, every control character as `\HH`.
pub fn escape_rdn_value(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    let last = v.chars().count().saturating_sub(1);
    for (i, c) in v.chars().enumerate() {
        match c {
            ',' | '+' | '"' | '\\' | '<' | '>' | ';' | '=' => {
                out.push('\\');
                out.push(c);
            }
            '#' if i == 0 => out.push_str("\\#"),
            ' ' if i == 0 || i == last => out.push_str("\\ "),
            // Every control character, NUL and DEL included, as hex pairs.
            c if c.is_control() => {
                let mut buf = [0u8; 4];
                for b in c.encode_utf8(&mut buf).bytes() {
                    out.push_str(&format!("\\{b:02X}"));
                }
            }
            _ => out.push(c),
        }
    }
    out
}

fn ou_hhtl_of(dn: &Dn128) -> Option<OuHhtl> {
    if dn.depth() > OU_LEVELS {
        return None;
    }
    let mut h = OuHhtl::ROOT;
    for (i, &c) in dn.codes().iter().enumerate() {
        h.0[i] = u16::from(c) + 1;
    }
    Some(h)
}

/// Project version `v` as AD entries under `naming_context` (e.g.
/// `DC=example,DC=de`). `dict` resolves locations to OU names: the
/// dictionary the observation was read with.
pub fn project(
    v: &View<'_>,
    source: Source<'_>,
    naming_context: &str,
    dict: &OuDictionary,
) -> Result<Projection, ProjectError> {
    let nc = Dn::parse(naming_context)
        .ok()
        .filter(|d| d.rdns.iter().all(|r| r.is("DC")))
        .ok_or_else(|| ProjectError::NamingContext(naming_context.to_string()))?;
    let nc: String = nc
        .rdns
        .iter()
        .map(|r| format!("DC={}", escape_rdn_value(&r.value)))
        .collect::<Vec<_>>()
        .join(",");
    let (cloud, synthetic) = match source {
        Source::Ad => (false, &[][..]),
        Source::Cloud { synthetic } => (true, synthetic),
    };
    let base_origin = if cloud {
        Origin::Mirrored
    } else {
        Origin::Observed
    };
    let snap = v.snapshot();
    let origin_of = |g: &Guid128| {
        if snap.user_ordinal(g).is_none() && snap.group_ordinal(g).is_none() {
            Origin::Simulated
        } else if cloud && synthetic.binary_search(g).is_ok() {
            Origin::Synthetic
        } else {
            base_origin
        }
    };
    let text = |id: ogar_dir_sim::ValueId| v.dicts().value(id).map(str::to_string);

    // OU path of every node, and every OU on the way.
    let mut ous: BTreeSet<Vec<String>> = BTreeSet::new();
    let mut nodes: Vec<(NodeKind, Guid128, String)> = Vec::new();
    for kind in [NodeKind::User, NodeKind::Group] {
        for i in 0..v.len_in(kind) {
            let Some(g) = v.guid_in(kind, i) else {
                continue;
            };
            let s = v.node_state(&g).expect("existing node");
            let dn = s.dn.ok_or(ProjectError::Unlocated { node: g })?;
            let path = ou_hhtl_of(&dn)
                .and_then(|h| dict.explain(&h))
                .filter(|p| p.len() == dn.depth())
                .ok_or(ProjectError::UnknownOu { node: g })?;
            for d in 1..=path.len() {
                ous.insert(path[..d].to_vec());
            }
            nodes.push((kind, g, render(&format!("CN={g}"), &path, &nc)));
        }
    }

    let mut out = Projection::default();
    let mut ou_entries: Vec<(usize, String, Entry)> = ous
        .into_iter()
        .map(|path| {
            let leaf = path.last().expect("non-empty").clone();
            let dn = render(
                &format!("OU={}", escape_rdn_value(&leaf)),
                &path[..path.len() - 1],
                &nc,
            );
            let origin = if cloud && path[0] == SYNTHETIC_OU {
                Origin::Synthetic
            } else {
                base_origin
            };
            let e = Entry {
                dn: dn.clone(),
                origin,
                node: None,
                attrs: vec![
                    ("objectClass", Value::Text("top".into())),
                    ("objectClass", Value::Text("organizationalUnit".into())),
                    ("ou", Value::Text(leaf)),
                    ("dirSimOrigin", Value::Text(origin.as_str().into())),
                ],
            };
            (path.len(), dn, e)
        })
        .collect();
    ou_entries.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    out.entries
        .extend(ou_entries.into_iter().map(|(_, _, e)| e));

    let dn_of: std::collections::BTreeMap<Guid128, String> =
        nodes.iter().map(|(_, g, d)| (*g, d.clone())).collect();
    let (members, dangling) = members(v);
    out.dangling_members = dangling;
    // Member DNs per group, built once: O(memberships), not O(groups × memberships).
    let mut by_group: std::collections::BTreeMap<Guid128, Vec<&String>> = Default::default();
    for (u, grp) in &members {
        if let Some(d) = dn_of.get(u) {
            by_group.entry(*grp).or_default().push(d);
        }
    }

    let mut objs: Vec<(NodeKind, String, Entry)> = Vec::with_capacity(nodes.len());
    for (kind, g, dn) in nodes {
        let s = v.node_state(&g).expect("existing node");
        let origin = origin_of(&g);
        let mut a: Vec<(&'static str, Value)> = Vec::new();
        let classes: &[&str] = match kind {
            NodeKind::User => &["top", "person", "organizationalPerson", "user"],
            NodeKind::Group => &["top", "group"],
        };
        for c in classes {
            a.push(("objectClass", Value::Text((*c).into())));
        }
        a.push(("cn", Value::Text(g.to_string())));
        a.push(("objectGUID", Value::Binary(g.to_ms_bytes().to_vec())));
        if kind == NodeKind::User {
            if let Some(u) = s.upn.and_then(text) {
                a.push(("userPrincipalName", Value::Text(u)));
            }
            if let Some(m) = v.mail(&g).and_then(text) {
                a.push(("mail", Value::Text(m)));
            }
        }
        let primary = s.primary_smtp.and_then(text);
        if let Some(p) = &primary {
            a.push(("proxyAddresses", Value::Text(format!("SMTP:{p}"))));
        }
        if kind == NodeKind::User {
            for p in observed_secondaries(v, &g, s.primary_smtp) {
                a.push(("proxyAddresses", Value::Text(p)));
            }
            if let Some(flag) = s.active {
                let uac = if flag { "512" } else { "514" };
                a.push(("userAccountControl", Value::Text(uac.into())));
            }
            if let Some(x) = v.exchange_guid(&g) {
                a.push(("msExchMailboxGuid", Value::Binary(x.to_ms_bytes().to_vec())));
            }
        } else {
            let mut m = by_group.remove(&g).unwrap_or_default();
            m.sort();
            for d in m {
                a.push(("member", Value::Text(d.clone())));
            }
        }
        a.push(("dirSimOrigin", Value::Text(origin.as_str().into())));
        objs.push((
            kind,
            dn.clone(),
            Entry {
                dn,
                origin,
                node: Some(g),
                attrs: a,
            },
        ));
    }
    objs.sort_by(|a, b| {
        let k = |k: NodeKind| matches!(k, NodeKind::Group);
        (k(a.0), &a.1).cmp(&(k(b.0), &b.1))
    });
    out.entries.extend(objs.into_iter().map(|(_, _, e)| e));
    Ok(out)
}

/// `leaf,OU=...,<nc>` with the path root-first.
fn render(leaf: &str, path: &[String], nc: &str) -> String {
    let mut s = leaf.to_string();
    for ou in path.iter().rev() {
        s.push_str(",OU=");
        s.push_str(&escape_rdn_value(ou));
    }
    s.push(',');
    s.push_str(nc);
    s
}

/// The observed non-primary proxies of a base user, as written, minus any
/// whose address is the effective primary.
fn observed_secondaries(
    v: &View<'_>,
    g: &Guid128,
    primary: Option<ogar_dir_sim::ValueId>,
) -> Vec<String> {
    let snap = v.snapshot();
    let Some(o) = snap.user_ordinal(g) else {
        return Vec::new();
    };
    let primary_key = primary.and_then(|p| v.dicts().key_of(p));
    let rel = snap.proxies();
    let mut out = Vec::new();
    for i in rel.owner_rows(u32::from(o.0)) {
        let r = rel.row(i);
        if r.primary || Some(r.key) == primary_key {
            continue;
        }
        let Some(addr) = v.dicts().value(r.value) else {
            continue;
        };
        let prefix = match r.kind {
            crate::ProxyKind::Smtp => "smtp:",
            crate::ProxyKind::X500 => "X500:",
            crate::ProxyKind::Sip => "SIP:",
            crate::ProxyKind::Other => "",
        };
        out.push(format!("{prefix}{addr}"));
    }
    out
}

/// The live `(user, group)` memberships of a version, by identity; and the
/// count of observed or added memberships naming a node this version does
/// not hold.
fn members(v: &View<'_>) -> (Vec<(Guid128, Guid128)>, usize) {
    let snap = v.snapshot();
    let live = v.live_rows();
    let mut out = Vec::new();
    for r in 0..snap.membership_rows() {
        if live[r / 64] & (1 << (r % 64)) == 0 {
            continue;
        }
        // A node cannot be deleted while it holds a membership
        // (`ApplyError::NodeHasMemberships`), so both ends of a live row exist.
        out.push((
            snap.users().ids[snap.m_user[r] as usize],
            snap.groups().ids[snap.m_group[r] as usize],
        ));
    }
    let added = v.added_rows();
    // Resolved against this view: both ordinals name existing nodes.
    for (u, g) in added.users.iter().zip(&added.groups) {
        if let (Some(u), Some(g)) = (
            v.guid_in(NodeKind::User, *u as usize),
            v.guid_in(NodeKind::Group, *g as usize),
        ) {
            out.push((u, g));
        }
    }
    let dangling = added.unresolved.len();
    out.sort_unstable();
    out.dedup();
    (out, dangling)
}

// ── LDIF (RFC 2849) ──────────────────────────────────────────────────────

/// The projection as LDIF: `version: 1`, then one record per entry.
/// Values that are not SAFE-STRINGs are base64 (`::`); lines are folded at
/// 76 columns.
pub fn to_ldif(p: &Projection) -> String {
    let mut out = String::from("version: 1\n");
    for e in &p.entries {
        out.push('\n');
        line(&mut out, "dn", e.dn.as_bytes(), false);
        for (name, v) in &e.attrs {
            match v {
                Value::Text(t) => line(&mut out, name, t.as_bytes(), false),
                Value::Binary(b) => line(&mut out, name, b, true),
            }
        }
    }
    out
}

fn safe(v: &[u8]) -> bool {
    let first_ok = !matches!(v.first(), Some(b' ' | b':' | b'<'));
    let last_ok = v.last() != Some(&b' ');
    first_ok
        && last_ok
        && v.iter()
            .all(|&b| b != 0 && b != b'\n' && b != b'\r' && b < 0x80)
}

fn line(out: &mut String, name: &str, v: &[u8], binary: bool) {
    let l = if !binary && safe(v) {
        format!("{name}: {}", std::str::from_utf8(v).expect("ascii"))
    } else {
        format!("{name}:: {}", base64(v))
    };
    fold(out, &l);
}

fn fold(out: &mut String, l: &str) {
    let mut rest = l;
    let mut width = 76;
    loop {
        if rest.len() <= width {
            out.push_str(rest);
            out.push('\n');
            return;
        }
        let mut cut = width;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push_str(&rest[..cut]);
        out.push_str("\n ");
        rest = &rest[cut..];
        width = 75;
    }
}

/// Standard base64 with padding.
pub fn base64(b: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i <= c.len() {
                s.push(A[(n >> shift & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}
