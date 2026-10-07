//! `ogar-ad` records (OGAR PR #313) → an [`Observation`].
//!
//! The ingestion boundary: values are read out of the record's value pool
//! once and handed to [`Snapshot::build`](crate::Snapshot::build) for
//! interning. "Active" is derived from `userAccountControl` bit `0x2`
//! (ACCOUNTDISABLE), and a record without the attribute is **unknown**, never
//! enabled; the primary SMTP is the `SMTP:` proxy and every other proxy is
//! kept raw for the [`ProxyRelation`](crate::ProxyRelation); the location is
//! the record's `OuHhtl` (the ingress wire format) converted to a [`Dn128`],
//! never a DN string. A parent with more than 256 children cannot be a
//! `Dn128` and the whole observation is refused — never hashed or truncated.
//! Memberships are relations that `ogar-ad` records do not carry; the
//! caller adds observed ones.
//!
//! How an AD and an Entra observation of the same person combine is
//! `ogar_dir_sim::effective_active` (V4); no path merges the two sources yet,
//! so a node carries the flag its one source reported.

use crate::snapshot::{NodeKind, Observation, ObservedNode};
use ogar_ad::{AdKind, SCHEMA_V1};
use ogar_dir_core::{DirRecord, DirectoryScope, Dn128, Dn128Error, Guid128, ValuePool};

const UAC_ACCOUNTDISABLE: u32 = 0x2;

fn slot(name: &str) -> usize {
    SCHEMA_V1
        .iter()
        .find(|d| d.name == name)
        .map(|d| d.slot as usize)
        .expect("ogar-ad schema v1")
}

/// Why a record could not be observed. Either way the whole observation is
/// refused: nothing is silently dropped or merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserveError {
    /// The record's `OuHhtl` is not a `Dn128` (e.g. a 257th child).
    Location {
        /// The record.
        node: Guid128,
        /// Why.
        error: Dn128Error,
    },
    /// The record belongs to another directory than the observation. A
    /// `Dn128` carries no scope (it is external context), so a record from
    /// another domain or tenant would be indistinguishable from a local one.
    ForeignScope {
        /// The record.
        node: Guid128,
        /// The record's own scope.
        scope: DirectoryScope,
    },
}

/// Users and groups among `records` (other kinds are skipped), located in
/// `scope`. Every user or group record must carry that scope.
pub fn from_ad(
    scope: DirectoryScope,
    records: &[DirRecord],
    pool: &ValuePool,
) -> Result<Observation, ObserveError> {
    let text = |r: &DirRecord, name: &str| {
        r.str_ref(slot(name))
            .and_then(|s| pool.get(s))
            .and_then(|b| std::str::from_utf8(b).ok())
            .map(str::to_string)
    };
    let mut obs = Observation {
        scope,
        ..Observation::default()
    };
    for r in records {
        let kind = match r.object_kind() {
            k if k == AdKind::User as u16 => NodeKind::User,
            k if k == AdKind::Group as u16 => NodeKind::Group,
            _ => continue,
        };
        let mut proxies: Vec<String> = r
            .str_ref(slot("proxyAddresses"))
            .and_then(|s| pool.get_multi(s))
            .map(|vs| {
                vs.into_iter()
                    .filter_map(|v| std::str::from_utf8(v).ok())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        // The first upper-case `SMTP:` is the primary attribute; every other
        // value stays a proxy row, raw.
        let primary_smtp = proxies
            .iter()
            .position(|v| v.starts_with("SMTP:"))
            .map(|i| proxies.remove(i)["SMTP:".len()..].to_string());
        let node = r.node_guid();
        if r.scope_guid() != scope.0 {
            return Err(ObserveError::ForeignScope {
                node,
                scope: DirectoryScope(r.scope_guid()),
            });
        }
        let dn = r
            .ou_hhtl()
            .map(|h| Dn128::from_ou_hhtl(&h))
            .transpose()
            .map_err(|error| ObserveError::Location { node, error })?;
        obs.nodes.push((
            node,
            ObservedNode {
                kind,
                active: r.num(0).map(|uac| uac & UAC_ACCOUNTDISABLE == 0),
                upn: text(r, "userPrincipalName"),
                primary_smtp,
                proxies,
                dn,
            },
        ));
    }
    Ok(obs)
}
