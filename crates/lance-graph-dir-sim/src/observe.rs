//! `ogar-ad` records (OGAR PR #313) → an [`Observation`].
//!
//! The ingestion boundary: values are read out of the record's value pool
//! once and handed to [`Snapshot::build`](crate::Snapshot::build) for
//! interning. "Active" is derived from `userAccountControl` bit `0x2`
//! (ACCOUNTDISABLE); the primary SMTP is the `SMTP:` proxy; the location is
//! the record's `OuHhtl` (the ingress wire format) converted to a [`Dn128`],
//! never a DN string. A parent with more than 256 children cannot be a
//! `Dn128` and the whole observation is refused — never hashed or truncated.
//! Memberships are relations that `ogar-ad` records do not carry; the
//! caller adds observed ones.
//!
//! **Open, recorded, not decided here:** a user record with no
//! `userAccountControl` value is read as **enabled** (`is_none_or`). Whether
//! an absent flag should mean enabled, disabled or "unknown" — and how an AD
//! and an Entra observation of the same person are merged — is an open
//! policy question; this module does not choose for it.

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

/// Why a record could not be observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocationError {
    /// The record.
    pub node: Guid128,
    /// Why its `OuHhtl` is not a `Dn128`.
    pub error: Dn128Error,
}

/// Users and groups among `records` (other kinds are skipped), located in
/// `scope`.
pub fn from_ad(
    scope: DirectoryScope,
    records: &[DirRecord],
    pool: &ValuePool,
) -> Result<Observation, LocationError> {
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
        let primary_smtp = r
            .str_ref(slot("proxyAddresses"))
            .and_then(|s| pool.get_multi(s))
            .and_then(|vs| {
                vs.into_iter()
                    .filter_map(|v| std::str::from_utf8(v).ok())
                    .find_map(|v| v.strip_prefix("SMTP:").map(str::to_string))
            });
        let node = r.node_guid();
        let dn = r
            .ou_hhtl()
            .map(|h| Dn128::from_ou_hhtl(&h))
            .transpose()
            .map_err(|error| LocationError { node, error })?;
        obs.nodes.push((
            node,
            ObservedNode {
                kind,
                active: r.num(0).is_none_or(|uac| uac & UAC_ACCOUNTDISABLE == 0),
                upn: text(r, "userPrincipalName"),
                primary_smtp,
                dn,
            },
        ));
    }
    Ok(obs)
}
