//! `ogar-ad` records (OGAR PR #313) → an [`Observation`].
//!
//! The ingestion boundary: values are read out of the record's value pool
//! once and handed to [`Snapshot::build`](crate::Snapshot::build) for
//! interning. "Active" is derived from `userAccountControl` bit `0x2`
//! (ACCOUNTDISABLE); the primary SMTP is the `SMTP:` proxy; the OU-HHTL is
//! taken as-is (never a DN string). Memberships are relations that `ogar-ad`
//! records do not carry; the caller adds observed ones.

use crate::snapshot::{NodeKind, Observation, ObservedNode};
use ogar_ad::{AdKind, SCHEMA_V1};
use ogar_dir_core::{DirRecord, ValuePool};

const UAC_ACCOUNTDISABLE: u32 = 0x2;

fn slot(name: &str) -> usize {
    SCHEMA_V1
        .iter()
        .find(|d| d.name == name)
        .map(|d| d.slot as usize)
        .expect("ogar-ad schema v1")
}

/// Users and groups among `records` (other kinds are skipped).
pub fn from_ad(records: &[DirRecord], pool: &ValuePool) -> Observation {
    let text = |r: &DirRecord, name: &str| {
        r.str_ref(slot(name))
            .and_then(|s| pool.get(s))
            .and_then(|b| std::str::from_utf8(b).ok())
            .map(str::to_string)
    };
    let mut obs = Observation::default();
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
        obs.nodes.push((
            r.node_guid(),
            ObservedNode {
                kind,
                active: r.num(0).is_none_or(|uac| uac & UAC_ACCOUNTDISABLE == 0),
                upn: text(r, "userPrincipalName"),
                primary_smtp,
                ou: r.ou_hhtl(),
            },
        ));
    }
    obs
}
