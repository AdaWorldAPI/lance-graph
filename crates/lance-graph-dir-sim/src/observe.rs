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
//! The Exchange recipient triplet (`msExchRemoteRecipientType`,
//! `msExchRecipientDisplayType`, `msExchRecipientTypeDetails`) and
//! `targetAddress` are read raw into an [`ObservedRecipient`]; the `SMTP:`
//! prefix of `targetAddress` is stripped here (in) and belongs to egress
//! (out). A record encoded with schema version 2 or later has read them —
//! absent attributes mean "not mail-enabled"; an older record has not, and
//! its recipient is unknown.
//!
//! How an AD and an Entra observation of the same person combine is
//! `ogar_dir_sim::effective_active` (V4); no path merges the two sources yet,
//! so a node carries the flag its one source reported.
//!
//! [`from_graph`] does the same for Entra users read through Microsoft Graph
//! (`ogar-az` records), so the directory can be shown as an Active Directory
//! emulation even where only the cloud was read. A synchronized user is
//! **mirrored**: its location is the OU path of `onPremisesDistinguishedName`,
//! interned into the on-premises domain's dictionary, exactly where the AD
//! record of the same object sits. A cloud-only user has no on-premises
//! location and is placed **synthetically** under one marked container,
//! [`SYNTHETIC_OU`], and reported as such ([`GraphObservation::synthetic`]).

use crate::snapshot::{NodeKind, Observation, ObservedNode, ObservedRecipient};
use ogar_ad::{AdKind, SCHEMA_V1};
use ogar_dir_core::{
    DirRecord, DirectoryScope, Dn128, Dn128Error, Guid128, HhtlError, OuDictionary, SchemaFamily,
    ValuePool,
};

const UAC_ACCOUNTDISABLE: u32 = 0x2;
/// The first `ogar-ad` schema version that carries the recipient triplet.
const RECIPIENT_SCHEMA: u16 = 2;
/// The first `ogar-ad` schema version that carries `msExchMailboxGuid`.
const EXCHANGE_GUID_SCHEMA: u16 = 6;

/// `targetAddress` without its `SMTP:` prefix (any case); other address
/// types are kept whole.
fn strip_smtp(s: &str) -> &str {
    match s.split_once(':') {
        Some((ns, addr)) if ns.eq_ignore_ascii_case("smtp") => addr,
        _ => s,
    }
}

fn slot(name: &str) -> usize {
    SCHEMA_V1
        .iter()
        .find(|d| d.name == name)
        .map(|d| d.slot as usize)
        .expect("ogar-ad schema v1")
}

/// Why a record could not be observed. Either way the whole observation is
/// refused: nothing is silently dropped or merged.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// The synthetic container cannot be added to the dictionary.
    Synthetic(HhtlError),
    /// A mirrored user's on-premises OU is the synthetic container or lies
    /// under it, so mirrored and synthetic placements would be
    /// indistinguishable.
    SyntheticClash {
        /// The mirrored user.
        node: Guid128,
    },
    /// A user reports `onPremisesDistinguishedName`, but its OU path was not
    /// encoded at ingest. It is synchronized, so it is not placed
    /// synthetically.
    UnplacedMirror {
        /// The user.
        node: Guid128,
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
                mail: text(r, "mail"),
                alias: text(r, "mailNickname"),
                exchange_guid: (r.schema().version >= EXCHANGE_GUID_SCHEMA)
                    .then(|| r.guid(slot("msExchMailboxGuid")))
                    .flatten(),
                recipient: (r.schema().version >= RECIPIENT_SCHEMA).then(|| ObservedRecipient {
                    remote_recipient_type: r.num(slot("msExchRemoteRecipientType")),
                    display_type: r.num(slot("msExchRecipientDisplayType")).map(|n| n as i32),
                    type_details: text(r, "msExchRecipientTypeDetails")
                        .and_then(|t| t.trim().parse().ok()),
                    target_address: text(r, "targetAddress").map(|t| strip_smtp(&t).to_string()),
                }),
            },
        ));
    }
    Ok(obs)
}

/// The container cloud-only users are placed under, as one OU level at the
/// root of the emulated tree. It is marked so it cannot be mistaken for an
/// on-premises OU; a mirrored user that sits in or under an OU of this name
/// refuses the observation ([`ObserveError::SyntheticClash`]).
pub const SYNTHETIC_OU: &str = "Cloud Only (emulated)";

/// Entra users as an AD emulation: the observation, plus which users were
/// placed synthetically.
#[derive(Clone, Debug, Default)]
pub struct GraphObservation {
    /// The users, located: mirrored where `onPremisesDistinguishedName` was
    /// read, synthetic otherwise.
    pub observation: Observation,
    /// The users with no on-premises location, placed under
    /// [`SYNTHETIC_OU`], sorted.
    pub synthetic: Vec<Guid128>,
}

/// Graph `user` records (`ogar-az`) in `tenant`, as an observation. Other
/// record kinds are skipped. `dict` must be the dictionary the records were
/// ingested with (the on-premises domain's, so a mirrored user's OU equals
/// its AD object's); the synthetic container is added to it.
///
/// Graph reports no Exchange recipient attributes and no
/// `msExchMailboxGuid`, so `recipient` is not read (`None`) and
/// `exchange_guid` is `None`. `accountEnabled` gives the flag; absent is
/// unknown, never enabled.
pub fn from_graph(
    tenant: DirectoryScope,
    records: &[DirRecord],
    pool: &ValuePool,
    dict: &mut OuDictionary,
) -> Result<GraphObservation, ObserveError> {
    let enabled_slot = ogar_az::SCHEMA_V1
        .iter()
        .find(|d| d.name == "accountEnabled")
        .map(|d| d.slot as usize)
        .expect("ogar-az schema v1 has accountEnabled");
    let text = |r: &DirRecord, name: &str| ogar_az::attr_str(r, pool, name).map(str::to_string);
    let synthetic_ou = dict
        .intern(&[SYNTHETIC_OU])
        .map_err(ObserveError::Synthetic)?;
    let synthetic_dn =
        Dn128::from_ou_hhtl(&synthetic_ou).map_err(|error| ObserveError::Location {
            node: Guid128::NIL,
            error,
        })?;
    let mut out = GraphObservation {
        observation: Observation {
            scope: tenant,
            ..Observation::default()
        },
        synthetic: Vec::new(),
    };
    for r in records.iter().filter(|r| {
        r.schema().family == SchemaFamily::MsGraph
            && r.object_kind() == ogar_az::AzKind::User as u16
    }) {
        let node = r.node_guid();
        if r.scope_guid() != tenant.0 {
            return Err(ObserveError::ForeignScope {
                node,
                scope: DirectoryScope(r.scope_guid()),
            });
        }
        let hybrid = ogar_az::attr_str(r, pool, "onPremisesDistinguishedName").is_some();
        let dn = match (hybrid, r.ou_hhtl()) {
            (true, Some(h)) => {
                if synthetic_ou.is_ancestor_of(&h) {
                    return Err(ObserveError::SyntheticClash { node });
                }
                Dn128::from_ou_hhtl(&h).map_err(|error| ObserveError::Location { node, error })?
            }
            // The on-premises DN was read but could not be placed: never
            // demote a synchronized user to the synthetic container.
            (true, None) => return Err(ObserveError::UnplacedMirror { node }),
            (false, _) => {
                out.synthetic.push(node);
                synthetic_dn
            }
        };
        let mut proxies: Vec<String> = ogar_az::attr_multi(r, pool, "proxyAddresses")
            .unwrap_or_default()
            .into_iter()
            .map(str::to_string)
            .collect();
        let primary_smtp = proxies
            .iter()
            .position(|v| v.starts_with("SMTP:"))
            .map(|i| proxies.remove(i)["SMTP:".len()..].to_string());
        out.observation.nodes.push((
            node,
            ObservedNode {
                kind: NodeKind::User,
                active: r.num(enabled_slot).map(|b| b != 0),
                upn: text(r, "userPrincipalName"),
                primary_smtp,
                proxies,
                dn: Some(dn),
                recipient: None,
                mail: text(r, "mail"),
                alias: text(r, "mailNickname"),
                exchange_guid: None,
            },
        ));
    }
    out.synthetic.sort_unstable();
    Ok(out)
}
