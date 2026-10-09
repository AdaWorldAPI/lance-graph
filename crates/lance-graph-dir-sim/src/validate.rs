//! Invariants as Quack programs over the version's lanes.
//!
//! | invariant            | relational form                                         | lowering                               |
//! |----------------------|---------------------------------------------------------|----------------------------------------|
//! | edge integrity       | `members ANTI JOIN users ∪ members ANTI JOIN groups`    | `Rows(¬Semijoin(user,·) ∨ ¬Semijoin(group,·))` — `MaskOp::Gather` over the live user and group planes |
//! | unique SMTP          | `GROUP BY key HAVING count(DISTINCT owner) > 1` over the active users' SMTP proxies | `GroupReduce Count` keyed on [`KeyId`] over one row per `(owner, key)` |
//! | unique UPN           | `GROUP BY key HAVING count > 1` over active users        | `GroupReduce Count` keyed on [`KeyId`]  |
//!
//! No user objects are built and no string is read: the integrity check
//! reads the two membership lanes and the two live-node planes, uniqueness
//! one key lane and one plane. A violation carries identities and the
//! [`KeyId`]; resolving a key to text is the reporter's job.
//!
//! Memberships whose endpoint never resolved to an ordinal are not in the
//! lanes (no sentinel ordinal exists); they are reported from the snapshot's
//! unresolved table and the overlay, both evidence-sized. A group nested in a
//! live group is held the same way but is a valid membership, not dangling.
//!
//! Materialisations, all at the evidence boundary and bounded by the number
//! of violations: the offending membership rows (`materialize_rows` of the
//! kept mask) and each duplicate key's owner rows.

use crate::cloud::CloudMailboxes;
use crate::exec::{group_count_into, keep, program};
use crate::proxy::meta;
use crate::snapshot::{bit, clear_bit, ones, NONE};
use crate::view::View;
use lance_graph_mask_risc::{Foreign, ForeignPlane as FPlane, LaneRef, Planes, Program};
use lance_graph_quack::{Agg, Cmp, Col, Filter, ForeignPlane, Mask};
use ogar_dir_core::Guid128;
use ogar_dir_sim::{AddressRole, Attribute, Endpoint, KeyId, NodeKind, Recipient, Violation};

/// The edge-integrity program over a membership relation: lane 0 = user
/// ordinal, lane 1 = group ordinal, plane 0 = live rows; foreign plane 0 =
/// the live user plane, 1 = the live group plane. An anti-join on each side.
pub fn dangling_program() -> Program {
    program(
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::or([
                Filter::negate(Filter::semijoin(Col(0), ForeignPlane(0))),
                Filter::negate(Filter::semijoin(Col(1), ForeignPlane(1))),
            ]),
        ]),
        Agg::Rows,
    )
}

/// Dangling memberships of a version.
///
/// Resolved rows (base still live, overlay added): one anti-join program
/// against the version's live user and group planes — the snapshot's, minus
/// deleted nodes, plus created ones, composed from resident planes and the
/// overlay, never taken from a program's output. Unresolved rows: reported
/// directly.
pub fn dangling(v: &View<'_>) -> Vec<Violation> {
    let s = v.snap;
    let (users, groups) = (v.existing(NodeKind::User), v.existing(NodeKind::Group));
    let fps = [
        FPlane {
            words: &users,
            rows: v.users_len(),
        },
        FPlane {
            words: &groups,
            rows: v.groups_len(),
        },
    ];
    let foreign = Foreign {
        planes: &fps,
        lanes: &[],
    };
    let p = dangling_program();
    let side = |uo: u32| {
        if bit(&users, uo as usize) {
            Endpoint::Group
        } else {
            Endpoint::User
        }
    };
    let mut out = Vec::new();

    let live = v.live_rows();
    let lanes = [LaneRef::U32(&s.m_user), LaneRef::U32(&s.m_group)];
    let masks: [&[u64]; 1] = [&live];
    let planes = Planes {
        n_rows: s.membership_rows(),
        masks: &masks,
        lanes: &lanes,
    };
    for r in keep(&p, &planes, &foreign).rows() {
        let (user, group) = s.member_guids(r as u32);
        out.push(Violation::DanglingMembership {
            user,
            group,
            missing: side(s.m_user[r]),
        });
    }

    // Identity-held rows (overlay adds, observed unresolved pairs) that
    // resolve in this version: delta-sized lanes, same program.
    let added = v.added_rows();
    let all = ones(added.users.len());
    let lanes = [LaneRef::U32(&added.users), LaneRef::U32(&added.groups)];
    let masks: [&[u64]; 1] = [&all];
    let planes = Planes {
        n_rows: added.users.len(),
        masks: &masks,
        lanes: &lanes,
    };
    for r in keep(&p, &planes, &foreign).rows() {
        let user = v.guid_in(NodeKind::User, added.users[r] as usize);
        let group = v.guid_in(NodeKind::Group, added.groups[r] as usize);
        if let (Some(user), Some(group)) = (user, group) {
            out.push(Violation::DanglingMembership {
                user,
                group,
                missing: side(added.users[r]),
            });
        }
    }

    // Identity-held rows that still do not resolve as `(user, group)` in
    // this version. A group nested in a group is held here too (the lanes are
    // user × group ordinals) and is not dangling while both groups exist.
    for &(user, group) in &added.unresolved {
        let member_is_group = v.group_ordinal(&user).is_some();
        let group_exists = v.group_ordinal(&group).is_some();
        if member_is_group && group_exists {
            continue;
        }
        let missing = if v.user_ordinal(&user).is_some() || member_is_group {
            Endpoint::Group
        } else {
            Endpoint::User
        };
        out.push(Violation::DanglingMembership {
            user,
            group,
            missing,
        });
    }
    out.sort();
    out
}

/// Address owners (enabled, or a live mail recipient) sharing a comparison key of `a`, read from the attribute's
/// key lane alone. For [`Attribute::PrimarySmtp`] that is the primary
/// address only; [`validate`] uses [`smtp_duplicates`], which also covers
/// the secondary SMTP proxies.
pub fn duplicates(v: &View<'_>, a: Attribute) -> Vec<Violation> {
    let s = v.snap;
    let u = &s.users;
    let k = v.dicts.key_count();
    if k == 0 {
        return Vec::new();
    }
    let base_key = match a {
        Attribute::Upn => &u.upn_key,
        Attribute::PrimarySmtp => &u.smtp_key,
    };
    let owners_plane = v.live_owners(a);
    let mut counts = vec![0i64; k];

    // Base: GROUP BY key COUNT(*) over the users still owning their observed value.
    let lanes = [LaneRef::U32(base_key)];
    let masks: [&[u64]; 1] = [&owners_plane];
    let base = Planes {
        n_rows: u.len(),
        masks: &masks,
        lanes: &lanes,
    };
    group_count_into(
        Filter::plane(Mask(0)),
        Col(0),
        &base,
        &Foreign::NONE,
        &mut counts,
    );

    // Delta rows — overrides of base users and created users, as one small
    // relation (ordinal, key) — through the same GROUP BY, admitted by a
    // semijoin of the ordinal against the version's active-user plane.
    let n = u.len() as u32;
    let ou = &v.ov.users;
    let (oo, ok): (Vec<u32>, Vec<u32>) = ou
        .overrides(a)
        .iter()
        .map(|(o, (_, key))| (u32::from(*o), *key))
        .chain(
            ou.created
                .key(a)
                .iter()
                .enumerate()
                .map(|(i, key)| (n + i as u32, *key)),
        )
        .unzip();
    // Address owners: enabled, or a live mail recipient.
    let active = v.owner_users();
    let all = ones(oo.len());
    let fps = [FPlane {
        words: &active,
        rows: v.users_len(),
    }];
    let foreign = Foreign {
        planes: &fps,
        lanes: &[],
    };
    let lanes = [LaneRef::U32(&oo), LaneRef::U32(&ok)];
    let masks: [&[u64]; 1] = [&all];
    let delta = Planes {
        n_rows: oo.len(),
        masks: &masks,
        lanes: &lanes,
    };
    group_count_into(
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::semijoin(Col(0), ForeignPlane(0)),
        ]),
        Col(1),
        &delta,
        &foreign,
        &mut counts,
    );

    let mut out = Vec::new();
    for (key, _) in counts.iter().enumerate().filter(|(_, c)| **c > 1) {
        let key = key as u32;
        // Owners: one gated equality over the base key lane (evidence boundary).
        let p = program(
            Filter::and([Filter::plane(Mask(0)), Filter::cmp(Col(0), Cmp::EqU32(key))]),
            Agg::Rows,
        );
        let mut owners: Vec<Guid128> = keep(&p, &base, &Foreign::NONE)
            .rows()
            .into_iter()
            .map(|o| u.ids[o])
            .collect();
        owners.extend(
            oo.iter()
                .zip(&ok)
                .filter(|(o, kk)| **kk == key && bit(&active, **o as usize))
                .filter_map(|(o, _)| v.guid_in(NodeKind::User, *o as usize)),
        );
        owners.sort();
        let key = KeyId(key);
        out.push(match a {
            Attribute::Upn => Violation::DuplicateUpn { key, owners },
            Attribute::PrimarySmtp => Violation::DuplicateSmtp { key, owners },
        });
    }
    out
}

/// The SMTP uniqueness program over the proxy relation: plane 0 = the first
/// SMTP row of each `(owner, key)` run; lane 0 = owner, lane 1 = key, lane 2
/// = meta; foreign plane 0 = owners whose secondaries count (active, live),
/// foreign plane 1 = owners whose observed primary still stands (the same,
/// minus primary-SMTP overrides). Keyed on lane 1.
fn smtp_filter() -> Filter {
    Filter::and([
        Filter::plane(Mask(0)),
        Filter::or([
            Filter::and([
                Filter::cmp(Col(2), Cmp::EqU32(meta::SMTP_SECONDARY)),
                Filter::semijoin(Col(0), ForeignPlane(0)),
            ]),
            Filter::and([
                Filter::cmp(Col(2), Cmp::EqU32(meta::SMTP_PRIMARY)),
                Filter::semijoin(Col(0), ForeignPlane(1)),
            ]),
        ]),
    ])
}

/// Address owners — enabled users and live mail recipients, so a disabled
/// shared, room or equipment mailbox counts — sharing a normalized SMTP address — primary or secondary,
/// any prefix case. Counts **distinct owners** per [`KeyId`]: one user
/// holding an address twice is one owner.
///
/// Base: one `GroupReduce Count` over the proxy relation's `smtp_first`
/// rows (one per owner and address). A primary-SMTP override replaces the
/// owner's observed primary row (dropped through foreign plane 1) and keeps
/// its secondaries. Delta: override and created addresses, through the same
/// GROUP BY, minus an override its owner already holds as a secondary (that
/// owner is already counted).
pub fn smtp_duplicates(v: &View<'_>) -> Vec<Violation> {
    let rel = &v.snap.proxies;
    let k = v.dicts.key_count();
    if k == 0 {
        return Vec::new();
    }
    let mut counts = vec![0i64; k];

    // Address owners: enabled, or a live mail recipient.
    let active = v.owner_users();
    let ou = &v.ov.users;
    let overrides = ou.overrides(Attribute::PrimarySmtp);
    let mut primary_ok = active.to_vec();
    for o in overrides.keys() {
        clear_bit(&mut primary_ok, usize::from(*o));
    }
    let fps = [
        FPlane {
            words: &active,
            rows: v.users_len(),
        },
        FPlane {
            words: &primary_ok,
            rows: v.users_len(),
        },
    ];
    let foreign = Foreign {
        planes: &fps,
        lanes: &[],
    };
    let lanes = [
        LaneRef::U32(&rel.owner),
        LaneRef::U32(&rel.key),
        LaneRef::U32(&rel.meta),
    ];
    let masks: [&[u64]; 1] = [&rel.smtp_first];
    let base = Planes {
        n_rows: rel.len(),
        masks: &masks,
        lanes: &lanes,
    };
    group_count_into(smtp_filter(), Col(1), &base, &foreign, &mut counts);

    let n = v.snap.users.len() as u32;
    let (oo, ok): (Vec<u32>, Vec<u32>) = overrides
        .iter()
        .map(|(o, (_, key))| (u32::from(*o), *key))
        .filter(|&(o, key)| key != NONE && !rel.has_secondary_smtp(o, key))
        .chain(
            ou.created
                .key(Attribute::PrimarySmtp)
                .iter()
                .enumerate()
                .map(|(i, key)| (n + i as u32, *key))
                .filter(|&(_, key)| key != NONE),
        )
        .unzip();
    let all = ones(oo.len());
    let dl = [LaneRef::U32(&oo), LaneRef::U32(&ok)];
    let dm: [&[u64]; 1] = [&all];
    let delta = Planes {
        n_rows: oo.len(),
        masks: &dm,
        lanes: &dl,
    };
    group_count_into(
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::semijoin(Col(0), ForeignPlane(0)),
        ]),
        Col(1),
        &delta,
        &foreign,
        &mut counts,
    );

    let mut out = Vec::new();
    for (key, _) in counts.iter().enumerate().filter(|(_, c)| **c > 1) {
        let key = key as u32;
        // Owners: the same program gated to one key (evidence boundary).
        let p = program(
            Filter::and([smtp_filter(), Filter::cmp(Col(1), Cmp::EqU32(key))]),
            Agg::Rows,
        );
        let mut owners: Vec<Guid128> = keep(&p, &base, &foreign)
            .rows()
            .into_iter()
            .map(|r| v.snap.users.ids[rel.owner[r] as usize])
            .collect();
        owners.extend(
            oo.iter()
                .zip(&ok)
                .filter(|(o, kk)| **kk == key && bit(&active, **o as usize))
                .filter_map(|(o, _)| v.guid_in(NodeKind::User, *o as usize)),
        );
        owners.sort();
        owners.dedup();
        out.push(Violation::DuplicateSmtp {
            key: KeyId(key),
            owners,
        });
    }
    out
}

/// The one-address-space and routing rules (OGAR `AddressConflict`,
/// `RoutingNotInProxies`, `RoutingMismatch`), over the users and groups of a
/// version.
///
/// Rows `(key, holder, role)`: UPN, primary and secondary SMTP and the
/// routing address of address owners (enabled, or a live mail recipient),
/// and the primary SMTP address of every live group. `mail` is not a row:
/// it is a label, and Exchange enforces uniqueness on proxy addresses (and
/// the UPN namespace), never on `mail`. A key is an `AddressConflict` when two of its holders
/// collide across attributes: a pair of users that already collides as SMTP
/// (`DuplicateSmtp`) or as UPN (`DuplicateUpn`) is not reported twice. A
/// collision with a group is always an `AddressConflict`: `DuplicateSmtp`
/// counts users only.
///
/// A live remote mailbox's routing address must be
/// `{alias}@{tenant}.mail.onmicrosoft.com` for its own `mailNickname`
/// (`RoutingMismatch`; not checked while the alias is unknown) and one of its
/// own SMTP addresses (`RoutingNotInProxies`). The proxy rule applies only to
/// the routing address the node was observed with: a routing address this
/// version introduces is stamped as a proxy by the lifecycle operation
/// (Enable-RemoteMailbox), so the pre-actuation version is not rejected for
/// lacking it. Both compare ids: the alias was parsed out of the address
/// when it was interned.
///
/// The rows are collected per user and sorted — `O(n log n)` over the users,
/// a validation pass rather than the simulation hot path.
pub fn address_rules(v: &View<'_>) -> Vec<Violation> {
    let mut out = Vec::new();
    let rows = address_rows(v, &mut out);
    for run in rows.chunk_by(|a, b| a.0 == b.0) {
        // Per holder: does it hold the key as SMTP, as UPN?
        let holders: Vec<(Guid128, bool, bool)> = run
            .chunk_by(|a, b| a.1 == b.1)
            .map(|h| {
                let has = |f: fn(AddressRole) -> bool| h.iter().any(|r| f(r.2));
                // `smtp_duplicates` counts users' SMTP proxies only, so a
                // group's SMTP never counts as "already reported there": a
                // collision with a group is an AddressConflict here.
                let user = v.group_ordinal(&h[0].1).is_none();
                (
                    h[0].1,
                    user && has(|r| {
                        matches!(r, AddressRole::PrimarySmtp | AddressRole::SecondarySmtp)
                    }),
                    has(|r| r == AddressRole::Upn),
                )
            })
            .collect();
        // Two holders conflict across attributes unless they already
        // collide as SMTP (DuplicateSmtp) or as UPN (DuplicateUpn).
        let cross = holders.iter().enumerate().any(|(i, a)| {
            holders[i + 1..]
                .iter()
                .any(|b| !(a.1 && b.1) && !(a.2 && b.2))
        });
        if !cross {
            continue;
        }
        out.push(Violation::AddressConflict {
            key: KeyId(run[0].0),
            holders: run.iter().map(|r| (r.1, r.2)).collect(),
        });
    }
    out
}

/// The single directory object that holds `key` in the one address space
/// [`address_rules`] checks, read from the same rows: UPN, primary and
/// secondary SMTP and the routing address of address owners; the primary
/// SMTP address of every live group. `mail` holds nothing.
///
/// - `Ok(None)`: no live directory object holds the key.
/// - `Ok(Some(owner))`: exactly one does, under one or more roles (a user
///   whose UPN is its primary SMTP is one owner).
/// - `Err(Violation::AddressConflict { key, holders })`: two or more do.
///   Every `(holder, role)` is listed, sorted by holder. No holder is
///   chosen: not the first, not by attribute, not by observation order.
///   This is reported for any shared key, including a same-attribute
///   collision that [`validate`] classes as `DuplicateSmtp`/`DuplicateUpn`
///   rather than `AddressConflict`, because either way the key does not
///   name one recipient.
///
/// `key` comes from an ingress lookup (`Dicts::key_lookup` of a recipient
/// address); this function compares ids only. `O(n log n)` over the users,
/// like [`address_rules`].
///
/// # Errors
///
/// `Violation::AddressConflict` when more than one object holds `key`.
pub fn address_owner(v: &View<'_>, key: KeyId) -> Result<Option<Guid128>, Violation> {
    let mut routing = Vec::new();
    let rows = address_rows(v, &mut routing);
    let start = rows.partition_point(|r| r.0 < key.0);
    let end = rows.partition_point(|r| r.0 <= key.0);
    let run = &rows[start..end];
    match run.first() {
        None => Ok(None),
        Some(first) if run.iter().all(|r| r.1 == first.1) => Ok(Some(first.1)),
        Some(_) => Err(Violation::AddressConflict {
            key,
            holders: run.iter().map(|r| (r.1, r.2)).collect(),
        }),
    }
}

/// The directory object mail to `key` is delivered to: the
/// [`address_owner`] of the key, kept only if it is a mail recipient
/// ([`View::is_mail_recipient`]).
///
/// [`address_owner`] answers who *holds* an address — the claim that keeps
/// anyone else from taking it. This answers who *receives* at it, so a
/// holder that is no longer a recipient (an enabled account that is not
/// mail-enabled, say) is `Ok(None)`: the address is reserved and delivers
/// nowhere. A shared key is still `Err`, since it names no single
/// recipient either way.
///
/// # Errors
///
/// `Violation::AddressConflict` when more than one object holds `key`.
pub fn address_recipient(v: &View<'_>, key: KeyId) -> Result<Option<Guid128>, Violation> {
    Ok(address_owner(v, key)?.filter(|g| v.is_mail_recipient(g)))
}

/// [`address_recipient`] with the cloud side observed: a holder whose
/// recipient type is a remote mailbox receives only when the hybrid
/// correspondence fold ties it to exactly one Exchange Online mailbox
/// ([`CloudMailboxes`]). On-premises mailboxes and groups are decided as in
/// [`address_recipient`].
///
/// Hybrid routing sends mail for a remote mailbox to its routing address in
/// the tenant; without a mailbox there, Exchange Online rejects it. The
/// match is on GUIDs only (anchor, backsync and Entra id), never on an
/// address.
///
/// # Errors
///
/// `Violation::AddressConflict` when more than one object holds `key`.
pub fn address_recipient_in(
    v: &View<'_>,
    key: KeyId,
    cloud: &CloudMailboxes,
) -> Result<Option<Guid128>, Violation> {
    Ok(
        address_recipient(v, key)?.filter(|g| match v.node_state(g).and_then(|s| s.recipient) {
            Some(Recipient::RemoteMailbox(_)) => cloud.contains(g),
            _ => true,
        }),
    )
}

/// The `(key, holder, role)` rows of the one address space, sorted and
/// deduplicated, with the routing violations (`RoutingNotInProxies`,
/// `RoutingMismatch`) pushed to `out` on the way.
fn address_rows(v: &View<'_>, out: &mut Vec<Violation>) -> Vec<(u32, Guid128, AddressRole)> {
    let p = &v.snap.users;
    let rel = &v.snap.proxies;
    let owners = v.owner_users();
    let n = p.len();
    let mut rows: Vec<(u32, Guid128, AddressRole)> = Vec::new();
    for i in 0..v.users_len() {
        let Some(g) = v.guid_in(NodeKind::User, i) else {
            continue;
        };
        if !bit(&owners, i) {
            continue;
        }
        let slot = (NodeKind::User, i);
        let mut smtp: Vec<u32> = Vec::new();
        for (a, role) in [
            (Attribute::Upn, AddressRole::Upn),
            (Attribute::PrimarySmtp, AddressRole::PrimarySmtp),
        ] {
            if let Some((_, k)) = v.attr_ids(slot, a).filter(|&(_, k)| k != NONE) {
                rows.push((k, g, role));
                if role == AddressRole::PrimarySmtp {
                    smtp.push(k);
                }
            }
        }
        if i < n {
            for r in rel.owner_rows(i as u32) {
                if rel.meta[r] == meta::SMTP_SECONDARY {
                    rows.push((rel.key[r], g, AddressRole::SecondarySmtp));
                    smtp.push(rel.key[r]);
                }
            }
        }
        let routing = match v.node_state(&g).and_then(|s| s.recipient) {
            Some(Recipient::RemoteMailbox(m)) => m.routing(),
            _ => None,
        };
        let Some(routing) = routing else {
            continue;
        };
        let Some(rk) = v.dicts.key_of(routing) else {
            continue;
        };
        rows.push((rk.0, g, AddressRole::Routing));
        // Only a routing address AD was observed with must already be a
        // proxy. One this version introduces (an enable, a create, a new
        // routing address) is stamped as a proxy by the operation itself, as
        // Enable-RemoteMailbox does; it stays in the address space as
        // `Routing`, so it still conflicts.
        let observed = match (i < n).then(|| p.recipient_of(i)).flatten() {
            Some(Recipient::RemoteMailbox(m)) => m.routing(),
            _ => None,
        };
        if observed == Some(routing) && !smtp.contains(&rk.0) {
            out.push(Violation::RoutingNotInProxies { node: g });
        }
        let alias = if i < n { p.alias_key[i] } else { NONE };
        if alias != NONE && v.dicts.routing_alias(routing) != Some(KeyId(alias)) {
            out.push(Violation::RoutingMismatch { node: g });
        }
    }
    // A mail-enabled group's primary SMTP address is in the same space: a
    // group is a recipient too (a distribution list), and Exchange refuses
    // an address another object holds, whatever kind that object is.
    for i in 0..v.groups_len() {
        let Some(g) = v.guid_in(NodeKind::Group, i) else {
            continue;
        };
        if let Some((_, k)) = v
            .attr_ids((NodeKind::Group, i), Attribute::PrimarySmtp)
            .filter(|&(_, k)| k != NONE)
        {
            rows.push((k, g, AddressRole::PrimarySmtp));
        }
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Every invariant, sorted. Empty = valid.
///
/// SMTP uniqueness is [`smtp_duplicates`] (every SMTP proxy); UPN is
/// [`duplicates`] over the UPN key lane.
pub fn validate(v: &View<'_>) -> Vec<Violation> {
    let mut out = dangling(v);
    out.extend(smtp_duplicates(v));
    out.extend(duplicates(v, Attribute::Upn));
    out.extend(address_rules(v));
    out.sort();
    out
}
