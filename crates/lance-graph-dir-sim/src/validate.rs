//! Invariants as Quack programs over the version's lanes.
//!
//! | invariant            | relational form                                         | lowering                               |
//! |----------------------|---------------------------------------------------------|----------------------------------------|
//! | edge integrity       | `members ANTI JOIN users ∪ members ANTI JOIN groups`    | `Rows(¬Semijoin(user,·) ∨ ¬Semijoin(group,·))` — `MaskOp::Gather` over the live user and group planes |
//! | unique SMTP / UPN    | `GROUP BY key HAVING count > 1` over active users        | `GroupReduce Count` keyed on [`KeyId`]  |
//!
//! No user objects are built and no string is read: the integrity check
//! reads the two membership lanes and the two live-node planes, uniqueness
//! one key lane and one plane. A violation carries identities and the
//! [`KeyId`]; resolving a key to text is the reporter's job.
//!
//! Memberships whose endpoint never resolved to an ordinal are not in the
//! lanes (no sentinel ordinal exists); they are dangling by construction and
//! reported from the snapshot's unresolved table and the overlay, both
//! evidence-sized.
//!
//! Materialisations, all at the evidence boundary and bounded by the number
//! of violations: the offending membership rows (`materialize_rows` of the
//! kept mask) and each duplicate key's owner rows.

use crate::exec::{group_count_into, keep, program};
use crate::snapshot::{bit, ones};
use crate::view::View;
use lance_graph_mask_risc::{Foreign, ForeignPlane as FPlane, LaneRef, Planes, Program};
use lance_graph_quack::{Agg, Cmp, Col, Filter, ForeignPlane, Mask};
use ogar_dir_core::Guid128;
use ogar_dir_sim::{Attribute, Endpoint, KeyId, NodeKind, Violation};

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

    // Identity-held rows that still do not resolve in this version.
    for &(user, group) in &added.unresolved {
        let missing = if v.user_ordinal(&user).is_some() {
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

/// Active users sharing a comparison key of `a`.
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
    let active = v.active_users();
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

/// Every invariant, sorted. Empty = valid.
pub fn validate(v: &View<'_>) -> Vec<Violation> {
    let mut out = dangling(v);
    out.extend(duplicates(v, Attribute::PrimarySmtp));
    out.extend(duplicates(v, Attribute::Upn));
    out.sort();
    out
}
