//! Invariants as Quack programs over the version's lanes.
//!
//! | invariant            | relational form                                         | lowering                               |
//! |----------------------|---------------------------------------------------------|----------------------------------------|
//! | edge integrity       | `members ANTI JOIN users ∪ members ANTI JOIN groups`    | `Rows(¬Semijoin(user,·) ∨ ¬Semijoin(group,·))` — `MaskOp::Gather` over the node kind planes |
//! | unique SMTP / UPN    | `GROUP BY key HAVING count > 1` over active users        | `GroupReduce Count` keyed on the normalized-key dictionary id |
//!
//! No user objects are built. The integrity check reads the two membership
//! lanes and two kind planes — it cannot see a string. Uniqueness reads one
//! key lane and one plane; strings are resolved only for the (few) keys that
//! actually collide, to report them.
//!
//! Materialisations, all at the evidence boundary and bounded by the number
//! of violations: the offending membership rows (`materialize_rows` of the
//! kept mask) and each duplicate key's owner rows.

use crate::exec::{group_count_into, keep, program};
use crate::snapshot::bit;
use crate::view::View;
use lance_graph_mask_risc::{Foreign, ForeignPlane as FPlane, LaneRef, Planes, Program};
use lance_graph_quack::{Agg, Cmp, Col, Filter, ForeignPlane, Mask};
use ogar_dir_core::Guid128;
use ogar_dir_sim::{normalize, Attribute, Endpoint, Violation};

/// The edge-integrity program over a membership relation: lane 0 = user
/// ordinal, lane 1 = group ordinal, plane 0 = live rows; foreign plane 0 =
/// the node table's user plane, 1 = its group plane. An anti-join on each
/// side; an unresolved endpoint (`NONE`) is out of range and gathers 0.
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

/// Dangling memberships of a version (base rows still live + added rows).
pub fn dangling(v: &View<'_>) -> Vec<Violation> {
    let s = v.snap;
    let fps = [
        FPlane {
            words: &s.user,
            rows: s.len(),
        },
        FPlane {
            words: &s.group,
            rows: s.len(),
        },
    ];
    let foreign = Foreign {
        planes: &fps,
        lanes: &[],
    };
    let p = dangling_program();
    let side = |uo: u32| {
        if bit(&s.user, uo) {
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

    // The overlay relation: delta-sized lanes, same program.
    let (au, ag): (Vec<u32>, Vec<u32>) = v.ov.added.values().copied().unzip();
    let ids: Vec<(Guid128, Guid128)> = v.ov.added.keys().copied().collect();
    let all = crate::snapshot::ones(au.len());
    let lanes = [LaneRef::U32(&au), LaneRef::U32(&ag)];
    let masks: [&[u64]; 1] = [&all];
    let planes = Planes {
        n_rows: au.len(),
        masks: &masks,
        lanes: &lanes,
    };
    for r in keep(&p, &planes, &foreign).rows() {
        let (user, group) = ids[r];
        out.push(Violation::DanglingMembership {
            user,
            group,
            missing: side(au[r]),
        });
    }
    out.sort();
    out
}

/// Active users sharing a normalized value of `a`.
pub fn duplicates(v: &View<'_>, a: Attribute) -> Vec<Violation> {
    let s = v.snap;
    let keys = &v.dicts.keys;
    let k = keys.len();
    if k == 0 {
        return Vec::new();
    }
    let base_key = match a {
        Attribute::Upn => &s.upn_key,
        Attribute::PrimarySmtp => &s.smtp_key,
    };
    let over = match a {
        Attribute::Upn => &v.ov.upn,
        Attribute::PrimarySmtp => &v.ov.smtp,
    };
    let owners_plane = v.live_owners(a);
    let mut counts = vec![0i64; k];

    // Base: GROUP BY key COUNT(*) over the users still owning their observed value.
    let lanes = [LaneRef::U32(base_key)];
    let masks: [&[u64]; 1] = [&owners_plane];
    let base = Planes {
        n_rows: s.len(),
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

    // Overrides: the same GROUP BY over the delta rows, admitted by a
    // semijoin of the overridden ordinal against the active-user plane.
    let (oo, ok): (Vec<u32>, Vec<u32>) = over.iter().map(|(o, (_, key))| (*o, *key)).unzip();
    let all = crate::snapshot::ones(oo.len());
    let fps = [FPlane {
        words: &s.active_user,
        rows: s.len(),
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
            .map(|o| s.ids[o])
            .collect();
        owners.extend(
            over.iter()
                .filter(|(o, (_, kk))| *kk == key && bit(&s.active_user, **o))
                .map(|(o, _)| s.ids[*o as usize]),
        );
        owners.sort();
        let value = keys.resolve(key).map(normalize).unwrap_or_default();
        out.push(match a {
            Attribute::Upn => Violation::DuplicateUpn { upn: value, owners },
            Attribute::PrimarySmtp => Violation::DuplicateSmtp {
                address: value,
                owners,
            },
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
