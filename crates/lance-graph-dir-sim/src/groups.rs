//! Group properties as a `WHERE` over the group population, separate from
//! nesting.
//!
//! Nesting is one global pattern ([`View::members_transitive`],
//! [`View::groups_transitive`]): it follows every group, whatever its
//! kind. What a group is, is a property, and the properties are
//! independent, so a group can have neither, either or both:
//!
//! * **mail-enabled**: the group has a primary SMTP address, a distribution
//!   group in the mail sense;
//! * **security-enabled**: the group has a SID, which carries the
//!   permissions granted to the group to its members.
//!
//! A caller combines the two halves: the groups a user is in
//! (`groups_transitive`) that are security-enabled (`groups_where`) are the
//! SIDs the user inherits.
//!
//! ```text
//!   groups_where(view, &GroupWhere::Is(SecurityEnabled))
//!      │  and([Plane(live), Plane(security)])     one Quack program
//!      ▼
//!   base groups (deleted and overridden rows gated out)
//!      + overridden groups, evaluated directly       (delta-sized)
//!      + created groups, the same program            (delta-sized)
//! ```

use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_quack::{Agg, Cmp, Col, Filter, Mask};
use ogar_dir_sim::Attribute;

use crate::snapshot::{self, NONE};
use crate::{exec, Kept, NodeKind, View};

/// A property a group has, independent of nesting and of the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupProperty {
    /// The group has a primary SMTP address: mail sent to it reaches its
    /// members.
    MailEnabled,
    /// The group is known to be security-enabled: it has a SID that carries
    /// permissions. A group whose flag was not read, or that a version
    /// created, does not have it.
    SecurityEnabled,
}

/// A `WHERE` over group properties, lowered to one Quack filter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupWhere {
    /// The group has the property.
    Is(GroupProperty),
    /// The negation.
    Not(Box<GroupWhere>),
    /// Every part holds; an empty `And` holds for every group.
    And(Vec<GroupWhere>),
    /// Some part holds; an empty `Or` holds for none.
    Or(Vec<GroupWhere>),
}

/// Plane 0: the live groups. Plane 1: the security plane.
const LIVE: Mask = Mask(0);
const SECURITY: Mask = Mask(1);
/// Lane 0: the primary SMTP comparison key (`NONE` when absent).
const SMTP: Col = Col(0);

impl GroupWhere {
    /// Both properties: a mail-enabled security group.
    #[must_use]
    pub fn both() -> Self {
        GroupWhere::And(vec![
            GroupWhere::Is(GroupProperty::MailEnabled),
            GroupWhere::Is(GroupProperty::SecurityEnabled),
        ])
    }

    fn filter(&self) -> Filter {
        match self {
            GroupWhere::Is(GroupProperty::MailEnabled) => Filter::cmp(SMTP, Cmp::NeU32(NONE)),
            GroupWhere::Is(GroupProperty::SecurityEnabled) => Filter::plane(SECURITY),
            GroupWhere::Not(w) => Filter::Not(Box::new(w.filter())),
            GroupWhere::And(ws) => Filter::and(ws.iter().map(Self::filter)),
            GroupWhere::Or(ws) => Filter::or(ws.iter().map(Self::filter)),
        }
    }

    /// The same predicate on one group's properties: the delta-sized rows a
    /// version overrode.
    fn holds(&self, mail: bool, security: bool) -> bool {
        match self {
            GroupWhere::Is(GroupProperty::MailEnabled) => mail,
            GroupWhere::Is(GroupProperty::SecurityEnabled) => security,
            GroupWhere::Not(w) => !w.holds(mail, security),
            GroupWhere::And(ws) => ws.iter().all(|w| w.holds(mail, security)),
            GroupWhere::Or(ws) => ws.iter().any(|w| w.holds(mail, security)),
        }
    }
}

/// `SELECT groups WHERE <filter>` over a version, as a bitmap over the
/// group ordinals (base groups, then the groups the version created).
#[must_use]
pub fn groups_where(v: &View<'_>, filter: &GroupWhere) -> Kept {
    let (pop, ov) = v.pop(NodeKind::Group);
    let p = exec::program(
        Filter::and([Filter::plane(LIVE), filter.filter()]),
        Agg::Rows,
    );
    let run = |live: &[u64], security: &[u64], smtp: &[u32]| {
        let lanes = [LaneRef::U32(smtp)];
        let masks: [&[u64]; 2] = [live, security];
        exec::keep(
            &p,
            &Planes {
                n_rows: smtp.len(),
                masks: &masks,
                lanes: &lanes,
            },
            &Foreign::NONE,
        )
    };
    let mut live = v.base_live(NodeKind::Group, &pop.all).into_owned();
    let overridden = ov.overrides(Attribute::PrimarySmtp);
    for o in overridden.keys() {
        snapshot::clear_bit(&mut live, usize::from(*o));
    }
    let mut base = run(&live, &pop.security, &pop.smtp_key);
    let alive = v.base_live(NodeKind::Group, &pop.all);
    for (o, (_, key)) in overridden {
        let i = usize::from(*o);
        if snapshot::bit(&alive, i) && filter.holds(*key != NONE, snapshot::bit(&pop.security, i)) {
            base.set(i);
        }
    }
    let created = ov.created.key(Attribute::PrimarySmtp);
    let none = vec![0u64; lance_graph_mask_risc::words_for(created.len())];
    base.concat(&run(&snapshot::ones(created.len()), &none, created))
}
