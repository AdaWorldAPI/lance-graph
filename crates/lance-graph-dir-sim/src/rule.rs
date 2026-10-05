//! Pure rules: `G(n+1) = R(G(n), evidence)`.
//!
//! A rule receives a [`View`] — borrowed lanes of one version — and returns
//! the [`Change`]s it proposes. It has no I/O handle and no way to mutate the
//! view. Rules select **populations**: a population is a bitmap over the
//! version's user ordinals, produced by Quack programs, never a loop that
//! runs a workflow per user. Values are [`ValueId`]s: a rule never compares
//! or builds a string.

use crate::exec::group_count_into;
use crate::view::View;
use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_quack::{Cmp, Col, Filter, Mask};
use ogar_dir_core::Guid128;
use ogar_dir_sim::NodeKind;
use ogar_dir_sim::{Attribute, Change, EvidenceRef, RuleId, ValueId};

/// A pure graph transformation.
pub trait Rule {
    /// Identity recorded in provenance.
    fn id(&self) -> RuleId;
    /// Proposed changes; must be deterministic in `(version, evidence)`.
    fn propose(&self, v: &View<'_>, evidence: &[EvidenceRef]) -> Vec<Change>;
}

/// Per-user membership count in `group` for one version:
/// `GROUP BY user COUNT(*) WHERE group = g` over the live base relation,
/// plus the overlay's added rows (delta-sized). One `K = |users|` sink,
/// indexed by user ordinal; no joined rows.
pub fn member_counts(v: &View<'_>, group: &Guid128) -> Vec<i64> {
    let s = v.snap;
    let mut counts = vec![0i64; v.users_len()];
    let Some(g) = v.group_ordinal(group) else {
        return counts;
    };
    let g = u32::from(g.0);
    let added = v.added_rows();
    for (uo, go) in added.users.iter().zip(&added.groups) {
        if *go == g {
            counts[*uo as usize] += 1;
        }
    }
    // A created group has no base rows.
    if g as usize >= s.groups.len() {
        return counts;
    }
    let live = v.live_rows();
    let lanes = [LaneRef::U32(&s.m_user), LaneRef::U32(&s.m_group)];
    let masks: [&[u64]; 1] = [&live];
    let planes = Planes {
        n_rows: s.membership_rows(),
        masks: &masks,
        lanes: &lanes,
    };
    group_count_into(
        Filter::and([Filter::plane(Mask(0)), Filter::cmp(Col(1), Cmp::EqU32(g))]),
        Col(0),
        &planes,
        &Foreign::NONE,
        &mut counts,
    );
    counts
}

/// Grant `group` to an explicit, evidence-sized list of users. Already
/// members are skipped, so the proposal is exactly the net change. The work
/// is proportional to the request, not to the directory.
pub struct GrantGroup {
    /// Rule identity.
    pub rule: RuleId,
    /// The group.
    pub group: Guid128,
    /// Who should receive it.
    pub to: Vec<Guid128>,
}

impl Rule for GrantGroup {
    fn id(&self) -> RuleId {
        self.rule
    }
    fn propose(&self, v: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        let mut to = self.to.clone();
        to.sort();
        to.dedup();
        to.into_iter()
            .filter(|u| !v.is_member(u, &self.group))
            .map(|user| Change::AddMembership {
                user,
                group: self.group,
            })
            .collect()
    }
}

/// Population rule: every active member of `source` is a member of `target`.
/// `active ∧ count(source) > 0 ∧ count(target) = 0`, from two folded
/// `GROUP BY` sinks and the resident active-user plane — no per-user workflow.
///
/// Why two programs and not one: "member of A and not member of B" is a
/// per-user fact over the membership relation, so a single program would
/// have to `Semijoin` users against a population another program produced.
/// That is the forbidden shape ([`crate::Kept`] makes it a compile error).
/// The two sinks are keyed by user ordinal and sized by the node universe,
/// never by membership rows, and are combined here at the consumer.
///
/// Not an `ogar-loco` program, for now. The only loco dialect that lowers to
/// mask-risc (`FoldDialect`) lives inside a test file in
/// `r2il-mask-abi-probe`, not in a library. It also combines only scalar
/// folds; its `GROUP_SUM` sink is zero-filled per fold and keeps just the
/// last one, so it cannot hold both counts. Copying it here would make a
/// second arity table. That dialect has to become a library first.
pub struct ImplyGroup {
    /// Rule identity.
    pub rule: RuleId,
    /// Membership that implies…
    pub source: Guid128,
    /// …membership here.
    pub target: Guid128,
}

impl Rule for ImplyGroup {
    fn id(&self) -> RuleId {
        self.rule
    }
    fn propose(&self, v: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        let (cs, ct) = (
            member_counts(v, &self.source),
            member_counts(v, &self.target),
        );
        let active = v.active_users();
        (0..v.users_len())
            .filter(|&o| crate::snapshot::bit(&active, o) && cs[o] > 0 && ct[o] == 0)
            .filter_map(|o| v.guid_in(NodeKind::User, o))
            .map(|user| Change::AddMembership {
                user,
                group: self.target,
            })
            .collect()
    }
}

/// Compare-and-set one user's primary SMTP against the version it reads.
pub struct SetPrimarySmtp {
    /// Rule identity.
    pub rule: RuleId,
    /// User.
    pub user: Guid128,
    /// New address, interned by the caller at ingress
    /// ([`VersionStore::intern`](crate::VersionStore::intern)).
    pub to: ValueId,
}

impl Rule for SetPrimarySmtp {
    fn id(&self) -> RuleId {
        self.rule
    }
    fn propose(&self, v: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        if v.user_ordinal(&self.user).is_none() {
            return Vec::new();
        }
        let from = v.attr(&self.user, Attribute::PrimarySmtp);
        if from == Some(self.to) {
            return Vec::new();
        }
        vec![Change::SetAttribute {
            node: self.user,
            attribute: Attribute::PrimarySmtp,
            from,
            to: Some(self.to),
        }]
    }
}
