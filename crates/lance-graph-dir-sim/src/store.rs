//! The version history — and the audit trail. There is no other log.
//!
//! * An observation is a root: an immutable [`Snapshot`] behind an `Arc`.
//! * A simulated version stores only `parent`, `origin` (rule + evidence) and
//!   `delta`. Its state is the root snapshot plus the overlay obtained by
//!   folding the lineage's deltas — work and memory proportional to the
//!   accumulated delta, never a copy of the population.
//! * Versions, snapshots and dictionaries are append-only. Simulating,
//!   validating or rejecting a version cannot alter another.

use crate::rule::Rule;
use crate::snapshot::{BuildError, Dicts, Observation, Snapshot};
use crate::validate::validate;
use crate::view::{ApplyError, Overlay, View};
use ogar_dir_core::Guid128;
use ogar_dir_sim::{
    Attribute, Change, EvidenceRef, ExecutionPlan, Origin, PlanError, Version, VersionId,
    Violation, TAG_DESIRED, TAG_OBSERVED,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Simulation failure. No version is created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimError {
    /// No such version.
    UnknownVersion(VersionId),
    /// The rule proposed nothing (no new version).
    EmptyProposal(ogar_dir_sim::RuleId),
    /// The proposal does not apply to the parent.
    Apply(ApplyError),
    /// The two versions do not share a node set (diff unsupported in this slice).
    NodeSetChanged,
}

/// Refusal to make a version desired. The version stays as evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    /// The rejected version.
    pub version: VersionId,
    /// Why.
    pub violations: Vec<Violation>,
}

/// Append-only version store.
#[derive(Debug, Default)]
pub struct VersionStore {
    dicts: Dicts,
    roots: BTreeMap<VersionId, Arc<Snapshot>>,
    versions: Vec<Version>,
    tags: BTreeMap<String, VersionId>,
    verdicts: BTreeMap<VersionId, Vec<Violation>>,
}

impl VersionStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    fn next_id(&self) -> VersionId {
        VersionId(self.versions.len() as u64)
    }

    /// Record an observation as a new root; tags it `"observed"`.
    pub fn observe(
        &mut self,
        source: &str,
        observed_at_ms: i64,
        obs: Observation,
    ) -> Result<VersionId, BuildError> {
        let snap = Snapshot::build(obs, &mut self.dicts)?;
        let id = self.next_id();
        self.versions.push(Version {
            id,
            parent: None,
            origin: Origin::Observed {
                source: source.into(),
                observed_at_ms,
            },
            delta: Vec::new(),
        });
        self.roots.insert(id, Arc::new(snap));
        self.tags.insert(TAG_OBSERVED.into(), id);
        Ok(id)
    }

    /// Provenance record.
    pub fn version(&self, v: VersionId) -> Option<&Version> {
        self.versions.get(v.0 as usize)
    }

    /// Root first.
    pub fn lineage(&self, v: VersionId) -> Result<Vec<VersionId>, SimError> {
        let mut path = Vec::new();
        let mut cur = Some(v);
        while let Some(c) = cur {
            path.push(c);
            cur = self.version(c).ok_or(SimError::UnknownVersion(c))?.parent;
        }
        path.reverse();
        Ok(path)
    }

    /// The shared snapshot under `v`.
    pub fn snapshot(&self, v: VersionId) -> Result<&Arc<Snapshot>, SimError> {
        let root = self.lineage(v)?[0];
        self.roots.get(&root).ok_or(SimError::UnknownVersion(root))
    }

    /// A coherent read of `v`: shared snapshot + folded overlay.
    pub fn view(&self, v: VersionId) -> Result<View<'_>, SimError> {
        let path = self.lineage(v)?;
        let snap = self
            .roots
            .get(&path[0])
            .ok_or(SimError::UnknownVersion(path[0]))?;
        let mut view = View::new(snap, &self.dicts, Overlay::default());
        for id in &path[1..] {
            for c in &self.versions[id.0 as usize].delta {
                view.apply(c).map_err(SimError::Apply)?;
            }
        }
        Ok(view)
    }

    /// Run a pure rule against `parent`, recording a hypothetical version.
    pub fn simulate(
        &mut self,
        parent: VersionId,
        rule: &dyn Rule,
        evidence: &[EvidenceRef],
    ) -> Result<VersionId, SimError> {
        let delta = rule.propose(&self.view(parent)?, evidence);
        if delta.is_empty() {
            return Err(SimError::EmptyProposal(rule.id()));
        }
        for c in &delta {
            if let Change::SetAttribute { to, .. } = c {
                self.dicts.intern_attr(to.as_deref());
            }
        }
        let mut view = self.view(parent)?;
        for c in &delta {
            view.apply(c).map_err(SimError::Apply)?;
        }
        let id = self.next_id();
        self.versions.push(Version {
            id,
            parent: Some(parent),
            origin: Origin::Simulated {
                rule: rule.id(),
                evidence: evidence.to_vec(),
            },
            delta,
        });
        Ok(id)
    }

    /// Validate `v` and record the verdict. Empty = valid.
    pub fn validate(&mut self, v: VersionId) -> Result<Vec<Violation>, SimError> {
        let violations = validate(&self.view(v)?);
        self.verdicts.insert(v, violations.clone());
        Ok(violations)
    }

    /// Recorded verdict.
    pub fn verdict(&self, v: VersionId) -> Option<&[Violation]> {
        self.verdicts.get(&v).map(Vec::as_slice)
    }

    /// Make `v` desired — only if valid. On failure the tag is untouched.
    pub fn promote_desired(&mut self, v: VersionId) -> Result<(), Rejection> {
        let violations = self.validate(v).map_err(|_| Rejection {
            version: v,
            violations: Vec::new(),
        })?;
        if !violations.is_empty() {
            return Err(Rejection {
                version: v,
                violations,
            });
        }
        self.tags.insert(TAG_DESIRED.into(), v);
        Ok(())
    }

    /// Version a tag points to.
    pub fn tag(&self, name: &str) -> Option<VersionId> {
        self.tags.get(name).copied()
    }

    /// Semantic difference `a → b`, sorted. Versions over the same snapshot
    /// compare only their overlays' touched keys (delta-sized); versions over
    /// different snapshots (reconciliation) merge the two sorted relations.
    pub fn diff(&self, a: VersionId, b: VersionId) -> Result<Vec<Change>, SimError> {
        let (va, vb) = (self.view(a)?, self.view(b)?);
        let mut out = if std::ptr::eq(va.snap, vb.snap) {
            diff_shared(&va, &vb)
        } else {
            diff_full(&va, &vb)?
        };
        out.sort();
        Ok(out)
    }

    /// Plan for the current desired version, from the LATEST observation.
    ///
    /// The basis is the version tagged observed, not the desired version's
    /// own lineage root: after a re-observation the directory may already
    /// carry part of the desired state, and the plan must cover only what is
    /// still missing, or its `NotMember` preconditions fail on execution.
    /// Before any observation is tagged, the lineage root is the basis.
    pub fn plan(&self, target: VersionId) -> Result<ExecutionPlan, PlanError> {
        if self.tag(TAG_DESIRED) != Some(target) {
            return Err(PlanError::NotDesired(target));
        }
        let root = self
            .lineage(target)
            .map_err(|_| PlanError::UnknownVersion(target))?[0];
        let basis = self.tag(TAG_OBSERVED).unwrap_or(root);
        let diff = self
            .diff(basis, target)
            .map_err(|_| PlanError::UnknownVersion(target))?;
        Ok(ExecutionPlan::from_diff(basis, target, diff))
    }

    /// Why does `v` contain membership `(user, group)`? The lineage from
    /// the observed root to the version whose delta last added it; a chain
    /// of length 1 means it was observed. `None` if `v` lacks it.
    pub fn explain_membership(
        &self,
        v: VersionId,
        user: &Guid128,
        group: &Guid128,
    ) -> Option<Vec<&Version>> {
        if !self.view(v).ok()?.is_member(user, group) {
            return None;
        }
        let path = self.lineage(v).ok()?;
        let add = Change::AddMembership {
            user: *user,
            group: *group,
        };
        let at = path
            .iter()
            .rposition(|id| self.versions[id.0 as usize].delta.contains(&add))
            .unwrap_or(0);
        Some(
            path[..=at]
                .iter()
                .map(|id| &self.versions[id.0 as usize])
                .collect(),
        )
    }
}

fn set_change(
    node: Guid128,
    attribute: Attribute,
    from: Option<&str>,
    to: Option<&str>,
) -> Option<Change> {
    (from != to).then(|| Change::SetAttribute {
        node,
        attribute,
        from: from.map(str::to_string),
        to: to.map(str::to_string),
    })
}

/// Same snapshot: only keys either overlay touched can differ.
fn diff_shared(a: &View<'_>, b: &View<'_>) -> Vec<Change> {
    let mut pairs: BTreeSet<(Guid128, Guid128)> =
        a.ov.added
            .keys()
            .chain(b.ov.added.keys())
            .copied()
            .collect();
    for v in [a, b] {
        if let Some(rm) = &v.ov.removed {
            for r in lance_graph_mask_risc::materialize_rows(rm, v.snap.membership_rows()) {
                pairs.insert(v.snap.member_guids(r as u32));
            }
        }
    }
    let mut out = Vec::new();
    for (u, g) in pairs {
        match (a.is_member(&u, &g), b.is_member(&u, &g)) {
            (false, true) => out.push(Change::AddMembership { user: u, group: g }),
            (true, false) => out.push(Change::RemoveMembership { user: u, group: g }),
            _ => {}
        }
    }
    for attr in [Attribute::Upn, Attribute::PrimarySmtp] {
        let touched: BTreeSet<u32> = match attr {
            Attribute::Upn => a.ov.upn.keys().chain(b.ov.upn.keys()).copied().collect(),
            Attribute::PrimarySmtp => a.ov.smtp.keys().chain(b.ov.smtp.keys()).copied().collect(),
        };
        for o in touched {
            out.extend(set_change(
                a.snap.ids[o as usize],
                attr,
                a.attr(o, attr),
                b.attr(o, attr),
            ));
        }
    }
    out
}

/// Different snapshots (actual vs desired): a merge of two sorted relations,
/// O(n + m) — the reconciliation path, not the simulation hot path.
fn diff_full(a: &View<'_>, b: &View<'_>) -> Result<Vec<Change>, SimError> {
    if a.snap.ids != b.snap.ids {
        return Err(SimError::NodeSetChanged);
    }
    let effective = |v: &View<'_>| -> BTreeSet<(Guid128, Guid128)> {
        let live = v.live_rows();
        lance_graph_mask_risc::materialize_rows(&live, v.snap.membership_rows())
            .into_iter()
            .map(|r| v.snap.member_guids(r as u32))
            .chain(v.ov.added.keys().copied())
            .collect()
    };
    let (ea, eb) = (effective(a), effective(b));
    let mut out: Vec<Change> = eb
        .difference(&ea)
        .map(|(u, g)| Change::AddMembership {
            user: *u,
            group: *g,
        })
        .collect();
    out.extend(ea.difference(&eb).map(|(u, g)| Change::RemoveMembership {
        user: *u,
        group: *g,
    }));
    for o in 0..a.snap.len() as u32 {
        for attr in [Attribute::Upn, Attribute::PrimarySmtp] {
            out.extend(set_change(
                a.snap.ids[o as usize],
                attr,
                a.attr(o, attr),
                b.attr(o, attr),
            ));
        }
    }
    Ok(out)
}
