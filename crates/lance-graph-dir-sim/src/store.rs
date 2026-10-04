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
    Attribute, Change, EvidenceRef, ExecutionPlan, NodeState, Origin, PlanError, Version,
    VersionId, Violation, TAG_DESIRED, TAG_OBSERVED,
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
        let mut delta = rule.propose(&self.view(parent)?, evidence);
        // Canonical order is also a safe application order (`Change`'s
        // variant order: removals, deletes, sets, creates, adds), so the version does
        // not depend on the order a rule emitted its changes in.
        delta.sort();
        delta.dedup();
        if delta.is_empty() {
            return Err(SimError::EmptyProposal(rule.id()));
        }
        for c in &delta {
            match c {
                Change::SetAttribute { to, .. } => {
                    self.dicts.intern_attr(to.as_deref());
                }
                Change::CreateNode { state, .. } => {
                    self.dicts.intern_attr(state.upn.as_deref());
                    self.dicts.intern_attr(state.primary_smtp.as_deref());
                }
                _ => {}
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
    /// different snapshots (reconciliation) merge the two sorted node lanes
    /// and the two membership relations. Node sets may differ: a node only
    /// in `b` is a [`Change::CreateNode`], one only in `a` a
    /// [`Change::DeleteNode`].
    pub fn diff(&self, a: VersionId, b: VersionId) -> Result<Vec<Change>, SimError> {
        let (va, vb) = (self.view(a)?, self.view(b)?);
        let mut out = if std::ptr::eq(va.snap, vb.snap) {
            diff_shared(&va, &vb)
        } else {
            diff_full(&va, &vb)
        };
        out.sort();
        Ok(out)
    }

    /// Plan for the current desired version: the work still outstanding
    /// against the LATEST observation.
    ///
    /// The intent is the desired version's own net delta over its lineage
    /// root (delta-sized, same snapshot). Each intended change is then
    /// checked against the latest observation (the version tagged observed;
    /// before any, the lineage root) and kept only if reality does not
    /// already show its effect, with the precondition read from that
    /// observation. So after a re-observation the plan holds no repeated
    /// operation and no stale precondition, and a change of reality outside
    /// the intent (a node or membership someone else created) is neither
    /// planned nor reverted.
    pub fn plan(&self, target: VersionId) -> Result<ExecutionPlan, PlanError> {
        if self.tag(TAG_DESIRED) != Some(target) {
            return Err(PlanError::NotDesired(target));
        }
        let root = self
            .lineage(target)
            .map_err(|_| PlanError::UnknownVersion(target))?[0];
        let basis = self.tag(TAG_OBSERVED).unwrap_or(root);
        // Stored versions re-fold by construction (every delta applied when
        // it was recorded), so the only failure here is an unknown id.
        let view = |v| self.view(v).map_err(|_| PlanError::UnknownVersion(v));
        let (vr, vt, vb) = (view(root)?, view(target)?, view(basis)?);
        let intent = diff_shared(&vr, &vt);
        let ops = outstanding(&vb, intent).map_err(|node| PlanError::Unconvergeable {
            basis,
            target,
            node,
        })?;
        Ok(ExecutionPlan::from_diff(basis, target, ops))
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

/// Upn and primary-SMTP changes between two present nodes.
fn attr_changes(node: Guid128, a: &NodeState, b: &NodeState, out: &mut Vec<Change>) {
    out.extend(set_change(
        node,
        Attribute::Upn,
        a.upn.as_deref(),
        b.upn.as_deref(),
    ));
    out.extend(set_change(
        node,
        Attribute::PrimarySmtp,
        a.primary_smtp.as_deref(),
        b.primary_smtp.as_deref(),
    ));
}

/// The change that takes node `g` from its state in `a` to its state in `b`.
fn node_changes(g: Guid128, a: Option<NodeState>, b: Option<NodeState>, out: &mut Vec<Change>) {
    match (a, b) {
        (None, Some(state)) => out.push(Change::CreateNode { node: g, state }),
        (Some(state), None) => out.push(Change::DeleteNode { node: g, state }),
        (Some(sa), Some(sb)) => attr_changes(g, &sa, &sb, out),
        (None, None) => {}
    }
}

/// Same snapshot: only keys either overlay touched can differ.
fn diff_shared(a: &View<'_>, b: &View<'_>) -> Vec<Change> {
    let mut out = Vec::new();
    let s = a.snap;

    // Node presence: created or deleted in either overlay.
    let mut nodes: BTreeSet<Guid128> = BTreeSet::new();
    for v in [a, b] {
        nodes.extend(v.ov.created.ids.iter().copied());
        nodes.extend(v.ov.deleted.iter().map(|&o| s.ids[o as usize]));
    }
    for g in &nodes {
        node_changes(*g, a.node_state(g), b.node_state(g), &mut out);
    }

    // Memberships either overlay touched.
    let mut pairs: BTreeSet<(Guid128, Guid128)> =
        a.ov.added.iter().chain(&b.ov.added).copied().collect();
    for v in [a, b] {
        pairs.extend(v.ov.removed.iter().map(|&r| s.member_guids(r)));
    }
    for (u, g) in pairs {
        match (a.is_member(&u, &g), b.is_member(&u, &g)) {
            (false, true) => out.push(Change::AddMembership { user: u, group: g }),
            (true, false) => out.push(Change::RemoveMembership { user: u, group: g }),
            _ => {}
        }
    }

    // Attribute overrides of base nodes present in both versions (a node
    // created or deleted between them is covered above).
    for attr in [Attribute::Upn, Attribute::PrimarySmtp] {
        let touched: BTreeSet<u32> =
            a.ov.overrides(attr)
                .keys()
                .chain(b.ov.overrides(attr).keys())
                .copied()
                .collect();
        for o in touched {
            let g = s.ids[o as usize];
            if nodes.contains(&g) {
                continue;
            }
            out.extend(set_change(g, attr, a.attr(o, attr), b.attr(o, attr)));
        }
    }
    out
}

/// Existing nodes of a view, ascending by identity.
fn live_ids(v: &View<'_>) -> Vec<Guid128> {
    let mut ids: Vec<Guid128> = (0..v.snap.len() as u32)
        .filter_map(|o| v.guid(o))
        .chain(v.ov.created.ids.iter().copied())
        .collect();
    ids.sort();
    ids
}

/// Different snapshots (actual vs desired): a merge of the two sorted node
/// lanes and the two membership relations, O(n + m) — the reconciliation
/// path, not the simulation hot path.
fn diff_full(a: &View<'_>, b: &View<'_>) -> Vec<Change> {
    let mut out = Vec::new();
    let (ia, ib) = (live_ids(a), live_ids(b));
    let (mut i, mut j) = (0, 0);
    while i < ia.len() || j < ib.len() {
        let g = match (ia.get(i), ib.get(j)) {
            (Some(x), Some(y)) => *x.min(y),
            (Some(x), None) => *x,
            (None, Some(y)) => *y,
            (None, None) => break,
        };
        i += usize::from(ia.get(i) == Some(&g));
        j += usize::from(ib.get(j) == Some(&g));
        node_changes(g, a.node_state(&g), b.node_state(&g), &mut out);
    }
    let effective = |v: &View<'_>| -> BTreeSet<(Guid128, Guid128)> {
        let live = v.live_rows();
        lance_graph_mask_risc::materialize_rows(&live, v.snap.membership_rows())
            .into_iter()
            .map(|r| v.snap.member_guids(r as u32))
            .chain(v.ov.added.iter().copied())
            .collect()
    };
    let (ea, eb) = (effective(a), effective(b));
    out.extend(eb.difference(&ea).map(|(u, g)| Change::AddMembership {
        user: *u,
        group: *g,
    }));
    out.extend(ea.difference(&eb).map(|(u, g)| Change::RemoveMembership {
        user: *u,
        group: *g,
    }));
    out
}

/// The part of `intent` the observation `basis` does not already show, with
/// compare-and-set expectations read from `basis`. Each check is one
/// identity lookup, so the work is proportional to the intent.
///
/// `Err(node)`: a node the intent creates already exists in `basis` with a
/// kind, enabled flag or OU that no change can converge (the algebra sets
/// only UPN and primary SMTP), so the create is neither done nor doable.
fn outstanding(basis: &View<'_>, intent: Vec<Change>) -> Result<Vec<Change>, Guid128> {
    let mut out = Vec::new();
    for c in intent {
        match c {
            Change::AddMembership { user, group } => {
                if !basis.is_member(&user, &group) {
                    out.push(Change::AddMembership { user, group });
                }
            }
            Change::RemoveMembership { user, group } => {
                if basis.is_member(&user, &group) {
                    out.push(Change::RemoveMembership { user, group });
                }
            }
            Change::SetAttribute {
                node,
                attribute,
                to,
                ..
            } => {
                // A node gone from reality has nothing left to set.
                if let Some(o) = basis.ordinal(&node) {
                    out.extend(set_change(
                        node,
                        attribute,
                        basis.attr(o, attribute),
                        to.as_deref(),
                    ));
                }
            }
            Change::CreateNode { node, state } => match basis.node_state(&node) {
                None => out.push(Change::CreateNode { node, state }),
                // Already exists: only its settable attributes may differ.
                Some(actual) => {
                    if (actual.kind, actual.active, actual.ou)
                        != (state.kind, state.active, state.ou)
                    {
                        return Err(node);
                    }
                    attr_changes(node, &actual, &state, &mut out);
                }
            },
            Change::DeleteNode { node, .. } => {
                if let Some(actual) = basis.node_state(&node) {
                    out.push(Change::DeleteNode {
                        node,
                        state: actual,
                    });
                }
            }
        }
    }
    Ok(out)
}
