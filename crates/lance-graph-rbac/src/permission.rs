//! Permission specifications tied to the ontology layer.

use lance_graph_contract::class_view::FieldMask;
use lance_graph_contract::property::PrefetchDepth;

/// What a role can do on a specific entity type.
///
/// # Depth is a level; projection is a view
///
/// [`max_depth`](Self::max_depth) is a *scalar level* (Identity < … < Full):
/// how far down the prefetch ladder a role may read — "more or less of the
/// same fields". It does NOT express **distinct** role-views of one class.
///
/// [`projection`](Self::projection) is the orthogonal axis: a [`FieldMask`]
/// over the class's `ClassView` field basis naming exactly which fields the
/// role may see. Two roles at the *same* depth can carry *disjoint*
/// projections — the mechanism behind `classid :: role :: membership`, where
/// the role IS the projection. A consumer (e.g. medcare-rs) gives
/// `health-personnel`, `invoice`, and `research` three distinct projections
/// of one clinical class and enforces that they never collapse (the research
/// projection disjoint from the identifier fields, etc.). The distinctness
/// enforcement is the consumer's; the projection slot is here.
#[derive(Clone, Debug)]
pub struct PermissionSpec {
    /// Entity type this permission applies to (e.g. "Customer", "Invoice").
    pub entity_type: &'static str,
    /// Maximum property prefetch depth this role can access.
    /// Identity = Required only, Full = everything including Free + episodic.
    pub max_depth: PrefetchDepth,
    /// The role's lawful **field projection** over this entity's `ClassView`
    /// field basis. [`FieldMask::FULL`] = no narrowing (depth governs); a
    /// narrowed mask is the per-role view that makes roles distinct rather
    /// than merely graduated.
    pub projection: FieldMask,
    /// Predicates this role can write. Empty = read-only for this entity.
    pub writable_predicates: &'static [&'static str],
    /// ActionSpec names this role can trigger. Empty = no actions.
    pub allowed_actions: &'static [&'static str],
}

impl PermissionSpec {
    /// Read-only at Identity depth (minimal access).
    pub const fn read_only(entity_type: &'static str) -> Self {
        Self {
            entity_type,
            max_depth: PrefetchDepth::Identity,
            projection: FieldMask::FULL,
            writable_predicates: &[],
            allowed_actions: &[],
        }
    }

    /// Full read + specified write predicates + specified actions.
    pub const fn full(
        entity_type: &'static str,
        writable: &'static [&'static str],
        actions: &'static [&'static str],
    ) -> Self {
        Self {
            entity_type,
            max_depth: PrefetchDepth::Full,
            projection: FieldMask::FULL,
            writable_predicates: writable,
            allowed_actions: actions,
        }
    }

    /// Read at a specific depth, no writes.
    pub const fn read_at(entity_type: &'static str, depth: PrefetchDepth) -> Self {
        Self {
            entity_type,
            max_depth: depth,
            projection: FieldMask::FULL,
            writable_predicates: &[],
            allowed_actions: &[],
        }
    }

    /// Narrow this permission to a specific field projection — the per-role
    /// view over the class's field basis. Builder; chains after any
    /// constructor (`read_at(..).with_projection(mask)`).
    pub const fn with_projection(mut self, projection: FieldMask) -> Self {
        self.projection = projection;
        self
    }

    /// Is field position `n` (a bit in the class's `ClassView` field basis)
    /// within this role's projection? `false` = the role may not see it,
    /// regardless of depth.
    pub const fn projects(&self, n: u8) -> bool {
        self.projection.has(n)
    }

    /// Check if this permission allows reading a predicate at the given depth.
    pub fn can_read_at(&self, depth: PrefetchDepth) -> bool {
        depth <= self.max_depth
    }

    /// Check if this permission allows writing a specific predicate.
    pub fn can_write(&self, predicate: &str) -> bool {
        self.writable_predicates.contains(&predicate)
    }

    /// Check if this permission allows triggering a specific action.
    pub fn can_act(&self, action_name: &str) -> bool {
        self.allowed_actions.contains(&action_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_defaults() {
        let p = PermissionSpec::read_only("Customer");
        assert_eq!(p.entity_type, "Customer");
        assert_eq!(p.max_depth, PrefetchDepth::Identity);
        // No narrowing by default — projection governs nothing until set.
        assert_eq!(p.projection, FieldMask::FULL);
        assert!(p.writable_predicates.is_empty());
        assert!(p.allowed_actions.is_empty());
    }

    #[test]
    fn distinct_roles_carry_disjoint_projections_of_one_class() {
        // classid :: role :: membership — two roles, SAME entity, SAME
        // depth, but DISTINCT views. Mirrors the medcare shape: a billing
        // role sees the coded/amount fields; a research role sees only
        // de-identified aggregate fields. The point isn't more-vs-less
        // depth — it's that the two field projections are disjoint.
        //
        // (Field positions index the entity's ClassView field basis; here
        // 0,1 are identifier/clinical slots, 5,6 the de-identified slots.)
        let invoice = PermissionSpec::read_at("Diagnosis", PrefetchDepth::Detail)
            .with_projection(FieldMask::from_positions(&[0, 1]));
        let research = PermissionSpec::read_at("Diagnosis", PrefetchDepth::Detail)
            .with_projection(FieldMask::from_positions(&[5, 6]));

        // Same level, distinct views.
        assert_eq!(invoice.max_depth, research.max_depth);
        assert!(
            invoice.projection.is_disjoint(research.projection),
            "the two roles must project distinct views of the class",
        );
        // The slot the invoice role sees, the research role does not.
        assert!(invoice.projects(0));
        assert!(!research.projects(0));
        assert!(research.projects(5));
        assert!(!invoice.projects(5));
    }

    #[test]
    fn full_access_allows_writes() {
        let p = PermissionSpec::full("Invoice", &["status", "payment_date"], &["approve"]);
        assert_eq!(p.max_depth, PrefetchDepth::Full);
        assert!(p.can_write("status"));
        assert!(p.can_write("payment_date"));
        assert!(!p.can_write("due_date"));
        assert!(p.can_act("approve"));
        assert!(!p.can_act("delete"));
    }

    #[test]
    fn can_read_at_depth() {
        let p = PermissionSpec::read_at("Customer", PrefetchDepth::Detail);
        assert!(p.can_read_at(PrefetchDepth::Identity));
        assert!(p.can_read_at(PrefetchDepth::Detail));
        assert!(!p.can_read_at(PrefetchDepth::Similar));
        assert!(!p.can_read_at(PrefetchDepth::Full));
    }
}
