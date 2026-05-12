//! Permission specifications tied to the ontology layer.

use lance_graph_contract::property::PrefetchDepth;

/// What a role can do on a specific entity type.
#[derive(Clone, Debug)]
pub struct PermissionSpec {
    /// Entity type this permission applies to (e.g. "Customer", "Invoice").
    pub entity_type: &'static str,
    /// Maximum property prefetch depth this role can access.
    /// Identity = Required only, Full = everything including Free + episodic.
    pub max_depth: PrefetchDepth,
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
            writable_predicates: writable,
            allowed_actions: actions,
        }
    }

    /// Read at a specific depth, no writes.
    pub const fn read_at(entity_type: &'static str, depth: PrefetchDepth) -> Self {
        Self {
            entity_type,
            max_depth: depth,
            writable_predicates: &[],
            allowed_actions: &[],
        }
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
        assert!(p.writable_predicates.is_empty());
        assert!(p.allowed_actions.is_empty());
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
