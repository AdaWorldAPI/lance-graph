//! Named roles with permission sets.

use crate::permission::PermissionSpec;

/// A named role with a set of permissions across entity types.
#[derive(Clone, Debug)]
pub struct Role {
    pub name: &'static str,
    pub permissions: Vec<PermissionSpec>,
}

impl Role {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            permissions: Vec::new(),
        }
    }

    /// Builder: add a permission and return self.
    pub fn with_permission(mut self, perm: PermissionSpec) -> Self {
        self.permissions.push(perm);
        self
    }

    /// Find the permission for a specific entity type.
    pub fn permission_for(&self, entity_type: &str) -> Option<&PermissionSpec> {
        self.permissions
            .iter()
            .find(|p| p.entity_type == entity_type)
    }

    /// Check if this role can read an entity type at a given depth.
    pub fn can_read(
        &self,
        entity_type: &str,
        depth: lance_graph_contract::property::PrefetchDepth,
    ) -> bool {
        self.permission_for(entity_type)
            .map(|p| p.can_read_at(depth))
            .unwrap_or(false)
    }

    /// Check if this role can write a predicate on an entity type.
    pub fn can_write(&self, entity_type: &str, predicate: &str) -> bool {
        self.permission_for(entity_type)
            .map(|p| p.can_write(predicate))
            .unwrap_or(false)
    }

    /// Check if this role can trigger an action on an entity type.
    pub fn can_act(&self, entity_type: &str, action_name: &str) -> bool {
        self.permission_for(entity_type)
            .map(|p| p.can_act(action_name))
            .unwrap_or(false)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Example roles — SMB domain
// ═══════════════════════════════════════════════════════════════════════════

/// Accountant: can see Detail on Customers, Full on Invoices,
/// can approve invoices, cannot delete anything.
pub fn accountant() -> Role {
    use lance_graph_contract::property::PrefetchDepth;
    Role::new("accountant")
        .with_permission(PermissionSpec::read_at("Customer", PrefetchDepth::Detail))
        .with_permission(PermissionSpec::full(
            "Invoice",
            &["status", "payment_date"],
            &["approve", "mark_paid"],
        ))
        .with_permission(PermissionSpec::read_at(
            "TaxDeclaration",
            PrefetchDepth::Similar,
        ))
}

/// Auditor: can see Full (L3) on everything but cannot write or act.
pub fn auditor() -> Role {
    use lance_graph_contract::property::PrefetchDepth;
    Role::new("auditor")
        .with_permission(PermissionSpec::read_at("Customer", PrefetchDepth::Full))
        .with_permission(PermissionSpec::read_at("Invoice", PrefetchDepth::Full))
        .with_permission(PermissionSpec::read_at(
            "TaxDeclaration",
            PrefetchDepth::Full,
        ))
}

/// Admin: full access everywhere.
pub fn admin() -> Role {
    Role::new("admin")
        .with_permission(PermissionSpec::full(
            "Customer",
            &[
                "customer_name",
                "tax_id",
                "address",
                "iban",
                "phone",
                "email",
                "industry",
                "description",
                "tag",
                "note",
            ],
            &["classify", "merge", "delete"],
        ))
        .with_permission(PermissionSpec::full(
            "Invoice",
            &["status", "payment_date", "due_date", "flagged"],
            &["approve", "mark_paid", "flag", "delete"],
        ))
        .with_permission(PermissionSpec::full(
            "TaxDeclaration",
            &["status", "submitted_date"],
            &["submit", "retract"],
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::PrefetchDepth;

    #[test]
    fn accountant_can_approve_invoice() {
        let role = accountant();
        assert!(role.can_act("Invoice", "approve"));
        assert!(role.can_act("Invoice", "mark_paid"));
        assert!(role.can_write("Invoice", "status"));
        assert!(role.can_write("Invoice", "payment_date"));
    }

    #[test]
    fn accountant_cannot_delete_customer() {
        let role = accountant();
        assert!(!role.can_act("Customer", "delete"));
        assert!(!role.can_write("Customer", "customer_name"));
        // accountant can read Customer at Detail
        assert!(role.can_read("Customer", PrefetchDepth::Detail));
        assert!(!role.can_read("Customer", PrefetchDepth::Full));
    }

    #[test]
    fn auditor_reads_full_cannot_write() {
        let role = auditor();
        assert!(role.can_read("Customer", PrefetchDepth::Full));
        assert!(role.can_read("Invoice", PrefetchDepth::Full));
        assert!(role.can_read("TaxDeclaration", PrefetchDepth::Full));
        assert!(!role.can_write("Customer", "customer_name"));
        assert!(!role.can_write("Invoice", "status"));
        assert!(!role.can_act("Invoice", "approve"));
    }

    #[test]
    fn admin_can_do_everything() {
        let role = admin();
        assert!(role.can_read("Customer", PrefetchDepth::Full));
        assert!(role.can_write("Customer", "customer_name"));
        assert!(role.can_act("Customer", "delete"));
        assert!(role.can_write("Invoice", "flagged"));
        assert!(role.can_act("Invoice", "delete"));
        assert!(role.can_act("TaxDeclaration", "submit"));
        assert!(role.can_act("TaxDeclaration", "retract"));
    }
}
