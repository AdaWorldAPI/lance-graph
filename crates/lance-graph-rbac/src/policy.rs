//! Policy: a collection of roles with lookup and evaluation.

use crate::access::AccessDecision;
use crate::role::Role;
use lance_graph_contract::property::PrefetchDepth;

/// A policy is a named set of roles. Users are assigned roles;
/// the policy resolves access decisions by checking the user's role.
#[derive(Clone, Debug)]
pub struct Policy {
    pub name: &'static str,
    pub roles: Vec<Role>,
}

impl Policy {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            roles: Vec::new(),
        }
    }

    pub fn with_role(mut self, role: Role) -> Self {
        self.roles.push(role);
        self
    }

    pub fn role(&self, name: &str) -> Option<&Role> {
        self.roles.iter().find(|r| r.name == name)
    }

    /// Evaluate an access request.
    pub fn evaluate(
        &self,
        role_name: &str,
        entity_type: &str,
        operation: Operation<'_>,
    ) -> AccessDecision {
        let role = match self.role(role_name) {
            Some(r) => r,
            None => {
                return AccessDecision::Deny {
                    reason: "unknown role",
                }
            }
        };

        match operation {
            Operation::Read { depth } => {
                if role.can_read(entity_type, depth) {
                    AccessDecision::Allow
                } else {
                    AccessDecision::Deny {
                        reason: "insufficient read depth",
                    }
                }
            }
            Operation::Write { predicate } => {
                if role.can_write(entity_type, predicate) {
                    AccessDecision::Allow
                } else {
                    AccessDecision::Deny {
                        reason: "predicate not writable",
                    }
                }
            }
            Operation::Act { action } => {
                if role.can_act(entity_type, action) {
                    AccessDecision::Allow
                } else {
                    AccessDecision::Deny {
                        reason: "action not allowed",
                    }
                }
            }
        }
    }
}

/// What the caller wants to do.
#[derive(Clone, Debug)]
pub enum Operation<'a> {
    Read { depth: PrefetchDepth },
    Write { predicate: &'a str },
    Act { action: &'a str },
}

/// Build the default SMB policy with accountant, auditor, admin roles.
pub fn smb_policy() -> Policy {
    use crate::role::{accountant, admin, auditor};
    Policy::new("smb-default")
        .with_role(accountant())
        .with_role(auditor())
        .with_role(admin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::PrefetchDepth;

    #[test]
    fn smb_policy_has_three_roles() {
        let policy = smb_policy();
        assert_eq!(policy.roles.len(), 3);
        assert!(policy.role("accountant").is_some());
        assert!(policy.role("auditor").is_some());
        assert!(policy.role("admin").is_some());
    }

    #[test]
    fn evaluate_accountant_read_customer_detail() {
        let policy = smb_policy();
        let decision = policy.evaluate(
            "accountant",
            "Customer",
            Operation::Read {
                depth: PrefetchDepth::Detail,
            },
        );
        assert_eq!(decision, AccessDecision::Allow);
    }

    #[test]
    fn evaluate_accountant_read_customer_full() {
        let policy = smb_policy();
        let decision = policy.evaluate(
            "accountant",
            "Customer",
            Operation::Read {
                depth: PrefetchDepth::Full,
            },
        );
        assert!(decision.is_denied());
    }

    #[test]
    fn evaluate_auditor_write_anything() {
        let policy = smb_policy();
        let decision = policy.evaluate(
            "auditor",
            "Invoice",
            Operation::Write {
                predicate: "status",
            },
        );
        assert!(decision.is_denied());
    }

    #[test]
    fn evaluate_admin_write_customer_name() {
        let policy = smb_policy();
        let decision = policy.evaluate(
            "admin",
            "Customer",
            Operation::Write {
                predicate: "customer_name",
            },
        );
        assert_eq!(decision, AccessDecision::Allow);
    }

    #[test]
    fn evaluate_unknown_role() {
        let policy = smb_policy();
        let decision = policy.evaluate(
            "ghost",
            "Customer",
            Operation::Read {
                depth: PrefetchDepth::Identity,
            },
        );
        assert!(decision.is_denied());
        assert_eq!(
            decision,
            AccessDecision::Deny {
                reason: "unknown role"
            }
        );
    }
}
