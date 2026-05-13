//! Access decisions — the output of policy evaluation.

/// Result of an RBAC policy evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessDecision {
    /// Access granted.
    Allow,
    /// Access denied with reason.
    Deny { reason: &'static str },
    /// Access requires escalation (human approval, MFA, etc.).
    /// Maps to FreeEnergy escalation in the cognitive loop.
    Escalate { reason: &'static str },
}

impl AccessDecision {
    pub const fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    pub const fn is_denied(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }

    pub const fn is_escalation(&self) -> bool {
        matches!(self, Self::Escalate { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_predicates() {
        let allow = AccessDecision::Allow;
        assert!(allow.is_allowed());
        assert!(!allow.is_denied());
        assert!(!allow.is_escalation());

        let deny = AccessDecision::Deny {
            reason: "no permission",
        };
        assert!(!deny.is_allowed());
        assert!(deny.is_denied());
        assert!(!deny.is_escalation());

        let escalate = AccessDecision::Escalate {
            reason: "needs MFA",
        };
        assert!(!escalate.is_allowed());
        assert!(!escalate.is_denied());
        assert!(escalate.is_escalation());

        assert_eq!(AccessDecision::Allow, AccessDecision::Allow);
        assert_eq!(
            AccessDecision::Deny { reason: "x" },
            AccessDecision::Deny { reason: "x" }
        );
        assert_ne!(AccessDecision::Allow, AccessDecision::Deny { reason: "x" });
    }
}
