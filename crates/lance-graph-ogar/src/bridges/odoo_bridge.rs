//! Odoo (odoo-rs) tenant bridge — a thin type alias over
//! [`crate::bridges::unified::UnifiedBridge`] parameterised by
//! [`ogar_vocab::ports::OdooPort`].
//!
//! The differences between bridges (namespace, bridge_id, public-name
//! → class_id alias table) all come from the OGAR class schema. The
//! `OdooPort` carries the `Odoo` namespace + `odoo` bridge_id + the
//! Odoo-model alias table (`account.move` → COMMERCIAL_DOCUMENT,
//! `res.partner` → BILLING_PARTY, …) plus the cross-arm bridge
//! `account.analytic.line` → `BILLABLE_WORK_ENTRY` (`0x0103`), the SAME
//! canonical id the OpenProject / Redmine `TimeEntry` ports resolve to —
//! the commerce-arm convergence pin (OGAR #94, Northstar §7 T10).

use crate::bridges::unified::UnifiedBridge;
// `OdooPort::NAMESPACE` / `::aliases()` are `PortSpec` associated items —
// the trait must be in scope for the resolution to work (codex P1 on
// PR #570). Same import in the test module below.
pub use ogar_vocab::ports::OdooPort;
use ogar_vocab::ports::PortSpec;

/// Odoo `NamespaceBridge` — alias over the generic harness, locked to the
/// `Odoo` namespace via [`OdooPort`].
pub type OdooBridge = UnifiedBridge<OdooPort>;

/// Canonical namespace name for Odoo. Mirrors `OdooPort::NAMESPACE` so
/// consumers that import the constant from this module keep building.
pub const NAMESPACE: &str = OdooPort::NAMESPACE;

/// Compatibility shim — re-exports `ogar_vocab::ports::ODOO_ALIASES`
/// under a `*_CODEBOOK` name for symmetry with the other per-port
/// bridges. New code should reach for `ogar_vocab::ports::ODOO_ALIASES`
/// (or `OdooPort::aliases()`) directly — going through the canonical
/// layer keeps lance-graph free of port-specific data.
#[deprecated(
    note = "use `ogar_vocab::ports::ODOO_ALIASES` (or `OdooPort::aliases()`) — the constant lives in OGAR"
)]
pub const ODOO_CODEBOOK: &[(&str, u16)] = ogar_vocab::ports::ODOO_ALIASES;

#[cfg(test)]
mod tests {
    use super::*;
    use ogar_vocab::class_ids;
    // PortSpec needed in scope for `OdooPort::aliases()` / `::class_id()`
    // (the methods are trait items — codex P1 on PR #570).
    use ogar_vocab::ports::PortSpec;

    #[test]
    fn namespace_and_bridge_id_mirror_the_port() {
        assert_eq!(NAMESPACE, "Odoo");
        assert_eq!(OdooPort::NAMESPACE, "Odoo");
        assert_eq!(OdooPort::BRIDGE_ID, "odoo");
    }

    #[test]
    fn port_resolves_account_move_to_commercial_document() {
        assert_eq!(
            OdooPort::class_id("account.move"),
            Some(class_ids::COMMERCIAL_DOCUMENT)
        );
        assert_eq!(OdooPort::class_id("account.move"), Some(0x0202));
    }

    #[test]
    fn port_analytic_line_converges_on_billable_work_entry() {
        // The cross-arm bridge (OGAR #94): Odoo's timesheet/cost line
        // resolves to the SAME canonical id as the planner ports.
        assert_eq!(
            OdooPort::class_id("account.analytic.line"),
            Some(class_ids::BILLABLE_WORK_ENTRY)
        );
        assert_eq!(OdooPort::class_id("account.analytic.line"), Some(0x0103));
    }

    #[test]
    fn port_returns_none_for_non_codebook_name() {
        assert_eq!(OdooPort::class_id("not.a.model"), None);
    }
}
