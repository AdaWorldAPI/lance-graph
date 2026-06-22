//! WoA (Work Order Application) tenant bridge — a thin type alias over
//! [`crate::bridges::unified::UnifiedBridge`] parameterised by
//! [`ogar_vocab::ports::WoaPort`].
//!
//! The differences between bridges (namespace, bridge_id, public-name
//! → class_id alias table) all come from the OGAR class schema. The
//! `WoaPort` carries the `WorkOrder` namespace + `woa` bridge_id + the
//! German/English alias table (Vorgang ≡ WorkOrder, Stundenzettel ≡
//! TimesheetActivity ≡ TimeEntry, Kunde ≡ Customer, …), so this bridge
//! is one line. `Stundenzettel` / `TimeEntry` resolve to the SAME
//! canonical `BILLABLE_WORK_ENTRY` (`0x0103`) as the OpenProject /
//! Redmine planner ports — the cross-fork convergence pin (OGAR #93).

use crate::bridges::unified::UnifiedBridge;
// `WoaPort::NAMESPACE` / `::aliases()` are `PortSpec` associated items —
// the trait must be in scope for the resolution to work (codex P1 on
// PR #570). Same import in the test module below.
use ogar_vocab::ports::PortSpec;
pub use ogar_vocab::ports::WoaPort;

/// WoA `NamespaceBridge` — alias over the generic harness, locked to the
/// `WorkOrder` namespace via [`WoaPort`].
pub type WoaBridge = UnifiedBridge<WoaPort>;

/// Canonical namespace name for WoA. Mirrors `WoaPort::NAMESPACE` so
/// consumers that import the constant from this module keep building.
pub const NAMESPACE: &str = WoaPort::NAMESPACE;

/// Compatibility shim — re-exports `ogar_vocab::ports::WOA_ALIASES` under
/// a `*_CODEBOOK` name for symmetry with the other per-port bridges. New
/// code should reach for `ogar_vocab::ports::WOA_ALIASES` (or
/// `WoaPort::aliases()`) directly — going through the canonical layer
/// keeps lance-graph free of port-specific data.
#[deprecated(
    note = "use `ogar_vocab::ports::WOA_ALIASES` (or `WoaPort::aliases()`) — the constant lives in OGAR"
)]
pub const WOA_CODEBOOK: &[(&str, u16)] = ogar_vocab::ports::WOA_ALIASES;

#[cfg(test)]
mod tests {
    use super::*;
    use ogar_vocab::class_ids;
    // PortSpec needed in scope for `WoaPort::aliases()` / `::class_id()`
    // (the methods are trait items — codex P1 on PR #570).
    use ogar_vocab::ports::PortSpec;

    #[test]
    fn namespace_and_bridge_id_mirror_the_port() {
        assert_eq!(NAMESPACE, "WorkOrder");
        assert_eq!(WoaPort::NAMESPACE, "WorkOrder");
        assert_eq!(WoaPort::BRIDGE_ID, "woa");
    }

    #[test]
    fn port_resolves_stundenzettel_to_billable_work_entry() {
        // The cross-fork convergence pin (OGAR #93): WoA's billable-hours
        // concept resolves to the SAME canonical id as the planner ports.
        assert_eq!(
            WoaPort::class_id("Stundenzettel"),
            Some(class_ids::BILLABLE_WORK_ENTRY)
        );
        assert_eq!(WoaPort::class_id("Stundenzettel"), Some(0x0103));
        // English synonym collapses to the same id.
        assert_eq!(WoaPort::class_id("TimeEntry"), Some(0x0103));
    }

    #[test]
    fn port_returns_none_for_non_codebook_name() {
        assert_eq!(WoaPort::class_id("NotAConcept"), None);
    }
}
