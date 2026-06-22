//! SMB (small-and-medium-business German office ERP) tenant bridge — a
//! thin type alias over [`crate::bridges::unified::UnifiedBridge`]
//! parameterised by [`ogar_vocab::ports::SmbPort`].
//!
//! The differences between bridges (namespace, bridge_id, public-name
//! → class_id alias table) all come from the OGAR class schema. The
//! `SmbPort` carries the `SMB` namespace + `smb` bridge_id + the
//! German/English alias table (Kunde ≡ Customer, Auftrag ≡ Order,
//! Rechnung ≡ Invoice, Stundenzettel ≡ TimeEntry, …). Sister of
//! [`super::woa_bridge::WoaBridge`]: SMB's `Stundenzettel` resolves to
//! the SAME canonical `BILLABLE_WORK_ENTRY` (`0x0103`) as the WoA and
//! planner ports — cross-fork convergence (OGAR #93).

use crate::bridges::unified::UnifiedBridge;
// `SmbPort::NAMESPACE` / `::aliases()` are `PortSpec` associated items —
// the trait must be in scope for the resolution to work (codex P1 on
// PR #570). Same import in the test module below.
use ogar_vocab::ports::PortSpec;
pub use ogar_vocab::ports::SmbPort;

/// SMB `NamespaceBridge` — alias over the generic harness, locked to the
/// `SMB` namespace via [`SmbPort`].
pub type SmbBridge = UnifiedBridge<SmbPort>;

/// Canonical namespace name for SMB. Mirrors `SmbPort::NAMESPACE` so
/// consumers that import the constant from this module keep building.
pub const NAMESPACE: &str = SmbPort::NAMESPACE;

/// Compatibility shim — re-exports `ogar_vocab::ports::SMB_ALIASES` under
/// a `*_CODEBOOK` name for symmetry with the other per-port bridges. New
/// code should reach for `ogar_vocab::ports::SMB_ALIASES` (or
/// `SmbPort::aliases()`) directly — going through the canonical layer
/// keeps lance-graph free of port-specific data.
#[deprecated(
    note = "use `ogar_vocab::ports::SMB_ALIASES` (or `SmbPort::aliases()`) — the constant lives in OGAR"
)]
pub const SMB_CODEBOOK: &[(&str, u16)] = ogar_vocab::ports::SMB_ALIASES;

#[cfg(test)]
mod tests {
    use super::*;
    use ogar_vocab::class_ids;
    // PortSpec needed in scope for `SmbPort::aliases()` / `::class_id()`
    // (the methods are trait items — codex P1 on PR #570).
    use ogar_vocab::ports::PortSpec;

    #[test]
    fn namespace_and_bridge_id_mirror_the_port() {
        assert_eq!(NAMESPACE, "SMB");
        assert_eq!(SmbPort::NAMESPACE, "SMB");
        assert_eq!(SmbPort::BRIDGE_ID, "smb");
    }

    #[test]
    fn port_resolves_kunde_to_billing_party() {
        assert_eq!(SmbPort::class_id("Kunde"), Some(class_ids::BILLING_PARTY));
        assert_eq!(SmbPort::class_id("Kunde"), Some(0x0204));
        // English synonym collapses to the same id.
        assert_eq!(SmbPort::class_id("Customer"), Some(0x0204));
    }

    #[test]
    fn port_resolves_stundenzettel_to_billable_work_entry() {
        // Same convergence pin as WoA (OGAR #93).
        assert_eq!(
            SmbPort::class_id("Stundenzettel"),
            Some(class_ids::BILLABLE_WORK_ENTRY)
        );
        assert_eq!(SmbPort::class_id("Stundenzettel"), Some(0x0103));
    }

    #[test]
    fn port_returns_none_for_non_codebook_name() {
        assert_eq!(SmbPort::class_id("NotAConcept"), None);
    }
}
