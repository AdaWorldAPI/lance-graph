//! Super-domain layer for the unified bridge surface
//! (per `.claude/plans/super-domain-rbac-tenancy-v1.md` §3.4-§3.8 / D-SDR-2).
//!
//! Adds the level-1 routing/governance unit above OGIT basins:
//!
//! ```text
//! Level 0 — META-ANCHORS                         (interop)
//!     Foundry / OWL / DOLCE / Wikidata cross-walks
//!
//! Level 1 — SUPER DOMAIN  ←── this module        (governance, ~7-10 today)
//!     Healthcare / Science / TicketTool / WorkOrderBilling / OSINT / ...
//!     1 byte; RBAC trust boundary + compliance regime + activation routing
//!
//! Level 2 — OGIT BASIN                           (per-codebook unit)
//!     1 byte (high byte of OwlIdentity); see `unified_bridge::OgitFamily`
//!
//! Level 3 — WITHIN-BASIN SLOT                    (the row identity)
//!     1 byte (low byte of OwlIdentity)
//! ```
//!
//! D-SDR-2 scope: type system + `FAMILY_TO_SUPER_DOMAIN` reverse-lookup.
//! The `UnifiedBridge::authorize()` wiring against these types is D-SDR-5.
//!
//! Current placement: this module sits in `lance-graph-callcenter` alongside
//! `unified_bridge.rs`. Per spec §8 Tier A the canonical landing is
//! `lance-graph-contract::rbac`; the move is a mechanical follow-up.

use crate::unified_bridge::OgitFamily;

// ═══════════════════════════════════════════════════════════════════════════
// SuperDomain — Level 1 activation root + RBAC trust boundary
// ═══════════════════════════════════════════════════════════════════════════

/// 1 byte. Activation root for spreading-activation queries +
/// RBAC enforcement gate (cross-domain queries denied per §13.4 hard-lock
/// partner matrix) + compliance regime tag.
///
/// Eight starter values; 256-slot capacity headroom.
///
/// Promotes the existing `holograph::dntree::WellKnown` constants
/// (CONCEPTS=0x01, ENTITIES=0x02, NSM primes 0x10-0x4F, ...) to
/// first-class business-named activation roots with formal cross-walks.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SuperDomain
{
    #[default]
    Unknown = 0,
    /// FMA, SNOMED, ICD10, RxNorm, LOINC, RadLex, MONDO, HPO, DRON, CHEBI
    Healthcare = 1,
    /// Physics, chemistry, math, materials
    Science = 2,
    /// Genes, sequences, expression, GO
    Genetics = 3,
    /// Quantum-specific (specialized lift from Science)
    QuantumPhysics = 4,
    /// Hiro, HubSpot, ServiceNow, Jira, Zendesk
    TicketTool = 5,
    /// WorkOrder, Billing, Tax, SMB, MRO, MRP, Accounting
    WorkOrderBilling = 6,
    /// Maltego, intel sources, social graph
    Osint = 7,
}

impl SuperDomain
{
    pub const fn raw(self) -> u8
    {
        self as u8
    }

    pub const fn is_known(self) -> bool
    {
        !matches!(self, SuperDomain::Unknown)
    }

    pub const fn as_str(self) -> &'static str
    {
        match self {
            SuperDomain::Unknown => "Unknown",
            SuperDomain::Healthcare => "Healthcare",
            SuperDomain::Science => "Science",
            SuperDomain::Genetics => "Genetics",
            SuperDomain::QuantumPhysics => "QuantumPhysics",
            SuperDomain::TicketTool => "TicketTool",
            SuperDomain::WorkOrderBilling => "WorkOrderBilling",
            SuperDomain::Osint => "Osint",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DolceMarker — upper-ontology category (per §3.5)
// ═══════════════════════════════════════════════════════════════════════════

/// 1 byte. DOLCE upper marker (compressed; only low bits used).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DolceMarker
{
    #[default]
    Unknown = 0,
    Endurant = 1,
    Perdurant = 2,
    Quality = 3,
    Abstract = 4,
}

// ═══════════════════════════════════════════════════════════════════════════
// MetaAnchors — Level 0 cross-walks to formal upper ontologies (§3.5)
// ═══════════════════════════════════════════════════════════════════════════

/// 4 cross-walk pointers — one per upper standard. Each optional.
/// Consulted only when interop with an external system is requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetaAnchors
{
    /// e.g. `Some("PhysicalSystem")`
    pub foundry_object_type: Option<&'static str>,
    /// e.g. `Some("BiomedicalOntology")`
    pub owl_upper_class: Option<&'static str>,
    /// DOLCE Endurant / Perdurant / Quality / Abstract
    pub dolce_marker: DolceMarker,
    /// Wikidata QID (e.g. `Some(11190)` for "medicine")
    pub wikidata_qid: Option<u64>,
}

impl MetaAnchors
{
    pub const EMPTY: Self = Self {
        foundry_object_type: None,
        owl_upper_class: None,
        dolce_marker: DolceMarker::Unknown,
        wikidata_qid: None,
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// ComplianceRegime — Level 1 regulatory tag (per §3.7)
// ═══════════════════════════════════════════════════════════════════════════

/// 1 byte. Tagged on each `SuperDomainEntry`. Drives certification surface
/// (audit log retention, key-rotation cadence, redaction defaults).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ComplianceRegime
{
    #[default]
    None = 0,
    /// Healthcare
    Hipaa = 1,
    /// WorkOrderBilling (financial reporting)
    Sox = 2,
    /// WorkOrderBilling (payment cards)
    PciDss = 3,
    /// Cross-cutting (any super domain with PII)
    Gdpr = 4,
    /// OSINT (tiered by classification)
    OsintClearance = 5,
    /// Science / QuantumPhysics (dual-use research)
    ItarEar = 6,
}

// ═══════════════════════════════════════════════════════════════════════════
// SuperDomainEntry — the static lookup row per super domain
// ═══════════════════════════════════════════════════════════════════════════

/// One entry per `SuperDomain`. Lives in static memory; ~30 bytes per entry
/// × 8 entries ≈ 240 B total.
///
/// `RoleGroup` (§3.6) + `hard_lock_partners` (§13.4) are deferred to a
/// follow-up commit — the minimum-viable shape ships the routing +
/// compliance fields now so `FAMILY_TO_SUPER_DOMAIN` lookups work.
#[derive(Clone, Copy, Debug)]
pub struct SuperDomainEntry
{
    pub super_domain: SuperDomain,
    /// Basins this super domain spans (10-30 per domain typically).
    pub basins: &'static [OgitFamily],
    /// Cross-walks to Foundry / OWL / DOLCE / Wikidata.
    pub meta: MetaAnchors,
    pub label: &'static str,
    /// Compliance regime governing this domain's audit + redaction policy.
    pub compliance: ComplianceRegime,
}

impl SuperDomainEntry
{
    /// Activation drill-down: super domain → constituent basins.
    /// Used by spreading-activation queries ("anything in Healthcare").
    #[inline]
    pub fn activate(&self) -> &'static [OgitFamily]
    {
        self.basins
    }

    /// Cross-walk: this super domain's anchor in an external upper ontology.
    #[inline]
    pub fn cross_walk(&self) -> &MetaAnchors
    {
        &self.meta
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Static registry — placeholder entries for the 8 starter super domains
// ═══════════════════════════════════════════════════════════════════════════

/// Starter `SuperDomainEntry` table. Basin lists + cross-walk QIDs are
/// best-effort placeholders; the canonical values land alongside the OGIT
/// namespace inventory (D-SDR-4 / D-SDR-37).
pub const SUPER_DOMAINS: &[SuperDomainEntry] = &[
    SuperDomainEntry {
        super_domain: SuperDomain::Unknown,
        basins: &[],
        meta: MetaAnchors::EMPTY,
        label: "Unknown",
        compliance: ComplianceRegime::None,
    },
    SuperDomainEntry {
        super_domain: SuperDomain::Healthcare,
        basins: &[],
        meta: MetaAnchors {
            foundry_object_type: Some("BiomedicalSystem"),
            owl_upper_class: Some("BiomedicalOntology"),
            dolce_marker: DolceMarker::Endurant,
            wikidata_qid: Some(11190),
        },
        label: "Healthcare",
        compliance: ComplianceRegime::Hipaa,
    },
    SuperDomainEntry {
        super_domain: SuperDomain::Science,
        basins: &[],
        meta: MetaAnchors {
            foundry_object_type: Some("ScientificDiscipline"),
            owl_upper_class: Some("ResearchOntology"),
            dolce_marker: DolceMarker::Abstract,
            wikidata_qid: Some(336),
        },
        label: "Science",
        compliance: ComplianceRegime::None,
    },
    SuperDomainEntry {
        super_domain: SuperDomain::Genetics,
        basins: &[],
        meta: MetaAnchors {
            foundry_object_type: Some("BiomedicalSystem"),
            owl_upper_class: Some("GeneOntology"),
            dolce_marker: DolceMarker::Endurant,
            wikidata_qid: Some(7162),
        },
        label: "Genetics",
        compliance: ComplianceRegime::Gdpr,
    },
    SuperDomainEntry {
        super_domain: SuperDomain::QuantumPhysics,
        basins: &[],
        meta: MetaAnchors {
            foundry_object_type: Some("ScientificDiscipline"),
            owl_upper_class: Some("PhysicsOntology"),
            dolce_marker: DolceMarker::Abstract,
            wikidata_qid: Some(944),
        },
        label: "QuantumPhysics",
        compliance: ComplianceRegime::ItarEar,
    },
    SuperDomainEntry {
        super_domain: SuperDomain::TicketTool,
        basins: &[],
        meta: MetaAnchors {
            foundry_object_type: Some("WorkflowSystem"),
            owl_upper_class: Some("ProcessOntology"),
            dolce_marker: DolceMarker::Perdurant,
            wikidata_qid: None,
        },
        label: "TicketTool",
        compliance: ComplianceRegime::None,
    },
    SuperDomainEntry {
        super_domain: SuperDomain::WorkOrderBilling,
        basins: &[],
        meta: MetaAnchors {
            foundry_object_type: Some("BusinessSystem"),
            owl_upper_class: Some("FinancialOntology"),
            dolce_marker: DolceMarker::Perdurant,
            wikidata_qid: Some(8413436),
        },
        label: "WorkOrderBilling",
        compliance: ComplianceRegime::Sox,
    },
    SuperDomainEntry {
        super_domain: SuperDomain::Osint,
        basins: &[],
        meta: MetaAnchors {
            foundry_object_type: Some("IntelligenceSystem"),
            owl_upper_class: Some("IntelOntology"),
            dolce_marker: DolceMarker::Perdurant,
            wikidata_qid: Some(7138900),
        },
        label: "OSINT",
        compliance: ComplianceRegime::OsintClearance,
    },
];

/// Reverse lookup: `OgitFamily` (basin) → `SuperDomain` it belongs to.
///
/// Single-member by default (per spec §3.4 tradeoff): each basin picks one
/// primary super domain; cross-cutting basins (e.g., HPO/MONDO straddling
/// Healthcare ↔ Genetics) explicitly activate multiple super-domain
/// entries when queried.
///
/// Populated at hydration time by `NamespaceRegistry::seed_defaults()`;
/// `Unknown` is the safe default for basins not yet classified.
static FAMILY_TO_SUPER_DOMAIN: [SuperDomain; 256] = [SuperDomain::Unknown; 256];

/// Lookup the super domain for a given OGIT basin.
///
/// Returns `SuperDomain::Unknown` for unclassified basins. The mapping is
/// seeded at hydration time; until then this always returns Unknown.
#[inline]
pub fn super_domain_for_family(family: OgitFamily) -> SuperDomain
{
    FAMILY_TO_SUPER_DOMAIN[family.raw() as usize]
}

/// Lookup the `SuperDomainEntry` for a super domain.
#[inline]
pub fn super_domain_entry(sd: SuperDomain) -> &'static SuperDomainEntry
{
    &SUPER_DOMAINS[sd as usize]
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn super_domain_known_predicate()
    {
        assert!(!SuperDomain::Unknown.is_known());
        assert!(SuperDomain::Healthcare.is_known());
        assert!(SuperDomain::TicketTool.is_known());
    }

    #[test]
    fn super_domain_raw_matches_repr_u8()
    {
        assert_eq!(SuperDomain::Unknown.raw(), 0);
        assert_eq!(SuperDomain::Healthcare.raw(), 1);
        assert_eq!(SuperDomain::Osint.raw(), 7);
    }

    #[test]
    fn super_domain_as_str_matches_variant()
    {
        assert_eq!(SuperDomain::Healthcare.as_str(), "Healthcare");
        assert_eq!(SuperDomain::WorkOrderBilling.as_str(), "WorkOrderBilling");
    }

    #[test]
    fn super_domain_default_is_unknown()
    {
        assert_eq!(SuperDomain::default(), SuperDomain::Unknown);
    }

    #[test]
    fn super_domains_table_indexed_by_enum_variant()
    {
        // The static SUPER_DOMAINS table must be ordered such that
        // `SUPER_DOMAINS[sd as usize].super_domain == sd` for every variant.
        // This is the contract `super_domain_entry()` relies on.
        for sd in [
            SuperDomain::Unknown,
            SuperDomain::Healthcare,
            SuperDomain::Science,
            SuperDomain::Genetics,
            SuperDomain::QuantumPhysics,
            SuperDomain::TicketTool,
            SuperDomain::WorkOrderBilling,
            SuperDomain::Osint,
        ] {
            assert_eq!(super_domain_entry(sd).super_domain, sd);
        }
    }

    #[test]
    fn super_domain_entry_carries_expected_compliance()
    {
        assert_eq!(super_domain_entry(SuperDomain::Healthcare).compliance, ComplianceRegime::Hipaa);
        assert_eq!(super_domain_entry(SuperDomain::WorkOrderBilling).compliance, ComplianceRegime::Sox);
        assert_eq!(super_domain_entry(SuperDomain::Osint).compliance, ComplianceRegime::OsintClearance);
        assert_eq!(super_domain_entry(SuperDomain::QuantumPhysics).compliance, ComplianceRegime::ItarEar);
    }

    #[test]
    fn meta_anchors_empty_const()
    {
        let m = MetaAnchors::EMPTY;
        assert!(m.foundry_object_type.is_none());
        assert!(m.owl_upper_class.is_none());
        assert_eq!(m.dolce_marker, DolceMarker::Unknown);
        assert!(m.wikidata_qid.is_none());
    }

    #[test]
    fn family_to_super_domain_unclassified_defaults_to_unknown()
    {
        // Without hydration the table is all-Unknown.
        assert_eq!(super_domain_for_family(OgitFamily(1)), SuperDomain::Unknown);
        assert_eq!(super_domain_for_family(OgitFamily(42)), SuperDomain::Unknown);
        assert_eq!(super_domain_for_family(OgitFamily(255)), SuperDomain::Unknown);
    }
}
