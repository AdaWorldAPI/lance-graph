//! Odoo → OWL → OGIT alignment cache (the "two-version bridge" leg).
//!
//! This is the static seed the odoo-richness harvest lanes
//! (`woa-rs/.claude/odoo/`) and the eventual woa-rs `skr_data` consumer read
//! to attach an OGIT identity — and therefore an inherited *thinking style* —
//! to every odoo concept they steal. It is the **single source**: consumers
//! depend on `lance-graph-callcenter` and call these functions; they MUST NOT
//! mirror the table (the spine goes dirty the moment two copies drift — see
//! the per-OGIT-storage invariant in `super-domain-rbac-tenancy-v1`).
//!
//! ## The chain (two legs, both O(1))
//!
//! ```text
//! odoo class ──owl:equivalentClass──► OWL pivot ──► OGIT family + slot ──► FamilyEntry
//!            resolve_odoo()           (fibo/schema)  (Option B: inherit)    OgitFamilyTable.lookup()
//!            └──────────────── resolve_odoo_to_family(class, &table) ───────────────┘
//!            └──────────────── resolve_odoo_entry(class, &table) ───────► &FamilyEntry
//! ```
//!
//! **Option B (locked):** no new CAM family, no freshly-minted slot per odoo
//! class. Each class *inherits* an existing foundry family + slot via its OWL
//! pivot. The four foundry families are defined authoritatively in
//! `data/family_registry.ttl` (BillingCore 0x61, SMBAccounting 0x62,
//! SmbFoundryCustomer 0x80, SmbFoundryInvoice 0x81); the constants below
//! restate those bytes so the alignment binds to the same basins hydration
//! loads. Classes with no existing family (`stock.move`, `sale.order`,
//! `hr.*`, `account.reconcile.model`, …) resolve to `None` — that is the
//! signal to author a Layer-2 alignment axiom, not to invent a family.

use crate::family_table::{FamilyEntry, OgitFamilyTable, OwlCharacteristics, SchemaKind};
use crate::super_domain::DolceMarker;
use crate::unified_bridge::{OgitFamily, OwlIdentity};

// ═══════════════════════════════════════════════════════════════════════════
// Foundry family bytes — restated from data/family_registry.ttl (Option B)
// ═══════════════════════════════════════════════════════════════════════════

/// `ogit:BillingCore` — billable items / billing surface. familyId 97.
pub const FAMILY_BILLING_CORE: OgitFamily = OgitFamily(0x61);
/// `ogit:SMBAccounting` — double-entry substrate (accounts, posting lines).
/// familyId 98.
pub const FAMILY_SMB_ACCOUNTING: OgitFamily = OgitFamily(0x62);
/// `ogit:SmbFoundryCustomer` — partner / legal-entity master data. familyId 128.
pub const FAMILY_SMB_FOUNDRY_CUSTOMER: OgitFamily = OgitFamily(0x80);
/// `ogit:SmbFoundryInvoice` — invoice / transaction document. familyId 129.
pub const FAMILY_SMB_FOUNDRY_INVOICE: OgitFamily = OgitFamily(0x81);
/// `ogit:ProductCatalog` — product catalogue + pricelist + UoM. familyId 100.
///
/// NOTE: the woa-rs `SAVANTS.md` proposal named `0x63` for this basin, but
/// `0x63` (=99) is already `ogit:MRORepair` in `data/family_registry.ttl`.
/// The lance-graph authoritative registry therefore assigns the next free
/// commercial-cluster byte `0x64` (=100). (Plan odoo-savant-reasoners-v1
/// D-ODOO-SAV-1; deviation from proposal documented there.)
pub const FAMILY_PRODUCT_CATALOG: OgitFamily = OgitFamily(0x64);
/// `ogit:HRFoundation` — employee / org / job / base-contract. familyId 144.
/// Base HR data only; payroll engine is odoo Enterprise (absent).
pub const FAMILY_HR_FOUNDATION: OgitFamily = OgitFamily(0x90);

// ═══════════════════════════════════════════════════════════════════════════
// OwlPivot — the resolved owl:equivalentClass landing (leg 1 output)
// ═══════════════════════════════════════════════════════════════════════════

/// What an odoo model resolves to: the OWL pivot URI plus the OGIT address
/// (family + slot) it inherits under Option B, plus its DOLCE marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwlPivot {
    /// `owl:equivalentClass` target, e.g. `"fibo:LegalEntity"`,
    /// `"fibo:Transaction"`, `"schema:Product"`.
    pub pivot_uri: &'static str,
    /// Inherited foundry basin.
    pub family: OgitFamily,
    /// Inherited within-family slot.
    pub slot: u16,
    /// DOLCE upper marker (Endurant / Perdurant / Quality / Abstract).
    pub dolce: DolceMarker,
}

impl OwlPivot {
    /// The 3-byte OGIT row identity this pivot inherits.
    #[inline]
    pub const fn identity(self) -> OwlIdentity {
        OwlIdentity::new(self.family, self.slot)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// dolce_odoo — DOLCE marker from odoo class suffix rules
// ═══════════════════════════════════════════════════════════════════════════

/// Classify an odoo class onto its DOLCE upper marker from structural suffix
/// rules. Independent of the seed table so unmapped-but-recognisable classes
/// (e.g. `sale.order`) still get a marker the harvest lanes can record.
///
/// - **Perdurant** — transactional events / processes: `*.move`,
///   `*.move.line`, `*.payment`, `*.order`, `*.order.line`, bank statements,
///   stock moves, pickings.
/// - **Abstract** — rules / classifications / models: `*.tax`,
///   `*.fiscal.position`, `*.reconcile.model`, `*.payment.term`.
/// - **Endurant** — persistent master-data objects: `res.*`, `*.account`,
///   `product.*`.
/// - **Unknown** — no rule matched.
pub fn dolce_odoo(class: &str) -> DolceMarker {
    if class.ends_with(".move")
        || class.ends_with(".move.line")
        || class.ends_with(".payment")
        || class.ends_with(".order")
        || class.ends_with(".order.line")
        || class.ends_with("bank.statement")
        || class.ends_with(".picking")
        || class == "stock.move"
    {
        return DolceMarker::Perdurant;
    }
    if class.ends_with(".tax")
        || class.ends_with("fiscal.position")
        || class.ends_with("reconcile.model")
        || class.ends_with("payment.term")
    {
        return DolceMarker::Abstract;
    }
    if class.starts_with("uom.") {
        // Units of measure are Qualities in DOLCE (dimensions of comparison).
        return DolceMarker::Quality;
    }
    if class.starts_with("res.") || class.ends_with(".account") || class.starts_with("product.") {
        return DolceMarker::Endurant;
    }
    DolceMarker::Unknown
}

// ═══════════════════════════════════════════════════════════════════════════
// family_default_style — inherited ThinkingStyle cluster per OGIT family
// (Plan odoo-savant-reasoners-v1 D-ODOO-SAV-3). SAVANTS.md inherits each
// savant's style from its family; this is the authoritative family→cluster map.
// ═══════════════════════════════════════════════════════════════════════════

pub use lance_graph_contract::thinking::StyleCluster;

/// The default `StyleCluster` a Savant inherits from its OGIT family. Pinned
/// per `SAVANTS.md` § "OGIT family map ... and inherited style":
///
/// - `0x60` WorkOrderCore       → Direct      (task execution)
/// - `0x61` BillingCore         → Analytical  (pricing math)
/// - `0x62` SMBAccounting       → Analytical  (ledger reasoning)
/// - `0x64` ProductCatalog      → Analytical  (catalogue / pricing)
/// - `0x80` SmbFoundryCustomer  → Empathic    (relationship + trust)
/// - `0x81` SmbFoundryInvoice   → Direct      (transaction processing)
/// - `0x90` HRFoundation        → Empathic    (people / org)
///
/// Returns `None` for families with no pinned default (caller proposes a
/// cluster with rationale, per the BRIEFING delegation discipline).
pub fn family_default_style(family: OgitFamily) -> Option<StyleCluster> {
    match family.raw() {
        0x60 => Some(StyleCluster::Direct),
        0x61 => Some(StyleCluster::Analytical),
        0x62 => Some(StyleCluster::Analytical),
        0x64 => Some(StyleCluster::Analytical),
        0x80 => Some(StyleCluster::Empathic),
        0x81 => Some(StyleCluster::Direct),
        0x90 => Some(StyleCluster::Empathic),
        _ => None,
    }
}

/// Both legs + style: resolve an odoo class all the way to the
/// `StyleCluster` its inherited family carries. `None` when the class is
/// unmapped or its family has no pinned default. O(1).
pub fn resolve_odoo_style(class: &str) -> Option<StyleCluster> {
    let pivot = resolve_odoo(class)?;
    family_default_style(pivot.family)
}

// ═══════════════════════════════════════════════════════════════════════════
// Seed — the realized rows the BRIEFING enumerates (res.partner, account.*,
// product.*, SKR). Each row owns a stable slot within its foundry family.
// ═══════════════════════════════════════════════════════════════════════════

struct OdooSeedRow {
    /// odoo model name (the resolvable key).
    odoo_class: &'static str,
    /// `owl:equivalentClass` pivot URI.
    pivot_uri: &'static str,
    family: OgitFamily,
    slot: u16,
    kind: SchemaKind,
    dolce: DolceMarker,
    /// Canonical OGIT label this slot carries inside the foundry family table.
    label_uri: &'static str,
    /// `dcterms:source` lineage stamped into the `FamilyEntry`.
    provenance: &'static str,
}

impl OdooSeedRow {
    /// Build the inline `FamilyEntry` this row populates in its family table.
    fn entry(&self) -> FamilyEntry {
        FamilyEntry {
            label_uri: self.label_uri,
            kind: self.kind,
            owl_characteristics: OwlCharacteristics::EMPTY,
            dolce_marker: self.dolce,
            axiom_blob: &[],
            provenance: self.provenance,
            verbs: &[],
        }
    }
}

/// The realized alignment rows. Small fixed table — `resolve_odoo`'s linear
/// scan over a handful of entries is effectively O(1).
static ODOO_SEED: &[OdooSeedRow] = &[
    OdooSeedRow {
        odoo_class: "res.partner",
        pivot_uri: "fibo:LegalEntity",
        family: FAMILY_SMB_FOUNDRY_CUSTOMER,
        slot: 1,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Endurant,
        label_uri: "ogit.SMB:Customer",
        provenance: "odoo res.partner (company facet) =owl:equivalentClass=> fibo:LegalEntity",
    },
    OdooSeedRow {
        odoo_class: "account.move",
        pivot_uri: "fibo:Transaction",
        family: FAMILY_SMB_FOUNDRY_INVOICE,
        slot: 1,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Perdurant,
        label_uri: "ogit.SMB:Invoice",
        provenance: "odoo account.move =owl:equivalentClass=> fibo:Transaction",
    },
    OdooSeedRow {
        odoo_class: "account.move.line",
        pivot_uri: "fibo:JournalEntryLine",
        family: FAMILY_SMB_ACCOUNTING,
        slot: 1,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Perdurant,
        label_uri: "ogit.SMBAccounting:JournalEntryLine",
        provenance: "odoo account.move.line =owl:equivalentClass=> fibo:JournalEntryLine",
    },
    OdooSeedRow {
        odoo_class: "account.account",
        pivot_uri: "fibo:Account",
        family: FAMILY_SMB_ACCOUNTING,
        slot: 2,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Endurant,
        label_uri: "ogit.SMBAccounting:Account",
        provenance: "odoo account.account =owl:equivalentClass=> fibo:Account",
    },
    OdooSeedRow {
        odoo_class: "account.account.template",
        pivot_uri: "fibo:Account",
        family: FAMILY_SMB_ACCOUNTING,
        slot: 3,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Endurant,
        label_uri: "ogit.SMBAccounting:SkrAccount",
        provenance: "SKR03/04 chart concept (odoo account.account.template) => fibo:Account",
    },
    OdooSeedRow {
        odoo_class: "product.template",
        pivot_uri: "schema:Product",
        family: FAMILY_BILLING_CORE,
        slot: 1,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Endurant,
        label_uri: "ogit.Billing:Product",
        provenance: "odoo product.template =owl:equivalentClass=> schema:Product",
    },
    OdooSeedRow {
        odoo_class: "product.product",
        pivot_uri: "schema:Product",
        family: FAMILY_BILLING_CORE,
        slot: 2,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Endurant,
        label_uri: "ogit.Billing:ProductVariant",
        provenance: "odoo product.product (variant) =owl:equivalentClass=> schema:Product",
    },
    // ── ProductCatalog 0x64 — catalogue STRUCTURE (pricing + measurement) ──
    // product.template / product.product stay on BillingCore (0x61): they are
    // billable ITEMS. The pricelist / UoM concepts are catalogue STRUCTURE and
    // get their own basin so L8's PricelistAssignmentAgent has a real family
    // instead of None. (Deviation from SAVANTS.md noted on FAMILY_PRODUCT_CATALOG.)
    OdooSeedRow {
        odoo_class: "product.pricelist",
        pivot_uri: "schema:PriceSpecification",
        family: FAMILY_PRODUCT_CATALOG,
        slot: 1,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Abstract,
        label_uri: "ogit.ProductCatalog:Pricelist",
        provenance: "odoo product.pricelist =owl:equivalentClass=> schema:PriceSpecification",
    },
    OdooSeedRow {
        odoo_class: "product.pricelist.item",
        pivot_uri: "schema:UnitPriceSpecification",
        family: FAMILY_PRODUCT_CATALOG,
        slot: 2,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Abstract,
        label_uri: "ogit.ProductCatalog:PricelistRule",
        provenance: "odoo product.pricelist.item =owl:equivalentClass=> schema:UnitPriceSpecification",
    },
    OdooSeedRow {
        // uom.uom ties into the QUDT Foundation namespace (qudt:Unit) — the
        // measurement spine the bO-4 hydrator already loads.
        odoo_class: "uom.uom",
        pivot_uri: "qudt:Unit",
        family: FAMILY_PRODUCT_CATALOG,
        slot: 3,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Quality,
        label_uri: "ogit.ProductCatalog:UnitOfMeasure",
        provenance: "odoo uom.uom =owl:equivalentClass=> qudt:Unit",
    },
    // ── HRFoundation 0x90 — employee / org / job / base-contract ──
    // Payroll ENGINE is odoo Enterprise (absent): only base HR data aligns here.
    OdooSeedRow {
        odoo_class: "hr.employee",
        pivot_uri: "vcard:Individual",
        family: FAMILY_HR_FOUNDATION,
        slot: 1,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Endurant,
        label_uri: "ogit.HR:Employee",
        provenance: "odoo hr.employee =owl:equivalentClass=> vcard:Individual",
    },
    OdooSeedRow {
        odoo_class: "hr.department",
        pivot_uri: "org:OrganizationalUnit",
        family: FAMILY_HR_FOUNDATION,
        slot: 2,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Endurant,
        label_uri: "ogit.HR:Department",
        provenance: "odoo hr.department =owl:equivalentClass=> org:OrganizationalUnit",
    },
    OdooSeedRow {
        odoo_class: "hr.job",
        pivot_uri: "org:Role",
        family: FAMILY_HR_FOUNDATION,
        slot: 3,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Abstract,
        label_uri: "ogit.HR:Job",
        provenance: "odoo hr.job =owl:equivalentClass=> org:Role",
    },
    OdooSeedRow {
        // Base employment contract only — payroll computation is Enterprise.
        odoo_class: "hr.contract",
        pivot_uri: "fibo:Contract",
        family: FAMILY_HR_FOUNDATION,
        slot: 4,
        kind: SchemaKind::Entity,
        dolce: DolceMarker::Abstract,
        label_uri: "ogit.HR:EmploymentContract",
        provenance: "odoo hr.contract (base, payroll is Enterprise/absent) =owl:equivalentClass=> fibo:Contract",
    },
];

// ═══════════════════════════════════════════════════════════════════════════
// Resolution surface — the three BRIEFING-named functions + entry lookup
// ═══════════════════════════════════════════════════════════════════════════

/// Leg 1: resolve an odoo class to its OWL pivot + inherited OGIT address.
///
/// Exact seed match first; then a `product.*` prefix fallback onto the
/// generic `product.template` row (schema:Product / BillingCore). Returns
/// `None` for any class with no existing family (the Layer-2-axiom signal).
pub fn resolve_odoo(class: &str) -> Option<OwlPivot> {
    if let Some(row) = ODOO_SEED.iter().find(|r| r.odoo_class == class) {
        return Some(OwlPivot {
            pivot_uri: row.pivot_uri,
            family: row.family,
            slot: row.slot,
            dolce: row.dolce,
        });
    }
    // Unseen product subtype → inherit the generic product slot.
    if class.starts_with("product.") {
        let row = ODOO_SEED
            .iter()
            .find(|r| r.odoo_class == "product.template")?;
        return Some(OwlPivot {
            pivot_uri: row.pivot_uri,
            family: row.family,
            slot: row.slot,
            dolce: dolce_odoo(class),
        });
    }
    None
}

/// Both legs: resolve `class` to `(family, slot)` and confirm the slot is
/// **live** in the supplied family's hydrated table. Returns `None` when the
/// class is unmapped, the caller passed a different family's table, or the
/// slot is not (yet) hydrated. One static lookup + one hash probe — O(1).
pub fn resolve_odoo_to_family(class: &str, table: &OgitFamilyTable) -> Option<(OgitFamily, u16)> {
    let pivot = resolve_odoo(class)?;
    if pivot.family != table.family {
        return None;
    }
    table
        .lookup(pivot.identity())
        .map(|_| (pivot.family, pivot.slot))
}

/// Both legs, landing on the inline `FamilyEntry` (carries the DOLCE marker,
/// OWL characteristics, label, and provenance the harvest lanes read for the
/// inherited thinking style). Same O(1) guarantees as
/// [`resolve_odoo_to_family`].
pub fn resolve_odoo_entry<'t>(class: &str, table: &'t OgitFamilyTable) -> Option<&'t FamilyEntry> {
    let pivot = resolve_odoo(class)?;
    if pivot.family != table.family {
        return None;
    }
    table.lookup(pivot.identity())
}

/// Populate `table` with the odoo-aligned slots that belong to its family.
/// Idempotent. The TTL overlay path + tests call this so the leg-2 lookup in
/// [`resolve_odoo_to_family`] has live slots to confirm against.
pub fn seed_family_table(table: &mut OgitFamilyTable) {
    for row in ODOO_SEED {
        if row.family == table.family {
            table.set(row.slot, row.entry());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_bytes_match_registry_ttl() {
        // Option B binding: these MUST equal the familyId values in
        // data/family_registry.ttl or the alignment lands in the wrong basin.
        assert_eq!(FAMILY_BILLING_CORE.raw(), 97);
        assert_eq!(FAMILY_SMB_ACCOUNTING.raw(), 98);
        assert_eq!(FAMILY_SMB_FOUNDRY_CUSTOMER.raw(), 128);
        assert_eq!(FAMILY_SMB_FOUNDRY_INVOICE.raw(), 129);
    }

    #[test]
    fn seed_rows_resolve_to_expected_pivots() {
        assert_eq!(
            resolve_odoo("res.partner").unwrap().pivot_uri,
            "fibo:LegalEntity"
        );
        assert_eq!(
            resolve_odoo("account.move").unwrap().pivot_uri,
            "fibo:Transaction"
        );
        assert_eq!(
            resolve_odoo("account.move.line").unwrap().pivot_uri,
            "fibo:JournalEntryLine"
        );
        assert_eq!(
            resolve_odoo("account.account").unwrap().pivot_uri,
            "fibo:Account"
        );
        assert_eq!(
            resolve_odoo("product.template").unwrap().pivot_uri,
            "schema:Product"
        );
    }

    #[test]
    fn seed_rows_inherit_expected_families() {
        assert_eq!(
            resolve_odoo("res.partner").unwrap().family,
            FAMILY_SMB_FOUNDRY_CUSTOMER
        );
        assert_eq!(
            resolve_odoo("account.move").unwrap().family,
            FAMILY_SMB_FOUNDRY_INVOICE
        );
        assert_eq!(
            resolve_odoo("account.move.line").unwrap().family,
            FAMILY_SMB_ACCOUNTING
        );
        assert_eq!(
            resolve_odoo("account.account").unwrap().family,
            FAMILY_SMB_ACCOUNTING
        );
        assert_eq!(
            resolve_odoo("product.product").unwrap().family,
            FAMILY_BILLING_CORE
        );
    }

    #[test]
    fn unmapped_classes_return_none() {
        // The "needs a Layer-2 alignment axiom" signal — NOT a minted family.
        // NOTE: hr.employee USED to be here but D-ODOO-SAV-1 mapped it to
        // HRFoundation (0x90); these are the classes that genuinely stay None.
        assert!(resolve_odoo("stock.move").is_none());
        assert!(resolve_odoo("sale.order").is_none());
        assert!(resolve_odoo("account.reconcile.model").is_none());
        assert!(resolve_odoo("account.analytic.distribution.model").is_none());
    }

    #[test]
    fn unseen_product_subtype_inherits_generic_slot() {
        let p = resolve_odoo("product.category").expect("product.* prefix fallback");
        assert_eq!(p.pivot_uri, "schema:Product");
        assert_eq!(p.family, FAMILY_BILLING_CORE);
        assert_eq!(p.slot, 1); // product.template's slot
    }

    #[test]
    fn dolce_marker_suffix_rules() {
        assert_eq!(dolce_odoo("account.move"), DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("account.move.line"), DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("sale.order"), DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("account.tax"), DolceMarker::Abstract);
        assert_eq!(dolce_odoo("account.fiscal.position"), DolceMarker::Abstract);
        assert_eq!(dolce_odoo("res.partner"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("account.account"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("product.template"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("ir.cron"), DolceMarker::Unknown);
    }

    #[test]
    fn end_to_end_against_live_table() {
        // SMBAccounting basin: seed it, then chain odoo class → family/slot.
        let mut table = OgitFamilyTable::empty(FAMILY_SMB_ACCOUNTING);
        seed_family_table(&mut table);
        assert!(!table.is_empty());

        let (fam, slot) = resolve_odoo_to_family("account.account", &table).expect("live slot");
        assert_eq!(fam, FAMILY_SMB_ACCOUNTING);
        assert_eq!(slot, 2);

        let entry = resolve_odoo_entry("account.move.line", &table).expect("live entry");
        assert_eq!(entry.label_uri, "ogit.SMBAccounting:JournalEntryLine");
        assert_eq!(entry.dolce_marker, DolceMarker::Perdurant);
    }

    #[test]
    fn wrong_family_table_yields_none() {
        // res.partner inherits SmbFoundryCustomer; confirming against an
        // SMBAccounting table must NOT match (and must not panic).
        let mut table = OgitFamilyTable::empty(FAMILY_SMB_ACCOUNTING);
        seed_family_table(&mut table);
        assert!(resolve_odoo_to_family("res.partner", &table).is_none());
        assert!(resolve_odoo_entry("res.partner", &table).is_none());
    }

    #[test]
    fn unhydrated_slot_is_not_live() {
        // Right family, but the table was never seeded → leg-2 lookup misses.
        let table = OgitFamilyTable::empty(FAMILY_SMB_ACCOUNTING);
        assert!(resolve_odoo_to_family("account.account", &table).is_none());
    }

    // ── D-ODOO-SAV-1: ProductCatalog (0x64) + HRFoundation (0x90) basins ──

    #[test]
    fn new_family_bytes_are_free_in_registry() {
        // ProductCatalog 0x64=100 (NOT the proposed 0x63=99 which is
        // MRORepair); HRFoundation 0x90=144. Both must be the values the
        // family_registry.ttl rows added in D-ODOO-SAV-1 carry.
        assert_eq!(FAMILY_PRODUCT_CATALOG.raw(), 100);
        assert_eq!(FAMILY_HR_FOUNDATION.raw(), 144);
        // Guard the deviation: 0x63 (=99) is MRORepair, NOT ProductCatalog.
        assert_ne!(FAMILY_PRODUCT_CATALOG.raw(), 0x63);
    }

    #[test]
    fn product_catalogue_structure_resolves_to_productcatalog() {
        // Pricing + UoM STRUCTURE lands on ProductCatalog (0x64)...
        let pl = resolve_odoo("product.pricelist").expect("pricelist seeded");
        assert_eq!(pl.family, FAMILY_PRODUCT_CATALOG);
        assert_eq!(pl.pivot_uri, "schema:PriceSpecification");
        assert_eq!(pl.dolce, DolceMarker::Abstract);

        let item = resolve_odoo("product.pricelist.item").expect("pricelist item seeded");
        assert_eq!(item.family, FAMILY_PRODUCT_CATALOG);

        let uom = resolve_odoo("uom.uom").expect("uom seeded");
        assert_eq!(uom.family, FAMILY_PRODUCT_CATALOG);
        assert_eq!(uom.pivot_uri, "qudt:Unit"); // ties into the bO-4 QUDT namespace
        assert_eq!(uom.dolce, DolceMarker::Quality);
    }

    #[test]
    fn billable_items_stay_on_billing_core() {
        // Deviation guard: product.template / product.product are billable
        // ITEMS and MUST stay on BillingCore (0x61), NOT move to ProductCatalog.
        assert_eq!(
            resolve_odoo("product.template").unwrap().family,
            FAMILY_BILLING_CORE
        );
        assert_eq!(
            resolve_odoo("product.product").unwrap().family,
            FAMILY_BILLING_CORE
        );
    }

    #[test]
    fn hr_base_classes_resolve_to_hrfoundation() {
        // hr.* resolved None before D-ODOO-SAV-1; now they land on 0x90.
        let emp = resolve_odoo("hr.employee").expect("hr.employee seeded");
        assert_eq!(emp.family, FAMILY_HR_FOUNDATION);
        assert_eq!(emp.pivot_uri, "vcard:Individual");
        assert_eq!(emp.dolce, DolceMarker::Endurant);

        assert_eq!(
            resolve_odoo("hr.department").unwrap().pivot_uri,
            "org:OrganizationalUnit"
        );
        assert_eq!(resolve_odoo("hr.job").unwrap().family, FAMILY_HR_FOUNDATION);
        assert_eq!(
            resolve_odoo("hr.contract").unwrap().pivot_uri,
            "fibo:Contract"
        );
    }

    #[test]
    fn end_to_end_productcatalog_live_table() {
        let mut table = OgitFamilyTable::empty(FAMILY_PRODUCT_CATALOG);
        seed_family_table(&mut table);
        assert!(!table.is_empty());
        let (fam, slot) = resolve_odoo_to_family("uom.uom", &table).expect("live uom slot");
        assert_eq!(fam, FAMILY_PRODUCT_CATALOG);
        assert_eq!(slot, 3);
    }

    #[test]
    fn end_to_end_hrfoundation_live_table() {
        let mut table = OgitFamilyTable::empty(FAMILY_HR_FOUNDATION);
        seed_family_table(&mut table);
        let entry = resolve_odoo_entry("hr.employee", &table).expect("live hr.employee entry");
        assert_eq!(entry.label_uri, "ogit.HR:Employee");
        assert_eq!(entry.dolce_marker, DolceMarker::Endurant);
    }

    #[test]
    fn classes_still_none_after_d1() {
        // D-ODOO-SAV-1 added ProductCatalog + HRFoundation only. The genuinely
        // cross-cutting classes stay None (Layer-2 axioms in D-ODOO-SAV-2, but
        // those record semantics in TTL, not a new foundry family).
        assert!(resolve_odoo("stock.move").is_none());
        assert!(resolve_odoo("account.analytic.distribution.model").is_none());
        assert!(resolve_odoo("account.account.tag").is_none());
    }

    // ── D-ODOO-SAV-3: family → inherited StyleCluster ──

    #[test]
    fn family_default_style_pins_savant_clusters() {
        assert_eq!(
            family_default_style(FAMILY_BILLING_CORE),
            Some(StyleCluster::Analytical)
        );
        assert_eq!(
            family_default_style(FAMILY_SMB_ACCOUNTING),
            Some(StyleCluster::Analytical)
        );
        assert_eq!(
            family_default_style(FAMILY_PRODUCT_CATALOG),
            Some(StyleCluster::Analytical)
        );
        assert_eq!(
            family_default_style(FAMILY_SMB_FOUNDRY_CUSTOMER),
            Some(StyleCluster::Empathic)
        );
        assert_eq!(
            family_default_style(FAMILY_SMB_FOUNDRY_INVOICE),
            Some(StyleCluster::Direct)
        );
        assert_eq!(
            family_default_style(FAMILY_HR_FOUNDATION),
            Some(StyleCluster::Empathic)
        );
        // Unpinned family → None (caller proposes with rationale).
        assert_eq!(family_default_style(OgitFamily(0xFE)), None);
    }

    #[test]
    fn resolve_odoo_style_chains_class_to_cluster() {
        // res.partner → SmbFoundryCustomer (0x80) → Empathic.
        assert_eq!(
            resolve_odoo_style("res.partner"),
            Some(StyleCluster::Empathic)
        );
        // product.pricelist → ProductCatalog (0x64) → Analytical.
        assert_eq!(
            resolve_odoo_style("product.pricelist"),
            Some(StyleCluster::Analytical)
        );
        // hr.employee → HRFoundation (0x90) → Empathic.
        assert_eq!(
            resolve_odoo_style("hr.employee"),
            Some(StyleCluster::Empathic)
        );
        // unmapped class → None.
        assert_eq!(resolve_odoo_style("stock.move"), None);
    }

    #[test]
    fn uom_dolce_suffix_rule() {
        assert_eq!(dolce_odoo("uom.uom"), DolceMarker::Quality);
        assert_eq!(dolce_odoo("uom.category"), DolceMarker::Quality);
    }
}
