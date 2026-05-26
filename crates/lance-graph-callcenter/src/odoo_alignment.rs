//! `OdooAlignment` — the precomputed odoo → OWL → OGIT-family resolution cache.
//!
//! ## Why this module exists
//!
//! The four-way alignment chain (odoo → OWL/DOLCE/FIBO → OGIT → lance-graph)
//! is specified in `woa-rs/.claude/reference/four_way_alignment_seam.md`. The
//! addendum ("CAM encoding & O(1) odoo inheritance") pins the precise reading
//! of **Seam decision 1 / Option B**:
//!
//! > odoo inherits; it does not alias. An odoo class owns **no** codebook
//! > slot — it inherits the OGIT family's slot through the OWL equivalence
//! > pivot. The inheritance chain is **OGIT → OWL → odoo** (root → pivot →
//! > leaf). The chain is walked **once** at hydration and the result is baked
//! > into the slot's inheritance bitmap → **O(1)** thereafter. No new family,
//! > no new slots, no query-time traversal.
//!
//! The `OGIT → OWL → CAM` machinery already exists in this crate
//! (`OgitFamilyTable::lookup(OwlIdentity) -> Option<&FamilyEntry>` is the O(1)
//! OWL→OGIT-family resolver). What was missing is the **odoo → OWL leg**: the
//! precomputed table that says *which OWL pivot (hence which OGIT family slot)
//! each odoo class resolves to*. This module fills exactly that leg, "to be on
//! the safe side" — a static, redundant resolver so the equivalence is baked
//! in rather than re-walked.
//!
//! ## Shape
//!
//! ```text
//! resolve_odoo("odoo:res.partner.Company")
//!     │  (this module — the odoo→OWL leg, O(1) sorted-slice match)
//!     ▼
//! OwlIdentity { family: SMB_FOUNDRY_CUSTOMER, slot: SLOT_LEGAL_ENTITY }
//!     │  (existing machinery — the OWL→OGIT-family leg)
//!     ▼ table.lookup(owl)
//! &FamilyEntry   ← the OGIT codebook slot the odoo class inherits
//! ```
//!
//! `resolve_odoo_to_family` chains the two legs with **no graph walk at call
//! time** — the equivalence is precomputed here and the family table probe is
//! a single hash lookup.
//!
//! ## Option B (no new CAM family)
//!
//! Per the seam, odoo is an *extraction source*, not a *domain*. It is NOT
//! allocated a CAM family. Every entry below points its `OwlIdentity` at an
//! **existing** WorkOrderBilling / SMB family (see `family_registry.ttl`:
//! `WorkOrderCore` 0x60, `BillingCore` 0x61, `SMBAccounting` 0x62,
//! `SmbFoundryCustomer` 0x80, `SmbFoundryInvoice` 0x81). This module adds no
//! variant, no slot, and does not perturb `OgitFamilyTable`'s layout.

use crate::family_table::{FamilyEntry, OgitFamilyTable};
use crate::super_domain::DolceMarker;
use crate::unified_bridge::{OgitFamily, OwlIdentity};

// ═══════════════════════════════════════════════════════════════════════════
// Canonical OGIT families the odoo classes inherit into (Option B targets)
// ═══════════════════════════════════════════════════════════════════════════
//
// Family IDs are the canonical WorkOrderBilling / SMB basins declared in
// `data/family_registry.ttl`. We DO NOT invent families — these already exist
// and already resolve through the hydrated `OgitFamilyTable`.

/// `ogit:SmbFoundryCustomer` (0x80) — the partner / legal-entity basin.
const FAM_SMB_CUSTOMER: OgitFamily = OgitFamily(0x80);
/// `ogit:SmbFoundryInvoice` (0x81) — the document / transaction basin.
const FAM_SMB_INVOICE: OgitFamily = OgitFamily(0x81);
/// `ogit:SMBAccounting` (0x62) — the ledger / chart-of-accounts basin.
const FAM_SMB_ACCOUNTING: OgitFamily = OgitFamily(0x62);
/// `ogit:BillingCore` (0x61) — the product / billing-catalogue basin.
const FAM_BILLING_CORE: OgitFamily = OgitFamily(0x61);

// Within-family slot identities for the OWL pivots. These are the OWL/FIBO/
// schema.org pivot classes the seam names; the slot number is a stable, small
// within-family index (the `OwlIdentity::slot` the family table is keyed by).
// The pivot a slot stands for is recorded in the doc-comment so a reader can
// trace the equivalence-class lineage back to the seam.

/// `fibo:LegalEntity` pivot (res.partner.Company).
const SLOT_LEGAL_ENTITY: u16 = 0x01;
/// `vcard:Individual` pivot (res.partner.Individual).
const SLOT_INDIVIDUAL: u16 = 0x02;
/// `fibo:Transaction` pivot (account.move journal entry).
const SLOT_TRANSACTION: u16 = 0x03;
/// `fibo:Account` pivot (account.account chart-of-accounts node + SKR concepts).
const SLOT_ACCOUNT: u16 = 0x04;
/// `schema:Product` pivot (product.product / product.template).
const SLOT_PRODUCT: u16 = 0x05;
/// `fibo:JournalEntryLine` pivot (account.move.line).
const SLOT_JOURNAL_LINE: u16 = 0x06;

// ═══════════════════════════════════════════════════════════════════════════
// OdooAlignment — one precomputed odoo→OWL row
// ═══════════════════════════════════════════════════════════════════════════

/// One precomputed odoo → OWL pivot mapping. `'static` because the table is
/// baked at compile time (Option B = no runtime authoring).
///
/// The `owl` field is the OWL equivalence pivot expressed as the
/// `OwlIdentity` (OGIT family + within-family slot) it inherits — i.e. the key
/// the existing `OgitFamilyTable` is indexed by. Resolving an odoo class to
/// its OGIT codebook slot is therefore two O(1) probes:
/// `resolve_odoo` (this table) → `OgitFamilyTable::lookup` (existing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OdooAlignment {
    /// odoo class IRI, e.g. `"odoo:res.partner.Company"`. Sorted-slice key.
    pub odoo_class: &'static str,
    /// The OWL pivot expressed as the OGIT-family `OwlIdentity` it inherits.
    pub owl: OwlIdentity,
    /// Human-readable OWL pivot label for provenance / audit
    /// (e.g. `"fibo:LegalEntity"`). Not used for lookup.
    pub owl_pivot_label: &'static str,
    /// DOLCE upper category from the suffix classifier (Seam decision 2).
    pub dolce: DolceMarker,
}

// ═══════════════════════════════════════════════════════════════════════════
// The table — seeded from the seam's worked classes + the SKR anchor
// ═══════════════════════════════════════════════════════════════════════════
//
// MUST stay sorted by `odoo_class` so `resolve_odoo` can binary-search. The
// `table_is_sorted` test enforces this; add new rows in lexicographic order.

/// Static odoo → OWL pivot table (Option B aliases into existing OGIT
/// families). Sorted by `odoo_class` for O(log n) binary search; n is small
/// (~12) so this is effectively O(1) and const-friendly.
pub const ODOO_ALIGNMENTS: &[OdooAlignment] = &[
    OdooAlignment {
        odoo_class: "odoo:account.account",
        owl: OwlIdentity::new(FAM_SMB_ACCOUNTING, SLOT_ACCOUNT),
        owl_pivot_label: "fibo:Account",
        dolce: DolceMarker::Endurant,
    },
    OdooAlignment {
        odoo_class: "odoo:account.move",
        owl: OwlIdentity::new(FAM_SMB_INVOICE, SLOT_TRANSACTION),
        owl_pivot_label: "fibo:Transaction",
        dolce: DolceMarker::Perdurant,
    },
    OdooAlignment {
        odoo_class: "odoo:account.move.line",
        owl: OwlIdentity::new(FAM_SMB_INVOICE, SLOT_JOURNAL_LINE),
        owl_pivot_label: "fibo:JournalEntryLine",
        dolce: DolceMarker::Perdurant,
    },
    OdooAlignment {
        odoo_class: "odoo:product.product",
        owl: OwlIdentity::new(FAM_BILLING_CORE, SLOT_PRODUCT),
        owl_pivot_label: "schema:Product",
        dolce: DolceMarker::Endurant,
    },
    OdooAlignment {
        odoo_class: "odoo:product.template",
        owl: OwlIdentity::new(FAM_BILLING_CORE, SLOT_PRODUCT),
        owl_pivot_label: "schema:Product",
        // product.template is odoo's MASTER product record (not a config
        // template) — the explicit Endurant override (Seam decision 2 table).
        dolce: DolceMarker::Endurant,
    },
    OdooAlignment {
        odoo_class: "odoo:res.partner.Company",
        owl: OwlIdentity::new(FAM_SMB_CUSTOMER, SLOT_LEGAL_ENTITY),
        owl_pivot_label: "fibo:LegalEntity",
        dolce: DolceMarker::Endurant,
    },
    OdooAlignment {
        odoo_class: "odoo:res.partner.Individual",
        owl: OwlIdentity::new(FAM_SMB_CUSTOMER, SLOT_INDIVIDUAL),
        owl_pivot_label: "vcard:Individual",
        dolce: DolceMarker::Endurant,
    },
];

// SKR chart-of-accounts concepts. The seam's SKR anchor: the SKR
// `account.account` chart concepts all resolve to `fibo:Account` (same OWL
// pivot as the bare `odoo:account.account`). These are kept as distinct rows
// so a SKR-specific concept IRI resolves directly without the caller having to
// normalise to `odoo:account.account` first.
//
// Maintained as a separate slice and merged at lookup-fallback time so the
// primary `ODOO_ALIGNMENTS` slice stays the seam's worked-class core.

/// SKR chart-of-accounts concept IRIs → `fibo:Account` pivot. Sorted.
pub const SKR_ACCOUNT_CONCEPTS: &[OdooAlignment] = &[
    OdooAlignment {
        odoo_class: "odoo:account.account.skr03",
        owl: OwlIdentity::new(FAM_SMB_ACCOUNTING, SLOT_ACCOUNT),
        owl_pivot_label: "fibo:Account",
        dolce: DolceMarker::Endurant,
    },
    OdooAlignment {
        odoo_class: "odoo:account.account.skr04",
        owl: OwlIdentity::new(FAM_SMB_ACCOUNTING, SLOT_ACCOUNT),
        owl_pivot_label: "fibo:Account",
        dolce: DolceMarker::Endurant,
    },
];

// ═══════════════════════════════════════════════════════════════════════════
// Resolution — the odoo→OWL leg (O(1)) and the chained odoo→OWL→family leg
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve an odoo class IRI to its OWL pivot `OwlIdentity` (the odoo→OWL
/// leg). O(log n) binary search over a ~12-entry static table; n is tiny so
/// this is effectively O(1). No graph walk — the equivalence is baked in
/// (per the addendum).
///
/// Searches the worked-class core first, then the SKR concept slice.
///
/// Returns `None` for any odoo class without a precomputed alignment — those
/// are correctly unaddressable at the CAM tier (Seam decision 1 Option B
/// "Cons": un-aligned classes shouldn't be cascade-resolvable). Callers MUST
/// NOT invent a family for an unresolved class.
#[inline]
pub fn resolve_odoo(odoo_class: &str) -> Option<OwlIdentity> {
    lookup_row(odoo_class).map(|row| row.owl)
}

/// Resolve an odoo class to the full precomputed alignment row (OWL pivot +
/// label + DOLCE category). O(1)-class, same lookup as [`resolve_odoo`].
#[inline]
pub fn resolve_odoo_alignment(odoo_class: &str) -> Option<&'static OdooAlignment> {
    lookup_row(odoo_class)
}

/// Chained odoo → OWL → OGIT-family resolution, **O(1) end to end, no graph
/// walk at call time**.
///
/// Leg 1: `resolve_odoo` (this module) — odoo class → OWL pivot `OwlIdentity`.
/// Leg 2: `OgitFamilyTable::lookup` (existing machinery) — `OwlIdentity` →
/// `&FamilyEntry`.
///
/// The equivalence (which OGIT family slot each odoo class inherits) is baked
/// into `ODOO_ALIGNMENTS` at compile time; only the family table is consulted
/// at runtime, and that is a single hash probe. Returns `None` if either leg
/// misses (unknown odoo class, or the OWL slot not yet hydrated into `table`).
#[inline]
pub fn resolve_odoo_to_family<'t>(
    odoo_class: &str,
    table: &'t OgitFamilyTable,
) -> Option<&'t FamilyEntry> {
    let owl = resolve_odoo(odoo_class)?;
    table.lookup(owl)
}

/// Internal: binary-search the worked-class core, then the SKR concept slice.
#[inline]
fn lookup_row(odoo_class: &str) -> Option<&'static OdooAlignment> {
    ODOO_ALIGNMENTS
        .binary_search_by(|row| row.odoo_class.cmp(odoo_class))
        .ok()
        .map(|i| &ODOO_ALIGNMENTS[i])
        .or_else(|| {
            SKR_ACCOUNT_CONCEPTS
                .binary_search_by(|row| row.odoo_class.cmp(odoo_class))
                .ok()
                .map(|i| &SKR_ACCOUNT_CONCEPTS[i])
        })
}

// ═══════════════════════════════════════════════════════════════════════════
// DOLCE suffix classifier (Seam decision 2)
// ═══════════════════════════════════════════════════════════════════════════

/// Classify an odoo class IRI into its DOLCE upper category by **suffix**
/// (Seam decision 2 of the four-way alignment seam).
///
/// The odoo namespace uses dotted lowercase model names where event/quality/
/// abstract semantics are encoded by suffix. Precedence: `product.template`
/// override → Perdurant → Quality → Abstract → default Endurant.
///
/// - `.move` / `.message` / `.activity` / `.attendance` / `.event` / `.log` /
///   `.picking` / ... → `Perdurant` (an occurrence / event)
/// - `.tag` / `.category` / `.type` / `.group` / `.tax` → `Quality`
///   (an attribute / classification / rate)
/// - `.template` / `.config` / `.policy` / `.rule` → `Abstract`
///   (a reference / configuration)
/// - `product.template` → `Endurant` (special-case override: odoo's
///   "template" here means the master product record, not a config template)
/// - default → `Endurant` (a persistent stateful object)
pub fn dolce_odoo(iri: &str) -> DolceMarker {
    // Strip the odoo: prefix and look at the model name only.
    let model = iri.trim_start_matches("odoo:");

    // Special-case override (Seam decision 2 table): odoo uses `.template`
    // for the master product record, NOT for a config template. This must be
    // checked BEFORE the generic `.template` Abstract rule below.
    if model == "product.template" {
        return DolceMarker::Endurant;
    }

    // Perdurant — name suffix indicating an event / occurrence.
    const PERDURANT_SUFFIXES: &[&str] = &[
        ".move",
        ".message",
        ".activity",
        ".attendance",
        ".transition",
        ".event",
        ".log",
        ".history",
        ".transaction",
        ".picking",
        ".scrap",
    ];
    // Quality — attributes / classifications / tags / rates.
    const QUALITY_SUFFIXES: &[&str] = &[".tag", ".category", ".type", ".group", ".tax"];
    // Abstract — references / configurations / templates.
    const ABSTRACT_SUFFIXES: &[&str] = &[".template", ".config", ".policy", ".rule", ".formula"];

    for suffix in PERDURANT_SUFFIXES {
        if model.ends_with(suffix) {
            return DolceMarker::Perdurant;
        }
    }
    for suffix in QUALITY_SUFFIXES {
        if model.ends_with(suffix) {
            return DolceMarker::Quality;
        }
    }
    for suffix in ABSTRACT_SUFFIXES {
        if model.ends_with(suffix) {
            return DolceMarker::Abstract;
        }
    }

    DolceMarker::Endurant
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family_table::{FamilyEntry, OgitFamilyTable, SchemaKind};

    // ── Both tables must stay sorted (binary-search contract) ─────────────────

    #[test]
    fn alignment_tables_are_sorted() {
        for w in ODOO_ALIGNMENTS.windows(2) {
            assert!(
                w[0].odoo_class < w[1].odoo_class,
                "ODOO_ALIGNMENTS not sorted: {} >= {}",
                w[0].odoo_class,
                w[1].odoo_class
            );
        }
        for w in SKR_ACCOUNT_CONCEPTS.windows(2) {
            assert!(
                w[0].odoo_class < w[1].odoo_class,
                "SKR_ACCOUNT_CONCEPTS not sorted: {} >= {}",
                w[0].odoo_class,
                w[1].odoo_class
            );
        }
    }

    // ── odoo→OWL leg: the seam's worked classes resolve ───────────────────────

    #[test]
    fn res_partner_company_resolves_to_legal_entity_pivot() {
        let owl = resolve_odoo("odoo:res.partner.Company").expect("Company must resolve");
        assert_eq!(owl, OwlIdentity::new(FAM_SMB_CUSTOMER, SLOT_LEGAL_ENTITY));
        let row = resolve_odoo_alignment("odoo:res.partner.Company").unwrap();
        assert_eq!(row.owl_pivot_label, "fibo:LegalEntity");
        assert_eq!(row.dolce, DolceMarker::Endurant);
    }

    #[test]
    fn account_move_resolves_to_transaction_pivot() {
        let row = resolve_odoo_alignment("odoo:account.move").unwrap();
        assert_eq!(row.owl_pivot_label, "fibo:Transaction");
        assert_eq!(row.dolce, DolceMarker::Perdurant);
        assert_eq!(row.owl.family(), FAM_SMB_INVOICE);
    }

    #[test]
    fn account_account_resolves_to_account_pivot() {
        let row = resolve_odoo_alignment("odoo:account.account").unwrap();
        assert_eq!(row.owl_pivot_label, "fibo:Account");
        assert_eq!(row.owl.family(), FAM_SMB_ACCOUNTING);
    }

    #[test]
    fn product_product_resolves_to_schema_product_pivot() {
        let row = resolve_odoo_alignment("odoo:product.product").unwrap();
        assert_eq!(row.owl_pivot_label, "schema:Product");
        assert_eq!(row.owl.family(), FAM_BILLING_CORE);
    }

    #[test]
    fn skr_account_concepts_resolve_to_account_pivot() {
        for skr in ["odoo:account.account.skr03", "odoo:account.account.skr04"] {
            let row = resolve_odoo_alignment(skr).unwrap_or_else(|| panic!("{skr} must resolve"));
            assert_eq!(row.owl_pivot_label, "fibo:Account");
            assert_eq!(row.owl.family(), FAM_SMB_ACCOUNTING);
        }
    }

    #[test]
    fn unknown_odoo_class_is_unresolved() {
        // Un-aligned classes are correctly unaddressable at the CAM tier.
        assert!(resolve_odoo("odoo:stock.warehouse").is_none());
        assert!(resolve_odoo("odoo:not.a.real.model").is_none());
        assert!(resolve_odoo("").is_none());
    }

    // ── Chained odoo→OWL→family leg: O(1), no walk ────────────────────────────

    #[test]
    fn resolve_odoo_to_family_chains_through_existing_table() {
        // Build a family table for the SMB-customer basin and populate the OWL
        // pivot slot the odoo Company class inherits.
        let mut table = OgitFamilyTable::empty(FAM_SMB_CUSTOMER);
        table.set(
            SLOT_LEGAL_ENTITY,
            FamilyEntry::plain_entity("ogit.SMB:Organization"),
        );

        let entry = resolve_odoo_to_family("odoo:res.partner.Company", &table)
            .expect("chained resolution must hit the populated slot");
        assert_eq!(entry.label_uri, "ogit.SMB:Organization");
        assert_eq!(entry.kind, SchemaKind::Entity);
    }

    #[test]
    fn resolve_odoo_to_family_misses_when_slot_unhydrated() {
        // Known odoo class, but the family table slot is empty → None (no panic).
        let table = OgitFamilyTable::empty(FAM_SMB_CUSTOMER);
        assert!(resolve_odoo_to_family("odoo:res.partner.Company", &table).is_none());
    }

    #[test]
    fn resolve_odoo_to_family_misses_for_unknown_class() {
        let table = OgitFamilyTable::empty(FAM_SMB_CUSTOMER);
        assert!(resolve_odoo_to_family("odoo:unknown.model", &table).is_none());
    }

    // ── DOLCE suffix classifier (Seam decision 2 test matrix) ─────────────────

    #[test]
    fn dolce_classifier_matches_seam_matrix() {
        // Endurant (persistent stateful objects)
        assert_eq!(dolce_odoo("odoo:res.partner"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:res.users"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:res.company"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:account.account"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:account.journal"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:product.product"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:stock.warehouse"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:crm.lead"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:hr.employee"), DolceMarker::Endurant);

        // product.template — explicit Endurant override (master record)
        assert_eq!(dolce_odoo("odoo:product.template"), DolceMarker::Endurant);

        // Perdurant (events / occurrences via suffix)
        assert_eq!(dolce_odoo("odoo:account.move"), DolceMarker::Perdurant);
        // NB: `account.move.line` is NOT caught by the suffix heuristic (its
        // suffix is `.line`, not an event suffix). The seam's `classify_odoo`
        // CODE returns Endurant here; the seam's test MATRIX lists Perdurant
        // ("child of Perdurant") — that finer verdict is carried by the
        // hand-curated alignment ROW, not the heuristic. The classifier is the
        // default; curated data refines it. See `account.move.line`'s row
        // (DolceMarker::Perdurant) and the divergence note in
        // `aligned_rows_dolce_consistent_with_classifier_where_applicable`.
        assert_eq!(dolce_odoo("odoo:account.move.line"), DolceMarker::Endurant);
        assert_eq!(dolce_odoo("odoo:stock.move"), DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("odoo:stock.picking"), DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("odoo:mail.message"), DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("odoo:hr.attendance"), DolceMarker::Perdurant);

        // Quality (attributes / classifications / rates via suffix)
        assert_eq!(dolce_odoo("odoo:account.tax"), DolceMarker::Quality);
        assert_eq!(dolce_odoo("odoo:product.category"), DolceMarker::Quality);
        assert_eq!(dolce_odoo("odoo:crm.tag"), DolceMarker::Quality);
        assert_eq!(
            dolce_odoo("odoo:account.account.type"),
            DolceMarker::Quality
        );

        // Abstract (templates / configs via suffix)
        assert_eq!(dolce_odoo("odoo:mail.template"), DolceMarker::Abstract);
        assert_eq!(
            dolce_odoo("odoo:account.chart.template"),
            DolceMarker::Abstract
        );
    }

    #[test]
    fn dolce_classifier_handles_missing_prefix() {
        // Works on bare model names too (prefix strip is a no-op).
        assert_eq!(dolce_odoo("account.move"), DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("res.partner"), DolceMarker::Endurant);
    }

    // ── Aligned rows carry a DOLCE marker consistent with the classifier ──────

    #[test]
    fn aligned_rows_dolce_consistent_with_classifier_where_applicable() {
        // For the worked classes whose DOLCE the suffix heuristic CAN derive,
        // the curated row's DOLCE must equal the classifier's verdict.
        for class in [
            "odoo:account.account",
            "odoo:account.move",
            "odoo:product.product",
            "odoo:product.template",
        ] {
            let row = resolve_odoo_alignment(class).unwrap();
            assert_eq!(
                row.dolce,
                dolce_odoo(class),
                "row DOLCE for {class} disagrees with dolce_odoo"
            );
        }

        // Documented divergence: `account.move.line` carries the matrix's
        // finer Perdurant verdict in its curated row, while the suffix
        // heuristic only sees `.line` and defaults to Endurant. Curated data
        // is allowed to refine the heuristic.
        let line = resolve_odoo_alignment("odoo:account.move.line").unwrap();
        assert_eq!(line.dolce, DolceMarker::Perdurant);
        assert_eq!(dolce_odoo("odoo:account.move.line"), DolceMarker::Endurant);
    }
}
