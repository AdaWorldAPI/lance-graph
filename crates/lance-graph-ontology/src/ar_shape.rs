// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ar_shape` — minimal smoke convergence between Rails-shaped curators
//! (OpenSourceBilling, future Spree/Solidus, future Redmine/OpenProject)
//! and Odoo via the OGAR canonical-concept layer.
//!
//! # What this is
//!
//! The first concrete instance of the synergy-registry framing the doctrine
//! (`docs/OGAR_AR_SHAPE_ENDGAME.md` §2 corrections, dated 2026-06-19) names:
//!
//! - Per-curator labels (e.g. OSB `InvoiceLineItem.item_unit_cost` / Odoo
//!   `account.move.line.price_unit`) are **leaf detail** that hangs off the
//!   OGAR class-inheritance edge.
//! - The ≥2-curator promotion rule (`E-OGAR-AR-SHAPE-ENDGAME` §3) requires
//!   ≥2 independent curators to surface the SAME primitive under different
//!   syntactic forms before a `CanonicalConcept` is admitted.
//! - Claude Code owns convergence detection; OGAR stores only stable
//!   canonical results after code/tests prove the overlap (per operator
//!   smoke-pass directive, 2026-06-19).
//!
//! # The shape today
//!
//! Hand-built fixtures per the operator directive *"Prefer hand-built Class
//! fixtures for the first smoke test if full repository extraction is too
//! heavy"*. The fixtures are typed `Class` instances carrying:
//!
//! - `source_curator` (`OpenSourceBilling`, `Odoo`, …)
//! - `source_domain` (`Billing`, `Erp`, …)
//! - `curator_label` — the curator's own class name (`InvoiceLineItem` /
//!   `account.move.line`), kept verbatim as leaf detail.
//! - `shape: ClassShape` — the structural form the overlap detector
//!   compares (today: only `ClassShape::LineItem`).
//! - `inherits` — curator-side composition labels.
//!
//! The overlap detector (`overlap_commercial_line_item`) returns
//! `Some(CanonicalConcept::CommercialLineItem)` exactly when the two
//! fixtures (a) come from *different* curators (≥2-curator promotion rule)
//! and (b) share the structural `LineItem` shape (both carry parent-doc
//! reference, quantity, unit-price, ≥1 tax binding, and a label field).
//!
//! # Scope discipline
//!
//! - **One** `CanonicalConcept` today (`CommercialLineItem`). The minimal
//!   step per operator acceptance #4 ("if absent, add the minimal canonical
//!   class or slot needed").
//! - **No** Rails / Odoo syntax leaks into OGAR Core: the canonical
//!   concept is a name only; curator labels stay on the fixture side.
//! - **Additive only**: this module introduces no changes to existing
//!   ontology types and does not require any change to
//!   `lance-graph-contract`.
//! - Future curators (Spree, Solidus, Redmine, OpenProject, future SAP)
//!   plug in by adding a `SourceCurator` variant and a fixture; the
//!   detector is reusable as-is for the LineItem shape, and grows by adding
//!   sibling `overlap_*` functions per `CanonicalConcept`.

/// The high-level domain a curator belongs to. Used as a coarse filter
/// (e.g. ERP vs commercial document vs project tracking) before the
/// structural shape test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceDomain {
    /// Customer-facing billing apps. OpenSourceBilling sits here.
    Billing,
    /// Full ERP with accounting + posting + tax finalization. Odoo sits
    /// here.
    Erp,
    /// E-commerce / sales-order-shaped apps. Spree, Solidus sit here.
    Commerce,
    /// Project / task / time tracking apps. Redmine, OpenProject sit
    /// here.
    Project,
}

/// A specific curator (a concrete upstream codebase). Maps 1-1 to a
/// namespace prefix at the harvest seam (`open_source_billing:` / `odoo:`
/// / …). New variants are added as new curators come online.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceCurator {
    /// `AdaWorldAPI/open-source-billing` — Ruby/Rails AR billing app.
    OpenSourceBilling,
    /// Odoo ORM (Python). Sourced via `tools/odoo-blueprint-extractor`.
    Odoo,
    /// Spree commerce platform (Rails AR). Future.
    Spree,
    /// Solidus (Spree fork, Rails AR). Future.
    Solidus,
    /// Redmine PM (Rails AR). Future.
    Redmine,
    /// OpenProject PM (Rails AR). Future.
    OpenProject,
}

impl SourceCurator {
    /// The namespace prefix this curator emits at the harvest seam. Stable
    /// `&'static str` per workspace canon (E-OGAR-AR-SHAPE-ENDGAME §11.1
    /// Inc 3: adapter target ids are `&'static str`).
    #[must_use]
    pub const fn namespace_prefix(self) -> &'static str {
        match self {
            Self::OpenSourceBilling => "open_source_billing:",
            Self::Odoo => "odoo:",
            Self::Spree => "spree:",
            Self::Solidus => "solidus:",
            Self::Redmine => "redmine:",
            Self::OpenProject => "openproject:",
        }
    }
}

/// The OGAR canonical concept — what ≥2 curators must agree on to promote.
///
/// Append-only. Each variant lands ONLY after at least two independent
/// curator fixtures overlap on its structural shape AND tests pin the
/// detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalConcept {
    /// `CommercialLineItem` — a per-line entry on a commercial document
    /// (invoice / journal / sales order line) carrying
    /// quantity × unit_price + tax bindings + parent-doc ref + label.
    /// Promoted from `{ osb:InvoiceLineItem, odoo:account.move.line }`
    /// pair on 2026-06-19.
    CommercialLineItem,
}

/// A typed fixture for one curator's class declaration. Hand-built today;
/// future ruff-side extraction will emit these from real corpora.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Class {
    /// Which curator surfaced this class. Drives the ≥2-curator promotion
    /// rule (same-curator pairs cannot promote).
    pub source_curator: SourceCurator,
    /// The high-level domain. Coarse filter / observability.
    pub source_domain: SourceDomain,
    /// The curator's own name for the class. Kept verbatim. Leaf detail
    /// (per doctrine §2 correction 1).
    pub curator_label: &'static str,
    /// The structural form the overlap detector compares.
    pub shape: ClassShape,
    /// Curator-side composition / inheritance — Rails `acts_as_*` /
    /// `include` / STI parents; Odoo `_inherit` chains. Names verbatim.
    pub inherits: &'static [&'static str],
}

/// The structural form of a class. Today only `LineItem`; sibling variants
/// (Document, Tax, Payment, …) land as new `CanonicalConcept`s prove out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassShape {
    /// A per-line entry on a commercial document. Shared by Rails
    /// `InvoiceLineItem`, Odoo `account.move.line`, Spree `LineItem`,
    /// future SAP BSEG.
    LineItem(LineItemShape),
}

/// The structural fields a `LineItem`-shaped class must carry. Field
/// *names* are curator-specific (leaf detail); what matters for overlap is
/// that each slot is present (non-empty).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineItemShape {
    /// Curator label of the parent document (`Invoice`, `account.move`,
    /// `Order`).
    pub parent_doc: &'static str,
    /// Curator label of the item/product reference, if any (`Item`,
    /// `product.product`, `Variant`).
    pub item_ref: Option<&'static str>,
    /// Curator-side quantity field name (`item_quantity`, `quantity`).
    pub quantity_field: &'static str,
    /// Curator-side unit-price field name (`item_unit_cost`, `price_unit`,
    /// `price`).
    pub unit_price_field: &'static str,
    /// Curator-side tax references. OSB uses two named slots (`tax_1`,
    /// `tax_2`); Odoo uses one M2M (`tax_ids`); both are non-empty for a
    /// line-item shape that can be promoted.
    pub tax_refs: &'static [&'static str],
    /// Curator-side label / description field name (`item_description`,
    /// `name`).
    pub label_field: &'static str,
}

// ─── Overlap detection ──────────────────────────────────────────────────

/// Detect a `CanonicalConcept::CommercialLineItem` overlap between two
/// curator fixtures. Returns `Some(CommercialLineItem)` exactly when:
///
/// 1. The two fixtures come from *different* curators (≥2-curator
///    promotion rule — same-curator pairs cannot promote).
/// 2. Both fixtures carry `ClassShape::LineItem`.
/// 3. Both fixtures have non-empty values for every structural slot
///    (`parent_doc`, `quantity_field`, `unit_price_field`, ≥1 `tax_refs`,
///    `label_field`).
///
/// Symmetric: `overlap_commercial_line_item(a, b) ==
/// overlap_commercial_line_item(b, a)`.
///
/// Deterministic: re-running on the same pair returns the same result
/// (no duplicate emissions per operator acceptance #5).
#[must_use]
pub fn overlap_commercial_line_item(a: &Class, b: &Class) -> Option<CanonicalConcept> {
    if a.source_curator == b.source_curator {
        return None;
    }
    let (ClassShape::LineItem(la), ClassShape::LineItem(lb)) = (&a.shape, &b.shape);

    let has_shape = |s: &LineItemShape| {
        !s.parent_doc.is_empty()
            && !s.quantity_field.is_empty()
            && !s.unit_price_field.is_empty()
            && !s.tax_refs.is_empty()
            && !s.label_field.is_empty()
    };
    if has_shape(la) && has_shape(lb) {
        Some(CanonicalConcept::CommercialLineItem)
    } else {
        None
    }
}

// ─── Hand-built curator fixtures ────────────────────────────────────────

/// `open_source_billing:InvoiceLineItem` fixture. Sourced from
/// `AdaWorldAPI/open-source-billing` commit `61cd6ed` (2026-06-19),
/// `app/models/invoice_line_item.rb`.
///
/// Notable curator-side facts (preserved as leaf detail):
///
/// - `belongs_to :tax1` / `:tax2` with FKs `tax_1` / `tax_2` (max two
///   taxes per line vs Odoo's M2M).
/// - `acts_as_archival` / `acts_as_paranoid` (soft-delete).
/// - `after_destroy :recalculate_invoice_total` (denormalized parent).
#[must_use]
pub const fn osb_invoice_line_item() -> Class {
    Class {
        source_curator: SourceCurator::OpenSourceBilling,
        source_domain: SourceDomain::Billing,
        curator_label: "InvoiceLineItem",
        shape: ClassShape::LineItem(LineItemShape {
            parent_doc: "Invoice",
            item_ref: Some("Item"),
            quantity_field: "item_quantity",
            unit_price_field: "item_unit_cost",
            tax_refs: &["tax_1", "tax_2"],
            label_field: "item_description",
        }),
        inherits: &["ApplicationRecord"],
    }
}

/// `odoo:account.move.line` fixture. Field names per the Odoo canonical
/// `account/models/account_move_line.py` surface (already grounded in
/// `lance-graph-ontology::odoo_blueprint::structural` and matched against
/// the #527 corpus). Inherits `analytic.mixin`.
#[must_use]
pub const fn odoo_account_move_line() -> Class {
    Class {
        source_curator: SourceCurator::Odoo,
        source_domain: SourceDomain::Erp,
        curator_label: "account.move.line",
        shape: ClassShape::LineItem(LineItemShape {
            parent_doc: "account.move",
            item_ref: Some("product.product"),
            quantity_field: "quantity",
            unit_price_field: "price_unit",
            tax_refs: &["tax_ids"],
            label_field: "name",
        }),
        inherits: &["analytic.mixin"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The headline smoke per operator directive: OSB::InvoiceLineItem +
    /// Odoo::account.move.line surface the same primitive (a per-line
    /// commercial entry carrying qty × unit_price + tax + parent + label)
    /// → promote to `CommercialLineItem`.
    #[test]
    fn open_source_billing_invoice_line_and_odoo_move_line_overlap_as_commercial_line_item() {
        let osb = osb_invoice_line_item();
        let odoo = odoo_account_move_line();

        let forward = overlap_commercial_line_item(&osb, &odoo);
        assert_eq!(forward, Some(CanonicalConcept::CommercialLineItem));

        // Symmetric — order should not matter.
        let reverse = overlap_commercial_line_item(&odoo, &osb);
        assert_eq!(reverse, forward);
    }

    /// Regression for operator acceptance #5: detection is deterministic
    /// and re-running it must not register a second canonical concept.
    /// (Idempotence at the function level; registry-side idempotence
    /// would come from a `BTreeSet<CanonicalConcept>` upstream.)
    #[test]
    fn rails_billing_and_odoo_do_not_create_duplicate_canonical_concepts() {
        let osb = osb_invoice_line_item();
        let odoo = odoo_account_move_line();

        let first = overlap_commercial_line_item(&osb, &odoo);
        let second = overlap_commercial_line_item(&osb, &odoo);
        assert_eq!(first, second);
        assert!(matches!(first, Some(CanonicalConcept::CommercialLineItem)));
    }

    /// The ≥2-curator promotion rule is STRUCTURAL: comparing one
    /// curator's fixture against itself MUST NOT promote.
    #[test]
    fn same_curator_self_compare_does_not_promote() {
        let a = osb_invoice_line_item();
        let b = osb_invoice_line_item();
        assert_eq!(overlap_commercial_line_item(&a, &b), None);

        let c = odoo_account_move_line();
        let d = odoo_account_move_line();
        assert_eq!(overlap_commercial_line_item(&c, &d), None);
    }

    /// Curator-label divergence is part of the design — the field-NAMES
    /// differ (`item_unit_cost` vs `price_unit`), but the shape still
    /// promotes. The leaf detail stays visible on the fixture for
    /// adapter generation.
    #[test]
    fn curator_field_names_diverge_but_shape_still_promotes() {
        let osb = osb_invoice_line_item();
        let odoo = odoo_account_move_line();

        let ClassShape::LineItem(osb_shape) = osb.shape;
        let ClassShape::LineItem(odoo_shape) = odoo.shape;

        assert_ne!(osb_shape.unit_price_field, odoo_shape.unit_price_field);
        assert_ne!(osb_shape.quantity_field, odoo_shape.quantity_field);
        assert_ne!(osb_shape.label_field, odoo_shape.label_field);

        // …yet they overlap.
        assert_eq!(
            overlap_commercial_line_item(&osb, &odoo),
            Some(CanonicalConcept::CommercialLineItem),
        );
    }

    /// Namespace prefixes are stable `&'static str` per
    /// `E-OGAR-AR-SHAPE-ENDGAME` §11.1 Inc 3 (adapter target ids are
    /// `&'static str`). Lock the two curators we actually use today.
    #[test]
    fn namespace_prefixes_for_today_curators_are_stable() {
        assert_eq!(SourceCurator::OpenSourceBilling.namespace_prefix(), "open_source_billing:");
        assert_eq!(SourceCurator::Odoo.namespace_prefix(), "odoo:");
    }

    /// Empty structural slot (e.g. a malformed fixture with no tax_refs)
    /// must NOT promote — the overlap test is conservative on absent
    /// shape.
    #[test]
    fn empty_tax_refs_block_promotion() {
        let mut osb = osb_invoice_line_item();
        let ClassShape::LineItem(ref mut shape) = osb.shape;
        shape.tax_refs = &[];
        let odoo = odoo_account_move_line();

        assert_eq!(overlap_commercial_line_item(&osb, &odoo), None);
    }
}
