//! # `class_resolver` — the ontology-side `impl ClassView` (the "parser" over the OGIT cache).
//!
//! This is the layer the contract [`ClassView`](lance_graph_contract::class_view::ClassView)
//! trait inverts onto: the contract owns the *vocabulary* (FieldMask presence bits,
//! the resolver trait); this crate owns the *answers* — it resolves a `class_id`
//! into its shape (template, DOLCE category, field labels) **late, from the live
//! `OntologyRegistry` cache**, so the SoA row never stores a label.
//!
//! ## What "meta lookup" means here (vs the leaf hashtable)
//!
//! [`OntologyRegistry`] today is the hashtable doing *single* lookups:
//! `enumerate_first_with_entity_type_id(class_id) -> MappingRow` is one key → one
//! row. [`RegistryClassView`] composes that leaf lookup with the per-class
//! [`ObjectView`](lance_graph_contract::ontology::ObjectView) field-set into the
//! *meta* lookup the class needs: `class_id -> (ordered fields + labels + template
//! + DOLCE)`.
//!
//! ## Honest scope (what resolves today vs what's deferred)
//!
//! - **Resolves live from the cache:** the class's *existence* + its DOLCE category
//!   (via [`classify_odoo`] over the row's OGIT URI — OD-DOLCE "use the ontology
//!   cache") + its render template.
//! - **Field-set is supplied, not yet enumerated:** a `MappingRow` is a single
//!   entity's leaf row; it does **not** carry an enumerated field-set with labels.
//!   That enumeration is the deferred D-CLS structural-signature audit. Until it
//!   lands, the per-class `ObjectView`s are passed in (the bit-basis), and this
//!   adapter resolves everything *else* from the cache. No field-set is fabricated.

use std::cell::RefCell;
use std::collections::HashMap;

use lance_graph_contract::class_view::{ClassId, ClassView, FieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef, ObjectView};

use crate::hydrators::dolce_odoo::{classify_odoo, DolceCategory};
use crate::registry::OntologyRegistry;

/// Stable `u8` ids for the DOLCE upper categories — the opaque category id the
/// contract [`ClassView::dolce_category_id`] returns (the contract has no DOLCE
/// enum; consumers map this back). Positions are append-only (N3 discipline).
///
/// [`ClassView::dolce_category_id`]: lance_graph_contract::class_view::ClassView::dolce_category_id
pub mod dolce_id {
    /// DOLCE Endurant (persistent stateful object) — the default.
    pub const ENDURANT: u8 = 0;
    /// DOLCE Perdurant (event / process).
    pub const PERDURANT: u8 = 1;
    /// DOLCE Quality (inhering property — VAT rate, currency).
    pub const QUALITY: u8 = 2;
    /// DOLCE Abstract object (template, OGIT class).
    pub const ABSTRACT: u8 = 3;
}

/// Map a resolved [`DolceCategory`] to its stable `u8` id (the cache's opaque
/// category id, per the contract `ClassView` contract).
fn dolce_to_id(c: DolceCategory) -> u8 {
    match c {
        DolceCategory::Endurant => dolce_id::ENDURANT,
        DolceCategory::Perdurant => dolce_id::PERDURANT,
        DolceCategory::Quality => dolce_id::QUALITY,
        DolceCategory::AbstractEntity => dolce_id::ABSTRACT,
    }
}

/// The ontology-side [`ClassView`] — resolves a class's shape from the live cache.
///
/// Holds a borrow of the [`OntologyRegistry`] (the cache) plus the per-class
/// [`ObjectView`] field-sets (the bit-basis the registry does not yet enumerate).
/// It RESOLVES; it does not STORE labels on any row (classes.md:39).
pub struct RegistryClassView<'a> {
    registry: &'a OntologyRegistry,
    /// `class_id -> its ObjectView` (ordered fields + template). The field
    /// enumeration is the deferred D-CLS audit; supplied here as the bit-basis.
    views: HashMap<ClassId, ObjectView>,
    /// Empty fallback so `fields()` can return a `&[FieldRef]` for unknown classes.
    empty: Vec<FieldRef>,
    /// Memo of `class_id -> resolved DOLCE id`. A class's DOLCE category is stable
    /// (its OGIT URI does not change), and the underlying registry lookup is an
    /// O(n) scan + full `MappingRow` clone (`registry::enumerate_first_with_entity_type_id`
    /// — a known perf gap; the `by_entity_type_id` index is a deferred registry slice).
    /// Memoizing here makes the scan happen at most once per class, not per render call.
    dolce_memo: RefCell<HashMap<ClassId, u8>>,
}

impl<'a> RegistryClassView<'a> {
    /// Build over the live registry with the per-class field-set views.
    pub fn new(registry: &'a OntologyRegistry, views: HashMap<ClassId, ObjectView>) -> Self {
        Self {
            registry,
            views,
            empty: Vec::new(),
            dolce_memo: RefCell::new(HashMap::new()),
        }
    }

    /// Does the cache know this class? (a real leaf lookup against the registry)
    pub fn is_known(&self, class: ClassId) -> bool {
        self.registry
            .enumerate_first_with_entity_type_id(class)
            .is_some()
    }
}

impl ClassView for RegistryClassView<'_> {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        self.views
            .get(&class)
            .map(|v| v.fields.as_slice())
            .unwrap_or(&self.empty)
    }

    fn template(&self, class: ClassId) -> DisplayTemplate {
        self.views
            .get(&class)
            .map(|v| v.display_template.clone())
            // Default render is the compact Card when no per-class view is supplied.
            .unwrap_or(DisplayTemplate::Card)
    }

    fn dolce_category_id(&self, class: ClassId) -> u8 {
        if let Some(&id) = self.dolce_memo.borrow().get(&class) {
            return id;
        }
        // Resolve DOLCE LATE from the cache: class_id -> MappingRow -> OGIT URI ->
        // classify_odoo. Never stored on the row (OD-DOLCE "use the ontology cache").
        let id = match self.registry.enumerate_first_with_entity_type_id(class) {
            Some(row) => dolce_to_id(classify_odoo(row.ogit_uri.as_str())),
            // Unknown class → the default persistent-object category.
            None => dolce_id::ENDURANT,
        };
        self.dolce_memo.borrow_mut().insert(class, id);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::ontology::DisplayTemplate;

    fn invoice_view() -> ObjectView {
        ObjectView::new(
            DisplayTemplate::Detail,
            vec![
                FieldRef::new("amount_total", "Total"),
                FieldRef::new("amount_tax", "Tax"),
                FieldRef::new("partner_id", "Partner"),
            ],
        )
    }

    #[test]
    fn resolves_field_set_and_template_from_supplied_view() {
        let reg = OntologyRegistry::new_in_memory();
        let mut views = HashMap::new();
        views.insert(7u16, invoice_view());
        let cv = RegistryClassView::new(&reg, views);

        // The class's ordered field-set IS the bit basis (positions 0/1/2).
        assert_eq!(cv.field_count(7), 3);
        assert_eq!(cv.field_label(7, 0), Some("Total"));
        assert_eq!(cv.field_label(7, 2), Some("Partner"));
        assert_eq!(cv.template(7), DisplayTemplate::Detail);

        // Unknown class: empty field-set, default Card template — no panic.
        assert_eq!(cv.field_count(999), 0);
        assert_eq!(cv.template(999), DisplayTemplate::Card);
    }

    #[test]
    fn projects_above_an_agnostic_class_mask_with_labels_from_the_resolver() {
        let reg = OntologyRegistry::new_in_memory();
        let mut views = HashMap::new();
        views.insert(7u16, invoice_view());
        let cv = RegistryClassView::new(&reg, views);

        // The "SoA" supplies only (class_id=7, mask). Tax (pos 1) is off.
        let mask = FieldMask::from_positions(&[0, 2]);
        let rendered: Vec<&str> = cv
            .project(7, mask)
            .filter(|(_, present)| *present)
            .map(|(f, _)| f.label.as_str())
            .collect();
        assert_eq!(
            rendered,
            vec!["Total", "Partner"],
            "labels resolved by the meta-DTO above the SoA; off-bit Tax skipped"
        );
    }

    #[test]
    fn dolce_resolves_from_the_cache_not_the_row() {
        // Empty registry → unknown class falls back to the default category, and
        // `is_known` honestly reports the leaf lookup miss (no fabrication).
        let reg = OntologyRegistry::new_in_memory();
        let cv = RegistryClassView::new(&reg, HashMap::new());
        assert!(!cv.is_known(7), "empty cache: class is not known");
        assert_eq!(
            cv.dolce_category_id(7),
            dolce_id::ENDURANT,
            "unknown class → default persistent-object category"
        );
        // Memoized: a second call returns the same id (and skips the O(n) re-scan).
        assert_eq!(cv.dolce_category_id(7), dolce_id::ENDURANT);
    }

    #[test]
    fn dolce_id_mapping_is_total_and_stable() {
        assert_eq!(dolce_to_id(DolceCategory::Endurant), dolce_id::ENDURANT);
        assert_eq!(dolce_to_id(DolceCategory::Perdurant), dolce_id::PERDURANT);
        assert_eq!(dolce_to_id(DolceCategory::Quality), dolce_id::QUALITY);
        assert_eq!(
            dolce_to_id(DolceCategory::AbstractEntity),
            dolce_id::ABSTRACT
        );
    }
}
