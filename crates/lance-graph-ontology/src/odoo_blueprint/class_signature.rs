//! # `class_signature` — structural-signature audit of the curated `OdooEntity` consts.
//!
//! The honest D-CLS-2/D-CLS-3: classes.md:41-44 says the class taxonomy is
//! **discovered, not hand-assigned** — "20,000 Odoo entities are NOT 20,000
//! shapes; they are instances of ~dozens of shape-families. Group by structural
//! signature (which fields, `_compute_*` shape, depends/emits pattern)."
//!
//! This is the **deterministic** group-by-on-structural-hash (NOT an Aerial+
//! clustering pass — `aerial` mines association *rules*, it does not cluster
//! entities; the brutal-review confirmed that entry point does not exist). Two
//! entities with the same structural signature ARE the same shape-family.
//!
//! It also derives the per-class [`ObjectView`] **field-set** (the bit-basis the
//! [`crate::class_resolver::RegistryClassView`] previously took as a supplied
//! placeholder). Field position `i` in the derived `ObjectView` is the stable
//! [`FieldMask`](lance_graph_contract::class_view::FieldMask) bit `i` (N3).
//!
//! Pure analysis over the `&'static` const data — no hot path, no mutation.

use lance_graph_contract::ontology::{DisplayTemplate, FieldRef, ObjectView};

use super::{OdooEntity, OdooFieldKind, OdooMethodKind};

/// A deterministic structural signature of an [`OdooEntity`] — the shape-family key.
///
/// Two entities sharing a `StructuralSignature` are the same shape-family
/// (classes.md:43). Built from the *structure* (kind + field-kind histogram +
/// method-kind histogram + state-machine presence), NOT the names — so
/// `account.move` and `sale.order` (both stateful compute-emitting models)
/// collapse to one family while a plain config model does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructuralSignature(pub u32);

/// FNV-1a 32-bit over the canonicalized structural tuple. Mirrors the workspace
/// idiom (`odoo_blueprint::style_recipe::fnv1a_recipe`). Collisions are intentional
/// — identical structure → identical family id.
fn fnv1a(bytes: &[u8]) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME: u32 = 0x0100_0193;
    let mut h = OFFSET;
    for b in bytes {
        h ^= u32::from(*b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// The 6 [`OdooFieldKind`] buckets the histogram counts (the closed structural axis).
fn field_kind_bucket(k: OdooFieldKind) -> usize {
    match k {
        OdooFieldKind::Char | OdooFieldKind::Text | OdooFieldKind::Html => 0, // textual
        OdooFieldKind::Integer | OdooFieldKind::Float | OdooFieldKind::Monetary => 1, // numeric
        OdooFieldKind::Boolean => 2,
        OdooFieldKind::Date | OdooFieldKind::Datetime => 3, // temporal
        OdooFieldKind::Selection => 4,
        // relational + the rest (Many2one/One2many/Many2many/Binary/…)
        _ => 5,
    }
}

/// The 5 [`OdooMethodKind`] buckets the histogram counts (the `_compute_*` shape axis).
fn method_kind_bucket(k: OdooMethodKind) -> usize {
    match k {
        OdooMethodKind::Compute | OdooMethodKind::Inverse => 0, // computed-value shape
        OdooMethodKind::Constrain => 1,                         // validation shape
        OdooMethodKind::Onchange => 2,                          // reactive-UI shape
        OdooMethodKind::Action => 3,                            // state-transition shape
        _ => 4,                                                 // cron/api/override/helper
    }
}

/// Compute the [`StructuralSignature`] of an entity (deterministic, name-independent).
pub fn signature(entity: &OdooEntity) -> StructuralSignature {
    // Canonical tuple: [kind_disc, field_hist x6, method_hist x5, has_state_machine].
    // Counts are saturated to u8 so the byte layout is stable + the hash deterministic.
    let mut buf = [0u8; 1 + 6 + 5 + 1];
    buf[0] = entity.kind as u8;

    let mut field_hist = [0u32; 6];
    for f in entity.fields {
        field_hist[field_kind_bucket(f.kind)] += 1;
    }
    for (i, c) in field_hist.iter().enumerate() {
        buf[1 + i] = (*c).min(255) as u8;
    }

    let mut method_hist = [0u32; 5];
    for m in entity.methods {
        method_hist[method_kind_bucket(m.kind)] += 1;
    }
    for (i, c) in method_hist.iter().enumerate() {
        buf[7 + i] = (*c).min(255) as u8;
    }

    buf[12] = u8::from(entity.state_machine.is_some());

    StructuralSignature(fnv1a(&buf))
}

/// Derive the per-class [`ObjectView`] **field-set** from an entity's declared
/// fields — the real bit-basis. Field position `i` here is the stable
/// [`FieldMask`](lance_graph_contract::class_view::FieldMask) bit `i` (N3).
///
/// Field order = declaration order (append-only stability: new fields append at
/// higher positions, existing positions never move). The first textual/`Char`
/// field becomes the `primary_label`; the `DisplayTemplate` is chosen by size
/// (`<= 4` fields → `Card`, else `Detail`). Capped at
/// [`FieldMask::MAX_FIELDS`](lance_graph_contract::class_view::FieldMask::MAX_FIELDS)
/// (64) — the mask cannot address beyond a `u64`.
pub fn object_view(entity: &OdooEntity) -> ObjectView {
    use lance_graph_contract::class_view::FieldMask;

    let cap = FieldMask::MAX_FIELDS as usize;
    let fields: Vec<FieldRef> = entity
        .fields
        .iter()
        .take(cap)
        .map(|f| FieldRef::new(f.name, f.name)) // label defaults to name; OGIT resolves the display label late
        .collect();

    let template = if fields.len() <= 4 {
        DisplayTemplate::Card
    } else {
        DisplayTemplate::Detail
    };

    let mut view = ObjectView::new(template, fields);
    // primary_label = the first textual field (the headline), if any.
    view.primary_label = entity
        .fields
        .iter()
        .find(|f| field_kind_bucket(f.kind) == 0)
        .map(|f| f.name.to_string());
    view
}

/// One audited entity row: its name, ORM kind, derived signature, field count.
#[derive(Debug, Clone)]
pub struct AuditRow {
    pub model_name: &'static str,
    pub signature: StructuralSignature,
    pub field_count: usize,
    pub method_count: usize,
    pub has_state_machine: bool,
}

/// Audit a slice of entities → their rows (read-only structural pass).
pub fn audit(entities: &[OdooEntity]) -> Vec<AuditRow> {
    entities
        .iter()
        .map(|e| AuditRow {
            model_name: e.model_name,
            signature: signature(e),
            field_count: e.fields.len(),
            method_count: e.methods.len(),
            has_state_machine: e.state_machine.is_some(),
        })
        .collect()
}

/// Group audited entities into shape-families by structural signature
/// (the discovered taxonomy, classes.md:43). Returns `(signature → member model
/// names)`, sorted by signature for deterministic output.
pub fn shape_families(entities: &[OdooEntity]) -> Vec<(StructuralSignature, Vec<&'static str>)> {
    use std::collections::BTreeMap;
    let mut families: BTreeMap<u32, Vec<&'static str>> = BTreeMap::new();
    for e in entities {
        families
            .entry(signature(e).0)
            .or_default()
            .push(e.model_name);
    }
    families
        .into_iter()
        .map(|(sig, members)| (StructuralSignature(sig), members))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::l1;

    #[test]
    fn signature_is_deterministic_and_name_independent() {
        // Same entity → same signature on repeat (deterministic).
        let a = signature(&l1::ACCOUNT_MOVE);
        let b = signature(&l1::ACCOUNT_MOVE);
        assert_eq!(a, b, "signature must be deterministic");
        // account.move and account.move.line have DIFFERENT structure (one is a
        // stateful header, the other a line) → different signatures.
        assert_ne!(
            signature(&l1::ACCOUNT_MOVE),
            signature(&l1::ACCOUNT_MOVE_LINE),
            "structurally different entities must not collide"
        );
    }

    #[test]
    fn object_view_derives_the_bit_basis_from_fields() {
        let v = object_view(&l1::ACCOUNT_MOVE);
        // Field count = the entity's declared fields (capped at 64), in order.
        assert_eq!(v.fields.len(), l1::ACCOUNT_MOVE.fields.len().min(64));
        // Position 0 is the first declared field (stable bit 0).
        assert_eq!(v.fields[0].predicate_iri, l1::ACCOUNT_MOVE.fields[0].name);
        // ACCOUNT_MOVE is stateful + field-rich → Detail template.
        assert_eq!(v.display_template, DisplayTemplate::Detail);
    }

    #[test]
    fn shape_families_group_deterministically() {
        let families = shape_families(l1::ENTITIES);
        // Every l1 entity is accounted for exactly once across families.
        let total: usize = families.iter().map(|(_, m)| m.len()).sum();
        assert_eq!(total, l1::ENTITIES.len());
        // Output is sorted by signature (deterministic).
        let sigs: Vec<u32> = families.iter().map(|(s, _)| s.0).collect();
        let mut sorted = sigs.clone();
        sorted.sort_unstable();
        assert_eq!(sigs, sorted, "families sorted by signature");
    }

    #[test]
    fn audit_row_count_matches_input() {
        let rows = audit(l1::ENTITIES);
        assert_eq!(rows.len(), l1::ENTITIES.len());
        assert!(rows.iter().all(|r| !r.model_name.is_empty()));
    }
}
