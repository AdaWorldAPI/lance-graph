//! # `class_view` — the class as a META lookup that flies ABOVE the SoA.
//!
//! ## The XML-parse framing
//!
//! Today OGIT (`lance-graph-ontology::OntologyRegistry`) is a **hashtable doing
//! single lookups**: `uri → row`, `entity_type_id → row` — one key, one value,
//! O(1), leaf. That is the *single* lookup. What a class needs is a **meta
//! lookup**: `class_id → the whole shape` (ordered field set + labels + template
//! + the presence-bit basis). The class composes many leaf lookups into one
//! shape — the way an XSD schema composes element declarations.
//!
//! ```text
//!   SoA row          =  the XML document   (agnostic bytes, no meaning)
//!   class / ObjectView =  the XSD schema     (the shape: which fields, in order)
//!   ClassView (this) =  the parser+schema  (projects row → typed view, late-bound)
//!   FieldMask        =  which optional elements are present  (structural)
//!   askama template  =  the XSLT            (renders the projected view)
//! ```
//!
//! ## Classes fly as a meta-DTO ABOVE the SoA — the SoA stays agnostic
//!
//! The load-bearing rule (`cognitive-risc-classes.md`:39 "the meta-DTO resolves;
//! it does not store"; `cognitive-risc-core.md` invariant #1 "nothing semantic in
//! the register file"): the SoA row carries **only** `class_id` + a presence
//! [`FieldMask`] + agnostic columns. **Zero labels in the bytes.** The
//! labels / template / DOLCE-category are resolved *at projection time* by the
//! flying meta-DTO from the OGIT cache — never hand-rolled onto the row.
//!
//! That makes the presence/semantics split (C2) fall out for free:
//! - **bit = presence** — structural, lives on the SoA ("field N is populated").
//! - **bit → field → label → template** — semantic resolution, lives in the
//!   meta-DTO *above* the SoA. A bit NEVER means "field N behaves differently."
//!
//! ## Layering (dependency inversion, same shape as `MailboxSoaView`)
//!
//! - **contract (here, zero-dep):** the agnostic surface — [`FieldMask`] presence
//!   bits + the [`ClassView`] resolver *trait*. Extends the existing
//!   [`crate::ontology::ObjectView`] (the per-class ordered field set = the bit
//!   basis), does not duplicate it.
//! - **ontology (one layer up):** *implements* [`ClassView`] — the "parser" that
//!   walks the class shape and resolves labels late from the OGIT hashmap.
//! - **render (a consumer):** reads the projected view + mask, picks the askama
//!   template, skips off-bits.

use crate::ontology::{DisplayTemplate, FieldRef};

/// Per-row class discriminator — the Cognitive-RISC `class_id` / `shape_id`.
///
/// A `u16` (≤ 65,535 shape-families; OD-CLASSID-WIDTH ratified). It is a
/// *discriminator*, never a content hash — it stays OUTSIDE the CAM identity
/// layer (`I-VSA-IDENTITIES`: never hashed-as-content, never superposed). Reuses
/// the width of the existing [`crate::soa_view::MailboxSoaView::class_id`] accessor.
pub type ClassId = u16;

/// A class's **presence bitmask** — one bit per field of its class
/// [`ObjectView`](crate::ontology::ObjectView), set iff that field is populated
/// on a given instance.
///
/// The instance's *delta from its class* (`cognitive-risc-classes.md`:48), as
/// **pure presence bits**. Bit position `N` = the `N`-th field in the class's
/// ordered field list — stable + append-only (N3): once instances persist, a
/// field's bit position never moves and retired bits are never reused. Zero-dep
/// (`u64`, no `bitflags`); mask width is bounded by the *class's* field count
/// (dozens), never the entity union.
///
/// **Presence, NEVER semantics (C2).** `has(n)` answers "is field n populated
/// here"; it must never gate "field n means something different here."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct FieldMask(pub u64);

impl FieldMask {
    /// The empty mask (no fields populated).
    pub const EMPTY: Self = Self(0);

    /// Maximum addressable field positions in one `u64` mask.
    pub const MAX_FIELDS: u32 = 64;

    /// Build a mask from the populated field positions.
    pub const fn from_positions(positions: &[u8]) -> Self {
        let mut bits = 0u64;
        let mut i = 0;
        while i < positions.len() {
            bits |= 1u64 << (positions[i] as u64 & 63);
            i += 1;
        }
        Self(bits)
    }

    /// Set field position `n` as populated.
    #[inline]
    pub const fn with(self, n: u8) -> Self {
        Self(self.0 | (1u64 << (n as u64 & 63)))
    }

    /// Is field position `n` populated? (presence — C2)
    #[inline]
    pub const fn has(self, n: u8) -> bool {
        self.0 & (1u64 << (n as u64 & 63)) != 0
    }

    /// Number of populated fields.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Is nothing populated?
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The class as a **meta lookup that flies above the SoA** — the resolver trait.
///
/// An implementor (in `lance-graph-ontology`, over the OGIT cache) is the
/// "parser+schema": given a `class_id` it resolves the class's ordered field set,
/// labels, DOLCE category, and render template — all LATE-bound from the cache,
/// none stored on the SoA row. The contract owns only the *vocabulary*; the cache
/// owns the *answers* (dependency inversion, like `PlannerContract`/`MailboxSoaView`).
///
/// "Single lookup" (leaf, today) vs "meta lookup" (the class, this trait): a
/// single lookup is `uri → row`; a meta lookup is `class_id → shape`, composing
/// many leaf lookups into one projected view.
pub trait ClassView {
    /// The class's ordered field set — the bit basis. Position `i` in this slice
    /// is the stable [`FieldMask`] bit `i` (N3 append-only). This IS the
    /// per-class [`ObjectView`](crate::ontology::ObjectView)'s `fields`.
    fn fields(&self, class: ClassId) -> &[FieldRef];

    /// Which askama template renders this class.
    fn template(&self, class: ClassId) -> DisplayTemplate;

    /// The DOLCE upper-category of this class, RESOLVED from the ontology cache
    /// (not a stored enum on the row — OD-DOLCE "use the ontology cache"). Returned
    /// as the cache's opaque category id; the consumer maps it to its own enum.
    fn dolce_category_id(&self, class: ClassId) -> u8;

    /// The label of field position `n` in `class`, resolved late from the cache
    /// (locale resolution is the consumer's job). `None` if `n` is out of range.
    fn field_label(&self, class: ClassId, n: u8) -> Option<&str> {
        self.fields(class).get(n as usize).map(|f| f.label.as_str())
    }

    /// The class's field count (mask width). Must be `<= FieldMask::MAX_FIELDS`.
    #[inline]
    fn field_count(&self, class: ClassId) -> usize {
        self.fields(class).len()
    }

    /// Project an instance: iterate `(field, populated?)` pairs in class order,
    /// gating each field by the presence `mask`. This is the render surface — the
    /// consumer skips off-bits (`cognitive-risc-classes.md`:49). The SoA supplied
    /// only `(class, mask)`; the labels come from the cache, above the SoA.
    fn project<'a>(&'a self, class: ClassId, mask: FieldMask) -> ClassProjection<'a> {
        ClassProjection {
            fields: self.fields(class),
            mask,
            pos: 0,
        }
    }
}

/// An iterator over a class's fields paired with their presence bit — the
/// projected view a render template consumes (off-bits are still yielded with
/// `present = false` so the template can `{% if present %}`-skip them).
pub struct ClassProjection<'a> {
    fields: &'a [FieldRef],
    mask: FieldMask,
    pos: usize,
}

impl<'a> Iterator for ClassProjection<'a> {
    /// `(field, present)` — `present` is the C2 presence bit, never a semantics bit.
    type Item = (&'a FieldRef, bool);

    fn next(&mut self) -> Option<Self::Item> {
        let f = self.fields.get(self.pos)?;
        let present = self.mask.has(self.pos as u8);
        self.pos += 1;
        Some((f, present))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{DisplayTemplate, FieldRef};

    /// A tiny in-contract ClassView fake — proves the trait is satisfiable and the
    /// meta-DTO projects above an agnostic (class, mask) input, no labels stored.
    struct FakeClasses {
        // class 7 = a 3-field shape ("invoice": amount, tax, partner)
        invoice: Vec<FieldRef>,
    }

    impl FakeClasses {
        fn new() -> Self {
            Self {
                invoice: vec![
                    FieldRef::new("amount_total", "Total"),
                    FieldRef::new("amount_tax", "Tax"),
                    FieldRef::new("partner_id", "Partner"),
                ],
            }
        }
    }

    impl ClassView for FakeClasses {
        fn fields(&self, class: ClassId) -> &[FieldRef] {
            match class {
                7 => &self.invoice,
                _ => &[],
            }
        }
        fn template(&self, _class: ClassId) -> DisplayTemplate {
            DisplayTemplate::Detail
        }
        fn dolce_category_id(&self, _class: ClassId) -> u8 {
            0 // Endurant, resolved from the cache in the real impl
        }
    }

    #[test]
    fn field_mask_is_presence_bits() {
        let m = FieldMask::from_positions(&[0, 2]); // amount + partner populated, tax absent
        assert!(m.has(0) && !m.has(1) && m.has(2));
        assert_eq!(m.count(), 2);
        assert!(!m.is_empty() && FieldMask::EMPTY.is_empty());
        assert_eq!(
            FieldMask::EMPTY.with(1).with(1),
            FieldMask::from_positions(&[1])
        );
    }

    #[test]
    fn meta_dto_projects_above_agnostic_class_mask() {
        let classes = FakeClasses::new();
        // The SoA supplied ONLY (class_id=7, mask) — no labels. The meta-DTO
        // resolves the labels from above.
        let mask = FieldMask::from_positions(&[0, 2]); // tax (pos 1) is off
        let projected: Vec<(&str, bool)> = classes
            .project(7, mask)
            .map(|(f, present)| (f.label.as_str(), present))
            .collect();
        assert_eq!(
            projected,
            vec![("Total", true), ("Tax", false), ("Partner", true)],
            "labels come from the cache above the SoA; presence comes from the mask"
        );
        // The render template skips off-bits: only present fields surface.
        let rendered: Vec<&str> = classes
            .project(7, mask)
            .filter(|(_, present)| *present)
            .map(|(f, _)| f.label.as_str())
            .collect();
        assert_eq!(rendered, vec!["Total", "Partner"], "off-bit (Tax) skipped");
    }

    #[test]
    fn field_label_resolves_late_from_class_not_row() {
        let classes = FakeClasses::new();
        assert_eq!(classes.field_label(7, 1), Some("Tax"));
        assert_eq!(classes.field_label(7, 9), None); // out of range
        assert_eq!(classes.field_count(7), 3);
        assert_eq!(classes.field_count(999), 0); // unknown class
    }
}
