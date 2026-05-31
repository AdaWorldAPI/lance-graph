//! # `wikidata_hhtl` — the N4 second-domain falsifier for the class-meta-DTO.
//!
//! `cognitive-risc-classes.md` N4: the class-meta-DTO (FieldMask presence +
//! StructuralSignature shape + ClassView resolution) must generalise BEYOND Odoo,
//! or it is overfit to one ERP. **Wikidata is the falsifier domain.** This module
//! routes a curated set of *real* Wikidata classes through the SAME machinery the
//! D-CLS arc (#441) built for Odoo — proving the Wikidata "D-CLS triple"
//! `(class_id, shape_hash, presence_bitmask)` = `(ClassId, StructuralSignature,
//! FieldMask)` is **domain-independent** (`wikidata-hhtl-load.md`:33).
//!
//! ## What is reused (not re-grown)
//! - **Routing:** [`NiblePath`](lance_graph_contract::hhtl::NiblePath) — the 16ⁿ
//!   Abstammung bucket router (`subClassOf`/P279 path). basin = the cache-resolved
//!   DOLCE id.
//! - **Presence:** [`FieldMask`](lance_graph_contract::class_view::FieldMask) — one
//!   bit per present property-id (the "facet-bitmask" of `wikidata-hhtl-load.md`).
//! - **Shape:** [`StructuralSignature`](crate::odoo_blueprint::class_signature::StructuralSignature)
//!   — the shape-family key, here a hash over the canonical property-id set (the
//!   Wikidata analogue of Odoo's field/method histogram). Same type, same collapse
//!   property (classes.md:42): structurally-identical classes dedup.
//! - **Resolution:** [`ClassView`](lance_graph_contract::class_view::ClassView) —
//!   [`WikidataClassView`] impls it, so the contract trait resolves a Wikidata
//!   class exactly as `RegistryClassView` resolves an Odoo one.
//!
//! ## Through-the-cache (OD-DOLCE, #441 `b31464d`)
//! Each class carries the cache-resolved **`dolce_id` u8** (0..3 =
//! [`dolce_id`](crate::class_resolver::dolce_id)), NOT a DOLCE enum — the basin is
//! that u8 directly. No DOLCE semantics live on the Wikidata path; the cache is the
//! authority. The 115M streaming load is a separate, deferred plan
//! (`wikidata-hhtl-load.md`).

use std::collections::HashMap;

use lance_graph_contract::class_view::{ClassId, ClassView, FieldMask};
use lance_graph_contract::hash::fnv1a;
use lance_graph_contract::hhtl::NiblePath;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};

use crate::class_resolver::dolce_id;
use crate::odoo_blueprint::class_signature::StructuralSignature;

/// A curated Wikidata class — the minimal shape the N4 falsifier needs (the full
/// streaming load is deferred). `properties` are the *representative core*
/// property-id set, in declaration order (position `i` = stable [`FieldMask`] bit
/// `i`, N3 append-only) — not the exhaustive Wikidata property list.
#[derive(Debug, Clone)]
pub struct WikidataClass {
    /// The class discriminator (reuses the #441 `ClassId` width — u16).
    pub class_id: ClassId,
    /// The Wikidata QID, e.g. `"Q5"` (human).
    pub qid: &'static str,
    /// The English label (resolved late in real loads; carried here for the fixture).
    pub label: &'static str,
    /// The **cache-resolved** DOLCE id (`0..3` = [`dolce_id`]). basin = this u8.
    pub dolce_id: u8,
    /// The P279 `subClassOf` nibble path BELOW the basin (each `0x0..=0xF`).
    pub subclass_path: &'static [u8],
    /// The present property-id set (P-numbers), declaration order = bit position.
    pub properties: &'static [&'static str],
}

impl WikidataClass {
    /// The HHTL nibble path: basin (DOLCE) + the P279 `subClassOf` descent. This is
    /// the entity's 16ⁿ bucket address (`wikidata-hhtl-load.md`:42).
    #[must_use]
    pub fn nibble_path(&self) -> NiblePath {
        let mut p = NiblePath::root(self.dolce_id);
        for &n in self.subclass_path {
            p = p.child(n);
        }
        p
    }

    /// The presence bitmask — bit `i` set for declared property `i` (the
    /// facet-bitmask). Reuses [`FieldMask`] verbatim.
    #[must_use]
    pub fn presence_mask(&self) -> FieldMask {
        let mut m = FieldMask::EMPTY;
        // Cap at MAX_FIELDS: positions beyond 64 are not addressable by the `u64`
        // mask (the bit-budget escape is a ref, deferred). `i < 64` here, so the
        // `i as u8` can never wrap (Codex P2 #442 — no aliasing onto low bits).
        let cap = (FieldMask::MAX_FIELDS as usize).min(self.properties.len());
        for i in 0..cap {
            m = m.with(i as u8);
        }
        m
    }

    /// The structural signature (shape-family key) — a deterministic hash over the
    /// DOLCE id + the **canonical (sorted, deduped) property-id set**. Label- and
    /// QID-independent, so two structurally-identical classes collapse to one
    /// family (classes.md:42, now on Wikidata). Reuses the canonical
    /// [`fnv1a`](lance_graph_contract::hash::fnv1a); truncated to the
    /// [`StructuralSignature`] `u32`.
    ///
    /// **Scale freeze (TD-WIKI-SCALE):** `StructuralSignature` is `u32` — ~50%
    /// birthday-collision near ~77k distinct shape-families. Safe for the curated
    /// corpus + Odoo; at full Wikidata load scale, widening to `u64` is a #441
    /// contract decision (flagged by the D-ARM-14 review of #442), to land WITH the
    /// deferred loader, not unilaterally here.
    #[must_use]
    pub fn signature(&self) -> StructuralSignature {
        let mut props: Vec<&str> = self.properties.to_vec();
        props.sort_unstable();
        props.dedup();
        let mut buf: Vec<u8> = Vec::with_capacity(1 + props.len() * 6);
        buf.push(self.dolce_id);
        for p in props {
            buf.extend_from_slice(p.as_bytes());
            buf.push(0); // separator — keeps {P1,P23} distinct from {P12,P3}
        }
        StructuralSignature(fnv1a(&buf) as u32)
    }

    /// The domain-independent **D-CLS triple** — identical in *shape* to the Odoo
    /// one: `(class_id, shape_hash, presence_bitmask)`. This identity IS the N4
    /// falsification target.
    #[must_use]
    pub fn dcls_triple(&self) -> (ClassId, StructuralSignature, FieldMask) {
        (self.class_id, self.signature(), self.presence_mask())
    }

    /// The class's fields as contract [`FieldRef`]s (property-id = `predicate_iri`;
    /// label defaults to the property-id, resolved late from the cache).
    fn field_refs(&self) -> Vec<FieldRef> {
        // Cap at MAX_FIELDS so `field_count() <= FieldMask::MAX_FIELDS` (the
        // ClassView contract) and `render_rows` never iterates past the
        // addressable mask — the same discipline as the Odoo `object_view()` path
        // (Codex P2 #442).
        self.properties
            .iter()
            .take(FieldMask::MAX_FIELDS as usize)
            .map(|p| FieldRef::new(*p, *p))
            .collect()
    }
}

/// The curated Wikidata corpus — 6 real classes spanning two DOLCE basins, with
/// two genuine structural twins (film ≡ TV series share the core AV-work shape)
/// and one genuine subclass chain (human ⊂ person). Honest minimal data: the
/// collapse + inheritance are properties of the *shapes*, asserted on the corpus,
/// not stipulated.
#[must_use]
pub fn curated_wikidata_classes() -> Vec<WikidataClass> {
    vec![
        WikidataClass {
            class_id: 1,
            qid: "Q215627",
            label: "person",
            dolce_id: dolce_id::ENDURANT,
            subclass_path: &[0x1],
            properties: &["P21", "P569", "P570"], // sex, birth, death
        },
        WikidataClass {
            class_id: 2,
            qid: "Q5",
            label: "human",
            dolce_id: dolce_id::ENDURANT,
            subclass_path: &[0x1, 0x2], // person → human (IS-A; path extends)
            properties: &["P21", "P569", "P570", "P106"], // + occupation (the delta)
        },
        WikidataClass {
            class_id: 3,
            qid: "Q515",
            label: "city",
            dolce_id: dolce_id::ENDURANT,
            subclass_path: &[0x3],
            properties: &["P1082", "P625", "P17"], // population, coords, country
        },
        WikidataClass {
            class_id: 4,
            qid: "Q11424",
            label: "film",
            dolce_id: dolce_id::PERDURANT,
            subclass_path: &[0x0],
            properties: &["P57", "P577", "P161"], // director, pubdate, cast
        },
        WikidataClass {
            class_id: 5,
            qid: "Q5398426",
            label: "television series",
            dolce_id: dolce_id::PERDURANT,
            subclass_path: &[0x1],
            properties: &["P57", "P577", "P161"], // SAME core AV-work shape as film
        },
        WikidataClass {
            class_id: 6,
            qid: "Q1656682",
            label: "event",
            dolce_id: dolce_id::PERDURANT,
            subclass_path: &[0x2],
            properties: &["P585", "P276"], // point-in-time, location
        },
    ]
}

/// Group the corpus into shape-families by [`StructuralSignature`] — the discovered
/// taxonomy (classes.md:43), now over Wikidata. Returns `(signature → member QIDs)`,
/// sorted by signature for deterministic output.
#[must_use]
pub fn shape_families(classes: &[WikidataClass]) -> Vec<(StructuralSignature, Vec<&'static str>)> {
    use std::collections::BTreeMap;
    let mut fams: BTreeMap<u32, Vec<&'static str>> = BTreeMap::new();
    for c in classes {
        fams.entry(c.signature().0).or_default().push(c.qid);
    }
    fams.into_iter()
        .map(|(k, v)| (StructuralSignature(k), v))
        .collect()
}

/// The ontology-side [`ClassView`] over a Wikidata corpus — the Wikidata analogue
/// of [`RegistryClassView`](crate::class_resolver::RegistryClassView). Proves the
/// #441 contract trait resolves a Wikidata class with no change.
pub struct WikidataClassView {
    fields: HashMap<ClassId, Vec<FieldRef>>,
    dolce: HashMap<ClassId, u8>,
    empty: Vec<FieldRef>,
}

impl WikidataClassView {
    /// Build the view over a curated corpus.
    #[must_use]
    pub fn new(classes: &[WikidataClass]) -> Self {
        let mut fields = HashMap::new();
        let mut dolce = HashMap::new();
        for c in classes {
            fields.insert(c.class_id, c.field_refs());
            dolce.insert(c.class_id, c.dolce_id);
        }
        Self {
            fields,
            dolce,
            empty: Vec::new(),
        }
    }
}

impl ClassView for WikidataClassView {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        self.fields
            .get(&class)
            .map_or(self.empty.as_slice(), |v| v.as_slice())
    }

    fn template(&self, class: ClassId) -> DisplayTemplate {
        // Same size heuristic as Odoo's `object_view` (≤4 fields → Card).
        match self.fields.get(&class) {
            Some(f) if f.len() <= 4 => DisplayTemplate::Card,
            _ => DisplayTemplate::Detail,
        }
    }

    fn dolce_category_id(&self, class: ClassId) -> u8 {
        // Resolved from the cache (carried as the opaque u8); default Endurant.
        self.dolce
            .get(&class)
            .copied()
            .unwrap_or(dolce_id::ENDURANT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_qid(qid: &str) -> WikidataClass {
        curated_wikidata_classes()
            .into_iter()
            .find(|c| c.qid == qid)
            .unwrap()
    }

    #[test]
    fn dolce_resolves_to_basin_through_the_cache_u8() {
        // basin = the cache-resolved dolce_id; the router never embeds DOLCE.
        assert_eq!(by_qid("Q5").nibble_path().basin(), Some(dolce_id::ENDURANT));
        assert_eq!(
            by_qid("Q515").nibble_path().basin(),
            Some(dolce_id::ENDURANT)
        );
        assert_eq!(
            by_qid("Q11424").nibble_path().basin(),
            Some(dolce_id::PERDURANT)
        );
        assert_eq!(
            by_qid("Q1656682").nibble_path().basin(),
            Some(dolce_id::PERDURANT)
        );
    }

    #[test]
    fn corpus_collapses_to_fewer_shape_families() {
        // The N4 falsification: the class-meta-DTO's collapse property (classes.md:42)
        // holds on Wikidata, not just Odoo — film ≡ TV series share the AV-work shape.
        let corpus = curated_wikidata_classes();
        let families = shape_families(&corpus);
        assert!(
            families.len() < corpus.len(),
            "corpus must collapse to FEWER shape-families ({} classes → {} families)",
            corpus.len(),
            families.len()
        );
        // film (Q11424) and TV series (Q5398426) are the genuine twins.
        assert_eq!(
            by_qid("Q11424").signature(),
            by_qid("Q5398426").signature(),
            "film and TV series share one structural signature"
        );
        // human carries an extra property → it is NOT person's twin.
        assert_ne!(by_qid("Q5").signature(), by_qid("Q215627").signature());
    }

    #[test]
    fn dcls_triple_shape_is_domain_independent() {
        // The Wikidata triple is the SAME (ClassId, StructuralSignature, FieldMask)
        // as Odoo's — the identity that proves the meta-DTO is not Odoo-overfit.
        let human = by_qid("Q5");
        let (id, sig, mask) = human.dcls_triple();
        assert_eq!(id, 2);
        assert_eq!(sig, human.signature());
        assert_eq!(
            mask.count(),
            human.properties.len() as u32,
            "one present bit per property"
        );
        assert!(mask.has(0) && mask.has(3) && !mask.has(4));
    }

    #[test]
    fn classview_resolves_a_wikidata_class_unchanged() {
        // The #441 ClassView trait resolves a Wikidata class with no modification.
        let corpus = curated_wikidata_classes();
        let view = WikidataClassView::new(&corpus);
        assert_eq!(view.dolce_category_id(2), dolce_id::ENDURANT); // human
        assert_eq!(view.field_count(2), 4);
        assert_eq!(view.template(2), DisplayTemplate::Card);
        assert_eq!(view.template(1), DisplayTemplate::Card); // person, 3 fields
                                                             // render_rows skips off-bits (C2 presence-only): mask with only bit 0 + bit 3.
        let mask = FieldMask::EMPTY.with(0).with(3);
        let rows = view.render_rows(2, mask);
        assert_eq!(rows.len(), 2, "only the two set bits render");
        assert_eq!(rows[0].predicate, "P21");
        assert_eq!(rows[1].predicate, "P106");
    }

    #[test]
    fn subclass_inherits_path_and_mask_as_delta() {
        // human ⊂ person: the P279 path extends AND the presence mask inherits.
        let person = by_qid("Q215627");
        let human = by_qid("Q5");
        assert!(
            person.nibble_path().is_ancestor_of(human.nibble_path()),
            "person's nibble path is a prefix of human's (IS-A reachability)"
        );
        // human's mask = person's mask inheriting the occupation (P106) delta bit.
        let delta = FieldMask::EMPTY.with(3); // P106 is human's 4th property (bit 3)
        assert_eq!(person.presence_mask().inherit(delta), human.presence_mask());
        // basin (DOLCE) is preserved down the subclass path.
        assert_eq!(
            person.nibble_path().basin(),
            human.nibble_path().basin(),
            "subclassing never changes the DOLCE basin"
        );
    }

    #[test]
    fn oversized_class_caps_to_fieldmask_width() {
        // A class with > MAX_FIELDS properties must cap — no `usize`→`u8` wraparound
        // aliasing, field_count within the ClassView contract (Codex P2 #442), the
        // same discipline as Odoo's `object_view()`.
        let many: Vec<&'static str> = (0..70)
            .map(|i| &*Box::leak(format!("P{i}").into_boxed_str()))
            .collect();
        let props: &'static [&'static str] = Box::leak(many.into_boxed_slice());
        let big = WikidataClass {
            class_id: 99,
            qid: "Qbig",
            label: "big",
            dolce_id: dolce_id::ENDURANT,
            subclass_path: &[],
            properties: props,
        };
        assert_eq!(
            big.presence_mask().count(),
            FieldMask::MAX_FIELDS,
            "presence mask caps at MAX_FIELDS — property 64+ never wraps onto a low bit"
        );
        let view = WikidataClassView::new(std::slice::from_ref(&big));
        assert_eq!(
            view.field_count(99),
            FieldMask::MAX_FIELDS as usize,
            "field_count <= MAX_FIELDS (the ClassView contract)"
        );
        // Rendering the full mask yields exactly MAX_FIELDS rows, not 70.
        let rows = view.render_rows(99, big.presence_mask());
        assert_eq!(rows.len(), FieldMask::MAX_FIELDS as usize);
    }
}
