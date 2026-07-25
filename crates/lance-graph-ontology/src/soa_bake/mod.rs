//! **SoA bake** — the codec-native ontology cache (issue #845).
//!
//! An RDF/OWL/OBO source file is *address-verbatim* — every triple respells
//! three IRIs as strings and pays label bytes in full on every relation. That
//! is what makes FMA 253 MB / LOINC 705 MB, and 90 %+ of those bytes are pure
//! repetition. The **SoA bake** replaces the verbatim form with a struct-of-
//! arrays layout whose columns compose four orthogonal compressions:
//!
//! | column | encoding | savings |
//! |---|---|---|
//! | address (canonical GUID) | fixed 16 B, prefix-sorted | O(1) `is_a` = prefix containment (edges free) |
//! | label | [`LabelColumn`] (token codebook + u16 indices) | ~3× on raw label bytes at simple encoding (measured on FMA); composes further with shared codebook + zstd + templating |
//! | edges (`part_of`, cross-walks) | [`EdgePair`] columns, delta-encoded | address pairs replace IRI triples |
//! | ClassView | one per class prefix (schema inheritance) | not per instance |
//!
//! Given the four columns above, the bake is `single-digit MB` for FMA
//! (source: 253 MB OWL) at mmap-time; consumer resident-set is the working
//! set, not the file size.
//!
//! ## Schema versioning ([`SchemaVersion`])
//!
//! Every bake carries an explicit [`SchemaVersion`] that names its
//! `(codebook_schema, classview_schema, address_schema)` triple. A store that
//! references the bake by numeric pointer (rather than by verbatim label)
//! resolves those pointers **against this pinned schema version**, so a
//! codebook re-mint that reshuffles token ids never corrupts stored
//! references. The `ClassView` (per class prefix) carries the same pointer —
//! its own schema version + the codebook version it was built against — so
//! evolving the schema is a versioned migration, not a silent breaking
//! change.
//!
//! ## Status
//!
//! This module ships the **label-codebook half** first (issue #845 empirical
//! anchor: ~3× at u16 on real FMA label bytes, measurably composable). Types
//! for the address column + edge pairs + `ClassView`-inheritance + bake
//! driver are scaffolded here and are the next milestones inside the same
//! crate; the dominant compression from source-file → cache size comes from
//! the address-trie half (IRI elimination via prefix containment) once that
//! lands.
//!
//! ## Follow-on (the trie codebook)
//!
//! The flat u16-index [`LabelColumn`] is the baseline (~3× on FMA). The next
//! algorithmic iteration is **trie codebook hydration** — a builder that
//! observes the token stream and emits a *shared prefix (or BPE-merged)
//! trie* whose interior nodes ARE the reused n-grams (`"of aorta"`,
//! `"systemic artery"`). Each label then resolves to a single leaf id in the
//! trie, and the trie itself compresses because every ancestry node is
//! shared across all descendants. The 206.9× *token reuse* factor measured
//! on FMA is the ceiling that this approach genuinely approaches (the flat
//! bake only realizes a fraction because it re-emits the index sequence per
//! label). Kept as a seam here; tracked as the next iteration of issue #845.
//!
//! Downstream stores separately switch from verbatim labels to
//! `SchemaVersion`-pinned codebook pointers — indirection saves both bytes
//! and re-mint safety.

pub mod label_codebook;

pub use label_codebook::{amortization, raw_bytes, LabelCodebook, LabelColumn};

/// The three-schema version triple every bake carries. Consumer pointers
/// resolve against this pinned version so a codebook re-mint / classview
/// evolution / address-scheme rev is a *versioned migration*, not a silent
/// break. See module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaVersion {
    /// Version of the label-codebook token table (bumps when tokens are
    /// added / renumbered).
    pub codebook: u16,
    /// Version of the `ClassView` layout for this bake (bumps when a class
    /// prefix's field basis / DOLCE upper changes).
    pub classview: u16,
    /// Version of the canonical-GUID address scheme (bumps only on a
    /// substrate change; see lance-graph `CANON` § *Minimal SoA node*).
    pub address: u16,
}

impl SchemaVersion {
    /// The initial version — codebook v1, classview v1, address v1 (the
    /// current canonical `NodeGuid` layout).
    pub const V1: Self = Self {
        codebook: 1,
        classview: 1,
        address: 1,
    };
}

/// One `(source, destination, kind)` relation, stored as an **address pair**
/// rather than a verbatim RDF triple. Two 16-B canonical GUIDs + a 1-B edge
/// kind = 33 B (vs three ~80-B IRIs = ~240 B verbatim, before dictionary
/// encoding); delta-encoded against a prefix-sorted source column, the src
/// bytes typically compress to near zero.
///
/// `is_a` edges are NOT stored here — the address trie's prefix already
/// carries them (a child's address extends its parent's prefix, so
/// subsumption = prefix containment). Only cross-cutting edges
/// (`part_of` across `is_a` families, cross-ontology cross-walks) need a
/// pair row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgePair {
    pub src: u128,
    pub dst: u128,
    pub kind: EdgeKind,
}

/// The catalogued edge types that need explicit pair rows (i.e. NOT `is_a`,
/// which is implicit in the address trie). Kept as a u8 enum for the pair
/// column's byte layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EdgeKind {
    PartOf = 1,
    /// Cross-ontology alignment (e.g. FMA ↔ RadLex, LOINC ↔ SNOMED).
    SameAs = 2,
    /// A domain-specific link (e.g. CPIC drug → RxNorm ingredient). The
    /// namespace-registered id resolves the specific semantics.
    Related = 3,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_version_v1_is_the_expected_triple() {
        // V1 pins the substrate everything else is compared against.
        assert_eq!(
            SchemaVersion::V1,
            SchemaVersion {
                codebook: 1,
                classview: 1,
                address: 1,
            }
        );
    }

    #[test]
    fn edge_pair_layout_is_the_documented_shape() {
        let e = EdgePair {
            src: 0x0A01_0000_0000_0000_0000_0000_0000_0001,
            dst: 0x0A01_0000_0000_0000_0000_0000_0000_0002,
            kind: EdgeKind::PartOf,
        };
        assert_ne!(e.src, e.dst);
        assert_eq!(e.kind as u8, 1);
    }
}
