//! `codebook` — per-family codebook (D-GV2-2, feature `guid-v2-tail`).
//!
//! The finer sibling of [`class_view`](crate::class_view) (`classid → ClassView`):
//! here **`family → Codebook`**. Each family owns a ≤256-entry vocabulary that its
//! nodes index by a **1-byte in-family adapter** — the 256×256 Morton centroid
//! tile of `E-UNIFORM-MORTON-TILE-PYRAMID`, with ≤256 leaves for the 1-byte index.
//!
//! Why per-family (not one global codebook): it dissolves the aiwar "60 noisy
//! families" at the root — each family's vocabulary is small and clean, and a
//! within-family reference is an **exact** index into that family's codebook (no
//! `& 0xFF` low-byte aliasing). The `family` tier (u16) selects the codebook
//! (head-only routing); the 1-byte index resolves within it. Cross-family edges
//! carry `(family, index)` and decode via [`FamilyCodebookRegistry::resolve`].
//!
//! A family that outgrows 256 entries **splits** (mint a sub-family — cheap with
//! the v2 16-bit family tier), never widens the byte ([`Codebook::intern`]
//! returns `None` on overflow as the split signal). The codebook is the family
//! node's **episodic basin** content (`E-MIXIN-IS-AN-ADDRESS-REFERENCE-NOT-A-COPY`):
//! members reference it by the 1-byte index, the shared vocabulary lives once.
//!
//! This module is the TYPE + in-memory registry (the `LazyLock` tier). The
//! Lance-backed persistence + `OntologyRegistry` integration are deferred to the
//! ontology-crate wiring step (see plan `guid-v2-tail-per-family-codebook-v1.md`).

use std::collections::HashMap;

/// Max entries per family codebook — the 1-byte in-family index cap. A family
/// that needs more SPLITS (mint a sub-family), never widens the byte.
pub const CODEBOOK_CAP: usize = 256;

/// A per-family codebook: insertion-ordered label interning, `index ↔ label`.
/// `index` is the 1-byte in-family adapter value (`0..len`). ≤[`CODEBOOK_CAP`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Codebook {
    entries: Vec<String>,          // index → label (insertion order)
    by_label: HashMap<String, u8>, // label → index
}

impl Codebook {
    /// An empty codebook.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern `label` → its 1-byte index (insertion order, deduped). Returns
    /// `None` if the codebook is full (256) and `label` is new — the caller must
    /// SPLIT the family (the `CODEBOOK_CAP` overflow signal). An already-present
    /// label always resolves (even at capacity).
    pub fn intern(&mut self, label: &str) -> Option<u8> {
        if let Some(&i) = self.by_label.get(label) {
            return Some(i);
        }
        if self.entries.len() >= CODEBOOK_CAP {
            return None;
        }
        let i = self.entries.len() as u8;
        self.entries.push(label.to_string());
        self.by_label.insert(label.to_string(), i);
        Some(i)
    }

    /// The 1-byte index of `label`, if interned.
    pub fn index_of(&self, label: &str) -> Option<u8> {
        self.by_label.get(label).copied()
    }

    /// The label at `index`, if present.
    pub fn label(&self, index: u8) -> Option<&str> {
        self.entries.get(index as usize).map(String::as_str)
    }

    /// Number of interned entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the codebook is at [`CODEBOOK_CAP`] (a new label would overflow →
    /// split the family).
    pub fn is_full(&self) -> bool {
        self.entries.len() >= CODEBOOK_CAP
    }
}

/// `family → Codebook` — the per-family codebook registry, the finer sibling of
/// `classid → ClassView`. In-memory (the `LazyLock` tier); a Lance-backed,
/// `OntologyRegistry`-integrated variant is deferred.
#[derive(Debug, Clone, Default)]
pub struct FamilyCodebookRegistry {
    books: HashMap<u16, Codebook>,
}

impl FamilyCodebookRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The codebook for `family`, creating an empty one if absent.
    pub fn entry(&mut self, family: u16) -> &mut Codebook {
        self.books.entry(family).or_default()
    }

    /// The codebook for `family`, if it exists (read-only).
    pub fn get(&self, family: u16) -> Option<&Codebook> {
        self.books.get(&family)
    }

    /// Intern `label` into `family`'s codebook → its 1-byte index. `None` on
    /// codebook overflow (split the family).
    pub fn intern(&mut self, family: u16, label: &str) -> Option<u8> {
        self.entry(family).intern(label)
    }

    /// Resolve a cross-family reference `(family, index)` → label — the decode of
    /// an out-of-family adapter / `references` edge.
    pub fn resolve(&self, family: u16, index: u8) -> Option<&str> {
        self.books.get(&family).and_then(|cb| cb.label(index))
    }

    /// Number of families with a codebook.
    pub fn families(&self) -> usize {
        self.books.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_dedups_and_assigns_sequential_indices() {
        let mut cb = Codebook::new();
        assert_eq!(cb.intern("Nation"), Some(0));
        assert_eq!(cb.intern("TechCompany"), Some(1));
        assert_eq!(cb.intern("Nation"), Some(0)); // dedup
        assert_eq!(cb.len(), 2);
        assert_eq!(cb.index_of("TechCompany"), Some(1));
        assert_eq!(cb.label(0), Some("Nation"));
        assert_eq!(cb.label(9), None);
    }

    #[test]
    fn codebook_overflow_signals_split() {
        let mut cb = Codebook::new();
        for i in 0..CODEBOOK_CAP {
            assert!(cb.intern(&format!("e{i}")).is_some());
        }
        assert!(cb.is_full());
        // a NEW label overflows → None (split the family)…
        assert_eq!(cb.intern("one_too_many"), None);
        // …but an already-interned label still resolves at capacity.
        assert_eq!(cb.intern("e0"), Some(0));
    }

    #[test]
    fn registry_scopes_codebooks_per_family() {
        // The SAME label gets INDEPENDENT indices in different families — the
        // whole point of per-family scoping (no global contamination).
        let mut reg = FamilyCodebookRegistry::new();
        assert_eq!(reg.intern(0x0001, "Issue"), Some(0));
        assert_eq!(reg.intern(0x0001, "Bug"), Some(1));
        assert_eq!(reg.intern(0x0002, "Issue"), Some(0)); // family 2's own index 0
        assert_eq!(reg.families(), 2);
        // cross-family resolve (family, index) → label
        assert_eq!(reg.resolve(0x0001, 1), Some("Bug"));
        assert_eq!(reg.resolve(0x0002, 0), Some("Issue"));
        assert_eq!(reg.resolve(0x0099, 0), None); // unknown family
    }
}
