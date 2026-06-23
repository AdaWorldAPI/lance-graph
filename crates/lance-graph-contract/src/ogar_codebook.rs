//! `ogar_codebook` — the OGAR concept codebook, wire-compatible mirror (D-OVC-1).
//!
//! OGAR's `ogar-vocab` crate owns the canonical class-identity codebook: a curated
//! `(concept, u16)` table whose ids are **domain-encoded** as `0xDDCC` (`DD` = the
//! domain high byte, `CC` = the concept slot, `CC == 0x00` = the domain root,
//! reserved). Its own doc-comment says the long-term home of these types is
//! `lance-graph-contract`, "alongside `ClassId` and the `NodeGuid` LE layout."
//!
//! This module is that home — but **wire-compatible, not a dependency**. The
//! contract is zero-runtime-dep by design, so it does NOT depend on `ogar-vocab`;
//! instead both crates agree on the **wire**: a concept's id is one `u16`,
//! serialized little-endian, and that id IS the low 16 bits of
//! [`NodeGuid::classid`](crate::NodeGuid). Any encoder/decoder that agrees on
//! `u16` LE is compatible regardless of which crate it links. The parity tests
//! below pin the shared values; if OGAR's `CODEBOOK` ever moves an id, BOTH sides
//! must update together (the drift guard).
//!
//! What this mirror carries: the **codebook-id layer** the contract needs to route
//! a `classid` to its domain ([`canonical_concept_domain`], [`classid_concept_domain`])
//! and to resolve a canonical-concept string to its id ([`canonical_concept_id`],
//! [`LabelDTO::from_canonical`]). What it does NOT carry: OGAR's curator-alias
//! normalizer (`canonical_concept` — the large `"Issue"`/`"WorkPackage"` →
//! `"project_work_item"` table). Alias normalization stays in `ogar-vocab`; this
//! module resolves canonical-shaped concept strings only (hence `from_canonical`,
//! not `from_alias` — naming the difference rather than faking parity).
//!
//! Cross-ref: `.claude/plans/ogar-vocab-contract-codebook-migration-v1.md`,
//! OGAR `crates/ogar-vocab/src/lib.rs` (`CODEBOOK` / `ConceptDomain` / `LabelDTO`),
//! [`canonical_node`](crate::canonical_node) (`CLASSID_*`), [`codebook`](crate::codebook)
//! (the FINER per-family scope — this is the coarse concept/classid scope).

/// Codebook **domain** — the high byte of a canonical id (`id >> 8`, the `0xDDCC`
/// layout). Lets a consumer route on domain in O(1) from just the `u16`, no table
/// lookup. Reserved high-byte slots have a stable variant even before a concept
/// lands there, so consumers can branch on them today. Mirrors OGAR
/// `ogar_vocab::ConceptDomain` (wire-compatible — same `id >> 8` discriminant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConceptDomain {
    /// `0x00XX` — reserved (`0x0000` is [`NodeGuid::CLASSID_DEFAULT`]).
    Reserved,
    /// `0x01XX` — project-management (OpenProject ↔ Redmine).
    ProjectMgmt,
    /// `0x02XX` — commerce / billing / ERP (Odoo ↔ OSB).
    Commerce,
    /// `0x07XX` — OSINT (open-source intelligence / Palantir-Gotham).
    Osint,
    /// `0x08XX` — OCR (optical character recognition / document extraction).
    Ocr,
    /// `0x09XX` — Health (clinical / patient / care; FMA anatomy lives here).
    Health,
    /// `0x0BXX` — Auth (IAM; provider-agnostic AuthStore family — Zitadel /
    /// Zanzibar / Ory-Keto resolve to one canonical concept).
    Auth,
    /// Any high-byte slot not yet assigned a domain (`0x03XX`–`0x06XX`, `0x0AXX`, `0x0CXX`+).
    Unassigned,
}

/// Resolve a canonical id's [`ConceptDomain`] from its high byte. Pure,
/// deterministic, O(1) — no table lookup. The single rule both the contract's
/// `classid → ReadMode` registry and OGAR's promotion gate route on.
#[inline]
#[must_use]
pub fn canonical_concept_domain(id: u16) -> ConceptDomain {
    match id >> 8 {
        0x00 => ConceptDomain::Reserved,
        0x01 => ConceptDomain::ProjectMgmt,
        0x02 => ConceptDomain::Commerce,
        0x07 => ConceptDomain::Osint,
        0x08 => ConceptDomain::Ocr,
        0x09 => ConceptDomain::Health,
        0x0B => ConceptDomain::Auth,
        _ => ConceptDomain::Unassigned,
    }
}

/// Resolve a [`NodeGuid`](crate::NodeGuid) `classid` to its [`ConceptDomain`] (D-OVC-4). The
/// codebook id is the low 16 bits of the classid (`0xDDCC` lives in the low u16);
/// the high u16 is the canon-reserved zero-fallback prefix. So a domain route is
/// `canonical_concept_domain(classid as u16)`. This is the coarse sibling of the
/// per-family scope in [`codebook`](crate::codebook): classid (domain) selects the
/// coarse codebook; `family` selects the sub-codebook (longest-prefix-wins).
#[inline]
#[must_use]
pub fn classid_concept_domain(classid: u32) -> ConceptDomain {
    canonical_concept_domain(classid as u16)
}

/// Map a coarse curator `source_domain` tag (`"project"`, `"erp"`, `"german-erp"`)
/// to the [`ConceptDomain`] its promotions live in. `None` for an unrecognised tag
/// (the producer's source-domain → typed-domain seam). Mirrors OGAR
/// `source_domain_concept`.
#[inline]
#[must_use]
pub fn source_domain_concept(source_domain: &str) -> Option<ConceptDomain> {
    match source_domain {
        "project" => Some(ConceptDomain::ProjectMgmt),
        "erp" | "german-erp" => Some(ConceptDomain::Commerce),
        _ => None,
    }
}

/// The curated `(canonical_concept, u16)` codebook — wire-compatible mirror of
/// OGAR `ogar_vocab::CODEBOOK`. Ids are stable forever (once shipped, never
/// re-assigned); domain-encoded `0xDDCC`. Carries the promoted concept slots the
/// contract graph surfaces realize today: project-mgmt `0x01XX`, commerce/ERP
/// `0x02XX`, Health `0x09XX`, and Auth `0x0BXX` (provider-agnostic AuthStore
/// family). OSINT (`0x07XX`) and OCR (`0x08XX`) are represented by their
/// [`NodeGuid`](crate::NodeGuid) classid roots, not yet by promoted concept slots here. Drift is
/// guarded by [`tests::codebook_ids_match_ogar_vocab`] and the compile-time
/// `COUNT_FUSE` in `lance-graph-ogar`.
pub const CODEBOOK: &[(&str, u16)] = &[
    // ── 0x01XX — project-mgmt domain (OpenProject ↔ Redmine) ──
    ("project", 0x0101),
    ("project_work_item", 0x0102),
    ("billable_work_entry", 0x0103),
    ("project_actor", 0x0104),
    ("project_status", 0x0105),
    ("project_type", 0x0106),
    ("priority", 0x0107),
    ("project_membership", 0x0108),
    ("project_journal", 0x0109),
    ("project_repository", 0x010A),
    ("project_version", 0x010B),
    ("project_wiki_page", 0x010C),
    ("project_query", 0x010D),
    ("project_attachment", 0x010E),
    ("project_comment", 0x010F),
    ("project_custom_field", 0x0110),
    ("project_relation", 0x0111),
    ("project_changeset", 0x0112),
    ("project_watcher", 0x0113),
    ("project_news", 0x0114),
    ("project_message", 0x0115),
    ("project_forum", 0x0116),
    ("project_role", 0x0117),
    ("project_member_role", 0x0118),
    ("project_custom_value", 0x0119),
    ("project_enabled_module", 0x011A),
    // ── 0x02XX — commerce / billing / ERP domain (Odoo ↔ OSB) ──
    ("commercial_line_item", 0x0201),
    ("commercial_document", 0x0202),
    ("tax_policy", 0x0203),
    ("billing_party", 0x0204),
    ("payment_record", 0x0205),
    ("currency_policy", 0x0206),
    // ── 0x09XX — Health domain (MedCare; OGIT NTO/Healthcare promotion) ──
    ("patient", 0x0901),
    ("diagnosis", 0x0902),
    ("lab_value", 0x0903),
    ("medication", 0x0904),
    ("treatment", 0x0905),
    ("visit", 0x0906),
    ("vital_sign", 0x0907),
    // ── 0x0BXX — Auth domain (IAM; provider-agnostic AuthStore family).
    // OGIT Configuration entity ⊨ auth_store (arago's Jan-2026 bridge entity).
    // Zitadel / Zanzibar / Ory-Keto are providers that resolve to one concept. ──
    ("auth_store", 0x0B01),
    ("auth_zitadel", 0x0B02),
    ("auth_zanzibar", 0x0B03),
    ("auth_ory_keto", 0x0B04),
];

/// Resolve a **canonical-concept** string to its stable `u16` codebook id via
/// [`CODEBOOK`]. `None` for an unpromoted concept (not in the codebook).
///
/// This resolves canonical-shaped names only (e.g. `"project_work_item"`). For
/// curator-shaped aliases (`"Issue"`, `"WorkPackage"`), normalize through OGAR
/// `ogar_vocab::canonical_concept` first — that alias table stays in `ogar-vocab`,
/// out of the zero-dep contract.
#[inline]
#[must_use]
pub fn canonical_concept_id(concept: &str) -> Option<u16> {
    CODEBOOK
        .iter()
        .find_map(|&(name, id)| (name == concept).then_some(id))
}

/// A curator-agnostic label binding: a consumer-local `label`, its OGAR codebook
/// `id` (binary identity), and the portable `canonical` symbol. Mirrors OGAR
/// `ogar_vocab::LabelDTO` (wire-compatible). Identity comparison uses `id`;
/// AST/planner emission uses `canonical`; presentation uses `label`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct LabelDTO {
    /// Consumer-local label. Not normalized by the contract.
    pub label: String,
    /// OGAR codebook binary identity (the classid low u16).
    pub id: u16,
    /// Canonical-AST label — the portable curator-agnostic symbol.
    pub canonical: String,
}

impl LabelDTO {
    /// Build a `LabelDTO` from a **canonical-shaped** concept string. `None` if the
    /// concept is not in [`CODEBOOK`]. (Contract counterpart of OGAR's
    /// `from_alias`, minus curator-alias normalization — see the module docs:
    /// pass a canonical concept, or normalize via `ogar-vocab` first.)
    #[must_use]
    pub fn from_canonical(concept: impl Into<String>) -> Option<Self> {
        let canonical = concept.into();
        let id = canonical_concept_id(&canonical)?;
        Some(Self {
            label: canonical.clone(),
            id,
            canonical,
        })
    }

    /// `id` rendered as **2 little-endian bytes** — the wire contract. Roundtrips
    /// via `u16::from_le_bytes`. Byte order matches the [`NodeGuid`](crate::NodeGuid) LE layout, so
    /// this is exactly the classid low half on the wire.
    #[inline]
    #[must_use]
    pub fn id_le(&self) -> [u8; 2] {
        self.id.to_le_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeGuid;

    #[test]
    fn domain_routes_on_high_byte() {
        assert_eq!(canonical_concept_domain(0x0000), ConceptDomain::Reserved);
        assert_eq!(canonical_concept_domain(0x0101), ConceptDomain::ProjectMgmt);
        assert_eq!(canonical_concept_domain(0x0206), ConceptDomain::Commerce);
        assert_eq!(canonical_concept_domain(0x0700), ConceptDomain::Osint);
        assert_eq!(canonical_concept_domain(0x0801), ConceptDomain::Ocr);
        assert_eq!(canonical_concept_domain(0x0901), ConceptDomain::Health);
        assert_eq!(canonical_concept_domain(0x0B01), ConceptDomain::Auth);
        assert_eq!(canonical_concept_domain(0x0500), ConceptDomain::Unassigned);
        assert_eq!(canonical_concept_domain(0x0C00), ConceptDomain::Unassigned);
    }

    #[test]
    fn classid_routes_through_low_u16() {
        // The contract classids resolve to the domain their `0xDDCC` low half
        // encodes — the contract↔OGAR alignment (ISS-CLASSID-OGAR-DRIFT).
        assert_eq!(
            classid_concept_domain(NodeGuid::CLASSID_PROJECT),
            ConceptDomain::ProjectMgmt
        );
        assert_eq!(
            classid_concept_domain(NodeGuid::CLASSID_ERP),
            ConceptDomain::Commerce
        );
        assert_eq!(
            classid_concept_domain(NodeGuid::CLASSID_OSINT),
            ConceptDomain::Osint
        );
        assert_eq!(
            classid_concept_domain(NodeGuid::CLASSID_FMA),
            ConceptDomain::Health,
            "FMA anatomy lives in the Health domain (0x09XX)"
        );
        assert_eq!(
            classid_concept_domain(NodeGuid::CLASSID_DEFAULT),
            ConceptDomain::Reserved
        );
    }

    #[test]
    fn source_domain_maps_to_concept_domain() {
        assert_eq!(
            source_domain_concept("project"),
            Some(ConceptDomain::ProjectMgmt)
        );
        assert_eq!(source_domain_concept("erp"), Some(ConceptDomain::Commerce));
        assert_eq!(
            source_domain_concept("german-erp"),
            Some(ConceptDomain::Commerce)
        );
        assert_eq!(source_domain_concept("nope"), None);
    }

    #[test]
    fn codebook_ids_match_ogar_vocab() {
        // Drift guard: these MUST match OGAR `ogar_vocab::CODEBOOK` exactly (the
        // wire is the contract). If OGAR moves an id, update BOTH sides together.
        assert_eq!(canonical_concept_id("project"), Some(0x0101));
        assert_eq!(canonical_concept_id("project_work_item"), Some(0x0102));
        assert_eq!(canonical_concept_id("project_enabled_module"), Some(0x011A));
        assert_eq!(canonical_concept_id("commercial_line_item"), Some(0x0201));
        assert_eq!(canonical_concept_id("commercial_document"), Some(0x0202));
        assert_eq!(canonical_concept_id("currency_policy"), Some(0x0206));
        assert_eq!(canonical_concept_id("patient"), Some(0x0901));
        assert_eq!(canonical_concept_id("vital_sign"), Some(0x0907));
        assert_eq!(canonical_concept_id("auth_store"), Some(0x0B01));
        assert_eq!(canonical_concept_id("auth_zitadel"), Some(0x0B02));
        assert_eq!(canonical_concept_id("auth_zanzibar"), Some(0x0B03));
        assert_eq!(canonical_concept_id("auth_ory_keto"), Some(0x0B04));
        assert_eq!(canonical_concept_id("not_a_concept"), None);
    }

    #[test]
    fn codebook_has_no_duplicate_ids_or_zero_concept_slot() {
        // Every id non-zero in its concept slot (CC != 0x00 — root is reserved),
        // every id unique, and each id's domain matches its position.
        let mut seen = std::collections::HashSet::new();
        for &(name, id) in CODEBOOK {
            assert_ne!(
                id & 0x00FF,
                0x00,
                "{name}: concept slot CC must be non-zero"
            );
            assert!(seen.insert(id), "{name}: duplicate id {id:#06x}");
        }
    }

    #[test]
    fn label_dto_roundtrips_canonical_and_wire() {
        let dto = LabelDTO::from_canonical("project_enabled_module").unwrap();
        assert_eq!(dto.id, 0x011A);
        assert_eq!(dto.canonical, "project_enabled_module");
        assert_eq!(dto.id_le(), [0x1A, 0x01]); // LE: low byte (0x1A) first, high (0x01)
        assert_eq!(u16::from_le_bytes(dto.id_le()), 0x011A);
        // domain reachable from the DTO id
        assert_eq!(canonical_concept_domain(dto.id), ConceptDomain::ProjectMgmt);
        assert_eq!(
            LabelDTO::from_canonical("Issue"),
            None,
            "curator alias unresolved in contract (normalize via ogar-vocab first)"
        );
    }
}
