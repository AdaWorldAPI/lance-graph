# PR: Add OGIT + OWL/DOLCE semantic layer + Ractor orchestration foundation in lance-graph-callcenter

## Summary

This PR introduces the foundation for turning `lance-graph-callcenter` into the **Rosetta Stone** and **Ractor orchestrator** between spear and the high-performance lance-graph substrate (DnPath + CAM-PQ).

It builds directly on the existing `DnPath` and CAM-PQ infrastructure without breaking changes.

## Goals

- Enable context-addressable semantic navigation (`DnPath` + OGIT + OWL/DOLCE)
- Map CAM Codebook archetypes and labels into a proper ontological space
- Prepare `lance-graph-callcenter` for Ractor-based orchestration
- Keep heavy semantic work behind the membrane so spear consumers stay lightweight

## Changes

### New Files
- `ogit_owl_dolce.rs` — Semantic mapping layer (CAM archetypes → DOLCE categories, codebook labels → OGIT)
- `dn_path_with_semantics.rs` — Proposed non-breaking extension to `DnPath` (`DnPathWithSemantics` + `SemanticContext`)
- `OGIT_OWL_DOLCE_RACTOR.md` — Architecture design document and phased implementation plan

### Modified Files (proposed)
- `dn_path.rs` — Can be extended with the content from `dn_path_with_semantics.rs`
- `lib.rs` — Will need small re-exports once the modules are integrated

## Design Highlights

- **Backward compatible**: Original `DnPath` and `scent()` behavior is unchanged.
- **CAM-PQ synergy**: Archetypes and codebook `label` column are mapped to OWL/DOLCE concepts.
- **Ractor-ready**: Structure prepared for future Ractor supervisor inside the membrane.
- **Clean separation**: Ontology work stays in `lance-graph-callcenter`; spear consumers only see clean facades.

## Next Phases (out of scope for this PR)

- Full Ractor membrane supervisor implementation
- Integration of `ogit_owl_dolce` helpers into `ExternalMembrane`
- Ontology-guided CAM training
- Spear-side consumer facades

## How to Review

1. Read `OGIT_OWL_DOLCE_RACTOR.md` for the overall vision.
2. Review `ogit_owl_dolce.rs` for the semantic mapping logic.
3. Review `dn_path_with_semantics.rs` for the proposed `DnPath` extension pattern.

This PR is intentionally small and focused so it can be reviewed and merged cleanly.
