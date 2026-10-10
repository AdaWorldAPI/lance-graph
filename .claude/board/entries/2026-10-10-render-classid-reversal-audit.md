# 2026-10-10 — RENDER-CLASSID-REVERSAL-AUDIT: reversing render_classid would mis-route silently; do not do it

**Status:** MEASURED (read-only). Three parallel agents read the code; key claims were spot-checked against source. Plan: `.claude/plans/cross-glove-business-parity-v1.md` §C.9.4.

**Hypothesis audited:** move the codebook 16-bit concept from the high half to the low half (`[domain:appid][concept]`). §C.9.3's correction (appid ≡ concept byte) shows the ruling does not ask for this. This record is why it must not be done.

## Headline findings
1. **No single switch.** The `(concept << 16) | prefix` math is hand-copied ~20 times. Verified sites:
   - `ogar-obo/src/lib.rs:220-222`
   - `ogar-loco/src/lib.rs:220-222`
   - `a2ui-server/src/action_stream.rs:93-95` (+ its wasm twin)
   - `MedCare-rs medcare-server/src/views/addressed.rs:167`
   - `spear/src/mail/iam.rs:236`
   - `tesseract-paperless/src/kv.rs:273`

   Further copies: `ogar-ro`, `ogar-dismech`, `ogar-obo registry/crosswalk`, Python/C#/askama SDK templates, MedCare `class_registry.rs:319` / `first-thought lib.rs:161` / `python/app/ogar_sdk.py`, `ruff_r2il facet.rs:232`, q2 `fma/converge.rs`, lance-graph `facet.rs:326-396` (`semantic_tiles`, hard-coded `>> 16`).
2. **A flip mis-routes; it does not fail.** Each of these reads the high half and would get a valid but different class:
   - `ClassGrant::permits` (`rbac.rs:538-540`, verified)
   - `RbacBinding::plugged` (`rbac_plug.rs:184`)
   - `graph_of` / SPOG tenant routing
   - `chain_admission::palette_of`
   - `ogar-loco resolve_classid`
   - `Namespace::from_concept_id`
   - a2ui `concept_of_key` → ClassView
   - MedCare `rails.rs` / `domain_block.rs` `is_obo_lane` / `facet_regime`
   - osm `identity.rs:117` / `cluster.rs:150` (returns `None`, i.e. "empty")

   Example: `0x0905` (domain 09 : appid 05) is the valid class `treatment` (`ogar-vocab/src/lib.rs:1328`).
3. **Domain routing survives by position.** `classid_concept_domain` and `FacetSchema::of_classid` read the top byte.
4. **Persisted data:**
   - OBO `.soa` bakes: re-key plus re-sort (`SpineLens` binary search);
   - MedCare `obo_slim_edges.tsv` (124 lines, verified), `obo_slim_labels.tsv`, `obo_core_classid_ranges.tsv`, joinmap JSON/YAML/TTL, `bakes.tsv` `@0x…` gates parsed by `bake_hydrate.rs`;
   - `ogar-from-ruff` emitted constants in consumer repos;
   - Lance `NodeGuid` key bytes;
   - the tesseract-paperless `document_guid` archive;
   - osm slabs, q2 osint bakes plus the `osint_v3_rebake_hilo.py` byte check;
   - ruff r2il VarnodeFacet bytes;
   - odoo-rs generated code and the `/compile` JSON;
   - the postgres DDL `classid INTEGER` column.
5. **Legacy forms collide.** Reversed ids share the shape of the pre-2026-07-02 `CanonLow` forms, so `classid_canon_compat`, `classify_form` and the `CLASSID_*_LEGACY` aliases become ambiguous.
6. **The ClassView field basis is safe.** `ClassView` keys a bare `u16` (`class_view.rs:54`); masks are `rbac ∩ present ∩ view` (op-server `viewfilter.rs:83-95`) / `surface ∩ role` (a2ui `project.rs:73-82`). App-prefix skin routing (`resolve_codebook`) exists only in `OGAR/docs/APP-CLASS-CODEBOOK-LAYOUT.md` §4.

## Follows automatically
These call OGAR or the contract and only need their pinned tests re-pinned: odoo-rs `od-ontology`, openproject `op-canon` wrappers, HubSPO `hubspo-port`, stockfish-web classview, rs-graph-llm `AUTH_STORE`, osm `row.rs` `CLASSID_GEO_V3`, MedCare `Namespace::render_classid` callers.

OPEN: the low-16 question in §C.9.3.
