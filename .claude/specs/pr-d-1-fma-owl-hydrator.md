> **W9-rev2 correction (main thread):** original W9 push went to AdaWorldAPI/ada-consciousness instead of AdaWorldAPI/lance-graph. Content recovered verbatim below and pushed to the correct repo by main thread via pygithub. Original ndarray-style misrouted commit lives at `AdaWorldAPI/ada-consciousness/claude/tier-1-implementation-specs` as harmless residue.

---

# PR-D-1: FMA OWL Hydrator (First Concrete Pattern D)

**Sprint:** 3
**Worker:** W9
**Pattern:** D — Meta-Structure Hydration
**Status:** DESIGN PHASE
**Branch:** `claude/tier-1-implementation-specs`

---

## Goal

Hydrate the **Foundational Model of Anatomy (FMA)** — ~75,000 anatomical
classes and ~168 properties — as the OGIT graph slot `G = FMA_V1`. This PR is
the *first concrete proof* that Pattern D works: an entire biomedical
ontology lands in the system as **data + a tiny Rust glue crate**, never as a
per-domain hand-written crate.

If this lands cleanly, the same `OwlHydrator` is reusable for SNOMED-CT,
GO, ChEBI, DOLCE, BFO, UBERON, and every future OWL ontology — Sprint-4 will
then need only `hydrate_xxx()` glue functions, not new crates.

---

## Why this matters (Pattern D in one paragraph)

Historically, every new ontology in Ada/OGIT spawned a bespoke crate
(`lance-anatomy`, `lance-chemistry`, ...). That doesn't scale: there are
~1,400 OBO-Foundry ontologies. Pattern D inverts the model — ontologies are
treated as *meta-structure data* loaded into a generic registry. The only
per-ontology code is a ~30-line glue function that picks the OWL parser,
declares the `G` slot, and whitelists the edge types. PR-D-1 is the first
end-to-end demonstration.

---

## Files to touch

| Path | Status | Purpose |
|------|--------|---------|
| `data/ontologies/fma.ttl` | NEW | FMA OWL/Turtle source (BioPortal, Apache-2.0) |
| `crates/lance-graph-ontology/src/hydrators/mod.rs` | NEW | Hydrator module map |
| `crates/lance-graph-ontology/src/hydrators/owl.rs` | NEW | Generic OWL → ContextBundle hydrator |
| `crates/lance-graph-ontology/src/hydrators/fma.rs` | NEW | FMA-specific glue (calls owl + custom whitelist) |
| `crates/lance-graph-ontology/Cargo.toml` | EDIT | Add `oxttl` (or `rio_xml`) dep |
| `crates/lance-graph-ontology/tests/owl_hydrator_minimal_ttl.rs` | NEW | 10-class smoke test |
| `crates/lance-graph-ontology/tests/owl_hydrator_subclass_traversal.rs` | NEW | rdfs:subClassOf cascade |
| `crates/lance-graph-ontology/tests/fma_hydrator_smoke.rs` | NEW | Full FMA load |
| `crates/lance-graph-ontology/tests/fma_part_of_heart.rs` | NEW | BFO:part_of cascade on heart |

---

## API sketch (~250 LOC for owl.rs, ~40 LOC for fma.rs)

```rust
// crates/lance-graph-ontology/src/hydrators/mod.rs
pub mod owl;
pub mod fma;

pub use owl::{MetaStructureHydrator, OwlHydrator, HydrateErr};
pub use fma::hydrate_fma;
```

```rust
// crates/lance-graph-ontology/src/hydrators/owl.rs
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use oxttl::TurtleParser;          // or rio_xml::RdfXmlParser
use crate::registry::{OntologyRegistry, OntologySlot, ContextBundle};
use crate::spo::SpoQuad;

pub trait MetaStructureHydrator {
    fn hydrate(&self, source: &Path, registry: &mut OntologyRegistry)
        -> Result<u32, HydrateErr>;
}

pub struct OwlHydrator {
    pub g: u32,
    pub version: u32,
    pub domain_name: String,
    pub inherits_from: Option<u32>,
    pub starting_entity_id: u32,    // default 100; lower IDs reserved for OWL builtins
}

impl MetaStructureHydrator for OwlHydrator {
    fn hydrate(&self, ttl_path: &Path, registry: &mut OntologyRegistry)
        -> Result<u32, HydrateErr>
    {
        // 1. Parse TTL via oxttl
        let mut parser = TurtleParser::new()
            .for_reader(File::open(ttl_path)?);

        let mut entity_map: HashMap<String, u32> = HashMap::new();
        let mut next_id: u32 = self.starting_entity_id;
        let mut spo_quads: Vec<SpoQuad> = Vec::with_capacity(1_000_000);

        let intern = |iri: &str, map: &mut HashMap<String, u32>, n: &mut u32| -> u32 {
            *map.entry(iri.to_string()).or_insert_with(|| {
                let id = *n;
                *n += 1;
                id
            })
        };

        for triple in parser {
            let triple = triple?;
            let s_id = intern(triple.subject.to_string().as_str(), &mut entity_map, &mut next_id);
            let p_id = intern(triple.predicate.to_string().as_str(), &mut entity_map, &mut next_id);
            let o_id = match triple.object {
                oxttl::Term::NamedNode(n) =>
                    intern(n.as_str(), &mut entity_map, &mut next_id),
                oxttl::Term::Literal(l) =>
                    hash_literal_to_u32(l.value()),    // 0x80000000 high-bit flag
                oxttl::Term::BlankNode(b) =>
                    intern(&format!("_:{}", b.as_str()), &mut entity_map, &mut next_id),
            };
            spo_quads.push(SpoQuad {
                s_id: s_id as u64,
                p_id: p_id as u64,
                o_id: o_id as u64,
                g: self.g,
                version: self.version,
            });
        }

        // 2. OntologySlot
        let ontology = OntologySlot {
            entity_count: entity_map.len() as u32,
            iri_to_id: entity_map,
        };

        // 3. ContextBundle (INERT — no consumer crate)
        let bundle = ContextBundle {
            g: self.g,
            version: self.version,
            domain_name: self.domain_name.clone().into(),
            inherits_from: self.inherits_from,
            ontology: Some(Arc::new(ontology)),
            consumer_pointer: None,
            ..ContextBundle::default()
        };
        registry.register(bundle);

        // 4. Bulk write SPO via PR-A-1 writer
        registry.spo_writer().bulk_insert(spo_quads)?;

        Ok(self.g)
    }
}
```

```rust
// crates/lance-graph-ontology/src/hydrators/fma.rs
use std::path::Path;
use crate::ogit::OGIT;
use super::owl::{OwlHydrator, MetaStructureHydrator, HydrateErr};
use crate::registry::OntologyRegistry;

pub fn hydrate_fma(registry: &mut OntologyRegistry) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::FMA_V1.0,
        version: OGIT::FMA_V1.1,
        domain_name: "fma".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate(Path::new("data/ontologies/fma.ttl"), registry)?;

    // FMA-specific edge whitelist — what cascade traversals will see.
    registry.register_edge_types(OGIT::FMA_V1.0, &[
        "rdfs:subClassOf",
        "BFO:part_of",
        "BFO:has_part",
        "FMA:regional_part_of",
        "FMA:constitutional_part_of",
        "FMA:systemic_part_of",
        "FMA:supplies_blood_to",
        "FMA:innervates",
        "FMA:adjacent_to",
        "FMA:continuous_with",
    ])?;

    Ok(OGIT::FMA_V1.0)
}
```

---

## Test plan

| Test | Asserts |
|------|---------|
| `owl_hydrator_minimal_ttl` | 10-class hand-written TTL hydrates; bundle exists; entity_count == 10; SPO row count matches triple count |
| `owl_hydrator_subclass_traversal` | Query `rdfs:subClassOf*` from synthetic root returns all 9 descendants in BFS order |
| `fma_hydrator_smoke` | Full FMA loads in <60 s; entity_count > 70_000; resolves IRI `http://purl.obolibrary.org/obo/FMA_7088` (heart) to a stable u32 |
| `fma_part_of_heart` | Cascade `BFO:part_of⁻¹` from FMA:heart yields 10+ entities including atrium, ventricle, valve IRIs |

All four tests must run via `cargo test -p lance-graph-ontology`.
The full-FMA tests are gated behind `--features heavy-data` so CI default
stays fast.

---

## Dependencies

**Internal (Sprint-3 in-flight):**
- **PR-A-1 / W2** — `SpoQuad` u32 G slot must exist; writer must accept bulk inserts.
- **PR-B-1 / W3** — `ContextBundle`, `OntologySlot`, `OntologyRegistry` types must exist.
- **OGIT enum (W1 master)** — `OGIT::FMA_V1` and `OGIT::DOLCE_V1` slot constants.

**External crates:**
- `oxttl = "0.1"` — modern Turtle parser, Apache-2.0 (recommended)
- *or* `rio_xml = "0.8"` — RDF/XML parser if FMA ships only as `.owl`/RDF-XML

**Data:**
- FMA OWL/Turtle from BioPortal (~30 MB compressed, ~280 MB inflated TTL)
- Apache-2.0 license verified at <http://purl.obolibrary.org/obo/fma.owl>

---

## Acceptance criteria

- [ ] `OwlHydrator` and `hydrate_fma()` compile and pass `cargo clippy`
- [ ] FMA TTL present at `data/ontologies/fma.ttl` (or fetched by build script if too large for git)
- [ ] `hydrate_fma()` registers `G = FMA_V1` with `consumer_pointer = None` (INERT bundle confirmed)
- [ ] SPO bulk-insert routes through PR-A-1's `SpoWriter`
- [ ] All 4 tests green
- [ ] Full FMA hydration completes in <60 s on a workstation (`cargo test --release --features heavy-data`)
- [ ] `cargo doc` renders `MetaStructureHydrator` trait with usage example

---

## Effort

**Medium.** ~600 LOC (250 owl.rs + 40 fma.rs + ~250 across 4 tests + ~60 mod.rs / boilerplate). **3–4 days** including BioPortal download, license confirmation, IRI mapping cache design, and test data curation.

---

## Open questions for the implementing engineer

1. **TTL storage.** Commit `fma.ttl` to git (~280 MB inflated, prohibitive)? Use git-lfs? Or download in `build.rs` from a pinned CDN/mirror?
   *Recommendation:* CDN download with sha256 pin. Avoids git bloat; license obligation is just attribution in `LICENSES/FMA.txt`.

2. **IRI → u32 mapping persistence.** Recompute on every hydration (~60 s) or persist `iri_to_id` to a Lance dataset?
   *Recommendation:* Persist. Hydration becomes O(1) cache load on warm starts; the map is ~3–5 MB serialized.

3. **TTL parser choice.** `rio_xml` (most stars), `oxttl` (modern, sans-IO, streaming), or `rdftk` (full-stack)?
   *Recommendation:* `oxttl` — pure-Rust, streaming, low alloc, Apache-2.0, actively maintained by the same author as oxigraph.

4. **Property whitelist policy.** Hand-curate per ontology (current spec) or accept all properties and filter at query time?
   *Recommendation:* Hand-curate for v1 (focus + traversal cost); v2 generalize once we have telemetry on which edges actually matter.

5. **Literal handling.** Hash literals to `u32` with high-bit flag (sketched), or split into a separate `LiteralStore`?
   *Recommendation:* Defer literals to PR-D-2; for FMA v1, drop datatype/string literals (FMA's semantics are entity-graph, not literal-heavy).

---

## Anatomy demo unlock

This PR is also **PR-ANATOMY-1** in `.claude/plans/anatomy-realtime-v1.md`.
Once it lands:

- **PR-ANATOMY-2** (DICOM hydrator) can ship — it depends on FMA IRIs existing as resolvable u32 entity IDs.
- **PR-ANATOMY-3** (FMA SPO-G edges into the live graph) becomes a one-liner over the registered bundle.
- **PR-ANATOMY-4** (Q2 cockpit-server 3D voxel render with FMA labels) unblocks for the Q2 milestone demo.

---

## Cross-references

- `.claude/plans/anatomy-realtime-v1.md` — PR-ANATOMY-1 entry
- `.claude/specs/pr-a-1-spo-g-u32-slot.md` (W2; **required**)
- `.claude/specs/pr-b-1-context-bundle.md` (W3; **required**)
- `.claude/specs/sprint-3-execution-plan.md` (W1 master)
- BioPortal FMA: <http://purl.obolibrary.org/obo/fma.owl>
- OBO Foundry FMA page: <http://www.obofoundry.org/ontology/fma.html>
- Pattern D rationale: see `.claude/BLACKBOARD.md` "Meta-Structure Hydration" section

---

*W9 / Sprint-3 / DESIGN PHASE / pygithub-first*

---

## CORRECTION (post-#360 substrate-recognition sweep)

**Defect:** This spec proposed a 600-LOC FMA-specific OWL hydrator + glue. **But the post-#355 substrate already ships ~50% of the hydrator pipeline:**

| Already shipped on main | File |
|---|---|
| `parse_ttl_directory_with_provenance` — generic TTL hydrator with per-attribute `dcterms:source` threading | `crates/lance-graph-ontology/src/ttl_parse.rs` |
| `OntologyRegistry::hydrate_once_sync` + `attach_provenance` — production hydration path | `crates/lance-graph-ontology/src/registry.rs` |
| `MappingRow.attribute_sources: Vec<AttributeProvenance>` — D-CASCADE-V1-7 column extension | (PR #355 cascade cols on MappingRow) |

**Pattern D is ~50% shipped.** The Turtle hydrator + provenance threading + Registry append + MappingRow attribute_sources column all ship. **Only the OWL/RDF-XML reader is genuinely new** (FMA ships in OWL/XML format, not Turtle, so a separate parser is needed for the input format) — the destination side is done.

### Re-scoped PR-D-1 (post-substrate-sweep)

**Adapter, not full hydrator:**
- Build OWL/RDF-XML reader (or use `oxttl` / `rio_xml`) → adapter that emits `MappingProposal` + `ProvenanceBundle`
- Adapter calls existing `OntologyRegistry::hydrate_once_sync` path
- Skip writing a parallel hydration pipeline; reuse what ships

**Net new work:** OWL/RDF-XML reader → adapter.

**Revised effort:** ~250 LOC, ~1-2 days (down from ~600 LOC, ~3-4 days).

**Pattern D status update:** PARTIALLY SHIPPED (was: design phase, "first concrete proof Pattern D works"). The hydrator SHAPE ships and is in production via existing TTL ingestion; FMA OWL/XML adapter is one input-format extension, not the proof. Future tier-0 doc should mark Pattern D as "primitives shipped" similar to Pattern M.

**Provenance:** post-#360 substrate-recognition sweep, flagged by reviewer.
