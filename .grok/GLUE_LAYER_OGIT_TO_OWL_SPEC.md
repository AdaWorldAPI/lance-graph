# GLUE LAYER SPEC — OGIT TTL → OWL + DOLCE + Packed Schema

**Version**: 0.1 (Initial fanout)  
**Owner**: Architecture team  
**Depends on**: `lance-graph-ontology` (ttl_parse.rs, foundry_map.rs), AdaWorldAPI/OGIT, INVARIANT_LAYER.txt

## Purpose
Define the exact, deterministic transformations that turn any OGIT TTL namespace into:
1. OWL/RDFS constructs (property characteristics, domain/range, equivalence)
2. DOLCE upper markers (Endurant vs Perdurant)
3. Content-addressable packed schema binary suitable for `lance-graph-owl-simd`

This glue layer is the **single source of truth** for schema assimilation. It runs at hydration time (when new OGIT TTL or customer extension arrives) and produces the L1-resident packed form used by the validator, MUL gate, planner, and investigation agent.

## 1. Overall Pipeline

```
OGIT TTL files (AdaWorldAPI/OGIT/NTO/<Namespace>/*.ttl)
        │
        ▼  (existing)
lance-graph-ontology::ttl_parse + foundry_map
        │
        ▼  (NEW — this spec)
ogit_to_owl_glue::Mapper
        │
        ├── Produces in-memory OWL model (classes, properties, annotations)
        ├── Applies DOLCE classification rules
        ├── Computes content hash
        └── Serializes to PackedSchema binary (Bgz-compressed optional)
        │
        ▼
Content-Addressable Registry (short 3-5 byte pointer)
        │
        ▼
lance-graph-owl-simd validator + MUL priors + Investigation Agent
```

## 2. Entity Type Transformation Rules

### Input (OGIT example)
```turtle
@prefix ogit: <http://www.ogit.org/> .
@prefix ogit:Automation: <http://www.ogit.org/Automation/> .

ogit:Automation:Ticket
    a ogit:EntityType ;
    ogit:parent ogit:Automation:Issue ;
    ogit:mandatory-attributes ( ogit:subject ogit:created-on ogit:status ) ;
    ogit:optional-attributes ( ogit:description ogit:priority ) ;
    ogit:scope "customer" ;
    rdfs:label "Ticket" ;
    rdfs:comment "Represents a support ticket in the IT automation domain." .
```

### Output (OWL + DOLCE)
```turtle
ogit:Automation:Ticket
    a owl:Class ;
    rdfs:subClassOf ogit:Automation:Issue ;
    rdfs:subClassOf dolce:Endurant ;          # ← DOLCE marker (see rules below)
    owl:hasKey ( ogit:subject ogit:created-on ) ;  # derived from mandatory
    rdfs:label "Ticket" ;
    rdfs:comment "..." ;
    ogit:originalScope "customer" .           # preserved for audit
```

**DOLCE Classification Rules** (implemented in glue):
- Default: `dolce:Endurant` (persistent, stateful objects: Ticket, Queue, Person, Mailbox, Document, Customer)
- `dolce:Perdurant` if:
  - Entity name contains `Decision`, `Intent`, `Event`, `Log`, `Record`, `Resolved`, `Fired`, `Created` (heuristic + explicit list)
  - Or explicit `ogit:perdurant true` annotation present
  - Or the entity only participates in immutable "happened" relationships

**Implementation note**: The single-bit DOLCE flag goes into the packed schema class entry. Fast path in validator: single bitwise AND.

## 3. Verb / Relationship Transformation Rules

### Input
```turtle
ogit:routes-to
    a ogit:Verb ;
    ogit:from ogit:Automation:Ticket ;
    ogit:to ogit:Automation:Queue ;
    ogit:cardinality "1" ;
    ogit:inverse ogit:has-ticket ;
    rdfs:label "routes to" .
```

### Output (OWL property characteristics)
```turtle
ogit:routes-to
    a owl:ObjectProperty ;
    rdfs:domain ogit:Automation:Ticket ;
    rdfs:range ogit:Automation:Queue ;
    owl:FunctionalProperty ;           # from cardinality "1" + mandatory semantics
    owl:inverseOf ogit:has-ticket ;
    rdfs:label "routes to" .
```

**Property Characteristics Bitfield** (1 byte per property in packed schema):
```
Bit 0: Functional
Bit 1: InverseFunctional
Bit 2: Transitive
Bit 3: Symmetric
Bit 4: Asymmetric
Bit 5: Reflexive
Bit 6: Irreflexive
Bit 7: Reserved / DOLCE-specific
```

**MUL Integration**:
- When planner returns multiple matches on a `FunctionalProperty` → `MUL::is_unskilled_overconfident()` receives explicit signal + property name → hard veto with human-readable explanation.

**TransitiveProperty handling**:
- Glue layer can optionally pre-compute transitive closure hints (stored in packed schema) for planner optimizations. Not required for correctness.

## 4. Attribute Transformation

- `ogit:mandatory-attributes` → contributes to `owl:hasKey` or `owl:minCardinality 1` on the owning class
- `ogit:indexed-attributes` → stored as annotation; used by lance-graph index planner / DataFusion pushdown hints
- Data type inference:
  - `ogit:created-on` etc. → `xsd:dateTime`
  - String fields → `xsd:string`
  - References → `owl:ObjectProperty`

## 5. Equivalence, Extension & Multi-Domain Assimilation

**Supported constructs** (to assimilate any domain ontology):
```turtle
# Customer-internal vocabulary equated to OGIT
customer:MyTicket owl:equivalentClass ogit:Automation:Ticket .

# External standard
itil:Incident owl:equivalentClass ogit:Automation:Ticket .
```

The glue layer:
- Records equivalence pairs in the packed schema (small table of short-ID pairs)
- Query planner and investigation agent resolve through equivalence transparently
- This is the mechanism that lets us cover "all domains" like Foundry without depending on it

## 6. Content-Addressable Pointer Generation

1. Canonicalize the OGIT TTL fragment for a namespace or extension (sorted triples, normalized prefixes).
2. Compute stable hash (BLAKE3 or SHA-256 truncated).
3. Short pointer = first 3–5 bytes of hash (collision probability extremely low for internal use; full hash stored alongside for verification).
4. Registry entry: `pointer → (packed_schema_bytes or Lance row address, source_ogit_hash, version)`

**Benefits**:
- Edges/columns in Lance store only the 3-5 byte pointer (massive space + cache win).
- O(1) schema lookup via perfect hash or small in-memory map.
- Versioning: new TTL → new pointer → new packed schema. Old data stays valid forever.

## 7. Packed Schema Emission

The glue layer calls into `PackedSchemaCompiler` (to be implemented in `lance-graph-owl-simd` or shared crate) which produces the exact binary format defined in `PACKED_SCHEMA_FORMAT.md`.

The compiler also:
- Uses ndarray (AMX/BLAS paths) for any pre-computation of bitmaps or transitive hints at hydration time.
- Optionally compresses the packed schema with Bgz tensor crate before storing in Lance.

## 8. Error Handling & Validation

- Glue layer must be **total** on valid OGIT (never panic on well-formed TTL).
- Unknown ogit: constructs → warning + best-effort mapping + audit log entry.
- Conflicting cardinality / property characteristics → hard error at hydration time (prevents bad schemas from ever reaching validator).
- Equivalence cycles → detected and rejected (or canonicalized).

## 9. Files to Implement / Extend

| File / Module                              | Responsibility                                      | Priority |
|--------------------------------------------|-----------------------------------------------------|----------|
| `lance-graph-ontology/src/glue/ogit_to_owl_mapper.rs` | Core transformation rules                           | P0      |
| `lance-graph-ontology/src/glue/dolce_classifier.rs`   | Endurant/Perdurant heuristics + explicit rules      | P0      |
| `lance-graph-ontology/src/content_addressable_registry.rs` | Pointer generation, lookup, versioning         | P0      |
| `lance-graph-owl-simd/src/packed_schema_compiler.rs`  | Binary emission + ndarray integration               | P0      |
| `spear/src/ontology/` (per vertical)       | Domain-specific extensions (e.g. Routing namespace) | P1      |

## 10. Testing Strategy

1. **Unit**: Property-based tests on synthetic OGIT fragments (mandatory → Functional, etc.).
2. **Integration**: Hydrate real `AdaWorldAPI/OGIT/NTO/Automation/` and `Routing/` (when created) and verify packed schema size + validator accepts valid data / rejects violations.
3. **End-to-end**: Routing vertical shadow mode — feed real ticket subjects, confirm MUL uses new OWL-derived priors.
4. **Equivalence**: Load a small customer extension TTL that uses `owl:equivalentClass` and verify queries resolve correctly.

---

**Status**: Ready for implementation. This spec is the contract between the ontology team and the SIMD / Spear vertical teams.

Next: Create `PACKED_SCHEMA_FORMAT.md` with exact byte layout.