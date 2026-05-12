# ROUTING NAMESPACE EXAMPLE — OGIT TTL + Mapped Packed Schema

**Purpose**: Concrete, minimal example of the first vertical (Routing) to demonstrate the full glue layer transformation, content-addressable pointer, and packed schema output. This is the pattern subsequent verticals follow.

## 1. OGIT TTL for Routing Namespace (to be placed in AdaWorldAPI/OGIT/NTO/Routing/)

```turtle
@prefix ogit: <http://www.ogit.org/> .
@prefix ogit:Routing: <http://www.ogit.org/Routing/> .
@prefix ogit:Automation: <http://www.ogit.org/Automation/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dolce: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> .

# =====================
# Entity Types
# =====================

ogit:Routing:RoutingRule
    a ogit:EntityType ;
    ogit:parent ogit:Automation:Issue ;
    rdfs:subClassOf dolce:Endurant ;
    ogit:mandatory-attributes (
        ogit:Routing:pattern
        ogit:Routing:targetQueue
    ) ;
    ogit:optional-attributes (
        ogit:Routing:template
        ogit:Routing:confidenceFloor
        ogit:Routing:priority
    ) ;
    rdfs:label "Routing Rule" ;
    rdfs:comment "Defines a pattern-based routing decision for incoming tickets." .

ogit:Routing:RoutingDecision
    a ogit:EntityType ;
    ogit:parent ogit:Automation:Event ;
    rdfs:subClassOf dolce:Perdurant ;
    ogit:mandatory-attributes (
        ogit:Routing:ticketRef
        ogit:Routing:matchedRule
        ogit:Routing:resolvedQueue
        ogit:Routing:mulOutcome
        ogit:Routing:timestamp
    ) ;
    rdfs:label "Routing Decision" ;
    rdfs:comment "Immutable record of a routing evaluation and its MUL-gated outcome." .

# =====================
# Verbs / Relationships
# =====================

ogit:Routing:routes-to
    a ogit:Verb ;
    ogit:from ogit:Routing:RoutingRule ;
    ogit:to ogit:Automation:Queue ;
    ogit:cardinality "1" ;
    rdfs:label "routes to" .

ogit:Routing:escalates-to
    a ogit:Verb ;
    ogit:from ogit:Routing:RoutingDecision ;
    ogit:to ogit:Automation:Person ;
    ogit:cardinality "0..1" ;
    rdfs:label "escalates to (manager on duty)" .

ogit:Routing:applies-template
    a ogit:Verb ;
    ogit:from ogit:Routing:RoutingRule ;
    ogit:to ogit:Automation:Template ;
    ogit:cardinality "0..1" ;
    rdfs:label "applies template" .

# =====================
# Attributes
# =====================

ogit:Routing:pattern
    a ogit:Attribute ;
    rdfs:range xsd:string ;
    rdfs:label "Regex or embedding pattern" .

ogit:Routing:targetQueue
    a ogit:Attribute ;
    rdfs:range ogit:Automation:Queue ;
    rdfs:label "Target queue reference" .

ogit:Routing:confidenceFloor
    a ogit:Attribute ;
    rdfs:range xsd:double ;
    rdfs:label "Minimum MUL confidence required to act autonomously" .

ogit:Routing:mulOutcome
    a ogit:Attribute ;
    rdfs:range xsd:string ;   # "confident" | "hold-for-human" | "vetoed"
    rdfs:label "Outcome of MUL gate" .
```

## 2. Content-Addressable Pointer Generation

Canonicalized TTL hash (BLAKE3 of sorted triples) → e.g. `0xA3F91B2C...`

**Short pointer** (3–5 bytes practical): `0xA3F91B` (example)

This pointer is stored in Lance edges/columns. Full hash + packed schema live in the content-addressable registry (backed by Lance + Bgz tensor for the schema blob).

## 3. Glue Layer Output (Conceptual In-Memory OWL Model)

After `ogit_to_owl_mapper` + `dolce_classifier`:

- `ogit:Routing:RoutingRule` → `owl:Class` + `dolce:Endurant`
- `ogit:Routing:RoutingDecision` → `owl:Class` + `dolce:Perdurant`
- `ogit:Routing:routes-to` → `owl:ObjectProperty` + `owl:FunctionalProperty` (from cardinality "1")
- `ogit:Routing:escalates-to` → `owl:ObjectProperty`
- `ogit:Routing:confidenceFloor` → contributes to MUL prior (used in `is_unskilled_overconfident`)
- Equivalence table: empty for this minimal namespace (can be extended later)

## 4. Packed Schema Binary (Conceptual Dump)

Header (64 bytes):
- Magic: OWLS
- Version: 1
- Content Hash: 0xA3F91B2C... (full)
- Short Pointer: 0xA3F91B
- Num Classes: 2
- Num Properties: 3 (routes-to, escalates-to, applies-template) + attribute properties

Class Hierarchy Bitmap (tiny for this example):
- RoutingRule is subclass of Automation:Issue + Endurant

Property Characteristics (16-byte records):
- routes-to: Functional=1, Domain=RoutingRule, Range=Queue
- escalates-to: Functional=0, ...
- ...

DOLCE Bitmap:
- RoutingRule: 0 (Endurant)
- RoutingDecision: 1 (Perdurant)

**Total size for this namespace**: << 1 KB (full OGIT is ~50 KB).

The packed schema is stored via Bgz tensor in Lance under the short pointer. The *existing* ndarray integration delivers **20–200 ns random access** to this packed schema or any SoA column it annotates.

## 5. Entropy Work / Uncertainty Handling in Routing

- **Multi-match ambiguity**: If multiple RoutingRules match a ticket subject, the planner returns a candidate set. Because `routes-to` is marked `FunctionalProperty` in the packed schema, MUL’s `is_unskilled_overconfident()` receives an explicit signal and can hard-veto to "hold-for-human" even if individual rules have high pattern confidence.
- **Confidence floor attribute**: Stored in the schema; MUL compares its internal assessment against the rule’s declared floor.
- **MUL outcome stored immutably** in `RoutingDecision` (Perdurant) → full audit trail of when the system chose to be uncertain.
- This is the core "entropy work": the system does not pretend perfect knowledge; it explicitly downgrades and escalates when data entropy (ambiguity) exceeds what the schema + MUL can confidently resolve.

This example is small enough to implement first and proves the entire glue → packed schema → MUL path works before scaling to the full investigation agent.