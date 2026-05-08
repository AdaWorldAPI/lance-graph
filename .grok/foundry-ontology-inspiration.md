# Foundry Ontology Inspiration – Key Patterns & Examples (2026-05-08)

**Source**: Palantir Foundry official docs + community examples  
**Focus**: High-signal patterns relevant to BindSpace + cognitive shaders + internal/external ontology direction.

## Core Modeling Principles (Lowest Entropy Wins)
- **Model reality, not systems**: Object types represent real-world entities (Employee, Customer, Order, Product, Flight, Vehicle, Assembly Line, Issue Report), not source tables or departments.
- **Keep object types focused**: One distinct entity per type. Avoid wide, sparse objects.
- **Use Interfaces for abstraction**: When entities share characteristics, define an Interface instead of duplicating properties across many object types.
- **Title Key + Primary Key**: Every object type has a clear display name (title key) and unique identifier.

## Common Object Type Examples
- **Simple domain**: Customer, Order, Product (with links: Order → Customer, Order ↔ Product via Line Items).
- **Event-oriented**: Rail Journeys or Issue Report (use timestamp properties to treat as events).
- **Manufacturing/Operational**: Assembly Line + Issue Report (link created via Action; property like "Active Issues Count" updated).
- **Project/Engagement**: Project, Engagement, Resource, Deliverable, Client, Billing Event, Risk, Milestone.

## Links & Relationships
- Explicit Link Types define relationships (e.g., "belongs to", "staffed on", "registered with").
- Links carry semantics and enable traversal/querying.

## Actions (Kinetic Layer)
- Actions mutate the ontology in controlled ways:
  - Create objects/links
  - Update properties
  - Complex functions (e.g., create Issue Report + link to Assembly Line + increment count)
- Three buckets: Create/Delete, Update, Multi-step functions.

## Three-Layer View (Semantic + Kinetic + Dynamic)
- **Semantic**: Shared language (objects, properties, links) — reconciles fragmented concepts into unified entities.
- **Kinetic**: Operational layer (actions, updates, lineage, backing datasets).
- **Dynamic**: Governance, security, access control, workflows.

## Relevance to BindSpace + Cognitive Shaders
- BindSpace as zero-copy internal ontology aligns with Foundry’s object model (typed, focused, reality-modeled).
- Cognitive shaders could map to "Actions" or dynamic behaviors on objects (resolution, truth revision, free energy updates).
- Interfaces could inspire reducing duplication in SoA/DTO clusters.
- External ontology integration (Gotham/Foundry + HubSpot) benefits from explicit Link Types and clean projection seams rather than ad-hoc bridges.

## Anti-Patterns to Avoid
- Modeling source systems instead of reality.
- Creating wide/sparse object types instead of using Interfaces.
- Making everything editable (only when necessary).
- Spreading objects across multiple ontologies without clear ownership.

## High-Signal Takeaway for Current Direction
The biggest opportunity is treating the **BindSpace ↔ External Ontology Projection** seam with Foundry-style discipline:
- Name the seam.
- Use focused object modeling + Interfaces inside BindSpace.
- Define explicit Link Types / projection mechanisms for external integration.
- Let Actions (or shader equivalents) handle mutation and cross-boundary effects.

This keeps internal cognitive ontology clean and zero-copy while enabling powerful external interoperability.

---

*Low-entropy summary for quick loading. Posted under .grok/ to support token-conscious sessions.*