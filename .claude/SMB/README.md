# .claude/SMB/ — SMB tenant schema reference

> **Read this when:** your task in lance-graph touches
> `lance-graph-ontology`, `lance-graph-callcenter::ontology_dto`,
> `lance-graph-callcenter::transcode`, `lance-graph-rbac::policy`,
> the `OntologyRegistry` hydration path, or the registry → DTO
> projection contract — and you need to know what a real tenant
> consumer is shaped like.

## Files

| File | Read when |
|---|---|
| `SCHEMA.md` | You're touching a contract that affects SMB consumers. Lists the 14-entity BSON shape, the 3-entity Foundry shape, and the projection rules between them. |

## Why a per-tenant folder

The tenant consumer (`AdaWorldAPI/smb-office-rs`) is a separate repo;
this lance-graph workspace doesn't load its source. Without these
reference docs, "is this lance-graph change safe for SMB" requires
cross-repo zipball + grep on every session. With them, the shape is
two pages and immediately readable.

This is reference material, not policy. Authoritative source is
smb-office-rs.

## Adding tenants

If a second tenant lands (MedCare, Stalwart, SAP, future Foundry
consumer), add a sibling folder under `.claude/<tenant>/` with the
same structure. Keep these per-tenant; do not mix shapes.
