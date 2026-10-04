# 2026-10-04 — SPOG × slab declaration resolves through the existing hot-plug (D-LXC-29 follow-up)

**Status:** VERIFIED-IN-CODE + TEST-PINNED (`crates/lance-graph-contract/src/hotplug.rs`, 7 tests for the 8 requested invariants; 6 guards disable-verified red-then-green; `lance-graph-ogar` 84/84 unchanged).

## What landed

The resolution seam asked for in `2026-10-04-deepnsm-v2-tenant-rails-6-vs-8.md` ("a `ValueSchema` entry that the hot-plug `ReadMode` resolves to"). No layout, kernel, fold or slab change.

- **SPOG** gives the concept: `spog_tenants::graph_of(key)`.
- **OGAR registry** gives the class reading: `Activation::read_mode_for(concept)`. If the concept is unplugged the call returns `NoReadingFor`; a slab declaration cannot stand in for it.
- **Slab metadata** is `SlabDeclaration {reading: SlabReading, value_schema, layout_version}`. It holds physical facts only:
  - no concept, because SPOG supplies semantic identity;
  - no tail or edge codec, because those read the key and the edge block, which the class owns.
- **Checks:**
  - an unknown tag returns `UnknownSlabReading`; it is never assumed to be Facet96;
  - a layout mismatch returns `SlabLayoutVersion`;
  - a value schema wider than the class's returns `SlabWidens`; a narrower one is allowed.
- `Activation::resolve_tenant_reading(key, Option<&SlabDeclaration>) -> ResolvedReading`:
  - with no declaration it returns exactly today's reading plus `Facet96`;
  - `ResolvedReading` is `Copy + Eq + Hash`, the future cache entry for a fold that resolves once per `(concept, declaration)`.

`SlabReading` has one variant, `Facet96` (the existing 4+12), at tag 0. No migration.

## OPEN

- Where a declaration is physically stored in the metadata envelope. Nothing writes one yet.
- New readings (a classid-free 128-bit register and its carvings), and authority validation of which concepts may opt into them.
- No caller is wired, and there is no cache yet.
