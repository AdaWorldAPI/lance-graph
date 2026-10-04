# 2026-10-04 — SPOG × slab declaration resolves through the existing hot-plug (D-LXC-29 follow-up)

**Status:** VERIFIED-IN-CODE + TEST-PINNED (`crates/lance-graph-contract/src/hotplug.rs`, 4 tests, 6 guards disable-verified red-then-green).

## What landed

The resolution path asked for in `2026-10-04-deepnsm-v2-tenant-rails-6-vs-8.md` ("a `ValueSchema` entry that the hot-plug `ReadMode` resolves to"), without any layout change:

- **SPOG** gives the concept: `spog_tenants::graph_of(key)`, read from the key.
- **OGAR registry** gives the class reading: `Activation::read_mode_for(concept)`; an unplugged concept bangs `NoReadingFor`, and a slab declaration cannot stand in for it.
- **Slab metadata** is `SlabDeclaration {concept, read_mode, layout_version}`. It is checked, not obeyed:
  - concept must equal the SPOG graph → else `SlabConceptMismatch`;
  - layout version must equal `ENVELOPE_LAYOUT_VERSION` → else `SlabLayoutVersion`;
  - tail and edge codec must equal the authority's (class semantics) → else `SlabReadingConflict`;
  - value schema may be a subset (fewer tenants written), never a superset → else `SlabReadingConflict`.
- `Activation::resolve_tenant_reading` returns the declared reading when it passes (it matches the bytes), the authority's when there is no declaration. No fallback reading on any path.

No second registry: the method lives on `Activation`, the hot-plug result.

## OPEN

- Where a declaration is physically stored (group header beside the slab, as palette gamma is) — not decided; nothing writes one yet.
- No caller is wired; consumers still use `read_mode_for` directly.
- Raw128 / 16-byte carvings, new tenants, NodeRow/FacetCascade: untouched by design.
- Whether a slab may ever change the edge codec (e.g. a Pq32x4 slab of a CoarseOnly class) is ruled "no" here; revisit if a real slab needs it.
