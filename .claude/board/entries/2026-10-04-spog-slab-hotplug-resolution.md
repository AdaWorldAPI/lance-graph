# 2026-10-04 — SPOG × slab declaration resolves through the existing hot-plug (D-HPS-1)

**Follows:** the D-LXC-29 tenant-rails entry.

**Status:** VERIFIED-IN-CODE + TEST-PINNED (`crates/lance-graph-contract/src/hotplug.rs`, 8 tests; 7 guards disable-verified red-then-green; `lance-graph-ogar` 84/84 unchanged).

## The four authorities, kept separate

| authority | carrier | decides |
|---|---|---|
| SPOG | `spog_tenants::graph_of(key)` | which concept the data belongs to |
| OGAR | `Activation::read_mode_for(concept)` | that the concept exists, and its CURRENT reading (dispatch for new writes) |
| slab metadata | `SlabDeclaration {reading, value_schema, layout_version}` | the physical truth about bytes already written |
| this build | `SlabReading::from_tag`, `ENVELOPE_LAYOUT_VERSION` | whether the binary implements that physical reading |

## Rules (`Activation::resolve_for_context`)

- Unknown concept → `NoReadingFor`, with or without a declaration.
- No declaration → the current OGAR reading, `slab: None`. Absence is inherited behaviour, never a Facet96 claim.
- Declaration with an unsupported tag or layout → `UnknownSlabReading` / `SlabLayoutVersion`.
- Otherwise the declaration wins for its own bytes: the declared value schema and reading are returned, whatever OGAR registers today. A class migration (e.g. yesterday G6D2, today G8D2) leaves old slabs readable; new writes use the new registration and record it.
- Tail and edge codec always come from OGAR (the key and edge block are not the slab's).
- There is no per-concept table of allowed readings.

`resolve_tenant_reading(key, …)` is the key wrapper (`graph_of` + the above). A caller holding one population resolves once with `resolve_for_context`; the population test shows that call shape only, not a fold or branch-freeness.

## OPEN

- Where a declaration is physically stored in the metadata envelope.
- A writer persisting the current reading into slab metadata.
- New readings (a classid-free 128-bit register and its carvings) as `SlabReading` variants.
- No caller and no cache are wired.
