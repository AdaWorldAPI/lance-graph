# 2026-10-04 — 6 vs 8 rails per 16-byte tenant, measured on KJV basin membership (D-LXC-29)

**Status:** MEASURED on the full KJV with `crates/deepnsm-v2/examples/toc_hydrate.rs` (debug-0, release), section "TENANT CAPACITY". No layout is changed.

## The proposal

The operator proposed using 8 × (u8:u8) per 16-byte tenant instead of 6 × (u8:u8):
- the V3 facet is classid(4) + 12 bytes = 6 rails;
- a tenant **inside a row whose key already carries the classid** repeats that classid;
- so the full 16 bytes could be rails, giving 8.

Suggested tenants:
- tenant 1: the verse address;
- tenant 2: basins (`part_of:is_a` style) or edges;
- many-to-many tenants as group nodes.

## Measurement

- One rail holds one 16-bit reference (a basin id; 1,187 basins > 255, so one `u8:u8` pair per basin).
- Per verse: the distinct promoted basins whose subject or object occurs in that verse's triples. 25,460 verses have triples.
- "Group nodes" = distinct overflowing reference sets, i.e. many-to-many group nodes with exact-set sharing.

| | median | p95 | max | fit in one tenant | overflow verses | group nodes |
|---|---|---|---|---|---|---|
| 6 rails (facet 4 + 12) | 3 | 6 | 12 | 95.66 % | 1,106 | 1,099 |
| **8 rails (16 B, classid in the slab)** | 3 | 6 | 12 | **99.41 %** | **149** | **146** |

**Findings.**
1. **8 rails cut the overflow about 7.4×.** The p95 sits exactly at 6, so 6 rails cut into the tail and 8 rails clear almost all of it.
2. **Exact-set group nodes do not deduplicate**: 1,099 of 1,106 overflowing sets are unique. A useful many-to-many group would be a frequent SUBSET (itemset) shared across verses, not an exact set. Not measured.
3. **The edge measure here is degenerate.** Links come from the reading-order chain, so every verse has exactly one next link (max 1). An edge tenant needs coreference or shared-basin links, not measured.
4. The verse address (book, chapter, verse, each ≤ 255) needs 2 rails under either carving.

**OPEN, a contract decision rather than a measurement.**
- The 4 + 12 facet is canon (`E-V3-FACET-4-PLUS-12`, `.claude/v3/soa_layout/le-contract.md` §3).
- A classid-free 8-rail register as a value-slab tenant is a new reading.
- It needs the envelope auditor (`v3-envelope-auditor`: field-isolation matrix, read-mode alias) and an operator ruling before any lane changes.
- BASIS: this table. REVISIT WHEN: the ruling lands, or edge/coreference degree is measured.
