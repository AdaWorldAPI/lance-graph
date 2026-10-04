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

## Correction: group nodes are explicit membership, not storage sharing

Operator: *"Group nodes wasn't about saving, it was about active directory like group memberships for explicit members as opposed to 24xi4 abstract qualities."*

Finding 2 above ("exact-set group nodes do not deduplicate") measured storage sharing, which is the wrong question.

- **A group node is an enumerated, crisp membership**, like an AD group: *the sons of Levi* = {Gershon, Kohath, Merari}.
- Groups nest: a member is itself a group.
- A verse refers to the group with one rail; members resolve through the group node's own member tenant, transitively through nested groups.
- Contrast: the 24×i4 tenant flavours hold graded, abstract qualities that are inferred, not enumerated.

**Exploratory scan.** A regex over the KJV for "the sons/children/… of X; A, B, and C" (uncommitted lab script, noisy):
- **74 explicit groups**, 380 members, median 5 members, maximum 14.
- **29 nested groups**: a member heads its own group. Example chain: sons of Israel → Levi → sons of Levi → Kohath → sons of Kohath → Amram, four levels.
- **Groups are named collectively far more often than they are enumerated.** Collective mentions: *sons of Aaron* 28, *sons of Levi* 21, *children of Israel* 647, *the twelve* 38, *twelve tribes* 10.

**Known noise:**
- Dinah (a daughter) is captured under the sons of Jacob.
- The variant spellings Gershom/Gershon are not merged.
- *Children of Israel* is never enumerated by this pattern.

**OPEN.**
- A group-node reader:
  - enumerations → member tenant (8 × u8:u8 per 16 B, chained for larger groups);
  - nested membership as group → group edges;
  - a collective mention → one group rail in the verse tenant.
- Then measure how many verse rails collective mentions replace.

## Addendum: the same 16 classid-free bytes, read as 16 × u8 or 32 × u4

The operator points out that a classid-free 16-byte register can be read three ways (the content-blind register idea, one size up from the 12-byte facet):
- 8 × (u8:u8) rails;
- 16 × u8 tags;
- 32 × u4 sparse tags.

**Current contract.**
- `CascadeShape` defines only 12-byte carvings (G6D2 / G4D3 / G3D4, G·D = 12).
- `CausalWitnessFacet` (the 24 × i4 tenant flavours) is exactly 12 bytes.
- A 16-byte register as 32 × u4 holds the 24 witness loci plus 8 spare nibbles.
- None of the 16-byte readings exist yet; adding them is a contract change.

**Natural fits:**
- 16 × u8: verb atom (144), STTS (54) or UPOS (17), DeReKo frequency class, per token;
- 32 × u4: case + number, lane, or sparse flags, per token. Zero means unbound.

**Measured: clause length (tokens between punctuation; lab scan, uncommitted).**

| corpus | median | p95 | ≤ 16 tokens | ≤ 32 tokens |
|---|---|---|---|---|
| KJV | 6 | 13 | 98.5 % | 100.0 % |
| Luther 1545 | 5 | 13 | 98.1 % | 100.0 % |
| Buddenbrooks | 5 | 13 | 98.2 % | 100.0 % |
| Effi Briest | 5 | 12 | 98.4 % | 99.9 % |
| UD HDT (news) | 7 | 19 | 91.4 % | 99.9 % |

One 16-byte tenant per clause therefore carries a per-token u8 tag for about 98 % of literary and biblical clauses (91 % of news), and a per-token nibble for essentially all clauses.

**OPEN.** This is the same contract decision as above. It needs `CascadeShape` variants for 16 bytes (G8D2, G16D1, nibble×32), the envelope auditor's field-isolation matrix, and an operator ruling.
