
## S7-W7 Run — 2026-05-13

**Deliverable:** `ogit.SMB.bson:` sub-namespace wiring in `parse_family_registry()` / `family_registry.ttl`.

**Files touched:**
- `crates/lance-graph-callcenter/data/family_registry.ttl` — +76 LOC (3 SMB Foundry entries + 14 SMB BSON entries)
- `crates/lance-graph-callcenter/src/hydration.rs` — +87 LOC (parse_super_domain_name extension + 4 new unit tests U6-U9)

**Slot ranges chosen:**
- SMB Foundry-shape (`ogit.SMB:`): `OgitFamily(0x80..=0x82)` = 128..130 (family IDs 128, 129, 130)
- SMB BSON-shape (`ogit.SMB.bson:`): `OgitFamily(0xA0..=0xAD)` = 160..173 (14 entities)
- Rationale: 0x80-0x82 and 0xA0-0xAD are unconflicted with all prior ranges

**enumerate("SMB") contamination check:** CONFIRMED = 3 (Foundry entities only). BSON entities under "SMB.bson" namespace are disjoint. U8 test locks this invariant.

**cargo check:** both crates clean (only pre-existing warnings)
**cargo test family_:** 20/20 passed
**cargo test hydration:** 9/9 passed
