# Consumer Map — who writes what, on whose behalf (audit 2026-07-02)

> READ BY: v3-mailbox-warden, every consumer-crate session, W5 planners.
> Evidence: consumer-audit fleet pass over the six local checkouts,
> file:line-cited. The iron rule it audits against: `write-on-behalf.md`.

## Status: FINDING — one live ORPHAN-WRITE found (smb-office-rs); everything else is bakes or pre-persistence

---

## §1 The adoption tiers (classid/addressing)

| Tier | Consumers | Pattern |
|---|---|---|
| **T1 — canonical resolvers** | openproject-nexgen-rs `op-canon` (reference quality: pure re-exports of `ogar_vocab::app::*`, bit math only in round-trip TEST asserts); MedCare-rs `auth_bridge` (pulls `0x0B01_0000` through the rbac membrane, "never hardcodes") | delegate to the ONE upstream composer |
| **T2 — canonical composer itself** | OGAR `ogar-vocab::app` (`render_classid/app_of/concept_of` — the single sanctioned home of the bit math); `ogar-from-ruff/mint.rs` (composes via it; unmapped models mint bootstrap `0`, tested) | the canon |
| **T3 — dep-free byte-identical hand-copies** (sanctioned interim, OGAR_CONSUMER_INTEGRATION §2.3) | q2 `fma/converge.rs` (the model), q2 `cpic` (Genetics `0x0E01_000N`, interim by its own doc), woa-rs `erp/canon.rs` (unwired module, zero callers yet, Phase-3 doc points at `render_classid`) | own the 16 bytes, assert byte-identity, skip the dep — three copies, zero shared code (M21) |
| **T4 — outside the ladder entirely** | smb-office-rs (ZERO classid/NodeGuid anywhere; addresses by `(namespace, table, EntityKey)` strings via `contract::repository`) | never entered the classid ladder — cannot use prefix routing / key-only scans |

## §2 Write paths (the warden's table)

| Consumer / path | What it writes | Ownership | Verdict |
|---|---|---|---|
| q2 `osint-bake` bins (body.rs:153, fma.rs:440), `fma/converge.rs:448` | `body.soa` / `fma.soa` / TSVs via `std::fs::write` — offline artifacts (Release assets, not git) | none — offline single-writer | **BOOTSTRAP-OK** (the documented interim exception) |
| q2 cockpit-server runtime | NO durable writes — `include_bytes!` embeds + tokio RwLock in-memory state only | n/a | clean |
| **smb-office-rs `LanceConnector::upsert`** (smb-bridge/lance.rs:176-201) ← called from smb-woa/customer.rs:189-195 | **LIVE `Dataset::write` to a real Lance dataset**, keyed `(namespace, "smb.customer", EntityKey)` | **none — no stamp, no envelope, no classid** | **ORPHAN-WRITE** — the first live migration target of W5; the ONLY online consumer write found fleet-wide |
| MedCare-rs | no writer exists yet — `soa_mapping.rs` is schema metadata for 7 entities; the `medcare-soa` writer crate is "forthcoming" | n/a | **day-one opportunity**: route through the stamp from the first line (don't repeat smb-office) |
| woa-rs | no SoA/Lance writes at all (K-steps write MySQL via sea-orm per its writer-parity rule); `erp/canon.rs` has zero callers | n/a | pre-adoption |
| openproject-nexgen-rs | no persistence in scope — pure classid resolution | n/a | pre-persistence |
| OGAR `ogar-from-ruff` | emits SOURCE TEXT (Rust/C#/Python codegen), no storage writes | n/a | n/a |

## §3 Defects & drift found (dispositioned)

1. **OGAR `emit.rs:260-265,324-329,382-387`** — three identical
   `facet_classid() as u16` sites label the LOW half "(concept 0x…)" in
   generated doc comments. Pre-flip that was the concept; **post-flip it's
   the APP PREFIX** (`account.move` → prints `0x0002` instead of concept
   `0x0202`). The `_CLASSID` const itself (full u32) is correct. Fix =
   `ogar_vocab::app::concept_of(...)` at all three sites (1 line × 3);
   propagates into every generated SDK until fixed. → queued as W5 row.
2. **q2 `data/osint-v3/osint_v3_codebook.json`** — stores PRE-flip order
   (`0x1000_0700/0x1000_0701`); q2's own newer code (osint-bake
   `0x0700_0000`, BodyV3.tsx dual-alias) already migrated. No reader of
   the raw strings found → **latent**, but the exact I-LEGACY failure
   shape if any future reader assumes canon-high. → collapse with the
   dual-bake milestone (M22).
3. **woa-rs false positive on the V3 marker**: `festschreibung.rs:117`'s
   `0x10000` is UTF-16 surrogate math, NOT the marker — recorded so a
   grep-only pass never miscounts woa-rs as V3-marker-aware.

## §4 Reference patterns (point consumers here)

- **Classid resolution**: `op-canon/src/app.rs` — thin re-exports, test-only
  bit math for round-trip verification.
- **Dual-alias read across the flip**: q2 `BodyV3.tsx:22-28,437-443` —
  `FMA_V3_CLASSID` + `_LEGACY`, both checked at render time; the cleanest
  V3-marker-aware code in the fleet.
- **Post-flip self-documentation**: q2 `osint_gotham.rs:507-513` — doc
  comment warning that `classid & 0xFFFF` now reads the custom half.
- **Zero-fallback minting**: `ogar-from-ruff/mint.rs` — unmapped model →
  bootstrap `0`, never a wrong stamp (tested).

## §5 W5 consequences (feeds INTEGRATION-PLAN)

1. smb-office-rs `LanceConnector::upsert` is **W5's first live migration**
   (stamp + batch-writer routing when W1 lands; until then: explicitly
   grandfather-or-flag, never silently assume it's a bake).
2. MedCare-rs `medcare-soa` writer: **born stamped** (design gate before
   first merge).
3. OGAR emit.rs 3-line fix + regen check.
4. q2: collapse the two OSINT bakes (crates/osint-bake canon-high vs
   data/osint-v3 stale pre-flip dual-GUID scheme) into one canon-high bake
   (M22); cpic contract pull with mereology dissolves the interim scheme
   (existing D-V3-W5b).
5. T3 hand-copies → one zero-dep `canon-node-bytes` extraction (M21) —
   ends three-way byte-layout re-derivation.

Cross-ref: `write-on-behalf.md` (corrected: "interim reality" now names
the smb-office exception), COMPONENT-MAP §7, ENTROPY-MILESTONES M21-M23,
board E-V3-FACET-4-PLUS-12.
