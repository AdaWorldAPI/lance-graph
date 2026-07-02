# soa_layout/ — the V3 byte-and-ownership contracts

> Per-tenant + consumer documentation mapping, LE contract, and routing —
> the operator-requested layout home (2026-07-02). Read in this order:

| Doc | Carries | Read when |
|---|---|---|
| **`le-contract.md`** | the 4+12 facet atom (`[domain\|appid\|classview] + 96-bit payload`), the L1–L8 payload catalogue + polymorphic (8:8) readings, slot purity ("labels and positions come from the ClassView"), the two-level LE contract + jc-pillar gate (§3b), classview-as-focus-lens + the 64-bit-cramp retirement, and the CODE ground truth (§5: FacetCascade/CascadeShape/hi-lo chains + the honest discrepancies) | touching any byte layout, facet, or payload reading |
| **`tenants.md`** | the 10 ValueTenant lanes with exact offsets/widths (ENVELOPE_LAYOUT_VERSION=2), ValueSchema presets, the classid→ReadMode registry, the in-RAM MailboxSoA mirror, and the four flagged seams (SoaEnvelope zero production impls; Meta/Plasticity width mismatches; MailboxId≠NiblePath; jc gate per lane) | reading/writing a tenant lane, adding a schema, wiring a consumer |
| **`consumer-map.md`** | the six-consumer audit: adoption tiers T1–T4, the warden write-path table (ONE live ORPHAN-WRITE: smb-office-rs), dispositioned defects, reference patterns, W5 consequences | any consumer-crate session; W5 planning |
| **`routing.md`** | address-as-router (canon-high clustered index, tier shift/mask, zero-fallback ladder), MailboxId/NiblePath routing, the cast→delegation→owner→board write route, read-mode aliasing, adoption-as-range-count | adding any lookup/scan/dispatch over V3 keys |

Iron context (don't re-derive): `../knowledge/v3-substrate-primer.md` is
the one-page doctrine; board `E-V3-FACET-4-PLUS-12` /
`E-V3-CLASSVIEW-FOCUS-LENS` / `E-V3-TWO-LEVEL-LE-JC-GATE` are the
canonical ruling texts; `CLAUDE.md § CANON` locks the 512-byte node.
Gates: every layout diff → `v3-envelope-auditor`; every write path →
`v3-mailbox-warden`; every new lane consumer → jc pillars (ICC / Spearman
ρ / Cronbach α) before its reading backs a claim.
