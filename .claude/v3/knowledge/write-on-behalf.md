# Write-on-Behalf — the fleet-wide consumer iron rule

> READ BY: mailbox-warden, envelope-auditor, every consumer-crate session
> (q2, MedCare-rs, woa-rs, smb-office-rs, openproject-nexgen-rs, OGAR,
> ladybug-rs), and any session adding a write path to SoA rows / Lance
> datasets / tenant lanes.

## Status: FINDING (operator-ruled 2026-07-02; enforcement mechanics EXTEND — batch writer pending)

---

## The rule

**Every consuming crate writes ON BEHALF OF the ractor owner mailbox (the
compile-time ownership guarantee). Always.** No consumer writes a SoA row, tenant lane, or Lance dataset as
itself. The write names its owner via the envelope stamp:

```rust
// the ONLY sanctioned shape (batch-writer pairing):
cast(on_behalf = envelope.mailbox_owner(), payload = BusDto { .. })
```

- `SoaEnvelope::mailbox_owner() -> MailboxId` is the nominal stamp
  (default `0` = bootstrap, zero-fallback ladder — dormant until minted).
- The envelope's `&self` views are the structural half: borrowing a view IS
  borrowing from the owner, proven at compile time.
- **`BusDto` never grows ownership fields.** Cognitive provenance (what was
  thought, how settled: converged, cycle_count) and write provenance (whose
  lane, who may write) are orthogonal and are paired AT CAST, not merged
  into one type. A PR adding `owner`/`mailbox`/`tenant_id` to `BusDto` (or
  any Φ/Ψ/B DTO) is a defect — reject and route to the envelope stamp.

## Why (the two directions of inheritance)

- **Down (Lance provenance):** every persisted row is attributable to a
  mailbox — the L4 learning loop (perturbation residue → tenant lane →
  next cycle's template) only works if lanes are owner-keyed.
- **Up (consumer clarity):** a consumer never needs to know SoA internals;
  it needs one MailboxId and the contract composers. Delegation questions
  (may THIS caster write into THAT lane?) are the batch writer's delegation
  cache, resolved at cast time — never the consumer's problem and never a
  RBAC re-implementation in the consumer.

## Consumer preflight (5 questions, 60 seconds)

Before authoring any consumer write path:

1. **Who is the owner?** Which mailbox does this write serve? If "none",
   you are writing bootstrap (`0`) — is that intended, or are you missing a
   mint?
2. **Where is the stamp?** Does the write route through an owner-stamped
   envelope / the batch writer, or does it hit Lance/SoA directly?
   Direct = defect (interim exception: bake pipelines, see below).
3. **Is the payload clean?** No ownership fields on the DTO; classids
   composed via `contract::render_classid` / `ogar_vocab::app::*`, never
   local bit math.
4. **Which tenant lane?** Named per `soa_layout/tenants.md`; new lanes need
   the field-isolation matrix (envelope-auditor gate).
5. **Is the write idempotent per cycle?** `last_active_cycle` is the
   same-cycle guard — respect it, don't reinvent it.

## Interim reality (audited 2026-07-02; CORRECTED same day by the consumer audit)

No consumer writes on-behalf yet — the batch writer does not exist
(INTEGRATION-PLAN W1). Almost all consumer writes are **bake pipelines**
(q2 `osint_scene.soa` / `fma.soa` / `body.soa`): offline, single-writer,
owner-less by construction — grandfathered as bootstrap-owner writes,
migrating in W5.

**ONE exception, found by audit and named so it is never assumed away:**
smb-office-rs `LanceConnector::upsert` (smb-bridge/lance.rs:176-201,
live caller smb-woa/customer.rs:189-195) is an **online** Lance write —
no stamp, no envelope, no classid. It is W5's first live migration
target (`consumer-map.md` §2), explicitly flagged rather than silently
grandfathered. Do not add NEW online write paths without the stamp; do
not model new writers on the smb-office path.

Cross-ref: `v3-substrate-primer.md` §2–3, `mailbox-kanban-model.md`,
`.claude/v3/soa_layout/consumer-map.md`, board `E-DTO-LADDER-OWNERSHIP-SPLIT`.
