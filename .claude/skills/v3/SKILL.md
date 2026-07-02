---
name: v3
description: >
  Bootstraps V3-substrate awareness for this session: the mailbox-kanban
  model (no singleton CollapseGate), the SoaEnvelope LE ownership contract
  (mailbox_owner + write-on-behalf), compiled thinking templates
  (elixir-template / StepMask / graph-flow / Rig oracle), classid canon-high
  + the 0x1000 adoption monitor, and the DTO ladder (PerturbationDto split).
  Use when a session touches SoA rows, tenants, kanban, templates, mailbox
  ownership, or asks "what is V3 / where do I start". Canonical files live
  under .claude/v3/ — this skill is the loader, not the content.
---

# /v3 — V3 Substrate Bootload

You just entered V3 territory. Load, in this order, reading fully:

1. `.claude/v3/README.md` — what V3 is, what shipped, the doc map.
2. `.claude/v3/knowledge/v3-substrate-primer.md` — the ruled model in one
   page (mailbox-kanban, envelope ownership, DTO ladder, templates,
   classid canon + monitor, the must-not-reinvent table).
3. The domain doc your task touches:
   - write paths / consumers → `.claude/v3/knowledge/write-on-behalf.md`
   - kanban / batch writer / executors → `.claude/v3/knowledge/mailbox-kanban-model.md`
   - templates / orchestration / oracle → `.claude/v3/knowledge/compiled-templates.md`
   - byte layout / tenants / routing → `.claude/v3/soa_layout/README.md`
     (then `le-contract.md`, `tenants.md`, `routing.md` as needed)
4. `.claude/v3/INTEGRATION-PLAN.md` — which wave (W0–W6) your work
   belongs to and what gates it.
5. `.claude/v3/agents/BOOT.md` — which V3 card to wake
   (`v3-mailbox-warden` / `v3-envelope-auditor` /
   `v3-kanban-executor-engineer` / `v3-template-smith`).

Then apply the two standing gates to your own diff before any commit:

- **Ownership gate:** every write on behalf of
  `envelope.mailbox_owner()`; no ownership fields on DTOs; no
  CollapseGate/baton/emission resurrection. (`/v3-audit` runs the greps.)
- **Layout gate:** byte offsets stable or version-gated; field-isolation
  matrix for any layout-touching change; legacy read modes intact.

Tier-0 workspace reads (`LATEST_STATE.md`, `PR_ARC_INVENTORY.md`) remain
mandatory per the main `.claude/BOOT.md` — this skill adds to them, it
does not replace them.
