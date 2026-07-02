---
description: >
  Mechanical V3 conformance audit of the working tree (or a given crate):
  forbidden classid bit math, singleton/baton resurrection, ownership
  fields on DTOs, unstamped write paths, layout-version hygiene.
  Read-only; reports findings with file:line, fixes nothing.
---

Run the V3 conformance audit. Scope: `$ARGUMENTS` if given (a crate or
directory), otherwise the full workspace diff against origin/main plus
the contract crate.

Read `.claude/v3/knowledge/v3-substrate-primer.md` §5–6 and
`.claude/v3/knowledge/write-on-behalf.md` first if not already loaded.

Then run these checks with the Grep tool (never bash grep) and report
every hit with file:line and a one-line disposition
(violation / legacy-gated OK / test-only OK / false positive):

1. **Forbidden classid discriminators** on composed u32 classids:
   pattern `(as u16|& *0xFFFF|>> *16|>> *8)` in files that also match
   `classid|class_id|CLASSID`. Sanctioned: `classid_canon`,
   `classid_canon_compat`, `compose_classid`, `split_classid`,
   `render_classid`, `ogar_vocab::app::*`.
2. **Resurrection scan**: `CollapseGateEmission|MailboxSoA::emit|wire_cost_bytes|last_emission_cycle`
   — any non-historical, non-doc hit is a violation
   (successor: zero-copy envelope + kanban tenant).
3. **Ownership fields on DTOs**: `owner|mailbox_id|tenant_id` as FIELDS
   inside `StreamDto|ResonanceDto|PerturbationDto|BusDto|ThoughtStruct`
   definitions — BusDto especially (E-DTO-LADDER-OWNERSHIP-SPLIT).
4. **Unstamped writes**: direct Lance/SoA writes (`Dataset::write|append|insert`)
   in consumer crates without an envelope/`mailbox_owner` in scope —
   classify bake pipelines as grandfathered-bootstrap, online paths as
   violations.
5. **Layout hygiene**: diffs touching `soa_envelope.rs|canonical_node.rs`
   — verify const size asserts unchanged or `ENVELOPE_LAYOUT_VERSION`
   bumped, and that no `_LEGACY` read-mode key was removed (retirement is
   corpus-proof-gated).

End with a verdict per the v3-mailbox-warden vocabulary (OWNED /
BOOTSTRAP-OK / ORPHAN-WRITE / RESURRECTION) plus LAYOUT-CLEAN /
LAYOUT-GATED / LAYOUT-BREAK for check 5, and a one-paragraph summary.
