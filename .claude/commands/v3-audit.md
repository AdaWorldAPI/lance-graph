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
6. **V1-mint forbid** (ISS-V1-TAIL-RESIDUE): pattern `NodeGuid::new\(`
   in non-test production code is a violation — new mints route through
   `NodeGuid::mint_for(classid_read_mode(c).tail_variant, …)` so the
   class registry drives the tail. Sanctioned: `#[cfg(test)]` modules,
   `mint_for`'s own V1 arm (`canonical_node.rs`), and legacy-compat
   *reads* (`family()`/`identity()` accessors are reads, not mints).

7. **Ack-paced advance scan** (E-ACK-VIOLATION-REGRADE-1 — the ack-gated
   advance was a hard architecture violation; do not let it regrow):
   pattern `ack` co-occurring with `advance_phase|try_advance|KanbanMove`
   in the same file, plus `\.await` or `ractor::(call!|cast)` on any path
   that also matches `advance_phase|KanbanColumn|cycle`. Sanctioned:
   `batch_writer.rs` (durability bookkeeping + the SLA-gate doc),
   callcenter/OGAR SLA-membrane surfaces, tests, TD-MESSAGE-RESIDUE
   sites already ledgered. Any NEW hit that paces a reasoning cycle on
   an awaited event is a violation (kanbanstep is the only reasoning
   advance).

End with a verdict per the v3-mailbox-warden vocabulary (OWNED /
BOOTSTRAP-OK / ORPHAN-WRITE / RESURRECTION) plus LAYOUT-CLEAN /
LAYOUT-GATED / LAYOUT-BREAK for check 5, and a one-paragraph summary.
