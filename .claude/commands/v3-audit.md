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

7. **Elimination scan — the ack mechanism does not exist**
   (E-ACK-ELIMINATED-1, operator 2026-07-17; after E-ACK-VIOLATION-REGRADE-1
   + E-SOA-OWN-BOARD-NO-SIDECAR-1): patterns `ack_and_propose`,
   `\bunacked\b`, `\backed\b`, `fn ack\b`, `\.ack\(`, `ack-gated`,
   `actionhandler`, `SLA gate` in code or live doctrine — ANY hit is a
   violation (there is no sanctioned home; durability evidence is the
   row's own LanceVersion via temporal.rs). Also flag any NEW stored
   confirmation ledger in a writer regardless of its name ("retire",
   "confirm", "settle" + version map = the same mechanism), and any
   `\.await`/`ractor::(call!|cast)` on a path that also matches
   `advance_phase|KanbanColumn|cycle` (kanbanstep is the only advance).
   Exempt: board-history entries (`E-ACK-*` names in EPIPHANIES /
   PR_ARC / TECH_DEBT / plan addenda) — history is append-only.

End with a verdict per the v3-mailbox-warden vocabulary (OWNED /
BOOTSTRAP-OK / ORPHAN-WRITE / RESURRECTION) plus LAYOUT-CLEAN /
LAYOUT-GATED / LAYOUT-BREAK for check 5, and a one-paragraph summary.
