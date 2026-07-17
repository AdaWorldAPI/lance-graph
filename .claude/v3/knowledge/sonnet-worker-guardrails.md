# Sonnet Worker Guardrails — V3 footgun catalogue + the worker preamble

> READ BY: **every Sonnet worker before its first tool call**, and by every
> orchestrator writing a worker brief. The orchestrator MUST paste §1 (the
> preamble) verbatim into every Sonnet worker prompt that touches this
> workspace. This doc exists because grindwork agents execute literally:
> every rule here is mechanical — no judgment calls required, ever.
> If a worker hits a situation not covered by a rule below, the rule is:
> **STOP and return the question; do not improvise.**

## Status: FINDING (operator directive 2026-07-02: "no foot gun at any time")

---

## §1 — The worker preamble (orchestrators: paste verbatim into every brief)

```text
WORKER IRON RULES (V3 workspace — mechanical, no exceptions):
1. SCOPE: touch ONLY the files named in this brief. If a fix seems to
   need another file, STOP and report; do not follow the thread.
2. READ FULLY: Read every file you will edit, entirely, before editing
   (offset/limit chunks for >2000 lines — ALL chunks). Never paraphrase
   from grep/snippet output. grep locates; Read comprehends.
3. NO INVENTION: never mint a new struct/trait/enum/module. If the brief
   needs a type, it names the existing one. A "missing" type = STOP+report.
4. CLASSIDS: compose ONLY via contract::render_classid / compose_classid
   (or ogar_vocab::app::* in OGAR). Discriminate ONLY via classid_canon /
   classid_canon_compat. NEVER write `as u16`, `& 0xFFFF`, `>> 8`, `>> 16`
   on a composed classid u32 — not even in tests.
5. BOARD FILES: .claude/board/*.md are append-only ledgers. Bash
   `tee -a` ONLY. Never Edit/Write/`>` them. Never reorder or delete rows.
6. BRANCH: work on the branch this brief names. Never checkout/switch/
   create branches. Never push unless the brief says push.
7. NO CARGO BUILDS: do not run `cargo build`/`cargo check`. A targeted
   `cargo test -p <crate> <filter>` or `cargo clippy -p <crate>` is
   allowed ONLY if the brief explicitly grants it. Verification is
   centralized in the orchestrator.
8. LEGACY IS LOAD-BEARING: never delete, rename, or "clean up" anything
   marked deprecated/legacy/_LEGACY/superseded (aliases, read modes,
   collapse_gate.rs contents, BindSpace). Retirement is proof-gated and
   never a worker task.
9. DTO PURITY: never add owner/mailbox_id/tenant_id fields to any DTO
   (StreamDto/ResonanceDto/PerturbationDto/BusDto/ThoughtStruct).
   Ownership lives in SoaEnvelope::mailbox_owner(), nowhere else.
10. CLAIMS: any "there is no X" / "all sites routed" claim in your report
    MUST name the exhaustive search that backs it (tool + pattern + scope),
    or be phrased as "not found in <scope searched>".
11. DONE = your diff + the named test/probe green + a report listing:
    files touched, searches run, anything you did NOT do. Partial work is
    reported as partial, never as done.
```

## §2 — Vocabulary disambiguation (the words that bite)

Sonnet workers MUST use this table; a brief using one of these words means
exactly the row below, never a neighbor.

| Word | In V3 it means | It does NOT mean | Never do |
|---|---|---|---|
| **tenant (value tenant)** | a lane in the 480-byte value slab, selected by `classid_read_mode(c).value_schema` | a customer/org (that's consumer-app tenancy) | invent a lane; new lanes are envelope-auditor-gated |
| **KanbanTenant (per-row)** | the existing per-row kanban state type | the per-mailbox board | extend it to carry board state |
| **kanban board (per-mailbox)** | W2a deliverable: a dedicated board ROW (classid-routed) whose aggregates live in the NEW append-only 10th ValueTenant `BoardAggregates` (row_offset 152) — 2026-07-02 envelope-audit ruling, plan Addendum-12a: LAYOUT-GATED, tests T1-T6 + batched classid mint mandatory | a global/singleton board; a reuse-reinterpretation of existing tenant bytes (pre-P4 inexpressible) | implement without Addendum-12a's T1-T6 + STOP list; let the board classid fall through to ReadMode::DEFAULT |
| **cascade** | HEEL/HIP/TWIG key tiers (GUID canon; 256×256 centroid tiles) | the perturbation field | rename anything containing "cascade" |
| **PerturbationDto** (was dto.rs `ResonanceDto`) | the MECHANICAL Morton-tile inverse-pyramid field (Ψ) | awareness/perspective | touch awareness_dto.rs during D-PERT-1 |
| **ResonanceDto (awareness_dto.rs)** | the PERSPECTIVAL (Piaget Three-Mountains) resonance — KEEPS its name | a duplicate to dedup | rename/merge it with the Ψ DTO |
| **CollapseGate** | a legacy MODULE NAME (`collapse_gate.rs` still hosts live types: MailboxId, KanbanPhase, MergeMode) | a live singleton gate semantic | re-implement gate semantics OR delete the module |
| **BindSpace** | the legacy global store, MIGRATION IN PROGRESS → MailboxSoA | something to extend or to delete now | add new writers to it; remove it |
| **baton / emission / CollapseGateEmission / emit()** | tombstoned concepts (removed from source) | anything to restore | reintroduce, even "for compatibility" |
| **owner / on_behalf** | `SoaEnvelope::mailbox_owner()` stamp + batch-writer cast pairing | a DTO field | add ownership fields to DTOs (§1 rule 9) |
| **0x1000 (custom half)** | the V3-adoption MONITOR marker (temporary by declaration) | a domain id or a semantic flag | branch business logic on it; mint new meanings for it |
| **canon / custom** | classid halves: hi u16 = canon concept, lo u16 = custom marker/render | "old/new" | read halves with bit math (§1 rule 4) |
| **template** | a compiled, replayable orchestration artifact (elixir-template DSL) | a prompt or a string | degrade a template into a prompt |
| **StepMask** | a QUEUED contract type (W3a) — does not exist yet | something to invent ad hoc | mint it without the W3a spec |
| **facet** | the V3 16-byte atom: 4B prefix (domain\|appid\|classview) + 96-bit payload (soa_layout/le-contract.md §3 catalogue L1–L8) | a free-form struct | add a layout outside the L1–L8 catalogue; put a label/position in a slot |
| **classview (lo u16)** | the ClassView selector in the classid custom half — labels + positions resolve THROUGH it | a place to store data bits | branch on raw classview values other than via read-mode helpers |
| **rail** | a 6×(8:8) payload plane of one-byte refs (part_of:is_a etc.) | an edge list to grow | change rail arity/stride |
| **CAM_PQ digital / analog** | digital = 6×(8:8) palette256² byte pairs (L4, LUT similarity); analog = 48-bit helix + 6B CAM-PQ (L8) | interchangeable encodings | mix the two styles in one lane; compare codes with float math (similarity = DeepNSM 4096 codebook table lookup) |

## §3 — Footgun catalogue (what broke before → the mechanical prevention)

| # | Footgun (documented incident) | Mechanical prevention |
|---|---|---|
| F1 | Snippet-read paraphrase errors in briefs/edits (woa-rs Round 9: 3 workers lost 30+ min each) | §1 rule 2. Full Read before edit; grep is a locator only |
| F2 | "All sites routed" claimed from plan inventory; rbac.rs missed (E-CLASSID-COMPAT-READER) | §1 rule 10: coverage claims name the whole-crate grep (pattern + scope) |
| F3 | Local classid bit math shipped 3 latent bugs pre-flip | §1 rule 4 + `/v3-audit` check 1 before commit |
| F4 | Board file edited/overwritten (append-only violation) | §1 rule 5; settings.json denies Edit/Write on the 8 files — do not work around a denial |
| F5 | Write-over-self: file regenerated from prompt instead of edited from state (~N ins/~N del diff signature) | Edit tool for existing files; Write only for NEW files; after a denial or conflict, re-Read then Edit |
| F6 | Shared-tree branch races between parallel workers | §1 rule 6; parallel writers get disjoint FILE lists (this workspace shares one checkout — no worktrees, per cargo-hygiene rule) |
| F7 | 12× cold `target/` build residue (~7 GB each) from fleet cargo runs | §1 rule 7; orchestrator compiles once, centrally |
| F8 | Resurrection of superseded framings copied from stale prose (old CLAUDE.md sections, old plans) | §2 rows CollapseGate/baton/BindSpace; when a doc contradicts `.claude/v3/knowledge/v3-substrate-primer.md` §6, the primer wins — cite it, don't "fix" the code to match stale prose |
| F9 | Deleting/renaming deprecated aliases as "cleanup" (retirement is corpus-proof-gated) | §1 rule 8; the word LEGACY in an identifier = hands off |
| F10 | Inventing a duplicate of an existing type (30-turn rediscovery tax; 4× Fingerprint, 3× ZeckF64 precedent) | §1 rule 3; the brief names types; MODULE-TABLE.md + LATEST_STATE.md Contract Inventory are the lookup — search before any `struct` keyword |
| F11 | New REST endpoint / Wire DTO added on the System-1 path | LAB-only doctrine: extend `UnifiedStep`/`OrchestrationBridge`; a worker asked to add an endpoint = STOP+report (needs lab-vs-canonical-surface.md, an orchestrator decision) |
| F12 | Test literals that only work in one classid order / one layout version (synthetic 0xAB12-style values) | test values come from the §2-sanctioned composers + realistic allocations; never hand-rolled hex composites |
| F13 | Worker "fixes" a failing unrelated test to get green | out of scope (§1 rule 1): report the failure, touch nothing |
| F14 | German PII labels or model identifiers in committed artifacts | never emit either in any file/commit (workspace-wide rule); chat-only |
| F15 | A cycle/phase advance paced on a completion/confirmation event, message, or awaited anything — or a stored confirmation ledger added to a writer (the eliminated ack mechanism; E-ACK-VIOLATION-REGRADE-1 → E-ACK-ELIMINATED-1) | kanbanstep is the ONLY advance (inline `on_version → try_advance_phase(&mut)`); the ack concept DOES NOT EXIST — durability evidence is the row's own LanceVersion via temporal.rs; ractor = compile-time ownership guarantee, never messaging (10⁴–10⁷× tax). A worker asked to await anything before an advance, or to add confirmation bookkeeping under ANY name = STOP+report |

## §4 — Sanctioned command palette (copy these, don't improvise)

- Pre-commit conformance: run the checks in `.claude/commands/v3-audit.md`
  (or ask the orchestrator to run `/v3-audit <crate>`).
- Board append (the ONLY board write shape):
  `tee -a .claude/board/<FILE>.md > /dev/null <<'EOF' … EOF`
- Coverage grep for classid discriminators (F2/F3):
  Grep pattern `(as u16|& *0xFFFF|>> *16|>> *8)` over the WHOLE crate you
  touched, then disposition every hit in your report.
- Existing-type lookup before writing any `struct`/`trait`/`enum`:
  Grep the name across `crates/` + check `.claude/v3/MODULE-TABLE.md` and
  `.claude/board/LATEST_STATE.md` § Contract Inventory.

## §5 — Escalation (STOP+report triggers — return these, don't resolve them)

A Sonnet worker returns the question to the orchestrator when ANY of these
appears, because each requires accumulation-tier judgment:

1. The brief's spec contradicts the code on disk (drift — needs
   preflight-drift-auditor, not a local fix).
2. A needed type/lane/mask does not exist (needs a mint decision).
3. Two docs disagree (needs a supersession ruling; primer §6 usually
   answers, but the RULING is not the worker's to make).
4. The change would touch bytes at rest (offsets, widths, stored forms —
   needs v3-envelope-auditor).
5. The change would add/modify a write path's ownership routing (needs
   v3-mailbox-warden).
6. Anything RBAC, PII-adjacent, or externally visible.
7. The change would make ANY cycle/phase advance wait on a completion
   or confirmation event, a message, a `ractor::call!`/`cast`, or any
   awaited event — or would add ANY stored confirmation ledger to a
   writer (the eliminated mechanism, E-ACK-ELIMINATED-1; durability
   evidence is the row's own LanceVersion read via temporal.rs, and no
   equivalent may be rebuilt under any name).

Cross-ref: `.claude/knowledge/autoattended-multiagent-pattern.md` (the
4-savant wave pattern these rules slot into), `v3-substrate-primer.md`
(the doctrine the rules protect), `.claude/v3/MODULE-TABLE.md` (the
existing-type lookup), main `.claude/agents/BOOT.md` (LD-1..5 proof-of-read
checks orchestrators may demand).
