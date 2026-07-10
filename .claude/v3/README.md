# .claude/v3/ — the V3 Substrate entry point

> **Start here** (or invoke the `/v3` skill — same bootload) for any work
> touching SoA rows, tenants, mailbox ownership, kanban, thinking
> templates, classids, or the DTO ladder. This folder consolidates what is
> NEW since the 2026-07-02 operator rulings (mailbox-kanban model, classid
> canon-high flip, the 4+12 facet, compiled templates, the DTO ladder
> splits) so sessions converge instead of re-deriving.

## What V3 is, in five sentences

One **Mailbox** owns one SoA and one Kanban board (a tenant, never a
singleton); ractor proves ownership at compile time and the batch writer
fires **ahead** kanban updates at write cast, on behalf of
`envelope.mailbox_owner()`. Every V3 unit is a **16-byte facet**:
`[1B domain | 1B appid | 2B classview] + 96-bit payload`, the payload
reading selected by the **ClassView as the focus lens the data shape
wants** — labels and positions come from the ClassView, never a slot.
Thinking is **compiled**: the elixir-like DSL × StepMask is to
orchestration what askama × FieldMask is to rendering; a successful LLM
(Rig oracle) run compiles DOWN to a deterministic replayable template
executed by graph-flow under inherited ownership. Thinking cycles follow
**standing async plans** within a 550 ms net budget; updates reprioritize,
never gate. The `0x1000` custom half is a temporary **adoption monitor**;
at 100% adoption (P4) it retires and the classview half opens for the 64k
render/template catalogue.

## Doc map (read order per task)

| You are… | Read |
|---|---|
| new to V3 | `knowledge/v3-substrate-primer.md` (one page) → this README's plan/waves row |
| planning / sequencing | `INTEGRATION-PLAN.md` (waves W0–W6, gates, adopted D-ids) |
| deciding reuse vs build | `COMPONENT-MAP.md` (every subsystem's verdict, file:line) |
| hunting collapse work | `ENTROPY-MILESTONES.md` (M1–M23; each row has a mechanical gate) |
| touching bytes / tenants / addresses | `soa_layout/README.md` → le-contract / tenants / consumer-map / routing |
| per-file lookup (3 core crates) | `MODULE-TABLE.md` (consumes / emits / LE / debt / duplication / wave) |
| about to spawn Sonnet workers | `knowledge/sonnet-worker-guardrails.md` — **§1 preamble pasted verbatim, non-negotiable** |
| kanban / batch writer / executors | `knowledge/mailbox-kanban-model.md` |
| templates / oracle / graph-flow | `knowledge/compiled-templates.md` (incl. the corrected NextAction mapping) |
| consumer write paths | `knowledge/write-on-behalf.md` + `soa_layout/consumer-map.md` |
| ruff/odoo transcode landings | `knowledge/multi-anchor-ast-resolution.md` |
| waking an agent | `agents/BOOT.md` (the four `v3-*` cards + trigger routing) |
| landing future design / post-V3 rulings | `FUTURE-DESIGN.md` (the meta board: ruling index, migration arc, thinking-engine gem wiring queue) |
| keeping the vision (the WHY, graded) | `VISION.md` — the AGI-aspiring canon re-grounded 2026-07-10; every claim [G]/[RULING]/[ASPIRATION]; filigree-reviewed |

Shortcuts: `/v3` (bootload), `/v3-audit` (pre-commit conformance greps).
Canonical ruling texts live on the board (`.claude/board/EPIPHANIES.md`,
entries `E-V3-*`, `E-MAILBOX-KANBAN-NO-COLLAPSEGATE`,
`E-COMPILED-THINKING-TEMPLATES`, `E-DTO-LADDER-OWNERSHIP-SPLIT`,
`E-TWO-RESONANCES-SPLIT`, `E-RUFF-ODOO-MULTI-ANCHOR-AST`,
`E-V3-PLANNER-TWO-NATURES-AND-SPEED-PROBE`) — cite those in PRs, not
these mirrors.

## The three load-bearing NEW pieces (everything else is wiring)

1. **W1** — ahead-firing batch writer + delegation cache
   (`cast(on_behalf = envelope.mailbox_owner(), payload = BusDto)`).
2. **W2a** — the per-mailbox kanban board as a TENANT (type + lane).
3. **W3a/b** — StepMask + the ElixirTemplate→graph-flow adapter that
   closes the (falsified) control-flow gap.

## Standing gates

Probe-first; `v3-mailbox-warden` + `v3-envelope-auditor` on every
write-path/layout diff; jc pillars (ICC/Spearman/Cronbach) before a new
lane reading backs any claim; board hygiene same-commit; Sonnet grindwork
/ Fable decisions with the guardrails preamble in every brief; wire,
don't invent — COMPONENT-MAP is the precomputed "does it exist" search.

Tier-0 workspace reads (`.claude/board/LATEST_STATE.md`,
`PR_ARC_INVENTORY.md`) remain mandatory per `.claude/BOOT.md`; this folder
adds to them, never replaces them.
