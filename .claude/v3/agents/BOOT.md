# V3 Agent Ensemble — activation table

> The V3 EXTENSION of `.claude/agents/BOOT.md`, not a replacement. The main
> ensemble (19 specialists + 5 meta-agents + PP-13/14/15/16 savants) stays
> authoritative for everything it already covers. This table adds the four
> V3-aware cards and routes V3 triggers to them.
>
> **Card files live in `.claude/agents/v3-*.md`** (the harness discovers
> agents there); this folder holds the activation table + the V3 knowledge
> the cards bootload.

## The four V3 cards

| Card | Kind | Fires on | Verdicts / Output |
|---|---|---|---|
| `v3-mailbox-warden` | reviewer | any SoA/Lance write path; ownership-ish fields on DTOs; CollapseGate/baton/emission resurrection; consumers writing as themselves | OWNED / BOOTSTRAP-OK / ORPHAN-WRITE / RESURRECTION |
| `v3-envelope-auditor` | reviewer | soa_envelope.rs / canonical_node.rs / collapse_gate.rs diffs; new tenant lanes; byte-offset changes; ENVELOPE_LAYOUT_VERSION | LAYOUT-CLEAN / LAYOUT-GATED / LAYOUT-BREAK (+ layout-diff table) |
| `v3-kanban-executor-engineer` | builder | style_strategy.rs / kanban_actor.rs / kanban_loop.rs; batch-writer, ahead-update, delegation-cache designs; 550 ms scheduling | probe-first implementation, gated by the two reviewers |
| `v3-template-smith` | builder | elixir-template / template-runtime / template-equivalence / cognitive-compiler; StepMask; graph-flow adapter; Rig oracle loop | probe-first implementation; replay-equivalence gate |

## Knowledge bootload (Tier-1 for V3 work)

Every V3 card reads, in order:

1. `.claude/v3/knowledge/v3-substrate-primer.md` — always.
2. Its own domain doc: `mailbox-kanban-model.md` (warden, engineer) /
   `write-on-behalf.md` (warden + every consumer session) /
   `compiled-templates.md` (smith).
3. `.claude/v3/soa_layout/le-contract.md` + `tenants.md` (auditor; anyone
   touching layout).
4. Tier-0 stays mandatory: `.claude/board/LATEST_STATE.md` +
   `PR_ARC_INVENTORY.md` (main BOOT.md rule).

## Trigger routing (V3 additions to the main Knowledge Activation table)

| Trigger phrase / diff surface | Wake | With knowledge |
|---|---|---|
| "write on behalf", "mailbox owner", "delegation", new write path | v3-mailbox-warden | write-on-behalf.md |
| "SoaEnvelope", "ColumnDescriptor", "tenant lane", "layout version", byte offsets | v3-envelope-auditor | soa_layout/le-contract.md + tenants.md |
| "kanban", "ahead update", "batch writer", "550ms", "KanbanMove", D-MBX-A6 | v3-kanban-executor-engineer | mailbox-kanban-model.md |
| "template", "StepMask", "replay", "oracle", "graph-flow", "compile down" | v3-template-smith | compiled-templates.md |
| "CollapseGate", "baton", "emission", "BindSpace sink" (non-legacy context) | v3-mailbox-warden (RESURRECTION scan) | v3-substrate-primer.md §6 |

## Relationship to existing cards

- `dto-soa-savant` keeps the four-column AGI-as-SoA lens; the warden adds
  the OWNERSHIP lens — they compose, not compete.
- `baton-handoff-auditor` keeps cross-crate boundary review; on V3
  surfaces it defers resurrection-scan findings to the warden's verdict
  vocabulary.
- `soa-review` (smb-office flavored) and `v3-envelope-auditor` split by
  repo: Arrow-schema consumer reviews there, LE register-file audits here.
- The PP-13/14/15/16 lifecycle savants apply unchanged to V3 waves.

## Model policy (per operator, 2026-07-02)

Sonnet 5 for grindwork (mechanical wiring, greps, censuses, doc stubs from
spec); Fable/Opus tier for decisions, plans needing nuance, reviews, and
anything accumulating multiple sources. Never Haiku.

**Sonnet footgun-proofing (operator, 2026-07-02):** every Sonnet worker
brief MUST paste the §1 preamble of
`.claude/v3/knowledge/sonnet-worker-guardrails.md` verbatim, and every
Sonnet worker reads that doc before its first tool call. The doc turns
every V3 rule into a mechanical check (no judgment calls) and defines the
STOP+report escalation triggers. Orchestrators: a worker that improvised
past a §5 trigger produced an invalid result — re-dispatch.
