---
name: 5plus3
description: >
  Convene the 5+3 council for delicate/ambiguous work: write a spec so
  detailed the council can't divert the path, cast 5 research savants
  (parallel, Sonnet default), consolidate FIRST into draft v2, then cast
  3 brutal reviewers on v2 only, fix, consolidate to ratified v3, land
  with board hygiene. The strict sequencing is the anti-mush protocol.
  Canonical harness spec: .claude/agents/5plus3-council.md.
---

# /5plus3 — spec-first council for delicate work

Bootload: **read `.claude/agents/5plus3-council.md` in full** — it is the
canonical harness (when to convene, the iron sequencing, the Phase-0 spec
bar, panel defaults, output contracts, token-economy table). This skill
file is only the invocation stub.

Execution checklist (each step gates the next — never overlap phases):

1. **Qualify.** Is this council-grade (≥3-crate refactor / canon-adjacent /
   silently-corrupting-if-wrong)? If not: do it directly or via one
   Sonnet worker with guardrails §1.
2. **Phase 0 — SPEC v1** (main thread): frozen decisions (cited),
   input inventory (file:line — run a pre-spec Explore if unknown),
   the committed resolution, non-goals, pre-registered gates,
   per-savant question sets (3–6 each, YES/NO/VIOLATES-answerable).
3. **Phase 1 — cast the 5** in ONE parallel spawn (Sonnet default).
   Full spec + own question set + output contract (≤10 findings,
   fixed verdict vocab, file:line evidence). No cross-briefing.
4. **Phase 2 — consolidate FIRST** → draft v2 + change ledger.
   No reviewer exists yet. Raw savant output is banked, never forwarded.
5. **Phase 3 — cast the 3** in ONE parallel spawn on draft v2 ONLY
   (overclaim-auditor / dilution-collapse-sentinel / firewall-warden
   charters; PASS / FIX / BLOCK per section).
6. **Phase 4 — fix.** BLOCK → resolve or re-spec; stricter verdict wins.
7. **Phase 5 — ratify v3, implement, run the pre-registered gates,
   commit with board hygiene same-commit** (AGENT_LOG names the run:
   which 5, which 3, verdict counts, v1→v2→v3 deltas).
