# 5+3 Council — spec-first hardening for delicate / ambiguous work

> **Operator directive (2026-07-10):** *"for delicate/ambiguous create the
> specs so detailed that a 5+3 council doesn't divert the path; cast the 5
> first, consolidate first, then run the 3, then fix, then consolidate —
> otherwise the 5+3 becomes mushy."*
>
> Prior art: OGAR `.claude/agents/README.md` (the original 5 savants + 3
> brutal reviewers) and this repo's `epiphany-brainstorm-council.md` (panel
> selection). This card is the **sequencing-hardened** synthesis of both —
> the anti-mush protocol is the contribution.

## When to convene (and when NOT to)

Convene the council for **delicate or ambiguous** work only:

- a dedup/refactor that touches ≥3 crates or retires a public taxonomy
- a spec whose wrong resolution silently corrupts downstream sessions
  (canon entries, LE-layout adjacent decisions, classid/mask semantics)
- promoting a CONJECTURE toward canon when the probe alone can't close it

Do NOT convene for: grindwork (one Sonnet worker + guardrails §1),
mechanical probes with pre-registered gates (just run them), anything a
single existing specialist card already owns end-to-end, or anything an
operator ruling has already decided (rulings are frozen; the council
verifies compliance, it never re-litigates).

## The iron sequencing (anti-mush)

```
Phase 0  ORCHESTRATOR writes SPEC v1        ← the real work happens HERE
Phase 1  cast the 5 savants, PARALLEL       ← they verify/harden, never design
Phase 2  CONSOLIDATE FIRST → draft v2       ← orchestrator only; before ANY review
Phase 3  cast the 3 reviewers, PARALLEL     ← they see ONLY draft v2, never raw fan-out
Phase 4  FIX                                 ← apply verdicts; BLOCK = stop or re-spec
Phase 5  CONSOLIDATE → ratified v3 + commit ← board hygiene same-commit
```

**Why the order is load-bearing (the mush failure modes):**

| Skipped step | What goes mushy |
|---|---|
| Spec too thin (Phase 0 rushed) | 8 agents each invent a different architecture; consolidation becomes design-by-averaging |
| Reviewers see raw savant output | 3 reviews of 5 divergent drafts = 15 cross-products of opinion; nothing is attackable |
| 5 and 3 cast together | reviewers attack a moving target; savant findings and review verdicts interleave into soup |
| Fix before consolidate | point-fixes to individual findings reintroduce the conflicts consolidation exists to resolve |
| No final consolidate | the ratified artifact is a thread of patches, not a spec; the next session inherits the mush |

## Phase 0 — the SPEC (the council-can't-divert bar)

The spec is written by the orchestrator (main thread, full depth) BEFORE
any agent is cast. A spec meets the bar when every section below is
present and a savant could answer its questions without making a single
design decision:

1. **FROZEN DECISIONS** — numbered list. Operator rulings, iron rules,
   prior council verdicts that apply. Each cites its source
   (EPIPHANIES id / CLAUDE.md section / plan row). *The council may flag
   a frozen decision only as VIOLATES with file:line evidence — never
   re-open it on taste.*
2. **INPUT INVENTORY** — every file:line the work touches, with shape
   (type/variant counts, consumers, conversions). No "somewhere in
   planner/" — exact paths. If the inventory needs discovery, that
   discovery is a PRE-spec Explore task, not council work.
3. **THE PROPOSED RESOLUTION** — the design, fully committed: target
   shapes, migration steps in order, what is deleted / kept / aliased,
   feature-gate and version-gate treatment (I-LEGACY-API-FEATURE-GATED).
4. **NON-GOALS** — explicitly out of scope, each with one line of why.
5. **PRE-REGISTERED GATES** — pass/fail criteria decided before any
   agent runs (test counts, clippy -D warnings, parity assertions,
   field-isolation matrix where layout is touched).
6. **PER-SAVANT QUESTION SETS** — 3–6 numbered questions per lens,
   answerable YES/NO/VIOLATES-with-evidence. This is what makes Sonnet
   sufficient for the savants: bounded input, fixed output shape.

## Phase 1 — the 5 (research savants, parallel, single lens each)

Default panel (swap lenses per domain; declare swaps in the spec header):

| # | lens | default card / charter | answers |
|---|---|---|---|
| 1 | prior art | `prior-art-savant` | is any part already shipped/named elsewhere? duplicate E-ids? |
| 2 | iron rules | `iron-rule-savant` | YIELDS/VIOLATES per iron rule + AP1-AP9 |
| 3 | code truth | runtime-archaeologist charter (via general-purpose) | is every file:line claim in the spec REAL? CODED vs CLAIMED vs ABSENT |
| 4 | cascade impact | `cascade-impact-savant` | every file/test/doc/board row that must change; mandatory vs follow-up |
| 5 | different views | `creative-explorer-savant` | the strongest alternative reading + the second-order consequence — WITHOUT redesigning |

Casting rules:
- **Model: Sonnet by default.** The detailed spec is precisely what makes
  each lens grindwork-shaped (one bounded input, one fixed output shape).
  Escalate an individual savant to Opus ONLY when its lens is genuinely
  multi-source accumulation (e.g. prior-art across ~100 board/knowledge
  docs). Never haiku (workspace floor).
- Each savant receives: the FULL spec + its question set + the output
  contract below. It does NOT receive the other savants' briefs.
- **Output contract (per savant):** ≤10 findings, each =
  `(question #, verdict from fixed vocab, file:line evidence, ≤2 sentences)`.
  Verdict vocab: `CONFIRMS / VIOLATES / GAP / PRIOR-ART-AT / RISK`.
  No prose essays. No redesigns. A savant that wants to redesign files a
  single `RISK` finding naming the concern and stops.
- Savants are read-only. The orchestrator is the only writer.

## Phase 2 — consolidate FIRST (orchestrator, main thread)

Before any reviewer exists: merge the ≤50 findings into **draft v2** =
spec v1 + a change ledger:

- every `VIOLATES` → spec amended or the frozen decision escalated to
  the operator (never silently overridden)
- every `GAP` → filled with a decision (committed, not optional)
- every `PRIOR-ART-AT` → reuse wired in, duplication removed
- conflicting findings → resolved by the orchestrator with one line of
  why; the losing finding is recorded, not deleted (anti-collapse)
- `RISK` findings → either absorbed as a gate or explicitly accepted

Draft v2 is a single self-contained document. Raw savant output is
banked (scratchpad or AGENT_LOG line) but never forwarded.

## Phase 3 — the 3 (brutal reviewers, parallel, on draft v2 ONLY)

| # | reviewer | charter (from OGAR; run via general-purpose if no local card) | blocks on |
|---|---|---|---|
| 1 | `overclaim-auditor` | every claim's grade matches its evidence; absolute words (`guarantee`/`proven`/`cannot`) only on [G] | grade inflation |
| 2 | `dilution-collapse-sentinel` | no two motifs conflated (dilution); no valid leg deleted because a facet was wrong (collapse) | either |
| 3 | `firewall-warden` | non-negotiables: no German PII labels, no model identifier in artifacts, no hot-path serialization, no prohibited shell, board hygiene same-commit | any hit |

Casting rules:
- **Model: Sonnet default** (draft v2 is self-contained); Opus when the
  review requires independent codebase verification beyond the draft.
- Output contract: per spec section, verdict `PASS / FIX(P1|P2) / BLOCK(P0)`
  + evidence. "Looks good" without naming why each section earns PASS is
  a malformed review — recast.
- Reviewers never see Phase-1 raw output and never talk to each other.

## Phase 4 — FIX (orchestrator)

- Every `BLOCK` → resolve or return to Phase 0 (re-spec); never argue a
  BLOCK away in the commit message.
- Every `FIX` → applied, with the change ledger extended.
- If two reviewers conflict, the stricter verdict wins by default;
  overriding the stricter one requires an operator escalation line.

## Phase 5 — final consolidate + commit

Ratified **v3** = the executable spec. Then: implement (or hand to
workers with guardrails §1), run the pre-registered gates, and land with
board hygiene in the SAME commit (EPIPHANIES if a finding emerged,
STATUS_BOARD row flip, AGENT_LOG entry naming the council run: which 5,
which 3, verdict counts, what changed between v1→v2→v3).

## Token-economy summary

| role | model | why |
|---|---|---|
| Phase 0 spec + Phase 2/4/5 consolidation | main thread (full depth) | accumulation by definition |
| the 5 savants | Sonnet (Opus only for true multi-source lenses) | the detailed spec bounds them into grindwork shape |
| the 3 reviewers | Sonnet on the self-contained draft; Opus if codebase re-verification needed | one input, fixed output |
| pre-spec discovery (inventory) | Explore/Sonnet | pattern matching |

If the whole item is small enough that writing the spec costs more than
doing the work twice — it was not council-grade; do it directly.
