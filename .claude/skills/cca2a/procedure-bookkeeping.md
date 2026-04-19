# Procedure: Prompt↔PR ledger for code awareness in ~90 seconds

Concrete how-to companion to `concepts.md`. Deployed successfully on
2026-04-19: PR #213 (lance-graph, 41 prompts mapped) + PR #110 (ndarray,
25 prompts mapped). Both shipped in ~90 seconds on a single Haiku
subagent, zero code reads, zero MCP.

## Goal

Produce a grep-addressable index pairing every authored prompt file
(brief) with the PR that delivered it. Closes three token-waste
channels simultaneously:

| Channel | Before | After | Discount |
|---|---|---|---|
| Cold-start (once per session) | 20-30 turns | 3-5 turns | ~6× |
| Find-code (per query) | ~25M tokens | ~25 tokens | **10⁷×** |
| Ambient arc knowledge (every turn) | 30-50% of tokens | ~0% | 2×-eternal |

Cost of deployment: **one Haiku subagent, ~90 seconds wall clock,
one PR**. Zero shipping of code.

## Prerequisites (what already exists in every workspace adopting CCA2A)

- `.claude/prompts/*.md` — scoped briefs authored by prior sessions
  (session / certification / probe / handover / research briefs)
- `.claude/*.md` (top-level) — capstones, plans, calibration reports,
  handover logs, integration-plan snapshots
- Working git history (`git log --oneline` reveals PR-merge commits by
  title pattern `Merge pull request #N`)
- `.claude/board/` folder exists with `cat >> file << 'EOF'` append-only
  governance (via `.claude/settings.json` deny rules on Edit/Write)

If any of these are missing, see `concepts.md` § Governance Rules for
setup.

## Three passes

### Pass 1 — Haiku bookkeeper (90 s, mechanical, one-shot)

Spawn a Haiku subagent (or run on the cheapest available tier). Paste
this prompt verbatim:

```
Bookkeeping only. No shipping. No synthesis. No reads beyond:
  ls .claude/prompts/*.md
  ls .claude/*.md
  git log --oneline
  .claude/board/PROMPTS_VS_PRS.md   (to skip pairs already logged)

For each prompt file not yet logged, find the matching PR in git log
by filename keyword or date proximity, and append one line:

  cat >> .claude/board/PROMPTS_VS_PRS.md << 'EOF'
  | YYYY-MM-DD | <prompt file path> | #N <PR title> | merged|none |
  EOF

If no PR matches: `| ... | — | none |`.

`cat >> file << 'EOF'` only. No Edit. No Write. No `>`. No MCP.
Exit when every prompt file has exactly one ledger line.
```

**Output:** one PR containing `.claude/board/PROMPTS_VS_PRS.md` with N
table rows (one per prompt file). Literal filename match — will miss
semantic overlap (that's Pass 2's job).

**Governance:** Haiku is acceptable *here* because the work is
single-source-mechanical with a known output shape. The "never Haiku"
rule (see `concepts.md` § Model policy) targets synthesis; this pass
does zero synthesis.

### Pass 2 — Opus meta-synthesizer (one-shot, ledger-only inputs)

Only reads the ledger + PR arc. Never touches code.

```
Read ONLY:
  .claude/board/PROMPTS_VS_PRS.md
  .claude/board/PR_ARC_INVENTORY.md
Nothing else. No code. No session history replay.

For each line tagged `none` in the ledger, determine whether the
prompt was implicitly resolved by an overlapping PR. Evidence is
subsystem-keyword match in PR titles + arc entries — no semantic
re-derivation from source code.

Append classifications to .claude/board/META_SYNTHESIS.md using
`cat >> file << 'EOF'` only:

  | <prompt file> | superseded by #N[,#M] | <subsystem keyword> |
  | <prompt file> | still open, adjacent to phase <P> | — |
  | <prompt file> | stale, no adjacency | — |

Append-only. Never edit prior rows.
```

**Why Opus here:** synthesis across N ledger rows + arc entries, holding
multiple PR contexts in mind to detect semantic overlap. Grindwork test
fails (see `concepts.md` § Model policy) → Opus mandatory.

**Output:** `.claude/board/META_SYNTHESIS.md` — a superseded/open/stale
classification of all `none` rows from Pass 1.

### Pass 3 — Main thread consumer (sub-second per query, infinite reuse)

From any subsequent session, answer "what's open / what's done / what
about X" in sub-second:

```
grep X .claude/board/PROMPTS_VS_PRS.md          # what shipped?
grep X .claude/board/META_SYNTHESIS.md          # what was superseded?
grep X .claude/phases/integration_phases.md      # what's live?
```

No code reads. ~25 tokens per answer.

## When to re-run

| Trigger | Pass | Cost |
|---|---|---|
| Every merged PR | Pass 1 on the new PR's branch | ~90 s, one commit appending one row |
| Monthly or on drift | Pass 2 to re-annotate | ~2 min, reads two files, appends classifications |
| Never | Retrofit old entries | — (append-only; corrections append as new dated rows) |

## Invariants (never violate)

- **Append-only.** `cat >> file << 'EOF'` is the only write method.
  No `Edit`, no `Write`, no `>` overwrite. Rows are historical record.
- **Ledger-only inputs in Passes 2 and 3.** Never grep code, never
  load prompt-file bodies beyond the filename, never re-read session
  history. If an answer needs code context, the ledger entry points at
  the PR — open the PR diff, don't grep the tree.
- **One line per prompt file.** Pass 1 enforces exactly one ledger
  row per `.claude/prompts/*.md` or top-level doc. Corrections in
  Pass 2 append; the Pass 1 row stays.

## Cross-references

- `concepts.md` — the governance this procedure implements (append-only,
  two-layer A2A, grindwork/accumulation split).
- `divergence.md` — how this pattern diverges from official Claude Code
  conventions.
- `.claude/board/EPIPHANIES.md` — 10⁷× finding (2026-04-19) +
  30-50% ambient-loss finding.
- `.claude/phases/integration_phases.md` § Open Prompts Adjacency —
  the phase-by-phase mapping of open prompts post-Pass-2.
- PR #213 (lance-graph) + PR #110 (ndarray) — first deployments, both
  ~90 s, both one-shot Haiku.
