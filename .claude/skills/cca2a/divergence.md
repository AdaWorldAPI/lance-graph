# CCA2A — Divergence from Official Claude Code

Audit of what CCA2A does vs what official Anthropic Claude Code
conventions prescribe. Per the 2026-04-19 audit of
`docs.claude.com/en/docs/claude-code/`.

## Aligned with official

| Area | Convention |
|---|---|
| Agent cards location | `.claude/agents/<name>.md` with YAML frontmatter |
| Frontmatter fields | `name`, `description`, `tools`, `model` |
| Settings files | `.claude/settings.json` (project, committed) + `.claude/settings.local.json` (local, gitignored) |
| Model values | `sonnet`, `opus` (CCA2A excludes `haiku` by policy) |
| CLAUDE.md location | Project root or `.claude/CLAUDE.md` |
| Permissions arrays | `permissions.ask`, `permissions.allow`, `permissions.deny` |
| Hook events | `SessionStart`, `PostCompact`, etc. all official |

## Extensions (invented by CCA2A, not in official docs)

| Feature | What it adds | Why |
|---|---|---|
| `.claude/BOOT.md` | Single page entry point referenced from CLAUDE.md | Explicit bootstrap contract; official doesn't have this concept |
| `.claude/knowledge/LATEST_STATE.md` | Current-state snapshot, updated after every PR | Prevents re-proposing shipped work |
| `.claude/knowledge/PR_ARC_INVENTORY.md` | APPEND-ONLY decision history per PR | Locks architectural history against rewrite-drift |
| `.claude/handovers/*.md` | Agent-to-agent state transfer files | Official has subagents but no multi-step chain handover protocol |
| Knowledge Activation trigger table | Domain → agent → knowledge docs mapping | Extends `READ BY:` headers into an orchestration table |
| Grindwork vs accumulation model split | Per-task model selection rule | Official allows `model: sonnet` but doesn't codify the grindwork/accumulation split |
| Zipball-for-reads policy | Use curl zipball + local grep for 3+ cross-repo reads | Official doesn't cover cross-repo read cost optimization |

These extensions do not conflict with official conventions. They
layer on top as workspace-specific conventions. A Claude Code
session without CCA2A installed still works with official
conventions; CCA2A just accelerates cold-start.

## Recommended adoptions from official (future work)

### 1. `.claude/rules/` with `paths:` frontmatter

Official path-scoped knowledge loading. CCA2A's READ BY headers
pattern could be replaced or complemented by:

```markdown
---
paths: ["src/billing/**"]
---
Load this rule when Claude works in src/billing/.
```

Deterministic (directory match) vs pattern-matched intent. Consider
migrating domain-specific knowledge docs into `.claude/rules/`
with path scopes where the mapping is literal.

### 2. `SessionStart` hook with `matcher: startup`

CCA2A already includes this. Official doc recommends it for
session-start context injection. Covered by the install.

### 3. `PostCompact` hook

CCA2A includes this. Critical — compaction otherwise drops the
workspace bootload, and the next turn rediscovers. Official doc
recommends this for compaction-resilient sessions.

### 4. `Skill` fork + `agent:` field

Skills can fork into named subagents via `agent: Explore` or similar
for read-only isolation. CCA2A doesn't use this yet; could be added
for pure-search skill variants (e.g. a read-only CCA2A audit skill
that reports divergence without installing).

### 5. Auto memory

`~/.claude/projects/<project>/memory/MEMORY.md` — Claude writes
automatically. CCA2A's LATEST_STATE.md is the curated alternative;
auto memory could be an unstructured addition for personal notes.

## Drift to avoid

- **Invented frontmatter fields.** Stick to the official schema for
  `.claude/agents/*.md` frontmatter. Don't add workspace-specific
  fields; put conventions in the prose body instead.
- **Custom hook JSON shapes.** Return only the fields official docs
  support (`hookSpecificOutput.additionalContext`,
  `hookSpecificOutput.hookEventName`, etc.).
- **Skill max frontmatter size.** 1024 chars. Keep `description`
  short; offload detail to subpages referenced by the skill.

## Compatibility

CCA2A installs work alongside any existing `.claude/` setup:
- Doesn't replace existing agent cards.
- Doesn't clobber existing `settings.json` rules (merges).
- Doesn't overwrite existing CLAUDE.md (prepends a new section if
  absent, skips if a Session Start section already exists).

A workspace can adopt CCA2A partially: install only the knowledge
docs + governance rules without the handover protocol, or add
handovers later without reinstalling anything.
