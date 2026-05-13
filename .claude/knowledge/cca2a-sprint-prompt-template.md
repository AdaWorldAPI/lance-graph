# CCA2A Sprint Prompt Template

> **READ BEFORE WRITING agent prompts** for any sprint that scopes new architectural work.
>
> This template captures the lessons learned from sprint-2 (PR #358/#359) and sprint-3 (PR #360/#361/sprint-3-rescope) about how to prevent the recurring failure modes.

## Failure modes captured to date

1. **"Designing What's Already Built" anti-pattern** (sprint-2 itself; recurred in sprint-3 PR-A-1/PR-C-1/PR-D-1 specs) — agents propose new types/files for substrate that already ships
2. **Wrong-repo error** (W7 sprint-2 → ndarray; W9 sprint-3 → ada-consciousness) — agents defer to `GITHUB_REPO` env var instead of explicit prompt
3. **Race-with-merge** (PR #358/#359, PR #360/#361, this PR) — main thread corrections land after PR merge; need follow-up PR
4. **Pattern letter divergence** (sprint-2 W2 invented A-G that conflicted with W1) — fixed by embedding canonical pattern letters in every prompt

## Substrate-grep checklist (mandatory before any new spec)

Before writing PR-X-Y spec for any pattern X, run:

```python
import os
from github import Github, Auth
tok = os.environ["GITHUB_TOKEN"].strip().strip('"').strip("'")
g = Github(auth=Auth.Token(tok))
repo = g.get_repo("AdaWorldAPI/lance-graph")  # CANONICAL — do not defer to GITHUB_REPO env

# 1. Read tier-0 doc — current pattern X status
tier0 = repo.get_contents(".claude/knowledge/tier-0-pattern-recognition.md", ref="main").decoded_content.decode()
# Find Pattern X section; note SHIPPED / PARTIALLY SHIPPED / DESIGN PHASE status

# 2. Read cross-source matrix — find Pattern X row
cross = repo.get_contents(".claude/knowledge/pattern-recognition-cross-source.md", ref="main").decoded_content.decode()
# Note: shipped substrate file:symbol citations

# 3. Substrate grep — confirm cited substrate exists
for path in cited_substrate_files:
    f = repo.get_contents(path, ref="main")
    # Verify symbols mentioned in tier-0 actually exist in the file

# 4. TD-X check — read corresponding TD row in TECH_DEBT.md
td = repo.get_contents(".claude/board/TECH_DEBT.md", ref="main").decoded_content.decode()
# Find TD-X row; status must align with tier-0

# 5. Recent PR check — has this pattern been touched in the last 30 days?
recent = list(repo.get_commits(since=datetime.now() - timedelta(days=30)))
# Scan for commits mentioning Pattern X or its TD-X
```

If any check reveals shipped substrate, the spec MUST cite it and re-scope to "extension of shipped" rather than "construction from scratch".

## Wrong-repo guardrail (mandatory in every agent prompt)

Every worker agent prompt should include this snippet:

```python
# WRONG-REPO GUARDRAIL: do not defer to GITHUB_REPO env var
import os
from github import Github, Auth
tok = os.environ["GITHUB_TOKEN"].strip().strip('"').strip("'")
g = Github(auth=Auth.Token(tok))
repo = g.get_repo("AdaWorldAPI/lance-graph")  # CANONICAL — explicit, not env-driven
assert repo.full_name == "AdaWorldAPI/lance-graph", \
    f"WRONG REPO: got {repo.full_name}, expected AdaWorldAPI/lance-graph"
```

This 8-line block has prevented zero wrong-repo errors so far (because it didn't exist). Going forward, every spawned agent should have it pre-pended to their write protocol.

## Pattern letter discipline

The canonical pattern letter assignment lives in:
1. `.claude/plans/unified-ogit-architecture-v1.md` (W1 master, sprint-2)
2. `.claude/patterns.md` Pattern Recognition Framework (W3 sprint-2)
3. `.claude/knowledge/tier-0-pattern-recognition.md` (post-PR #359 corrected)

Every worker agent prompt MUST embed the canonical 15-letter table inline. Do not invent new letters; do not re-letter substrate inventory items.

If a worker discovers a genuinely new pattern, they MUST surface it as a meta-1-review escalation, not invent letter P / Q / etc. unilaterally.

## Race-with-merge mitigation

When the main thread (or meta agent) catches a defect AFTER the PR opens but BEFORE merge:
1. Push the fix to the same sprint branch (PR will pick it up on next push)
2. Comment on the PR with the fix summary so reviewer sees it before merge
3. If the merge happens before the fix lands: open a follow-up PR off main (this is what PR #359/#361/this PR does)

## Sprint scaffolding pattern

Every sprint should pre-create:
1. Branch (`claude/<sprint-N>-<scope>`) off main via pygithub
2. SPRINT_LOG-N.md scaffolding at `.claude/board/sprint-log-N/SPRINT_LOG.md` (worker roster, deliverables table, coordination notes, **canonical pattern letters embedded**)
3. Worker prompt template (this template)

Then spawn 12+meta agents in parallel; each agent's prompt includes:
- Sprint context (compact)
- Their specific deliverable (file path, target size, acceptance criteria)
- Wrong-repo guardrail snippet
- Substrate-grep checklist
- Canonical pattern letter table (inline, not by reference)
- pygithub-first protocol with quote-stripped GITHUB_TOKEN

## Cross-references

- `.claude/knowledge/tier-0-pattern-recognition.md` — canonical pattern letters + status
- `.claude/knowledge/pattern-recognition-cross-source.md` — A-O ↔ Pillars ↔ .grok/ ↔ shipped substrate matrix
- `.claude/board/sprint-log-2/sprint-summary.md` — sprint-2 lessons (W2 invented pattern letters, W7 wrong-repo)
- `.claude/board/sprint-log-3/sprint-summary.md` — sprint-3 lessons (W9 wrong-repo, recurring; specs over-scoped)
- `.claude/patterns.md` Pattern Recognition Framework + Anti-Pattern subsection

## Provenance

Captured post-#360 review pass. Reviewer flagged:
> "Suggested guardrail for the next sprint: bake into every worker prompt a first-action verify step — `git remote -v` (if local) or `mcp__github__get_me` + repo env confirmation before any write. Cost: ~10 LoC per prompt. Saves 1 wrong-repo-rev per sprint."

This template formalizes that guardrail + extends to substrate-grep + pattern-letter discipline.
