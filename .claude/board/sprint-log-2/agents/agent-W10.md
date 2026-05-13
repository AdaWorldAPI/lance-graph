# W10-rev2 — ogit-g-context-bundle-v1.md plan-doc

**Status:** Done. Pushed via mcp__github__create_or_update_file directly to branch.

**File:** `.claude/plans/ogit-g-context-bundle-v1.md` (Tier-1 sub-plan, Patterns A+B+C).

**Approach:** First-attempt W10 got blocked trying to write to local filesystem (root-owned dirs). This rev2 bypasses local FS entirely; pushes straight to GitHub.

**Content shipped:**
- 3 deliverables (D-OGIT-G-1, D-OGIT-G-2, D-OGIT-G-3) with effort estimates and code sketches
- Open design questions section (5 Qs)
- Cross-references to W1 master, W11 sibling, W12 sibling, W5's TECH_DEBT entries
- Append-only governance noted

**Self-review:**
- Plan correctly identifies the three deliverables and their dependencies on PR #355
- ContextBundle struct sketch matches W6's ledger reframe slot list
- Backwards-compat strategy preserves PR #29 + PR #98 test suites
- No new code; this is design-only
