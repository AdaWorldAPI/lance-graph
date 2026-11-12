# 2026-11-12 — D-SCHEMA-WARDROBE-P0: the resident-write chain exists in parts, not as one wired path

**Kind:** audit + falsifier suite. No production code changed except one `cfg(doctest)` module and one dev-dependency feature.
**Grade:** VERIFIED-IN-CODE for the call graph (every cited site read at its line);
MEASURED for the suite (`cargo test -p lance-graph-supervisor --features cycle-driver --test ownership_preservation`: 9 pass;
`cargo test -p cognitive-shader-driver --doc ownership_laws`: 7 pass; 4 of 4 disable runs red).
**Plan:** `.claude/plans/schema-wardrobe-ownership-audit-v1.md`. **Linear (existing):** EXO-253.

- `MailboxSoA::cast_on_behalf` stamps the owner's own id, and has no caller outside tests.
- `BatchWriter::cast` is public and takes a plain mailbox id; a payload-only cast under a foreign id is not refused anywhere. A cross-owner MOVE is refused in `persist_cycle`.
- The only non-test submitter to the background `CycleWriter` is the DR map log, which builds its slots itself and does not pass `MailboxSoA` or `BatchWriter`.
- The wired seal path carries owned payload bytes and copies them twice before the Lance append (retry copy, freeze). The descriptor route exists in examples and tests only.
- Three knowledge documents under `.claude/v3/knowledge/` describe callers and a sink that do not match the code; listed in the plan §5, not edited here.

**Agent runs (grindwork tier):** four read-only runs (two fact maps, one call-site census, one quote harvest) and one drafting run for the test file. Checked by reading: `cast_on_behalf` at its stated lines, the four non-test cast sites, `seal_cycle`, `persist_cycle`, `try_submit`; the drafted test file read in full before compiling. Agent items wrong: 0. One orchestrator error caught before commit: a truncated search made `cognitive_pass` look absent; reading `cycle_driver.rs` showed it exists and calls the adapter with a fleet-resolved owner id, and the audit was corrected. One map run stopped on a missing brief field and was re-dispatched.

**Open:** the `on_behalf` enforcement seam; descriptors in `collect_casts`; the owner of a resident codegraph population (BLOCKED); the stale documents.
