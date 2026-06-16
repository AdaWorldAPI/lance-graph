# 2026-06-15 20:50 UTC — Autoattended wave kickoff: OpenProject AR-shape extraction

> **From:** orchestrator (main thread, claude-opus-4-8[1m])
> **To:** 4-savant ensemble (dto-soa-savant, prior-art-savant, truth-architect, integration-lead)
> **Plan:** `.claude/plans/openproject-ar-shape-extraction-v1.md`
> **Branch:** `claude/openproject-ar-shape-extraction-v1` (lance-graph)
> **Round:** 1 — research, parallel
>
> **APPEND-ONLY.** Each savant prepends an `AGENT_LOG.md` entry on return.

## What I did

1. **Measured the 100 %-coverage surface** of `OpenProject/app/models/` →
   78 distinct class-body DSL names across 941 files / 1696 declarations.
   Census saved at `/tmp/cov-repro/openproject-78-name-surface.txt`.
   Methodology: `grep -hE '^  [a-z_]'` on every `.rb`, suppress Ruby
   keywords / scope markers, `sort | uniq -c | sort -rn`.

2. **Classified all 78 names** into 67 emit categories + 11 scope markers
   (`private`, `protected`, `class`, `module`, `self`, `class_attribute`,
   `private_class_method`, `private_constant`, `module_function`, plus `def`/`end`).
   Plan §2 has the full classification table with predicate names.

3. **Identified the 22 new predicates** the closed vocab needs (existing
   set: 7; planned additions: 22; coverage of OpenProject = 100 %).

4. **Wrote the plan** with 7 deliverables D-AR-{1..7}, 3-round sequencing,
   and the 4-savant brief below.

## FINDING

- `ruff_ruby_spo` is the SCAFFOLD that this plan completes (per its own
  lib header: *"Point `extract` at an OpenProject `app/models/` tree"*).
- Coverage gap measured: current `RubyClass.associations: Vec<String>` =
  ~21 % of declarations (351 / 1696); drops 305 nested association
  options to flat names.
- The OpenProject corpus has **only 78 distinct class-body DSL names**.
  That is bounded and exhaustively enumerable — "100 %" is a real number
  on this corpus, not a hand-wave.

## CONJECTURE (savants confirm or correct)

- `has_dsl_call{name, args}` is the right catch-all for OpenProject custom
  registrations (`register_journal_formatter`, `activity_provider_for`,
  `deprecated_alias`, `associated_to_ask_before_destruction`,
  `has_details_table`). Alternative: give each its own predicate. Argued
  against in §2 (loses bulk + couples the closed vocab to OpenProject), but
  the council may overrule — dto-soa-savant decides.
- `Provenance::OpenProjectExtracted{file, line}` is the right new variant.
  Truth (f, c) defaults: unknown — truth-architect calibrates.

## Blockers

None for Round 1 — every input the savants need is local (plan §1
references, /tmp/cov-repro/openproject-78-name-surface.txt, AdaWorldAPI/ruff
zipball at /tmp/sources/AdaWorldAPI-ruff-5179bc0/).

## Open questions for the savants

1. **dto-soa-savant:** does any of the 22 proposed predicates collide
   semantically with the 7 existing ones? Specifically — does
   `has_callback` overlap `emitted_by`? Does `has_dsl_call` overlap
   `has_function`? Naming review.
2. **prior-art-savant:** is the §2 67-emit / 11-scope split implementable
   purely as additions to `RubyClass` and `Predicate`, or does it need
   new traits / types elsewhere in `ruff_spo_triplet`? Flag any drift.
3. **truth-architect:** propose `(f, c)` defaults for
   `Provenance::OpenProjectExtracted`. Compare to `Extracted` (Odoo
   Python) and `Aerial+::Mined`. Cite `I-NOISE-FLOOR-JIRAK` per the
   iron-rule annotation policy.
4. **integration-lead:** is the §5 sequencing correct? Should D-AR-1
   (predicates) and D-AR-2 (IR) land in the same ruff PR, or split? Does
   `op-surreal-ast` need to wait for D-AR-4 (coverage proof), or can it
   skeleton against the existing 7 predicates as a contract-only stub?

## What "Round 2" looks like after consolidation

Main thread reads the 4 savant responses, prepends a consolidation entry
to `AGENT_LOG.md`, then either (a) opens the D-AR-1 PR on `AdaWorldAPI/ruff`
with the agreed predicate enum, or (b) loops Round 1 with sharper questions
if any savant flags a blocking concern.
