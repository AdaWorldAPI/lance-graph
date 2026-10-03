---
name: coresearch
description: >
  Convene the co-research council for an OPEN question where outside
  knowledge may change the answer: 5 scouts in two rings (inside: code
  cartographer + internal prior art; outside: arXiv literature, known
  systems such as DuckDB / Odoo, ontologies and concepts), a crosswalk of
  outside ideas onto inside surfaces, then 3 co-architects (bridge,
  firewall/fit, falsifier) and an exploration map with ADOPT-NOW / PROBE /
  PARK / SKIP per idea. Ratifies nothing; a chosen design goes to a plan or
  /5plus3. Canonical harness: .claude/agents/coresearch-council.md.
---

# /coresearch — explore the code and the outside world together

Bootload: **read `.claude/agents/coresearch-council.md` in full.** It is the
canonical harness. This file is only the invocation stub.

Checklist (each step gates the next):

1. **Qualify.** Is the question open, and could outside knowledge change the
   answer? If it is decided, run nothing. If it is a committed design, use
   `/5plus3`. If two sources answer it, read them.
2. **Phase 0 — QUESTION BRIEF** (main thread): the question, anchors
   (cited, never re-opened), the internal surface by path, external
   domains with seed terms and the knowledge docs that already cover them,
   what would count as an answer, and the budget.
3. **Phase 1 — cast the 5 scouts** in ONE parallel spawn: code
   cartographer, internal prior art, literature, systems, concepts and
   ontology. Each item carries a source id and a read grade
   (`READ-IN-FULL` / `SECTION-READ` / `ABSTRACT-ONLY` / `SECONDHAND`).
4. **Phase 2 — CROSSWALK** (main thread): one row per idea, with the
   relation `ALREADY-HAVE` / `PARTIAL` / `NEW` / `CONFLICTS-ANCHOR`. Raw
   scout output is banked, never forwarded.
5. **Phase 3 — cast the 3 co-architects** in ONE parallel spawn, on the
   crosswalk only: the bridge architect, the firewall and fit critic, and
   the falsifier designer.
6. **Phase 4 — EXPLORATION MAP**: ADOPT-NOW / PROBE / PARK / SKIP per idea,
   the named probes, and what was NOT searched.
7. **Phase 5 — land** one `entries/YYYY-MM-DD-coresearch-<topic>.md` plus
   `entries_index.py --write`. Then ask the operator which ideas to take
   forward. Nothing is adopted by the council itself.
