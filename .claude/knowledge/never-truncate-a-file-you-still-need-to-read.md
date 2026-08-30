# Never open a file for writing in the same expression that still reads it

> READ BY: every session and worker that writes any board, ledger, or
> append-only file (`.claude/board/*`, `EPIPHANIES.md`, `PR_ARC_INVENTORY.md`,
> plans, census docs) — and any executor running a scripted prepend.

## Status: CONTRACT (locked by operator directive, 2026-08-30)

Operator-directed prohibition of the faulty one-liner, issued after the
pattern destroyed 5876 lines of `PR_ARC_INVENTORY.md`. Operator words are not
reproduced in committed artifacts; the directive's authority is the lock
above, not a quotation.

## The One-Line Rule

A file opened for writing/truncation may not be READ anywhere in the same
expression, statement, or shell pipeline — prepend is read-into-variable →
compose-in-memory → write, or the `Edit` tool (read-anchored by construction).

## The incident (evidence class: observed)

The #1081 board-hygiene commit ran:

```python
open(p, "w").write(entry + "\n" + open(p).read())   # PROHIBITED
```

Python evaluates `open(p, "w")` — truncating the file to zero bytes — before
the argument's `open(p).read()` runs. The read-back returned `""`; the
append-only ledger collapsed **5876 → 32 lines** on main, leaving a dangling
#1079 self-reference as the visible wound. Restored same day in #1082
(merge `82679c3a`), byte-identical from the parent blob `cdf8c15a^1`.

## Consequences (all load-bearing)

1. Prohibited in every language and shape: `sort f > f` (shell truncates `f`
   before `sort` reads it), any Python expression combining `open(p, "w")`
   with a read of `p`, any equivalent. Do not rely on remembering which
   language evaluates arguments first — the rule is shape-based, not
   language-based.
2. Safe prepend is three steps: `body = read(p)` → compose → `write(p)`.
   For board files the default is the `Edit` tool anyway.
3. This is the sharper edge of the existing "Read before Write, always" P0:
   read-before-write applies WITHIN a single expression's evaluation order,
   not just across tool calls.

## Falsifier (machine-runnable, mandatory after every ledger write)

```bash
wc -l <file>   # compare against the pre-write count
```

**An append-only file that got SHORTER is always a defect** — no exceptions.
Supports: catching any truncation immediately. Does **not** prove: content
correctness or ordering — a same-length corruption passes this check.

A non-blocking PreToolUse guard (`.claude/hooks/anti-pattern-matching.sh`)
injects this rule when a Bash command matches the write-while-reading shape.

## Cross-reference / retrieval footer

- Restore PR: lance-graph **#1082** (merge `82679c3a`); prohibition PR **#1083**
  (merge `352005d3`); this file's Goldstandard-format landing: its own PR.
- Board entry: `EPIPHANIES.md` `E-DESTRUCTIVE-PREPEND-TRUNCATES-BEFORE-READ-1`
  (2026-08-30).
- Census trap 10: `docs/architecture/COGNITIVE-FABRIC-CENSUS-2026-08-30.md` §8.3.
- CLAUDE.md § In-Session Orchestration Discipline carries the short pointer.
