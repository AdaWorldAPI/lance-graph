# Never open a file for writing in the same expression that still reads it

> READ BY: every session and worker that writes any board, ledger, or
> append-only file (`.claude/board/*`, `EPIPHANIES.md`, `PR_ARC_INVENTORY.md`,
> plans, census docs) — and any executor running a scripted prepend.

## Status: CONTRACT (locked by operator directive, 2026-08-30)

Operator-directed prohibition of the truncating write pattern, issued after
it destroyed 5876 lines of `PR_ARC_INVENTORY.md`. Operator words and
phrasings are not reproduced in committed artifacts; the directive's
authority is the lock above, not a quotation.

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

## Recurrence — 2026-09-02, EPIPHANIES.md, 25,172 -> 61 lines

The rule fired again, in a session that had this file's own prohibition in its
context. A board-hygiene pass prepended a new entry with
`open(p,'w').write(E + open(p).read())` -- the exact prohibited shape --
inside a Python heredoc that ALSO carried two CORRECT prepends
(`b = open(p).read()` first, then `open(p,'w')`) for LATEST_STATE and
PR_ARC_INVENTORY. Writing the safe form twice in the same script did not
prevent writing the unsafe form once.

What caught it: the mandatory post-write `wc -l` comparison against
`origin/main`, printed for every touched board file in the same command. The
line `EPIPHANIES.md main=25172 now=61` was unmissable, and restoration was a
`git checkout` plus a re-prepend, because the destruction happened in the
working tree and was never committed.

Two lessons this recurrence adds:

1. **The guard has to be in the same breath as the write.** The rule as
   stated is a prohibition an author must remember; the `wc -l` check is a
   detector that runs whether or not they remembered. Print the before/after
   line counts in the SAME command that writes, every time, and never commit a
   board pass without reading that output.
2. **Mixed-safety scripts are the dangerous shape.** A heredoc containing
   several prepends is where this hides: the correct ones make the script
   look reviewed. Prefer one helper used for every prepend in a pass over
   three hand-written ones.

## Cross-reference / retrieval footer

- Restore PR: lance-graph **#1082** (merge `82679c3a`); prohibition PR **#1083**
  (merge `352005d3`); this file's Goldstandard-format landing: its own PR.
- Board entry: `EPIPHANIES.md` `E-DESTRUCTIVE-PREPEND-TRUNCATES-BEFORE-READ-1`
  (2026-08-30).
- Census trap 10: `docs/architecture/COGNITIVE-FABRIC-CENSUS-2026-08-30.md` §8.3.
- CLAUDE.md § In-Session Orchestration Discipline carries the short pointer.
