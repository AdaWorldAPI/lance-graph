# First-hand source law — search finds, read proves

> READ BY: every session before its first search of a task; every
> orchestrator writing a worker brief.
>
> Scope: what may be CLAIMED from which operation, and how a human decision
> is recorded. It deliberately owns nothing else.
>
> Not duplicated here: the agent tiers and the one-writer rule
> (`tiered-agent-execution-protocol.md`), the worker iron rules and
> STOP+escalate triggers (`.claude/v3/knowledge/sonnet-worker-guardrails.md`
> §1/§5). This file is cited from both; it does not restate them.

## A. Search is navigation, never evidence

Search, grep, rg, ugrep and Glob may establish exactly one thing:

```text
"candidate locations are X, Y, Z"
```

They may NOT establish what a type or function means, architectural
ownership, caller semantics, dependency direction, or any global negative —
`unused`, `no consumers`, `not implemented`, `only`, `all`, `never`.

A load-bearing claim requires first-hand reading of:

```text
the defining semantic item
enough enclosing contract/module context to interpret it
the direct callers/consumers the claim depends on
the tests/falsifiers that pin the behaviour
```

A cross-crate or cross-repo claim additionally requires the manifests
(dependency direction) and BOTH sides of the seam.

If the evidence stays incomplete, the answer is `UNKNOWN` / `OPEN` /
`PARTIAL` — never an inferred completion.

### The one mechanical form of the whole rule

> **A search result or snippet may never be the last evidence before an
> architectural conclusion.**

```text
FORBIDDEN:  search output          -> conclusion
REQUIRED:   search -> locate -> READ the semantic unit
                   -> caller/test census where relevant -> conclusion
```

## B. Source inspection

For understanding source, `sed` / `head` / `tail` / `awk` are prohibited:
they produce an arbitrary textual slice rather than a semantic unit, so the
cut can fall anywhere — before the decisive `impl`, between a definition and
its invariant.

`grep` / `rg` / `ugrep` are allowed as LOCATORS. What is prohibited is the
epistemic misuse, never the implementation: a search tool is not suspect
because some build routes it through Bash or ugrep internally.

Limiting a NON-search command's output (`cargo test 2>&1 | tail -30`) is not
source inspection and stays allowed.

## C. Auto-deepen

Deepen before writing the claim if ANY of these holds:

```text
 1. the search returned zero hits and an absence claim is contemplated
 2. the statement would use: none / no consumer / unused / never /
    only / all / every / not implemented
 3. the basis is only a search snippet
 4. a trait, macro, re-export, generated or feature-gated item is involved
 5. the claim crosses a crate or repo boundary
 6. a search or read result was truncated, partial, capped, or errored
 7. several similarly-named implementations exist
 8. the conclusion would delete code, mint a carrier, define ownership,
    change a contract, or become canonical documentation
```

A zero-result search means `no candidates found by this search`, never `the
thing does not exist`. **A global negative requires an explicitly CLOSED
search space** — name the tool, the pattern and the scope, or phrase the
claim as "not found in \<scope\>".

## D. Paging

A partial Read is not evidence for a whole-file or whole-section claim.

When the read surface reports continuation or truncation, continue from the
exact next offset/page until the relevant semantic item or section is
complete. Never first-page-plus-last-page and infer the middle.

Read semantic units, not byte ranges: a 200k-token file is not an obligation
when one complete `impl` is what the claim rests on.

```text
HARD GUARD: no WHOLE-FILE / ALL-CALLERS / NO-CONSUMERS claim
            while a relevant read remains PARTIAL.
```

## E. Context exhaustion

> **Context exhaustion must reduce scope, never evidence quality.**

When the required evidence does not fit the session, the two permitted
moves are to SHARD the census or to report `PARTIAL` / `UNKNOWN`. Never
substitute a shallower search for required first-hand reading because
context is getting tight.

## F. Delegation and escalation

Volume and ambiguity are different problems:

```text
large VOLUME      -> shard to grindwork workers
high AMBIGUITY    -> the main/strong agent
insufficient EVIDENCE -> STOP
```

Delegate mechanical work automatically: an exhaustive caller census, 10+
relevant files, ~25k-40k+ tokens of contiguous relevant material, repeated
same-shaped checks, multi-repo manifest inventory, a large test/falsifier
inventory.

A worker assignment is a CLOSED evidence task, never an architectural
question:

```text
"Enumerate every CallMask producer/consumer. Read each relevant item.
 Return file/symbol/role/evidence/uncertainty.
 Make no architecture recommendation."
```

Worker return shape:

```text
STATUS: DONE | PARTIAL | ESCALATE
Observed:
Evidence:
Unresolved:
Files/semantic items read:
Search space closed?  yes/no
```

The tiers themselves and the STOP+escalate trigger list are owned by the two
files named in the header; a worker follows those, not a second copy here.

## G. Authority and evidence are different things

```text
HUMAN AUTHORIZATION IS PROVENANCE, NOT VALIDATION.
```

A user may choose direction, scope, policy, naming, and acceptable risk.
`the user chose X` must never become `X is technically true` without
independent evidence.

```text
human decision  !=  evidence
                !=  technical truth
                !=  correctness
                !=  safety
```

So in NEW canonical material these are not technical status labels and are
not used as one: `operator-ruled`, `operator-pinned`, `operator-locked`,
`operator-confirmed`.

Use an evidence-bearing state instead:

| state | means |
|---|---|
| `MEASURED` | a command produced this number; the command is named |
| `VERIFIED-IN-CODE` | read first-hand at a named location |
| `TEST-PINNED` | a test fails if this stops holding |
| `CURRENT-CONTRACT` | what the shipped types/signatures require today |
| `WORKING-MODEL` | in use, not yet falsified either way |
| `HYPOTHESIS` | stated so it can be tested |
| `PROPOSED` | not in force |
| `OPEN` / `DEFERRED` | unresolved, deliberately |
| `SUPERSEDED` | replaced; the replacement is named |
| `REJECTED-BY-FALSIFIER` | a measurement killed it |

A real user decision is recorded as a decision, not as a proof:

```text
DECISION:     what was chosen
SCOPE:        where it applies
BASIS:        preference / risk tolerance / policy / cost — or the evidence
REVISIT WHEN: the condition that would reopen it
```

A decision and a measurement may coexist on one item; they are two fields,
never one label.

## Enforcement

Mechanical, in `.claude/hooks/anti-pattern-matching.sh`
(`PreToolUse(Grep|Bash|Edit|Write)`), tested by
`.claude/hooks/tests/anti-pattern-matching.test.sh`:

| | |
|---|---|
| DENY | a slicer (`sed`/`head`/`tail`/`awk`) whose argument list names a source/config file — §B |
| DENY | a search (`grep`/`rg`/`ugrep`/`find`/`fd`/`ls`) piped into a slicer — §C.6, the cap that hides itself |
| DENY | an edit that INTRODUCES one of the four authority labels — §G |
| INJECT | the law's summary + the §C triggers, on the Grep tool and on any search-shaped Bash command |

Guidance-only, because no regex decides it: §A's "enough enclosing context",
§D's paging, §E's shard-or-report choice, §F's ambiguity judgement, and
whether a chosen state label in §G is the right one.
