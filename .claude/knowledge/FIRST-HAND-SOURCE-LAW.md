# First-hand source law — search finds, read proves

> READ BY: every session before its first search of a task.
>
> Owns exactly two things nothing else does: what may be CLAIMED from which
> operation, and how a human decision is recorded. Everything else is a
> pointer.
>
> Worker mechanics — full-read-before-edit, paging, the report shape, the
> STOP+escalate triggers: `.claude/v3/knowledge/sonnet-worker-guardrails.md`
> §1/§5. Agent tiers and one-writer:
> `.claude/knowledge/tiered-agent-execution-protocol.md`. Not restated here.

## Search is navigation, never evidence

Search, grep, rg, ugrep and Glob establish exactly one thing: **candidate
locations**. Never what a type or function means, architectural ownership,
caller semantics, dependency direction, or a global negative (`unused`,
`no consumers`, `not implemented`, `only`, `all`, `never`).

A load-bearing claim requires reading, first-hand: the defining semantic
item, enough enclosing contract to interpret it, the callers the claim
depends on, and the tests that pin the behaviour. Across a crate or repo
boundary, also the manifests and both sides of the seam.

Incomplete evidence is written `UNKNOWN` / `OPEN` / `PARTIAL`, never an
inferred completion.

> **A search result or snippet may never be the last evidence before an
> architectural conclusion.**

## Slicing: two categories, no exception

```text
EVIDENCE INPUT    source · docs · manifests · tests · plans · contracts
                  · search results used as evidence
                  -> tail/head/sed/awk PROHIBITED, direct or through a pipe

EPHEMERAL OUTPUT  build · test · lint · benchmark · runtime logs
                  -> may be visually limited; truncation is NEVER
                     sufficient failure analysis
```

`tail` is not an evidence tool. A failing command is not understood because
its last lines were read: the root diagnostic is routinely far above the
window, and the final lines are the epilogue (`aborting due to previous
error`, `build failed`, `process exited`). So capture the full output,
locate the FIRST relevant error, read its complete diagnostic block, and
distinguish root cause from cascade.

`grep`/`rg`/`ugrep` are allowed as LOCATORS. What is prohibited is the
epistemic misuse, never the implementation.

## Auto-deepen

Deepen before writing the claim if ANY holds: the search returned zero hits
and an absence claim is contemplated · the statement would use
none/no-consumer/unused/never/only/all/every/not-implemented · the basis is
only a snippet · a trait, macro, re-export, generated or feature-gated item
is involved · the claim crosses a crate or repo boundary · a search or read
result was truncated, partial, capped or errored · several similarly-named
implementations exist · the conclusion would delete code, mint a carrier,
define ownership, change a contract, or become canonical documentation.

Zero results means `no candidates found by this search`, never `it does not
exist`. **A global negative requires an explicitly CLOSED search space** —
name tool, pattern and scope, or phrase it "not found in \<scope\>".

A partial read is not evidence for a whole-file claim: **no WHOLE-FILE /
ALL-CALLERS / NO-CONSUMERS claim while a relevant read is PARTIAL.**

**Context exhaustion must reduce SCOPE, never evidence quality** — shard the
census or report PARTIAL; never a shallower search because context is tight.

## Authority and evidence are different things

```text
HUMAN AUTHORIZATION IS PROVENANCE, NOT VALIDATION.
```

A person chooses direction, scope, policy, naming, acceptable risk.
`the user chose X` never becomes `X is technically true` without independent
evidence. So `operator-ruled` / `operator-pinned` / `operator-locked` /
`operator-confirmed` are not technical status labels in new material.

| state | means |
|---|---|
| `MEASURED` | a command produced this; the command is named |
| `VERIFIED-IN-CODE` | read first-hand at a named location |
| `TEST-PINNED` | a test fails if this stops holding |
| `CURRENT-CONTRACT` | what the shipped types require today |
| `WORKING-MODEL` | in use, not yet falsified |
| `HYPOTHESIS` / `PROPOSED` | stated to be tested / not in force |
| `OPEN` / `DEFERRED` | unresolved, deliberately |
| `SUPERSEDED` | replaced; the replacement is named |
| `REJECTED-BY-FALSIFIER` | a measurement killed it |

A real decision is recorded as `DECISION` / `SCOPE` / `BASIS` /
`REVISIT WHEN` — two fields beside any measurement, never one label.

## Enforcement

Partly mechanical, in `.claude/hooks/anti-pattern-matching.sh` — its header
lists exactly what is DENIED, what is injected, what stays guidance-only, and
the one measured gap still open. Tested by
`.claude/hooks/tests/anti-pattern-matching.test.sh`.
