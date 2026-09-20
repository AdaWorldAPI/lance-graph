# First-hand source law — search is navigation, never evidence

> READ BY: every session, before the first Grep/Glob/Bash-search of a task.
> Enforced (partly) by `.claude/hooks/anti-pattern-matching.sh`; tested by
> `.claude/hooks/tests/anti-pattern-matching.test.sh`.
>
> Operator ruling 2026-09-20, sharpening the 2026-07-21 anti-pattern-matching
> directive (PR #793, issued after code was DELETED having been only
> pattern-matched, never read). That directive said "do not act on a match
> before a full Read". It did not stop the failure that followed, because the
> failure was not about acting — it was about **claiming**.

## 0. The two shapes this exists to kill

```text
SEARCH HIT  ->  snippet read  ->  meaning assumed  ->  architecture asserted

SEARCH = 0  ->  "it does not exist"
```

The second is the worse one, and it is the one no tool reports as an error.

The diagnosis the operator recorded: roughly **20 % tool drift, 80 % weak
evidence gates**. Stacking more shell prohibitions treats the 20 %. This
document disarms search epistemically, which is the 80 %.

> **Premise, attributed and NOT independently verified here:** the operator
> reports that Claude Code's 2026 releases changed the search tooling surface
> (embedded ugrep/bfs behind the Bash path on some native builds), that
> releases around 2026-09-19 fixed cases where Grep/Glob could look like "no
> matches" under large search spaces or resource errors, and that Read now
> carries a page limit (~25k tokens) with partial views. This document does
> not depend on those specifics — every rule below is keyed to what a tool
> RESULT says (zero hits, truncated, partial), never to a version number or a
> constant. That is deliberate: a rule pinned to an unverified constant is a
> rule that stops firing the day the constant moves.

## 1. THE LAW

```text
FIRST-HAND SOURCE LAW

 1. Search is navigation, never evidence.
 2. grep/rg/ugrep/Glob may locate candidate files/symbols only.
 3. sed/head/tail/awk are prohibited for source inspection.
 4. Every load-bearing claim requires a first-hand Read of the defining item.
 5. Caller claims require reading the relevant callers.
 6. Cross-crate claims require reading manifests + both sides of the seam.
 7. Search = 0 proves nothing.
 8. Global negatives require a CLOSED and explicitly named search space.
 9. Truncated/partial/error/capped search automatically triggers deeper
    inspection.
10. Traits/macros/re-exports/generated/feature-gated code automatically
    trigger deeper inspection.
11. Search snippets may never be cited as semantic authority.
12. No architectural ruling may immediately follow a Search operation.
13. Read tests/falsifiers before promoting behaviour into a contract.
14. If the evidence remains incomplete, write UNKNOWN / OPEN.
15. Deletion beats a guessed replacement.
```

### What search may and may not establish

```text
MAY:     "candidate locations are X, Y, Z"

MAY NOT: what a type means
         what a function guarantees
         that a consumer does not exist
         that a mechanism is unused
         that a repository has no implementation
         architectural ownership
         caller semantics
         dependency direction
```

### Rule 12, as the mechanical form of the whole law

> **A search must never be the last tool result before an architectural
> conclusion.**

If the most recent evidence came from a search, at least one Read must follow
before the claim is written. This is the one rule short enough to actually be
obeyed under pressure, and it implies most of the others.

## 2. AUTO-DEEPEN — the session must escalate, not answer

Deepen automatically if ANY of these holds:

```text
 1. search returned zero hits AND you are about to claim absence
 2. the claim contains: none / no consumer / unused / never / only /
    all / every / not implemented
 3. the only basis is a search snippet
 4. the symbol is a trait, macro, generated, re-export, alias, or
    feature-gated
 5. the claim crosses a crate or repository boundary
 6. the search output was truncated, capped, errored, partial, or
    unusually small for the expected search space
 7. several implementations or similarly named symbols exist
 8. the conclusion would change architecture, delete code, mint a
    carrier, or create doctrine
```

### After every relevant hit

```text
READ the complete defining item
READ its enclosing contract / module docs
READ the direct callers / consumers relevant to the claim
READ the relevant tests / falsifiers
   -> only then may a claim be written
```

### The zero-hit protocol

```text
search "foo" -> 0

NOT:  "foo has no consumers"
BUT:  "the search found no candidates"

then: enumerate the relevant source files
      search aliases and re-exports
      search trait-method call sites
      search feature-gated modules
      inspect the manifests
      inspect generated paths where applicable

only with a CLOSED search space:
      "no consumer found in <explicitly named set>"
```

Measured instances of rule 7 failing in this fleet are on the board:
a `CapabilityAuthority` impl written fully-qualified so `impl.*Trait` missed
it; a seam declared "unwired both directions" where the correct grep supported
a false conclusion because the file that explained it was never opened.
**A correct grep can support a false conclusion.**

## 3. PAGING LAW — a partial Read is not evidence for a whole-file claim

```text
PAGING LAW

A partial Read is not evidence for a whole-file claim.

If Read truncates or reports a continuation:
    MUST continue from the returned next offset/page until one of
      a) the complete semantic item is read
      b) the complete relevant section is read
      c) the scope is EXPLICITLY narrowed and the claim narrowed with it

Never skip pages.
Never jump to the tail.
Never infer the unseen middle.
```

And the mechanical guard, simple enough to be followed:

> **No WHOLE-FILE / ALL-CALLERS / NO-CONSUMERS claim may be emitted while any
> relevant Read is marked PARTIAL.**

A huge file does not oblige a full read. If the claim concerns `impl CallMask`,
the obligation is: locate the impl → Read the COMPLETE impl → callers → tests.
Not 200k tokens of surrounding documentation. **Narrow the scope, never the
evidence.**

## 4. The one rule above all the others

> **Context exhaustion must reduce scope, never evidence quality.**

When a session notices "reading all the callers will not fit", the permitted
responses are to SHARD the census or to report PARTIAL / UNKNOWN. The
forbidden response is the cheap one:

```text
grep -> 3 hits -> "probably there are no others"
```

## 5. Delegation and escalation

Volume and ambiguity are different problems with different answers:

```text
too much VOLUME     -> shard mechanically
too much AMBIGUITY  -> escalate intelligence
too little EVIDENCE -> stop
```

### Levels

```text
LEVEL 0  main session      navigation + architectural synthesis
LEVEL 1  Sonnet worker     bounded mechanical census
LEVEL 2  Sonnet shards     large but separable census
LEVEL 3  Opus / main       contradictions, cross-repo seams, architecture
LEVEL 4  stop for operator genuinely underdetermined semantic decision
```

### Auto-delegate when

```text
> ~25k-40k tokens of contiguous relevant material
> ~10-15 relevant files
> 2+ repositories requiring a mechanical census
  an exhaustive caller/test inventory
  repeated same-shaped verification over many files
```

A worker gets a CLOSED evidence task, never an architectural question:

```text
Worker A: enumerate every CallMask producer and consumer. Read each
          defining item and each direct caller. Return file / symbol /
          role / evidence / uncertainty. NO architectural conclusion.
Worker B: enumerate every mask-algebra implementation. Diff operation
          sets and tail semantics. NO recommendation.
Worker C: inspect cargo dependency paths and package identities.
          Return the exact dependency DAG. NO redesign.
```

### A worker must STOP+ESCALATE, never improvise

```text
STATUS: ESCALATE if
  evidence conflicts
  two plausible architectural interpretations remain
  a global negative is required
  cross-repo ownership is unclear
  deleting or minting a type is being considered
  a new semantic contract would be needed
  the docs disagree with the executable code
  caller behaviour cannot be inferred mechanically
  the task exceeds the assigned shard
  more than a couple of unexpected branches appear
```

Report shape:

```text
STATUS: ESCALATE
Observed: ...
Unresolved fork:  A ...  B ...
Evidence needed: ...
Files already read: ...
```

### Escalation is BIDIRECTIONAL

```text
Opus discovers "this is actually 37 mechanical caller reads"
      -> delegate to Sonnet
      -> receive evidence tables
      -> Opus synthesises
```

Otherwise the expensive agent burns its context on grindwork — which is itself
a cause of thin evidence, not merely a cost.

### The flow

```text
                SEARCH
                   |
                   v
             scope identified
                   |
        +----------+-----------+
        |                      |
  small / semantic      large / mechanical
        |                      |
        v                      v
    main / Opus          Sonnet shard(s)
        |                      |
        |                evidence ONLY
        +----------+-----------+
                   v
             synthesis / gate
                   |
              ambiguity?
             no   |    yes
              v   |     v
            land  |  Opus / operator
```

## 6. Enforcement — what is mechanical and what is not

`.claude/hooks/anti-pattern-matching.sh` (`PreToolUse(Grep|Bash)`):

| behaviour | trigger |
|---|---|
| **DENY** | a slicer (`sed`/`head`/`tail`/`awk`) whose argument list names a SOURCE/config file — rule 3 |
| **DENY** | a SEARCH (`grep`/`rg`/`ugrep`/`find`/`fd`/`ls`) piped into a slicer — the cap that hides itself, rule 9 |
| **INJECT** | the Grep tool, and any other search-shaped Bash command — the law summary + the auto-deepen triggers |
| **INJECT** | the destructive-prepend rule (pre-existing, separate law) |
| silent | everything else |

Both DENY branches are disable-verified (remove the branch, the corresponding
rows of the test go from DENY to INJECT); the test is committed and runs in
one second.

### The scope correction, stated because it departs from the instruction

The instruction was *"sed/head/tail komplett verbieten, nicht einmal zum
Reinschauen"*. Implemented as the LAW TEXT's own wording — **prohibited for
source inspection** — and not as a blanket ban, for a measured reason:
`cargo test 2>&1 | tail -30` is REQUIRED elsewhere in this fleet (the
guarded-executor tail-30 output discipline). A deny that fires on every build
command is worked around within the hour and then guards nothing. The line the
hook draws is the one that matches the stated harm: a numeric slice of a
**file** has no semantic boundary; limiting a non-search command's **output**
does not fabricate one.

### What is deliberately NOT hooked, and why

**Read truncation.** The paging law above is prose, not a guard. A
`PostToolUse(Read)` hook could detect a partial read and inject §3 — but it
would have to match whatever marker the Read tool emits, and that marker was
NOT verified in this session. A guard keyed to a guessed string is a guard
that cannot fire, which this workspace forbids on its own terms. Closing it
requires reading one real truncated Read result first; until then the rule is
carried by §3 and by rule 14.

**A `SEARCH_UNRESOLVED` marker** that persists until a deeper Read clears it
was proposed and is not built: hooks here are stateless per invocation, so the
marker needs a session-scoped store that does not exist yet. Recorded so the
next attempt starts from the constraint rather than rediscovering it.

## 7. Anti-patterns, each with the right shape

| anti-pattern | right shape |
|---|---|
| `rg X \| head -20` then "these are the call sites" | Grep tool with `head_limit` (it REPORTS the cap), or an uncapped search plus a named search space |
| `sed -n '1,80p' foo.rs` to "get the gist" | locate the symbol, Read the complete item |
| 0 hits → "no consumer" | "no candidates found"; then close the search space, then claim |
| a snippet quoted as what a type means | Read the defining item and its module docs |
| "reading every caller will not fit" → guess | shard the census, or report PARTIAL |
| worker meets ambiguity → picks one reading | STATUS: ESCALATE with the fork named |
| a wrong claim corrected by a prettier wrong claim | write UNKNOWN / OPEN (rule 14/15) |
