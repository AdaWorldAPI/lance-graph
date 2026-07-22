READ BY: any session orchestrating subagent fleets in this workspace. PROPOSES a narrow exception to the Model Policy's blanket "NEVER haiku" line for ONE role (guarded executor); the exception is ratified if and only if the operator merges the PR carrying this doc — the merge is the authorization, not this text.

> **Provenance (operator request, in-session 2026-07-22, quoted verbatim):**
> "can you use opus agents for filigrane and sonnet 5 agents for grindwork
> and write down a complete set of plans, then PR, then I merge and switch
> to Opus medium and continue with Sonnet 5 agents, or even haiku if the
> agents are guardrailed enough to execute bash style tasks with clear
> start stop and append agent logs to be read by the supervising higher
> order agents / the idea is that running compilation with Haiku might
> drain tools token usage slower"
>
> The haiku permission is CONDITIONAL ("if the agents are guardrailed
> enough") — the contract below IS the guardrail that the condition
> requires. Absent the contract, the blanket "NEVER haiku" line stands.

# Tiered agent execution — Opus filigree / Sonnet grindwork / Haiku guarded execution

## The three tiers

| tier | role | may | must never |
|---|---|---|---|
| Opus (or main thread) | FILIGREE: synthesis, plan authorship, adjudication of probe results vs gates, board rulings, review | read everything; write plans/knowledge/board | be used for mechanical loops |
| Sonnet | GRINDWORK: write-file-from-spec, scoped edits, drafting from a complete brief | edit code/docs per brief | synthesize across sources; decide gates; touch board files unless brief says so |
| Haiku | GUARDED EXECUTION ONLY: run a pre-written bash sequence with clear START/STOP, capture output, append a log entry | run the exact commands in its brief; retry per an explicit retry table | write or edit ANY source file; interpret ambiguous failures; deviate from the command list; run commands not in the brief |

## Why Haiku for execution

Compilation/test loops drain token budget mostly through TOOL OUTPUT tokens,
not reasoning. A Haiku executor burns the cheapest tokens on the
highest-volume output. The guardrails make the role safe: Haiku never
authors, never decides — it executes and reports.

## The Haiku executor guardrail contract (paste into every Haiku brief verbatim)

1. You will run EXACTLY the numbered commands below, in order. No additions,
   no substitutions, no flags changed.
2. STOP conditions: any command exits non-zero AND is not covered by the
   retry table → STOP immediately, do not attempt fixes, write your log entry
   with status=BLOCKED and the last 30 lines of output.
3. Retry table: network-flavored git/curl failures → up to 3 retries with
   2s/4s/8s backoff. Nothing else retries.
4. Output discipline: capture only the LAST 30 lines of each command
   (`| tail -30`); never dump full build logs into your reply.
5. Log append (MANDATORY, your final act): append (never overwrite) one
   entry to `.claude/board/AGENT_LOG.md` in the format below. Your reply to
   the orchestrator is a copy of that entry, nothing more.
6. You are forbidden from: editing any file other than the log; running
   `git commit`/`git push` unless they appear in your numbered list;
   interpreting WHY something failed (that is the supervisor's job).

## The log entry format (append-only; read by supervising agents)

```
### <UTC timestamp> — haiku-exec — <task-slug>
- commands: <n>/<total> completed
- status: GREEN | BLOCKED@cmd<k>
- gates: <one line per gate command: name → PASS/FAIL/SKIPPED>
- tail: <last ≤10 lines of the failing command, only if BLOCKED>
```

## Supervision loop

- The Opus/main supervisor writes the numbered command list (the "execution
  card"), spawns the Haiku executor, and reads ONLY the log entry — never the
  raw transcript.
- A BLOCKED entry escalates to a Sonnet fix-agent (with the log tail as
  brief) or to the supervisor; Haiku is never asked to fix.
- Multiple Haiku executors may run in parallel ONLY on disjoint
  crates/directories (the shared-target cargo-hygiene rule applies: prefer
  one executor per shared target/ to avoid cold-compile duplication).

## Session-tier switching (the operator's token-economy pattern)

Capture phase: Opus main thread + Opus filigree agents write the plan set →
PR → operator merges. Execution phase: main thread drops to Opus-medium;
Sonnet agents implement from the plans; Haiku executors run the gates. The
plans must therefore be COMPLETE (a Sonnet implementer never needs session
memory — the plan is the memory).

## Proposed amendment to the Model Policy (ratified by operator merge)

The CLAUDE.md Model Policy line "NEVER haiku for any subagent" gains ONE
narrow exception once the operator merges the PR carrying this doc: Haiku is
permitted for the GUARDED EXECUTOR role ONLY, under the contract above,
which satisfies the operator's stated condition ("if the agents are
guardrailed enough"). Haiku remains forbidden for synthesis, drafting,
review, and any file edit. This doc is the canonical statement; cite it when
spawning. If a future session finds this doc on a branch whose PR was NOT
merged, the exception is not in force.
