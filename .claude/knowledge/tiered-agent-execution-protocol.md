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
>
> **Two halves of "guardrailed enough" (operator clarification, same
> session):** the guardrail is (1) *behavioural* — the executor never
> authors, decides, or edits any file but the log (§ contract items 1-6);
> AND (2) *resource* — the executor never creates **exponential build
> residue** (§ contract item 7 + § No exponential target residue). A fleet
> that each cold-compiles its own `target/` is NOT guardrailed even if every
> executor is behaviourally perfect: N executors × a ~7 GB `target/` in N
> worktrees is the exact 12×-residue failure `.../ndarray/.claude/rules/agent-cargo-hygiene.md`
> exists to prevent, and on this remote environment's **fixed per-session
> disk allowance** it converts "compile the workspace" into "fill the disk
> and fail every write." Cheap tokens are worthless if the disk is gone.

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
4. Output discipline: capture only the LAST 30 lines of each command, but
   NEVER let `tail` mask the command's exit status — `cmd | tail -30`
   reports `tail`'s success even when `cmd` failed. Run each command under
   `set -o pipefail` (or read `${PIPESTATUS[0]}` before evaluating the
   retry/STOP rule); the STOP condition (item 2) tests the PRODUCER's status,
   not the pipeline's. Never dump full build logs into your reply.
5. Run-record (MANDATORY, your final act): write your terse run-record to
   your OWN per-run file `.claude/board/exec-runs/<task-slug>.txt`
   (create the dir if absent) in the format below — one executor, one file,
   so parallel executors never contend on a shared sink. You do NOT touch
   `.claude/board/AGENT_LOG.md`: that canonical board file has a strict
   schema + read-before-start + prepend-after-commit lifecycle you cannot
   satisfy, so the SUPERVISOR consolidates your run-record into it (see
   § Supervision loop). Your reply to the orchestrator is a copy of your
   run-record, nothing more.
6. You are forbidden from: editing any file other than your own
   `exec-runs/<task-slug>.txt`; touching `AGENT_LOG.md` or any other board
   file; running `git commit`/`git push` unless they appear in your numbered
   list; interpreting WHY something failed (that is the supervisor's job).
7. **No exponential build residue.** You run in the SHARED session checkout
   against the ONE shared `target/`. You are forbidden from: creating a git
   worktree or clone (`git worktree add`, `git clone`); setting or moving
   `CARGO_TARGET_DIR`; any command not in your numbered list that would
   materialize a second `target/`. Your card's cargo commands are already
   scoped (`-p <crate>`); run them exactly. If a build fails with a
   disk/`No space left on device` error, that is a STOP → `status=BLOCKED`,
   `gates: disk → FAIL` — never try to free space yourself; the supervisor
   prunes and re-cards.

## No exponential target residue (the resource half of the guardrail)

This is the cargo-hygiene rule (`ndarray/.claude/rules/agent-cargo-hygiene.md`)
made mandatory for the executor tier, because the executor is the ONE tier
whose whole job is to compile:

- **One checkout, one `target/`, ever.** The supervisor spawns executors
  WITHOUT `isolation: "worktree"` (the default here is the shared checkout —
  keep it). An executor that clones or worktrees is a policy violation, not a
  performance choice.
- **Compilation is centralised, not fanned out.** Prefer a SINGLE executor
  running the full gate sequence (`fmt` → `clippy` → `test`, scoped `-p`)
  serially against the one `target/`. That is already the cheapest shape: the
  build is CPU/IO-bound and cargo takes a per-`target/` lock, so N parallel
  executors on the same crate would serialise on that lock anyway — you pay
  the coordination cost for zero speedup.
- **If you must parallelise, disjoint crates + STILL one `target/`.** Two
  executors on genuinely disjoint crates may overlap, but they share the ONE
  `target/` (cargo's lock serialises the contended parts; correctness over
  speed). Never give an executor its own target dir to "avoid contention" —
  that trades a lock wait for a disk blow-up.
- **The card, not the executor, owns scope.** Every cargo line the supervisor
  writes is `-p <crate>` (or `--manifest-path <crate>/Cargo.toml`) — never a
  bare `cargo build`/`test`/`--all`/`--workspace` that would cold-compile the
  whole tree or follow a path-dep into a sibling repo (the tesseract-rs
  `--all` disaster). The executor just runs the card; the discipline is
  upstream, in the filigree tier that authored it.

## The run-record format (one file per executor; read by the supervisor)

The executor writes exactly this to `.claude/board/exec-runs/<task-slug>.txt`
(its own file — never the shared `AGENT_LOG.md`). It is a machine-terse
receipt, NOT a board entry; the supervisor turns it into the board entry.

```text
### <UTC timestamp> — haiku-exec — <task-slug>
- commands: <n>/<total> completed
- status: GREEN | BLOCKED@cmd<k>
- gates: <one line per gate command: name → PASS/FAIL/SKIPPED>
- tail: <last ≤10 lines of the failing command, only if BLOCKED>
```

## Supervision loop

- The Opus/main supervisor writes the numbered command list (the "execution
  card"), spawns the Haiku executor, and reads ONLY its run-record — never
  the raw transcript.
- **The supervisor owns the board write.** It reads the executor's
  `exec-runs/<task-slug>.txt` receipt(s) and — following the full workspace
  blackboard contract (read `AGENT_LOG.md` first, PREPEND after committing,
  include the deliverables / commit / tests / outcome fields the schema
  requires) — writes the ONE consolidated `AGENT_LOG.md` entry. This is why
  the executor never touches the board: only an agent that runs the whole
  lifecycle can satisfy the board schema, and a single writer means no
  interleave and no append-vs-prepend conflict. The `exec-runs/*.txt` files
  are transient receipts (gitignore or prune after consolidation), not the
  audit trail — the `AGENT_LOG.md` entry the supervisor prepends IS the
  audit trail.
- A BLOCKED run-record escalates to a Sonnet fix-agent (with the receipt's
  tail as brief) or to the supervisor; Haiku is never asked to fix.
- Multiple Haiku executors may run in parallel ONLY on disjoint
  crates/directories, ONLY sharing the one `target/` (never
  `isolation: "worktree"`, never a per-executor target dir — see
  § No exponential target residue), AND ONLY writing disjoint
  `exec-runs/<task-slug>.txt` receipts (one executor, one file — the shared
  `AGENT_LOG.md` is written once, later, by the supervisor). The default and
  safest shape is a SINGLE executor running the gate sequence serially.
- A `status=BLOCKED` with `gates: disk → FAIL` is a supervisor action item,
  never an executor one: the supervisor prunes stale build artifacts / caches
  (deletes still succeed when writes fail) and re-cards; the executor is never
  asked to free space.

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
