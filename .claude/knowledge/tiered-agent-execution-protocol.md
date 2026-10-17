READ BY: any session orchestrating subagent fleets in this workspace. Defines the three tiers; since 2026-10-14 Haiku is allowed for read-only harvest and guarded execution (§ Haiku harvest tier), no longer only the executor role. The earlier proposal text below is kept as history.

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

# Tiered agent execution — Opus filigree / Sonnet grindwork / Haiku harvest and guarded execution

## The three tiers

| tier | role | may | must never |
|---|---|---|---|
| Opus (or main thread) | FILIGREE: synthesis, plan authorship, adjudication of probe results vs gates, board rulings, review | read everything; write plans/knowledge/board | be used for mechanical loops |
| Sonnet | GRINDWORK: write-file-from-spec, scoped edits, drafting from a complete brief | edit code/docs per brief | synthesize across sources; decide gates; touch board files unless brief says so |
| Haiku | HARVEST (read-only discovery: candidates, counts, verbatim quotes over a closed space) and GUARDED EXECUTION (a pre-written command card) | return what it found, with locations and the commands used; run the exact commands in its card; retry per an explicit retry table | write or edit ANY source or doc file; state a conclusion, an absence or an explanation; append to any SHARED file; interpret ambiguous failures; deviate from the brief |

## The load-bearing invariant — ONE WRITER PER FILE (all tiers)

**No agent ever appends to a file another agent also writes.** Each agent
writes exactly ONE file it alone owns, keyed by its agent tag / id (an
executor → `exec-runs/<slug>.txt`; any worker that must leave a record →
`<agentId>.md`). A shared file — `AGENT_LOG.md`, any board file — has exactly
ONE writer, ALWAYS: the orchestrating main thread, which reads the set of
per-agent tag-files and consolidates. This is not a Haiku-tier workaround; it
is the general law, and the executor's receipt is just its first instance.

Why it is load-bearing, not hygiene:

- **Concurrent append to one file is a lost-write race.** Two agents doing
  read → append → write (or `>>` of multiline records) interleave or drop
  entries. Per-agent tag-files have no contention by construction — one writer,
  one file.
- **It is the SAME law the runtime substrate already proves.** One-writer-
  per-mailbox is `SoaEnvelope::mailbox_owner` + write-on-behalf +
  no-singleton-sink (E-CE64-MB-4): aliasing/data-race becomes a *compile
  error* downstream precisely because nothing co-writes a shared cell. A
  shared session-log that every agent appends to is that eliminated
  shared-mutable-sink, re-created one layer up.
- **A shared append-log is NOT A2A.** Real agent-to-agent coordination is a
  message to an *owned* mailbox; you own your lane and the reader reads
  *across* lanes. Co-appending to one log is pseudo-A2A with a race attached —
  use per-agent tag-files + a single consolidator, never a shared sink.

> **Corrects the workspace Layer-2 A2A convention (operator-ruled 2026-07-22,
> `E-AGENT-LOG-SHARED-SINK-ANTIPATTERN-1`).** CLAUDE.md §"Layer 2 — Session
> A2A" previously said "Every agent run gets one append-only entry" and every
> agent prompt "MUST … prepend your own entry" to `AGENT_LOG.md`. That is the
> shared-mutable-sink anti-pattern: sub-agents own tag-files; the **orchestrator
> is the SOLE writer** of `AGENT_LOG.md` and every other shared board file. The
> operator's words: *"agents are never supposed to append to the same file, but
> instead it's an agent tag file … appending to the same log is illogical."*

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
   with status=BLOCKED and the root diagnostic block (item 4).
3. Retry table: network-flavored git/curl failures → up to 3 retries with
   2s/4s/8s backoff. Nothing else retries.
4. Diagnostic discipline: run the command so its COMPLETE output is
   retained (redirect to a file, or capture it whole) whenever failure
   analysis may be needed, and preserve the command's TRUE exit status —
   `cmd | tail -30` reports `tail`'s success even when `cmd` failed, so run
   under `set -o pipefail` (or read `${PIPESTATUS[0]}` before evaluating the
   retry/STOP rule); the STOP condition (item 2) tests the PRODUCER's status,
   not the pipeline's. On failure, locate the FIRST relevant diagnostic (the
   root, not the last thing printed) and read that complete diagnostic block
   — a compiler's later errors are usually consequences of the first one, and
   a trailing summary can omit the root entirely. A short tail or summary is
   DISPLAY ONLY and is never sufficient evidence; report the root diagnostic
   block, not a line count. Still never dump a full build log into your reply.
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
- root_diagnostic: <the FIRST relevant diagnostic block of the failing
  command, complete, only if BLOCKED — e.g. the first `error[E...]` and its
  whole body. Bounded: if that block exceeds ~40 lines, give its first 40
  and say how many were omitted. NEVER a trailing tail: a compiler prints
  the causal error FIRST and its consequences after, so the last lines are
  usually the cascade, not the cause.>
- tail: <last ≤10 lines, only if BLOCKED — context ONLY, never the evidence>
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
  root_diagnostic as the evidence, tail as context) or to the supervisor;
  Haiku is never asked to fix.
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

## Haiku harvest tier (DECISION 2026-10-14)

DECISION (operator, 2026-10-14): "Use Haiku agents for grindwork and take
Haiku off the guard list, but watch the agents closely and start using them
for very focused and scoped tasks that you use for cheap harvest discovery."
SCOPE: read-only discovery, deep mapping, exact spec edits and guarded
execution in this workspace; authoring stays with Sonnet. Follow-up
(operator, same day): "Goal is to outsource deeper mapping due to enormous
token equivalent of the substrate", and Haiku may apply exact edits from a
written spec. BASIS: harvest output is navigation (FIRST-HAND-SOURCE-LAW),
so a cheap model is enough to find things as long as reading and deciding
stay with a stronger one. REVISIT WHEN: the run ledger below shows the spot
checks costing more than a Sonnet run, or Haiku is proposed for edits.

### Cards

| card | returns | tools |
|---|---|---|
| `harvest-scout` | every candidate as `path:line` + the matching line | Read, Glob, Grep |
| `census-counter` | raw counts, each with the exact command | Read, Glob, Grep, read-only Bash |
| `quote-harvester` | verbatim text with line numbers; FOUND / NOT FOUND checks | Read, Grep, Glob |
| `substrate-mapper` | a fact map of one bounded unit, written to one named file | Read, Glob, Grep, Write (its map only) |
| `spec-editor` | per-edit APPLIED / REFUSED for exact edits to one file | Read, Edit |
| `guarded-executor` | a run-record for a pre-written command card | Bash, Read, Write (its own run-record only) |

### Reading is free: briefs never ration it (DECISION 2026-10-17)

DECISION (2026-10-17): "Their tokens are so immensely cheap that reading
becomes the normal at no cost, that's the whole point." BASIS: the first
wiring maps (five slices over PRs #1447-#1596) came back PARTIAL. The
orchestrator's brief had said "read the first 120 lines", and the mappers
substituted grep scans and 40-line windows for reading. A Haiku card reads
every file it reports on in full, every page; a search hit is only where to
read. A brief must never cap reading depth (first N lines, one section); it
caps the search SPACE. A card given such a cap reads the whole file anyway
and notes it. Each card carries this as its "Read in full" section.
REVISIT WHEN: a single file exceeds what a Haiku context can hold; then the
brief splits the file into named pages, all of which are read.

### What makes a brief Haiku-sized

- ONE question, answerable as a list, a count or a quote.
- A CLOSED search space: exact paths or globs. "Somewhere in the repo" is
  not a brief.
- The patterns written out by the orchestrator, including aliases,
  fully-qualified spellings and re-exports (the measured grep failures in
  CLAUDE.md were all spelling misses).
- A hit cap.
- No step that needs holding two sources in mind together. If it does, it is
  an Opus task.

### Watching closely (the supervision rules)

1. **Harvest is navigation.** The orchestrator Reads every item it relies on
   before using it in a claim, board entry, PR body or decision. A Haiku list
   is never cited as evidence.
2. **Re-run one count per reply.** For `census-counter`, the orchestrator
   re-runs at least one reported command and compares.
3. **Absence is never taken from Haiku.** A `hits: 0` sends the orchestrator
   to Read the place the thing would live, as for any search.
4. **Record every run.** The orchestrator's `AGENT_LOG.md` entry for a wave
   that used Haiku lists, per card: runs, items spot-checked, items wrong,
   items missed (found later by a stronger agent or a Read).
5. **Drop back on repeat failure.** A card that misses a hit, miscounts or
   states a conclusion twice on one kind of task is not used for that kind
   again; that kind goes to Sonnet, and the case is noted in the ledger.
6. **No authored content.** Only `spec-editor` edits, and only text the
   orchestrator wrote out in full; the orchestrator reads its diff before
   committing. `substrate-mapper` writes only its map, `guarded-executor`
   only its run-record. Other cards have no write tool.
7. **Maps are checked against source.** For every map, the orchestrator
   Reads at least two mapped items at their stated locations before relying
   on the map, and treats `mentioned by` as candidates, never as a call graph.

### Mapping a large surface

Split the surface into units small enough to read in full (one module file,
or one directory of a few thousand lines). Run one `substrate-mapper` per
unit, in parallel only over disjoint units, each with its own output file.
The orchestrator reads the maps, not the source, and Reads source only where
a decision rests on it.
