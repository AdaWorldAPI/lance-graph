# Issues Log — Open + Resolved (double-entry, append-only)

> **Append-only ledger.** Every issue (bug, regression, invariant
> violation, blocker) gets a dated entry here. Entries move from
> Open → Resolved by status-flip; they are NEVER deleted.
>
> **Format invariant:** every entry starts with `## YYYY-MM-DD — `
> followed by a short title. Body is short — one paragraph of
> problem + cross-references. Full repro / fix / test details go
> in the PR or in a dedicated doc and are LINKED, not duplicated.
>
> **Mutable field:** `**Status:**` line only (Open / Resolved /
> Wontfix / Superseded). Resolved entries keep a `**Resolution:**`
> line pointing at the PR + commit SHA that fixed them.

---

## Double-entry discipline

Every issue has TWO corresponding rows, both in this file:
1. **Open section** — issue captured when first seen.
2. **Resolved section** — same entry, appended when closed, with
   `**Resolution:**` line pointing at fix.

The resolved entry cites the open entry's date as anchor. Old
"Open" entry's **Status:** flips to `Resolved YYYY-MM-DD` — it
stays in the Open section (never moved) so chronology is
preserved. The Resolved section accumulates fixes for discovery.

This is **bookkeeping discipline**, not a storage optimization:
- Open section = what broke and when.
- Resolved section = how and when it was fixed.
- Both sections keep the same row forever; the view depends on
  which section you're reading.

---

## Governance

- **Append-only.** Never delete a row from either section.
- **Mutable:** `**Status:**` and `**Resolution:**` fields only.
- **`permissions.ask` on Edit** (same rule as PR_ARC_INVENTORY).
  Write for appends stays unprompted.
- **Supersedure:** if an issue turns out to be a duplicate of an
  older one, Status → `Superseded by YYYY-MM-DD <title>`; old entry
  stays.

## Cross-references

- `PR_ARC_INVENTORY.md` — which PR shipped the fix.
- `STATUS_BOARD.md` — deliverable-level view (an issue may block
  one or more D-ids).
- `EPIPHANIES.md` — if debugging surfaced an architectural
  insight, that lands in Epiphanies; this file tracks the concrete
  fix.
- `TECH_DEBT.md` — if an issue is knowingly deferred rather than
  fixed, it moves (via cross-ref) into technical debt.

---

## Kanban Format (priority + scope on every entry)

Every issue carries:
- **Priority** — `P0` blocker / `P1` high / `P2` medium / `P3` low.
- **Scope** — which agent / deliverable / domain owns it. One or
  more of: `@<agent-name>`, `D<N>` (plan D-id),
  `domain:<grammar|codec|infra|arigraph|...>`.

Together they form the ticket tag: `[P1 @truth-architect D5 domain:grammar]`.
Agents filter by their own `@`-mention or their domain; nothing
gets buried.

## Open Issues

## 2026-04-20 — [E-MEMB-1] Python↔Rust slice layouts are incompatible at the 10 kD membrane

**Status:** Open
**Priority:** P1
**Scope:** @integration-lead @truth-architect domain:membrane

PR #210's `role_keys.rs` (Rust) defines disjoint slices of the 10K VSA: Subject [0..2000), Predicate [2000..4000), Object [4000..6000), Modifier [6000..7500), Context [7500..9000), TEKAMOLO [9000..9900), Finnish [9840..9910), tenses [9910..9970), NARS [9970..10000). Python `adarail_mcp/membrane.py::DIMENSION_MAP` uses a different layout entirely: [0..500) "Soul Space" (qualia_16 / stances_16 / verbs_32 / tau_macros / tsv), dim 285 = hot_level, [2000..2018) = qualia_pcs_18. Any vector round-tripped across the two stacks will be reinterpreted by the other side's slice geometry → semantic noise, silent mis-binding.

**Impact:** blocks cross-language reconciliation for the AGI-as-glove surface (Ada σ/τ/q ↔ Rust BindSpace SoA). Until resolved, the Membrane cannot use raw 10K transfer — only serialized σ/τ/q at the REST edge.

**Secondary blocker:** E-MEMB-7 (Ada has its own 3-space incoherence between `membrane.py` 10kD, `rosetta_v2.py` 1024D Jina, and Fingerprint<256> 16K-bit — reconcile internally before Python↔Rust).

**Substrate constraint (added 2026-04-20 per [FORMAL-SCAFFOLD] reclassification):** any bridge between Python-membrane and Rust-role_keys MUST respect E-SUBSTRATE-1. An identity-map between the two layouts would violate bundle associativity — the two layouts encode different algebraic structures over d=10000. The reconciliation doc must EITHER pick one layout as canonical (likely Rust's `role_keys` disjoint slices) and re-express Python's into it, OR define a projector that preserves commutativity of bundle under translation. **A naive bit-by-bit remap is not acceptable** — it would silently break the Markov guarantee that D7 and the rest of the NARS revision stack rely on (see I-SUBSTRATE-MARKOV in CLAUDE.md).

**Next action (when queued):** author a `slice-layout-reconciliation.md` knowledge doc mapping every Python DIMENSION_MAP region to either (a) a Rust role_keys slice, (b) a dropped region, or (c) a new Rust slice to add. The doc MUST include the substrate-respect analysis above. Not yet scheduled.

Cross-ref: `.claude/board/EPIPHANIES.md` 2026-04-20 E-MEMB-1; `.claude/board/EPIPHANIES.md` E-SUBSTRATE-1 + [FORMAL-SCAFFOLD]; Deposit log E-MEMB-7; PR #210 role_keys.rs; `adarail_mcp/membrane.py::DIMENSION_MAP`; CLAUDE.md I-SUBSTRATE-MARKOV.

---

## 2026-05-13 — ndarray:master missing `hpc-extras` feature (latent downstream build break)
**Status:** Open (upstream-blocked)
**Priority:** P2
**Scope:** domain:infra D-NDARRAY-MASTER-HPC-EXTRAS

The `hpc-extras` feature on `ndarray` lives on `AdaWorldAPI/ndarray` branch `claude/burn-A1-dep-gating` (PR #116, **never merged to master**). lance-graph PR #364 (`a3c753f`) declares `features = ["hpc-extras"]` on its `ndarray` path dep — this works for us because the local `/home/user/ndarray` checkout is on the integration branch that carries the feature. **Any consumer that points at `ndarray:master` (post-#142, pre-#116) will hit `feature hpc-extras not found`** — surfaced by MedCare-rs PR #118 (doc-only investigation, merged 2026-05-13). The fix is upstream: `ndarray PR #116 → master`. Outside this session's scope; tracked here so it doesn't get rediscovered.

Cross-ref: MedCare-rs#118, lance-graph PR #364 commit `a3c753f`, ndarray PR #116 (`claude/burn-A1-dep-gating`), ndarray PR #142 (VBMI+Inf clamp, merged but does NOT add hpc-extras to master).

---

## 2026-05-16 — [W-F9-X1] Subagent Edit/Write permission isolation gap — workers must use python3 heredoc fallback

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a @adk-coordinator
**Filed by:** W-F9 (sprint-12 Wave F sweep); originally surfaced per E-META-8

The Claude Code SDK subagent context used in sprint-11 CCA2A workers had `Edit`, `Write`, and `MultiEdit` tools blocked by permission policy. Every worker that needed to write files was forced to use `python3 << 'PYEOF'` heredocs via the Bash tool as a fallback. This pattern works but is awkward, undiscoverable, and error-prone (heredoc quoting rules differ from Edit semantics). Workaround: explicitly instruct workers in their prompt ("Edit/Write blocked — use `python3` heredocs"). Resolution requires either an upstream SDK permission fix or acceptance of the heredoc pattern as the CCA2A standard for write operations in restricted subagent contexts.

Cross-ref: EPIPHANIES.md E-META-8; `.claude/agents/BOOT.md` subagent spawn policy; sprint-11 W-D2/W-F1..W-F9 agent logs.

---

## 2026-05-16 — [W-F9-X2] Stop-hook fires on uncommitted in-flight state during subagent handoff

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a domain:hooks
**Filed by:** W-F9 (sprint-12 Wave F sweep)

When a CCA2A subagent stops mid-task with uncommitted files, the stop-hook fires and may trigger board-hygiene checks or branch guards against a dirty state. Subsequent workers or branch switches then require a stash dance (`git stash` / `git stash pop`) before they can proceed. The workaround is: commit incrementally and stash before any branch switch. A proper resolution would require the stop-hook to detect known-active-worker state (e.g., via a sentinel file or `STATUS_BOARD.md` marker) and tolerate mid-task uncommitted changes without erroring.

Cross-ref: `.claude/hooks/` (stop-hook scripts); `.claude/board/STATUS_BOARD.md`; sprint-11 Wave D multi-step stash dance notes.

---

## 2026-05-16 — [W-F9-X3] Workspace disk quota at 91%+ during cargo builds; ENOSPC risk recurring

**Status:** Open
**Priority:** P1
**Scope:** domain:infra domain:build
**Filed by:** W-F9 (sprint-12 Wave F sweep); first hit during PR #386 rebase cycle

During the sprint-11 PR #386 cycle the workspace hit ENOSPC mid-rebase; 21 GB was freed by running `cargo clean`. The `target/` directory accumulates incrementally built artifacts from multiple workers building different crates in parallel, and the quota ceiling (~91% at the time of the incident) leaves insufficient headroom for rebase + build operations. Risk is recurring: every sprint with heavy parallel cargo work will approach the ceiling. Resolution options: (a) periodic `cargo clean` as a sprint-start hygiene step, (b) smaller per-worker `CARGO_TARGET_DIR` so artifacts don't accumulate in one location, (c) larger disk quota.

Cross-ref: PR #386 (sprint-11); sprint-11 Wave D rebase log.

---

## 2026-05-16 — [W-F9-X4] `cargo check -p lance-graph` may fail locally due to missing `protoc` binary

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:build crate:lance-graph
**Filed by:** W-F9 (sprint-12 Wave F sweep)

`lance-encoding` (a transitive dependency of `lance-graph`) requires the `protoc` system binary for its build script. In sprint-11 this binary was absent from the default environment; W-D2 installed it manually. As a result, `cargo check -p lance-graph` (and any other command that pulls `lance-encoding`) will fail with an opaque `protoc not found` error on any worker environment that has not had the binary pre-installed. **CI is the canonical validator**; workers should note that a local compile failure of `lance-graph` may be an environment issue, not a code issue. Resolution: automate `protoc` installation in workspace setup (see TECH_DEBT.md TD-PROTOC-ENV-SETUP-1).

Cross-ref: TECH_DEBT.md TD-PROTOC-ENV-SETUP-1; D-CSV-6a agent log (W-D2 manual install); sprint-11 Wave D build notes.

---

## 2026-05-16 — [W-F9-X5] Background-worker file collisions during main-thread rebase require multi-step stash dance

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a
**Filed by:** W-F9 (sprint-12 Wave F sweep)

During sprint-11 Wave D, a background worker had modified workspace files while the main thread needed to rebase onto updated `main`. The conflict required a multi-step stash dance: stash local changes → rebase → pop stash → resolve conflicts → continue. The pattern works but is fragile: if the stash contains large or structurally complex diffs the pop may produce confusing three-way conflicts. Proper resolution would coordinate worker commits with main-thread rebase windows (e.g., all workers commit before any rebase is initiated), or use per-worker branches that are rebased independently.

Cross-ref: Sprint-11 Wave D / sprint-12 Wave D rebase log; TECH_DEBT.md TD-PROTOC-ENV-SETUP-1 (related infra gap); `.claude/agents/BOOT.md` handover protocol.

(No other tracked open issues. New issues PREPEND here
in reverse chronological order. Format below.)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>

<one paragraph: what's broken, where it surfaces, rough impact>

Cross-ref: <file:line or PR # or knowledge doc>
```

---

## Resolved Issues

(No resolved issues at initial commit. When an Open issue is fixed,
APPEND a copy here with the same date anchor + `**Resolution:**`
line. Old Open entry's Status flips to `Resolved YYYY-MM-DD`. Old
entry stays in the Open section for chronology.)

```
## YYYY-MM-DD — <same title as Open entry>
**Status:** Resolved YYYY-MM-DD
**Resolution:** PR #NNN (commit SHA) — <one-line description>

<original problem paragraph, verbatim>

Cross-ref: <same as Open entry>
```

---

## How to use this file

**When an issue is found** — prepend to **Open Issues** section with
today's date + `**Status:** Open` + one-paragraph description.

**When an issue is fixed** — append to **Resolved Issues** section
with the same title and date anchor + `**Status:** Resolved
YYYY-MM-DD` + `**Resolution:** PR #NNN`. Don't edit the Open entry
body; just flip its Status to `Resolved YYYY-MM-DD`.

**When an issue is a duplicate** — append a new entry in Resolved
section noting `**Resolution:** duplicate of YYYY-MM-DD <title>`;
flip Open entry to Superseded.

**When an issue is deferred knowingly** — leave it Open here but
also append a row to `TECH_DEBT.md` with cross-ref back.
