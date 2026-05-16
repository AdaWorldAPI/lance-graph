# Preflight Drift Patterns — PD1..PD10 Catalogue

> **READ BY:** `preflight-drift-auditor` (canonical); also any Opus meta-reviewer
> running a manual sprint preflight when the full agent is not spawned; any
> main-thread orchestrator verifying planner-fleet outputs before worker dispatch.
>
> **Status:** OPERATING PROCEDURE (every pattern below was caught at least once
> by W-Meta-Opus during sprint-13 preflight of the Wave H fleet, or is a
> first-order generalization of a caught pattern. CSI citations are exact.)
>
> **Primary source:** `.claude/board/sprint-log-13/preflight-meta-review-opus.md`
> — W-Meta-Opus manual audit of PP-1..PP-13 Wave H outputs (2026-05-16).
>
> **Predecessors:**
> - `.claude/board/sprint-log-12/meta-review.md` CSI-7..18 (Wave G cross-cutting
>   findings that sprint-13 preflight fleet was supposed to prevent)
> - `.claude/board/sprint-log-11/meta-review-opus.md` CSI-1..6 (Wave F root
>   observations; PD1 traces to CSI-9 here)
>
> **Promotion track:** patterns that fire N ≥ 3 times across distinct preflight
> runs become permanent SPAWN-GATE rules per the ceremony in
> `.claude/agents/preflight-drift-auditor.md` §6.

---

## §1 What the Preflight-Drift-Auditor Sees — Six Verification Axes

### Axis 1 — Stale-Claim Verification

A plan document cites a git state, file:line, or symbol existence that was true
at the time of writing but has since diverged from the actual working tree, OR
was never true and was an error in the planner's self-report. The canonical failure
mode: planner states "CSI-N resolved via commit X" but the commit is not in the
target repo's master branch. Because impl workers read the plan as ground truth,
a stale claim sends them into a nonexistent state. The verification mechanism is
always: run the actual git command, read the actual file, grep the actual symbol —
never trust the planner's assertion. This is the axis that W-Meta-Opus's opening
move covers: `cd /home/user/ndarray && git log master --oneline` before accepting
any claim about ndarray's state.

### Axis 2 — Cross-Planner D-id / OQ-id Collision

In a multi-planner preflight fleet, each planner chooses D-CSV-N and OQ-CSV-N
identifiers independently. Without a designated ID coordinator, two planners can
assign the same D-CSV number to different deliverables (reserved by PP-X for
purpose A, reserved by PP-Y for purpose B), or use conflicting OQ number ranges
(PP-1 lists OQ-CSV-7..12 while PP-11 lists OQ-CSV-7..19). The failure mode is
that worker prompts reference the D-id or OQ-id and receive contradictory
instructions from two upstream planners. This axis also catches single-spec
internal contradictions where a planner cites a D-id with the wrong owner or
wrong purpose — three errors in one sentence is the PD6 benchmark.

### Axis 3 — Pending-vs-Canonical Confusion

A symbol, iron rule, or architectural constraint is cited as "canonical" or
"on main" when it actually lives on an open PR branch. Because specs are read
by impl workers who assume canonical state, a pending-state citation causes
workers to implement against a state that may never land, or to treat as
ratified what is still under review. The critical instance: PP-2 listed
`I-LEGACY-API-FEATURE-GATED` in its "Canonical iron rules" axis table, and
five downstream specs then cited it as canonical — but `grep I-LEGACY-API-FEATURE-GATED
CLAUDE.md` on main returned empty. The rule was on PR #390. The clean pattern
(PP-5's precedent) is explicit disclosure: "formerly `CamPqIndexPlaceholder`
until CSI-15 rename in PR #390." The failure pattern omits the qualifier entirely.

### Axis 4 — Cross-Repo PR-Merge Claim

A planner states that a pull request in a sibling repository (ndarray, crewai-rust,
n8n-rs) has been merged to its master branch. Because the planner cannot directly
run `git log` in the sibling repo at plan-write time (or runs it against a stale
local state), the claim may be false. The canonical failure: PP-12 stated "ndarray
PR #147 merged 2026-05-16T04:35:05Z" — actual `git log master --oneline` in
`/home/user/ndarray` showed master HEAD was `2a3885d2` (PR #146 merge), no PR #147
in master. The streams commit existed only on a local branch. When two planners
(PP-1 and PP-12) both ship the same false cross-repo merge claim, the compounding
effect increases the chance that a worker will be dispatched against the false state.

### Axis 5 — Renamed-Symbol Downstream Drift

A type rename, function rename, or module rename that is in-flight on a PR branch
creates a window where planners cite either the old name or the new name without
proper disclosure of the pending status. Two failure modes: (a) spec cites the
NEW name as canonical while the rename is still on the branch (impl worker looks
for the new name on main, finds the old name, stalls); (b) spec cites the OLD
name as current while the rename is live on a branch and should be disclosed
(impl worker implements against a name that is about to change). The clean pattern:
PP-5 §0 cited "formerly `CamPqIndexPlaceholder` until CSI-15 rename in PR #390"
— this is the ACCEPTABLE form. Any cite of the new name `WitnessIndexHashMap` as
canonical without this disclosure would be drift.

### Axis 6 — Iron-Rule Citation Drift

A spec cites an iron rule by its canonical ID (I-SUBSTRATE-MARKOV,
I-NOISE-FLOOR-JIRAK, I-VSA-IDENTITIES, I-LEGACY-API-FEATURE-GATED) and uses it
to justify an architectural decision, but the rule is not yet in `CLAUDE.md` on
main (it exists only on a PR branch or in a planning doc). Workers reading the
spec will treat the rule as enforcement-ready and may reject valid code that
doesn't satisfy a rule that hasn't landed yet, or accept code that violates a
rule they believe is pending but will actually ship. The verification is a single
grep: `grep -n 'I-RULE-ID' /home/user/lance-graph/CLAUDE.md`. If it returns
empty, the rule is not canonical. The spec must qualify the citation.

---

## §2 Preflight Drift-Pattern Catalogue — PD1..PD10

Each entry: Name | Verification axis | Symptom | Verification command | The rule |
Fix pattern | W-Meta-Opus precedent | Promotion track.

---

### PD1 — Stale-CSI-Resolved-Via-Commit

**Verification axis:** Axis 1 (stale-claim)

**Symptom:** A planner spec or plan document states "CSI-N resolved via commit X"
or "CSI-N status: CLOSED, fixed in d4e5bbc" but the fix commit exists only in
lance-graph (the planning repo), while the actual structural fix required in a
sibling repo (ndarray, crewai-rust) has not landed on that repo's master.

**Verification command:**
```bash
# For each sibling-repo CSI claim, run in the target repo:
cd /home/user/ndarray && git log master --oneline | head -20
# Then check the specific commit hash the planner cited:
cd /home/user/ndarray && git show --stat <cited-commit-hash> 2>/dev/null || echo "COMMIT NOT IN REPO"
# Also verify the file the fix was supposed to touch:
grep -rn '<symbol_name>' /home/user/ndarray/src/hpc/stream/mod.rs
```

**The rule:** A CSI that requires a change in a sibling repo is OPEN until that
repo's master branch contains the fix commit. A lance-graph-side aggregation commit
that adds governance docs does NOT resolve a sibling-repo structural requirement.

**Fix pattern:** Change the plan status from "RESOLVED/CLOSED" to "OPEN (cross-repo
fix pending on <repo>; local branch <branch-name> contains the fix; needs PR and
master merge before sprint spawn)."

**W-Meta-Opus precedent:** CSI-9 — PP-1 §0.1 claimed CSI-9 "Risk REDUCED" via
`d4e5bbc` (a lance-graph commit). W-Meta-Opus ran `cd /home/user/ndarray && git log
master --oneline` and found master HEAD `2a3885d2` with no stream registration.
`qualia.rs` and `splat_field.rs` were not registered in `hpc/stream/mod.rs` on
master. PP-7 sprint-12 meta-review correctly flagged CSI-9 as OPEN/HARD BLOCKER.
PP-1 and PP-12 both propagated the false "reduced" status.

**Promotion track:** Fire count = 2 (PP-1 + PP-12 in sprint-13 wave). One more fire
in sprint-14 preflight → promote to Option C SPAWN-GATE (mandatory pre-spawn check:
every sibling-repo CSI must show git log master verification before spawn-clear).

---

### PD2 — Cross-Planner-D-id-Collision

**Verification axis:** Axis 2 (ID collision)

**Symptom:** Two or more planners assign the same D-CSV-N number to different
deliverables, OR planner A's OQ-CSV range is a strict subset of planner B's range
with no acknowledgment of the discrepancy.

**Verification command:**
```bash
# Extract all D-CSV mentions from all planner outputs:
grep -rhn 'D-CSV-[0-9]\+' /home/user/lance-graph/.claude/plans/ \
     /home/user/lance-graph/.claude/specs/ \
     /home/user/lance-graph/.claude/knowledge/ \
     /home/user/lance-graph/.claude/board/sprint-log-13/ \
  | sed 's/.*\(D-CSV-[0-9]\+\).*/\1/' | sort | uniq -c | sort -rn
# For each D-id appearing in multiple files, compare what each file says:
grep -rn 'D-CSV-16' /home/user/lance-graph/.claude/ | head -30
```

**The rule:** Each D-CSV-N and OQ-CSV-N must map to exactly one deliverable description
across all planners. The canonical assignment is the one in the primary plan doc
(e.g., cognitive-substrate-convergence-v3.md §11 D-table). Any planner that cites
a D-id in a cross-reference must use the canonical description.

**Fix pattern:** Run the cross-reference table in the audit report. For each collision,
identify which planner has the canonical assignment (the one in the primary plan §11
D-table) and correct every other planner's cross-reference to match.

**W-Meta-Opus precedent:** CSI-19 — PP-1 plan v3 §11 correctly assigned D-CSV-16 to
CAM-PQ (PP-5). PP-3 §0 said "D-CSV-16 reserved by PP-2 (sprint-13 splat on-Think
method migration)" — three errors: PP-2 is iron-rules-doctrine; PP-4 is splat on-Think;
D-CSV-16 is CAM-PQ. OQ-CSV range: PP-1 lists OQ-CSV-7..12 (6 entries); PP-11 lists
OQ-CSV-7..19 (13 entries). The PP-1 undercount would have left 7 open OQs invisible
to the coordinator.

**Promotion track:** Fire count = 1 (sprint-13 wave). One more fire → promote to
Option B worker-template-v2 §5 item (worker must cite D-id from primary plan table,
not from sibling planner output).

---

### PD3 — Pending-Iron-Rule-Cited-As-Canonical

**Verification axis:** Axis 3 (pending-vs-canonical)

**Symptom:** A planner spec cites an iron rule by its I-* ID in the "iron rules in
force" or "canonical rules" context without a "(pending PR-N merge)" qualifier, but
the rule is not yet in `CLAUDE.md §Substrate-level iron rules` on main.

**Verification command:**
```bash
# For each iron rule cited by a planner, verify it is in CLAUDE.md on main:
grep -n 'I-LEGACY-API-FEATURE-GATED' /home/user/lance-graph/CLAUDE.md
grep -n 'I-SUBSTRATE-MARKOV' /home/user/lance-graph/CLAUDE.md
grep -n 'I-NOISE-FLOOR-JIRAK' /home/user/lance-graph/CLAUDE.md
grep -n 'I-VSA-IDENTITIES' /home/user/lance-graph/CLAUDE.md
# Also check iron-rules-doctrine §6 for "pending" annotations:
grep -n 'pending' /home/user/lance-graph/.claude/knowledge/iron-rules-doctrine.md
```

**The rule:** A spec may CITE a pending iron rule by name ONLY if it simultaneously
includes the disclosure "(pending PR-N merge; not yet canonical on main; treat as
advisory until merged)." A spec that uses a pending rule to JUSTIFY an architectural
decision that workers will implement without the pending qualifier is SPAWN-CAUTION.

**Fix pattern:** Add "(pending PR-#N merge)" inline after every bare cite of the
pending rule. If the spec uses the rule to gate a yes/no architectural decision
(e.g., "we reject Option B because it violates I-LEGACY-API-FEATURE-GATED"), flag
for SPAWN-CAUTION and add a note: "this decision is provisional pending PR-#N merge;
if PR-#N does not merge, revisit option selection."

**W-Meta-Opus precedent:** CSI-20 — PP-2 §1 axis table listed I-LEGACY-API-FEATURE-GATED
as a canonical peer of the other three iron rules. `grep I-LEGACY-API-FEATURE-GATED
/home/user/lance-graph/CLAUDE.md` returned empty — the promotion was on PR #390 branch
only. PP-6 line 8 and PP-13 §3 then cited the rule without pending qualifiers.
W-Meta-Opus grade: PP-2 A− with caveat; PP-6 A− with caveat; PP-13 A− with caveat.

**Promotion track:** Fire count = 3+ in sprint-13 wave (PP-2, PP-6, PP-13). Already
at promotion threshold. Recommend promoting to Option C SPAWN-GATE: "before any
worker spawn, verify every cited iron rule ID is present in CLAUDE.md on main."

---

### PD4 — False-Cross-Repo-PR-Merge

**Verification axis:** Axis 4 (cross-repo PR claim)

**Symptom:** A planner states that a PR in a sibling repository has been merged to
master on a specific date, but the sibling repo's actual `git log master` does not
contain the merge commit.

**Verification command:**
```bash
# Always verify against actual git log, not planner self-report:
cd /home/user/ndarray && git log master --oneline | head -20
# If planner cited a specific commit hash or PR number, check:
cd /home/user/ndarray && git branch --contains <cited-commit-hash> 2>/dev/null
# Cross-check against PR_ARC_INVENTORY.md:
grep 'PR #147\|ndarray.*147' /home/user/lance-graph/.claude/board/PR_ARC_INVENTORY.md
```

**The rule:** A cross-repo PR merge claim is only valid if:
(a) the merge commit appears in the target repo's `git log master`, AND
(b) the merge appears in `.claude/board/PR_ARC_INVENTORY.md` with a Confidence
annotation (or at minimum, the git log is authoritative). A local-branch-only commit
is NOT a merge regardless of the planner's timestamp.

**Fix pattern:** Change the claim from "PR #N merged YYYY-MM-DD" to "PR #N open on
branch `<branch-name>` (verified via `git log master --oneline`; NOT yet merged to
master as of audit date <date>)." Update all downstream plans that propagated the
false claim.

**W-Meta-Opus precedent:** PP-12 grade C+ — "ndarray PR #147 merged 2026-05-16T04:35:05Z"
verified FALSE by W-Meta-Opus running `cd /home/user/ndarray && git log master --oneline
| head -15`. Master HEAD `2a3885d2` = PR #146 merge. Commit `2a1a1e38` (the streams
scaffold) existed only on branch `claude/sprint-12-qualia-stream-w-f4`. PP-1 §13.8
and §0.1 also propagated this false claim. Two planners, one false claim = double
the downstream blast radius.

**Promotion track:** Fire count = 2 planners in sprint-13 wave (PP-1 + PP-12). One
more fire → promote to Option C SPAWN-GATE (mandatory check: run `git log master -20`
in every sibling repo before marking any cross-repo CSI CLOSED).

---

### PD5 — Renamed-Symbol-Downstream-Drift

**Verification axis:** Axis 5 (symbol rename)

**Symptom:** A planner spec cites a symbol by its NEW name as if it is canonical on
main (but the rename is on a branch), OR cites the OLD name for a concept that has
been renamed on a branch without disclosing the rename.

**Verification command:**
```bash
# Check whether the new name exists on main:
grep -rn 'WitnessIndexHashMap' /home/user/lance-graph/crates/ | head -10
# Check whether the old name still exists on main:
grep -rn 'CamPqIndexPlaceholder' /home/user/lance-graph/crates/ | head -10
# Find which branch the rename is on:
cd /home/user/lance-graph && git log --all --oneline --grep='CSI-15\|WitnessIndex' | head -10
```

**The rule:** A spec that references a renamed symbol MUST:
- Cite the OLD name as the current canonical name on main.
- Disclose the rename with: "Rename to `NewName` is queued in PR #N / CSI-N; the
  canonical name on main at spawn time is `OldName`; workers should use `OldName`
  until PR #N merges, then migrate."
- A spec that uses the new name as if it already exists on main MUST be flagged
  SPAWN-CAUTION; a worker following this spec will fail to find the symbol.

**Fix pattern:** Replace bare `NewName` references with "currently `OldName` on main
(rename to `NewName` pending PR #N / CSI-N)." Worker prompts should be updated to
include the pending-rename disclosure.

**W-Meta-Opus precedent:** CSI-15 / PP-5 A− — PP-5 §0 cited "formerly
`CamPqIndexPlaceholder` until CSI-15 rename in PR #390" — this is the CLEAN reference
pattern. W-Meta-Opus verified: `grep CamPqIndexPlaceholder witness_corpus.rs` returned
a match on main (still the old name); `WitnessIndexHashMap` existed only on branch
`claude/sprint-12-wave-g-fleet`. PP-5 was flagged A− precisely because this disclosure
was present and correct. A spec that cited `WitnessIndexHashMap` as canonical would
have been SPAWN-CAUTION.

**Promotion track:** Fire count = 0 actual fires (PP-5 was the CLEAN example; the
DIRTY pattern did not occur in sprint-13). Watch list for sprint-14+. If a dirty
instance fires once, track; two fires → promote to Option B worker-template-v2 item.

---

### PD6 — Spec-Internal-Contradiction

**Verification axis:** Axis 2 (ID collision)

**Symptom:** A single sentence or paragraph within one planner's output makes
mutually inconsistent claims about a D-id's owner, purpose, or associated worker.
The sprint-13 benchmark: three errors in one sentence.

**Verification command:**
```bash
# Extract the D-id claim and compare to primary plan table:
grep -n 'D-CSV-16' /home/user/lance-graph/.claude/specs/pr-sprint-13-rayon-streams.md
grep -n 'D-CSV-16' /home/user/lance-graph/.claude/plans/cognitive-substrate-convergence-v3.md
# For the owner claim, check the named planner's actual output file:
ls /home/user/lance-graph/.claude/knowledge/iron-rules-doctrine.md  # is PP-2 really splat?
ls /home/user/lance-graph/.claude/specs/pr-sprint-13-think-methods.md  # PP-4 is splat
```

**The rule:** Every D-id cross-reference in a planner spec must be consistent with
the primary plan document's §11 D-table. Owner is the planner assigned in that table;
purpose is the description in that table. Any discrepancy in a cross-reference is an
internal contradiction that needs pre-spawn repair.

**Fix pattern:** Look up the D-id in the primary plan §11 D-table. Replace the
contradictory cross-reference sentence with: "D-CSV-N (assigned to PP-M:
`<actual-purpose>`, per `.claude/plans/<primary-plan>.md §11`)."

**W-Meta-Opus precedent:** CSI-19 / PP-3 grade B — PP-3 §0: "D-CSV-17 — chosen
to not collide with D-CSV-13/14/15, and to follow the D-CSV-16 slot reserved by PP-2
(sprint-13 splat on-Think method migration)." Error 1: PP-2 is iron-rules-doctrine,
not splat. Error 2: PP-4 is splat on-Think, not PP-2. Error 3: D-CSV-16 is CAM-PQ
(PP-5), not splat. W-Meta-Opus: "three errors in one sentence. Implementation
correctness is independent of this; ID coordination is what failed."

**Promotion track:** Fire count = 1 (PP-3 in sprint-13 wave). One more fire →
promote to Option B worker-template-v2 §3 item (worker must verify D-id cross-refs
against primary plan table before reporting DONE on spec writing).

---

### PD7 — Iron-Rule-On-Wrong-Branch

**Verification axis:** Axis 6 (iron-rule citation drift)

**Symptom:** A spec cites an iron rule by ID in a "rules in force for this
implementation" context, but the rule is not in `CLAUDE.md` on main at the time
of the audit (it exists on a PR branch or in a planning doc only).

**Verification command:**
```bash
# The definitive check — run for each iron rule cited by the fleet:
grep -n 'I-LEGACY-API-FEATURE-GATED' /home/user/lance-graph/CLAUDE.md
# If empty, check where it IS:
grep -rn 'I-LEGACY-API-FEATURE-GATED' /home/user/lance-graph/.claude/ | head -10
# Confirm the iron-rules-doctrine pending annotation:
grep -n 'pending' /home/user/lance-graph/.claude/knowledge/iron-rules-doctrine.md
```

**The rule:** An iron rule is canonical when and only when it appears in
`CLAUDE.md §Substrate-level iron rules` on the main branch. A rule that appears
only in a planning doc, a spec, or on a PR branch is PENDING. Specs may reference
pending rules for context but MUST NOT treat them as enforcement gates.

**Fix pattern:** For each bare iron-rule cite in a spec, run the verification
command. If the rule is pending, add the qualifier "(pending PR-#N merge; advisory
only until landed on main)." If the spec uses the pending rule as a hard gate that
rejects an implementation option, flag SPAWN-CAUTION and note: "this rejection
is provisional pending rule canonicalization."

**W-Meta-Opus precedent:** CSI-20 — PP-6 SIMD spec line 8 cited "I-LEGACY-API-FEATURE-GATED"
in "iron rules in force" without the pending qualifier. Iron-rules-doctrine.md §6
had a note "pending sprint-13 ratification" but PP-6 did not inherit the qualifier.
The rule landed on PR #390. W-Meta-Opus flagged this as a P1 pre-spawn fix.

**Promotion track:** Fire count = 3 (PP-2, PP-6, PP-13 in sprint-13 wave). Already
at promotion threshold. Candidate for Option C SPAWN-GATE (pre-spawn mandatory: grep
every cited iron rule ID in CLAUDE.md main).

---

### PD8 — Missing-Integration-Responsibility

**Verification axis:** Axis 1 (stale-claim)

**Symptom:** A worker prompt or spec omits an explicit module-registration, lib.rs
inclusion, Cargo.toml dependency addition, or aggregation step that prior meta-reviews
have flagged as an orphan-prone pattern. The worker ships a file that compiles standalone
but is invisible to the rest of the workspace because it was never registered.

**Verification command:**
```bash
# Check if the new module file would be registered — look for lib.rs pattern:
grep -n 'pub mod' /home/user/lance-graph/crates/<crate>/src/lib.rs
# Check if worker-template-v2 §5 integration checklist is cited in the spec:
grep -n 'worker-template-v2\|lib\.rs\|pub mod\|integration responsibility' \
  /home/user/lance-graph/.claude/specs/<spec-file>.md | head -20
# For cross-repo: check ndarray mod.rs:
grep -n 'pub mod' /home/user/ndarray/src/hpc/stream/mod.rs
```

**The rule:** Every spec or worker prompt that creates a new `crates/<C>/src/<M>.rs`
MUST include an explicit instruction: "add `pub mod <M>;` to `crates/<C>/src/lib.rs`
in the same commit." This is worker-template-v2 §5.1 integration responsibility.
A worker prompt that lacks this instruction is PD8 drift; the worker will follow the
spec and create an orphan module.

**Fix pattern:** Add the §5.1 integration responsibility block to the worker prompt:
"Module registration: add `pub mod <module_name>;` to `crates/<crate>/src/lib.rs`
as part of this PR. Do NOT skip this step — an unregistered module creates a
compile-invisible orphan (CSI-8 pattern, sprint-11 Wave F)."

**W-Meta-Opus precedent:** CSI-13 / PP-8 A — CSI-13 in sprint-11 Wave F: worker
prompts said "main thread aggregates" for lib.rs registration, creating an invisible
coordination phase that nobody owned. CSI-8 specifically: `AttentionMask` and
`AttentionMaskActor` not registered in `cognitive-shader-driver/src/lib.rs`.
PP-8 worker-template-v2 §5.1 closes this by making registration the worker's
explicit responsibility. PD8 fires when a worker prompt omits §5.1.

**Promotion track:** Fire count = 2+ (CSI-8 + CSI-13 in sprint-11/12). Partially
promoted: PP-8 worker-template-v2 §5.1 is the Option B promotion. PD8 now fires
specifically when a new spec omits the §5.1 cite — verifying the fix was applied.

---

### PD9 — TD-Placeholder-Consumed-As-Resolved

**Verification axis:** Axis 3 (pending-vs-canonical)

**Symptom:** A sprint-N plan cites a tech-debt placeholder (TD-*) or deferred item
as resolved in sprint-N+1 planning without verifying that the actual resolution
commit landed on main.

**Verification command:**
```bash
# Look up the TD entry in TECH_DEBT.md:
grep -n 'TD-SIGMA-TIER-THRESHOLDS-1\|TD-COLLAPSE-GATE' \
  /home/user/lance-graph/.claude/board/TECH_DEBT.md | head -10
# Verify resolution commit is in main:
cd /home/user/lance-graph && git log main --oneline | grep -i 'sigma\|jirak\|collapse' | head -5
# Check LATEST_STATE.md for the resolved marker:
grep -n 'TD-SIGMA-TIER-THRESHOLDS-1' /home/user/lance-graph/.claude/board/LATEST_STATE.md
```

**The rule:** A TD-* entry is resolved when the resolution commit appears in
`git log main --oneline` AND TECH_DEBT.md has a "RESOLVED" annotation with the
commit hash. A planner citing a TD entry as resolved based on its own memory of
"we fixed this" without the git-log verification is PD9 drift.

**Fix pattern:** For each TD citation in a planner spec, run the verification command.
If the TD is not resolved in main, change the spec language from "resolved" to
"pending resolution (TD-* in TECH_DEBT.md, resolution commit <hash> on branch
<branch>; verify against main before treating as canonical)."

**W-Meta-Opus precedent:** Sprint-12 close handover pattern — several TD-* entries
from sprint-11 were carried forward as "pending" in PP-7 sprint-12 meta-review
correctly. PP-10 LATEST_STATE refresh had a slight drift on ndarray streams: "productization
sprint-12" when ndarray master had no stream module yet. The same class of drift:
treating an in-flight item as shipped.

**Promotion track:** Fire count = 1 (PP-10 ndarray streams claim in sprint-13 wave).
One more fire → promote to Option B worker-template-v2 §3 item (worker must verify
TD status in main before citing TD as resolved in spec).

---

### PD10 — Plan-Board-Inventory-Drift

**Verification axis:** Axis 1 (stale-claim)

**Symptom:** A planner spec proposes a new type, module, or contract item that
`LATEST_STATE.md § Current Contract Inventory` already records as shipped. The
worker spawned on this spec will spend time "implementing" something that already
exists, potentially creating a duplicate type or a conflicting implementation.

**Verification command:**
```bash
# Check LATEST_STATE.md for the proposed type name:
grep -n '<ProposedTypeName>' /home/user/lance-graph/.claude/board/LATEST_STATE.md
# Cross-check in the actual crates:
grep -rn '<ProposedTypeName>' /home/user/lance-graph/crates/ | head -10
# Also check PR_ARC_INVENTORY.md for recent additions:
grep -n '<ProposedTypeName>' /home/user/lance-graph/.claude/board/PR_ARC_INVENTORY.md | head -5
```

**The rule:** Before proposing any new type, contract item, or module in a spec,
the planner must have read `LATEST_STATE.md § Current Contract Inventory`. A spec
that proposes something already in the inventory is either a rediscovery (the planner
didn't read the inventory) or a deliberate extension (which must be stated explicitly:
"extending existing type X, not creating a new one"). CLAUDE.md § The Stance: "Proposing
a type that already exists is a 30-turn rediscovery tax — check first."

**Fix pattern:** Mark the proposed item as "already exists — use existing type X per
LATEST_STATE.md §Contract Inventory; do not create a duplicate." If the spec intends
to EXTEND the existing type, change the language to "extending `ExistingType` with
new field Y per §N ratification."

**W-Meta-Opus precedent:** No direct CSI in sprint-13 wave, but LATEST_STATE.md
§Contract Inventory exists precisely because the rediscovery problem recurred across
multiple sessions. PP-10's mandate (refresh LATEST_STATE) is the prevention mechanism.
PD10 fires whenever a spec proposes something already in the inventory without the
extension qualifier.

**Promotion track:** Fire count = 0 confirmed in sprint-13 wave (PP-10 refresh
prevented it). Watch list for sprint-14+. First fire → document. Second fire →
promote to Option B worker-template-v2 §3 item.

---

## §3 Severity Convention

| Severity | Meaning | Spawn action | PD examples |
|----------|---------|--------------|-------------|
| **P0 / SPAWN-BLOCKED** | Drift that would cause a worker to implement against a false ground state, creating work that must be thrown away or that would actively corrupt the workspace | Do NOT dispatch any worker until drift is repaired | PD1 (CSI marked resolved when not), PD4 (false PR merge), PD6 (spec-internal contradiction that would send worker to wrong owner) |
| **P1 / SPAWN-CAUTION** | Drift that is documented and visible but not immediately fatal; workers can proceed if the caveat is explicitly included in their prompts | Augment worker prompts with the drift caveat; dispatch with caution | PD2 (D-id undercount in one planner), PD3 (pending iron rule without qualifier), PD7 (iron rule on branch), PD5 (renamed symbol without disclosure) |
| **P2 / SPAWN-CLEAR with note** | Drift that is minor, has no downstream worker impact, or is a planning artifact that does not affect impl | Flag in audit report; dispatch normally; note for next preflight calibration | PD8 (missing integration step in one spec; worker-template-v2 covers it), PD10 (inventory check advisory) |
| **SPAWN-CLEAR** | No drift found on this axis/pattern | Dispatch normally | (All axes pass) |

The SPAWN-BLOCKED / SPAWN-CAUTION / SPAWN-CLEAR verdict in the agent card §3 output
maps from this P0/P1/P2/CLEAR table. P0 → SPAWN-BLOCKED. P1 → SPAWN-CAUTION.
P2 or CLEAR → SPAWN-CLEAR (with notes). Any single P0 finding blocks the entire fleet.

---

## §4 When NOT to Spawn This Agent — Anti-Trigger List

**Do NOT invoke preflight-drift-auditor when:**

- **Post-impl code review:** the sprint workers have already shipped Rust code and
  you need a code quality gate. Route to **PP-13 brutally-honest-tester** (clippy,
  audit, AP1..AP8 anti-patterns on the diff).
- **Cross-crate DTO boundary mismatch during impl:** two workers shipped incompatible
  DTO shapes and you need a boundary reconciliation. Route to **PP-15 baton-handoff-
  auditor** (cross-crate API contract verification on running code).
- **Positive expansion ideation before any planning:** you want to propose new
  0-friction implementation boundaries before planners write specs. Route to
  **PP-14 convergence-architect** (pre-plan divergent expansion + boundary proposal).
- **Single-planner sanity check with no parallel fleet:** only one spec was written
  and there are no cross-planner ID collision risks. Run the W-Meta-Opus manual pattern
  directly (grep the cited files, run the cited git commands) rather than spawning the
  full preflight agent — the agent overhead is only justified for fleets of N ≥ 3
  planners where cross-planner consistency is a real risk.
- **Post-sprint retrospective or grade writing:** you are grading completed work, not
  gating a spawn. Route to W-Meta-Opus (main-thread Opus post-commit review) instead.

---

## §5 Workflow Integration — The PRE-SPAWN Slot

```
[PP-14 convergence-architect]         [PRE-PLAN, divergent expansion]
           |
           v
[N Opus planners run in parallel]     [PLANNING phase — W-Meta-Opus monitors]
           |
           v
[planners report DONE: N spec docs]
           |
           v ◄── preflight-drift-auditor fires HERE
[SPAWN GATE: Axes 1-6, PD1-PD10]     [PRE-SPAWN — this agent]
           |                |
      BLOCKED           CAUTION → augment worker prompts
           |                |
     fix + re-audit    ┌────┘
           |            |
           └────────────┘
           |
           v
[N Sonnet impl workers dispatched]
           |
           v ◄── PP-15 baton-handoff-auditor fires HERE
[DURING-IMPL: cross-crate DTO/API]    [PP-15 boundary gate]
           |
           v ◄── PP-13 brutally-honest-tester fires HERE
[POST-IMPL: Rust code review]         [PP-13 pre-commit gate]
           |
           v
[commit + push]
           |
           v ◄── W-Meta-Opus fires HERE
[POST-COMMIT: per-wave meta-review]   [cross-spec CSI ledger, grades]
```

**Arrow connections to siblings:**
- **PP-14 convergence-architect** feeds INTO planners; planners feed into this agent.
  PP-14's "0-friction boundary" proposals become the claims this agent verifies.
- **PP-15 baton-handoff-auditor** catches the CODE-LAYER manifestation of drift that
  this agent catches at the PLAN layer. PD-patterns in this doc that "survive" to
  impl show up as baton-handoff failures — track the correlation.
- **PP-13 brutally-honest-tester** catches the CODE-QUALITY fallout AFTER impl.
  Planner drift caught here is cheaper than code drift caught there.
- **W-Meta-Opus** is the post-commit independent honest review. This agent reduces
  the W-Meta-Opus finding count by automating the pre-spawn checks W-Meta-Opus would
  have caught anyway. The goal: W-Meta-Opus reports "no CSI drift found in Wave N;
  preflight-drift-auditor caught it all pre-spawn."

---

## §6 Maintenance Protocol

This knowledge doc is **APPEND-ONLY within the PD catalogue** (§2). When a new
preflight run surfaces a drift pattern that doesn't fit PD1..PD10:

1. **Triage** in the relevant sprint-log meta-review (Wave N+ CSI-N) — does it fit
   an existing PD? If yes, update the fire count. If no, continue.
2. **New PD entry** — append PD11 (or next number) with the full eight-field format.
   Do NOT renumber existing entries; append only.
3. **Promotion tracking** — if the new pattern fires N ≥ 3 across distinct preflight
   runs AND has at least one SPAWN-BLOCKED instance AND has a confirmed copy-paste
   verification command, run the graduation ceremony in `preflight-drift-auditor.md`
   §6.

**Cross-reference to baton-handoff-anti-patterns:** when a PD-pattern fire at the
plan layer corresponds to a boundary-leak find later at the code layer, add a
`**Baton correlation:** PD-N plan-layer → <baton-handoff pattern name> code-layer`
annotation to the PD entry. This correlation is the leading indicator that the
pattern is load-bearing enough for permanent SPAWN-GATE promotion.

**Sprint-to-sprint carry-forward:** at the end of each sprint, the W-Meta-Opus
review should update the "Fire count" in each PD entry. The promotion thresholds
apply across sprints, not within a single sprint. A pattern that fires 3× in one
sprint is suspicious (the fleet may be systematically broken); a pattern that fires
once per sprint over 3 sprints is a genuine recurrence worth promoting.

---

## §7 Cross-References

- **`.claude/agents/preflight-drift-auditor.md`** — the agent card that runs this
  catalogue. §2 there has the one-line PD summaries; §1 has the six verification axes.
- **`.claude/board/sprint-log-13/preflight-meta-review-opus.md`** — primary source for
  every PD pattern; CSI-9/19/20/21/22/23 are the direct precedents.
- **`.claude/board/sprint-log-12/meta-review.md`** — CSI-7..18 baseline (Wave G
  cross-cutting findings).
- **`.claude/board/sprint-log-11/meta-review-opus.md`** — CSI-1..6 root observations
  (Wave F honest review); PD1 traces to CSI-9 here.
- **`.claude/agents/brutally-honest-tester.md`** (PP-13) — POST-impl code gate;
  runs AP1..AP8 on Rust diffs. Complementary: this doc covers PLAN drift; that doc
  covers CODE quality.
- **`.claude/agents/convergence-architect.md`** (PP-14) — PRE-plan expansion; proposes
  0-friction implementation boundaries that planners operationalize.
- **`.claude/agents/worker-template-v2.md`** (PP-8) — §5 integration responsibility
  checklist; PD8 fires when a spec omits the §5.1 module-registration step.
- **`.claude/knowledge/iron-rules-doctrine.md`** (PP-2) — canonical iron-rule list;
  Axis 6 and PD3/PD7 verification runs against §6 of this doc.
- **`.claude/board/LATEST_STATE.md`** — current contract inventory; Axis 1 and PD10
  verification source.
- **`.claude/board/PR_ARC_INVENTORY.md`** — APPEND-ONLY PR ledger; Axis 4 and PD4
  cross-check source.
- **`CLAUDE.md §Substrate-level iron rules`** — the four canonical iron rules; Axis 6
  and PD3/PD7 primary source.
- **`.claude/knowledge/codex-p1-anti-patterns.md`** — sibling knowledge doc for PP-13;
  AP1..AP8 patterns for post-impl code review. Structurally mirrors this doc.

---

## §8 One Sentence That Should Survive Any Refactor

**A spec that cites a state it did not verify has already started drifting; this
catalogue is the ledger of what the working tree will say when you finally ask it.**

---

*Authored W-Sprint-13-PP-16 (Sonnet 4.6 worker, 2026-05-16). Primary source:
`.claude/board/sprint-log-13/preflight-meta-review-opus.md` W-Meta-Opus audit.*
*Mirrors structure of `.claude/knowledge/codex-p1-anti-patterns.md` (PP-13 sibling).*
