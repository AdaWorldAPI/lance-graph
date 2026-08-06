# Idle-flush dataset eviction — plan v1

> **Status:** PROPOSAL. Nothing here is implemented; nothing here is measured.
> **Scope:** design + acceptance criteria for a feature-gated local-copy
> eviction policy over Lance datasets. **This plan does not authorize the
> implementation** — it states what the implementation would owe.
>
> **Prerequisite reading:** `.claude/knowledge/s3-hydration-lifecycle.md` — the
> three-layer model (object store hydrates / local dir stores / volume only
> decides whether hydration repeats), the four lifecycle states, and the one
> reported measurement this plan's cost model rests on. This plan **extends**
> that lifecycle with an automatic `hydrated → flushed` trigger; it restates
> none of it.

## 0. Evidence grading (workspace rule: label everything)

| claim | status |
|---|---|
| The four-state lifecycle and its legal transitions | **FINDING** (mechanism) — see the knowledge doc |
| Rehydration of a tens-of-MB dataset is ~1.4 s | **reported measurement**, single observation, provider- and region-dependent, not re-run here |
| The default policy: age floor **3 days** + soft budget **~300 MB**, pressure-driven and age-ordered (§2) | **OPERATOR-SET POLICY** — a heuristic starting point, explicitly **not measured**. Both are config; these are defaults. |
| Age-ordering (rather than a `size × idleness` key) is the right default (§2) | **CONJECTURE** — argued (rehydration cost is also size-proportional), not measured; the size-weighted variant is deferred, not rejected |
| A watermark-driven sweep dominates both a bare timer and a bare allocation-failure signal (§3) | **CONJECTURE** — argued from the two failure modes, no deployed instance |
| The Lance dataset version is a sufficient dirty-detector (§4) | **CONJECTURE**, with a named verification gate that must close before implementation |
| At a 3-day floor the flush/read race is negligible, so check-then-act suffices and a lease protocol is disproportionate (§5) | **OPERATOR-SET SCOPE RULING** — the requirement is *does not corrupt*, not *cannot occur*; revisit if the threshold drops to hours |

**No probe has run for any row marked CONJECTURE.** The falsifiers are §7.

## 1. What this buys — cost SMOOTHING, not capacity

**The win is the shape of the bill, not a capacity ceiling.** This is not a
mechanism for fitting a working set into a disk that is too small, and the plan
does **not** justify itself with "otherwise you run out of disk." It exists so
that datasets nobody is reading stop being paid for continuously.

The economics are structural: **local disk is billed continuously for capacity
provisioned; object storage is billed for what is kept.** A dataset touched once
and then retained forever pays the continuous rate for a one-time access.
Eviction flattens that curve — a one-time transfer plus storage-at-rest, instead
of a standing charge for bytes nobody reads. The population this targets is
large single-use material: one-off corpora, rebake inputs, derivations touched
once and never again.

### The cost model is incomplete as stated, and the omission has a direction

**Raised by review on PR #901, and correct.** "Object storage is billed for what
is kept" reduces the object-store side to **bytes at rest**. Real object stores
bill several further dimensions, and every one of them is charged on the side of
the ledger this policy *increases*:

- **request count** — each hydration is many requests, not one, since a dataset is
  a multi-file directory;
- **retrieval** — non-hot storage classes bill per byte retrieved, separately from
  storage;
- **data transfer / egress** — charged when the read crosses a boundary the
  provider prices;
- **storage-management features** — inventory, versioning, lifecycle rules and
  similar, where enabled.

So the honest form of the argument is **not** "storage-at-rest is cheaper than
provisioned disk" but: *the retained-byte saving must exceed the request +
retrieval + transfer cost of the rehydrations the policy causes.* Both sides scale
with **how often eviction is paid**, which is exactly what §7's thrash criterion
measures — the two sections are the same question asked as money and as latency,
and it is worth noticing that the plan already gates on the right quantity even
though the first draft priced only one term of it.

**The assumptions this leaves standing, named so they are checkable rather than
implied:** a single deployment and a single storage class, both unnamed and
neither varied; a hot/standard-tier assumption (a colder class trades storage
against a retrieval charge and can invert the conclusion); and no measured
request-count-per-hydration for a representative dataset. **The 3-day and ~300 MB
defaults are not derived from any of this** — §0 already grades them
OPERATOR-SET, and this subsection is the reason that grading must not drift toward
FINDING: the cost model they would have to be derived *from* is not complete
enough to derive them.

Two consequences of the framing, both binding on the design:

- **Never fail an operation to hold the number.** The budget is a soft
  watermark (§2); there is no admission control, because a smoothing mechanism
  that breaks a workload has traded a billing improvement for an outage.
- **The trade is explicit:** a flushed dataset's next access costs a
  rehydration (~1.4 s at the reported tens-of-MB scale — single observation,
  §0). Worthwhile exactly when access is sparse enough that this is paid rarely.
  When it is paid often the policy is *worse* than doing nothing, which is why
  §7's thrash falsifier is a gating acceptance criterion rather than an
  afterthought.

## 2. The shipped default policy (operator-set) — answers Q1 and Q2

**Two conditions, BOTH required before anything is evicted:**

| condition | default | is |
|---|---|---|
| **Age** — dataset idle since last use | **> 3 days** | a flush *candidate* |
| **Budget** — total local footprint across all datasets | **> ~300 MB** | eviction *engages at all* |

**The trigger is pressure-driven and age-ordered.** Under the budget, **nothing
is ever evicted, no matter how stale** — a small deployment pays nothing, not
even churn. Over the budget, the **stalest candidates go first** until the
footprint is back under.

**Both numbers are configuration; these are the defaults.** They are
**operator-set, not measured** — grade them as policy, never as findings. `~300 MB`
in particular is a heuristic starting point.

### "Soft spot" is load-bearing

**~300 MB is the point where pressure BEGINS, not a hard cap that must never be
exceeded.** Three consequences, and each is a design constraint rather than a
nicety:

1. **No operation may ever fail because the budget is exceeded.** The sweep
   evicts what it can and carries on. There is no admission control, no
   back-pressure onto callers, no error path keyed to the watermark.
2. **A dataset actively in use, larger than the whole budget, stays resident.**
   **Correctness beats the watermark, always.** The budget cannot evict a live
   in-use dataset (§5), and the single-dataset-over-budget case is an ordinary
   steady state rather than a special case to handle.
3. **Eviction may therefore fail to reach the target**, when everything resident
   is in use or nothing is old enough. That is a **legitimate steady state, not
   an error** — see the observability requirement below.

### Q1 — what the age is measured from **ANSWERED**

**Age is time since last USE — read or write.** Either touch is evidence the
dataset is in the working set. (Dirtiness is a *separate* axis: a write also
makes the dataset dirty, and dirtiness governs whether flushing is **legal**
(§4, §7) — never whether it is **desirable**.)

**Ordering among candidates is by age: stalest first.** The 3-day floor is what
does the real work of keeping a hot working set intact, and it does so
*regardless of size* — which is why the default policy does not need a
size-weighted key.

> **Deferred refinement (CONJECTURE, not the default):** ranking candidates by
> `bytes_on_disk × idle_seconds` instead of age alone would evict the largest
> disk-seconds first and reach the budget in fewer evictions. It is **not** the
> shipped policy because it cuts both ways: rehydration cost is *also*
> proportional to size, so a size-weighted key preferentially evicts what is
> most expensive to get back. Age-ordering under a hard age floor is the
> conservative choice. Revisit only with measured access-pattern data — which
> nobody has.

### Observability — the requirement that keeps the soft watermark honest

**A deployment silently over budget must not look identical to one under it.**
The sweep therefore reports, every time it runs and reaches no target:

- current footprint vs budget, and **why the sweep stopped** — distinguishing
  **"no candidate was old enough"** from **"every candidate was in use"**. These
  are different operational situations with different responses and must not
  collapse into one "could not evict" line.

**The over-budget-but-nothing-stale case is INTENDED, not a bug.** If the
footprint exceeds ~300 MB but nothing has been idle for 3 days, **nothing is
evicted** — that is precisely the age floor protecting a hot working set from
being thrashed by budget pressure. What an operator sees in that case is the
first reason above: *over budget, zero candidates past the age floor.* If that
line persists, the working set genuinely exceeds the budget, and the response is
to raise the budget or reduce the working set — a **capacity finding**, not a
policy failure.

## 3. Q2 — who triggers the flush? **ANSWERED (watermark, per the policy above)**

Neither a bare timer nor a bare pressure signal. Both fail in a stated way:

| trigger | failure mode |
|---|---|
| bare timer | wakes and does work on datasets nobody is asking about, and pays the sweep cost even when disk is abundant |
| bare allocation-failure signal | only acts when it must — but by then it is **inside a request**, so the eviction *and* any subsequent rehydration are charged to a caller who is waiting |

**The design is a watermark-driven background sweep**, which is what the
operator policy selects. The sweep task exists but **does nothing while the
footprint is under budget** — that removes the bare timer's cost, because the
common case is a no-op check. Over budget, it evicts age-ordered candidates past
the floor, off the request path, until the footprint is back under **or it runs
out of candidates** (§2 — a legitimate stop, reported).

**A last-resort synchronous fallback on genuine allocation failure may exist,
but it is not part of the budget mechanism** and must carry its own counter.
Because the watermark is soft and never fails an operation, this path is a
disk-actually-full condition, not a budget condition — conflating the two would
smuggle admission control in through the back door.

## 4. Q3 — dirty detection without hashing **ANSWERED, with a verification gate**

**Do not hash.** Hashing 35 MB to answer a boolean is the wrong cost class and
scales with the thing being avoided.

**The mechanism is the Lance dataset version.** Lance datasets are versioned;
record `version_at_hydration` when the local copy is established, and define
**dirty ⇔ `current_local_version != version_at_hydration`**. This is an integer
comparison against a generation counter the storage layer already maintains for
its own reasons — no new bookkeeping, no content scan, and it is correct across
compaction and append alike because those are what bump it.

**mtime is a corroborating signal, never the authority.** It is cheap and can
serve as a fast pre-check, but it is not reliable enough across filesystems and
maintenance operations to gate a destructive action. If mtime and version
disagree, **version wins and the disagreement is logged** — a disagreement is
itself worth seeing.

> **VERIFICATION GATE (must close before implementation, not assumed):**
> confirm that the current local dataset version can be read **cheaply and
> without a full dataset open**, or that a local open is cheap enough to run per
> sweep candidate. If neither holds, this mechanism is a **BLOCKER**, not a
> design — and the honest response is to say so and stop, not to substitute a
> heuristic. Stating it plainly: *this plan assumes a cheap local version read
> exists; that assumption has not been checked against the API.*

## 5. Q4 — the flush/read race: cheap check-then-act, NOT a lease protocol

> **Scope correction (operator).** An earlier draft called this "the part most
> likely to be subtly wrong" and specified a refcounted guard type. **That is
> walked back as disproportionate.** *"Given 3 days not used it's just
> flattening the payment curve and hardly attributing to race conditions."*

**At a 3-day idle threshold the race is vanishingly rare.** A dataset untouched
for three days is not plausibly under an active mmap at the moment the sweeper
decides to flush it. The design must therefore be **safe if it happens**, not
**engineered around the possibility**.

**A lease / refcount / guard-type protocol was CONSIDERED AND REJECTED** — and
that is recorded here rather than left silent, so a future reader does not
re-add it believing it was overlooked. Reason: it puts a permanent cost (a guard
in every read signature, an atomic state-machine step, a second gate-off code
path per §6) on **every** read, to close a window that the age floor already
makes negligible. The protocol is priced for a hot cache; this is not one.

**What the design does instead — cheap check-then-act:**

- **Skip the flush if a read is in flight.** A cheap, non-authoritative check is
  sufficient; it does not need to be race-free, because losing the race is not
  harmful (below).
- **If a read begins mid-flush, let it rehydrate.** Do not block it, do not
  coordinate with the sweeper. The reader's own hydration path is already the
  recovery mechanism — `absent → hydrated` is idempotent and safe to repeat.
- **The failure mode must be RECOVERABLE, never CORRUPTING.** Worst case is a
  **wasted rehydration** (~1.4 s at the reported scale). Never a torn read,
  never a partially-visible dataset, never bytes removed from under a live
  mapping observed as corruption.

The bar the implementation must clear is therefore **"does not corrupt"**, not
**"cannot occur"** — and §8 asserts it that way.

> **Revisit condition, stated so the decision is falsifiable rather than
> permanent:** if the idle threshold ever drops from **days to hours**, the
> calculus changes — the race stops being negligible and the protocol question
> **reopens**. Tie any such threshold change to a re-read of this section.

## 5a. The atomic publish boundary — what actually makes §5's bar true

**Added after review on PR #901; four review comments converged on this gap and
they were right.** §5 rejects a lease protocol and asserts the worst case is a
wasted rehydration. That assertion **did not follow from anything §5 stated**, and
the reason is in the knowledge doc's §4a: a Lance dataset is a **multi-file
directory**, so "the reader can just rehydrate" is only a recovery when the reader
can tell *hydrated* from *half-hydrated* — and neither a partial fetch nor an
in-progress reclaim is distinguishable from a complete dataset by opening the
directory and looking.

Concretely, the interleavings §5 leaves open:

1. reader resolves the path → sweeper begins deleting → reader opens a directory
   that is losing files underneath it;
2. reader's hydration begins → sweeper's delete finishes → sweeper removes files
   the hydration just wrote, leaving a directory that is neither;
3. hydration fails mid-transfer → the partial directory is later treated as a
   present dataset rather than as debris.

None of these is a wasted rehydration. (1) and (2) are torn reads; (3) is a silent
wrong answer. The age floor makes them **rare**; it does not make them
**recoverable**, and §5's own bar is *does not corrupt*.

**The requirement (not a lease, and compatible with §5's rejection):**

> **Hydrate aside; publish by rename; retire by rename.** A hydration fetches into
> a private temporary directory and becomes visible by **one atomic directory
> rename**. A reclaim renames the published directory **away first**, then deletes
> the renamed copy. A reader resolves the published name **once**, at open, and
> holds what it resolved.

Consequences, and each is why this is the cheap answer rather than the protocol
§5 walked back:

- **The read path does not change.** No guard in a signature, no atomic
  state-machine step per read, no refcount, no lease, no second gate-off code path
  — the objections that priced out the protocol in §5 do not apply, because this
  costs the *sweeper* a rename and the *reader* nothing.
- **Every observable state is complete.** The published name is either absent or a
  whole dataset. Interleaving (1) and (2) degrade to exactly what §5 claimed:
  the reader sees *absent* and hydrates, at worst redundantly.
- **Failure debris is self-identifying.** An unpublished temporary directory was
  never visible, so a sweep may delete it with no coordination and no risk of
  removing live data — which closes (3) without a partial-state protocol.
- **`hydrated → flushed` gets its barrier for free.** The rename-away IS the
  transition; the dirty check happens before it, and after it there is nothing
  left to race against.
- **It does not fix multi-process.** A rename is atomic on one filesystem, so two
  processes over one directory see consistent *published* state — but the
  reclaim-then-rehydrate decision is still uncoordinated and can duplicate work.
  §9.3 stays open; the bar it now inherits is the honest one (duplicated work,
  not corruption), which is the claim §9.3 previously made without support.

**Assumption this rests on, stated rather than assumed:** directory rename is
atomic on the filesystem in use. That is true of ordinary local filesystems and is
part of the "supported, mmap-capable local filesystem" requirement (knowledge doc
§1); it is **not** guaranteed on the network-mount cases that requirement already
excludes. Same excluded set, one more reason.

## 6. Q6 — why a feature gate, and what "off" excludes **ANSWERED**

**Off by default.** A consumer with ample local disk must pay *nothing* — and
"nothing" is meant literally, not "a cheap timer".

With the gate off, the following do not exist in the binary: the sweep task and
its wakeups, the per-dataset size and last-read accounting, the eviction key
computation, the watermark checks, and the object-store push path used for
`dirty → hydrated`. The store opens the local path exactly as it does today.

**Because §5 rejects the guard type, the read path does not change shape with
the gate** — which removes the `I-LEGACY-API-FEATURE-GATED` hazard the earlier
draft had to design around. The in-flight check is a cheap, non-authoritative
read of the same bookkeeping the sweep uses; when the gate is off that
bookkeeping does not exist and neither does the check.

**The constraint that remains: the same function name must not mean different
things under different gate states.** Gate-off must be *inert*, never *subtly
different* — asserted by T12.

**Orthogonality note:** idle-flush *requires* the object-store feature (§3 of
the knowledge doc) but does not *imply* it. Two independent gates; enabling
idle-flush without the object-store provider must be a **build-time or
startup-time refusal**, never a runtime surprise on the first eviction — the
scheme-error diagnosis trap, one layer up.

## 7. Q5 — what it costs when wrong, and the falsifiers **ANSWERED**

Rehydration is ~1.4 s at the reported tens-of-MB scale (single observation). **A
thrashing policy converts that from a rare cost into a per-request cost** — the
policy then makes the system strictly worse than not having it, while appearing
to function.

**The thrash falsifier — the gating acceptance criterion:**

> Over a measurement window, compute
> **`eviction_caused_rehydrations / distinct_datasets_accessed`**.
> A value **> 0** means at least one dataset was evicted and re-fetched — and
> because the window is bounded to the age floor (below), that re-fetch is
> necessarily *within its own working set*, which is the definition of thrash.
> A **second hydration of the same dataset inside one age-floor window** is the
> same finding at single-dataset granularity, and is the sharper signal.

**The metric's definition is load-bearing, and the first draft's was not
usable — review caught three independent defects in it, all real:**

1. **The numerator counted the wrong events.** A bare `rehydrations` count also
   includes first hydrations, process restarts, failed-hydration retries, manual
   invalidation, and version-driven reloads. None of those is eviction, and a
   metric that cannot attribute cannot falsify. **Fix: the sweeper stamps an
   eviction generation on each dataset it reclaims, and only a hydration that
   finds such a stamp increments `eviction_caused_rehydrations`.** First
   hydrations are excluded by construction — there is no stamp to find.
2. **An unbounded window produced false alarms.** With a window longer than the
   age floor, a dataset can be accessed, go correctly idle past the floor, be
   correctly evicted, and later start a *new* working-set interval. That is the
   policy working exactly as designed, and it would have registered as thrash.
   **Fix: the window is bounded to the age floor.** Inside one floor-length
   window, a correctly-evicted dataset cannot legitimately be re-accessed —
   re-access within the floor is precisely what "the floor was too short for this
   dataset" means.
3. **The `> 1.0` threshold could not fire at the granularity that matters.** With
   the numerator correctly restricted to eviction-caused rehydrations, one
   thrashing dataset among many gives a ratio well below 1.0 while being exactly
   the condition the criterion exists to catch. **Fix: the threshold is `> 0` on
   the corrected numerator**, and the ratio is retained as a *severity* measure
   (how widespread), not as the trigger.

**Datasets already resident when the window opens** carry no eviction stamp, so
they contribute to the denominator (they were accessed) and not to the numerator
until this policy actually evicts them. That is the intended asymmetry: the metric
measures what the policy *did*, never what the deployment inherited.

Supporting measurement: **`total_hydration_seconds / total_read_seconds`**. This
is the amortization ratio; if hydration time approaches read time, the feature
is paying for itself with the thing it was supposed to make cheaper.

**Both must be instrumented before the policy is enabled anywhere**, or a
thrashing deployment is indistinguishable from a working one — the observation
that motivates this whole section.

## 8. Acceptance criteria — the tests the implementation owes

Per the workspace P0 falsifiability rule, **every guard owes a can-fire test AND
a can-stay-silent test, both on non-trivial inputs.** An eviction policy that
never evicts and one that evicts everything are equally useless and both pass a
naive assertion. Enumerated:

| # | test | what it proves |
|---|---|---|
| T1 | **Eviction CAN fire** — over budget **and** a past-the-floor stale candidate present ⇒ something is evicted | the policy acts, and only when BOTH conditions hold |
| T2a | **Silent UNDER BUDGET** — footprint under the budget with a **very stale** dataset present ⇒ **nothing** is evicted | staleness alone must NOT evict; a policy that evicted on staleness alone would pass T1 and be wrong |
| T2b | **Silent OVER BUDGET, all fresh** — footprint over budget but no candidate past the age floor ⇒ **nothing** is evicted, and the sweep reports *zero candidates past the floor* | the age floor genuinely protects a hot working set, and the intended no-op is **observable** |
| T2c | **Discrimination within one sweep** — a within-floor dataset is spared *while* a past-the-floor one in the same sweep IS evicted | the policy discriminates; the paired candidate is what makes it non-vacuous (a silence test on an empty candidate set proves only that emptiness is handled) |
| T3 | **Age ordering is load-bearing** — among several past-the-floor candidates, the stalest is evicted first | ordering is by age, per the default policy |
| T4 | **Age-floor inertness** — raising the floor silences an eviction a lower value admits; lowering it admits one a higher value silences | the parameter is not decoration |
| T5 | **Budget inertness** — raising the budget above the current footprint silences a sweep that a lower budget engages; the sweep stops as soon as the footprint is back under | the budget is the trigger, and it is a threshold rather than decoration |
| T5b | **The budget is SOFT** — a single in-use dataset larger than the whole budget stays resident and **no operation fails**; the sweep reports *every candidate in use* | correctness beats the watermark; there is no admission control |
| T6 | **`dirty → flushed` is REFUSED** — a dirty dataset offered to the flush path is rejected, not silently accepted | the destructive edge is checked, not assumed (the knowledge doc's one rule) |
| T6b | **A dirty candidate is SKIPPED and SAID** — a past-the-floor, over-budget, dirty candidate is not reclaimed, no push-back is attempted, and the sweep reports *dirty* as a **distinct** stop reason (not folded into "in use" or "none old enough"); **and the same candidate IS evicted once clean** | §9a's clean-eviction-only decision, in both directions. Without the silence half the sweep could be refusing everything; without the report half a permanently-stuck deployment is invisible |
| T7 | **Dirty is DETECTED** — a mutation makes the version differ and the dataset reads dirty; **and an unmutated dataset reads clean after a full sweep** | the detector discriminates rather than always-firing or never-firing |
| T8 | **In-flight read is skipped** — a dataset with a read in flight is not flushed; **and the same dataset IS flushed once the read completes** | the cheap check discriminates in both directions (not always-skip, not never-skip) |
| T9 | **A LOST race does not corrupt** — force **each** of §5a's three interleavings (read resolves then reclaim begins; hydration overlaps a reclaim; hydration fails mid-transfer) and assert the reader observes the published name as **either absent or a complete dataset**, never a partial one. The cost may be a wasted rehydration; the result may **never** be a torn or partial read | §5's actual bar: *does not corrupt*, NOT *cannot occur* — this test deliberately makes the race happen rather than proving it impossible. Enumerating the three interleavings is what stops it degenerating into a single easy one |
| T9b | **Publish and retire are atomic at the name** — a reader that resolves the published name mid-reclaim holds a complete dataset; and a failed hydration leaves **no** visible published directory (only unpublished debris a sweep may delete) | §5a's boundary is asserted rather than assumed — without this, T9 could pass by timing luck |
| T10 | **Rehydrate is byte-identical** — flush → rehydrate → read equals the pre-flush read | the round trip is lossless |
| T11 | **Thrash detector CAN fire** — inside one age-floor window, a synthetic pattern that re-accesses an evicted dataset produces `eviction_caused_rehydrations > 0`; **and a well-behaved sparse pattern — accessed, correctly idled past the floor, evicted, then re-accessed in a LATER window — produces `0`** | §7's corrected metric discriminates. The silence half is deliberately the case the first draft's definition got wrong: normal sparse usage must NOT read as thrash |
| T11b | **Attribution is real** — a restart, a failed-hydration retry and a manual invalidation each produce a hydration that does **not** increment `eviction_caused_rehydrations` | the numerator counts eviction, not hydration; without this the metric is a hydration counter wearing a thrash label |
| T12 | **Gate-off is inert** — with the feature off, no sweep runs, no accounting is kept, and the read path is unchanged (per §6, either one code path or a proven-equivalent second one) | the gate costs nothing when off |

**T2a/T2b, T7, T8 and T11 are the ones that matter most** — each is a paired
fire/silence test on the exact guard whose degenerate always-on or always-off
form would otherwise pass review. **T2a is the sharpest:** a policy that evicted
on staleness alone would pass T1 and still be wrong.

**T9 is deliberately shaped as a corruption test, not an impossibility proof.**
Asserting the race "cannot happen" would be exactly the vacuous assertion the P0
rule forbids — implied by the code, falsifiable by nothing.

## 9. Open items (explicitly NOT answered)

1. **The §4 verification gate.** Cheap local version read — assumed, unchecked.
   Closing this is the first task; if it fails, the plan needs a different
   dirty-detector and this document is wrong rather than incomplete.
2. **Whether the default values are right.** The 3-day floor and ~300 MB budget
   are operator-set starting points, not derived from a measured access-pattern
   distribution. They are config precisely because that distribution is unknown;
   the deferred size-weighted ranking (§2) should be revisited only once it is.
3. **Multi-process access to one local directory.** The in-flight check is
   in-process. Two processes over one directory is out of scope and would need a
   different mechanism — named here so it is not assumed handled. (Note it does
   not change §5's bar: the worst case stays a wasted rehydration.)
4. **Partial hydration.** Whether a subset (fragment / column range) can be
   hydrated instead of a whole dataset is unexplored. It would change the size
   term of the eviction key, so it is a policy question, not just an I/O one.
5. ~~**Interaction with the push-back direction.**~~ **DECIDED after review on
   PR #901 — see §9a below.** It was recorded here as undecided while the board
   summaries described push-back as part of eviction; that contradiction is
   resolved in favour of **skip**, and the open item is now only the narrower
   question of whether a *separately triggered* push-back should exist.

### 9a. The sweep NEVER initiates push-back — clean eviction only

**Review on PR #901 found the plan and its two board summaries disagreeing** about
whether a dirty candidate is skipped or pushed back first. Those are different
data-integrity contracts, so the disagreement is resolved here rather than left to
the reader.

**The decision: the sweep performs CLEAN EVICTION ONLY.** A dirty candidate is
**skipped** — reported, never reclaimed, never pushed. Push-back is a **separate
operation with its own trigger**, and the sweep does not invoke it.

Two mechanisms, deliberately not one:

| operation | what it does | who triggers it |
|---|---|---|
| **clean eviction** | reclaims the local copy of a dataset that is *identical to the object store* | the watermark sweep (§3) |
| **push-back** (`dirty → hydrated`) | uploads a diverged local copy | an operator/operational step — **never the sweep** |

Why this and not the other choice — it follows from what the plan already says
rather than being a new preference:

- The knowledge doc's §4 rule is *flush is legal only from `hydrated`*. A sweep
  that pushes first would be **manufacturing** the precondition for its own
  destructive step, on a background timer, unattended. That inverts a safety
  check into a workflow.
- §4 of that same doc establishes push-back as the expensive,
  fragmentation-sensitive direction and rules it "an ops step, never a boot path".
  A background sweep is not a boot path, but it *is* unattended — which is the
  property that made it an ops step in the first place.
- A failed push mid-sweep leaves the only copy of diverged data in an ambiguous
  state, with nobody watching. Skipping has no such failure mode: the worst case
  is that the sweep reaches no target, which §2 already establishes as a
  **legitimate reported steady state**.

**Consequence to make observable:** a deployment whose footprint is dominated by
*dirty* datasets will sit permanently over budget with the sweep unable to act. §2
already requires the sweep to say *why* it stopped; **"every candidate was dirty"
is a third distinct reason** alongside "none old enough" and "every candidate in
use", and it must not be collapsed into either. It is the signal that a push-back
is owed — which is the correct place for a human to enter the loop.

**Remaining open question** (narrowed from the original item 5): whether a
separately-triggered push-back operation should exist in this feature at all, or
whether it belongs entirely outside it. Undecided; it does not block the sweep,
which never calls it either way.

## Cross-refs

`.claude/knowledge/s3-hydration-lifecycle.md` (the lifecycle, the three-layer
model, the reported measurements) · `.claude/knowledge/zero-copy-lens-law.md`
(why the local copy is not optional — the mapped bytes that law depends on are
what §5's "does not corrupt" bar protects) · `I-LEGACY-API-FEATURE-GATED` (§6's
same-name-different-semantics constraint).
