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
> **`rehydrations / distinct_datasets_accessed`**.
> A value **> 1.0** means at least one dataset was evicted and re-fetched
> *within its own working set* — the definition of thrash.
> A **second hydration of the same dataset inside one age-floor window** is the
> same finding at single-dataset granularity, and is the sharper signal.

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
| T7 | **Dirty is DETECTED** — a mutation makes the version differ and the dataset reads dirty; **and an unmutated dataset reads clean after a full sweep** | the detector discriminates rather than always-firing or never-firing |
| T8 | **In-flight read is skipped** — a dataset with a read in flight is not flushed; **and the same dataset IS flushed once the read completes** | the cheap check discriminates in both directions (not always-skip, not never-skip) |
| T9 | **A LOST race does not corrupt** — force the interleaving (read begins mid-flush) and assert the reader gets a **correct, complete dataset** via rehydration. The cost may be a wasted rehydration; the result may **never** be a torn or partial read | §5's actual bar: *does not corrupt*, NOT *cannot occur* — this test deliberately makes the race happen rather than proving it impossible |
| T10 | **Rehydrate is byte-identical** — flush → rehydrate → read equals the pre-flush read | the round trip is lossless |
| T11 | **Thrash detector CAN fire** — a synthetic access pattern designed to thrash produces a ratio > 1.0; **and a well-behaved pattern produces ≤ 1.0** | §7's metric discriminates — a detector that fires on everything carries no information |
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
5. **Interaction with the push-back direction.** `dirty → hydrated` is the
   expensive direction and is currently an operational step. Whether a sweep
   may *initiate* a push-back (making eviction possible) or only skip dirty
   candidates is undecided; the plan currently assumes **skip**, which is the
   conservative choice and possibly the wrong one for the target workload.

## Cross-refs

`.claude/knowledge/s3-hydration-lifecycle.md` (the lifecycle, the three-layer
model, the reported measurements) · `.claude/knowledge/zero-copy-lens-law.md`
(why the local copy is not optional — the mapped bytes that law depends on are
what §5's "does not corrupt" bar protects) · `I-LEGACY-API-FEATURE-GATED` (§6's
same-name-different-semantics constraint).
