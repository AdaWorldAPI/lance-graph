# measure-64k-axes v3 — the three arms Stage A0 earned (operator-directed, 2026-08-05)

> **Reads with:** v1 (arms + MEASURED RESULTS), v2 (the rolling-epoch-closure
> model). This file adds ONLY what the operator's review of the A0 results
> designated as next. Nothing here is measured.

## The takeaway that orders this file

A0 found the expensive part is **not** 64k owners and the unstable part is
**not** sealing — the instability lives in
`filesystem → page cache → writeback → allocator interaction`. So optimisation
effort belongs in **temporal chunk scheduling, Morton ordering, rolling
closure, and batch geometry** rather than in redesigning the ownership model.
**That is a hypothesis A0 makes worth testing, not a finding A0 proved.** The
three arms below are how it gets tested.

Sequencing (operator-directed): **M-arm and O-arm first, rolling closure
before any encryption, A-arm when the layout question comes back.**

---

## M-arm — MORTON REORDER: the experiment that answers the architectural question

A0 measured `logical order → seal → WAL`. It never measured the pipeline the
architecture actually proposes:

```
A0 (measured):   logical order              → seal → WAL
M-arm (new):     logical order → MORTON REORDER → seal → WAL
```

**Design.** Same 65,536 owners, same canonical frame, same host discipline;
the ONLY difference is a Morton reorder inserted before the seal
(`WriteOrderKey { morton_chunk, lane, cycle_position }` per v2 D1 — identity
stays on `MailboxId`). Measure the reorder as its OWN phase
(`morton_reorder_ns`, a new CSV column) so it is never folded into seal or
write time.

**Pre-registered readings, both directions named in advance:**
- **Reorder cost** — the reorder phase in isolation. It is a cost until proven
  otherwise; it must be paid for downstream or the arm is a KILL for Morton.
- **Downstream gain** — seal + write + temporal T1, ordered vs unordered.
- **The verdict is the SUM**, never the gain alone: `Δtotal = reorder_cost −
  (seal + write + T1 savings)`. Morton wins only if Δtotal < 0.
- **The temporal fast path is where the real gain should live** (v2 D2): with
  chunk headers proving version/chunk/lane/stream monotonicity, T1 becomes
  validate-and-append instead of regroup-and-sort. A0 measured T1 at
  **78–86 ms** over 1,048,576 rows — that is the number the fast path must
  beat, and it is stable enough across runs to be a real target.
- **Digest identity is mandatory**: ordered and unordered trajectories must
  produce identical digests, or the reorder changed semantics rather than
  layout, and the arm is void regardless of speed.

**Stability inheritance:** the A0 spread guard applies unchanged. If the WAL
phase is unstable in an M-arm run, the reorder's downstream half is
unreadable and only the reorder COST may be reported.

---

## O-arm — ORDERING SOURCE: where does the ordering actually come from?

The hypothesis that has been running through months of discussion —
*`temporal.rs` already provides the ordering* — has never been isolated. It
gets its own measurement:

```
O-A:  cast → seal → WAL → temporal replay      (today's pipeline)
O-B:  cast → temporal replay → seal → WAL      (ordering sourced first)
```

**What makes this decisive rather than merely interesting:** if O-B produces
**byte-identical trajectories** to O-A, then the seal's ordering work is
*redundant with* temporal's, and the seal can be re-scoped to closure +
batching + version publication alone (which is what A0 already showed it
fundamentally is — the seal was never cryptographic, and this asks whether it
must be ordering either). If the trajectories DIFFER, the hypothesis is dead
and the seal's ordering is load-bearing — equally valuable, and cheaper to
learn now than after a redesign.

**Pre-registered:**
- Primary observable: **digest identity** O-A vs O-B (a boolean, decided
  before any timing is looked at — timing must not be able to rescue a
  semantic difference).
- Secondary: per-phase time for both pipelines.
- **Kill condition:** if O-B cannot be constructed without duplicating
  ordering work that O-A does once, say so and report the arm as
  not-constructible rather than reporting a rigged comparison.
- **Firewall:** O-B must not consult the sealed stream to build its own order
  (that would be O-A wearing a disguise). Enforced by a compile-time self-scan
  in the probe, the pattern the shipped probes already use.

---

## A-arm — ALLOCATOR vs ARCHITECTURE (the confound A0 names but cannot split)

A0's L1a build delta of **−171 ms** is a SUM of at least four phenomena:
fewer allocation calls · better locality · allocator arena reuse · fewer cache
misses. Reporting it as one number is honest only while it is *labelled* as a
sum, which v1 now does.

**Decomposition design (deferred until the layout question returns):**
- **allocation count** — instrument an allocation counter (a counting global
  allocator behind a probe-local feature) and report calls, not just time.
- **arena reuse** — run each layout FIRST in a fresh process (the reuse
  A0 hit is why B1b's in-process RSS delta read 0). Separate processes, one
  arm each.
- **locality / cache misses** — needs perf counters; A0 emits `llc_misses`
  EMPTY by design rather than fabricating it. This sub-arm is BLOCKED on
  perf-counter access and must stay blocked rather than be estimated.
- **pure allocation cost** — a control that allocates the same shapes and does
  nothing else.

Until those run, the standing wording holds: *the chunked layout is faster to
build*, never *allocation is the cause*.

---

## What does NOT change

- Encryption stays out (v2 ⊘ D5): rolling closure is measured before any
  crypto, and the seal path remains verified crypto-free.
- The WAL knee stays unclaimed until a quiet host with headroom + O_DIRECT or
  a per-config cache barrier exists. No arm here weakens that.
- D-KIA-A2 stays frozen; EXP-KIA-A2-ROLLING-CLOSURE remains the non-claiming
  exploratory override.
- Every number keeps implementation-scoped wording: *this implementation,
  this workload, this host*.
