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


---

# MEASURED RESULTS — M-arm and O-arm (2026-08-05, release, 16 GiB free host)

Both arms produced **negative** results. Both were pre-registered as
two-sided, so both are findings rather than failures.

## M-arm — MORTON DOES NOT WIN under this workload/host

Digest identity **MATCHED** (`68128e3662df105c` both pipelines), so the
comparison is valid — the reorder changed layout, not semantics.

| phase | natural | morton |
|---|---|---|
| reorder | — | **9.4 ms** |
| seal | 11.6 ms | 15.6 ms |
| write | 257.3 ms | 254.3 ms |
| sync | 48.4 ms | 54.5 ms |
| T1 | 320.9 ms | 339.7 ms |

**SUM verdict (the pre-registered criterion): reorder_cost 9.4 ms,
downstream savings −25.8 ms (Morton is SLOWER downstream), Δtotal
= +35.2 ms ⇒ MORTON LOSES.** The ordered-chunk fast path
(350.9 ms) was also **slower than the generic path** (339.7 ms) while
producing an identical digest — so validate-and-append did not beat
regroup-and-sort here either.

> **⚠ CAVEAT THAT BLOCKS ONE COMPARISON (found by this run, not papered
> over).** The M-arm's T1 baseline is **320–340 ms**, roughly **4× A0's
> 78–86 ms** over the same nominal 1,048,576 rows. Until that gap is
> explained, the fast-path number **must NOT be compared against A0's
> 78–86 ms** — the two T1s are not commensurable. The natural-vs-Morton
> comparison IS valid (same harness, same run, same row count); only the
> cross-run comparison to A0 is void. Likely suspects: the M-arm's
> `BenchRow` materialisation inside the timed region, and the
> `stream_position` relabeling the harness needs because `freeze` always
> sorts by that field. **This is an open measurement defect, not a
> result.**

## O-arm — DIVERGED: the seal's ordering is LOAD-BEARING

Primary observable, computed and printed **before any timing** as
pre-registered: **O-A `64565f362db2e4a5` ≠ O-B `3e71c2aa7be8e325` —
DIVERGED.**

**Verdict:** ordering sourced from temporal replay does NOT reproduce the
seal's ordering. Under this construction the seal's ordering is
**load-bearing and cannot be re-scoped away** — which retires, for this
construction, the long-standing "temporal.rs already provides the
ordering" hypothesis. **Honest scope: this falsifies the hypothesis FOR
THIS O-B CONSTRUCTION; it does not prove no construction could match.**

**Firewall held** (after a real fix — see below): the region contains no
`scan_sealed` and no sealed-store read, and `local_trajectories` IS
present, so O-B's scan mechanism is proven live rather than absent.

**Kill-condition check: CONSTRUCTIBLE.** O-B's derivation
(`local_trajectories` grouping via BTreeMap) is a different code path
from O-A's seal-side `order_cycle_stably` Vec sort — not a disguised
O-A. Reported honestly: at one row per owner per cycle the two are doing
comparable asymptotic work, and the redundancy the plan asks about is
SEMANTIC, not code-sharing.

Timing (secondary): O-A cast 55.5 / seal 20.1 / commit 13.2 / T1 397.1 ms;
O-B cast 72.4 / **order_derive 64.9** / seal 34.9 / commit 5.1 /
T1 523.8 ms. O-B is slower on every phase except commit.

## Three defects caught at the gate (not shipped)

1. **The firewall fired on its own comment.** The self-scan matched the
   token inside a *comment* describing the check — a guard tripping on
   documentation tests the documentation, not the code. Fixed by
   stripping line comments before scanning, **plus a positive control**
   asserting the detector still finds a real call (otherwise a silent
   guard and a broken guard are indistinguishable).
2. **T1 read 18 cycles where the spec says 16.** Both arms scanned the
   unfiltered history, including the two warm-ups (1,179,648 rows vs
   1,048,576). Scoped to the measured window via
   `scan_sealed(Some(WARMUP))` — an unscoped T1 is not comparable to
   anything.
3. **A pre-registered outcome was coded as a panic.** O-arm divergence
   `assert!`-ed, which turns a designed falsification into a crash and
   discards every number after it. Both branches now report.
