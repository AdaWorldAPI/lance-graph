# What the seal computes that `temporal.rs` does not encode

> **READ BY:** any session touching `persist_sink::{freeze, order_cycle_stably,
> DetachedCycleBatch}`, `temporal::{local_trajectories, local_trajectory_of,
> LocalCausalRow}`, the cycle-loop closure driver, or any proposal to "re-scope
> the seal" / "let temporal.rs provide the ordering".
>
> **Status:** the four differences below are **read off the shipped source** and
> are therefore FINDINGs about the code as it stands. The *consequences* for
> future design are labelled separately. Born from the O-arm digest divergence
> (`measure-64k-axes-v3.md`, measured 2026-08-05).

---

## The question this answers (operator-framed, 2026-08-05)

The O-arm asked "can ordering be sourced from temporal replay instead of the
seal?" and got **DIVERGED** — O-A `64565f362db2e4a5` ≠ O-B `3e71c2aa7be8e325`.
The operator's reframing is the one worth keeping:

> Not *"can we remove temporal ordering?"* but **"what information does the seal
> compute that `temporal.rs` does not currently encode?"** — because the digest
> divergence means one of them knows something the other doesn't, and that
> something is probably the real architectural asset.

The standing position that follows, and which this doc records as the workspace
default until measured otherwise:

- **`temporal.rs` = the authoritative TEMPORAL model** (what a reader may see;
  what one owner actually did, in its own order).
- **The seal = the authoritative ORDERING model** (one total order, one fold,
  one cohort, one published version).
- **The gap between them is an explicit research question, not a redundancy to
  be resolved by deleting one side.**

---

## The answer: four things, all of them absent from the temporal surface

### 1. A cross-owner TOTAL order — where temporal computes a PARTIAL one

`LocalCausalRow::cast_seq`'s contract says it outright
(`temporal.rs:400-402`):

> *"Cross-owner values are never compared; only rows sharing an `owner()` are
> ordered against each other."*

`local_trajectories` therefore produces a **forest of per-owner chains** — a
partial order globally. `DetachedCycleBatch::freeze` produces **one total
order** over all owners (`order_cycle_stably(&mut casts, |s| s.stream_position)`,
`persist_sink.rs:197`).

**A partial order does not determine a total order.** The O-arm divergence is
the expected signature of that difference, not evidence of a defect on either
side. Any future "let temporal source the ordering" proposal has to start by
supplying a globally comparable key — which means *widening the very contract
the deinterlace exists to keep narrow*.

### 2. Arrival as an ordering INPUT — recorded nowhere else, durably

The seal's key is `stream_position`, and its sort is **stable**, so equal keys
keep arrival order. In the measured harness `stream_position` IS the arrival
rank. Nothing in `temporal.rs` records arrival at all: `LocalCausalRow` is
exactly `(owner, cast_seq)`.

So **the seal is the only place cross-owner arrival enters the durable record**
— and once it is in the record, it *is* the durable fact that every later read
returns (`scan_sealed` is contractually forbidden from re-sorting:
`persist_sink.rs:292-293`, *"this seam NEVER sorts (order is a write-side
property, fixed before the append)"*; restated at `:315-317`). O-B cannot reproduce it
because the information is not in the per-owner projection it reads.

> **⚠ Scope, stated so it is not overread.** The O-arm deliberately scrambled
> arrival (bit-reversal of the owner id) precisely so the two orders were FREE
> to diverge — without that, every owner casts in ascending id order and the
> comparison would coincide vacuously. So the measured result says *the seal
> preserves an arrival order temporal cannot see*, **not** *the seal always
> disagrees with temporal*. On an arrival-ascending workload they would agree,
> and that agreement would prove nothing.

### 3. The per-row coalescing FOLD — a row concept temporal does not have

`freeze` also builds `image: BTreeMap<row, payload>` — *"the coalesced final
image: `row -> last payload in stream order`"* (`persist_sink.rs:185-186`,
`:198-201`). Which of N writes to one row survives is decided **by the
cross-owner total order**, and two different owners can write the same row.

`temporal.rs` has no row concept whatsoever. Last-writer-wins at row
granularity is computed **nowhere else in the system**. This is the most
concretely irreplaceable of the four: it is not an ordering that could be
re-derived, it is a *destructive fold* whose result depends on the ordering.

### 4. The cohort boundary and its read horizon

`CycleFrame { cycle, base_version }` (`persist_sink.rs:104-109`) carries two
facts the temporal surface has no field for:

- **Cohort membership** — which casts belong to the same atomic, all-or-nothing
  publication (one WAL append → one `DatasetVersion`).
- **The read horizon `base_version`** — *"the sealed predecessor every thought
  in this cycle reads (`Vn`)"*. This is the "what did this cohort see" fact, and
  it is epistemically load-bearing everywhere else in the stack.

A per-owner trajectory carries neither. Reconstructing "these 65,536 casts read
the same `Vn` and landed together" from per-owner chains is not a matter of
sorting harder; the grouping key is simply absent.

---

## The consequence for design (labelled: consequence, not measurement)

**The two mechanisms are not competing implementations of one function.** They
compute different mathematical objects:

| | seal (`freeze`) | `temporal.rs` (`local_trajectories`) |
|---|---|---|
| output | one total order over all owners | forest of per-owner chains (partial order) |
| ordering input | `stream_position`, stable ⇒ arrival breaks ties | `cast_seq`, per-owner only, cross-owner comparison forbidden |
| fold | yes — `row → last payload` | none (no row concept) |
| cohort | yes — `CycleFrame{cycle, base_version}` | none |
| question it answers | *what became durable, in what order, as one unit* | *what a reader may see; what ONE owner did* |

So "can one replace the other" was the wrong shape of question, and the O-arm's
value is that it **failed semantically before it failed on performance** — which
makes the performance numbers (O-B slower on every phase but commit) almost
irrelevant to the decision.

**The minimal change that would make temporal able to source the ordering** is
therefore not "make temporal smarter" but *"give `LocalCausalRow` a globally
comparable key"* — a contract widening that re-couples the owners the
deinterlace exists to decouple, and that would still not supply the fold (3) or
the cohort (4). Anyone proposing it owns that cost explicitly.

---

## Probe queue (falsifiable, none run)

- **PROBE-SEAL-TIE-DENSITY** — do cross-owner `stream_position` ties actually
  occur in a realistic cast pattern? If they do, the seal's total order depends
  on non-durable arrival sequencing at those points, and *replay from durable
  data alone cannot reproduce it*. PASS = zero ties (order fully determined by
  the key); FAIL = ties exist (order partly determined by arrival, which is not
  stored). **Either outcome is a result.** Cheap: instrument `freeze`'s input.
- **PROBE-FOLD-COLLISION-RATE** — how often do two owners write the same `row`
  in one cycle? Zero would mean the fold is currently inert under this workload
  and its irreplaceability (item 3) is structural-but-unexercised; non-zero
  means it is live. Must be measured before item 3 is cited as load-bearing *in
  practice* rather than *in principle*.
- **PROBE-ARRIVAL-ASCENDING-CONTROL** — re-run the O-arm with arrival ORDER =
  owner-ascending. Pre-registered expectation: digests MATCH. This is the
  can-stay-silent twin of the divergence result: if they diverge even there, the
  difference is larger than the four items above account for and this doc is
  incomplete.

---

## Cross-refs

- `.claude/plans/measure-64k-axes-v3.md` § O-arm MEASURED RESULTS (the divergence).
- `.claude/board/EPIPHANIES.md` `E-SEAL-AND-TEMPORAL-ARE-DIFFERENT-OBJECTS-1`.
- `crates/lance-graph-planner/src/persist_sink.rs` — `order_cycle_stably`,
  `DetachedCycleBatch::freeze`, `CycleFrame`, `WalSink::scan_sealed`.
- `crates/lance-graph-planner/src/temporal.rs` — `LocalCausalRow`,
  `local_trajectories`, `local_trajectory_of`.
