# D-DR-SELECT-K — one versioned DR selection, bound once, reused by every fold

**Status:** PROPOSED, docs only. Read at `main` `54c9b98a4` (2026-11-12).
Hypothesis record: PR 2032 (`dr-widefieldmask-dt-classview-v1.md` §1a.1,
`dr-dt64-class-governance-v3.md` §2c.1). Coordination: EXO-253. Laws this
plan must keep: EXO-220 (locality), EXO-255 (mapping is versioned, content
needs its own witness), EXO-46 and EXO-140 (admission), EXO-114 (order).

Only D-DR-SELECT-K-1 (the lab) is a candidate for implementation, and it is
not authorized yet. Everything after it depends on what it measures.

## 1. Order of things

The choice of a physical representation is a DR addressing decision over a
deterministically bound version. It is not a storage or residency decision,
and it is not a query optimizer in front of DR.

```
                 DR
                  │
        deterministic version V
                  │
          ┌───────┴────────┐
          ▼                ▼
     index order         filter
          │                │
          └───────┬────────┘
                  ▼
          DR{dynamic}[K]
                  │
         reusable selection
                  │
          Quack / mask-RISC
                  │
               ndarray
```

- An index gives an order; a filter gives a subset. They are not exclusive
  and compose over the same witnessed version:
  `S_V = Filter_V(IndexOrder_V(N))`, then `DR{dynamic}[S_V] → Compute(K)`.
- That is one composition order. For `WHERE` plus Top-K the semantically
  correct order (eligibility before the limit) is kept, whatever order the
  physical execution uses.
- Lance may do the index or the filter work. DR binds the result once.
  Quack and mask-RISC consume the existing selection and compute only what
  remains. No further selection, no second read of the source, no second
  authority.
- Constraint that stays: folds never touch Lance. Intake and binding happen
  once per version and are reported in µs, not gated (the cold-cost
  exception of `CLAUDE.md`); folds are under the no-millisecond rule.

## 2. Ownership

- Canonical values stay with the source owner.
- `NodeGuid` stays the logical identity.
- DR's existing `MapTable` stays the authority for the versioned
  global-to-local numeric mapping.
- The source adapter owns the snapshot-scoped translation to Lance row
  identity or address. **Not shown yet:** how a `NodeGuid` resolves to the
  canonical global id and then to a row of one snapshot, using only
  existing authorities. Until that chain is demonstrated, D-DR-SELECT-K-2
  is blocked.
- A K-row Arrow batch is not zero copy because it is small. Where it is
  needed it is counted as bounded, derived staging owned by the source or
  the focus lease. It is never a second canonical owner of values. Borrow
  where borrowing is supported and shown to work.

## 3. Representation choice

The choice belongs to the versioned DR binding. It follows the
representation that already exists and the operation that comes next:

| view | chosen when |
|---|---|
| **IndexVector** | the order already exists and the consumer can use it |
| **SelectionVector** | a sparse sequence of local positions is cheaper |
| **FocusMask** | Boolean algebra and SIMD folds over the local population are cheaper |
| **implicit `ALL(K)`** | the upstream selection is already exact and complete |

One buffer may serve several roles. A further representation is allocated
only when its added benefit is measured. The choice is deterministic: the
same version, binding and next operation give the same view. If one view
wins everywhere, only that view is used.

> This replaces the input list of §1a.1's "Cold deterministic physical
> choice" paragraph and the size matrix of §2c.1's W7 addendum for planning
> purposes. Both stay in place as the recorded hypothesis of PR 2032.

## 4. D-DR-SELECT-K-1: the lab (not authorized yet)

**Question.** For one versioned DR selection, which existing physical
representation is cheaper for a given operation and a given reuse count?
The lab does not decide "cold Lance against resident DR".

A small, independent lab crate. No `StrategySelector`, cache, mapper,
executor or writer.

| dimension | values |
|---|---|
| N | 1,000,000 |
| K | 256 and 65,536 |
| reuse R | 1 and 100 |
| address pattern | clustered and random, where it matters |
| negative control | one case with 75% survivors |
| next operation | Boolean fold with count, aggregate over a lane, ordered Top-K |
| view | IndexVector, SelectionVector, FocusMask, implicit `ALL(K)` where valid |

**Setup.** One version V. The selection is produced once, by an index
order, a filter, or both composed, and bound once. Every view is taken
from that one binding; a view that has to be converted reports the
conversion.

**Oracle.** An independent row-at-a-time answer; every view equals it,
including the ordered `NodeGuid` sequence where order is requested.

**Reported per cell:** producing the selection (index, filter, bytes
copied), bind, conversion if any, fold, retained memory, and the total for
each R. Production and bind in µs, labelled "reported, not gates"; fold
terms under the no-millisecond rule. A counter shows that no fold reads the
source again.

**Pass or stop.** The lab passes if every view equals the oracle and each
number is attributable to one term. It does not have to produce a winner
per operation; "one view wins everywhere" is a valid result and ends the
strategy question.

## 5. Gates before D-DR-SELECT-K-2

1. The ownership chain `NodeGuid` → canonical global id → DR coordinate →
   row address of one Lance snapshot is established from existing
   authorities.
2. Exact WHERE results are told apart from candidate-only results, with
   residual filters and Top-K ordering covered.
3. A real workload exists where binding K and folding over it wins after
   producing the selection, staging and amortization are counted.
4. The Admission B, source, temporal, IAM and version witnesses are kept.

If these hold, D-DR-SELECT-K-2 may propose the smallest deterministic
handoff rule the measurements support.

## 6. Contingent, not planned

Each of these waits on §5 and gets its own plan if it is ever started:
binding the selected population in a lab (K-2), running the existing
Quack / RelAlg / mask-RISC over K-sized planes (K-3), temporal binding
(K-4), prefetch and lease (K-5), sparse write-through (K-6), an external
Arrow consumer (K-7).

## 7. Not verified

- Which Lance 14 scanner or index call returns a compact, exact survivor
  set, and at what copy cost. The lab's first task.
- Whether mask-RISC accepts an implicit `ALL(K)` without a materialized
  plane.
- The `NodeGuid` to row chain of §2.
