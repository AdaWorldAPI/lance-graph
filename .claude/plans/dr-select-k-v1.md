# D-DR-SELECT-K — storage selects K once per frame, DR folds over K many times

**Status:** PROPOSED, docs only. Read at `main` `54c9b98a4` (2026-11-12).
Hypothesis record: PR 2032 (`dr-widefieldmask-dt-classview-v1.md` §1a.1,
`dr-dt64-class-governance-v3.md` §2c.1). Coordination: EXO-253. Laws this
plan must keep: EXO-220 (locality), EXO-255 (mapping is versioned, content
needs its own witness), EXO-46 and EXO-140 (admission), EXO-114 (order).

Only D-DR-SELECT-K-1 (the lab) is a candidate for implementation, and it is
not authorized yet. Everything after it depends on what it measures.

## 1. Scenario

Two paths, with different budgets.

| | Cold path | Hot path |
|---|---|---|
| when | once per frame | many times per frame |
| what | Lance snapshot → optional indexed WHERE → K survivors → bind DR | Quack / mask-RISC / ndarray folds over K |
| touches Lance | yes | never |
| cost rule | reported in µs, not gated (the cold-cost exception of `CLAUDE.md`) | the no-millisecond rule applies in full |

1. **Resident data.** When the source-owned data and the DR planes are
   already resident, execute through Quack / mask-RISC / ndarray. Do not go
   back to Lance for WHERE, Top-K or another selection pass. The 188 µs
   resident fold at 1,000,000 rows (PR 2010) is the baseline here.
2. **Cold or changed frame.** When Lance owns non-resident data and can
   evaluate an indexed WHERE more cheaply than loading and preparing N rows,
   storage produces the exact K survivors. They are bound once into DR local
   coordinates. Every later fold runs over K.
3. **Reuse.** The point is to pay intake, binding, global-to-local mapping
   and prefetch once and spread them over R uses of the frame. It is not to
   put a storage access into every query.

The comparison the lab has to settle, with R the number of uses of a frame:

```
T_A = T_full_intake(N)                 + R × T_fold(N)
T_B = T_indexed_intake(K) + T_bind(K)  + R × T_fold(K)
```

Index lookup, copying and validation all count inside the terms where they
occur.

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

No query optimizer. Three roles that already exist, over one witnessed
selection:

- **IndexVector:** order and address mapping.
- **SelectionVector:** sparse explicit local positions.
- **FocusMask:** dense local Boolean selection.

`ALL(K)` is implicit when storage has evaluated the predicate exactly. One
buffer may serve several roles. A further representation is allocated only
when its added benefit is measured.

The first rule is deliberately small (illustration, not an API):

```
resident                              → reuse the existing DR binding
cold, measured index path wins        → bind the Lance survivors once
cold, otherwise                       → full intake, then bind
```

The rule is deterministic: the same witnessed context gives the same choice.
It becomes finer only if the lab shows that sparse positions and dense masks
each win for different downstream operations. If one view wins everywhere,
only that view is used.

> This replaces the input list of §1a.1's "Cold deterministic physical
> choice" paragraph and the size matrix of §2c.1's W7 addendum for planning
> purposes. Both stay in place as the recorded hypothesis of PR 2032.

## 4. D-DR-SELECT-K-1: the lab (not authorized yet)

A small, independent lab crate. No `StrategySelector`, cache, mapper,
executor or writer.

| dimension | values |
|---|---|
| N | 1,000,000 |
| K | 256 and 65,536 |
| reuse R | 1 and 100 |
| address pattern | clustered and random, where it matters |
| negative control | one case with 75% survivors |

**Arms.**

- Cold: full intake + resident DR execution over N, against indexed intake
  + DR execution over K.
- Resident (no Lance I/O): the existing mask path, the indexed-position
  path and the Focus path over the same selection.

**Oracle.** An independent row-at-a-time answer; every arm equals it,
including the ordered `NodeGuid` sequence where order is requested.

**Reported per arm:** intake, index lookup, bytes copied, bind, fold,
retained memory, amortized total for each R. Cold terms in µs, labelled
"reported, not gates"; fold terms under the no-millisecond rule.

**Pass or stop.** The lab passes if every arm equals the oracle and the
numbers are attributable per term. It does not have to show that the index
path wins; a clear loss ends the sequence here.

## 5. Gates before D-DR-SELECT-K-2

1. The ownership chain `NodeGuid` → canonical global id → DR coordinate →
   row address of one Lance snapshot is established from existing
   authorities.
2. Exact WHERE results are told apart from candidate-only results, with
   residual filters and Top-K ordering covered.
3. A real workload exists where selecting K before the DR bind wins after
   intake, staging and amortization are counted.
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
