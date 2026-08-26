## 2026-08-23 — E-THE-VIEW-MOVES-THE-POPULATION-DOES-NOT-1 — three incompatible selector families compose over one stationary cognitive state, provenance intact, zero population copy

**Status:** FINDING (measured — `PROBE-REVISION-ATTENTION-VIEW-1`, 8/8 gates
green). **Confidence:** High for what it measures; the scope fence below is
part of the finding, not a disclaimer.

**Measured.** One `BeliefArena` holding rungs `[0, 1, 2]` and 16 stationary
`NodeRow`s (8 KiB). Three selector families with NO common representation —
`BoundAt(Locus)` (signed nibble in a 12-byte witness register, read through
`WitnessLens`), `RungBand{lo,hi}` (`u32` arena field), `GapSubject(u16)`
(scan-derived `Vec<ReasoningGap>`) — compose into one narrowing view:

```
  A [BoundAt(SupportedBy)]                      -> 8 rows
  + Push(RungBand{1,2})
  B [BoundAt, RungBand]                         -> 4 rows
  + Push(GapSubject(1))
  C [BoundAt, RungBand, GapSubject]             -> 3 rows
```

- **Provenance survives.** 3 rows admitted, **4 blamed by exactly one**
  selector, **9 by two or more** — non-uniform blame, so the channel carries
  information. This is what a fused `union()`/`intersect()` on a packed mask
  cannot answer, and it is retained because the plan is a stack of typed
  descriptors while only the LOWERED artifact is opaque.
- **Zero population copy.** Population digest and base pointer byte-identical
  across all three lowerings. Descriptors allocated: **36 B**, against 16 × 512 B
  of population never touched. Reported separately, per the charter: the
  invariant is zero POPULATION copy, not zero allocation.
- **Typed edit reconstructs exactly.** `BEFORE + EDIT == AFTER` on BOTH layers
  (descriptor stack and lowered visible set), and the inverse `RemoveAt`
  restores `BEFORE`.
- **Non-destructive.** Rung bands and population bytes unchanged after every
  view change. Only visibility moved.
- **Controls.** Empty plan admits all 16; a contradictory plan admits 0. So the
  composition can both speak and stay silent.

**A measured surprise, kept visible rather than tidied away.** The first fixture
used `rcr_abduce` as the absence-shaped selector and the gate FAILED with
`gap=0`. Cause, measured not guessed: on a fully-closed transitive chain
`rcr_abduce` returns **20 candidates and ZERO gaps** — its gap channel fires on
a different condition (no shared middle / hub exclusion / budget), not on a
complete chain. `tr_diverge` on a siblingless focus emits the real
`ReasoningGap { kind: NoSibling, subject: Some(1) }`. The fix was to use the
call that actually produces the signal; the assertion was never weakened, and
the probe still prints `rcr_abduce`'s empty channel so a future reader sees the
distinction instead of rediscovering it.

**What this does NOT establish**, stated because the charter demands nothing
stronger: no behavioral BPE (this is the receipt substrate only); no wall-clock
or thread parallelism (occupancy is semantic — `close_transitive` is a
sequential fixpoint); no Rubicon persistence (`KanbanMove` carries no attention
provenance, `calcify` is `todo!()`); and no production Revision surface —
**F-REVISION-FOCUS-1 is ABSENT** and the `ViewEdit` used here is an explicitly
probe-local adapter. `RungElevator` does not appear in this probe at all.

**Existing-container audit ran first**, per the charter's probe law.
`contract::selection::{NamedView, ViewRegistry}` is the shipped precedent with
exactly the right two-layer shape — `union_of(&[ViewId]) -> WideFieldMask`
takes retained descriptor identities and returns one fused artifact. It was not
reused for one honest reason: it composes a SINGLE family over one
representation, and heterogeneity is the whole question. `CycleFrame` records a
cycle, `SupportReceipt` records support; neither is a view. The probe-local
`ViewPlan` follows `ViewRegistry`'s shape rather than inventing one, and does
not propose replacing it.

**No `CommonMask` was built. No NOT/XOR was added.** The three families meet
only at the lowering boundary, as one `impl Fn(usize) -> bool` — the seam that
already exists seven times over in `witness_fabric`.

