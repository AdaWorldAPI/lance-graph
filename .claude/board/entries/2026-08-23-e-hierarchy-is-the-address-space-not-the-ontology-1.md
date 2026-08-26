## 2026-08-23 — E-HIERARCHY-IS-THE-ADDRESS-SPACE-NOT-THE-ONTOLOGY-1 — HHTL is the universal address grammar, beneath the V3/V4 distinction; and globality is geometry ONLY WITH provenance

**Status:** ROOT LAW proposed by the operator, with one part [MEASURED]
and one part [CONJECTURE] — the split is stated below rather than blurred.
**Confidence:** High for the address-space law and the measured corollary;
the universal-addressability claim is explicitly NOT demonstrated for
`Belief`.

### The law (operator, 2026-08-23)

> **Hierarchy is the address space, not the ontology.**
>
> HHTL need not claim that reality is a tree. It claims that every resident
> datum can receive a hierarchical address FROM WHICH non-hierarchical
> relations can be expressed **without changing its physical identity — and
> zero-copy where the carrier is ABI-resident.**

**Why that qualifier is exact, not hedging.** Hierarchical addressability
by itself guarantees nothing about copies; the PHYSICAL ABI does. An
address over a non-resident carrier (an owned container, a materialized
window) is still an address, and it is still not zero-copy — which is
precisely the `WitnessStream` situation recorded in its own honesty note.
Keeping the two clauses separate stops the law from silently importing a
performance guarantee it does not itself supply.

The Active Directory lesson, exactly. A DN gives an object a deterministic
hierarchical home —
`CN=Jan,OU=Engineering,DC=example,DC=com` — without asserting that
everything about Jan is ancestry. Group membership, manager links, ACLs and
mail routing are cross-links between objects that already have hierarchical
identities. The directory hierarchy solves **where**; the references solve
**how things relate**.

```
  EVERY COGNITIVE DATUM GETS AN HHTL HOME.

  Relations may then: inherit · cross-link · point sideways · backward ·
  forward · contradict · support · cause · observe · intervene
  — without changing the object's addressability.
```

**What this corrects (a narrowness in my own prior caution).** The earlier
entry `E-STREAM-ORDER-VS-PREFIX-TREE-NEITHER-ACCUMULATES-1` framed
"stream-order Markov" and "prefix-tree HHTL" as an opposition. Under this
law that opposition is **weaker than it was stated** — see that entry's own
regrade note. A temporal datum is not a separate non-HHTL universe; it can
be HHTL-addressed (`episode → version window → event region → event
identity`) with the signed ±i4 offsets as LOCAL traversal inside an
addressed neighborhood. Stream ordering remains temporal semantics; HHTL
gives it its home. The two are layered, not rival. Same for graphs: a node
has a canonical HHTL home while edges are `address A → address B`, and the
graph itself needs no tree structure at all — AriGraph, `CausalEdge`,
`EpisodicEdge`, anaphora pointers, support/premise relations and R2IL
operations can all point between HHTL-resident things.

### HHTL is BENEATH the V3/V4 distinction

```
                 HHTL ADDRESS SPACE
                       │
         ┌─────────────┼─────────────┐
        V3            V3            V4
      state         witness       behavior
         │             │             │
     ClassView     ClassView     ClassView
         │             │             │
  route semantics  route semantics  route semantics
```

V3 and V4 do not own the railway. They are payload/read contracts for
objects that **already have a hierarchical home**. This is a cleaner
statement than the earlier "V4 may ride V3 routes" framing.

### The brutal admission rule (the sharp test this gives us)

> **If a datum exists but cannot be assigned a meaningful HHTL address
> without using `Vec` position as identity, it has not yet been normalized
> into the memory ABI.**

This names precisely what was wrong with the withdrawn "trivial
per-arena-position address" proposal (`BELIEF-ABI-RESTORATION-1` Step 1,
recut): **`arena[37]` is not a semantic hierarchy — it is an implementation
accident wearing an HHTL costume.** A real address would land through
semantic coordinates (relation class → subject basin → predicate basin →
evidence context → instance; exact carving to be PROVEN, not invented).

### The corollary, and where it breaks — [MEASURED]

> **Evidence rises only as high in the HHTL tree as its independent support
> generalizes.**

The prize is deleting a metadata system: no `enum Scope {Local, Regional,
Global}`, no global-concern score, no scheduler that "promotes" a belief.
Globality becomes geometry. `PROBE-EVIDENCE-RISES-BY-GENERALIZATION-1`
(7/7) measured it on shipped operators:

- **[CODE]** The pieces already exist and were not written for this:
  `AttentionFocusFacet::common_prefix` is the MEET (deepest focus covering
  both, and it *never invents an address*); `RowFocusMask` is literally
  "antichain of HHTL regions" with absorbing `union` and conservative
  `difference`; `TruthValue::revise` (`nars/truth.rs:57`) pools by
  `evidence_weight() = c/(1−c)`.
- **[MEASURED]** Support rises exactly as far as it generalizes (G1 one
  basin stays local; G2 three independent siblings rise to the common
  ancestor with pooled `c=0.9444 > 0.85`; G6 cross-region support rises
  COARSER, never sideways), the operator can refuse (G7: no common ancestor
  across classes ⇒ `None`), a dissenting region stays ADDRESSABLE instead
  of being averaged away (G4), and nothing moves (G5: children byte-
  identical; the parent acquires a derived reading at an address that
  already existed — **children stay, parent learns**).
- **⚠ [MEASURED] THE BOUNDARY — G3.** *Geometry alone over-generalizes.*
  One source observed through three sibling basins pools naively to
  `c=0.9444` — **bit-identical to three genuinely independent sources.**
  The two situations are *geometrically indistinguishable*. So:

  > `globality = geometry` is TRUE **only with provenance**.

  This is not a caveat, it is the load-bearing constraint: the hard problem
  was never aggregation (NARS already revises correctly) but **independence
  detection** — are these siblings independent evidence, or descendants of
  one observation? That is exactly what `Stamp`'s `disjoint → revise /
  overlap → CHOICE` guard protects today, and `spo::truth`'s revision doc
  already states the precondition verbatim: *"combine two truth values with
  **independent** evidence."*

### The correct decomposition (with the operator's own correction applied)

```
  NARS frequency   observed proportion / effect estimate
  NARS confidence  effective evidence mass  ← NOT frequency
  HHTL address     the scope at which it holds
  HHTL ancestry    how far that support generalizes
```

Frequency alone is not sample count; `evidence_weight() = c/(1−c)` in the
shipped `revise` confirms confidence is the evidence-mass side.

### [CONJECTURE] — stated as such

That **every** cognitive datum CAN receive a lawful HHTL address is not
demonstrated. For `Belief` specifically the audit found the opposite of a
demonstration: zero `FacetCascade`/`facet_classid` occurrences anywhere in
`nars/`. The law says such an address is *possible in principle*; whether
the semantic carving exists for beliefs is open, and is the real content of
`BELIEF-ABI-RESTORATION-1` Step 2. Nothing here mints a tenant, an address,
or a classid.

**One consequence worth recording if the conjecture holds:** attention
would need no foreign mask representation at all — `Attention = set /
antichain / scoped difference of HHTL regions` is `RowFocusMask` as it
already ships; epistemic potholes become scoped ABSENCE within an addressed
universe (still the conservative P* reading, `RowFocusMask::difference`
keeps partially-overlapped entries whole); and R2IL applicability points
into the same coordinate grammar. `DATA` has an address, `ATTENTION` is
region selection, `UNKNOWN` is scoped absence, `BEHAVIOR` has applicability
routes, `PROVENANCE` points to evidence, `CAUSALITY` links resident state —
and C operates on the surviving addressed slices.
