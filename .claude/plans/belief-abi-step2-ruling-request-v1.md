# BELIEF-ABI-RESTORATION-1 — Step 2: the ruling request

> Status: **RULING REQUEST — awaiting operator decision. This document makes
> no ruling.** Step 2 of the charter's ladder is *"Operator ruling on the
> residue: existing-tenant composition vs one new tenant mint (per residue
> item, not wholesale."* An agent prepares the decision; it does not take it.
>
> Every recommendation below is labelled **[RECOMMENDATION]** and is
> non-binding. Every item states what would falsify it.

## Why the picture changed since Step 1

Step 1 (#1006) produced a residue table and two [ABSENT] verdicts. Five
probes since (#1007, #1009, #1010, #1011, and the `Copula` probe in this
PR) have **moved four of the six items** — two toward elimination, one to a
different shape than assumed, and one to *provably irreducible*. The ruling
should be made against this state, not Step 1's.

| item | Step 1 said | now measured |
|---|---|---|
| `stmt`/`cop` | "open — needs its own audit" | **SETTLED: not rail-expressible** (C1–C4) |
| `truth` | structural fit, unwired | unchanged; confidence = evidence mass confirmed |
| `stamp` | "likely residue, do not invent yet" | **PROVABLY irreducible to geometry** (#1009 G3) |
| `rung` | "likely residue" | **conflates TWO axes**; depth may be derivable (#1011 E3, #1007 A2) |
| `premises` | arity ≤ 2 | unchanged; width still open |
| `contradiction` | "wire, don't reinvent" | **wrong shape**: a magnitude cannot express Auslöschung (#1010 F1) |

---

## Item 1 — `stmt.cop` (the copula) · **needs a home**

**Settled this PR (`PROBE-MASK-ALGEBRA-INVARIANCE-1`, C1–C4).** No `Copula`
variant is expressible in rail geometry, for a principled reason:

```
  RAILS      transitive (prefix containment IS transitivity)
             antisymmetric (a strict ancestry order)
             COMMITTED (RailPath = {len, slots}: no truth, no polarity)

  COPULAS    selectively transitive (`transits()`: only Inh, Sim)
             sometimes symmetric (Sim)
             always DEFEASIBLE (a Belief carries (frequency, confidence))
```

`Impl` and `Rel` fail because rails are *unconditionally* transitive; `Sim`
fails because rails are antisymmetric; `Inh` — the only rail-SHAPED one —
fails because **a rail placement is committed and a belief is defeasible.**
A rail IS the taxonomy; a belief is a CLAIM ABOUT the taxonomy, and storing
the claim as a placement silently promotes a hypothesis to structure.

> **⊘ RECOMMENDATION RETRACTED (operator-directed, 2026-08-23).** The text
> that stood here recommended `Copula → relation concept → classid
> reference`. **That was a drift and is withdrawn**: C1–C4 established only
> `COPULA ≠ RAIL PLACEMENT`, which does NOT establish `COPULA = CLASSID` —
> unrelated conclusions. Routing relation content through classid would
> smuggle content into the reading selector. The law:
>
> ```
>   CONTENT NEVER TRAVELS IN CLASSID.
>   CLASSID SELECTS THE READING.
> ```
>
> The replacement analysis — machinery audit, measured distribution
> (`PROBE-COPULA-GROUP-MASK-1`, 9/9), the Active-Directory group/membership
> interpretation, candidate homes A–F with A/B rejected as sole carriers
> and F not proposed — lives in
> `.claude/plans/belief-abi-step2-addendum-copula-v1.md`. The question is
> no longer *"where do we encode Copula?"* but *"what is the cheapest
> lawful resident relation + selection geometry from which Copula is merely
> an ergonomic reading?"* Current measurements favour sparse relation rows
> (copula content RESIDENT in the row) with group masks as lossy-by-design
> selection ergonomics; **no mint until workload-scale distributions rule
> out composition.**

---

## Item 2 — `truth (f32, f32)` · **compose, do not mint**

`spo::truth::TruthValue { frequency, confidence }` (`truth.rs:15-17`) is
byte-identical in shape and documented *"Each SPO edge carries a
TruthValue."* Real, shipped, **unwired**.

#1009 additionally confirmed the semantics: `revise()` pools by
`evidence_weight() = c/(1−c)`, so confidence carries the evidence mass —
frequency is the estimate, not the sample count.

**[RECOMMENDATION] — existing-tenant composition.** Wire it; mint nothing.

**Open sub-question for the ruling:** *where* per-relation truth physically
RESIDES (an SPO row vs a value lane) is a placement decision this document
does not attempt.

---

## Item 3 — `stamp: u64` · **PROVABLY irreducible — the strongest mint candidate**

**This is the item the probes settled most sharply, and it settled AGAINST
elimination.** #1009 G3: three sibling observations derived from ONE source
pool to `c=0.9444` — **bit-identical** to three genuinely independent
sources. The two situations are *geometrically indistinguishable*.

> Scope generalization is geometry. **Warranted** generalization is geometry
> **+ provenance.** Provenance is not metadata garnish; it is the promotion
> warrant, and it cannot be derived from the address space.

#1011 E2 then showed the same requirement from the other side: a closure
receipt is what separates a falsifier from a search accelerator.

**Whatever replaces `stamp` must preserve its IDENTITY semantics**, not
merely "accumulate": disjointness detection, overlap detection, source-set
union, no-double-count (`belief.rs:39-48`) — plus the modulo-64 folding
that is **conservative by design** (*"folding can only create false overlap,
never false disjointness"*), which is a soundness property a replacement
must not quietly drop.

**[RECOMMENDATION] — this is the one item where a mint is genuinely
warranted**, because no composition of the existing tenants supplies
independence detection. But the shape is the operator's call: a provenance
tenant, a witness-corpus/merkle composition, or a closure-receipt-shaped
carrier that serves both this and #1011's requirement.

**What would falsify it:** a demonstration that an existing tenant already
carries source identity with disjointness testable. Not found in this
repo's audit; not proven absent everywhere.

---

## Item 4 — `rung: u32` · **not one item — TWO, and both may avoid storage**

**#1011 E3 measured that `rung` conflates two independent axes:**

```
                    BROADER SCOPE
                         ↑
      shallow proof      │      deep proof
      broad support      │      broad support
   ──────────────────────┼──────────────────────→ DERIVATIONAL DEPTH
      shallow proof      │      deep proof
      local support      │      local support
                         ↓
                    LOCAL SCOPE
```

Two claims with the same depth at different scopes read differently, and
vice versa. A scalar collapses both.

- **Derivational depth** — #1007 A2 derived it from the premise DAG ALONE
  (never reading `b.rung`) and it reproduced the stored scalar **10/10 on
  one fixture**. If that generalizes, depth need not be stored at all.
- **Generalization scope** — #1009 showed this is geometric: support rises
  by `common_prefix` exactly as far as it generalizes. Already an address
  property; storing it would duplicate the geometry.

**[RECOMMENDATION] — split the item, and treat both halves as
elimination candidates rather than mint candidates.** Neither obviously
needs a tenant.

**What would falsify it:** the A2 result is **one fixture of one shape**.
Before depth is declared derivable, it should be measured on shapes that
could break it — unbalanced DAGs, diamond derivations, and the CHOICE
replacement path where a premise's own rung was later raised. **A
one-fixture result is not a general one**, and this recommendation is the
weakest in the document.

---

## Item 5 — `premises: Vec<u32>` · **unchanged; blocked on the address question**

Step 1 established real cardinality ≤ 2 (11 `admit_derived` sites, 4
`tactics.rs` mint sites) — so this is not the "cardinality = more rows"
case. Step 1's recut also established what is NOT known: that two `u32`
identities *fit* in two tiles or two nibbles. Cardinality and physical
width are different facts.

**[RECOMMENDATION] — defer.** The width question is downstream of whether a
belief can acquire an identity-derived address at all (the charter's open
Q3). Ruling on premise width before that is ruling on a representation for
an identity that does not yet exist.

---

## Item 6 — `contradiction: f32` · **wrong SHAPE, not merely unwired**

Step 1 recorded this as *"wire, don't reinvent"* against
`Locus::Contradiction`. **#1010 F1 changed the requirement.**

A single f32 magnitude cannot express Auslöschung: `net(+3, −3)` and
`net(unset)` are both `0`, so a summed/collapsed representation **cannot
distinguish "support and refutation met and annihilated" from "nothing was
ever asserted."** Those license opposite actions — a licence to LEARN vs a
licence to LOOK.

**[RECOMMENDATION] — the target is a RETAINED-POLARITY reading, not a
magnitude field.** Constructive and falsifying evidence must both survive;
cancellation is a projection over them, never a storage collapse. This
matches the standing rule that a contradiction is *committed and preserved*,
not resolved away — and #1010 F6 adds that an exclusion likewise needs its
own signed channel rather than a subtracted prefix.

---

## What the ruling actually has to decide

1. **`cop`** — *(reframed by the addendum; the classid-reference option is
   retracted)* accept sparse relation rows (copula content resident in the
   row) + group-mask ergonomics as the working hypothesis, pending the
   workload-scale distribution measurements? See
   `belief-abi-step2-addendum-copula-v1.md` §8.
2. **`truth`** — confirm compose-don't-mint, and rule on WHERE it resides.
3. **`stamp`** — mint what, exactly? Provenance tenant vs
   witness/merkle composition vs a closure-receipt-shaped carrier serving
   both this and #1011.
4. **`rung`** — accept the split into depth + scope? And is the
   one-fixture A2 result enough to pursue depth-as-derived, or should it be
   measured on breaking shapes first? *(The document recommends the latter.)*
5. **`premises`** — accept the deferral to the address question?
6. **`contradiction`** — accept retained-polarity as the target shape?

## What this document deliberately does NOT do

- It rules nothing. Every item above is a recommendation with its falsifier.
- It mints nothing, and proposes no layout, address, or classid.
- It does not answer the charter's Q3 (can a belief acquire an
  identity-derived address). Items 4 and 5 are partly blocked on it, and
  that blocker is stated rather than routed around.
