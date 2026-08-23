# TARSKI-MARKOV-HHTL — open questions register (NOT a plan)

> Status: **HELD — OPEN QUESTIONS ONLY. This file proposes no mechanism and
> licenses no work.** An earlier revision was a design plan for a specific
> composition (a stream-order reading bound to a tree address); that
> proposal is **withdrawn** — see § The withdrawn proposal. What remains is
> the register of what is genuinely unknown after
> `E-STREAM-ORDER-VS-PREFIX-TREE-NEITHER-ACCUMULATES-1`, so a future
> session inherits the questions without inheriting an unearned answer.

## What is established

`BELIEF-ABI-RESTORATION-1`'s Step 1 audit found the operator-ruled
tree-overlay delegation for `Belief.rung`/`stamp` has no implementing code.
The follow-up finding (`.claude/board/EPIPHANIES.md`,
`E-STREAM-ORDER-VS-PREFIX-TREE-NEITHER-ACCUMULATES-1`) checked the two
closest shipped candidates and found neither is the mechanism:

```
  deepnsm-v2::wave::WitnessStream      a TOTAL ORDER over time
    flat Vec<(version, register)>, signed-offset walk in a version window
    disclaims accumulation BY NAME (E-NO-BUNDLE-STANDING-WAVE-1)

  AttentionFocusFacet                   a PARTIAL ORDER over address
    covers / common_prefix — prefix containment, the meet
    answers "is A an ancestor of B", never "what do B's children give A"
```

That is the whole result. It is negative.

## The withdrawn proposal (recorded so it is not re-derived)

This file previously concluded that since neither candidate is the
mechanism, Step 3 must **compose** them — a `WitnessStream`-shaped window
rebased from version order into "address-scoped stream order," feeding an
accumulate operator. **Withdrawn**, for three reasons:

1. **The inference is invalid.** From *A is not it* and *B is not it*, the
   only conclusion is *we do not have it* — not *A × B is it*. The
   composition is one hypothesis among unknown others; a negative result
   cannot promote it.
2. **It proposed semantic type drift.** Rebasing a `VersionRange` (or a
   version-ordered stream) to mean an address span makes a type whose
   meaning depends on who holds it. **Time is time. Address is address.**
   Sharing a memory ABI does not make two topologies interchangeable —
   preventing exactly this is why the DOCK/ROUTE separation is written the
   way it is.
3. **It risked re-entering the memory escape.** A "`WitnessStream`-shaped
   window" that materializes another `Vec<(u64, CausalWitnessFacet)>` is
   the escape through a different door. `wave.rs`'s own honesty note
   already records that `WitnessStream` is TODAY *"a parallel OWNED
   container beside `TemporalStream`, not a zero-copy projection over the
   real `ValueTenant::CausalWitness` lane."*

## Regrade (2026-08-23) — the opposition was overstated

`E-HIERARCHY-IS-THE-ADDRESS-SPACE-NOT-THE-ONTOLOGY-1` weakens the framing
above. The two candidates are **layered, not rival**: a temporal datum can
be HHTL-addressed (`episode → version window → event region → event
identity`) with the signed ±i4 offsets as LOCAL traversal inside that
addressed neighborhood. Time stays time; HHTL supplies the home. There is
no "separate temporal universe" to reconcile.

This does NOT revive the withdrawn composition — it removes the motivation
for it. The withdrawn proposal wanted to re-scope a version-ordered type to
mean an address span; the address-space law says the correct move was
always to give the temporal datum its own address and leave the ordering
alone. The fence in § The withdrawn proposal is therefore *strengthened*,
not relaxed.

What survives unchanged: neither candidate implements a children-and-
siblings accumulate fold, which was and remains this register's reason to
exist.

## The open questions (each stated as a question, none as a direction)

- **Q1** — Is derivation depth reconstructible from SUPPORT topology (the
  premise DAG) alone? This is answerable with no address at all, and is
  the question `rung = HHTL tree depth` should have been.
  *Partial evidence:* `PROBE-TARSKI-SIGNED-WITNESS-1` gate A2 (PR #1007)
  derived depth from the premise DAG alone and it reproduced the arena's
  stored `rung` 10/10 — **on one fixture, of one shape**. That is a
  measurement, not a general result, and it says nothing about HHTL depth.
- **Q2** — What would a replacement for `Stamp` have to preserve? Not "a
  commutative fold": `Stamp`'s load-bearing behaviour is IDENTITY
  semantics — disjointness detection, overlap detection, source-set union,
  no-double-count (`belief.rs:39-48`), plus the documented modulo-64
  folding that is conservative BY DESIGN ("folding can only create false
  overlap, never false disjointness"). A `vsa_bundle` or any generic
  commutative accumulate is not automatically a source-set union and must
  not be assumed equivalent.
- **Q3** — Can a `Belief` acquire a canonical address at all — one derived
  from node/relation IDENTITY rather than from arena position? Step 1's
  recut records why a position-derived address is not an answer: it turns
  a `Vec` index into a pretty 16-byte `Vec` index.
- **Q4** — Do two proof routes with tied truth carry different provenance?
  `PROBE-TARSKI-SIGNED-WITNESS-1` observed that `close_transitive`'s
  `HashMap` iteration order makes both premise indices AND tie-broken
  derivation routes vary across identical builds. Whether that is a
  reproducibility defect or an acceptable degree of freedom depends on
  whether route identity is ever load-bearing — unresolved.
- **Q5** — Is there a mechanism neither candidate resembles? Unasked so
  far, and the most likely place a real answer lives, precisely because
  nobody has looked there.

## Standing fences (apply to any future answer, whatever it turns out to be)

- A time-ordered type must not be re-scoped to mean an address span.
- Any address-scoped read must be a BORROWED VIEW over ABI-resident rows,
  never a new owned container.
- A new reading of shared geometry gets its OWN ClassView and its OWN
  vocabulary; it may not use another ClassView's semantic API to mean
  something that ClassView forbids (the A9 "loci, not magnitudes" lesson,
  `PROBE-TARSKI-SIGNED-WITNESS-1` recut).
- Anti-vacuity: a fold that reproduces `max(premise rungs)+1` regardless of
  tree shape has proven nothing (this repo's own falsifiability rule).

## Trigger

Nothing here is startable. The next legitimate step is
`BELIEF-ABI-RESTORATION-1` Step 2 — the operator ruling on the Step 1
residue — which may or may not make any of Q1–Q5 relevant.
