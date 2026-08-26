## 2026-08-23 — E-STREAM-ORDER-VS-PREFIX-TREE-NEITHER-ACCUMULATES-1 — the two closest shipped candidates for the belief tree-overlay delegation are different topologies, and NEITHER implements an accumulate-fold

**Status:** FINDING (grounds and sharpens the `BELIEF-ABI-RESTORATION-1`
Step 1 audit's [ABSENT] verdict on the tree-overlay delegation mechanism;
does not overturn it). **⚠ PARTIALLY REGRADED 2026-08-23 — see the note
immediately below.**
**Confidence:** High for the [ABSENT] verdicts (the entry's actual result).
**Lowered for the "different topologies" FRAMING**, which overstated an
opposition.

> **⊘ REGRADE (2026-08-23, `E-HIERARCHY-IS-THE-ADDRESS-SPACE-NOT-THE-
> ONTOLOGY-1`).** This entry framed candidate A (stream order over time)
> and candidate B (prefix tree over address) as an *opposition* — "a total
> order over TIME, a partial order over ADDRESS," presented as if a datum
> had to live in one universe or the other. **That framing is too strong.**
> Under the address-space law, hierarchy is the ADDRESS SPACE, not the
> ontology: a temporal datum is not a separate non-HHTL universe, it can be
> HHTL-addressed (`episode → version window → event region → event
> identity`) with the signed ±i4 offsets as LOCAL traversal *inside* an
> addressed neighborhood. Stream ordering remains temporal semantics; HHTL
> supplies its home. The two are **layered, not rival**, and no "separate
> temporal universe" needs to exist.
>
> **What survives the regrade unchanged — the entry's actual finding:**
> neither candidate implements a children-and-siblings accumulate fold
> (`fn accumulate` / `children.*sibling`: zero hits;
> `E-NO-BUNDLE-STANDING-WAVE-1` disclaims accumulation by name). Both
> [ABSENT] verdicts stand. Also unchanged: the withdrawal of the
> composition conclusion, and the fence against rebasing a version-ordered
> type to mean an address span — that fence is *strengthened* here, because
> the right move was never to re-scope `VersionRange` but to give the
> temporal datum its own address and leave time as time.

**The question that prompted this.** Step 1 of `BELIEF-ABI-RESTORATION-1`
found that `Belief.rung`/`stamp`'s operator-ruled delegation — "rung = HHTL
tree depth, stamp = children-and-siblings accumulation, inherit from
parent" — names a mechanism that is currently prose and precedent, not
code (`.claude/plans/belief-abi-restoration-v1.md` §Step 1: zero
`FacetCascade`/`facet_classid` occurrences in `nars/`, zero `fn accumulate`
or `children.*sibling` hits in `lance-graph-contract/src/`). A follow-up
asked directly: is the "24×i4 Markov left-right-corner parsing" the
missing mechanism, compared against "AriGraph-style HHTL basins"? Grounding
both candidates in shipped code answers: **they are not rival
implementations of the same job — one is a total order over TIME, the
other a partial order over ADDRESS — and neither one accumulates.**

**Candidate A — `deepnsm-v2::wave::WitnessStream` (the G24N4/Markov
reading).** `wave.rs` module doc, verbatim: *"Each versioned event owns its
`CausalWitnessFacet` loci; the wave **reads** a version-range window and
mutates nothing — there is no accumulator and no shared register. The
Markov property is STREAM ORDER (`E-MARKOV-TEMPORAL-STREAM-1`), never a
superposition into one carrier."* Concretely: `events: Vec<(u64,
CausalWitnessFacet)>` in append order; `ground_at`/`resolve_at` walk a
SIGNED OFFSET through that flat line within a version-visibility window
(`TemporalPov::at`). This is where left-corner parsing theory actually
enters the codebase — but only as a CITATION justifying the ±8 nibble
range (Manning & Carpenter 1997, IWPT-97 Table 7: max left-corner stack
depth over the whole binarized WSJ treebank is 8), never as an executing
parser. There is no tree here. `E-NO-BUNDLE-STANDING-WAVE-1` is a standing
refusal to accumulate, not an oversight.

**Candidate B — `AttentionFocusFacet` (the AriGraph/HHTL-basin reading).**
`covers`/`common_prefix` (`attention_facet.rs:297,314`) give coarse→fine
PREFIX containment over the same 12-byte cascade — ancestor/descendant by
shared byte prefix, ordinal position, never stream position.
`common_prefix` computes the meet (deepest common ancestor) two focuses
share, which IS the shape a "rung = tree depth" claim would need to walk.
But nothing here sums, folds, or aggregates evidence across a node's
children into a parent value — `covers`/`common_prefix` answer "is A an
ancestor of B" / "what do A and B share", never "what do B's children
contribute to A". Grepped independently for this pass: zero `fn
accumulate` hits and zero `children.*sibling` hits anywhere under
`lance-graph-contract/src/attention_facet.rs`, `facet.rs`, or `hhtl.rs`.

**A third, previously-conflated shipped thing, named so it stops being
confused with either candidate:** `insight_right_corner_read.rs`
(`lance-graph-planner/examples/`) is a REAL left-corner/right-corner SVO
clause parser running on real KJV text — but it is a `Basins`-driven token
scan with its own `RightCornerReason`/`Triple` types and touches
`CausalWitnessFacet`/G24N4 **not at all**. "Left-corner parsing" and
"G24N4" are two unconnected machines in this codebase; only `wave.rs`'s
module-doc citation bridges them, and only as a numeric-range
justification, not a shared mechanism.

(A fourth precedent worth naming for completeness, since it was checked in
the same pass and is the closest thing to a REAL KJV+G24N4 example:
`probe_binding_not_heuristic.rs` resolves `Locus::Antecedent` — a single
locus, single-hop structural anaphora binding — on Gen 3:1/3:7. It proves
the chip is load-bearing, not decorative. It is not an accumulate-fold
either: it is a pointer write/read on one register, the same shape
`wave.rs`'s single-owner events already generalize.)

**The consequence — and the inference NOT to draw.** The finding is
negative and stops there:

```
  A is not the mechanism
  B is not the mechanism
  ⇒ WE DO NOT HAVE THE MECHANISM
```

**⇒ NOT `therefore A × B is the mechanism`.** An earlier revision of this
entry concluded that Step 3 must COMPOSE the two (a stream-order walk
bound to a tree address). **That leap is withdrawn.** "Neither of the two
things I checked is it" licenses no claim whatever about what it is; the
composition is one hypothesis among unknown others, and elevating it to
"what Step 3 must build" would smuggle a design decision in through a
negative result.

**The specific danger that withdrawal avoids, recorded as a fence.** The
withdrawn composition proposed rebasing `WitnessStream` from VERSION
order into "address-scoped stream order," possibly repurposing
`VersionRange`. That is precisely the semantic type drift this
architecture exists to prevent: **time is time, address is address.**
Sharing a memory ABI does not make two topologies interchangeable, and a
`VersionRange` that sometimes means an address span is a type whose
meaning depends on who is holding it. Further: any future address-scoped
fold must be a BORROWED VIEW over ABI-resident rows — if it materializes
another `Vec<(u64, CausalWitnessFacet)>`, the memory escape has simply
re-entered through a different door (`E-TYPE-COMPLEXITY-EXPOSED-A-MEMORY-
ABI-ESCAPE-1`), and `wave.rs`'s own honesty note already records that
`WitnessStream` is TODAY *"a parallel OWNED container beside
`TemporalStream`, not a zero-copy projection."*

**Two inherited assumptions this entry does not carry forward.** Neither
is established, and the negative finding does not support either: (a)
`rung = HHTL tree depth` — the answerable question is whether derivation
depth is reconstructible from SUPPORT topology, which needs no address at
all; (b) `stamp = commutative accumulation` — any replacement must
reproduce `Stamp`'s load-bearing IDENTITY semantics (disjointness,
overlap, source-set union, no double-count, `belief.rs:39-48`), and a
`vsa_bundle`-style fold is not automatically a source-set union.

**Bounds respected:** no code changed by this finding; no tenant minted;
no mechanism proposed; no canonizing on either candidate. The open
questions this finding leaves are registered — explicitly as questions,
not a plan — in `.claude/plans/tarski-markov-hhtl-seam-v1.md`.

