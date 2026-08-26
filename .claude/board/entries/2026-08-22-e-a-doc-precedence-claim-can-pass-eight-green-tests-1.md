## 2026-08-22 — E-A-DOC-PRECEDENCE-CLAIM-CAN-PASS-EIGHT-GREEN-TESTS-1 — a two-gate ORDER is unfalsifiable until a fixture trips both

**Status:** FINDING (measured, this session — the mutation was run, not
reasoned about). **Confidence:** High; the instance is reproducible from the
commit.

`recipe_vocab::refusal_of` (D-ACR-9 second half) applies two gates and its doc
committed to a precedence: *"a recipe that is both too deep AND ungrounded
reports `AboveCeiling`, deterministically."* Eight tests were green, including
one written specifically for the refusal surface, asserting all three variants
fire.

**Swapping the two checks changed nothing.** All eight still passed. The
`AboveCeiling` assertion used a fully grounded fixture, so `nan_disqualifier`
returned `None` for every recipe and the order could never be observed. The
claim was **doc-only** while looking tested.

The fix was a fixture, not a rewrite: search `dispatch_order()` for an id where
`rung(id) > ceiling` **and** `nan_disqualifier(&holed, id).is_some()` — recipe
17 (CDI, Revision, rung 6) under an emptied belief set — and assert the outer
gate is what gets reported. The mutation then fired with exactly that id named.

**The generalization, and it is the point:** the falsifiability rule already
says *"a doc-comment claim is not a behaviour"* and *"a guard needs a
can-it-fire test"*. This instance adds the case those two do not cover — **a
claim about the INTERACTION of two guards that each fire correctly on their
own.** Every individual variant here had a real fixture; what had none was the
overlap. So:

> When a function applies N gates and documents which one wins, the test suite
> needs a fixture in the INTERSECTION. Fixtures that trip one gate each prove
> the gates, never the precedence — and N passing single-gate tests read
> exactly like coverage.

The overlap fixture is also the one that must be searched for rather than
hardcoded: writing `op_of(17)` would pin a recipe id that a re-tiering moves,
and the search doubles as the anti-vacuity assertion (`.expect("some deep
recipe must also read the emptied beliefs")`).

Method note: the only reason this was caught is that the mutation was
**executed**. Eight green tests plus a plausible doc sentence is precisely the
shape that survives review.

