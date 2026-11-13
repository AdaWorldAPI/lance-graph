# Hidden-state twins: one rule shape, two domains

**Date:** 2026-11-13. **Status:** lab, run locally; 4 tests pass. **Scope:** unadmitted research only.

**Parent:** section 8.5 of the [CE64 logic-neutrino scratchpad](2026-11-13-ce64-logic-neutrino-grammar-spine.md), which proposed this gate. No ClassId, recipe, CE64 bit, F### word, evaluator or production crate is changed.

## Question

Can one guarded-candidate rule shape serve two unrelated domains without a domain-specific guard type, when each domain has twins that look the same and are not?

## The shared shape

`crates/hidden-state-twins-probe/src/lib.rs`, with no chess and no Rust-syntax knowledge:

- a family of at most eight candidate rules;
- a candidate set per row as one byte, bit `i` meaning rule `i` is proposed;
- a WHEN per rule: a conjunction of `lane ∈ {values}` tests over the row's own lanes, stored as a DT64 `Guard` on a registered method and lowered through Quack to one mask-risc program;
- a producer identity carried in every guard as `SetExpr::Sym`.

A rule is admitted, checked against its bound witness and lowered only if some row proposes it.

## Results

| Arm | Twins | Authority | Result |
|---|---|---|---|
| Chess, castling | positions A and B: one board, castling right present / absent | `shakmaty` `legal_moves` | With the right copied into a lane, the guards agree with shakmaty on A, B and D. Without it, B is given a castle shakmaty does not generate. |
| Chess, attacked square | position C: right present, squares empty, a rook attacks f1 | `shakmaty` `legal_moves` | Both guards hold; shakmaty generates neither move. **Pinned as a limit.** |
| Chess, draw claim | two games ending on the initial arrangement, repeated three times / twice | the FIDE rule, read off the move lists | The final positions are equal as `Chess` values and as `Setup` structs, clocks included. A lane counting occurrences separates them; no raw field does. |
| Rust source | two sites reading `(Foo,`: a struct in a return type, a function used as a value | rust-analyzer `StaticIndex`, committed pin | With "is in a signature" as a lane, exactly one rule holds at each of six sites and it names rust-analyzer's symbol. Without it, both twins are given both symbols. |

## What the lanes had to be

The same guard type worked in both domains. What differed is where each lane came from:

1. **A raw field of the row.** The castling right; the byte after an identifier.
2. **A fold over neighbouring text.** Whether a site lies between `fn` and the body's `{`.
3. **A fold over time.** How often a position has occurred. Two producers were tried: FIDE equality counts 3 and 2; structural equality with clocks counts 1 and 1, and the same guard then refuses both claims.
4. **Something only the authority can compute.** Whether a square is attacked. No raw field states it, and a producer for it would be the chess rules themselves.

Kinds 2 and 3 are the same construction as the grammar lab's `head_frame` lane: information from outside the row, written into the row by a producer.

## Kill conditions, as set before the run

- *The two domains need different guard types.* Not triggered. Both arms use one `Family`, one `Rule`, one composition path.
- *The producer cannot be pinned in the method identity.* Not triggered inside the lab: lanes described as coming from another producer are refused (`DescriptionMismatch`). **Limit:** the caller states the producer digest. The lanes carry no provenance of their own, so a caller that lies is not caught.
- *Red-first arm, the draw claim.* Confirmed: positions that are equal field for field receive different answers only through a history lane.

## What this does not show

- That a row-local guard is enough in general. Position C shows it is not, and a move that must not leave the king in check has the same shape: a condition on the state a move would produce.
- Anything about Go or Mississippi Queen. No rules authority for either is available locally.
- Any execution of a transition. Every method here is read-only; the result is a candidate mask.

## Provenance

- `fixtures/ns_twins_foo.pin.tsv` is rust-analyzer's output for `fixtures/ns_twins` (ra_ap_ide 0.0.358, through `crates/sag-ra-probe`), not hand-written. The command is in the file's header.
- `shakmaty` is the AdaWorldAPI fork, a sibling checkout, licensed GPL-3.0-or-later. It is a dev-dependency: only this crate's test binaries link it.

## How to run

~~~sh
cargo test --manifest-path crates/hidden-state-twins-probe/Cargo.toml
~~~

Needs sibling checkouts of `ndarray` and `shakmaty` beside `lance-graph`.
