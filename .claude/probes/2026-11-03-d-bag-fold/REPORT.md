# D-BAG-FOLD: can a fold establish a relationship that was never stored?

Question: `∃d. P(d)`, `P(d) = R(a,b(d)) ∧ S(b(d),c(d)) ∧ T(c(d),d)`, evaluated
natively, with every edge-presence witness preserved and no composed relation
stored. Probe and expected output: `README.md`.

**Short answer.** Yes, in two lawful-looking arms, both under explicit,
checked witnesses. Each arm meets exactly one gate, and they are different gates:

- **Arm P (projection).** Gather-free, but it copies presence bits.
- **Arm L (lens).** Zero-copy, but it executes `MaskOp::Gather`.

Which arm is admitted is an ownership decision, not a missing capability.

A coordinate locates a candidate. Presence, read inside the fold, decides it.

## The two arms

Both arms use the same preparation, and preparation is ADMISSION B's shape
(`entries/2026-10-20-six-decisions.md` §1). It does counted, version-witnessed,
same-table seeks over an attested sorted key, keyed by each candidate's own
address (`parent(d)`, `parent(parent(d))`).

- **Arm P (projection).** Preparation re-addresses each relation's own presence
  onto the candidate coordinates: `R'[j] = R_in[b(d_j)]`, `S'[j]`, `T'[j]`. No
  plane is a composition. Execution is one Program, `And(R', S', T') → Any | Count`,
  with row-aligned ops only and no `Gather` (asserted).
- **Arm L (lens).** Preparation builds only the pointer tables `$i` (`ib`, `ic`:
  the rows of b and c for each row). Execution is one Program over the node lane:
  `Range(subtree a) ∧ T_in ∧ Semijoin(ib → R_in) ∧ Semijoin(ic → S_in)`.
  `R_in` and `S_in` are the table's own resident planes, read in place. This is
  the zero-copy template of `CLAUDE.md` ("reads column 0 from lane A and column 1
  from lane B at position `$i`").

## PROVEN (executed against an independent nested-loop oracle)

| claim | evidence |
|---|---|
| Both arms answer `∃d P(d)` and the count of distinct valid `d` in one Program each, with no stored composition, no tuple and no cross-program mask handoff | v2 cases 1, 2a-c, 3, 3b, 4, 7, 8: 0 mismatches |
| Each edge's presence is load-bearing in the fold | cases 2a-c; arm-L disables of either `Semijoin` go red |
| Branch-crossing decoys are rejected (S under a b without R; T under a c without S) | cases 3 and 4 |
| A chain held entirely by another classid is not counted, and an edge crossing the classid boundary is refused at build | case 7 |
| `Filter::Semijoin` over resident same-table planes runs in today's mask-risc with no change | arm L, `execute_into` with `Foreign` |

## CONDITIONAL (correct only while a checked invariant holds; silent when the check is removed)

| invariant | unchecked failure (measured) | who checks it today |
|---|---|---|
| plane order = sealed key order | case 5: `(false,0)` vs oracle `(true,1)` | probe recomputes the key digest. In production it is the caller's documented obligation: `quack/src/lib.rs:859-885`, `ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER` OPEN |
| key uniqueness | case 6: `(true,2)` vs `(true,1)` | probe seal refuses. `CountKeyRunsU32` enforces order for its own terminal only |
| presence/version currency | case 8: stale `(true,1)` after S deleted | probe compares versions. `OrderedLaneWitness` (`contract/src/ordered_lane.rs:51-56`) carries a `LanceVersion` for the key lane, nothing for presence planes |
| edge refines its source (child = one-tier extension) | not representable: build refuses | probe only. No production guard located in quack, mask-risc or lance-graph-xlog (closed set, read for this) |

## FALSIFIED

- Prefix-only existence: it says true with R or S deleted (cases 2a, 2b, 4).
  Address compatibility is not edge membership.
- `hi − lo` or `Count` as a distinct count when keys can repeat (case 6; v1 case 2).
- A concept-only (16-bit) prefix as a class boundary (v1 case 3).

## BLOCKED (existing rulings, not missing capability)

- **Arm L:** it executes `MaskOp::Gather`. ADMISSION B requires call-site
  classification of `Filter::Semijoin` / `Pred::EqU32Via` / `MaskOp::Gather`
  before any ruling (six-decisions §1), and that classification has no owner
  (EXO-140). Whether a same-table, address-derived `$i` counts as "a position
  derived from another relational column" is exactly the unanswered question.
- **Arm P:** it copies presence bits into a prepared projection. ADMISSION B
  (2026-10-20) permits "construct a projection for subsequent co-addressed
  evaluation". The zero-copy law in `CLAUDE.md` (2026-10-21, later) admits only
  the pointer table `$i` as a materialization. The two rulings disagree on this
  arm; neither supersedes the other in text.
- **Reusing any fold's output mask as the next hop's input** (the recursive
  route): the scatter survival law (`ir.rs:329-342`) and #1791 (not authorized).

## UNKNOWN

- Arm L's cost, and whether its `Gather` stays tile-local at scale. Not measured;
  the no-ms rule applies before any claim.
- Variable path length or depth not fixed by the address: not tested. It needs a
  per-depth union, or an unbounded `$i` chain.
- General graphs whose edges do not refine their source key: outside this layout,
  so the parent of a node is not address arithmetic. Not shown either way.
- The Bag question (next section): the obstacle is located, not resolved.

## The Bag question

Can an admitted DT64 method ask "does this Bag contain a valid three-hop
consequence"? Traced:

- `MethodBinding` (`dt64/src/lib.rs:197-213`) has ON (change trigger), WHEN
  (`Option<Guard>`) and DO (`effects`). The existence test belongs in WHEN or DO.
  It is not ON: ON detects change, not existence.
- WHEN: `r2il::prepare` refuses any guarded binding: `GuardNotExecutable`
  (`r2il/src/binding.rs:290-291`).
- DO: the only xlog-admitted Bag method shape is
  `[GroupCount, GroupCount, MulI64]`, a two-side count (`xlog/src/lib.rs:384-403`).
  No admitted word sequence expresses `And` over three re-addressed presences
  followed by `Any`.
- Frames: `prepare` checks `r.guards.check(frames, now)` before anything else
  (`binding.rs:284`).

**Location of the obstacle.** It is not the operand layout: both arms above
show the conjunction is expressible. It is not execution capability: both arms
run on today's executor. It is three things:

1. **Method binding.** No admitted word shape, and WHEN is not executable.
2. **Witness admission.** No contract witness covers plane order or presence
   version.
3. **The P-vs-L ownership ruling.**

## Smallest missing pieces

| # | location | owner / contract | smallest missing piece | falsifier | authorized? |
|---|---|---|---|---|---|
| 1 | `quack/src/lib.rs:859-885`, `contract/src/ordered_lane.rs:51` | D-DIAMOND-1, `ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER` | the witness must name the planes it aligns (order + version), checked before execution | v2 case 5 and case 8 unchecked arms must become refusals | no |
| 2 | six-decisions §1 vs `CLAUDE.md` zero-copy law | operator ruling | which arm is lawful: copied projection (P) or `$i` + same-table `Gather` (L) | the same eight cases, under whichever arm is ruled | no; operator decision |
| 3 | `quack` / `mask-risc` call sites of `Gather` / `Semijoin` | EXO-140 (no owner) | classify same-table, address-derived `$i` | arm L green, with the gather classification asserted | no |
| 4 | `r2il/src/binding.rs:290-291`, `xlog/src/lib.rs:399` | DT64 / BindStore | an admitted word shape for presence conjunction → `Any`, or an executable WHEN | a binding that runs v2 case 3 through `prepare`/`run` and is refused for a stale frame | no |
| 5 | layout | storage (mint) | a guard that every stored edge refines its source key | inject a non-refining edge; it must be refused, never stamped | no |

EXO-108 (Bag multiplicity) is untouched. `Count` here counts distinct valid
endpoints because the keys are unique; it is not a path count.
