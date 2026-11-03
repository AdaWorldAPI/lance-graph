# D-BAG-AWARENESS — co-addressed folding hypothesis (read-only, lance-graph `514948203`)

Question: can `∃B,C. R(A,B) ∧ S(B,C) ∧ T(C,D)` be evaluated as ONE lawful fold, with no
intermediate stored relation, no computed-mask handoff, no value-directed gather and no
tuple materialization? Governance findings from SIDE-BOUNDARY.md are taken as given.

## Verdict

**RESTRICTED.** It is one fold only when the composition was already decided by the
ADDRESS at mint time: each hop is "the target key extends the source key by one tier".
Then the three-hop existential is a single key interval on one sealed lane: one
`Cmp::Range` and a `Count`/`Any` terminal that answers from `hi - lo`. For arbitrary
relations it is **BLOCKED**: every route on record needs a gather, a scattered-mask
intermediate, a stored composed relation, #1791 or a pull lane.

So co-addressing is a LAYOUT answer for refinement relations and a NEW CAPABILITY for
everything else. Laying the path out at one coordinate for arbitrary R, S, T means
storing R∘S∘T, which is the intermediate relation the hypothesis excludes.

## Findings

| # | Claim | Class | Evidence |
|---|---|---|---|
| 1 | `fold_ranges` merges `Range` leaves on the SAME `col` that are direct siblings of one `And`. Nothing else: no cross-column binding, no flattening, no merge across `Or`/`Not`. It cannot state `∃B,C`. | PROVEN for its scope; BLOCKED as a chain mechanism | `quack/src/lib.rs:1932-1983` (doc + body); tests `range_fold_tests` 5094ff |
| 2 | A range can come from an ADDRESS without reading a row: `Filter::prefix_facet` turns a `SemanticPrefix` into `Cmp::Range{lo,hi}` under an `OrderedLaneWitness` the sealed lane accepts. Without a witness it falls back to a `MatchU64` sweep and reports the reason. | PROVEN (single lane) | `lib.rs:888-940`; `an_address_prefix_selects_exactly_a_contiguous_trie_subtree` (3869, two-sided halving); `a_witnessed_prefix_is_a_range_on_either_sweep` (4972) |
| 3 | For refinement relations (`B` has prefix `A`, `C` has prefix `B`, `D` has prefix `C`), `R∘S∘T(a)` = the keys on T's D lane with prefix `a` at depth `d+3`. That is one range, one terminal, zero rows read under `Count`/`Any`. This is the canon's "is_ancestor_of = centroid-tree containment" and "path = HEEL+HIP+TWIG". | RESTRICTED; the collapse is NOT VERIFIED (no test chains relations, it follows from prefix semantics only) | items 1-2; OGAR CLAUDE.md tier interpretation |
| 4 | The plane-order precondition of #2 is unenforced: a `Range` is ordinals in the SEALED lane's order, `Pred::Range` reads no lane, `Planes` carry no order identity. | RESTRICTED (open hazard) | `lib.rs:859-885`; `ISSUES.md:753` `ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER` OPEN |
| 5 | The refinement property itself is not checked anywhere. A relation that is lowered as a prefix but contains one non-refining edge gets a wrong answer, silently. It is a property of the mint, with no guard. | NOT VERIFIED (no guard found in quack, mask-risc or the xlog cube) | absence within those three crates only; open negative, read as "no guard located" |
| 6 | Star joins (one shared center: `R(x,c), S(c,z), T(c,q)`) are expressible co-addressed with no gather: per-side fold, `GroupReduce Count` by slot, `And` of slot planes, bag count as a co-addressed dot product. A CHAIN through two join domains is "not built". | LAB ONLY (correctness measured, ms gate FAILED); chains NOT VERIFIED | `entries/2026-10-21-u4-coaddressed-join.md` (Status, Open: "a chain through two domains is not built") |
| 7 | General composition as a gather-free tile product: `C \|= broadcast(row y of A) & spread(bit y of B)`, a `Ternlog 0xF8` Program per coordinate `y`. Gather-free in execution, disable-verified. But it (a) STORES the intermediate `A∘B` before composing with T, (b) feeds a `Keep` mask back as `Plane(0)` 256 times per tile product from a host loop, (c) copies rows into `bcast`/`spread` buffers. So it breaks no-intermediate, is a computed-mask handoff and materializes copies. | LAB ONLY; fails the hypothesis on three counts | `crepe-plan-probe/examples/walk_tile.rs:20-32, 319-405`; `entries/2026-10-21-d-ru-6-tile-walk-probe.md` (1.0-9.3x slower count-only, ADMISSION B not demonstrated) |
| 8 | Arbitrary R, S, T as one fold. | BLOCKED | needs a gather (#1590 ban, ADMISSION B), a scattered intermediate (`ir.rs:329-342` survival law), a pull lane on the target rows (RF-CHAIN alternative, `cypher-mask-lowering-v2.md:291-292`, not ruled), #1791 (not authorized) or OP-14 (open) |
| 9 | Bag counting over the chain (walk counts). Not addressed by any of the above; a range count is a SET count of distinct D keys, never a path multiplicity. | Separate: EXO-108 stays OPEN | cypher-quack gap doc `lib.rs:172-186` ("needs a foreign-value sum") |

## Invariants the restricted shape needs (every one is load-bearing)

1. **Refinement at mint.** Each hop's target key extends its source key by exactly one
   tier. This is the whole reason the composition collapses. It is unguarded (#5).
2. **One sealed lane, in execution order.** The final range is ordinals in T's sealed
   D lane, and the planes must be that lane's rows in that order (#4, open).
3. **Witness currency.** `OrderedLaneWitness` version and digest match the lane; stale
   means sweep, never a silent range (`a_prefix_aperture_under_a_stale_witness_sweeps_and_says_why`).
4. **Prefix, not aperture-with-hole.** A care mask with a hole is a predicate and
   always sweeps (`an_aperture_with_a_hole_sweeps_even_under_a_witness`). Still one
   fold, but no longer O(1).
5. **ClassView scope.** The carving that makes a byte a tier is per classid
   (longest-prefix codebook binding). A chain that crosses classids crosses codebooks,
   and nothing shows that the prefix reading survives that. NOT VERIFIED.

## Failure cases

- A non-refining edge anywhere in R, S or T: wrong answer, silent (#5).
- Planes not in sealed-lane order: unrelated rows selected, silent (#4).
- Hops with different depths per row (variable-length paths): no single depth, so
  there is no single prefix width. Expressible only as an `Or` of ranges, which
  `fold_ranges` leaves unmerged.
- Any non-hierarchical relation (cross-links, many-to-many, cycles): not an interval;
  falls to #8.
- Bag semantics (how many paths): a range counts distinct keys; EXO-108.
- Star shape: lab only (#6); a chain through two domains is not built.

## Falsifier (proposed, not built)

On a sealed fixture with three refinement relations, compute `R∘S∘T` existence as one
`prefix_facet` range plus `Count`, and assert it equals the oracle (`dr-math-binding-probe`
`Chain::eval` or a nested loop). Two disables, both required to go red:

- (a) permute the planes relative to the seal. If the test stays green, #4 is
  confirmed as an undetectable hazard.
- (b) inject one non-refining edge into S. The range answer must diverge from the
  oracle. That proves invariant 1 is load-bearing, and it shows a guard is needed.

Silence twin: the same program on a fixture whose S is genuinely a refinement must stay
equal to the oracle at every depth.

Nothing was implemented, mask-risc was not touched, nothing pushed.

---

# Addendum: the missing logical invariant, one falsifier (closing)

Probe: `/tmp/claude-0/-home-user/89dae588-85f2-539f-94b2-003b808f478e/scratchpad/cofold/src/main.rs`.
It sits outside the repo and runs the real `lance_graph_quack::lower` and the real mask-risc
`execute_into`. Run it with `cargo +1.99.0 run` (ndarray 0.17.2 needs rustc 1.99). One row
per node, sorted by key. Key layout: `u64 = classid(32) | t1 | t2 | t3 | t4`, 8 bits per tier.
Presence planes `R_in`, `S_in` and `T_in` mark each edge at its child row.

Two methods:

- **P, the single-range reduction from the verdict above:** `Count(T_in ∧ prefix(a))`.
- **C, one fold over co-addressed presence for fixed endpoints a..d:**
  `Or(R_in@b, S_in@c, T_in@d)` must select exactly 3 rows. `b` and `c` are prefixes of
  the query constant `d`, located by their ordinal in the sealed order. That is
  representation translation of a constant, never a value read from a row. One
  Program, one terminal, no gather, no produced mask, no intermediate.

## Measured output

| case | oracle | P | C |
|---|---|---|---|
| 4 valid hierarchy | true | 1 → true | 3/3 → true |
| **0 R(a,b) present, S(b,c) DELETED, T(c,d) present** | false | **1 → true (FALSE POSITIVE)** | 2/3 → false |
| 0b R deleted | false | 1 → true (false positive) | 2/3 → false |
| 0c T deleted | false | 0 → false | 2/3 → false |
| 1 planes stored reversed against the sealed ordinals | true | 1 → true (the sweep reads the lane) | **1/3 → false, no error raised** |
| 2 D lane `[d, d, d2]` | — | range rows = 3, distinct D = 2 | — |
| 3 other classid, same concept hi u16 and same tiers | — | 40-bit prefix (full classid) = 1; 16-bit prefix (concept only) = 2 | — |

## Classification

- **P (single-range reduction): FALSIFIED for its intended hierarchy.** Hierarchical
  address compatibility is not relational composition. P reads only T's presence, so a
  deleted R or S edge is invisible to it by construction. P would be equivalent only
  under prefix-closed edge membership: T present at d implies every ancestor link is
  present. No contract states that and no guard enforces it, so P is at best
  CONDITIONALLY equivalent under an assumed invariant. In practice it is false: a
  deletion breaks the invariant. This supersedes finding #3 above: RESTRICTED is
  downgraded to FALSIFIED unless prefix-closed membership is enforced.
- **C (direct conjunction over co-addressed presence): CONDITIONALLY equivalent.** It
  matches the oracle on all four presence cases, and each of R, S and T is load-bearing
  (0, 0b, 0c). It evaluates the relational conjunction itself in one fold. Its open
  assumption is plane order (case 1: a silent wrong answer, the same hazard as
  `ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER`). Not PROVEN-under-enforced-invariants
  until storage exposes the seal's permutation, or execution checks the lane digest.
- **Distinct count (2): FALSIFIED as stated.** `hi − lo` counts lane rows, not distinct
  keys. It equals the distinct count only on a key-unique lane. `Agg::CountDistinctOrderedU32`
  is the shipped terminal for that (it enforces key order), and only for u32 lanes.
  Path multiplicity remains EXO-108.
- **Cross-ClassId (3): RESTRICTED.** The prefix stays inside one classid only when it
  covers all 32 classid bits. A concept-only prefix spans app prefixes, and so spans
  their codebooks. Tiers alone cannot distinguish the classes.

## What this does and does not show about the larger idea

The single-range shortcut fails. The larger idea survives: a fold CAN evaluate the
relational conjunction directly from co-addressed presence, for fixed endpoints, with
none of the excluded mechanisms (method C).

**Open, not shown:** the existential `∃B,C,D under a`. That needs, at each candidate d,
its ancestors' presence. In a sorted sparse lane the parent's ordinal is not
address-arithmetic, so that read is either a derived prefix-closed plane (forbidden here)
or a value-positioned read (gather). In a DENSE co-addressed level layout (canon
256-cell tiles), `parent(d)` is a fixed shift. There the check is a broadcast lens of
the level-3 S plane over its 256 children. mask-risc has no such lens op, and
`walk_tile` emulates it with copied `spread` buffers. Status: NOT VERIFIED, and it needs
a mask-risc capability (frozen, EXO-105).

Investigation closed. No repo files changed, mask-risc untouched, nothing pushed.
