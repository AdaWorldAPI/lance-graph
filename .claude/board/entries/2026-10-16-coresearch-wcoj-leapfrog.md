# 2026-10-16 — co-research: worst-case-optimal joins (leapfrog triejoin family) for the join path

**Status:** EXPLORATION MAP (nothing adopted; the operator chooses). Feeds D-ASM-11.

Harness: `.claude/agents/coresearch-council.md`. Five scouts (code, internal prior
art, literature, systems, concepts), a main-thread crosswalk, three
co-architects (bridge, firewall and fit, falsifier). Every literature item below
was read at section level or better unless marked; numbers marked "measured"
are the papers' own, re-checked against the extracted texts for 2411.04042,
2301.10841 and 1602.08721.

## The question, after the premise gate

The first framing set leapfrog against "the semijoin chain" as two competing
joins. The premise audit returned PREMISE-SPLIT: a semijoin chain produces a
head-table MASK (existence), leapfrog produces cross-table BINDINGS (answers).
The question became two:

- **Q-E (existence half):** for cyclic bodies, can a per-variable leapfrog
  intersection over sealed lanes give the exact head-row mask where the semijoin
  chain over-approximates?
- **Q-A (answer half):** is leapfrog enumeration the carrier D-ASM-3 lacks for
  cross-table projection and foreign fan-out, emitting keyed counts instead of
  tuples, with `materialize_bindings` as the one exit?

## What the outside world settles

1. **Acyclic bodies need no WCOJ.** Semijoins give the exact existence mask on
   α-acyclic bodies: root the join tree at the head relation and one bottom-up
   pass leaves exactly the head rows that extend to an answer (Yannakakis; FAQ
   1504.04044 §2.2). A plan is instance-optimal when every shrinking step runs
   before any expanding step, and such a plan exists iff the body is acyclic
   (Shredded Yannakakis 2411.04042; measured in DataFusion: faster on 85.3% of
   1,849 queries, up to 62.5x). Every system that measured it (Umbra, Free Join,
   LogicBlox, Graphflow) ships a hybrid: WCOJ only on the cyclic or growing
   part. This matches the earlier internal doctrine (D-V4-FOLD lab: "WCOJ only
   for cyclic patterns"), which now has its sources.
2. **Cyclic bodies break the semijoin chain.** Triangle Q(a,b,c) :- R(a,b),
   S(b,c), T(a,c) with R = {(0,0),(1,1)}, S = {(0,1),(1,0)}, T = {(0,0),(1,1)}:
   Q is empty, yet every semijoin keeps every row (checked by hand). 2+2+2 is the
   smallest such instance. No cyclic fixture exists in the workspace today.
3. **The cyclic existence mask is a range intersection per head row.** For each
   head row (a,b): is S_b ∩ T_a non-empty? Total Õ(√(|R||S||T|)) = N^1.5 by
   Cauchy–Schwarz (Ngo 1803.09930 §2); optimal for any variable order provided
   each one-variable intersection costs about its smallest set (Generic Join
   1310.3314). No combinatorial method is expected to do much better: triangle
   detection's best known bound is m^1.41 via fast matrix multiplication
   (1402.0054).
4. **One algebra, two semirings, answers both halves.** FAQ / InsideOut
   (1504.04044): variable elimination over a commutative semiring. The Boolean
   semiring gives the existence mask (other atoms enter as indicator
   projections, i.e. a simultaneous semijoin reduction); (ℕ,+,×) with the head
   key free gives keyed counts without enumerating bindings. EmptyHeaded runs a
   generic join inside each decomposition bag with semiring early aggregation;
   CLFTJ caches counts at decomposition adhesions (measured, 5-cycle memory
   accesses: LFTJ 45e9, Yannakakis 16e9, CLFTJ 1.4e9; 246x when fully cached).
5. **Tries are a view, not a store.** LFTJ needs only `seek`, `next`, `open`,
   `up` over sorted data; `seek` must be amortized O(1 + log(N/m)), which needs
   a seek that gallops forward from the cursor (1210.0481 §3; SIMD galloping is
   the right kernel above a length ratio of ~1000, 1401.6399; EmptyHeaded
   switches at 32:1, so the threshold must be measured on v4).
6. **Free-connex is the enumeration boundary.** Free-connex acyclic queries
   enumerate with constant delay after linear preprocessing; for self-join-free
   queries that is tight under the BMM assumption (2003.01075). D-ASM-3's
   cross-table refusal therefore has a hardness argument, not only a policy one.

## What we already have (verified in code)

- `SealedFacetLane::bound_aperture` (`lance-graph-contract/src/ordered_lane.rs:344-365`):
  two fresh whole-lane `partition_point`s per call; no cursor, no galloping.
- `SemanticAperture` (`facet.rs:634-748`): a bit-granular key prefix = one closed
  interval, so a trie level exists as "a longer aperture"; no next-distinct step.
- quack `Filter::Semijoin` → `MaskOp::Gather` over a resident foreign plane;
  `fold_ranges` intersects same-column ranges only; D-MTP-1 (two-hop == datafrog
  10²..10⁶) and D-MTP-1b (hop as a preimage range, 0.0001–0.0004 ns/row on v4).
- `GroupReduce{Count|Min|Max|SumSym}` into a dense `Out::I64`, keyed Local, Via or
  same-table Pair.
- Absent: a cyclic fixture; a second sealed ordering of one relation; a join node
  in relalg; any galloping or sorted-set intersection in `ndarray::simd`;
  `materialize_bindings` (a plan name only). The planner's `WcoJoin` is a logical
  node with a cost and no executor. datafrog is a SET oracle (sorted,
  deduplicated); its fork carries `leapjoin` and `gallop`, unused by our probes.

## Crosswalk and verdicts

Firewall seams that every adopted row must keep:
- the per-variable frontier stays inside ONE mask-risc program and never crosses
  as a `ForeignPlane` (the 2026-10-03 PREMISE-WRONG ruling);
- a second ordering is a permutation index (sorted key, row ordinal), sealed at
  the same version and witness as the base lane, never a payload copy;
- semirings on this path are integer only (Boolean, ℕ, min/max). They share
  one intersection kernel and differ in the sink: Boolean writes a mask (a
  population, the Q-E answer), ℕ / min / max write `GroupReduce`'s keyed
  `Out::I64` (the Q-A answer). No new executor;
- routing lands in relalg / R2IL, not in the planner (DataFusion grace period).

| # | idea | relation | verdict | next step |
|---|---|---|---|---|
| X1 | galloping seek from the cursor | PARTIAL | **PROBE P-1** | `seek_from` on the sealed lane; kernel in `ndarray::simd` |
| X2 | trie level = aperture (`open`/`up`) | PARTIAL | folded into X1/X7 | — |
| X3 | GYO acyclicity test | NEW | **PROBE P-3** | compile-time, over a coordinate-only join body in relalg |
| X4 | Yannakakis rooted at the head | ALREADY-HAVE (rule unstated) | **ADOPT-NOW (rule)** | write the root-at-head rule beside the semijoin lowering |
| X5 | shrink-before-expand | PARTIAL | **ADOPT-NOW (rule)** | no expansion before every shrinking step; expansion only via `materialize_bindings` |
| X6 | cyclic over-approximation fixture | NEW | **ADOPT-NOW (fixture)** | 2+2+2 triangle, ×k, committed with the D-ASM-11 probe |
| X7 | per-head-row range intersection (Q-E) | NEW | **PROBE P-7** | first an insufficiency proof that existing mask-risc ops cannot express it (session rule: no opcode just to pass an experiment); then one fused op if needed |
| X8 | second sealed ordering | NEW | **PROBE P-8** | does one ordering plus `bound_aperture` keep the cyclic slope ≤ 1.5? |
| X9 | hybrid rule (WCOJ only on the cyclic part) | doctrine, no code | **PROBE (D-ASM-11 arm A3)** | router in relalg/R2IL from the X3 verdict |
| X10 | GHD: WCOJ in bags, Yannakakis across | NEW | PARK | after X3 and X7; barbell fixture |
| X11 | one intersection kernel, two sinks | PARTIAL | **PROBE P-11** | X7's kernel feeding (i) a mask for the Boolean semiring and (ii) `GroupReduce{Count}` for ℕ; the second closes the foreign fan-out undercount without tuples |
| X12 | count passing (dot of folds) | PARTIAL | **PROBE P-12** | promote `TwoFoldsAndDot` to the acyclic count arm; typed refusal for count-distinct across factors |
| X13 | factorized bag | NEW | PARK | only behind `materialize_bindings` when a projection demands it |
| X14 | free-connex boundary | ALREADY-HAVE | **ADOPT-NOW (doc)** | cite 2003.01075 beside D-ASM-3's refusal |
| X15 | mask × sorted-run kernel menu | PARTIAL | PARK | after P-7 shows intersection dominates |
| X16 | smallest range proposes, others confirm | PARTIAL | folded into P-7 | `bound_aperture` width is already the count step |
| X17 | bag oracle | NEW | **ADOPT-NOW (test infra)** | datafrog + a multiplicity column, checked against a brute-force bag; gates P-11/P-12 |
| X18 | hash tries | CONFLICTS-ANCHOR (A3) | SKIP | lazy levels and singleton pruning are implied by apertures plus early exit |
| X19 | MORK leapfrog + differential corpus | oracle only | SKIP (code), keep the corpus shape | MORK stays an external oracle binary (D-ASM-10) |
| X20 | worst-case cost N^1.5 | concept | prediction | the AGM / Cauchy–Schwarz bound N^1.5 is reached only on the adversarial F3b; F2 is expected near 1.0. A slope well under 1.5 on F3b means the fixture is not adversarial (m^1.41 is the best known detection bound via fast matrix multiplication, not a floor on any one instance) |
| X21 | i-cost (range lengths touched) | NEW | **ADOPT-NOW (metric)** | reported beside v4 ns/element, never instead |
| X22 | heavy/light with Boolean matrix products | NEW | PARK | skewed cyclic fixture only; bitset/popcount products only |

Licences of systems read for concepts (nothing borrowed as code): datafrog
Apache-2.0/MIT (dev-dependency oracle only), PathMap MIT, DuckDB MIT, Soufflé
UPL-1.0, MORK undeclared (oracle binary only).

## Probes (pre-registered, nothing run)

Fixtures: F1 acyclic chain R(a,b),S(b,c),T(c,d), 10²..10⁶, with 90% dangling
tuples and n:1 / n:m (degree 1, 10); F2 uniform triangle (average degree 8);
F3a skewed triangle (one hub of degree √N); F3b adversarial grid (√N×√N, S_b even
keys, T_a odd keys, every intersection empty); F4 the 2+2+2 counterexample ×k.
Oracles: datafrog SET for Q-E, the X17 bag for Q-A. Cost: v4 ns/element plus
i-cost; "slope" = d log(i-cost) / d log N.

| probe | kill condition | disable run (must go red) |
|---|---|---|
| P-1 seek_from | comparisons > m·(2·log₂(N/m)+2), or no speedup over fresh `partition_point` at N/m ≥ 10³ | reset the cursor every seek: comparisons equal fresh |
| P-3 GYO | any misclassification of chain, star, triangle, 4-cycle, barbell | force "acyclic": the triangle goes red |
| P-7 cyclic mask | mask ≠ SET on F2/F3/F4; F3b slope outside 1.5 ± 0.1; F2 slope > 1.2 | always iterate the larger side: F3 slope rises toward 2 |
| P-8 second ordering | if one ordering keeps slope ≤ 1.5 for both variable orders, the second is unnecessary | drop it: T_a stops being one range, slope ≥ 2 |
| P-11 two semirings | count ≠ bag on n:m (would reproduce the fan-out undercount), or allocation grows with \|out\| | swap × for max: n:m red and n:1 green |
| P-12 dot of folds | mismatch with the bag, or cost grows with \|out\| | + instead of ×: red; count-distinct across factors refused |
| X17 bag oracle | any disagreement with a brute-force nested loop | dedup inside the oracle: n:m red, n:1 green |

**D-ASM-11 arms:** A0 semijoin chain; A1 leapfrog, variable order (a,b,c); A2
leapfrog, order (c,a,b); A3 hybrid, routed by two predicates: for Q-E,
α-acyclic → A0 and cyclic → leapfrog; for Q-A, acyclic and free-connex → the
X12 dot of folds and cyclic → leapfrog feeding `GroupReduce{Count}`; the
leapfrog arms let the smallest range propose and seek by galloping. A1′ = A1
with a fixed proposer and no galloping (the disable arm).

| fixture | A0 | A1 | A2 | A3 | A1′ |
|---|---|---|---|---|---|
| F1 chain (Q-E) | == SET, fastest | == SET, ns ≥ A0 | == SET | within 10% of A0 | — |
| F4 ×k | keeps all 6k rows (wrong by design) | ∅ | ∅ | ∅ | ∅ |
| F2 uniform | strict superset of SET | == SET, slope ≈ 1.0 | within 2× of A1 | == A1 | ≈ A1 |
| F3a skewed | strict superset | slope ≤ 1.5 | i-cost within 3× of A1 | ≤ 1.1 × min(A1, A2) | i-cost ≥ 10× A1 at 10⁶ |
| F3b adversarial | strict superset | slope 1.5 ± 0.1 | same | same | slope ≈ 2 |
| F1 n:m (Q-A) | n/a (a mask cannot carry the count) | keyed count == bag | == bag | == bag via X12, no tuples allocated | — |

D-ASM-11 is refuted if any leapfrog arm's mask differs from the SET on
F2/F3/F4, if A3 loses to the best single arm by more than 10%, if the F3b slope
falls outside 1.5 ± 0.1, or if the A1′ disable fails to move i-cost by at least 10×.
If A0 equals the SET on F4 the fixture is wrong and the run is void. X17 and P-3
must be green before any timing is taken.

## Not searched / not read

Ngo survey beyond §1–3; boxing 1501.06689; the Minesweeper PODS'14 certificate
paper; factorized size-bound proofs; BKN Lookup & Expand; the 2601.00098 survey;
Han et al. SIMD intersection; LogicBlox / RelationalAI internals; Kùzu primary
sources (its documentation domain did not resolve). "Not found" in this map means
not found in these sources.

## OPEN

- Whether the X7 kernel needs a new mask-risc op, or is expressible with existing
  ops — the insufficiency proof comes first.
- Whether a second sealed ordering is needed at all (P-8).
- Hyperon and PeTTa licences are not recorded here; record them before relying
  on them as bag oracles.

## Premise gate, second pass (on the map's options)

Three rows were corrected before landing: X11 (one kernel, two sinks: a mask is
not a keyed count); the A3 router (the Q-A boundary is free-connex, so acyclic
counting goes to X12, not to the semijoin chain); X20 (N^1.5 is the adversarial
worst case; 1.41 is a detection bound, not a floor). The A1′ ratio is in i-cost.
