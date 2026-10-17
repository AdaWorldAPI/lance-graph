# PG-Ring idea catalogue (clean room)

**Status:** WORKING-MODEL. Research only; no code, no decision.
**Source:** Gómez-Brandón, Hogan, Navarro, *Uplifting the Superpowers of
Worst-Case-Optimal Join Algorithms*, ISWC 2026, arXiv 2608.03840v1 (read in
full, appendices A–F included). Implementation structure from a
`ruff_cpp_spo` harvest of `AdaWorldAPI/pg-ring` at `ce012ba` (class, field
and method signatures and call graphs only).
**Licence fence:** PG-Ring is GPL-3.0. Nothing below is code from it; each
row is an idea, stated in our terms, with the paper as the citation.
**Feeds:** D-ASM-11 (leapfrog probe; banner: gather, correct plans D-HOLO
H14c/H14d). This catalogue is prior art to learn from; it authorizes nothing.

## 1. What PG-Ring is

The Ring (Arroyuelo et al., TODS 2024) stores the N triples of a graph as
three wavelet-tree sequences `L_SPO`, `L_OSP`, `L_POS` (the last component of
each triple in three cyclic sort orders) plus count arrays `A_S`, `A_P`,
`A_O` (for each value, how many entries sort below it). Trie nodes of all six
orders become position ranges in one of the three sequences, so leapfrog
triejoin (LTJ) runs in worst-case-optimal time in about the space of the
data. PG-Ring adds property graphs and pushes filters into LTJ: every
condition becomes a **leaptor**, an ordered set answering
`leap(c) = min{c' ∈ S, c' ≥ c}`, and the WHERE clause is a syntax tree of
leaptors intersected inside LTJ instead of before (pre-filter) or after
(post-filter) the join.

Measured (paper §6, Appendix C/E): on Wikidata (109 M nodes, 720 M edges,
1,000 log queries) pushing filters in is 16 % faster than post-filtering
and 15× faster than pre-filtering on average, with one query 5,000× faster
than post-filter. On LSQB it wins the cyclic queries (Q2, Q3, Q6) and loses
the acyclic ones to MillDB, at 32 % of MillDB's space.

## 2. Catalogue

Verdicts: **HAVE** (we already do it), **ADOPT** (idea worth carrying over),
**REFUSED** (the shape breaks a substrate rule; the row says which and what
replaces it), **OPEN** (needs a measurement or a ruling).

| # | PG-Ring idea (paper §) | Our counterpart / gap | Verdict |
|---|---|---|---|
| P1 | Leaptor: any ordered set with `leap(c)`; LTJ intersects leaptors, not just trie children (§4) | `OrderedLane::seek_from` (galloping lower bound from a cursor, `O(1 + log gap)`), `crates/lance-graph-contract/src/ordered_lane.rs:343` | HAVE (the primitive); the leaptor-as-interface is the D-ASM-11 arm A question |
| P2 | Filters pushed into the intersection, not pre/post (§4.1, §6.2: 15× vs pre, 16 % vs post) | Our doctrine already folds every predicate into one mask before the terminal (photolithography; D-HOLO H14d). Difference: they intersect *ordered sets* by seeking, we AND *planes* word-wise | HAVE in principle; the dense-vs-sparse carrier choice is OPEN (see P11) |
| P3 | Alternation complexity: cost of an intersection is the number of switches between sorted sequences, not their sizes (§3.2) | No cost model in our planner uses it; word-wise AND costs `n/64` regardless of alternation | OPEN: a probe on clustered vs interleaved sets would show where seek beats AND |
| P4 | Property range filter as a grid `M_π`; `leap` = orthogonal range successor `leftmost([c,∞]×[a,b])` (§4.1) | Property values sit at the row's coordinate, so `a ≤ x.π ≤ b` is a range compare to a mask over the property plane, and successor is the next set bit | HAVE (as mask + next set bit); no grid structure needed because the value is co-located |
| P5 | Same-variable range conditions merged into one before execution (App. B.1) | `fold_ranges` merges `Cmp::Range` on one column inside an `And`, `crates/lance-graph-quack/src/lib.rs:1918` | HAVE |
| P6 | Labels as sparse bitvectors; `leap(c) = select1(B, rank1(B, c−1)+1)`; the complement leaptor comes free from `select0` (§4.3) | Presence bit + rank in its own mask is our one allowed gather-like operation; next-unset-bit is the complement | ADOPT: name "next set / next unset bit after c" as the label leaptor; no new carrier |
| P7 | Label membership as runs: `label_nodes` keeps run bits, a run rank and a next-zero per run, giving `next_node` / `next_neg_node` (harvest `include/label_nodes.hpp`) | Run-shaped presence (`Runs` IR, D-TELEPORT range intersections) | ADOPT: run-encoded presence answers both `next` and `next_neg` in O(1) per run; worth a v4 measurement against bit scan on clustered labels |
| P8 | Edge identity = position in the POS order; no stored edge id (§5.2) | "A value's position is its address"; implicit registers | HAVE (same doctrine, independently reached) |
| P9 | Edge-label Boolean expressions compiled to a set of disjoint position ranges: complement = two ranges, OR = union, AND = intersection (§5.2) | Schnittpunkt computation: intersections of coordinate ranges worked out symbolically before execution (D-TELEPORT #1484/#1487, `fold_ranges`) | HAVE; PG-Ring shows the same trick works for label algebra when edges are sorted by label first |
| P10 | Boolean leaptor tree: negations pushed to leaves, per-variable reduction (`trivial AND X = X`, `trivial OR X = trivial`), bound variables fold conditions to true/false (App. A.3) | relalg `normalize`; D-HOLO H14d logic stage (34 % of 2,000 programs decided symbolically) | HAVE (normalization); ADOPT the per-variable reduction as an explicit rewrite with its own proof test |
| P11 | AND children ordered most-selective first, restarting at the leftmost after each hit (App. B.1) | Fold order in a compiled mask program does not change the result; for a dense AND it barely changes cost | OPEN: matters only on a sparse (seek) carrier; ties to the fold-carrier question |
| P12 | Count arrays `A_*` give each constant's range in O(1): the range of `(s,*,*)` is `[A_S[s]+1, A_S[s+1]]` (§4.3) | Prefix-sum lanes computed at seal, the open decision "prefix-sum lanes at seal vs fold over the aperture" | ADOPT as evidence for the prefix-sum option: for a query constant this is a range bound, not a data-indexed read |
| P13 | Tracking a triple across orders, `i' = A_O[o] + rank_o(L_SPO, i)` with `o = L_SPO[i]` (LF mapping, §4.3) | Reads a position computed from another sequence's value: a data-indexed read | REFUSED (gather). We co-locate the pair at one coordinate instead (`s_dst[i][a]:rb[i][a]`). Consequence: where the Ring stores 3 orders and tracks between them, we must store every order we read, so the space trade differs; record it, do not port it |
| P14 | Three cyclic orders suffice for all six trie orders because of tracking (§4.3) | Without tracking, each needed order is its own sealed lane | OPEN: count which orders our workloads actually read before assuming six |
| P15 | Adaptive variable elimination order: choose the next variable after each binding, by the smallest locus range length (App. B.1, after the Ring paper) | Range length is a population count we already have per aperture | ADOPT for D-ASM-11's arm A; the paper reports adaptive beats static |
| P16 | VEO classes: non-lonely node variables first, then non-lonely edge variables, then lonely variables last (App. B.1) | No VEO in our stack yet | ADOPT as the default order heuristic for the probe |
| P17 | Selectivity: AND = product, OR = 1−(1−s1)(1−s2); labels by fraction of set bits; property comparisons by uniform area of the min/max box; exact counting only for the static VEO because it costs O(log) per branch (App. B.1) | Popcount gives the exact fraction for free on a sealed plane | ADOPT, with popcount in place of the uniformity guess where the plane is resident |
| P18 | Identifier comparisons (`x < y`) as a leaptor: `leap(c) = c` if allowed else +∞ (App. A.2), used to drop symmetric duplicates | A range bound on the variable's coordinate given the bound one | HAVE in shape (range bound); no work |
| P19 | Order-preserving integer encoding per value type, plus a front-coded string dictionary; strings only at the edges of a query (App. B) | Label codebook; "strings are never an address, only a label" | HAVE (same rule) |
| P20 | Bag semantics, homomorphic matching (App. B) | D-ASM-11 Q-A is checked as a bag against datafrog | HAVE (oracle already bag-shaped) |
| P21 | Read-only index; updates via dynamic or adaptive dynamic bitvectors at a query-time penalty (§7) | Sparse delta per thought table, merged on read (D-HL) | HAVE (different mechanism, same goal) |
| P22 | Benchmarks: LSQB Q1–Q6 (cyclic Q2, Q3, Q6) and a 1,000-query Wikidata log sample with filters (App. C, E) | Our cyclic fixture is F4 (#1587) | ADOPT: LSQB Q2/Q3/Q6 shapes as additional cyclic fixtures; Q1/Q4/Q5 as acyclic controls where a semijoin chain should suffice (GYO, P-3) |

## 3. What this changes for D-ASM-11

1. Arm A already has its leap: `seek_from`. What PG-Ring adds is the
   interface around it (P1), the adaptive order (P15–P17) and the evidence
   that filters belong inside the intersection (P2).
2. The Ring's space advantage rests on tracking (P13), which our rules
   refuse. A clean-room port therefore cannot reproduce its space numbers;
   the probe should compare time per element on the v4 arm only, and record
   lane count separately.
3. Dense AND vs sparse seek is the real question (P2, P3, P11): PG-Ring
   wins where sets are sparse and clustered. A falsifiable probe: the same
   cyclic query on interleaved vs clustered identifiers, both carriers,
   answers equal, ns per element printed with the tier.

## 4. Open points

- O1: Is P7 (run-encoded label presence) faster than a bit scan on v4 for
  realistic label clustering? Not measured.
- O2: Which sort orders do our workloads read (P14)? Not counted.
- O3: The extended version with appendices was cited as "to be published";
  the arXiv text read here already contains Appendices A–F.
