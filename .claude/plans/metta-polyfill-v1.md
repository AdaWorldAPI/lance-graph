# metta-polyfill-v1 — Hyperon/MeTTa over the masking-algebra polyfill

**Status:** PROPOSAL (D-MTP-0..5). Nothing beyond D-MTP-0 (this map and its
measurements) is authorized. No crate exists yet.
**Date:** 2026-10-14.
**Repos:** lance-graph (`lance-graph-relalg`, `lance-graph-r2il`,
`lance-graph-mask-risc`, `lance-graph-quack`, `lance-graph-nal`),
`AdaWorldAPI/hyperon-experimental` (fork, MIT), `AdaWorldAPI/petta` (MIT).
**Parents:** `.arxiv/2026-10_hyperon_metta_to_lance_graph_parity_map.md`
(guardrails, HY-Q01..15, probes P0–P7), `nal-polyfill-v1.md` (the template:
text front end → bind → procedures → `FoldDialect` → mask-risc),
`lance-graph-relalg` (#1546, the describing IR).

## 1. The question

Can MeTTa serve as an IR for the masking algebra, and where does
hyperon-experimental attach so that the atoms MeTTa uses are served by the
polyfill rather than by Hyperon's own trie? Should the crate be
`lance-graph-cog` or `lance-graph-metta`?

## 2. What Hyperon is, at the seams (VERIFIED-IN-CODE, fork `1ffdbd6`)

- **Atoms:** `Symbol`, `Variable`, `Grounded`, `Expression`
  (`hyperon-atom/src/lib.rs`), matched by `match_atoms` / `Bindings`
  (`matcher.rs`).
- **Interpreter:** minimal MeTTa is one dispatch over 12 instruction symbols
  (`lib/src/metta/interpreter.rs:415-458`: `eval`, `evalc`, `chain`,
  `function`, `collapse-bind`, `unify`, `decons-atom`, `cons-atom`,
  `superpose-bind`, `metta`, `context-space`, `call-native`; `function` is an
  internal state that panics if reached). The full language (types, `let`,
  `if`, `case`, …) is MeTTa code in `stdlib.metta` plus grounded ops on top.
- **The space seam:** `hyperon_space::Space::query(&self, &Atom) ->
  BindingsSet` (`hyperon-space/src/lib.rs:175`) and `SpaceMut`
  (`add`/`remove`/`replace`, `:219`). The interpreter reaches data only
  through `query` (`interpreter.rs:604`, `eval_impl` → `query`).
- **Conjunctions:** `GroundingSpace::query` (`lib/src/space/grounding/mod.rs:147`)
  calls `complex_query` (`hyperon-space/src/lib.rs:340`), which splits
  `(, q1 q2 …)` and joins **tuple at a time**: for every binding of the
  conjuncts so far it substitutes into the next conjunct and probes the trie
  again (`:345-368`). The trie (`index/trie.rs`) is the only index.

## 3. Measured (2026-10-14, v4 host, whole-run seconds; engine comparison, not a primitive timing)

Two Datalog-shaped MeTTa programs, generated, identical for both engines:
`join_n` = n facts `(parent nᵢ n₍₇ᵢ₊₃₎ mod n)` and the two-hop conjunctive
query `(match &self (, (parent $x $y) (parent $y $z)) ($x $z))`, counted;
`anc_n` = a chain of n−1 `parent` facts and the recursive ancestor rule,
`(anc n0 $z)` counted. Both engines return the correct counts wherever they
finish (100 / 1,000 / 10,000 pairs; 49 / 199 / 799 ancestors).

| program | PeTTa (SWI-Prolog 10.0.2), total | Hyperon (fork, release, native), run |
|---|---|---|
| startup only (`!(+ 1 1)`) | 0.32 | 0.000 (init 0.016–0.020) |
| join_100 | 0.29 | 0.11 |
| join_1000 | 0.41 | 5.92 |
| join_10000 | 1.47 | **panic** (`trie.rs:539`, see §3.1) |
| anc_50 | 0.29 | 0.28 |
| anc_200 | 0.33 | 7.37 |
| anc_800 | 0.42 | 416.9 |

PeTTa compiles MeTTa to Prolog clauses and stays near its startup cost; Hyperon
grows about n^1.7 on the join (×54 for ×10) and n^2.4–2.9 on the recursion
(×26 then ×56 per ×4). Single runs; the ratios are far outside the 40 % noise
band, the absolute seconds are not a primitive claim.

### 3.0 A Rust Datalog oracle (datafrog 2, measured 2026-10-14)

`datafrog` (the semi-naive engine behind Polonius), consumed from the fork
`AdaWorldAPI/datafrog` (`master`, `edd7cfc`, version 2.0.1), on the same generated programs, release,
`target-cpu=native`, program-printed tier `avx512f=true`, best of 5, three runs
within 5–15 %. Relation construction (sorting) is charged. Counts asserted
against the known answers, which match Hyperon's and PeTTa's.

| program | datafrog | per output tuple |
|---|---|---|
| join, n = 10² / 10⁴ / 10⁶ | 3 µs / 0.51–0.62 ms / 65–69 ms | 33 / 51–62 / 65–69 ns |
| ancestors from n0 (the MeTTa query, seeded), n = 800 / 10⁴ / 10⁶ | 0.14 ms / 2.2 ms / 0.29 s | 179–185 / 222 / 286–298 ns |
| full closure on the chain, n = 2,000 (2.0 M tuples) | 0.28 s | 139–143 ns |

It finishes join_10000 in 0.5 ms where Hyperon does not finish in 600 s and
PeTTa needs 1.47 s (mostly its 0.32 s startup plus compilation). The seeded
ancestor query costs more per tuple than the join because a chain derives one
tuple per semi-naive round, so per-round overhead dominates. These are
per-tuple engine costs on hash-free sorted merges, not mask-risc primitives;
they are the bar a folded lowering is compared against, not a production path.
First run against the crates.io 2.0.1 release, re-run against the fork: every count
identical, timings within the noise band (join 10⁶ 65.0 ns/pair; seeded ancestors
10⁶ 0.31 s). Probe source: local crate `/home/user/datalog-oracle-probe` (not committed).

### 3.1 A second Hyperon trie bug (distinct from the fix already on the fork)

The measurements above ran on the fork's `fix-trie-key-value` branch (`1ffdbd6`),
which already carries the earlier fix: `TrieKey::value()` decoded ids ≥ 1,024
wrongly (`& (MASK - 1024)` instead of `- 1024`), so a space with more than 1,024
distinct atoms returned wrong atoms or panicked. That fix is in place and is not
what follows. The panic below is a separate limit that shares the constant 1,024.

Any expression with 1,024 or more children passed to a MeTTa-defined function
panics: `(let $r (0 … 1022) (size-atom $r))` returns 1023; the same with 1,024
items aborts with `assertion failed: size < TK_MAX_EXPRESSION_SIZE`
(`hyperon-space/src/index/trie.rs:539`, `TrieKey::start_expr`;
`TK_MAX_EXPRESSION_SIZE = 1 << 10`, `:511`). The interpreter queries the
space for `(= (let $r <expr> …) $X)`, and the trie cannot key an expression
that wide. `(size-atom (0 … 1023))` called directly does not panic, because a
grounded op is executed without a space query. This is what stopped join_10000
(its `collapse` result has 10,000 children). Not fixed here; reproduced on `1ffdbd6`.
⊘ Corrected 2026-10-16 (the fork was read, not the board): TWO fixes exist on `AdaWorldAPI/hyperon-experimental` — `1ffdbd6` (`fix-trie-key-value`, id decoding above 1023) is upstream PR trueagi-io#1107 (`refs/pull/1107/head` == `1ffdbd6`, open against upstream main `3f76dc4`), and `5c9afd4` on `ccr-9f3b8075-74u47l-trie-expr-width` (2026-10-14; `b5ae248` adds the failing ≥ 1,024-children tests first, `5c9afd4` splits the `TrieKey` value bits so expression width is bounded by trie recursion depth, ~1,100 children debug / 2,048 release on a 2 MB stack, not by `TK_MAX_EXPRESSION_SIZE`). Neither is on the fork's `main`; the lance-graph pins (`rev = 3f76dc4`) therefore still carry the panic. What waits on the operator is merging `5c9afd4` to the fork's main and/or upstreaming it, not writing the fix.

## 4. Can MeTTa work as an IR?

**As a declaration IR: yes. As the execution IR: no.**

- MeTTa expressions are typed S-expressions over four atom kinds; rules
  (`=`), facts and queries (`match`, `(, …)`) are data. That is what the
  stack needs from a front end: text parsed once, names resolved through the
  label codebook to addresses (nal-polyfill §4 H), nothing after the binder
  holding a string.
- Minimal MeTTa is an interpreter for nondeterministic, tuple-at-a-time
  rewriting (`superpose-bind`, `collapse-bind`, `cons-atom`,
  `decons-atom`, `call-native`). It constructs new terms and runs grounded
  host code. That is the opposite of a folded mask program, and §3 shows what
  executing it naively costs.
- So the arrangement is the one NAL and SQL already use: **MeTTa is a
  front end; `lance-graph-relalg` describes; R2IL / `FoldDialect` executes;
  mask-risc is the only engine.** Three tiers of the language map to three
  outcomes:

| MeTTa fragment | lowers to | outcome |
|---|---|---|
| facts in a space, `match` over one pattern, `(, …)` conjunctions with shared variables, no new terms | `Rel` selections per conjunct, joins on shared variables (`EqU32Via` / `Gather`), one fold, bindings only at the terminal | **polyfill** (one program per query, not one trie probe per binding) |
| recursive `=` rules that are Datalog (no function symbols in heads, no grounded calls) | relalg `μ` driven by the external driver d′ (`2026-10-03-coresearch-chained-hop.md`: recursion stays outside mask-risc), each iteration one fold | **polyfill, gated** (D-MTP-3) |
| term construction, `superpose`/`collapse` bags, grounded ops, dependent types, state | Hyperon's own interpreter, unchanged | **stays in Hyperon** (typed refusal on our side, never approximated) |

## 5. Where it attaches

```text
MeTTa text ─► hyperon parser ─► Hyperon interpreter (fork, unchanged, 12 ops)
                                        │  Space::query(&Atom) / SpaceMut
                                        ▼
                        lance-graph-metta: LanceSpace (impl Space + SpaceMut)
                          atoms → addresses (label codebook), facts → rows
                          query atom → Rel (σ per conjunct, ⋈ on shared vars)
                                        ▼
                   lance-graph-relalg (normalize, symbolic ∅ / count)
                                        ▼
                   lance-graph-r2il FoldDialect ─► mask-risc ─► ndarray::simd
```

- **The seam is `Space`, not the interpreter.** `complex_query` is a helper a
  space chooses to call, not part of the trait, so a `LanceSpace` that is
  queried directly receives the whole `(, q1 q2 …)` atom and can lower the full
  conjunction into one program.
- **But only as a named space, never inside `&self`.** `ModuleSpace`
  (`lib/src/space/module.rs:33-48`) runs `complex_query` itself and hands its
  main space and every dependency one conjunct at a time; a dependency that is
  not a `ModuleSpace` panics (`:44`). So `LanceSpace` is bound as its own token
  (`!(bind! &kb (new-lance-space …))`, `(match &kb (, …) …)`), and the module's
  `&self` stays an ordinary `GroundingSpace` holding rules, grounded atoms and
  everything outside the Datalog subset. `match` hands its pattern unchanged to
  the named space's `query` (`lib/src/metta/runner/stdlib/core.rs:163`).
  Wrapped in a `ModuleSpace`, it would degrade to one fold per conjunct per
  binding, which is the cost §3 measures. Whether `match &self` can reach the
  conjunction without patching Hyperon is open (D-MTP-2).
- **Recursive rules live in the named space too.** The interpreter evaluates
  `=` rules by querying the context space (`interpreter.rs:604`), i.e. `&self`,
  one call per reduction step. A recursive Datalog rule left there never
  reaches the polyfill. D-MTP-3 therefore stores the rules as algebra values in
  `LanceSpace` and answers a pattern over the derived relation
  (`(match &kb (anc n0 $z) $z)`) with the closure computed by d′, not by
  `=` rewriting.
- **Atoms are addresses, not strings.** A `Symbol` resolves once to a
  `classid × NodeGuid` (or a term id within a class); an `Expression` fact of
  fixed shape `(functor a₁ … aₖ)` is a row of a class keyed by the functor and
  arity, one lane per argument. `Variable` never reaches a row. Shapes the
  class does not declare are refused, never stored as text. ⊘ Council v3 (2026-10-16, `metta-datalog-assimilation-v1-council-v3.md`
  R-5): `SpaceMut::add` returns `()`, so such shapes are never stored as ROWS but stay atoms in the named space's inner
  `GroundingSpace`, counted by kind; the module's `&self` remains a separate space as above.
- **Multiplicity is part of the answer.** `BindingsSet` is a bag; the
  terminal must emit one `Bindings` per match (the `b4_nondeterm` kill
  condition in the parity map). A row mask alone is not an answer.
- **The NAL polyfill is the precedent and stays separate.** PeTTa's
  `lib_nars.metta` / `lib_pln.metta` are oracles; NAL operations go through
  `lance-graph-nal` (`nal::{Op}`, global core methods, no classid plug), not
  through MeTTa rules.

## 6. `lance-graph-cog` or `lance-graph-metta`

Recommendation: **`lance-graph-metta`**, workspace-excluded like
`lance-graph-relalg`, depending on the fork's `hyperon-atom` and
`hyperon-space` only (not the interpreter crate).

- It holds exactly one concern: the MeTTa front end and the `Space` adapter.
- "cog" would pull in OpenCog Classic AtomSpace, PLN, the miner and
  attention, which the parity map assigns elsewhere (#1497 owns
  scheduling/attention; Classic AtomSpace is AGPL and a different engine).
- An earlier note proposed a separate `metta-rs` repo. That fits a full
  transcode of the interpreter; this plan does not transcode it, so a crate
  beside relalg is enough. Revisit if the interpreter itself is ever ported.

## 6a. D-MTP-1 pre-registration (2026-10-14, before any code)

**Correction to §4:** `lance-graph-relalg` cannot lower the join: its own
crate docs say "recursion (`μ`) and the hop (preimage) are not built". The
existing join surface is Quack's `Filter::Semijoin { fk, foreign }`, which
lowers to mask-risc `MaskOp::Gather` over a RESIDENT foreign plane. D-MTP-1
uses that path; relalg gains the hop later or not at all.

**Layout (ingest, charged separately).** Symbols resolve once to addresses.
The `parent` facts become a table over the address space of their first
argument: row `a` holds the fact whose subject has address `a`, the `dst` lane
holds the object's address (a coordinate, not a key), and a `valid` plane marks
the addresses that have a fact. Functional relations only: one fact per
subject. A second fact for the same subject is refused at ingest, not
approximated (one-to-many needs `ScatterOrU32`/a CSR and is out of scope here).

**Query.** `(match &kb (, (parent $x $y) (parent $y $z)) ($x $z))` lowers to
`Filter::and([Plane(valid), Semijoin { fk: dst, foreign: valid }])`. Two
terminals: `Agg::Count` (the MeTTa `size-atom` of the collapsed result) and
`Agg::Rows` (`Keep`) followed by binding emission `(x, dst[dst[x]])` over the
kept bits only — the one place pairs are materialized.

**Fixtures** (generated, identical in all engines):
- F1 `join_n`: `parent(i, (7i+3) mod n)` for every `i` (total, every row
  survives; count = n).
- F2 `join_n_partial`: the same, only for `i mod 3 ≠ 0` (the semijoin rejects
  rows whose target has no fact; count < n).
- n = 10², 10³, 10⁴, 10⁵, 10⁶.

**Oracles.** datafrog (fork) at every n: the emitted pairs as a sorted vector
must equal datafrog's, not only the count. Hyperon and PeTTa on F1 and F2 at
n = 10², 10³ (and PeTTa 10⁴): counts must match.

**Disable runs** (commit first, assert each patch applied once):
1. drop the `Semijoin` leaf → F2 count and pairs must go red (F1 stays green,
   which is why F2 exists);
2. emit `(x, dst[x])` instead of `(x, dst[dst[x]])` → pairs red on F1 and F2.

**Measured, reported, no threshold claimed before the run** (v4 arm,
program-printed tier, best of 5): ingest ns/fact; count program ns/row;
Rows + emission ns/pair; datafrog ns/pair beside it (its relation build
charged, as in §3.0). Expected only as a direction, not a gate: the folded
count costs a small fraction of a nanosecond per row, emission a few ns per
pair, both below datafrog's 65 ns/pair, because ingest has already turned
the join key into a coordinate. If ingest is charged to the query, the
comparison is reported both ways.

**Home.** First probe of `crates/lance-graph-metta` (own workspace root,
excluded, like `lance-graph-relalg`), so the datafrog git dev-dependency never
enters the main workspace's resolution.

## 6b. D-MTP-1 results (2026-10-14)

Probe `crates/lance-graph-metta/examples/join_probe.rs` (5 tests, run as part
of `cargo test`). v4 host, program-printed `avx512f=true`, release, best of 5,
three runs; ranges below span the three.

**Correctness.** The emitted pairs equal datafrog's as sorted vectors at every
n on F1 and F2. Counts equal Hyperon (F1 100 / 1,000; F2 46 / 476) and PeTTa
(F2 46 / 476 / 4,762). Disable runs, after committing: dropping the
`Semijoin` leaf fails F2 and the pinned counts while F1 stays green; emitting
`(x, dst[x])` fails the pairs on F1 and F2. Both as pre-registered.

| fixture, n = 10⁶ | ingest ns/fact | count ns/row | Rows + emit ns/pair | datafrog ns/pair |
|---|---|---|---|---|
| F1 (10⁶ pairs) | 2.5–3.0 | 1.72–1.79 | 3.97–4.06 | 59–62 |
| F2 (476,190 pairs) | 2.3–2.5 | 1.21–1.24 | 5.4–5.7 | 66–71 |

From 10³ to 10⁶ the count stays at 1.1–2.4 ns/row and emission at 3.0–5.7
ns/pair; at n = 100 fixed costs (lowering, scratch) dominate (8–9 ns/row,
10–21 ns/pair).

**Against the pre-registration.**
- Pairs and counts identical: met.
- Emission below datafrog: met, by 11–15× per pair at 10⁶. With ingest
  charged to the query, normalized per emitted pair (F2 ingests 666,666 facts
  for 476,190 pairs, so its ingest counts 1.4×): F1 6.5–7.1 ns/pair against
  59–62 (8.4–9.6×), F2 8.6–9.2 ns/pair against 66–71 (7.2–8.2×).
- "The folded count costs a small fraction of a nanosecond per row": NOT met.
  It costs 1.2–1.8 ns/row. The semijoin is a `Gather` through `dst`, one
  dependent random read per row (`(7i+3) mod n` scatters the targets), and
  that read, not the mask algebra, sets the cost. A layout where the hop is a
  range (rule: hops are coordinate ranges, chained hops are intersections) is
  what would remove it; this fixture has no such locality to exploit.

**Scope.** Functional relations over a dense address space, integer facts
given to every engine. Symbol → address resolution through the label codebook
is not charged to anyone here; one-to-many relations, recursion and the
`Space` adapter are not covered.

## 6c. D-MTP-1b pre-registration: the hop as a range (2026-10-14, before any code)

**Question.** D-MTP-1's count costs 1.2–1.8 ns/row because the semijoin is a
`Gather`: one dependent random read per row. When the link lane is monotone in
row order, can the hop become interval arithmetic instead (the hop as a
coordinate range, chained hops as intersections)?

**Mechanism.** If `dst` is non-decreasing over rows, the rows whose target lies
in `[a, b)` are exactly `[pp(dst < a), pp(dst < b))` (`pp` = `partition_point`,
as in `teleport_probe`). So `Semijoin(dst, valid)` restricted to rows in
`valid` is the intersection of two interval sets: `valid`'s runs, and the
preimages of `valid`'s runs. The count is the length of that intersection;
the pairs `(x, dst[dst[x]])` are emitted by walking the intervals.

**Precondition, checked not assumed.** Monotonicity of `dst` is verified once at
ingest (an O(n) pass, charged to ingest) and recorded as a flag; the range path
refuses a lane without it and the caller falls back to the gather. This is a
probe-local witness, not the contract's `OrderedLaneWitness` (which covers
sealed facet lanes only, `teleport-probe` OPEN item).

**Fixtures** (generated, n = 10³..10⁶):
- F3 BFS tree, fanout 4: `parent(x, (x-1)/4)` for every `x ≥ 1` (root has no
  fact); `dst` monotone; `valid` is one run.
- F4: F3 without the facts of nodes whose depth is a multiple of 3; `valid` is
  one run per kept level band (about depth·2/3 runs), so both the source and the
  target restriction bite.
- F3s / F4s: the same facts under a random permutation of addresses (the
  scattered control; no range path exists there).

**Arms.**
- G-scattered: D-MTP-1's gather program on F3s/F4s.
- G-clustered: the same gather program on F3/F4 (same algebra, local reads).
- R-scan: `valid`'s runs extracted from the plane by a word scan, then the
  interval path.
- R-declared: `valid`'s runs recorded at ingest (no scan), then the interval path.

**Oracle and tests.** datafrog pairs (sorted) equal every arm's pairs at every
n; counts equal. The range path refuses a non-monotone `dst`. Disable runs
(commit first, assert each patch applied once): (1) intersect with the
preimage runs only, skipping `valid`'s own runs → F4 red, F3 green; (2) take
`valid`'s runs as the answer, skipping the preimage → F4 red.

**Expected, as a direction only (no threshold before the run).** R count costs
O(runs · log n), far below a nanosecond per row at 10⁶; G-clustered sits
between G-scattered and R; pair emission on the range path is sequential and
should not exceed G-clustered's. Reported per row and per pair, v4,
program-printed tier, best of 5, three runs; ingest (including the
monotonicity pass and, for R-declared, the run recording) reported separately.

## 6d. D-MTP-1b results (2026-10-14)

Probe `crates/lance-graph-metta/examples/join_range_probe.rs` (3 tests, shared
fixture code in `examples/common/mod.rs`). v4 host, program-printed
`avx512f=true`, release, best of 5, three runs; ranges span the three.

**Correctness.** Every arm's pairs equal datafrog's (sorted) on F3, F4 and the
scattered twins at n = 100, 1,000, 4,097; counts equal at every n up to 10⁶.
The range path refuses the scattered (non-monotone) lane. Disable runs, after
committing: (1) taking the preimage of `valid`'s runs without intersecting
with `valid` itself fails F4 (`banded=true`) while F3 stays green; (2) taking
`valid`'s runs without the preimage fails already on F3. Both as
pre-registered. A fixture test pins that both halves of the intersection
change the answer on F4.

| n = 10⁶ | G-scattered count ns/row | G-clustered count ns/row | R-scan count ns/row | R-declared count ns/row | emit ns/pair G-clustered / R | datafrog ns/pair |
|---|---|---|---|---|---|---|
| F3 (999,995 pairs, 1 run) | 1.96–3.13 | 1.74–1.99 | 0.037–0.055 | 0.00010–0.00011 | 3.4–5.2 / 1.5–1.8 | 16–24 |
| F4 (66,576 pairs, 4 runs) | 1.52–2.43 | 1.27–2.08 | 0.037–0.051 | 0.00026–0.00038 | 21–35 / 1.2–2.0 | 144–196 |

Ingest is charged separately: 3.4–7.1 ns/row clustered, of which the
monotonicity pass is 0.5–0.9 ns/row (scalar `windows(2)`), and 7.8–10.2 ns/row
scattered. G-clustered's emission per pair on F4 is high because the gather
pass runs over all 10⁶ rows for 66,576 pairs.

**Against the pre-registration.**
- Pairs identical; two disable runs red: met.
- "R count far below a nanosecond per row": met. R-declared is
  O(runs · log n), about 0.1–0.4 µs for the whole count at 10⁶, 3,000–24,000×
  below the gather. R-scan pays the word scan of the plane (n/64 words) and
  sits at 0.04–0.05 ns/row.
- "G-clustered between G-scattered and R": met only barely. Clustering the
  same gather program saves 11–36 %; the gather's per-row work, not its cache
  misses, sets most of its cost at this size.
- "R emission not above G-clustered": met (1.2–2.0 ns/pair; the remaining
  cost is the dependent read `dst[dst[x]]`).
- The fixture note "about depth·2/3 runs" was wrong: adjacent kept levels
  merge, so F4 has one run per band of two levels (2 runs at 4,097, 4 at 10⁶).

**Scope.** The range path needs the order witness; it is a probe-local flag
set at ingest, not the contract's `OrderedLaneWitness`, which covers sealed
facet lanes only. A scattered layout (F3s/F4s) has no range path and keeps
the gather. Pair emission still reads `dst` twice per pair; a chained range
(the second hop as a range too) is not built here.

## 7. Deliverables

| D-id | scope | gate |
|---|---|---|
| D-MTP-0 | this map, measurements §3, bug §3.1 | — (done) |
| D-MTP-1 | pre-registered probe (§6a): the two-hop join lowered by hand through Quack `Semijoin` (relalg has no hop) against datafrog, Hyperon and PeTTa, n = 10²..10⁶, v4 | answers identical (count and bag of pairs); cost reported, no threshold claimed before the run — **measured, §6b** |
| D-MTP-2 | `LanceSpace` adapter (`Space + SpaceMut`) for the non-recursive Datalog subset, bound as a named space; Hyperon's golden corpus (`hyperon-probe/golden`, 32 of 33 passing) rerun with facts moved into it | every golden file that passes today still passes; refusals named; the probe records whether the space received whole conjunctions or single conjuncts |
| D-MTP-3 | recursive Datalog rules declared into `LanceSpace` → relalg `μ` through d′, queried by `match` over the derived relation | `anc_n` answers identical; no fixpoint state inside mask-risc |
| D-MTP-4 | the §3.1 expression-width panic as a fix on the fork, test first (separate from `1ffdbd6`) — ⊘ 2026-10-16: the fix is written, `5c9afd4` on `ccr-9f3b8075-74u47l-trie-expr-width` (tests `b5ae248` first); `1ffdbd6` is upstream PR #1107; open item = merge to fork main / upstream | red before, green after; upstream only with the operator's word |
| D-MTP-1b | the hop as a range (§6c): preimage of `valid`'s runs under a monotone `dst`, against the gather on clustered and scattered layouts | pairs == datafrog on F3/F4 and their scattered twins; two disable runs red; costs reported — **measured, §6d** |
| D-MTP-6 | datafrog as the exact Rust oracle in D-MTP-1/3 probes and the `ruff_metta_spo` lowerable tag (test-only dependency) | answers identical to Hyperon and PeTTa as bags of pairs, not only counts |
| D-MTP-5 | PeTTa as the second oracle in the probes (needs SWI-Prolog ≥ 9.3; 10.0.2 installed from the official PPA) | divergences between PeTTa and Hyperon recorded, never averaged |

## 8. Fences

- No second engine or interpreter inside lance-graph; no new opcode; no
  MeTTa text past the binder.
- Hyperon's interpreter is not modified for this; the fork's upstream fixes
  are separate PRs.
- An unsupported shape is a typed refusal answered by the in-memory space,
  never an approximation.
- Licenses: Hyperon and PeTTa MIT; no Classic AtomSpace (AGPL) code.
