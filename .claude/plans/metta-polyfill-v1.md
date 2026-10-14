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
  class does not declare are refused, never stored as text.
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

## 7. Deliverables

| D-id | scope | gate |
|---|---|---|
| D-MTP-0 | this map, measurements §3, bug §3.1 | — (done) |
| D-MTP-1 | pre-registered probe: `join_n` lowered by hand through relalg/R2IL against the Hyperon and PeTTa answers, n = 10²..10⁶, v4, ns per row, build and terminal charged | answers identical (count and bag of pairs); cost reported, no threshold claimed before the run |
| D-MTP-2 | `LanceSpace` adapter (`Space + SpaceMut`) for the non-recursive Datalog subset, bound as a named space; Hyperon's golden corpus (`hyperon-probe/golden`, 32 of 33 passing) rerun with facts moved into it | every golden file that passes today still passes; refusals named; the probe records whether the space received whole conjunctions or single conjuncts |
| D-MTP-3 | recursive Datalog rules declared into `LanceSpace` → relalg `μ` through d′, queried by `match` over the derived relation | `anc_n` answers identical; no fixpoint state inside mask-risc |
| D-MTP-4 | the §3.1 expression-width panic as a fix on the fork, test first (separate from `1ffdbd6`) | red before, green after; upstream only with the operator's word |
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
