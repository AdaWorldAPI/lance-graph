# D-SAG-1 — Software-awareness graph: the methods, programs and evidence of lance-graph as graph rows

**Status:** MEASURED for one concept's methods (#1504's `BillableWorkEntry` set).
Code: `crates/lance-graph-glove-parity/src/awareness.rs`, tests
`tests/awareness.rs` (16), example `examples/awareness_loop.rs`.

**Research question.** Can lance-graph represent its own methods, dependencies,
semantic contracts and execution evidence as graph rows in its existing
substrate, and then use that representation to find and check structural
optimizations?

**Answer so far.** Yes, for one concept and one hop.
- The rows are produced by projecting a `MethodTable`.
- They are queried with an ordinary Quack `Draft`, lowered and run on mask-risc.
- The graph found one optimization candidate (H1). That candidate has a
  measured cost gap and an open list of obligations.
- The first blocker sits exactly where a question needs more than one hop
  (§9).

**SoC** in this repository means *separation of concerns* (about 55 of 88
occurrences; "system on chip" occurs 0 times). "SoC self-optimization" is read
here as: detect where concerns are duplicated or tangled across methods, and
propose a factoring. Nothing in this plan concerns hardware.

## Checklist

- [x] 1. Inventory of reusable components (§1)
- [x] 2. Typed schema (§2)
- [x] 3. One real method projected, with provenance (§3)
- [x] 4. Dependency and validation queries, through Quack (§3)
- [x] 5. Optimization candidate from graph evidence: H1 (§6)
- [x] 6. Correctness oracle or falsifier for each claim (§4)
- [x] 7. MDL experiment with stated assumptions (§5)
- [x] 8. Capability ledger (§10)
- [x] 9. Plan for G1 compositions without a second registry (§8)
- [ ] Multi-hop questions (blocked, §9)
- [ ] Persisting the graph as a Lance dataset; source revisions from git

## 1. Inventory: what exists, read in code

### Multilingual Bible work

**No seven-language graph exists in code.**

What does exist:
- **The KJV in English, as triples.**
  - `deepnsm-v2/examples/bible_wave.rs`.
  - Stored as `Spo` packed into 48 bits, kept in a `TemporalStream` whose
    version number is the verse index.
  - No provenance per triple; no snapshots.
- **A lexeme fixture across 6 lanes.**
  - `lance-graph-planner/examples/probe_babel_stances.rs` plus
    `data/babel/lanes.rs`.
  - Lanes: English, German, Czech, Latin, Greek and Aramaic, 6 rows each.
  - A concept is a WordNet synset (`Coord`). A surface form (`LaneLex`) is
    linked to its concept only by a string `pair` plus a verse address.
- **German silver labels**, from an optional external directory.

The figure "9 lanes / 7 languages" appears only in
`.claude/plans/cycle-loop-closure-driver-v1.md`, and that plan itself retracts
the claim that the probe's lanes were corpora. No Bible query runs through
Quack.

**Reused here: the principle, not the machinery.** An identity is separate from
its spellings:
- A method is a 16-byte `MethodKey`; its label is display text only.
- A field is (concept, schema seal, canonical name), never a label or an
  ordinal.

### Identity contracts

| concept | identity | where | minted by |
|---|---|---|---|
| class | u16 codebook id | `ogar-vocab` `Class::canonical_id`, `class_ids` | OGAR |
| field | canonical name of the class's `ClassView`, in basis order | `basis::canonical_fields()` → `OgarClassView::fields` | OGAR |
| schema | `SchemaSeal` (FNV over the basis) | `method.rs:72` | computed locally |
| method | `MethodKey([u8;16])` | `method.rs:66` | **caller-issued; OGAR mints no method id** (`ActionDef` is keyed by an identity string; `ogar-loco` `Inventory::key_of` is unconnected) |
| reading | `ReadingId(u16)` | `method.rs:94` | literal in tests |
| effect | `EffectClass` | `method.rs:104` (the enum OGAR #357 proposes) | — |

**OGAR #357** exists only locally, as the branch `pr357`. It is documentation
only (`ALGEBRAIC-LOGIC-COMPILER-RFC.md` plus a falsification document), and
the `LogicMethodSpec` it proposes is not coded. Its line 36 calls the function
inventory "not implemented", but `ogar-loco/src/inventory.rs` already exists.

### Provenance pattern (dir-sim)

`ogar-dir-sim/src/provenance.rs` provides:
- `VersionId`, a logical clock;
- `RuleId { name, version }`, where a behaviour change is a new version, never
  an edit;
- `Origin::{Observed, Simulated}`.

The graph's `Tier` follows the same rule: how a fact is known is part of the
fact.

### Query substrate

The edge table is a Quack table. Ingredients:
- **Binder:** `SoftwareGraph` implements `quack::bind::Binder`.
- **Front end:** `Draft`, the ordinary one.
- **Terminal:** `Agg::ScatterOrU32`, the one-hop result.

Additional facts read in code:
- **mask-risc `Program::lowering()`** is derived from the program text: `Range`,
  `Ternlog`, … or `Tiled`. `FusedFold` has only `Count` and `Any`, and
  `fused_ternlog` declines any `Pred`.
- **#1311's traversal adapter** is test code: Quack has no edge type and no hop
  operator, only public building blocks.

## 2. Schema

### Nodes

A node's kind is its identity prefix (`kind_of`). One identity has exactly one
kind; interning asserts it, and `claim` refuses an unknown prefix.

| kind | identity | rule |
|---|---|---|
| class | `class:0x0103` | OGAR codebook id |
| method | `method:<key hex>` | never the label |
| field | `field:<concept>/<seal>/<name>` | the same label in two classes gives two nodes |
| cond | `cond:<field>=<lit>@r<reading>` | the reading is part of the identity |
| fold | `fold:sum(<field>)@r<reading>`, `fold:count(*)@r…` | bag `COUNT(*)` only; no set fold exists in a method body |
| law | `law:values`, `law:nars_revision(...)` | |
| program | `program:<key hex>#<fold index>` | |
| primitive | `prim:<MaskOp or Terminal variant>` | read from the program |
| lowering | `lowering:Tiled` … | `Program::lowering()` |
| refusal | `refusal:<PrepareError variant>` | |
| claim | `claim:<native field>:<grade>` | |
| capability | `cap:<name>` | |
| test | `test:<path>::<name>` | |

### Edges

Each edge is a row `(src, rel, dst, tier, ctx, prov)`.
- `ctx` is the method (or class) whose projection wrote the edge.
- `prov` points to (file, revision, seal, reading).

| rel | direction | written at |
|---|---|---|
| HAS_METHOD | class → method | declared |
| READS_FIELD, FILTERS, FOLDS, FINISHED_BY | method → field / cond / fold / law | declared |
| DEPENDS_ON_CLAIM | field → claim | derived |
| COMPUTES_FROM | field → field | derived |
| LOWERS_TO | method → program | compiled |
| USES_PRIMITIVE | program → primitive | compiled; on any other tier it is a *claim* and is checked (falsifier 8) |
| LOWERS_AS | program → lowering | compiled |
| REFUSED_FOR | method or class → refusal | compiled (the compiler's verdict) |
| BLOCKED_BY | refusal → capability | compiled, and only when the refusal itself is compiled |
| MAY_REQUIRE | method → capability | hypothesis |
| VERIFIED_BY | method → test | verified, under a `Certificate` |

The five tiers are kept apart: **declared, derived, compiled, verified,
hypothesis**. A recorded claim never becomes a verified capability by being
recorded.

## 3. The experiment on #1504's methods

The subjects are `count_billable`, `count_by_duration`, `count_not_billable`,
`billable_hours` (`COUNT(*)`, `SUM(duration)` WHERE `billable = true`) and
`hours_by_billable`. Each is projected with the `MethodTable` that prepared
it. Test names below are from `tests/awareness.rs`.

- **Projection** (`billable_hours_is_projected_with_tiers_and_provenance`):
  - class, fields, condition, folds, law;
  - two programs, one per fold;
  - primitives counted against the real program;
  - claim grades;
  - provenance.
- **A. Dependency** (`dependency_query_runs_through_quack_and_matches_the_programs`).
  The query is:
  ```
  edges WHERE rel = uses_primitive AND tier = compiled AND dst = prim:X
  → ScatterOrU32(ctx)
  ```
  - It is checked against the programs themselves for `MaskedSumI32`, `Count`
    and `GroupSumI32`.
  - A primitive that no program uses is not a node, so it does not bind.
    Nothing is minted.
- **B. Validation.** The verdicts are the compiler's; the queries read them
  back.
  - Under SAP's real claims, `billable_hours` is `refused_for UnmappedRequired`,
    and its `duration` field `DEPENDS_ON_CLAIM claim:hours_logged:hypothesized`.
    The graph explains a refusal down to the claim grade behind it
    (`a_hypothesized_claim_refuses_the_method_and_the_graph_says_why`).
  - Cross-class field labels, compute cycles and foreign readings: falsifiers
    1–3.
- **C. Capability gaps** (`a_declared_need_never_ships_a_capability`).
  - `prim:MaskedSumI32` is shipped: a compiled program contains it.
  - `cap:FirstOfRun`:
    - starts as *claimed* (IAM and SAP needs as hypotheses, plus one
      declared use);
    - becomes *demonstrated* only after the compiler's
      `EvidenceNotIndependent` refusal of NAL revision is projected and linked;
    - a hand-noted refusal does not demonstrate it.
- **D. Repeated structure** (`repeated_structure_is_found_by_identity`).
  - `cond:billable=true@r1` is shared by `count_billable` and `billable_hours`.
    This is G1's `is_billable`, rediscovered from the rows.
  - `!= true` is a different condition and is not counted as a repeat.
  - `billable_hours` scans its selection once per fold. That is a
    shared-selection candidate whose obligation stays open: one mask feeding
    two terminals is a population intermediate the executor forbids, so it
    needs a multi-terminal program.

## 4. Falsifiers

Every row has a test. Every guard was disabled once and its test went red
(14 disable runs; the tree was restored each time).

| # | fault | outcome | test | disable that turned it red |
|---|---|---|---|---|
| 1 | same field label in two classes | two field nodes; `WrongConcept` | `the_same_field_label_in_two_classes_is_two_fields` | field identity without the concept |
| 2 | compute cycle | `refusal:ComputeCycle` from `compute_dag_is_acyclic` | `a_compute_cycle_is_the_contracts_refusal` | verdict not taken from the contract |
| 3 | identical folds under two readings | distinct nodes; `ReadingMismatch`; not paired | `identical_folds_under_two_readings_stay_apart` | identity without the reading |
| 4 | stale certificate | `StaleBody`, `StaleSeal` | `a_certificate_goes_stale_when_the_body_or_seal_changes` | digest check, seal check |
| 5 | bag vs set count | different terminal primitives over identical ops | `a_bag_count_and_a_set_count_are_different_primitives` | hop lands on `src` |
| 6 | fusion changes NULL behaviour | obligation open whenever a validity plane gates the program | `the_fusion_candidate_keeps_its_null_obligation_open_when_a_validity_plane_gates` | obligation discharged unconditionally |
| 7 | better on one workload, worse on another | `Verdict::Mixed`; a tie is not faster | `a_mixed_result_is_not_an_improvement` | "any faster" read as improves |
| 8 | claimed primitive not in the lowered program | `ClaimMismatch` | `a_claimed_primitive_must_be_in_the_lowered_program` | check ignores the owning method |
| 9 | capability "shipped" because a method declares it | `Claimed`, never `Shipped` | `a_declared_need_never_ships_a_capability` | shipped from any tier; `blocked_by` without a compiled refusal; kind guessed from the prefix |
| 10 | benchmark from one tier attributed to another | `TierMismatch` | `a_measurement_answers_only_for_its_own_tier` | tier ignored |

**Two disable runs found two real defects.**
1. `blocked_by`'s compiled-refusal guard first stayed green: no test created a
   refusal node that was not compiled. That gap was in the test.
2. Writing the missing test case exposed a defect in the code: `claim` picked a
   node's kind from a narrow prefix check, so a hand-noted refusal became a
   Capability node, and the compiler's later refusal reused that wrong-kind
   node. Fixed with `kind_of` (`eaa58f6`).

## 5. Entropy and MDL

There are three separate measurements, never one number:
- **Structural redundancy:** `shared_predicates`, `shared_selections`.
- **Epistemic uncertainty:** refusals traced to `hypothesized` claims,
  `Claimed` vs `Demonstrated` capabilities, stale certificates.
- **Execution cost:** `Measurement`s per program, rows and tier.

**Objective (research, not law):** `J = L(P) + L(M|P) + λC`.
- `description_length`:
  - `L(P)` charges each atom `log2(atoms)` bits, and each composite one
    `log2|V|` reference per atom it is defined over.
  - `L(M|P)` charges each body token `log2|V|` bits.
- Correctness, authority, evidence independence and version compatibility are
  **hard constraints**: a candidate that violates them is never scored.

**Results** (`description_length_rewards_only_typed_multi_condition_reuse`):
- Factoring a **single** shared condition never shortens the description. G1's
  one-condition `is_billable` has no syntactic description-length gain. Its
  value is edit-once semantics: change it once and both consumers move. A
  measure over syntax cannot see that. This is a stated limitation of the
  measure, not of G1.
- Factoring a two-condition set shared by three methods does shorten it.
- **Counterexample** (identical labels, different meaning): the same three
  bodies, read by label, repeat and appear to gain bits. Read by typed
  identity (each class's own fields) they do not repeat, and factoring gains
  nothing. A label-level detector would propose a cross-class factoring that
  the typed graph refuses.
- **The first metric was wrong, and its own counterexample caught it.**
  Defining a composite removed its atoms from the vocabulary, so even a
  single-use factoring looked profitable. A composite is spelled in atoms, so
  its atoms stay in P.

The λC term is **unscored** for H1: the cost after fusion has not been
measured (§6), and a scored J would invent it.

## 6. H1: fuse the predicate into the masked sum

**What the graph found.** `billable_hours`'s sum program:
- uses `MaskedSumI32`;
- has `LOWERS_AS lowering:Tiled`;
- reads only the live plane.

Generated by `fusion_candidates`.

**Measured with `examples/awareness_loop.rs`** (1,048,576 rows, 51 reps,
interleaved; the answers are asserted equal before anything is timed). The
machine is shared and noisy.

| build | detected AVX-512 | program median | scalar reference median | ratio |
|---|---|---|---|---|
| native (`avx512f` compiled) | yes | 603.5 µs | 379.1 µs | 1.59 |
| native, second run | yes | 619.6 µs | 362.8 µs | 1.71 |
| `config-v3` (`avx512f` not compiled) | yes | 617.4 µs | 347.6 µs | 1.78 |
| #1504 ledger, x86-64-v3 | — | 600 µs | 366 µs | 1.64 |

- **Observed:** the tiled program costs 1.6–1.8× a single-pass scalar loop on
  both tiers.
- **Hypothesized, not shown:** a fused mask-risc terminal would close that gap.
  The scalar loop is a reference for what one pass costs, not a fused SIMD
  terminal.

**What is actually missing.** Two pieces, both in mask-risc's recognisers:
1. Fusion over a lane `Pred`: `fused_ternlog` declines any `Pred`.
2. A sum fold: `FusedFold` is only `Count` and `Any`.

**Obligations:**
- NULL/UNKNOWN: every validity plane must still gate the sum. Discharged here,
  because the program reads only the live plane.
- The `MASKED_SUM_I32_MAX_ROWS` bound.
- The fused terminal itself.

**Falsifiers that already exist:** the row-loop oracles in `tests/method.rs`
and `tests/awareness.rs`, and the Quack == R2IL parity.

**Who would benefit:** every program the dependency query returns for
`MaskedSumI32` with `lowering:Tiled`.

## 7. Missing algebra from repeated needs

`FirstOfRun` (a first-occurrence mask over an ordered key) is the shared
candidate.
- **NAL revision:** the need is demonstrated by a compiler refusal.
- **IAM reconcile and SAP import:** recorded as `MAY_REQUIRE` hypotheses.
  Neither was checked against IAM or SAP code here.

A shared physical primitive does not equate the domain laws. NARS evidence
independence, IAM identity uniqueness and SAP transaction identity remain
separate laws, each needing its own proof. The graph may recommend the
primitive; it records no semantic equivalence.

## 8. The loop, and integrating G1

**States:** candidate → structurally valid → semantically verified →
performance measured → rejected / approved for implementation.

**Executable today:**
- projection;
- queries A to D;
- compiler verdicts;
- certificates;
- tier-labelled measurements;
- verdicts across workloads.

**Not executable today:** acting on a candidate. Nothing rewrites code, a
declaration, a deployment or an external system.

**G1 (#1508)** needs no second registry. `Cond::Method` references become
`CALLS` edges (method → method, declared), and the compiler's
`MethodCycle`, `MethodRefContract`, `MethodRefNotPredicate` and
`UnknownMethodRef` refusals become `REFUSED_FOR` edges. The projection keeps
reading the `MethodTable`; the graph never resolves a method.

A separate hypothesis, not attempted here: a graph representation *as* a
method body. That would need its own parity test against
`MethodTable::prepare`.

## 9. Blockers and gaps

1. **Multi-hop.** "Methods using X" is one hop only because every edge
   carries `ctx`, the method it was written within.
   - "Methods two calls away from X" would need a mask from one program fed to
     the next. Quack documents that as a forbidden population intermediate
     (`Filter::Semijoin`).
   - The legal routes are the factored `EqU32Via` read or a relation carrier
     (D-CMH-G1-4).
   - **This is the first concrete blocker.** It is not worked around with a
     separate traversal engine.
2. **Method identity is caller-issued.** OGAR mints no method key.
3. **The body digest** is FNV over a spec's `Debug` rendering.
   - It is stable within one build.
   - A toolchain change makes every certificate stale (never falsely fresh).
   - A canonical byte encoding of `MethodSpec` would remove that.
4. **The graph is in memory.** It is not yet a Lance dataset, and the source
   revision is a caller string (`SAG_REV`), not read from git.
5. **Certificates are issued by tests,** not by a running service.

## 10. Capability ledger

| item | status | evidence |
|---|---|---|
| projection of prepared methods, tiers, provenance | implemented | 16 tests |
| dependency query through Quack | implemented, one hop | query A test; example output |
| compiler verdicts as edges (refusals, compute cycle) | implemented | falsifiers 1–3, query B |
| certificates and staleness | implemented | falsifier 4 |
| capability status | implemented | falsifier 9 |
| shared predicates / shared selections | implemented (detection only) | query D test |
| MDL measure | implemented, crude, with its limitation | MDL test |
| H1 fusion candidate | candidate; cost gap measured; fused terminal absent | `awareness_loop` |
| `FirstOfRun` | demonstrated gap (NAL); hypothesis (IAM, SAP) | query C test |
| multi-hop dependency | blocked | §9.1 |
| applying a transformation | out of scope | — |
