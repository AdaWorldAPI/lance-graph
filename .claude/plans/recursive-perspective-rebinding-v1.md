# D-RPR: recursive perspective rebinding over composable relational registers

**Status: LAB EXPLORATION, 2026-10-20. Architecture and experimental design only. Nothing here is admitted to production, no production code may be rewired to it, and no implementation is authorized by this document.** Every claim carries its grade: IMPLEMENTED (code on `main`), MEASURED (a recorded probe run, with its source), PROBE (probe code exists, run or not), PROPOSED (plan text only), OPEN (undecided or unknown).

**Question.** Can globally consequential reasoning be decomposed into a sequence of bounded changes of perspective, each reusing the relationships its predecessors already resolved and resolving only what the next question adds? This is a question about addressing, memory organization and reuse. It proposes no graph engine, interpreter, world simulator or neural architecture.

**Sources read for this distillation (origin/main, 2026-10-20):**
- `d-holo-h14c-thought-first-v1.md`, `d-holo-h14d-logic-mask-read-v1.md` (MEASURED);
- `d-rmap-implicit-relationship-register-v1.md` (D-RMAP-1), `d-rmap-separate-tables-thought-palettes-v1.md` (D-RMAP-1A), `d-rmap-2-context-tile-algebra-probe-v1.md`;
- `crates/mork-pathmap-probe/src/{rmap,join,boundary,lib}.rs` and its tests;
- `mork-pathmap-schnellverschluss-v1.md`;
- `metta-datalog-assimilation-v1.md`, its council `-v3.md`, `metta-polyfill-v1.md`, `crates/lance-graph-metta`;
- `thought-git-api-v1.md` (D-TGA), `causal-pickaxe-resident-git-lance13-temporal-v1.md` (D-CPG), `ce64-hypothesis-layers-v1.md` (givens layer);
- D-TELEPORT-0/1/2 board entries; `relstore-quack-petgraph-traversal-v1.md` (D-RST).

## 1. Experience and navigation are different things

`Idea(WHERE, WHAT, HOW, WHY)` names kinds of question. It is not a schema, and nothing here stores it as four columns or four registers. A question's kind determines which relationships must become addressable next; it does not determine how.

`WORLD × ANGLE × THOUGHT` names the navigation, and each of the three already has a home:

| Navigation term | Existing component | Grade |
|---|---|---|
| WORLD | canonical tables at a sealed version; within a thought table, thought 0 is the givens of that version ("Thoughts `1..=63` branch off main (thought 0, the givens of the current sealed version)", D-TGA:6–9) | givens layer IMPLEMENTED as a probe (`cognitive-shader-driver/examples/givens_layer_probe.rs`, refusal `WriteError::GivensLayer`); D-TGA PROPOSED |
| ANGLE | a selection of relationship maps plus a focus binding: `MapPalette` (0..255 local map selectors → `MapId` + generation), `FocusBinding` (global ids → focus slots), the prepared relation planes | `Focus`, `Csr`, tile IMPLEMENTED in the probe `rmap.rs`; `MapPalette`/`ThoughtMapPaletteBinding` PROPOSED (D-RMAP-1A §1) |
| THOUGHT | one of 64 slots (0 givens, 1..63 hypotheses) holding sparse typed deltas over the shared frame | `ThoughtDelta` PROBE in `rmap.rs`; D-TGA thought table PROPOSED |

So the brief does not need new nouns. What it adds is a **relation between successive angles** (§4), which nothing on `main` names yet.

> **Correction (2026-10-20, §11.1):** WORLD, ANGLE and THOUGHT are metaphors for thought. They name no type, register, field or modification. Thought identity is #1605's `ThoughtCommitId` (speculative) and `DurableRef` (durable).

## 2. The addressing operation is the H14c/H14d canon

The brief's `TableA[i] : TableB[i]` is the measured canonical expression `Thought(t)[T_a[i] : T_b[i]]` (H14c, MEASURED 2026-10-15; H14d, MEASURED 2026-10-16). `i` is a shared coordinate, presence says whether a value is there and never where, and "the join is the layout". The council v3 header of the MeTTa assimilation plan names these two plans as the correct path and marks the preimage-join design (D-ASM-2, D-ASM-11/11A/11B) as a gather violation.

H14d runs four stages, each shrinking what the next touches: logic reduced symbolically → address mask over presence planes → masked read at the source (Lance 13 `with_row_addr_prefilter`, the D-TELEPORT-2 probe) → terminal. Only addresses are materialized before a terminal.

**Where translation happens.** Turning a relationship between two independently keyed tables into a shared `i` requires resolving one side's identities in the other's coordinate space. That is the global → local binding. D-RMAP-1 §6: "A pre-fold global->local binding is only a candidate new admissible *membrane* and must be independently reviewed. Until accepted, such operations are a typed refusal." The `rmap.rs` probe performs it (`Focus::bind`, `slot_of` by binary search) and states that it measures it without settling its legality. **Admission A1 is OPEN and is the single precondition for everything below.**

**Prefetch** is two existing, measured mechanisms, not a new one: the address mask restricting the Lance read (D-TELEPORT-2, H14d stage 3), and the prepared relation planes of a focus (D-RMAP-1A §8). Temporary working copies are admitted by this brief; authoritative ownership stays with the canonical tables.

## 3. Composable registers: what exists, at what size

| Register | Concrete form in the probe | Size measured (D-RMAP-1A §8.3) |
|---|---|---|
| WORLD, 0..255 local slots | `struct Focus { ids: Vec<u32>, by_id: Vec<(u32, u8)>, pub generation: u64 }`, `SLOTS = 256` (`rmap.rs:58–65, 201–208`) | binding 3 072 B |
| ANGLE, per predicate | `CsrPred { p, offsets: [u32; SLOTS+1], dst: Vec<u8>, row: Vec<u32>, escaped: Vec<(u8, u32, u32)>, has_escape: SlotSet }`; the 256 × 256 tile built from it | CSR 18–66 KB; tile 8 KiB per predicate |
| THOUGHT, 0..63 | `ThoughtDelta` sparse edits over the shared tile | 32 B per thought as edits, 16 KiB lowered to executor planes; per-eval 2.61 / 2.66 / 2.79 µs at 1 / 8 / 64 thoughts (sparse d2), "the shared tile is stored once" (§8.5) |

Nothing is a three-dimensional tensor. The world's values are never copied into an angle, and an angle is never copied into a thought.

> **Reading note (2026-10-20, §11.1):** the first column above uses the metaphors as labels. The registers are `Focus`, `CsrPred` and its tile, and `ThoughtDelta`, and nothing should be renamed after the metaphors.

**Arity.** The probe's facts are `[u32; 3]` (subject, predicate, object), so its angles are predicate-indexed binary relations; `join.rs` co-addresses one shared variable. General n-ary binding over several shared variables, with repeated variables inside one pattern, has **no design on `main`** (OPEN). The triangle needs two variables co-addressed at once; only the 256 × 256 tile gives that, inside one focus (Schnellverschluss §7b).

## 4. Recursive rebinding, built from existing pieces

The brief's `F_{k+1} = Rebind(W, A_{k+1}, F_k)` already has a measured kernel. `rmap.rs` `rebind` replaces k slots of a focus with one forward seek each, re-resolves the old escapes against the new focus, remaps or escapes edges from unchanged members, and rebuilds the tile. The test `incremental_rebind_equals_full_preparation` pins it to a full preparation. D-RMAP-1A §8.4 measured it (µs, incremental / full):

| k replaced | sparse d2 | sparse d8 | dense |
|---|---|---|---|
| 1 | 41.8 / 357.2 | 223.9 / 1657.7 | 308.2 / 892.9 |
| 16 | 60.4 / 433.1 | 298.1 / 1985.2 | 752.6 / 744.9 |
| 128 | 156.5 / 310.7 | 850.4 / 937.7 | 970.3 / 833.2 |
| 256 | 344.3 / 305.3 | 1238.4 / 971.3 | 1011.0 / 644.9 |

Rebind wins for small k and loses at k = 256, and on the dense relation from k ≥ 16. **Whether recursive questioning is cheap is therefore a question about the distribution of k along a real chain of questions.** No probe has measured that distribution.

**The three operations stay distinct, and each already has its mechanism:**

| Change | Mechanism | What it costs or refuses |
|---|---|---|
| THOUGHT (another hypothesis, same angle) | a new sparse delta; nothing rebinds | measured flat across 1 / 8 / 64 thoughts (§8.5) |
| ANGLE (other relationships, other slots) | `rebind` for slots, a new `CsrPred` for each new predicate | measured above; a new predicate costs its own prepare |
| WORLD (an admitted canonical change) | a new sealed version | every frame built from the old one refuses: `Witness { facts, seal, focus, scope }`, `Refusal::{FactsChanged, IndexResealed, FocusChanged, ScopeChanged, ThoughtMismatch, OrderingMismatch, IncompleteFocus}` (IMPLEMENTED in the probe) |

**What produces the next angle.** The brief leaves "identify missing dependency" open. The probe already emits the signal. A scoped reach records the global ids its edges left the focus for (`ScopedReach { reached, escaped: BTreeSet<u32> }`, exact and never truncated). `global()` refuses with `IncompleteFocus` while any escape exists. **Hypothesis H1:** the escapes of angle `k` are the seeds of angle `k+1`, so "the explanation lies outside the focus" is detected, not assumed, and `rebind` brings exactly those identities in. H1 is PROPOSED and measured nowhere.

**Ownership along the chain.** A frame has one owner at a time. The mailbox moves to the frame (rotating ownership), so a child angle borrows within the mailbox that holds the frame. Hand-over is a move, which the type system checks (E-CE64-MB-4). The number of hand-overs per chain is a cost to count, not a blocker.

**Reconsidering the parent.** A deeper angle may contradict an assumption of an earlier one:
- **Contradiction.** A contradiction of a given is recorded and never merged: "A hypothesis write that offers a different value for a given's lane is refused and recorded as a contradiction" (ce64-hypothesis-layers givens rule, IMPLEMENTED in the probe).
- **Retracting what depended on it.** That is delete-and-rederive (D-TGA Rule 2, PROPOSED). It needs, per derived cell, the cells it was derived from, and that provenance is undecided (D-TGA-3, OPEN).
- **Making a revision canonical.** That is `land` through the revision gate G1–G7 (D-TGA-4, PROPOSED).

Until D-TGA-3 is decided, a parent can be **refused** on a changed premise, but its dependants cannot be **retracted** selectively.

## 5. PathMap, MORK, Hyperon/MeTTa: what each contributes

**PathMap (navigation to the next angle).** Measured in D-RMAP-1A §8.6:
- Prefix seek beats a sorted run on sparse relations (7.0 vs 8.5 µs; 100.9 vs 114.7 µs) and loses on dense (1 184 vs 566 µs).
- The trie costs 4–6× the run to build, and its keys take at least 3× the bytes.
- On postings, a zipper merge takes 2.32 µs, a sorted merge 0.41 µs, and a mask-risc `And` + `Count` 0.082 µs.

So a trie is a candidate *navigation* index for choosing seeds, not a store and not the executor. A Lance-owned zipper over sorted relation orders is D-MPS-6, BLOCKED.

**MORK (matching).** MORK and PathMap store sets (`PathMap<()>`), so bags are refused (`Refusal::SetOnly`). Its leapfrog route was read in source, never observed running. D-ASM-10 keeps MORK as an oracle outside the workspace. The native shared-variable join (MEASURED, Schnellverschluss §7b, `cargo run --release --example join_cost`, medians of 5) took the following at n = 50 000:

| Step | Native | PathMap | MORK |
|---|---|---|---|
| Prepare / index build | 2.75 ms | 14.65 ms | 22.87 ms |
| Bag count | 35.5 µs | — | — |
| Answers streamed | 0.33 ms | — | — |
| Answers into a map | 7.39 ms | — | — |

Only the intersection runs on mask-risc; enumeration and counting are plain loops. One join shape was measured, on synthetic data, warm, with no reuse across different queries.

**Hyperon/MeTTa.** The existing classification (`metta-polyfill-v1.md:128–132`, Decision H) sorted into the brief's four classes:

| Class | Operations | Source |
|---|---|---|
| 1. Exactly representable | facts in a space; `match` over one pattern; conjunctions with shared variables; no new terms | polyfill (PROPOSED) |
| 2. Representable after binding or materialization | recursive `=` rules that are Datalog (no function symbols in heads, no grounded calls), through the external driver d′, one fold per iteration, bounded, "refusing an unbounded closure rather than truncating" | Decision D, gated (PROPOSED) |
| 3. Requires functionality not yet built | repeated variables inside one pattern (no plan text addresses it); bags whose multiplicity lives on a foreign table (the keyed count undercounts, council N1c); nested expressions (refused in W1, N2); n-ary co-addressing (§3) | OPEN |
| 4. Kept external | term construction, `superpose`/`collapse`, grounded operations, `new-state`/`change-state!`, dependent types, `=` rules with a function symbol in the head: answered by Hyperon's interpreter, refused or routed on our side, never approximated | Decision H, council F6 |

`crates/lance-graph-metta` is a stub: "Nothing is shipped here yet". No `Space` implementation exists (D-ASM-4/D-MTP-2 queued, not authorized). Oracles: Hyperon, PeTTa, datafrog, and MORK as a fourth, compared per category (bag / set / count), never averaged (Decision G, R-8).

**Assimilation rule** (unchanged from those plans): take the algorithm and its semantic contract into RelAlg / Quack / R2IL / mask-risc only where exactness is shown against the oracle. At the boundary, reconstruct MeTTa-compatible bindings only on request. No foreign runtime objects, S-expressions or JSON on a hot path, and no Hyperon interpreter as a hidden second evaluator.

## 6. Reuse is permitted only under a witness

A prepared frame is reused only while its `Witness` equals the current one. Otherwise it refuses and names the stale component (IMPLEMENTED in the probe). D-RMAP-1A §7.4 specifies nine stale-state cases, none of them run. Among them:
- X1: datasets that collide at the same version;
- X2: compaction without stable row ids;
- X3: palette ABA;
- X4: a reused thought slot;
- X5: a dirty thought;
- X6: a rebase;
- X7: a revoked scope;
- X8: lineage after a squash;
- X9: a deletion on main.

Durable versions are `DurableRef { dataset_id, branch, version }` and speculative ones `ThoughtCommitId`. Neither type exists in `crates/` (PROPOSED, D-CPG). Their never-reused rule is OPEN (D-TGA gaps).

## 7. Three kinds of plasticity, three admission levels

| Kind | What changes | Admission |
|---|---|---|
| Epistemic | conclusions within one angle: thought deltas, never the givens | none beyond the thought's own write rules |
| Perspective | which relationships and slots are addressable: rebind, palette selection | no canonical write; frames refuse on stale witnesses |
| Ontological | canonical facts, classifications or relationships | `land` through G1–G7 with retraction (D-TGA-3/-4), authorized, and the strongest evidence; PROPOSED, nothing built |

A bounded frame does not establish global consistency. The only guard on `main` is `IncompleteFocus`: a frame with escapes cannot answer globally. Any experiment below must show that guard firing, not just that answers match.

## 8. Minimal experiments (pre-registration sketch; none authorized, none run)

**D-RPR-1: recursive question chain over one corpus.** Probe in `mork-pathmap-probe`, lab only. It reuses `Focus`, `discover`, `Csr`, `rebind`, `ThoughtDelta`, `Witness` and `join.rs`, and adds no new types.

**Corpus.** The KJV SPO triples exported by `deepnsm-v2/examples/bible_wave.rs --export spo.tsv`. They need Gutenberg #10 and the `v0.1.0-cam96-data` release assets, which are fetched, not committed. A committed synthetic world is a fallback and must be labelled as such. The figures "31 327 triples" and "606 subjects" were not found in the example's source and must be re-measured before they are cited.

**Chain.** Angle `A_0` is seeded by one question's entities and predicates. Each later angle is seeded by the previous one's escapes (H1), plus the predicates the next question kind adds.

**Arms:**
- (a) MORK / PathMap repeated matching: set semantics only, as an external oracle;
- (b) native direct lookup with no retained frame (the probe's "direct/run" arm);
- (c) native, with a full prepare per angle;
- (d) native with `rebind` and retained predicate CSRs.

**Measured per angle, kept apart:**
1. the initial binding cost;
2. the marginal cost of the next angle, given what is retained;
3. relationships reused;
4. new identities and predicates resolved;
5. indexed reads and bytes prefetched;
6. fold time;
7. 1 / 8 / 64 thoughts;
8. refusal on a changed premise;
9. resident bytes and the reuse crossover;
10. exact parity: bags against a nested-loop oracle, sets against MORK.

Also measured: the distribution of k per rebind and the number of ownership hand-overs.

**Reuse must fail, each case with a named refusal:**

| Case | Refusal |
|---|---|
| no shared slots with the parent | full prepare, reported as k = 256 |
| a changed predicate order | `OrderingMismatch` |
| a new canonical version | `FactsChanged` / `IndexResealed` |
| a scope change | `ScopeChanged` |
| a rejected parent hypothesis | `ThoughtMismatch` |
| an answer outside the focus | `IncompleteFocus` |
| a recursive dependency cycle | **no refusal exists yet: OPEN** |

**Acceptance.** Exact parity on every arm. Every case above refuses rather than answers. Arm (d)'s chain total is reported against (b) and (c) with its crossover.

**Falsified if** arm (d)'s marginal cost per angle is not below arm (c)'s for the measured k distribution, or if any narrow frame returns an answer without `IncompleteFocus` when its oracle needs out-of-focus facts.

**D-RPR-2: premise revision and selective retraction.** BLOCKED on D-TGA-3, which leaves the per-derived-cell provenance undecided. Until then only refusal is testable, not retraction.

**D-RPR-3: MeTTa-facing parity of a chain's answers.** BLOCKED on D-ASM-4 (the `LanceSpace` harness), which is not authorized.

## 9. Reuse before creating anything

| Need | Existing | Grade |
|---|---|---|
| shared-coordinate operation | `Thought(t)[T_a[i] : T_b[i]]` | MEASURED (H14c/H14d) |
| focus, binding, escapes, refusals, witness | `rmap.rs` | PROBE |
| incremental rebind | `rmap.rs` `rebind`, §8.4 table | MEASURED |
| shared-variable join with bags | `join.rs`, Schnellverschluss §7b | MEASURED |
| source-restricted reads | Lance 13 row-address prefilter, D-TELEPORT-2 | MEASURED (probe) |
| thoughts over one frame | `ThoughtDelta`, §8.5 | MEASURED |
| givens immutability and contradiction records | `givens_layer_probe.rs` | PROBE |
| versions, branches, land, retraction | D-TGA, D-CPG | PROPOSED |
| map selection per thought | `MapPalette`, `ThoughtMapPaletteBinding` (D-RMAP-1A) | PROPOSED |
| MeTTa boundary and oracles | `metta-polyfill-v1.md`, D-ASM-4/-10 | PROPOSED; crate stub |
| relationship store ownership | D-RMAP-1A §1, D-RST (lab) | PROPOSED |

**Nothing new is needed for D-RPR-1** except the chain driver and the measurements.

## 10. Open, in order of consequence

1. **A1:** is the pre-fold global → local binding an admitted membrane? Everything here presupposes it.
2. **H1:** do escapes predict the next angle's seeds in a real question chain?
3. **The k distribution** along real chains; it decides whether rebind pays (§4).
4. **n-ary co-addressing and repeated variables** (§3, §5 class 3).
5. **Selective retraction provenance** (D-TGA-3).
6. **A refusal for recursive dependency cycles.**
7. **`ThoughtCommitId`'s never-reused rule.**

| D-id | Content | Status |
|---|---|---|
| D-RPR-0 | this distillation | PROPOSED, lab only |
| D-RPR-1 | recursive chain probe | not authorized; waits on A1 and an operator go |
| D-RPR-2 | premise revision with retraction | BLOCKED on D-TGA-3 |
| D-RPR-3 | MeTTa-facing parity | BLOCKED on D-ASM-4 |

## 11. Folded in from #1624 (2026-10-20)

#1624 distilled the same brief independently, before this plan merged. This plan is the canonical one. This section adds only what #1624 had and this plan lacked. Its falsifiers are sub-cases of D-RPR-1, not new D-ids. #1624's own plan file and board rows are withdrawn. A draft of the chain it describes is banked, not merged, in `.claude/harvest/d-rpr-1-chain-probe-draft/` (§11.9).

### 11.1 Corrections

- **WORLD, ANGLE, THOUGHT are metaphors.** Operator, 2026-10-20: "angle is a metaphor for thought and does not allow any modifications". No type, register, field, opcode or refactor follows from these words. The tables in §1 and §3 map them to existing components and nothing more.
- **Thought identity is #1605's.** A speculative thought is a `ThoughtCommitId` with parent ids, held in CE64 W slots 0..63 (0 = givens, 1..=63 = branches); "an in-RAM thought commit is not a Lance commit". Durable state is `DurableRef{dataset, branch, version}`. In the probe, `ThoughtDelta.thought: u64` and `Witness.facts` stand for these (their doc comments in `rmap.rs` say so). A chain step that reads the previous step's answer reads resident speculative output, which #1605 §5 point 4 admits only through a declared dependency.
- **D-RST.** #1624 said D-RST resolved to no plan. That was written before this merge added `relstore-quack-petgraph-traversal-v1.md`, and it is withdrawn.
- **Two meanings of `ThoughtMask`.** `lance-graph-contract::recipe_kernels::ThoughtMask` is the set of thought fields known before a recipe wave (`rung_schedule.rs`). #1605's `ThoughtMask::selected(&[24, 48])` selects thought slots. Recorded, not resolved: OPEN.
- **The Hyperon class-2 path.** The measured D-MTP-1 two-hop lowering runs `MaskOp::Gather` behind quack `Filter::Semijoin` (D-ASM council v3), so under the 2026-10-16 ruling it is gather-shaped and cannot be the class-2 route.

### 11.2 D-RPR-1 sub-cases

Each sub-case has a kill condition and a disable run that must turn its test red.

| sub-case | probe | kill condition | disable run |
|---|---|---|---|
| 1a | the chain fixture and its oracles, before any timing | any arm's `reached` or `escaped` differs from the scan oracle at any depth; or a step never escapes | stop seeding the next focus from the escapes; the anti-vacuity check (§11.3) goes red |
| 1b | marginal cost: arm (d) against (c) and (b), at the natural k of each step | (d) is not below (c) at the measured k on the dense fixture | make `rebind` treat every slot as changed; the "rows reused > 0" assertion goes red |
| 1c | invalidation: facts, seal, scope or ordering changed between steps | a retained CSR or tile answers after the change, or the rebuild is not reported | remove `Witness::check` from `reach`; a stale frame answers |
| 1d | the narrow-frame guard under the chain | a depth whose answer depends on escapes is reported as global | remove the `escaped.is_empty()` test in `global()` |
| 1e | a two-variable step: a closing atom that needs `a` and `c` together | a silent answer from a one-variable domain; the expected outcome today is a refusal | none; the row records the boundary, not a mechanism |
| 1f | thought rejection: a parent hypothesis is discarded between steps | a child frame built under the discarded thought's `ThoughtDelta` still folds | reuse the old thought id; `ThoughtMismatch` must fire, and disabling it is red |

### 11.3 Anti-vacuity

- Every step must escape at least once, or it tests nothing. Assert escapes > 0 at every depth.
- The first focus must not already contain the first answer. The banked draft measured this failure: `discover` over all predicates to 256 slots contained the 3-hop reach over one predicate, so step 1 never escaped. Seed the first focus narrower than the first question's reach.
- Reuse must be real. Some canonical rows kept from the first preparation must remain, and some must be newly resolved.
- A thought must change at least one answer, and the shared tile must be unchanged afterwards.
- Every guard gets a firing case and a silent twin. Example: an exhausted focus refuses when no slot is evictable, and answers when one is.

### 11.4 Independent oracles

- Global reachability by a full scan over all facts, with no index and no focus, per step, compared on `reached` and on `escaped`.
- For each thought, an in-focus scan over the edited facts.
- For any join-shaped step, the nested-loop bag oracle in `reference.rs`. MORK and PathMap are oracles for sets only.
- Incremental against full preparation: equal edge set, escape set and tile plane for every predicate.

### 11.5 Protocol

- Oracles and disable runs go green → red → green before any timing.
- Commit before disabling, assert that each disable patch applied exactly once, and restore from the committed file.
- The program prints the tier. Release, debug 0, medians of 5.
- Differences inside the ~40 % run-to-run noise of this VM (H14c) are not read.
- Fixtures are frozen after the first measurement. FALSIFIED and BOUNDED are valid outcomes. Timings go to a dated entry as CSV.

### 11.6 Refusals the chain must show, by axis

| axis | refusal | exercised by the draft |
|---|---|---|
| version | `FactsChanged`, `IndexResealed`; `JoinRefusal::Stale` in `join.rs` | `FactsChanged` only |
| focus | `FocusChanged` | no |
| scope | `ScopeChanged` | no |
| ordering | `OrderingMismatch` | no |
| thought identity | `ThoughtMismatch`; the `ThoughtCommitId` never-reuse rule is OPEN (§10 item 7) | `ThoughtMismatch` |
| narrow frame | `IncompleteFocus` | yes, as the re-ask condition |
| capacity | the needed identities do not fit beside the kept ones: refuse, never answer from an incomplete focus | yes; a probe-local refusal, not in `rmap.rs` |
| no shared slots | full prepare, reported as k = 256 | no |
| dependency cycle | none exists | OPEN |

### 11.7 Preparation and reuse costs already measured

Copied from D-RMAP-1A §8 and Schnellverschluss §7b, not re-measured:

| quantity | sparse d2 | sparse d8 | dense |
|---|---|---|---|
| CSR break-even reuses (arithmetic from medians) | 76 | 11 | 1.8 |
| 3-hop per-eval, CSR | 3.33 µs | 5.58 µs | 12.38 µs |
| 3-hop per-eval, tile | 5.08 µs | 5.74 µs | 4.53 µs |

- **The tile wins only on the dense relation.** Neither layout is a default.
- **Report escapes as source slots when the next step allows it.** At degree 8, escapes reported as identities cost 50.4 µs per eval, and as source slots 5.6 µs. The chain's next step consumes identities, so it pays the higher figure.
- **Bytes:** binding 3 072 B; CSR 18–66 KB; tile 8 KiB per predicate. At n = 50 000, the join table is 403 KB, and a thought's filter plane is 1 568 B and costs 5.2 µs per thought for a bag count.
- **Zero-copy against register copy is an open tension (§8.7).** A count over canonical lanes takes 20–24 µs with 0 B copied. Loading a 3 072 B register first takes 0.78 µs, and counting in it 0.38–1.27 µs.

### 11.8 Two-variable and multiplicity limits

- `join.rs` co-addresses one shared variable, plus a star of relations on that variable (`prepare_star`). A cyclic body needs two variables at once, "which a one-variable domain cannot give" (§7b). OPEN.
- "Stepping from `y` results to `z` slots over a large domain would need a scatter, which stays forbidden" (§7b). A deeper step takes that shape when its question lands on a relation the parent did not co-address. OPEN.
- The focus register answers Boolean reachability only: "Bag counts and path counts are not answered here" (`rmap.rs:40–44`). A chain over `rmap` therefore preserves no multiplicity. Carrying bags through a step needs `join.rs` at that step, and that is untested.
- 256 slots is the probe constant `SLOTS`, not a decided register width.

### 11.9 Banked draft probe

`.claude/harvest/d-rpr-1-chain-probe-draft/` holds `chain-probe.patch` and a README. The patch applies to `main` at `4dbd085` and at `8b570ed`, after #1623.

- **Result on `main`:** 5 of 6 tests pass. The failing test is the §11.3 first-focus case. Its per-step parity with the global scan oracle passes.
- **Deviations from D-RPR-1:**
  - it adds probe-local glue types;
  - it uses the synthetic fixture, not KJV;
  - the escapes of a question complete that question, and its answer, not its escapes, seeds the next one, so H1 as worded is not tested;
  - it takes no timings.
- **Why it is not merged:** D-RPR-1 is not authorized.

