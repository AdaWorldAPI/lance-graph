# D-GRAMMAR-WHEN-1 — relational guard binding × DT64 × global WHERE

**Date:** 2026-11-03. **Status:** SOURCE AUDIT + LAB PROBE.

- **Base:** `main` @ `0df42b9` (the #1947 merge).
- **Probe:** `crates/grammar-constraints-probe`.
  - New module `src/when.rs`.
  - New tests `tests/when.rs`.
  - New example `examples/when_cost.rs`.
- **Not changed:** no production code, no ClassId mint, no `F###`, no executor. Nothing here is an admission.

**Question:** can DT64 bind and invoke the meaning of a grammatical WHEN lawfully? The case is two read-only methods that share `obl:arg ∧ Case=Dat` and differ in the head verb's frame.

---

## 0. Correction to #1947 (stated first, because it changes the answer)

#1947 said the head hop `Pred::EqU32Via` "executes; its call sites are EXO-140-unclassified". That is true but understated. Source:

- **ADMISSION B, the operator's words** (`.claude/board/entries/2026-10-20-six-decisions.md:26-34`):
  > "A position derived from another relational column or table is not admissible as a read address merely because the lookup occurs during preparation. … Execution remains co-addressed."
- **The same entry, `:46-48`:**
  > "`Filter::Semijoin`, `Pred::EqU32Via` and `MaskOp::Gather` need call-site classification before any removal."
- **The 2026-10-16 banner** (`entries/2026-10-16-wcoj-probes-1-3.md:3`) calls a read
  > "AT A POSITION TAKEN FROM ANOTHER LANE'S VALUES"

  a violation. The correct shape is `Thought(t)[T_a[i] : T_b[i]]`: "THE JOIN IS THE LAYOUT".
- **Sibling atom.** `EqU32ViaLane`, a sibling closing atom, is marked "⛔ VIOLATION — GATHER. DO NOT BUILD" (`STATUS_BOARD.md:580`, D-ASM-11A).

`EqU32Via` reads `foreign[fk[i]]`. That is exactly a position taken from another lane's value. **No text admits it. It is not forbidden by an explicit ruling either: it is unclassified, and gather-shaped by the rulings' own definition.** #1947's lab result stands as LAB. Its physical mechanism is in the disputed class.

## 1. Source-backed wiring map

| arrow | exists? | source |
|---|---|---|
| Global WHERE (Quack `Plane`/`Keep`) | yes, production | `lance-graph-quack/src/lib.rs` `Filter::Plane` |
| DR population + frame check | yes | `lance-graph-dt64/src/dr.rs` `LocalMap`, `MapTable::map` |
| DT64 `MethodBinding` ON × WHEN × DO | yes | `lance-graph-dt64/src/lib.rs:197` |
| WHEN = `Guard{SetExpr, Logic}` | yes, single population | `lib.rs:152`; `set_algebra.rs:135` (`Sym`, `Res::{I32,U32}`, `And/Or/Not`) |
| `admit` checks the guard | **witness equality only** | `lib.rs:832-858` |
| `validate` checks guard lanes | **no.** ON/reads/effects are checked; WHEN lanes are not | `lib.rs:900-955`; test `dt64_does_not_validate_guard_lanes` |
| A read-only ("returns") DO | **no.** Every `Effect` writes a `target` field | `lib.rs:128-137` |
| `SetExpr` ↔ relational description | relalg `Rel` ↔ `SetExpr` (single table) | `lance-graph-relalg/src/normalize.rs:82-105` |
| `SetExpr` → mask-risc | yes, `set_lower` (D-CMR-2, same-table runs) | `lance-graph/src/fold/set_lower.rs` |
| RelAlg hop / join atom | **no**: `Rel` = `All/Empty/Plane/Atom/And/Or/Not` | `lance-graph-relalg/src/ir.rs:47` |
| Quack head hop | `Filter::EqU32Via` → `Pred::EqU32Via` | `lance-graph-quack/src/lib.rs:358`; `mask-risc/src/ir.rs:158` |
| R2IL WHEN execution | **refused**: `GuardNotExecutable` (`binding.rs:291`), `ThreeValuedNotExecutable` (`:293`) | re-run: `what_the_reading_cannot_honour_is_refused_at_prepare` ok, r2il integration 52/52 |
| R2IL is mandatory? | **no**: "A consumer executes its own prepared native plan … without consulting any codegraph" | `binding.rs:6-10` |
| Read-only admission policy | LAB: `Admitted::Pure` | `crates/dt64-admission-probe/src/lib.rs` |
| Vector × tile fold (co-addressed hop) | **no** (LAB `relcomp` only) | `entries/2026-11-03-d-dr-fold-map.md` (C) |

## 2. Actual DT64 / execution refusals

The first typed refusal on each path:

- **Production R2IL path.** `prepare` returns `ExecRefusal::GuardNotExecutable` for any WHEN, before any effect is read (`binding.rs:290-292`).
- **DT64 itself.**
  - No refusal is reachable for the frame difference: `SetExpr` cannot state it.
  - The only hop-shaped leaf is `Sym(u32)`, whose meaning "is the consumer's business" (`set_algebra.rs:140-142`).
  - `admit` therefore accepts any `Sym` number the witness carries. Nothing checks what it means.
- **Lab composer (approach C).** The lab composer adds `DescriptionMismatch`, `NotReadOnly` and `LexiconRevision`. These are lab refusals, not production refusals.

## 3. Approaches A / B / C

| | A: hop leaf in `SetExpr` | B: WHEN references a prepared relational description | C: cold composition under a shared WHERE |
|---|---|---|---|
| What it is | e.g. `HeadLemmaEq(fk, foreign, v)` | WHEN carries an identity of a Quack/RelAlg description | admit → check description against the witness → lower description ∧ WHERE into ONE program |
| Expressible today | no | the description, **yes, in Quack** (`Filter::and[cmp, cmp, or[eq_u32_via…]]`). Not in RelAlg (`Rel` has no hop). DT64 can only point at it through an opaque `Sym` | **yes, lab** (`when::compose`); its physical op is whatever the description lowers to |
| Witness keeps the relation | yes, if the leaf names (fk lane, foreign lane, value) | **only by convention**: `Sym(digest)` (lab). The store does not verify the digest | via the description check (lab) |
| Duplicates RelAlg/Quack | yes: a second hop vocabulary | no | no |
| Leaks physical navigation into the guard | yes | no | no |
| Global→local authorization | none: the leaf would read `foreign[fk[i]]` in the hot path | same as the description | same |
| Admission | ADMISSION B text applies directly (gather-shaped) | the same, once lowered: `Pred::EqU32Via` | the same, **unless** the description is co-addressed (C′) |
| Hot cost (lab, 2 methods, 40 tokens) | n/a | n/a | Via form 0.56–0.57 µs/run, 12 ops |

**C′, the co-addressed form ("the join is the layout").** The head's frame code is a lane at the DEPENDENT's own coordinate, `head_frame[i]`.

- **WHEN.** It becomes row-local and fully stated in today's `SetExpr`: `And[eq_u32(DEPREL,·), eq_u32(CASE,·), eq_u32(HEAD_FRAME, GIVE)]`.
- **Witnesses.** They are distinct with no `Sym`.
- **Query.** It reads every lane at address `i`: no `EqU32Via`, 10 ops, 0.41–0.47 µs/run.
- **What remains.** The cross-row read moves into the producer, which writes `head_frame[i] = frame(lemma[head[i]])` once. The probe counts 35 lookups for 35 non-root tokens.
- **Open membrane.** That producer step is the EXO-46 / D-RMAP-1 membrane. ADMISSION B leaves one door open for it ("rewritten as a sequential, witnessed relationship projection it could come under B", `six-decisions.md:40-44`). The lab producer is a plain loop, not that witnessed projection.

**Verdict.**
- C reuses existing algebra (Quack `Filter` + `lower` + mask-risc; no new executor). But C over the Via form inherits the admission problem.
- C′ is the only form whose query side matches the rulings as written. It moves the open question to the producer.
- C does not win on its own. **C′ = C with a co-addressed description** is the smallest lawful candidate.

## 4. Falsifier (`tests/when.rs`, 16/16; `tests/p0.rs` from #1947 still 12/12)

**Setup.**
- **Fixtures and gold:** #1947's.
- **Independent oracle:** a string-level evaluator over the token structs. It uses no ids and no programs.
- **Bindings:** `CandidateRecipient` (`MethodId 1`) and `CandidateExperiencer` (`MethodId 2`), registered in a real `BindStore`.
  - Lab stand-in classids `0x7F21_0000` (Via) and `0x7F21_0001` (co-addressed); unminted.
  - No effects.
  - `Population::Distinct`, `Truth::TwoValued`.

**Results.**
- Both forms equal the oracle and the gold: recipient = {105, 405}, experiencer = {302}.
- The recipient and experiencer witnesses are distinct in both forms. Stripped of the frame part, they coincide (the #1947 finding).
- **UNKNOWN stays UNKNOWN.** *helfen* gives no candidate, its `head_frame` is `FRAME_NONE` (no entry, not a negative role), and #1947's abstention rule still reports it.
- The Direction/Purpose ambiguity is kept.
- Candidates are read-only: `effects` is empty and `writes` is empty. A method with an effect is refused (`NotReadOnly`).

**Negative controls.** Each prediction was written in the test's doc comment before it ran.

| # | control | predicted = observed |
|---|---|---|
| 1 | head-frame condition removed | recipient = {105, 302, 405, 505}; against the bound guard: `DescriptionMismatch` |
| 2 | GIVE ↔ EXPERIENCE swapped in the lexicon | Via: `DescriptionMismatch` (digest moved). Co-addressed: `LexiconRevision{bound 1, population 2}`; recomposed honestly under rev 2, the answers swap |
| 3 | Dat → Acc | no candidate |
| 4 | dependency reversed | no candidate |
| 5 | referenced head removed | typed refusal `Map(Unmapped)` at production, never a FALSE role |
| 6 | foreign frame | `Stale` |
| 7 | foreign ClassId | `UnknownClass` from `admit` |
| 8 | deprel operand removed | `DescriptionMismatch`; under its own binding the instrument 408 becomes a recipient |
| 9 | Direction/Purpose | both kept (#1947 rule, same population) |
| 10 | one WHERE plane (tokens of A and C) shared by both methods | recipient {105}, experiencer {302}; equals unscoped ∩ W, in both forms |

**Source disables.** Each anchor was asserted and read from the `test result` line; 9 guards, all red:

| guard disabled | red |
|---|---|
| witness check | n01, n02, n08 |
| lexicon revision | n02 |
| co-addressed frame | 5 |
| Via frame | 5 |
| Sym digest | witnesses, n02 |
| producer reads head (not self) | 5 |
| read-only check | 1 |
| WHERE plane | n10 |
| frame check | n06 |

## 5. EXO-140 classification proposal (NONBINDING)

`Pred::EqU32Via` reads `foreign.lanes[key][fk[i]]` (`ir.rs:148-158`). Its properties:

- It is an indexed per-row load at a position taken from a value. It is **not** a vertical lane read, and not a same-table attested seek.
- The mapping (`fk` = HEAD) is resident. In this probe it is versioned only through the population's frame check.
- It runs in the hot path, per row, every run. It is not bound once in preparation.
- It cannot cross a tile: `fk` indexes one foreign lane slice.

**Proposed classification:** gather-class read (ADMISSION B "position derived from another relational column", option C), not admitted. **Smallest admissible scope for an operator to consider:** none for the query path. Instead, admit the *producer* of a co-addressed relation lane under B's "sequential, witnessed relationship projection". This is an operator decision.

**What RelAlg lowers to.** RelAlg cannot lower to `EqU32Via`: it has no hop. Quack's `Filter::EqU32Via` lowers to exactly that primitive. The co-addressed description lowers to `Pred::EqU32` only.

## 6. Frame lexicon: reuse candidates

| existing | fits? |
|---|---|
| `lance-graph-contract/src/grammar/verb_lexicon.rs:47` `FAMILY_LEXICON` (lemma → `VerbFamily`, 12 families → TEKAMOLO `SlotPrior`) | no slot → role mapping; the file calls itself "starter knowledge" |
| `grammar/witness.rs:134-137` names `ArgumentStatus{CoreArgument, GovernedOblique, CircumstantialOblique, Unknown}` | named, not implemented |
| `revision.rs:104` `LanguageId(pub u16)` (in `InterpretiveHorizon`) | language scope exists, as an "opaque" id |
| deepnsm-v2 `haspelmath.rs` `FormAttestation` | form → function set, not lemma → frame |
| UD | HEAD/DEPREL/FEATS only; no frames |

**Gap.** No contract holds `(language, lemma, slot, role, source revision)`. The lab `Lexicon{rev, language, entries}` is fixture data.

**Requirements for an owner design.**
- A frame lexicon needs a revision that the produced lane carries. Control 2 shows why: a swapped lexicon is otherwise undetectable in the co-addressed form.
- Multiple frames per lemma need either a mask-test leaf in `SetExpr` or multiple candidate rows. `SetExpr::Res::U32` is eq/ne only; Quack already has `Cmp::MatchU32`.
- A missing entry = `FRAME_NONE` = abstain, never a negative role.

## 7. Two role vocabularies

| function | CE64 `Orientation` (`causal-edge/src/orientation.rs:17`, 3-bit code) | deepnsm-v2 `DativeFunction` (`haspelmath.rs:27`, bit in a `u8` set) |
|---|---|---|
| none / plain core argument | `Core` = 0 | — |
| Direction | 1 | 0 |
| Recipient | 2 | 1 |
| Experiencer | 3 | 2 |
| Purpose | 4 | 3 |
| Beneficiary | 5 | 4 |
| Predicative possessor | 6 | 5 |
| External possessor | 7 | 6 |
| Judicantis | — | 7 |

- **Overlap.** The vocabularies share 7 functions, with ordinals offset by one.
- **Not subsets.** Neither is a subset of the other: CE64 has `Core`, deepnsm-v2 has Judicantis.
- **Not equivalent.** CE64 holds ONE code per edge. `DativeFunction` is a SET (`u8` bitmask).
- **Projection.** CE64 is a lossy projection for any ambiguous reading (Direction|Purpose) and has no Judicantis.
- **Recommendation for representation, not identity.** Keep candidate roles as separate masks/relations, one per function, as both probes do. Canonical identity is an owner decision.

## 8. Candidate-to-fact boundary

**Existing boundary.**
- Layer 0 GIVENS are "written only by hydration (pearl::hydrate), never overwritten" (`knowledge/epistemic-frontier-capstone.md:34`). Layers 1..63 are HYPOTHESES (`:38`).
- A write is "the Rubicon decision … landed into Lance only when it is satisfactory" (`plans/thought-git-basics-v1.md:45`), gated by `mul::GateDecision {Flow, Hold, Block}` (`:155`).
- D-STAUNEN-3: "candidate relations in THOUGHT, not WORLD … independent observation permits promotion" (`docs/research/learning/d-staunen-3-xlog-nsl-hydration.md:7`).

A grammatical candidate is a layer ≥1 hypothesis. Minimum conditions before promotion:
1. The supporting observations are named: the token gids and the source annotation revision.
2. The frame lexicon revision is recorded with the candidate. The produced lane carries it (control 2).
3. An explicit selection policy decides ambiguous candidates. None exists; nothing is chosen today.
4. The write goes through hydration / the Rubicon gate, never through a method `Effect`.
5. UNKNOWN (no frame, no Case) stays UNKNOWN.

**Out of scope.** No confidence score is invented here, and no causal certification. CG-3 SELECT/REMOVE is excluded.

## 9. Measurements

`cargo run --release --example when_cost`, Xeon @ 2.10 GHz, three runs:
- 40 tokens; 2 programs per form.
- One tile read per lane per run.

**Cold (reported, not gates):**

| phase | cost |
|---|---|
| vocab bind | 4.7–7.0 µs |
| DR map | 5.9–6.4 µs |
| producing `head_frame` (35 lookups) | 9.1–10.0 µs |
| registering 4 methods | 8.3–8.9 µs |
| compose ×2 (admit + witness check + lower), Via | 21.4–22.0 µs |
| compose ×2, co-addressed | 16.0–18.6 µs |

- The lab producer uses a linear search per token. It is not a production algorithm.

**Hot (both methods, including the frame and lexicon-revision checks):**

| form | mask ops | cost |
|---|---|---|
| Via | 12 | 557–574 ns/run |
| Co-addressed | 10 | 413–465 ns/run |

**Other properties.**
- **Allocation:** 0 B per hot run.
- **Working set:** 5 `u32` lanes × 40 = 800 B, plus scratch.
- **Not measured:** binding enumeration (masks only). #1947's 1.2–1.3 µs covers 5 rules and is not comparable one-to-one.
- **Scale:** no corpus-scale claim.

## 10. Owner decisions (with evidence)

1. **EXO-140.** Classify `Pred::EqU32Via` at the grammar call site. Proposed: gather-class, not admitted on the query path (§5).
2. **EXO-46 / D-RMAP-1.** Admit, or refuse, a producer that writes a co-addressed relation lane (`head_frame[i]`) as a "sequential, witnessed relationship projection" under ADMISSION B.
3. **D-DT64-6.**
   - Should WHEN carry a description identity the store verifies, rather than an opaque `Sym`?
   - Should `validate` check guard lanes against the basis?
   - Should a read-only "returns" DO exist?
4. **Frame lexicon owner** (OGAR codebook, DT64 class data or neither), with revision and language scope (`LanguageId` exists).
5. **Role vocabulary identity:** CE64 `Orientation` vs `DativeFunction`. Representation stays per-function masks.
6. **Promotion policy** for candidates (selection policy; hydration / Rubicon gate).

## Final verdict

- **WIRE NOW:** none in production. In the lab, `compose` connects existing parts only (`admit`, `Guard::witness`, Quack `Filter` + `lower`, mask-risc `execute_into`, dt64 `MapTable` frames).
- **PROBE:**
  - a witnessed sequential producer for `head_frame` at GSD scale;
  - the multi-frame lexicon (a mask-test leaf);
  - the R2IL path with WHEN once D-DT64-6 rules.
- **OWNER DECISION:** items 1–6 above.
- **PARK:**
  - cross-language runtime;
  - full CG-3 ordered SELECT/REMOVE;
  - grammar learning;
  - recursion (EXO-73).
- **SKIP:**
  - a new grammar executor;
  - a grammar AST;
  - a second relation store;
  - copied presence planes;
  - scratch-mask handoff between programs;
  - a new `F###` family. The F000 polyfill words `IntEqual` / `IntAnd` / `IntOr` already name every op the co-addressed WHEN needs.
