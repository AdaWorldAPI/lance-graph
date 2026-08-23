# Architecture report — is there a learnable behavioural micro-IR already here?

> **Fathoming report, 2026-08-22.** Reconstructed from code, not analogy.
> Grades: **[G]** read in current code / merged decision · **[H]** plausible,
> bounded · **[S]** analogy only. Every [G] carries `file:line`.
> Nothing here is minted, no opcode is allocated, no type is proposed.

## A. Executive verdict

| # | hypothesis | verdict |
|---|---|---|
| 1 | **cognitive micro-IR** | **REAL MECHANISM** — it is a bytecode, not a metaphor: `Call{function: FnIndex(u8), values: [u8; N]}`, packed in-slab, with bodies, branches, reference resolution and a validator. |
| 2 | **BPE behavioural macros** | **BLOCKED ON MISSING TRACE** — the algorithm is published and positive; the *input* does not exist. Nothing executes the IR, so there is no trace to merge. |
| 3 | **potholes as training events** | **BLOCKED ON MISSING TRACE**, and separately **damaged at the label**: the refusal that would be the training label discards one of two simultaneous causes before it is ever computed. |
| 4 | **R2IL ↔ cognitive shared behavioural layer** | **RHYME ONLY — against the pairing the prompt proposed.** R2IL is not in ruff. But the prompt's *shape* is right about something else: there are **two** op-streams in this workspace, they have already converged structurally without importing each other, and **one of them executes**. |

**The one-sentence answer to the session's question:** we did not accidentally
build a learnable behavioural substrate — we built **the front half of a
compiler** (a byte-exact instruction format, a validator, a refusal taxonomy,
an operand addressing scheme) and **none of the back half** (an interpreter, an
executed trace, a profile). Learning is not blocked by a missing neural
mechanism. It is blocked by a missing **interpreter**, and that is a far smaller
and far more ordinary thing to be missing.

---

## B. Code-truth map — the three op-streams

### B1. `ogar_loco::Call` — the cognitive IR. REAL, VALIDATED, **NEVER EXECUTED**

`OGAR/crates/ogar-loco/src/lib.rs:631-638` [G]:
```rust
pub struct Call {
    /// Which function — an index into the scope's `<256` codebook.
    pub function: FnIndex,
    /// Immediate value bytes, execution-order.
    pub values: [u8; MAX_VALUES_PER_CALL],
}
```
- `FnIndex(pub u8)` (`:374`); `DOMAIN_FLOOR = 0x90` (`:347`) is **stored-byte ABI**
  — the const-assert says moving it "reinterprets every persisted" body [G].
- `LaneShape` (`:267`) fixes immediates per call: `Pairs` → 1 value byte,
  `Triples` → 2, `Quads` → 3 [G].
- `FunctionBody` holds calls **in execution order**, with `calls_per_function()`
  as a hard budget and `remaining()` as the split trigger (`:855-895`) [G].
- `call_in_slab(slab, shape, index)` (`:990`) reads call *i* **in place — no
  gather, no copy** — from a node's 480-byte value slab (`CONTENT_SLOTS × 16`
  facets of `classid(4) + payload(12)`) [G].
- `Program` (`program.rs:49`) has an `entry` body, `references_are_resolvable`
  (`:92`) and `branches_of` (`:124`) — **so nesting and control flow exist** [G].
- Validation gates are named and tallied: `StackUnderflow`, `Uncovered`,
  `SharedCoreDrift`, `ShapeTooNarrowForRefs` → `RefusalGate` / `FunnelTally`
  (`telemetry.rs:33-158`) [G].

**And there is no interpreter.** A crate-wide search for `fn execute`, `fn eval`,
`fn interpret`, `fn step`, `fn run(` returns **nothing** [G]. `telemetry.rs:17-19`
states the boundary itself: *"this crate has no fitness signal to report in the
first place — it only knows whether a candidate parses, casts, and segments."*

### B2. `recipe_vocab` — the 34 recipes lowered onto that IR. **STATIC ORDERING ONLY**

`lance-graph/crates/lance-graph-ogar/src/recipe_vocab.rs` [G]:
- recipe id `1..=34` ↔ op byte `0x90..=0xB1`; `RECIPE_OP_END = 0xB2` (`:99`) —
  **this is where the prompt's "0xB2..0xFF" free space comes from** [G].
- `ladder_program() -> Vec<FnIndex>` (`:126`) is `dispatch_order()` filtered
  through `op_of`. **It is a static ordering — ascending rung, then id.** [G]
- The operand is **not** a flat index. Module doc `:41-58`: a recipe operand
  resolves against a **prefix-scoped `ValueCodebook`** — *"a codebook scoped to
  the classid prefix the call's own body lives under, not a vocabulary-wide
  table"*; one byte selects one level of refinement and *"depth is bought by
  stacking levels rather than by widening the byte"* [G].
- The convergence the prompt guessed at is **stated in the source**: that
  `6×(u8:u8) = 12` carving is *"byte-identical to
  `contract::attention_facet::AttentionFocusFacet` under `CascadeShape::G6D2`,
  with neither side importing the other (the measured six-site convergence)"*
  (`:52-55`) [G].

**The module disclaims execution in its own words** (`:63-65`): *"It does not
execute a recipe, does not write a `ReasoningBand` (the only writer stays
`with_reasoning_band`), and does not resolve the codebook it declares."* [G]

And the named writer is not a trace writer: `with_reasoning_band` is a
**CausalEdge64 bit-field setter for bits 61-63**, and every call site in the
workspace is in `causal-edge/src/v2_layout_tests.rs` [G].

### B3. `LgjOpDesc` — the op-stream that **actually executes**

`lance-graph-java/native/lgj-abi/src/abi.rs:274-288` [G]:
```rust
pub struct LgjOpDesc {
    pub op:       u32,   // LGJ_OP_*
    pub lane_id:  u32,   // which lane of the resource to read
    pub operand:  i64,   // needle / threshold, sign-extended
    pub combine:  u32,   // 0 = AND (narrow), 1 = OR (widen)
    pub _reserved:u32,
}
```
- `View.where()` crosses **zero** times — it appends a `Predicate`; a terminal
  op marshals the whole chain to `LgjOpDesc[]` and makes **one** `lgj_plan_eval`
  downcall (`docs/abi.md:342-360`) [G].
- `plan_eval_impl`'s `for op in ops` loop folds `kernels::eval_predicate` /
  `combine_into` through `ndarray::simd` (`exports.rs:1007-1020`) [G].
- The mask is a packed bitmap (`MASK_WORD` = u64 of 64 row bits), never row IDs
  (`exports.rs:351`); `long[]` as a frontier is **forbidden as normal execution
  state**, materialisation only via methods named `materialize*` (`CLAUDE.md`) [G].
- Maturity: **PROOF-OF-CONCEPT.** 388 disable-verified checks, but every number
  comes from a deterministic SplitMix64 **synthetic** fixture; the real
  `ClassView`/`NodeRow`/`SoaEnvelope` types are *"not built yet, by design"*
  (`docs/architecture.md:119-140`) [G].

### B4. R2IL — **not in ruff**

`/home/user/ruff` crates are the SPO family (`ruff_cpp_spo`, `ruff_python_spo`,
`ruff_ruby_spo`, `ruff_csharp_spo`, `ruff_spo_address`, `ruff_spo_triplet`,
`ruff_sqlalchemy_spo`, `ruff_source_file`). Searches for `R2ILOp`, `Varnode`,
`r2sleigh`, `SpaceId`, `sleigh`, `Ghidra`, `pcode`, `R2IL` return **zero** in
the local checkout [G — for this checkout]. A dedicated inventory pass across
remote branches and `git log --all --grep` was dispatched and had not returned
when this report was written — **so this row is [G] for the working tree and
[H] for the repository as a whole.** The last council's own lesson applies:
a search that comes back empty is not absence. **Do not treat "R2IL is absent"
as settled until that pass lands.**

What ruff *does* have is the SPO harvest arm — AST → `(subject, predicate,
object)` facts feeding the OGAR transpiler (`ruff/AGENTS.md`) [G]. That is a
**declarative fact stream**, not an instruction stream with operands, blocks and
dependencies. It is not the external half of the prompt's diagram.

---

## C. The comparison table the prompt asked for

Filled against what exists, with the prompt's two-column shape replaced by the
three streams actually present.

| property | `LgjOpDesc` (executes) | `loco::Call` (validated, unexecuted) | R2IL | verdict |
|---|---|---|---|---|
| operation identity | `op: u32` (LGJ_OP_*) | `function: FnIndex(u8)`, `0x90..0xB1` minted | not found | **SAME MECHANISM** (both a small typed opcode) |
| operand identity | `operand: i64` immediate | `values: [u8; N]` immediates | not found | **COMPATIBLE ABSTRACTION** |
| operand scope | `lane_id: u32` | prefix-scoped `ValueCodebook`, stackable | not found | **COMPATIBLE** — loco's is strictly richer (hierarchical vs flat) |
| operand width/type | fixed i64 | `LaneShape` → 1/2/3 bytes | not found | **COMPATIBLE ABSTRACTION** |
| address space | resource + lane | classid prefix + cascade level | not found | **RHYME ONLY** — one is storage, one is ontology |
| control flow | none (flat fold) | `Program::branches_of`, `references_are_resolvable` | not found | **MISSING ON ONE SIDE** (loco has it, Lgj does not) |
| data dependencies | `combine: AND/OR` accumulator | stack discipline (`StackUnderflow` gate) | not found | **RHYME ONLY** — a fold is not a dependency graph |
| side effects | mask allocation | *none — nothing runs* | not found | **MISSING ON ONE SIDE** |
| failure / refusal | ABI `i32` status | `RefusalGate` + `FunnelTally` taxonomy | not found | **COMPATIBLE**, loco far richer |
| provenance | none | none | not found | **MISSING ON BOTH** |
| historical version | none on the op | none on the op | not found | **MISSING ON BOTH** |
| reversible expansion | n/a | n/a — no macro layer exists | not found | **MISSING ON BOTH** |
| nested blocks | none | `FunctionBody` + refs + split budget | not found | **MISSING ON ONE SIDE** |

**Reading of the table.** The prompt asked whether external-code IR and
cognitive execution share enough to justify one notion of *typed operation +
typed operand + dependency + transition + provenance + version + reversible
expansion*. Against the evidence: the **first three** properties are genuinely
shared and already convergent; **provenance, version, and reversible expansion
are absent on every side**. So the common denominator is real but **much smaller
than the prompt's list** — it is "typed op + typed operand + scope", which is
the definition of *an instruction*, not of a behavioural IR.

Calling that a shared "behavioural compiler substrate" today would be naming a
convergence that has not happened yet. **The honest claim: two instruction
formats in one workspace have independently converged on the same 12-byte
`6×(u8:u8)` carving without importing each other** — and `recipe_vocab.rs:52-55`
already records that as *"the measured six-site convergence"*. That is a real
finding and it is smaller than the hypothesis.

---

## D. What already exists (do not rebuild any of this)

1. **A byte-exact instruction format with stored-byte ABI discipline** —
   `Call`, `FnIndex`, `LaneShape`, `DOMAIN_FLOOR` const-asserted [G].
2. **In-slab zero-copy call access** — `call_in_slab`, no gather [G].
3. **Bodies, branches, reference resolution, and a split budget** [G].
4. **A refusal taxonomy already shaped as data** — `RefusalGate`,
   `FunnelTally::from_results` [G]. This is the single most learning-ready
   artifact in the workspace and it is about *parse validity*, not behaviour.
5. **A 34-recipe vocabulary minted onto stable opcodes**, `0x90..=0xB1`, with
   `0xB2..0xFF` free [G].
6. **A hierarchical operand addressing scheme** that a flat table cannot express
   — prefix-scoped, stackable, unbounded subtree per shallow prefix [G].
7. **An executing op-stream with a measured profile** — `LgjOpDesc` +
   `bench/RESULTS.md` crossover tables, and the stated intent that *"a future
   planner could even choose the side per-operation using exactly the crossover
   tables"* [G]. **That is profile-guided dispatch with the profile already
   collected.**
8. **The progression doctrine that tells you where a trace belongs** —
   `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`: *think → seal → publish Lance
   version → next cycle reads the published version* [G].

## E. What is genuinely missing — one thing, and it is not a type

**An interpreter.** Nothing consumes a `FunctionBody` and produces effects.

Everything else the prompt lists as missing follows from that one absence:
- no executed trace, because nothing executes;
- no episode boundaries, because there are no episodes;
- no outcome labels, because no outcome is produced;
- no profile, because there is nothing to profile.

Two smaller gaps are **independent** of the interpreter and worth naming
separately, because they will silently corrupt a corpus even after one exists:

- **The refusal label is lossy by construction.** `refusal_of`
  (`recipe_vocab.rs:313-326`) checks the ceiling first and **returns**, so
  `nan_disqualifier` is *never called* when both gates trip. The second cause is
  not hidden — it is **never computed**. A learner would see an `AboveCeiling`
  class silently merging "too deep" with "too deep AND ungrounded". There is
  even an anti-vacuity test proving both can trip at once (`:604-615`) [G].
- **Rubicon cannot be a feature or a label.** `overlap()` divides
  `intersect.len()` by `union.len()`, but `union` absorbs into one minimal
  antichain while `intersect` inserts the deeper of each covering pair
  *independently*: with `self={X1,X2}` siblings and `other={C}` covering both,
  `union.len()==1` and `intersect.len()==2` → **overlap 2.0**, breaking the
  Jaccard bound that `persistence_gain` and `verdict` rest on. Add no
  phase-identity check, no epsilon guard, and no mailbox/version/rung stamp, and
  a bad reading cannot even be attributed back [G]. **Excluded from probe 1.**

---

## F. The smallest falsifier — and it is not a learning experiment

The prompt's §8 designs a BPE-over-traces probe. **That probe cannot be built
today**, because step 1 ("gather actual owner-local executed traces") has no
source. Running it on `ladder_program()` would be running it on a static
ordering — the exact error the prompt itself forbids.

So the smallest experiment that can kill or advance the whole idea is one step
earlier, and it is a **compiler** experiment, not a learning one:

> **PROBE-LOCO-INTERPRETER-1.** Write a minimal interpreter for `FunctionBody`
> over a bounded real corpus. Record, per executed call, exactly:
> `(mailbox_owner, seq, FnIndex, values, pre_version, post_version, outcome)`.
> Nothing else. No macro learner, no BPE, no policy.

Pre-registered kill conditions, decided now:

- **KILL if the 34 recipes cannot be given effects that are separable.** If
  executing recipe R requires a context so large that R's identity stops
  predicting its behaviour, the vocabulary is not an opcode set and hypothesis 1
  degrades from REAL MECHANISM to RHYME.
- **KILL if execution cannot be made deterministic under a fixed version
  horizon.** Replay is a stated invariant; an interpreter that cannot reproduce
  its own trace makes every downstream claim unfalsifiable.
- **KILL if traces are trivially short.** If a typical episode is 1-3 calls,
  there is no sequence to merge and hypothesis 2 dies on arity, not on
  algorithm. *Pre-register the threshold: median episode length ≥ 5 calls.*
- **KILL if every episode is the same sequence.** If executed order just
  reproduces `ladder_program()`, the "trace" carries no information the static
  ordering did not, and there is nothing to learn. **This is the single most
  likely outcome and it must be checked first.**

Only if all four survive does the §8 macro probe become constructible — and
then it must carry the literature's baselines, not compression alone (below).

## F1. Results — PROBE-LOCO-INTERPRETER-1 was run (2026-08-23)

**[G] — this is no longer a proposal.** A minimal interpreter for
`ogar_loco::FunctionBody` was built and run:
`AdaWorldAPI/OGAR` branch `claude/probe-loco-interpreter-1`,
`crates/ogar-loco/examples/interpret_probe.rs`. Corpus: four hand-authored
real algorithms with independently-known-correct answers (GCD, a
summation via `REPEAT`, FizzBuzz-style classification via nested
`IF_ELSE`, Collatz step counts via `WHILE`), 44 real inputs total, each
run twice for a determinism check. Every arithmetic answer was checked
against a ground-truth function that shares no code with the
interpreter.

Kill conditions, as pre-registered above:

- **KC2 (deterministic replay): PASS.** All 44 episodes replayed
  byte-identical call-for-call.
- **KC3 (median episode length ≥ 5): PASS.** Median = 23 calls across 44
  episodes (Collatz(27) alone traced 2089 calls).
- **KC4 (traces are not all `ladder_program()`'s static ordering):
  PASS — and decisively.** 44 distinct call sequences across 44
  episodes; this was the single most-likely-to-kill condition and it
  did not fire. Real, input-dependent branching over `FunctionBody` is
  now [G], not [H].
- **Independent correctness check: all 44 episodes correct** against the
  ground-truth functions — the interpreter is not merely deterministic,
  it computes the right answers.
- **KC1 (the 34 lance-graph-ogar recipes have separable effects):
  explicitly NOT TESTED, as scoped.** This interpreter covers only the
  shared computational core below `DOMAIN_FLOOR`; the recipes' semantics
  live in `lance-graph-ogar`'s `ThoughtCtx`/`recipe_dispatch` wiring,
  which this run did not pull in. This remains the next required step,
  not a result this run can report — do not read KC1 as passed by
  proximity to the other three.

**A genuine, unregistered finding surfaced while building the
interpreter, not while running it:** the shared core's own declared
`pushes_result` table marks `VAR_SET`/`VAR_CHANGE` as *pushing* a result
(chainable-assignment semantics), and there is no `DROP`/`POP` primitive
in the shared core. That makes `ogar_loco::statements::statement_bounds`
(the crate's whole-body segmentation, built for step-mask masking)
correctly *refuse* — `DanglingOperands` — any ordinary imperative
"set a; set b; …" sequence, because nothing consumes the leftover pushed
values. Nobody had built a real multi-statement program against this ABI
before this probe, so the property was real but untested. It is not a
defect in the probe or in `statement_bounds` — they are answering
different questions (masking vs. execution) — but it is a fact about the
ABI a future macro/learning layer needs to know: **`statement_bounds`
cannot be reused, unmodified, as the unit boundary for a learned-macro
scheme**, because it refuses on exactly the ordinary-assignment bodies a
real program is made of. The interpreter therefore does not use
`statement_bounds` for dispatch; it walks each function body as one
linear program-counter pass (treating `VAR_SET`/`VAR_CHANGE` as void),
using a local backward operand-span scan only where `WHILE`/
`REPEAT_UNTIL` must re-run a condition. See the module doc in
`interpret_probe.rs` for the full account.

**Consequence for §G/§H below:** the falsifier fired in the *surviving*
direction. §H ("if it fails") does not apply. §G's sketch — a learned
macro as a `MacroId` reference with a deopt-shaped fallback, never a new
opcode — is now reasoning about a substrate with a confirmed-executing
IR underneath it, not a hypothetical one. It remains [H]: nothing about
macro induction, BPE-over-traces, or the 34 recipes was tested by this
run, and KC1 in particular gates whether hypothesis 1 (learnable
opcode vocabulary) extends past the shared core at all.

## F2. What the literature demands of that later probe

Primary sources, so the probe is not designed in a vacuum:

- **The mechanism is published and positive over ACTION sequences** — FAST
  (Pertsch et al., arXiv:2501.09747, 2025) applies DCT + BPE to continuous robot
  actions; **Subwords as Skills** (Yunis et al., arXiv:2309.04459, NeurIPS 2024)
  applies BPE to discretized RL action sequences and beats skill-generation
  baselines with orders of magnitude less compute [G].
- **Exact reversibility is a solved primitive** — Sequitur (Nevill-Manning &
  Witten, JAIR 7, 1997) and Re-Pair (Larsson & Moffat, DCC 1999) produce
  grammars/SLPs that re-expand losslessly [G].
- **Merging executed op sequences is the compiler tradition** —
  superinstructions (Ertl & Gregg, TOPLAS 27(1) 2005; up to 4.55×) and
  trace compilation (TraceMonkey, PLDI 2009; 2-20×) [G].
- ★ **The macro utility problem is the named failure mode.** Macro-FF (Botea et
  al., JAIR 24, 2005) ships a four-stage pipeline *because* raw frequency-derived
  candidates are not usable without a filter/rank stage; **Newton & Levine**
  (ECAI 2010) report a measured case where a macro used without control rules
  performs **worse than the no-macro baseline**, and state that prioritising
  frequently-used sequences over ones *meaningful for solving the task* is a
  known limitation [G].
- **Neither literature line accepts "did compression happen" as sufficient.**
  Outcome-level performance is the load-bearing baseline in both [G].

So "frequency is not success" stops being a house rule and becomes a citation.

## G. Architecture if the falsifier fires (interpreter exists, traces are real)

Sketched only to the depth the evidence supports; nothing minted.

- **Do NOT allocate `0xB2..0xFF` for learned macros.** The prompt's Option B is
  the one that survives the constraints: a macro is a **reference** to a
  versioned macro table (`MacroId → expansion[] → hash → scope → provenance`),
  not a new opcode. Reasons from code, not taste: `DOMAIN_FLOOR` is
  **stored-byte ABI** (`lib.rs:352` const-assert says moving it "reinterprets
  every persisted" body), so a learned identity that becomes an opcode becomes
  ABI; and the workspace's own killed-dead-ends list already forbids
  pointers-becoming-magnitudes (EPIPHANIES:8560). A `MacroId` is a pointer.
- **The deopt story is the elegant part and it is already available.** A learned
  macro is a speculative trace with a guard; on guard failure it expands to its
  primitive `Call` sequence and re-executes. That is exactly TraceMonkey's
  shape, and it satisfies "learned macro must expand exactly into canonical
  primitives" for free — because expansion IS the fallback path, not an
  afterthought. **[H]** — plausible and bounded, not established.
- **Profile-guided dispatch already has its profile on the other side of the
  workspace** — `bench/RESULTS.md`'s crossover tables plus the stated intent
  that a planner could choose per-operation. That is the same mechanism a macro
  dispatcher needs. **[H]**

## H. Architecture if it fails

If PROBE-LOCO-INTERPRETER-1's fourth kill condition fires — executed order just
reproduces `ladder_program()` — then:

- **Hypothesis 2 is retired**, not deferred. There is no procedural vocabulary
  to induce because the procedure is fixed.
- **Hypothesis 3 narrows to a diagnostics question**: potholes remain worth
  recording as structured events (and the refusal-flattening defect is worth
  fixing regardless), but they are a *dashboard*, not a training set.
- **Hypothesis 1 survives** — a bytecode is still a bytecode, and the IR keeps
  earning its place as a compact, in-slab, branch-capable call format.
- What gets retired workspace-wide is the framing that **learning needs a
  separate neural mechanism**. It does not. It needs an interpreter, and if the
  interpreter shows a fixed procedure, then the honest conclusion is that this
  system does not have procedural variety to learn from — which is a finding,
  not a failure.

---

## I. The compiler interpretation (prompt §14) — graded

Is the architecture better read as *primitive ops → traces → profile-guided
optimization → superinstructions → context-sensitive dispatch* than as
*input → neural learner → opaque policy*? **Yes, and it is not close** — but
the reading also exposes exactly which half is built.

| compiler idea | maps to | grade |
|---|---|---|
| instruction format | `Call{FnIndex, values}`, stored-byte ABI | **[G]** built |
| opcode allocation | `0x90..=0xB1` minted, `0xB2..` free | **[G]** built |
| operand addressing modes | prefix-scoped `ValueCodebook`, stackable | **[G]** built |
| basic block / body | `FunctionBody`, split budget | **[G]** built |
| branch resolution | `branches_of`, `references_are_resolvable` | **[G]** built |
| static verifier | `RefusalGate` (stack underflow, uncovered, drift) | **[G]** built |
| **interpreter** | — | **ABSENT** |
| **profile** | — | ABSENT for loco; **[G] built** for `LgjOpDesc` (crossover tables) |
| superinstructions | — | absent; literature-supported **[G]**, unbuilt |
| trace compilation | — | absent |
| guard failure / deoptimization | — | absent, but the natural fit **[H]** |
| versioned IR | Lance versions + the progression doctrine | **[G]** available, unwired to any trace |
| inline caches / specialization | — | absent |

The instructive pattern: **every front-end box is built and every back-end box
is empty.** That is a very specific and very fixable shape. It also explains why
the "learning" framing kept reaching for a neural mechanism — the missing piece
sits in a part of the pipeline nobody had named, so it got mistaken for a
missing *kind* of machinery rather than a missing *stage*.

## J. Pending — do not read this report as complete

Five investigation lanes were dispatched and had not returned when this was
written. Their absence bounds three claims:

1. **ruff / R2IL across all branches + `git log --all --grep`.** §B4 is [G] for
   the working tree and [H] for the repository. If R2IL exists on a branch, the
   §C table's third column changes and hypothesis 4 must be re-judged.
2. **lance-graph symbol census** (recipe_dispatch, recipe_kernels, MailboxSoA,
   attention_facet, episodic, counterfactual surfaces).
3. **PR bodies #978-#981, #986, #988 head** — the merged-decision record.
4. **A dedicated recipe-substrate read** — independent confirmation of §B1/B2.
5. **The trace-corpus field matrix** — which of `mailbox / seq / op / operand /
   before-version / after-version / rung / focus / witness / failure / outcome /
   causal validity` have a real source. §E asserts the interpreter is the only
   missing thing; that matrix is what would falsify or confirm it.

**Anything in this report contradicted by those lanes should be corrected in
place, not defended.**

## K. Governing constraints — compliance check

| constraint | status |
|---|---|
| no new CE64 bits | **honoured** — nothing proposed touches CE64 |
| no widening `6×2×8bit` | **honoured** — the operand scheme stacks levels, by design |
| no magnitude stuffing | **honoured** — §G chooses MacroId (a pointer) over an opcode |
| pointer/reference before inline learned payload | **honoured** — Option B |
| no global mutable learning dictionary | **honoured** — the macro table is versioned and frozen; §G states no ownership argument has been made, so none is claimed |
| one-writer-per-mailbox | **untouched** — no write path proposed |
| no hindsight leakage | **pre-registered** as a kill condition in §F |
| replay under original version horizon | **pre-registered** as a kill condition |
| macro expands exactly into primitives | satisfied structurally by the deopt shape **[H]** |
| primitive meanings never mutate | **honoured** — macros are references, base ops immutable |
| learner proposes; causal/NARS/provenance gates stay authoritative | **honoured** |
| `kanban_actor` is visibility | **verified** — its module header IS the tombstone for `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` [G] |
| do not use `ladder_program()` as evidence of learned behaviour | **enforced** — §F kill condition 4 exists precisely for this |
| frequency ≠ success | **enforced, and now cited** (Macro-FF; Newton & Levine) |
| negative controls mandatory | **pre-registered** |

## L. One open item the session raised and could not close

`max_rung_admitted` (`recipe_vocab.rs:213-222`) computes an admissible rung from
`census(CognitiveWork) − census(Commit)`. It takes **no `MailboxId`**, and
`PhaseCensus::observe` is documented as a pass over *"an iterator of
`MailboxSoAViews"*. So fleet load can shape one mailbox's admissible thinking
depth — the falsifier the prompt asked for holds **by signature**.

**But as shipped it is unwired**: `max_rung_admitted`, `admitted_program`,
`grounded_program` and `refusal_of` are called only from `#[cfg(test)]` inside
`recipe_vocab.rs` itself [G]. So this is a latent property of the design, not an
observed cross-mailbox effect. Recorded, not fixed — the prompt said not to fix
it here, and nothing in the probe depends on it.

---

## M. Required corrections to PR #988

#988's rewrite retired a BPE thread wholesale. That retraction was **correct for
one hypothesis and wrong for another**, and it conflated them — the same
homonym failure the document itself was written to record.

**Three distinct claims wear the word "BPE" in this workspace:**

| | claim | status |
|---|---|---|
| **A** | BPE is the *mechanism* behind the `6×2×8bit` centroid/ontology codebooks | **RETIRED, correctly.** `E-BPE-IS-RHYME-VQ-IS-THE-MECHANISM-FOR-6X2X8BIT-1` — *"RHYME, not mechanism"*, with the hazards named (*loss of `is_ancestor_of`*, *frequency-optimal replacing distance-optimal*). Those are **codebook** hazards. |
| **B** | BPE (or Sequitur/Re-Pair) induces reusable behavioural **macros over executed `(FnIndex : Value)` traces** | **NOT TESTED, NOT RETIRED — and published-positive.** FAST (arXiv:2501.09747) and Subwords as Skills (arXiv:2309.04459) do exactly this over action sequences. Blocked here only by the missing interpreter. |
| **C** | BPE merge **rank** is a frequency statistic usable as a cheap surprisal proxy | separable from both; untouched by the cited finding. |

**#988 collapsed B into A.** The evidence that the collapse was invalid: A's
falsifiers are all about centroid allocation and prefix containment — none of
them says anything about whether a *trace* can be merged into macros. A
retraction requires evidence just as an assertion does, and A's evidence does
not reach B.

**Exact edits #988 needs:**

1. **§0 row 3** currently retracts "a BPE reading of `6×2×8bit` — the *analogy*
   half". Scope it explicitly to **A**, and state that **B and C were not tested
   by the cited finding.**
2. **§3's restored "BPE merge RANK" pointer** should be split into the **B** and
   **C** entries above — it currently carries C only, leaving B unrecorded
   anywhere despite being the strongest of the three.
3. **§0's closing pattern paragraph** should gain this as a fourth instance of
   *a name taken for a mechanism* — "BPE" now demonstrably spans three claims,
   alongside BNN (binary vs Bayesian), plasticity (three objects) and the
   `6×2×8bit` homonym.
4. **§1 T-B (potholes as escalation-routing labels)** is superseded in an
   important way by this report: its gate assumed pothole → dispatch labels are
   available. They are not, because nothing executes; and the refusal label is
   lossy at the source (`refusal_of` short-circuits before computing the second
   cause). T-B should point at PROBE-LOCO-INTERPRETER-1 as its prerequisite and
   name the refusal defect as a blocker.
   **Update (2026-08-23): the prerequisite is now half-met.** The interpreter
   exists and runs (§F1) — but only over the shared core. T-B's actual
   dispatch labels live in the 34 lance-graph-ogar recipes, which the probe
   explicitly did not execute (KC1 untested). T-B is therefore un-blocked on
   "does an interpreter exist" and still blocked on "does the recipe layer
   execute" and "is `refusal_of`'s short-circuit fixed" — two separate,
   still-open prerequisites, not one.
5. **§4's honesty box** should gain the citation for "frequency is not success" —
   Macro-FF (JAIR 24, 2005) and Newton & Levine (ECAI 2010) — replacing the
   house-rule phrasing with the actual literature.

**What #988 got right and should keep:** the §0 error table's discipline, the
per-atom Learned-lane defect, the retraction-needs-evidence rule (which is what
caught this), and the two restored sub-ideas. The document's shape is sound; one
row of its content was wrong in the direction it was most primed to be wrong.
