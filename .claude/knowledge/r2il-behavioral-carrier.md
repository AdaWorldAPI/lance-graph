# KNOWLEDGE: The Behavioral Carrier — What May Carry a Learned Macro

## READ BY: ALL AGENTS. MANDATORY before any work that touches
##          behavioral compression, learned/frozen macros, R2IL or
##          machine-code carriers, thinking-style microcode, or ANY
##          proposal to add a BPE table, a token vocabulary, a
##          reward model, or a "learner" subsystem.

## P0 RULE: The learner already exists and is 16 bytes of shipped types.
##          Before proposing a subsystem, read § "Already shipped" and
##          `nars/truth.rs` + `nars/belief.rs`. Proposing a type that
##          already exists is the 30-turn rediscovery tax `CLAUDE.md`
##          names by that phrase.

## P0 RULE: Sequential adjacency is NOT composition. On the pass-1
##          SEVEN-OPCODE PROJECTION of two binaries, the single top
##          recurring opcode trigram over-admits 99.7% (379 of 380
##          occurrences are not def-use linked), against an all-pairs
##          def-use base rate of 31.5%. This is NOT a claim about
##          x86-64, and NOT a claim about window matchers in general —
##          see F-1 and the boundary in F-11 before quoting it.

---

## Scope: two BPEs that do not transfer

```
  TOKEN BPE       intake tokenization of symbol streams into the
                  existing 12-byte 6×(8:8) payload
                  → status CAN-FIT, NOT YET BUY  (F-6)

  BEHAVIORAL BPE  compression of recurring typed #1001/R2IL
                  transformations into resident macros
                  → status BEHAVIORAL COMPRESSION CARRIER: UNDECIDED
```

**Neither result transfers to the other, in either direction.** The scope
fence is stated in `probe_token_bpe_geometry.rs` (module docs) and again
in `E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1`. They may later share recurrence
machinery; they do not share semantics.

---

## The measured findings

| # | Claim | Status | Probe / gate | Source |
|---|---|---|---|---|
| F-1 | The top opcode trigram over-admits **99.7%** on the seven-opcode projection (not a general window-matcher claim — see F-11) | FINDING [MEASURED] | `PROBE-R2IL-REAL-EPISODES-1` gate **E5** | `probe_r2il_real_episodes.rs` |
| F-2 | Real op streams show **type-collapse**, not higher top-1 counts | FINDING [MEASURED] (pre-registration REFUTED in place) | gate **E3** | same |
| F-3 | The `Stamp` mod-64 ceiling **BINDS** at 143 episodes | FINDING [MEASURED] | gate **E4** | same |
| F-4 | A happy-path signal reinforces a **contract-violating** macro above the trust bar | FINDING [MEASURED] | `PROBE-R2IL-FRONTIER-PHASE2-1` gate **R6** | `probe_r2il_frontier_phase2.rs` |
| F-5 | The frontier learner is **already shipped**; no subsystem is needed | FINDING [MEASURED] | `PROBE-STYLE-MICROCODE-FRONTIER-1` gate **S9** | `probe_style_microcode_frontier.rs` |
| F-6 | Token BPE fits the fixed `6×(8:8)` geometry reconstructibly | FINDING [MEASURED] | `PROBE-TOKEN-BPE-GEOMETRY-1` gates T-A/T-C | `probe_token_bpe_geometry.rs` |
| F-7 | A BPE merge tree is **not** lawful HHTL ancestry | FINDING [MEASURED] | gate **T-B** | same |
| F-8 | Membership is **participation**, not ancestry (relation topology, not HHTL) | FINDING [MEASURED] | `PROBE-MULTI-GROUP-MEMBERSHIP-1` M1–M5 | `E-MEMBERSHIP-IS-PARTICIPATION-NOT-ANCESTRY-1` |

### F-1 — the carrier must be the def-use chain (the headline)

Measured on the ruff-side R2IL pass-1 harvest: 143 real functions from real
x86-64 ELF64 binaries, 17,557 typed `FlatFact` rows, 5,340 ops, 2
symtab-bearing binaries (gate E1 cross-validates every count against ruff's
independently committed census; gate E2 partitions the episodes).

```
  top recurring opcode trigram   (int_add, copy, store)
  real occurrences                380
  dataflow-chained (def-use)        1
  NOT chained                     379   →  99.7% over-admission
  adjacency base rate            31.5% of ALL adjacent op pairs are def-use linked
```

The top idiom chains **far below** the base rate: it is an ADDRESSING idiom
(address computed, value staged, store issued), not a dataflow pipe. Real
code interleaves independent chains.

**SSA coverage is total** — 0 of 11,653 operand rows lack a `ValueId` — so a
linkage failure is a dataflow fact, never missing coverage. E5 enforces
non-vacuity both ways (some occurrence must fail the contract; some must
pass), per the `CLAUDE.md` falsifiability rule.

> **Consequence (scoped):** on this projection the macro carrier must be the
> **DEF-USE CHAIN**, not the linear opcode window. Generalizing the phrasing to
> "real machine code" outruns the evidence — see F-11.

**Falsifier:** a corpus where an opcode-window matcher's admissions are
predominantly def-use-chained, or where the top idiom chains at or above the
adjacency base rate.

### F-2 — the refuted pre-registration, recorded not adjusted away

The first E3 pre-registration was *"real top-1 trigram occurrence count >
shuffled"* and it **FAILED** (real 380 vs shuffled 387): shuffling a
`copy`/`int_add`-dominated marginal CREATES monotone `(copy,copy,copy)` runs,
so the control's top-1 grows. Top-1 count was the wrong statistic. The real
structure is **type-collapse**: 97 distinct trigram types real vs 264 under
shuffle (>2.7×; 264 is this probe's own LCG run — the 260 quoted inside
`probe_r2il_real_episodes.rs` is the independent python cross-check), top-10 occupancy 50.5% real vs 33.6% shuffled. The gate pins
the shuffle's top type to the predicted monotone-run artifact — the mechanism
of the refutation, not just its outcome.

### F-3 — the stamp ceiling is a real capacity note

`Stamp::source(id) = 1u64 << (id % 64)` (`nars/belief.rs`). Folding is
**conservative by construction**: it can create false overlap, never false
disjointness, so no-double-count survives. At 143 episodes it BINDS: for
`(int_add, copy, store)` the loop **revised 64** and **CHOICE-dropped 23**
(e = 0.999) — ~26% of the evidence for the widest idiom discarded. Sound, but
no longer a footnote at real corpus size.

### F-4 — happy-path RL would have learned the clobber

`explore-reckless` computes the RIGHT value into the WRONG register,
clobbering callee-saved `r3`. Under a deliberately sloppy oracle ("the doubled
sum exists in SOME register") it succeeds every episode and reaches
**e = 0.812 — above the 0.75 trust bar**. It is refused ONLY by the
falsification intervention checking the actual contract (result in `r2` AND
`r3` bit-preserved). Recurrence + success is not enough; only what SURVIVES
the intervention is learned.

> The falsification-first admission predicate (`LearnedSurvivedTests`) is not
> a nicety at the behavioral level — it is the difference between a learned
> macro and a **learned bug**.

**Falsifier:** an admission rule that freezes on expectation alone and still
refuses the clobber; or a real corpus where value-correct/contract-wrong
macros do not arise.

---

## Already shipped — do not reinvent

The frontier loop is **`TruthValue::revise` + `Stamp` disjointness + CHOICE by
`expectation()`**. Per-macro learned state is exactly one `TruthValue` (8 B) +
one `Stamp` (8 B); gate S9 pins the sizes so a smuggled subsystem fails.

Exact mechanism, read from source (`nars/truth.rs`, `nars/belief.rs`):

- `TruthValue { frequency: f32, confidence: f32 }`;
  `evidence_weight() = c / (1 - c)` (`f32::MAX` at `c >= 1.0`).
- `revise(other)`: `f' = (f₁w₁ + f₂w₂)/(w₁+w₂)`, `c' = (w₁+w₂)/(w₁+w₂+1)`;
  returns `TruthValue::default()` when `w_sum < f32::EPSILON`.
- `expectation() = c·(f − 0.5) + 0.5` — the dispatch/CHOICE key.
- `Stamp(u64)`: `source(id) = 1 << (id % 64)`, `disjoint = (a & b) == 0`,
  `union = a | b`.
- `BeliefArena::revise_at` is the codified guard: **non-empty** incoming stamp
  AND disjoint → `revise` + stamp union + preserved `|f₁−f₂|` contradiction
  depth, in place, rung untouched (`ReviseOutcome::Revised { synthesis_c,
  depth }`); otherwise → CHOICE, keep the higher-confidence truth
  (`ReviseOutcome::Chosen { kept_existing }`); absent statement →
  `ReviseOutcome::Admitted`.
- **The empty-stamp guard is load-bearing:** `Stamp::default()` is disjoint
  from every stamp, so unsourced evidence must NOT pool — it competes by
  CHOICE, or confidence inflates without bound.

The three behavioral probes construct their own per-macro `truth`/`stamp`
pair and call the same two primitives directly; the arena is where the same
discipline is codified for beliefs. **No gradient, no bandit, no Q-table, no
reward-model type exists, and the arc measured that none is needed** (F-5).

Frozen microcode is **bit-immutable**: evolution mints a NEW explore group
(S8). The population does not move; the frontier does.

---

## The standing fences

```
  CONTENT NEVER TRAVELS IN CLASSID.   CLASSID SELECTS THE READING.
    classid = HOW these bytes may be read
    HHTL    = WHERE the resident thing lives
    mask    = WHAT part / group / region conducts
    edges   = HOW addressed things relate
```

(operator law, `E-CONTENT-NEVER-TRAVELS-IN-CLASSID-1`.) No per-macro,
per-copula or per-group classids; no predicate, relation, group or belief
identity smuggled into classid. Companion laws: SHARE THE HIERARCHY, NOT
NECESSARILY THE PAYLOAD; AN INDEX OR MASK MAY ACCELERATE THE ABI, IT MUST
NEVER BECOME A SECOND ABI; MEASURE THE DISTRIBUTION BEFORE BUYING THE
REPRESENTATION.

Further fences, all currently in force:

- **No mint without a ruling.** The V4 plane classid stays provisional (O5
  gate). No probe in this arc mints, reserves, or canonizes anything.
- **`R2IL × BPE` / OGAR-loco routing / V4-as-thinking-dynamic is a THREE-IF
  HYPOTHESIS**, recorded in the mandated conditional phrasing: IF measured
  recurrent typed R2IL behavior requires a resident macro representation, the
  recurrence machinery MAY compress ordered groups into reconstructible
  macros; IF that produces reusable routing structure, OGAR-loco-shaped
  routing MAY carry it; and V4-shaped behavior geometry is ONE possible
  future carrier. **Three IFs, zero decisions.**
- **`ruff_r2il` is never imported** (separate cargo workspace). Probe-local
  mirrors are cited as mirrors; gate E1 exists to catch a schema misread —
  and did (the TSV kind column is CamelCase, not Census's snake_case).
- **Encodability ≠ hierarchy** (F-7): every BPE merge is `(left:right)` with
  both ids u8, yet **3 same-depth token pairs are prefixes of each other**, so
  "siblings" overlap. A binary merge DAG over strings is not a radix prefix
  partition. Do not confuse a merge tree with the ontology tree.
- **`u8:u8` stays two separate bytes**, never widened to a u16.
- **Authority order:** canonical source AUTHORITATIVE → tokenized form exact
  and reconstructible → compressed shorthand must round-trip or is
  non-canonical.
- **If production recurrence turns out too rare, the correct result is NO
  BPE** (`E-MEMBERSHIP-IS-PARTICIPATION-NOT-ANCESTRY-1`).

---

## Honest boundaries (do not over-read these numbers)

| Boundary | What it limits |
|---|---|
| **2 binaries, 143 functions, one source**, at the pass-1 harvest convention (9 opcodes in the census) | F-1/F-2/F-3 are high-confidence *for these binaries*; this is not "all real code" |
| **Toy 4-register machine**, both oracles probe-local | F-4 proves the LOOP and the contract-level falsification, not that real corpora behave this way |
| **Toy world oracle** in Phase 1 (`PushBoundAt` before `PushGapSubject`) | F-5 proves loop MACHINERY; no claim that any real style wins |
| **`Stamp` mod-64 conservatism** | costs ~26% of evidence for the widest idiom at 143 episodes; worse at larger corpora |
| **Fixture-scale BPE corpus** (in-tree KJV Genesis 2–3, 1125 bytes) | every F-6/F-7 number is fixture-scale; COCA, whole-KJV, R2IL streams and AST intakes are ABSENT from this checkout and reported absent, never simulated |
| **B2 recurrence is the mechanical driver's** (same typed pattern per subject) | proves detection machinery; production recurrence is UNMEASURED |

Fixture-scale surprises worth keeping (F-6): scoped per-chapter vocabularies
produced **19% MORE** tokens than one global table; the vocabulary **saturated
at 180 of 255** (the corpus, not the cap, set it); **every** verse overflows
one particle (p50 = 4, max = 8), so continuation rows are the norm; chapter
token-usage Jaccard **0.32** — BPE stayed orthogonal to scope here.

---

## What would change our mind

Each line is a measurement that would supersede a finding above.

1. **F-1** — a real-code corpus (different sources, richer opcode
   convention, optimized vs unoptimized) where opcode-window admissions are
   predominantly def-use-chained, or where the top idiom chains at/above the
   adjacency base rate. Also: a def-use-chain-keyed macro that measurably
   *fails* to reduce over-admission.
2. **F-2** — a corpus where the marginal-preserving shuffle does NOT collapse
   type counts, i.e. real/shuffled distinct-trigram counts within 2×.
3. **F-3** — a stamp representation with capacity proportional to real source
   counts, measured to drop no evidence at the same corpus size while
   preserving the never-false-disjointness property.
4. **F-4** — an admission rule that reaches the same refusals from
   expectation and cost alone, with a can-fire AND can-stay-silent test.
5. **F-5** — a measured workload where `revise` + CHOICE demonstrably cannot
   express the required credit assignment. Absent that, a proposed learner
   subsystem is a rediscovery, not a capability.
6. **F-6/F-7** — a scale corpus (present, not simulated) where a BPE carrier
   measurably BUYS something the canonical source does not already provide,
   with reconstruction still byte-exact. Until then: CAN-FIT, NOT YET BUY.
7. **The UNDECIDED verdict** — behavioral compression is admitted only if ALL
   of the operator's conditions hold at once: typed-IR source units, measured
   recurrence, exact reconstruction, order preserved, applicability preserved,
   truth/provenance/warrants survive, falsification history survives, no
   copy/repack, carrier follows the measured distribution, and no second
   cognitive universe.

---

## Wave results — F-9..F-12 (measured; this section was authored empty and
## filled by the orchestrator after the gates ran)

| # | Claim | Status | Probe / gate |
|---|---|---|---|
| F-9 | The idiom vocabulary **survives optimization completely** | FINDING [MEASURED] (pre-registration REFUTED in place) | `PROBE-R2IL-OPTIMIZATION-TRANSFER-1` T3/T4 |
| F-10 | The def-use chain carrier yields a **3.6x smaller signature vocabulary** with ~1.8x the top-10 mass (no occurrence-size control run) | FINDING [MEASURED] | `PROBE-R2IL-DEFUSE-MACROS-1` C2/C4/C5 |
| F-11 | Residual magnitude is **comparable to or larger than** the classified output (ratio 0.478, **units unverified** — see the note) | FINDING [MEASURED] | `PROBE-R2IL-SLAG-BOUNDARY-1` S5 |
| F-12 | `Stamp` loss is **0 at N≤64 and 55.2% at N=143** | FINDING [MEASURED] | `PROBE-STAMP-CAPACITY-1` K2/K3 |

### F-9 — optimization does not prune the vocabulary, it ADDS to it

TRAIN = `stress_test` (71 fns / 3040 ops), TEST = `stress_test_opt`
(72 fns / 2300 ops), same source, disjoint address keys — a real held-out split.

```
  top-10 idiom transfer        10/10   (COMPLETE, not partial)
  TRAIN trigram types          64  →  61 survive into TEST, 3 TRAIN-only
  TEST trigram types           94  →  33 of them TEST-ONLY
  Spearman rho (shared top-10) 0.600  (order partially preserved)
  ops per function             42.82 → 31.94  (density cut, as predicted)
```

The pre-registration was PARTIAL survival, on the theory that a slice of the
top idioms are unoptimized-compilation artifacts an optimizer removes. **It
was refuted**: the top idioms are optimization-INVARIANT here, and the
optimizer *added* 33 new trigram types while cutting ops-per-function. The
half that held was the density cut. Recorded in place in the probe's own
module docs.

**Falsifier:** a build pair where top-K transfer drops below K.

### F-10 — the chain carrier, measured against the window it replaces

Forward def-use chains (X→Y→Z where a `ValueId` defined by X is consumed by Y,
then Y by Z), built on the same 143 episodes and compared head-to-head with the
adjacency window:

```
  length-3 def-use chains found      1872
  distinct chain signatures            27   vs   97 window trigram types
  top-10 occupancy share            0.887   vs   0.505 window
  top chain occurrences skipping
    >=1 intervening op               95.9%  (439/458)
  chain span (z_pos - x_pos)         min 2, median 6, max 148
```

The chain carrier collapses the vocabulary **3.6× harder** than the window and
concentrates nearly twice as much mass in its top-10 — while 95.9% of the top
chain's occurrences are *invisible to a window matcher at any width the data
would justify* (median span 6, max 148). F-1 said the carrier must be the def-use chain; F-10 is that prescription
executed, and the concentration figures favour it.

**What F-10 does NOT show (meta-review P1):** F-1's content is an
*over-admission rate*, and the chain carrier's over-admission is **0 by
construction** — which the probe's C3 explicitly refuses to count as evidence.
What remains is an *uncontrolled* concentration comparison: 1,872 chain
occurrences against ~5,054 window occurrences over the same 9-opcode alphabet,
so a smaller signature count is partly a sample-size effect. No
occurrence-matched window control was run. The honest claim is the vocabulary
and mass figures above — not "the prescription is proven".

**Falsifier:** a corpus where chain signatures are as numerous as window
trigrams, or concentrate less.

### F-11 — the boundary, and what it does to every number above

The harvest's addressed residual ("slag") is of **comparable or larger
magnitude** than its classified output:

```
  classified rows        17557
  residual count         36747      ratio classified/residual = 0.478
  largest single reason  opcode_not_in_convention  32388/36747 = 88.1%
  conservation           every by_address shape sums EXACTLY to its grouped count
```

The residual is **convention-bounded, not diffuse** — one named reason carries
88.1%, and nothing is dropped unnamed. That is the furnace behaving correctly.

> **Cross-slice consequence (visible only with F-1 and F-11 together, and it
> qualifies every finding in this document):** F-1's 99.7% over-admission, F-2's
> type-collapse and F-10's chain vocabulary are all measured over the
> **classified** rows — the pass-1 seven-opcode convention — while a residual of comparable or
> larger magnitude sits outside it, 88% of it simply "opcode not in convention." None
> of these findings are claims about x86-64; they are claims about the
> **seven-opcode projection** of two binaries. A wider convention could move any
> of them.

> **Units caveat (meta-review P1):** the numerator counts ore TSV **rows**
> (`FlatFact` rows across 5 kinds); the denominator sums the residual ledger's
> **`count` column**, and nothing in the probe establishes that one residual
> count unit equals one `FlatFact` row. The ratio is therefore a magnitude
> comparison, not a like-for-like row ratio. Establishing the unit is the next
> gate this finding needs.

**Falsifier:** a harvest at a wider convention where the ratio inverts and the
findings above survive unchanged (that would strengthen them); or one where
they do not (that would bound them further).

### F-12 — the stamp loss curve

`Stamp::source(id) = 1 << (id % 64)`, measured through the shipped type:

```
  N sources     8   16   32   64    96    143    256    512
  dropped       0    0    0    0    32     79    192    448
  loss %      0.0  0.0  0.0  0.0  33.3   55.2   75.0   87.5
```

Loss is exactly 0 up to 64 distinct sources and strictly positive past it. At
the real corpus size (143 distinct `(binary, function)` evidence sources)
**55.2%** of sources would be CHOICE-dropped if each contributed evidence for
the same macro — the upper bound; F-3's 26% is the measured figure for one
actual idiom, which appears in fewer than all 143 episodes. Modelled wider
registers (128/256-bit) recover 15/0 dropped at N=143 — **modelled only, not a
proposal, not implemented, and the memory/cache/wire costs were not measured.**
Any width change is the operator's ruling.

**Falsifier:** as F-3.

---

## Files

```
crates/lance-graph-planner/examples/probe_r2il_real_episodes.rs      E1–E5
crates/lance-graph-planner/examples/probe_r2il_frontier_phase2.rs    R1–R7
crates/lance-graph-planner/examples/probe_style_microcode_frontier.rs S1–S9
crates/lance-graph-planner/examples/probe_token_bpe_geometry.rs      T-*
crates/lance-graph-planner/examples/probe_r2il_optimization_transfer.rs  T1-T6  (F-9)
crates/lance-graph-planner/examples/probe_r2il_defuse_macros.rs          C1-C6  (F-10)
crates/lance-graph-planner/examples/probe_r2il_slag_boundary.rs          S1-S6  (F-11)
crates/lance-graph-planner/examples/probe_stamp_capacity.rs              K1-K6  (F-12)
crates/lance-graph-planner/src/nars/truth.rs                         revise/expectation
crates/lance-graph-planner/src/nars/belief.rs                        Stamp/ReviseOutcome
.claude/board/EPIPHANIES.md                                          the seven entries above
```

Lock in truths. Measure conjectures. Label everything.
