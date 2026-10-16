# CE64 grammar wiring v1 — every upper-half field of a parsed triple from its source

**Status:** D-CGW-0 SHIPPED IN PR (this branch); D-CGW-1..12 PROPOSAL, test-first,
nothing built. Board entry: `entries/2026-10-16-grammar-coordinates-annotation-wiring.md`
§7–§11 (what landed, with measurements and disable runs).

**Scope.** `crates/deepnsm-v2/src/wire.rs` and the readers of the edges it writes.
A parsed verse becomes rows; each row's `CausalEdge64` bits 0..63 are written from
one named source. The probe `crates/lance-graph-quack/examples/wired_reasoning_probe.rs`
checks the field rules as Datalog over presence planes against a row oracle.

**Reading order.** The CE64 field meanings follow
`docs/architecture/ce64-semantic-upper-half.md` (WORKING-MODEL) and the frozen
revision invariants G1–G7 (`entries/2026-10-14-revision-layer-invariants.md`).
Where this plan and those disagree, they win and this plan is corrected.

## 0. What D-CGW-0 delivers (this PR)

| bits | field | source in `wire.rs` |
|---|---|---|
| 0..23 | S, P, O | the frozen operand ABI of `run_harness` |
| 24..31, 32..39 | F, C | `arm_to_truth_u8`, rule `(S, P) → O`, counted only after the Gadamer gate admits |
| 40..42 | Pearl ladder | `PO` when a Kausal lane is stated on the event, else `SO` |
| 43..45 | orientation | the Haspelmath dative-map function (0 core argument, 1..7 Direction, Recipient, Experiencer, Purpose, Beneficiary, Predicative possessor, External possessor) from the forms attested in the clause span, narrowed by TEKAMOLO lanes |
| 46..49 | activation | the change of the statement's arm frequency this sentence caused, in eighths |
| 50..52 | novelty → wisdom | rung 1 Staunen ..= 7; climbs on an admitted, resolved observation with positive support balance; a contest never resets it |
| 53..58 | W | slot into the cohort's `WitnessTable<64>`: the witness of the same statement's prior state (the Markov arc) |
| 59..63 | epistemic state | topology from the clause path, certification from support balance and the causes arena |

Besides the edge, every row carries `CausalWitnessFacet` loci (Antecedent, TEKAMOLO,
Quorum, Contradiction, all −8..+7 over the row stream), the qualia register
(`FacetCascade`, `G6D2`), the gate's `RevisionKind` and `contested`.

Verification: 229 deepnsm-v2 tests; clippy `-D warnings` on the lib; the probe
(741 rows, 5 rules, 0 violations, each rule broken by a one-bit flip); disable runs
red for every guard listed in the board entry (§7–§11).

## 1. Gate: nothing below leaves the probe until D-CGW-1 lands

Two fields written by `wire.rs` already have other readers with other meanings:

| bits | wire writes | existing readers read |
|---|---|---|
| 43..45 | Haspelmath function code | the per-plane pathology triad: `CausalEdge64::{s,p,o}_pathological`, `concern_level`, `network.rs` Simpson's-paradox detection, `compose` (`edge.rs:808`), `moore_nars16_probe`, `pearl_ladder_probe`, `witness_angle_probe` |
| 46..49 | activation (signed change) | the inference mantissa (+6 Intervention, −6 Counterfactual, NARS rule index): `pearl.rs`, `chain_counterfactual.rs`, `contract::counterfactual::deposit_counterfactual`, `cognitive-shader-driver/src/mailbox_soa.rs` (production) |

So a wire edge handed to `CausalNetwork` would read code 1 (Direction) as
"S pathological", and one handed to the mailbox would read activation +4 as a NARS
rule. Today wire edges go only to the probe. **They must not go anywhere else
before D-CGW-1.**

## 2. Work items

Each item lists: goal; tests to write first (can fire + stays silent, non-trivial
inputs); files; dependencies; the decision it needs, if any.

### D-CGW-1 — versioned reading binding for 43..45 and 46..49

- **Goal.** A wire edge declares which reading its bits carry, and every reader
  refuses a reading it does not implement. `ce64-semantic-upper-half.md`
  §"Encoding, reading, operation": readings are told apart by a versioned binding,
  never inferred. `I-LEGACY-API-FEATURE-GATED` applies.
- **Tests first.**
  - A triad reader (`s_pathological`, `CausalNetwork`) given an edge bound to the
    grammar reading refuses it (error / `None`), and still reads a triad-bound edge.
  - A mantissa reader (`pearl.rs` dispatch, `chain_counterfactual`, `mailbox_soa`)
    refuses a grammar-bound edge; the −6 counterfactual tag still round-trips on a
    mantissa-bound edge.
  - Field-isolation matrix over bits 43..49 for both readings.
- **Files.** `causal-edge/src/{edge.rs,layout.rs,v2_layout_tests.rs}`; the binding
  site (where an edge's reading is known: the tenant's ClassView, not a new bit);
  `docs/architecture/ce64-semantic-upper-half.md` (the "code today" column).
- **Decision.** Where the binding lives. Options: (a) the tenant / ClassView of the
  row that holds the edge; (b) a field outside the edge. A bit inside the edge is
  excluded (no free bit). Recommendation: (a), because a reading is a property of
  the class that holds the register.

### D-CGW-2 — frequency-ranked default, antecedent-tested promotion of the alternative reading

- **Goal.** Corpus frequency gives each ambiguous token a default reading (a
  best-effort median, never a decision). When the readings contradict, the
  alternative reading is promoted as a constituent of a different parse if the
  grammar test passes: the relative pronoun's antecedent within −8..+7, with
  morphological agreement.
- **Measured basis (2026-10-16, DeReKo-2014 local only, CC BY-NC, never committed).**
  The relative-pronoun reading is a minority of every article form: `der` ART
  241.4 M / PRELS 11.5 M (4.6 %), `die` 13 %, `das` PDS 11.6 M / PRELS 6.3 M (9.6 %), `den`
  1.2 %, `dem` 3.3 %; `welche` PRELS dominant. COCA has no relativizer tag: `that`
  c 8.3 M / d 5.0 M / r 0.1 M, `which` d only, `who` p only. So the most frequent
  reading is wrong exactly at a relative clause, and in English frequency cannot
  even propose it.
- **Tests first.**
  - `der Mann, der den Hund sah, ...`: the second `der` (frequency default ART)
    is promoted to Rel because an antecedent noun of matching gender/number lies
    within −8..−1 and the following clause lacks a subject. Stays silent: `der
    Mann gab der Frau das Buch` (no comma clause, no gap) keeps ART.
  - English `the man that saw the dog bit the cat`: `that` promoted to Rel by the
    antecedent test although COCA has no Rel reading; silent on `he said that the
    dog bit the cat` (no antecedent noun directly before).
  - A promotion is recorded as a contradiction resolved by the grammar (the gate's
    resistance), never as a silent re-tag.
- **Files.** `deepnsm-v2/src/{fsm.rs,lexical.rs,coca.rs}`; a DeReKo adapter beside
  `coca.rs` (reads a local path, like `examples/dereko_profile.rs`; nothing
  committed); `wire.rs` only consumes the result.
- **Dependencies.** None for English; the DeReKo adapter for German.
- **Decision.** Agreement features: STTS has no gender/case, so agreement needs UD
  features (`Gender`, `Number`, `Case`) or a small article table. Recommendation:
  UD features when present, else the closed German article paradigm as a table.

### D-CGW-3 — case morphology decides argument order

- **Goal.** `Den Hund biss der Mann` yields `SPO(Mann, biss, Hund)`: when case
  contradicts position, case wins, and the triple is built from it in the parser
  (the role decision is the FSM's, not a post-hoc swap in `wire.rs`).
- **Tests first.** OVS with unambiguous case flips roles; SVO with agreeing case is
  unchanged; syncretic case (`die Frau sah die Katze`) keeps position and marks the
  token ambiguous (the row is not resolved, nothing climbs).
- **Files.** `deepnsm-v2/src/fsm.rs` (`Reading` gains an optional case reading);
  `wire.rs` (none beyond reading the parse).
- **Measure.** On UD German GSD dev (fetched locally, 15,041 lines, 5,604 `Case=`
  features): share of transitive clauses where case and position disagree; parse
  agreement with UD `nsubj`/`obj` before and after.
- **Dependency.** D-CGW-2's agreement source.

### D-CGW-4 — counterfactual lanes: `CausalCandidate` → `Causes`

- **Goal.** A row is certified `Causes` only when `RevisionVerdict::is_acceptable`
  holds: the gate said `IncreaseEligible` AND a counterfactual said `Necessary`.
  The counterfactual removes the cause's contributing verses from the causes arena
  and the effect's arc and re-evaluates the effect's support (W0 vs W_X, D-SAG-CF-1
  §0). Candidates run in parallel, one per hypothesis lane (64-lane budget per
  batch; 72 candidates in the probe corpus → two batches).
- **Tests first.** A cause stated by two independent verses whose effect is
  supported only through it → `Necessary` → `Causes`; the same cause where the
  effect has an independent third source → `Dispensable` → stays `CausalCandidate`;
  a candidate never attacked stays `CausalCandidate` (`NotRun`, G6).
- **Files.** `wire.rs` (record contributors per candidate); the replay over the
  arenas; `contract::revision` (no change; consumes `CounterfactualVerdict`).
- **Constraints.** A counterfactual belief is never written as observed truth; the
  −6 tag does not go into 46..49 of a grammar-bound edge (D-CGW-1). Single cuts
  miss overdetermination (D-SAG-CF-1 §1): joint cuts are a follow-up.

### D-CGW-5 — concessive markers as resistance and as a refuted implication

- **Goal.** `obwohl X, Y` / `X, aber Y`: the writer marks a contradiction on
  purpose. The clash enters the gate as `resistance` (it ranks above an unmarked
  one) and `obwohl X, Y` seeds the counterfactual test `X ⇒ ¬Y` refuted.
- **Tests first.** Marked contest → resistance set, kind unchanged where G1–G7
  decide it; unmarked contest → no resistance. `obwohl` adds one refuted-implication
  observation to the causes arena with the verse stamp.
- **Files.** `tekamolo.rs` (`GrammarHeuristics` gains a concessive row class),
  `wire.rs` (`Group::encounter` passes resistance).
- **Decision.** Concessive is not one of the four TEKAMOLO lanes. Options: a fifth
  lane in the heuristics table, or a separate marker table. Recommendation: a
  separate marker table, so the four lane addresses stay unchanged.

### D-CGW-6 — witness arcs across cohorts

- **Problem.** A cohort is 63 rows, so 196 of 741 probe rows started a new arc
  (`arc_breaks`). Whole books break arcs at every boundary.
- **Options to measure.** (a) the first row of a statement in a new cohort writes a
  back-link entry whose `spo_fact_ref` names the same facts-arena id, so the arc is
  continued by fact identity, not by slot; (b) cohorts per statement group instead
  of per reading position.
- **Tests first.** A statement seen in rows 10, 70, 130 walks back 130 → 70 → 10
  under the chosen option; `arc_breaks` = 0 for that statement.
- **Measure.** `arc_breaks` on the probe corpus and on a KJV book.

### D-CGW-7 — loci: 24 → 32 × i4 over the full 16-byte lane

- **Goal.** Tenant 14 reads `G24N4` (24 loci in the 12-byte payload of a 4+12
  facet, slots 16..23 reserved). With the classid already in the row key
  (D-LXC-29), all 16 bytes can hold 32 loci.
- **Needs.** `v3-envelope-auditor` review; `ENVELOPE_LAYOUT_VERSION` decision;
  EW64 OD-5 (`ew64-witness-unification-v1.md`).
- **Tests first.** Field isolation over 32 slots; slots 16..31 read 0 until set;
  the existing 16 named loci keep their offsets.

### D-CGW-8 — S/P/O as 16-byte identities (8 × u8:u8) instead of the u8 palette

- **Goal.** The frozen operand ABI caps a verse at 256 operands (8:8:8 in bits
  0..23). The row already has a 16-byte identity and a 16-byte edge facet; reading
  them as 8 rails of `u8:u8` (D-LXC-29 measured 8 rails: one tenant fits 99.41 % of
  KJV verses vs 95.66 % with 6) removes the cap.
- **Decision.** Whether the edge's 0..23 stays the palette (hot, 3 bytes) with the
  identity beside it, or the identity replaces it. Recommendation: keep 0..23 as the
  palette and resolve through the identity rails; changing 0..23 breaks every ISA
  reader.

### D-CGW-9 — bits 46..49 ownership (activation vs inference mantissa)

- **Goal.** Settle D-CGW-1's second half: either activation keeps 46..49 under the
  grammar binding and counterfactuals live only in their own lanes, or activation
  moves. Approved direction (2026-10-16 table, row 6): activation owns the bits for
  grammar-bound edges; the counterfactual lane is separate by design.
- **Tests first.** `deposit_counterfactual` on a grammar-bound edge is refused;
  on a mantissa-bound edge unchanged.

### D-CGW-10 — the clause-path → orientation correction downstream

- **Goal.** The board entry §7 table and the plan text that named 43..45 as the
  clause path are corrected (done in §11 of the entry); `ce64-semantic-upper-half.md`
  gains the grammar reading of 43..45 as a declared reading beside the triad.
- **Files.** `docs/architecture/ce64-semantic-upper-half.md` only.

### D-CGW-11 — Judicantis and multifunctional forms

- **Open.** Judicantis has no code (seven states named, one core-argument state).
  A form allowing several functions with no TEKAMOLO lane leaves 0 and the row
  unresolved. Measure how often on a real corpus with attested forms before
  deciding whether a context rule beyond Kausal → Purpose / Lokal → Direction is
  needed.

### D-CGW-12 — word → qualia index source (deferred)

- **Deferred by the operator.** Candidate: WordNet senses measured against the 219
  patches' Jina embeddings (`data/qualia/calibrated_219.bin1024`). Until then the
  caller supplies indices.

## 3. Order

D-CGW-1 → D-CGW-9 (same binding) → D-CGW-2 → D-CGW-3 → D-CGW-5 → D-CGW-4 →
D-CGW-6 → D-CGW-7 / D-CGW-8 (layout, need the envelope audit) → D-CGW-10 / 11.
D-CGW-12 waits for the operator.

## 4. Not in scope

- No new opcode, interpreter or executor; the probe's rules run as quack filters.
- No production write path for wire rows.
- No change to `contract::revision` (G1–G7 are frozen).
