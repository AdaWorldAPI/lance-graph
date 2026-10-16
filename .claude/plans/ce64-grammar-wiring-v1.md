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
before D-CGW-1.** Under the contract rule (D-CGW-1) this is not a reading mismatch
to bind around: these are contract violations by `wire.rs`, to be fixed by
amending the contract (43..45) or by `wire.rs` conforming (46..49).

## 2. Work items

Each item lists: goal; tests to write first (can fire + stays silent, non-trivial
inputs); files; dependencies; the decision it needs, if any.

### D-CGW-1 — one global meaning per CE64 field, held in the contract

> ⊘ **Superseded 2026-10-16, same day:** the first version of this item was a
> per-tenant "versioned reading binding" for 43..45 and 46..49. That is wrong.
> CE64 is an ISA wire format, like TCP/IP or BF16: every field has one meaning
> everywhere, fixed by the contract, and nothing interprets it per tenant or per
> edge. The mantissa bits are law. Like GMT, the edge is the fixed point every
> reading is measured against, so it carries no qualia of its own: qualia and
> other readings sit beside it (the row's qualia register), never inside it.

- **Rule.** The meaning of each field lives once, in the `causal-edge` contract
  (`layout.rs`, the `edge.rs` accessors, and `docs/architecture/ce64-semantic-upper-half.md`).
  Every writer conforms to it, `wire.rs` included. A field changes meaning only by
  amending the contract once, with every reader migrated in the same change
  (`I-LEGACY-API-FEATURE-GATED`).
- **Where `wire.rs` breaks it today.**
  - 43..45: the contract says S/P/O pathology triad; `wire.rs` writes the
    Haspelmath function. The operator ruled 43..45 = orientation / syntax
    (Haspelmath for text). So the contract is amended to orientation, and the
    triad's readers (`*_pathological`, `concern_level`, `network.rs` Simpson
    check, `compose`) migrate or move their signal elsewhere.
  - 46..49: the contract says inference mantissa, which the ISA executes
    (+6 Intervention, −6 Counterfactual, NARS rule index). `wire.rs` writes
    activation there. Because the mantissa is law, activation cannot share these
    bits; see D-CGW-9.
- **Tests first.** For each amended field: the contract accessor, a writer and
  every migrated reader agree on one table of codes; the field-isolation matrix
  over 43..49; a reader of the old meaning that is not migrated fails to compile
  (the accessor is renamed), never silently misreads.
- **Files.** `causal-edge/src/{layout.rs,edge.rs,network.rs,v2_layout_tests.rs}`,
  the readers listed in §1, `ce64-semantic-upper-half.md`.
- **Open.** Moore for non-verbal Cartesian data and Haspelmath for text are two
  code tables for 43..45. Under one global meaning they must be one table, or
  orientation needs a code space both share.

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

### D-CGW-9 — bits 46..49: the inference mantissa is law; where does activation go?

- **Contract.** 46..49 is the inference mantissa the ISA executes. It keeps that
  meaning globally (D-CGW-1).
- **Consequence for `wire.rs`.** Writing activation into 46..49 contradicts the
  contract. Until the contract gives activation a field, `wire.rs` must write the
  mantissa (or nothing) there and keep activation outside the edge (the row, or a
  lane beside it).
- **Supersedes** the row-6 direction of the 2026-10-16 table ("activation owns the
  bits for grammar-bound edges"), which assumed per-tenant readings.
- **Tests first.** Every edge `wire.rs` emits decodes through the ISA mantissa
  decoder as the inference it states; activation is read from its own home and
  equals the frequency change.
- **Decision.** Where activation lives: a row field beside the edge, or an amended
  contract field. Recommendation: a row field until the contract is amended,
  because no 4-bit field of the edge is free.

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

### D-CGW-13 — every oracle reads out on the same CE64 fields

- **Why now.** Because each CE64 field has one global meaning (D-CGW-1), the
  answers of independent reasoners can be compared on the same register: F and C
  (24..39), the Pearl question (40..42), orientation (43..45), the inference
  mantissa (46..49) and the certification ladder (59..63). This is the readout of
  the existing oracle plan `d-oracle-fidelity-matrix-v1.md` (D-ORACLE), not a new
  harness: Java OpenNARS (Pei Wang's lineage, `opennars/opennars`), ONA, MeTTa
  (Hyperon and PeTTa, golden corpus #48), OpenCog (temporary oracle only, entry
  `2026-10-16-opencog-oracle-scope.md`) and MORK (D-ASM-10).
- **Tests first.** One fixture, all oracles, each native answer recorded
  verbatim; then a per-oracle projection onto the CE64 fields, where a field the
  oracle has no notion of is written as "not comparable", never as 0. Falsifier:
  two oracles that disagree in their native output must not project to equal
  edges unless the disagreement lies in a field marked not comparable.
- **Fences.** Native outputs stay separate (D-ORACLE F9: non-parity does not mean
  one is wrong). Oracle code runs outside the workspace; only inputs we generated
  and recorded answers are committed. No blended score.
- **Needs.** D-CGW-1 (the contract amendment), D-ORACLE runners pinned.

### D-CGW-14 — OpenPSI motivation as a qualia device (#1571)

- **Shape.** OpenPSI is another *device* in the ICC sense of
  `contract::qualia_profile` (#1571, D-QPROF-0), beside the engine telemetry and
  the Ekman observer: its modulators (activation/arousal, resolution level,
  selection threshold, securing rate) get a device → PCS profile, measured, not
  declared. Arousal and resolution meet the PCS axes `arousal`,
  `resolution_hunger` and `clarity` by name; every other mapping is fitted from
  probe pairs, like the A2B CLUT the module says it does not ship yet.
- **What does not move.** The edge carries no qualia (D-CGW-1: the edge is the
  fixed point). OpenPSI *demands* are goals, not felt states: they are never
  revised by counting and stay with the Rubikon planning/evaluation path
  (revision invariants G1–G7); only the modulators enter the qualia profile.
  D-ORACLE row D already marks the OpenPSI ↔ Rubikon correspondence as a
  hypothesis.
- **Tests first.** Drive one OpenPSI scenario as an oracle (outside the
  workspace), record its modulator trace, fit the shared-axis transform, and
  check it with a held-out trace; a guard that maps every trace to the same
  palette index fails.
- **Source.** Written from the published PSI/MicroPsi model and OpenPSI's
  observed behaviour, never from its source.

## 3. Order

D-CGW-1 → D-CGW-9 (same contract amendment) → D-CGW-2 → D-CGW-3 → D-CGW-5 → D-CGW-4 →
D-CGW-6 → D-CGW-7 / D-CGW-8 (layout, need the envelope audit) → D-CGW-10 / 11.
D-CGW-12 waits for the operator. D-CGW-13 follows D-CGW-1; D-CGW-14 follows
D-CGW-13's OpenPSI runner.

## 4. Not in scope

- No new opcode, interpreter or executor; the probe's rules run as quack filters.
- No production write path for wire rows.
- No change to `contract::revision` (G1–G7 are frozen).
