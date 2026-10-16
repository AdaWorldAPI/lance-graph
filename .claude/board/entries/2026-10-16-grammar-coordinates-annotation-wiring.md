# 2026-10-16 — Grammar on coordinates: annotations, TEKAMOLO × Antecedent loci, AriGraph episodic vs epistemic, CE64 bits 42 / 52 / 59..63

**Status:** OPEN. A breadcrumb ledger for one design thread; it feeds the
loose ends of `2026-10-16-board-catchup-open-items.md` (#1589). It decides
nothing. Every claim cites the file it was read from (read in full, or through
a Haiku substrate map whose items were spot-checked against source); every
open point says what would close it.

## 0. What landed in this thread (local commits, not pushed)

| commit | what | measured (v4, tier printed by the program) |
|---|---|---|
| lance-graph `b0046b30` | `lance-graph-quack/examples/ruff_taint_probe.rs`: ruff_metta_spo's interpreter-only fixpoint recomputed as `Any(And(Plane(E), Plane(T_rep)))` per head extent | Hyperon 147 heads / PeTTa 1005 heads: 0 mismatches vs ruff; zero steps mismatch by exactly the call-tainted heads (9 / 71); dropping `T_rep` from the AND goes red |
| lance-graph `b4a35521` | `lance-graph-quack/examples/semantic_map_probe.rs`: Haspelmath (2003) grams as a gram × function plane over a function × function plane; contiguity = the same closure restricted to the gram's row | all 18 grams of Figs. 8.1/8.6/8.9/8.12 contiguous; (4b) broken by French à, (4c) by German zu; Turkish old present (23) caught; both disables red |
| ruff `4513c50af9` | `ruff_metta_spo/examples/metta_census.rs --dump` writes `heads.tsv` / `calls.tsv` | input of the taint probe |

**Prior art missed, recorded so it is not repeated:**
`crates/deepnsm-v2/src/haspelmath.rs` already implements Fig. 8.1 continuity
(`continuity`, `FormAttestation` with `evidence_root`, `MapVerdict::Incomplete`
for incomplete coverage) at the FSM → hydrate → CE64 seam. The quack probe is
the same check on planes and covers Figs. 8.6–8.12 and (4)/(23), which
haspelmath.rs does not; it has no `Incomplete` verdict. OPEN: fold one into the
other (haspelmath.rs is the in-tree home; the probe's extra maps become its
fixtures) — not done.

## 1. Datalog vs NAL vs ONA vs relalg as annotation readings of one fact

| reading | where it lives (read) | what it adds to a fact at a coordinate |
|---|---|---|
| relalg / Datalog | `lance-graph-relalg/src/ir.rs` (`Rel`, `Atom`, `Measure`; no string by const assert); plan `relalg-v1.md` D-RA-3 (rules as algebra values), D-RA-4 (bounded μ) | set membership only; no truth, no stamp. μ and joins are "not built" (`lib.rs` doc, plan §4) |
| NAL | `lance-graph-nal/src/lib.rs`: key = NodeGuid read as `nal-statement-v1` `[classid, S, copula, P, id, 0, 0]` (lib.rs:7); f/c are value lanes beside the key; `HORIZON = 1` (lib.rs:216), `MAX_CONFIDENCE_CODE = 252` (lib.rs:220); revision refuses evidence without a stamp, `NalError::MissingProvenance` (lib.rs:515, nal.rs:298) | truth (f, c) + evidential stamp, revision gated on independence |
| ONA | oracle only (rules: no NARS clone); the NAL crate's constants are named after it | external answers to compare against |
| MeTTa | ruff `ruff_metta_spo` (harvest + classify) | which heads lower, call graph |

Working model (not decided): one fact = one coordinate; each reading is an
**annotation plane or lane beside the key**, never a second store and never a
string. relalg supplies membership, NAL supplies (f, c, stamp) lanes, ONA and
Hyperon are oracles. Gap to watch: AriGraph's `TripletGraph::revise_with_evidence`
(`lance-graph/src/graph/arigraph/triplet_graph.rs:904`) revises on an exact
string match with **no stamp**, so it cannot tell independent evidence from a
repeat — the exact case the NAL crate refuses (and the rule "FirstOfRun is not
evidential independence").

## 2. TEKAMOLO × relativPronomen (Antecedent) — the 24 × i4 loci in DeepNSM-v2

Read in full: `deepnsm-v2/src/tekamolo.rs`, `deepnsm-v2/src/loci.rs`,
`lance-graph-contract/src/causal_witness.rs` (`Locus`), and the fsm.rs map.

- The register: `CausalWitnessFacet`, 12 bytes = 24 signed i4 loci (−8..+7
  event offsets); 16 are named (`causal_witness.rs:116`): Temporal 0, Kausal 1,
  Modal 2, Lokal 3, S/P/O meaning 4–6, **Antecedent 7** ("relativPronomen → its
  antecedent"), BasinAnchor 8, SupportedBy 9, Supports 10, RunbookEvidence 11,
  QualiaReference 12, MeaningLevel 13, Quorum 14, Contradiction 15. 8 slots are
  unnamed.
- Who writes loci today: only `loci.rs` (`loci_from_triples`), and only
  `Locus::Antecedent`, only for **pronoun subjects**, by recency or selectional
  fit, refusing thin margins and anything beyond −8 (a basin edge, not an
  overflow). A test pins that no other locus is ever written.
- **Gap G1 — relative-clause antecedent is computed but not exported.** The FSM's
  one-level relative-clause sub-machine holds the antecedent in a register
  (`fsm.rs:146` field `antecedent`, set at `fsm.rs:277`) and emits it inside the
  SPO triple (object-relative: `Spo::new(es, verb, antecedent)`, `fsm.rs:232`).
  The module doc calls it "the FSM-side feeder for the ±8 antecedent pointer in
  `wave`" (`fsm.rs:32-33`, `:62-63`), but no code writes `Locus::Antecedent`
  from it. Closing it: the parser reports, per emitted triple, the stream index
  of the triple whose subject is the antecedent; loci writes `-(i - j)` when it
  is within reach.
- **Gap G2 — TEKAMOLO lanes are filled, TEKAMOLO loci are not.** `tekamolo.rs`
  writes the `ValueTenant::Tekamolo` facet (`LaneAddress {relation, rank}`, what
  kind of circumstance); `LaneHypothesis.at` (`tekamolo.rs:162`) records the
  TOKEN index that stated it. Loci 0–3 are EVENT offsets in the triple stream
  (`loci.rs`: "one event per triple, at version = its stream index"). Unit
  mismatch: a token index is not an event offset, so `at` cannot be copied into
  a locus; it needs the token → triple-index map from the parse.
- The two meet: a relative clause ("der Mann, **der** gestern kam") carries
  both an Antecedent pointer and its own TEKAMOLO lanes; the circumstance
  belongs to the embedded event, the pointer to the matrix one. Which event a
  lane attaches to is open.

## 3. AriGraph episodic vs epistemic

- **Episodic** exists: `EpisodicMemory` (`episodic.rs:64`), episodes with
  `step`, observation text and `triplets: Vec<String>` (`episodic.rs:50`) —
  strings as content, against the no-strings rule; `EpisodicBasins::rail_rows`
  (`episodic.rs:215`) moves basins onto `ValueTenant::EpisodicBasin` (D-ACR-6);
  `episodic_search` = AriGraph Eq. 1.
- **Epistemic** has no AriGraph type. It lives in two places outside it: CE64
  bits 59..63 (`EpistemicState5`, `causal-edge/src/layout.rs:102-115`; meaning
  per `(classid, rail, generation)` in `lance-graph-contract/src/epistemic_state5.rs`)
  and the planner's read gate `EpistemicMode` (`lance-graph-planner/src/temporal.rs:77`,
  Strict / Aware / Retro by rung).
- Working model: episodic = WHEN and IN WHICH EXPERIENCE (versions, steps,
  basins); epistemic = HOW WELL KNOWN (topology × certification). Neither should
  be inferred from the other. OPEN: AriGraph triplets carry `TruthValue` f32 and
  a `timestamp`, but no epistemic state and no stamp.

## 4. CE64 bits 42, 52, 59..63 (`causal-edge/src/layout.rs`, `pearl.rs`, `plasticity.rs`, `isa.rs`)

| bits | field (code) | what the named bit means |
|---|---|---|
| 40..42 | Pearl `CausalMask`, 3 plane bits | **bit 42 = the S plane bit** (`CausalMask::S = 0b100`): set in S, SO (association), SP (confounder detection), SPO (counterfactual); clear in O, P, PO (intervention) |
| 50..52 | Plasticity, per plane hot/frozen (`isa.rs:360` `Field::Plasticity`) | **bit 52 = O-plane hot** (`PlasticityState::O_HOT = 0b100`) |
| 59..63 | `EpistemicState5`, one 5-bit field = topology (59-60) | certification << 2 (61-63) | D-EPI-CANON-0 |

- **Two readings of 50..52 disagree.** The code names them plasticity; session
  rule S7 (2026-10-12) records them as carrying the Staunen/witness ladder plus
  a free-energy follow-up signal. #1589's N1 adds: `syllogize` writes bits 50..52
  as `0b111` and zeroes witness and epistemic bits. OPEN: which reading is the
  contract; until decided, nothing new writes 50..52.
- `haspelmath.rs:10-13` keeps CE64 bits 24..63 at ZERO for grammar-only edges
  ("syntax is not NARS confidence, Pearl causation, evidence independence or
  epistemic maturity"). Any wiring of annotations into CE64 must keep that
  boundary: a grammatical fit never sets Pearl, plasticity or epistemic bits.

## 5. Open decisions (for the operator)

| # | Decision | Recommendation | Why | Caveats | Needs from you |
|---|---|---|---|---|---|
| 1 | Annotation shape | lanes/planes beside the key, one per reading | one fact, many readings, no second store | relalg has no truth; NAL needs stamps the parser does not produce | OK on the model |
| 2 | G1 (relative antecedent → `Locus::Antecedent`) | build first, test-first, in deepnsm-v2 | the value is already computed; only the export is missing | needs triple-index provenance from `parse_readings` | go / no-go |
| 3 | G2 (TEKAMOLO → loci 0..3) | after G1, with the token → event map | lanes say what, loci say where | unit mismatch token vs event | go / no-go |
| 4 | CE64 bits 50..52 | settle the contract before any writer | code and rule S7 disagree; N1 shows a writer already clobbers it | — | ruling |
| 5 | Meaning of "42 52" in this request | treat as S-plane presence (42) and O-plane plasticity (52) per code | that is what the code says | may have meant a range | confirm |
| 6 | Haspelmath duplication | fold the quack probe's maps into `haspelmath.rs` fixtures | one home | haspelmath.rs is deepnsm-v2, the probe is quack | go / no-go |
| 7 | AriGraph revision without stamps | route revision through the NAL stamp gate | independence is otherwise unknown | AriGraph triplets are strings | go / no-go |

## 6. Not searched

ONA source (oracle only, not opened here); Hyperon's own grammar of
contexts beyond ruff's harvest; the 8 unnamed loci's intended use; any plan
that may already assign them (grep `Locus` in `.claude/plans` not run).

## 7. Wiring landed (2026-10-16, later the same day)

`crates/deepnsm-v2/src/wire.rs` fills every field of the CE64 semantic
upper half on each parsed triple, each from one named source (module doc
has the table). `run_harness` is unchanged and still emits the grammar-only
edge; `wire` is the annotation layer above it.

| bits | field | source |
|---|---|---|
| 24..39 | F/C | `lance-graph-arm-discovery::arm_to_truth_u8`, rule `(S,P) → O` over independent verses, k = 1 |
| 40..42 | Pearl | TEKAMOLO Kausal lane on the event → `PO`, else `SO` |
| 43..45 | orientation | clause path (`fsm::parse_to_spo_traced`): 1 Main, 2 Carried, 3 SubjRelative, 4 ObjRelative, 0 untraced |
| 46..49 | activation | pressure `pos − neg`, clamped i4 (`CausalEdge64::activation`, same bits as `inference_mantissa`) |
| 50..52 | novelty → wisdom | per-statement Markov rung, 1 Staunen ..= 7 wisdom (`novelty_wisdom`, same bits as plasticity) |
| 53..58 | belief update | W1/W2/W3 window of the last three updates: 1 corroborate, 2 contest, 3 dependent |
| 59..63 | epistemic | topology from the clause path; Associated → Related (≥2 independent, pressure > 0) → CausalCandidate (≥2 independent Kausal verses); never Causes |

Closes G1 (relative antecedent → `Locus::Antecedent`) and G2 (TEKAMOLO →
loci 0..3, via `ClauseRead::at` and the triple's `closed_at`).

Tests: 10 in `wire`, each guard disable-verified red on exactly its named
test (rival reset, dependent kind, resolved gate, pressure gate, two causal
verses, Kausal → PO, orientation code, antecedent locus, contest shift, map
link). `f_and_c_against_nal_revision_of_units` pins that arm `f` equals NAL
revision of unit evidence, and arm `c` equals it only when uncontested.

Reasoning AST: `lance-graph-quack/examples/wired_reasoning_probe.rs` states
five field rules as Datalog (R1 `pearl_po ↔ kausal`, R2 `cert_rel ↔
support2 ∧ press_pos`, R3 `candidate → cert_rel`, R4 `climbed → support2`,
R5 `direct ↔ main`) and runs each as a quack filter over presence planes of
541 wired rows: 0 violations, counts equal to a row oracle, every plane
non-trivial, one flipped output bit breaks exactly its rule.

Not done, recorded so it is not lost:

- The orientation map's contiguity check has no consumer: over the two
  relative functions it is always true, so a per-relativizer probe would be
  vacuous.
- `deepnsm_v2::belief::BeliefArena` and `lance-graph-nal` do not consume
  wired rows yet. The arena's `Stamp` folds sources modulo 64 (false
  overlap past 64 verses); `lance-graph-nal` has no relational copula for
  `(S, verb, O)` (R10).
- AriGraph `revise_with_evidence` still revises without stamps.
- The 144 verbs (`contract/src/grammar/verb_table.rs`, 12×12 in a 16×16
  space with TEKAMOLO slot priors) are not connected to the P code or the
  Pearl rung.

## 8. Kausal claims through the belief arena; qualia separation measured

- `wire` now returns `Wired { rows, report, facts, causes, events }`. Every
  row is a unit observation `S –Rel(P)→ O` in a facts `BeliefArena`; every
  `weil` claim a unit observation `cause ⇒ effect` (event terms) in a causes
  arena, both stamped `Stamp::source(verse)`. `CausalCandidate` now requires
  the implication to rest on ≥2 disjoint stamps; the per-statement
  causal-verse counter is gone. Five disables red (causes stamp, facts stamp,
  two sources, no self-implication, cause = stating event); the two-sources
  test was vacuous until its fixture put the plain verse first.
- Reasoning probe on 741 rows: 0 violations on R1–R5, 72 candidates, 201
  implications, 138 pooled.
- Qualia palette (#1571, 219 patches, PCS hundredths): nearest-neighbour L1
  min 25, p10 50, median 90, max 250; no duplicates; 95 / 219 nearest
  neighbours in the same family. So any PCS reading within L1 12 of a patch
  decodes to that patch uniquely. A "3σ" statement needs the noise σ of a
  reading, which is the unmeasured A2B side (`ISS-QUALIA-A2B-CLUT-UNMEASURED`).

## 9. Qualia register (approved proposals #1, #2, #4)

- `WiredRow::qualia` is a `FacetCascade` (class id 0) read `G6D2`, six
  `u8:u8` pairs: 0..2 = `(operand code : qualia index)` for S, P, O; pair 3 =
  `(verb_table::morton_cell : 1)`, or `(0 : 0)` when the verb does not
  classify. An adjective's index lands on its noun's S/O pair, an adverb's
  on its event's P pair: one index, two meanings, told apart by the pair.
- Deviation from the proposal, found while building: `morton_cell(Becomes,
  Present) == 0`, a real verb, so the cell cannot share a pair whose zero
  means "absent", and a reserved cell may not be borrowed as a sentinel. The
  cell got its own pair with a presence byte.
- `saw` stays unclassified (`verb_lexicon` refuses the homograph); German
  surfaces never classify (the lexicon is English).
- Five disables red. Two tests were vacuous on their first fixture and were
  fixed until the disable bit: the single-implication case (plain verse
  first) and the sentence-bound adjective (only bites when the next
  sentence's noun also occurs in the current sentence's triples).
- OPEN (#3, deferred by the operator): the word → qualia index source.
  Candidate for later: WordNet senses measured against the 219 patches'
  Jina embeddings (`data/qualia/calibrated_219.bin1024`). Until then the
  caller supplies indices and every index defaults to 0.

## 10. W is the Markov arc; activation is what the update caused (`e31cd087`)

- W (53..58) indexes a per-cohort `WitnessTable<64>` (63 rows per cohort,
  slot 0 = no prior). Each row's entry is `(verse + 1, facts-arena id)`; its
  W points at the entry of the PRIOR state of the same statement, so walking
  W from the newest row visits that statement's verses newest to oldest and
  skips rival rows. A prior in an earlier cohort starts a new arc
  (`arc_breaks`). The earlier 3-slot window (`shift_window`) is gone: it
  ignored the existing `WitnessTable` contract.
- Activation (46..49) = the change of the statement's arm frequency over the
  prior state, in eighths (`128` = unknown prior). A repeat of a settled
  belief activates 0. The ladder and certification read the support balance
  (`pos > neg`) instead.
- Tests: `w_walks_the_statement_arc_backwards`,
  `a_cohort_boundary_breaks_the_arc`, plus re-pinned activation values. Four
  disables red (activation as pos − neg, cohort check removed, balance
  always true, W = previous row).
- Probe R2 now reads `2·support > antecedent`; 741 rows, 0 violations.
- OPEN: bits 46..49 are also `InferenceType`'s mantissa, and
  `contract::counterfactual::deposit_counterfactual` writes −6 there to mark
  the road not taken. Activation and that marker cannot share the bits.
