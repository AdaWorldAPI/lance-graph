# Ledger revision, 2026-10-16

Maps the PRs since the last mapping (#1560–#1576) onto
`LEDGER-CORRECTION-2026-10-15.md` (merged in #1560). That file and the
2026-10-14 revision are not edited; this file supersedes the rows it names.
Board entry: `.claude/board/entries/2026-10-16-ledger-revision-after-1576.md`.

**Scope.** 15 merged PRs (#1560, #1562–#1571, #1573–#1576) and two closed
unmerged drafts (#1561 withdrawn, #1572 rejected). Read from the merged PR
bodies (fetched verbatim), the code diffs of #1562, #1565 and #1574, the board
entries each PR added, and the arc entries.

## 1. New closed ends

| id | what closed | evidence |
|---|---|---|
| C22 | O22, the first `CounterfactualVerdict` producer, including the flip direction (a cut that repairs an inconsistent chain is `Dispensable`) | #1560; `pearl.rs` tests `o22_*`; 6 disable runs red |
| C23 | R1 covers the whole verdict grid: all nine (effect × counterfactual) verdicts from `CognitiveWork`; none reopens the goal; the one accepted verdict moves on to `Evaluation`, which is not a goal revision | #1565 (strengthens the #1560 test) |
| C24 | Bits 59..60 are one field under two vocabularies (`TrustTexture` ≡ `CausalTopology`, identity mapping); contradiction is a pothole the MUL reacts to, not a cell | #1574 code-doc correction in `layout.rs`, `band_reading.rs`, `epistemic_state5.rs` |
| C25 | The global-wire ruling (R-b) holds on 0..23 too: the Haspelmath harness replaced a per-verse operand palette with one frozen, version-identified dictionary per epoch | #1570 |
| C26 | N15 groundwork read from source: no foreign system carries a 59..63 or a 40..42 (both are ours to write); abduction/induction argument order matches OpenNARS Java, ONA and PeTTa swap it; evidential-base width differs per engine | #1574 mapping inventory (pinned SHAs) |
| C27 | The stamp's place: an evidence lane beside the atom, never inside it, reserved `3` = `NO_PROVENANCE` | #1576 council (premise-auditor PREMISE-WRONG on "stamp as atom") |
| C28 | The coordinate machinery the O25 fix and K2 need exists: `VersionClock` maps a Lance version to the row positions it holds; the writer keeps it current | #1562 (contract), #1567 (writer) |
| C29 | A per-predicate guard on `hydrate` is not needed: a chain handed to `hydrate` is one relation's chain by construction, and `ChainStep.0` is not the predicate | #1572 rejected and reverted; closeout in its body |
| C30 | Board catch-up: arc entries for #1560, #1570, #1571, #1574, #1575, #1576; LATEST_STATE section | this PR |

## 2. Open items, revised (merged, deduplicated)

Fruit: **L** low (an afternoon, no decision needed), **H** high (a day or a
measured probe), **Heavy** (multi-PR or blocked on a decision). Tier is who
should do the grindwork.

| id | item | status after #1576 | fruit | tier |
|---|---|---|---|---|
| **O25 + K2** (merged) | Replay revises at every hop. Fix: compose along one trajectory; revise only across distinct evidence roots. The root is the `temporal.rs` coordinate, resolvable through `VersionClock` (C28) and carried as an evidence lane (C27). | OPEN, now fully specified. Two pinned tests in `chain_confidence.rs` flip with a recorded reason. D-HL-5 H5 (NAL overlap vs Gadamer gate, pre-registered in #1564) is the paired check. | H | O designs the root type, S implements, guarded-executor gate |
| **O30** (new, from C26) | Evidential-base width: `Stamp(u64)` bitset of ≤ 64 sources (planner), `&[u64]` roots (nal), ONA 10, OpenNARS 20,000. Pin the policy before revision runs over large populations. | OPEN (decision) | — | O |
| **K6 = N1 rest + decision 2** (merged) | What a derived conclusion writes into 53..63. #1564 pre-registered it as D-HL-5 H4 with candidate laws (a) 0, (b) `IndirectKnown × Open`, (c) weakest premise topology made indirect × `Open`, (d) `Unknown × Open` then hydrated, and a decision rule. | OPEN, experiment written, not run | H | S runs the lane, O reads the table, operator picks |
| **O29** (new, from #1565) | Should an accepted verdict move `CognitiveWork` on mid-action, or wait for the action to finish? | OPEN (decision) | — | O |
| **D-46** | One global meaning for 46..49: opcode (code) or activation (design). | unchanged | Heavy | O |
| **D-MATURATION3-1** | The pyramid as the plasticity ladder in 50..52: entry code (7 vs "new"), `learn`'s rule, migration gate for codes 1..6 (`learn`, `MooreNars16`, `CausalEdgeV3` byte 3 in lockstep). | unchanged | Heavy | O, then S behind a version gate |
| **ABL** (item 6) | Ablation through the N15 families. Its machinery now exists: D-ASM §7 "validate against the original as the oracle first" and the pinned upstream SHAs (C26). | OPEN; a `/coresearch` run is the next step | H | coresearch council |
| **N15 → D-ASM** (merged) | Pearl × NAL × others as views on one wire. Executable path: D-ASM-1..4 (binder, join as preimage, bag terminal, `LanceSpace` + three-way harness), ratified in #1576, nothing authorized. First step: measure the 26-script Hyperon baseline (`hyperon-probe/golden` is absent). | Queued (authorization) | Heavy | O authorizes; S builds W1 |
| **O23** | TruthOnly: a fourth verdict or a remap. Caveat unchanged: at the verdict level TruthOnly and "no cut" both read `NotRun`. | unchanged | — | O |
| **R2–R4** | Rubikon tests: agreement as `HorizonFusion` (needs "independent root for a person" decided), goal never revised by counting, folding only under the gate. | unchanged | L for R3/R4 once worded; R2 blocked | S |
| **O26** | `causal_bridge` Intentional → Causes, dead code behind a non-compiling `wip` feature. | unchanged (downgraded) | — | — |
| **O7, O13, O15** | model-identifier rule, phase/activation rule, Experiment A. | unchanged | — | O |
| **DOC-WIRE** (merged: README tension + #1574 leftovers) | (1) `docs/architecture/ce64-semantic-upper-half.md` lets each tenant declare its own reading of 43..52; R-b says one global meaning. (2) Stale prose still saying "two readings" on 59..60: `.claude/knowledge/causal-plane-inventory.md:55`, `docs/TYPE_DUPLICATION_MAP.md:15` ("Pearl-3 epistemic lens"), `.claude/plans/causaledge64-mailbox-rename-soa-v1.md:57` (`Murky/Dissonant … Contradiction`). | OPEN; (1) awaits a yes; (2) is mechanical | L | spec-editor (Haiku) per file, O reads each diff |
| **BAND-RETIRE** (new) | `band_reading` has no production caller (only the `ClassView::band_reading` trait method at `class_view.rs:1338` and two examples); `project_truth` still refuses the other vocabulary with `LensMismatch`, a gate on a distinction C24 removed. Either retire the 59..60 half of the declaration surface or keep it for 61..63 `ReasoningBand` only. | OPEN (small decision) | L/H | S after the decision |
| **N16** (new, from #1570) | A canonical, cross-system operand registry: `u16 WordId → u8` operand per sealed epoch. Until it exists every 8:8:8 operand dictionary is a fixture, and C25 is a rule without a registry. | OPEN | Heavy | O designs |
| **N17** (new, from #1575) | `ISS-RELALG-ABSORPTION-AFTER-DE-MORGAN`: `normalize` widens 2 of 2,000 programs (absorption missed after De Morgan). Answers stay correct; the cost is width and a refusal. | OPEN, reproducer in the H14d CSV | L/H | S with the two programs as fixtures |
| **N18** (new, from #1562/#1567) | The version clock is not wired into a reader; compaction and deletes break its prefix property (fragment-id remap); the clock is not seeded from storage on open; the 60-minute soak is outstanding. | OPEN | H | S |
| **N19** (new, from #1564/#1566–#1575) | Layout stream: no A/B/C layout verdict yet. Open costs: the merged point read (~21 overlay entries per row, 2.3 µs), page-bound masked reads (bytes saved only when survivors are page-local), the payload stage's fixed word scan. | OPEN, measured, no decision depends on it yet | Heavy | O |
| **N20** (new, from #1571) | `ISS-QUALIA-A2B-CLUT-UNMEASURED`: the device → PCS mapping for 12 of 17 axes; the Σ-codebook slot sharing between 219 named patches and 64 glyphs. | OPEN | H | S runs the chart probe |
| **N21** (new, from #1570) | The Haspelmath harness's tests were committed without a local run. | OPEN, verify | L | guarded-executor (Haiku) runs the five commands in the PR body |

## 3. Connecting the dots

- **K2' — one evidence root.** C27 (stamp as an evidence lane), C28 (version →
  row positions) and the O25 fix are one object: an evidence root is a
  (lane, row position) pair, resolvable as of any version, carried beside the
  atom. O30 (width) is the only decision left before it can be built. Once it
  exists, `nars::belief::Stamp`, `nal::Evidence.stamp`, contract
  `independent_roots` and the probes' roots key on it, replay stops pooling
  self-evidence, and D-HL-5 H5 can be read.
- **K6 — the derived packet is an experiment, not a debate.** #1564 already
  wrote the four candidate laws and the rule that picks between them. Running
  H4 answers decision 2 and the rest of N1 with numbers.
- **K7 — ablation, N15 and D-ASM are one stream.** The families the operator
  named for ablation are exactly D-ASM's oracles (Hyperon, PeTTa, ONA,
  OpenNARS), with the SHAs already pinned. A `/coresearch` run on "what does
  removing one piece of evidence mean in each family" uses that inventory
  rather than re-reading upstream.
- **K8 — the wire is now global in practice, not just in prose.** C24 (59..60),
  C25 (0..23) and R-b agree. What still says otherwise is documentation
  (DOC-WIRE) and one dormant gate (BAND-RETIRE).

## 4. Fruits

- **Low:** DOC-WIRE (2) — three stale lines; N21 — run five commands; N17 —
  two reproducers become tests, then the absorption rule; R3/R4 wording.
- **High:** O25 + K2' once O30 is decided; D-HL-5 H4 (K6); the ablation
  `/coresearch`; N18 reader wiring; N20 chart probe.
- **Heavy:** D-ASM W1 (authorization); D-MATURATION3-1 migration; D-46; N16
  registry; N19 layout verdict.

## 5. Decisions for the operator

| # | question | recommendation |
|---|---|---|
| O30 | evidential-base width | a bounded root list keyed on (lane, row position), width pinned per class, overflow refused rather than truncated |
| O29 | accepted verdict mid-action | wait for the action to finish; record the verdict, move at the boundary |
| DOC-WIRE (1) | amend the diagram doc to one global reading of 43..52 | yes, with the version-gate sentence kept |
| BAND-RETIRE | retire the 59..60 lens gate | keep the declaration for 61..63 only; drop `TruthLens` refusal |
| D-ASM | authorize W1 | yes for the Hyperon baseline measurement only |

## 6. Agent ledger (Haiku harvest tier)

| card | runs | items checked | wrong | missed |
|---|---|---|---|---|
| quote-harvester (PR bodies) | 1 | — | — | refused correctly: the card has no GitHub or Write tool |
| general-purpose on Haiku (PR bodies #1561–#1576) | 1 | 7 bodies read in full and used; titles/merged state cross-checked against `git log` | 0 | 0 |
| harvest-scout (band_reading callers, drift lines) | 1 | positives `class_view.rs:1338`, `band_reading.rs` sites; 3 zero-hit claims re-checked | 0 | **3** (patterns 5, 6, 8 present at the cited lines) |

Two misses on zero-hit searches over Markdown (one 2026-10-14, three here):
that kind of task goes back to Sonnet. Positive-hit harvesting on Haiku stays.
