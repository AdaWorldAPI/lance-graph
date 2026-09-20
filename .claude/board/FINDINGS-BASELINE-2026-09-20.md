# Findings baseline — 181bb2c28005 (`EPIPHANIES.md`, post-2026-08-06 entries)

> **What this is.** The ONE historical catch-up over the findings that went
> into the `EPIPHANIES.md` monolith after the 2026-08-06 split watermark.
> It is a consolidated current-state checkpoint — a **k-frame**. After it,
> routine closeout is DELTA ONLY and never censuses the monolith again.
> Like `PLAN-INVENTORY-2026-09-07.md` it mints **no D-ids**, so
> `supersession_index.py` and `plan_dids.py` do not see it — by design.
> A FROZEN snapshot, not a live artifact: the classifier that produced it
> was the instrument for this one historical pass and is retired. Git holds
> it at `2374b1d` if the measurement ever needs reproducing.
>
> **The historical prose is FROZEN, not reconciled away.** `EPIPHANIES.md`
> is untouched: nothing was migrated, re-split, deleted or rewritten, and
> no entry files were created for these findings. Frozen means *not reread
> by routine closeout*; it does NOT mean adjudicated — 228 of 306 were not.
>
> **PROCESSED_THROUGH_SHA = `181bb2c28005c185a967edd61367008c6722eb5c`** — every eligible finding visible
> through that source revision is consumed into this baseline. The marker
> names the CONSUMED INPUT, never this file's own commit: a commit cannot
> contain its own hash. Machine-readable: `.claude/board/PROCESSED_THROUGH`.

---

## 0. The numbers

| | count |
|---|---|
| population (level-2 post-watermark entries with an E-id) | **306** |
| OPEN | 39 |
| CLOSED | 34 |
| SUPERSEDED | 5 |
| AMBIGUOUS | 228 |

Excluded and counted so the exclusion is visible, not asserted: **3**
level-2 bare date-group headers (no E-id) and **3** level-3
sub-headings (sections *inside* an entry). 306 + 3 + 3 =
312 dated headings at or after the watermark.

### Mechanical reachability — the union of join keys

A first estimate put the ceiling at 198 by taking `306 − 108 without a D-id
or PR`. That was wrong: the E-id → board joins are **independent keys** and
most of them land inside that 108.

| join key | entries | answers |
|---|---|---|
| `eid_board` | 192 | named on a board surface — provenance |
| `did_statusboard` | 101 | a referenced D-id has a STATUS_BOARD row — deliverable status |
| `pr` | 128 | a PR is referenced — landing evidence ONLY |
| `cite_live` | 48 | a cited path still exists — implementation reality |
| `did_unknown` | 12 | referenced D-id has NO status-bearing board row — dangling |
| `cite_dead` | 8 | cited path is GONE — stale citation |

**278** entries carry ≥ 1 usable key; **27** carry none and are
therefore automatically AMBIGUOUS. The other 201 ambiguous rows are
ambiguous for a different and more interesting reason — §1.

## 1. Why AMBIGUOUS is the largest bucket

**303 of the 306 entries carry their own `Status:` line, and 295 of those lead
with an EPISTEMIC GRADE rather than a work status:**

| leading token | entries |
|---|---|
| `FINDING` | 204 |
| `RULING` | 31 |
| `OPERATOR` | 15 |
| `CORRECTION` | 11 |
| `MEASURED` | 6 |
| `OPERATOR-RULED` | 5 |
| `⊘` | 5 |
| `SHIPPED` | 4 |
| `PROPOSAL` | 4 |
| `FENCE` | 2 |

`FINDING`, `RULING`, `CORRECTION`, `MEASURED` answer *how well established
is this claim*. They do not answer *is the work done*. Only **8** entries
lead with a work-shaped token.

That is the substantive result: **post-watermark `EPIPHANIES.md` was being
used as a findings log, not a deliverable tracker.** For most rows
OPEN/CLOSED is the wrong axis — the live question is *is this still true?*,
which no join answers mechanically. Reading their prose to manufacture a
status is what this pass was told not to do, so they stay AMBIGUOUS with
their grade recorded.

Not a comparable number: `PLAN-INVENTORY-2026-09-07.md` reached 40/211
ambiguous **with a human read of every status line in context**, and records
that naive substring matching produced ≥ 6 false positives in its corpus.
This pass is mechanical-only by instruction; the larger residue is the price
of that, not a worse measurement.

## 2. How to read a verdict

Vocabulary reused from `PLAN-INVENTORY`; nothing new minted. Each join
answers only the question it can answer:

| evidence | role | may decide status? |
|---|---|---|
| STATUS_BOARD D-id row | deliverable status | **yes** |
| ISSUES section | unresolved / resolved | **yes** |
| TECH_DEBT section | implementation debt | **yes** (OPEN) |
| the entry's own status line | only if its leading token is work-shaped | **yes** |
| INTEGRATION_PLANS | integration ownership | no — context |
| PR state | landing evidence | **no — MERGED ≠ CLOSED** |
| live code citation | implementation reality | no |
| `entries/`, `LATEST_STATE` mention | provenance | no |

Conflicting decisive evidence ⇒ **AMBIGUOUS** (23 rows), never averaged
into certainty. Absent decisive evidence ⇒ **AMBIGUOUS** (178 joined rows +
27 unjoinable). The *implementation* column carries landing and
code-liveness facts precisely so they cannot be mistaken for closure.

Three traps the tool encodes, each measured: STATUS_BOARD's status column
is **per-table** (28 schemas, index 1..6, absent in two); a `Status:` line's
**leading token only** is read; and cross-supersession needs **directional**
phrasing, because a bare `⊘`-proximity rule read `caveat (⊘ in E-FOO-1)` —
a sibling citing this entry's caveat — as the sibling superseding it.

## 3. The rows

### OPEN (39)

| source id | status | implementation | outcome/open point |
|---|---|---|---|
| `E-THE-GOLDEN-STEP-IS-THE-WRONG-STEP-AT-SMALL-Q-1` | OPEN | PR #932 merged | live ISSUES entry |
| `E-A-TOTAL-FUNCTION-THAT-CANNOT-REFUSE-IS-A-CORRUPTION-PATH-1` | OPEN | PR #948 merged | STATUS_BOARD row not done |
| `E-CONTRACT-INFERENCETYPE-INVERTS-THE-COUNTERFACTUAL-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-CAPABILITY-IS-NOT-REACHABILITY-1` | OPEN | PR #971 merged | live TECH_DEBT entry |
| `E-THE-FILTER-WAS-FILTERING-ON-THE-WRONG-PREDICATE-1` | OPEN | PR #971 merged | live TECH_DEBT entry |
| `E-THE-RECIPE-SURFACE-IS-CAUSALLY-BLIND-1` | OPEN | no implementation evidence | live TECH_DEBT entry |
| `E-A-DOC-COMMENT-CAN-GIVE-THE-WRONG-REASON-FOR-A-CORRECT-GUARD-1` | OPEN | no implementation evidence | live TECH_DEBT entry |
| `E-A-CORRECTION-CAN-SUBSTITUTE-ONE-WRONG-NOUN-FOR-ANOTHER-1` | OPEN | PR #1112 merged | STATUS_BOARD row not done |
| `E-ONLY-TWO-OF-FOUR-STANCES-MAY-CUT-A-CANDIDATE-SET-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-BLW5-FIRST-MEASUREMENT-1` | OPEN | 1 cited path(s) live | STATUS_BOARD row not done |
| `E-EVERYTHING-WIRES-TO-SOA-V3-CE64-IS-ALU-LEGACY-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-NXG-10` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-NXG-11` | OPEN | PR #1160 merged | STATUS_BOARD row not done |
| `E-NXG-16` | OPEN | no implementation evidence | STATUS_BOARD row not done; live TECH_DEBT entry |
| `E-NXG-5` | OPEN | no implementation evidence | STATUS_BOARD row not done; own status line: PROPOSAL |
| `E-NXG-6` | OPEN | no implementation evidence | STATUS_BOARD row not done; own status line: PROPOSAL |
| `E-NXG-9` | OPEN | PR #1153, #1154 merged | own status line: PROPOSAL |
| `E-AN-EXCLUDED-CRATE-ON-AN-X86-ONLY-FLEET-IS-CODE-NO-CI-HAS-EVER-COMPILED-1` | OPEN | PR #146, #844, #1194 +2 merged; 1 cited path(s) live | live ISSUES entry |
| `E-A-CONSUMER-THAT-OPENS-A-DATASET-HAS-ALREADY-LOST-1` | OPEN | PR #879, #911, #912 merged | STATUS_BOARD row not done |
| `E-A-DYNAMIC-DOMAIN-MASK-IS-A-SECOND-WITNESS-AND-ITS-ALIGNMENT-IS-CALIBRATION-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-LANCE-GRAPH-OWNS-THE-AGNOSTIC-THINKING-CONSUMERS-BIND-DOMAIN-1` | OPEN | PR #1220 merged | STATUS_BOARD row not done |
| `E-SPOG-IS-FOUNDRY-WITH-AN-ABI-SHAPED-SUBSTRATE-1` | OPEN | 1 PR ref(s), state not cached | STATUS_BOARD row not done |
| `E-T1-HAS-TWO-SIBLING-ALGEBRAS-THE-AXIS-IS-SYNTAX-VS-EXECUTION-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1` | OPEN | PR #1220 merged | STATUS_BOARD row not done |
| `E-A-CHECK-THAT-CANNOT-RUN-IS-INDISTINGUISHABLE-FROM-A-CHECK-THAT-PASSES-1` | OPEN | PR #1190, #1235 merged | live ISSUES entry |
| `E-POPCOUNT-FINDS-ELEPHANT-WHALE-BECAUSE-IT-IS-POSITION-BLIND-THE-TREES-METRIC-IS-LZCNT-AND-THE-BOARD-ALREADY-FILED-IT-1` | OPEN | no implementation evidence | STATUS_BOARD row not done; live ISSUES entry |
| `E-POPCOUNTS-UPPER-RANGE-SIMILARITY-IS-THE-HEXAGONS-RAUMGEWINN-AND-BOARD-GAMES-MAKE-IT-FALSIFIABLE-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-RAUMGEWINN-NEEDS-A-HORIZON-SMALLER-THAN-THE-BOARD-TIC-TAC-TOE-HAS-NONE-SO-ARM-1-IS-F0-DEGENERATE-NOT-A-KILL-1` | OPEN | 3 cited path(s) live | STATUS_BOARD row not done |
| `E-THE-ACCUMULATOR-GATE-OUTRANKED-THE-PLANE-AND-SILENTLY-DROPPED-IT-1` | OPEN | PR #1235 merged | live ISSUES entry |
| `E-THE-NET-ARM-RANKED-ON-A-PARTIAL-SUM-AND-ITS-ONLY-APPARENT-SIGNAL-WAS-THAT-BUG-1` | OPEN | 2 cited path(s) live | STATUS_BOARD row not done |
| `E-THREE-CARRIERS-THREE-FOLDS-1` | OPEN | PR #1244 merged; 1 cited path(s) live | live ISSUES entry |
| `E-A-MASK-EXPRESSION-DOES-NOT-IMPLY-A-BITMAP-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-A-VARNODE-IS-NOT-A-BUFFER-R2IL-IS-MICROCODE-FOR-MASKED-THINKING-1` | OPEN | no implementation evidence | live ISSUES entry |
| `E-DO-NOT-BACK-DATE-A-NEW-LAW-ONTO-AN-OLD-DOCTRINE-1` | OPEN | no implementation evidence | live ISSUES entry |
| `E-FOLD-AND-MASK-ARE-SIBLING-PHYSICAL-PLANS-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-LAYER-0-IS-T1-AND-MASK-RISC-IS-ALREADY-ITS-ISA-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-THE-CENTER-IS-SPOG-PLUS-FC-EVERYTHING-ELSE-IS-CAST-1` | OPEN | no implementation evidence | STATUS_BOARD row not done |
| `E-THREE-CONVERGENCES-TARSKI-SHANNON-JC-AND-THE-BAND-IS-A-SANDBOX-1` | OPEN | 1 referenced PR(s) closed unmerged; 1 cited path(s) live | STATUS_BOARD row not done |
| `E-WE-THINK-WITH-OGAR-GRAPHS-OGAR-DOES-NOT-DO-THE-THINKING-1` | OPEN | no implementation evidence | live ISSUES entry |

### CLOSED (34)

| source id | status | implementation | outcome/open point |
|---|---|---|---|
| `E-A-HORSE-RACE-IS-NOT-A-CROSS-SWAP-1` | CLOSED | PR #927, #928, #930 +2 merged | STATUS_BOARD row done |
| `E-THE-HYPOTHESIS-REFUTED-CLEANLY-AND-REVERSED-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-THE-METRIC-THAT-SEPARATES-ONE-COMPARISON-IS-BLIND-TO-ANOTHER-1` | CLOSED | PR #926 merged | STATUS_BOARD row done |
| `E-PIN-LANCE9-LANCEDB033-DF541-ARROW58-NO-DF53-1` | CLOSED | PR #879, #911, #912 +1 merged | ISSUES entry resolved |
| `E-ATTENTION-MASK-IS-A-RENAME-REGISTER-FILE-NOT-A-RESIDUE-CARRIER-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-FROM-V1-DROPS-PROVENANCE-AND-THE-COUNCIL-CAUGHT-THE-CONTRACT-ABOUT-TO-TRUST-IT-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-THE-ATTENTION-ATOM-WAS-ALREADY-SHIPPED-WHAT-WAS-MISSING-WAS-A-COMPOSITION-THAT-IS-NOT-OR-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-A-DOC-PRECEDENCE-CLAIM-CAN-PASS-EIGHT-GREEN-TESTS-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-TWO-KEY-ELEVATION-WINDOW-IS-NARROW-AND-THE-CORPUS-STRADDLES-IT-1` | CLOSED | PR #997, #998 merged | STATUS_BOARD row done |
| `E-THE-FUSED-PAYLOAD-IS-INERT-AT-EVERY-EXECUTION-GATE-THAT-CONSUMES-IT-1` | CLOSED | PR #1045 merged | STATUS_BOARD row done |
| `E-BELIEF-ARENA-DEDUP-IS-PAYABLE-AND-W0-MUST-CITE-IT-1` | CLOSED | PR #1078 merged; 1 referenced PR(s) closed unmerged | STATUS_BOARD row done |
| `E-THE-ORACLE-WAS-CITED-AS-A-PHILOSOPHY-AND-NEVER-AS-A-METHOD-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-A-DETERMINISM-GATE-IS-TRIVIALLY-SATISFIED-BY-A-KERNEL-THAT-DOES-NOTHING-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-A-WITNESS-THAT-DROPS-THE-RELATION-IS-NOT-A-WITNESS-1` | CLOSED | PR #1120 merged | STATUS_BOARD row done |
| `E-EVERY-DEFECT-IN-A-MEASUREMENT-WAS-IN-ITS-FIXTURE-NOT-ITS-CODE-1` | CLOSED | PR #1118 merged | STATUS_BOARD row done |
| `E-PILLAR-11-GREEN-FOR-LATTICE-WALKS-LENGTH-PARAMETERIZED-1` | CLOSED | PR #1129, #1133 merged | own status line: SHIPPED |
| `E-QUALIA-IS-RANK-INERT-AT-THE-FRONTIER-AND-POPULATION-LOSES-TO-COUNTING-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-THE-CALIBRATION-GATE-REVERSED-THE-DECLARED-FLOOR-1` | CLOSED | 1 cited path(s) live | STATUS_BOARD row done |
| `E-A-DOCUMENTED-MODULE-THAT-WAS-NEVER` | CLOSED | 3 cited path(s) live | own status line: SHIPPED |
| `E-A-PRODUCER-IS-A-PURE-FUNCTION-OF-THE-CONTENT-LOCI-1` | CLOSED | no implementation evidence | STATUS_BOARD row done; own status line: SHIPPED |
| `E-THE-VACANCY-RULE-IS-NOT-ABOUT-ENTROPY-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-TWO-FATE-PROBES-KILL-DIFFERENT-WAYS-1` | CLOSED | PR #1144 merged | STATUS_BOARD row done |
| `E-A-SWEEP-IS-COMPLETE-ONLY-WITHIN-THE-TARGET-KINDS-ITS-GATE-COMPILES-1` | CLOSED | PR #1194 merged | TECH_DEBT entry resolved |
| `E-NXG-18` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-NXG-2` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-NXG-21` | CLOSED | 1 cited path(s) live | STATUS_BOARD row done; own status line: SHIPPED |
| `E-NXG-22` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-SEVEN-HARVEST-SOURCES-ONE-OBJECT-THE-VERSION-KEYED-MASK-SET-1` | CLOSED | PR #1218 merged; 1 referenced PR(s) closed unmerged; 1 cited path(s) live | STATUS_BOARD row done |
| `E-A-FLOOR-PASSED-AT-ITS-BOUND-IS-A-DEAD-FIXTURE-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-THE-VOCABULARY-IS-THE-RECOGNITION-ORGAN-THE-LAW-IS-THE-TRANSFER-ORGAN-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-256-BY-256-IS-EXACTLY-64K-THE-RAILS-SKIP-UNIT-IS-ITS-HI-BYTE-AND-A-QUARTER-BLOCK-IS-A-REMAINDER-1` | CLOSED | 1 cited path(s) live | STATUS_BOARD row done |
| `E-A-SPREAD-WITHOUT-A-SURROUND-IS-A-BLUR-INHIBITION-IS-THE-FREE-HALF-1` | CLOSED | no implementation evidence | STATUS_BOARD row done |
| `E-I-GRAFTED-HELIX-ONTO-HEXAGON-AND-THEN-DEPRECATED-THE-OPERATORS-TENANTS-ON-MY-OWN-AUTHORITY-1` | CLOSED | PR #1233 merged; 5 cited path(s) live | STATUS_BOARD row done |
| `E-THE-TWO-FAMILY-NAMINGS-INVERT-AND-FROM-BE-BYTES-IS-THE-PLAUSIBLE-WRONG-JOIN-1` | CLOSED | no implementation evidence | ISSUES entry resolved |

### SUPERSEDED (5)

| source id | status | implementation | outcome/open point |
|---|---|---|---|
| `E-A7A-IS-THE-NAME-NOT-LITERALLY-DUMB-1` | SUPERSEDED | no implementation evidence | own heading/status ⊘ note |
| `E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1` | SUPERSEDED | no implementation evidence | own heading/status ⊘ note |
| `E-THE-COVERAGE-FIX-IS-REAL-AND-ASYMMETRIC-1` | SUPERSEDED | no implementation evidence | own heading/status ⊘ note |
| `E-A-CRATE-WITH-ZERO-CONSUMERS-IS-BUILT-BY-NOTHING-AND-CAN-BE-MERGED-BROKEN-1` | SUPERSEDED | PR #957, #981 merged | own heading/status ⊘ note |
| `E-THE-RUNG-LADDER-HAS-A-STORAGE-DESIGN-AND-NO-WRITER-1` | SUPERSEDED | no implementation evidence | own heading/status ⊘ note |

### AMBIGUOUS (228)

| source id | status | implementation | outcome/open point |
|---|---|---|---|
| `E-THREE-NAMED-PROBES-ARE-ONE-MEASUREMENT` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-COMMENT-THAT-RESTATES-A-PINNED-VALUE-GOES-STALE-EVERY-BUMP-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-CLAUDE-MD-KEY-DEPENDENCIES-WENT-STALE-AND-PROPAGATED-A-WRONG-PIN-INTO-A-PLAN-1` | AMBIGUOUS | PR #915 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-JC-AND-NDARRAY-BOTH-SHIP-A-RELIABILITY-BATTERY-WITH-DIFFERENT-DEGENERATE-CONTRACTS-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-DOCUMENTED-PROXY-BYPASS-IS-FOR-PUSH-DENIALS-NOT-CLONE-AUTH-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-CORRECTION-IS-A-CLAIM-AND-CARRIES-A-CLAIM-S-BURDEN-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-JITTER-AMPLITUDE-YOU-CHOSE-IS-NOT-AN-UNCERTAINTY-YOU-MEASURED-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-CYCLONE-ASYMMETRY-IS-ONE-DIPOLE-1` | AMBIGUOUS | PR #926 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-HELIX360-IS-THE-NORMALIZED-SUBSTRATE-NOT-A-BIT-BUDGET-1` | AMBIGUOUS | PR #498 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-JUDGE-THE-FIELD-NOT-THE-ELEMENT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-MY-OWN-PRE-REGISTRATION-HAD-A-GAP-AND-I-NAMED-IT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-N-EQUALS-TWO-LOOKED-LIKE-PHYSICS-AND-WAS-HALF-COIN-FLIP-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-SPINE-FOUND-MODERATORS-MISSING-1` | AMBIGUOUS | PR #926 merged | graded OPERATOR — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-BYTE-WAS-ONLY-THE-SELECTOR-THE-PAIR-IS-THE-CARRIER-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-DOCTRINE-DOC-EXISTED-AND-I-NEVER-READ-IT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-FRAME-WAS-ALREADY-SHIPPED-FOUR-TIMES-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-HEADLINE-NUMBER-MEASURED-A-MODEL-NOBODY-CLAIMED-1` | AMBIGUOUS | PR #926 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-OFFSET-WAS-THE-APPARATUS-THE-LADDER-WAS-THE-PHYSICS-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-RESCUE-THAT-WEAKENED-ITSELF-UNDER-SCRUTINY-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-REUSE-IS-THE-PROCESS-AND-IT-EXPOSED-A-FIT-PROBLEM-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-TRANSFORM-MUST-MATCH-THE-DISTRIBUTION-SHAPE-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-TOPOLOGY-PICKS-THE-TABLE-NOT-THE-DOMAIN-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1` | AMBIGUOUS | PR #935 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-FIGURE-CITED-TWICE-IS-NOT-CONFIRMED-ONCE-1` | AMBIGUOUS | PR #945 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-ON-A-GOLDEN-LATTICE-LOCALITY-IS-FIBONACCI-MEMBERSHIP-1` | AMBIGUOUS | PR #936, #937, #938 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-CONTROL-SCORED-THE-HEADLINE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-DISPLACEMENT-FILTER-ATE-THE-STRANDED-STRATUM-1` | AMBIGUOUS | PR #940 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-REGIME-LADDER-MEASURED-RANGE-NOT-TURBULENCE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-DISABLE-PROBE-CAN-ITSELF-BE-VACUOUS-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-FIGURE-YOU-TALLIED-YOURSELF-IS-A-DERIVED-FIGURE-1` | AMBIGUOUS | PR #950 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-REAL-GATE-RAN-AND-QUALIFIED-NOT-RETRACTED-THE-CLAIM-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-IDENTITY-QUAD-4X24-RATIFIED-PERMANENT-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-FORD-REAL-PUBLICATION-IDENTITY-IS-ARRIVAL-DEPENDENT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-OGAR-CODEBOOK-MIRROR-DOMAIN-DRIFT-SYNCED-1` | AMBIGUOUS | PR #275, #276, #277 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A7A-STORNO-THE-EXCLAMATION-WAS-NOT-A-NAME-1` | AMBIGUOUS | no implementation evidence | graded CORRECTION — an epistemic grade, not a work status; no deliverable attached |
| `E-ARCHITECTURE-RESET-DUMB-STORAGE-HHTL-EPISTEMIC-1` | AMBIGUOUS | PR #968 merged | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-CROSS-VERSION-IDENTITY-MIGRATES-BLIND-SO-IT-FAILS-CLOSED-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-E2-REVERIFIED-SCATTER-CONTESTED-PMU-ABSENT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-HIERARCHY-NODE-IS-ALGEBRA-NEVER-A-CROSSWALK-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-LOTUS-IS-A-REGISTER-GRID-NOT-A-BYTE-GRID-1` | AMBIGUOUS | PR #968 merged | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-REPLAY-IS-CANONICAL-COMPACTION-IS-ECONOMICS-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-RP-SEAL-PASS1-THE-MAXIM-WORKED-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-SEAL-IS-ACCUMULATED-ON-THE-HOT-PATH-NOT-A-PASS-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-CANONICAL-ROW-WAS-READ-OFF-A-FIXTURE-1` | AMBIGUOUS | 2 cited path(s) GONE | graded CORRECTION — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-OU-COLUMN-EXISTS-AND-NOTHING-WRITES-IT-1` | AMBIGUOUS | 1 cited path(s) GONE | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-STRONG-HIERARCHY-EXISTS-AS-FIVE-DISCONNECTED-ISLANDS-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-TIER0-CANONICAL-REPLAY-LANDED-DV-IS-EPISTEMIC-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-TWO-WITNESS-SHAPES-CONTEST-ONE-LANDING-ZONE-1` | AMBIGUOUS | PR #446, #448 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-XC21-HARNESS-CONFIRMS-C2-AND-FINDS-DEAD-CODE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-LOCAL-DERIVATION-CANNOT-OVERRULE-A-MEASURED-COUNTEREXAMPLE-1` | AMBIGUOUS | PR #875 merged; 1 referenced PR(s) closed unmerged | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-DISMECH-CORPUS-CENSUS-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NIBLEPATH-DEPTH-IS-NOT-HHTL-DIMENSIONALITY-1` | AMBIGUOUS | 1 referenced PR(s) closed unmerged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-S3-0-NEEDED-NO-NEW-ADDRESS-1` | AMBIGUOUS | 1 referenced PR(s) closed unmerged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-AUDIT-GATE-WAS-PINNING-THE-BUG-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-COMPAT-ENUM-WAS-EATING-HALF-THE-REGISTER-1` | AMBIGUOUS | PR #970, #971 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-V3-IS-REPRESENTATION-INVARIANT-ON-THE-PLANNER-CE64-LEG-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-WORDNET-IS-A-LOCALITY-PRIOR-NOT-AN-IDENTITY-ENCODING-1` | AMBIGUOUS | PR #875 merged; 1 referenced PR(s) closed unmerged | graded CORRECTION — an epistemic grade, not a work status; no deliverable attached |
| `E-ABBREVIATION-GREP-MANUFACTURED-AN-ABSENCE-1` | AMBIGUOUS | PR #876 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-ACADEMIC-CARVE-UNDERFILLS-ROWS-ARE-NOT-WORDS-1` | AMBIGUOUS | PR #975 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-ADDRESS-FROM-THE-THING-NOT-THE-ACCIDENT-1` | AMBIGUOUS | no implementation evidence | graded SYNTHESIS — an epistemic grade, not a work status; no deliverable attached |
| `E-DISMECH-KNOWN-INTERMEDIATES-ARE-PROSE-NOT-IDENTITIES-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-HHTL-IS-MINTED-IN-THE-ARTIFACT-NOBODY-CITES-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-R2IL-VARNODEFACET-IS-A-G3-CARVING-AND-` | AMBIGUOUS | 1 cited path(s) GONE | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-ORACLE-POPULATION-IS-64-PERCENT-AND-A-GATE-HARDCODES-THE-OTHER-36-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-V4-IS-THE-100-PERCENT-TIER-V3-UNCHANGED-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-CONSTANT-OFFSET-CANNOT-ALIGN-TWO-VERSIFICATIONS-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-BPE-IS-RHYME-VQ-IS-THE-MECHANISM-FOR-6X2X8BIT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-HHTL-NAMES-TWO-STRUCTURES-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-GATE-IS-A-HAND-MAINTAINED-ALLOWLIST-NOT-THE-WORKSPACE-1` | AMBIGUOUS | PR #984 merged; 1 cited path(s) live | CONFLICT: live ISSUES entry vs ISSUES entry resolved |
| `E-A-WARRANT-MUST-BE-ABLE-TO-SAY-NO-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-AN-IMPORT-EDGE-IS-NOT-AN-ARCHITECTURAL-RELATION-1` | AMBIGUOUS | PR #103, #104 merged | graded CORRECTION — an epistemic grade, not a work status; no deliverable attached |
| `E-CONTENT-NEVER-TRAVELS-IN-CLASSID-1` | AMBIGUOUS | no implementation evidence | graded ROOT — an epistemic grade, not a work status; no deliverable attached |
| `E-HAPPY-PATH-RL-WOULD-HAVE-LEARNED-THE-CLOBBER-1` | AMBIGUOUS | PR #1001, #1011 merged; 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-HHTL-COMPILES-HIERARCHY-INTO-MASK-GEOMETRY-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-HIERARCHY-IS-THE-ADDRESS-SPACE-NOT-THE-ONTOLOGY-1` | AMBIGUOUS | no implementation evidence | graded ROOT — an epistemic grade, not a work status; no deliverable attached |
| `E-MEMBERSHIP-IS-PARTICIPATION-NOT-ANCESTRY-1` | AMBIGUOUS | PR #1001 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-METACOGNITIVE-TRIANGLE-ARROW-1` | AMBIGUOUS | PR #995, #997 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-OGAR-LOCO-INTERPRETER-RUN-1` | AMBIGUOUS | PR #989 merged; 1 cited path(s) GONE | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-ONE-RECEIPT-MANY-BORROWED-CONSUMERS-1` | AMBIGUOUS | PR #1012, #1016 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-REAL-CODE-INTERLEAVES-THE-OPCODE-MACRO-IS-NOT-A-DATAFLOW-PIPE-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-RECIPE-DISPATCH-BRIDGE-1` | AMBIGUOUS | PR #992, #995 merged; 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-RECIPE-EXECUTION-SEPARABILITY-1` | AMBIGUOUS | PR #992 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-STREAM-ORDER-VS-PREFIX-TREE-NEITHER-ACCUMULATES-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-SUDOKU-COGNITIVE-CORPUS-1` | AMBIGUOUS | PR #995, #996 merged; 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-FIRST-PARTICLE-1` | AMBIGUOUS | PR #1001 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-FRONTIER-LEARNER-IS-ALREADY-SHIPPED-1` | AMBIGUOUS | PR #1001, #1011 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-SEVEN-OPCODE-PROJECTION-IS-NOT-X86-AND-THE-CHAIN-CARRIER-WINS-1` | AMBIGUOUS | PR #1014 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-VIEW-MOVES-THE-POPULATION-DOES-NOT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1` | AMBIGUOUS | PR #1001 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-TYPE-COMPLEXITY-EXPOSED-A-MEMORY-ABI-ESCAPE-1` | AMBIGUOUS | PR #1004 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-BPE-OVER-DEFUSE-CHAINS-BEATS-LINEAR-AND-FITS-LOCO-1` | AMBIGUOUS | PR #998 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-GIT-SOURCED-CRATE-CANNOT-PATH-DEP-OUTSIDE-ITS-REPO-1` | AMBIGUOUS | PR #1019 merged; 2 cited path(s) live | graded FIX — an epistemic grade, not a work status; no deliverable attached |
| `E-GIT-SOURCED-CRATE-CANNOT-PATH-DEP-OUTSIDE-ITS-REPO-1` | AMBIGUOUS | 1 cited path(s) live | graded FIX — an epistemic grade, not a work status; no deliverable attached |
| `E-PHI-WEYL-STAMP-CASCADE-PRECISION-RULING-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-R2IL-BPE-RECOMBINATION-FALSIFIERS-CONFIRMED-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-R2IL-MACRO-VOCABULARY-TRANSFERS-ACROSS-COMPILER-AND-LANGUAGE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-W0-THE-SPACE-ORDINAL-IS-A-RANK-RELATIVE-TO-A-TABLE-THE-CLASSID-NEVER-NAMES-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-COUPLED-MATERIALS-NOT-A-CHOOSER-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded CORRECTION |
| `E-V4-EXECUTABLE-CONTENT-THREE-TIER-JIT-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded DOCTRINE |
| `E-A-RUNG-WRITE-PATH-ALREADY-SHIPPED-IN-A-SIBLING-REPO-1` | AMBIGUOUS | PR #561, #565, #590 merged; 1 cited path(s) GONE | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-A-FLOOD-THROTTLE-IS-NOT-A-DISCRIMINATOR-1` | AMBIGUOUS | PR #1079 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-DESTRUCTIVE-PREPEND-TRUNCATES-BEFORE-READ-1` | AMBIGUOUS | PR #1079, #1081, #1082 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-MONITOR-KEYED-ON-THE-PR-HEAD-CAN-CERTIFY-THE-WRONG-COMMIT-1` | AMBIGUOUS | PR #1120 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-REVIEW-REMEDY-HAS-A-SHELF-LIFE-1` | AMBIGUOUS | PR #1120, #1122, #1123 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-THRESHOLD-IS-BOUND-TO-ITS-STATISTIC-AND-ITS-SAMPLE-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-CONSUMER-PINS-ON-INTERNAL-SIBLINGS-PROHIBITED-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded OPERATOR-RULED |
| `E-DEPTH-INF-CONVERSE-IS-QUADRATIC-IN-LEVY-AREA-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-I-PINNED-THE-DEFECT-AS-THE-GUARD-WHILE-FIXING-A-REVIEW-COMMENT-1` | AMBIGUOUS | PR #1120, #1122 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-LEVEL-SCALED-NORMALIZATION-IS-THE-SIGNATURE-PARITY-GATE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-LEVY-AREA-COEFFICIENT-BEATS-REFINEMENT-1` | AMBIGUOUS | PR #350 merged; 2 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-MONOTONE-STREAM-LEVEL2-IS-DISCRIMINATION-NOT-MAGNITUDE-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NECESSARY-CONDITIONS-ARE-NOT-A-PSD-TEST-1` | AMBIGUOUS | PR #291 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-ORIENTATION-BIT-PARTIAL-NIBBLE-SUFFICES-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-PRESENCE-2BIT-CHEAPER-SIBLING-1` | AMBIGUOUS | PR #1099, #1103 merged; 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-Q8-THE-SIX-DOES-NO-WORK-A-DEGREE-ABLATION-COLLAPSES-THE-HEX-OVERLAYS-ENTIRE-ADVANTAGE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-SUPERSESSION-GATE-WATCHED-TWO-OF-ITS-FOUR-INPUTS-1` | AMBIGUOUS | PR #1123 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-TWO-REVIEWERS-FOUND-THE-SAME-THREE-DEFECTS-AND-ONE-OF-THEM-WAS-MINE-ALONE-1` | AMBIGUOUS | PR #1120 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-W0-MEASURED-THE-MASK-HALF-DOMINATES-AND-THE-PLAN-WAS-UNDER-CITED-1` | AMBIGUOUS | PR #1117 merged | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-AN-HHTL-POSITION-IS-A-NODE-AND-A-NODE-HAS-A-VALUE-1` | AMBIGUOUS | PR #1127 merged | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-ASKING-WHERE-THE-MAP-LIVES-IS-ASKING-WHERE-THE-OUS-ARE-IN-A-DN-1` | AMBIGUOUS | PR #1127 merged | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-G24N4-ALREADY-SHIPS-AND-THAT-IS-WHY-W2B-CANNOT-USE-IT-1` | AMBIGUOUS | 1 cited path(s) GONE | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-LITERATURE-HARVEST-POST-1132-TWO-PILLAR-CORRECTIONS-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded HARVEST |
| `E-ONE-HOP-UP-ONE-HOP-DOWN-A-PARENT-SPEAKS-ONLY-ITS-CHILDREN-1` | AMBIGUOUS | no implementation evidence | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-THE-24-AXIS-BASIS-V3-EVERY-AXIS-IS-A-GROUNDED-PRESSURE-1` | AMBIGUOUS | PR #296 merged | graded BUILT — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-PALETTE-MARGIN-IS-SPENT-AND-GROWTH-MOVES-TO-LOCO-1` | AMBIGUOUS | PR #1125 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-SIGNED-NET-WAS-FALSIFIED-NOT-LIMITED-AND-THE-LOCI-LAW-WAS-SCOPED-TOO-WIDE-1` | AMBIGUOUS | PR #1127 merged | graded OPERATOR — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-STATE-LAYER-IS-A-BELNAP-BILATTICE-AND-THE-JOIN-IS-THE-ACCUMULATOR-1` | AMBIGUOUS | PR #1129 merged | graded OPERATOR — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-THRESHOLD-AXIS-WAS-SATURATED-AND-THE-GATE-WOULD-HAVE-BEEN-VACUOUS-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-THREE-BRANCHES-ONE-REGISTER-THE-AUDIT-AFTER-THE-COLLISION-1` | AMBIGUOUS | PR #1125, #1126, #1127 merged | graded RECONCILIATION — an epistemic grade, not a work status; no deliverable attached |
| `E-THREE-KINDS-OF-MENGENLEHRE-AND-W2-SHIPPED-THE-NARROWEST-1` | AMBIGUOUS | no implementation evidence | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-A-GHOST-TRACE-IS-NOT-THE-COUNTERFACTUAL-LANE-1` | AMBIGUOUS | PR #1137 merged | CONFLICT: live TECH_DEBT entry vs STATUS_BOARD row done |
| `E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1` | AMBIGUOUS | 1 cited path(s) live | CONFLICT: live TECH_DEBT entry vs STATUS_BOARD row done |
| `E-PILLAR-11-PUBLISHED-BOUND-NEEDS-ITS-OWN-NUMERIC-GUARD-1` | AMBIGUOUS | PR #1133 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-SIGNATURE-PDE-SWEEP-SHIPPED-W1` | AMBIGUOUS | PR #293, #348 merged; 2 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-SIX-SEMANTIC-FAMILIES-MUST-NOT-IMPERSONATE-EACH-OTHER-1` | AMBIGUOUS | PR #295, #1125, #1128 +3 merged; 1 referenced PR(s) closed unmerged | graded OPERATOR — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-DTO-LADDER-IS-THE-ALU-BUS-AND-WAS-ALREADY-RULED-1` | AMBIGUOUS | PR #1051 merged | graded CORRECTION — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-LIFT-GATE-FOUND-A-TIE-BLIND-SPEARMAN-1` | AMBIGUOUS | 1 cited path(s) live | CONFLICT: live TECH_DEBT entry vs STATUS_BOARD row done |
| `E-THE-PERIPHERY-OF-A-STRATUM-IS-THE-OTHER-STRATA-1` | AMBIGUOUS | PR #1141 merged; 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THINKING-ENGINE-LIVE-FOOTPRINT-IS-ONE-TRAIT-AND-HOUSE-IS-SHIPPED-IN-PIECES-1` | AMBIGUOUS | PR #387 merged | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-A-CENSUS-IS-A-FUNCTION-OF-ITS-REGEX-SO-GATE-THE-PROPERTY-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-CITATION-IS-NOT-A-DEPENDENCY-AND-A-FORCED-COPY-NEEDS-A-GATE-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-CORRECTION-IS-ONLY-AS-GOOD-AS-ITS-MERGE-1` | AMBIGUOUS | PR #1092 merged; 2 referenced PR(s) closed unmerged; 1 cited path(s) GONE | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-CROSS-REPO-SYMBOL-GREP-IS-ONLY-AS-FRESH-AS-THE-SIBLING-CHECKOUT-1` | AMBIGUOUS | PR #1157 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-HAND-SWEEP-UNDERCOUNTS-TOWARD-DONE-AND-THE-CRITERION-IS-THE-WHOLE-DESIGN-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-A-PROBE-CAN-STATE-A-MEASUREMENT-THAT-WAS-FALSE-WHEN-IT-WAS-WRITTEN-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-RULED-HOME-NEEDS-A-FIRST-CONSUMER-OR-IT-IS-A-VACANCY-1` | AMBIGUOUS | PR #1152 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-M8-COLLAPSE-TARGET-ALREADY-EXISTED-1` | AMBIGUOUS | PR #1151 merged; 1 cited path(s) live | CONFLICT: STATUS_BOARD row not done + live ISSUES entry vs STATUS_BOARD row done |
| `E-THE-ENTROPY-HOME-WAS-RULED-AND-LEFT-EMPTY-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-FIX-FOR-A-REVIEW-FINDING-SHIPS-UNREVIEWED-BY-DEFAULT-1` | AMBIGUOUS | PR #1154, #1160 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-FREE-MITIGATION-WAS-FREE-FOR-TWO-HOURS-1` | AMBIGUOUS | PR #1160 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-GATE-INHERITS-THE-BLIND-SPOT-OF-WHOEVER-WROTE-IT-1` | AMBIGUOUS | PR #1167, #1168, #1169 +1 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-SKETCH-THAT-MISSED-TWICE-WILL-MISS-A-THIRD-TIME-1` | AMBIGUOUS | PR #293, #294 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-COLUMN-OF-INDICES-INTO-A-CODEBOOK-THAT-DOES-NOT-EXIST-1` | AMBIGUOUS | 4 cited path(s) live | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-A-MACHINE-APPLICABLE-FIX-IS-A-SUGGESTION-NOT-A-PROOF-1` | AMBIGUOUS | PR #302, #1194, #1195 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NARS-EXPECTATION-CHOICE-PREFERS-IGNORANCE-TO-A-CONFIDENT-NEGATIVE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-1` | AMBIGUOUS | no implementation evidence | CONFLICT: live TECH_DEBT entry + own status line: PROPOSAL vs STATUS_BOARD row done |
| `E-NXG-12` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-NXG-13` | AMBIGUOUS | PR #288, #295 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-14` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-NXG-15` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-NXG-17` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-19` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-20` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-3` | AMBIGUOUS | 2 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-4` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-7` | AMBIGUOUS | PR #296, #1134, #1159 merged; 1 referenced PR(s) closed unmerged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-NXG-8` | AMBIGUOUS | PR #1129 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1` | AMBIGUOUS | PR #1185 merged | graded OPERATOR-RULED — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-UNFINISHED-FUNCTION-WAS-NOT-THE-DEBT-1` | AMBIGUOUS | PR #1188 merged; 1 cited path(s) GONE | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-UNFINISHED-UDF-WAS-NOT-THE-DEBT-1` | AMBIGUOUS | PR #1185 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-VERSIONED-GRAPH-OVERWRITES-SO-ROW-ADDRESSES-ALIAS-ACROSS-VERSIONS-1` | AMBIGUOUS | PR #1190 merged; 1 cited path(s) live | CONFLICT: STATUS_BOARD row not done + live TECH_DEBT entry vs STATUS_BOARD row done |
| `E-I-CITED-THE-RIGHTMOST-REGISTER-AND-CALLED-IT-THE-ADDRESS-1` | AMBIGUOUS | PR #174, #175, #658 +1 merged | graded OPERATOR — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-AARCH64-PATH-HAD-NEVER-BEEN-COMPILED-1` | AMBIGUOUS | PR #146 merged; 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-DOC-COMMENT-IS-NOT-A-FAIL-CLOSED-MECHANISM-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-A-PLAN-INVENTORY-FINDS-THE-BOARD-LAGS-THE-TREE-IN-BOTH-DIRECTIONS-1` | AMBIGUOUS | PR #1198 merged | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-A-V3-MINT-MUST-NEVER-DEGRADE-TO-V1-1` | AMBIGUOUS | PR #1207 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-AN-EMPTY-RANGE-AFTER-A-RESET-IS-NOT-EVIDENCE-1` | AMBIGUOUS | PR #1201, #1203, #1217 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-EVERY-DOMAIN-IS-A-TABLE-AND-A-CROSSWALK-IS-A-CHAIN-OF-MASKS-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-PLUG-AND-PLAY-IS-THE-DECLARATION-NOT-A-TABLE-1` | AMBIGUOUS | PR #1207, #1216 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-RUNG-BAND-AND-PLASTICITY-ARE-THREE-AXES-NEVER-ONE-LEVEL-FIELD-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FENCE |
| `E-THE-FUSED-AND3-HOP-WAS-NEVER-SHIPPED-LGJ-HOP-IS-TWO-ANDS-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-V1-GUARD-WAS-TESTED-THE-V3-GUARD-THAT-REPLACED-IT-WAS-NOT-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-SIGMA-CHAIN-IS-A-PROVEN-UG-SURVIVAL-AND-PHI-IS-A-HOMONYM-1` | AMBIGUOUS | 1 cited path(s) live | graded FOSSIL — an epistemic grade, not a work status; no deliverable attached |
| `E-TRIPLE-MODEL-DKPOSITION-IS-AN-UNWIRED-DUPLICATE-1` | AMBIGUOUS | 4 cited path(s) live | graded FOSSIL — an epistemic grade, not a work status; no deliverable attached |
| `E-LE-IS-THE-UNIVERSAL-DTO-LAYER-TYPED-SYNTAX-MEANS-A-VERSIONED-LE-SCHEMA-1` | AMBIGUOUS | PR #1154, #1222, #1223 merged | CONFLICT: STATUS_BOARD row not done + live TECH_DEBT entry vs STATUS_BOARD row done |
| `E-HEX-TENANT-RAIL-IS-DIRECTION-CHAIN-IS-FREE-SHIFT-IS-THE-COST-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded — |
| `E-HEX-TENANT-RAIL-IS-DIRECTION-CHAIN-IS-FREE-SHIFT-IS-THE-COST-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-DISABLE-CAN-GO-RED-FOR-THE-WRONG-REASON-AND-THE-TWO-PEAK-FIGURES-WERE-NEVER-IN-CONFLICT-1` | AMBIGUOUS | PR #1233 merged | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-HORIZON-CUT-AND-AN-UNBOUND-MEET-ARE-NOT-THE-SAME-ANSWER-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-A-THOUGHT-MASKS-ITSELF-BY-ITS-DISTANCE-FROM-ROOT-THE-V3-FACET-IS-THE-MASK-AND-THE-RADIUS-IS-STEPLESS-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-UNIFORM-WEIGHT-ARM-CANNOT-MEASURE-EVIDENCE-ITS-ARGMIN-IS-INVARIANT-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-BOUNDED-ATTENTION-BUYS-REACH-AND-A-DISTANT-HOP-IS-A-TERNLOG-NOT-A-SEMIRING-1` | AMBIGUOUS | no implementation evidence | CONFLICT: live ISSUES entry vs STATUS_BOARD row done |
| `E-DENSITY-IS-FALSIFIED-THE-VARIABLE-IS-PATH-LENGTH-SPREAD-AND-THIS-RE-OPENS-A5-1` | AMBIGUOUS | no implementation evidence | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done + ISSUES entry resolved |
| `E-DEPTH-RANK-REPRODUCES-MOST-SPECIFIC-BUT-ONLY-ON-A-TAXONOMY-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-FAMILY-HAS-FOUR-WIDTHS-AND-4096-HAS-FIVE-REFERENTS-PIN-THE-UNIT-BEFORE-THE-ARITHMETIC-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-FUSING-FORFEITS-THE-SKIP-AND-ADAPTIVEFILTER-FAILS-IN-TWO-PLACES-NOT-ONE-1` | AMBIGUOUS | no implementation evidence | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-I-DECLARED-A-JOIN-ABSENT-BY-GREPPING-ONE-FILE-AND-COMPOSE-IS-THE-SAME-XOR-A-THIRD-TIME-1` | AMBIGUOUS | 5 cited path(s) live | graded CORRECTION — an epistemic grade, not a work status; no deliverable attached |
| `E-POPCOUNT-TIMES-SELF-THE-EXACT-PREFIX-IS-THE-K-EQUALS-ZERO-HAMMING-BALL-AND-THE-FUSED-ROW-PREDICATE-IS-THE-GAP-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-SIX-SEAMS-EVERY-CAUSAL-SELECTOR-SHIPS-AND-NONE-IS-WIRED-AT-THE-HOP-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-THE-BOXCAR-HORIZON-IS-NOT-A-DISCOUNT-IT-REVERSES-THE-OTHER-WAY-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-THE-CANON-SPECIFIED-THE-WHOLE-MASKED-O1-CHAIN-AND-ITS-LOAD-BEARING-LINKS-ARE-STUBS-1` | AMBIGUOUS | no implementation evidence | CONFLICT: live ISSUES entry vs STATUS_BOARD row done |
| `E-THE-RAIL-IS-A-NEEDLE-NOT-A-MASK-256-BY-256-IS-THE-EXACT-ROW-ADDRESS-AND-A-MASK-OVER-THE-AREA-IS-ANOTHER-OBJECT-1` | AMBIGUOUS | no implementation evidence | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-REVIEW-FOUND-A-REAL-BUG-THAT-FALSIFIED-MY-OWN-ISSUES-PREMISE-AND-I-BROKE-MY-OWN-RULE-IN-THE-FILE-STATING-IT-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-THE-RLHF-SHAPED-PROMOTION-LOOP-IS-IMPLEMENTED-END-TO-END-IN-A-PROBE-AND-HAS-NO-SRC-PROMOTER-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-THE-SEMIRING-IS-FREE-THE-COST-IS-CARRIER-WIDTH-AND-THE-JOIN-IS-THE-SAME-XOR-1` | AMBIGUOUS | 1 cited path(s) live | CONFLICT: live ISSUES entry vs ISSUES entry resolved |
| `E-THE-SKIP-LEVER-LIVES-ONLY-BELOW-THE-DENSITY-WHERE-D-GTM-0N-SAYS-SWITCH-TO-SPARSE-AND-THE-CLUSTERED-99-90-IS-PREFIX-ARITHMETIC-1` | AMBIGUOUS | 1 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-SLOWEST-GATE-IS-THE-ONE-YOUR-OWN-PUSH-CADENCE-CANCELS-1` | AMBIGUOUS | PR #1235 merged; 2 cited path(s) live | graded FINDING — an epistemic grade, not a work status; no deliverable attached |
| `E-FORMAT-SLOT-FOLD-IS-THE-SAME-OP-AS-THE-VL-DESCENT-1` | AMBIGUOUS | PR #310, #1242, #1243 +1 merged | graded MEASURED — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-SPINE-IS-WHATEVER-THE-READER-ALREADY-HAS-AN-ADDRESS-FOR-1` | AMBIGUOUS | PR #1085, #1240 merged; 1 cited path(s) live | graded OPERATOR-RULED — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-SECOND-FACET-IS-NOT-AN-EDGE-BLOCK-1` | AMBIGUOUS | no implementation evidence | graded OPERATOR-RULED — an epistemic grade, not a work status; no deliverable attached |
| `E-BYTES-ARE-STORED-INTEGERS-ARE-PROJECTED-1` | AMBIGUOUS | PR #1245, #1246 merged | graded OPERATOR-RULED — an epistemic grade, not a work status; no deliverable attached |
| `E-NO-FOLD-REPORTS-AN-O-POPULATION-COST-1` | AMBIGUOUS | no implementation evidence | graded — — an epistemic grade, not a work status; no deliverable attached |
| `E-1224-CLOSED-FOR-BEING-WRONG-NOT-FOR-LACKING-CONSUMERS-1` | AMBIGUOUS | 1 referenced PR(s) closed unmerged | graded CORRECTION — an epistemic grade, not a work status; no deliverable attached |
| `E-1224-WAS-A-BIDIRECTIONAL-DOMAIN-INVERSION-NOT-A-LEAK-1` | AMBIGUOUS | 1 referenced PR(s) closed unmerged | graded SHARPENING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-BORROW-IS-NOT-A-REPLAY-CARRIER-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded RULING |
| `E-A-BOUND-AND-A-TILE-ARE-INTERVALS-IN-DIFFERENT-ORDERS-1` | AMBIGUOUS | no implementation evidence | CONFLICT: STATUS_BOARD row not done vs STATUS_BOARD row done |
| `E-A-DOMAIN-IS-AN-OPTIONAL-CONSUMER-THROUGH-OGAR-NEVER-A-CO-DEFINER-1` | AMBIGUOUS | 1 referenced PR(s) closed unmerged | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-A-POSITIONAL-INDEX-ADDED-TO-A-KEY-DIGEST-ATTESTS-NOTHING-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded FINDING |
| `E-A-THOUGHT-IS-A-REPLAYABLE-OPERATOR-NOT-A-MAINTAINED-STATE-1` | AMBIGUOUS | PR #1245, #1250 merged | graded RULING — an epistemic grade, not a work status; no deliverable attached |
| `E-ATTENTION-IS-NOT-EVIDENCE-AND-FIRE-IS-NOT-DURABILITY-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded RULING |
| `E-FOLDS-ARE-ZERO-COPY-PERIOD-PEEK-NOT-BORROW-BUILD-FOLD-1` | AMBIGUOUS | no implementation evidence | graded LAW — an epistemic grade, not a work status; no deliverable attached |
| `E-FROZEN-IS-FINE-MARCHING-IS-THE-DISASTER-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded RULING |
| `E-MASKING-IS-AN-OPERATION-A-MASK-IS-A-CARRIER-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded RULING |
| `E-ONE-OBSERVABLE-IS-NOT-THREE-INSTRUMENTS-AND-IMPORTS-ARE-PROBES-1` | AMBIGUOUS | 1 referenced PR(s) closed unmerged | graded FENCE — an epistemic grade, not a work status; no deliverable attached |
| `E-REPLAY-CAN-BE-CHEAPER-THAN-STORAGE-1` | AMBIGUOUS | no implementation evidence | graded CONJECTURE — an epistemic grade, not a work status; no deliverable attached |
| `E-THE-1224-DETOUR-CLEANUP-PASS-WHAT-WAS-CONTAMINATION-AND-WHAT-SURVIVES-1` | AMBIGUOUS | 1 referenced PR(s) closed unmerged | graded CLEANUP — an epistemic grade, not a work status; no deliverable attached |
| `E-ZERO-COPY-IS-NOT-A-SIZE-THRESHOLD-1` | AMBIGUOUS | no implementation evidence | no join key at all; graded RULING |

---

## 4. What this baseline does NOT claim

- It does not claim the 228 AMBIGUOUS rows are resolved, wrong, or safe
  to delete. They are unadjudicated, and that is recorded, not hidden.
- It does not claim a merged PR closed the finding attached to it.
- It does not claim the frozen prose was reviewed. **FROZEN ≠ RECONCILED.**
- It is not a licence to start another archaeology pass over the ambiguous
  rows. If one matters later it resurfaces as live work and enters the
  transient tier like anything else.

## 5. Steady state after this checkpoint

```
new work → .claude/board/entries/ → reconcile against current state
         → OPEN | CLOSED | SUPERSEDED | AMBIGUOUS
         → rare Eureka promotion to EPIPHANIES.md (must cite its entry)
         → advance PROCESSED_THROUGH_SHA to the captured source head
```

MIRROR dies. Corrections die. Failed probes normally die. Git keeps the
route. Only surviving state crosses the checkpoint.
