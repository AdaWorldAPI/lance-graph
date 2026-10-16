# Ledger revision, 2026-10-16 (c): #1582–#1591

Maps the merges since the last mapping (`LEDGER-UPDATE-2026-10-16b.md`, #1581)
onto the open list. That file and its predecessors are not edited; this one
supersedes the rows it names. Board entry:
`.claude/board/entries/2026-10-16-ledger-revision-1582-1591.md`.

**Scope.** Ten merged PRs, #1582–#1591. Read from the merged PR bodies (Haiku
harvest, verbatim), the code diffs of #1587, #1588 and #1591, and the board
files each PR changed. #1583 and #1589 are board-only; #1590 is board and plans
only.

## 1. The ruling that reorders everything: gather is forbidden (#1590)

#1590 put a VIOLATION banner on every plan, entry and STATUS row that designs a
read at a position taken from another lane's values: gather, foreign lanes, and
the preimage hop. The correct design it names is the D-HOLO pair
(`d-holo-h14c-thought-first-v1.md`, `d-holo-h14d-logic-mask-read-v1.md`):
`Thought(t)[T_a[i] : T_b[i]]`, both values of a pair at the same coordinate, so
**the join is the layout**.

Rows of this ledger the ruling touches:

| row | before | after |
|---|---|---|
| D2 (one preimage op serving four callers, by reference from #1579) | OPEN, fruit H2 | **EXCLUDED.** D-CMR-5 is bannered; the op is not to be built. |
| D-RA-4 (bounded recursion over D-CMR-5 preimage) | waits on D-RA-2 | **EXCLUDED** (bannered). D-RA-2 itself is not touched: it is about where single-table rules live, and only relalg's "joins" line is bannered. |
| N15 → D-ASM | Queued (authorization) | **Re-spec needed.** The council-v3 §0 preimage hop, D-ASM-2's existence half (Semijoin / `EqU32Via`) and D-ASM-3's foreign fan-out (`Semijoin` × `GroupReduce{Count}`) are all bannered; D-ASM-11 / 11A / 11B too, although 11A/B were marked authorized the same day. D-ASM-1 (binder) and the Hyperon baseline measurement are untouched. |
| N19 (layout costs, no A/B/C verdict) | OPEN, no decision depends on it | **Promoted.** "The join is the layout" makes the write layout the join strategy. H14d's measurement (bytes fall only when survivors are page-local) is now the cost model for every relation, not one probe's footnote. |

What the ruling did NOT do, and is still open:

- **No DECISION record.** The ruling exists only as banners. No entry carries DECISION / SCOPE / BASIS / REVISIT WHEN, so a session cannot read what is in scope (is a same-lane `seek` a gather? is a codebook lookup?) without re-deriving it.
- **The code surface is untouched**, by design ("a separate change"): quack `Filter::Semijoin`, `Pred::EqU32Via`, `MaskOp::Gather`, and #1587's `SealedFacetLane::seek_from` / `gallop_partition_point` and relalg `is_alpha_acyclic`. The last two shipped with D-ASM-11 as their only planned consumer, which is now bannered.

## 2. New closed ends

| id | what closed | evidence |
|---|---|---|
| C31 | D8 doc drift, lance-graph part: `.cargo/config.toml` AVX2-oracle note (native for local, v4 / sapphirerapids for shipped), `CARGO-COMPUTE-SUBSTRATE.md` rev-pin template, TD-NDARRAY-SIMD-POPCNT-NATIVE item (b) withdrawn | #1582 |
| C32 | OpenCog's role: a temporary validation oracle, harvested outside the workspace; AGPL with linking exception means an adapter does not change our licence, a port does | #1582, entry `2026-10-16-opencog-oracle-scope.md` (DECISION form) |
| C33 | CATS optional fields bind three-valued through validity planes; reading the sentinel lane alone kept NULL rows | #1588; disable run red |
| C34 | WCOJ pre-work measured: semijoin chain over-approximates on the cyclic F4 fixture; seek bound `m·(2·log₂(N/m)+4)` holds; GYO classifies 7 acyclic / 4 cyclic bodies; the exact cyclic mask needs one disjunct per value with today's ops; DuckDB core has no WCOJ | #1586, #1587 |
| C35 | G1 (relative-clause antecedent → `Locus::Antecedent`) and G2 (TEKAMOLO → loci 0..3) | #1591 `wire.rs`, tests |
| C36 | The Gadamer gate runs on real parsed data: every observation passes `GadamerRevision` before it is counted; a contest never resets a rung; echoes count nothing | #1591, four disable runs red |
| C37 | O30 has a measured failure: `deepnsm_v2::belief::Stamp` folds sources modulo 64, so two verses 64 apart read as overlapping evidence | #1591 entry §7 |
| C38 | Arc catch-up for 115 merged PRs | #1589 |

## 3. Open items, revised

Fruit: **L** low, **H** high, **Heavy**. Tier: who does the grindwork.

| id | item | status after #1591 | fruit | tier |
|---|---|---|---|---|
| **WIRE-50** (new, highest) | Bits 50..52 now have two encodings in code with opposite order. The pyramid (`knowledge/maslov-pyramid-of-cognition.md`, `planner/tests/maturation_ladder.rs`): 7 Astonishment → 0 Wisdom. `CausalEdge64::with_novelty_wisdom` (#1591): 0 = not on the ladder, 1 Staunen ..= 7 wisdom. Under one global wire (R-b) a wire-written 7 reads as Astonishment. D-CGW-1's gate (wire edges stay in the probe) contains it today, but D-CGW-1 names only 43..45. | OPEN, decision | — | O |
| **D-MATURATION3-1** (merged with WIRE-50) | Entry code (7 vs a separate "new"), `learn`'s rule, migration gate. #1591 is a concrete proposal for decision 3: code 0 = "not on the ladder". It costs Wisdom its code, which the pyramid gives to code 0. | OPEN, now one decision with two candidates in code | Heavy | O |
| **DOC-WIRE (1)** (merged with D-CGW-1 / D-CGW-10) | One global meaning for 43..52 in `ce64-semantic-upper-half.md`. #1591 writes 43..45 as a Haspelmath dative-map function, beside the direction triad's `DIR_SHIFT = 43`. D-CGW-1 (Queued) is the same item: one code table in `causal-edge`, triad readers migrated. | OPEN, now has a queued owner | H | O decides, S migrates |
| **D-46** (merged with D-CGW-9) | 46..49: D-CGW-9 proposes the ISA inference mantissa stays the law and activation leaves the edge. That answers D-46 in favour of the code. | OPEN, recommendation exists | — | O |
| **EDGE-DOC** (new) | `CausalEdge64::novelty_wisdom`'s doc says "a writer picks one vocabulary per class and says which" — the per-class reading R-b withdrew (K1) and D-CGW-1 now forbids. | OPEN | L | spec-editor |
| **GATHER-DECISION** (new) | Write the ruling as one DECISION entry (scope: which reads count as gather; whether same-lane seek and codebook lookups are in scope; REVISIT WHEN). | OPEN | L | O words it, S records |
| **GATHER-CODE** (new) | The existing gather surface (`Filter::Semijoin`, `Pred::EqU32Via`, `MaskOp::Gather`) and #1587's orphans (`seek_from`, `gallop_partition_point`, `is_alpha_acyclic`). Each needs keep / gate / remove. `is_alpha_acyclic` may stay as a refusal test (refuse cyclic bodies). | OPEN | H | O decides per item |
| **D-ASM** (re-spec) | See §1. What survives: D-ASM-1 binder, the Hyperon 26-script baseline, D-ASM-1b, -8, -10, -12, -13 where they do not route through a foreign lane. | OPEN, re-spec | Heavy | O, then /5plus3 |
| **O25 + K2'** (evidence root) | Unchanged in shape; new concrete sites: `BeliefArena`'s modulo-64 stamp (C37) and AriGraph `revise_with_evidence`, which revises on a string match with no stamp (#1591 §1). | OPEN, needs O30 | H | O, then S |
| **O30** | Evidential-base width. C37 is the measured cost of the current answer (64, folded). | OPEN, decision; now evidenced | — | O |
| **K6** (derived packet, D-HL-5 H4) | #1591 writes 53..63 for parsed rows: W = the statement's previous state, topology from the clause path, certification from support balance, never `Causes`. That is the observation case; H4 asks the syllogism case. The two should agree on 59..63 vocabulary. | OPEN, one data point | H | S runs H4 |
| **D-CGW-4 ↔ C22** (new connection) | `CausalCandidate` → `Causes` only with `IncreaseEligible ∧ Necessary` is exactly `RevisionVerdict::is_acceptable()`, whose first producer is `pearl.rs` `Reaction::verdict` (#1560). D-CGW-4 should call it, not re-derive it. | OPEN | H | S |
| **ABL** | Ablation through the oracle families. #1582 fixed OpenCog's role; D-CGW-13 queues a cross-oracle readout on the same CE64 fields (Java OpenNARS, ONA, MeTTa, OpenCog, MORK); D-ORACLE-0 (#1549) pre-registers 100–150 fixtures. These are one harness. | OPEN | H | coresearch, then S |
| **N16** (operand registry) | Related to D-CGW-8 (S/P/O identity as 8 × u8:u8 rails beside the u8 palette). | OPEN | Heavy | O |
| **N17** | Unchanged: the absorption fix waits on D-RA-2. Not bannered. | OPEN | L tests / fix waits | S |
| **N18** | Unchanged. | OPEN | H | S |
| **N20** (qualia CLUT) | #1591 measured the palette's separation (nearest-neighbour L1 min 25, median 90) and showed a "3σ" claim needs the unmeasured A2B noise; D-CGW-14 (OpenPSI as a qualia device) is a candidate device profile. | OPEN | H | S |
| **TD-ONTOLOGY-REGISTRY** (new, #1582) | `OntologyRegistry` resolves through string-keyed hash maps; review against codebook ids + masks, measured on one OGIT tenant. | OPEN, review | H | S |
| **#1589 stale rows** | Its open-items entry lists K1, O25-as-policy, O26 and N1 as owed decisions; the ledger withdrew K1, made O25 a defect, downgraded O26 and narrowed N1. #1591's entry §4 repeats the N1 / rule-S7 reading of 50..52 that ruling R-c settled. | OPEN | L | spec-editor, dated correction lines |
| **ARC-GAPS** | #1549 and #1581 had no arc entry. | **Closed in this PR** | — | — |
| **STATUS D-CGW-0** | reads "In PR"; #1591 merged. | OPEN (owner session) | L | — |
| O23, O29, R2–R4, O26, O7/O13/O15, BAND-RETIRE | Unchanged. | OPEN | — | O |

## 4. Connecting the dots

- **K9 — the join is the layout, and the layout is already measured.** The
  gather ban (§1), N19's page-local cost (H14d), H14c's thought-first reads and
  #1591's rows (each row already carries S, P, O, F/C and its loci at one
  coordinate) are one design: whatever a query will pair must be written at
  the same coordinate. That turns N19 from a cost note into the write-side
  contract, and it is the replacement D-ASM needs for its existence half.
- **K10 — one CE64 contract, written once.** WIRE-50, DOC-WIRE (1), D-46,
  EDGE-DOC and D-CGW-1/9/10 are one change: a single code table in
  `causal-edge` for 40..63, with `wire.rs`, `learn`, the pyramid tests and the
  triad readers reading it. D-CGW-1's gate makes it safe to decide before any
  wire edge leaves the probe.
- **K11 — the docket is real now.** C22 built the verdict producer, C36 runs the
  gate on parsed data, D-CGW-4 needs the verdict. Wiring D-CGW-4 to
  `Reaction::verdict` closes the path from a sentence to an admitted `Causes`
  with no new type.
- **K12 — evidence roots have their failing cases.** C37 (modulo-64 stamps)
  and AriGraph's stamp-less revision are the two concrete failures O30 and K2'
  exist to prevent; both can become failing tests before the root type exists.
- **K7 (from before) — ablation, oracles, D-ORACLE-0 and D-CGW-13 are one
  harness.**

## 5. Fruits

- **Low:** GATHER-DECISION record; EDGE-DOC one-line fix; dated corrections on
  #1589's and #1591's stale rows; failing tests for C37 and AriGraph's
  stamp-less revision; N17's two reproducers as failing tests.
- **High:** WIRE-50 + D-MATURATION3-1 decided and written into one code table
  (K10); D-CGW-4 through `Reaction::verdict` (K11); GATHER-CODE triage; K6/H4;
  the oracle harness (K7); N18; N20; the ontology registry review.
- **Heavy:** D-ASM re-spec around layout joins (K9); O25 + K2' evidence root
  after O30; N16; the write-side layout contract (N19 promoted).

## 6. Decisions for the operator

| # | question | recommendation | caveat |
|---|---|---|---|
| WIRE-50 | which order owns 50..52 | keep the pyramid (7 Astonishment → 0 Wisdom): it is the ruled reading (R-c) and has tests; give "not on the ladder" its own meaning only through D-MATURATION3-1 decision 3 | `wire.rs` and its probe re-pin; the probe's rule R2 reads rung order |
| GATHER-DECISION | scope of "gather" | one DECISION entry; same-lane `seek` and frozen codebook lookups stated explicitly in or out | without it, every new plan re-asks |
| GATHER-CODE | existing gather code | gate behind a named feature first, remove after D-ASM re-spec; keep `is_alpha_acyclic` as a refusal test | removal touches quack, mask-risc and lgj-abi |
| D-46 | 46..49 | accept D-CGW-9: the mantissa is law, activation leaves the edge | activation needs a home (a lane beside the key) |
| O30 | root width | unchanged: bounded root list keyed on (lane, row position), overflow refused | C37 shows folding is the wrong default |

## 7. Agent ledger (Haiku harvest tier)

| card | runs | items checked | wrong | missed |
|---|---|---|---|---|
| general-purpose on Haiku (PR bodies + file lists #1582–#1591) | 1 | #1588's two test names re-found in `tests/binder.rs`; #1587 / #1591 file counts equal `git diff --stat` (17, 18) | 0 | 0 |
