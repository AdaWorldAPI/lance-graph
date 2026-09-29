# deepnsm-v2-lexical-evidence-consumer-v1

**Status:** PROPOSAL, ratified by a 5+3 council (2026-09-29; v1 → v2 → v3 in
the change ledger below). No code authorized. Written against `main`
`5282dfa3` (the #1299 merge).

## Overview

#1299 made DeepNSM-v2 keep every counted PoS reading per `WordId`
(`crates/deepnsm-v2/src/lexical.rs`). Nothing reads it yet.
`examples/bible_wave.rs::load_pos` (`:1023`) still tags tokens by
`entry().or_insert_with`: first `lemmas_5k.csv` (keyed by the LEMMA string),
then `word_forms.csv` (keyed by the SURFACE string), first reading wins, counts
dropped.

This plan (**D-LXC-1**) changes the consumer only. The library stays
count-only: it preserves evidence and does not perform the CausalEdge64
epistemic transition.

## Inherited rulings (not re-opened)

- F1 #1299 boundary: evidence, not transition; no dep on causal-edge,
  cognitive-shader-driver, arm-discovery or the deprecated deepnsm crate.
- F2 Routing `WordId`s and Cam96 `codes[word_id]` do not move.
- F3 Integer counts; unknown = `None`; aggregates checked; unknown beats overflow.
- F4 Each guard has a can-fire and a can-stay-silent test; disable runs turn tests red.
- F5 Search is navigation, not evidence (workspace policy).
- F6 Proposal only for now; no force push; no model names or attribution in commits or PR text.
- F7 Commit `68955ecb` deleted the deepnsm-v2 tagger module as redundant with
  the planner's `insight_coca_read`; taggers stay inlined in the examples
  (`bible_wave.rs:976-979`, `genre_shapes.rs:200-203`). Cited from the commit
  message; the deleted module itself was not re-read.
- F8 `lexical.rs:50-54`: `PosCode` is deliberately not mapped onto `fsm::Pos`
  in the library; a consumer maps at its own boundary.

## Two evidence classes, kept apart

- **Lemma-keyed** (`lemmas_5k.csv`): one tag per lemma, applied today to any
  surface token with the same spelling.
- **Surface-keyed, counted** (`word_forms.csv` via `LexicalEvidence`): every
  reading of a surface with its `wordFreq`.

Neither is a role-based reading. The workspace's own homograph thesis selects
the reading by SPO role (`E-SURFACE-FORM-COLLAPSE-1`,
`deepnsm/examples/homograph_collapse.rs:84-92`); that needs several readings
per token in the FSM (D-LXC-2). The count-based pick here is a separate,
interim leg, not a step toward proving the role-based one.

## Measured (orchestrator, offline over the two CSVs)

Method: Python over `lemmas_5k.csv` + `word_forms.csv`, fold `n|p→Noun,
v→Verb, j→Adj, a|d→Det, else Other`, counted pick = highest summed `wordFreq`
per folded state, tie order Noun > Verb > Adj > Det > Other. Words compared on
the union of both tables.

| variant | order | tags changed vs today |
|---|---|---|
| **B (committed)** | lemma table first (as today), counted pick replaces forms first-wins | **105** |
| A (alternative) | counted pick first, lemma table only as fallback | 288, of which 183 are lemma-derived (e.g. work V→N, thought N→V, used Adj→V, left Adj→V, open V→Adj) |

Whether either variant tags KJV better is **unmeasured**; the counts say
only how many tags move. B is committed because it is the smaller change and
deletes no evidence class.

Also found, both pre-existing and out of scope:
- `art` is already Noun today (`word_forms.csv:1047`), so `archaic_pos`
  (`bible_wave.rs:993`) never fires for COCA-known words → **D-LXC-9**.
- `Pos::Rel` appears only as match arms in `src/fsm.rs` and one FSM unit test
  (`fsm.rs:272`); no tagger in the crate's source (search space: `*.rs` under
  `crates/deepnsm-v2`, excluding `target/`; no other crate depends on
  deepnsm-v2) produces it → **D-LXC-10**.

## Checklist (D-LXC-1; D-LXC-7 rides with it)

- [ ] Tests first (G3), each seen red before the change.
- [ ] In `bible_wave` only: per `WordId`, read `LexicalEvidence::readings`,
      fold with the example's `coca_pos`, sum known `form_count` per state
      with `checked_add` (a state with any unknown count is unknown), pick
      the highest known sum with the fixed tie order; no known count → `None`.
- [ ] Order (variant B): lemma table → counted pick → `archaic_pos` →
      `Pos::Other`. The lemma layer is unchanged.
- [ ] Give the pick a test home that CI runs: `[[example]] name =
      "bible_wave"` with `test = true`, or a `tests/` file including the
      function via `#[path]`; G1 decides which.
- [ ] Measure on the KJV (Gutenberg #10 `pg10.txt`, input only, not
      committed): before and after for B, and for A as a reported
      alternative — triples, subjects, same-subject links, % beyond ±5/±8,
      tokens whose `Pos` changed, `WordFormsReport` (rows, stored,
      empty_surface, unrouted).
- [ ] `.claude/settings.json` `"attribution": {"commit": "", "pr": ""}`.
- [ ] Not touched: `lexical.rs` / `lib.rs` (library), `genre_shapes.rs`
      (deferred with its academic_20k layering, retained as D-LXC-4).

## Pre-registered gates

- G1 `cargo test --manifest-path crates/deepnsm-v2/Cargo.toml` green; a
  disable run under exactly that command turns the new tests red (proves they run).
- G2 clippy `--all-targets -D warnings` and `fmt --check` clean.
- G3 (a) `record` has n and v readings; (b) n+p fold sums, overflow errors;
  (c) an unknown count makes that state unknown; (d) the pick gives Noun for
  `record`, and a first-wins disable turns (d) red; (e) tie order pinned;
  (f) no known count → `None` → `archaic_pos`/Other; (g) stay-silent: a
  single-reading word keeps its tag; (h) a lemma-table word keeps its
  lemma-table tag under B (e.g. `work` stays Verb), and a lemma-only word
  (`stair`) keeps its tag.
- G4 #1299 routing and Cam96 invariance tests stay green.
- G5 **Blocking:** KJV before/after recorded in the PR and the board entry.
  No merge without them. If any of `E-WHOLE-BOOK-WAVE-1`'s four in-code gates
  flips, that is reported as a finding before merge.
- G6 The PR re-derives the offline tag-change count for B: exactly 105, or
  the PR fails and the difference is investigated.

## Board (same commit as the implementation)

- STATUS_BOARD: D-LXC-1 row flip.
- `.claude/board/entries/<date>-*.md` with the numbers, then
  `python3 .claude/tools/entries_index.py --write`.
- Append-only files get NEW dated entries that cite what they correct, never
  a line inserted beside an old one: a prepended entry citing
  `E-WHOLE-BOOK-WAVE-1` (LATEST_STATE), notes for D-SRS-1/3/4, TECH_DEBT item 3
  (`TECH_DEBT.md:613-620`, which this PR falsifies), and one note per
  downstream plan whose cited number moved (`INTEGRATION_PLANS.md:2113`,
  `plans/self-reasoning-substrate-v1.md:18-19,187`,
  `plans/literature-probe-ladder-v1.md:28,362-368`,
  `plans/dismech-causality-v3-v1.md:227`, `probes/README.md:104,108`).
  `EPIPHANIES-ARCHIVE` is frozen and not touched; `PROCESSED_THROUGH` not advanced.
- AGENT_LOG is written by the orchestrator only. LATEST_STATE and
  PR_ARC_INVENTORY on merge, as a separate hygiene commit.
- `SUPERSESSION-INDEX.md` regenerated last.

## Non-goals

- Any library change (F1, F7, F8).
- `genre_shapes` and academic_20k (D-LXC-4); multi-reading FSM (D-LXC-2).
- The archaic override (D-LXC-9); making `Pos::Rel` reachable (D-LXC-10).
- CausalEdge64 / mask-risc seam (D-LXC-6); history scrub (D-LXC-8).

## Council change ledger

v1 → v2 (5 savants: prior art, iron rules, code truth, cascade, different views):
1. Fold and pick moved out of the library into `bible_wave`: F7 + F8
   (`lexical.rs:50-54`) forbid the library mapping; a library pick is an
   interpretation, contra F1.
2. No known count → `None`, not a fabricated Noun.
3. v1's "only `stair` is lost" was wrong: 288 tag changes, 183 lemma-derived
   (re-computed by the orchestrator).
4. Fallback order made explicit; the `art` defect is pre-existing (filed D-LXC-9).
5. KJV before/after made a blocking gate.
6. Explicit test home: no CI job runs the examples; clippy only compiles them.
7. `genre_shapes` deferred (switching its fold alone changes nothing
   observable, per the code-truth read of `genre_shapes.rs:55-73`); retained as D-LXC-4.
8. Canon and board lists expanded; role-based prior art cited.
Recorded, not adopted: `lexicon.tsv` as a counted source — the asset is not in
the repo, so whether it carries counts is unknown.

v2 → v3 (3 reviewers: overclaim, dilution/collapse, firewall; 0 BLOCK, 9 FIX):
1. Dropping `lemmas_5k` reversed: variant B keeps the lemma layer (105 changes)
   and A is reported, not adopted; neither is framed as a correctness fix.
2. Lemma-keyed and surface-keyed evidence named as separate classes; the count
   pick and the role-based route named as separate legs.
3. `Pos::Rel` claim restated with its closed search space.
4. Measurement method stated; G6 made an exact equality.
5. G3(h) added for the lemma layer.
6. Append-only files get new dated entries, not inserted lines; AGENT_LOG
   orchestrator-only; merge-time hygiene separate.
7. No sourcing commentary on the KJV input.
8. F1–F8 labelled inherited; F7 marked as cited from the commit message.

## All candidate next parts (2026-09-26 .. 09-29 thread)

Ranked. Item 1 is the checklist above; the rest are recorded so the
convergence session and the operator see the whole set.

1. **D-LXC-1** · **Lexical-evidence consumer** (this plan). Smallest; closes the first-wins
   loss at its only caller; gives any later hydration step a real counted
   input.
2. **D-LXC-2** · **FSM takes several readings per token.** The honest end state of item 1:
   `Tagged` carries candidates, `parse_to_spo` resolves ambiguity with
   context instead of a counted guess. Needs an FSM change; separate PR.
3. **D-LXC-3** · **`lemmas_5k.csv` order.** The council kept the lemma
   table first (variant B, 105 tag changes). Switching to counted-first
   (variant A, 288 changes) is an operator decision, taken on the KJV
   numbers the D-LXC-1 PR reports.
4. **D-LXC-4** · **`academic_20k.csv` loader.** Blocked on a ruling: 3 pairs of rows share
   (word, PoS) with different counts. Disjoint and summable, or duplicates?
   Until ruled, the builder refuses the second as `DuplicateReading`.
5. **D-LXC-5** · **Counts for the Cam96 vocabulary.** `bible_vocab.txt` is surface-only (no
   counts, no PoS). The Tigris bucket (checked 2026-09-26 via the `AWS_*`
   credentials) holds only `academic_20k.csv` (identical to the committed
   copy) and a first-wins codebook TSV — no lemma-level or per-form counts.
   KJV-own counts would have to be computed from the corpus, not fetched.
6. **D-LXC-6** · **Mask-risc fold → next cycle.** From the 2026-09-26 `.claude/v3` review:
   the smallest missing executable seam across the cycle boundary is a
   mask-risc fold result at Lance version v entering cycle v+1 as a staged
   cast. Most value, not small, and it touches the CausalEdge64 area whose
   PRs #1293–#1295 were reverted. Needs its own plan.
7. **D-LXC-7** · **Repo attribution setting.** `.claude/settings.json`
   `"attribution": {"commit": "", "pr": ""}`; rides with item 1.
8. **D-LXC-8** · **History scrub of the `Opus 5.5` trailers.** Deferred by the operator:
   no force push. The two #1299 commits (`31f7d26f`, `eee1b17a`) and seven
   older ones keep the trailer; new commits carry none.

## Convergence

A second session holds additional ideas for this area. Its brief is
`.claude/prompts/deepnsm-v2-lexical-consumer-converge.md`. Its answer lands as
a v2 of this plan, never as parallel code.
