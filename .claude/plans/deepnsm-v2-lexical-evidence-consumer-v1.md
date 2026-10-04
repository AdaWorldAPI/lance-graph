# deepnsm-v2-lexical-evidence-consumer-v1

**Status:** CORRECTED 2026-09-30 — see "Correction: frequency is evidence,
not a lexical decision" below; it supersedes the counted pick, G3(b)-(f) and
G6. Originally: PROPOSAL. No code authorized. Written against `main` `5282dfa3`
(the #1299 merge). Rewritten 2026-09-29 after every claim was re-read in
source with the Read tool; the earlier council version relied on shell
searches and on files it never opened (change ledger at the end).

## Overview

#1299 made DeepNSM-v2 keep every counted PoS reading per `WordId`
(`crates/deepnsm-v2/src/lexical.rs`). Nothing reads it yet. The one tagger
that would benefit, `bible_wave::load_pos` (`examples/bible_wave.rs:1020-1042`),
still builds a `HashMap<String, Pos>` with `entry().or_insert_with`: first
`lemmas_5k.csv` keyed by the lemma string, then `word_forms.csv` keyed by the
surface string, first row wins, counts dropped.

This plan (**D-LXC-1**) changes that one consumer. The library does not
change: it preserves evidence and does not perform the CausalEdge64
epistemic transition.

## Where this sits in the DeepNSM → DeepNSM-v2 migration

Read in full before this rewrite:

- **The v2 rebuild** (`LATEST_STATE.md:3584-3587`,
  `E-DEEPNSM-V2-PALETTE-ARCHITECTURE-1`; `deepnsm-v2/Cargo.toml:7-15`).
  DeepNSM-v2 is a parallel rebuild on the V3 palette256² substrate; the `deepnsm`
  crate is untouched. The v1→v2 mapping keeps the "frequency-ranked vocab →
  6-state PoS FSM → SPO" signature and replaces the substrate under it.
  Tagging is part of the preserved signature, not of the replaced substrate.
- **The inbound-leg ruling** (`LATEST_STATE.md:3566`,
  `E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1`).
  DeepNSM-v2 encodes forward and emits a belief stream; reasoning lives in the
  lance-graph planner. `TD-DEEPNSM-V2-BELIEF-DUP` (`TECH_DEBT.md:1344`) records
  the V0 `belief.rs` arena as superseded. A counted PoS choice is inbound
  tagging, so it belongs in DeepNSM-v2; any belief or truth derived from the
  counts does not.
- **The V3 convergence plan** (`.claude/plans/deepnsm-v3-convergence-v1.md`,
  `E-V3-DEEPNSM-IS-THE-ENCODER-NOT-A-MIGRATION-1`, D-DNV-1..4 at
  `STATUS_BOARD.md:1231-1240`). DeepNSM fills tenants V3 already reserved; it
  is not ported. None of D-DNV-1..4 covers PoS or lexical counts, so D-LXC-1
  neither advances nor blocks them.
- **The Morton-comma facet design** (`.claude/plans/deepnsm-morton-comma-facet-v1.md`
  §0, §2, §3a; confirmed by `E-DEEPNSM-FACET-BLIND-CONVERGENCE-1`).
  A DeepNSM word's classid header is `norm(prefix, frequency, PoS)`; frequency
  and PoS are routing/header signals and must not enter the 6×(8:8) payload
  (PF=Payload is the measured defection). Counted PoS evidence is therefore
  header material. D-LXC-1 does not land anything in a facet; a later landing
  of this evidence goes into the header, never the payload.
- **The tagger history** (commits `ec50f07b` and `68955ecb`, both 2026-08-22;
  the deleted module read in full from `68955ecb^:crates/deepnsm-v2/src/lexicon.rs`).
  `ec50f07b` added `word_forms.csv` under the lemma table in one library
  `Lexicon` (lemma → forms → archaic → Other). It pinned "lemma-first: the
  forms table can only fill silence, never overrule" with the test
  `adding_forms_never_retags_a_word_the_lemmas_knew`, and measured
  40,767 → 70,393 triples over 31,102 verses. `68955ecb` deleted that module
  as redundant with `lance-graph-planner/examples/insight_coca_read.rs` and
  inlined the tagger into the two examples, without the tests.
  `TD-DEEPNSM-V2-SESSION-RESIDUE` item 3 (`TECH_DEBT.md:613-620`) records the
  layering as having no falsifier since.
- **The deleted module's stated premise:** "Both tables take the FIRST row for
  a given key — they are frequency-ranked, so the first is the dominant
  reading" (`lexicon.rs:101-102` as deleted). For `word_forms.csv` this is
  false. The file is ordered by lemma rank, not by `wordFreq`, and 259 surfaces
  have a first row that is not their most frequent reading (measured below).
  D-LXC-1 fixes exactly that premise and keeps the lemma-first rule.
  ⊘ 2026-09-30: struck. The premise is false, but replacing it with "the most
  frequent row wins" is the same kind of resolver. D-LXC-1 now measures the
  reading order and does not tag from it (see the Correction).
- **v1 prior art:** `deepnsm/src/vocabulary.rs:131-135,158-184` keeps the
  first occurrence for lemmas and adds forms only when absent (first-wins, no
  counts). `deepnsm/examples/homograph_collapse.rs:3-15,84-92` selects a
  homograph's reading by SPO role, not by count (`E-SURFACE-FORM-COLLAPSE-1`).
  The planner's `insight_coca_read.rs:96-108` keeps one (lemma, PoS) per
  surface with `HashMap::insert` (last row wins, no counts).

## Inherited rulings (not re-opened)

- F1 #1299 boundary: evidence, not transition; no dependency on causal-edge,
  cognitive-shader-driver, arm-discovery or the deprecated `deepnsm` crate.
- F2 Routing `WordId`s and Cam96 `codes[word_id]` do not move.
- F3 Integer counts; unknown = `None`; aggregates checked; unknown beats
  overflow (`lexical.rs:293-306`).
- F4 Each guard has a can-fire and a can-stay-silent test; disable runs turn
  tests red.
- F5 Search is navigation, not evidence; conclusions rest on reading the file.
- F6 Proposal only for now; no force push; no model names or attribution in
  commits or PR text.
- F7 Taggers stay in the examples (`68955ecb`; restated in
  `bible_wave.rs:976-979` and `genre_shapes.rs:200-203`).
- F8 `PosCode` is deliberately not mapped onto `fsm::Pos` in the library; "a
  consumer maps at its own boundary" (`lexical.rs:48-54`).
- F9 Lemma-first: the forms layer only fills silence, never overrules the
  lemma table (`ec50f07b`; docstring kept at `bible_wave.rs:1020-1022`).
- F10 Frequency and PoS are classid-header signals, never facet payload
  (`deepnsm-morton-comma-facet-v1.md` §2, §3a).
- F11 DeepNSM-v2 is the inbound leg; reasoning over its output lives in the
  lance-graph planner (`E-DEEPNSM-V2-IS-INBOUND-LEG-...`).

## Two evidence classes, kept apart

- **Lemma-keyed:** `lemmas_5k.csv`, one tag per lemma string, applied today to
  any surface token of the same spelling.
- **Surface-keyed, counted:** `word_forms.csv` through `LexicalEvidence`,
  every reading of a surface with its `wordFreq`.

Neither is a role-based reading. ⊘ 2026-09-30: "The counted pick is an
interim leg" is struck — frequency may not pick a reading, interim or not. The
role-based route (`E-SURFACE-FORM-COLLAPSE-1`) needs several readings per token
in the FSM, which is D-LXC-2.

## Measured

Method: Python over the committed `lemmas_5k.csv` and `word_forms.csv`, plus
`bible_vocab.txt` from the `v0.1.0-cam96-data` release. Surfaces are
lowercased as `load_pos` does. Fold: `n|p→Noun, v→Verb, j→Adj, a|d→Det,
else Other`. Counted pick: highest summed `wordFreq` per folded state, ties in
the order Noun > Verb > Adj > Det > Other.

| | over both tables | within `bible_vocab.txt` (12,543 words) |
|---|---|---|
| surfaces whose first `word_forms` row is not the counted pick | 259 | — |
| **variant B** (lemma table first, counted pick fills the rest; F9 holds) | 105 tags change | **25** |
| variant A (counted pick first, lemma table as fallback; breaks F9) | 288 (183 lemma-derived) | 141 |

The 25 in-vocabulary changes under B: bases, bears, cares, causes, changes,
closer, cooks, cries, drinks, fastest, flies, gains, kisses, locks, marks,
millions, mounts, nearer, promises, resting, rests, rolls, seals, sticks, traps.

Whether B tags the KJV better is **not measured**. These numbers count
changed tags, not correct ones.

Other facts read for this plan:

- `word_forms.csv` has 3 non-lowercase surfaces (`True`, `False`,
  `reElection`); `bible_vocab.txt` has none. `load_word_forms_csv` matches
  surfaces exactly (`lexical.rs:415-420,472`) and `load_pos` lowercases, so the
  consumer must lowercase before the lookup or those rows go unrouted.
- `academic_20k.csv` has exactly 3 duplicate (word, PoS) pairs with different
  counts: wastewater/n, disproportionately/r, instill/v.
- `archaic_pos` (`bible_wave.rs:993-1008`) runs only after the COCA lookup
  misses (`:149-153`). `art` is already Noun today (`word_forms.csv:1047`), so
  the archaic `art → Verb` entry never fires; pre-existing, filed D-LXC-9.
- `Pos::Rel` (`fsm.rs:57`) is consumed by `parse_to_spo` (`fsm.rs:142-205`).
  The in-crate taggers `coca_pos` (`bible_wave.rs:980-988`,
  `genre_shapes.rs:204-212`) and `archaic_pos` never return it, so no
  in-crate tagger produces it. `Tagged`/`parse_to_spo` are public, so
  external callers can. Filed D-LXC-10.
- The board's `E-WHOLE-BOOK-WAVE-1` numbers (`LATEST_STATE.md:3572`: 31,327
  triples, 23,145 verses) predate two changes to `bible_wave`: the OT-only
  truncation fix (`bible_wave.rs:85-104`) and the forms layering
  (`ec50f07b`: 70,393 triples over 31,102 verses). Current code does not
  reproduce 31,327, and D-LXC-1 is not what moved it.
  `plans/dismech-causality-v3-v1.md:226-228` already records these numbers as
  stale.

## Checklist (D-LXC-1; D-LXC-7 rides with it)

⊘ 2026-09-30: the counted-pick items below (per-`WordId` pick, resolution
order) record what was built and then removed; they are not the current
state. The current state is the Correction below.

- [x] Tests first (G3). Each was shown red by a disable run (see Results).
- [x] In `bible_wave` only: build `LexicalEvidence` with `load_word_forms_csv`
      against `nsm.vocab`, after lowercasing the `word` column.
- [x] Per `WordId`: read `readings`, fold each `PosCode` with the example's
      `coca_pos`, sum known `form_count` per state with `checked_add` (a state
      with any unknown count is unknown), take the highest known sum with the
      fixed tie order; no known count gives `None`.
- [x] Resolution order, variant B (F9): lemma table → counted pick →
      `archaic_pos` → `Pos::Other`. The lemma layer is unchanged.
- [x] Give the pick a test home that CI runs: `[[example]] name = "bible_wave"`
      with `test = true`.
- [x] Measure on the KJV (Gutenberg #10 `pg10.txt`, input only, not
      committed): before and after for B, and A as a reported alternative.
- [x] Record the tag-fold duplication: two copies of `coca_pos` remain
      (`bible_wave.rs`, `genre_shapes.rs`) under F7; a change to one is a change
      to both.
- [x] `.claude/settings.json` `"attribution": {"commit": "", "pr": ""}`.
- [x] Not touched: `lexical.rs`, `lib.rs`, `fsm.rs`, `genre_shapes.rs`.

## Results (2026-09-29)

⊘ 2026-09-30: this Code block and the "after (B)" column describe the
removed counted pick; kept as the historical measurement. Current code and
numbers: the Correction below.

**Code.** `crates/deepnsm-v2/examples/bible_wave.rs`:
- `counted_pos` implements the fold and pick.
- `Tagger` is lemma table → counted pick → `archaic_pos` → Other.
- `lowercase_word_column` fixes the capitalised COCA surfaces.
- `load_pos_first_wins` is the old rule, kept only for G6.
- Nine tests.

`crates/deepnsm-v2/Cargo.toml` adds `[[example]] name = "bible_wave"`
`test = true`.

**G1.** `cargo test --manifest-path crates/deepnsm-v2/Cargo.toml` runs 124 lib
tests plus the 9 new example tests; all pass. Each disable run below turned at
least one of them red under the example's tests:

| disable | red |
|---|---|
| first reading wins (the old rule) | 6 tests, including (d) |
| no sum across folded tags | (b) |
| unknown count treated as zero | (c) and (f) |
| ties go to the later state | (e) |
| no known count becomes Noun | (f) |
| counted pick before the lemma table | (h)/G7 |
| no lowercasing | (i) |
| overflow wraps | (b) |

A first attempt at the first-wins disable (take the first state in tie order)
stayed green. It was not the old rule, so it was replaced.

**G2.** clippy `--all-targets -D warnings` and `fmt --check` are clean.

**G4.** The #1299 invariance tests are unchanged and green.

**G5/G6: KJV.** The `v0.1.0-cam96-data` release artifacts were used. Before is
`main` `5282dfa3`; after is this branch. Every in-code gate passes in all three
runs.

| | before (first-wins) | **after (B)** | A (counts first) |
|---|---|---|---|
| verses | 31,102 | 31,102 | 31,102 |
| triples | 70,393 | **70,396** | 71,088 |
| distinct subjects | 1,227 | **1,237** | 1,243 |
| distinct predicates | 1,941 | **1,931** | 1,944 |
| same-subject links | 60,947 | **60,944** | 61,554 |
| beyond ±5 / ±8 | 60.3% / 52.7% | **60.3% / 52.7%** | 60.3% / 52.6% |
| in-vocab tags moved (G6) | — | **25** | 141 |

- **Token scope.** Of 771,176 in-vocabulary tokens, 107 change tag under B.
  The largest are bases 16, locks 15, promises 13 and flies 10.
- **`WordFormsReport`:** 11,460 rows, 4,284 readings stored, 4 empty
  surface, 7,172 not in the KJV vocabulary.
- ⊘ 2026-09-30: the reading below is struck. The 25 moved tags and 3 added
  triples were frequency changing semantics, which the Correction forbids.
- **Reading.** B is a small correction. It moves 25 words' tags, adds 3
  triples, and leaves the long-range shares unchanged. Whether the moved tags
  are more correct is still not measured. A moves 5.6× more tags and adds 695
  triples.
- **G7.** The lemma-first property is pinned again by
  `the_forms_layer_never_retags_a_lemma_table_word`. That is the falsifier
  `TD-DEEPNSM-V2-SESSION-RESIDUE` item 3 asked for.

### Amendment: frequency is the register (2026-09-29, operator-directed)

⊘ 2026-09-30: the storage order and coverage below stand; "frequency is the
register's preference" and `dominant_pos` are struck (see the Correction).

The counted pick summed counts at tag time. Frequency is a dimension, so it
belongs in the storage order, expressed as percentile coverage:
- `LexicalEvidenceBuilder::finish` stores each word's readings most frequent
  first (unknown last, ties in file order). `LexicalEvidence::coverage(id)`
  gives each reading's cumulative percentile coverage (integer `0..=100`,
  `None` when any count of the word is unknown). `changes`: noun first, 89.
- `bible_wave`'s `dominant_pos` reads position 0 when coverage is known.
  `counted_pos`, `PICK_ORDER` and per-state summing are removed.
- Behaviour dropped with the summing: folded tags no longer add (n 60 + p 50
  loses to v 100), and ties keep file order instead of Noun > Verb.
- KJV: identical to the summed run — 25 moved tags, 70,396 triples, same
  shares. The summing moved nothing on this corpus.
- This supersedes the "no library change" non-goal (F1): the change is storage
  order plus a derived integer, not interpretation.

## Correction: frequency is evidence, not a lexical decision (2026-09-30)

Operator ruling on #1304. Frequency calibrates ambiguity; it never chooses,
ranks for selection, or eliminates a reading. Neither count magnitude nor
source-row order may choose meaning or PoS.

**Kept (measurement):** the count-ordered storage, unknown-last and stable
ties; cumulative percentile coverage with `u128` arithmetic; the D-LXC-11
band population, rank rule and cuts; lowercasing; the `test = true` home.

**Removed (selection):** `dominant_pos` and every path where
`readings()[0]` becomes the tag. `Tagger::pos` now reads
`load_pos_legacy_first_wins` → `archaic_pos` → `Pos::Other` — `main`'s
`load_pos` verbatim, renamed so the boundary is visible. It reads no count.

**Inherited debt, not precedent:** the legacy map still depends on
source-row order (the first lemma row, the first form row). That is not an
authorized resolver; it stays until D-LXC-2 (several readings into the
parser) and D-LXC-3 (the lemma-table migration). F9 is unchanged in #1304
and is documented as this same inherited exception.

**G6 replaced.** The historical G6 = 25 (and 70,396 triples) measured
frequency changing tags — unintended semantic interference, not a
correction. The new G6 is a paired non-interference invariant, pinned by
`counts_change_evidence_never_the_readings_or_the_tag`:
- count changes MUST NOT change the reading set or the tag;
- count changes MAY change the ordering, coverage and statistics.
A disable run that re-derives the tag from `readings()[0]` turns exactly
that test red. KJV after the repair: 70,393 triples, 1,227 subjects,
1,941 predicates — identical to `main`.

**Gates now:** G3(a), (g), (h), (i) stand. G3(b)-(f) tested the counted
pick and are retired with it; their tests were deleted, and (f) became
`legacy_tagging_falls_through_to_archaic_then_other`. G6 is the invariant
above. G5 stands with the numbers above.

**Multiplicity.** #1304 stops at exact readings plus calibrated evidence.
No parser fold, `PosSet`, reading mask, Det/Adj licensing, or ambiguity
marking in the `Spo` high bits — the reading-set ABI boundary is specified
first.

### Sizing evidence for D-LXC-2 / D-LXC-3 (not a design)

Python simulation of a reading-set fold over the parser's three control
states, run on the KJV verse stream with the same vocabulary; not committed.
- F9 kept: 3,363 tokens carry more than one folded reading; at most 4 live
  parser configurations at once; 4,304 alternative triples.
- A Det/Adj licensing probe resolves 1,895 of those tokens (56%), giving
  71,400 triples.
- F9 lifted: 91,054 multi-reading tokens; peak 17 live configurations.
- At most 6 readings per surface, so a `u8` reading mask over a word's span
  fits.
- Configurations are bounded per clause (polynomial), not one register
  tuple per state.

## Pre-registered gates

⊘ 2026-09-30: G3(b)-(f) and G6 below are superseded by the Correction above.

- G1 `cargo test --manifest-path crates/deepnsm-v2/Cargo.toml` green; a
  disable run under exactly that command turns the new tests red.
- G2 clippy `--all-targets -D warnings` and `fmt --check` clean.
- G3
  - (a) `record` returns both n and v readings.
  - (b) An n+p fold sums, and overflow is an error.
  - (c) An unknown count makes that state unknown.
  - (d) The pick gives Noun for `changes`. Its first `word_forms.csv` row is
    the verb (`:707`, v 13,624) against the noun (`:853`, n 113,085), and it is
    not a lemma-table key, so first-wins gives Verb. A first-wins disable turns
    (d) red. `record` cannot serve here: its noun row comes first, so both
    rules pick Noun.
  - (e) The tie order is pinned.
  - (f) No known count gives `None`, then `archaic_pos`, then Other.
  - (g) Stay-silent: a single-reading word keeps its tag.
  - (h) F9 holds under B: a lemma-table word keeps its lemma-table tag
    (`work` stays Verb).
  - (i) A non-lowercase `word_forms` surface is routed after lowercasing.
- G4 #1299 routing and Cam96 invariance tests stay green.
- G5 **Blocking:** KJV before/after recorded in the PR and the board entry.
  No merge without them. If one of `bible_wave`'s in-code gates (G1, G1b,
  G2-G4) fails, report it before merge.
- G6 With `bible_vocab.txt` from `v0.1.0-cam96-data`, exactly 25 in-vocabulary
  tags change under B. Any other count fails the PR.
- G7 This PR is the falsifier `TD-DEEPNSM-V2-SESSION-RESIDUE` item 3 asks for:
  a test pins that the forms layer never retags a lemma-table word (the
  property of the deleted `adding_forms_never_retags_a_word_the_lemmas_knew`).

## Board (same commit as the implementation)

- STATUS_BOARD: flip the D-LXC-1 row.
- `.claude/board/entries/<date>-*.md` with the numbers, then
  `python3 .claude/tools/entries_index.py --write`.
- Append-only files get NEW dated entries that cite what they correct, never
  a line inserted beside an old one:
  - LATEST_STATE: the KJV numbers from the current pipeline, citing
    `E-WHOLE-BOOK-WAVE-1` as the older measurement. That older figure predates
    the forms layering, so D-LXC-1 is not what moved it.
  - TECH_DEBT item 3 closed by G7.
  - A note in each downstream plan whose cited number the new run changes
    (`INTEGRATION_PLANS.md:2113`, `plans/self-reasoning-substrate-v1.md:18-19,187`,
    `plans/literature-probe-ladder-v1.md:28,362-368`,
    `plans/dismech-causality-v3-v1.md:227`, `probes/README.md:104,108`).
  - `EPIPHANIES-ARCHIVE` is frozen; `PROCESSED_THROUGH` is not advanced.
- AGENT_LOG is written by the orchestrator only. LATEST_STATE and
  PR_ARC_INVENTORY entries land on merge, as a separate hygiene commit.
- `SUPERSESSION-INDEX.md` is regenerated last.

## Non-goals

- Any library change (F1, F7, F8).
- Landing counts or PoS in a V3 facet. When that happens they go in the
  classid header (F10); that is a separate plan.
- Any belief, truth value or reasoning over the counts (F11).
- `genre_shapes` and academic_20k (D-LXC-4); multi-reading FSM (D-LXC-2).
- The archaic override (D-LXC-9); making `Pos::Rel` reachable (D-LXC-10).
- CausalEdge64 or the mask-risc seam (D-LXC-6); history scrub (D-LXC-8).

## All candidate next parts

1. **D-LXC-1** · Lexical-evidence consumer (this plan).
2. **D-LXC-2** · The FSM takes several readings per token and resolves them by
   role (`E-SURFACE-FORM-COLLAPSE-1`). Needs an FSM change; separate PR.
3. **D-LXC-3** · Lemma-table order (⊘ 2026-09-30: neither A nor B may pick
   by count; the migration target is a reading set, not a new order). B keeps F9 (25 in-vocabulary changes);
   A drops it (141). Switching to A overrides F9 and is the operator's call,
   taken on the D-LXC-1 KJV numbers.
4. **D-LXC-4** · `academic_20k.csv` loader, plus the duplicate `coca_pos` in
   `genre_shapes.rs`. Blocked on a ruling about the 3 duplicate (word, PoS)
   pairs: disjoint and summable, or duplicates? Until ruled, the builder
   refuses the second as `DuplicateReading`.
5. **D-LXC-5** · Counts for the Cam96 vocabulary. `bible_vocab.txt` is
   surface-only (no counts, no PoS); KJV-own counts would have to be computed
   from the corpus.
6. **D-LXC-6** · Mask-risc fold result at Lance version v entering cycle v+1
   as a staged cast. Named by the 2026-09-26 `.claude/v3` review as the
   smallest missing cycle-boundary seam; not re-verified in this rewrite. Most
   value, not small, and in the CausalEdge64 area whose PRs #1293–#1295 were
   reverted. Needs its own plan.
7. **D-LXC-7** · Repo attribution setting (rides with item 1).
8. **D-LXC-8** · History scrub of the model-naming `Co-Authored-By`
   trailer. 120 commits on `main` carry it (`git log --grep`, 2026-09-29).
   Deferred by the operator: no force push. New commits carry none.
9. **D-LXC-9** · `archaic_pos` cannot override COCA-known KJV words (`art`).
10. **D-LXC-10** · No in-crate tagger produces `Pos::Rel`.

## Change ledger

**5+3 council (2026-09-29).**
- The 5: prior art, iron rules, code truth, cascade, different views.
- The 3: overclaim, dilution/collapse, firewall. They returned 0 BLOCK and 9 FIX.
- Outcomes: the fold and pick moved out of the library into `bible_wave`; no
  known count gives `None`, not a fabricated Noun; the lemma table stays
  first; the KJV before/after is blocking; an explicit test home; `Pos::Rel`
  narrowed to in-crate taggers; `changes` as the first-wins falsifier.

**Rewrite with Read (2026-09-29, operator-directed).** The council version
rested on shell `grep`/`sed`/`head`/`tail` output and missed the migration
documents. Corrections:
1. Migration context added from full reads: the v2 rebuild, the inbound-leg
   ruling, the V3 convergence plan, the Morton-comma facet design, and the
   tagger history. This adds F9–F11.
2. Lemma-first is a pinned rule (F9), not the council's own preference.
   Variant A is recast as an F9 override.
3. The deleted module's "first row is dominant" premise is named, and shown
   false for `word_forms.csv` (259 surfaces).
4. Counts re-measured against the real vocabulary: 25 (B) and 141 (A) within
   `bible_vocab.txt`. G6 now pins 25.
5. The canon's 31,327 is shown to predate the forms layering; the board plan no
   longer implies D-LXC-1 moved it.
6. New gates G3(i) (lowercasing; 3 non-lowercase surfaces) and G7 (restores
   the deleted monotonicity test).
7. The private-bucket reference in D-LXC-5 is removed from this public repo.
8. D-LXC-8 corrected from "7 older commits" to 120 commits on `main`.
9. Every file:line re-read with the Read tool.

## Convergence

A second session holds additional ideas for this area. Its brief is
`.claude/prompts/deepnsm-v2-lexical-consumer-converge.md`. Its answer lands as
a v2 of this plan, never as parallel code.

## D-LXC-2 landed in PR (2026-10-03)

DECISION 2026-10-03 (operator): D-LXC-3 = B, the lemma table order is kept;
English data first, with the FSM reading-set ABI language-neutral.

- `fsm::parse_readings` takes a `PosSet` per token and keeps what structure
  cannot separate (certain vs alternative triples, per-token survivors).
  The one elimination is relative Det/Adj → Verb licensing.
- `coca` holds the single COCA letter fold. ⊘ F7 and F8 are narrowed, not
  struck: the per-corpus tagger stays in the example, `lexical` still stores
  raw `PosCode`, and the fold moved to a boundary module so the copies in
  `bible_wave`, `genre_shapes` and tesseract-rs's `probe_token_seam` can
  converge on it.
- KJV numbers and the OPEN points: board entry
  `2026-10-03-deepnsm-v2-multi-reading-fsm.md`.

## D-LXC-13: position decides noun/verb homographs (2026-10-04)

Direction (operator, 2026-10-04): a word between subject and object must not
become a noun because COCA counts it as a noun more often. D-LXC-3's B stands
for row order, but the first row no longer settles a noun/verb homograph:
`coca::predicate_alternatives` widens it, and `fsm`'s slot rule (after a fresh
subject) and licensing (after a determiner) pick by position. A sentence-level
"needs a predicate" rule was measured and rejected (KJV fragments). Numbers
and OPEN points: board entry `2026-10-04-deepnsm-v2-predicate-by-position.md`.

## D-LXC-14: the position rules on gold tags (2026-10-04)

`examples/ud_pos_eval.rs` scores every rule against Universal Dependencies
gold tags (English, German, French), with the typology measured from each
train split rather than written per language. Noun/verb position rules beat
COCA's frequency pick (96.7 % vs 85.1 % on decided tokens); the adjective/adverb
clauses do not beat frequency in any language consistently and ship disabled.
Adjective licensing is gated on the measured typology. Numbers: board entry
`2026-10-04-deepnsm-v2-position-rules-on-ud-gold.md`.

