# DeepNSM-v2 counted PoS pick: tag changes and the KJV run (2026-09-29)

**Status:** MEASURED. D-LXC-1 is implemented in `bible_wave` (this PR).

The removed `deepnsm-v2/src/lexicon.rs` (`68955ecb^`) assumed "the first row
[of a frequency-ranked table] is the dominant reading". For `word_forms.csv`
that is false: the file is ordered by lemma rank, and 259 surfaces have a
first row that is not their most frequent reading. For example, `changes` has
its verb row first (v 13,624 at `:707`), but the noun row (`:853`) counts
113,085.

D-LXC-1 replaces that first-wins layer with a counted pick: the highest summed
`wordFreq` per folded parser state, ties in the order Noun > Verb > Adj > Det
> Other. The lemma table stays first.

| | before (first-wins) | after (B, shipped) | A (counts first) |
|---|---|---|---|
| in-vocabulary tags moved | — | 25 | 141 |
| KJV triples (31,102 verses) | 70,393 | 70,396 | 71,088 |
| distinct subjects | 1,227 | 1,237 | 1,243 |
| same-subject links beyond ±5 / ±8 | 60.3% / 52.7% | 60.3% / 52.7% | 60.3% / 52.6% |

Under B, 107 of 771,176 in-vocabulary KJV tokens change tag. The offline
predictions (25 and 141 words) match the in-code G6 count exactly. Whether the
moved tags are more correct is not measured.

Method: Python over `crates/deepnsm/word_frequency/{lemmas_5k,word_forms}.csv`
and `bible_vocab.txt` (release `v0.1.0-cam96-data`); `bible_wave` run on
Gutenberg #10 with the same release artifacts, `main` `5282dfa3` against this
branch. The KJV text is input only and is not committed.

Reproduce: fetch the release assets per `crates/deepnsm-v2/data/README.md`,
then `DEEPNSM_V2_DATA=<dir> cargo run --release --example bible_wave --
<pg10.txt>`. The in-code G6 count recomputes the 25 and asserts it. The 141
(variant A) was measured once in Python and is not re-run by any gate. The
`bible_vocab.txt` used had SHA-256
`8dc3a65dcd3af38a2f53308fb96ef5fb5b34336c14587f6971b75ac966c3e212` (recorded
as provenance, not a gate).

Pre-existing observations filed with the plan:
- `archaic_pos` never fires for COCA-known words such as `art` (D-LXC-9).
- No in-crate tagger produces `Pos::Rel`; external callers can still pass it
  (D-LXC-10).

**Amended (same day).** The summed pick was replaced by the register:
`LexicalEvidence` stores readings most frequent first with a cumulative
percentile coverage, and the tagger reads position 0. The KJV run is identical
to "after (B)" above (25 moved, 70,396 triples); the summing moved nothing.

Plan: `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md`.

**Correction (2026-09-30, operator ruling on #1304).** Frequency is evidence,
not a lexical decision. The 25 moved tags and the 70,396 triples above record
frequency changing the tag: unintended semantic interference, not a
correction. The counted pick and `dominant_pos` are removed. The tag is
`main`'s tagging again, named `load_pos_legacy_first_wins`; its dependence on
source-row order is inherited debt pending D-LXC-2/D-LXC-3, not an authorized
resolver. KJV after the repair: 70,393 triples, 1,227 subjects, 1,941
predicates — identical to `main`. G6 is now a paired invariant pinned by
`counts_change_evidence_never_the_readings_or_the_tag`: count changes may move
the order and the coverage, never the reading set or the tag. The table above
is kept as the historical measurement.
