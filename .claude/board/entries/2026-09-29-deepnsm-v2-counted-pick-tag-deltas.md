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

Pre-existing observations filed with the plan:
- `archaic_pos` never fires for COCA-known words such as `art` (D-LXC-9).
- No in-crate tagger produces `Pos::Rel`; external callers can still pass it
  (D-LXC-10).

**Amended (same day).** The summed pick was replaced by the register:
`LexicalEvidence` stores readings most frequent first with a cumulative
percentile coverage, and the tagger reads position 0. The KJV run is identical
to "after (B)" above (25 moved, 70,396 triples); the summing moved nothing.

Plan: `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md`.
