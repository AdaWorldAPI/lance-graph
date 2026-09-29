# DeepNSM-v2 counted PoS pick: tag-change counts (2026-09-29)

**Status:** MEASURED (offline, over committed CSVs plus the released KJV
vocabulary). No code changed.

The removed `deepnsm-v2/src/lexicon.rs` (`68955ecb^`) assumed "the first row
[of a frequency-ranked table] is the dominant reading". For `word_forms.csv`
that is false: the file is ordered by lemma rank, and 259 surfaces have a
first row that is not their most frequent reading (e.g. `changes`: first row
v 13,624 at `:707`, noun n 113,085 at `:853`).

Replacing that first-wins layer in `bible_wave::load_pos` with a counted pick
(highest summed `wordFreq` per folded parser state, tie order Noun > Verb >
Adj > Det > Other) changes:

| variant | over both tables | within `bible_vocab.txt` (12,543) |
|---|---|---|
| B: lemma table first, as pinned by `ec50f07b` (the plan's choice) | 105 | **25** |
| A: counted pick first, lemma table as fallback | 288 (183 lemma-derived) | 141 |

Method: Python over `crates/deepnsm/word_frequency/{lemmas_5k,word_forms}.csv`
and `bible_vocab.txt` from release `v0.1.0-cam96-data`. Surfaces are
lowercased as `load_pos` does. Fold: `n|p→Noun, v→Verb, j→Adj, a|d→Det,
else Other`. Whether either variant tags the KJV better is not measured; the
KJV before/after is gate G5 of D-LXC-1.

Pre-existing observations filed with the plan:
- `archaic_pos` never fires for COCA-known words such as `art` (D-LXC-9).
- No in-crate tagger in deepnsm-v2 produces `Pos::Rel`; external callers can
  still pass it (D-LXC-10).

Plan: `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md`.
