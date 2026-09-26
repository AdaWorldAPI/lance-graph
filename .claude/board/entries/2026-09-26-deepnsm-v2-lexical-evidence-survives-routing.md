# 2026-09-26 — DeepNSM-v2 keeps counted lexical evidence beside routing

**Status:** MEASURED · DONE — `crates/deepnsm-v2/src/lexical.rs`

## The loss point
`PaletteVocab::from_frequency_ranked` (`vocab.rs`) admits surface strings, first wins. The counts and PoS behind the ranking never reach v2 storage. `examples/bible_wave.rs::load_pos` repeats the loss for PoS: `word -> one Pos`, counts dropped.

## What landed
`LexicalEvidence`, built against an existing `PaletteVocab` and never changing it. One `WordId` owns N `LexicalReading { pos: PosCode, lemma: Option<LemmaRef>, form_count: Option<u64> }`. `LemmaEntry { source_key, lemma, pos, count }` is keyed by the source's `lemRank`. Queries: `surface_count`, `surface_pos_count`, `lemma_pos_count`, `lemma_count`. All counts are `Option<u64>`: an empty field is unknown, a literal `0` is zero. Conflicting lemma entries and duplicate readings are refused, not first-wins. Loader: `load_word_forms_csv` (exact header `lemRank,lemma,PoS,lemFreq,wordFreq,word`).

## Measured on the committed COCA `word_forms.csv`
11,460 rows → 11,456 readings + 4 empty-surface rows (reported), 0 unrouted when the vocab covers every surface, 5,050 lemma entries (= distinct `lemRank`). 1,165 surfaces carry more than one reading, e.g. `record` n 120,048 / v 13,014 under lemma totals 187,057 / 51,375.

## Open
- The trained Cam96 path (`bible_vocab.txt`) is surface-only: no counts, no PoS. KJV routing gets evidence only when a caller joins a counted source to it.
- `academic_20k.csv` (surface × PoS × COCA-All/Acad, no lemma) has no loader. It holds 3 same-(word, PoS) row pairs with different counts, so a loader must decide whether those are disjoint before summing. The builder refuses a second (word, PoS, no-lemma) reading as `DuplicateReading` until that is decided.
