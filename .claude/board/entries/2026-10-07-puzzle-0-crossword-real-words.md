# 2026-10-07 — D-PUZZLE-0 step 2b: real-word crosswords (COCA, optional DeReKo)

**Probe:** `cognitive-shader-driver/examples/crossword_real_words_probe.rs`,
over `examples/shared/population_fold.rs` (unchanged).
**Status:** MEASURED (this machine), TEST-PINNED (10 tests, 3 disable runs red).

## What changed from step 2

The step-2 crossword drew its words from a synthetic 3-letter alphabet. This
probe fills the same 5×5 template from a real dictionary: the 1000 most
frequent words per slot length.

- **English:** COCA word forms from the committed
  `crates/deepnsm/word_frequency/word_forms.csv`, the file DeepNSM-v2's
  lexical layer reads. Alphabetic ASCII surfaces, scored by their largest
  `wordFreq`. COCA holds only 902 such four-letter forms, so that length is the
  whole list.
- **German:** the DeReKo-2014 STTS frequency list, only when `DEREKO_PATH`
  names the file at run time; skipped otherwise. Proper nouns (`NE`) are
  excluded and frequencies summed per lowercase form. DeReKo is © IDS
  Mannheim, CC BY-NC 3.0. No German word, list or derived artifact is
  committed; this entry carries counts only.

The probe reads the COCA CSV itself. DeepNSM-v2's `load_word_forms_csv` routes
each surface through a `PaletteVocab` (the KJV-trained vocabulary), which would
drop most of the words a crossword needs.

An instance is a random fill of the template from the dictionary. Given slots
are then added in random order until the dictionary admits exactly one fill
(backtracking). Law, claims, questions, fold and oracle are the step-2 ones.

## Measured (release, `avx2=true avx512f=false`, median of 7)

| | English COCA | German DeReKo |
|---|---|---|
| edges | 1,002,858 | 1,000,847 |
| instances | 1,338 | 1,483 |
| build time | 4.2 s | 4.7 s |
| given | 5,673 | 6,545 |
| forced | 1,574 | 1,713 |
| entailed, not yet forced | 6,133 | 6,572 |
| live candidates | 989,478 | 986,017 |
| filter fold, ns/edge | 0.34 | 0.36 |
| histogram, ns/edge | 2.08 | 2.02 |
| decode oracle, ns/edge | 3.87 | 3.83 |

Every count is equal three ways (filter, histogram, decode) and to the
instances' own counters. Every instance holds exactly 10 asserted claims.

Givens needed for uniqueness (givens: instances), EN:
`1:102 2:249 3:214 4:202 5:172 6:169 7:122 8:77 9:31`. DE:
`1:94 2:240 3:246 4:221 5:212 6:189 7:140 8:94 9:47`. These are minimal
along a random order, not globally minimal; a smarter order would need fewer.

The filter and the oracle cost the same as for Sudoku and the synthetic
crossword. The histogram is slower again (2.0 ns against 1.7 synthetic and
1.0 Sudoku). That fits the step-2 hypothesis: here 98.6 % of edges share one
code, against 77 % there. It fits, but nothing here measures the cause.

## Falsifiers

- Uniqueness check returning 1 always → `instances_are_unique_and_givens_are_needed`
  red (the givens must be needed for at least one instance).
- Index ignoring fixed letters → four tests red, including
  `the_index_agrees_with_a_scan` (bitset index against a direct scan).
- `NE` rows kept → `dereko_rows_are_parsed_by_tag_and_summed` red.

## Open

- German runs only where the DeReKo file is present; CI covers English only.
- Chess in `stockfish-rs` and how it reaches the shared fold: still open.
