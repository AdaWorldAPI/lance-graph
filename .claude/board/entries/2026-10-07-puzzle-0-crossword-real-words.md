# 2026-10-07 — D-PUZZLE-0 step 2b: real-word crosswords (DeepNSM vocabularies, optional DeReKo)

**Probe:** `cognitive-shader-driver/examples/crossword_real_words_probe.rs`,
over `examples/shared/population_fold.rs` (unchanged).
**Status:** MEASURED (this machine), TEST-PINNED (11 tests, 6 disable runs red).

## What changed from step 2

The step-2 crossword drew its words from a synthetic 3-letter alphabet. This
probe fills the same 5×5 template from a real vocabulary. The dictionary is
every word of a slot's length in that vocabulary.

- **English:** exactly the two vocabularies DeepNSM reads, both committed
  under `crates/deepnsm/word_frequency/`. These are v1's 4096-word table
  (`word_rank_lookup.csv`, ranks 1..=4096) and v2's academic list
  (`academic_20k.csv`), taken as their union. Alphabetic ASCII only, scored by
  the larger COCA frequency. `word_forms.csv` is not a source. Dictionary:
  1,250 four-letter, 1,697 five-letter words.
- **German:** the 20,000 most frequent DeReKo-2014 forms, after excluding
  proper nouns (`NE`) and non-alphabetic forms and summing frequencies per
  lowercase form. Read only when `DEREKO_PATH` names the file at run time and
  skipped otherwise. DeReKo is © IDS Mannheim, CC BY-NC 3.0. No German word,
  list or derived artifact is committed; this entry carries counts only.
  Dictionary: 802 four-letter, 1,466 five-letter words.

The first version of this probe (`cc887e3`) used the top 1000 `word_forms.csv`
surfaces per length. It was replaced on review because that list is not a
vocabulary DeepNSM uses.

An instance is a random fill of the template from the dictionary. Given slots
are then added in random order until the dictionary admits exactly one fill
(backtracking). Law, claims, questions, fold and oracle are the step-2 ones.

## Measured (release, `avx2=true avx512f=false`, median of 7, one run)

| | English (v1 4096 + academic) | German (DeReKo top 20k) |
|---|---|---|
| edges | 1,005,146 | 1,000,074 |
| instances | 1,209 | 1,390 |
| build time | 7.4 s | 6.7 s |
| given | 5,370 | 6,118 |
| forced | 1,278 | 1,400 |
| entailed, not yet forced | 5,442 | 6,382 |
| live candidates | 993,056 | 986,174 |
| filter fold, ns/edge | 0.35 | 0.36 |
| histogram, ns/edge | 2.23 | 2.10 |
| decode oracle, ns/edge | 4.01 | 3.79 |

Every count is equal three ways (filter, histogram, decode) and to the
instances' own counters. Every instance holds exactly 10 asserted claims.

Givens needed for uniqueness (givens: instances), EN:
`1:23 2:246 3:208 4:204 5:161 6:129 7:109 8:88 9:41`. DE:
`1:53 2:250 3:263 4:208 5:200 6:152 7:132 8:80 9:52`. These are minimal
along a random order, not globally minimal.

The filter and the oracle cost the same as for Sudoku and the synthetic
crossword. The histogram is slower again (2.1–2.2 ns, against 1.7 synthetic
and 1.0 Sudoku). That fits the step-2 hypothesis: here 98.6–98.8 % of edges
share one code, against 77 % there. It fits, but nothing here measures the
cause.

## Falsifiers (disable-verified red)

- Uniqueness check returning 1 always → `instances_are_unique_and_givens_are_needed`.
- Index ignoring fixed letters → 4 tests, including `the_index_agrees_with_a_scan`.
- `NE` rows kept → `dereko_rows_are_parsed_by_tag_and_summed`.
- 4096 rank cut removed → `the_english_dictionary_is_deepnsm_v1_plus_academic`
  (words ranked past 4096 and absent from the academic list leak in).
- Academic list dropped → the same test (academic-only words missing).
- German 20k truncation removed → `german_keeps_the_top_20k_forms`.

## Open

- German runs only where the DeReKo file is present; CI covers English only.
- Chess in `stockfish-rs` and how it reaches the shared fold: still open.
