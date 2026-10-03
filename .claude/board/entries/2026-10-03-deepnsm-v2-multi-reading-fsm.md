# 2026-10-03 — deepnsm-v2: the FSM takes every reading (D-LXC-2)

**Status:** MEASURED (`cargo run --release --example bible_wave -- pg10.txt`,
`v0.1.0-cam96-data` assets).

## What changed

- `fsm::parse_readings(&[Reading]) -> ReadingParse`. A token enters with a
  `PosSet` (bitmask over the FSM alphabet; empty = lexically unknown). The
  parser steps a set of configurations with the unchanged transition table
  (`Core::step`, now shared with `parse_to_spo`), merges configurations with
  equal registers and equal triples, and reports per-sentence `certain`
  triples (on every surviving path), `alternative` triples (on some), and
  each ambiguous token's `entered` / `survived` readings.
- One elimination primitive: a verb reading is not licensed right after a
  determiner or adjective. It is relative — it never removes a token's last
  admissible reading — so single-reading words are never rejected and
  one-reading-per-token input parses exactly as `parse_to_spo`.
- `coca::{fsm_pos, fsm_pos_tag, reading_set}` — the one COCA letter → FSM tag
  fold; the copies in `bible_wave` and `genre_shapes` now call it.
  `reading_set` returns `None` for a word with no reading (unknown ≠ zero).
- Frequency is not read anywhere in this path.

## KJV (`bible_vocab.txt`, lemma table order kept per D-LXC-3 = B)

| | value |
|---|---|
| tokens / single / ambiguous / unknown | 771,176 / 683,805 / 3,363 / 84,008 |
| ambiguous words | 141 (= the D-LXC-11 band population) |
| narrowed / still ambiguous | 1,908 / 1,455 |
| legacy tag eliminated | 179 tokens, 35 words (means 31, wonders 24, holds 20, locks 13, promises 12, …); 14 of the 35 are on the D-LXC-3 25-word list |
| triples: legacy → certain + alternative | 70,393 → 69,670 + 1,716 |
| legacy → alternative / lost / new certain | 732 / 113 / 122 |
| peak configurations / overflow flushes | 16 / 0 |
| unexplained changes | 0 (the run KILLs on any) |

Licensing disabled: 0 narrowed, 3,363 still ambiguous, 4,289 alternatives —
the rule is what resolves the 1,908.

Whether the narrowed readings are *correct* is not measured; these count
changes, not accuracy.

## Superseded

Plan F7 ("taggers stay in the examples") and F8 ("a consumer maps at its own
boundary") are narrowed: the per-corpus tagger (lemma table, archaic list)
stays in the example; the COCA letter fold moves into `deepnsm_v2::coca`, a
boundary module, so the three copies converge. `lexical` still stores raw
`PosCode`.

## OPEN

- Whether Det/Adj licensing is right for COCA `d` words used pronominally
  ("some say", "all went") — single-reading verbs after them are safe by the
  relative rule; an ambiguous verb after them is narrowed.
- Downstream (`--export`) carries certain triples only; alternatives are
  counted, not exported.
- `tesseract-paperless::consistency` still bridges v1 tags through
  `map_pos`; moving it to `reading_set` + `parse_readings` waits on this
  merging (tesseract CI builds against lance-graph main).
