# 2026-10-04 — deepnsm-v2: the position rules scored on gold tags in three languages (D-LXC-14)

**Status:** MEASURED (`cargo run --release --example ud_pos_eval -- TRAIN TEST [--coca DIR]`,
Universal Dependencies r2.15: UD_English-EWT, UD_German-GSD, UD_French-GSD,
CC BY-SA 4.0, fetched, never committed; sha256 in the example's run notes below).

Operator direction (2026-10-04): decide adjective vs adverb by position as
D-LXC-13 does noun vs verb, feed the TEKAMOLO tenant, and validate across
languages instead of writing rules per language.

## What changed

- `Pos::Adv` — the 8th tag (bit 7, after `Stop`, so no earlier bit moves).
  Skipped in the core like `Other`; opens no nominal group. COCA `r` folds to
  it. KJV numbers are unchanged (a `{Other}` set became `{Adv}`).
- `Typology { adjective: AdjectiveOrder, adjective_opens_nominal,
  attribute_rules }` + `parse_readings_with`. `parse_readings` =
  `Typology::ENGLISH`. Determiner licensing applies everywhere; adjective
  licensing only where `adjective_opens_nominal`.
- `AttributeRule` (4 clauses) + `attribute_rule` (which clause fires) +
  `attribute_readings`. As shipped no clause narrows (`ENGLISH.attribute_rules
  = &[]`): each was measured below frequency (table below).
- `examples/ud_pos_eval.rs` — the gold harness. The lexicon is the treebank's
  train split (or COCA with `--coca`), folded by one UPOS table; the typology
  is measured from train (`amod` head direction; share of adjectives directly
  followed by a verb), not written per language.

## Measured typology

| | adjective before its noun (`amod`) | adjective followed by a verb | → typology |
|---|---|---|---|
| English | 97.0 % | 1.2 % | Before, opens nominal |
| German | 98.7 % | 7.1 % | Before, does not |
| French | 30.4 % | 7.4 % | Both, does not |

Gating adjective licensing on this fixed the cross-language failures: with
English licensing everywhere, German "kept Noun" was 37.5 % right ("sehr
lecker **war**", "möglich **ist**") and French 50 %.

## Noun/verb: position vs frequency on the same tokens

| lexicon / language | decided | position right | frequency right |
|---|---|---|---|
| **COCA → English EWT** | 457 / 1,632 | **96.7 %** | 85.1 % |
| — licensing (kept Noun) | 349 | 98.0 % | 85.4 % |
| — slot rule (kept Verb) | 108 | 92.6 % | 84.3 % |
| treebank → English | 155 / 2,921 | 89.0 % | 89.7 % |
| treebank → German | 46 / 693 | 87.0 % (slot 100 %, 36) | 91.3 % |
| treebank → French | 30 / 524 | 100 % (slot 100 %, 28) | 90.0 % |

With COCA — the lexicon `bible_wave` and tesseract read — position beats
COCA's frequency pick by 11.6 points on the tokens it decides
(position-then-frequency 87.4 % vs frequency 84.2 % over all 1,632). This
supports D-LXC-13. With a treebank lexicon (every tag the form ever carried,
in-domain counts) the advantage is gone: almost every word "can be a noun".

## Adjective/adverb: each clause vs frequency on the same tokens

| clause | COCA → English | treebank English | German | French |
|---|---|---|---|---|
| after determiner → Adj | 89.3 vs 86.0 (121) | 1 token | 1 token | 1 token |
| before adjective → Adv | 86.7 vs 100 (60) | 63.4 vs 89.6 (164) | 75.0 vs 68.8 (32) | 1 token |
| subject _ verb → Adv | 93.1 vs 86.2 (29) | 29.6 vs 92.6 (108) | 57.9 vs 94.7 (19) | 2 tokens |
| next to noun → Adj | 79.8 vs 92.9 (84) | 85.6 vs 93.6 (327) | 38.3 vs 81.5 (81) | 61.5 vs 82.1 (39) |

No clause beats frequency in every setting; the two COCA wins are 4 and 2
tokens. Position does not decide adjective vs adverb better than frequency,
so no clause is enabled.

## Not done, and why

- `lance-graph-mask-risc`: the rules read a 3-token window of the borrowed
  `&[Reading]` in place; no plane is built and nothing is copied. A mask
  program would need ndarray in this zero-dependency crate and either a
  materialised reading lane or a byte view of `Reading`. Worth it only for a
  bulk pass (archive scale), with a measured clause to run.
- Animal Farm (1945): under US copyright until 2041, so not fetched; the UD
  treebanks give gold tags, which a plain text does not.
- TEKAMOLO producer (`insight_spo_tekamolo_read`, planner) still uses its cue
  lists; `Pos::Adv` makes a COCA adverbial reading visible to it. Wiring it is
  a separate change.

## OPEN

- German determiner licensing: 10 decided, 4 right — not inspected.
- The adjective/adverb ambiguity is lexical more than positional; a lexical
  class (temporal / manner / place adverb) is what the TEKAMOLO lanes need,
  not a position rule.

UD inputs (r2.15, sha256): en_ewt train `a049fc40…c566`, test `0612e459…036`;
de_gsd train `cb1326ed…c77`, test `9d6776ee…465f`; fr_gsd train `a539328e…30e`,
test `f786c60a…19de`.
