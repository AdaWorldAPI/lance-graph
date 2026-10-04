# 2026-10-04 — deepnsm-v2: a learned position table beats frequency on adjective/adverb in every language (D-LXC-16)

**Status:** MEASURED (`ud_pos_eval`, UD r2.15 en/de/fr; Animal Farm silver-tagged
by spaCy `en_core_web_sm` 3.8.0 from Project Gutenberg Australia `0100011.txt`,
sha256 d67d621f…, public domain in AU/EU, fetched not committed).

Operator direction (2026-10-04): research mode — if a position method does not
work yet, work harder; Animal Farm as the SPO / frequency-vs-position
reference; keep German case for morphology.

## Method

Hand-written clauses (D-LXC-14) captured a sliver. The principled version is
learned: for each ambiguous pair, count gold tags per **positional context**
— (previous readings, previous word is a copula, next readings) — on a
treebank's train split, with the evaluation lexicon's readings. The word
itself is never part of the key. Contexts with < 10 train tokens back off to
the more decisive half (next-only / previous-only). Scored alone (decide at
p ≥ 0.75 or ≤ 0.25) and combined with the word's frequency as log-odds
(`logit p_word + logit p_ctx − logit prior`).

Copulas: the forms train marks `cop` (measured). Case: a language
capitalises nouns when non-initial NOUN is > 90 % capitalised and other
words < 10 % (German 98.1 % / 0.8 % → kept; English, French folded).

## Results (all tokens of the pair; frequency vs position × frequency)

| setting | adj/adv freq | adj/adv pos×freq | noun/verb freq | noun/verb pos×freq |
|---|---|---|---|---|
| Animal Farm (COCA; table from UD English train) | 78.1 % | **88.4 %** | 82.6 % | **92.6 %** |
| UD English (COCA) | 89.7 % | **92.0 %** | 84.2 % | **90.8 %** |
| UD English (treebank lexicon) | 90.0 % | **90.6 %** | 90.7 % | **92.8 %** |
| UD German (case kept) | 80.8 % | **85.5 %** | 97.9 % | 97.6 % |
| UD French | 82.0 % | **92.0 %** | 88.9 % | **94.7 %** |

Adjective/adverb: 5 of 5 above frequency. Noun/verb: 4 of 5 (German is
−0.3 once case is kept, because capitalisation already halves its noun/verb
ambiguity, 693 → 333 tokens, and frequency is at 97.9 %).

On Animal Farm the hand clauses already beat or tied frequency (after
determiner 84.5 vs 78.3, before adjective 92.1 vs 81.6); they lost only on UD
English web text.

## Caveats

- Animal Farm labels are silver (spaCy, itself contextual) — they may favour
  contextual methods. Its table is learned on a different text (UD English).
- The table is a CC BY-SA derivative of UD (counts per context).

## OPEN

- Library form: a `PositionTable` type and whether the FSM should narrow on
  a confident combined posterior — not built; harness only.
- The KJV (German-casing silver) has not been scored with the table: its
  tagger's reading sets differ from the table's.
