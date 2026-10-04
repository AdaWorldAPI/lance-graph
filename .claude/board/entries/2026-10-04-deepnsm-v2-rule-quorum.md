# 2026-10-04 — deepnsm-v2: literal rules, priorities and a quorum for word-class homographs (D-LXC-19)

**Status:** MEASURED (`UD_RULES=1 [UD_DE_INVENTORY=DIR] cargo run --release
--example ud_pos_eval -- TRAIN TEST [--coca ../deepnsm/word_frequency]`, UD r2.15
en_ewt / de_gsd / fr_gsd, Animal Farm spaCy silver; debug-0 profile).

Operator (2026-10-04): instead of picking one rule, build ~20 rules with
priorities and a quorum; collect the literal grammar rules (Wechsel, case,
TEKAMOLO).

## What was built (`examples/ud_pos_eval.rs`, eval harness only)

- **Rules:** 14 for noun/verb and 16–20 for adj/adv.
  - three learned position tables (neighbours, the 2³ answered-question mask, both);
  - frequency, split at a 0.9 share;
  - determiner, adjective, slot, infinitive, modal and copula rules;
  - the four attribute clauses;
  - English `-ly`;
  - German capitalisation;
  - with `UD_DE_INVENTORY`: German inflection against the lemma (uninflected → adv; lemma + ending → adj; uninflected right before a noun → adv) and TEKAMOLO adverbial cues.
- **Training:** tables learn on 90 % of train. Each rule's precision is measured on the other 10 %, and that held-out precision is its weight.
- **Three decisions on test:**
  - priority — the highest-precision rule that fires;
  - summed quorum — Σ ±logit(precision);
  - joint quorum — a logistic regression over all votes plus the frequency log-odds, fitted on held-out.
- **German inventories:** built with `build_de_codebook.py` from `de_gsd-ud-train.conllu` only. The release `de-codebook` was built from GSD + HDT, and its manifest does not name the splits, so it may contain the test sentences. It is not used against GSD test.

## Results (all tokens of the pair; the same run's position × frequency for reference)

| run | pair | frequency | position × frequency | priority | summed quorum | joint quorum |
|---|---|---|---|---|---|---|
| UD English (train lexicon) | noun/verb | 89.5 | 92.7 | 93.3 | 91.4 | **93.7** |
| | adj/adv | 89.0 | 90.1 | 88.8 | 85.4 | **90.4** |
| UD English (COCA) | noun/verb | 84.2 | **91.1** | 89.0 | 88.1 | 90.5 |
| | adj/adv | 89.3 | 91.0 | 88.5 | 87.1 | **92.0** |
| Animal Farm (COCA) | noun/verb | 82.6 | 91.6 | 90.8 | 89.4 | **91.9** |
| | adj/adv | 77.1 | **88.2** | 86.5 | 85.7 | 84.8 |
| German (train lexicon) | noun/verb | 98.2 | 97.6 | 97.3 | 84.9 | **98.8** |
| | adj/adv, + inventories | 79.3 | **82.9** | 82.5 | 81.2 | 81.6 |
| French (train lexicon) | noun/verb | 89.9 | 94.8 | 94.0 | 89.7 | **94.8** |
| | adj/adv (50 tokens) | 82.0 | 90.0 | 88.0 | 88.0 | **90.0** |

Train-lexicon runs build the lexicon from the 90 % fit split. Their quorum
token sets are therefore a few percent smaller than the position-table line's,
and the frequency column is the quorum line's own.

## Findings

1. **A summed quorum fails.** The three position tables agree almost always,
   so summing logits counts one opinion three times. German noun/verb drops to
   84.9 % against 98.2 % frequency. The joint fit spreads one weight across
   correlated voters and is best or tied in 7 of 10 rows.
2. **The collapse is correlation, not lexicon leakage.** Disable run: with the
   all-train lexicon (held-out sentences included), German noun/verb's summed
   quorum is still 85.0 %. The fit-only lexicon makes the held-out weights
   honest. It also has a cost: it covers 90 % of train, and with the full
   lexicon German adj/adv's joint quorum reaches 82.9 % (315 tokens instead
   of 309).
3. **Animal Farm adj/adv loses with the joint fit (84.8 % vs 88.2 %).** The
   weights are fitted on EWT held-out (web text) and do not transfer to
   Orwell's prose, where the question table alone is 91.9 %. Weights are
   genre-dependent; a quorum fitted on one register is not the default for
   another.
4. **German rule precision (test):**
   - "Next to a noun → adj" is right only 27.8 % of the time; "uninflected right before a noun → adv" 80.0 %. A German attributive adjective must inflect, so the English noun-adjacency rule is wrong for German, and inflection is the German position signal.
   - "Before an adjective → adv" 96.7 %.
   - TEKAMOLO adverbial cues 71.0 %.

## OPEN

- German adj/adv stays at 81–83 %; no combination beats position × frequency.
- Fit the weights per register (or on dev of the target genre) instead of one held-out set.
- The case and PP-lane questions (Dativ/Akkusativ, Wechsel static vs. directional → Lokal, relative-pronoun case) are separate ambiguities. They are not scored here, and their inventories (`wechsel.tsv`, `article_case.tsv`, `valency.tsv`, `relative_pronoun.tsv`) need their own gold harness on UD `Case`.
- Re-measure right-corner TEKAMOLO on edited text (GSD) against Luther.
