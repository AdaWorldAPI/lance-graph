# 2026-10-04 — deepnsm-v2: KJV noun/verb decisions checked against German casing; slot rule vs clause rule; WordNet as a parallel coordinate (D-LXC-15)

**Status:** MEASURED. KJV arm: `ROSETTA_DIR=<dir> cargo run --release --example
bible_wave -- pg10.txt`, with `<dir>` holding `pd-texts/`, `rosetta-gpl/`, `de/`
from release `v0.1.0-codebooks-2026-07-26` (sha256 de a5575ad1…, pd-texts
d141d76c…, rosetta-gpl 66acc1ed…, wordnet f670f1f3…). UD arm: `ud_pos_eval`
with `UD_CLAUSE=1` / `UD_WORDNET=…` / `UD_WORDNET_FILTER=1`.

Operator questions (2026-10-04): the slot rule vs "we have no verb in this
sentence, which one could it be"; use the German/Czech/Greek Bible releases;
WordNet as a coordinate in parallel.

## What changed

- `Typology::predicate_required` — the clause rule as a switch, off by
  default. At a sentence end it drops reading combinations on which no verb
  took a predicate slot, if some combination's verb did.
- `bible_wave` optional Rosetta arm (`ROSETTA_DIR`): silver noun/verb labels
  for KJV homographs from Luther 1545 casing via the release's en→de
  alignment. Contrastive: a word is used only if its associates include a
  German lexicon noun AND a verb; capitalised non-initial noun → Noun,
  lowercase verb → Verb; all associates found must agree. Psalms excluded
  (documented Luther 1545 versification offset). Verse key (book, chapter,
  verse) from `corpus` markers, 66 books asserted. `serde_json` is a
  dev-dependency only.
- `ud_pos_eval`: `UD_CLAUSE`, `UD_WORDNET` (sense-count pick), `UD_WORDNET_FILTER`
  (WordNet as an existence filter on COCA readings).

## Slot rule vs clause rule (kept Verb)

| | slot rule only | + clause rule | clause rule's own additions |
|---|---|---|---|
| UD English, COCA readings | 108 at 92.6 % | 177 at 68.4 % | ~69, ~30 % right |
| UD German | 36 at 100 % | 133 at 94.7 % | ~97, ~93 % (frequency on them ~97 %) |
| UD French | 28 at 100 % | 56 at 89.3 % | ~28, ~79 % |
| KJV, German-casing silver | 20 at 60.0 % | 34 at 58.8 % | 14, ~57 % |

The clause rule stays off: its additions are below the slot rule everywhere
and below frequency in every language.

## KJV against German casing (1,751 labelled of 34,608 homograph tokens)

| policy | decided | right | COCA lemma tag on the same tokens |
|---|---|---|---|
| COCA lemma tag (forced) | 1,751 | 67.2 % | — |
| slot + licensing | 756 | **85.8 %** | 74.9 % |
| — licensing (kept Noun) | 736 | 86.5 % | |
| — slot rule (kept Verb) | 20 | 60.0 % | |

Labels are silver. Of the slot rule's 8 misses, 3 are label noise (collocate
associates: regard→Gott, light→Licht, walking→Meer) and 5 are vocatives /
a list ("ye fools", "thou judge", "faith hope charity"). The non-contrastive
first version (7,531 labels) gave the same ordering — licensing 95.9 %,
slot 56.0 %, clause 45.3 %, lemma tag 80.3 % — but with collocate noise
(`rose` → `des Morgens`).

## WordNet as a parallel coordinate (UD English, COCA readings)

- As a chooser (reading with more WordNet senses): **59.3 %** right vs COCA
  frequency 84.2 % — sense counts do not track usage.
- As an existence filter (drop a COCA reading WordNet lacks): changes **1**
  form; WordNet confirms nearly every COCA noun/verb homograph. Neutral.

## Not used yet

- Czech (BKR) and Greek (Tischendorf): no noun capitalisation; labels would
  need a morphological lexicon. The en-cs / en-el alignments are in the
  release.
- `de/tekamolo.tsv` (adverbial lemma → TEKAMOLO lane, from German UD
  `advmod`/`mark`) is the lexical adverb class the TEKAMOLO tenant needs;
  through the en→de alignment it can reach English adverbs. Not wired.

## OPEN

- Slot-rule errors: vocatives/address ("ye fools", "thou judge"), archaic
  `mine`/`thine` as subject, nouns after a carried object.
- Licensing errors on the KJV: a verb after a predicative adjective ("make
  alive wound"), `both`/`that` read as determiners.
- Silver-label noise: collocates and translation shifts; a cleaner label
  would align word to word, not verse to verse.
