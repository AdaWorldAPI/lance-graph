# 2026-10-04 — gender of unseen German nouns from compound head, lemma, stem and nominalisation (D-LXC-27)

**Status:** MEASURED (debug-0, release) on UD r2.15 German GSD and HDT. Instrument: `crates/deepnsm-v2/examples/ud_gender_eval.rs`. Kill bars were fixed before the run.

Operator: *"if lemma and word forms have a match, stemmer should mostly help to resolve the gazillions of German prefix and zusammengesetzte Substantive and substantivierte Verben"*.

## Method

The unit is a test noun with a single gold `Gender=` whose form is **unseen** in train (HDT: 13,157; GSD: 885). Each method is scored alone, then as a cascade.
- **lemma**: DeReKo-2014 STT `NN` rows (form → lemma; CC BY-NC, read from a local path, never committed) give the lemma, whose gender comes from the train lemmas.
- **infinitive**: a form that is a train VERB lemma ending in *-n* is Neut (*das Essen*).
- **compound**: the gender of the longest proper suffix (≥ 3 letters) that is a train noun form. A compound takes its last part's gender.
- **stem**: the `frostem` German Snowball stem (the tesseract-paperless search stemmer) → majority gender.

## Results

| method | HDT coverage | HDT precision | GSD coverage | GSD precision | bar |
|---|---|---|---|---|---|
| lemma (DeReKo) | 4.7 % | 98.4 % | 14.1 % | 87.2 % | — |
| infinitive → Neut | 0.8 % | 91.5 % | 1.4 % | 8.3 % (12 tokens) | 95 %: **KILL** |
| compound head | **83.7 %** | **98.1 %** | 59.7 % | 88.8 % | 90 %: PASS on HDT, KILL on GSD |
| stem | 15.5 % | 88.0 % | 19.8 % | 73.7 % | 85 %: PASS on HDT, KILL on GSD |
| **cascade** | **89.9 %** | **97.7 %** | 71.1 % | 86.8 % | — |

- The majority-gender baseline is Fem: 39.4 % on HDT, 41.1 % on GSD.
- **The compound head carries the result.** It covers most unseen German nouns at 98 % precision on HDT.
- The lemma route is highly precise but adds little coverage, because train already covers most lemmas.
- **GSD is weaker throughout.** Its train set is about 3× smaller, and its gold gender is noisier (the D-LXC-23 note on GSD Case gold applies too).

**OPEN.**
- Nominalised infinitives are rare as unseen nouns and miss the 95 % bar on HDT.
- On GSD, the 12 infinitive-shaped "nouns" are mostly not neuter. Plural-noun / verb-lemma collisions such as *Rennen* are suspected, not checked.
- Next: feed this gender into R6 (D-LXC-26) for unseen nouns. Number for unseen nouns is still open.
