# 2026-10-04 — gender + number of the head noun shrink article case ambiguity (D-LXC-26)

**Status:** MEASURED (debug-0, release) on UD r2.15 German GSD and HDT. Instrument: `crates/deepnsm-v2/examples/ud_case_eval.rs`, rule R6. The kill bars below were fixed before the run.

Operator: *"Gender would greatly reduce morphology ambiguity"*. Their suggested stemmer is the one tesseract-paperless search uses: tantivy `Stemmer` with `Language::German`, which is the `frostem` crate (pure-Rust Snowball, zero deps, BSD-3). No AdaWorldAPI fork exists (checked via `list_repos`), so the registry crate is used. It is a **dev-dependency** of deepnsm-v2, German only, floating at major `1`.

## Method

- **Noun lexicon (train only):** form → (gender, number) counts.
  - Fallback: the noun's German Snowball stem → counts.
- **Paradigm table:** (article form, head-noun gender, head-noun number) → case counts.
- **Head noun:** the first capitalised token within three tokens of the article.
- **Prediction:** the cell's majority case.
- **Baseline:** the majority case of the article form alone, on the same tokens.

## Results

| | GSD | HDT |
|---|---|---|
| train case purity, article form alone | 66.4 % | 65.3 % |
| train case purity, form + gender + number | **79.1 %** | **79.1 %** |
| R6, seen noun | 74.7 % vs 62.4 % (1,323) | **76.4 % vs 64.1 % (30,916)** |
| R6, stem fallback | 59.6 % vs 64.4 % (146) | 63.7 % vs 66.9 % (1,606) |
| R6, all | 73.2 % vs 62.6 % | 75.8 % vs 64.2 % (71.6 % of articles) |

**Verdict.**
- The main bar passes: **+11.6 points** over the form majority on HDT.
- The stem-fallback bar is **KILL**: 63.7 % against 76.4 % for seen nouns, which is worse than the form majority. So R6 as a whole is KILL.
- Gender + number removes about 14 points of article case ambiguity on both treebanks.

**Diagnosis (exploratory, after the run).** A Snowball stem keeps gender but erases number (*Firma* / *Firmen* share a stem), so it supplies the wrong number for plurals. Reading the (article, gender) table only:
- HDT: **68.4 % vs 66.9 %** (1,640 tokens);
- GSD: 67.5 % vs 63.6 %.

The gain is small, and unseen nouns are only about 5 % of articles.

**OPEN.**
- Unseen-noun number from morphology: plural suffix, or the article itself when unambiguous.
- Combining R6 with position. "der + feminine singular" leaves Dat vs Gen; the 2,665 Nom→Gen confusion on HDT is the next target.
- A gender column in the German codebook builder (`build_de_codebook.py`); UD already carries `Gender=`.
