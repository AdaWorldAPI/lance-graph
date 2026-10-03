# data/ — trained Cam96 artifacts (NOT committed; fetched from the release)

The trained codebook artifacts live as a GitHub Release on `AdaWorldAPI/lance-graph`:
**`v0.1.0-cam96-data`** — "Cam96 Trained Codebook (KJV vocab, Jina-v3 96d)".

Download the three assets into this directory before running `bible_wave`:

```sh
BASE=https://github.com/AdaWorldAPI/lance-graph/releases/download/v0.1.0-cam96-data
curl -L -o cam96_codebook.bin "$BASE/cam96_codebook.bin"   # CAM96CB1, 12x256x8d f32 + d_max, 96 KB
curl -L -o cam96_codes.bin    "$BASE/cam96_codes.bin"      # CAM96WD1, 12,543 x 12 B, 147 KB
curl -L -o bible_vocab.txt    "$BASE/bible_vocab.txt"      # frequency-ranked vocab, one word/line
```

Provenance + held-out metrics: `../probes/README.md` §4 (producer scripts:
`../probes/{embed_bible_vocab,train_codebook}.py`). Loaded by
`deepnsm_v2::codebook::{load_cam96_space, load_cam96_codes}`.

## Lexical evidence (counts + PoS) — not in the Cam96 release

`bible_vocab.txt` carries surface forms only, in rank order: no counts, no
part of speech. Counted evidence is read separately by
`deepnsm_v2::lexical::load_word_forms_csv`, whose input is COCA
`word_forms.csv` (`lemRank,lemma,PoS,lemFreq,wordFreq,word`, committed at
`crates/deepnsm/word_frequency/`). It is stored beside the routing
`PaletteVocab`, never inside it and never inside the Cam96 codes.

## Lexical source contract

`bible_wave` reads the COCA tables through `lexicon_file`: by default from
`crates/deepnsm/word_frequency/` (where they are committed), or from the
directory named by `DEEPNSM_V2_LEXICON`. That location is not an ownership
claim — v2 links no v1 code. What v2 depends on is the bytes and the row
order: `lemmas_5k.csv` (the lemma table, first row per lemma wins, F9) and
`word_forms.csv` (every reading, through `load_word_forms_csv`).

The Tigris bake `lance-graph/codebooks/deepnsm-v2-academic-coca-v1/` carries
`academic_20k.csv` byte-identical to the committed copy (md5
`70ad802b6a9f8ac4ed3d08983e39133b`, sha256 `1dfd5eda…6e2d` per its
`MANIFEST.json`) plus the derived `PaletteVocab` carve. No code reads the
bake yet; it is the published copy a future loader can verify against.

The COCA letter → FSM tag fold lives in one place, `deepnsm_v2::coca`.
