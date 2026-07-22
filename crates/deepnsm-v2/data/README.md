# data/ — trained Cam96 artifacts (NOT committed; fetched from the release)

The trained codebook artifacts live as a GitHub Release on `AdaWorldAPI/q2`:
**`v0.1.0-cam96-data`** — "Cam96 Trained Codebook (KJV vocab, Jina-v3 96d)".

Download the three assets into this directory before running `bible_wave`:

```sh
BASE=https://github.com/AdaWorldAPI/q2/releases/download/v0.1.0-cam96-data
curl -L -o cam96_codebook.bin "$BASE/cam96_codebook.bin"   # CAM96CB1, 12x256x8d f32 + d_max, 96 KB
curl -L -o cam96_codes.bin    "$BASE/cam96_codes.bin"      # CAM96WD1, 12,543 x 12 B, 147 KB
curl -L -o bible_vocab.txt    "$BASE/bible_vocab.txt"      # frequency-ranked vocab, one word/line
```

Provenance + held-out metrics: `../probes/README.md` §4 (producer scripts:
`../probes/{embed_bible_vocab,train_codebook}.py`). Loaded by
`deepnsm_v2::codebook::{load_cam96_space, load_cam96_codes}`.
