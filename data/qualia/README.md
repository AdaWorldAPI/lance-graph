# data/qualia — the calibration set behind `contract::qualia_profile`

`calibrated_219.json` is the reference chart: 219 named felt states, each a
measured patch in a 17-axis profile connection space (PCS). Pulled from
Upstash `ada:qualia:calibrated:*` on 2026-10-15 (`source: MERGED_17D.json`,
index created 2025-01-24). It is the set `thinking-engine/src/qualia.rs` cites
as "220 calibrated pairs in Upstash" and did not carry.

| file | what | size |
|---|---|---|
| `calibrated_219.json` | `{id, label, family, subfamily, vector_17d, qualia_tags}` × 219, axes in PCS order, values on a 0.05 grid in `[-0.40, 1.00]` | 131 KB |
| `calibrated_219.bin1024` | header `QBIN1024` + `u32 count`, then 128 bytes per row: the row's 1024-d Jina embedding sign-binarized (`bit i = emb[i] > 0`), same order as the JSON | 28 KB |

`scripts/gen_qualia_profile.py` turns the JSON into
`crates/lance-graph-contract/src/qualia_profile_table.rs`: rows sorted by
`id`, palette index = position + 1, PCS in hundredths as `[i8; 17]`, output
rustfmt'd. `PROFILE_ID` (sha256 of the JSON) and `TABLE_FNV` (FNV-1a over the
emitted rows) pin the table; `qualia_profile::tests::table_fnv_matches_source`
fails on a single edited hundredth.

## The ICC reading

| ICC | here |
|---|---|
| device space | `qualia::AXIS_LABELS` — convergence observables (`clarity = 1/cycles`): what the engine emits |
| PCS | `qualia_profile::PCS_AXES` — valence, arousal, dominance, warmth, brightness, tension, clarity, social, nostalgia, sacredness, desire, grief, awe, shame, hope, edge, resolution_hunger |
| named-colour tag (`ncl2`) | `NAMED` — the 219 patches |
| profile ID | `PROFILE_ID` |
| B2A (PCS → index) | `nearest_named` → palette byte + residual |
| K | the residual: what no name covers |
| gamut | `Gamut`, a 256-bit mask per mode / persona / voice |
| rendering intent | `RenderingIntent::{AbsoluteColorimetric, RelativeColorimetric, Perceptual}` |
| ink mix | `Mix` — up to three patches with weights |
| A2B (device → PCS) | **unmeasured** for 12 of 17 axes — `ISS-QUALIA-A2B-CLUT-UNMEASURED` |

In the workspace's CMYK/RGB words: the PCS axes are the subtractive side
(grief, shame, tension absorb from a white of full presence); an observer
vocabulary such as the six Ekman archetypes is the additive side; the device
axes are neither — they are the print head.

## Families (38)

Devotion 25 · Longing 17 · Integration 9 · Endurance 2 · Liberation 1 · and
33 families of exactly 5: Anger, Authorship, Awe, Becoming, Communion,
Courage, Curiosity, Death_symbolic, Desire, Discovery, Ending, Faith, Grief,
Innocence, Letting_Go, Loss_of_Control, Love_non_erotic, Meaning, Play,
Power, Presence, Purpose, Rebirth, Release, Repair, Return, Risk, Silence,
Stillness, Surrender, Transformation, Trust, Witness.

## Regenerate

```
python3 scripts/gen_qualia_profile.py
cargo test -p lance-graph-contract --lib qualia_profile
```
