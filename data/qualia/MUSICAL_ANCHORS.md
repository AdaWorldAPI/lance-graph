# Calibrated 219: original musical-qualia annotations

This is an **additive, provenance-preserving** companion to the artifacts introduced by PR #1571.

## Files and boundaries

- `calibrated_219.json`: **unchanged** production-facing snapshot from PR #1571, preserving the existing schema, 17D coordinates and qualitative tags.
- `calibrated_219.bin1024`: **unchanged** sign-binarized embedding fixture from PR #1571.
- `calibrated_219.musical_anchors.json`: the **original verbatim 219-record** rich corpus, copied from [AdaWorldAPI/ada-consciousness `qualia_17D/MERGED_17D.json`](https://github.com/AdaWorldAPI/ada-consciousness/blob/main/qualia_17D/MERGED_17D.json), source Git blob `2ebe6454618e1a590e928b53116cb65f4a96c264`.

Each of the 219 enriched records has the original `id`, `label`, `family`, `subfamily`, `vector` (17D), `qualia`, `melodic_motions`, `harmonic_bias`, `truth_anchor_candidates`, and `neighbors`.

The source data's `vector` corresponds exactly to `calibrated_219.json`'s `vector_17d`, and its `qualia` corresponds exactly to `qualia_tags`. Records are joined by **`id`**, never by ordering, fuzzy labels, or geometric nearest neighbors. All 219 IDs, labels, coordinates, tags, families and subfamilies matched during preparation.

This file is **archival and cold-path only**: no hot-path ABI modification, palette-index reassignment, binary regeneration, implicit migrations or changes to #1571 readers. No new canonical qualia states are asserted.

## Interpretation and source chronology

The musical fields preserve the earlier compositional intent: felt texture, scale-degree motion, modal/harmonic behavior, specific musical choices to *retain* (truth anchors), and related states. Some fields are descriptors or composition constraints rather than measured musicological facts. `neighbors` can reference states outside the calibrated 219; do not assume the references resolve within this subset.

Upstream documentation: [Ada music module](https://github.com/AdaWorldAPI/ada-consciousness/tree/main/core/music), including `qualia_codex.py` and `SCHEMA.md`. The codex contains example-song calibration anchors, but the 219-file itself is **not** a per-record song-to-state catalog. Do not fabricate those links.

The 219-item corpus was extracted from [AdaWorldAPI/Chat/music/Interval_Qualia_and_Archetypes_694dbdbf.json](https://github.com/AdaWorldAPI/Chat/tree/main/music), as documented by `ada-consciousness/qualia_17D/INDEX.json`, with 322 parsed items in total, of which 219 had 17D vectors. A separately documented 800-microstate compositional *target* is not equivalent to 800 calibrated states. The attempted mechanically generated 806 extra atoms were rejected in `ada-consciousness/anti-patterns/sigma1024_failure`.

## Verification

Run `python3 data/qualia/verify_musical_anchors.py` from the repository root.

This asserts 219 unique IDs in both files and exact correspondence for IDs, labels, families, subfamilies, vectors and descriptive tags. It also checks that the four musical fields are non-empty strings and match a pinned SHA-256 of their canonical JSON, so an altered or emptied annotation fails. Checks raise explicitly (no `assert`), so they also run under `python -O`; negative controls live in `test_verify_qualia.py` and run in CI (`.github/workflows/qualia-verify.yml`). The original `calibrated_219.json` and `.bin1024` are never rewritten by the script.
