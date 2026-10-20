# Musical enrichment of the calibrated 219 qualia

> **Plan and board (2026-10-20):** owned by [`ce64-grammar-wiring-v1`](../../.claude/plans/ce64-grammar-wiring-v1.md). Board: [2026-10-20-chatgpt-prs-1601-1617-plan-links.md](../../.claude/board/entries/2026-10-20-chatgpt-prs-1601-1617-plan-links.md). Merged in #1608.

This is an **additive archival sidecar**, not a replacement for the #1571 ABI, palette, or binary Jina fixture.

- Canonical/runtime: `calibrated_219.json` and `calibrated_219.bin1024` (**unchanged**).
- Companion: `calibrated_219.music_enriched.json`, keyed by the same 219 `id` values.
- Source: `AdaWorldAPI/ada-consciousness/qualia_17D/MERGED_17D.json` (Git blob `2ebe6454618e1a590e928b53116cb65f4a96c264`).
- Origin: `AdaWorldAPI/Chat/music/Interval_Qualia_and_Archetypes_694dbdbf.json`; extraction manifest dated 2025-12-30.
- Added fields: `melodic_motions`, `harmonic_bias`, `truth_anchor_candidates`, `neighbors` copied verbatim from the original record. This is not an automatic inference or LLM reconstruction.
- The original source's `vector` is identical to the calibrated `vector_17d`; the original `qualia` is identical to `qualia_tags`. All 219 ID, label, family, subfamily, vector and qualia mappings were compared before the sidecar was authored.

The sidecar intentionally does not replicate the canonical numeric coordinates. Join on `id`. Not all neighbor IDs are represented in the 219 calibrated subset, because some are reference anchors to the wider source vocabulary. Preserve those references as written; do not silently remap them.

**Lab / provenance only.** JSON is a preservation and validation fixture, not a proposed production interface or new runtime serialization. No consumer is enabled by this PR.

Run:

```sh
python3 data/qualia/verify_music_enrichment.py
```

The validator checks that all 219 IDs match, that there are no duplicates, and that the four musical fields are nonempty arrays. Existing #1571 files remain unmodified.

## Examples

- `devotion_worship_without_church`: Dorian natural 6 and held 4; preserve its modal characteristic.
- `desire_slow_burn`: minor-second friction and suspended pedal; keep the friction unresolved.
- `longing_doorway_pause`: withhold the cadence instead of smoothing it into closure.

Related original codex: `AdaWorldAPI/ada-consciousness/core/music/qualia_codex.py`, which also provides example-song anchors such as *Alegría*, *Lean on Me*, and *Fix You*. Those song associations are codex-level references and **are not automatically assigned to all 219 individual entries**.
