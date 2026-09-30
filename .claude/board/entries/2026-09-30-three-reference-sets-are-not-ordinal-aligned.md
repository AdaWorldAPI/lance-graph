# 2026-09-30 — The three COCA reference sets are neither nested nor ordinal-aligned

**Status:** MEASURED · OPEN — plan `.claude/plans/deepnsm-v2-cam96-pairwise-v5.md` §3.7 (F23, D-CPW-14, G-REF)

## Measured (python over the committed CSVs and the Tigris artifact)
| reference | rows | distinct keys | key |
|---|---|---|---|
| COCA4096 — `crates/deepnsm/word_frequency/word_rank_lookup.csv`, ranks ≤ 4096 | 4,096 ranks | 3,559 words | rank (homographs share a rank) |
| COCA5K_LEMMA — `lemmas_5k.csv` | 5,050 | 4,380 lemmas / 5,050 (lemma, PoS) | (lemma, PoS) |
| COCA20K_ACAD — `academic_20k.csv`; Tigris `lance-graph/codebooks/deepnsm-v2-academic-coca-v1/` | 20,845 | 18,559 words / 20,842 (word, Pos) | `word_id` = admission order |

- 5k ∩ 20k: 4,895 of 5,050 (lemma, PoS); 4,264 words; **116 of the 5k lemmas are absent from the 20k**.
- 4096 ∩ 20k: 3,462 of 3,559 words.
- **Ordinal alignment: 3 of 4,264 shared words carry the same ordinal in the 5k and in the 20k.**
- Ambiguity differs per set: the 20k carve drops 2,286 same-word-different-Pos duplicates to one id (MANIFEST: 18,559 of 20,480 reserved slots; basins 73..79 empty); the 4096 shares ranks across homographs; the 5k keeps (lemma, PoS) distinct.
- The Tigris "academic codebook" TSV is a `PaletteVocab::from_frequency_ranked` carve (`word_id, basin, slot, …`), not a trained centroid codebook. Its `academic_20k.csv` sha256 equals the committed file's.

## Consequence
A reading contract names (reference id, version, digest); an ordinal is stable only within its reference; correspondence is an explicit (lemma, PoS) map with `Ambiguous{n}` / `Missing` as values; switching references must preserve the declared map and never reuse an ordinal. Gate G-REF re-derives this table and diffs it.

## Open
- No embedding artifact exists for the academic reference (v5 Q7: own codebook vs correspondence-mapped centroids).
- `range`/`disp` are loaded by no `src/` code; `Vocabulary::load` never opens `lemmas_5k.csv`.
