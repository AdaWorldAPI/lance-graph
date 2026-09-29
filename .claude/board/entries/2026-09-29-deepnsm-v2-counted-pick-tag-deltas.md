# DeepNSM-v2 counted PoS pick: offline tag-change counts (2026-09-29)

**Status:** MEASURED (offline, over the committed CSVs). No code changed.

Replacing `bible_wave::load_pos`'s first-wins tagging with a counted pick
(highest summed `wordFreq` per folded parser state, tie order Noun > Verb >
Adj > Det > Other) changes:

- **105 tags** when the `lemmas_5k.csv` layer stays first (variant B, the plan's choice);
- **288 tags**, 183 of them lemma-derived, when the counted pick goes first (variant A).

Method: Python over `crates/deepnsm/word_frequency/{lemmas_5k,word_forms}.csv`,
fold `n|p→Noun, v→Verb, j→Adj, a|d→Det, else Other`, union of both tables.
Whether either variant tags the KJV better is not measured. The KJV
before/after is gate G5 of D-LXC-1.

Two pre-existing observations from the same council: `archaic_pos` never fires
for COCA-known words such as `art` (D-LXC-9), and no tagger in deepnsm-v2
produces `Pos::Rel` (D-LXC-10).

Plan: `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md`.
