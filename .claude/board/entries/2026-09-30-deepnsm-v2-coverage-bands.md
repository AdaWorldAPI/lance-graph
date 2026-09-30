# DeepNSM-v2 coverage bands at population-calibrated thresholds (2026-09-30)

**Status:** MEASURED. D-LXC-11 is implemented in `bible_wave` (report only;
no tag changes).

Each word's dominant reading share (`coverage(id)[0]`, from D-LXC-1's
frequency-ordered register) is banded Contested / Leaning / Decisive at the
quartiles of the band population, calibrated once at load. The population is
the vocabulary words that are not lemma-table keys, have known coverage, and
have readings that fold to at least two parser states.

On the release KJV vocabulary (`v0.1.0-cam96-data`):

| quantity | value |
|---|---|
| band population | 141 (of 635 cross-state words; 494 are lemma-table keys) |
| cuts (lo, hi), `rank_per_10000` indices 35 / 105 | 72, 97 |
| share range | 50 … 99 |
| contested / leaning / decisive | 34 / 71 / 36 |
| absolute alternative "< 60 contested" | 20 |

Contested words include flies, leaves, lies, promises, needs, means.

The Rust gate G8c and the Python receipt in the plan give the same numbers.
The KJV run is otherwise unchanged (70,396 triples, G6 = 25).

Open for the operator: relative bands (this) vs an absolute floor.

Ready-to-paste STATUS_BOARD row (the file is deny-listed for the agent):

`| D-LXC-11 | coverage bands (population quartiles) in bible_wave | In PR | deepnsm-v2-coverage-bands-v1 |`

Plan: `.claude/plans/deepnsm-v2-coverage-bands-v1.md`.
