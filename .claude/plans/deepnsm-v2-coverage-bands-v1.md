# DeepNSM-v2 coverage bands at population-calibrated thresholds (D-LXC-11)

**Status:** RATIFIED v3 (5+3 council, 2026-09-30). Implementation in this PR.
**Parent:** `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md` (D-LXC-1,
the frequency-ordered register this reads).

## Overview

D-LXC-1 stores each word's readings most frequent first, with a cumulative
percentile coverage per reading. The operator asked that frequency be read as
percentile coverage with thresholds at prevalence boundaries:

> "It should simply be percentile coverage"
> "Compared with HDR popcount stacking early exit Belichtungsmesser statistical
> confidence interval thresholds preheating rolling floor bucket assignment.
> It's thresholds at certain Prävalenz boundaries/percentile coverage.
> Akin to using Gini palette when looking at poor countries. Or IQ buckets in
> gaussian distribution."

This plan turns the dominant reading's share into a three-level
`CoverageBand` whose cut points are quartiles of the measured population, so
each band holds a known share of it. The cuts are calibrated once at load from
the evidence present. That is a load-time snapshot, not a streaming rolling
floor; the streaming form exists as ndarray `hpc::rolling_floor::RollingFloor`.

## Checklist

- [x] Spec v1, 5 research savants, draft v2, 3 reviewers, v3 (this file)
- [x] `CoverageBand`, `BandCuts`, `calibrate` in `bible_wave.rs`
- [ ] Tests T1-T8, each disable-verified red before landing
- [x] Gates G8a-G8c
- [x] Board entry + indexes in the same commit

## Frozen decisions

- F1 `LexicalEvidence` is evidence, not interpretation. It stores readings
  frequency-first with cumulative coverage (`lexical.rs` `finish()` and
  `coverage()`, PR #1304). It gets no band logic.
- F2 The PoS fold (`coca_pos`) and all tagging live in the consumer example
  `crates/deepnsm-v2/examples/bible_wave.rs` (commit `68955ecb`).
- F3 Lemma table first (commit `ec50f07b`, pinned by
  `the_forms_layer_never_retags_a_lemma_table_word`). The register is read only
  for words the lemma table does not know.
- F4 DeepNSM-v2 is the inbound leg: no reasoning, NARS truth or belief in it.
- F5 Unknown is not zero: a word with unknown coverage has no band.
- F6 No model identifiers or advertising; Read for source inspection; the
  board files denied in `.claude/settings.json` are not edited by the agent.
- F7 Search is navigation, never evidence.
- F8 The band key is the dominant READING's share, `coverage(id)[0]`. The
  operator rejected summing at tag time ("why are you counting … it should
  simply be percentile coverage"), so no parser-state sum is formed.
- F9 Bands are relative by design: the operator asked for cuts at prevalence
  boundaries ("IQ buckets", "Gini palette"), i.e. cuts that each hold a known
  share of the population. The absolute floor is the recorded alternative.

## Measured

Method: the Python receipt below, run from the repo root against the release
`bible_vocab.txt` (`v0.1.0-cam96-data`, SHA-256
`8dc3a65dcd3af38a2f53308fb96ef5fb5b34336c14587f6971b75ac966c3e212`). The
in-code gate G8c recomputes the band numbers and asserts them.

| quantity | value |
|---|---|
| `word_forms.csv` rows with an empty `wordFreq` | 0 |
| in-vocab words with ≥1 reading | 3,556 |
| … whose readings fold to ≥2 parser states | 635 |
| … of those, lemma-table keys (register never read) | 494 |
| **band population** | **141** |
| top reading shares its state with another reading | 0 of 141 |
| rank indices (`rank_per_10000`, n = 141) | 35, 105 |
| cuts (lo, hi) | 72, 97 |
| share range | 50 … 99 |
| contested / leaning / decisive | 34 / 71 / 36 |
| absolute alternative "< 60 contested" | 20 |

Because no top reading shares its state with another reading, reading share
and state share are equal for all 141 words: F8 changes nothing here.

Other vocabularies (same receipt, different word set):
- The COCA 5k lemma vocab: population 0. Every word is a lemma-table key.
- All COCA surfaces: population 388, cuts (66, 94), bands 96 / 189 / 103;
  "< 60" gives 62.

```python
# receipt: python3 receipt.py <bible_vocab.txt>  (from the lance-graph root)
import csv,collections,sys
vs=set(l.strip() for l in open(sys.argv[1]) if l.strip())
lem={}
for r in list(csv.reader(open("crates/deepnsm/word_frequency/lemmas_5k.csv")))[1:]:
    lem.setdefault(r[1].lower(),r[2])
fold=lambda p:{'n':'N','p':'N','v':'V','j':'J','a':'D','d':'D'}.get(p,'O')
rd=collections.defaultdict(list)
for r in list(csv.reader(open("crates/deepnsm/word_frequency/word_forms.csv")))[1:]:
    w=r[5].lower()
    if w in vs: rd[w].append((fold(r[2]),int(r[4])))
pop=[w for w,rs in rd.items() if len({p for p,_ in rs})>=2 and w not in lem]
sh=sorted(max(c for _,c in rd[w])*100//sum(c for _,c in rd[w]) for w in pop)
n=len(sh); rk=lambda p:min(p*n//10000,n-1)
lo,hi=sh[rk(2500)],sh[rk(7500)]
print(n,lo,hi,sum(s<lo for s in sh),sum(lo<=s<hi for s in sh),sum(s>=hi for s in sh))
```

## Design

In `bible_wave.rs` only:

1. `enum CoverageBand { Decisive, Leaning, Contested }`. It is a separate
   implementation from the planner's `NestedBands` quantile buckets
   (`lance-graph-planner/src/nested_bands.rs`), which are cited as prior art
   but live in a crate this one does not depend on. Doc comments say
   "reading share" and never "confidence", "σ", "CI" or "reliability":
   shares of rare and common words weigh equally.
2. `struct BandCuts { lo: u8, hi: u8, population: usize }`, computed once in
   `Tagger::load` by `calibrate(&lemmas, &evidence, &vocab)` (the vocabulary gives each id's word for the lemma check and its length).
   - Population: every id in `0..vocab.len()` that (a) is not a key of the
     tagger's own `lemmas` map (as built, first row wins), (b) has known
     coverage, and (c) has readings that fold to ≥2 parser states.
   - Key: `coverage(id)[0]`.
   - Rank rule: ndarray `rank_per_10000`, index `p·n / 10000` (integer
     division) clamped to `n − 1`, with p = 2500 for `lo` and 7500 for `hi`.
     Hand-rolled in a few lines, because deepnsm-v2 builds with no ndarray
     dependency (`Cargo.toml:17-20`). This rule is not nearest-rank: the two
     differ when p·n/10000 is an exact integer (n = 4 gives index 1 here, 0
     under nearest-rank). They agree at n = 141.
   - Empty population: `None` (no cuts). The function must not panic for
     n = 0 or n = 1; the unit-test fixtures build 1-5 word vocabularies.
   - The calibrator reads `coverage` directly and never the tagger's
     archaic/`Other` fallback.
3. Assignment: share < lo is Contested; share ≥ hi is Decisive; otherwise
   Leaning. Words outside the population get `None`. The three reasons for
   `None` (lemma-table, single state, unknown coverage) are merged
   deliberately: a reader asks only "is there a band", and the tests keep the
   reasons apart.
4. Storage: `bands: Vec<Option<CoverageBand>>` indexed by `WordId` (at most
   65,536 entries, since `WordId` is `u16`).
5. No tag changes. `Tagger::pos` is untouched. `main` prints one `BANDS` line
   (population, cuts, share range, counts per band), and G8c asserts it.
6. Integer only: no floating point in calibration.

## Non-goals

- Any change to `lexical.rs` (F1).
- Changing any tag, including an early exit on Decisive: role resolution
  (D-LXC-2) is the reader that acts on bands.
- A word-frequency axis: blocked by D-LXC-5 (`bible_vocab.txt` has no counts).
- A streaming rolling floor: the evidence is loaded once.
- Landing bands in a V3 facet header (parent plan F10). The cuts would have
  to travel with the bands, since a band means different things per
  vocabulary.
- Configurable quantiles or more than three bands.
- The dormant lowercasing duplicate-key case in `lowercase_word_column`.

## Pre-registered gates

Numbering continues the parent plan's G1-G7.

- G8a `cargo test` (deepnsm-v2) green; clippy `--all-targets -D warnings` and
  `fmt --check` clean.
- G8b KJV run unchanged from D-LXC-1: 70,396 triples, 1,237 subjects, G6 = 25.
- G8c In `main`: population 141, cuts (72, 97), bands 34 / 71 / 36. This is a
  regression pin for the release KJV vocabulary only.
- Unit tests. Each must be disable-verified red before landing.
  - T1 The cuts are computed, not constant: a population shifted upward
    raises both cuts.
  - T2 On a mixed population, both Contested and Decisive are non-empty.
  - T3 Three fixtures each get no band: a lemma-table word, a word whose
    readings fold to one state, and a word with unknown coverage. Each has its
    own disable run.
  - T4 An empty population gives no cuts; a one-word population does not
    panic.
  - T5 Boundaries on explicit `BandCuts`: share == lo is Leaning, and
    share == hi is Decisive.
  - T6 Can stay silent: identical shares give zero Contested and all Decisive.
  - T7 The key is reading share: a word with a 40% top reading and a larger
    state sum is banded on 40.
  - T8 The rank rule is `rank_per_10000`, not nearest-rank: at n = 4 the lo
    index is 1.

## Commit contents (one commit, in this order)

1. `crates/deepnsm-v2/examples/bible_wave.rs`: code and tests.
2. This plan (it carries D-LXC-11 for the `added-plans-have-dids` check).
3. `.claude/board/entries/2026-09-30-deepnsm-v2-coverage-bands.md`, then
   `python3 .claude/tools/entries_index.py --write`.
4. `SUPERSESSION-INDEX.md` regenerated last.
5. `AGENT_LOG.md` entry naming the council run.

STATUS_BOARD, INTEGRATION_PLANS and PR_ARC_INVENTORY are deny-listed for the
agent. The entry carries a ready-to-paste STATUS_BOARD row for the operator:

`| D-LXC-11 | coverage bands (population quartiles) in bible_wave | In PR | deepnsm-v2-coverage-bands-v1 |`

## Open for the operator

- Relative (this plan) vs absolute ("< 60") bands. Relative follows the stated
  intent; absolute marks 20 instead of 34 words contested on the KJV
  vocabulary, and 62 instead of 96 on all COCA surfaces.

## Change ledger

**v1 → v2 (5 savants: prior art, iron rules, code truth, cascade, different
views; 0 VIOLATES).**
- Rank rule taken from ndarray `rank_per_10000`; `NestedBands` cited.
- The proposed new id D-LXC-12 was dropped, because it duplicated D-LXC-5.
- `CoverageBand` naming.
- Relative bands kept, with T6 and a printed share range; the absolute
  alternative is recorded with numbers.
- Reading-share key made explicit (F8) and measured.
- T3 split into three conditions.
- Wording kept free of statistics.
- Population defined by the tagger's own lemma map; calibrator uses no
  fallback.
- `vocab.len()` passed in; n = 0/1 safe.
- G8 numbering continues the parent plan.
- Deny-list-compliant board path.
- "Rolling floor" reworded.

**v2 → v3 (3 reviewers: overclaim, dilution/collapse, firewall; 0 BLOCK;
3 P1, 9 P2).**
- One rank rule named everywhere (it had been described both as nearest-rank
  and as `rank_per_10000`; they differ).
- Receipt, method and asset digest added for every number.
- The "0 differ" result explained: no top reading shares its state.
- `CoverageBand` used consistently.
- Gates renamed G8a-c; G8c labelled a KJV-only pin.
- The tests are required to be disable-verified before landing; they are not
  described as already verified.
- T5 uses explicit cuts. T6 asserts all Decisive. T4 gains n = 1. T8 added.
- Merged `None` reasons stated as a decision.
- The early exit kept visible as a non-goal.
- The undefined "variant A" removed.
- The dependency claim cited (`Cargo.toml:17-20`).
- Commit contents listed; the STATUS_BOARD row handed to the operator.
