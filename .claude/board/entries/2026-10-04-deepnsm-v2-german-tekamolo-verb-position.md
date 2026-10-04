# 2026-10-04 — deepnsm-v2: German TEKAMOLO cues decided by verb position; the Te-Ka-Mo-Lo order law fails on real text (D-LXC-17)

**Status:** MEASURED (`cargo run --release --example tekamolo_de -- de_gsd-ud-train.conllu
de_gsd-ud-test.conllu`; `ud_pos_eval` for the adverbial-adjective fold; UD r2.15).

Operator (2026-10-04): "In German we have TEKAMOLO" — and DeepNSM had it.

## Where TEKAMOLO lived

- deepnsm-v2 had `src/tekamolo.rs` (718 lines, `7bae6c84`): German left-corner
  lane hypotheses over Luther 1545 / Elberfelder 1905 from `de/tekamolo.tsv`,
  committed at the clause's right corner. Deleted in `68955ecb` as redundant to
  `lance-graph-planner/examples/insight_coca_read.rs` (English, COCA tables).
- deepnsm v1: TEKAMOLO role slices in `markov_bundle`; `parser.rs` leaves
  `TekamoloSlots` empty ("until D3").
- `de/tekamolo.tsv` (release `v0.1.0-codebooks-2026-07-26`): lanes from a
  hand cue list (`build_de_codebook.py`), counted by UD relation — no gold lanes.

## 1. The Te < Ka < Mo < Lo order law (lab, UD German GSD gold dependencies)

| test | pairs | agree |
|---|---|---|
| cue-lexicon adverbials, same head, any position | 297 train / 46 test | 60.3 % / 65.2 % |
| Mittelfeld only (after the finite verb), advmod/obl | 92 / 17 | 45.7 % / 58.8 % |
| `obl` laned by a single-lane preposition | 37 / 0 | 54.1 % / — |

Ka-before-Te violations are subordinators opening their clause; the cue
lexicon mostly hits particles (noch, schon, so, sehr) whose position follows
scope. Laned PPs rarely co-occur. On this text the order law cannot decide a
lane.

## 2. Cue words by verb position (Rust, `tekamolo_de`, no gold at test)

Signatures learned over every cue token of train (word-independent):
V2 (finite verb next) → adv 180 / mark 7 / case 1; verb-final segment →
mark 507 / case 293 / adv 84; noun phrase → case 2160 / adv 336 / mark 179.
Finiteness measured from train (2,394 forms), case kept.

| | n | position | frequency | position × frequency |
|---|---|---|---|---|
| all 11 cues | 208 | 57.2 % | 84.1 % | **88.5 %** |
| `da` | 16 | 68.8 % | 37.5 % | **87.5 %** |
| `während` | 12 | 50.0 % | 33.3 % | 50.0 % |
| `als` | 44 | 81.8 % | 77.3 % | 77.3 % |

Lane mapping the classes feed: `da`+mark → Kausal, `da`+adv → Lokal/Temporal;
`als`/`während`/`bis`/`seit`/`nachdem` → Temporal; `damit`+mark → Kausal
(final); `so`+adv → Modal.

## 3. Adverbially used adjectives fill the Modal lane

`ud_pos_eval`'s fold now maps UPOS ADJ with relation `advmod` to `Adv`
(language-neutral). German Adj/Adv tokens 234 → 315; the position table alone
decides 182 at 87.4 % vs frequency 82.4 % on them; position × frequency
82.9 % vs 80.0 % overall. A "copula anywhere in the clause" context feature
(`UD_NO_CLAUSE_CTX` ablates it) helps German (+0.4 adj/adv, +0.6 noun/verb)
and costs English/French (up to −2.0); per-language selection on dev is OPEN.

## OPEN

- Lane-level gold does not exist; lanes stay a mapping from the decided class.
- The deleted left-corner module could be re-measured against this signature
  method on the Luther/Elberfelder lanes.
- `so`/`damit`: position alone is wrong (mid-clause adverbs in verb-final
  segments); frequency carries them.
