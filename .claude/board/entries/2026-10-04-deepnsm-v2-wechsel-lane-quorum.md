# 2026-10-04 — TIME / PLACE / FIG for Wechsel phrases: a quorum over noun, adverb, verb and position evidence (D-LXC-25)

**Status:** MEASURED (debug-0, release), German HDT. The first round is pre-registered (KILL). The second is exploratory, and a fresh confirmation round has been pre-registered and is pending. Instrument: `crates/deepnsm-v2/examples/ud_lane_quorum.rs`.

Operator:
- *"Time/place Pronomen with abstract can only be time or figuratively"*;
- *"add tekamolo position and adverb and noun context and you have a quorum"*.

## Labels (silver)

- German UD has no lane gold. Independent Sonnet labelers read the full sentence and labelled each phrase TIME / PLACE / FIG, using `examples/data/hdt_wechsel_lanes.BRIEF.md`.
- Train has 800 items, one labeler each. Test has 400 items, two labelers each.
- **Cohen κ = 0.958** (98.0 % agreement): RELIABLE.
- The file `examples/data/hdt_wechsel_lanes.tsv` holds ids and labels only, no HDT text (HDT is CC BY-SA 4.0).
- **κ caveat.** The two labelers on a test item are two runs of the same model, not independent annotators.
  - On the confirmation sample, fresh_1 A and B agreed 200/200 without reading each other's files; their transcripts were checked.
  - So κ measures model self-consistency. It is an upper bound on label quality, not human-grade agreement.
- **One labeler discarded.** fresh_0 B overwrote its reading-based labels with a rule-based script, against the brief. A replacement labeler (C) was dispatched.

## Results

**Operator hypothesis: PASS.** Of 105 abstract-head items, 4 are PLACE (3.8 % ≤ 5 %): 95 FIG, 6 TIME.

**Round 1 (pre-registered): KILL.** The quorum scores 72.7 %, below the noun voter alone (80.4 %). Every voter has PLACE F1 ≈ 0, because nothing separates a physical place from an institutional figurative one.

**Round 2 (exploratory; changes made after seeing the test errors).**
- Added a `name` noun class: heads whose train majority UPOS is PROPN.
- Time nouns also mined from bare `obl` nouns: no `case` child and no `nummod` child.
  - The first attempt used `obl*` and mined 1,052 nouns. That included the German `obl:arg` bare datives (*Kunden*) and measure phrases (*Prozent*, *Dollar*).
  - The corrected version mines 411.
- Compound-head match: *Geschäftsquartal* → *quartal*.

| | accuracy | TIME F1 | PLACE F1 | FIG F1 |
|---|---|---|---|---|
| noun voter | 86.5 % | 0.830 | 0.603 | 0.905 |
| quorum (all voters) | 88.0 % | 0.831 | 0.655 | 0.918 |

- Ablation: dropping the noun voter costs 19 points. Dropping any other voter costs ≤ 1 point.
- The quorum misses its "+2 points over the best single voter" bar: noun context carries almost all the signal.
- TEKAMOLO position, adverb and verb add little on this sample.

**Round 3 (pre-registered before its labels existed; design frozen).** Data: a fresh sample of 400 HDT test phrases. fresh_0 used labelers A + C, fresh_1 used A + B. κ = 0.969, subject to the same-model caveat above.

| bar | result | verdict |
|---|---|---|
| κ ≥ 0.6 | 0.969 | — |
| noun voter TIME F1 ≥ 0.80 | **0.831** | PASS |
| noun voter PLACE F1 ≥ 0.55 | 0.476 | **KILL** |
| quorum ≥ noun + 2 points | 85.3 % vs 84.8 % (+0.5) | **KILL** |
| abstract head never PLACE (≤ 5 %) | 3 of 94 (3.2 %) | PASS (replicated) |

**Verdict.**
- Noun context detects TIME reliably, and the abstract-head rule replicates.
- PLACE is not solved.
- On HDT, TEKAMOLO position, adverb, verb and article add nothing measurable over the noun voter.

**Exceptions to the abstract rule.** The abstract-head PLACE items across all labelled samples:
- concrete *-ung* artifacts: *Verpackung*, *Packung*;
- a direction: *in Richtung*;
- an institution: *an der Bauhaus-Universität*.

The suffix class mixes event and abstract nouns with object nouns (*Wohnung*, *Zeitung*, *Leitung*, *Heizung*).

**Register.** Operator: colloquial *auf der Arbeit* reads as PLACE. HDT is formal IT-news text (heise/c't), so colloquial place readings are under-sampled here. A colloquial frequency source is a prerequisite for that axis.

**OPEN.**
- A PLACE signal beyond names: object nouns, a physical-artifact lexicon, gender or semantic class.
- The German lexicon (`build_de_codebook.py`, UD-derived, COCA shape) has no gender column, although UD carries `Gender=`.

## Addendum: adverbs of time (operator: *früh*) — exploratory, both samples already seen

**Coverage.** *früh* was **not** covered.
- HDT tags it ADJ + `advmod` (an uninflected adjective used adverbially, the same German adj/adv split as D-LXC-20), and the first miner took UPOS ADV only.
- It occurs about 17 times in HDT train (formal IT news).
- The first "temporal adverb" list was mined as "the word right before *seit/während/bis*". It was mostly focus particles and other lanes: *auch*, *nur*, *sogar*, *aber*, *hier*, *unten*, *daher*.

| adverb source | test quorum | confirm quorum | confirm, without the adverb voter |
|---|---|---|---|
| word before *seit/während/bis* (first version) | 88.0 % | 85.3 % | 84.8 % |
| lift-mined (ADV, or ADJ-`advmod`, ≥ 2× in clauses with a time-only preposition) | 88.0 % | 85.8 % | 86.3 % |
| codebook `TEMPORAL` list (`build_de_codebook.py`: *früh, spät, heute, damals, …*; `LANE_ADVERBS=codebook`) | 88.3 % | 85.3 % | 86.3 % |

**Direction (all 1,586 agreed labels).** Testing the operator's Sudoku idea that a filled Te slot pushes the PP to another lane:
- a clause with a time adverb has **more** TIME phrases (32.8 % vs 21.0 %), not fewer;
- time adverbs mostly modify the time PP itself (*erst in 12 Monaten*, *schon vor der Fahrt*) rather than fill a separate Te slot;
- the signal is real but largely redundant with the time-noun voter, so the adverb voter adds nothing.

**OPEN.**
- A stand-alone temporal adverb (*früh*, *heute*) as a TEKAMOLO Te element is a word-class question, the adj/adv quorum of D-LXC-20, not a PP-lane question.
- The lane reader does not yet place bare adverbs into Te.

> **⊘ Corrected 2026-10-04 (PR #1321 review): the abstract-head counts included the shared train labels.**
> Both rounds counted the 800 train items, so the two rounds were not independent. On held-out items only:
> - exploratory test: 1 of 35 abstract heads is PLACE (2.9 %), PASS;
> - fresh confirmation: 0 of 24 (0 %), PASS.
>
> Both are small samples.
>
> `LANE_SPLIT=confirm` now prints the Round 3 bars it registered:
> - noun-voter TIME F1 0.831: PASS;
> - PLACE F1 0.476: KILL;
> - quorum ≥ noun + 2 points: KILL.
>
> The verdicts are unchanged.
