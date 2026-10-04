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
