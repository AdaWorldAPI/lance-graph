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

**Round 3 (pre-registered before its labels existed): pending.**
- Data: a fresh sample of 400 HDT test phrases, two labelers each.
- Bars:
  - κ ≥ 0.6;
  - noun voter TIME F1 ≥ 0.80 and PLACE F1 ≥ 0.55;
  - quorum ≥ noun + 2 points.
