# 2026-10-04 — WordNet may feed CLAM/CHAODA; grammar reads by ReadParams, not thinking styles (D-LXC-22)

## DECISION — WordNet hypernym metric for CLAM/CHAODA

Operator, 2026-10-04: *"Yes Wordnet would allow also clam Chaoda."* The
operator gave this answer to the scope question put in
`.claude/knowledge/german-grammar-rule-inventory.md` §6.

- **DECISION:** ontology concept ids remain identities. CAM-PQ and any learned or embedded distance over them stay prohibited (`48405aa2` unchanged).
- **SCOPE:** an exact graph metric derived solely from WordNet's own hypernym edges (shortest path, minimised over multiple inheritance) MAY serve as the `Distance` for `ndarray::hpc::clam::ClamTree` and its CHAODA anomaly scores. It is used offline, as an evaluation and research instrument. It is never stored as a concept attribute, never mixed into the CAM-PQ / palette space, and never consulted on the substrate hot path.
- **BASIS:** the metric is defined by the ontology's own edges, not trained. That is the difference from the CAM-PQ case `48405aa2` rejected.
- **REVISIT WHEN:**
  - a proposal moves it to run time or into a quorum voter;
  - a proposal extends it to a non-WordNet ontology;
  - its swap AUC fails to beat frequency (the probe in the inventory, rank 18).
- **Carried constraint (second premise gate):** a path distance is symmetric. Points must be role-indexed, `(verb, role, synset)`, or the items restricted to verb swaps.

## Grammar is not a thinking style — `tekamolo.rs` reads by `ReadParams`

**Checked:**
- The 36 `ThinkingStyle` adjectives and the 12 `StyleFamily` macros contain no grammar style.
- Grammar is rung 2 of the content ladder: the 144 universal-grammar verbs and `verb_table`'s TEKAMOLO slot priors (`.claude/v3/knowledge/persona-vs-rung-ladder.md`).
- The restored `tekamolo.rs` drove its reading knobs from the persona adjectives through `V2StyleProvider`, which is the wrong ladder.

**Changed:**
- `read_clause` takes `ReadParams` (commit point, fan-out, margin, admit-ambiguous), with the presets `ReadParams::LEFT_CORNER` and `ReadParams::RIGHT_CORNER`.
- The style mapping is kept, not dropped, as the adapter `ReadParams::from_style` (plus `scan_for_style`).
- Its destination is the planner, and long term a thinking dialect on `ogar-loco`. The operator named `ogar-loco` / `ogar-r2il` as the long-term IR substrate; `ogar-loco`'s basin doc already treats a thinking IR as a caller that plugs in its vocabulary.

**Test:** `the_presets_equal_the_adapter_rows` pins the presets to the adapter's Analytical and Exploratory rows. Disable-verified (a changed preset fails it). deepnsm-v2 has 195 lib tests; clippy is clean.

**OPEN:** `V2StyleProvider` still lives in deepnsm-v2. Moving it out is its own change, once the planner side or a loco dialect exists to receive it.

## WordNet gives the address, COCA + tokens give the torque (operator, 2026-10-04)

*"The 'negative' about WordNet is that it doesn't contain linguistic torque. It
needs COCA and tokens to have a qualia about the text. In return it offers a
Cartesian address."*

| source | gives | does not give |
|---|---|---|
| WordNet | the Cartesian address: parent/child, the HHTL DN path, masks across branches | torque: frequency, collocation, position, how a word moves in a sentence |
| COCA + tokens | the torque: frequency, word forms, position, context, i.e. the qualia of the text | a stable concept address |

**Consistent with the measured record.** D-LXC-15: WordNet helped noun/verb neither as a chooser nor as a reading filter, while COCA frequency × position did.

**Consequence for the probes.** WordNet never judges alone.
- Rank 17's animacy mask is the coordinate. The plausibility signal comes from COCA/token counts and the slot position.
- Rank 18's CHAODA points combine both: (WordNet address of S/O, verb, COCA frequency band, slot position). Density over addresses alone would measure an unusual concept location, not an implausible sentence.
