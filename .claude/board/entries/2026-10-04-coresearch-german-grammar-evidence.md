# 2026-10-04 — /coresearch: German grammar as position + morphology evidence (exploration map)

**Status:** OPEN — an exploration map. Nothing is adopted; the operator chooses.
Map: `.claude/knowledge/german-grammar-rule-inventory.md`.

**Council:**
- Brief and premise gate: PREMISE-SPLIT into Q1 word class, Q2 case, Q3 anaphora, Q4 TEKAMOLO, Q5 plausibility.
- 5 scouts: code, internal prior art, literature, linguistic resources, grammar concepts.
- A 23-row crosswalk.
- 3 co-architects: bridge, firewall, falsifier.

## Key results

**Already have.**
- Satzklammer adj/adv rules: 88.5 % / 90.0 %.
- German adj/adv quorum: 85.1 %.
- The TEKAMOLO order law is killed as a decider; the cue-by-verb-position rule scores 88.5 %.

**Inventories without consumers.**
- Five mined TSVs have no Rust reader: `article_decidability`, `valency`, `wechsel`, `relative_pronoun`, `reflexive`.
- They become one Q2 case-constraint evaluator (`ud_case_eval`), not five voters.
- Shared precondition: `next_form` in `Tok`, plus the tables re-mined on the fit split.

**Data facts that change the design.**
- UD German has `iobj` = 0: a dative object is obj/obl + Case=Dat.
- ADP carries no Case.
- GSD Case gold is about 10 % noisy, so Case kills must hold on HDT.

**Cheapest probes first.**
- Fixed-case prepositions.
- der/die/das DET vs PRON.
- Relative-pronoun case.
- Paradigm-exact monoflexion (read on HDT).
- Expletive es.
- Separable particles.
- Auxiliary selection.

**Operator additions (2026-10-04).**
- **Belief transitions:** re-filed by the second premise gate. Revision history is a planner belief-arena record (magnitudes ordered across books), not 24×i4 loci; only a peer pointer within ±8 may be a locus. Probe: the history must predict later contradiction beyond the final TruthValue.
- **WordNet → ndarray CLAM/CHAODA:** outside the letter of `48405aa2`, inside its spirit. Clean only offline, under a recorded DECISION (draft wording in the map, §6). Second premise gate: a path distance is symmetric, so S↔O swaps are invisible; points must be role-indexed `(verb, role, synset)`, or the items restricted to verb swaps.
- **hydrate = node creation:** ALREADY-HAVE (restored, green on the KJV).

## Not searched

- adverbial vs predicative ADJD accuracy;
- a computational Wechselpräposition accuracy;
- German Winograd beyond Wino-X;
- adverbial-order corpus studies;
- left-corner parsing for German.
